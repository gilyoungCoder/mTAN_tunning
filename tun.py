import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from random import SystemRandom
from sklearn.model_selection import ParameterSampler
import models
import utils
from setmodels import *


def main(args):
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    num_tp = data_obj["num_tp"]
    print(f"num tp : {num_tp}")

    rec = models.enc_mtan_rnn(
        dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)

    dec = models.dec_mtan_rnn(
        dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)
        
    classifier = models.create_classifier(args.latent_dim, args.rec_hidden).to(device)

    aug = models.TimeSeriesAugmentation(dim*2+1, args.augh1, args.augh2, dim*2+1, num_outputs=args.aug_ratio*num_tp).to(device)
    
    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()) + list(aug.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier), utils.count_parameters(aug))
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    best_val_loss = float('inf')
    best_test_auc = 0
    best_val_auc = 0
    total_time = 0.
    
    val_losses = []
    test_aucs = []
    
    for itr in range(1, args.niters + 1):
        train_recon_loss, train_ce_loss, train_reg_loss = 0, 0, 0
        mse = 0
        train_n = 0
        train_acc = 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.99**(itr - wait_until_kl_inc))
        else:
            kl_coef = 1
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len = train_batch.shape[0]
            ## data augmentation
            observed_data, observed_mask, observed_tp = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            
            x_aug, tp_aug = aug(observed_tp, torch.cat((observed_data, observed_mask), 2))
                    
            mask_aug = torch.where(
                x_aug[:, :, dim:2*dim] < 0.5,  # 조건
                torch.zeros_like(x_aug[:, :, dim:2*dim]),  # 조건이 True일 때 적용할 값
                x_aug[:, :, dim:2*dim]  # 조건이 False일 때 적용할 값
            )    
            data_aug = x_aug[:, :, :dim]
            
            data = torch.cat((observed_data, data_aug), -2)
            mask = torch.cat((observed_mask, mask_aug), -2)

            tt = torch.cat((observed_tp, tp_aug), -1)

            reg_loss = utils.diversity_regularization(tt, drate=0.5)

            out = rec(torch.cat((data, mask), 2), tt)
            
            qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
            epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_y = classifier(z0)
            
            pred_x = dec(z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])  # nsample, batch, seqlen, dim
            
            logpx, analytic_kl = utils.compute_losses(dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            label = label.unsqueeze(0).repeat_interleave(args.k_iwae, 0).view(-1)
            ce_loss = criterion(pred_y, label)
            loss = recon_loss + args.alpha * ce_loss + args.beta * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_ce_loss += ce_loss.item() * batch_len
            train_recon_loss += recon_loss.item() * batch_len
            train_reg_loss += reg_loss.item() * batch_len
            train_acc += (pred_y.argmax(1) == label).sum().item() / args.k_iwae
            train_n += batch_len
            mse += utils.mean_squared_error(observed_data, pred_x.mean(0), observed_mask) * batch_len
            
        total_time += time.time() - start_time
        val_loss, val_acc, val_auc = utils.evaluate_classifier(rec, aug, val_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        test_loss, test_acc, test_auc = utils.evaluate_classifier(rec, aug, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)

        val_losses.append(val_loss)
        test_aucs.append(test_auc)
        
        if val_loss <= best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            # best_test_auc = test_auc
            rec_state_dict = rec.state_dict()
            dec_state_dict = dec.state_dict()
            classifier_state_dict = classifier.state_dict()
            optimizer_state_dict = optimizer.state_dict()
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
        cur_reg_loss = args.beta * train_reg_loss / train_n
        print('Iter: {}, recon_loss: {:.4f}, ce_loss: {:.4f}, reg_loss: {:.4f}, acc: {:.4f}, mse: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
              .format(itr, train_recon_loss / train_n, args.alpha * train_ce_loss / train_n, cur_reg_loss,
                      train_acc / train_n, mse / train_n, val_loss, val_acc, test_acc, test_auc))
        
        if best_val_loss * 1.2 < val_loss:
            print("early stop")
            break

    print("Best Validation Loss: ", best_val_loss)
    print("Test AUC at Best Validation Loss: ", best_test_auc)
    print(total_time)
    return best_test_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--std', type=float, default=0.01)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--rec-hidden', type=int, default=32)
    parser.add_argument('--gen-hidden', type=int, default=50)
    parser.add_argument('--embed-time', type=int, default=128)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--enc', type=str, default='mtan_rnn')
    parser.add_argument('--dec', type=str, default='mtan_rnn')
    parser.add_argument('--fname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--n', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--quantization', type=float, default=0.1, 
                        help="Quantization on the physionet dataset.")
    parser.add_argument('--classif', action='store_true', 
                        help="Include binary classification loss")
    parser.add_argument('--freq', type=float, default=10.)
    parser.add_argument('--k-iwae', type=int, default=10)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--learn-emb', action='store_true')
    parser.add_argument('--dataset', type=str, default='physionet')
    parser.add_argument('--alpha', type=int, default=100)
    parser.add_argument('--beta', type=int, default=1000000)
    parser.add_argument('--gamma', type=int, default=10000)
    parser.add_argument('--old-split', type=int, default=1)
    parser.add_argument('--nonormalize', action='store_true')
    parser.add_argument('--enc-num-heads', type=int, default=1)
    parser.add_argument('--dec-num-heads', type=int, default=1)
    parser.add_argument('--num-ref-points', type=int, default=128)
    parser.add_argument('--classify-pertp', action='store_true')
    parser.add_argument('--aug-ratio', type=int, default=2)
    parser.add_argument('--augh1', type=int, default=300)
    parser.add_argument('--augh2', type=int, default=256)

    args = parser.parse_args()

    param_grid = {
        'alpha': [10, 50, 100, 200, 500],
        'niters': [100, 200, 300, 400, 500],
        'lr': [0.01, 0.001, 0.0001, 0.00001],
        'batch_size': [16, 32, 50, 64, 128],
        'rec_hidden': [64, 128, 256, 512],
        'gen_hidden': [50, 100, 150],
        'latent_dim': [10, 20, 32, 64],
        'enc': ['mtan_rnn'],
        'dec': ['mtan_rnn'],
        'n': [8000],
        'quantization': [0.016],
        'save': [1],
        'classif': [True],
        'norm': [True],
        'kl': [True],
        'learn_emb': [True],
        'k_iwae': [1, 5, 10, 20],
        'dataset': ['physionet'],
        'aug_ratio': [3, 5, 7, 10],
        'augh1': [128, 256, 300, 512],  # 추가된 파라미터 augh1
        'augh2': [128, 256, 512]        # 추가된 파라미터 augh2
}

    n_iter_search = 10
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter_search, random_state=args.seed))

    best_score = 0
    best_params = None

    for param_set in param_list:
        for param, value in param_set.items():
            setattr(args, param, value)
        print(f"Running with parameters: {param_set}")
        best_test_auc = main(args)
        # Assuming you save the best_test_auc in main() function
        if best_test_auc > best_score:
            best_score = best_test_auc
            best_params = param_set

    print(f"Best score: {best_score} with parameters: {best_params}")
