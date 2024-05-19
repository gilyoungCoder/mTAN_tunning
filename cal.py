import re
import numpy as np

# 예시 데이터 (실제 데이터로 교체)
data_string = """
Iter: 6, recon_loss: 40.7038, ce_loss: 40.3677, reg_loss: 20.3955, q_loss: 0.0000, acc: 0.8613, mse: 0.0086, val_loss: 0.4156, val_acc: 0.8547, val_auc: 0.4488, test_acc: 0.8675, test_auc: 0.4632
Iter: 7, recon_loss: 40.2193, ce_loss: 40.3883, reg_loss: 19.6430, q_loss: 0.0000, acc: 0.8613, mse: 0.0086, val_loss: 0.4135, val_acc: 0.8547, val_auc: 0.5371, test_acc: 0.8675, test_auc: 0.4932
Iter: 8, recon_loss: 39.9651, ce_loss: 40.3893, reg_loss: 19.0972, q_loss: 0.0000, acc: 0.8613, mse: 0.0085, val_loss: 0.4153, val_acc: 0.8547, val_auc: 0.4752, test_acc: 0.8675, test_auc: 0.4324
Iter: 9, recon_loss: 39.8176, ce_loss: 40.3692, reg_loss: 18.6951, q_loss: 0.0000, acc: 0.8613, mse: 0.0085, val_loss: 0.4144, val_acc: 0.8547, val_auc: 0.5183, test_acc: 0.8675, test_auc: 0.4897
Iter: 10, recon_loss: 39.6891, ce_loss: 40.3689, reg_loss: 18.4021, q_loss: 0.0000, acc: 0.8613, mse: 0.0085, val_loss: 0.4130, val_acc: 0.8547, val_auc: 0.5836, test_acc: 0.8675, test_auc: 0.4793
Iter: 11, recon_loss: 40.1329, ce_loss: 40.3134, reg_loss: 18.1450, q_loss: 0.0000, acc: 0.8613, mse: 0.0085, val_loss: 0.4152, val_acc: 0.8547, val_auc: 0.4780, test_acc: 0.8675, test_auc: 0.5367
Iter: 12, recon_loss: 40.5165, ce_loss: 40.3858, reg_loss: 17.9744, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4151, val_acc: 0.8547, val_auc: 0.4839, test_acc: 0.8675, test_auc: 0.5233
Iter: 13, recon_loss: 40.7980, ce_loss: 40.3358, reg_loss: 17.8042, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4149, val_acc: 0.8547, val_auc: 0.4861, test_acc: 0.8675, test_auc: 0.5191
Iter: 14, recon_loss: 40.9365, ce_loss: 40.3647, reg_loss: 17.6851, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4123, val_acc: 0.8547, val_auc: 0.5878, test_acc: 0.8675, test_auc: 0.5723
Iter: 15, recon_loss: 40.9731, ce_loss: 40.3518, reg_loss: 17.5425, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4140, val_acc: 0.8547, val_auc: 0.5311, test_acc: 0.8675, test_auc: 0.5136
Iter: 16, recon_loss: 40.8998, ce_loss: 40.3667, reg_loss: 17.4089, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4143, val_acc: 0.8547, val_auc: 0.5230, test_acc: 0.8675, test_auc: 0.5407
Iter: 17, recon_loss: 40.7883, ce_loss: 40.4128, reg_loss: 17.3539, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4128, val_acc: 0.8547, val_auc: 0.5723, test_acc: 0.8675, test_auc: 0.4954
Iter: 18, recon_loss: 40.6318, ce_loss: 40.2073, reg_loss: 17.2541, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4142, val_acc: 0.8547, val_auc: 0.5220, test_acc: 0.8675, test_auc: 0.5301
Iter: 19, recon_loss: 40.4430, ce_loss: 40.2830, reg_loss: 17.2829, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4138, val_acc: 0.8547, val_auc: 0.5282, test_acc: 0.8675, test_auc: 0.6102
Iter: 20, recon_loss: 40.3663, ce_loss: 40.1833, reg_loss: 17.3420, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4103, val_acc: 0.8547, val_auc: 0.6109, test_acc: 0.8675, test_auc: 0.6254
Iter: 21, recon_loss: 40.3018, ce_loss: 40.1777, reg_loss: 17.1956, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4095, val_acc: 0.8547, val_auc: 0.6272, test_acc: 0.8675, test_auc: 0.5588
Iter: 22, recon_loss: 40.1546, ce_loss: 40.1280, reg_loss: 17.1801, q_loss: 0.0000, acc: 0.8613, mse: 0.0084, val_loss: 0.4086, val_acc: 0.8547, val_auc: 0.6122, test_acc: 0.8675, test_auc: 0.6158
Iter: 23, recon_loss: 39.8445, ce_loss: 39.8931, reg_loss: 17.2595, q_loss: 0.0000, acc: 0.8613, mse: 0.0083, val_loss: 0.4109, val_acc: 0.8547, val_auc: 0.5702, test_acc: 0.8675, test_auc: 0.6029
Iter: 24, recon_loss: 39.6586, ce_loss: 39.8965, reg_loss: 17.1131, q_loss: 0.0000, acc: 0.8613, mse: 0.0083, val_loss: 0.4040, val_acc: 0.8547, val_auc: 0.6348, test_acc: 0.8675, test_auc: 0.6237
Iter: 25, recon_loss: 39.0398, ce_loss: 39.6453, reg_loss: 17.2682, q_loss: 0.0000, acc: 0.8613, mse: 0.0081, val_loss: 0.4002, val_acc: 0.8547, val_auc: 0.6582, test_acc: 0.8675, test_auc: 0.6486
Iter: 26, recon_loss: 38.5995, ce_loss: 39.6189, reg_loss: 17.0566, q_loss: 0.0000, acc: 0.8613, mse: 0.0081, val_loss: 0.4019, val_acc: 0.8547, val_auc: 0.6265, test_acc: 0.8675, test_auc: 0.6634
Iter: 27, recon_loss: 38.0826, ce_loss: 39.4053, reg_loss: 17.2639, q_loss: 0.0000, acc: 0.8613, mse: 0.0080, val_loss: 0.3973, val_acc: 0.8547, val_auc: 0.6659, test_acc: 0.8675, test_auc: 0.6521
Iter: 28, recon_loss: 37.6185, ce_loss: 39.4201, reg_loss: 17.1180, q_loss: 0.0000, acc: 0.8613, mse: 0.0079, val_loss: 0.3973, val_acc: 0.8547, val_auc: 0.6609, test_acc: 0.8675, test_auc: 0.6517
Iter: 29, recon_loss: 37.1026, ce_loss: 39.1393, reg_loss: 16.8730, q_loss: 0.0001, acc: 0.8613, mse: 0.0078, val_loss: 0.3940, val_acc: 0.8547, val_auc: 0.6672, test_acc: 0.8675, test_auc: 0.6726
Iter: 30, recon_loss: 36.8905, ce_loss: 39.0154, reg_loss: 17.1241, q_loss: 0.0001, acc: 0.8613, mse: 0.0077, val_loss: 0.3946, val_acc: 0.8547, val_auc: 0.6663, test_acc: 0.8675, test_auc: 0.6721
Iter: 31, recon_loss: 36.7578, ce_loss: 39.1451, reg_loss: 16.9688, q_loss: 0.0001, acc: 0.8613, mse: 0.0077, val_loss: 0.3887, val_acc: 0.8547, val_auc: 0.6959, test_acc: 0.8675, test_auc: 0.6952
Iter: 32, recon_loss: 36.4142, ce_loss: 38.7669, reg_loss: 17.1178, q_loss: 0.0002, acc: 0.8613, mse: 0.0076, val_loss: 0.3911, val_acc: 0.8547, val_auc: 0.6805, test_acc: 0.8675, test_auc: 0.6703
Iter: 33, recon_loss: 36.2318, ce_loss: 38.6459, reg_loss: 16.9844, q_loss: 0.0004, acc: 0.8613, mse: 0.0076, val_loss: 0.3815, val_acc: 0.8547, val_auc: 0.7112, test_acc: 0.8675, test_auc: 0.6748
Iter: 34, recon_loss: 36.3434, ce_loss: 38.7967, reg_loss: 16.9429, q_loss: 0.0007, acc: 0.8613, mse: 0.0076, val_loss: 0.3923, val_acc: 0.8547, val_auc: 0.6763, test_acc: 0.8675, test_auc: 0.6963
Iter: 35, recon_loss: 36.3484, ce_loss: 38.4502, reg_loss: 17.1218, q_loss: 0.0021, acc: 0.8613, mse: 0.0076, val_loss: 0.3828, val_acc: 0.8547, val_auc: 0.7086, test_acc: 0.8675, test_auc: 0.7173
Iter: 36, recon_loss: 35.9522, ce_loss: 38.3186, reg_loss: 16.9044, q_loss: 0.0038, acc: 0.8613, mse: 0.0076, val_loss: 0.3808, val_acc: 0.8547, val_auc: 0.7113, test_acc: 0.8675, test_auc: 0.6977
Iter: 37, recon_loss: 35.7706, ce_loss: 38.3783, reg_loss: 16.9572, q_loss: 0.0042, acc: 0.8613, mse: 0.0075, val_loss: 0.3756, val_acc: 0.8547, val_auc: 0.7318, test_acc: 0.8675, test_auc: 0.7276
Iter: 38, recon_loss: 35.7951, ce_loss: 38.0295, reg_loss: 17.0590, q_loss: 0.0043, acc: 0.8613, mse: 0.0075, val_loss: 0.3839, val_acc: 0.8547, val_auc: 0.6981, test_acc: 0.8675, test_auc: 0.6995
Iter: 39, recon_loss: 35.9188, ce_loss: 38.1925, reg_loss: 17.0398, q_loss: 0.0053, acc: 0.8613, mse: 0.0076, val_loss: 0.3748, val_acc: 0.8547, val_auc: 0.7420, test_acc: 0.8675, test_auc: 0.7288
Iter: 40, recon_loss: 35.5656, ce_loss: 37.6456, reg_loss: 16.8937, q_loss: 0.0100, acc: 0.8613, mse: 0.0075, val_loss: 0.3760, val_acc: 0.8547, val_auc: 0.7291, test_acc: 0.8675, test_auc: 0.7163
Iter: 41, recon_loss: 35.7161, ce_loss: 37.5383, reg_loss: 17.1120, q_loss: 0.0154, acc: 0.8613, mse: 0.0075, val_loss: 0.3681, val_acc: 0.8547, val_auc: 0.7498, test_acc: 0.8675, test_auc: 0.7341
Iter: 42, recon_loss: 35.5080, ce_loss: 37.6912, reg_loss: 16.8684, q_loss: 0.0146, acc: 0.8613, mse: 0.0075, val_loss: 0.3777, val_acc: 0.8547, val_auc: 0.7266, test_acc: 0.8675, test_auc: 0.7211
Iter: 43, recon_loss: 35.6089, ce_loss: 37.5105, reg_loss: 16.9094, q_loss: 0.0148, acc: 0.8613, mse: 0.0075, val_loss: 0.3678, val_acc: 0.8547, val_auc: 0.7531, test_acc: 0.8675, test_auc: 0.7462
Iter: 44, recon_loss: 35.4133, ce_loss: 37.3585, reg_loss: 16.8869, q_loss: 0.0124, acc: 0.8613, mse: 0.0075, val_loss: 0.3711, val_acc: 0.8547, val_auc: 0.7494, test_acc: 0.8675, test_auc: 0.7330
Iter: 45, recon_loss: 35.3041, ce_loss: 37.0212, reg_loss: 16.9579, q_loss: 0.0100, acc: 0.8613, mse: 0.0075, val_loss: 0.3685, val_acc: 0.8547, val_auc: 0.7500, test_acc: 0.8675, test_auc: 0.7425
Iter: 46, recon_loss: 35.2826, ce_loss: 37.1824, reg_loss: 17.2737, q_loss: 0.0106, acc: 0.8613, mse: 0.0074, val_loss: 0.3663, val_acc: 0.8547, val_auc: 0.7655, test_acc: 0.8675, test_auc: 0.7534
Iter: 47, recon_loss: 35.1246, ce_loss: 37.1130, reg_loss: 16.8558, q_loss: 0.0096, acc: 0.8613, mse: 0.0074, val_loss: 0.3657, val_acc: 0.8547, val_auc: 0.7646, test_acc: 0.8675, test_auc: 0.7507
Iter: 48, recon_loss: 34.9915, ce_loss: 37.0394, reg_loss: 16.8674, q_loss: 0.0059, acc: 0.8613, mse: 0.0074, val_loss: 0.3739, val_acc: 0.8547, val_auc: 0.7481, test_acc: 0.8675, test_auc: 0.7492
Iter: 49, recon_loss: 35.0318, ce_loss: 36.7728, reg_loss: 16.9537, q_loss: 0.0060, acc: 0.8613, mse: 0.0074, val_loss: 0.3684, val_acc: 0.8547, val_auc: 0.7627, test_acc: 0.8675, test_auc: 0.7468
Iter: 50, recon_loss: 35.0312, ce_loss: 36.7422, reg_loss: 16.9400, q_loss: 0.0085, acc: 0.8613, mse: 0.0074, val_loss: 0.3659, val_acc: 0.8547, val_auc: 0.7696, test_acc: 0.8675, test_auc: 0.7487
Iter: 51, recon_loss: 34.8663, ce_loss: 36.6978, reg_loss: 16.9421, q_loss: 0.0070, acc: 0.8613, mse: 0.0074, val_loss: 0.3621, val_acc: 0.8547, val_auc: 0.7718, test_acc: 0.8675, test_auc: 0.7510
Iter: 52, recon_loss: 34.6328, ce_loss: 36.7095, reg_loss: 17.0039, q_loss: 0.0062, acc: 0.8613, mse: 0.0073, val_loss: 0.3633, val_acc: 0.8547, val_auc: 0.7706, test_acc: 0.8675, test_auc: 0.7734
Iter: 53, recon_loss: 34.6972, ce_loss: 36.3780, reg_loss: 16.9944, q_loss: 0.0049, acc: 0.8613, mse: 0.0073, val_loss: 0.3654, val_acc: 0.8547, val_auc: 0.7747, test_acc: 0.8675, test_auc: 0.7495
Iter: 54, recon_loss: 34.6094, ce_loss: 36.6464, reg_loss: 16.9824, q_loss: 0.0050, acc: 0.8613, mse: 0.0073, val_loss: 0.3618, val_acc: 0.8547, val_auc: 0.7709, test_acc: 0.8675, test_auc: 0.7567
Iter: 55, recon_loss: 34.5212, ce_loss: 36.5765, reg_loss: 16.9573, q_loss: 0.0046, acc: 0.8613, mse: 0.0073, val_loss: 0.3703, val_acc: 0.8547, val_auc: 0.7691, test_acc: 0.8675, test_auc: 0.7583
Iter: 56, recon_loss: 34.4901, ce_loss: 36.7118, reg_loss: 16.9480, q_loss: 0.0044, acc: 0.8613, mse: 0.0073, val_loss: 0.3658, val_acc: 0.8547, val_auc: 0.7779, test_acc: 0.8675, test_auc: 0.7524
Iter: 57, recon_loss: 34.3466, ce_loss: 36.2509, reg_loss: 16.9293, q_loss: 0.0042, acc: 0.8613, mse: 0.0073, val_loss: 0.3651, val_acc: 0.8547, val_auc: 0.7680, test_acc: 0.8675, test_auc: 0.7615
Iter: 58, recon_loss: 34.2731, ce_loss: 36.2624, reg_loss: 16.8911, q_loss: 0.0069, acc: 0.8613, mse: 0.0073, val_loss: 0.3608, val_acc: 0.8547, val_auc: 0.7846, test_acc: 0.8675, test_auc: 0.7494
Iter: 59, recon_loss: 34.1738, ce_loss: 36.4722, reg_loss: 16.8406, q_loss: 0.0050, acc: 0.8613, mse: 0.0072, val_loss: 0.3629, val_acc: 0.8547, val_auc: 0.7836, test_acc: 0.8675, test_auc: 0.7513
Iter: 60, recon_loss: 34.1787, ce_loss: 36.2836, reg_loss: 16.8304, q_loss: 0.0040, acc: 0.8613, mse: 0.0072, val_loss: 0.3642, val_acc: 0.8547, val_auc: 0.7697, test_acc: 0.8675, test_auc: 0.7444
Iter: 61, recon_loss: 34.3785, ce_loss: 36.3100, reg_loss: 17.6830, q_loss: 14.6328, acc: 0.8613, mse: 0.0073, val_loss: 0.3627, val_acc: 0.8547, val_auc: 0.7795, test_acc: 0.8675, test_auc: 0.7540
Iter: 62, recon_loss: 34.5166, ce_loss: 36.6399, reg_loss: 17.1626, q_loss: 0.0000, acc: 0.8613, mse: 0.0073, val_loss: 0.3603, val_acc: 0.8547, val_auc: 0.7703, test_acc: 0.8675, test_auc: 0.7623
Iter: 63, recon_loss: 33.9395, ce_loss: 36.3062, reg_loss: 16.9695, q_loss: 0.0000, acc: 0.8613, mse: 0.0072, val_loss: 0.3573, val_acc: 0.8547, val_auc: 0.7741, test_acc: 0.8675, test_auc: 0.7665
Iter: 64, recon_loss: 33.9740, ce_loss: 36.4530, reg_loss: 16.8701, q_loss: 0.0000, acc: 0.8613, mse: 0.0072, val_loss: 0.3615, val_acc: 0.8547, val_auc: 0.7716, test_acc: 0.8675, test_auc: 0.7585
Iter: 65, recon_loss: 33.8807, ce_loss: 36.2314, reg_loss: 16.8814, q_loss: 0.0000, acc: 0.8613, mse: 0.0072, val_loss: 0.3595, val_acc: 0.8547, val_auc: 0.7853, test_acc: 0.8675, test_auc: 0.7589
Iter: 66, recon_loss: 33.8785, ce_loss: 36.2122, reg_loss: 17.0946, q_loss: 0.0000, acc: 0.8613, mse: 0.0072, val_loss: 0.3621, val_acc: 0.8547, val_auc: 0.7711, test_acc: 0.8675, test_auc: 0.7724
Iter: 67, recon_loss: 33.6929, ce_loss: 36.2273, reg_loss: 16.8610, q_loss: 0.0000, acc: 0.8613, mse: 0.0071, val_loss: 0.3653, val_acc: 0.8547, val_auc: 0.7657, test_acc: 0.8675, test_auc: 0.7765
Iter: 68, recon_loss: 33.5189, ce_loss: 36.0326, reg_loss: 16.8401, q_loss: 0.0000, acc: 0.8613, mse: 0.0071, val_loss: 0.3597, val_acc: 0.8547, val_auc: 0.7825, test_acc: 0.8675, test_auc: 0.7697
Iter: 69, recon_loss: 33.4091, ce_loss: 35.9328, reg_loss: 16.8747, q_loss: 0.0000, acc: 0.8613, mse: 0.0071, val_loss: 0.3618, val_acc: 0.8547, val_auc: 0.7730, test_acc: 0.8675, test_auc: 0.7721
Iter: 70, recon_loss: 33.3615, ce_loss: 36.1000, reg_loss: 16.8861, q_loss: 0.0000, acc: 0.8613, mse: 0.0071, val_loss: 0.3600, val_acc: 0.8547, val_auc: 0.7828, test_acc: 0.8675, test_auc: 0.7779
Iter: 71, recon_loss: 33.1440, ce_loss: 35.5655, reg_loss: 16.8773, q_loss: 0.0000, acc: 0.8613, mse: 0.0070, val_loss: 0.3585, val_acc: 0.8547, val_auc: 0.7836, test_acc: 0.8675, test_auc: 0.7690
Iter: 72, recon_loss: 33.1535, ce_loss: 35.9365, reg_loss: 16.8808, q_loss: 0.0000, acc: 0.8613, mse: 0.0070, val_loss: 0.3564, val_acc: 0.8547, val_auc: 0.7831, test_acc: 0.8675, test_auc: 0.7734
Iter: 73, recon_loss: 33.0691, ce_loss: 35.6719, reg_loss: 16.8920, q_loss: 0.0000, acc: 0.8613, mse: 0.0070, val_loss: 0.3592, val_acc: 0.8547, val_auc: 0.7796, test_acc: 0.8675, test_auc: 0.7747
Iter: 74, recon_loss: 32.9059, ce_loss: 35.6897, reg_loss: 16.8813, q_loss: 0.0000, acc: 0.8613, mse: 0.0070, val_loss: 0.3567, val_acc: 0.8547, val_auc: 0.7852, test_acc: 0.8675, test_auc: 0.7673
Iter: 75, recon_loss: 32.7618, ce_loss: 35.5968, reg_loss: 16.9015, q_loss: 0.0000, acc: 0.8613, mse: 0.0069, val_loss: 0.3525, val_acc: 0.8547, val_auc: 0.7920, test_acc: 0.8675, test_auc: 0.7697
Iter: 76, recon_loss: 32.6614, ce_loss: 35.5738, reg_loss: 16.8952, q_loss: 0.0000, acc: 0.8613, mse: 0.0069, val_loss: 0.3558, val_acc: 0.8547, val_auc: 0.7820, test_acc: 0.8675, test_auc: 0.7736
Iter: 77, recon_loss: 32.4804, ce_loss: 35.6634, reg_loss: 16.9117, q_loss: 0.0000, acc: 0.8613, mse: 0.0069, val_loss: 0.3517, val_acc: 0.8547, val_auc: 0.7939, test_acc: 0.8675, test_auc: 0.7798
Iter: 78, recon_loss: 32.5518, ce_loss: 35.4479, reg_loss: 16.9103, q_loss: 0.0000, acc: 0.8613, mse: 0.0069, val_loss: 0.3566, val_acc: 0.8547, val_auc: 0.7827, test_acc: 0.8675, test_auc: 0.7909
Iter: 79, recon_loss: 32.5377, ce_loss: 35.2649, reg_loss: 16.9264, q_loss: 0.0000, acc: 0.8613, mse: 0.0069, val_loss: 0.3528, val_acc: 0.8547, val_auc: 0.7874, test_acc: 0.8675, test_auc: 0.7821
Iter: 80, recon_loss: 32.3556, ce_loss: 35.3297, reg_loss: 16.9042, q_loss: 0.0000, acc: 0.8613, mse: 0.0069, val_loss: 0.3588, val_acc: 0.8547, val_auc: 0.7769, test_acc: 0.8675, test_auc: 0.7820
Iter: 81, recon_loss: 32.2364, ce_loss: 35.0367, reg_loss: 16.9259, q_loss: 0.0000, acc: 0.8613, mse: 0.0068, val_loss: 0.3509, val_acc: 0.8547, val_auc: 0.7939, test_acc: 0.8675, test_auc: 0.7854
Iter: 82, recon_loss: 32.1051, ce_loss: 34.9808, reg_loss: 16.9187, q_loss: 0.0001, acc: 0.8613, mse: 0.0068, val_loss: 0.3494, val_acc: 0.8547, val_auc: 0.7961, test_acc: 0.8675, test_auc: 0.7745
Iter: 83, recon_loss: 32.1476, ce_loss: 34.9360, reg_loss: 16.9257, q_loss: 0.0001, acc: 0.8613, mse: 0.0068, val_loss: 0.3504, val_acc: 0.8547, val_auc: 0.8076, test_acc: 0.8675, test_auc: 0.7830
Iter: 84, recon_loss: 32.1225, ce_loss: 35.1422, reg_loss: 16.9390, q_loss: 0.0001, acc: 0.8613, mse: 0.0068, val_loss: 0.3518, val_acc: 0.8547, val_auc: 0.7907, test_acc: 0.8675, test_auc: 0.7809
Iter: 85, recon_loss: 31.8996, ce_loss: 35.0332, reg_loss: 16.9283, q_loss: 0.0001, acc: 0.8613, mse: 0.0068, val_loss: 0.3503, val_acc: 0.8547, val_auc: 0.7920, test_acc: 0.8675, test_auc: 0.7781
Iter: 86, recon_loss: 31.7113, ce_loss: 34.9514, reg_loss: 16.9305, q_loss: 0.0001, acc: 0.8613, mse: 0.0067, val_loss: 0.3480, val_acc: 0.8547, val_auc: 0.7904, test_acc: 0.8675, test_auc: 0.7751
Iter: 87, recon_loss: 31.6226, ce_loss: 34.7866, reg_loss: 16.9440, q_loss: 0.0001, acc: 0.8613, mse: 0.0067, val_loss: 0.3494, val_acc: 0.8547, val_auc: 0.7929, test_acc: 0.8675, test_auc: 0.7856
"""

def parse_data(data_string):
    pattern = r"Iter: (\d+), recon_loss: ([\d.]+), ce_loss: ([\d.]+), reg_loss: ([\d.]+), q_loss: ([\d.]+), acc: ([\d.]+), mse: ([\d.]+), val_loss: ([\d.]+), val_acc: ([\d.]+), val_auc: ([\d.]+), test_acc: ([\d.]+), test_auc: ([\d.]+)"
    matches = re.findall(pattern, data_string)
    
    data = []
    for match in matches:
        iter_num, recon_loss, ce_loss, reg_loss, q_loss, acc, mse, val_loss, val_acc, val_auc, test_acc, test_auc = match
        data.append((int(iter_num), float(recon_loss), float(ce_loss), float(reg_loss), float(q_loss), float(acc), float(mse), float(val_loss), float(val_acc), float(val_auc), float(test_acc), float(test_auc)))
    return data

def calculate_stats(data):
    # val_loss 기준 상위 5개
    top_5_val_loss = sorted(data, key=lambda x: x[7])[:3]
    val_loss_test_auc = [x[11] for x in top_5_val_loss]
    
    # val_auc 기준 상위 5개
    top_5_val_auc = sorted(data, key=lambda x: x[9], reverse=True)[:3]
    val_auc_test_auc = [x[11] for x in top_5_val_auc]
    
    # 평균 및 표준편차 계산
    mean_val_loss_test_auc = np.mean(val_loss_test_auc)
    std_val_loss_test_auc = np.std(val_loss_test_auc)
    
    mean_val_auc_test_auc = np.mean(val_auc_test_auc)
    std_val_auc_test_auc = np.std(val_auc_test_auc)
    
    return (mean_val_loss_test_auc, std_val_loss_test_auc), (mean_val_auc_test_auc, std_val_auc_test_auc)

# 데이터 파싱
parsed_data = parse_data(data_string)

# 데이터 처리 및 통계 계산
(mean_val_loss_test_auc, std_val_loss_test_auc), (mean_val_auc_test_auc, std_val_auc_test_auc) = calculate_stats(parsed_data)

print(f"val_loss 기준 상위 5개 데이터의 test_auc 평균: {mean_val_loss_test_auc:.4f}, 표준편차: {std_val_loss_test_auc:.4f}")
print(f"val_auc 기준 상위 5개 데이터의 test_auc 평균: {mean_val_auc_test_auc:.4f}, 표준편차: {std_val_auc_test_auc:.4f}")
