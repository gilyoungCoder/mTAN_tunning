import torch

# 예시 텐서 생성
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# 텐서 전체에서 최대 값을 찾기
max_value = torch.max(tensor)

# 최대 값 출력
print(max_value)
