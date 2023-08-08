import torch

y_target = torch.tensor([2,1], dtype=torch.long) # 샘플이 두개 -> y_target을 나타낸다. shape은 (2000,) 이런식의 일차원 텐서
y_pred = torch.tensor([[0.0001, 0.0008, 5.2], [0.0001, 5.4, 0.0001]], dtype=torch.float) # 네트워크 출력, target의 index의 확률이 압도적으로 높아야함
criterion = torch.nn.CrossEntropyLoss()  

loss = criterion(y_pred, y_target)

print("손실: ", loss.item())