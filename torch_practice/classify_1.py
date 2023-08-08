import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

samples, labels = datasets.make_blobs(n_samples=2000, centers=3, random_state=0) #샘플 갯수, 분류 갯수


#데이터셋의 정체를 알아보자
print(samples.shape, samples.dtype) # (2000, 2) float64 2차원 평면위의 두 점(좌표)
print(labels.shape, labels.dtype) # (2000,) int32


#훈련과정
import torch
model = torch.nn.Sequential(
    torch.nn.Linear(2, 512),
    torch.nn.Linear(512, 512),
    torch.nn.Linear(512, 512),
    torch.nn.Linear(512, 3), #----- 이게 문제였네,, 점 2개에서 3개의 구역으로 분류 -----
)
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)


#실제 분류 훈련
def train(x_target, y_target, model, criterion, optimizer, num_epochs):

    loss_history = [] # epoch의 갯수와 동일함... 점점 loss가 감소하려나
    train_accuracy_history = []
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        y_pre = model(x_target) # Tensor여야한다, y_pre는 (2000, 3)
        loss = criterion(y_pre, y_target) # Tensor여야한다
        loss.backward() 
        optimizer.step()

        loss_history.append(loss.item())

        # train 분류 정확도 확인
        train_accuracy = model(x_target).argmax(dim=1).__eq__(y_target).sum().item() / y_target.shape[0] * 100.0 #2000개 중에 model이 예측에 성공한게 몇개인지 백분율로 나타낸다
        train_accuracy_history.append(train_accuracy)
      
    
    '''
    할일) 훈련이 다 끝난 모델을 대상으로 테스트를 수행한다
    '''
    '''
    y_test = model(x_test)
    '''

    return loss_history, train_accuracy_history


#메인함수
loss_history, train_accuracy = train(torch.tensor(samples, dtype=torch.float), torch.tensor(labels,dtype=torch.long), model, criterion, optimizer, 40)
print("손실 추이: ")
for i in loss_history:
    print(i)
print("잘 분류한 정도를 백분율로 나타냄: ")
for i in train_accuracy:
    print(i)


