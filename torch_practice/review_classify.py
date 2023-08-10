import review_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

star_ratings = review_preprocess.load_star_ratings()
reviews = review_preprocess.load_reviews()

#---------- vectorize

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

tfidf_vectorizer = TfidfTransformer()
X = tfidf_vectorizer.fit_transform(X)

#---------- train

import torch

model = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], 512),
    torch.nn.Linear(512, 512),
    torch.nn.Linear(512, 512),
    torch.nn.Linear(512, 2), #----- input갯수: X.shape[1], output갯수: 2 -----
)

criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1) # 할일) learning rate 바꿔보기

def train(x_target, y_target, model, criterion, optimizer, num_epochs):

    loss_history = [] 
    train_accuracy_history = []

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        y_pre = model(x_target) # Tensor여야한다, y_pre의 shape은??
        loss = criterion(y_pre, y_target) # Tensor여야한다
        loss.backward() 
        optimizer.step()

        loss_history.append(loss.item())

        # train 분류 정확도 확인
        train_accuracy = model(x_target).argmax(dim=1).__eq__(y_target).sum().item() / y_target.shape[0] * 100.0 #model이 예측에 성공한게 몇개인지 백분율로 나타낸다
        train_accuracy_history.append(train_accuracy)
      
    
    '''
    할일) 훈련이 다 끝난 모델을 대상으로 테스트를 수행한다
    
    y_test = model(x_test)
    '''

    return loss_history, train_accuracy_history

#x_target = torch.tensor(X, dtype=torch.long)


i = [[0, 1, 1],
     [2, 0, 2]]
v =  [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 3))
print(s)
print(s.to_dense())


'''loss_history, train_accuracy = train()
print("손실 추이: ")
for i in loss_history:
    print(i)
print("잘 분류한 정도를 백분율로 나타냄: ")
for i in train_accuracy:
    print(i)'''