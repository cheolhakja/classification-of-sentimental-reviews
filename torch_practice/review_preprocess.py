from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

corpus = []

with open("parsed_review_data.pkl", "rb") as f:
    try:
        while True:
            value = pickle.load(f)
            li = []
            for i in value:
                tmp = i.split("/")
                li.append(tmp[0])
            str = ' '.join(li)
            corpus.append(str)
            
    except EOFError:
        print("파일 읽기 종료")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

tfidf_vectorizer = TfidfTransformer()
X = tfidf_vectorizer.fit_transform(X)
print(X.shape)
print(type(X))
print(X[0])

print("----------")

print(X[8,:])

print("----------")

stars = None
with open("star_rate.pkl", "rb") as f:
    try:
        while True:
            stars = pickle.load(f) #value는 list이다. list객체를 그대로 직렬화했음
            
            
    except EOFError:
        print("파일 읽기 종료")

print(stars)
print("별점 갯수: ", len(stars))

'''
이제 

corpus로 만든 X(행렬)와  stars(list)로 모델을 학습한다
'''

def load_star_ratings() -> list:
    with open("star_rate.pkl", "rb") as f:
        try:
            while True:
                stars = pickle.load(f) #value는 list이다. list객체를 그대로 직렬화했음
                
        except EOFError:
            print("파일 읽기 종료")

    return stars

def load_reviews():
    pass