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
