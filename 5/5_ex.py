from sklearn import tree
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

df = load_dataset("emotion")

train = df["train"]

X = train["text"]
y = train["label"]

vectorizer = CountVectorizer()
vec2 = TfidfVectorizer()
X0 = vectorizer.fit_transform(X)
X1 = vec2.fit_transform(X)

clf = tree.DecisionTreeClassifier()
clf0 = clf.fit(X0, y)
clf1 = clf.fit(X1, y) 

test = df["test"]

X_test = test["text"]
y_test = test["label"]

X_test = vectorizer.transform(X_test)
pred = clf0.predict(X_test)
pred0 = clf1.predict(X_test)

print(accuracy_score(y_test, pred)) 
print(f1_score(y_test, pred, average="macro"))

print(accuracy_score(y_test, pred0))
print(f1_score(y_test, pred, average="weighted"))





