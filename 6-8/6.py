import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd
from datasets import load_dataset 
from nltk.downloader import Downloader 
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string 
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = load_dataset("emotion")

#stop_words = set(stopwords.words("english"))

def tag(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemma(text):

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = nltk.WordNetLemmatizer()
    
    lemmatized_tokens = [lemmatizer.lemmatize(token, tag(pos)) for token, pos in
                         pos_tags if token not in string.punctuation]
    
    return " ".join(lemmatized_tokens)

train = df["train"].to_pandas()

X = train["text"]
y = train["label"]
X_0 = X.apply(lemma)

vec = TfidfVectorizer()
vec_lemma = TfidfVectorizer()
X_new = vec.fit_transform(X)
X_with_lemma = vec_lemma.fit_transform(X_0)

svd = TruncatedSVD(n_components=100)
svd_lemma = TruncatedSVD(n_components=100)
X_lsa = svd.fit_transform(X_new)
X_lsa_lemma = svd_lemma.fit_transform(X_with_lemma)

base = tree.DecisionTreeClassifier(max_depth=1)

#Здесь у меня адабуст для обработанных и необратботанных слов
clf = AdaBoostClassifier(
        estimator=base,
        n_estimators=100,
        random_state=0
)
clf = clf.fit(X_new, y)
clf_with_lemm0 = AdaBoostClassifier(
        estimator=base,
        n_estimators=100,
        random_state=0
)
clf_with_lemm0 = clf_with_lemm0.fit(X_with_lemma, y)

test = df["test"].to_pandas()

X_test = test["text"]
y_test = test["label"]

X_test0 = X_test.apply(lemma)
X_test_new = vec.transform(X_test)
X_test_with_lemma = vec_lemma.transform(X_test0)

X_test_lsa = svd.transform(X_test_new)
X_test_lsa_lemma = svd_lemma.transform(X_test_with_lemma)

pred = clf.predict(X_test_new)
pred_with_l0 = clf_with_lemm0.predict(X_test_with_lemma)

clf1 = GradientBoostingClassifier(n_estimators=10)
clf1 = clf1.fit(X_new, y)
clf_with_lemm1 = GradientBoostingClassifier(n_estimators=10)
clf_with_lemm1 = clf_with_lemm1.fit(X_with_lemma, y)

pred1 = clf1.predict(X_test_new)
pred_with_l1 = clf_with_lemm1.predict(X_test_with_lemma)

#Здесь добавляю lsa
clf_lsa = AdaBoostClassifier(estimator=base, n_estimators=100, random_state=0)
clf_lsa.fit(X_lsa, y)

clf_lsa_lemma = AdaBoostClassifier(estimator=base, n_estimators=100, random_state=0)
clf_lsa_lemma.fit(X_lsa_lemma, y)

pred_lsa = clf_lsa.predict(X_test_lsa)
pred_lsa_lemma = clf_lsa_lemma.predict(X_test_lsa_lemma)

results = []

results.append({
    "preprocessing": "lsa_no_lemm",
    "model": "AdaBoost",
    "f1_average": "micro",
    "f1_score": f1_score(y_test, pred_lsa, average="micro")
})

results.append({
    "preprocessing": "lsa_with_lemm",
    "model": "AdaBoost",
    "f1_average": "micro",
    "f1_score": f1_score(y_test, pred_lsa_lemma, average="micro")
})

results.append({
    "preprocessing": "no_lemm",
    "model": "AdaBoost",
    "f1_average": "weighted",
    "f1_score": f1_score(y_test, pred, average="weighted")
})
results.append({
    "preprocessing": "no_lemm",
    "model": "AdaBoost",
    "f1_average": "macro",
    "f1_score": f1_score(y_test, pred, average="macro")

})
results.append({
    "preprocessing": "no_lemm",
    "model": "AdaBoost",
    "f1_average": "micro",
    "f1_score": f1_score(y_test, pred, average="micro")

})
results.append({
    "preprocessing": "with_lemm",
    "model": "AdaBoost",
    "f1_average": "weighted",
    "f1_score": f1_score(y_test, pred_with_l0, average="weighted")
})
results.append({
    "preprocessing": "with_lemm",
    "model": "AdaBoost",
    "f1_average": "macro",
    "f1_score": f1_score(y_test, pred_with_l0, average="macro")
})
results.append({
    "preprocessing": "with_lemm",
    "model": "AdaBoost",
    "f1_average": "micro",
    "f1_score": f1_score(y_test, pred_with_l0, average="micro")
})

results.append({
    "preprocessing": "no_lemm",
    "model": "GradientBoost",
    "f1_average": "weighted",
    "f1_score": f1_score(y_test, pred1, average="weighted")
})
results.append({
    "preprocessing": "no_lemm",
    "model": "GradientBoost",
    "f1_average": "macro",
    "f1_score": f1_score(y_test, pred1, average="macro")
})
results.append({
    "preprocessing": "no_lemm",
    "model": "GradientBoost",
    "f1_average": "micro",
    "f1_score": f1_score(y_test, pred1, average="micro")
})
results.append({
    "preprocessing": "with_lemm",
    "model": "GradientBoost",
    "f1_average": "weighted",
    "f1_score": f1_score(y_test, pred_with_l1, average="weighted")
})
results.append({
    "preprocessing": "with_lemm",
    "model": "GradientBoost",
    "f1_average": "macro",
    "f1_score": f1_score(y_test, pred_with_l1, average="macro")
})
results.append({
    "preprocessing": "with_lemm",
    "model": "GradientBoost",
    "f1_average": "micro",
    "f1_score": f1_score(y_test, pred_with_l1, average="micro")
})

df_results = pd.DataFrame(results)
print(df_results)










