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
X_new = vec.fit_transform(X)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_new, y)

test = df["test"].to_pandas()

X_test = test["text"]
y_test = test["label"]

X_test0 = X_test.apply(lemma)
X_test_new = vec.transform(X_test)

pred = clf.predict(X_test_new)
print(f1_score(y_test, pred, average="weighted"))















