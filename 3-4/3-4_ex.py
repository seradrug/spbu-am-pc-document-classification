from datasets import load_dataset 
import pandas as pd
from collections import Counter
import re
import nltk 
from sklearn.feature_extraction.text import CountVectorizer 
import string 
from datasets import load_dataset 
from nltk.downloader import Downloader 
from nltk.corpus import wordnet
from nltk.corpus import stopwords

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
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

ds = load_dataset("emotion")
df = load_dataset("SetFit/20_newsgroups")

categories = {
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.graphics",
    "comp.windows.x"
}

stop_words = set(stopwords.words("english"))

HEADER_RE = re.compile(
    r'^(from|subject|organization|lines|reply-to|nntp-posting-host|'
    r'distribution|keywords|summary|x-newsreader|xref|path|newsgroups):.*$',
    flags=re.IGNORECASE | re.MULTILINE)
FROM_ARTICLE_RE = re.compile(r'^from article .*$', flags=re.IGNORECASE | re.MULTILINE)
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
ANGLE_ID_RE = re.compile(r'<[^>]+>')    
QUOTE_RE = re.compile(r'(?m)^\s*>.*\n?') 

def clean_20ng(text: str):
    text = text.replace('\\n', '\n')

    text = QUOTE_RE.sub('', text)

    text = HEADER_RE.sub('', text)

    text = FROM_ARTICLE_RE.sub('', text)

    text = EMAIL_RE.sub(' ', text)
    text = URL_RE.sub(' ', text)
    text = ANGLE_ID_RE.sub(' ', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def keep(exp):
    return exp["text"] in categories

data1 = ds["train"].to_pandas()
data2 = df["train"].to_pandas()

data1 = data1.head(30)
data2 = data2.head(30)

data1 = data1["text"]
data2 = data2["text"]

def top_tokens(text, n=30):
   cnt = Counter()

   for t in text:
       cnt.update(t)
   
   return cnt.most_common(n)
      
clean_top = top_tokens([clean_20ng(t) for t in data2], 30)

def get_wordnet_pos(treebank_tag):
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
    text = clean_20ng(text).lower()
    text = text.translate(str.maketrans('','', string.punctuation))
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = nltk.WordNetLemmatizer()
    remove_pos = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}

    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags if pos not in remove_pos and token not in string.punctuation]

    clean = [
        token for token in lemmatized_tokens if token not in stop_words]

    return clean
print(top_tokens(data2, 30))
l = data2.apply(lemma)
print(top_tokens(l, 30))
print(l)
