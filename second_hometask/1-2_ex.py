import nltk 
import pandas as pd 
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

dataset1 = load_dataset("emotion")
dataset2 = load_dataset("SetFit/20_newsgroups")
categories = {
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.graphics",
    "comp.windows.x"
}

def keep(example):
    return example["label_text"] in categories

dataset = dataset1["train"].to_pandas() 
dataset0 = dataset2["train"].filter(keep).to_pandas()
f11 = dataset0.head(10)
f11_text = f11["text"]
f10 = dataset.head(10)
f10_text = f10["text"]
stop_words = set(stopwords.words("english"))

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
    text = text.lower()
    text = text.translate(str.maketrans('','', string.punctuation))
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = nltk.WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags if token not in string.punctuation]

    clean = [
        token for token in lemmatized_tokens if token not in stop_words]

    return clean

l = f10_text.apply(lemma)
l1 = f11_text.apply(lemma)
print(l, l1)
