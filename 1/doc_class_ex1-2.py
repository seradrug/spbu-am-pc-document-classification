import nltk
import pandas as pd
import string 
from datasets import load_dataset
from nltk.downloader import Downloader
from nltk.corpus import wordnet
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
dataset = dataset1["train"].to_pandas()
f10 = dataset.head(10)
f10_text = f10["text"]

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
                    
def stem(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stemmer = nltk.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def lemma(text):
    text = text.lower()
    text = text.translate(str.maketrans('','', string.punctuation))
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
    return lemmatized_tokens

s = f10_text.apply(stem)
l = f10_text.apply(lemma)
print(s, l, "\n")
