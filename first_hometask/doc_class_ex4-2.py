import nltk 
from datasets import load_dataset
import pandas as pd
import string
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
nltk.download("punkt_tab")

df1 = load_dataset("SetFit/20_newsgroups")
categories = {
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.graphics",
    "comp.windows.x"
}

def keep(example):
    return example["label_text"] in categories

df = df1["train"].filter(keep).to_pandas()
df_text1 = df.head(10)["text"]
df_text = "Hi, my name is Sergey, I'm from Russia, Tyumen. I love my home-sity so much!"

token = df_text1.apply(nltk.word_tokenize)
print(token) # до удаления знаков препинания

clean_text = df_text1.str.translate(str.maketrans('', '', string.punctuation))
token1 = clean_text.apply(nltk.word_tokenize)
print(token1) # это мы получили токенизацию с учётом того, что удалили все знаки препинания до самой процедуры

clear = token.apply(lambda x:[t for t in x if t.isalpha()])
print(clear) # удалили все знаки препинания после токенизации

