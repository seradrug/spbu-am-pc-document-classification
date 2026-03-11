import nltk 
from datasets import load_dataset
import pandas as pd
import string
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
nltk.download("punkt_tab")

df1 = load_dataset("emotion")
df = df1["train"].to_pandas()
df_text1 = df.head(10)["text"]
df_text = "Hi, my name is Sergey, I'm from Russia, Tyumen. I love my home-sity so much!"

token = nltk.word_tokenize(df_text)
print(token) # до удаления знаков препинания

clean_text = df_text.translate(str.maketrans('', '', string.punctuation))
token1 = nltk.word_tokenize(clean_text)
print(token1) # это мы получили токенизацию с учётом того, что удалили все знаки препинания до самой процедуры

clear = (lambda x:[t for t in x if t.isalpha()])(token)
print(clear) # удалили все знаки препинания после токенизации

