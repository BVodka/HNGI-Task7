#!/usr/bin/env python
# coding: utf-8

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pylint.lint import Run
import string
import re
import pandas as pd
import nltk

nltk.download("stopwords")

df=pd.read_csv(r'C:\Users\okunato oluwanifemi\Desktop\New folder (2)\New folder\saved assignments\2_count.csv')
df.head()

#soring df by productid

df.sort_values(["ProductId"], axis=0, ascending=True, inplace=True)

#we want to know how many positive and negative reviews we have
#we import the first sql table we created
FILTER = pd.read_csv(r'C:\Users\okunato oluwanifemi\Desktop\New folder (2)\New folder\saved assignments\1_filter.csv')
FILTER.head()

#we add a column which displays which is positive and negative
FILTER["Result"] = FILTER["Score"].apply(lambda score: "positive" if score > 3 else "negative")
FILTER.head(5)

#total negative and positive reviews
CONN = FILTER["Result"].value_counts()

#Remove HTML tags
TAG_RE = re.compile(r'<[^>]+>$[...] ')

def remove_tags(text):
    return TAG_RE.sub('', text)

df['Text']=df["Text"].apply(remove_tags)

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df["Text"] = df['Text'].apply(remove_punctuations)

df['Non Alphanumeric'] = df['Text'].str.count(r'[^a-zA-Z0-9 ]')

#check to see if the length of the word is greater than 2

df.dropna(inplace=True)

df['Text_len'] = df['Text'].astype(str).str.len()

#convert words o lower case

LOWER = df['Text'].str.lower()
LOWER.head()

#now we have to remove stopwords

df['Text'] = df['Text'].str.lower().str.split() 

STOP = stopwords.words('english')
df['Text'] = df['Text'].apply(lambda x: [item for item in x if item not in STOP])

ENGLISH_STEMMER = SnowballStemmer('english', ignore_stopwords=True)
df['stemmed'] = df['Text'].apply(lambda x: [ENGLISH_STEMMER.stem(y) for y in x])

df.head()

RESULTS = Run(['task 7.py'], do_exit=False)

# `exit` is deprecated, use `do_exit` instead
print(RESULTS.linter.stats['global_note'])
