#!/usr/bin/env python
# coding: utf-8

import re
import sys
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from nltk.sentiment.vader import SentimentIntensityAnalyzer

train = str(sys.argv[1])
test = str(sys.argv[2])

tsv_read = pd.read_csv(train, sep='\t',header=None)
tsv_read.columns = ['instance_number', 'tweet_text', 'sentiment']

test_db = pd.read_csv(test, sep='\t',header=None)
test_db.columns = ['instance_number', 'tweet_text', 'sentiment']

regex_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

train_list=[]
test_list=[]

for i in range(len(tsv_read)):
    train_list.append(tsv_read.iloc[i]['tweet_text'])
    
for i in range(len(test_db)):
    test_list.append(test_db.iloc[i]['tweet_text'])

def my_tokenizer(text):

    text = re.sub(regex_url, '', text)
    text = re.sub('[^a-zA-Z0-9\@\#\_\$\% ]', '', text)
    text = re.sub(r'\b\w{0,1}\b', '', text)
    text = ' '.join( [w for w in text.split() if len(w)>1] )
    Z = text.split()
   
    return Z


cv = CountVectorizer(tokenizer=my_tokenizer, lowercase = False)
X_train_bag_of_words = cv.fit_transform(train_list)

def predict_and_test(model, X_test_bag_of_words):
    predicted_y = model.predict(X_test_bag_of_words)
    
    final = {test_db.iloc[i]['instance_number']:predicted_y[i] for i in range(len(predicted_y))}
    for i,j in final.items():
        print(f"{i} {j}")


X_test_bag_of_words = cv.transform(test_list)

y = tsv_read['sentiment']
clf = BernoulliNB()
model = clf.fit(X_train_bag_of_words, y)
predict_and_test(model, X_test_bag_of_words)





