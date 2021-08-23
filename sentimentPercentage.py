import nltk
from nltk import word_tokenize
import string
from nltk.stem import WordNetLemmatizer
import numpy as np
import math
wordnet_lemmatizer = WordNetLemmatizer()
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
import spacy
nlp = spacy.load("en_core_web_sm")

# def read_data(filename):
#     data=pd.read_csv(filename)
#     return data


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score
    
def percentage(part,whole):
    return 100 * float(part)/float(whole)


def posnegpercentage(data):
    # df['Original_Message']=df['Original_Message'].to_frame()
    # df.columns=['Original_Message']
    positive = 0
    negative = 0
    neutral = 0
    comment_list = []
    negative_list = []
    positive_list = []
    sentiment_list = []
    pos_score = []
    neg_score = []
    noOfComments = len(data)
    for comment in data:
        comment_list.append(comment)
        score = sentiment_analyzer_scores(comment)
        neg = score['neg']
        pos = score['pos']
        comp = score['compound']
        pos_score.append(pos)
        neg_score.append(neg)
        if pos > neg:
            sentiment_list.append('POSITIVE')
            positive_list.append(comment)
            positive += 1
        
        else: 
            sentiment_list.append('NEGATIVE')
            negative_list.append(comment)
            negative += 1

    positive = percentage(positive, noOfComments)
    negative = percentage(negative, noOfComments)
    return positive,negative