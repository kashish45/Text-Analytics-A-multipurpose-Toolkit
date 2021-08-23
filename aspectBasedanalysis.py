import nltk
from nltk import word_tokenize
import string
from nltk.grammar import pcfg_demo
from nltk.stem import WordNetLemmatizer
import numpy as np
import math
wordnet_lemmatizer = WordNetLemmatizer()
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import yake



import spacy
nlp = spacy.load("en_core_web_sm")

STOP_WORDS=nlp.Defaults.stop_words
#print ('Before adding custom stop words=',len(STOP_WORDS))
mystopwords = ['I', 'x', '', ' ','To', 'Mr', 'Please','yet','want', 'kind regards', 'didn', 'regard', 'kind', 'Kind','ve','cassie', 'thanks', 'tell','take','Kind regards' 'will', 'xA', 'have', 'Dear', 'please', 'hello', 'Hello', 'confirm','to', 'dear', 'i', 'XA', 'Xa', 'i', 'From', 'hi', 'Hi','from', 'sincerely', 'Thank', 'regarding', 'Subject', 'subject', 'date', 'Date', 'Thank', 'thank', 'yours', 'Yours', 'told', 'dealing']
for i in mystopwords :
    STOP_WORDS.add(i)
    
STOP_WORDS.add('message')
STOP_WORDS.add('thanks')
STOP_WORDS.add('dear')
STOP_WORDS.add('thank')
STOP_WORDS.add('sincerely')

STOP_WORDS.add('customer')
STOP_WORDS.add('relations')
STOP_WORDS.add('gmail')
STOP_WORDS.add('hotmail')
STOP_WORDS.add('i')

STOP_WORDS.add('wishes')
STOP_WORDS.add('regards')
STOP_WORDS.add('like')
STOP_WORDS.add('hi')

STOP_WORDS.add('morning')
STOP_WORDS.add('good')

STOP_WORDS.add('hello')
STOP_WORDS.add('message')
STOP_WORDS.add('thanks')
STOP_WORDS.add('dear')
STOP_WORDS.add('thank')
STOP_WORDS.add('sincerely')
STOP_WORDS.add('fidelity')
STOP_WORDS.add('customer')
STOP_WORDS.add('relations')
STOP_WORDS.add('gmail')
STOP_WORDS.add('hotmail')

STOP_WORDS.add('hello')
STOP_WORDS.add('kind')
STOP_WORDS.add('best')
STOP_WORDS.add('gilford')
STOP_WORDS.add('s')
STOP_WORDS.add('let')
STOP_WORDS.add('asap')
STOP_WORDS.add('getting')
STOP_WORDS.add('way')
STOP_WORDS.add('use')
STOP_WORDS.add('dec')
STOP_WORDS.add('day')
STOP_WORDS.add('don')
STOP_WORDS.add('wanted')
STOP_WORDS.add('way')
STOP_WORDS.add('jan')
STOP_WORDS.add('mr')
STOP_WORDS.add('tell')


def newKeys(text):
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 18
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None,stopwords=STOP_WORDS)
    keywords = custom_kw_extractor.extract_keywords(text.lower())
    keys=[]
    for kw in keywords:
        keys.append(kw[0])
    return keys


analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score
    
def percentage(part,whole):
    return 100 * float(part)/float(whole)
#df=pd.read_csv('topic1_messages.csv')
def pos_neg_key_count(data):
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
    comments_text = ' '.join(comment_list).lower()
    positiveComments = ' '.join(positive_list).lower()
    negativeComments = ' '.join(negative_list).lower()
    key_features=newKeys(comments_text)
    count_pos=[]
    count_neg=[]
    for keyword in key_features:
        count_pos.append(positiveComments.count(keyword))
        count_neg.append(negativeComments.count(keyword))
    return key_features,count_pos,count_neg