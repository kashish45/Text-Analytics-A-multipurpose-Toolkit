import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score
    
def percentage(part,whole):
     return 100 * float(part)/float(whole)

    

def get_comments(data):
    positive = 0
    negative = 0
    compound_list = []
    comment_list = []
    negative_list = []
    positive_list = []
    sentiment_list = []
    pos_score = []
    neg_score = []
    noOfComments = len(data['Original_Message'])
    for comment in data['Original_Message']:
        comment_list.append(comment)
        score = sentiment_analyzer_scores(comment)
        neg = score['neg']
    #neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        pos_score.append(pos)
        neg_score.append(neg) 
        compound_list.append(comp)
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
    comment_list = pd.DataFrame(comment_list)
#neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)
    sentiment_list = pd.DataFrame(sentiment_list)
    pos_score = pd.DataFrame(pos_score)
    neg_score = pd.DataFrame(neg_score)
    senti_output = pd.concat([comment_list, sentiment_list, pos_score, neg_score], axis=1)
    senti_output.columns = ['Review', 'Sentiment', 'Positive Score', 'Negative Score']
    Pos_Comments=senti_output[senti_output['Sentiment']=='POSITIVE'].head(10)
    Neg_Comments=senti_output[senti_output['Sentiment']=='NEGATIVE'].head(10)
    pos_com=[]
    neg_com=[]
    for i in Pos_Comments['Review']:
        pos_com.append(i)
    for j in Neg_Comments['Review']:
        neg_com.append(j)
    return pos_com,neg_com