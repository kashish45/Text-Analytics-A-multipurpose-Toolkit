import spacy
from collections import Counter
from string import punctuation
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim.corpora  as corpora
nlp = spacy.load('en_core_web_sm')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu, levene, shapiro
from statsmodels.stats.power import ttest_power
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm
from   statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import zscore
from sklearn import svm
from platform import python_version
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
from wordcloud import WordCloud
import glob
from collections import Counter

STOP_WORDS=nlp.Defaults.stop_words
# print ('Before adding custom stop words=',len(STOP_WORDS))
mystopwords = ['I', 'x', '', ' ','To', 'Mr', 'Please','yet','want', 'kind regards', 'didn', 'regard', 'kind', 'Kind','ve','cassie', 'thanks', 'tell','take','Kind regards' 'will', 'xA', 'have', 'Dear', 'please', 'hello', 'Hello', 'confirm','to', 'dear', 'i', 'XA', 'Xa', 'i', 'From', 'hi', 'Hi','from', 'sincerely', 'Thank', 'regarding', 'Subject', 'subject', 'date', 'Date', 'Thank', 'thank', 'yours', 'Yours', 'told', 'dealing']
for i in mystopwords :
    STOP_WORDS.add(i)
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
STOP_WORDS.add('i')
STOP_WORDS.add('xa')
STOP_WORDS.add('wishes')
STOP_WORDS.add('regards')
STOP_WORDS.add('like')
STOP_WORDS.add('hi')
STOP_WORDS.add('xa')
STOP_WORDS.add('morning')
STOP_WORDS.add('good')
STOP_WORDS.add('th')
STOP_WORDS.add('x')
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
STOP_WORDS.add('wishes')
STOP_WORDS.add('regards')
STOP_WORDS.add('like')
STOP_WORDS.add('hi')
STOP_WORDS.add('xa')
STOP_WORDS.add('morning')
STOP_WORDS.add('good')
STOP_WORDS.add('th')
STOP_WORDS.add('x')
#STOP_WORDS.add('account')
#STOP_WORDS.add('accounts')
#STOP_WORDS.add('cash')
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
STOP_WORDS.add('acc')
STOP_WORDS.add('rathbone')
STOP_WORDS.add('acc')
STOP_WORDS.add('set')
STOP_WORDS.add('james')
#STOP_WORDS.add('fund')
STOP_WORDS.add('fidelity')


def read_messages(filename,challenges):
    file1 = open(filename, 'r', encoding = 'utf-8') 
    raw_lines = file1.readlines() 
    for line in raw_lines:
        challenges.append(line.strip())
    return challenges


def find_top_unigrams(tokenized_sentences_series, top_number):
  token=[]
  token_freq=[]
  for sentence in tokenized_sentences_series:
    token=token+sentence
  return(Counter(token).most_common(top_number))

def find_all_unigrams(tokenized_sentences_series):
  token=[]
  token_freq=[]
  for sentence in tokenized_sentences_series:
    token=token+sentence
  return(token)


def create_challenges(df):
    challenges=[]
    for i in df['Original_Message']:
        challenges.append(i)
    
    data = pd.DataFrame(columns=['Comments'],data=(challenges))
    count=0
    for idx,row in data.iterrows():    
        message = row['Comments']
    
        if 'Fidelity Message' in message:
            client_msg = message[0:message.index('Fidelity Message')]
        else:
            client_msg = message
        
        data.at[idx,'Comments']=client_msg
    count=0
    for idx,row in data.iterrows():    
        message = row['Comments']
        data.at[idx,'Comments']=message

    data=data.drop(data[data['Comments']==''].index)


    data['Original_Comment']=data['Comments']
    data['Comments'].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=[" "," "], regex=True, inplace=True)
    spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–","-"]
    for char in spec_chars:
        data['Comments'] = data['Comments'].apply(lambda x:x.replace(char, ' '))
    #Extra spaces
    data['Comments'].replace('\s+', ' ', regex=True, inplace=True)
#Remove non ASCII words
    data['Comments']=data['Comments'].apply(lambda x:x.encode("ascii","ignore").decode())

#Tokenizing
    data['Comments']=data['Comments'].apply(lambda x:x.lower().split())

#Remove stopwords
    data['Cleaned_Comments']=data['Comments'].apply(lambda x:[item for item in x if item not in STOP_WORDS])
    return data[1:]



def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    
    
    coherence_values = []
    model_list = []
    
    model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                                num_topics=4, 
                                                id2word=dictionary,
                                                random_state=100,
                                                chunksize=100,
                                                passes=6,
                                                alpha='auto')
    model_list.append(model)
    coherencemodel = gensim.models.ldamodel.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def get_topic_name(topwords):
    tokens = nltk.tokenize.word_tokenize(topwords)
    tagged_tokens = nltk.pos_tag(tokens)
    nouns_and_adjs = [token[0] for token in tagged_tokens if token[1] in ['NN', 'JJ']]
    frequency = nltk.FreqDist(nouns_and_adjs)
    return frequency.most_common(2)[0][0]+"-"+frequency.most_common(2)[1][0] #pair top words as title


def format_topics_sentences(ldamodel, corpus ,df):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topic_label = get_topic_name(topic_keywords)
        
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords, topic_label]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Topic_Label']

    return(sent_topics_df)

def computation(df):
    data=create_challenges(df)
    id2word = corpora.Dictionary(data['Cleaned_Comments'])
    texts =  data['Cleaned_Comments']
    corpus = [id2word.doc2bow(text) for text in texts]
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, 
                                                        corpus=corpus, 
                                                        texts=texts, 
                                                        start=2, 
                                                        limit=8, 
                                                        step=1)
    cvList = []
    topicNumber = []
    limit=8; start=2; step=1
    x = range(start, limit, step)
    for m, cv in zip(x, coherence_values):
        cvList.append(cv)
        topicNumber.append(m)

    optimal_model = model_list[cvList.index(max(cvList))]

    model_topics = optimal_model.show_topics(formatted=False)
    df_topic_sents_keywords = format_topics_sentences(optimal_model, corpus,data)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Topic_Label']
    df_dominant_topic['Text']=data['Cleaned_Comments'].tolist()
    df_dominant_topic['Original_Message']=data['Original_Comment'].tolist()
    return df_dominant_topic