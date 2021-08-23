import pandas as pd
import spacy
import re
from collections import Counter    
nlp = spacy.load("en_core_web_sm")
STOP_WORDS=nlp.Defaults.stop_words
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
STOP_WORDS.add('account')
STOP_WORDS.add('accounts')
STOP_WORDS.add('cash')
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
STOP_WORDS.add('fund')
STOP_WORDS.add('fidelity')
# df=pd.read_csv('topic1_messages.csv')
# comments_text=""
# for i in df['Original_Message']:
#     comments_text+=i
# comments_text=comments_text.lower()
# comments_text

def process(text):
    # comments_text=""
    # for i in df['Original_Message']:
    #     comments_text+=i
    # text=comments_text.lower()
    
    
    cleaned_text = ' '.join([word for word in text.split() if word not in (STOP_WORDS)])
    cleaned_text = re.sub(r'\.\s+', ".", cleaned_text)
    #li = re.split("\.+", cleaned_text)
    shortword = re.compile(r'\W*\b\w{1,2}\b')
    x=shortword.sub('', cleaned_text)
    li=re.split("\.+", x)
    clean_list=[]
    for i in li:
        text_clean = ' '.join([word for word in i.split() if word not in (STOP_WORDS)])
        clean_list.append(text_clean)
    for a in clean_list:
        if len(a.split(' '))<2:
            clean_list.remove(a)
        else:
            pass
    final_text='.'.join(clean_list)
    final_text = re.sub(r'\.+', ".", final_text)
    words =final_text.split(' ')
    result = dict(Counter(words))
    de={}
    l=[]
    for key,value in result.items():
        l.append({'text':key,'value':value})

    de["data"]=l  
    return de