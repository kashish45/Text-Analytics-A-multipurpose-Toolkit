import imp
from operator import pos
from typing_extensions import final
import uvicorn
import docx
from Key_Summarizers import keys_summarize
from fastapi import FastAPI 
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from extractive_summarizer import article_summarize
#from bart_summarizer import bart_model
from key_points import generate_summary
#from bart_summarizer import t5_summarize
# from keywords import key_words
from topic_modelling import computation
from pos_neg_comments import get_comments
from KeyWordds import Key_Words_Project
from WordCountCloud import process
from io import BytesIO
from pydantic import BaseModel
import PyPDF2
import io
import os
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
import time
from sentimentPercent import posnegpercentage
from AspectBasedSentiment import pos_neg_key_count
from fastapi import FastAPI, Request, File, UploadFile
import io

import numpy as np
from New_Keywords import newKeys

app = FastAPI(
    title="Text Summarization API",
    description="A simple API that use NLP Textranking algorithms to extract important sentences from the text",
    version="0.1",
)

@app.post('/api/v1/summarizer_Upload')
async def summarizer_upload(file: UploadFile=File(...)):
    if file.filename.endswith('.txt'):
        contents=await file.read()
        contents=contents.replace('\n','')
        contents=contents.replace('\r','')
        return {"Contents":contents}
    
    
    elif file.filename.endswith('.pdf'):
        file_content = await file.read()
        
        file_content_io = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfFileReader(file_content_io)
        count=pdf_reader.numPages
        page_info="`"
        for i in range(0,count):
            page_info+= pdf_reader.getPage(i).extractText()
        page_info=page_info.replace('\n','')
        page_info=page_info.replace('\r','')
        return {"Info":page_info}

    elif file.filename.endswith('.docx'):
        f_content=await file.read()
        f_content_io=io.BytesIO(f_content)
        document = docx.Document(f_content_io)
        docText = '\n\n'.join(
        paragraph.text for paragraph in document.paragraphs)
        docText=docText.replace('\n','')
        docText=docText.replace('\r','')
        return {'Para':docText}


@app.post("/api/v2/theme_extraction_upload/word_cloud")
async def theme_extraction_upload_word_cloud(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(BytesIO(file.file.read()))
        df_comments = df['Original_Message']
        comments=df_comments.to_json(orient='records')
        comments=eval(comments)
    text=""
    for i in comments:
        text+=i
    text=text.lower()
    wc=process(text)
    # keyss,pos,neg= pos_neg_key_count(comments)
    # pos_per,neg_per= posnegpercentage(comments)
    return {"word_cloud":wc}

@app.post("/api/v2/theme_extraction_upload/aspect_sentiment")
async def aspect_sentiment(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(BytesIO(file.file.read()))
        df_comments = df['Original_Message']
        comments=df_comments.to_json(orient='records')
        comments=eval(comments)
    # text=""
    # for i in comments:
    #     text+=i
    # text=text.lower()
    # wc=process(text)
    keyss,pos,neg= pos_neg_key_count(comments)
    # pos_per,neg_per= posnegpercentage(comments)
    return {"Keys":keyss,"Positive_Count":pos,"Negative_Count":neg}

@app.post("/api/v2/theme_extraction_upload/sentiment_percent")
async def theme_extraction_upload_sentiment_percent(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(BytesIO(file.file.read()))
        df_comments = df['Original_Message']
        comments=df_comments.to_json(orient='records')
        comments=eval(comments)
    # text=""
    # for i in comments:
    #     text+=i
    # text=text.lower()
    # wc=process(text)
    # keyss,pos,neg= pos_neg_key_count(comments)
    pos_per,neg_per= posnegpercentage(comments)
    return {"Positive_Percentage":pos_per,"Negative_Percentage":neg_per}




@app.get("/summarize")
async def summarize_text(text:str):
    if(len(text.split(' '))<1000):
        final_summary=article_summarize(text)
        #final_summary=t5_summarize(str(final_summary))
    else:
        final_summary=article_summarize(text)
        #final_summary=bart_model(str(final_summary))
    result={"summary":final_summary}
    return result

@app.get("keypoints")
async def keyPoints(text:str):
    ls=[]
    keyP=generate_summary(text,4)
    ls.append(keyP)
    res={"Key Points":ls}
    return res

@app.get("/keywords")
async def keyWords(text:str):
    
    ress={"Keywords":newKeys(text)}
    return ress


def create_topic_files(df):
    topic_1=df[df['Dominant_Topic']==0.0]
    topic_2=df[df['Dominant_Topic']==1.0]
    topic_3=df[df['Dominant_Topic']==2.0]
    topic_4=df[df['Dominant_Topic']==3.0]
    return topic_1,topic_2,topic_3,topic_4

@app.post("/api/v3/theme_extraction")
async def topic_modelling(file: UploadFile = File(...)):
    if file.filename.endswith('.csv'):
        # df = pd.read_csv(BytesIO(file.file.read()))
        # df_comments = df['Original_Message']
        # comments=df_comments.to_json(orient='records')
        # comments=eval(comments)
        topics=pd.read_csv(BytesIO(file.file.read()))
        com=computation(topics)
        topic_1,topic_2,topic_3,topic_4=create_topic_files(com)

        topic1_comments=topic_1['Original_Message']
        topic2_comments=topic_2['Original_Message']
        topic3_comments=topic_3['Original_Message']
        topic4_comments=topic_4['Original_Message']

        
        posT1,negT1=get_comments(topic_1)
        posT2,negT2=get_comments(topic_2)
        posT3,negT3=get_comments(topic_3)
        posT4,negT4=get_comments(topic_4)




        T1_comments=topic1_comments.to_json(orient='records')
        T1_comments=eval(T1_comments)
        T2_comments=topic2_comments.to_json(orient='records')
        T2_comments=eval(T2_comments)
        T3_comments=topic3_comments.to_json(orient='records')
        T3_comments=eval(T3_comments)
        T4_comments=topic4_comments.to_json(orient='records')
        T4_comments=eval(T4_comments)
        
        Keys_T1,Pos_T1,Neg_T1=pos_neg_key_count(T1_comments)
        Keys_T2,Pos_T2,Neg_T2=pos_neg_key_count(T2_comments)
        Keys_T3,Pos_T3,Neg_T3=pos_neg_key_count(T3_comments)
        Keys_T4,Pos_T4,Neg_T4=pos_neg_key_count(T4_comments)

        
        text_1=""
        for i in T1_comments:
            text_1+=i
        text_1=text_1.lower()
        wc_1=process(text_1)

        text_2=""
        for i in T2_comments:
            text_2+=i
        text_2=text_2.lower()
        wc_2=process(text_2)

        text_3=""
        for i in T3_comments:
            text_3+=i
        text_3=text_3.lower()
        wc_3=process(text_3)

        text_4=""
        for i in T4_comments:
            text_4+=i
        text_4=text_4.lower()
        wc_4=process(text_4)

        # topic1=topic_1.to_json(orient='records')
        # topic2=topic_2.to_json(orient='records')
        # topic3=topic_3.to_json(orient='records')
        # topic4=topic_4.to_json(orient='records')
        # topic1=eval(topic1)
        # topic2=eval(topic2)
        # topic3=eval(topic3)
        # topic4=eval(topic4)

        pos_per_1,neg_per_1= posnegpercentage(T1_comments)
        pos_per_2,neg_per_2= posnegpercentage(T2_comments)
        pos_per_3,neg_per_3= posnegpercentage(T3_comments)
        pos_per_4,neg_per_4= posnegpercentage(T4_comments)


        
    #return {'Topic 1':topic1,"Topic 2":topic2,"Topic 3":topic3,"Topic 4":topic4}
    return {'Topic 1 Keys':Keys_T1,'Topic 1 Positive':Pos_T1,'Topic 1 Negative':Neg_T1,'Topic 2 Keys':Keys_T2,'Topic 2 Positive':Pos_T2,'Topic 2 Negative':Neg_T2,'Topic 3 Keys':Keys_T3,'Topic 3 Positive':Pos_T3,'Topic 3 Negative':Neg_T3,'Topic 4 Keys':Keys_T4,'Topic 4 Positive':Pos_T4,'Topic 4 Negative':Neg_T4,'Wordcloud_1':wc_1,'Wordcloud_2':wc_2,'Wordcloud_3':wc_3,'Wordcloud_4':wc_4,'Pos_Per_1':pos_per_1,'Pos_Per_2':pos_per_2,'Pos_Per_3':pos_per_3,'Pos_Per_4':pos_per_4,'Neg_Per_1':neg_per_1,'Neg_Per_2':neg_per_2,'Neg_Per_3':neg_per_3,'Neg_Per_4':neg_per_4,'PositveComm1':posT1,'PositiveComm2':posT2,'PositiveComm3':posT3,'PositiveComm4':posT4,
                                                           'NegativeComm1':negT1,'NegativeComm2':negT2,'NegativeComm3':negT3,'NegativeComm4':negT4}