import pandas as pd
import numpy as np
import  re
import nltk
import random
import string as st
from nltk.corpus import stopwords
from textblob import TextBlob

#run for each company
data_sentiment=pd.read_csv('C:/Users/leand/OneDrive/Desktop/final_project/FINAL_CLEANED_DATA/PRESENT/Toyota_present.csv')
data_sentiment.columns=['index','tweet_id','language','date','text','location','followers','retweets','likes','month','company','PERIOD']
data_sentiment=data_sentiment.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]


# function to get the subjectivity
def getSubjectivity(text):
   return TextBlob(str(text)).sentiment.subjectivity
  
# function to get the polarity
def getPolarity(text):
   return TextBlob(str(text)).sentiment.polarity

data_sentiment['Subjectivity'] = data_sentiment['text'].apply(getSubjectivity)
data_sentiment['Polarity'] = data_sentiment['text'].apply(getPolarity)


def getSentiment(score):
    if score < 0:
       return -1
    elif score == 0:
       return 0
    else:
       return 1

data_sentiment['sentiment'] =data_sentiment['Polarity'].apply(getSentiment )