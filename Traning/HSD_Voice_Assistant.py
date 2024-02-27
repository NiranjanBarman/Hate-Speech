import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import speech_recognition as sr 
from googletrans import Translator 
import re
import nltk
from nltk.util import pr
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))

import speech_recognition as sr 
from googletrans import Translator 

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from time import sleep

def Listen():

    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source,0,9) 
    
    try:
        print("Recognizing...")
        query = r.recognize_google(audio,language="hi")

    except:
        return ""
    
    query = str(query).lower()
    return query



def TranslationHinToEng(Text):
    line = str(Text)
    translate = Translator()
    result = translate.translate(line)
    data = result.text
    print(data)
    return data



def MicExcution():
    query = Listen()
    data = TranslationHinToEng(query)
    return data



chrome_options = Options()
chrome_options.add_argument('--log-level=3')
chrome_options.headless = True
Path = "C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe"
driver = webdriver.Chrome(Path,options=chrome_options)
driver.maximize_window()

website = r"https://ttsmp3.com/text-to-speech/British%20English/"
driver.get(website)
ButtonSelection = Select(driver.find_element(by=By.XPATH,value='/html/body/div[2]/div[2]/form/select'))
ButtonSelection.select_by_visible_text('British English / Brian')

def Speak(Text):

    lengthoftext = len(str(Text))

    if lengthoftext==0:
        pass

    else:
        print("")
        print(Text)
        print("")
        Data = str(Text)
        xpathofsec = '//*[@id="voicetext"]'
        driver.find_element(By.XPATH,value=xpathofsec).send_keys(Data)
        driver.find_element(By.XPATH,value='//*[@id="vorlesenbutton"]').click()
        driver.find_element(By.XPATH,value="/html/body/div[2]/div[2]/form/textarea").clear()

        if lengthoftext>=30:
            sleep(4)

        elif lengthoftext>=40:
            sleep(6)

        elif lengthoftext>=55:
            sleep(8)

        elif lengthoftext>=70:
            sleep(10)

        elif lengthoftext>=100:
            sleep(13)

        elif lengthoftext>=120:
            sleep(14)

        else:
            sleep(2)
    


def HateSpeech():
    
    df = pd.read_csv("twitter_data.csv")
    #print(df.head())
    df['labels'] = df['label'].map({0: "NO Hate Speech Detected!", 1: "Hate Speech detected!",}) 
    #print(df.head())
    df = df[['tweet','labels']]
    #print(df.head())

    def clean(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]','',text)
        text = re.sub('https?://\s+|www\.\s+','',text)
        text = re.sub('<.*?>+', '',text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text=" ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text=" ".join(text)
        return text
    df["tweet"] = df["tweet"].apply(clean)
    #print(df.head())
        
    df.isnull()
    df.isnull().sum().sum()
    df.dropna(inplace=True)

    x = np.array(df["tweet"])
    y = np.array(df["labels"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)

    test_data= MicExcution()
    df = cv.transform([test_data]).toarray()
    print(clf.predict(df))
    Speak(clf.predict(df))

HateSpeech()
