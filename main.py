from concurrent.futures import thread

import time

import requests
import cfg
from calendar import Calendar, calendar
from distutils.command.config import config
from math import remainder
from typing import Dict
import torch
from click import password_option

import pyjokes
from transformers import pipeline
import random
import wikipedia as wiki
from duckduckgo_search import ddg
import datetime
import time
from threading import Thread
import threading
from tokenize import String
import speech_recognition
import pyttsx3
import random
import torch.nn as nn
import transformers
from transformers import AutoTokenizer
import cfg as config 
from googlesearch import search as gsearch
import torch

import joblib
from wolframalpha import *
from copy import deepcopy

from datetime import date
from datetime import datetime
import random
import speech_recognition

import os

from functools import partial
import json
import pyttsx3
import asyncio

from kivy.lang import Builder
from kivy.properties import StringProperty, ListProperty

from kivymd.app import MDApp
from kivymd.theming import ThemableBehavior
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.list import OneLineIconListItem, MDList
from kivymd.font_definitions import theme_font_styles 
from kivymd.uix.label import MDLabel
from kivy.core.window import Window
import tensorflow as tf
import encodings
from importlib.resources import path
from tracemalloc import start
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import keras
import tensorflow as tf
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from abc import ABCMeta, abstractmethod

import random
import json
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import nltk
# from nltk.stem import WordNetLemmatizer

import tensorflow as tf
import tensorflow.lite as lt
from keras.models import load_model
import keras
#from cfg import TOKENIZER
import tensorflow.lite.python.lite

import json
import datetime
from mimetypes import init
#from mimetypes import list
import time
import os
import calendar as cln


data_path = ""

from multiprocessing.connection import Client
from re import search
from typing import List
from youtubesearchpython import VideosSearch
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from googletrans import Translator
from pandas import json_normalize
import pandas as pd
import webbrowser
import requests
import json
import wolframalpha
import geocoder

key = "1e8c18c48cb146a3836fd8d827ad4d35"

class News:
    def __init__(self) -> None:
        self.channels = {
            "bbc":'https://www.bbc.com/news',
            "ny":"https://www.nytimes.com/international/",
            "aljazeera":"https://www.aljazeera.com/"
        }
        self.newsapi =  NewsApiClient(api_key=key)
    def get(self,channel,loc="global"):
        pass
    def get_headlines(self,chnl = "bbc"):
        response = requests.get(self.channels[chnl])
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find('body').find_all('h3')
        unwanted = ['BBC World News TV', 'BBC World Service Radio','News daily newsletter', 'Mobile app', 'Get in touch']
        hdl = []
        for x in list(dict.fromkeys(headlines)):
            if x.text.strip() not in unwanted:
                print(x.text.strip()) 
                hdl.append(x.text.strip())
        return hdl

    def get_api_headline(self,category,country="us"):
        newsapi = NewsApiClient(api_key=key)
        top_headlines =newsapi.get_top_headlines(category=category,language='en',country=country)     
        top_headlines=json_normalize(top_headlines['articles'])
        newdf=top_headlines[["title","url"]]
        dic=newdf.set_index('title')['url'].to_dict()
        print(dic)
    def get_news_url(self,q):
        newsapi = NewsApiClient(api_key=key)
        everything = newsapi.get_everything(q=q)
        everything = json_normalize(everything['articles'])
        print(everything["url"][0])
        return random.choice(everything["url"])


class Weather:
    def __init__(self) -> None:
        pass
    def get(self):
        g = geocoder.ip('me')
        
        wth = str(g.latlng[0]) + "," + str(g.latlng[1])
        htmldata = self.getdata("https://weather.com/en-IN/weather/today/l/"+wth+"?par=google&temp=c/")
        
        soup = BeautifulSoup(htmldata, 'html.parser')
        wth_data = soup.find_all("div",class_="WeatherDetailsListItem--wxData--2s6HT")
        for wth in wth_data:
            pass
        
        current_temp = soup.find_all("span", class_= "TodayDetailsCard--feelsLikeTempValue--Cf9Sl")[0].text
        current_wind = soup.find_all("span",class_ = "Wind--windWrapper--3aqXJ undefined")[0].text.split("Wind Direction")[1]
        temp_hl = soup.find_all("div",class_="WeatherDetailsListItem--wxData--2s6HT")[0].text.split("/")
        humidity = soup.find_all("div",class_="WeatherDetailsListItem--wxData--2s6HT")[2].text
        chances_rain = soup.find_all("div", class_= "CurrentConditions--phraseValue--2Z18W")
        temp = current_temp
        wthh = []
        for i in soup.find_all("div",class_="WeatherDetailsListItem--wxData--2s6HT"):
            wthh.append(i.text)
        
        wth_rtn = {
            "humidty":wthh[3],
            "wind":current_wind,
            "temp":current_temp,
            "moon":wthh[-1],
            "overall":chances_rain[0].text
        }
        return wth_rtn

    def getdata(self,url):
        print(url)
        r = requests.get(url)
        return r.text

class TransLL:
    translator = Translator()
    motherLang = "en"
    def translate(self,snt):
        translations = self.translator.translate(snt, dest=self.motherLang)
        temp = []
        if isinstance(translations,List):
            temp = [x.text for x in translations]
            return temp
        else:
            return translations.text

class Search:
    def __init__(self):
        self.p = ""     
        self.c = ""
  
    def gglsearch(self,tag):
        pass

    def ytsearch(self,tag):
        videosSearch = VideosSearch(tag, limit = 2).result()
        return videosSearch

    def open(self,lnk):
        webbrowser.open(lnk)

class ScraperCSV:
    def _init__(self)->None:
        self.sources = []

    def get_data(self,params):
        pass

def wl_Answer(question):
    app_id = '378KAR-YR5HKRQEAJ'
    client = wolframalpha.Client(app_id)
    res = client.query(question)
    try:
        answer = next(res.results).text
        return answer
    except:
        return False

class Lists:
    def __init__(self) -> None:
        self.path = "./data/lists.json"
        self.data = json.load(open(self.path))
        
    def handle():
        pass
    def make(self,name):
        for i in self.data.keys():
            if i == name:
                return "list with same name already exists"
        self.data.update({name:[]})
        self.save()
        return f"successfully created list {name}"
    def append_to_lst(self,lst_nm,dta,date_time):
        self.data[lst_nm].append({"data":dta,"time":date_time})
        self.save()
    def get(self,name):
        return self.data[name]
    def save(self):
        with open(self.path, 'w') as outfl:
            outfl.write(json.dumps(self.data))
    def remove(self,ln,dn):
        for i in self.data.items():
            if i[0] == ln:
                for ii in i[1]:
                    if ii["data"] == dn:
                        self.data[ln].remove(ii)
 


class Alarms:
    def __init__(self) -> None:
        self.path = "./data/alarms.json"
        self.data = json.load(open(self.path))
    def set(self,time,flag = "normal"):
        self.data.append({"time":time,"flag":flag})
        self.save()
    def get(self):
        return self.data
    def remove(self,time):
        for i in self.data:
            if i["time"] == time:
                self.data.remove(i)
        self.save()
    def save(self):
        with open(self.path, 'w') as outfl:
            outfl.write(json.dumps(self.data))


class Calenders:
    def __init__(self) -> None:
        self.events = []
        self.caln  = cln.calendar(2022)
    def set_event(self,event,date):
        x = self.events.append({"date":date,"event":event})

    def get_event(self,date):
        pass

class LocalPlay:
    def __init__(self) -> None:
        pass
    def Search(self,ext,name):
        pass
    def Open(self,oath):
        pass

class AssModel(keras.Model):
    def __init__(self):
        pass


class QuerkyAnswer:
    def __init__(self,intents) -> None:
        self.intents = intents
        self.model_name = "probot"
        self.words = []

        if intents.endswith(".json"):
            self.load_json_intents(intents)

        #elf.lemmatizer = WordNetLemmatizer()
    def load_json_intents(self, intents):
        self.intents = json.loads(open(intents).read())
    def train_model(self):
        self.words = []
        self.classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']

        for intent in self.intents['talks']:
            for pattern in intent['pattern']:
                word = pattern.split(" ")
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [w.lower() for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))



        training = []
        output_empty = [0] * len(self.classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [word.lower() for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

        sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        print(train_x," ",len(train_x))
        print(train_y," ",len(train_y))
    
    def save_model(self, model_name=None):
        self.model.save(f"{self.model_name}.h5", self.hist)
        with open(f'{self.model_name}_words.json', 'w') as outfl:
            outfl.write(json.dumps(self.words))
        with open(f'{self.model_name}_classes.json', 'w') as outfl:
            outfl.write(json.dumps(self.classes))

    def load_model(self, model_name=None):
        self.words = json.loads(open(f'./data/{self.model_name}_words.json', 'r').read())
        self.classes = json.loads(open(f'./data/{self.model_name}_classes.json', 'r').read())
        self.model = load_model(model_name)

    def _get_response(self, ints, intents_json):
        result = ""
        try:
            tag = ints[0]['intent']
            print("Answering with prob" + ints[0]['probability'])
            list_of_intents = intents_json['talks']
            for i in list_of_intents:
                if i['tag']  == tag:
                    result = random.choice(i['response'])
                    break
        except IndexError:
            result = "sorry I didnt get that. would you want to search it to web"
        return result
    def _predict_class(self, sentence):
        p = self.bows(sentence, self.words)
        print(np.array([p]))
        res = self.model.predict(np.array([p]))[0]
        print(res)
        ERROR_THRESHOLD = 0.9
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def _clean_up_sentence(self, sentence):
        sentence_words = sentence.split(" ")
        sentence_words = [word.lower() for word in sentence_words]
        return sentence_words

    def bows(self, sentence, words):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag,np.float32) 
    def request(self, message):
        ints = self._predict_class(message)
        return self._get_response(ints, self.intents)
    def saveLite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open('probot.tflite', 'wb') as f:
            f.write(tflite_model)


class QAmdl:
    def __init__(self,pth):
        #self.model = tf.saved_model.load("D:\dev\clients\charlie\AI\models")
        self.model = BertForQuestionAnswering.from_pretrained(pth)
        self.tokenizer = BertTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        self.nlp = pipeline("question-answering",model=self.model,tokenizer=AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2"))
        self.data = ""
        

    def question_answer(self,q,t):
        tt = q.split(".")
        
        answers = []
        confidence = []
        for j,i in enumerate(tt):
            inp  = self.tokenize(t,i)
            outp = self.model(input_ids=torch.tensor([inp[0]]), token_type_ids=torch.tensor([inp[1]]))

            start_index = torch.argmax(outp.start_logits)
            end_index = torch.argmax(outp.end_logits)
            
            if ' '.join(inp[2][start_index:end_index+1]) != " ":
                confidence.append(torch.max(outp.start_logits).detach().numpy())
                answers.append(' '.join(inp[2][start_index:end_index+1]))

        corrected_answer = ''
        result = np.where(confidence == np.amax(confidence))
        
        print(confidence)
        print(result)
        print(answers)
        ind = 0
        
        if answers[int(np.argmax(confidence))] != "" :
            ind = 0
        

        for word in answers[ind].split():
            
            #If it's a subword token
            if word[0:2] == '##':
                corrected_answer += word[2:]
            else:
                corrected_answer += ' ' + word
        return corrected_answer

    def tokenize(self,question,text):
        encodes = self.tokenizer.encode_plus(text=question,text_pair=text,max_length=512,truncation=True)
        inputs = encodes['input_ids']
        tokens = self.tokenizer.convert_ids_to_tokens(inputs)
        sentence_embedding = encodes['token_type_ids'] 
        return inputs,sentence_embedding,tokens


    def read_doc(self):
        pass


KV = '''
# Menu item in the DrawerList list.
<ItemDrawer>:
    theme_text_color: "Custom"
    on_release: self.parent.set_color_item(self)

    IconLeftWidget:
        id: icon
        icon: root.icon
        theme_text_color: "Custom"
        text_color: root.text_color


<ContentNavigationDrawer>:
    orientation: "vertical"
    padding: "8dp"
    spacing: "8dp"



    ScrollView:

        DrawerList:
            id: md_list



MDScreen:

    MDNavigationLayout:

        ScreenManager:

            MDScreen:

                MDBoxLayout:
                    orientation: 'vertical'

                    MDToolbar:
                        title: "Charlie"
                        elevation: 10
                        left_action_items: [['menu', lambda x: nav_drawer.set_state("open")]]

                    Widget:
                    MDBoxLayout:
                        orientation: "vertical"
                        size_hint_y:None 
                        height:450
                        ScrollView:
                            pos_hint: {'center_y':0.5}

                            MDList:
                                id: box
                    MDBoxLayout:
                        orientation: 'horizontal'
                        padding:10
                        MDTextField:
                            padding:10
                            id:cmd
                            hint_text:"command anything"
                        MDRoundFlatIconButton:
                            padding:10
                            icon: "arrow-right-bold"
                            text: "execute"
                            on_press:app.execute(root.ids.cmd)
                        MDRoundFlatIconButton:
                            padding:10
                            icon: "microphone"
                            text: "use voice"
                            on_press:app.voice_cmd()
                        
                        


        MDNavigationDrawer:
            id: nav_drawer

            ContentNavigationDrawer:
                id: content_drawer
'''



class NLUModel(nn.Module):
    def __init__(self,num_entity, num_intent, num_scenario):
        super(NLUModel,self).__init__()
        self.num_entity = num_entity
        self.num_intent = num_intent
        self.num_scenario = num_scenario

        self.bert = transformers.BertModel.from_pretrained(
            config.BASE_MODEL
        ) 
        self.drop_1 = nn.Dropout(0.3)
        self.drop_2 = nn.Dropout(0.3)
        self.drop_3 = nn.Dropout(0.3)

        self.out_entity = nn.Linear(768,self.num_entity)
        self.out_intent = nn.Linear(768,self.num_intent)
        self.out_scenario = nn.Linear(768,self.num_scenario)

    def forward(self, ids,mask,token_type_ids):
        out = self.bert(input_ids=ids,
                              attention_mask=mask,
                              token_type_ids=token_type_ids
                              )
        hs, cls_hs = out['last_hidden_state'], out['pooler_output']
        entity_hs = self.drop_1(hs)
        intent_hs = self.drop_2(cls_hs)
        scenario_hs = self.drop_3(cls_hs)

        entity_hs = self.out_entity(entity_hs)
        intent_hs = self.out_intent(intent_hs)
        scenario_hs = self.out_scenario(scenario_hs)

        return entity_hs,intent_hs,scenario_hs


class NLUEngine:
    def __init__(self,model_path):
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.device = config.DEVICE
        #self.model = torch.jit.load(model_path).to(self.device).eval()
        
        self.meta_data =joblib.load('./models/meta_data.bin')

        self.enc_entity = self.meta_data['enc_entity']
        self.enc_intent = self.meta_data['enc_intent']
        self.enc_scenario = self.meta_data['enc_scenario']

        self.num_entity = len(self.enc_entity.classes_)
        self.num_intent = len(self.enc_intent.classes_)
        self.num_scenario = len(self.enc_scenario.classes_)
        self.model = NLUModel(self.num_entity,self.num_intent,self.num_scenario)
        self.model.load_state_dict(torch.load("./models/model_trace.pth"))
        self.model.to(config.DEVICE)
        self.model.eval()

    @staticmethod
    def classes_to_scores_json(classes,intent_scores):
        dict = {'scores':[]}
        for scores in intent_scores:
            dict['scores'] += [{c:s for c,s in zip(classes,scores)}]
        return dict

    @staticmethod
    def sentence_to_labels_json(labels,num_sentence):
        return {i:l for l,i in zip(labels, range(num_sentence))}
            

    @staticmethod
    def words_to_labels_json(word_pieces,labels):
        return {w:l for w,l in zip(word_pieces,labels)}

    @staticmethod
    def words_to_scores_json(words_pieces, scores):
        return {w:cs for w,cs in zip(words_pieces, scores)}


    @staticmethod
    def to_yhat(logits):
        logits = logits.view(-1, logits.shape[-1]).cpu().detach()
        probs = torch.softmax(logits, dim=1)
        y_hat = torch.argmax(probs, dim=1)
        return probs.numpy(),y_hat.numpy() 
    
    
    def process_sentence(self,sentence):
        """ Given a sentence stirng it will return rquired inputs for NLU model's forward pass

        """
        sentence = str(sentence)
        sentence = " ".join(sentence.split())
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length = self.max_len
        )
        
        tokenized_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        word_pieces = self.tokenizer.decode(inputs['input_ids']).split()[1:-1] #the first token an the last token are special token

        #padding
        padding_len = self.max_len - len(tokenized_ids)
            
        ids = tokenized_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        
        ids = torch.tensor(ids,dtype=torch.long).unsqueeze(0).to(self.device)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(self.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        return ids,mask,token_type_ids,tokenized_ids,word_pieces
    
    def sentence_prediction(self,ids,mask,token_type_ids):
        entity_hs,intent_hs,scenario_hs  = self.model(ids,mask,token_type_ids)
        return entity_hs,intent_hs,scenario_hs   
    
    def entity_extraction(self,entity_hs,word_pieces,tokenized_ids):
        """Given logits of (entity) from model, it will generate class labels and scores

        Args:
            entity_hs ([torch.tensor]): [num_sentence x seq_len x classes]
            word_pieces ([Array]): [Sentence word peices from bert tokenizers]
            tokenized_ids ([Array]): [Token ids from bert tokenizers pre-padding stage]

        Returns:
            words_labels [dict]: [Dictionary of shape (words_pieces x prediction_labels) eg. {'wake':'O', '5':'time'...}]
            words_scores  [dict]: [Dictionary of shape (words_pieces x num_classes) eg. {'wake':[class1:score,......], '5':[class1:score,.....]...}]
        """
        entity_scores,entity_preds = self.to_yhat(entity_hs)
        entity_scores = entity_scores[1:len(tokenized_ids)-1, :]
        enitity_labels = self.enc_entity.inverse_transform(entity_preds)[1:len(tokenized_ids)-1]
        words_labels_json = self.words_to_labels_json( word_pieces , enitity_labels)
        words_scores_json = self.words_to_scores_json( word_pieces, entity_scores)
        return words_labels_json,words_scores_json
        
    def classification(self,logits,task='intent'):
        """Given logits of (intent or scenario) from model, it will generate class labels and scores
        Args:
            logits ([torch.tensor]): [Tensor of shapee (Num_sentences x classes)]
            task (str, optional): [The classification task to perform]. Defaults to 'intent'.

        Returns:
            sentence_labels_json [dict]:  [Dictionary of shape (Num_sentences,) eg. {0:intent_for_sentence_0, 1:intent_for_sentence2}]  
            class_scores_json [dict]: [Dictionary of shape (num_classes x num_classes), eg,{'scores':[{'call':0.5,.....},{..sentence2..}]}
        """
        if task == 'intent':
            enc = self.enc_intent
        else:
            enc = self.enc_scenario
        class_scores,class_preds = self.to_yhat(logits)
        sentence_labels_json = self.sentence_to_labels_json(enc.inverse_transform(class_preds), len(class_preds))
        class_scores_json = self.classes_to_scores_json(enc.classes_,class_scores)
        return sentence_labels_json, class_scores_json
    
    def predict(self,sentence):
        """Given a sentence it will return NLU model prediction for entity,intent and scenario, wrapped as  
        json object.
        Args:
            sentence ([str]): [Input string]
        Returns:
            words_labels_json [dict]: [Dictionary of shape (words_pieces x prediction_labels) eg. {'wake':'O', '5':'time'...}]
            words_scores_json  [dict]: [Dictionary of shape (words_pieces x num_classes) eg. {'wake':[class1:score,......], '5':[class1:score,.....]...}]
            intent_sentence_labels_json [dict]: [Dictionary of shape (Num_sentences,), eg. {0:intent_for_sentence_0, 1:intent_for_sentence2}]  
            intent_class_scores_json [dict]: [Dictionary of shape (num_classes x num_classes), eg,{'scores':[{'call':0.5,.....},{..sentence2..}]}]
            scenario_sentence_labels_json [dict]:  [Dictionary of shape (Num_sentences,) eg. {0:intent_for_sentence_0, 1:intent_for_sentence2}]  
            scenario_class_scores_json [dict]: [Dictionary of shape (num_classes x num_classes), eg,{'scores':[{'call':0.5,.....},{..sentence2..}]}
        """
        ids,mask,token_type_ids,tokenized_ids,word_pieces = self.process_sentence(sentence)

        #forward the inputs throught the model and get logits
        entity_hs,intent_hs,scenario_hs = self.sentence_prediction(ids,mask,token_type_ids)

        #entity extraction
        words_labels_json,words_scores_json = self.entity_extraction(entity_hs,word_pieces,tokenized_ids)

        # intent and scenario classification
        intent_sentence_labels_json, intent_class_scores_json = self.classification(intent_hs,task='intent')
        scenario_sentence_labels_json, scenario_class_scores_json = self.classification(scenario_hs,task='scenario')
             
        return (words_labels_json, 
                words_scores_json, 
                intent_sentence_labels_json, 
                intent_class_scores_json, 
                scenario_sentence_labels_json, 
                scenario_class_scores_json)

class ChrBrain:
    def __init__(self,mdl):
        self.nlu_mdl = NLUEngine(mdl)
        self.recognizer = speech_recognition.Recognizer()
        self.speaker = pyttsx3.Engine()
        self.snts = ""
        self.nt_rcg = False
        self.threshold = 0.25
        self.prev_voice_id = self.speaker.getProperty("voice")
    def change_voice(self):
        voice = self.speaker.getProperty('voices')
        if voice[0].id == self.prev_voice_id:
            self.speaker.setProperty('voice', voice[1].id)
            self.prev_voice_id = voice[1].id
        else:
            self.speaker.setProperty('voice', voice[0].id)
            self.prev_voice_id = voice[0].id

    def speak(self,snt):
        def runnn():    
            self.speaker.say(snt)
            self.speaker.runAndWait()
        x = threading.Thread(target=runnn)
        x.start()
        x.join(2.0)
    def listens(self):
        def runn():
            self.nt_rcg = False
            with speech_recognition.Microphone() as mic:
                print("recgnizing")
                self.recognizer.adjust_for_ambient_noise(mic,0.2)
                audio = self.recognizer.listen(mic)
                print("recgnizzing")
                self.snts = self.recognizer.recognize_google(audio)
                self.nt_rcg = True
        runn()
    def predicts(self,cmd):
        (words_labels, 
        words_scores, 
        intent_sentence_labels,
        intent_class_scores,
        scenario_sentence_labels,
        scenario_class_scores)= self.nlu_mdl.predict(cmd)
        intent = intent_sentence_labels[0]
        scenario = scenario_sentence_labels[0]
        print(intent_class_scores)
        ic = intent_class_scores["scores"][0][intent]
        sc = scenario_class_scores["scores"][0][scenario]
        icb = ic >= self.threshold
        scb = sc>= self.threshold
        
        out = {
            "intent":{"name":intent,"probs":ic,"threshold":icb},
            "scenario":{"name":scenario,"probs":sc,"threshold":scb},
            "entity":words_labels
        }
        return out


class ContentNavigationDrawer(MDBoxLayout):
    pass


class ItemDrawer(OneLineIconListItem):
    icon = StringProperty()
    text_color = ListProperty((0, 0, 0, 1))


class DrawerList(ThemableBehavior, MDList):
    def set_color_item(self, instance_item):
        """Called when tap on a menu item."""

        # Set the color of the icon and text for the menu item.
        for item in self.children:
            if item.text_color == self.theme_cls.primary_color:
                item.text_color = self.theme_cls.text_color
                break
        instance_item.text_color = self.theme_cls.primary_color

ss = ""

class Charlie(MDApp):
    def __init__(self, **kwargs):
        self.tsk = TaskEngine()
        self.brain = ChrBrain(cfg.MODEL_PATH)
        super().__init__(**kwargs)
        self.tsk = TaskEngine()
        self.state = ""
        self.running = False 
    def build(self):
        screen = Builder.load_string(KV)
        self.theme_cls.theme_style = "Dark"  # "Light"

        return screen

    def on_start(self):
        self.running = True
        self.fps_monitor_start()
        icons_item = {
            "home":"main page",
            "cog":"Settings"
        }
        for icon_name in icons_item.keys():
            self.root.ids.content_drawer.ids.md_list.add_widget(
                ItemDrawer(icon=icon_name, text=icons_item[icon_name],on_press=partial(self.chng,icons_item[icon_name]))
            )
        if(self.state == "listen"):
            self.voice_cmd()

    def chng(self,nm,obj):
        #this func should change the application screen according to navbar clickz
        print(nm)

    def RUN(self,vcmd):
        self.state = vcmd
        self.run()

    def outputs(self,text):
        self.root.ids.box.add_widget(MDLabel(
            text=f"Charlie:: {text}",
            font_style="H5",
            size_hint_y= None,
            theme_text_color = "Error"
        ))
        self.brain.speak(text)

    def voice_cmd(self):
        self.outputs("Listening")
        self.brain.listens()
        
        if self.brain.snts != "":
            self.executes(self.brain.snts)
    def execute(self,bx):
        self.executes(bx.text)
    
    def executes(self,cmd):
        print(cmd)
        if self.tsk.mode != "qa":
            self.root.ids.box.add_widget(MDLabel(
                        text=f"USER:: {cmd}",
                        font_style="H5",
                        size_hint_y= None
                    ))
        if self.tsk.mode == "":
            outp = self.brain.predicts(cmd)
            print(outp)
            self.tsk.execute(outp,cmd)
            self.outputs(self.tsk.outp)
            self.tsk.outp = ""
        elif self.tsk.mode == "trn":
            oop = self.tsk.translate(cmd)
            if oop:
                self.outputs(oop)
            else:
                self.outputs("translation mode off")
        elif self.tsk.mode == "chng":
            self.brain.change_voice()
            self.tsk.mode = ""
            outp = self.brain.predicts(cmd)
            self.tsk.execute(outp,cmd)
            self.outputs(self.tsk.outp)
        elif self.tsk.mode == "qa":
            if "exit" in cmd or "close" in cmd or "mode off" in cmd:
                self.tsk.mode = ""
                self.outputs("Question answering mode off")
            else:
                if self.tsk.inp_flg == "qa_text":
                    if cmd == str(0) or "mode on" in cmd:
                        pass
                    else:
                        self.tsk.knowledge += cmd
                    self.tsk.inp_flg = "qa_q"
                    open("./data/knowledge.txt","w").write(self.tsk.knowledge)
                    self.outputs("ask the question")
                elif self.tsk.inp_flg == "qa_q":
                    self.outputs(self.tsk.qa_mdl.question_answer(self.tsk.knowledge,cmd)+ "..\n Enter the text or enter 0")
                    self.tsk.inp_flg = "qa_text"
                    # self.outputs("Enter the text or enter 0")
        elif self.tsk.mode == "excel":
            pass



class TaskEngine:
    def __init__(self):
        self.curr_task = ""
        self.state = ""
        self.outp = ""
        self.last_outp = ""
        self.knowledge = open("./data/knowledge.txt","r").read()
        self.q = QuerkyAnswer("./data/cmd.json")
        self.q.load_model("./models/probot.h5")
        self.alm = Alarms()
        self.wth = Weather()
        self.news = News()
        self.rmn = Calenders()
        self.translator= TransLL()
        self.srch = Search()
        self.lst = Lists()
        self.qa_mdl = QAmdl("./models/qa/")
        self.dta = ScraperCSV()
        self.brt = 50

        self.inp_flg = ""
        self.mode = ""
    def translate(self,snt):
        if snt == "exit" or snt == "close":
            self.mode = ""
            return False
        else:
            return self.translator.translate(snt)
    def extract_entity(self,ent : Dict):
        ents = []
        for i in ent.items():
            if i[1]!= "O":
                ents.append({"word":i[0],"val":i[1]})
        if len(ents) >= 1:
            return ents
        return False
    def execute(self,outp,snt):
        ir = outp["intent"]
        sr = outp["scenario"]
        er = outp["entity"]
        ers = self.extract_entity(er)
        
        if "change" in snt.split(" ") and "voice" in snt.split(" "):
            self.outp = " voice changed"
            self.mode = "chng"
        if "scrape" in snt.split(" "):
            self.outp = "scraping mode on"
            self.mode = "scr"
        if "search" in snt.split(" "):
            self.outp = "Ok now enter the query that needs to be searched"
            self.mode = "srch"
        elif "translat" in snt.split(" ") or "translation" in snt:
            self.outp = "translation mode on"
            self.mode = "trn"
        elif "question" in snt.split(" ") or "qa" in snt.split(" "):
            self.mode = "qa"
            self.outp = "Question answering mode on. Enter the text or enter 0"
            self.inp_flg = "qa_text"
        elif "excel" in snt.split(" "):
            self.mode = "excel"
            self.outp = "Excel spreadsheet mode on "

        elif sr["name"] == "general":
            if ir["name"]== "quirky" or ir["name"] == "greet":
                self.outp = self.q.request(snt)
            elif ir["name"] == "confirm":
                self.outp = random.choice(["confirming","OK","getting it done"])
            elif ir["name"] == "negate":
                self.outp = random.choice(["taking the value as negative","OK"])
            elif ir["name"] == "affirm":
                self.outp = random.choice(["thanks","All Correct"])
            elif ir["name"] == "repeat":
                self.outp = self.last_outp
            elif ir["name"] == "commandstop":
                self.outp = random.choice(["Stopping","OK"])
            elif ir["name"] == "praise":
                self.outp = random.choice(["thanks","I dont worth that praise","You are very welcome","My pleasure"])
            elif ir["name"] == "joke":
                resp = requests.get("https://api.chucknorris.io/jokes/random")
                joke = json.loads(resp.text)["value"].split("Norris")[1]
                self.outp = joke
            else:
                self.outp = random.choice(["I didnt Understand","I am confused","I am little bit confused about what you saying","I am not sure about what you are saying"])
        elif sr["name"] == "email":
            self.outp = "I currently cant handle emails"
        elif sr["name"] == "qa":
            soutp = wl_Answer(snt)
            if soutp:
                self.outp = soutp.replace("|"," ")
            else:
                query = snt.replace(" ","+")
                resp = json.loads(requests.get("http://api.duckduckgo.com/?q="+query+"&format=json").text)
                print(resp)
                if resp["Abstract"] != "": 
                    self.outp = resp["Abstract"]
                else:
                    self.outp = "I found this on google"
                    webbrowser.open("https://www.google/search?q="+query+"&aqs=chrome.0.69i59j69i60.9117j0j7&sourceid=chrome&ie=UTF-8")
        elif sr["name"] == "social":
            self.outp = "I currently cant do anything in social media"
        elif sr["name"] == "calendar":
            self.outp = random.choice("This featur is not implemented yet","Currently I cant do that","I am really sorry But I cant do that right now","I am not trained for doing this")
        elif sr["name"] == "datetime":
            if ir["name"] == "query":
                today = date.today()
                now = datetime.now()
                current_time = now.strftime("%H:%M").split(":")
                self.outp = "current datetime is " + current_time[0] + " hours " + current_time[1] +"minutes " + today.strftime("%B %d, %Y")
            elif ir["name"] == "convert":
                self.outp = "ok should I search google about it"
            else:
                self.outp = "sorry I cant do that"

        elif sr["name"] == "news":
            if ir["name"] == "query": 
                if ers:
                    medias = []
                    places = []
                    news_topic = []
                    for i in ers:
                        if i["val"] == "place_name":
                            places.append(i["word"])
                        elif i["val"] == "media_type":
                            medias.append(i["word"])
                        elif i["val"] == "news_topic":
                            news_topic.append(i["word"])
                    arr = None
                    
                    if len(medias) != 0  and len(places) != 0:
                        self.outp = "getting news about " + places[0] + " in " + medias[0] + " media"
                        url = self.news.get_news_url("news in "+ places[0] + " "+ medias[0])
                        webbrowser.open(url)
                    elif len(medias) != 0:
                        self.outp = "getting news in " + medias[0] + " media"
                        url = self.news.get_news_url(medias[0]+" news")
                        webbrowser.open(url)
                    elif len(places) != 0:
                        self.outp = "getting news  of " + places[0]
                        url = self.news.get_news_url(places[0]+" ")
                        webbrowser.open(url)
                    elif len(news_topic) != 0:
                        self.outp = "getting news about " + news_topic[0]
                        url = self.news.get_news_url(news_topic[0])
                        webbrowser.open(url=url)
                    else:
                        ents = ""
                        for i in ers:
                            ents += i["word"] + " "
                        url = self.news.get_news_url(ents)
                        webbrowser.open(url)
                else:
                    news = self.news.get_headlines()
                    newss = ""
                    for i in news:
                        if random.random() > 0.8:
                            newss += i + "\n"

                    self.outp = "here is some latest news headlines I got from bbc " + newss
        elif sr["name"] == "weather":
            if ir["name"] == "query":
                wtd = ""
                datt = ""
                if ers:
                    for i in ers:
                        if i["val"] == "weather_descriptor":
                            wtd += i["word"] + "  "
                        elif i["val"] == "date":
                            datt += i["word"] + "  "
                    weather = self.wth.get()
                    if "umbrella" in wtd or "rain" in wtd:
                        if "cloudy" in weather["overall"]:
                            self.outp = "Weather reports says "+ datt +" is " + weather["overall"] + " I think you should bring an umbrella"
                        else:
                            self.outp = "Weather reports says "+ datt +" is " + weather["overall"]

                    elif "snow" in wtd:
                        if int(weather["temp"][0]) < 1:
                            self.outp = random.choices(["there might be a possibility of snowing","there maybe snowing","I think it's snowfalling","current temperature is" + weather["temp"][0]])
                        else:
                            self.outp = "todays temperature is " + weather["temp"] + " degree celsius. So I think there is no chance of snowfall"
                    else:
                        self.outp += datt +"'s weather report is"
                        for i in weather.items():
                            self.outp += " " + i[0] + ":  " +i[1]
                else:
                    weather = self.wth.get()
                    self.outp += "todays weather report is"
                    for i in weather.items():
                        self.outp += " \n" + i[0] + ": " +i[1]
            else:
                weather = self.wth.get()
                self.outp += "todays w eather report is"
                for i in weather.items():
                    self.outp += " "+ i[0] + ":  " +i[1]
        elif sr["name"] == "play":
            if ers:
                plt = ""
                qry = ""
                person = ""
                for i in ers:
                    if i["val"] == "media_type":
                        plt = i["word"]
                    elif i["val"] == "song_name":
                        qry += i["word"] + " "
                    elif i["val"] == "person":
                        person += i["word"] + " "
                if plt == "" or "yt" in plt or plt =="youtube":
                    ytt = self.srch.ytsearch(qry)
                    print(ytt["result"])
                    self.outp = "playing a video I found on youtube titled:" + ytt["result"][0]["title"]
                    webbrowser.open(ytt["result"][0]["link"])
                elif person != "":
                    self.outp = "playing a video of " + person + "on youtube titled:" + ytt["result"][0]["title"]
                    ytt = self.srch.ytsearch(person)
                    print(ytt["result"])
                    webbrowser.open(ytt["result"][0]["link"])
                else:
                    ytt = self.srch.ytsearch(snt)
                    print(ytt["result"])
                    self.outp = "playing a media I found on youtube titled:" + ytt["result"][0]["title"]
                    webbrowser.open(ytt["result"][0]["link"])
            else:
                ytt = self.srch.ytsearch(snt)
                print(ytt["result"])
                self.outp = "playing a media I found on youtube titled:" + ytt["result"][0]["title"]
                webbrowser.open(ytt["result"][0]["link"])
        elif sr["name"] == "audio":
            if ir["name"] == "play":
                if ers:
                    plt = ""
                    qry = ""
                    for i in ers:
                        if i["val"] == "media_type":
                            plt = i["word"]
                        elif i["val"] == "song_name":
                            qry += i["word"] + " "
                    if plt == "youtube" or "yt" in plt:
                        ytt = self.srch.ytsearch(qry)
                        print(ytt["result"])
                        self.outp = "playing a video I found on youtube titled:" + ytt["result"][0]["title"]
                        webbrowser.open(ytt["result"][0]["link"])
                else:
                    ytt = self.srch.ytsearch(snt)
                    print(ytt["result"])
                    self.outp = "playing a media I found on youtube titled:" + ytt["result"][0]["title"]
                    webbrowser.open(ytt["result"][0]["link"])
            else:
                self.outp = "audio operations are currently not supported"
        elif sr["name"] == "alarm":
            self.outp = random.choice("This featur is not implemented yet","Currently I cant do that","I am really sorry But I cant do that right now","I am not trained for doing this")
        elif sr["name"] == "music":
            self.outp = random.choice("This featur is not implemented yet","Currently I cant do that","I am really sorry But I cant do that right now","I am not trained for doing this")
        elif sr["name"] == "iot":
            if ir["name"] == "hue_lightdim":
                if self.brt > 20:
                    import wmi
                    c = wmi.WMI(namespace="wmi")
                    self.brt -= 10
                    methods = c.WmiMonitorBrightnessMethods()[0]
                    methods.WmiSetBrightness(self.brt,0)
                    self.outp = "brightness decreased by 10%"
                else:
                    self.outp = "brightness cant be decreased anymore"
            if ir["name"] == "hue_lightup":
                if self.brt >= 100:
                    self.outp = "brightness cant be increased anymore"
                else:
                    import wmi
                    c = wmi.WMI(namespace="wmi")
                    self.brt += 10
                    methods = c.WmiMonitorBrightnessMethods()[0]
                    methods.WmiSetBrightness(self.brt,0)
                    self.outp = "brightness increased by 10%"                    
        elif sr["name"] == "list":
            self.outp = random.choice("This featur is not implemented yet","Currently I cant do that","I am really sorry But I cant do that right now","I am not trained for doing this")
        elif sr["name"] == "takeaway":
            self.outp = "I currently cant do that"
        elif sr["name"] == "recommendation":
            self.outp = "I found thses recomendation on google"
            webbrowser.open("https://www.google/search?q="+snt.replace(" ","+")+"&aqs=chrome.0.69i59j69i60.9117j0j7&sourceid=chrome&ie=UTF-8")
        elif sr["name"] == "cooking":
            self.outp = random.choice("This feature is not implemented yet","Currently I cant do that","I am really sorry But I cant do that right now","I am not trained for doing this")
        self.last_outp = self.outp  
if __name__ == "__main__":
    Charlie().run()