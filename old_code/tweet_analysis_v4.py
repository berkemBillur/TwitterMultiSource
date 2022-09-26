from textblob import TextBlob
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import os.path
import nltk
import nltk.data
import time
import string
from datasets import load_dataset

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
import json
import pickle
import joblib

import language_tool_python
import preprocessor as p

import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import readability

## DATA
from datasets import Dataset

### POLITENESS
from politeness.polite_script import *

### topic modelling
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import expit

### irony
import urllib.request
from scipy.special import softmax
import csv

## offensiveness
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier

#hate
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet

# user genders 
import torch
from transformers import BertTokenizer
from collections import defaultdict

nltk.download('omw-1.4')

# if 'google.colab' in str(get_ipython()):
#   print('Running on CoLab')
# else:
#   print('Not running on CoLab')
#   os.chdir('G:\My Drive\MSc_project\.MAIN\offensiveness')


class Analyzer(object):
    def __init__(self, hashtag):

        self.hashtag = hashtag
        self.save_path = f'informer_results{os.path.sep}{hashtag}'
        self.tool = language_tool_python.LanguageTool('en-US')
        self.max_len = 160

    def get_device(self):
        if torch.cuda.is_available():    

            # Tell PyTorch to use the GPU.    
            self.device = torch.device("cuda")

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            print('We will use the GPU:', torch.cuda.get_device_name(0))

        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")


    def load_informer_data(self):
        path = f'tweets{os.path.sep}{self.hashtag}{os.path.sep}{self.hashtag}_ms_cases.json'
        with open(path) as jf:
            data = json.load(jf)
        return data

    def load_user_feeds(self):
        path = f'tweets/{self.hashtag}/100_feeds'
        jsons = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
        all_js = {}
        for file in jsons:
            with open(os.path.join(f'{path}/' + file)) as jf:
                all_js = { **all_js, **json.load(jf) }
        print(f'pulled data on {len(all_js)} users')
        return all_js


    @staticmethod
    def get_the_tweets(database):
        all_tweets = {}
        for key,value in database.items():
            #store tweets by tweet id
            all_tweets.update( {key:{'text':value['tweet-text'],'user_id':value['user-id'],'tweet_id':key}} )

            infector = value['infector-info']
            i = [k for k in infector]
            infector = infector[i[0]]
            all_tweets.update( {i[0]:{'text':infector['tweet-text'],'user_id':infector['user-id'],'tweet_id':infector['id']}} )

            for informer in value['informers-data']:
                all_tweets.update({informer['id']:informer['tweet-text']})
                all_tweets.update( {informer['id']:{'text':informer['tweet-text'],'user_id':informer['user-id'],'tweet_id':informer['id']}} )
        return all_tweets

    @staticmethod
    def store_by_tweets(database):
        all_tweets = {}
        for key,value in database.items():
            if value in all_tweets:
                new = all_tweets[value].append(key)
                all_tweets[value] = new
            else:
                all_tweets[value] = [key]

        return all_tweets

    @staticmethod
    def get_users(database):
        users = {}
        for key,value in database.items():
            users.update( { value['user-id']:{'description': value['description'], 'feed':[]} } )
            infector = value['infector-info']
            i = [k for k in infector]
            infector = infector[i[0]]
            users.update(  { infector['user-id']:{'description': infector['description'],'feed':[] } } )
            for informer in value['informers-data']:
                users.update( { informer['user-id']:{'description': informer['description'],'feed':[] } } )
        return users

    def add_feeds(self,users):
        feeds = self.load_user_feeds()
        pulled_feeds = feeds.keys()
        users_got = users.keys()
        users_needed = list(set(pulled_feeds) & set(users_got))
        for id in users_needed:
            users[id]['feed'] = feeds[id]
        return users

    @staticmethod
    def sort_by_tweet(all_tweets): 

        df = pd.DataFrame.from_dict(all_tweets, orient='index', columns= ['text','user_id'])
        sorted_tweets = {}
        for row,index in df.groupby('text').groups.items():
            key = tuple(index.values.tolist())
            sorted_tweets.update({key:row})

        new_df = pd.DataFrame.from_dict(sorted_tweets, orient='index', columns= ['text'])
        
        return new_df



###########################################
#######         PREPROCESSING       #######
###########################################

        
    def tweet_cleaner(self,tw_list):
        remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
        rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
        hash = lambda x: re.sub(r'#', "", x)
        amp = lambda x: re.sub(r'&amp', "", x)


        tw_list['grammartext'] = tw_list.text.map(remove_rt).map(rt)
        tw_list['clean_text'] = tw_list.text.map(remove_rt).map(rt)
        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)
        tw_list["grammartext"] = tw_list.grammartext.map(p.clean)
        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.NUMBER)
        tw_list["clean_text"] = tw_list.clean_text.map(p.clean).map(hash).map(amp)
        tw_list["clean_text"] = tw_list.clean_text.str.lower()

        #Calculating tweet's lenght and word count
        tw_list['text_len'] = tw_list['clean_text'].astype(str).apply(len)
        tw_list['text_word_count'] = tw_list['clean_text'].apply(lambda x: len(str(x).split()))
        tw_list['punct'] = tw_list['clean_text'].apply(lambda x: self.remove_punct(x))
        tw_list['tokenized'] = tw_list['punct'].apply(lambda x: self.tokenization(x.lower()))
        tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: self.remove_stopwords(x))
        tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: self.stemming(x))
        return tw_list

    @staticmethod
    def hugging_preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def remove_punct(self,text):
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0–9]+', '', text)
        return text


    def remove_stopwords(self,text):
        self.stopword = nltk.corpus.stopwords.words('english')
        text = [word for word in text if word not in self.stopword]
        return text

    def stemming(self,text):
        self.ps = nltk.PorterStemmer()
        text = [self.ps.stem(word) for word in text]
        return text

    def clean_text(self,text):
        text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
        text_rc = re.sub('[0-9]+', '', text_lc)
        tokens = re.split('\W+', text_rc)    # tokenization
        text = [self.ps.stem(word) for word in tokens if word not in self.stopword]  # remove stopwords and stemming
        return text

    @staticmethod
    def tokenization(text):
        text = re.split('\W+', text)
        return text

################################################################################################
################################################################################################
############################                 METRICS                ############################
################################################################################################
################################################################################################

    @staticmethod
    def get_sentiment(df,index,row):
        score = SentimentIntensityAnalyzer().polarity_scores(row.clean_text)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        if neg > pos:
            df.loc[index, 'sentiment'] = "negative"
        elif pos > neg:
            df.loc[index, 'sentiment'] = "positive"
        else:
            df.loc[index, 'sentiment'] = "neutral"
        df.loc[index, 'neg'] = neg
        df.loc[index, 'neu'] = neu
        df.loc[index, 'pos'] = pos
        df.loc[index, 'compound'] = comp

    def get_grammar(self,df,index, row):
        # https://michaeljanz-data.science/deepllearning/natural-language-processing/scoring-texts-by-their-grammar-in-python/
        scores_word_based_sentence = []
        scores_sentence_based_sentence = []
        s1 = time.perf_counter()
        sentences = nltk.tokenize.sent_tokenize(row.grammartext)
        e1 = time.perf_counter()
        # sentences = self.split_into_sentences(row)
        for sentence in sentences:
        # for sentence in helpers.text_to_sentences(text):
            matches = self.tool.check(sentence)
            count_errors = len(matches)
            # only check if the sentence is correct or not
            scores_sentence_based_sentence.append(np.min([count_errors, 1]))
            scores_word_based_sentence.append(count_errors)
            
        word_count = len(nltk.tokenize.word_tokenize(row.grammartext))
        sum_count_errors_word_based = np.sum(scores_word_based_sentence)
        score_word_based = 1 - (sum_count_errors_word_based / word_count)
        
        sentence_count = len(sentences)       
        sum_count_errors_sentence_based = np.sum(scores_sentence_based_sentence)
        score_sentence_based = 1 - np.sum(sum_count_errors_sentence_based / sentence_count)

        df.loc[index, 'grammar-word-scoure'] = score_word_based
        df.loc[index, 'grammar-sentence-score'] = score_sentence_based

    @staticmethod
    def get_readability(df,index,row):
        results = readability.getmeasures(row.tokenized,lang='en')
        # readability grades
        df.loc[index, 'Kincaid'] = results['readability grades']['Kincaid']
        df.loc[index, 'ARI'] = results['readability grades']['ARI']
        df.loc[index, 'Coleman-Liau'] = results['readability grades']['Coleman-Liau']
        df.loc[index, 'FleschReadingEase'] = results['readability grades']['FleschReadingEase']
        df.loc[index, 'GunningFogIndex'] = results['readability grades']['GunningFogIndex']
        df.loc[index, 'LIX'] = results['readability grades']['LIX']
        df.loc[index, 'SMOGIndex'] = results['readability grades']['SMOGIndex']
        df.loc[index, 'RIX'] = results['readability grades']['RIX']
        df.loc[index, 'DaleChallIndex'] = results['readability grades']['DaleChallIndex']
        # sentence info
        # self.df.loc[index,'characters_per_word'] = results['sentence info']['characters_per_word']
        # self.df.loc[index,'syll_per_word'] = results['sentence info']['syll_per_word']
        # self.df.loc[index,'words_per_sentence'] = results['sentence info']['words_per_sentence']
        # self.df.loc[index,'sentences_per_paragraph'] = results['sentence info']['sentences_per_paragraph']
        # self.df.loc[index,'type_token_ratio'] = results['sentence info']['type_token_ratio']
        # self.df.loc[index,'characters'] = results['sentence info']['characters']
        # self.df.loc[index,'syllables'] = results['sentence info']['syllables']
        # self.df.loc[index,'words'] = results['sentence info']['words']
        # self.df.loc[index,'wordtypes'] = results['sentence info']['wordtypes']
        # self.df.loc[index,'long_words'] = results['sentence info']['long_words']
        df.loc[index,'complex_words'] = results['sentence info']['complex_words']
        df.loc[index,'complex_words_dc'] = results['sentence info']['complex_words_dc']


    def get_topic(self,df,index,row):
        # tokens = self.topic_tokenizer(row.clean_text,return_tensors='pt')
        output = self.topic_model(**row.topic_tokens)
        scores = output[0][0].detach().numpy()
        scores = expit(scores)
        pred = np.argmax(scores)

        df.loc[index,'topic'] = pred
        for i in range(18):
            label = str(self.topic_classes[i])
            df.loc[index, label] = scores[i]

    def get_topic_single(self,df,index,row):
        # tokens = self.topic_tokenizer(row.clean_text,return_tensors='pt')
        output = self.topic_model(**row.topic_tokens)
        scores = output[0][0].detach().numpy()
        scores = expit(scores)
        pred = np.argmax(scores)

        df.loc[index,'topic_single'] = pred
        for i in range(6):
            label = str(self.topic_classes[i])
            df.loc[index, label] = scores[i]

    def get_politeness(self,df,index,row):
        df.loc[index, 'politeness'] = row.politeness    

    def get_offensive(self,df,index,row):
        df.loc[index, 'offensive'] = row.offensive  

    def get_irony(self,df,index,row):
        output = self.irony_model(**row.cardiff_tokens)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        df.loc[index, 'irony'] = ranking[0]

    def get_emoji(self,df,index,row):
        output = self.emoji_model(**row.cardiff_tokens)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        df.loc[index, 'emoji'] = ranking[0] 

    def get_hate(self,df,index,row):
        for i in range(3):
            df.loc[index, self.hate_labels[i]] = row.hate_output.probas[self.hate_labels[i]]

    def get_emotion(self,df,index,row):
        for i in range(6):
            df.loc[index, self.emo_labels[i]] = row.emo_output.probas[self.emo_labels[i]]

    @staticmethod
    def get_gender_model(df):
        path = 'user_gender_class/model/logistic_gender'
        mod = joblib.load(path)
        predictions = mod.predict(df)
        return predictions


####################################################################################################
############################                 LOAD MODELS            ################################
####################################################################################################

    def load_gender_model(self,tweet_df):
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        stop_words.extend(['u', 'wa', 'ha', 'would', 'com'])

        print('starting user gender classification')

        remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
        rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)

        print('now cleaning')

        p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.NUMBER)
        self.user_feeds['clean_description'] = self.user_feeds.description.map(remove_rt).map(rt).map(p.clean).str.lower()
        self.user_feeds['clean_text'] = self.user_feeds.text.map(remove_rt).map(rt).map(p.clean).str.lower()
        n = len(self.user_feeds)

        # call the user feeds df to df just for ease
        df = self.user_feeds

        print(df.columns)
        df['sep'] = ['.' for i in range(n)]
        df['txt'] = df['clean_description'] + df['sep'] + df['clean_text']
        df['txt'] = df['txt'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

        user_info = df.txt
        print('finished cleaning')

        print('now predicting the gender of each tweet and descrption')
        text_predictions = self.get_gender_model(user_info)
        print('model finished predicting gender')

        df['gender'] = text_predictions
        male_txt = df[df['gender']==1]
        female_txt = df[df['gender']!=1]
        print('male tweets')
        print(len(male_txt))
        print('female tweets')
        print(len(female_txt))

        # both these contain the users feeds and their feeds
        self.user_feeds = df


        del self.user_feeds['sep']
        del self.user_feeds['clean_text']
        del self.user_feeds['clean_description']

        user_ids = [k for k in self.all_users]
        self.df.set_index('user_id')

        self.df['gender'] = np.nan
        self.df['num_male'] = np.nan
        self.df['num_female'] = np.nan



        print(user_ids)
        for id in user_ids:
            info_user = df[df['user_id']==id]
            if info_user.empty:
                print('dont have useres feeds')
            else:
                gen = info_user['gender'].mode().values[0]
                self.df.loc[id,'gender'] = gen
                self.df.loc[id,'num_male'] = len(info_user[info_user['gender']==1])
                self.df.loc[id,'num_female'] = len(info_user[info_user['gender']!=0])
                
        self.df.set_index('tweet_id')

        print('gender results')
        male_usr = self.df[self.df['gender']==1]
        female_usr = self.df[self.df['gender']!=1]
        print('male users')
        print(len(male_usr))
        print('female users')
        print(len(female_usr))
        print('finished gender')
        print('---------\n---------\n')
        

    def load_topic_model(self,tweet_df):

        MODEL = f"cardiffnlp/tweet-topic-21-multi"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        tokens = tweet_df.text.apply(lambda row: tokenizer(row, return_tensors='pt'))
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        self.topic_classes = self.topic_model.config.id2label

        s_MODEL = f"cardiffnlp/tweet-topic-21-single"
        self.topic_model_single = AutoModelForSequenceClassification.from_pretrained(s_MODEL)
        self.topic_classes_single = self.topic_model_single.config.id2label

        tweet_df['topic_tokens'] = tokens
        print('loaded topic model')

        print('now loading the topic tokens for all user feeds')
        feed_tokens = tweet_df.text.apply(lambda row: tokenizer(row, return_tensors='pt'))
        self.user_feeds['topic_tokens'] = feed_tokens
        print('---------\n---------\n')
        
        

    @staticmethod
    def run_offensive_model(test):
        current = os.getcwd()
        new_dir = current+'/offensiveness'

        os.chdir(new_dir)
        df_scraped = pd.read_csv('labeled_tweets.csv')
        df_public = pd.read_csv('public_data_labeled.csv')
        df_scraped.drop_duplicates(inplace = True)
        df_scraped.drop('id', axis = 'columns', inplace = True)
        df_public.drop_duplicates(inplace = True)
        df = pd.concat([df_scraped, df_public])
        df['label'] = df.label.map({'Offensive': 1, 'Non-offensive': 0})
        X_train, X_test, y_train, y_test = train_test_split(df['full_text'], 
                                                    df['label'], 
                                                    random_state=42)

        os.chdir(current)
        # Instantiate the CountVectorizer method
        count_vector = CountVectorizer(stop_words = 'english', lowercase = True)

        # Fit the training data and then return the matrix
        training_data = count_vector.fit_transform(X_train)
        testing_data = count_vector.transform(test)
        model = SGDClassifier()
        model.fit(training_data, y_train)
        preds = model.predict(testing_data)
        return preds
    

    def load_offensive_model(self,tweet_df):
        test_data = tweet_df.text
        preds = self.run_offensive_model(test_data)
        tweet_df['offensive'] = preds
        print('loaded offensive model')

        print('loading offensive model for users')
        feed_test = self.user_feeds.text
        feed_preds = self.run_offensive_model(feed_test)
        self.user_feeds['offensive'] = feed_preds
        print('---------\n---------\n')



    def load_irony_model(self,tweet_df):
        task='irony'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokens = tweet_df.text.apply(lambda row: tokenizer(row, return_tensors='pt'))
        # download label mapping
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        self.irony_labels = [row[1] for row in csvreader if len(row) > 1]
        # PT
        self.irony_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        tweet_df['cardiff_tokens'] = tokens
        print('loaded irony model')

        print('loading hugging face tokens')
        feed_tokens = self.user_feeds.text.apply(lambda row: tokenizer(row, return_tensors='pt'))
        self.user_feeds['cardiff_tokens'] = feed_tokens
        print('---------\n---------\n')



    def load_emoji_model(self,tweet_df):
        task='emoji'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokens = tweet_df.text.apply(lambda row: tokenizer(row, return_tensors='pt'))
        # download label mapping
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        self.emoji_labels = [row[1] for row in csvreader if len(row) > 1]

        self.emoji_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        print('loaded emoji model')
        print('---------\n---------\n')

    def load_politeness(self,tweet_df):
        ## Scoring each tweet based on politeness 
        class SimpleDataset:
            def __init__(self, tokenized_texts):
                self.tokenized_texts = tokenized_texts
            
            def __len__(self):
                return len(self.tokenized_texts["input_ids"])
            
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.tokenized_texts.items()}
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


        # Tokenize same way as training data
        model_name = 'roberta-base'
        path = f'politeness{os.path.sep}results/checkpoint-52500/'
        print('loaded politeness')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print('politeness per tweet')
        model = AutoModelForSequenceClassification.from_pretrained(path)
        tweets = tweet_df.grammartext.tolist()
        test_encodings = tokenizer(tweets , truncation=True, padding=True, max_length=256)
        test_dataset = SimpleDataset(test_encodings)       


        trainer = Trainer(model=model)
        predictions = trainer.predict(test_dataset)
        tweet_df['politeness'] = predictions[0]


        print('politeness per user')
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(self.device)
        feeds = self.user_feeds.text.tolist()
        feed_encodings = tokenizer(feeds , truncation=True, padding=True, max_length=256)
        print('tokenized feeds')
        feed_dataset = SimpleDataset(feed_encodings)       

        print('now predicting')
        feed_trainer = Trainer(model=model)
        feed_predictions = feed_trainer.predict(feed_dataset)
        self.user_feeds['politeness'] = feed_predictions[0]
        
        print('loaded politeness model for users')
        print('---------\n---------\n')

        

    def load_psysentimento_model(self,tweet_df):
            
        tweets = tweet_df.text.to_list()

        
        # hateful
        analyzer = create_analyzer(task="hate_speech", lang="en")
        self.hate_labels = ['hateful', 'targeted', 'aggressive']


        predictions = [ analyzer.predict(preprocess_tweet(txt)) for txt in tweets ]
        print('predicted hate of tweets')
        # predictions = process_(analyzer,tweets)
        tweet_df['hate_output'] = predictions

        
        print('loaded hate model')

        #emotion
        e_analyzer = create_analyzer(task="emotion", lang="en")
        self.emo_labels = ['joy','sadness','others','anger','surprise','disgust','fear']     

        # e_predictions = process_(e_analyzer,tweets)
        e_predictions = [ e_analyzer.predict(preprocess_tweet(txt)) for txt in tweets ]
        print('predicted emotion of tweets')
        tweet_df['emo_output'] = e_predictions

        print('loaded emotion model')


        feeds = self.user_feeds.text.to_list()
        print('loading psysentimento for user feeds')

        print('tokenizing and predicting user feeds hate')
        # fd_hate_pred = process_(analyzer,feeds)
        fd_hate_pred = [ analyzer.predict(preprocess_tweet(txt)) for txt in feeds ]
        self.user_feeds['hate_output'] = fd_hate_pred
        print('predicted hate of feeds')

        print('tokenizing and predicting user feeds emotion')
        # fd_e_predictions = process_(e_analyzer,feeds)
        fd_e_predictions = [ e_analyzer.predict(preprocess_tweet(txt)) for txt in feeds ]
        self.user_feeds['emo_output'] = fd_e_predictions
        print('predicted emotion of feeds')



#######
# MAIN FUNCTION TO RUN THE ANALYSIS
########


    def tweet_analysis(self):

        informer_db = self.load_informer_data()
        self.get_device()

        all_tweets = self.get_the_tweets(informer_db)
        all_users = self.get_users(informer_db)
        self.all_users = self.add_feeds(all_users)
        
        all_tweets_data = [ {'user_id':key, 'description':value['description'],'text':tweet['tweet-text']} for key,value in all_users.items() for tweet in value['feed']  ]
        self.user_feeds = pd.DataFrame(all_tweets_data, columns = ['user_id','description', 'text'])
        print(f'have {len(all_tweets_data)} tweets users feeds!!')

        all_descriptions = [ {'user_id':key, 'description':value['description'] } for key,value in all_users.items() ]
        self.user_descriptions = pd.DataFrame(all_descriptions, columns = ['user_id','description'])
        self.user_descriptions = self.user_descriptions[self.user_descriptions['description'].notna()]

        # self.df contains all the tweets but is sorted by tweet id. therefore contains many duplicates as the same tweet is retweeted i think
        # self.df = pd.DataFrame.from_dict(all_tweets)

        df = pd.DataFrame.from_dict(all_tweets, orient='index', columns= ['text','user_id','tweet_id'])
        self.df = self.tweet_cleaner(df)

        self.df[['polarity', 'subjectivity']] = self.df['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

        tweet_df = self.df.copy()

        ##########################################################################
        ##########################################################################
        ### LOADING IN NECESSARY MODELS

        # self.load_gender_model(tweet_df)

        # self.df.to_csv('user_gender_class/trial_output.csv')

        # self.load_politeness(tweet_df)

        # self.load_topic_model(tweet_df)

        # self.load_irony_model(tweet_df)

        # self.load_offensive_model(tweet_df)

        # self.load_emoji_model(tweet_df)

        # self.load_psysentimento_model(tweet_df)

        ###############################
        # CLASSIFYING EACH TWEET

        # index the rows of the database that contains the unique tweets
        for index, row in tweet_df.iterrows():
            self.get_sentiment(self.df,index,row)
            self.get_grammar(self.df,index,row)
            self.get_readability(self.df,index,row)
            # self.get_politeness(self.df,index,row)
            # self.get_offensive(self.df,index,row)
            # self.get_topic(self.df,index,row)
            # self.get_irony(self.df,index,row)
            # self.get_emoji(self.df,index,row)
            # self.get_emotion(self.df,index,row)
            # self.get_hate(self.df,index,row)

        print('finished classifying all the tweets')
        #################################
        #################################


        # CLASSIFY EACH USER
        # index the rows of the database that contains the unique tweets
        for index, row in self.user_feeds.iterrows():
            self.get_sentiment(self.user_feeds,index,row)
            self.get_grammar(self.user_feeds,index,row)
            self.get_readability(self.user_feeds,index,row)
            self.get_politeness(self.user_feeds,index,row)
            self.get_offensive(self.user_feeds,index,row)
            self.get_topic(self.user_feeds,index,row)
            self.get_irony(self.user_feeds,index,row)
            self.get_emoji(self.user_feeds,index,row)
            self.get_emotion(self.user_feeds,index,row)
            self.get_hate(self.user_feeds,index,row)


        ###############################################################
        ###############################################################
        # AVERAGE FOR EACH USERS FEED!!!!!
        
        
                

        print('finished getting the metrics!!!!!!!')

        df_path = f'{self.save_path}_emotion_classification.csv'
        self.df.to_csv(df_path)
        
        # refilter the database to only contain the unique tweets
        unique_df = self.df.copy()
        unique_df.drop_duplicates('text', inplace=True)


hashtags = ['avengers']

informer_db_path = f'informer_results{os.path.sep}'

for hashtag in hashtags:

    a = Analyzer(hashtag)

    a.tweet_analysis()