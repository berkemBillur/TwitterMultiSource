from turtle import done
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

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
import json
import pickle
import joblib
import torch

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


### topic modelling
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import expit

#hate
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet

#politeness
from politeness.polite_script import *

# grammar
import language_tool_python

nltk.download('omw-1.4')


class Analyzer(object):
    def __init__(self, hashtag):

        self.hashtag = hashtag
        self.tool = language_tool_python.LanguageTool('en-US')

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
        from itertools import islice
        
        def take(n, iterable):
            "Return first n items of the iterable as a list"
            return list(islice(iterable, n))

        path = f'tweets{os.path.sep}{self.hashtag}{os.path.sep}{self.hashtag}_ms_cases.json'
        with open(path) as jf:
            data = json.load(jf)
        if len(data.keys()) > 1500:  
            print('only getting user feeds for 1.5k of our tweets. we must shorten our subset')
            return dict(take(1500,data.items()))
        else:
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
            all_tweets.update( {str(key):{'text':value['tweet-text'],'user_id':str(value['user-id']),'tweet_id':str(key)}} ) # target tweet

            inf = value['infector-info']
            k = list(inf.keys())[0]
            infector = inf[k]

            all_tweets.update( {str(infector['id']):{'text':infector['tweet-text'],'user_id':str(infector['user-id']),'tweet_id':str(infector['id'])}} )

            for i,informer in enumerate(value['informers-data']):
                if i <5:
                    all_tweets.update( {str(informer['id']):{'text':informer['tweet-text'],'user_id':str(informer['user-id']),'tweet_id':str(informer['id'])}} )
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
            users.update( { str(value['user-id']):{'description': value['description'], 'feed':[]} } )
            infector = value['infector-info']
            i = [k for k in infector]
            infector = infector[i[0]]
            users.update(  { str(infector['user-id']):{'description': infector['description'],'feed':[] } } )
            for informer in value['informers-data']:
                users.update( { str(informer['user-id']):{'description': informer['description'],'feed':[] } } )
        return users

    def add_feeds(self,users):
        feeds = self.load_user_feeds()
        pulled_feeds = feeds.keys()
        users_got = users.keys()
        users_needed = list(set(pulled_feeds) & set(users_got))
        tweet_ids = []
        for id in users_needed:
            users[id]['feed'] = feeds[id][0:10]
            tweet_ids.extend( tw['id'] for tw in feeds[id][0:10]  )
        return users,tweet_ids

    @staticmethod
    def sort_by_tweet(all_tweets): 

        df = pd.DataFrame.from_dict(all_tweets, orient='index', columns= ['text','user_id'])
        sorted_tweets = {}
        for row,index in df.groupby('text').groups.items():
            key = tuple(index.values.tolist())
            sorted_tweets.update({key:row})

        new_df = pd.DataFrame.from_dict(sorted_tweets, orient='index', columns= ['text'])
        
        return new_df

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
        return tw_list

    def remove_punct(self,text):
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0–9]+', '', text)
        return text


    def clean_text(self,text):
        text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
        text_rc = re.sub('[0-9]+', '', text_lc)
        tokens = re.split('\W+', text_rc)    # tokenization
        text = [self.ps.stem(word) for word in tokens if word not in self.stopword]  # remove stopwords and stemming
        return text



################################################################################################
################################################################################################
############################                 METRICS                ############################
################################################################################################
################################################################################################

    def get_grammar(self,row):
        # https://michaeljanz-data.science/deepllearning/natural-language-processing/scoring-texts-by-their-grammar-in-python/
        scores_word_based_sentence = []
        scores_sentence_based_sentence = []
        s1 = time.perf_counter()
        sentences = nltk.tokenize.sent_tokenize(row)
        e1 = time.perf_counter()
        # sentences = self.split_into_sentences(row)
        for sentence in sentences:
        # for sentence in helpers.text_to_sentences(text):
            matches = self.tool.check(sentence)
            count_errors = len(matches)
            # only check if the sentence is correct or not
            scores_sentence_based_sentence.append(np.min([count_errors, 1]))
            scores_word_based_sentence.append(count_errors)
            
        word_count = len(nltk.tokenize.word_tokenize(row))
        sum_count_errors_word_based = np.sum(scores_word_based_sentence)
        if word_count == 0:
            score = 0
        else: 
            score = 1 - (sum_count_errors_word_based / word_count)

        return score


    @staticmethod
    def get_readability(row):
        if not row:
            return [0]*23
            # print('sentence has no real text')
        else:
            results = readability.getmeasures(row,lang='en')
            # [ df.loc[index, score] = results['readability grades'][score] for score in \
            #  ['Kincaid','ARI', 'Coleman-Liau', 'FleschReadingEase', 'GunningFogIndex', \
            #   'LIX', 'SMOGIndex', 'RIX', 'DaleChallIndex'] ]

            return [ grade[t] for grade in [results['readability grades'], results['sentence info'] ] for t in grade ]

    @staticmethod
    def get_sentiment(row):
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        if neg > pos:
            
            label = "negative"
        elif pos > neg:
            
            label = "positive"
        else:
            
            label = "neutral"
        return [label, neg,neu,pos, comp]


    def get_topic(self,row):
        # tokens = self.topic_tokenizer(row.clean_text,return_tensors='pt')
        output = self.topic_model(**row.to(self.device))
        scores = output[0][0].detach().cpu().numpy()
        scores = expit(scores)
        pred = np.argmax(scores)
        return [pred] + scores.flatten().tolist()


    def get_hate(self,row):
        return [row.probas[self.hate_labels[i]] for i in range(3) ]

    def get_emo(self,row):
        return [row.probas[self.emo_labels[i]] for i in range(7) ]


####################################################################################################
############################                 LOAD MODELS            ################################
####################################################################################################

    def load_topic_model(self,user_feeds):

        MODEL = f"cardiffnlp/tweet-topic-21-multi"

        with torch.no_grad():
            self.topic_model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.device)

        self.topic_classes = self.topic_model.config.id2label#

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokens = user_feeds.clean_text.apply(lambda row: tokenizer(row, return_tensors='pt'))

        print('loaded topic model')

        return tokens


    def load_psysentimento_model(self, user_feeds):
            
        tweets = user_feeds.clean_text.to_list()

        # hateful
        analyzer = create_analyzer(task="hate_speech", lang="en")
        self.hate_labels = ['hateful', 'targeted', 'aggressive']

        hate_out = [ analyzer.predict(preprocess_tweet(txt)) for txt in tweets ]
        print('predicted hate of tweets')

        
        print('loaded hate model')

        #emotion
        e_analyzer = create_analyzer(task="emotion", lang="en")
        self.emo_labels = ['joy','sadness','others','anger','surprise','disgust','fear']     

        # e_predictions = process_(e_analyzer,tweets)
        emo_out = [ e_analyzer.predict(preprocess_tweet(txt)) for txt in tweets ]
        print('predicted emotion of tweets')

        print('loaded emotion model')

        return hate_out, emo_out



    def load_politeness_model(self, user_feeds):
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
        with torch.no_grad():
            model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)

        t1 = time.time()
        print('NOW PREDICTING POLITENESS SCORES OF USER FEEDS\n----')

        tweets = user_feeds.grammartext.tolist()
        test_encodings = tokenizer(tweets , truncation=True, padding=True, max_length=256)
        test_dataset = SimpleDataset(test_encodings)       


        trainer = Trainer(model=model)
        predictions = trainer.predict(test_dataset)

        t2 = time.time()

        print(f'got prediction scores in {(t2-t1)/3600} hours')
        
        return predictions[0]

        print('loaded politeness model for users')
        print('---------\n---------\n')

#######################################
# MAIN FUNCTION TO RUN THE ANALYSIS
########################################


    def tweet_analysis(self):

        informer_db = self.load_informer_data()
        self.get_device()

        print('loaded informer data')

        #####################
        # LOAD IN THE DATA!!!!!!!
        #####################

        # ALL THE TWEETS IN THE MULTI-SOURCE CASE - store by tweet id
        all_tweets = self.get_the_tweets(informer_db)
        tweet_df = pd.DataFrame.from_dict(all_tweets, orient='index')
        tweet_df.drop_duplicates()
        user_ids = tweet_df.user_id.copy().tolist()
        print('loaded in ms tweets')

        # ALL THE USER FEEDS!!!!! STORED BY THE USER ID !!!!!
        all_users = self.get_users(informer_db)
        all_users,tweet_ids_feeds = self.add_feeds(all_users)
        ## ALL USER FEEDS!! STORED BY THE TWEET ID!! WILL!!!
        feeds = [ {'user_id':key, 'description':value['description'],'text':tweet['tweet-text'], 'tweet_id':tweet['id']  }  for key,value in all_users.items() for tweet in value['feed']  ]
        user_feeds = pd.DataFrame(feeds)
        user_feeds['tweet_ids'] = tweet_ids_feeds
        user_feeds.set_index('tweet_ids', inplace=True)
        print(f'\n\n\n raw feed len {len(user_feeds)}')
        user_feeds.drop_duplicates()
        print(f'\n\n\n feed len after drop duplibats {len(user_feeds)}' )
        user_feeds = user_feeds.loc[user_feeds['user_id'].isin(user_ids)]
        print(f'\n\n\n feed len after not considering the feeds we didnt pull {len(user_feeds)}' )

        print('loaded in user feeds')


        print('loaded in necessary data')

        ##################
        # DROPPING DUPLICATE TWEETS FROM BOTH DATAFRAMES....
        
        print(f'have {len(tweet_df)} tweets loaded in')
        tweet_df = tweet_df.copy()[~tweet_df.index.duplicated(keep='first')]
        print(f'now considering {len(tweet_df)} tweets')


        # ## SUBSET TESTING

        tweet_df = tweet_df.iloc[0:3]
        user_ids = tweet_df['user_id'].tolist()
        print('reduced df for testing\n')
        user_feeds = user_feeds.loc[user_feeds['user_id'].isin(user_ids)]
        print('loading test data')


        # print('loading test data')
        ## END
        ##########################################
        
        ###############################
        ###############################
        ### LOADING IN NECESSARY MODELS

        user_feeds = self.tweet_cleaner(user_feeds)

        topic_in = self.load_topic_model(user_feeds)
        hate, emo = self.load_psysentimento_model(user_feeds)
        polite_out = self.load_politeness_model(user_feeds)

        # readability metrics
        read_cols = ['Kincaid', 'ARI', 'Coleman-Liau', 'FleschReadingEase', 'GunningFogIndex', 
                     'LIX', 'SMOGIndex', 'RIX', 'DaleChallIndex','characters_per_word', 'syll_per_word', 
                     'words_per_sentence', 'sentences_per_paragraph', 'type_token_ratio', 'characters', 'syllables', 
                     'words', 'wordtypes', 'sentences', 'paragraphs', 'long_words', 'complex_words', 'complex_words_dc']    



        ##########################################
        ########################################################################
        ### SCORE EVERY FEED FOR USERS!!!!!!

        print('NOW SCORING EACH TWEET IN USER FEED')
        tall= time.time()
        # [ func(user_feeds_df,idx,tweet) for func in funcs for tweet, idx in user_feeds_dum.groupby('text').iterrows() ]

        sent = user_feeds.copy().clean_text.to_list()
        grammer_in = user_feeds.grammartext.tolist()
        read_in = user_feeds.clean_text.tolist()
        
        t1 = time.time()
        sent_output = [ self.get_sentiment(row) for row in sent]
        print(f'got sentiment: {time.time()-t1} seconds')

        t1 = time.time()
        topic_output = [ self.get_topic(row) for row in topic_in]
        print(f'got topic: {time.time()-t1} seconds')

        t1 = time.time()
        hate_output = [ self.get_hate(row) for row in hate]
        print(f'got hate: {time.time()-t1} seconds')

        hate_output = abs(np.array(hate_output) - np.mean(hate_output, axis=0)).tolist() # fix error in hate classification
        hate_output = abs(np.array(hate_output) - np.mean(hate_output, axis=0)).tolist()
        t1 = time.time()
        emo_output = [ self.get_emo(row) for row in emo]
        print(f'got emo: {time.time()-t1} seconds')

        t1 = time.time()
        grammar_out = [ self.get_grammar(row) for row in grammer_in]
        print(f'got grammar scores in {(time.time()-t1)/60} minutes')

        t1 = time.time()
        read_out = [ self.get_readability(row) for row in read_in]
        print(f'got readability scores in {(time.time()-t1)/60} minutes')

        t2 = time.time()
        print(f'time to score tweets: {t2-tall}')

        
        # PLACING ALL INTO THE USER FEEDS DF!!!!!!
        all = np.hstack((sent_output,topic_output,hate_output,emo_output, grammar_out, read_out, polite_out))

        topics = [ val for _,val in self.topic_classes.items()]


        cols = ['sentiment','neg','neu','pos','comp','topic'] + topics + self.hate_labels + self.emo_labels + ['grammar-score'] + read_cols + ['politeness']

        # all = list(zip(np.array(sent_output).T.tolist(),np.array(topic_output).T.tolist(),np.array(hate_output).T.tolist()))

        user_feeds = pd.DataFrame(all.tolist())
        user_feeds.columns = cols

        user_feeds['tweet_id'] = list(user_feeds_df.index.values)
        user_feeds['user_id'] = list(user_feeds_df.user_id.tolist())

        # making sure these are floats
        int_cols = ['neg','neu','pos','comp'] + topics + self.hate_labels + self.emo_labels + ['grammar-score'] + read_cols + ['politeness']

        user_feeds[int_cols] = user_feeds.copy()[int_cols].astype(float)

    
        del user_feeds_df # delete the old user_feeds_df
        ########################################################################
        ########################################################################
        ### AVERAGE PER USER SCORE


        tweet_df = tweet_df[ tweet_df['user_id'].notna()]
        user_feeds = user_feeds[ user_feeds['user_id'].notna()]

        fid = list(set(user_feeds.user_id.tolist()))

        tweet_df = tweet_df.copy()[tweet_df['user_id'].isin(fid)]

        user_ids = tweet_df.user_id.tolist()

        hates = ['hateful', 'targeted', 'aggressive']
        emos = ['joy','sadness','others','anger','surprise','disgust','fear']     

        del self.topic_classes, self.hate_labels, self.emo_labels

        print('now averaging users feeds')
        

        score_cols = ['user_id','neg','neu','pos','comp']+topics + hates + emos + ['grammar-score'] + read_cols + ['politeness']

        score_df = user_feeds[score_cols]

        label_df = user_feeds[['user_id','sentiment','topic']]

        def Average(lst):
            return sum(lst) / len(lst)
          
        import random
        from collections import Counter
        from itertools import groupby
        def cust_mode(l):
            freqs = groupby(Counter(l).most_common(), lambda x:x[1])
            return [val for val,count in next(freqs)[1]]

        all_s = []
        all_mode = []
        all_count = []

        t1= time.time()

        for user in user_ids:

            s_df = score_df[score_df['user_id']==user].drop('user_id',axis=1).values.tolist()
            
            l_df = label_df[label_df['user_id']==user].drop('user_id',axis=1).values.tolist()

            all_s.append([ Average(x) for x in zip(*s_df) ])

            all_mode.append( [ random.choice(cust_mode(x)) for x in zip(*l_df)] )

            counts = [ x for x in zip(*l_df) ] 

            sent_count = [ counts[0].count(s) for s in ['negative','neutral','positive'] ]
            topic_count = [ counts[1].count(str(float(s))) for s in range(19) ]

            all_count.append(sent_count + topic_count)


        t2= time.time()
        print(f'finshed averaging users feed in {(t2-t1)/3600} hours')

        new_cols = [ f'user_{x}_mean' for x in ['neg','neu','pos','comp']+topics + hates + emos + ['grammar-score'] + read_cols + ['politeness'] ]

        tweet_df[new_cols]  = all_s

        new_col = [ f'user_{x}_mode' for x in ['sentiment','topic']]

        tweet_df[new_col] = all_mode

        tweet_df.reset_index(drop=False)
        tweet_df.set_index('tweet_id', inplace = True)
      
        save_path = f'tweets/{self.hashtag}/{self.hashtag}_USER_scores_10_feeds.csv'

        tweet_df.to_csv(save_path)


a = Analyzer('avengers')
a.tweet_analysis()