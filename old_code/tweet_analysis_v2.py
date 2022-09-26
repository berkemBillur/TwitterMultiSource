from enum import unique
from msilib import sequence
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

# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('stopwords')
import json
import pickle

import language_tool_python
import preprocessor as p

import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk import word_tokenize
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

class Analyzer(object):
    def __init__(self, hashtag):

        self.hashtag = hashtag
        self.save_path = f'informer_results{os.path.sep}{hashtag}'

        self.tool = language_tool_python.LanguageTool('en-US')
        self.max_len = 160



    def load_informer_data(self):
        path = f'tweets{os.path.sep}{self.hashtag}{os.path.sep}informer_database.json'
        with open(path) as jf:
            data = json.load(jf)
        return data

    @staticmethod
    def get_the_tweets(database):
        all_tweets = {}
        for key,value in database.items():
            all_tweets.update( {key:value['tweet-text']})
            for informer in value['informers-data']:
                all_tweets.update({informer['id']:informer['tweet-text']})
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

        tw_list['hugging'] = tw_list.text.apply(lambda x: self.hugging_preprocess(x))
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



    def get_sentiment(self,index,row):
        score = SentimentIntensityAnalyzer().polarity_scores(row.clean_text)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        if neg > pos:
            self.df.loc[index, 'sentiment'] = "negative"
        elif pos > neg:
            self.df.loc[index, 'sentiment'] = "positive"
        else:
            self.df.loc[index, 'sentiment'] = "neutral"
        self.df.loc[index, 'neg'] = neg
        self.df.loc[index, 'neu'] = neu
        self.df.loc[index, 'pos'] = pos
        self.df.loc[index, 'compound'] = comp


    def get_grammar(self,index, row):
        # https://michaeljanz-data.science/deepllearning/natural-language-processing/scoring-texts-by-their-grammar-in-python/
        scores_word_based_sentence = []
        scores_sentence_based_sentence = []
        s1 = time.perf_counter()
        sentences = nltk.tokenize.sent_tokenize(row.grammartext)
        e1 = time.perf_counter()
        print(f'tokenizer took {e1-s1}s to complete')
        print(sentences)
        # sentences = self.split_into_sentences(row)
        for sentence in sentences:
        # for sentence in helpers.text_to_sentences(text):
            s2 = time.perf_counter()
            matches = self.tool.check(sentence)
            e2 = time.perf_counter()
            print(f'language tool took {e2-s2}s ')
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

        self.df.loc[index, 'grammar-word-scoure'] = score_word_based
        self.df.loc[index, 'grammar-sentence-score'] = score_sentence_based

    def get_readability(self,index,row):
        results = readability.getmeasures(row.tokenized,lang='en')
        # readability grades
        self.df.loc[index, 'Kincaid'] = results['readability grades']['Kincaid']
        self.df.loc[index, 'ARI'] = results['readability grades']['ARI']
        self.df.loc[index, 'Coleman-Liau'] = results['readability grades']['Coleman-Liau']
        self.df.loc[index, 'FleschReadingEase'] = results['readability grades']['FleschReadingEase']
        self.df.loc[index, 'GunningFogIndex'] = results['readability grades']['GunningFogIndex']
        self.df.loc[index, 'LIX'] = results['readability grades']['LIX']
        self.df.loc[index, 'SMOGIndex'] = results['readability grades']['SMOGIndex']
        self.df.loc[index, 'RIX'] = results['readability grades']['RIX']
        self.df.loc[index, 'DaleChallIndex'] = results['readability grades']['DaleChallIndex']
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
        self.df.loc[index,'complex_words'] = results['sentence info']['complex_words']
        self.df.loc[index,'complex_words_dc'] = results['sentence info']['complex_words_dc']

    def get_topic(self,index,row):
        tokens = self.topic_tokenizer(row.clean_text,return_tensors='pt')
        output = self.topic_model(**tokens)
        scores = output[0][0].detach().numpy()
        scores = expit(scores)
        predictions = (scores >= 0.5) * 1
        pred = np.argmax(scores)

        self.df.loc[index,'topic'] = pred
        for i in range(6):
            label = str(self.topic_classes[i])
            self.df.loc[index, label] = scores[i]


    def get_gender(self,index,row):
        pass

    def move_politeness(self,index,row):
        self.df.loc[index, 'politeness'] = row.politeness    

    def move_offensive(self,index,row):
        self.df.loc[index, 'offensive'] = row.offensive  

    def get_irony(self,index,row):
        encoded_input = self.irony_tokenizer(row.hugging, return_tensors='pt', truncation = True, max_length = 128, padding = True)
        output = self.irony_model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        self.df.loc[index, 'offensive'] = ranking[0]

    def get_emoji(self,index,row):
        encoded_input = self.emoji_tokenizer(row.hugging, return_tensors='pt', truncation = True, max_length = 128, padding = True)
        output = self.emoji_model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        self.df.loc[index, 'emoji'] = ranking[0] #  update this

####################################################################################################
############################                 LOAD MODELS            ################################
####################################################################################################

    def load_topic_model(self,tweet_df):
        MODEL = f"cardiffnlp/tweet-topic-21-single"
        self.topic_tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # PT
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        self.topic_classes = self.topic_model.config.id2label
        print('loaded topic model')




    def load_gender_model(self, tweet_df):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        max_tokencount = min(max_tokencount, 510)
        
        encoding = tokenizer.batch_encode_plus(tweet_df.grammertext.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=max_tokencount, add_special_tokens=True)
        
        tweet_df['bert_tokens'] = encoding

        MODEL = 'genderBERT/models/reddit_bert_base_epoch_4'

        self.gender_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL,
            attention_probs_dropout_prob=0.2,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
        print('loaded gender model')

    def load_politeness(self,tweet_df):
        ## Scoring each tweet based on politeness 
        
        tweets = tweet_df.grammartext.tolist()

        # Tokenize same way as training data
        model_name = 'roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        test_encodings = tokenizer(tweets , truncation=True, padding=True, max_length=256)
        test_dataset = SimpleDataset(test_encodings)

        path = f'politeness{os.path.sep}results/checkpoint-52500/'

        model = AutoModelForSequenceClassification.from_pretrained(path)
        print('loaded politeness')

        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            compute_metrics=compute_metrics_for_regression,     # the callback that computes metrics of interest
        )

        predictions = trainer.predict(test_dataset)
        tweet_df['politeness'] = predictions[0]


    @staticmethod
    def run_offensive_model(test):
        current = os.getcwd()

        os.chdir('G:\My Drive\MSc_project\.MAIN\offensiveness')
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


    def load_irony_model(self):
        task='irony'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        self.irony_tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # download label mapping
        labels=[]
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        self.irony_labels = [row[1] for row in csvreader if len(row) > 1]

        # PT
        self.irony_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        print('loaded irony')

    def load_emoji_model(self):

        task='emoji'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        self.emoji_tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # download label mapping
        labels=[]
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        self.emoji_labels = [row[1] for row in csvreader if len(row) > 1]

        # PT
        self.emoji_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        print('loaded emoji model')

    def load_hate_model(self):
        
        hate_tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")

        model = AutoModelForSequenceClassification.from_pretrained("pysentimiento/robertuito-sentiment-analysis")







################################################################################################
############################                 FIGURES                ############################
################################################################################################
    



    @staticmethod
    def count_values_in_column(data,feature):
        total=data.loc[:,feature].value_counts(dropna=False)
        percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
        return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])


    def make_sentiment_pi_chart(self,tw_list):
        # create data for Pie Chart
        pichart = self.count_values_in_column(tw_list,"sentiment")
        names= pichart.index
        size=pichart["Percentage"]
        
        # Create a circle for the center of the plot
        my_circle=plt.Circle( (0,0), 0.7, color='white')
        plt.pie(size, labels=names, colors=['green','blue','red'])
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        # plt.show()
        plt.savefig(f'{self.save_path}_sentiment_summarisation_pichart.png')

    def create_wordcloud(self,text,label):
        path = f'{self.save_path}_wordcloud_{label}.png'
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width=1600, height=800,max_font_size=200,max_words=1000,stopwords=stopwords, background_color='white', repeat=False).generate(str(text))
        plt.figure(figsize=(12,10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(path)
        # plt.show()


    def get_wordclouds(self,tw_list):
        
        tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]['clean_text']
        tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]['clean_text']
        tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]['clean_text']

        #Creating wordcloud for all tweets
        self.create_wordcloud(tw_list["clean_text"].values,label='all')

        #Creating wordcloud for positive sentiment
        self.create_wordcloud(tw_list_positive.values,label ='positive')

        #Creating wordcloud for negative sentiment
        self.create_wordcloud(tw_list_negative.values,label = 'negative')

        #Creating wordcloud for neutral sentiment
        self.create_wordcloud(tw_list_neutral.values,label = 'neutral')


    def get_density_of_words(self,tw_list):

        #Appliyng Countvectorizer
        countVectorizer = CountVectorizer(analyzer=self.clean_text) 
        countVector = countVectorizer.fit_transform(tw_list['stemmed'])
        print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
        #print(countVectorizer.get_feature_names())
        count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())

        # Most Used Words
        count = pd.DataFrame(count_vect_df.sum())
        countdf = pd.DataFrame(count.sort_values(0,ascending=False))
        countdf.to_csv(f'{self.save_path}_most_used_words.csv')

    @staticmethod
    def sort_by_tweet(all_tweets): 

        df = pd.DataFrame.from_dict(all_tweets, orient='index', columns= ['text'])
        sorted_tweets = {}
        for row,index in df.groupby('text').groups.items():
            key = tuple(index.values.tolist())
            sorted_tweets.update({key:row})

        new_df = pd.DataFrame.from_dict(sorted_tweets, orient='index', columns= ['text'])
        
        return new_df




#######
# MAIN FUNCTION TO RUN THE ANALYSIS
########


    def tweet_analysis(self):

        self.informer_db = self.load_informer_data()

        all_tweets = self.get_the_tweets(self.informer_db)
        # self.df contains all the tweets but is sorted by tweet id. therefore contains many duplicates
        self.df = pd.DataFrame.from_dict(all_tweets, orient='index', columns= ['text'])

        # tweet_df is the dataframe that contains all the tweets
        tweet_df = self.sort_by_tweet(all_tweets)

        tweet_df = self.tweet_cleaner(tweet_df)
        self.df = self.tweet_cleaner(self.df)

        # self.df[['polarity', 'subjectivity']] = self.df['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

        ##########################################################################
        ##########################################################################
        ### LOADING IN NECESSARY MODELS

        # self.load_politeness(tweet_df)

        # self.load_topic_model(tweet_df)

        self.load_irony_model()


        # self.load_offensive_model(tweet_df)

        # self.load_emoji_model()

        self.load_hate_model(tweet_df)
        
        # self.load_gender()

        ###############################
        # CLASSIFYING EACH TWEET
        


        # index the rows of the database that contains the unique tweets
        for index, row in tweet_df.iterrows():
            # self.get_sentiment(index,row)
            # self.get_grammar(index,row)
            # self.get_readability(index,row)
            # self.move_politeness(index,row)
            self.get_topic(index,row)
            self.move_offensive(index,row)
            self.get_irony(index,row)
            self.get_emoji(index,row)



        ### Scoring Politeness
        

        print('finished getting the metrics!!!!!!!')

        df_path = f'{self.save_path}_emotion_classification.csv'
        self.df.to_csv(df_path)
        
        # refilter the database to only contain the unique tweets
        unique_df = self.df.copy()
        unique_df.drop_duplicates('clean_text', inplace=True)

        # sentiment_count = self.count_values_in_column(unique_df,"sentiment")
        # sentiment_count.to_csv(f'{self.save_path}_sentiment_summarisation.csv')

        # # make a pi chart of the result of sentiments
        # self.make_sentiment_pi_chart(unique_df)

        # ### Create word clouds of all 
        # self.get_wordclouds(unique_df)

        # # get thedensity of the words used
        # self.get_density_of_words(unique_df)


hashtags = ['blm']

informer_db_path = f'informer_results{os.path.sep}'

for hashtag in hashtags:

    a = Analyzer(hashtag)

    a.tweet_analysis()