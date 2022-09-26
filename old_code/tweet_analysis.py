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
        return tw_list

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

    def get_emotion(self,index,row):
        pred = self.em_model.predict(np.expand_dims(row.sequences,axis=0))
        prediction = np.argmax(pred)
        self.df.loc[index, 'emotion'] = self.index_to_classes[prediction]

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
        self.df.loc[index,'characters_per_word'] = results['sentence info']['characters_per_word']
        self.df.loc[index,'syll_per_word'] = results['sentence info']['syll_per_word']
        self.df.loc[index,'words_per_sentence'] = results['sentence info']['words_per_sentence']
        self.df.loc[index,'sentences_per_paragraph'] = results['sentence info']['sentences_per_paragraph']
        self.df.loc[index,'type_token_ratio'] = results['sentence info']['type_token_ratio']
        self.df.loc[index,'characters'] = results['sentence info']['characters']
        self.df.loc[index,'syllables'] = results['sentence info']['syllables']
        self.df.loc[index,'words'] = results['sentence info']['words']
        self.df.loc[index,'wordtypes'] = results['sentence info']['wordtypes']
        self.df.loc[index,'long_words'] = results['sentence info']['long_words']
        self.df.loc[index,'complex_words'] = results['sentence info']['complex_words']
        self.df.loc[index,'complex_words_dc'] = results['sentence info']['complex_words_dc']
        




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

    @staticmethod
    def get_sequences(tokenizer, tweets):
        sequences = tokenizer.texts_to_sequences(tweets)
        padded_sequences = pad_sequences(sequences, truncating='post', maxlen=50, padding='post')
        return padded_sequences

    @staticmethod
    def create_model():
        model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000, 16, input_length=50),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(6, activation='softmax') ])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model

    def load_emotion_data(self):
        def get_tweets(data):
            tweets=[x['text'] for x in data]
            labels=[x['label'] for x in data]
            return tweets, labels
        
    
        dataset = load_dataset('emotion')

        train=dataset['train']
        test = dataset['test']

        tweets, labels = get_tweets(train)

        #Tokenize
        self.tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
        self.tokenizer.fit_on_texts(tweets)

        self.classes_to_index = {'anger': 3, 'fear': 4, 'joy': 1, 'love': 2, 'sadness': 5, 'surprise': 0}
        self.index_to_classes = {0: 'surprise', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'sadness'}
    
        em_path = f'models{os.path.sep}emo_6_weights.h5'
        self.em_model = self.create_model()
        self.em_model.load_weights(em_path)

        test_tweets, test_labels = get_tweets(test)
        test_sequences = self.get_sequences(self.tokenizer, test_tweets)

        _ = self.em_model.evaluate(x=test_sequences, y=np.array(test_labels))


#######
# MAIN FUNCTION TO RUN THE ANALYSIS
########


    def tweet_analysis(self):

        self.informer_db = self.load_informer_data()

        all_tweets = self.get_the_tweets(self.informer_db)
        self.df = pd.DataFrame.from_dict(all_tweets, orient='index', columns= ['text'])

        tweet_df = self.sort_by_tweet(all_tweets)

        tweet_df = self.tweet_cleaner(tweet_df)
        self.df = self.tweet_cleaner(self.df)

        self.df[['polarity', 'subjectivity']] = self.df['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

        # load in emotion model stuff
        print('loading in pretrained emotion model')
        self.load_emotion_data()
        sequences = self.get_sequences(self.tokenizer,tweet_df.clean_text.to_list())
        seq = [a for a in sequences]
        tweet_df['sequences'] = seq


        # index the rows of the database that contains the unique tweets
        for index, row in tweet_df.iterrows():

            # self.get_sentiment(index,row)
            # self.get_grammar(index,row)
            # self.get_emotion(index,row)
            self.get_readability(index,row)

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