## Part of stage 2 to grab the necessary info of the multi-source case.
# grab the base information of the infectors of the tweet.

import json 
import os
from types import new_class
import pandas as pd

class crawler(object):
    def __init__(self, hashtag): 

        self.hashtag = hashtag
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################


    @staticmethod
    def process_loaded_all(tweet):
        d = {'id':tweet['id_str'], 'tweet-text': tweet['full_text'],'created_at': tweet['created_at'], 'user-id':tweet['user']['id_str'],
        'location': tweet['user']['location'],
        'num-followers': tweet['user']['followers_count'], 'num-following':  tweet['user']['friends_count'], 'description': tweet['user']['description'],
        'retweet_count': tweet['retweet_count'], 'favourite_count':tweet['favorite_count']}
        return d




####################################################################################

# FUNCTINOS INVOLVING GETTING THE TWEETS WITH HASHTAGS

####################################################################################

    def load_in_tweets(self):
        path = f'{self.hashtag}{os.path.sep}{self.hashtag}_all_tweets.json'
        with open(path) as jf:
            data = json.load(jf)
        return data

    def load_informer_df(self):
        path = f'{self.hashtag}{os.path.sep}informer_database.json'
        with open(path) as jf:
            data = json.load(jf)
        return data

    def load_new_informer_df(self):
        path = f'{self.hashtag}{os.path.sep}updated_informers_data.json'
        with open(path) as jf:
            data = json.load(jf)
        return data



    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
        '''

        THE MAIN FUNCTION!!!!!!!!!!!!!!!!!!!!!!!!
        
        '''
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################

    def infector_extraction(self):


        # WHEN LOADING TWEETS IN
        tweet_list = self.load_in_tweets()
        ########################################################

        informer_df = self.load_informer_df()

        dic=  {}
        for tweet in tweet_list:
            dic.update({tweet['id_str']:tweet})
        
        print('turn list to dic')

        new_dic = {}

        # get the data of the informer-db
        for key, value in informer_df.items():
            value['num-informers'] = len(value['informers-data'])
            try: 
                infector_data = dic[key]['retweeted_status']
                value['infector-info'] = {infector_data['user']['id_str']:self.process_loaded_all(infector_data)}
                if infector_data['user']['id'] in value['friend-ids']:
                    value['infector-is-friend'] = 1
                else: 
                    value['infector-is-friend'] = 0
                new_dic.update( {key:value})
            except:
                print(f'source was not a retwet. delete from multi-source df')
        print('added infector data of all')

        path = f'{self.hashtag}{os.path.sep}updated_informers_data.json'
        with open(path, 'w') as fp:
            json.dump(informer_df,fp)

    def get_num_of_cases_info(self):

        data = self.load_new_informer_df()

        df = pd.DataFrame.from_dict(data,orient='index')

        # infector is friend
        ms_cases = df['infector-is-friend'].sum()
        print(f'\n{self.hashtag}')
        print(f'INFECTOR IS FRIEND CASES: {ms_cases}. Out of {len(df)}')

        # SINGLE SOURCE CASES
        single_source = df [ df['num-informers'] == 1 ] 
        print(f'SINGLE SOURCE CASES: {len(single_source)}')


        # BREAKDOWN OF THE NUMBER OF INFORMERS A TWEET HAS!!!
        breakdown = pd.DataFrame(df['num-informers'].value_counts(normalize=True) * 100).sort_index()

        info = pd.DataFrame(breakdown.iloc[0:10])

        info.loc[len(info)+1] = {'num-informers':breakdown.iloc[10:20].sum().values[0]}
        info.loc[len(info)+1] = {'num-informers':breakdown.iloc[20:40].sum().values[0]}
        info.loc[len(info)+1] = {'num-informers':breakdown.iloc[40:80].sum().values[0]}
        info.loc[len(info)+1] = {'num-informers':breakdown.iloc[80:].sum().values[0]}

        path = f'{hashtag}{os.path.sep}{hashtag}_breakdown.csv'
        info.to_csv(path)


hashtags = ['gaza','loveisland','monkeypox','NHS','Olivianewtonjohn','Supercup','UkraineWar']
# hashtags = ['avengers','blm','borisjohnson','brexit','climatechange','covid','gaza','loveisland','monkeypox','NHS','Olivianewtonjohn','Supercup','UkraineWar']
#   DO FOR COVID

for hashtag in hashtags:

    c = crawler(hashtag)

    c.infector_extraction()
    c.get_num_of_cases_info()
        