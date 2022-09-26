import tweepy
from tweepy import OAuthHandler
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from datetime import datetime
import json
import time
import glob
import os
from collections import defaultdict
import configparser
from copy import deepcopy
import multiprocessing
from joblib import Parallel, delayed



class crawler(object):
    def __init__(self, maxTweets,friend_path,hashtag,file_names,num_keys): 

        # consumer_key = str(consumer_key)
        # consumer_secret = str(consumer_secret)
        # access_token = str(access_token)
        # access_secret = str(access_secret)

        self.maxTweets = maxTweets
        self.friends_path = friend_path
        self.hashtag = hashtag
        self.sep = os.path.sep  
        self.file_names = file_names
        self.num_keys = num_keys


################################################################################################################
################################################################################################################
################################################################################################################

    def process_targets(self,tweet):
        # OLD
        # d = {'tweet-text': tweet.full_text,'created_at': tweet.created_at, 'user-id':tweet.author.id,
        # 'check-idx':None,'informers-data':None,'friend-ids':None, 'user-feed': None, 'location': tweet.author.location,
        # 'num-followers': tweet.author.followers_count, 'num-following':  tweet.author.friends_count, 'description': tweet.author.description}

        d = {'check-idx':None,'informers-data':None,'friend-ids':None } # the extra information we want from the tweets.

        return { **tweet._json, **d}

    @staticmethod
    def process_loaded_targets(tweet):
        d = {'check-idx':None,'informers-data':None,'friend-ids':None } # the extra information we want from the tweets.
        return {**d,**tweet}

    def write_stats(self):

        file1 = open(f"{self.hashtag}{self.sep}tweets_description.txt","w")
        n = len(self.tweet_df)
        all_users = self.tweet_df['user'].to_dict()
        all_ids = [dic['id'] for key,dic in all_users.items() ]
        num_users = len(pd.unique(all_ids))

        last_date = datetime.strptime( self.tweet_df.iloc[0]['created_at'] , '%a %b %d %H:%M:%S +%f %Y')
        first_date =  datetime.strptime( self.tweet_df.iloc[n-1]['created_at'], '%a %b %d %H:%M:%S +%f %Y') 

        file1.write(f'For the hashtag #{self.hashtag}\nTotal Tweets Extracted: {n} \nTotal Retweets Extracted: {len(self.all_retweets)}')
        file1.write(f'\nCrawled tweets from {num_users} different users')
        file1.write(f'\nAll tweets were found in a region of {first_date-last_date}\nTime first tweet was posted {first_date} \nTime Last tweets was posted {last_date}')

        file1.close() #to change file access modes

    def get_friends_db(self):
        jsons = [pos_json for pos_json in os.listdir(f'user-friends{self.sep}') if pos_json.endswith('.json')]
        all_js = {}
        for file in jsons:
            with open(os.path.join(f'user-friends{self.sep}' + file)) as jf:
                all_js = { **all_js, **json.load(jf) }
        print(f'pulled data on {len(all_js)} users')
        return all_js



    def my_search_api(self):

        consumer_key = '8a4Pfd428qOu4fG8INDr9IXYU'
        consumer_secret = 'OTayly3aQ4QckJVurGkVJllHpISWD4Mdk2zdd8jTkvA8MjjyEe'
        access_token = '2165355935-n9L0XTw1wDtFZyFii4MvflaXH512NbIw27c82Ak'
        access_secret = 'l6Kou2M8YFUz1PrapMV6N4eR4B6JcxqMIEn70zVY3JXOL'

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)   

        return api      

###########################################################
###########################################################
# API KEY FUNCTIONS
###########################################################
###########################################################


    def get_api_list(self,filename,num_keys):
        settings_file = f"apikeys{self.sep}{filename}"
        # Read config settings
        config = configparser.ConfigParser()
        config = configparser.ConfigParser(interpolation=None)
        config.readfp(open(settings_file))

        # Create API objects for each of the API keys
        # 1-based indexing of config file
        start_idx = 1
        end_idx = num_keys
        num_api_keys = end_idx - start_idx + 1

        apis = []

        print("Creating api objects for {} API keys".format(num_api_keys))
        for api_idx in range(start_idx, end_idx + 1):
            consumer_key = config.get('API Keys ' + str(api_idx), 'consumer_key')
            consumer_secret = config.get('API Keys ' + str(api_idx), 'consumer_secret')
            access_token_key = config.get('API Keys ' + str(api_idx),
                                        'access_token')
            access_token_secret = config.get('API Keys ' + str(api_idx),
                                            'access_secret')

            # Connect to Twitter API
            try:
                auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
                auth.set_access_token(access_token_key, access_token_secret)
                api = tweepy.API(auth, wait_on_rate_limit=True)
            except Exception as e:
                print("Error while creating API object: " + str(e))
                continue
            else:
                apis.append(api)
        print(f'found {len(apis)} apis from {filename}')
        idx = []
        for i, apii in enumerate(apis):
            try: 
                a = apii.verify_credentials()
            except tweepy.TweepyException as e:
                idx.append(i)
                # print(e)
        
        working_api = np.delete(apis,idx)
        print(f'{len(working_api)} api work')
        return working_api.tolist()

    def make_api(self):
        consumer_key = '3rJOl1ODzm9yZy63FACdg'
        consumer_secret = '5jPoQ5kQvMJFDYRNE8bQ4rHuds4xJqhvgNJM4awaE8'

        tokens="1273621486201450497-2KsGK9JgSSdabGYxFlDNB214MXwi35:kbLULiG4k7SvUlwFkr0p6ESTdvARIThfmI1lQ5GuQmH8s,"
        # tokens+="1186873796982128641-RHYpdyKJIfSX6KtRfA6k73emHMOhBY:Zth72rrtcZGfqJ4B9TK8dH1RhS8HC6KOlTDqA3bJnMnGL," DOES NOT WORK
        tokens+="1186715756437868546-U9LOFzfSEk4tTOYHoKn7X08vWFm3An:kWWnOy6JLbHI9nG8GihGIjB1SeH5nJE8oqi4KiRIiHB2B,"
        tokens+="1186707518799650816-KOvTi7HOrZ1KiVRXyAvNjpJofF3lY8:EsXAVrtyqkekdmtr0GcZCiVk9RwAJC4v9Kj3fo0RWPpgp,"
        tokens+="1186873046726041600-rNJoDJulul6m5GQUbXUG7zNfpnJywj:OsR7WIHZ9PEdlmIE7K36V1CrM3OiMKIzAboNXMdEJm1UJ,"
        tokens+="1186872285782794240-X4x7t8KRaf3Ce3oNI2ka7DO7rapVys:VUyNnBIRr6oRpBr10lbEF0MzlGz6eJJW3STUWOGyXgVbr,"
        tokens+="1261898258739126273-Dw4HYHPkKj82V4RcgYhpxLsnJdMALV:384AYTeZqRykw5ADdVXhi1BXZadOuflhzOiKljh1Pt0bi,"
        tokens+="1261898258739126273-Dw4HYHPkKj82V4RcgYhpxLsnJdMALV:384AYTeZqRykw5ADdVXhi1BXZadOuflhzOiKljh1Pt0bi,"
        tokens+="1261901714564583424-odKz6Xonlzg1QVg32o6hdqXbGAZ6nt:lx7hysS64y7kyc0pr4IMdqXNJr8yqbgY9vZ7qhnL6ldGY,"
        tokens+="1261905369485373440-mxYEQlU9tEjFvFClg3YTZU2Krh4ezO:mcvotgeJvuOgkJb5pXejP9qzjDBcALZSHqGJesH7FzOOs,"
        tokens+="1261908827961585664-YkhHP2z6XczyTwBt8pT7IYmYw4XFTN:OQJhcBCQiHWCOS1lcUHlZp8Minc7drmpW861tFNwweABC,"
        tokens+="1261911771528364032-uFSYzCtvZeQPLvSIR2NZM3MwYnPDx9:gjLJUDRkiC6qVhh6zN8czDLzcE3Bnhge9Sgq7QIQsRfZp,"
        tokens+="1261913654967992321-dq90nrKd8OTIAiBwK4rmDPR1hvBwTb:6lL8f15qO9DOMczUzImxwUjMmDvPHlQcd3eJBtUzQ53ck,"
        tokens+="1261918659942871040-ThZsmIIeIWLfTuhNB8uD3HORhunMVS:D3QNGphJ1R870yzBV41QSvezqS6rysRVgEG1RU3c7tH30,"
        tokens+="1261929369691283456-0CdW3UgD4ONxFCshwmBqroq3m8BFG7:YpeYZ3hWRNkFOzfoJGiIPXnW65vzul5e7ofDN1q3zZErV,"
        tokens+="1261930395513192448-14EJHzMXDGzDN65NAx9HLXaz8Y1OHf:97qLbZ53tPQl5xwmNPDBlADBrSlZZeXd659N5XXDG2O5E,"
        tokens+="1261931615837581317-Bdgkok64OnA0HWB9q3NZfFF12FpBjD:qeYMydCMxt2ctLOkfNV3yix6erJoxti7NiOScNCQ0zHf6,"
        tokens+="1261932804046114816-5s1MVYfHsfwjxz1pZ2jhLE3jgO0enO:8XmjhdkKTakxF9a3ZKCMYJzlNUb9ARChEZ74XH91qZVqg,"
        tokens+="1259732232433856512-knm0EqhoG7tyEjTsgIQc6974cEy976:lavQjBjsOdZgUy5U64ZtK8YYdfwQeknl9Oy2CP7nJ3uE9,"
        tokens+="1262372803174531072-uyPlAvb0BHwSuB1kgPOroGMyjFPtzj:Bsnfo8WTYNqEzr5S9fW9gzfWFc9E2ZqM0wXwUGIUgSwPo,"
        tokens+="1262374702003044353-GCVqUDfNPkRGnBzmKHTPeZmmsBHjxx:pzqueROwD89WH3hZyr7qDV4ftnu8OPZM3OfhyNSM4pi9T,"
        tokens+="1261903040946171905-mLBFRyOpShKunFrxk5o5DDoJERTEE1:Osvx77SI7qS0qIIYqlI43eLOCOS6UL1e3XSYcUyrYpCHz"

        connections=[]

        for tk in tokens.split(","):
            token=tk.split(":")
            auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret,token[0], token[1])
            api = tweepy.API(auth, wait_on_rate_limit=True)
            connections.append(api) 

        print(f'\nfound {len(connections)} apis from juans stuff')
        idx = []
        for i, apii in enumerate(connections):
            try: 
                a = apii.verify_credentials()
            except tweepy.TweepyException as e:
                idx.append(i)
                # print(e)
        
        working_api = np.delete(connections,idx)
        print(f'{len(working_api)} api work')
        return working_api.tolist()


####################################################################################

# FUNCTINOS INVOLVING GETTING THE TWEETS WITH HASHTAGS

####################################################################################

    def load_in_tweets(self):
        path = f'{self.hashtag}{os.path.sep}{self.hashtag}_all_tweets.json'
        with open(path) as jf:
            data = json.load(jf)

        return data

    def search_for_tweets(self):

        start = time.perf_counter()
        print(f'\nSearching for tweets with hashtag {self.hashtag}...')


        search_api = self.my_search_api()
        tweets = tweepy.Cursor(search_api.search_tweets,
                                q=f'#{self.hashtag}',
                                tweet_mode='extended', 
                                count = 100,
                                lang='en').items(self.maxTweets)

        all_tweets = []
        tweet_counter = 0
        for tweet in tweets:
            tweet_counter +=1 

            if tweet_counter == 449:
                self.api_counter +=1 
            
            all_tweets.append(tweet)

        fin = time.perf_counter()
        total = fin - start
        total /= 60
        print(f'search took {total} mins to complete')

        ### Save all extracted tweets to directory
        json_save = [ tweet._json for tweet in all_tweets ]

        directory = f'{self.hashtag}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f'{self.hashtag}{self.sep}{self.hashtag}_all_tweets.json', 'w') as fp:
            json.dump(json_save,fp)
        
        return all_tweets

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
    '''

    THE MAIN FUNCTION!!!!!!!!!!!!!!!!!!!!!!!!
    
    '''
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

    def stage1_extraction(self):

        print(('changeshave been made'))

        # friend_db = self.get_friends_db()

        juan_api = self.make_api()

        everyone_api = self.get_api_list(self.file_names[0],self.num_keys[0])
        just_my_api = self.get_api_list(self.file_names[1],self.num_keys[1])

        all_api = juan_api + everyone_api + just_my_api
        num_api = len(all_api)
        print(f'total working api is {num_api}')

        ### FIND THE INITIAL TWEETS

        #  we will keep track of the api that we use
        self.api_counter = 0
        # tweet_list = self.search_for_tweets()
        tweet_list = self.load_in_tweets()

        ################################################################################################################
        ########################################################
        ### PROCESS INTO NECESSARY FORMAT

        # WHEN SEARCHING FOR TWEETS

        ########################################################
        # all_tweets = {tweet.id:self.process_all(tweet) for tweet in tweet_list}
        # all_tweets = {tweet.id: tweet._json for tweet in tweet_list}
        # self.tweet_df = pd.DataFrame.from_dict(all_tweets, orient='index')
        # self.all_retweets = {tweet.id:self.process_targets(tweet) for tweet in tweet_list if "RT @" in tweet.full_text }
        # print(f'found {len(self.tweet_df)} tweets all together\nfound {len(self.all_retweets)} retweets')



        # WHEN LOADING TWEETS IN
        ########################################################
        all_tweets = {tweet['id']: tweet for tweet in tweet_list}
        self.tweet_df = pd.DataFrame.from_dict(all_tweets, orient='index')

        # self.all_retweets = {tweet['id']:self.process_loaded_targets(tweet) for tweet in tweet_list if "RT @" in tweet['full_text'] }
        all_rt_path = f'{self.hashtag}{self.sep}{self.hashtag}_all_retweets.json'
        with open(all_rt_path) as jf:
            self.all_retweets = json.load(jf)
    
        print(f'found {len(self.tweet_df)} tweets all together\nfound {len(self.all_retweets)} retweets')

        # users_done = self.get_friends_db()


        ########################################################

        ########### CHECK FOR TWEETS THAT ARE 24 HOURS BEFORE

        ########################################################



        # print('\nnow finding tweets that are posted 24 hours before, for each tweet')

        # for key, value in list(self.all_retweets.copy().items()):

        #     tweet_time =  datetime.strptime( value['created_at'] , '%a %b %d %H:%M:%S +%f %Y')
        #     all_tweet_times = pd.DataFrame([ datetime.strptime( row , '%a %b %d %H:%M:%S +%f %Y') for row in self.tweet_df.loc[:,'created_at'] ])

        #     c1 = ( tweet_time - all_tweet_times < timedelta(days=1)).values.astype(int)
        #     c2 = ( all_tweet_times < tweet_time ).values.astype(int)

        #     idx = np.where( c1+c2 == 2 ) # find the tweets that are tweeted 24 hours before target tweet
        #     if idx[0].size == 0:
        #         del self.all_retweets[key]
        #         print(f'dropped tweet {key} from consideration. Had no tweets that were posted 24 hours before')
        #     else:
        #         self.all_retweets[key]['check-idx'] = idx[0].tolist()

        print('found!!')

        # dump this to our ting

        # with open(f'{self.hashtag}{self.sep}{self.hashtag}_all_retweets.json', 'w') as fp:
        #     json.dump(self.all_retweets,fp)

        #################################################################
        
        ### NOW GET THE LIST OF THE ACCOUNTS EACH TWEET AUTHOR FOLLOWS.
        
        #################################################################


        print(f'Getting the accounts that user of each target tweet follows')
        start = time.perf_counter()
        storage = {}
        counter = self.api_counter # counter count the number of accounts we request to pull in this batch
        api_idx = 0

        users_done = self.get_friends_db()

        unique_users = users_done

        for tweet_id, data in list(self.all_retweets.copy().items()):

            if counter % 15: # can only make 15 requests per API. So after 15 requests, move onto next API
                # print('using next api')
                api_idx +=1     

            if counter % 400 == 0: # once we have collected the information of 500 users. Dump the information to a json
                print(f'\n\n HAVE FOUND {counter} users friends!! The code still works!! Wow!!\n\n')
                stamp = datetime.today().strftime('%Y%m%d_%H%M%S')
                with open(f'{self.friends_path}{self.sep}{self.hashtag}_dump_{stamp}.json', 'w') as fp:
                    json.dump(storage,fp)
                storage.clear()

            if api_idx == num_api-1:
                api_idx = 0     
                print('passed through all api')

            followings = [] # initialise the list of friends each user will have
            user = data['user']['id']
            if  user in unique_users:
                self.all_retweets[tweet_id]['friend-ids'] = unique_users[user]['friend-ids']
                print('done this mug')
            else: # the user has not already been found
                try:
                    for friend in tweepy.Cursor(all_api[api_idx].get_friend_ids, user_id=user, count = 4999).items():
                        followings.append(friend)
                        # update the unique users file
                        unique_users.update({user:{'friend-ids':followings}} )
                        self.all_retweets[tweet_id]['friend-ids'] = followings
                    # store this users friends in our storgae dictionary. this will later be dumped into 
                    storage.update( {user: {'friend-ids':followings}} )
                    counter+=1 # have collected data on one more user
                    if followings:
                        del self.all_retweets[tweet_id]
                except tweepy.TweepyException as e: # we've got an error with the tweepy API
                    print(e) 
                    del all_api[api_idx]
                    num_api -=1 # working with one less api
                    api_idx += 1 # move onto the next
            
        fin = time.perf_counter()
        ttl = fin - start
        print(f'got the accounts that users of informer tweets follow in {ttl/60/60} hours')

        ### Check if we already have data on users
        # user_data = self.get_friends_db()

        informer_df = deepcopy(self.all_retweets)

        ##########################################################
        ##########################################################
        ##### FINAL STEP --- CHECK IF USER FRIENDS ARE IN THE DATABASE

        ### Check the target tweets for users that follow them in this database.
        for key, value in list(informer_df.copy().items()):

            users_friends = value['friend-ids']

            if not users_friends: 
                del informer_df[key]
            else: # if the user has no friend, then delete. Get outta here 
                # get the users that we must check are informers of the current tweet. ie if they are friends of the target user
                user_obj = self.tweet_df.iloc[ value['check-idx'] ] ['user']
                check_users = [ row.get('id') for row in user_obj]

                check = (set(check_users) & set(users_friends) )

                # returns a list of the indices of the tweet authors that are informers!!
                informers = np.where([friend in check for friend in check_users])[0].tolist()

                if not informers:
                    del informer_df[key]
                else:
                    informer_data = {}
                    for informer in informers:
                        informer_data.update({self.tweet_df.iloc[informer]['user']['id']: self.tweet_df.iloc[informer].to_dict()})
                    informer_df[key]['informers-data'] = informer_data

        print(f'Found all informers for this batch of tweets with hashtag {self.hashtag}')
        print(f'In total found {len(informer_df)} multi-source cases \nOnto the next')

        if not informer_df.keys():
            print('Found no informers in this search. Sorry Berkem.')
        else: # WE ACTUALLY FOUND SOME DATA!!! WOOHOOOOOO
            print(f'Found all informers for this batch of tweets with hashtag {self.hashtag}')
            print(f'In total found {len(informer_df)} multi-source cases \nOnto the next')

            file1 = open(f"{self.hashtag}{os.path.sep}tweets_description.txt","a")
            file1.write(f'\n\nFOUND {len(informer_df)} MULTI-SOURCE CASES')
            file1.close()

            # SAVE THIS TO JSON DERULO
            save_path = f'{self.hashtag}{os.path.sep}informer_database.json'
            with open(save_path, 'w') as fp:
                json.dump(informer_df, fp)
            self.all_retweets


# hashtag = 'anime'
# friend_path  = f'user-friends{os.path.sep}'
# maxTweets = 500
# key_paths = ['apikeys.txt', 'apikeys2.txt']
# num_of_keys = [45,19]

# c = crawler( maxTweets, friend_path, hashtag, key_paths, num_of_keys)

# c.stage1_extraction()

# # c.get_api_list()
