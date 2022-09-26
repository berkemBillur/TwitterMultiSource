import tweepy
from tweepy import OAuthHandler
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import json
import time

"""
FUNCTIONS TO USE
"""
def process_dic(tweet):
    cols = ['tweet-id','tweet-text','tweet-time', 'user-id','check1-idx','informer-tweets','following-accounts']
    d = {key: None for key in cols}


    d['tweet-id'] = tweet.id
    d['tweet-text']= tweet.full_text
    d['tweet-time']= tweet.created_at
    d['user-id']= tweet.author.id
    # d['user-following'] remains blank for now
    return d

def process_tweet(tweet):
    return [tweet.id,tweet.full_text, tweet.created_at,  tweet.author.id, [], [], [] ]

def write_stats(all_tweets,all_retweets, hashtag):
    file1 = open(f"{hashtag}/tweets_description.txt","w")
    n = len(all_retweets)
    num_users = len(pd.unique(all_tweets['user-id']))

    first_date = all_tweets.at[0,'tweet-time']
    last_date = all_tweets.at[n-1,'tweet-time']

    file1.write(f'For the hashtag #{hashtag}\nTotal Tweets Extracted: {n} \nTotal Retweets Extracted: {len(all_retweets)}')
    file1.write(f'\nCrawled tweets from {num_users} different users')
    file1.write(f'\nAll tweets were found in a region of {first_date-last_date}\nTime first tweet was posted {first_date} \nTime Last tweets was posted {last_date}')

    file1.close() #to change file access modes

def save_all_tweets_to_json(tweet_list):
    pass

"""
TWITTER KEYS
"""

consumer_key = '8a4Pfd428qOu4fG8INDr9IXYU'
consumer_secret = 'OTayly3aQ4QckJVurGkVJllHpISWD4Mdk2zdd8jTkvA8MjjyEe'
access_token = '2165355935-n9L0XTw1wDtFZyFii4MvflaXH512NbIw27c82Ak'
access_secret = 'l6Kou2M8YFUz1PrapMV6N4eR4B6JcxqMIEn70zVY3JXOL'

# INITIALISE API
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_secret)
api = tweepy.API(
    auth,
    # Making sure this is True, so the data will be complete.
    wait_on_rate_limit=True)

"""
CRAWL PARAMETERS
"""

hashtag = 'brexit'
maxTweets = 1000
data_path = f'{hashtag}/{hashtag}_tweets.json'


"""
BEGIN THE SEARCH
"""
start = time.perf_counter()
print(f'\nSearching for tweets with hashtag {hashtag}...')

tweets = tweepy.Cursor(api.search_tweets,
                        q=f'#{hashtag} filter:retweets',
                        tweet_mode='extended', 
                        lang='en').items(maxTweets)

list_tweets = [tweet for tweet in tweets]

tweet_list = []
for tweet in list_tweets:
    tweet_list.append(tweet)

fin = time.perf_counter()
total = fin - start
total /= 60
print(f'search took {total} mins to complete')



### PROCESSING THE DATA 

# initialise the dataframe
cols = ['tweet-id','tweet-text','tweet-time', 'user-id','check1-idx','informer-tweets','following-accounts']
all_tweets = pd.DataFrame([],columns = cols)
all_retweets = pd.DataFrame([],columns = cols)

# FILTERING THROUGH THE TWEETS TO PROCESS ALL AS WELL AS FIND ALL RETWEETS
start = time.perf_counter()
for i, tweet in enumerate(tweet_list):
    all_tweets.loc[len(all_tweets)] = process_tweet(tweet)
    if "RT @" in tweet.full_text:
        all_retweets.loc[len(all_retweets)] = process_tweet(tweet)
end = time.perf_counter()
total = fin - start
print(f'processing took {total} secs to complete')
print(f'found {len(all_tweets)} tweets all together\nfound {len(all_retweets)} retweets')

### Storing the number of retweets
file1 = open("myfile.txt","w")
L = ["This is Delhi \n","This is Paris \n","This is London \n"] 
  
# \n is placed to indicate EOL (End of Line)
file1.write("Hello \n")
file1.writelines(L)
file1.close() #to change file access modes


# NOW CHECKING THE TIMES
print('\nnow finding tweets that are posted 24 hours before, for each tweet')

for i, retweet in all_retweets.iterrows():

    # check if potential informer tweet is made within 24 hours of target tweet
    c1 = (all_tweets.loc[:,"tweet-time"] - retweet.loc['tweet-time'] < timedelta(days=1)).values.astype(int)

    # check if potenial informer tweet is posted before the target tweet
    c2 = (all_tweets.loc[:,"tweet-time"] < retweet.loc['tweet-time']).values.astype(int)

    idx = np.where( c1+c2 == 2 ) # find the tweets that are tweeted 24 hours before target tweet

    # if the array is full of 0s
    if idx[0].size == 0:
        all_retweets.drop(labels = i, axis=0) # delete that entry from the all_retweets daatabase, that stores further 
        print(f'dropped the {i}th row from consideration. Had no tweets that were posted 24 hours before')
    else:
         all_retweets.at[i,'check1-idx'] = idx[0].tolist()

# cols = ['tweet-id','tweet-text','tweet-time', 'user-id','check1-idx','informer-tweets','user-following']

### NOW GET THE LIST OF THE ACCOUNTS EACH TWEET AUTHOR FOLLOWS.

print(f'Getting the accounts that user of each target tweet follows')
start = time.perf_counter()

storage = {}
for j, tweet in all_retweets.iterrows():

    followings = []
    for friend in tweepy.Cursor(api.get_friend_ids, user_id=tweet.loc['user-id'], count = 200).items(200):
        followings.append(friend)
    # all_retweets.at[j,'following-accounts'] = followings
    all_retweets.at[j,'following-accounts'] = followings
    storage.update({tweet.at['user-id']:{'friends-id':followings}})

    if j%100:
        with open(f'user-friends\{hashtag}_dump_{j}.json', 'w') as fp:
            json.dump(storage,fp)
        storage.clear

fin = time.perf_counter()
ttl = fin - start
print(f'got the accounts that users of informer tweets follow in {total} seconds')


### Check the target tweets for users that follow them in this database.

# Check that the users 
