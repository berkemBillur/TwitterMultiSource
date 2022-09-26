import os
from hash_crawler_v3 import crawler

hashtags = ['BLM']
friend_path  = f'user-friends{os.path.sep}'
maxTweets = 10000
job = ['load']
key_paths = ['apikeys.txt', 'apikeys2.txt']
num_of_keys = [45,83]
storage_save_freq = 100
# num_of_keys = [1,1]


for i,hashtag in enumerate(hashtags):

    c = crawler( job[i] , maxTweets, friend_path, hashtag, key_paths, num_of_keys,storage_save_freq)

    # c.get_api_list('apikeys.txt', 45)
    
    c.stage1_extraction()
