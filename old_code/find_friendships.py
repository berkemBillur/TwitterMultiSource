import json
import pandas as pd
import os.path

    
def load_informer_df():
    path = f'{hashtag}{os.path.sep}informer_database.json'
    with open(path) as jf:
        data = json.load(jf)
    return data

hashtag = 'UkraineWar'

informer_data = load_informer_df()

full_df = pd.DataFrame.from_dict(informer_data,orient='index')

df1 = full_df.filter(['user-id','friend-ids'],axis=1)

df = pd.DataFrame(df1.groupby(['user-id']), columns = ['user-id','friend-ids'])

users = set(df['user-id'].to_list())

print(len(informer_df))