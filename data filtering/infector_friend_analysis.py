## This script looks into how the number of informers effects the probability of an infector being a friend of the source

import json
import pandas as pd
import time
from datetime import datetime
import os.path


hashtags = ['avengers','blm','borisjohnson','brexit','climatechange','covid','gaza','loveisland','monkeypox','nhs','olivianewtonjohn','supercup','UkraineWar']
# hashtags = ['UkraineWar']

results = {}

for hashtag in hashtags:

    path = f'{hashtag}{os.path.sep}updated_informers_data.json'
    with open(path) as jf:
        data = json.load(jf)

    df = pd.DataFrame.from_dict(data,orient='index')
    max_infrmr = df['num-informers'].max()

    info = {}

    for i in range(max_infrmr+1):
        db = df.loc[df['num-informers']==i]
        n = len(db) # the number of tweets with i informers
        if n != 0:
            num_infector_friends = db['infector-is-friend'].sum()
            info.update( {i : [ num_infector_friends, n] } )
        else:
            info.update( {i : [ None, None ] } )
    print(f'done {hashtag}')

    breakdown = pd.DataFrame.from_dict(info,orient='index', columns = ['infector-friends','number-of-i-informers'])
    n = len(breakdown)

    info = pd.DataFrame(breakdown.iloc[1:10])
    info[len(info)+1] = pd.DataFrame(breakdown.iloc[10:20].sum())
    info[len(info)+1] = pd.DataFrame(breakdown.iloc[20:40].sum())
    info[len(info)+1] = pd.DataFrame(breakdown.iloc[40:80].sum())
    info[len(info)+1] = pd.DataFrame(breakdown.iloc[80:].sum())

    results.update( {hashtag:find_percent(info) } )