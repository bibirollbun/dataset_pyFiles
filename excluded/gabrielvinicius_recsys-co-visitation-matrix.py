### import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm.notebook import tqdm
import glob
import numpy as np
import multiprocessing
import os
import pickle

import glob
import gc
from collections import Counter
import itertools


class CFG:
    train_chunks_path = '/kaggle/input/otto-validation/train_parquet'
    test_chunks_path = '/kaggle/input/otto-validation/test_parquet'

    debug = True
    top_aids = 20

    type_labels = {'clicks':0,'carts':1,'orders':2}
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}


%%time
all_data_df = pd.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/train.parquet')
print(all_data_df.shape)
all_data_df.head()


print('Nยบ of sessions (Users) = ', all_data_df['session'].nunique())
print('Nยบ of aids (Products) = ', all_data_df['aid'].nunique())


all_data_df.info()


del all_data_df


chunks = glob.glob(CFG.train_chunks_path + '/*')
chunks = chunks[:10] if CFG.debug else chunks
print('Debug Mode = ', CFG.debug,'\nNยบ of Chunks = ', len(chunks))


def get_covisitation_matrix(df_chunk, max_session_interactions = 30):
    
    #FILTER MAX INTERACTIONS OF SESSIONS
    df = df_chunk.reset_index(drop = True)
    df['n'] = df.groupby('session').cumcount()
    df = df.loc[df.n<max_session_interactions].drop('n',axis=1)
    
    #GET PAIRS: ITENS VISETED AT THE SAME DAY
    df = df.merge(df,on='session')
    df = df.loc[ ((df.ts_x - df.ts_y).abs()< 24 * 60 * 60) & (df.aid_x != df.aid_y)]
    df = df[['session', 'aid_x', 'aid_y','type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])

    # ASSIGN WEIGHTS
    df['wgt'] = 1
    #df['wgt'] = 1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800)
    df = df[['aid_x','aid_y','wgt']]
    df.wgt = df.wgt.astype('float32')
    df = df.groupby(['aid_x','aid_y']).wgt.sum().reset_index()

    return df







pairs_df = pd.DataFrame()
for chunk in tqdm(chunks, desc='Get pairs'):
    
    chunk_df = pd.read_parquet(chunk)
    chunk_df.ts = (chunk_df.ts/1000).astype('int32')
    chunk_df['type'] = chunk_df['type'].map(CFG.type_labels).astype('int8')
    
    pairs_chunk_df = get_covisitation_matrix(chunk_df)
    pairs_df = pd.concat([pairs_df, pairs_chunk_df])

    del chunk_df,pairs_chunk_df 
    gc.collect()

# CONVERT MATRIX TO DICTIONARY
pairs_df = pairs_df.reset_index()
pairs_df = pairs_df.sort_values(['aid_x','wgt'],ascending=[True,False])
# SAVE TOP 
pairs_df = pairs_df.reset_index(drop=True)
pairs_df['n'] = pairs_df.groupby('aid_x').aid_y.cumcount()
pairs_df = pairs_df.loc[pairs_df.n<CFG.top_aids].drop('n',axis=1)

pairs_df.to_parquet(f'top_{CFG.top_aids}_clicks.parquet', index = False)


del pairs_df


def load_df(folder_path):
    df = pd.DataFrame()
    for chunk_path in tqdm(glob.glob(folder_path + '/*')):
        chunk = pd.read_parquet(chunk_path)
        chunk.ts = (chunk.ts/1000).astype('int32')
        chunk['type'] = chunk['type'].map(CFG.type_labels).astype('int8')
        df = pd.concat([df,chunk])
    return df.reset_index(drop = True)

def pqt_to_dict(df):
    return df.groupby('aid_x').aid_y.apply(list).to_dict()


test_df = load_df(CFG.test_chunks_path)
print(test_df.shape)
test_df.head()


print('Nยบ of sessions (Users) = ', test_df['session'].nunique())
print('Nยบ of aids (Products) = ', test_df['aid'].nunique())


# TOP CLICKS AIDs
top_clicks = test_df.loc[test_df['type']==0,'aid'].value_counts().index.values[:20]
top_clicks


covis_clicks = pqt_to_dict(pd.read_parquet(f'top_{CFG.top_aids}_clicks.parquet'))


def suggest_clicks(df):
    # USER HISTORY AIDS AND TYPES
    aids=df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1] ))
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids)>=20:
        weights=np.logspace(0.1,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * CFG.type_weight_multipliers[t]
        sorted_aids = [k for k,v in aids_temp.most_common(20)]
        return sorted_aids
    # USE "CLICKS" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[covis_clicks[aid] for aid in unique_aids if aid in covis_clicks]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[:20 - len(unique_aids)]
    # USE TOP20 TEST CLICKS
    return result + list(top_clicks)[:20-len(result)]


%%time
pred_df_clicks = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
    lambda x: suggest_clicks(x)
)


pred_df_clicks = pred_df_clicks.reset_index()
pred_df_clicks = pred_df_clicks.rename(columns = {0:'pred_labels'})
pred_df_clicks


%%time
test_labels = pd.read_parquet('../input/otto-validation/test_labels.parquet')
test_labels = test_labels[test_labels['type'] == 'clicks']


test_labels = test_labels.merge(pred_df_clicks, 
                                   on='session', 
                                   how = 'left'
                                  )

test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.pred_labels))), axis=1)
test_labels['length_labels'] = test_labels.ground_truth.str.len().clip(0,20)
test_labels.head(10)


recall = test_labels['hits'].sum()/test_labels['length_labels'].sum()
print('Clicks Recall@20 = ', recall)

