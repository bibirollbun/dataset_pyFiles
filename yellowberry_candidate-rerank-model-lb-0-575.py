VER = 5

import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
from collections import Counter
import cudf, itertools
print('We will use RAPIDS version',cudf.__version__)


%%time
# CACHE FUNCTIONS
def read_file(f):
    return cudf.DataFrame( data_cache[f] )
def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts/1000).astype('int32')
    df['type'] = df['type'].map(type_labels).astype('int8')
    return df

# CACHE THE DATA ON CPU BEFORE PROCESSING ON GPU
data_cache = {}
type_labels = {'clicks':0, 'carts':1, 'orders':2}
files = glob.glob('../input/otto-chunk-data-inparquet-format/*_parquet/*')
for f in files: data_cache[f] = read_file_to_cache(f)

# CHUNK PARAMETERS
READ_CT = 5
CHUNK = int( np.ceil( len(files)/6 ))
print(f'We will process {len(files)} files, in groups of {READ_CT} and chunks of {CHUNK}.')


%%time
type_weight = {0:1, 1:6, 2:3}

# USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
DISK_PIECES = 4
SIZE = 1.86e6/DISK_PIECES

# COMPUTE IN PARTS FOR MEMORY MANGEMENT
for PART in range(DISK_PIECES):
    print()
    print('### DISK PART',PART+1)
    
    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
    # => OUTER CHUNKS
    for j in range(6):
        a = j*CHUNK
        b = min( (j+1)*CHUNK, len(files) )
        print(f'Processing files {a} thru {b-1} in groups of {READ_CT}...')
        
        # => INNER CHUNKS
        for k in range(a,b,READ_CT):
            # READ FILE
            df = [read_file(files[k])]
            for i in range(1,READ_CT): 
                if k+i<b: df.append( read_file(files[k+i]) )
            df = cudf.concat(df,ignore_index=True,axis=0)
            df = df.sort_values(['session','ts'],ascending=[True,False])
            # USE TAIL OF SESSION
            df = df.reset_index(drop=True)
            df['n'] = df.groupby('session').cumcount()
            df = df.loc[df.n<30].drop('n',axis=1)
            # CREATE PAIRS
            df = df.merge(df,on='session')
            df = df.loc[ ((df.ts_x - df.ts_y).abs()< 24 * 60 * 60) & (df.aid_x != df.aid_y) ]
            # MEMORY MANAGEMENT COMPUTE IN PARTS
            df = df.loc[(df.aid_x >= PART*SIZE)&(df.aid_x < (PART+1)*SIZE)]
            # ASSIGN WEIGHTS
            df = df[['session', 'aid_x', 'aid_y','type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = df.type_y.map(type_weight)
            df = df[['aid_x','aid_y','wgt']]
            df.wgt = df.wgt.astype('float32')
            df = df.groupby(['aid_x','aid_y']).wgt.sum()
            # COMBINE INNER CHUNKS
            if k==a: tmp2 = df
            else: tmp2 = tmp2.add(df, fill_value=0)
            print(k,', ',end='')
        print()
        # COMBINE OUTER CHUNKS
        if a==0: tmp = tmp2
        else: tmp = tmp.add(tmp2, fill_value=0)
        del tmp2, df
        gc.collect()
    # CONVERT MATRIX TO DICTIONARY
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
    # SAVE TOP 40
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    tmp = tmp.loc[tmp.n<15].drop('n',axis=1)
    # SAVE PART TO DISK (convert to pandas first uses less memory)
    tmp.to_pandas().to_parquet(f'top_15_carts_orders_v{VER}_{PART}.pqt')


%%time
# USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
DISK_PIECES = 1
SIZE = 1.86e6/DISK_PIECES

# COMPUTE IN PARTS FOR MEMORY MANGEMENT
for PART in range(DISK_PIECES):
    print()
    print('### DISK PART',PART+1)
    
    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
    # => OUTER CHUNKS
    for j in range(6):
        a = j*CHUNK
        b = min( (j+1)*CHUNK, len(files) )
        print(f'Processing files {a} thru {b-1} in groups of {READ_CT}...')
        
        # => INNER CHUNKS
        for k in range(a,b,READ_CT):
            # READ FILE
            df = [read_file(files[k])]
            for i in range(1,READ_CT): 
                if k+i<b: df.append( read_file(files[k+i]) )
            df = cudf.concat(df,ignore_index=True,axis=0)
            df = df.loc[df['type'].isin([1,2])] # ONLY WANT CARTS AND ORDERS
            df = df.sort_values(['session','ts'],ascending=[True,False])
            # USE TAIL OF SESSION
            df = df.reset_index(drop=True)
            df['n'] = df.groupby('session').cumcount()
            df = df.loc[df.n<30].drop('n',axis=1)
            # CREATE PAIRS
            df = df.merge(df,on='session')
            df = df.loc[ ((df.ts_x - df.ts_y).abs()< 14 * 24 * 60 * 60) & (df.aid_x != df.aid_y) ] # 14 DAYS
            # MEMORY MANAGEMENT COMPUTE IN PARTS
            df = df.loc[(df.aid_x >= PART*SIZE)&(df.aid_x < (PART+1)*SIZE)]
            # ASSIGN WEIGHTS
            df = df[['session', 'aid_x', 'aid_y','type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = 1
            df = df[['aid_x','aid_y','wgt']]
            df.wgt = df.wgt.astype('float32')
            df = df.groupby(['aid_x','aid_y']).wgt.sum()
            # COMBINE INNER CHUNKS
            if k==a: tmp2 = df
            else: tmp2 = tmp2.add(df, fill_value=0)
            print(k,', ',end='')
        print()
        # COMBINE OUTER CHUNKS
        if a==0: tmp = tmp2
        else: tmp = tmp.add(tmp2, fill_value=0)
        del tmp2, df
        gc.collect()
    # CONVERT MATRIX TO DICTIONARY
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
    # SAVE TOP 40
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    tmp = tmp.loc[tmp.n<15].drop('n',axis=1)
    # SAVE PART TO DISK (convert to pandas first uses less memory)
    tmp.to_pandas().to_parquet(f'top_15_buy2buy_v{VER}_{PART}.pqt')


%%time
# USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
DISK_PIECES = 4
SIZE = 1.86e6/DISK_PIECES

# COMPUTE IN PARTS FOR MEMORY MANGEMENT
for PART in range(DISK_PIECES):
    print()
    print('### DISK PART',PART+1)
    
    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
    # => OUTER CHUNKS
    for j in range(6):
        a = j*CHUNK
        b = min( (j+1)*CHUNK, len(files) )
        print(f'Processing files {a} thru {b-1} in groups of {READ_CT}...')
        
        # => INNER CHUNKS
        for k in range(a,b,READ_CT):
            # READ FILE
            df = [read_file(files[k])]
            for i in range(1,READ_CT): 
                if k+i<b: df.append( read_file(files[k+i]) )
            df = cudf.concat(df,ignore_index=True,axis=0)
            df = df.sort_values(['session','ts'],ascending=[True,False])
            # USE TAIL OF SESSION
            df = df.reset_index(drop=True)
            df['n'] = df.groupby('session').cumcount()
            df = df.loc[df.n<30].drop('n',axis=1)
            # CREATE PAIRS
            df = df.merge(df,on='session')
            df = df.loc[ ((df.ts_x - df.ts_y).abs()< 24 * 60 * 60) & (df.aid_x != df.aid_y) ]
            # MEMORY MANAGEMENT COMPUTE IN PARTS
            df = df.loc[(df.aid_x >= PART*SIZE)&(df.aid_x < (PART+1)*SIZE)]
            # ASSIGN WEIGHTS
            df = df[['session', 'aid_x', 'aid_y','ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = 1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800)
            df = df[['aid_x','aid_y','wgt']]
            df.wgt = df.wgt.astype('float32')
            df = df.groupby(['aid_x','aid_y']).wgt.sum()
            # COMBINE INNER CHUNKS
            if k==a: tmp2 = df
            else: tmp2 = tmp2.add(df, fill_value=0)
            print(k,', ',end='')
        print()
        # COMBINE OUTER CHUNKS
        if a==0: tmp = tmp2
        else: tmp = tmp.add(tmp2, fill_value=0)
        del tmp2, df
        gc.collect()
    # CONVERT MATRIX TO DICTIONARY
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
    # SAVE TOP 40
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    tmp = tmp.loc[tmp.n<20].drop('n',axis=1)
    # SAVE PART TO DISK (convert to pandas first uses less memory)
    tmp.to_pandas().to_parquet(f'top_20_clicks_v{VER}_{PART}.pqt')


# FREE MEMORY
del data_cache, tmp
_ = gc.collect()


def load_test():    
    dfs = []
    for e, chunk_file in enumerate(glob.glob('../input/otto-chunk-data-inparquet-format/test_parquet/*')):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts/1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_labels).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

test_df = load_test()
print('Test data has shape',test_df.shape)
test_df.head()


def load_train():    
    dfs = []
    for e, chunk_file in enumerate(glob.glob('../input/otto-chunk-data-inparquet-format/train_parquet/*')):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts/1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_labels).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

train_df = load_train()
print('Test data has shape',train_df.shape)
train_df.head()


# %%time
# def pqt_to_dict(df):
#     return df.groupby('aid_x').aid_y.apply(list).to_dict()
# # LOAD THREE CO-VISITATION MATRICES
# top_20_clicks = pqt_to_dict( pd.read_parquet(f'top_20_clicks_v{VER}_0.pqt') )
# for k in range(1,DISK_PIECES): 
#     top_20_clicks.update( pqt_to_dict( pd.read_parquet(f'top_20_clicks_v{VER}_{k}.pqt') ) )
# top_20_buys = pqt_to_dict( pd.read_parquet(f'top_15_carts_orders_v{VER}_0.pqt') )
# for k in range(1,DISK_PIECES): 
#     top_20_buys.update( pqt_to_dict( pd.read_parquet(f'top_15_carts_orders_v{VER}_{k}.pqt') ) )
# top_20_buy2buy = pqt_to_dict( pd.read_parquet(f'top_15_buy2buy_v{VER}_0.pqt') )

# # TOP CLICKS AND ORDERS IN TEST
# top_clicks = test_df.loc[test_df['type']=='clicks','aid'].value_counts().index.values[:20]
# top_orders = test_df.loc[test_df['type']=='orders','aid'].value_counts().index.values[:20]

# print('Here are size of our 3 co-visitation matrices:')
# print( len( top_20_clicks ), len( top_20_buy2buy ), len( top_20_buys ) )


import numpy as np
import pandas as pd
import gc, glob
from tqdm import tqdm
import xgboost as xgb


def pqt_to_dict(df):
    return df.groupby('aid_x').aid_y.apply(list).to_dict()

top_20_clicks = {}
for k in range(DISK_PIECES):
    top_20_clicks.update(
        pqt_to_dict(pd.read_parquet(f'top_20_clicks_v{VER}_{k}.pqt'))
    )

top_20_buys = {}
for k in range(DISK_PIECES):
    top_20_buys.update(
        pqt_to_dict(pd.read_parquet(f'top_15_carts_orders_v{VER}_{k}.pqt'))
    )

top_20_buy2buy = pqt_to_dict(
    pd.read_parquet(f'top_15_buy2buy_v{VER}_0.pqt')
)


FEATURE_COLS = [
    'cnt', 'recency', 'max_type',
    'in_click_cov', 'in_buy_cov', 'in_buy2buy'
]

def generate_candidates(hist_aids, max_candidates=100):
    recents = list(dict.fromkeys(hist_aids[::-1]))[:20]
    covisit = []

    for a in recents:
        covisit += top_20_clicks.get(a, [])
        covisit += top_20_buys.get(a, [])
        covisit += top_20_buy2buy.get(a, [])

    return list(dict.fromkeys(recents + covisit))[:max_candidates]


def make_features(hist_df, candidates):
    last_ts = hist_df['ts'].max()

    aid_counts = hist_df['aid'].value_counts()
    aid_last_ts = hist_df.groupby('aid')['ts'].max()
    aid_type_max = hist_df.groupby('aid')['type'].max()

    X = np.zeros((len(candidates), 6), dtype=np.float32)

    for i, aid in enumerate(candidates):
        X[i, 0] = aid_counts.get(aid, 0)
        X[i, 1] = last_ts - aid_last_ts.get(aid, last_ts)
        X[i, 2] = aid_type_max.get(aid, -1)
        X[i, 3] = aid in top_20_clicks
        X[i, 4] = aid in top_20_buys
        X[i, 5] = aid in top_20_buy2buy

    return X


def make_labels(candidates, future_aids):
    return np.fromiter(
        (aid in future_aids for aid in candidates),
        dtype=np.int8,
        count=len(candidates)
    )


import numpy as np
import pandas as pd
import glob, os, gc
import xgboost as xgb

os.makedirs("train_chunks", exist_ok=True)

MAX_SESSIONS_PER_CHUNK = 250_000  # adjust so each chunk fits in memory
chunk_id = 0

X_buf, y_buf, group_buf = [], [], []
session_count = 0

train_files = glob.glob('../input/otto-chunk-data-inparquet-format/train_parquet/*')

def flush_chunk(X_buf, y_buf, group_buf, chunk_id):
    X_chunk = np.vstack(X_buf)
    y_chunk = np.concatenate(y_buf)
    group_chunk = np.array(group_buf, dtype=np.int32)
    np.savez_compressed(f"train_chunks/chunk_{chunk_id}.npz",
                        X=X_chunk, y=y_chunk, group=group_chunk)
    # print(f"Saved chunk {chunk_id}, shape {X_chunk.shape}")
    X_buf.clear(); y_buf.clear(); group_buf.clear()
    del X_chunk, y_chunk, group_chunk
    gc.collect()


NUM_FEATURES = 6


import cudf, pandas as pd, numpy as np, gc, glob, os
from tqdm import tqdm
import xgboost as xgb

VER = 5
DISK_PIECES = 4  # adjust based on GPU memory
MAX_CANDIDATES = 100  # max candidates per session
FEATURE_COLS = ['cnt','recency','max_type','in_click_cov','in_buy_cov','in_buy2buy']
NUM_FEATURES = len(FEATURE_COLS)
MAX_SESSIONS_PER_CHUNK = 250_000  # tune to fit GPU memory

os.makedirs("train_chunks", exist_ok=True)

# # load co-visitation matrices (already computed with GPU)
# def pqt_to_dict(df):
#     return df.groupby('aid_x').aid_y.apply(list).to_dict()

# top_20_clicks = {}
# top_20_buys = {}
# top_20_buy2buy = {}
# for k in range(DISK_PIECES):
#     top_20_clicks.update(pqt_to_dict(pd.read_parquet(f'top_20_clicks_v{VER}_{k}.pqt')))
#     top_20_buys.update(pqt_to_dict(pd.read_parquet(f'top_15_carts_orders_v{VER}_{k}.pqt')))
#     top_20_buy2buy.update(pqt_to_dict(pd.read_parquet(f'top_15_buy2buy_v{VER}_{k}.pqt')))

# GPU-friendly generate_candidates
def generate_candidates(hist_aids):
    recents = list(dict.fromkeys(hist_aids[::-1]))[:20]
    covisit = []
    for a in recents:
        covisit += top_20_clicks.get(a, [])
        covisit += top_20_buys.get(a, [])
        covisit += top_20_buy2buy.get(a, [])
    candidates = list(dict.fromkeys(recents + covisit))
    return candidates[:MAX_CANDIDATES]

# GPU-friendly feature generation
def make_features(hist_df, candidates):
    last_ts = hist_df['ts'].max()
    aid_counts = hist_df['aid'].value_counts()
    aid_last_ts = hist_df.groupby('aid')['ts'].max()
    aid_type_max = hist_df.groupby('aid')['type'].max()
    feats = []
    for aid in candidates:
        feats.append([
            aid_counts.get(aid,0),
            last_ts - aid_last_ts.get(aid,last_ts),
            aid_type_max.get(aid,-1),
            int(aid in top_20_clicks),
            int(aid in top_20_buys),
            int(aid in top_20_buy2buy)
        ])
    return np.array(feats, dtype=np.float32)

def make_labels(candidates, future_aids):
    return np.array([1 if aid in future_aids else 0 for aid in candidates], dtype=np.int8)

def flush_chunk(X_buf, y_buf, group_buf, chunk_id):
    np.savez_compressed(f"train_chunks/chunk_{chunk_id}.npz",
                        X=X_buf, y=y_buf, group=group_buf)
    X_buf.fill(0); y_buf.fill(0); group_buf.fill(0)
    gc.collect()

# Read train files in chunks on GPU
train_files = glob.glob('../input/otto-chunk-data-inparquet-format/train_parquet/*')

for PART in range(DISK_PIECES):
    print(f"\nProcessing PART {PART+1}/{DISK_PIECES}")
    
    # preallocate buffers
    X_buf = np.zeros((MAX_SESSIONS_PER_CHUNK * MAX_CANDIDATES, NUM_FEATURES), dtype=np.float32)
    y_buf = np.zeros((MAX_SESSIONS_PER_CHUNK * MAX_CANDIDATES,), dtype=np.int8)
    group_buf = np.zeros((MAX_SESSIONS_PER_CHUNK,), dtype=np.int32)
    buf_ptr = 0
    session_ptr = 0
    chunk_id = 0
    
    for f in tqdm(train_files, desc="Files"):
        # load chunk to GPU
        df = cudf.read_parquet(f)
        df.ts = (df.ts // 1000).astype('int32')
        df['type'] = df['type'].map({'clicks':0,'carts':1,'orders':2}).astype('int8')

        # process only aid range for this PART
        aid_min = PART * 1.86e6 / DISK_PIECES
        aid_max = (PART+1) * 1.86e6 / DISK_PIECES
        df = df[(df['aid'] >= aid_min) & (df['aid'] < aid_max)]

        # sort per session
        df = df.sort_values(['session','ts'], ascending=[True,False])
        df['n'] = df.groupby('session').cumcount()
        df = df[df.n < 30].drop('n', axis=1)

        sessions_in_file = df['session'].nunique()
        for session, df_sess in df.groupby('session', sort=False):
            if len(df_sess) < 3:
                continue

            split = int(len(df_sess) * 0.7)
            hist = df_sess.iloc[:split].to_pandas()
            future = df_sess.iloc[split:].to_pandas()
            future_aids = set(future['aid'].values)

            candidates = generate_candidates(hist['aid'].values)
            X_sess = make_features(hist, candidates)
            y_sess = make_labels(candidates, future_aids)
            n_cand = len(candidates)

            X_buf[buf_ptr:buf_ptr+n_cand, :] = X_sess
            y_buf[buf_ptr:buf_ptr+n_cand] = y_sess
            group_buf[session_ptr] = n_cand

            buf_ptr += n_cand
            session_ptr += 1

            if session_ptr == MAX_SESSIONS_PER_CHUNK:
                flush_chunk(X_buf[:buf_ptr,:], y_buf[:buf_ptr], group_buf[:session_ptr], chunk_id)
                buf_ptr = 0
                session_ptr = 0
                chunk_id += 1

        del df
        gc.collect()

    if session_ptr > 0:
        flush_chunk(X_buf[:buf_ptr,:], y_buf[:buf_ptr], group_buf[:session_ptr], chunk_id)


model = xgb.XGBRanker(
    objective='rank:pairwise',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=42
)

chunk_files = sorted(glob.glob("train_chunks/chunk_*.npz"))

for f in tqdm(chunk_files):
    data = np.load(f)
    X_chunk = data['X']
    y_chunk = data['y']
    group_chunk = data['group']
    
    model.fit(
        X_chunk, y_chunk, group=group_chunk,
        xgb_model=model if chunk_files.index(f) > 0 else None,
        verbose=True
    )
    
    del X_chunk, y_chunk, group_chunk, data
    gc.collect()


test_files = glob.glob('../input/otto-chunk-data-inparquet-format/test_parquet/*')
preds = {}

for f in tqdm(test_files):
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype('int32')
    df['type'] = df['type'].map({'clicks':0,'carts':1,'orders':2}).astype('int8')

    for session, df_sess in df.groupby('session', sort=False):
        candidates = generate_candidates(df_sess['aid'].values)
        X = make_features(df_sess, candidates)
        scores = model.predict(X)
        top_idx = np.argsort(-scores)[:20]
        preds[session] = np.array(candidates)[top_idx]

    del df
    gc.collect()


rows = []

for session, aids in preds.items():
    labels = " ".join(map(str, aids))
    rows.append((f"{session}_clicks", labels))
    rows.append((f"{session}_carts", labels))
    rows.append((f"{session}_orders", labels))

submission = pd.DataFrame(rows, columns=["session_type", "labels"])
submission.to_csv("submission.csv", index=False)


X_train = np.vstack(X_buf)
y_train = np.concatenate(y_buf)
group = np.array(group_buf, dtype=np.int32)

del X_buf, y_buf, group_buf
gc.collect()

model = xgb.XGBRanker(
    objective='rank:pairwise',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=42
)

model.fit(X_train, y_train, group=group)


test_files = glob.glob('../input/otto-chunk-data-inparquet-format/test_parquet/*')
preds = {}

for f in tqdm(test_files):
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype('int32')
    df['type'] = df['type'].map({'clicks':0,'carts':1,'orders':2}).astype('int8')

    for session, df_sess in df.groupby('session', sort=False):
        candidates = generate_candidates(df_sess['aid'].values)
        X = make_features(df_sess, candidates)

        scores = model.predict(X)
        top_idx = np.argsort(-scores)[:20]

        preds[session] = np.array(candidates)[top_idx]

    del df
    gc.collect()


rows = []

for session, aids in preds.items():
    labels = " ".join(map(str, aids))
    rows.append((f"{session}_clicks", labels))
    rows.append((f"{session}_carts", labels))
    rows.append((f"{session}_orders", labels))

submission = pd.DataFrame(rows, columns=["session_type", "labels"])
submission.to_csv("submission.csv", index=False)


# import pandas as pd
# import numpy as np
# import glob
# import itertools
# import xgboost as xgb
# from collections import Counter
# import os

# type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}
# DISK_PIECES = 4
# VER = 1


# def generate_candidates(hist_aids):
#     unique_aids = list(dict.fromkeys(hist_aids[::-1]))
#     covisit_clicks = list(itertools.chain(*[top_20_clicks.get(a, []) for a in unique_aids if a in top_20_clicks]))
#     covisit_buy = list(itertools.chain(*[top_20_buy2buy.get(a, []) for a in unique_aids if a in top_20_buy2buy]))
#     candidates = list(dict.fromkeys(unique_aids + covisit_clicks + covisit_buy))
#     return candidates


# np.random.seed(42)

# sess_with_buy = (
#     train_df
#     .groupby('session')['type']
#     .max()
#     .loc[lambda x: x >= 1]
#     .index
# )

# keep_sessions = np.random.choice(
#     sess_with_buy,
#     # size=min(300_000, len(sess_with_buy)),
#     size=len(sess_with_buy),
#     replace=False
# )

# train_df = train_df[train_df['session'].isin(keep_sessions)]
# train_df = train_df.sort_values(['session','ts'])


len(train_df)


def generate_candidates(hist_aids, max_candidates=100):
    recents = list(dict.fromkeys(hist_aids[::-1]))[:20]

    covisit = []
    for a in recents:
        covisit += top_20_clicks.get(a, [])
        covisit += top_20_buys.get(a, [])
        covisit += top_20_buy2buy.get(a, [])

    candidates = list(dict.fromkeys(recents + covisit))
    return candidates[:max_candidates]

def make_features(hist_df, candidates):
    feats = []
    last_ts = hist_df['ts'].max()

    aid_counts = hist_df['aid'].value_counts()
    aid_last_ts = hist_df.groupby('aid')['ts'].max()
    aid_type_max = hist_df.groupby('aid')['type'].max()

    for aid in candidates:
        feats.append([
            aid,
            aid_counts.get(aid, 0),
            last_ts - aid_last_ts.get(aid, last_ts),
            aid_type_max.get(aid, -1),
            aid in top_20_clicks,
            aid in top_20_buys,
            aid in top_20_buy2buy,
        ])

    return pd.DataFrame(
        feats,
        columns=[
            'aid',
            'cnt',
            'recency',
            'max_type',
            'in_click_cov',
            'in_buy_cov',
            'in_buy2buy'
        ]
    )
    
def make_labels(candidates, future_aids):
    return np.array([1 if aid in future_aids else 0 for aid in candidates], dtype=np.int8)


# infer feature dimension
dummy_feats = make_features(
    train_df.iloc[:10],
    generate_candidates(train_df.iloc[:10]['aid'].values)
)
FEATURE_COLS = [c for c in dummy_feats.columns if c != 'aid']
D = len(FEATURE_COLS)


D


from tqdm import tqdm
import numpy as np

X_buf, y_buf, group_buf = [], [], []

MAX_SESSIONS_PER_CHUNK = 5000

def flush_chunk(X_buf, y_buf, group_buf, chunk_id):
    X = np.vstack(X_buf).astype(np.float32)
    y = np.concatenate(y_buf).astype(np.int8)
    group = np.array(group_buf, dtype=np.int32)

    np.savez_compressed(
        f"xgb_chunk_{chunk_id}.npz",
        X=X,
        y=y,
        group=group
    )

    X_buf.clear()
    y_buf.clear()
    group_buf.clear()


chunk_id = 0
session_count = 0

for session, df_sess in tqdm(
        train_df.groupby('session', sort=False),
        total=train_df['session'].nunique()
    ):

    if len(df_sess) < 3:
        continue

    split = int(len(df_sess) * 0.7)
    hist = df_sess.iloc[:split]
    future = df_sess.iloc[split:]
    future_aids = set(future['aid'].values)

    candidates = generate_candidates(hist['aid'].values)
    feats = make_features(hist, candidates)

    X_buf.append(
        feats[FEATURE_COLS].to_numpy(dtype=np.float32)
    )
    y_buf.append(
        make_labels(candidates, future_aids)
    )
    group_buf.append(len(candidates))

    session_count += 1

    if session_count % MAX_SESSIONS_PER_CHUNK == 0:
        flush_chunk(X_buf, y_buf, group_buf, chunk_id)
        chunk_id += 1

# flush remainder
if X_buf:
    flush_chunk(X_buf, y_buf, group_buf, chunk_id)


# from tqdm import tqdm

# X_all, y_all, group = [], [], []

# for session, df_sess in tqdm(
#         train_df.groupby('session'),
#         total=train_df['session'].nunique()
#     ):
#     if len(df_sess) < 3:
#         continue

#     split = int(len(df_sess) * 0.7)
#     hist = df_sess.iloc[:split]
#     future = df_sess.iloc[split:]
#     future_aids = set(future['aid'].values)

#     candidates = generate_candidates(hist['aid'].values)
#     feats = make_features(hist, candidates)
    
#     # Drop 'aid' before appending to training data
#     X_all.append(feats.drop(columns='aid', errors='ignore'))
#     y_all.append(make_labels(candidates, future_aids))
#     group.append(len(candidates))


X_train = pd.concat(X_all, ignore_index=True)
y_train = np.concatenate(y_all)
group = np.array(group)


import xgboost as xgb

model = xgb.XGBRanker(
    objective='rank:pairwise',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42
)

model.fit(
    X_train,
    y_train,
    group=group
)


from tqdm import tqdm

sessions = test_df['session'].values
unique_sessions, session_ptrs = np.unique(sessions, return_index=True)

preds = {}

for i, sess in enumerate(tqdm(unique_sessions, total=len(unique_sessions))):
    start = session_ptrs[i]
    end = session_ptrs[i+1] if i+1 < len(session_ptrs) else len(test_df)

    df_sess = test_df.iloc[start:end]

    candidates = generate_candidates(df_sess['aid'].values)
    X = make_features(df_sess, candidates)
    aids = X['aid'].values
    
    # convert all features to numeric
    X_model = X.drop(columns='aid').copy()
    X_model = X_model.fillna(0)        # optional: handle NaNs
    X_model = X_model.astype(float)     # convert int/bool to float
    
    # prediction
    scores = model.predict(X_model)
    top_idx = np.argsort(-scores)[:20]
    
    preds[sess] = aids[top_idx]


import pandas as pd

rows = []

for session, aids in preds.items():
    labels = " ".join(map(str, aids))

    rows.append((f"{session}_clicks", labels))
    rows.append((f"{session}_carts",  labels))
    rows.append((f"{session}_orders", labels))

submission = pd.DataFrame(rows, columns=["session_type", "labels"])
submission.to_csv("submission.csv", index=False)

submission.head()

