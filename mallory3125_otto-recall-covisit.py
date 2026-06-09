import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
import cudf, cupy
import gc
import glob
import itertools
print('Using RAPIDS version',cudf.__version__)


def cache_data_to_memory(files):
    """
    Cache all parquet files to CPU RAM for faster GPU processing.
    """
    print("Caching data to CPU RAM...")
    data_cache = {}
    for f in files:
        data_cache[f] = pd.read_parquet(f)
    
    print(f'Cached {len(files)} files to memory.')
    return data_cache

def read_file(f):
    return cudf.DataFrame(data_cache[f])


# Step 1: Split your large files into smaller chunks
def split_large_files(input_files, output_dir='./chunked_data', rows_per_chunk=5_000_000):
    """Split large parquet files into smaller chunks"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_files = []
    for file_path in input_files:
        print(f"Splitting {file_path}...")
        df = pd.read_parquet(file_path)
        
        # Split into chunks
        num_chunks = int(np.ceil(len(df) / rows_per_chunk))
        for i in range(num_chunks):
            start_idx = i * rows_per_chunk
            end_idx = min((i + 1) * rows_per_chunk, len(df))
            chunk_df = df.iloc[start_idx:end_idx]
            
            # Save chunk
            chunk_file = f'{output_dir}/chunk_{os.path.basename(file_path)}_{i}.parquet'
            chunk_df.to_parquet(chunk_file)
            chunk_files.append(chunk_file)
            # print(f"  Created {chunk_file} with {len(chunk_df)} rows")
    
    return chunk_files



def build_covisit_matrix_gpu(
    data_cache,
    type1_filter=None,
    type2_filter=None,
    type1_weights=None,
    type2_weights=None,
    max_session_length=30,
    max_days_elapsed=1,
    top_k=15,
    disk_pieces=6,
    read_ct=5,
    output_prefix='covisit_matrix'
):
    """
    Build co-visit matrix using cuDF GPU acceleration from cached data.

    """
    
    # Set default weights
    if type1_weights is None:
        type1_weights = {0: 1, 1: 1, 2: 1}
    if type2_weights is None:
        type2_weights = {0: 1, 1: 1, 2: 1}
    

    files = list(data_cache.keys())
    
    # CHUNK PARAMETERS
    CHUNK = int(np.ceil(len(files) / 6))
    print(f'Processing {len(files)} files, in groups of {read_ct} and chunks of {CHUNK}.')
    
    # Calculate size for disk pieces
    SIZE = 1.86e6 / disk_pieces
    
    # PROCESS IN PARTS FOR MEMORY MANAGEMENT
    for PART in range(disk_pieces):
        print(f'\n### DISK PART {PART + 1}')
        
        # OUTER CHUNKS
        for j in range(6):
            a = j * CHUNK
            b = min((j + 1) * CHUNK, len(files))
            print(f'Processing files {a} thru {b-1} in groups of {read_ct}...')
            
            # INNER CHUNKS
            for k in range(a, b, read_ct):
                # READ FILES
                df = [read_file(files[k])]
                
                for i in range(1, read_ct):
                    if k + i < b:
                        df.append(read_file(files[k + i]))
                df = cudf.concat(df, ignore_index=True, axis=0)
                df = df.sort_values(['session', 'ts'], ascending=[True, False])
                
                # USE TAIL OF SESSION
                df = df.reset_index(drop=True)
                df['n'] = df.groupby('session').cumcount()
                df = df.loc[df.n < max_session_length].drop('n', axis=1)
                
                # CREATE PAIRS
                df = df.merge(df, on='session', suffixes=('_x', '_y'))
                df = df.loc[((df.ts_x - df.ts_y).abs() < max_days_elapsed * 24 * 60 * 60) & (df.aid_x != df.aid_y)]
                
                # FILTER BY TYPE
                if type1_filter is not None:
                    df = df.loc[df.type_x.isin(type1_filter)]
                  
                if type2_filter is not None:
                    df = df.loc[df.type_y.isin(type2_filter)]
                  
                
                # MEMORY MANAGEMENT - COMPUTE IN PARTS
                df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]
               
                # ASSIGN WEIGHTS (multiply both type weights)
                df = df[['session', 'aid_x', 'aid_y', 'type_x', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
                
                # Map weights for both types
                df['wgt_x'] = df.type_x.map(type1_weights).fillna(1)
                df['wgt_y'] = df.type_y.map(type2_weights).fillna(1)
                df['wgt'] = (df.wgt_x * df.wgt_y).astype('float32')
                
                df = df[['aid_x', 'aid_y', 'wgt']]
                df = df.groupby(['aid_x', 'aid_y']).wgt.sum()
                
                # COMBINE INNER CHUNKS
                if k == a:
                    tmp2 = df
                else:
                    tmp2 = tmp2.add(df, fill_value=0)
                print(k, ', ', end='')
            
            print()
            # COMBINE OUTER CHUNKS
            if a == 0:
                tmp = tmp2
            else:
                tmp = tmp.add(tmp2, fill_value=0)
            del tmp2, df
            gc.collect()
        
        # CONVERT MATRIX TO DATAFRAME
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])
        
        # SAVE TOP K
        tmp = tmp.reset_index(drop=True)
        tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
        tmp = tmp.loc[tmp.n < top_k].drop('n', axis=1)
        
        # SAVE PART TO DISK
        output_file = f'{output_prefix}_{PART}.pqt'
        tmp.to_pandas().to_parquet(output_file)
        print(f'Saved {output_file}')
        
        del tmp
        gc.collect()
    
    print(f'\nCompleted! Output saved as {output_prefix}_*.pqt')



# Split your files first
original_files = glob.glob('../input/otto-full-optimized-memory-footprint/*.parquet')
chunked_files = split_large_files(original_files, rows_per_chunk=5_000_000)



%%time
# Now cache the chunked files
data_cache = cache_data_to_memory(chunked_files)


%%time
DISK_PIECES=4
# Process with appropriate parameters for ~20 chunks
build_covisit_matrix_gpu(
    data_cache=data_cache,
    type1_filter=[0,1,2],
    type2_filter=[1,2],
    type1_weights={0: 1,1: 2,2: 2},
    type2_weights={1:1, 2: 3},
    disk_pieces=DISK_PIECES,  # Now you can use more pieces
    read_ct=3,      # Process 3 chunks at a time
    output_prefix='covisit_all_to_cart-order'
)

build_covisit_matrix_gpu(
    data_cache=data_cache,
    type1_filter=[0,1,2],
    type2_filter=[0],
    type1_weights={0: 1,1: 2,2: 2},
    type2_weights={0:1},
    disk_pieces=DISK_PIECES,  # Now you can use more pieces
    read_ct=3,      # Process 3 chunks at a time
    output_prefix='covisit_all_to_click'
)

build_covisit_matrix_gpu(
    data_cache=data_cache,
    type1_filter=[0],
    type2_filter=[1,2],
    type1_weights={0: 1,},
    type2_weights={1: 1, 2: 3},
    disk_pieces=DISK_PIECES,  # Now you can use more pieces
    read_ct=3,      # Process 3 chunks at a time
    output_prefix='covisit_click_to_buy'
)




# Clean up cache when done
del data_cache
gc.collect()


test_df = pd.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/test.parquet')
print('Test data has shape',test_df.shape)
test_df.head()


%%time
def pqt_to_dict(df):
    return df.groupby('aid_x').aid_y.apply(list).to_dict()
    
top_20_all_to_cart_order = pqt_to_dict( pd.read_parquet(f'/kaggle/working/covisit_all_to_cart-order_0.pqt') )
for k in range(1,DISK_PIECES): 
    top_20_all_to_cart_order.update(pqt_to_dict( pd.read_parquet(f'/kaggle/working/covisit_all_to_cart-order_{k}.pqt') ) )

top_20_all_to_click = pqt_to_dict( pd.read_parquet(f'/kaggle/working/covisit_all_to_click_0.pqt') )
for k in range(1,DISK_PIECES): 
    top_20_all_to_click.update(pqt_to_dict( pd.read_parquet(f'/kaggle/working/covisit_all_to_click_{k}.pqt') ) )

top_20_click_to_buy = pqt_to_dict( pd.read_parquet(f'/kaggle/working/covisit_click_to_buy_0.pqt') )
for k in range(1,DISK_PIECES): 
    top_20_click_to_buy.update(pqt_to_dict( pd.read_parquet(f'/kaggle/working/covisit_click_to_buy_{k}.pqt') ) )





# TOP CLICKS AND ORDERS IN TEST
top_clicks = test_df.loc[test_df['type']=='0','aid'].value_counts().index.values[:20]
top_carts  = test_df.loc[test_df['type']=='1','aid'].value_counts().index.values[:20]
top_orders = test_df.loc[test_df['type']=='2','aid'].value_counts().index.values[:20]

print('Here are size of our 3 co-visitation matrices:')
print( len( top_20_all_to_cart_order ), len(top_20_all_to_click), len(top_20_click_to_buy)  )


type_weight_multipliers = {0: 1, 1: 3, 2: 6}
def suggest_clicks(df):
    aids = df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))  # Reverse for recency
    
    # Long sessions: rerank by recency + type
    if len(unique_aids) >= 20:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        
        return [k for k, v in aids_temp.most_common(20)]
    
    # Short sessions: use co-visitation
    aids2 = list(itertools.chain(*[top_20_clicks[aid] for aid in unique_aids if aid in top_20_clicks]))
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids]
    
    result = unique_aids + top_aids2[:20 - len(unique_aids)]
    return result + list(top_clicks)[:20 - len(result)]



# %%time
# pred_df_clicks = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
#     lambda x: suggest_clicks(x)
# )


# clicks_pred_df = pd.DataFrame(pred_df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()

# orders_pred_df = pd.DataFrame(pred_df_clicks.add_suffix("_orders"), columns=["labels"]).reset_index()
# carts_pred_df =  pd.DataFrame(pred_df_clicks.add_suffix("_carts"),  columns=["labels"]).reset_index()

# clicks_pred_df


# pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
# # pred_df=clicks_pred_df
# pred_df.columns = ["session_type", "labels"]
# pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str,x)))
# pred_df.to_csv("submission.csv", index=False)
# pred_df.head()

