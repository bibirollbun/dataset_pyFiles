DIR = '/kaggle/input/pump-fun-graduation-february-2025'


!ls {DIR}


import pandas as pd
import os
import catboost


train = pd.read_csv(os.path.join(DIR, 'train.csv'))

train.shape


train.columns


test = pd.read_csv(os.path.join(DIR, 'test_unlabeled.csv'))

test.shape


token_info = pd.read_csv(os.path.join(DIR, 'dune_token_info_v2.csv'))


onchain_divers_token_info =  pd.read_csv(os.path.join(DIR, 'token_info_onchain_divers_v2.csv'))


mints = pd.concat([train, test]).mint.unique()


creation_info = pd.concat([
    onchain_divers_token_info.assign(onchain_divers=1).drop_duplicates(subset='mint'),
    token_info.rename(columns={'token_mint_address': 'mint', 'created_at': 'block_time'}).assign(onchain_divers=0).drop_duplicates(subset='mint'),
])[['mint', 'block_time', 'onchain_divers']].sort_values(['mint', 'onchain_divers'], ascending=True)


creation_info = creation_info[creation_info.mint.isin(mints)]


creation_info = creation_info.drop_duplicates(subset=['mint']) # .onchain_divers.value_counts()


all_mints = pd.Series(mints)
all_mints[~all_mints.isin(creation_info.mint)].sample(50)


# all mints above created too early, we exclude them from train/test


filenames = !ls {DIR}/chunk*.csv
filenames



from tqdm.auto import tqdm

def mint_first_swap(filenames):
    all_data = []
    for chunk_filename in tqdm(filenames):
        all_data.append(
            pd.read_csv(chunk_filename)
        )
    data = pd.concat(all_data)
    data.info()
    features = data.groupby('base_coin').agg({
        'block_time': 'min',
        'slot': 'min',
    })
    return features

first_swap = mint_first_swap(filenames)



first_swap.sample(10)


mint_time_with_first_swap = creation_info.merge(first_swap, left_on='mint', right_index=True, how='outer', suffixes=['_mint', '_swap'])


mint_time_with_first_swap.query('block_time_mint == block_time_swap')


mint_time_with_first_swap['block_time_mint'] = pd.to_datetime(mint_time_with_first_swap['block_time_mint'], utc=True)


mint_time_with_first_swap['block_time_swap'] = pd.to_datetime(mint_time_with_first_swap['block_time_swap'], utc=True)


from datetime import timedelta
(mint_time_with_first_swap['block_time_mint'] - mint_time_with_first_swap['block_time_swap']) == timedelta(hours=8)


mint_time_with_first_swap['is_valid'] = (
    # Mints are selected starting from February, 1st
    (~mint_time_with_first_swap['block_time_mint'].isna()) & 
    # Minting and first swap should be in the same transaction, unfortunately timezones are different in the tables
    ((mint_time_with_first_swap['block_time_mint'] - mint_time_with_first_swap['block_time_swap']) == timedelta(hours=8))
)


mint_time_with_first_swap['is_valid'].value_counts()


mint_time_with_first_swap.sample(10)


train = train.merge(mint_time_with_first_swap[['mint', 'is_valid']], on='mint', how='left')


test = test.merge(mint_time_with_first_swap[['mint', 'is_valid']], on='mint', how='left')


train[train['is_valid']].to_csv('train_valid_only.csv')


test[test['is_valid']].to_csv('test_valid_only.csv')




