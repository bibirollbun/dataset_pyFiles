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


token_info = pd.read_csv(os.path.join(DIR, 'dune_token_info.csv'))


onchain_divers_token_info =  pd.read_csv(os.path.join(DIR, 'token_info_onchain_divers.csv'))


token_info_mints = set(token_info['token_mint_address'])
train_mints = set(train['mint'])
test_mints = set(test['mint'])
onchain_mints = set(onchain_divers_token_info['mint'])


len(token_info_mints), len(train_mints), len(test_mints), len(onchain_mints)


len(token_info_mints.intersection(train_mints)) / len(train_mints), len(token_info_mints.intersection(test_mints)) / len(test_mints)


len(onchain_mints.intersection(train_mints)) / len(train_mints), len(onchain_mints.intersection(test_mints)) / len(test_mints)


all_mints_info = onchain_mints.union(token_info_mints)


len(all_mints_info.intersection(train_mints)) / len(train_mints), len(all_mints_info.intersection(test_mints)) / len(test_mints)


token_info = pd.read_csv(os.path.join(DIR, 'dune_token_info_v2.csv'))


onchain_divers_token_info =  pd.read_csv(os.path.join(DIR, 'token_info_onchain_divers_v2.csv'))


token_info_mints = set(token_info['token_mint_address'])
train_mints = set(train['mint'])
test_mints = set(test['mint'])
onchain_mints = set(onchain_divers_token_info['mint'])


len(token_info_mints.intersection(train_mints)) / len(train_mints), len(token_info_mints.intersection(test_mints)) / len(test_mints)


len(onchain_mints.intersection(train_mints)) / len(train_mints), len(onchain_mints.intersection(test_mints)) / len(test_mints)


all_mints_info = onchain_mints.union(token_info_mints)


len(all_mints_info.intersection(train_mints)) / len(train_mints), len(all_mints_info.intersection(test_mints)) / len(test_mints)




