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


mints = pd.concat([train, test]).mint.unique()


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



set(test['mint']) - set(first_swap.index)


pd.concat([
    train.assign(is_train=1)[['mint', 'is_train']],
    test.assign(is_train=0)[['mint', 'is_train']],
]).merge(first_swap.assign(has_swaps=1)[['has_swaps']], left_on='mint', right_index=True, how='outer')[['is_train', 'has_swaps']].fillna(-1).value_counts()


first_swap.shape


1


mint = '5heBo3Q7s5VeAnyn6nTZTPVuhsB2oAK8Sfw1eM3zpump'


train[train.mint == mint]


test[test.mint == mint]


len(set(test['mint']) - set(first_swap.index))


first_swap[first_swap.index == 'ELEbUbiG7yPrs7zpXAXNn52zXq42eA5GkUt1pVZnpump']


from tqdm.auto import tqdm

for chunk_filename in tqdm(filenames[::-1]):
    data = pd.read_csv(chunk_filename)
    sample = data[data['base_coin'] == 'ELEbUbiG7yPrs7zpXAXNn52zXq42eA5GkUt1pVZnpump']
    if sample.shape[0] > 0:
        print(chunk_filename)
        print(sample)
        break




