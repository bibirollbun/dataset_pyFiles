DIR = '/kaggle/input/pump-fun-graduation-february-2025'


!ls {DIR}


import pandas as pd
import os
import catboost


train = pd.read_csv(os.path.join(DIR, 'train.csv'))

train.shape


train.columns


filenames = !ls {DIR}/chunk*.csv
filenames


from tqdm.auto import tqdm
def generate_features(filenames):
    all_data = []
    for chunk_filename in tqdm(filenames):
        all_data.append(
            pd.read_csv(chunk_filename)
        )
    data = pd.concat(all_data)
    data.info()
    features = data.groupby('base_coin').agg({
        'quote_coin_amount': 'sum', # Total trade volume in SOL
    })
    return features

features = generate_features(filenames)


feature_names = features.columns


features


Xy = train[['mint', 'has_graduated']].merge(features, left_on='mint', right_on='base_coin', how='left')


model = catboost.CatBoostClassifier()


model.fit(Xy[feature_names], Xy['has_graduated'], metric_period=100)


test = pd.read_csv(os.path.join(DIR, 'test_unlabeled.csv'))


X_test = test[['mint']].merge(features, left_on='mint', right_on='base_coin', how='left')


p = model.predict_proba(X_test[feature_names])[:, 1]


submission = X_test[['mint']]


submission['has_graduated'] = p


submission


assert submission.shape[0] == test.shape[0]


submission.to_csv('submission.csv', index=False)




