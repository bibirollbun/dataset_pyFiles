import pandas as pd
import glob
import os
from tqdm.auto import tqdm
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


DIR = '/kaggle/input/pump-fun-graduation-february-2025'

train = pd.read_csv(os.path.join(DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DIR, 'test_unlabeled.csv'))

chunk_files = glob.glob(os.path.join(DIR, 'chunk*.csv'))


def generate_features(filenames):
    all_data = []
    for chunk_filename in tqdm(filenames, desc="Loading chunks"):
        chunk = pd.read_csv(chunk_filename)
        all_data.append(chunk)
    data = pd.concat(all_data)
    print("Shape of concatenated data:", data.shape)
    
    features = data.groupby('base_coin').agg({
        'quote_coin_amount': ['sum', 'mean', 'max', 'count'],
        'base_coin_amount': ['sum', 'mean', 'max'],
        'signing_wallet': 'nunique',  # количество уникальных кошельков
        'direction': lambda x: (x == 'buy').sum() / len(x)  # доля покупок
    })
    
    features.columns = ['_'.join(col) for col in features.columns]
    features.reset_index(inplace=True)
    
    return features

features = generate_features(chunk_files)


Xy = train[['mint', 'has_graduated']].merge(features, left_on='mint', right_on='base_coin', how='left')
X_test = test[['mint']].merge(features, left_on='mint', right_on='base_coin', how='left')

Xy.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

feature_names = [col for col in Xy.columns if col not in ['mint', 'has_graduated', 'base_coin']]


X_train, X_val, y_train, y_val = train_test_split(Xy[feature_names], Xy['has_graduated'], 
                                                  test_size=0.2, random_state=42, stratify=Xy['has_graduated'])


model = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='Logloss',
    random_seed=42,
    verbose=100
)

train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)
model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)


val_preds = model.predict_proba(X_val)[:, 1]
print("Validation LogLoss:", log_loss(y_val, val_preds))

test_preds = model.predict_proba(X_test[feature_names])[:, 1]


submission = pd.DataFrame({
    'mint': X_test['mint'],
    'has_graduated': test_preds
})

assert submission.shape[0] == test.shape[0]

submission.to_csv('submission.csv', index=False)




