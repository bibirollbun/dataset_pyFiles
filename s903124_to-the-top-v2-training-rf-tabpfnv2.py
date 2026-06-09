!pip install tabpfn


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import KFold
import numpy as np


from kaggle_secrets import UserSecretsClient
import os
user_secrets = UserSecretsClient()
os.environ["TABPFN_TOKEN"] = user_secrets.get_secret("HF_TOKEN")


X_train = pd.read_csv('../input/to-the-top-v2/X_train.csv',index_col=[0]).reset_index(drop=True)
X_test = pd.read_csv('../input/to-the-top-v2/X_test.csv',index_col=[0]).reset_index(drop=True)
y_train = pd.read_csv('../input/to-the-top-v2/y_train.csv',index_col=[0])


def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc = np.array(auc, dtype='f')
    auc /= (nfalse * (n - nfalse))
    return auc




def process_in_chunks(X_train, y_train, chunk_size=100000):
    fold_score = []
    kf = KFold(n_splits=10)
    
    # Calculate number of chunks
    n_samples = len(X_train)
    n_chunks = int(np.ceil(n_samples / chunk_size))
    
    for chunk in range(n_chunks):
        start_idx = chunk * chunk_size
        end_idx = min((chunk + 1) * chunk_size, n_samples)
        
        X_chunk = X_train.iloc[start_idx:end_idx]
        y_chunk = y_train.iloc[start_idx:end_idx]
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_chunk)):
            # print(fold)
            clf = TabPFNClassifier(model_path="/kaggle/input/tabpfn-weight/tabpfn-v3-classifier-v3_20260417_binary.ckpt")
            
            # Train on chunk's training data
            clf.fit(X_chunk.iloc[train_idx], y_chunk.iloc[train_idx].values.ravel())
            
            # Evaluate on chunk's test data
            fold_score.append(
                fast_auc(
                    y_chunk.iloc[test_idx], 
                    clf.predict_proba(X_chunk.iloc[test_idx])[:,1]
                )
            )
            
            # Save model for this fold and chunk
            model_name = f'model_fold_{fold}.joblib'
            joblib.dump(clf, model_name)
    
    return fold_score

# Use the function
fold_scores = process_in_chunks(X_train, y_train, chunk_size=100000)


# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# secret_value_0 = user_secrets.get_secret("HF_TOKEN")


# fold_score = []
# kf = KFold(n_splits=10)
# chunk_size = 10000
# # Calculate number of chunks
# n_samples = len(X_train)
# n_chunks = int(np.ceil(n_samples / chunk_size))

# for chunk in range(n_chunks):
#     start_idx = chunk * chunk_size
#     end_idx = min((chunk + 1) * chunk_size, n_samples)
    
#     X_chunk = X_train.iloc[start_idx:end_idx]
#     y_chunk = y_train.iloc[start_idx:end_idx]
#     print(chunk)
    
#     for fold, (train_idx, test_idx) in enumerate(kf.split(X_chunk)):
#         print(fold)
#         clf = TabPFNClassifier()
        
#         # Train on chunk's training data
#         clf.fit(X_chunk.iloc[train_idx], y_chunk.iloc[train_idx].values.ravel())
        
#         # Evaluate on chunk's test data
#         fold_score.append(
#             fast_auc(
#                 y_chunk.iloc[test_idx], 
#                 clf.predict_proba(X_chunk.iloc[test_idx])[:,1]
#             )
#         )


print('Cross Validation score = %1.6f' % np.mean(fold_scores))


result = []

for fold in range(10):
    clf = joblib.load('model_fold_' +str(fold)+ '.joblib')
    result.append(clf.predict_proba(X_test))


submission_df = pd.read_csv('../input/santander-customer-satisfaction/sample_submission.csv')


submission_df['TARGET'] = 1-np.mean(result,axis=0)


submission_df.to_csv('submission.csv', index=False)

