import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool


PATH_DATASET = '/kaggle/input/equity-post-HCT-survival-predictions/'


data_dictionary_df = pd.read_csv(PATH_DATASET + 'data_dictionary.csv')
train_df = pd.read_csv(PATH_DATASET + 'train.csv')
submit_df = pd.read_csv(PATH_DATASET + 'test.csv')

data_dictionary_df.shape, train_df.shape, submit_df.shape


# Primary identification of attributes
target_column = 'efs'
target_columns = ['efs', 'efs_time']
numerical_columns = list(data_dictionary_df[(~data_dictionary_df['variable'].isin(target_columns))&(data_dictionary_df['type'] == 'Numerical')]['variable'].values)
categorical_columns = list(data_dictionary_df[(~data_dictionary_df['variable'].isin(target_columns))&(data_dictionary_df['type'] == 'Categorical')]['variable'].values)
feature_columns = numerical_columns + categorical_columns

len(target_columns), len(numerical_columns), len(categorical_columns), len(feature_columns)


# Fill Na for Categorical
train_df[categorical_columns] = train_df[categorical_columns].fillna('null')
submit_df[categorical_columns] = submit_df[categorical_columns].fillna('null')


# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(train_df[feature_columns], train_df[target_column], test_size = 0.05, random_state = 53)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5, random_state = 53)

(X_train.shape, y_train.shape, 
X_val.shape, y_val.shape, 
X_test.shape, y_test.shape)


%%time
model_clf = CatBoostClassifier(
                eval_metric="AUC", 
                early_stopping_rounds=200, 
                iterations=2000,
                random_state=53, 
                cat_features=categorical_columns, 
                learning_rate=0.01,
                task_type='GPU'
)
val_pool = Pool(X_val, y_val, cat_features=categorical_columns)
model_clf.fit(X_train, y_train, eval_set=val_pool, plot=True, verbose=False)


submit_df['prediction'] = model_clf.predict_proba(submit_df[feature_columns])[:,1]
submit_df[['ID', 'prediction']].to_csv('submission.csv', index=False)




