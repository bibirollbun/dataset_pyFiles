import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


df = pd.read_csv('../input/playground-series-s5e2/train.csv', index_col = 'id')
df_test = pd.read_csv('../input/playground-series-s5e2/test.csv', index_col = 'id')
submission = pd.read_csv('../input/playground-series-s5e2/sample_submission.csv')


df.head()


df.shape, df_test.shape


print('Null Values in train set: ', df.isnull().sum().sum())
print('Null Values in test set: ', df_test.isnull().sum().sum())


df['Price'].isnull().sum()


#obj_cols = [c for c in df.columns if df[c].dtype == 'object']
cat_cols = [c for c in df.columns if df[c].nunique() < 10]


num_cols = [c for c in df.columns if c not in cat_cols]
num_cols


num_cols.remove('Price')
num_cols


sets = []
for col in cat_cols:
    num = df[col].nunique()
    sets.append(num)

sets_categorical = dict(zip(cat_cols, sets))
print(sets_categorical)  


cat_imputer = SimpleImputer(strategy = 'most_frequent')
num_imputer = SimpleImputer(strategy = 'mean')
num_imputed_df = pd.DataFrame(num_imputer.fit_transform(df[num_cols]), columns = num_cols)
num_imputed_df_test = pd.DataFrame(num_imputer.transform(df_test[num_cols]), columns = num_cols)

num_imputed_df.index = df.index
num_imputed_df_test.index = df_test.index


cat_imputed_df = pd.DataFrame(cat_imputer.fit_transform(df[cat_cols]), columns = cat_cols)
cat_imputed_df_test = pd.DataFrame(cat_imputer.transform(df_test[cat_cols]), columns = cat_cols)


print('Null Values: ', num_imputed_df.isnull().sum().sum())
print('Null Values in test set: ',num_imputed_df_test.isnull().sum().sum())


encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_df = pd.DataFrame(encoder.fit_transform(cat_imputed_df[cat_cols]), columns=encoder.get_feature_names_out())
encoded_df_test = pd.DataFrame(encoder.transform(cat_imputed_df_test[cat_cols]), columns=encoder.get_feature_names_out())

encoded_df.index = df.index
encoded_df_test.index = df_test.index


print('Null Values: ', encoded_df.isnull().sum().sum())
print('Null Values in test set: ',encoded_df_test.isnull().sum().sum())


X = pd.concat([num_imputed_df, encoded_df], axis=1)
X_test = pd.concat([num_imputed_df_test, encoded_df_test], axis=1)
y = df['Price'].values


X.info(memory_usage='deep')


df.shape, X.shape, df_test.shape, X_test.shape


y_preds = np.zeros(len(X_test))
scores = []

n_splits = 5
folds = KFold(n_splits = n_splits, shuffle = True)
for fold, (trn_id, val_id) in enumerate(folds.split(X,y)):
    X_train, X_valid = X.iloc[trn_id], X.iloc[val_id]
    y_train, y_valid = y[trn_id], y[val_id]

    xgb_model = XGBRegressor(n_estimators=200, 
                             max_depth=4, learning_rate=0.01,
                             subsample=0.8, colsample_bytree=0.8,
                             objective='reg:squarederror',
                             random_state=42)

    xgb_model.fit(X_train, y_train)
    preds = xgb_model.predict(X_valid)

    score = mean_squared_error(y_valid, preds, squared = False)

    print(f'Fold #: {fold}, Score: {score}')
    scores.append(score)

    y_preds += xgb_model.predict(X_test) / n_splits

print(f'Mean Score: {np.mean(scores)}')


submission.head()


submission.shape, y_preds.shape


submission['Price'] = y_preds
submission.to_csv('submission.csv', index = False)

