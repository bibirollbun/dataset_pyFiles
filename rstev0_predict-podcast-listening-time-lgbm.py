import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col = 'id')#.sample(frac = 0.2, random_state = 1111).reset_index(drop = True)
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col = 'id')
sample = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


import warnings

msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category = RuntimeWarning, message = msg)
print(f"{train[train.duplicated(keep = False)]}\n")
train.head()


train.info()


import warnings

# look at the numeric columns
def num_plotter(data, target = None):
    for col in data.select_dtypes(["int", "float"]):
        if col != target:
            plt.figure(figsize = (6,1))
            sns.boxplot(data = data, x = col, y = target)
            plt.show();

with warnings.catch_warnings(): # Disabling FutureWarning for the plots
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    num_plotter(train)


# look at the non-numeric columns
def cat_bar_plotter(df, normalize = False):
    for col in df.select_dtypes("object").columns:
        plt.figure(figsize = (6,3))
        df[col].value_counts(normalize = normalize, dropna = False).plot.bar()
        plt.show();

cat_bar_plotter(train)


def clean_df(df):
    df['Episode_Num'] = df['Episode_Title'].str[8:]
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(
        value = df['Number_of_Ads'].median()
    )
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(
        value = df['Episode_Length_minutes'].median()
    )
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(
        value = df['Guest_Popularity_percentage'].median()
    )
    df['Ads_Per_Hour'] = np.where(df['Episode_Length_minutes'] > 0, (60 * df['Number_of_Ads'] / df['Episode_Length_minutes']), 0)
    return df.drop(columns = ['Episode_Title'])

train = clean_df(train)
test = clean_df(test)


from itertools import combinations

cat_features = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Episode_Num']
encode_columns = ['Episode_Length_minutes', 'Genre', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Episode_Num']
encoded_columns = []
# pair_size = [2, 3, 4]
pair_size = [2]

for r in pair_size:
    for col in list(combinations(encode_columns, r)):
        new_col_name = '_'.join(col)
        encoded_columns.append(new_col_name)
        train[new_col_name] = train[list(col)].astype(str).agg('_'.join, axis=1)
        # train[new_col_name] = train[new_col_name].astype('category')
        
        test[new_col_name] = test[list(col)].astype(str).agg('_'.join, axis=1)
        # test[new_col_name] = test[new_col_name].astype('category')

for col in cat_features:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
for col in encoded_columns:
    agg = train.groupby(col)['Listening_Time_minutes'].agg('mean')
    train[col] = train[col].map(agg).astype(float)
    test[col] = test[col].map(agg).astype(float)


train.head()


X = train.drop(columns = ['Listening_Time_minutes'])
y = train['Listening_Time_minutes']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .2, random_state = 1111)


lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = cat_features)
lgb_valid = lgb.Dataset(X_val, y_val, categorical_feature = cat_features)
params = {
    'n_estimators': 10_000,
    'learning_rate': 0.03,
    'objective': 'l2',
    'metric': 'rmse',
    'max_depth': -1,
    'num_leaves': 2048,
    'colsample_bytree': 0.7,
    'max_bin': 1024,
    'verbosity': -1,
}
model = lgb.train(
    params = params,
    train_set = lgb_train,
    valid_sets = [lgb_train, lgb_valid],
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
)


y_pred = model.predict(X_val)
print(f"RMSE: {mse(y_val, y_pred, squared = False):.4f}")


# 13.1544 for LabelEncoder
# 12.7360 with test hyperparameters
# 12.7328 with adding Ads_Per_Hour
# 12.6923 if filling nan with global median instead of per Podcast_Name

# 12.6301 with different parameters
# 12.6182 after removing Ads_Per_Hour (?)
# 12.5943 learning_rate 0.02
# 12.5652 correcting LabelEncoder

# 13.6394 LabelEncoder() + LGBMRegressor() to Dataset()


importance_types = ['split', 'gain']

for itype in importance_types:
    score = model.feature_importance(importance_type = itype)
    score_series = pd.Series(score, index = model.feature_name()).sort_values(ascending = False)

    plt.figure(figsize = (10, 6))
    score_series.head(30).plot(kind = 'bar')
    plt.title(f"Feature Importance - {itype}")
    plt.ylabel(itype)
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()


sample['Listening_Time_minutes'] = model.predict(test)
sample.to_csv('submission.csv',index = False)




