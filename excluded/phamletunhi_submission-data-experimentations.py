import numpy as np
import pandas as pd


sample = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sample


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")#.drop(["Listening_Time_minutes", "id"], axis = 1)

# Remove samples with listening time > episode length 
train = train[train["Listening_Time_minutes"] < train["Episode_Length_minutes"]]
train


# Remove outliner 
train = train[train['Number_of_Ads'] < 10]
train = train[train['Episode_Length_minutes'] < 300]
train = train[train['Episode_Length_minutes'] != 0]
train = train[train['Host_Popularity_percentage'] <= 100]
train = train[train['Guest_Popularity_percentage'] <= 100]
train.loc[
    train['Episode_Length_minutes'] < train['Listening_Time_minutes'],
    'Episode_Length_minutes'
] = train['Listening_Time_minutes']



# Drop na 
train = train.dropna()
train


# from sklearn.preprocessing import LabelEncoder

# Encoding
train['Episode'] = train['Episode_Title'].str.split(" ").apply(lambda x: x[1]).astype(np.int8)

# # Podcast 
# podcasts = pd.get_dummies(train['Podcast_Name'], prefix='Podcast', prefix_sep='__')
# train = pd.concat([train, podcasts], axis=1)

# Genre
genre = pd.get_dummies(train['Genre'], prefix='Genre', prefix_sep='__')
train = pd.concat([train, genre], axis=1)

# Publication Date
pub_date = pd.get_dummies(train['Publication_Day'], prefix='PubDate', prefix_sep='__')
train = pd.concat([train, pub_date], axis=1)

# # Publication Time
pub_time = pd.get_dummies(train['Publication_Time'], prefix='PubTime', prefix_sep='__')
train = pd.concat([train, pub_time], axis=1)

# One-hot encoding for Episode Sentiment
ep_sent = pd.get_dummies(train['Episode_Sentiment'], prefix='Sentiment', prefix_sep='__')
train = pd.concat([train, ep_sent], axis=1)

train = train.loc[:, ~train.columns.duplicated()]


from sklearn.preprocessing import StandardScaler

columns_to_normalize = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

# normalize values 
scaler = StandardScaler()
train[columns_to_normalize] = scaler.fit_transform(train[columns_to_normalize])


# Feature engineer
def check_weekend(x):
    if x in ['Saturday', 'Sunday']: 
        return True 
    return False

train['Is_Weekend'] = train['Publication_Day'].apply(lambda x: check_weekend(x))
# train['No_Ads'] = train['Number_of_Ads'] == 0
# train['Collab_Support'] = train['Host_Popularity_percentage'] * train['Guest_Popularity_percentage']




train.drop(['Episode_Title', 'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'], axis = 1, inplace=True)


train.drop('id', axis=1, inplace=True)


train.head()


from sklearn.model_selection import train_test_split

# Prepare train and validation dataset
X = train.drop('Listening_Time_minutes', axis=1)
y = train['Listening_Time_minutes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def evaluate_kfold(model, X, y, k=5, task='regression'):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if task == 'regression':
            score = mean_squared_error(y_val, y_pred, squared=False)  # RMSE
        elif task == 'classification':
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_val, y_pred)
        else:
            raise ValueError("Task must be 'regression' or 'classification'")
        
        scores.append(score)

    print(f'{k}-Fold Avg Score: {np.mean(scores):.4f}')
    return scores


# Try some estimating
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=500,         # number of boosting rounds (trees)
    learning_rate=0.05,       # lower learning rate for better generalization
    max_depth=6,              # deeper trees capture more complexity
    subsample=0.8,            # use 80% of rows per tree to reduce overfitting
    colsample_bytree=0.8,     # use 80% of features per tree
    reg_alpha=0.1,            # L1 regularization (sparsity)
    reg_lambda=1,             # L2 regularization (ridge-style)
    random_state=42,          # reproducibility
    verbosity=0               # cleaner output
)


model.fit(X_train, y_train)


np.sqrt(np.mean((y_val - model.predict(X_val))**2))


# evaluate_kfold(model, X, y)


from lightgbm import LGBMRegressor

model2 = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=1024,
        colsample_bytree=0.7,
        max_depth=-1,
        objective='l2',
        metric='rmse'
)


model2.fit(X_train, y_train)


# evaluate_kfold(model2, X, y)


np.sqrt(np.mean((y_val - model2.predict(X_val))**2))


def preprocess_data(df):
    # Encoding Episode
    df['Episode'] = df['Episode_Title'].str.split(" ").apply(lambda x: x[1]).astype(np.int8)

    # # # One-hot encoding for Podcast Name
    # podcasts = pd.get_dummies(df['Podcast_Name'], prefix='Podcast', prefix_sep='__')
    # df = pd.concat([df, podcasts], axis=1)

    # One-hot encoding for Genre
    genre = pd.get_dummies(df['Genre'], prefix='Genre', prefix_sep='__')
    df = pd.concat([df, genre], axis=1)

    # One-hot encoding for Publication Day
    pub_date = pd.get_dummies(df['Publication_Day'], prefix='PubDate', prefix_sep='__')
    df = pd.concat([df, pub_date], axis=1)

    # One-hot encoding for Publication Time
    pub_time = pd.get_dummies(df['Publication_Time'], prefix='PubTime', prefix_sep='__')
    df = pd.concat([df, pub_time], axis=1)

    # One-hot encoding for Episode Sentiment
    ep_sent = pd.get_dummies(df['Episode_Sentiment'], prefix='Sentiment', prefix_sep='__')
    df = pd.concat([df, ep_sent], axis=1)

    # Feature Engineering: Weekend check
    def check_weekend(x):
        if x in ['Saturday', 'Sunday']:
            return True
        return False

    df['Is_Weekend'] = df['Publication_Day'].apply(lambda x: check_weekend(x))
    # df['No_Ads'] = df['Number_of_Ads'] == 0
    # df['Collab_Support'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']

    # Drop duplicate columns (in case of duplicated column names)
    df = df.loc[:, ~df.columns.duplicated()]

    df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
    df.drop(['id', 'Episode_Title', 'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'], axis = 1, inplace=True)

    return df


test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test = preprocess_data(test)
test


set(test.columns) - set(X.columns)
# len(X.columns)


pred = model2.predict(test)
# pred = model.predict(test)



submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission['Listening_Time_minutes'] = pred


submission.to_csv('submission.csv', index=False)


train

