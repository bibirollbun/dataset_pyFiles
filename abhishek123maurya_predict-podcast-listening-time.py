import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from tqdm import tqdm, trange

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df.head()


df.drop('id', axis=1, inplace=True)
df.shape


df.info()


df.describe()


df.isnull().sum()


df.head()


df['ELM_IsNull'] = df['Episode_Length_minutes'].isnull().astype('int')
df['GPP_IsNull'] = df['Guest_Popularity_percentage'].isnull().astype('int')
df.head()


temp = df[(df['Publication_Time']=='Afternoon') & (df['Publication_Day']=='Saturday')\
    & (df['Podcast_Name']=='Funny Folks') & (df['Episode_Title']=='Episode 93')]


temp


(temp['Episode_Length_minutes'].isnull()).index


# for idx in tqdm(df[df['Episode_Length_minutes'].isnull()].index):
#     if not np.isnan(df.loc[idx, 'Episode_Length_minutes']): continue
#     row = df.iloc[idx]

#     temp = df[(df['Podcast_Name']==row.Podcast_Name)]
#     temp = temp[temp['Episode_Title']==row.Episode_Title]
#     temp = temp[temp['Publication_Time']==row.Publication_Time]
#     temp = temp[temp['Publication_Day']==row.Publication_Day]
    
#     t1 = temp['Episode_Length_minutes']    
#     null_idx = t1[t1.isnull()].index
#     df.loc[null_idx, 'Episode_Length_minutes'] = t1.mean()

#     t2 = temp['Guest_Popularity_percentage']
#     null_idx = t2[t2.isnull()].index
#     df.loc[null_idx, 'Guest_Popularity_percentage'] = t2.mean()

# df[['Episode_Length_minutes', 'Guest_Popularity_percentage']].isnull().sum()


def fill_na_cols(df):

    group_cols = ['Podcast_Name', 'Genre', 'Episode_Title', 'Publication_Day', 'Publication_Time']
    while df.isnull().sum().sum()>0:
    
        # Fill Episode_Length_minutes by the group mean
        df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(
            df.groupby(group_cols)['Episode_Length_minutes']
              .transform('mean')
        )
        
        # Fill Guest_Popularity_percentage by the group mean
        df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(
            df.groupby(group_cols)['Guest_Popularity_percentage']
              .transform('mean')
        )

        group_cols.pop()
        if len(group_cols)==0: break

    for col in df.columns:
        if df[col].isnull().sum()>0:
            df[col] = df[col].fillna(df[col].mean())

    return df

df = fill_na_cols(df)


# Its a sanity check to determine in how many rows the null value happened incorrectly. 
# One way to do this is comapring "Episode_Length_minutes" with "Listening_Time_minutes"
# Logically "Episode_Length_minutes" >= "Listening_Time_minutes"
df[df['Episode_Length_minutes']<df["Listening_Time_minutes"]].shape


df['Number_of_Ads'].value_counts()


df[df['Number_of_Ads']>10]


df = df[df['Number_of_Ads']<10]


cat_cols = []
for col in df.columns:
    if df[col].dtype=='object':
        print(f"{col} has {df[col].nunique()} unique features.")
        cat_cols.append(col)


num_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage',
            'Host_Popularity_percentage', 'Number_of_Ads',
            'Listening_Time_minutes']


for col in cat_cols[2:]:
    print(df[col].value_counts())
    print()
print(df['Podcast_Name'].value_counts())


pub_time = df['Publication_Time'].value_counts()
ep_sent = df['Episode_Sentiment'].value_counts()

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].pie(pub_time, labels=pub_time.index,
            autopct='%1.1f%%', startangle=140)
axes[0].set_title('Distribution across Publication Time')

axes[1].pie(ep_sent, labels=ep_sent.index,
            autopct='%1.1f%%', startangle=140)
axes[1].set_title('Distribution across Episode Sentiment')

plt.tight_layout()
plt.show()


plt.subplots(figsize=(10, 5))
for i, col in enumerate(num_cols):  
    plt.subplot(2, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()


temp = df[df['Episode_Length_minutes']>120]
temp.shape


df = df[df['Episode_Length_minutes']<120]
df.head()


def add_one_hot(df):
    one_hot_cols = ['Genre', 'Publication_Time', 'Episode_Sentiment']
    
    for col in one_hot_cols:
        temp = pd.get_dummies(df[col]).astype('int')
        
        df.drop(col, axis=1, inplace=True)
        df = pd.concat([df, temp], axis=1)
    return df
    
df = add_one_hot(df)
df.shape


df.head()


# Ordinal Encoding of Episode_Title 
pod_name = df['Podcast_Name'].unique()
pod_name_dict = {name: i for i, name in enumerate(pod_name)}
df['Podcast_Name'] = df['Podcast_Name'].map(pod_name_dict)

# Ordinal Encoding of Publication_Day
pub_day = df['Publication_Day'].unique()
pub_day_dict = {name: i for i, name in enumerate(pub_day)}
df['Publication_Day'] = df['Publication_Day'].map(pub_day_dict)

# Keep the episode number as it is.
df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '')
df['Episode_Title'] = df['Episode_Title'].astype('int')

df.head()


plt.figure(figsize=(15, 10))
sb.heatmap(df.corr()>0.8, annot=True, cbar=False)
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

target = df['Listening_Time_minutes']
features = df.drop('Listening_Time_minutes', axis=1)

x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.2)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def objective(trial):
    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':    trial.suggest_int('max_depth', 3, 12),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        'subsample':    trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha':    trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda':   trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'gamma':        trial.suggest_uniform('gamma', 0.0, 5.0)
    }
    model = XGBRegressor(**param)
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    preds = model.predict(x_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)


print("Best params:", study.best_params)
print("Best RMSE:", study.best_value)


model = XGBRegressor(**study.best_params)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Training RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")


test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
temp = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

test['ELM_IsNull'] = test['Episode_Length_minutes'].isnull().astype('int')
test['GPP_IsNull'] = test['Guest_Popularity_percentage'].isnull().astype('int')
temp['ELM_IsNull'] = temp['Episode_Length_minutes'].isnull().astype('int')
temp['GPP_IsNull'] = temp['Guest_Popularity_percentage'].isnull().astype('int')

temp.drop(['id', 'Listening_Time_minutes'], inplace=True, axis=1)
test.drop('id', inplace=True, axis=1)

test['label'] = 'test'
temp['label'] = 'train'

dft = pd.concat([temp, test], axis=0)
dft = fill_na_cols(dft)


dft.isnull().sum()


test = dft[dft['label']=='test']
test.drop('label', inplace=True, axis=1)
test.head()


import gc
del dft, temp
gc.collect()


# Add one hot cols
test = add_one_hot(test)

# Ordinally Encoding cols
test['Podcast_Name'] = test['Podcast_Name'].map(pod_name_dict)
test['Publication_Day'] = test['Publication_Day'].map(pub_day_dict)

# Keep the episode number as it is.
test['Episode_Title'] = test['Episode_Title'].str.replace('Episode ', '')
test['Episode_Title'] = test['Episode_Title'].astype('int')

print(test.shape)
test.head()


ss = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
preds = model.predict(scaler.transform(test))
ss['Listening_Time_minutes'] = preds
ss.head()


ss.to_csv('Submission.csv', index=False)




