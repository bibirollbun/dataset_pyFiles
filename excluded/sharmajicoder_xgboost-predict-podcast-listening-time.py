# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline 
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df


df.info()


df.drop_duplicates()


df.isnull().sum()


print(f'Total Size: {df["Number_of_Ads"].size}')
print(f'Number of NULL values: {df["Number_of_Ads"].isnull().sum()}')
df[df["Number_of_Ads"].isnull()]


print(df["Number_of_Ads"].value_counts())
print("-"*50)
print(df["Number_of_Ads"].value_counts(normalize = True) * 100)


df["Number_of_Ads"].fillna(df["Number_of_Ads"].mode()[0], inplace = True)


df["Number_of_Ads"].isnull().sum()


print(df["Episode_Length_minutes"].size)
print(df["Episode_Length_minutes"].isnull().sum())
df[df["Episode_Length_minutes"].isnull()]


print(df["Episode_Length_minutes"].value_counts())
print("-"*60)
print(df["Episode_Length_minutes"].value_counts(normalize = True) * 100)


#filling the values with Frequency-Weighted Random Sampling to keep data more naturally balanced.
#This randomly assigns missing values based on how often each value appears.

value_counts = df["Episode_Length_minutes"].value_counts(normalize = True) # gives relative frequency that sum upto 1
nan_indices = df["Episode_Length_minutes"].isnull() # This creates a Boolean Series indaicating which rows have NaN in the column

df.loc[nan_indices, "Episode_Length_minutes"] = np.random.choice(value_counts.index,size = nan_indices.sum(),p = value_counts.values)
#value_counts.index = list of values to choose
#size = nan_indices.sum() = the number of values to generate
#p = value_counts.values = the probabilities for each value (sum to 1)


df["Episode_Length_minutes"].isnull().sum()


print(df["Guest_Popularity_percentage"].size)
print(df["Guest_Popularity_percentage"].isnull().sum())

df[df["Guest_Popularity_percentage"].isnull()]


print(df["Guest_Popularity_percentage"].value_counts())
print("-" * 60)
print(df["Guest_Popularity_percentage"].value_counts(normalize = True) * 100)


value_counts = df["Guest_Popularity_percentage"].value_counts(normalize = True)
nan_indices = df["Guest_Popularity_percentage"].isnull()
df.loc[nan_indices, "Guest_Popularity_percentage"] = np.random.choice(value_counts.index,size = nan_indices.sum(),p = value_counts.values)


df["Guest_Popularity_percentage"].isnull().sum()


df.isnull().sum()


df.describe()


scatter = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
axes = axes.flatten()

for ax, i in zip(axes, scatter):
    sns.scatterplot(data = df,x = i, y = 'Listening_Time_minutes',ax = ax)
    ax.set_title(f'Scatterplot describe correlation between {i} and Listening_Time_minutes')
    ax.set_xlabel(i)
    ax.set_ylabel('Listening_Time_minutes')
    ax.set_xlim(0, df[i].max() * 1.1)
    for i in ax.get_xticklabels():
        i.set_rotation(45)
plt.tight_layout()
plt.show()


def popular(df):
    avg_popularity = 0
    if pd.isna(df['Guest_Popularity_percentage']):
        avg_popularity = df['Host_Popularity_percentage']
    else:
        avg_popularity = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']) / 2
    if avg_popularity <= 20:
        return 'Not Very Popular'
    elif 20 < avg_popularity <= 40:
        return 'Not Popular'
    elif 40 < avg_popularity <= 60:
        return 'Average'
    elif 60 < avg_popularity <= 85:
        return 'Popular'
    else:
        return 'Very Popular'

df['Popular_Level'] = df.apply(popular, axis=1)
df.drop(columns=['Host_Popularity_percentage', 'Guest_Popularity_percentage'], inplace=True)
df_test['Popular_Level'] = df_test.apply(popular, axis=1)
df_test.drop(columns=['Host_Popularity_percentage', 'Guest_Popularity_percentage'], inplace=True)


df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '').astype(int)
df_test['Episode_Title'] = df_test['Episode_Title'].str.replace('Episode ', '').astype(int)


df['Podcast_Name'].value_counts()


def Podcast_Name(df):
    if df['Podcast_Name'] in ['Tech Talks','Tech Trends', 'Gadget Geek', 'Digital Digest', 'Innovators']:
        return 'Tech'
    elif df['Podcast_Name'] in ['Game Day', 'Sports Weekly', "Athlete's Arena", 'Sports Central', 'Sport Spot']:
        return 'Sports'
    elif df['Podcast_Name'] in ['Business Insights', 'Business Briefs', 'Finance Focus', 'Money Matters', 'Market Masters']:
        return 'Business'
    elif df['Podcast_Name'] in ['Global News', 'World Watch', 'Current Affairs', 'Daily Digest', 'News Roundup']:
        return 'News'
    elif df['Podcast_Name'] in ['Funny Folks', 'Humor Hub', 'Comedy Corner', 'Joke Junction', 'Laugh Line']:
        return 'Comedy'
    elif df['Podcast_Name'] in ['Melody Mix', 'Tune Time', 'Sound Waves', 'Music Matters']:
        return 'Music'
    elif df['Podcast_Name'] in ['Style Guide', 'Fashion Forward', 'Lifestyle Lounge', 'Home & Living', 'Life Lessons']:
        return 'Lifestyle'
    elif df['Podcast_Name'] in ['Study Sessions', 'Learning Lab', 'Educational Nuggets', 'Brain Boost']:
        return 'Education'
    elif df['Podcast_Name'] in ['Detective Diaries', 'Crime Chronicles', 'True Crime Stories', 'Criminal Minds', 'Mystery Matters']:
        return 'True Crime'
    elif df['Podcast_Name'] in ['Fitness First', 'Wellness Wave', 'Mind & Body', 'Healthy Living', 'Health Hour']:
        return 'Health'
    else:
        return 'Other'


df['Podcast_Name'] = df.apply(Podcast_Name, axis=1)
df_test['Podcast_Name'] = df_test.apply(Podcast_Name, axis=1)
df['Podcast_Name'].value_counts()


def genre(df):
    if df['Genre'] in ['Comedy', 'Music', 'Lifestyle']:
        return 'Entertainment'
    elif df['Genre'] in ['News', 'True Crime', 'Business']:
        return 'Events News'
    elif df['Genre'] in ['Health', 'Education', 'Technology']:
        return 'Information'
    elif df['Genre'] in ['Sports']:
        return 'Sports'
    else:
        return 'Other'
df['Genre'] = df.apply(genre, axis=1)
df_test['Genre'] = df_test.apply(genre, axis=1)


df['ratio_Ads'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
df_test['ratio_Ads'] = df_test['Number_of_Ads'] / df_test['Episode_Length_minutes']


objects = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Popular_Level', 'Podcast_Name']
for column_name in objects:
    df[column_name].value_counts().plot(kind = "bar")
    plt.xlabel(f"{column_name}",fontsize = 16)
    plt.ylabel("Frequency", fontsize = 16)
    plt.title(f"Bar graph of {column_name}", fontsize = 20)
    for i, bar in enumerate(plt.gca().patches):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)),
             ha='center', va='bottom')
    plt.show()


fig, axes = plt.subplots(6, 2, figsize=(20, 30)) 
axes = axes.flatten()

for idx, col in enumerate(objects):
    sns.countplot(data=df, x=col, ax=axes[2*idx])
    axes[2*idx].set_title(f'Countplot for {col}')
    axes[2*idx].set_xlabel(col)
    for label in axes[2*idx].get_xticklabels():
        label.set_rotation(45)
    
    sns.boxplot(data=df, x=col, y='Listening_Time_minutes', ax=axes[2*idx+1])
    axes[2*idx+1].set_title(f'Boxplot for {col} vs Listening_Time')
    axes[2*idx+1].set_xlabel(col)
    for label in axes[2*idx+1].get_xticklabels():
        label.set_rotation(45)

plt.tight_layout()
plt.show()


df_test


df_test.drop_duplicates()


df_test.info()


df_test.isnull().sum()


print(df_test["Episode_Length_minutes"].size)
print(df_test["Episode_Length_minutes"].isnull().sum())
df_test[df_test["Episode_Length_minutes"].isnull()]


print(df_test["Episode_Length_minutes"].value_counts())
print("-" * 60)
print(df_test["Episode_Length_minutes"].value_counts(normalize = True) * 100)


value_counts = df_test["Episode_Length_minutes"].value_counts(normalize = True)
nan_indices = df_test["Episode_Length_minutes"].isnull()
df_test.loc[nan_indices, "Episode_Length_minutes"] = np.random.choice(value_counts.index,size = nan_indices.sum(),p = value_counts.values)


df_test["Episode_Length_minutes"].isnull().sum()


print(df_test["ratio_Ads"].size)
print(df_test["ratio_Ads"].isnull().sum())
df_test[df_test["ratio_Ads"].isnull()]


print(df_test["ratio_Ads"].value_counts())
print("-" * 60)
print(df_test["ratio_Ads"].value_counts(normalize = True) * 100)


value_counts = df_test["ratio_Ads"].value_counts(normalize = True)
nan_indices = df_test["ratio_Ads"].isnull()
df_test.loc[nan_indices, "ratio_Ads"] = np.random.choice(value_counts.index,size = nan_indices.sum(),p = value_counts.values)


df_test["ratio_Ads"].isnull().sum()


df_test.info()


df_test.describe()


mood = ['Negative', 'Neutral', 'Positive']
popularity = ['Not Very Popular','Not Popular','Average', 'Popular','Very Popular']
ord = OrdinalEncoder(categories=[mood, popularity])

columns_ordinal_encode = ['Episode_Sentiment', 'Popular_Level']
columns_label_encode = ['Genre', 'Publication_Day', 'Publication_Time', 'Podcast_Name']
num_cols = ['Episode_Length_minutes', 'Number_of_Ads', 'ratio_Ads', 'Episode_Title']


preprocessing = ColumnTransformer([
    ('Ordinal', ord, columns_ordinal_encode),
    ('LabelEncoder', OrdinalEncoder(), columns_label_encode),
    ('StandardScale', StandardScaler(),num_cols)
])
input = [
    ('preprocess', preprocessing),
    ('model', Ridge())]
pipe = Pipeline(input)


X = df.drop(columns=['id', 'Listening_Time_minutes'])
Y = df['Listening_Time_minutes']


param_grid = [
    {
        'model': [xgb.XGBRegressor()],
        'model__n_estimators': [100],
        'model__learning_rate': [0.1],
        'model__max_depth': [12],
        'model__subsample': [0.9],
        'model__gamma': [0.4],
        'model__min_child_weight': [1],
        'model__n_jobs': [-1],
        'model__reg_lambda': [1],
        'model__reg_alpha': [0.5]
    }
]


k_cv = KFold(n_splits= 10, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipe, param_grid, cv=k_cv, scoring= 'r2')
grid_search.fit(X, Y)

print('Best score of the model is: ')
print(f"Best score: {grid_search.best_score_}")
print(f"Which will use the model: \n {grid_search.best_estimator_}")


Y_pred = grid_search.best_estimator_.predict(X)
print(mean_squared_error(Y , Y_pred))


print(f'Root_mean_square_error is: {np.sqrt(mean_squared_error(Y , Y_pred))}')


df_test['predict'] = grid_search.best_estimator_.predict(df_test.drop(columns = ['id']))
df_submission = pd.DataFrame({
    'id': df_test['id'], 
    'Listening_Time_minutes' : df_test['predict']
})


df_submission.to_csv('submission.csv', index = False)
df_submission.info()


df_submission.to_csv("/kaggle/working/submission.csv", index = False)

