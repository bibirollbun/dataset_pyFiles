import pandas as pd
import numpy as np
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



df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')



print(df.info())
print(df.isnull().sum())


# There are 3 columns with missing values: 'Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads'
missing_percentages = (df.isnull().mean() * 100).sort_values(ascending=False)
missing_values = missing_percentages[missing_percentages > 0]
print(missing_values)
print('Test data missing values')
print('-'*40)
missing_percentages = (df_test.isnull().mean() * 100).sort_values(ascending=False)
missing_values = missing_percentages[missing_percentages > 0]
print(missing_values)


df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(), inplace=True)
df_test['Episode_Length_minutes'].fillna(df_test['Episode_Length_minutes'].median(), inplace= True)

# same result with the test data
df_test['Episode_Length_minutes'].fillna(df_test['Episode_Length_minutes'].median(), inplace= True)


print(df.info())


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
# Except Episode_Length_minutes, the other columns dont have a linear correlation with Listening_Time_minutes
# So we will feature all these columns


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
    elif 60 < avg_popularity <= 80:
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


df_test.info()


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
df['Genre'] = df.apply(genre, axis=1)
df_test['Genre'] = df_test.apply(genre, axis=1)





df['ratio_Ads'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
df_test['ratio_Ads'] = df_test['Number_of_Ads'] / df_test['Episode_Length_minutes']


objects = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Popular_Level', 'Podcast_Name']
fig, axes = plt.subplots(3, 2, figsize=(20, 10))
axes = axes.flatten()
for ax, i in zip(axes, objects):
    sns.histplot(data = df,x = i,ax = ax, kde=True, stat='density', bins=30)
    ax.set_title(f'Hisplot describe for {i}')
    ax.set_xlabel(i)
    for i in ax.get_xticklabels():
        i.set_rotation(45)
plt.tight_layout()
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


df['ratio_Ads'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
df_test['ratio_Ads'] = df_test['Number_of_Ads'] / df_test['Episode_Length_minutes']


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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


param_grid = [
    {
        'model': [xgb.XGBRegressor()],
        'model__n_estimators': [100],
        'model__max_depth': [12],
        'model__learning_rate': [0.055356453546653, 0.085465646353584, 0.064563215458],
        'model__subsample': [0.8645634],
        'model__colsample_bytree': [0.8742245],
        'model__gamma': [1.403452865],
        'model__reg_alpha': [2.785587517],
        'model__reg_lambda': [3.576579],
        'model__min_child_weight': [6],
        'model__tree_method': ['exact'],  
    }
]


k_cv = KFold(n_splits= 4, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipe, param_grid, cv=k_cv, scoring= {'r2': 'r2', 'mse': 'neg_mean_squared_error'},refit='mse')
grid_search.fit(X_train, Y_train)

print('Best score of the model is: ')
print(f"Best score: {grid_search.best_score_}")
print(f"Which will use the model: \n {grid_search.best_estimator_}")


Y_pred_train = grid_search.best_estimator_.predict(X_train)
print(mean_squared_error(Y_train , Y_pred_train))


print(r2_score(Y_train , Y_pred_train))


Y_pred_test = grid_search.best_estimator_.predict(X_test)
print(mean_squared_error(Y_test , Y_pred_test))


print(r2_score(Y_test , Y_pred_test))


print(f'Root_mean_square_error is: {np.sqrt(mean_squared_error(Y_train , Y_pred_train))}')


print(f'Root_mean_square_error is: {np.sqrt(mean_squared_error(Y_test , Y_pred_test))}')


df_test.info()


df_test['predict'] = grid_search.best_estimator_.predict(df_test.drop(columns = ['id']))
df_submission = pd.DataFrame({
    'id': df_test['id'], 
    'Listening_Time_minutes' : df_test['predict']
})


df_submission.to_csv('submission.csv', index = False)
df_submission.info()

