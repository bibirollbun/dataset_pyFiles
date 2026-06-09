import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


train.head()


train.info()


train.describe()


unique_count = train['Episode_Title'].nunique()
print(unique_count)



train['Episode'] = train['Episode_Title'].str.extract('(\d+)').astype(int)


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3 ,figsize=(16, 8))
ax1.hist(train['Episode_Length_minutes'])
ax1.set_title('Episode Length (minutes)')


ax2.hist(train['Listening_Time_minutes'])
ax2.set_title('Listening Time (minutes)')

ax3.hist(train['Guest_Popularity_percentage'])
ax3.set_title('Guest Popularity (%)')

ax4.hist(train['Host_Popularity_percentage'])
ax4.set_title('Host Popularity (%)')

ax5.hist(train['Episode'])
ax5.set_title('Episode')


ax6.hist(train['Number_of_Ads'], bins=4)
ax6.set_title('Number of Ads')


print(train['Episode_Length_minutes'].quantile(0.999), train['Number_of_Ads'].quantile(0.999))
print(train['Episode_Length_minutes'].quantile(0.9999), train['Number_of_Ads'].quantile(0.9999))
print(train['Episode_Length_minutes'].quantile(0.99999), train['Number_of_Ads'].quantile(0.99999))
print(train['Episode_Length_minutes'].quantile(0.999999), train['Number_of_Ads'].quantile(0.999999))


train = train[(train['Episode_Length_minutes'] < 121) | (train['Episode_Length_minutes'].isnull())]
train = train[(train['Number_of_Ads'] <= 3) | (train['Number_of_Ads'].isnull())]


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

days=train['Publication_Day'].value_counts()
days = pd.DataFrame(days)
days.reset_index(inplace=True)

times=train['Publication_Time'].value_counts()
times = pd.DataFrame(times)
times.reset_index(inplace=True)

genres=train['Genre'].value_counts()
genres = pd.DataFrame(genres)
genres.reset_index(inplace=True)

sentiment=train['Episode_Sentiment'].value_counts()
sentiment = pd.DataFrame(sentiment)
sentiment.reset_index(inplace=True)

ax1.barh(days['Publication_Day'], days['count'])
ax2.barh(times['Publication_Time'], times['count'])
ax3.barh(genres['Genre'], genres['count'])
ax4.barh( sentiment['Episode_Sentiment'], sentiment['count'])

plt.tight_layout()
plt.show()


days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
train_1=train.copy()
train_1['Publication_Day']=pd.Categorical(train_1['Publication_Day'], categories=days_order, ordered= True)

time_order=['Morning', 'Afternoon', 'Evening', 'Night']
train_1['Publication_Time']=pd.Categorical(train_1['Publication_Time'], categories = time_order, ordered = True)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(21, 12))


sns.boxplot(y=train_1['Listening_Time_minutes'], x=train_1['Publication_Day'], ax=ax1)
sns.boxplot(y=train_1['Listening_Time_minutes'], x=train_1['Publication_Time'], ax=ax2)
sns.boxplot(y=train_1['Listening_Time_minutes'], x=train_1['Genre'], ax=ax3)
sns.boxplot(y=train_1['Listening_Time_minutes'], x=train_1['Episode_Sentiment'], ax=ax4)


sns.set_theme(rc={'figure.figsize':(16, 8)})
ax=sns.boxplot(y=train['Listening_Time_minutes'], x=train['Publication_Day'], hue=train['Publication_Time'])
ax.legend(loc='upper right')


def features_create(df):
    df['guest*host']=df['Host_Popularity_percentage']*df['Guest_Popularity_percentage']

    df.loc[df['Number_of_Ads'] == 0, 'length/ads'] = df.loc[df['Number_of_Ads'] == 0,'Episode_Length_minutes']
    df.loc[df['Number_of_Ads'] > 0, 'length/ads'] = df['Episode_Length_minutes']/ df['Number_of_Ads']
    return df


features_create(train)


num_cols = ['Episode_Length_minutes','Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode', 'Listening_Time_minutes', 'guest*host', 'length/ads']
cat_cols= ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


correlation_matrix = train[num_cols].corr()
spearman_matrix = train[num_cols].corr(method='spearman')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax1, cbar=False)
ax1.set_title('Pearson Correlation')
sns.heatmap(spearman_matrix, annot=True, cmap='coolwarm', ax=ax2, cbar=False)
ax2.set_title('Spearman Correlation')
plt.subplots_adjust(wspace=0.65)

plt.show()


X_train = train.drop(['id','Episode_Title','Listening_Time_minutes'], axis=1)
y_train= train['Listening_Time_minutes']


print(X_train.shape, y_train.shape)


print(train.columns)


num_cols = ['Episode_Length_minutes','Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode', 'guest*host', 'length/ads']


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])


col_transformer= ColumnTransformer(transformers = [
    ('num_pipeline', num_pipeline, num_cols),
    ('cat_pipeline', cat_pipeline, cat_cols)
],
remainder= 'drop',
n_jobs= -1
)


X_prepared = col_transformer.fit_transform(X_train)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lr=LinearRegression()
linear_model_scores = cross_val_score(lr, X_prepared, y_train, cv=4, scoring='neg_mean_squared_error')


print((-linear_model_scores)**(1/2))


for col in cat_cols:
    X_train[col] = X_train[col].astype('category')


from xgboost import XGBRegressor


params = {'n_estimators': 856,
          'max_depth': 18,
          'learning_rate': 0.01862086690084558,
          'colsample_bytree': 0.5446339868372041,
          'gamma': 2.900622391304954,
          'reg_alpha': 1.1723017326650842,
          'reg_lambda': 8.838105586253267}

    
xgb = XGBRegressor(**params, enable_categorical=True)
xgb.fit(X_train, y_train)


test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


test.info()


test['Episode'] = test['Episode_Title'].str.extract('(\d+)').astype(int)
features_create(test)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
ax1.hist(test['Episode_Length_minutes'])
ax1.set_title('Episode Length (minutes)')

ax2.hist(test['Guest_Popularity_percentage'])
ax2.set_title('Guest Popularity (%)')

ax3.hist(test['Host_Popularity_percentage'])
ax3.set_title('Host Popularity (%)')


ax4.hist(test['Number_of_Ads'], bins=4)
ax4.set_title('Number of Ads')

plt.show()



print(test['Episode_Length_minutes'].quantile(0.99999))
print(test['Number_of_Ads'].quantile(0.99999))


for i in test.index:
    if test.loc[i,'Episode_Length_minutes'] > 121:
        test.loc[i, 'Episode_Length_minutes'] = 121

for i in test.index:
    if test.loc[i,'Number_of_Ads'] > 4:
        test.loc[i, 'Number_of_Ads'] = 4


test= test.drop(['id', 'Episode_Title'], axis=1)


test.describe()


test[cat_cols]= test[cat_cols].astype('category')


pred_xgb = xgb.predict(test)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


sample_submission['Listening_Time_minutes'] = pred_xgb


sample_submission.to_csv('/kaggle/working/submission.csv', index=False)

