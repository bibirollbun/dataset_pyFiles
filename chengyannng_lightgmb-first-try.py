import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output


data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
data.head()


data.info()


data.duplicated().sum()


data = data.dropna(subset=['Number_of_Ads'])


plt.figure(figsize=(8, 6))
sns.kdeplot(x=data['Episode_Length_minutes'])
sns.histplot(x=data['Episode_Length_minutes'], color='r', stat='density', alpha=0.3)
plt.show()


data[data['Episode_Length_minutes'] > 125]


data = data[(data['Episode_Length_minutes'] <= 125) | (data['Episode_Length_minutes'].isna())]
data.info()


plt.figure(figsize=(8, 6))
sns.kdeplot(x=data['Episode_Length_minutes'])
sns.histplot(x=data['Episode_Length_minutes'], color='r', stat='density', alpha=0.3)
plt.show()


data['Episode_Length_minutes'] = data['Episode_Length_minutes'].fillna(data.groupby('Podcast_Name')['Episode_Length_minutes'].transform('mean'))
data['Episode_Length_minutes'].isna().sum()


plt.figure(figsize=(8, 6))
sns.barplot(x=data['Genre'].value_counts().index, y=data['Genre'].value_counts().values, color='skyblue')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 6))
sns.barplot(x=data['Publication_Time'].value_counts().index, y=data['Publication_Time'].value_counts().values, color='skyblue')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 6))
sns.barplot(x=data['Episode_Sentiment'].value_counts().index, y=data['Episode_Sentiment'].value_counts().values, color='skyblue')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(22, 15))
sns.barplot(x=data['Podcast_Name'].value_counts().index, y=data['Podcast_Name'].value_counts().values, color='skyblue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.kdeplot(x=data['Listening_Time_minutes'])
sns.histplot(x=data['Listening_Time_minutes'], color='r', stat='density', alpha=0.3)
plt.show()


data[data['Listening_Time_minutes'] == 0]


data['Listening_Time_minutes'] = data.groupby('Podcast_Name')['Listening_Time_minutes'] \
    .transform(lambda x: x.replace(0, x[x != 0].mean()))


plt.figure(figsize=(8, 6))
sns.kdeplot(x=data['Listening_Time_minutes'])
sns.histplot(x=data['Listening_Time_minutes'], color='r', stat='density', alpha=0.3)
plt.show()


sampled_data = data.sample(n=1000, random_state=42)
sns.scatterplot(x=sampled_data['Host_Popularity_percentage'], y=sampled_data['Guest_Popularity_percentage'])


plt.figure(figsize=(8, 6))
sns.barplot(x=data.groupby('Publication_Time')['Listening_Time_minutes'].mean().index, y=data.groupby('Publication_Time')['Listening_Time_minutes'].mean().values, color='skyblue')
plt.xticks(rotation=45)
plt.show()



plt.figure(figsize=(8, 6))
sns.barplot(x=data.groupby('Episode_Sentiment')['Listening_Time_minutes'].mean().index, y=data.groupby('Episode_Sentiment')['Listening_Time_minutes'].mean().values, color='skyblue')
plt.xticks(rotation=45)
plt.show()



grouped = data.groupby('Genre')[['Host_Popularity_percentage', 'Guest_Popularity_percentage']].mean().sort_values('Guest_Popularity_percentage', ascending=False)
grouped_long = grouped.reset_index().melt(id_vars='Genre', 
                                          value_vars=['Host_Popularity_percentage', 'Guest_Popularity_percentage'],
                                          var_name='Index', value_name='Meanvalues')
plt.figure(figsize=(12, 8))
sns.barplot(data=grouped_long, x='Genre', y='Meanvalues', hue='Index')
plt.xticks(rotation=45)
plt.show()


grouped = data.groupby('Podcast_Name')[['Host_Popularity_percentage', 'Guest_Popularity_percentage']].mean().sort_values('Guest_Popularity_percentage', ascending=False)
grouped_long = grouped.reset_index().melt(id_vars='Podcast_Name', 
                                          value_vars=['Host_Popularity_percentage', 'Guest_Popularity_percentage'],
                                          var_name='Index', value_name='Meanvalues')
plt.figure(figsize=(12, 8))
sns.barplot(data=grouped_long, x='Podcast_Name', y='Meanvalues', hue='Index')
plt.xticks(rotation=45)
plt.show()


data['Guest_Popularity_percentage'] = data['Guest_Popularity_percentage'].fillna(data.groupby('Podcast_Name')['Guest_Popularity_percentage'].transform('mean'))
data['Guest_Popularity_percentage'].isna().sum()


data.info()


corr_matrix = data.select_dtypes(include=['number']).corr()
sns.heatmap(data=corr_matrix, fmt='.2f', annot=True)


sampled_data = data.sample(n=1000, random_state=42)
sns.scatterplot(x=sampled_data['Episode_Length_minutes'], y=sampled_data['Listening_Time_minutes'])


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor


df = data.copy().drop(columns=['id', 'Episode_Title'])
df = pd.get_dummies(df, dtype=float)


X = df.drop(columns='Listening_Time_minutes')
y = df['Listening_Time_minutes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


lgb = LGBMRegressor(random_state=42, learning_rate=0.1)
model = lgb.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'mse: {mean_squared_error(y_pred, y_test)}')
print(f'rmse: {np.sqrt(mean_squared_error(y_pred, y_test))}')
print(f'r2: {r2_score(y_pred, y_test)}')


test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test.head()


test.info()


test['Episode_Length_minutes'] = test['Episode_Length_minutes'].fillna(test.groupby('Podcast_Name')['Episode_Length_minutes'].transform('mean'))
test['Episode_Length_minutes'].isna().sum()


test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(test.groupby('Podcast_Name')['Guest_Popularity_percentage'].transform('mean'))
test['Guest_Popularity_percentage'].isna().sum()


test.isnull().sum()


test_df = test.copy().drop(columns=['id', 'Episode_Title'])
test_df = pd.get_dummies(test_df, dtype=float)
test_df = test_df.reindex(columns=X_train.columns, fill_value=0)


y_test_pred = model.predict(test_df)
y_test_pred = np.round(y_test_pred, 3)


results = pd.DataFrame({'id': test['id'],
                        "Listening_Time_minutes": y_test_pred})
results.to_csv('submission.csv', index=False)

