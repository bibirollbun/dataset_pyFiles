import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# importing data
df = pd.read_csv('train.csv')
df


#checking for null values
df.isnull().sum()



genere_listen_time = df.groupby('Genre')['Listening_Time_minutes'].sum().reset_index()


plt.bar(genere_listen_time["Genre"],genere_listen_time['Listening_Time_minutes'],color = 'skyblue')
plt.title('Listening Time for each Genre')
plt.xlabel('Genre',)
plt.xticks(rotation = 45)
plt.ylabel('Listening Time')
plt.show()


df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].mean())
df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())
df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].median())
df.isna().sum()


publication_day_listening = df.groupby('Publication_Day')['Listening_Time_minutes'].sum().reset_index()
plt.title("Listening TIme by Publication Day")
plt.bar(publication_day_listening['Publication_Day'],publication_day_listening['Listening_Time_minutes'])
plt.xlabel('Days')
plt.xticks(rotation = 45)
plt.ylabel('Listening TIme')
plt.show()


df.dtypes


df['Episode_Sentiment'].unique()


le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
df['Publication_Day'] = le.fit_transform(df['Publication_Day'])
df['Publication_Time'] = le.fit_transform(df['Publication_Time'])
df['Episode_Sentiment'] = le.fit_transform(df['Episode_Sentiment'])


df = df.drop(columns=['id','Podcast_Name','Episode_Title'])
df


plt.figure(figsize=(10,8))
plt.title("Correlation Matrix")
sns.heatmap(df.corr(),annot= True , fmt = "0.2f")
plt.show()


x = df.drop(columns=['Listening_Time_minutes'])
y = df['Listening_Time_minutes']
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state= 42, test_size= 0.2)


model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    max_samples=0.8,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)


root_mean_squared_error(y_true=y_test,y_pred=y_pred)


df_test = pd.read_csv('test.csv')
df_test


df_test.isna().sum()


df_test['Episode_Length_minutes'] = df_test['Episode_Length_minutes'].fillna(df_test['Episode_Length_minutes'].mean())
df_test['Guest_Popularity_percentage'] = df_test['Guest_Popularity_percentage'].fillna(df_test['Guest_Popularity_percentage'].median())


df_test.dtypes


df_test['Genre'] = le.fit_transform(df_test['Genre'])
df_test['Publication_Day'] = le.fit_transform(df_test['Publication_Day'])
df_test['Publication_Time'] = le.fit_transform(df_test['Publication_Time'])
df_test['Episode_Sentiment'] = le.fit_transform(df_test['Episode_Sentiment'])
df_test = df_test.drop(columns=['Podcast_Name','Episode_Title'])
df_test


modelfinal = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    max_samples=0.8,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
x_test_final = df_test.drop(columns=['id'])
modelfinal.fit(x,y)
y_pred_final = modelfinal.predict(x_test_final)


y_pred_final = model.predict(x_test_final)


yy = pd.DataFrame({'id': df_test['id'],
    'Listening_Time_minutes' : y_pred_final
    })
yy


yy.to_csv('submission.csv',index=False)




