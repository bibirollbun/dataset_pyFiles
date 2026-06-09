import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df.head()


df.info()


for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"{col} -> {df[col].isnull().sum()}\n")


df['Num_Episode'] = df['Episode_Title'].apply(lambda x: int(x.split()[1]))
df = df.drop('Episode_Title', axis=1)


df.dropna(subset=['Number_of_Ads'], inplace=True)


px.histogram(df, x="Episode_Length_minutes", nbins=100)


mean1 = df['Episode_Length_minutes'].mean()
df['Episode_Length_minutes'].fillna(mean1, inplace=True)


px.histogram(df, x="Guest_Popularity_percentage", nbins=100)


mean2 = df['Guest_Popularity_percentage'].mean()
df['Guest_Popularity_percentage'].fillna(mean2, inplace=True)


df.head()


X = df.drop(['id', 'Podcast_Name', 'Listening_Time_minutes'], axis=1)
y = df['Listening_Time_minutes']


numCols = [col for col in X.columns if X[col].dtype != object]
catCols = [col for col in X.columns if X[col].dtype == object]


X = pd.get_dummies(X, columns=catCols, dtype='int8')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


scaler = MinMaxScaler()


X_train[numCols] = scaler.fit_transform(X_train[numCols])
X_test[numCols] = scaler.transform(X_test[numCols])


model = LinearRegression()


model.fit(X_train, y_train)


preds = model.predict(X_test)


np.sqrt(mean_squared_error(y_test, preds))


dfTest = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

idCol = dfTest['id']

dfTest['Episode_Length_minutes'].fillna(mean1, inplace=True)
dfTest['Guest_Popularity_percentage'].fillna(mean2, inplace=True)

dfTest['Num_Episode'] = dfTest['Episode_Title'].apply(lambda x: int(x.split()[1]))
dfTest = dfTest.drop(['id', 'Episode_Title', 'Podcast_Name'], axis=1)

dfTest[numCols] = scaler.transform(dfTest[numCols])
dfTest = pd.get_dummies(dfTest, columns=catCols, dtype='int8')


predictions = model.predict(dfTest)


submission_df = pd.DataFrame({'id': idCol, 'Listening_Time_minutes': predictions})
submission_df.to_csv('submission.csv', index=False)

