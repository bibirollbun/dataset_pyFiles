import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


df.head()


df.describe()


df.info()


df.isnull().sum()


sns.histplot(df['Episode_Length_minutes'], kde=True, bins=30)
plt.title('Episode_Length_minutes')
plt.show()


sns.histplot(df['Guest_Popularity_percentage'], kde=True, bins=30)
plt.title('Guest_Popularity_percentage')
plt.show()


missing_features = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

for feature in missing_features:
    df[feature].fillna(df[feature].mean(), inplace=True)
df.isnull().sum()


df.dropna(inplace=True)
df.isnull().sum()


df.columns


features = ['Episode_Length_minutes', 
            'Host_Popularity_percentage',
        'Guest_Popularity_percentage', 'Number_of_Ads',
    'Listening_Time_minutes']

for col in features:
    sns.boxplot(x=df[col])
    plt.show()


deleted_features = ['Episode_Length_minutes', 'Number_of_Ads']
df_cleaned = df.copy()

for col in deleted_features:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 只保留当前列在正常范围内的数据（逐步过滤）
    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]


features = ['Episode_Length_minutes', 
            'Host_Popularity_percentage',
        'Guest_Popularity_percentage', 'Number_of_Ads',
    'Listening_Time_minutes']

for col in features:
    sns.boxplot(x=df_cleaned[col])
    plt.show()


df_cleaned['Genre'].describe()


from sklearn.preprocessing import OneHotEncoder

one_hot_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_array = ohe.fit_transform(df_cleaned[one_hot_features])
encoded_cols = ohe.get_feature_names_out(one_hot_features)

encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_cleaned.index)

df_cleaned_encoded = pd.concat([df_cleaned.drop(columns=one_hot_features), encoded_df], axis=1)


df_cleaned_encoded.drop(columns=['id', 'Podcast_Name', 'Episode_Title'], inplace=True)


from sklearn.ensemble import RandomForestRegressor

X = df_cleaned_encoded.drop(columns=['Listening_Time_minutes'])
y = df_cleaned_encoded['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

