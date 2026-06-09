import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df.head()


test_df.head()


print("The shape of train data is: ",train_df.shape)
print("The shape of test data is: ",test_df.shape)


train_df.info()


train_df.describe()


test_df.describe()


train_df.head()


plt.figure(figsize=(10,8))
# Select only numeric features for correlation calculation
numeric_df = train_df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(),annot=True)



plt.figure(figsize=(10, 6))
sns.barplot(
    x='Genre',
    y='Listening_Time_minutes',
    data=train_df,
    palette='viridis',
    edgecolor='black'
)


sns.histplot(train_df['Episode_Length_minutes'], kde=True, color='teal', bins=10)
plt.title('Distribution of Episode Lengths (in minutes)')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Frequency')
plt.show()



sns.countplot(x='Episode_Sentiment', data=train_df, palette='coolwarm')
plt.title('Distribution of Episode Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


sns.heatmap(train_df.isnull())
plt.title('Missing Data Heatmap')
plt.show()


train_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


train_df.drop(['id','Episode_Title'],axis=1,inplace=True)


test_df.drop(['id','Episode_Title'],axis=1,inplace=True)


for i in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
    train_df[i].fillna(train_df[i].median(), inplace=True)
    test_df[i].fillna(test_df[i].median(), inplace=True)


train_df.isnull().sum()


train_df.dropna(inplace=True)
test_df.dropna(inplace=True)


train_df.shape


train_df.info()


train_df['is_train'] = 1
test_df['is_train'] = 0

df=pd.concat([train_df,test_df])


df['Is_Long_Episode']=df['Episode_Length_minutes'].apply(lambda x:1 if x>100 else 0)


df.head()


def check_popularity(df):
  avg_popularity = (df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']) / 2
  if avg_popularity<=20:
    return 'Not popular'
  elif 20 < avg_popularity <= 50:
    return 'Average popularity'
  elif 50 < avg_popularity <= 70:
    return 'Popular'
  else:
    return 'Highly popular'


df['popularity_level']=df.apply(check_popularity,axis=1)


df.drop(['Host_Popularity_percentage','Guest_Popularity_percentage'],axis=1,inplace=True)


def handle_day(df):
  if df['Publication_Day'] in ['Saturday', 'Sunday']:
    return 'Weekend'
  else:
    return 'Weekday'

df['Publication_Day'] = df.apply(handle_day, axis=1)


df.drop(['Podcast_Name'],axis=1,inplace=True)


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


df.head()


df['Genre']=df['Genre'].map({'Entertainment':0,'Events News':1,'Information':2,'Sports':3,'Other':4})


df['Episode_Sentiment']=df['Episode_Sentiment'].map({'Positive':0,'Neutral':1,'Negative':2})


df.head()


df['popularity_level']=df['popularity_level'].map({'Not popular':0,'Average popularity':1,'Popular':2,'Highly popular':3})


df['Is_Weekend'] = df['Publication_Day'].apply(lambda x: 1 if x == 'Weekend' else 0)



time_order = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
df['Publication_Time'] = df['Publication_Time'].map(time_order)


df.head()


df.drop(['Publication_Day'],axis=1,inplace=True)


df.head()


train = df[df['is_train'] == 1].drop('is_train', axis=1)
test = df[df['is_train'] == 0].drop(['is_train', 'Listening_Time_minutes'], axis=1)  # test had no target



train.head()


test.head()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_to_scale = ['Episode_Length_minutes']
train[features_to_scale] = scaler.fit_transform(train[features_to_scale])


from sklearn.model_selection import train_test_split

# Separate features and target
X = train.drop('Listening_Time_minutes', axis=1)
y = train['Listening_Time_minutes']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 20% test, for example
)


X.head()


y.head()


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# Step 3: Fit the model
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate using RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Validation RMSE:", rmse)


model1 = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model1.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model1.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Validation RMSE: {rmse:.4f}")


test.head()


test['Episode_Length_minutes'] = scaler.fit_transform(test[['Episode_Length_minutes']])


y_test_pred = model1.predict(test)


y_test_pred


sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub["Listening_Time_minutes"] = y_test_pred
sub.to_csv("submission.csv", index=False)




