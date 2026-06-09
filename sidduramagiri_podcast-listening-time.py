import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
train_df.head()


train_df.shape


test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


test_df.head()


test_df1 = pd.merge(test_df, sample_df, on='id')
test_df1.head()


np.max(train_df['Listening_Time_minutes'])


numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
colors = ['blue','blue','red','green','purple']

plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(train_df[col].dropna(), bins=200, kde=True, color=colors[i])
plt.tight_layout()
plt.show()



categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

plt.figure(figsize=(12, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 2, i)
    train_df[col].value_counts().nlargest(10).plot(kind='bar', color='green')
plt.tight_layout()
plt.show()


plt.figure(figsize=(20,5))
sns.histplot(train_df['Listening_Time_minutes'], bins=150, kde=True)
plt.show()


train_df.info()


train_df.isnull().sum()


train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)



train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(0).astype(int)
mode_val = train_df['Number_of_Ads'].mode()[0]
train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(mode_val).astype(int)


print(test_df1.isnull().sum())
sns.heatmap(test_df1.isnull(), cmap="coolwarm") 
plt.show()


test_df1['Episode_Length_minutes'].fillna(test_df1['Episode_Length_minutes'].median(), inplace=True)
test_df1['Guest_Popularity_percentage'].fillna(test_df1['Guest_Popularity_percentage'].median(), inplace=True)


test_df1['Number_of_Ads'] = test_df1['Number_of_Ads'].fillna(0).astype(int)
mode_val = test_df1['Number_of_Ads'].mode()[0]
test_df1['Number_of_Ads'] = test_df1['Number_of_Ads'].fillna(mode_val).astype(int)


train_df


from itertools import combinations


train_df['Episode_Title'] = train_df['Episode_Title'].str.split(" ", expand=True)[1].astype(int)


train_df['Publication_Day'].replace({
    "Sunday": 0, 
    "Monday": 1, 
    "Tuesday": 2, 
    "Wednesday": 3, 
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}, inplace=True)

train_df['SinWeekDay'] = np.sin(2 * np.pi * train_df['Publication_Day'] / 7)
train_df['CosWeekDay'] = np.cos(2 * np.pi * train_df['Publication_Day'] / 7)


train_df['Publication_Time'].value_counts().sum


train_df['Publication_Time'].replace({
    "Morning": 0, 
    "Afternoon": 1, 
    "Evening": 2, 
    "Night": 3,
}, inplace=True)

train_df['SinTime'] = np.sin(2 * np.pi * train_df['Publication_Time'] / 4)
train_df['CosTime'] = np.cos(2 * np.pi * train_df['Publication_Time'] / 4)


train_df['EpLen_Int'] = np.floor(train_df['Episode_Length_minutes'])
train_df['EpLen_Dec'] = train_df['Episode_Length_minutes'] - train_df['EpLen_Int']


train_df['SinEpLen'] = np.sin(2 * np.pi * train_df['Episode_Length_minutes'] / 60)
train_df['CosEpLen'] = np.cos(2 * np.pi * train_df['Episode_Length_minutes'] / 60)


train_df["Number_of_Ads"] = train_df["Number_of_Ads"].astype(str)


cat_cols = [
    "Podcast_Name", "Episode_Title", "Genre", "Number_of_Ads", "Episode_Sentiment", "EpLen_Int"
]
train_df[cat_cols] = train_df[cat_cols].astype("string")

for col1, col2 in combinations(cat_cols, 2) :
    train_df[f"{col1}-{col2}"] = train_df[col1] + "-" + train_df[col2]


train_df


test_df1['Episode_Title'] = test_df1['Episode_Title'].str.split(" ", expand=True)[1].astype(int)

test_df1['Publication_Day'].replace({
    "Sunday": 0, 
    "Monday": 1, 
    "Tuesday": 2, 
    "Wednesday": 3, 
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}, inplace=True)

test_df1['SinWeekDay'] = np.sin(2 * np.pi * test_df1['Publication_Day'] / 7)
test_df1['CosWeekDay'] = np.cos(2 * np.pi * test_df1['Publication_Day'] / 7)


test_df1['Publication_Time'].replace({
    "Morning": 0, 
    "Afternoon": 1, 
    "Evening": 2, 
    "Night": 3,
}, inplace=True)

test_df1['SinTime'] = np.sin(2 * np.pi * test_df1['Publication_Time'] / 4)
test_df1['CosTime'] = np.cos(2 * np.pi * test_df1['Publication_Time'] / 4)

test_df1['EpLen_Int'] = np.floor(test_df1['Episode_Length_minutes'])
test_df1['EpLen_Dec'] = test_df1['Episode_Length_minutes'] - test_df1['EpLen_Int']


test_df1['SinEpLen'] = np.sin(2 * np.pi * test_df1['Episode_Length_minutes'] / 60)
test_df1['CosEpLen'] = np.cos(2 * np.pi * test_df1['Episode_Length_minutes'] / 60)

test_df1["Number_of_Ads"] = test_df1["Number_of_Ads"].astype(str)

cat_cols = [
    "Podcast_Name", "Episode_Title", "Genre", "Number_of_Ads", "Episode_Sentiment", "EpLen_Int"
]
test_df1[cat_cols] = test_df1[cat_cols].astype("string")

for col1, col2 in combinations(cat_cols, 2) :
    test_df1[f"{col1}-{col2}"] = test_df1[col1] + "-" + test_df1[col2]


print(train_df.info(), test_df1.info())


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

train = train_df
test = test_df1
string_cols = [col for col in train.columns if train[col].dtype == 'string']
encoders = {}

for col in string_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    encoders[col] = le
for col in string_cols:
    if col in test.columns:
        le = encoders[col]
        test[col] = test[col].astype(str).map(
            lambda x: x if x in le.classes_ else '-1'
        ).map(
            {val: idx for idx, val in enumerate(np.append(le.classes_, '-1'))}
        ).astype(int)


train.info(), test.info()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


X = train.drop(columns=['id','Listening_Time_minutes'])
y = train['Listening_Time_minutes']

X_test = test.drop(columns=['id', 'Listening_Time_minutes'])


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=1024,
    callbacks=[early_stop],
    verbose=1
)


val_loss, val_mae = model.evaluate(X_val, y_val)
print(f"Validation MAE: {val_mae:.2f} minutes")

predictions = model.predict(X_test)


print(predictions)


from catboost import CatBoostRegressor, Pool

# Define features and target
target = "Listening_Time_minutes"
features = train_df.columns.drop(["id", target])  

cat_features = [
    "Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time",
    "Number_of_Ads", "Episode_Sentiment", "EpLen_Int",
]

train_pool = Pool(data=train_df[features], label=train_df[target], cat_features=cat_features)

# Train the model
model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function='RMSE',
    verbose=100,
    random_seed=42
)
model.fit(train_pool)



from catboost import Pool

test_pool = Pool(data=test_df1[features], cat_features=cat_features)
predictions = model.predict(test_pool)


predictions


from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
import numpy as np

cat_features = [
    'Podcast_Name',
    'Episode_Title',
    'Genre',
    'Publication_Day',
    'Publication_Time',
    'Episode_Sentiment',
    'Podcast_Name-Episode_Title',
    'Podcast_Name-Genre',
    'Podcast_Name-Number_of_Ads',
    'Podcast_Name-Episode_Sentiment',
    'Podcast_Name-EpLen_Int',
    'Episode_Title-Genre',
    'Episode_Title-Number_of_Ads',
    'Episode_Title-Episode_Sentiment',
    'Episode_Title-EpLen_Int',
    'Genre-Number_of_Ads',
    'Genre-Episode_Sentiment',
    'Genre-EpLen_Int',
    'Number_of_Ads-Episode_Sentiment',
    'Number_of_Ads-EpLen_Int',
    'Episode_Sentiment-EpLen_Int'
]

target = 'Listening_Time_minutes'
features = [col for col in train_df.columns if col != target]

train_pool = Pool(data=train_df[features], label=train_df[target], cat_features=cat_features)
test_pool = Pool(data=test_df1[features], cat_features=cat_features)

model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=100,
    random_seed=42
)

model.fit(train_pool)

predictions = model.predict(test_pool)



predictions


submission = pd.DataFrame({
    'id': test_df1['id'].values,
    'Listening_Time_minutes': predictions})
submission.to_csv('submission.csv', index=False)




