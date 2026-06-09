import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e7/test.csv')


test_df.shape


train_df.shape


#Check for Nan Values
train_df.isna().sum()


#Filling Nan values for numeric columns
train_df= train_df.fillna(train_df.mean(numeric_only=True))
test_df= test_df.fillna(test_df.mean(numeric_only=True))



train_df.isna().sum()


#Filling Nan values for non-numeric columns

mode_fear_train = train_df['Stage_fear'].mode().iloc[0]
mode_drained_train = train_df['Drained_after_socializing'].mode().iloc[0]
train_df.fillna({'Stage_fear':mode_fear_train, 'Drained_after_socializing':mode_drained_train}, inplace=True)

mode_fear_test = test_df['Stage_fear'].mode().iloc[0]
mode_drained_test = test_df['Drained_after_socializing'].mode().iloc[0]
test_df.fillna({'Stage_fear':mode_fear_test, 'Drained_after_socializing':mode_drained_test}, inplace=True)


train_df['Stage_fear'].unique()


train_df.isna().sum()


# Encoding Yes/No values
le = LabelEncoder()
train_df['Stage_fear'] = le.fit_transform(train_df['Stage_fear'])
train_df['Drained_after_socializing'] = le.fit_transform(train_df['Drained_after_socializing'])

# Encoding Introvert/Extrovert values
le_y = LabelEncoder()
train_df['Personality'] = le_y.fit_transform(train_df['Personality'])

test_df['Stage_fear'] = le.fit_transform(test_df['Stage_fear'])
test_df['Drained_after_socializing'] = le.fit_transform(test_df['Drained_after_socializing'])



train_df.head()


# Data split
y_train = train_df['Personality']
X_train = train_df.drop(columns={'Personality'})

X_test = test_df


print(X_train.shape)
print(test_df.shape)


# Normailzing Data
ss = StandardScaler()
scaled_train = ss.fit_transform(X_train)
scaled_test = ss.fit_transform(X_test)


# Training Model
model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


predictions = le_y.inverse_transform(y_pred)


predictions


id = range(18524, 18524+len(predictions))
Personality = predictions

data = {
  'id':id,
  'Personality':Personality
}
result = pd.DataFrame(data)


result


# Creating submission.csv
result.to_csv('output.csv', index=False)

