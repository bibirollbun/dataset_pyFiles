import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train


train.info()


for col in train:
    if train[col].dtype == 'object':
        train[col] = train[col].fillna('not listed')
    if train[col].dtype == 'int64' or train[col].dtype == 'float64':
        train[col] = train[col].fillna(-99)


train.isna().sum()


test


test.isna().sum()


for col in test:
    if test[col].dtype == 'object':
        test[col] = test[col].fillna('not listed')
    if test[col].dtype == 'int64' or test[col].dtype == 'float64':
        test[col] = test[col].fillna(-99)


test.isna().sum()


submission


train = train.drop('id', axis = 1)
test = test.drop('id', axis = 1)

train.shape, test.shape


# Create the histogram
plt.hist(train['Personality'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Personality Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


train['Personality'].value_counts()


fear = train['Stage_fear'].value_counts()
fear


labels = [fear.index[0], fear.index[1], fear.index[2]]
sizes = [fear.iloc[0], fear.iloc[1], fear.iloc[2]]

print(labels)
print(sizes)

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels)


# Create the histogram
plt.hist(train['Stage_fear'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Stage_fear Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


drained = train['Drained_after_socializing'].value_counts()
print(drained)


labels = [drained.index[0], drained.index[1], drained.index[2]]
sizes = [drained.iloc[0], drained.iloc[1], drained.iloc[2]]

print(labels)
print(sizes)

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels)


# Create the histogram
plt.hist(train['Drained_after_socializing'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Drained_after_socializing Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()



# Create the histogram
plt.hist(train['Time_spent_Alone'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Time_spent_Alone Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


train['Time_spent_Alone'].value_counts()


# Create the histogram
plt.hist(train['Social_event_attendance'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Social_event_attendance Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


train['Social_event_attendance'].value_counts()


# Create the histogram
plt.hist(train['Going_outside'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Going_outside Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


train['Going_outside'].value_counts()


# Create the histogram
plt.hist(train['Friends_circle_size'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Friends_circle_size Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


train['Friends_circle_size'].value_counts()


# Create the histogram
plt.hist(train['Post_frequency'], bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Post_frequency Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()



train['Post_frequency'].value_counts()


replacement_dict = {
    "Yes": 1,
    "No": 0,
    "not listed": 2
}

# Replace words in the column
train['Stage_fear'] = train['Stage_fear'].replace(replacement_dict)
train['Drained_after_socializing'] = train['Drained_after_socializing'].replace(replacement_dict)

test['Stage_fear'] = test['Stage_fear'].replace(replacement_dict)
test['Drained_after_socializing'] = test['Drained_after_socializing'].replace(replacement_dict)



train.info()


y = train.pop('Personality')
X = train
X_test = test


corr = train.corr()
sns.heatmap(corr)


scaler = StandardScaler()

X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


model = SVC(random_state=42).fit(X_train, y_train)
model.score(X_train, y_train)


y_pred = model.predict(X_val)
y_pred


# Get unique values and their counts
unique_values, counts = np.unique(y_pred, return_counts=True)

# Display results
print("Unique values:", unique_values)
print("Counts:", counts)


acc = accuracy_score(y_val, y_pred)
acc


pred = model.predict(X_test)
pred


# Get unique values and their counts
unique_values, counts = np.unique(pred, return_counts=True)

# Display results
print("Unique values:", unique_values)
print("Counts:", counts)


# Create the histogram
plt.hist(pred, bins=30, color='blue', edgecolor='black', alpha=0.7)

# Add labels and title
plt.title('Prediction Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()


submission['Personality'] = pred
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

