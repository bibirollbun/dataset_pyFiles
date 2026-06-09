# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train



test


train = train.drop(columns='id')
test = test.drop(columns='id')


train.describe()


print(train.isnull().sum())
print(train.isnull().sum().sum())


type_counts = train['Personality'].value_counts()

max_train = type_counts.max()
min_train = type_counts.min()
percent_diff_train = round(((max_train - min_train) / max_train) * 100)

print("Количество наблюдений по категориям в train:")
print(type_counts)
print("\nРазница между максимальной и минимальной категорией в train:", percent_diff_train, "%")


cols = ['Stage_fear', 'Drained_after_socializing']
train[cols] = train[cols].replace({'Yes': 1, 'No': 0})
train['Personality'] = train['Personality'].replace({'Extrovert': 1, 'Introvert': 0})
test[cols] = test[cols].replace({'Yes': 1, 'No': 0})


train


correlation_matrix = train.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="crest", annot=True)
plt.title("Correlation map", fontweight='bold', fontsize=16)
plt.show()


plt.figure(figsize=(12, 6))
sns.kdeplot(data=train, x='Time_spent_Alone', hue='Personality', fill=True, common_norm=False)

for activity in train['Personality'].unique():
    sns.kdeplot(data=train[train['Personality'] == activity], 
                x='Time_spent_Alone', 
                label=activity, 
                fill=True, 
                common_norm=False)

plt.axvline(x=3.475, color='black', linestyle='-')
plt.title('Distribution density Time_spent_Alone for each personality')
plt.xlabel('Time_spent_Alone')
plt.ylabel('Time_spent_Alone')
plt.legend(title='Personalities', loc='upper right')

plt.show()


plt.figure(figsize=(12, 6))
sns.kdeplot(data=train, x='Social_event_attendance', hue='Personality', fill=True, common_norm=False)

for activity in train['Personality'].unique():
    sns.kdeplot(data=train[train['Personality'] == activity], 
                x='Social_event_attendance', 
                label=activity, 
                fill=True, 
                common_norm=False)

plt.axvline(x=3.57, color='black', linestyle='-')
plt.title('Distribution density Social_event_attendance for each personality')
plt.xlabel('Social_event_attendance')
plt.ylabel('Social_event_attendance')
plt.legend(title='Personalities', loc='upper right')

plt.show()


plt.figure(figsize=(12, 6))
sns.kdeplot(data=train, x='Going_outside', hue='Personality', fill=True, common_norm=False)

for activity in train['Personality'].unique():
    sns.kdeplot(data=train[train['Personality'] == activity], 
                x='Going_outside', 
                label=activity, 
                fill=True, 
                common_norm=False)

plt.axvline(x=2.85, color='black', linestyle='-')
plt.title('Distribution density Going_outside for each personality')
plt.xlabel('Going_outside')
plt.ylabel('Going_outside')
plt.legend(title='Personalities', loc='upper right')

plt.show()


plt.figure(figsize=(12, 6))
sns.kdeplot(data=train, x='Friends_circle_size', hue='Personality', fill=True, common_norm=False)

for activity in train['Personality'].unique():
    sns.kdeplot(data=train[train['Personality'] == activity], 
                x='Friends_circle_size', 
                label=activity, 
                fill=True, 
                common_norm=False)

plt.axvline(x=5.6, color='black', linestyle='-')
plt.title('Distribution density Friends_circle_size for each personality')
plt.xlabel('Friends_circle_size')
plt.ylabel('Friends_circle_size')
plt.legend(title='Personalities', loc='upper right')

plt.show()


plt.figure(figsize=(12, 6))
sns.kdeplot(data=train, x='Post_frequency', hue='Personality', fill=True, common_norm=False)

for activity in train['Personality'].unique():
    sns.kdeplot(data=train[train['Personality'] == activity], 
                x='Post_frequency', 
                label=activity, 
                fill=True, 
                common_norm=False)

plt.axvline(x=3, color='black', linestyle='-')
plt.title('Distribution density Post_frequency for each personality')
plt.xlabel('Post_frequency')
plt.ylabel('Post_frequency')
plt.legend(title='Personalities', loc='upper right')

plt.show()


def predict(df: pd.DataFrame):
    predictions = []  

    for _, row in df.iterrows():
        intr_flag = extr_flag = 0
        if row['Time_spent_Alone'] >= 3.475:
            intr_flag += 1
        else:
            extr_flag += 1

        if row['Social_event_attendance'] <= 3.57:
            intr_flag += 1
        else:
            extr_flag += 1

        if row['Going_outside'] <= 2.85:
            intr_flag += 1
        else:
            extr_flag += 1

        if row['Friends_circle_size'] <= 5.6:
            intr_flag += 1
        else:
            extr_flag += 1

        if row['Post_frequency'] <= 3:
            intr_flag += 1
        else:
            extr_flag += 1

        if row['Stage_fear'] == 1:
            intr_flag += 1
        else:
            extr_flag += 1

        if row['Drained_after_socializing'] == 1:
            intr_flag += 1
        else:
            extr_flag += 1

        if intr_flag > extr_flag:
            predictions.append('Introvert') 
        else:
            predictions.append('Extrovert')

    return np.array(predictions)
        

def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    return (y_true == y_pred).mean()


from sklearn.model_selection import train_test_split

X = train.drop(columns=['Personality'])  
y = train['Personality']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
train['Personality'] = train['Personality'].replace({'1': 'Extrovert', '0': 'Introvert'})



train_preds = predict(X_train)
test_preds = predict(X_val)
print(f"Accuracy on train set => {round(accuracy_score(y_train, train_preds) * 100, 2)} %")
print(f"Accuracy on test set => {round(accuracy_score(y_val, test_preds) * 100, 2)} %")


test_preds = predict(test)
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✅ Submission file created successfully!")

