# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
map_binary = {'Yes': 1, 'No': 0}
map_label = {'Introvert': 1, 'Extrovert': 0}
df_train['Stage_fear'] = df_train['Stage_fear'].map(map_binary)
df_train['Drained_after_socializing'] = df_train['Drained_after_socializing'].map(map_binary)
df_train['Personality'] = df_train['Personality'].map(map_label)
df_train = df_train.fillna(-1)


#amount of rows that are introvert
true_positive = len(df_train[((df_train['Stage_fear'] == 1) | (df_train['Drained_after_socializing'] == 1)) & (df_train['Personality'] == 1)])
true_negative = len(df_train[((df_train['Stage_fear'] == 0) & (df_train['Drained_after_socializing'] == 0)) & (df_train['Personality'] == 0)])
print(f"True positive: {true_positive}")
print(f"True negative: {true_negative}")
print(f"Total explained: {true_positive+true_negative}")


false_positive = len(df_train[((df_train['Stage_fear'] == 0) & (df_train['Drained_after_socializing'] == 0)) & (df_train['Personality'] != 0)])
false_negative = len(df_train[((df_train['Stage_fear'] == 1) | (df_train['Drained_after_socializing'] == 1)) & (df_train['Personality'] != 1)])
#amount of rows that are introvert
print(f"False positve (without missing values): {false_positive}")
print(f"False negative: {false_negative}")
print(f'Total unexplained without missing values: {false_positive+false_negative}')


from sklearn.neighbors import KNeighborsClassifier


def missing_filler(df, col):
    # split train and test
    X_train_col = df[df[col] != -1].drop(columns=['Stage_fear','Drained_after_socializing'])
    y_train_col = df[df[col] != -1][col]
    X_test_col = df[df[col] == -1].drop(columns=['Stage_fear','Drained_after_socializing'])
    # train
    model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    model.fit(X_train_col, y_train_col)
    # fill
    y_pred = model.predict(X_test_col)
    df.loc[df[col] == -1, col] = y_pred
    return df


df_clone = missing_filler(df_train.drop(columns=['id', 'Personality']), 'Stage_fear')
df_clone = missing_filler(df_clone, 'Drained_after_socializing')
df_clone['Personality'] = df_train['Personality']
df_clone.head()


#amount of rows that are introvert
true_positive_clone = len(df_clone[((df_clone['Stage_fear'] == 1) | (df_clone['Drained_after_socializing'] == 1)) & (df_clone['Personality'] == 1)])
true_negative_clone = len(df_clone[((df_clone['Stage_fear'] == 0) & (df_clone['Drained_after_socializing'] == 0)) & (df_clone['Personality'] == 0)])
print(f"True positive: {true_positive_clone}")
print(f"True negative: {true_negative_clone}")
print(f"Total explained rows: {true_positive_clone+true_negative_clone}")


false_positive_clone = len(df_clone[((df_clone['Stage_fear'] == 0) & (df_clone['Drained_after_socializing'] == 0)) & (df_clone['Personality'] != 0)])
false_negative_clone = len(df_clone[((df_clone['Stage_fear'] == 1) | (df_clone['Drained_after_socializing'] == 1)) & (df_clone['Personality'] != 1)])
#amount of rows that are introvert
print(f"False positve: {false_positive_clone}")
print(f"False negative: {false_negative_clone}")
print(f'Total unexplained rows: {false_positive_clone+false_negative_clone}')


import statsmodels.formula.api as smf
import statsmodels.api as sm


 model_logist = smf.glm(
        formula='Personality ~ Stage_fear + Drained_after_socializing',
        data=df_clone,
        family=sm.families.Binomial()
 ).fit()
model_logist.summary()


print(f"True positive (treshold 0.5): {len(df_clone[((df_clone['Stage_fear'] == 1) & (df_clone['Drained_after_socializing'] == 0)) & (df_clone['Personality'] == 1)])}")
print(f"True negative (treshold > 0.5): {len(df_clone[((df_clone['Stage_fear'] == 1) & (df_clone['Drained_after_socializing'] == 0)) & (df_clone['Personality'] != 1)])}")


from sklearn.metrics import accuracy_score


 model_logist = smf.glm(
        formula='Personality ~ Time_spent_Alone + Stage_fear + Social_event_attendance + Going_outside + Drained_after_socializing + Friends_circle_size + Post_frequency',
        data=df_clone,
        family=sm.families.Binomial()
 ).fit()
model_logist.summary()


probs = model_logist.predict(df_clone.drop(columns=['Personality']))
y_pred = (probs > 0.5).astype(int)
acc = accuracy_score(df_clone['Personality'], y_pred)
print(f'Accuracy of Logistic model with all parameters: {acc}')


df_res = pd.DataFrame({'Personality': y_pred})
df_res.head()


true_positive = df_clone[(df_clone['Personality'] == 1) & (df_res['Personality'] == 1)]
true_negative = df_clone[(df_clone['Personality'] == 0) & (df_res['Personality'] == 0)]
false_positive = df_clone[(df_clone['Personality'] == 0) & (df_res['Personality'] == 1)]
false_negative = df_clone[(df_clone['Personality'] == 1) & (df_res['Personality'] == 0)]
print(f'True positive: {len(true_positive)}')
print(f'True negative: {len(true_negative)}')
print(f'Total explained: {len(true_positive) + len(true_negative)}')
print(f'False positive: {len(false_positive)}')
print(f'False negative: {len(false_negative)}')
print(f'Total unexplained: {len(false_positive) + len(false_negative)}')


# compare them
true_positive.describe()
#false_negative.describe()


# compare them 
true_negative.describe()
#false_positive.describe()


# read the data
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test['Stage_fear'] = df_test['Stage_fear'].map(map_binary)
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].map(map_binary)
df_test = df_test.fillna(-1)


df_test_clone = missing_filler(df_test.drop(columns=['id']), 'Stage_fear')
df_test_clone = missing_filler(df_test_clone, 'Drained_after_socializing')
df_test_clone.head()


probs = model_logist.predict(df_test_clone)
y_pred = (probs > 0.5).astype(int)
df_res = pd.DataFrame({'id': df_test['id'],'Personality': y_pred})
map_label_inverse = {
    0: 'Extrovert',
    1: 'Introvert'
}
df_res['Personality'] = df_res['Personality'].map(map_label_inverse)
df_res.to_csv('/kaggle/working/submission.csv', index=False)
df_res.head()

