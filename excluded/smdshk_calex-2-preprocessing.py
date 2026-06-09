#0. Basic libraries
import pandas as pd
import numpy as np


#1. Importing data, separating y and converting logs
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train_id = train['id']
test_id = test['id']
train.drop('id', axis = 1, inplace = True)
test.drop('id', axis = 1, inplace = True)
y = train['Calories']
y_log = np.log1p(y)
num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


#2. Outlier row removal (IQR)
def drop_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in num_cols:
    train = drop_outliers(train, col)


#3. Create BMR feature
train['BMR'] = 10 * train['Weight'] + 6.25 * train['Height'] - 5 * train['Age'] + train['Sex'].map({'male': 5, 'female': -161})
test['BMR'] = 10 * test['Weight'] + 6.25 * test['Height'] - 5 * test['Age'] + test['Sex'].map({'male': 5, 'female': -161})


#4. Height and Weight Variables Normalize by Gender Groups
hm_mean = np.mean(train[train['Sex']=='male']['Height'])
hm_std = np.std(train[train['Sex']=='male']['Height'])
hf_mean = np.mean(train[train['Sex']=='female']['Height'])
hf_std = np.std(train[train['Sex']=='female']['Height'])
wm_mean = np.mean(train[train['Sex']=='male']['Weight'])
wm_std = np.std(train[train['Sex']=='male']['Weight'])
wf_mean = np.mean(train[train['Sex']=='female']['Weight'])
wf_std = np.std(train[train['Sex']=='female']['Weight'])

train['Height'] = train.apply(lambda row: (row['Height'] - hm_mean) / hm_std if row['Sex'] == 'male' else (row['Height'] - hf_mean) / hf_std, axis=1)
train['Weight'] = train.apply(lambda row: (row['Weight'] - wm_mean) / wm_std if row['Sex'] == 'male' else (row['Weight'] - wf_mean) / wf_std, axis=1)

test['Height'] = test.apply(lambda row: (row['Height'] - hm_mean) / hm_std if row['Sex'] == 'male' else (row['Height'] - hf_mean) / hf_std, axis=1)
test['Weight'] = test.apply(lambda row: (row['Weight'] - wm_mean) / wm_std if row['Sex'] == 'male' else (row['Weight'] - wf_mean) / wf_std, axis=1)


#5. Create more features
train['BMI'] = train['Weight'] / (train['Height']**2)
train['HR_Duration'] = train['Heart_Rate'] * train['Duration']
train['Temp_Duration'] = train['Body_Temp'] * train['Duration']
train['Age_HR_ratio'] = train['Heart_Rate'] / train['Age']
train['BMI_Duration'] = train['BMI'] * train['Duration']
train['Weight_Duration'] = train['Weight'] * train['Duration']
train['Temp_HR'] = train['Body_Temp'] * train['Heart_Rate']

test['BMI'] = test['Weight'] / (test['Height']**2)
test['HR_Duration'] = test['Heart_Rate'] * test['Duration']
test['Temp_Duration'] = test['Body_Temp'] * test['Duration']
test['Age_HR_ratio'] = test['Heart_Rate'] / test['Age']
test['BMI_Duration'] = test['BMI'] * test['Duration']
test['Weight_Duration'] = test['Weight'] * test['Duration']
test['Temp_HR'] = test['Body_Temp'] * test['Heart_Rate']

num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 
             'BMI', 'HR_Duration', 'Temp_Duration', 'Age_HR_ratio', 
             'BMI_Duration', 'Weight_Duration', 'Temp_HR', 'BMR']


#6. Normalizing X Variables - Not Required in Trees and Ensembles
# from sklearn.preprocessing import PowerTransformer

# pt = PowerTransformer(method='yeo-johnson')
# train[num_cols] = pt.fit_transform(train[num_cols])
# test[num_cols] = pt.transform(test[num_cols])


#7. Regression trees to find bin classification
from sklearn.tree import DecisionTreeRegressor

def get_sp(df, cols, target, max_leaf_nodes, min_samples_leaf, min_impurity_decrease):
    sp_dict = {}
    for col in cols:
        tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, min_samples_leaf = min_samples_leaf, min_impurity_decrease = min_impurity_decrease, random_state=32)
        tree.fit(df[[col]], df[target])
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
        sp_dict[col] = sorted(thresholds.tolist())
    return sp_dict

sp = get_sp(train, num_cols, 'Calories', 7, 75000, 0.25)
for k, v in sp.items():
    print(f"{k}: {v}")


def bin(df, sp_dict):
    binned_cols = {}
    for col, thresholds in sp_dict.items():
        bins = [-float('inf')] + thresholds + [float('inf')]
        labels = [f"{col}_{i}" for i in range(len(thresholds)+1)]
        binned_cols[col] = pd.cut(df[col], bins=bins, labels=labels, right=False)

    return pd.DataFrame(binned_cols)


bin_train = bin(train, sp)
bin_test = bin(test, sp)
train = pd.concat([train,bin_train], axis = 1)
test = pd.concat([test,bin_test], axis = 1)

print(train.shape)
print(test.shape)


train.info()


#8. One-hot encoding
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)
print(train.shape)
print(test.shape)
train.head()


train = train.astype(float)
test = test.astype(float)

