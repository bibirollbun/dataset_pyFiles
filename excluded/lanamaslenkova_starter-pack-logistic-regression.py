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


import warnings

warnings.filterwarnings("ignore")


path_to_train_data = "/kaggle/input/playground-series-s5e7/train.csv"

data = pd.read_csv(path_to_train_data)
print("N of rows:", data.shape[0])
print("N of rows:", data.shape[1])
print("Columns names: \n", list(data.columns))

data.head()


# let's see also the test set
path_to_test_data = "/kaggle/input/playground-series-s5e7/test.csv"

test_data = pd.read_csv(path_to_test_data)
print("N of rows:", test_data.shape[0])
print("N of rows:", test_data.shape[1])

test_data.head(1)


# see the descriptive statistics of the numerical columns
data.describe()


# See the counts for categorical columns
categorical_columns = ['Stage_fear', 'Drained_after_socializing']

print(data['Stage_fear'].value_counts())
print()
print(data['Drained_after_socializing'].value_counts())


print(data['Personality'].value_counts())


data.isna().sum()


data.head(1)


from sklearn.preprocessing import LabelEncoder

# create an encoder
label_encoder = LabelEncoder()
# make the encoder to learn the possible labels
label_encoder.fit(data['Personality'])
# let it transform our target variable column values into numerical categories (Extravert: 0, Introvert: 1) 
data['Personality'] = label_encoder.transform(data['Personality'])


# now we do the same for other categorical columns
label_encoder = LabelEncoder()
# we will use fit_transform() method which combines both fit() and transform() from above
data['Stage_fear'] = label_encoder.fit_transform(data['Stage_fear'])
data['Drained_after_socializing'] = label_encoder.fit_transform(data['Drained_after_socializing'])
data.head(1)


from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.15, stratify=data_y, random_state=42)
ids_train, ids_val = train_test_split(data['id'].values, test_size=0.15, stratify=data['Personality'], random_state=42)

print("Num of train samples: ", len(ids_train))
print("Num of validation samples: ", len(ids_val))


def fill_missing(data, columns):
    for column in columns:
        data[column].fillna(np.round(data[column].describe().median()), inplace=True)
    return data

features_columns = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
target_column = ['Personality']

data_train = data[data['id'].isin(ids_train)]
data_val = data[data['id'].isin(ids_val)]

data_train_processed = fill_missing(data_train, columns=features_columns)
data_val_processed = fill_missing(data_val, columns=features_columns)


X_train = data_train_processed[features_columns].to_numpy()
y_train = data_train_processed[target_column].to_numpy()
X_val = data_val_processed[features_columns].to_numpy()
y_val = data_val_processed[target_column].to_numpy()

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_val: ", X_val.shape)
print("y_val: ", y_val.shape)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42).fit(X_train, y_train)


y_pred = model.predict(X_val)


from sklearn.metrics import accuracy_score

accuracy_score(y_true=y_val, y_pred=y_pred)


from sklearn.metrics import confusion_matrix

confusion_matrix(y_val, y_pred)


import matplotlib.pyplot as plt
import seaborn as sns

numerical_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(12, 6))
sns.boxplot(data=data_train_processed[numerical_columns], orient='h')

plt.title('Box Plot of Numerical Variables')
plt.xlabel('Values')
plt.ylabel('Variables')

plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model_tuned = LogisticRegression(random_state=42).fit(X_train, y_train)
y_pred = model_tuned.predict(X_val)
accuracy = accuracy_score(y_true=y_val, y_pred=y_pred)

print(f"Validation Accuracy: {accuracy:.4f}")
confusion_matrix(y_val, y_pred)


import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1, 
    random_state=42
)  

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
confusion_matrix(y_val, y_pred)


do_search = False
if do_search:
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
    }
    
    xgb_model_tuned = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb_model_tuned,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        # verbose=3
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Accuracy: {best_score:.4f}")


best_params = {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'subsample': 0.9}

xgb_best_model = xgb.XGBClassifier(
                        **best_params,
                        objective='binary:logistic', 
                        random_state=42
                        )
xgb_best_model.fit(X_train, y_train)
y_pred = xgb_best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
confusion_matrix(y_val, y_pred)


path_to_test_data = "/kaggle/input/playground-series-s5e7/test.csv"

test_data = pd.read_csv(path_to_test_data)
print("N of rows:", test_data.shape[0])
print("N of rows:", test_data.shape[1])

test_data.head(1)


def fill_missing(data, columns):
    for column in columns:
        data[column].fillna(np.round(data[column].describe().median()), inplace=True)
    return data
    
def prepare_data(data):
    ids = data['id']
    data['Stage_fear'] = data['Stage_fear'].apply(lambda x: 0 if x=='No' else 1 if x=="Yes" else x)
    data['Drained_after_socializing'] = data['Drained_after_socializing'].apply(lambda x: 0 if x=='No' else 1 if x=="Yes" else x)
    data_proccessed = fill_missing(data, columns=['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', \
                          'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency'])
    
    data_x = data_proccessed[['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', \
                          'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']].to_numpy()
    return data_x, ids

def create_submission(model, data_x, ids):
    y_pred = model.predict(data_x)
    pred_df = pd.DataFrame()
    pred_df['id'] = ids
    pred_df['Personality'] = ['Extrovert' if pred==0.0 else 'Introvert' for pred in y_pred]
    return pred_df
    
data_x_test, ids = prepare_data(test_data)
submission = create_submission(xgb_best_model, data_x_test, ids)
submission


submission.to_csv("/kaggle/working/submission.csv",index=False)

