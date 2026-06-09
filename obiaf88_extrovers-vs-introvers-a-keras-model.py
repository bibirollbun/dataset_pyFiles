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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.shape, (train.isnull().sum()/ train.shape[0])*100


test.shape, (test.isnull().sum()/ test.shape[0])*100


(train['Personality'].value_counts()/train.shape[0])*100


#mapping personality to 0 or 1 to check correlations with numerical variables
train['Personality_mapped'] = train['Personality'].map({'Introvert':1, 'Extrovert':0})
original['Personality_mapped'] = original['Personality'].map({'Introvert':1, 'Extrovert':0})


train['Null_counts'] = train.isnull().sum(axis = 1)


num_cols = train.select_dtypes(include = 'number').columns


corr = train[num_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Pearson Correlation – Numeric Features")
plt.show()


for col in [col for col in train.columns if col not in ['Personality','Personality_mapped','Null_counts']]:
    train[f'{col}_is_null'] = train[col].isnull().astype(int)


for col in test.columns:
    test[f'{col}_is_null'] = test[col].isnull().astype(int)


train.head(1)


for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside','Friends_circle_size', 'Post_frequency']:
    median_value = original[original['Personality_mapped'] == 1][col].median() 
    train[col].fillna(median_value, inplace = True)
    test[col].fillna(median_value, inplace = True)


train.isnull().sum()


test.isnull().sum()


train.groupby(['Personality','Stage_fear'])['Personality'].value_counts()


train.groupby(['Personality','Drained_after_socializing'])['Personality'].value_counts()


train['Stage_fear'].fillna('Yes',inplace = True)
test['Stage_fear'].fillna('Yes',inplace = True)
train['Drained_after_socializing'].fillna('Yes',inplace = True)
test['Drained_after_socializing'].fillna('Yes',inplace = True)


train.isnull().any().sum(),test.isnull().any().sum()


train['Stage_fear'] = train['Stage_fear'].map({'Yes':1,'No':0})
test['Stage_fear'] = test['Stage_fear'].map({'Yes':1,'No':0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes':1,'No':0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes':1,'No':0})


train.head(1)


test.head(1)


test_id = test['id']
target = train['Personality_mapped']


train.drop(columns = ['id','Personality','Null_counts','id_is_null'], axis = 1, inplace = True)
test.drop(columns = ['id','id_is_null'], axis = 1 , inplace = True)


scaler = MinMaxScaler()


train_scaled = pd.DataFrame(scaler.fit_transform(train[[col for col in train.columns if col not in ['Personality_mapped']]]), columns = scaler.get_feature_names_out())
test_scaled = pd.DataFrame(scaler.transform(test), columns = scaler.get_feature_names_out())


assert train_scaled.shape[1] == test_scaled.shape[1]


train_scaled.shape


X_train, X_valid, y_train, y_valid = train_test_split(train_scaled, target, test_size=0.33, random_state=42, stratify = target)


model = keras.Sequential([
    layers.Dense(42, activation='relu', input_shape=[train_scaled.shape[1]]),
    layers.Dropout(0.3),
    layers.Dense(14, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1,activation = 'sigmoid')
])



model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  
    metrics=['accuracy']
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=100,
    #class_weight=class_weight_dict
    verbose=0, # hide the output because we have so many epochs
)


history_df = pd.DataFrame(history.history)


history_df.head(1)


history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
history_df.loc[:, ['loss', 'val_loss']].plot()


introvers_threshold = {0.5,0.55, 0.6,0.64,0.65,0.66,0.67,0.68,0.69, 0.7, 0.8,0.9,0.95,}
accuracy = {}


for t in introvers_threshold:
    accuracy[t] = accuracy_score(np.where(model.predict(X_valid) >t,1,0).flatten(), y_valid.values)



fig,ax = plt.subplots(figsize = (10,5))
ax.plot(sorted(accuracy.keys()), accuracy.values())
ax.set_title("Optimal threshold for Introvers classification")
ax.axvline(max(accuracy, key=accuracy.get), color = 'red', label = 'optimal threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('Accuaracy')
ax.legend()


best_threshold =  max(accuracy, key=accuracy.get)


print(f"Optimal thresholfd is: { best_threshold}")


submission = pd.DataFrame({
    'id': test_id,         
    'Personality': np.where(model.predict(test_scaled).flatten() > best_threshold,'Introvert','Extrovert')
})


# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
print("Submission created")

