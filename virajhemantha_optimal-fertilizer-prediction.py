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


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head()


train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)


train.head()


import matplotlib.pyplot as plt


values = train['Soil Type'].value_counts().reset_index()
values.columns = ['Soil Type', 'Count']
values


plt.figure()
cmap = plt.get_cmap('Set2')
colors = cmap(range(len(values['Soil Type'])))
plt.bar(values['Soil Type'], values['Count'], color=colors)
plt.title('Soil Type Distribution')

plt.show()


plt.figure()
plt.pie(values['Count'],labels=values['Soil Type'], autopct='%1.1f%%')
plt.title('Soilt Type Distribution')
plt.show()


Fertilize = train['Fertilizer Name'].value_counts().reset_index()
Fertilize.columns = ['Fertilizer Name', 'Count']


fig,(ax_bar, ax_pie) = plt.subplots(1,2, figsize=(10,5))
cmap = plt.get_cmap('Set2')
colors = cmap(range(len(Fertilize['Fertilizer Name'])))
ax_bar.bar(Fertilize['Fertilizer Name'],Fertilize['Count'], color=colors)
ax_bar.set_title('Fertilizer Vise Distribution')
ax_bar.set_xlabel('Fertilizer Name')
ax_bar.set_ylabel('Count')
ax_bar.tick_params(axis='x', rotation=30)
ax_pie.pie(Fertilize['Count'],labels=Fertilize['Fertilizer Name'],colors=colors, autopct='%1.1f%%')
ax_pie.set_title('Fertilizer Vise Distribution')
plt.tight_layout()
plt.show()



train


categorical_columns=['Soil Type','Crop Type']
train_x = train.drop('Fertilizer Name', axis=1)
train_x
train_y = train.pop('Fertilizer Name')
train_y


train_encoded = pd.get_dummies(train_x, prefix=categorical_columns).astype(int)
train_encoded.head()
test_encoded = pd.get_dummies(test, prefix=categorical_columns).astype(int)


from sklearn.preprocessing import StandardScaler
num_cols = list(train.select_dtypes(exclude=['object','category']).columns)
scaler = StandardScaler()
train_encoded[num_cols] = scaler.fit_transform(train_encoded[num_cols])
test_encoded[num_cols] = scaler.transform(test_encoded[num_cols])


import seaborn as sns 
corelation_matrix = train_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corelation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


train_encoded


from sklearn.preprocessing import LabelEncoder

# 2.1 Instantiate
le = LabelEncoder()

# 2.2 Fit on all labels (learn the mapping)
le.fit(train_y)

# 2.3 Transform into numeric codes
y_encoded = le.transform(train_y)

# Now y_encoded is an array of ints in [0..6]
print(y_encoded[:10])    
print(le.classes_)       # shows: array(['10-26-26','14-35-14',…, 'Urea'], dtype=object)



import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

y_train_one_hot = tf.keras.utils.to_categorical(y_encoded, 7)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_encoded,y_train_one_hot,test_size=0.1, random_state=42)





inputs = layers.Input(shape=(22,))
x = tf.keras.layers.Dense(256,activation='relu')(inputs)
x = tf.keras.layers.Dense(128,activation='relu')(x)
x = tf.keras.layers.Dense(64,activation='relu')(x)
x = tf.keras.layers.Dense(32,activation='relu')(x)
output = tf.keras.layers.Dense(7, activation='softmax')(x)


model = models.Model(inputs=inputs,outputs=output)
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)


# param_grid = {
#     'n_estimators': [1000, 2000, 3000],
#     'max_depth': [30, 40, 50],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.4, 0.8, 1.0],
#     'colsample_bytree': [0.4, 0.8, 1.0],
# }



# from xgboost import XGBClassifier

# xgb = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric='mlogloss',
#     random_state=42,
#     objective='multi:softprob'
# )


# from sklearn.model_selection import RandomizedSearchCV
# search = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_grid,
#     n_iter=20,            # number of parameter settings sampled
#     scoring='accuracy',   # or any other metric
#     cv=3,
#     random_state=42,
#     n_jobs=-1
# )
# search.fit(X_train, y_train)


# best_params  = search.best_params_
# final_model  = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
# final_model.fit(X_train, y_train)





import numpy as np
import pandas as pd

submission_data=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
# 1. Get predicted probabilities for each class
probs = model.predict(test_encoded)
#    shape = (n_samples, n_classes)

# 2. Find the indices of the top 3 probabilities for each sample
#    np.argsort sorts ascending; we take last 3 columns, then reverse
top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
#    shape = (n_samples, 3)

# 3. Convert those indices back to class labels
#    We need to flatten, inverse-transform, then reshape
flat_idx     = top3_idx.flatten()
flat_labels  = le.inverse_transform(flat_idx)
top3_labels  = flat_labels.reshape(top3_idx.shape)
#    shape = (n_samples, 3)

# 4. Join each row’s labels with a space
top3_strings = [" ".join(row) for row in top3_labels]

# 5. (Optional) Put into a DataFrame alongside your original data
results = pd.DataFrame({
    'id': submission_data['id'],  # if you want ground truth
    'Fertilizer Name': top3_strings
})
results.to_csv('submission.csv', index=False)
print(results.head())


