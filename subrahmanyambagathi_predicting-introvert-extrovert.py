import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




# load data
df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df.head()


df.shape


# imbalanced dataset
df['Personality'].value_counts()


df = df.dropna()


# check is there any  missing values are present in dataset
df.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis',yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()


# filling missing values in numerical coiumns with median
num_cols = df.select_dtypes(include =['float64','int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# filling missing values in categorical columns with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)



# convert categorical variables to numerical

df['Stage_fear'] = df['Stage_fear'].map({'Yes':1,'No':0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes':1,'No':0})

df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})


df.head()


df_numeric = df.copy()
df_numeric['Personality'] = df_numeric['Personality']
correlation = df_numeric.corr()

plt.figure(figsize=(12,6))
sns.heatmap(correlation,cmap='coolwarm',annot=True)


# divide the independent and dependent variables
X = df.drop(['id','Personality'], axis=1)
y = df['Personality']


# train_test_split
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,stratify=y,random_state=0)


# scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# it tells the mode "pay more attention  "to the minority calsses

from sklearn.utils.class_weight import compute_class_weight

class_weight = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(y_train),
    y = y_train
)

class_weight_dict = {i : class_weight[i] for i in range(len(class_weight))}
print("Class weights:", class_weight_dict)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# initialize ANN
classifier = Sequential()
# adding the input layer
classifier.add(Dense(units=8,activation='relu',input_dim=X_train_scaled.shape[1]))
# adding first hidden layer
classifier.add(Dense(units=6,activation='relu'))
# adding second hidden layer
classifier.add(Dense(units=4,activation='relu'))
# adding output layer
classifier.add(Dense(1,activation='sigmoid'))



classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



import tensorflow as tf
early_stop=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)


model_history = classifier.fit(X_train_scaled,y_train,validation_data=(X_val_scaled,y_val),batch_size=32,epochs=70,class_weight=class_weight_dict,callbacks=[early_stop])


from sklearn.metrics import classification_report, confusion_matrix

y_pred = classifier.predict(X_val_scaled)
y_pred = (y_pred > 0.5).astype(int)

print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))



test_df=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
test_df.head()


ids = test_df['id']


#  Drop non-feature columns (id, Personality) if present
X_test_raw = test_df.drop(['id', 'Personality'], axis=1, errors='ignore')

#  Fill missing values (use new method, avoid warning)
X_test_raw = X_test_raw.ffill()

#  Encode categoricals (same encoding as training!)
X_test_raw['Stage_fear'] = X_test_raw['Stage_fear'].map({'Yes': 1, 'No': 0})
X_test_raw['Drained_after_socializing'] = X_test_raw['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

#  Scale using training scaler
X_test_scaled = scaler.transform(X_test_raw)



y_pred_test = classifier.predict(X_test_scaled)
y_pred_test = (y_pred_test > 0.5).astype(int).flatten()



labels = ['Introvert' if i == 0 else 'Extrovert' for i in y_pred_test]
output = pd.DataFrame({'id': ids, 'Personality': labels})
output.to_csv("submission.csv", index=False)


import os
print("Exists:", os.path.exists("submission.csv"))



 pd.read_csv("submission.csv",nrows=5)











