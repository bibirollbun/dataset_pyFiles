import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from tensorflow.keras.metrics import Precision, Recall, F1Score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")
df.head()


df.info()


df.isna().sum()


df.describe()


ax = sns.countplot(data=df, x ='Target', palette = 'viridis')
for container in ax.containers:
    ax.bar_label(container, fontweight = 'black')
plt.title('Target Distribution')
plt.show()


for i in df.columns:
    print(f'{i} => {df[i].nunique()}')
    print(30 * '-')


cols = ['Daytime/evening attendance','Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
plt.figure(figsize=(15, 10))
for ind, val in enumerate(df[cols]):
    plt.subplot(2,4, ind+1)
    sns.countplot(data=df, x=val, hue='Target', palette = 'viridis')
    plt.title(f'{val} Distribution by Target')

plt.tight_layout()
plt.show()


df.drop('id', axis=1, inplace=True)


X = df.drop('Target', axis = 1)
y = df['Target']


encoder = OneHotEncoder(sparse_output = False)
y = encoder.fit_transform(y.values.reshape(-1,1))


cols_scale = ['Course', 'Previous qualification (grade)', 'Admission grade', 'Previous qualification']
scaler = StandardScaler()

for i in cols_scale:
    X[i] = scaler.fit_transform(X[[i]])


X_train, X_dummy, y_train, y_dummy = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_dummy, y_dummy, test_size = 0.5, random_state = 42, stratify = y_dummy)


X_train.shape


y_train.shape


model = Sequential([
    Dense(1024, activation = 'relu', input_dim=36),
    Dropout(0.2),
    Dense(1024, activation = 'relu'),
    Dense(512, activation = 'relu'),
    Dense(256, activation = 'relu'),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),

    Dense(3, activation = 'softmax')
])

model.compile(optimizer=Adam(learning_rate = 0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.summary()


model.compile(optimizer=Adam(learning_rate = 0.001), loss='categorical_crossentropy', metrics=['accuracy', Recall()])


modelcheckpoints = ModelCheckpoint('model.keras', monitor = 'val_loss', save_best_only = True, save_weights_only = False)
earlystopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
logger = CSVLogger('model.csv')


hist = model.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = 50, batch_size = 32, callbacks = [modelcheckpoints, earlystopping, logger])


model.evaluate(X_valid, y_valid)


epochs = [i+1 for i in range(len(hist.history['accuracy']))]

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']

train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']


plt.figure(figsize = (15,6))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, color = 'green')
plt.plot(epochs, val_loss, color = 'red')
plt.legend(['train_loss', 'val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss & Val_loss')

plt.subplot(1,2,2)
plt.plot(epochs, train_acc, color = 'green')
plt.plot(epochs, val_acc, color = 'red')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy & Val_Accuracy')

plt.show()


X = df.drop('Target', axis = 1)
y = df['Target']


encoder = LabelEncoder()
y = encoder.fit_transform(y)


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)


rf = RandomForestClassifier(n_estimators = 100, max_depth = 9)

rf.fit(X_train, y_train)


print(rf.score(X_train, y_train))


print(rf.score(X_test, y_test))


xgb = XGBClassifier(n_estimators = 30, max_depth = 6)
xgb.fit(X_train, y_train)


print(xgb.score(X_train, y_train))
print(xgb.score(X_test, y_test))


lgbm = LGBMClassifier(n_estimators = 200, max_depth = 6)
lgbm.fit(X_train, y_train)


print(lgbm.score(X_train, y_train))
print(lgbm.score(X_test, y_test))


df_test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
df_test.head()


prep_data = scaler.fit_transform(df_test.iloc[:, 1:])


pred = lgbm.predict(prep_data)


pred


pred = encoder.inverse_transform(pred)


pred


submission = pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')
submission


submission['Target'] = pred


submission.to_csv('submission.csv', index = False)




