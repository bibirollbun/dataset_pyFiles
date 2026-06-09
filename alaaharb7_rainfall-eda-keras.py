import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid",font_scale=1.5)
import warnings 
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, F1Score



df_train= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_subm= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


df_train.head()


df_train.shape


df_train.info()


df_train.duplicated().sum()


df_train.isna().sum()


df_test.isna().sum().sort_values(ascending = False)


cols = df_train.columns
cols = cols.drop(['day', 'rainfall'])


plt.figure(figsize = (15, 10))
for ind, val in enumerate(cols):
    plt.subplot(2, 5, ind + 1)
    plt.boxplot(df_train[val])
    plt.title(f'Boxplot of {val}')

plt.tight_layout()
plt.show()


df_train['rainfall'].value_counts()


plt.figure(figsize = (10, 6))
ax = sns.countplot(data = df_train, x = 'rainfall', palette = 'Set2')
for container in ax.containers:
    ax.bar_label(container, fontweight = 'black', size = 15)
plt.xticks(ticks=ax.get_xticks(), labels=['Not Rain', 'Rain'])
plt.title(f'rainfall Distribution', fontweight = 'black')


plt.figure(figsize = (15, 8))
sns.kdeplot(df_train['day'], fill = True)
plt.title('Day Distribution', fontweight = 'black', fontsize = 20, pad = 15)
plt.xlabel('Day')
plt.ylabel('Frequency')
plt.show()


df_train['pressure'].describe()


plt.figure(figsize = (15, 8))
sns.histplot(df_train['pressure'], kde = True)
plt.title('pressure Distribution', fontweight = 'black', fontsize = 20, pad = 15)
plt.xlabel('pressure')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize = (15, 8))
sns.kdeplot(df_train['maxtemp'], fill = True)
plt.title('maxtemp Distribution',fontweight = 'black', fontsize = 20, pad = 15)
plt.xlabel('maxtemp')
plt.ylabel('Frequency')
plt.show()


print('maxtemp Describe:')
print(df_train['maxtemp'].describe())
print("-" * 30)
print('temparature Describe:')
print(df_train['temparature'].describe())
print("-" * 30)
print('mintemp Describe:')
print(df_train['mintemp'].describe())


plt.figure(figsize = (15, 8))
sns.kdeplot(df_train['maxtemp'],fill = True)
sns.kdeplot(df_train['temparature'],fill = True)
sns.kdeplot(df_train['mintemp'],fill = True)
plt.legend(['maxtemp', 'temparature', 'mintemp'])
plt.title('maxtemp Vs temparature Vs mintemp Distribution',fontweight = 'black', fontsize = 20, pad = 15)
plt.xlabel('maxtemp & temparature & mintemp')
plt.ylabel('Frequency')
plt.show()


df_train['dewpoint'].describe()


plt.figure(figsize = (15, 8))
sns.kdeplot(df_train['dewpoint'])
plt.title('dewpoint Distribution', fontweight = 'black', fontsize = 20, pad = 15)
plt.xlabel('dewpoint')
plt.ylabel('Frequency')
plt.show()


df_train['humidity'].describe()


plt.figure(figsize = (15, 8))
plt.subplot(1,2,1)
sns.kdeplot(df_train['humidity'], fill = True)
plt.title('humidity Distribution', fontweight = 'black', fontsize = 20, pad = 15)
plt.xlabel('humidity')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
sns.kdeplot(df_train['cloud'], fill = True)
plt.title('cloud Distribution', fontweight = 'black', fontsize = 20, pad = 15)
plt.xlabel('cloud')
plt.ylabel('Frequency')

plt.show()


df_train['sunshine'].describe()


plt.figure(figsize = (15, 8))
sns.histplot(df_train['sunshine'])
plt.title('sunshine Distribution',fontweight = 'black', fontsize = 20, pad = 15)
plt.xlabel('sunshine')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize = (15, 8))
plt.subplot(1,2,1)
sns.kdeplot(df_train['winddirection'], fill = True)
plt.title('winddirection Distribution', fontweight = 'black', fontsize = 20, pad = 15)
plt.xlabel('winddirection')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
sns.kdeplot(x = df_train['windspeed'], fill = True)
plt.title('windspeed Distribution', fontweight = 'black', fontsize = 20, pad = 15)
plt.show()


plt.figure(figsize = (15, 10))
for ind, val in enumerate(cols[:6]):
    plt.subplot(2, 3, ind + 1)
    sns.histplot(data = df_train, x = val, hue = 'rainfall', kde = True)
    plt.title(f'{val} Vs Rainfall',fontweight = 'black', fontsize = 20, pad = 15)

plt.tight_layout()
plt.show()


plt.figure(figsize = (15, 10))
for ind, val in enumerate(cols[6:]):
    plt.subplot(2, 2, ind + 1)
    sns.histplot(data = df_train, x = val, hue = 'rainfall', kde = True)
    plt.title(f'{val} Vs Rainfall',fontweight = 'black', fontsize = 20, pad = 15)

plt.tight_layout()
plt.show()


plt.figure(figsize = (15, 8))
plt.subplot(1,2,1)
sns.scatterplot(data = df_train, x = 'dewpoint', y = 'temparature', hue = 'rainfall')
plt.title('dewpoint Vs temparature Relation',fontweight = 'black', fontsize = 20, pad = 15)

plt.subplot(1,2,2)
sns.scatterplot(data = df_train, x = 'humidity', y = 'temparature', hue = 'rainfall')
plt.title('humidity Vs temparature Relation',fontweight = 'black', fontsize = 20, pad = 15)

plt.show()


plt.figure(figsize = (15, 8))
plt.subplot(1,2,1)
sns.scatterplot(data = df_train, x = 'cloud', y = 'temparature', hue = 'rainfall')
plt.title('cloud Vs temparature Relation',fontweight = 'black', fontsize = 20, pad = 15)

plt.subplot(1,2,2)
sns.scatterplot(data = df_train, x = 'pressure', y = 'temparature', hue = 'rainfall')
plt.title('pressure Vs temparature Relation',fontweight = 'black', fontsize = 20, pad = 15)

plt.show()


corr = df_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', lw=0.2,cmap = 'coolwarm', annot_kws={'size': 10})
plt.title('Correlation Matrix of Features and Rainfall',fontweight = 'black', fontsize = 20, pad = 15)  
plt.show()



df_train["month"] = ((df_train["day"] - 1) // 30) % 12 + 1 
df_test["month"] = ((df_test["day"] - 1) // 30) % 12 + 1 


df_train['month'].value_counts()


df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].median())


X = df_train.drop(['rainfall'], axis = 1)
y = df_train['rainfall']


smote= SMOTE(sampling_strategy='minority')
X_resample,y_resample=smote.fit_resample(X,y)


y_resample.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.show()


X_train = X.drop(columns=['day'])
y_train = y
X_test = df_test.drop(columns=['day'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = Sequential([
    Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.3),
    
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    
    Dense(16, activation='relu', kernel_initializer='he_normal'),
        
    Dense(1, activation='sigmoid')
])


model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'binary_crossentropy', metrics=['accuracy', Recall()])


model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, min_lr = 0.000001)


history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2, 
                    callbacks=[early_stopping, reduce], verbose=1)


epochs = [i+1 for i in range(len(history.history['accuracy']))]

train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


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


y_pred_keras = model.predict(X_test_scaled).flatten()



# Save Submission
df_subm['rainfall'] = y_pred_keras
df_subm.to_csv('submission.csv', index=False)
df_subm.head()

