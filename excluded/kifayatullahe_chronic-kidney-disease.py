import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('/kaggle/input/ckdisease/kidney_disease.csv')


data


data.shape


data.isnull().sum()


data.duplicated().sum()


data.info()


data.describe()


data.classification.value_counts()





from sklearn.impute import SimpleImputer

mode  =  SimpleImputer(missing_values= np.nan, strategy='most_frequent')
df_imputer = pd.DataFrame(mode.fit_transform(data))
df_imputer.columns = data.columns


df_imputer


df_imputer.isnull().sum()


for i in df_imputer.columns:
    print("-------------------", i, "--------------------")
    print()
    print(set(df_imputer[i].tolist()))


print(df_imputer['pcv'].mode())
print(df_imputer['wc'].mode())
print(df_imputer['rc'].mode())


df_imputer['classification'] = df_imputer['classification'].apply(lambda x:'ckd' if x =='ckd\t' else x)
df_imputer['cad'] = df_imputer['cad'].apply(lambda x: 'no' if x == '\tno' else x)

df_imputer['dm'] = df_imputer['dm'].apply(lambda x: 'no' if x == '\tno' else x)
df_imputer['dm'] = df_imputer['dm'].apply(lambda x: 'yes' if x == '\tyes' else x)
df_imputer['dm'] = df_imputer['dm'].apply(lambda x: 'yes' if x == 'yes' else x)

df_imputer['rc'] = df_imputer['rc'].apply(lambda x: 5.2 if x == '\t?' else x)

df_imputer['wc'] = df_imputer['wc'].apply(lambda x: 9800 if x == '\t8400' else x)
df_imputer['wc'] = df_imputer['wc'].apply(lambda x: 9800 if x == '\t6200' else x)
df_imputer['wc'] = df_imputer['wc'].apply(lambda x: 9800 if x == '\t?' else x)

df_imputer['pcv'] = df_imputer['pcv'].apply(lambda x: 41 if x == '\t43' else x)
df_imputer['pcv'] = df_imputer['pcv'].apply(lambda x: 41 if x == '\t?' else x)




for i in df_imputer.columns:
    print("-------------------", i, "--------------------")
    print()
    print(set(df_imputer[i].tolist()))


df_imputer.classification.value_counts()


plt.figure(figsize=(8,5))
sns.countplot(x=df_imputer['classification'], palette='viridis')
plt.grid(linestyle='--', axis='y')


df_imputer.dtypes


data.select_dtypes(exclude = ['object']).columns


for c in data.select_dtypes(exclude = ['object']).columns:
    df_imputer[c]  = df_imputer[c].apply(lambda x: float(x))


df_imputer.dtypes


sns.pairplot(df_imputer)


df_imputer.hist()
plt.show()


plt.figure(figsize=(10,5))
df_imputer.boxplot()


from sklearn.preprocessing import LabelEncoder

df_encoded = df_imputer.copy()  

for col in df_encoded.select_dtypes(include=['object']).columns:  
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str)) 

df_encoded  



df_encoded.to_csv('Final_preprocessing_data.csv', index=False)






corr = df_encoded.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True, cmap='YlOrBr')


X  = df_encoded.drop(columns=(['id', 'classification']))
y = df_encoded['classification']


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X, y , test_size=0.3, random_state=2)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping



model = Sequential()
model.add(Dense(15, activation='relu', input_dim=24))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(15, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor='val_loss',
                        min_delta=0.0001,
                        patience=20,
                        verbose=1,
                        mode='auto',
                        baseline=None,
                        restore_best_weights=False)


history = model.fit(X_train_scaled, y_train, epochs=500, validation_data=(X_test_scaled, y_test), callbacks=callback)


plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.show()


test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test_scaled)
y_pred_labels = (y_pred > 0.5).astype(int)  # Threshold at 0.5

print(classification_report(y_test, y_pred_labels))



cm = confusion_matrix(y_test, y_pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()





