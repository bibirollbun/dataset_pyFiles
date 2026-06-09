import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')


traindf = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
testdf = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


traindf


testdf


traindf.info()


traindf.describe()


for col in traindf:
    print(col,"=",traindf[col].nunique())


traindf = traindf.drop_duplicates()


plt.figure(figsize=(12,7))
plots=[traindf['Temparature'],traindf['Humidity'],traindf['Moisture'],traindf['Nitrogen'],traindf['Potassium'],traindf['Phosphorous']]
plt.boxplot(plots)
plt.show();


plt.figure(figsize=(12,7))
sns.countplot(x=traindf['Fertilizer Name'],hue=traindf['Temparature'])
plt.show()





plt.figure(figsize=(12,7))
sns.countplot(x=traindf['Soil Type'],hue=traindf['Fertilizer Name'])
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()


plt.figure(figsize=(12,7))
sns.countplot(x=traindf['Fertilizer Name'],hue=traindf['Crop Type'])
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()


num_cols = []
for col in traindf:
    if(traindf[col].dtype==int and col!='id'):
        num_cols.append(col)
        plt.figure(figsize=(12,7))
        sns.violinplot(data=traindf,x=traindf['Fertilizer Name'],y=traindf[col])
        plt.show()


plt.figure(figsize=(12,7))
corr = traindf[num_cols].corr()
sns.heatmap(corr,annot=True)
plt.show()


from sklearn.preprocessing import LabelEncoder
cat_cols = ['Soil Type', 'Crop Type']
for col in cat_cols:
    le = LabelEncoder()
    le.fit(traindf[col])
    
    traindf[col] = le.transform(traindf[col])
    testdf[col] = le.transform(testdf[col])


# traindf = pd.get_dummies(traindf, columns=['Fertilizer Name'], drop_first=True)


traindf.info()


unique_fertilizers = traindf['Fertilizer Name'].unique()
fertilizer_to_id = {name: idx for idx, name in enumerate(unique_fertilizers)}
id_to_fertilizer = {idx: name for name, idx in fertilizer_to_id.items()}

traindf['Fertilizer Name'] = traindf['Fertilizer Name'].map(fertilizer_to_id)
traindf.drop('id',axis=1,inplace=True)
testdf.drop('id',axis=1,inplace=True)


from sklearn.model_selection import train_test_split
X = traindf.drop('Fertilizer Name', axis=1)
y = traindf['Fertilizer Name']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
testdf=scaler.fit_transform(testdf)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  
)
X.shape


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Input





model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(1800, activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))


model.summary()


model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


 model.fit(X_train,y_train,epochs=25,validation_split=0.2)


yprob = model.predict(X_test)
ypred = yprob.argmax(axis=1)



from sklearn.metrics import accuracy_score
accuracy_score(y_test,ypred)


y_pred = model.predict(testdf)


y_pred = y_pred.argmax(axis=1)


y_pred_names = [id_to_fertilizer[int(i)] for i in y_pred]
ids = np.arange(750000, 750000 + len(y_pred))

df_pred = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': y_pred_names
})

df_pred.to_csv("fertilizer_predictions.csv", index=False)

# Preview the output
print(df_pred.head())





