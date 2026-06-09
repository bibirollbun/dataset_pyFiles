# code by Mohammed Mossad at 21/2/2025 8:28 PM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



df_train = pd.read_csv(r'/kaggle/input/playground-series-s4e6/train.csv')


df_train.head(10)


df_train.shape


df_train.info()


df_train.describe()


for i in df_train.columns:
    print(f'\n {i} column\n min value: {df_train[i].min()} \n max value: {df_train[i].max()}')
    print("\n ---------------------------------------")



df_train.columns


df_train.isna().sum()


df_train.duplicated().sum()


for i in df_train.columns:
    print(f'\n column {i} have: ({df_train[i].nunique()}) unique values')
    print("\n---------------------------")
    


df_train['Target'].value_counts()


# Visualizing target distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df_train['Target'])
plt.title("Target Variable Distribution")
plt.show()


# Visualizing admission grades
plt.figure(figsize=(8,5))
sns.histplot(df_train['Admission grade'], bins=30, kde=True)
plt.title("Distribution of Admission Grades")
plt.show()


# Box plot of admission grades by target category
plt.figure(figsize=(8,5))
sns.boxplot(x=df_train['Target'], y=df_train['Admission grade'])
plt.title("Admission Grade vs Target")
plt.show()


# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_train.corr(numeric_only = True), annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Define categorical features
features = ['Gender', 'Scholarship holder', 'International']

# Create subplots
fig, axes = plt.subplots(1, len(features), figsize=(18, 5))

for i, feature in enumerate(features):
    sns.countplot(x=feature, hue='Target', data=df_train, ax=axes[i])
    axes[i].set_title(f"{feature} Distribution by Target")

plt.tight_layout()
plt.show()



plt.figure(figsize=(8,5))
sns.boxplot(x=df_train['Target'], y=df_train['Age at enrollment'])
plt.title("Age Distribution by Target")
plt.xlabel("Target")
plt.ylabel("Age at Enrollment")
plt.show()


financial_issues = df_train.groupby('Target')['Debtor'].sum().reset_index()
financial_issues.columns = ['Target', 'Financial Issues Students']
financial_issues




plt.figure(figsize=(8,5))
sns.barplot(x='Target', y='Financial Issues Students', data=financial_issues, palette='viridis')
plt.title("Financial Issues by Target Category")
plt.ylabel("Number of Students")
plt.xlabel("Target")
plt.show()


df_train.drop(['id'] , axis = 1 , inplace = True)


X = df_train.drop(['Target'], axis = 1)
Y = df_train['Target']


X


Y


features = [
    'Daytime/evening attendance', 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Target'
]
features_to_scale = [col for col in df_train.columns if col not in features]

scaler = MinMaxScaler()



encoder = LabelEncoder()
Y = encoder.fit_transform(Y)


Y


x_train , x_test , y_train , y_test = train_test_split(X,Y,train_size=0.8,shuffle=True,random_state=42)


x_train[features_to_scale] = scaler.fit_transform(x_train[features_to_scale])
x_test[features_to_scale] = scaler.transform(x_test[features_to_scale])


x_train.head()


x_train.shape


x_test.shape


y_train.shape


y_test.shape


input_dim = x_train.shape[1]
model = Sequential()
model.add(Dense(128,activation='relu',input_dim=input_dim))
model.add(Dropout(0.25))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=12,          
    restore_best_weights=True  
)


model.summary()


history = model.fit(x_train,y_train,epochs=100 ,validation_split=0.2 , callbacks=[early_stopping])


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='best')  
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(loc='best') 
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


y_pred = model.predict(x_test)


model.evaluate(x_test,y_test)


y_pred_classes = np.argmax(y_pred, axis=1)  
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm,annot=True,fmt='d')
plt.show()




