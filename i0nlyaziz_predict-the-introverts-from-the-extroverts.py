import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import the necessary libraries


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# load the data 


train.head()


test.head()


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columns = ['Stage_fear','Drained_after_socializing','Personality']

for i in columns :
  train[i] = encoder.fit_transform(train[i])

# preprocess the train data 


train.isnull().sum().sum()

# check for missing values  


train.fillna(train.mean(),inplace=True)


train.duplicated().sum()

# check for duplicates values


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columns = ['Stage_fear','Drained_after_socializing']

for i in columns :
  test[i] = encoder.fit_transform(test[i])

# preprocess the test data


test.isnull().sum().sum()

# check for missing values  


test.fillna(test.mean(),inplace=True)


test.duplicated().sum()

# check for duplicates values


plt.figure(figsize=(8,8))
sns.countplot(x='Personality',data=train)
plt.title('Extrovert vs Introvert')
plt.xlabel(" 0 = Extrovert , 1 = Introvert ")
plt.show()

# explore the data with countplot 


values = train['Friends_circle_size'].value_counts()

plt.figure(figsize=(8,8))
plt.pie(values, labels=values.index, autopct='%1.1f%%')
plt.title("Friends Circle Size")
plt.show()

# explore the data with pie plot 


values = train['Time_spent_Alone'].value_counts()

plt.figure(figsize=(8,8))
plt.pie(values, labels=values.index, autopct='%1.1f%%')
plt.title("Time Spent Alone")
plt.show()

# explore the data with pie plot 


plt.figure(figsize=(8,8))
sns.heatmap(train.corr(),annot=True)
plt.show()

# explore the data with heatmap


x = train.drop(columns='Personality',axis=1)
y = train['Personality']


from sklearn.model_selection import train_test_split

x_train , x_valid , y_train , y_valid = train_test_split(x,y,test_size=0.3,random_state=42)

# split the data


from sklearn.ensemble import RandomForestClassifier

Model = RandomForestClassifier()
Model.fit(x_train,y_train)

# train the model


y_pred = Model.predict(x_valid)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_valid,y_pred)
print(f"""The Accuracy : {accuracy * 100:.2f}%""")

# evaluate the model with accuracy score


from sklearn.metrics import classification_report

print(classification_report(y_valid,y_pred))

# evaluate the model with classification report


importanc = Model.feature_importances_
columns = x.columns

plt.figure(figsize=(10,10))
plt.barh(columns,importanc)
plt.show()

# Visualize feature importance


print(f"""Train Score : {Model.score(x_train,y_train) * 100:.2f}%""")
print(f"""Test Score : {Model.score(x_valid,y_valid) * 100:.2f}%""")

# display training and test accuracy 


predictions = Model.predict(test)


submission = pd.DataFrame({
    "id": test["id"],
    "target": predictions
})
submission.to_csv("submission.csv", index=False)


# Creating submission with model predictions
submission = pd.DataFrame({'id': test['id'], "Personality": predictions})

# Converting 1s back to Extrovert and 0s back to Introvert
submission['Personality'] = submission['Personality'].replace({0: 'Extrovert', 1: 'Introvert'})


submission

