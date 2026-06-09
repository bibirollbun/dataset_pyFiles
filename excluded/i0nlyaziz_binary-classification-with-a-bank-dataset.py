import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import the necessary libraries


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# load the data


train.head()


test.head()


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columnns = ['job','marital','education','default','housing','loan','contact','month','poutcome']

for i in columnns :
  train[i] = encoder.fit_transform(train[i])

# preprocess the train data


train.isnull().sum().sum()

# check for missing values  


train.duplicated().sum()

# check for duplicates values


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columnns = ['job','marital','education','default','housing','loan','contact','month','poutcome']

for i in columnns :
  test[i] = encoder.fit_transform(test[i])

# preprocess the test data


test.isnull().sum().sum()

# check for missing values  


test.duplicated().sum()

# check for duplicates values


plt.figure(figsize=(8,8))
sns.countplot(x='loan',data=train)
plt.title('Loan Status')
plt.xlabel('Rejected = 0 , Accepted = 1')
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.countplot(x='education',data=train)
plt.title('Distribution of Education')
plt.xlabel('primary = 0 , secondary = 1 , tertiary = 2 , unknown = 3 ')
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.countplot(x='marital',data=train)
plt.title('Distribution of Marital')
plt.xlabel('divorced = 0 , married = 1 , single = 2 ')
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.countplot(x='housing',data=train)
plt.title('Distribution of Housing Status')
plt.xlabel('No = 0 , Yes = 1 ')
plt.show()

# explore the data with countplot


plt.figure(figsize=(8,8))
sns.histplot(x='age',data=train,kde=True,bins=30)
plt.title('Distribution of Age')
plt.show()

# explore the data with histplot


plt.figure(figsize=(8,8))
sns.heatmap(train.corr(),annot=True)
plt.show()

# explore the data with heatmap


x = train.drop(columns=['y'],axis=1)
y = train['y']


from sklearn.model_selection import train_test_split

x_train , x_valid , y_train , y_valid = train_test_split(x,y,test_size=0.3,random_state=42)

# split the data


from catboost import CatBoostClassifier

Model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='Accuracy',
    verbose=100
)

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


print(f"""Train Score : {Model.score(x_train,y_train) * 100:.2f}%""")
print(f"""Valid Score : {Model.score(x_valid,y_valid) * 100:.2f}%""")

# display training and valid accuracy


importances = Model.feature_importances_
columns = x.columns



plt.figure(figsize=(8,8))
plt.barh(columns , importances)
plt.show()

# visualize feature importance


predictions = Model.predict_proba(test)[:, 1]


submission = pd.DataFrame({
    "id": test["id"],
    "y": predictions
})
submission.to_csv("submission.csv", index=False)


submission

