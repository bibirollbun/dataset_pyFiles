# --- Libraries ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Load dataset ---
train = pd.read_csv("/kaggle/input/titanic/train.csv")

# Quick look at data
print(train.head())
print(train.info())



# --- Load dataset ---
train = pd.read_csv("/kaggle/input/titanic/train.csv")

# Quick look
print(train.head())
print(train.info())



# --- Data Preprocessing ---

# Fill missing values
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# Convert categorical to numeric
train['Sex'] = train['Sex'].map({'male':0, 'female':1})
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)

train.head()



# Survival count
sns.countplot(x='Survived', data=train)
plt.title("Survival Counts")
plt.show()



# Survival by Sex
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title("Survival by Sex")
plt.show()



# Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title("Survival by Passenger Class")
plt.show()



# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()



# Fare distribution
plt.figure(figsize=(8,5))
sns.histplot(train['Fare'], bins=40, kde=True)
plt.title("Fare Distribution of Passengers")
plt.show()



# Correlation heatmap (only numeric columns)
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



# Features + Target
X = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']]
y = train['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test,y_pred))


