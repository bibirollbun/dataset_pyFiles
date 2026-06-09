import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.head()


train.isnull().sum()


train.info()


train.describe()


train.corr(numeric_only=True)


sns.set_style('darkgrid')
plt.figure(figsize=(11,6))
sns.countplot(x=train['Sex'], palette='Set1')
plt.title('Distribution Of Sex', fontsize=16)
plt.xlabel('Sex', fontsize=13)
plt.ylabel('Count', fontsize=13)
plt.show()



plt.figure(figsize=(11,6))
sns.scatterplot(x=train['Age'], y=train['Height'], s=100, alpha=0.9, color='steelblue')
plt.title('Age Vs. Height', fontsize=16)
plt.xlabel('Age', fontsize=13)
plt.ylabel('Height', fontsize=13)
plt.show()


plt.figure(figsize=(11,6))
sns.scatterplot(x=train['Height'], y=train['Weight'], hue=train['Sex'], s=100, alpha=0.9, palette='Reds')
plt.title('Height Vs. Weight With Sex', fontsize=16)
plt.xlabel('Height', fontsize=13)
plt.ylabel('Weight', fontsize=13)
plt.legend()
plt.show()


plt.figure(figsize=(11,6))
sns.scatterplot(x=train['Duration'], y=train['Body_Temp'], hue=train['Sex'], s=100, alpha=0.9, palette='mako')
plt.title('Duration Vs. Body Temperature With Sex', fontsize=16)
plt.xlabel('Duration', fontsize=13)
plt.ylabel('Body Temperature', fontsize=13)
plt.legend()
plt.show()


plt.figure(figsize=(11,6))
sns.scatterplot(x=train['Duration'], y=train['Calories'], hue=train['Sex'], s=100, alpha=0.9, palette='Blues')
plt.title('Duration Vs. Calories With Sex', fontsize=16)
plt.xlabel('Duration', fontsize=13)
plt.ylabel('Calories', fontsize=13)
plt.legend()
plt.show()


plt.figure(figsize=(15,20))
for k, cols in enumerate(['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp']):
    plt.subplot(3, 2, k+1)
    sns.kdeplot(x=cols, data=train, color='orange', shade=True)
    plt.title(f"Distribution Of {cols}")
    plt.tight_layout(pad=4.0)

plt.show()


plt.figure(figsize=(11,6))
sns.kdeplot(x=train['Calories'], shade=True)
plt.show()


plt.figure(figsize=(15,20))
for k, cols in enumerate(['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp', 'Calories']):
    plt.subplot(4, 2, k+1)
    sns.boxplot(x=cols, data=train, color='orange')
    plt.title(f"Distribution Of {cols}")
    plt.tight_layout(pad=4.0)

plt.show()


plt.figure(figsize=(11,6))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap='summer')
plt.show()


le  = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.fit_transform(test['Sex'])



X_train = train.drop(["id", "Calories"], axis=1)
y = train["Calories"]

X_test = test.drop(["id"],axis = 1)


random_f = RandomForestRegressor()
random_f.fit(X_train, y)
y_pred_randomforest = random_f.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "Calories": y_pred_randomforest
})

submission.to_csv("submission_RandomForest.csv", index=False)

