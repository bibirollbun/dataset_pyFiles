import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head(3)


train.info()


train.describe().T


train.dtypes


train.isna().sum()


test.head(2)


test.shape


test.info()


test.describe().T


test.isna().sum()


sns.countplot(x='Personality',data=train,palette='Set3')
plt.title('Distribution of Personality Types')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()



#Distribution of Numeric Values
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1) 
    sns.histplot(train[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=train[col], color='lightgreen')
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()


#Relationship between Personality
plt.figure(figsize=(18, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='Personality', y=col, data=train, palette='coolwarm')
    plt.title(f'{col} by Personality')
    plt.xlabel('Personality')
    plt.ylabel(col)
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


#correlation between Numeric values
corre_mat=train[numerical_cols].corr()
plt.figure(figsize=(8, 7))
sns.heatmap(corre_mat, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


cat_cols=['Stage_fear','Drained_after_socializing']
plt.figure(figsize=(10, 5))
for i, col in enumerate(cat_cols):
    plt.subplot(1, 2, i + 1) # Adjust subplot grid
    sns.countplot(x=col, data=train, palette='cividis')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

for col in cat_cols:
    print(f"\nCounts for {col}:\n", train[col].value_counts())


cols=['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 'Personality']
sns.pairplot(train[cols], hue='Personality', diag_kind='kde')
plt.suptitle('Pair Plot of Selected Features by Personality', y=1.02) # Adjust title position
plt.show()


numeric=['int64','float64']
cat=['object']
for i in train:
    if (train[i].dtypes in numeric) and (train[i].isna().sum()!=0):
        mean_num=train[i].mean()
        train[i]=train[i].fillna(mean_num)
    elif (train[i].dtypes in cat) and (train[i].isna().sum()!=0):
        mode_cat=train[i].mode()[0]
        train[i]=train[i].fillna(mode_cat)
    else:
        print("Column doesn't match the above criteria",i)


print(f"Total Null values on Train\n",train.isna().sum())


numeric=['int64','float64']
cat=['object']
for i in test:
    if (test[i].dtypes in numeric) and (test[i].isna().sum()!=0):
        mean_num=test[i].mean()
        test[i]=test[i].fillna(mean_num)
    elif (test[i].dtypes in cat) and (test[i].isna().sum()!=0):
        mode_cat=test[i].mode()[0]
        test[i]=test[i].fillna(mode_cat)
    else:
        print("Column doesn't match the above criteria",i)


print(f"Total Null values on Test\n",test.isna().sum())


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cat_cols=['Stage_fear','Drained_after_socializing']


for i in cat_cols:
    train[i]=le.fit_transform(train[i])
    test[i]=le.transform(test[i])


X=train.drop(columns=['id','Personality'])
y=train['Personality']

X_test=test.drop(columns=['id'])


model=LabelEncoder()
y=model.fit_transform(y)


from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)


cat=CatBoostClassifier( random_state=42,
        verbose=0)


cat.fit(X_train,y_train)


y_pred=cat.predict(X_val)


accuracy_score(y_val,y_pred)


y_test=cat.predict(X_test)


y_test_labels = model.inverse_transform(y_test)

submission = test[['id']].copy()
submission['Personality'] = y_test_labels

submission.to_csv('submission.csv', index=False)


submission

