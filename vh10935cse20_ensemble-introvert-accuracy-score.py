import pandas as pd
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
org=pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')


train.head(3)


org=(org.rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)


cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']


train = train.merge(org, how='left', on=cols)
test = test.merge(org, how='left', on=cols)


train.head(3)


train.info()


train.dtypes


train.describe().T


train.shape


train.isna().sum()


test.head(3)


test.dtypes


test.shape


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
plt.suptitle('Pair Plot of Selected Features by Personality', y=1.02) 
plt.show()


# Corrected null value handling for categorical columns
numeric=['int64','float64']
cat_dtypes = ['object'] 

for df_iter in [train, test]: 
    for col in df_iter.columns:
        if df_iter[col].isnull().any(): 
            if df_iter[col].dtypes in numeric:
                mean_val = df_iter[col].mean()
                df_iter[col] = df_iter[col].fillna(mean_val)
            elif df_iter[col].dtypes in cat_dtypes:
                mode_val = df_iter[col].mode()[0]
                df_iter[col] = df_iter[col].fillna(mode_val)


train.isna().sum()


test.isna().sum()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cat_cols=['Stage_fear','match_p','Drained_after_socializing']


for i in cat_cols:
    train[i]=le.fit_transform(train[i])
    test[i]=le.transform(test[i])


X=train.drop(columns=['id','Personality'])
y=train['Personality']
X_test=test.drop(columns=['id'])


model=LabelEncoder()
y=model.fit_transform(y)


from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)


categorical_features_names = X.select_dtypes(include='object').columns.tolist()
categorical_features_names


cat=CatBoostClassifier()
lgbm=LGBMClassifier()
rf=RandomForestClassifier()
df=DecisionTreeClassifier()
xgb=XGBClassifier()


cat.fit(X_train,y_train)
lgbm.fit(X_train,y_train)
rf.fit(X_train,y_train)
df.fit(X_train,y_train)
xgb.fit(X_train,y_train)


y_test_cat=cat.predict(X_test)
y_test_lgbm=lgbm.predict(X_test)
y_test_rf=rf.predict(X_test)
y_test_df=df.predict(X_test)
y_test_xgb=xgb.predict(X_test)


raw_preds=np.array([
    y_test_cat,y_test_lgbm,y_test_rf,y_test_df,y_test_xgb
])
raw_trans=raw_preds.T


y_ensembled_num,_=mode(raw_trans, axis=1, keepdims=False)
y_ensembled_num=y_ensembled_num.flatten()


y_labels_test=model.inverse_transform(y_ensembled_num)


submission=test[['id']].copy()
submission['Personality']=y_labels_test
submission.to_csv('submission.csv',index=False)


submission

