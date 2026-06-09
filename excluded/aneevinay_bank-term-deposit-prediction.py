import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



df=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


df.shape


df.head()


df.info()


df.isnull().sum()


df.duplicated().sum()


df['y'].value_counts()


plt.figure(figsize=(12, 6))
sns.countplot(x='y', data=df)
plt.title('Target variable Distribution')
plt.xlabel('target variable Type')
plt.ylabel('Count')
plt.show()
plt.savefig('Target variable distribution.png')


df.nunique()


low_card=['job','marital','education','default','housing','loan','month','contact','poutcome']
high_card = ['balance','duration','pdays','age','day','campaign','previous']


for col in low_card:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for col in high_card:
    plt.figure(figsize=(7,4))
    sns.boxplot(x='y', y=col, data=df)
    plt.title(f"Boxplot of {col} vs y")
    plt.show()


num_cols = df[high_card]
corr = num_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


df.describe()


df['age'] = pd.cut(df['age'],
                         bins=[0, 30, 60, 100],
                         labels=['Youth','midage', 'Senior'])

sns.countplot(x='age', data=df, hue='y')
plt.title("age vs Target (y)")
plt.show()


df['balance'] = pd.cut(df['balance'], bins= [-999999, 0, 3000, 1000000], labels=['low','medium','high'], right=False)


sns.countplot(x='balance', data=df, hue='y')
plt.title("Balance Group vs Target (y)")
plt.show()


df = df.drop(columns=['day'])


ids = df['id']

X = df.drop(columns=['y', 'id'])
y = df['y']


num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include='object').columns



num_pipeline = Pipeline([
    ('scaler',StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num',num_pipeline, num_cols),
    ('cat',cat_pipeline, cat_cols)
])


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1,random_state = 10)


model_pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',RandomForestClassifier(n_estimators=100, random_state=42))
])


model_pipeline.fit(X_train, y_train)


y_pred = model_pipeline.predict(X_test)
accuracy_score(y_test,y_pred)


test_ids = test['id']

X_final_test = test.drop(columns=['id'])

y_preds = model_pipeline.predict(X_final_test)

submission = pd.DataFrame({
    'id': test_ids,
    'y': y_preds
})

submission.to_csv('submission.csv', index=False)




