import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head()


test.head()


train.describe()


train.shape, test.shape


train.info()


train.isnull().sum()


train_df = train.rename(columns={'y': 'target'}).copy()
train_df


train.describe().T


train_df['target'].value_counts(normalize=True)

sns.countplot(data=train_df, x='target')
plt.title('Target Variable Distribution')
plt.show()


numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
train[numeric_cols].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()


sns.heatmap(train_df[numeric_cols + ['target']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cat_cols:
    plt.figure(figsize=(8, 3))
    sns.countplot(data=train_df, x=col, hue='target')
    plt.xticks(rotation=45)
    plt.title(f'{col} vs Target')
    plt.tight_layout()
    plt.show()


sns.boxplot(data=train_df, x='target', y='duration')
plt.title('Call Duratioin vs Target')
plt.show()


train_df['pdays'].value_counts().head()

train_df[train_df['pdays'] != -1]['pdays'].hist(bins=50)
plt.title('pdays (Days since last contact)')
plt.show()

sns.histplot(data=train_df, x='previous', hue='target', bins=30)
plt.title('Number of Previous Contacts vs Target')
plt.show()


sns.boxplot(data=train_df, x='target', y='balance')
plt.title('Balance vs Target')
plt.show()

sns.violinplot(data=train_df, x='target', y='age')
plt.title('Age vs Target')
plt.show()


job_duration = train_df.groupby('job')['duration'].mean().sort_values()
job_duration.plot(kind='barh', figsize=(10, 6))
plt.title("Average Call Duration by Job Type")
plt.show()


train_df.drop('id', axis=1, inplace=True)


train_df.head(5)


y = train_df['target']
X = train_df.drop('target', axis=1)
X_test = test.drop(columns=['id'])


y.head(5)


X.head(5)


cat_cols = X.select_dtypes(include=['object']).columns


X.head()


X.columns


test_ids = test['id']


cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()


preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_enc', TargetEncoder())
])


X[cat_cols] = preprocessor.fit_transform(X[cat_cols], y)


X_test[cat_cols] = preprocessor.transform(X_test[cat_cols])


X[num_cols] = X[num_cols].fillna(X[num_cols].median())


X_test[num_cols] = X_test[num_cols].fillna(X[num_cols].median())


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


model.fit(X_train, y_train)


y_val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation ROC AUC Score: {auc:.4f}')


y_test_pred = model.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({'id': test_ids, 'y': y_test_pred})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully!")

