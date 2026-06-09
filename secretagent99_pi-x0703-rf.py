# Notebook contains EDA+MODEL Development
# Install Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import xgboost as xgb


# Input data files are available in the read-only "../input/" directory
import warnings
warnings.filterwarnings('ignore')


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data = "/kaggle/input/playground-series-s5e7/train.csv"
test_data = "/kaggle/input/playground-series-s5e7/test.csv"


df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)


print(f"|-----------TRAIN DATA----------| and its shape is --| {df_train.shape}")
df_train.head()


print(f"|--------TEST DATA--------| and its shape is --| {df_test.shape}")
df_test.head()


# Display basic info
print("Train Data Info:")
print(df_train.info())
print("\nTest Data Info:")
print(df_test.info())

# Check missing values
print("\nTrain Missing Values:")
print(df_train.isnull().sum())
print("\nTest Missing Values:")
print(df_test.isnull().sum())

# Check target distribution
print("\nPersonality Distribution:")
print(df_train['Personality'].value_counts())


print("Meta Description about Train Data:\n")
df_train.describe()


print("Meta Description about Test Data:\n")
df_test.describe()


# Set style
sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(x='Personality', data=df_train)
plt.title('Personality Distribution')
plt.show()


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


cat_cols = ['Stage_fear', 'Drained_after_socializing']

plt.figure(figsize=(12, 5))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, 2, i)
    sns.countplot(x=col, hue='Personality', data=df_train)
    plt.title(f'{col} by Personality')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
corr = df_train[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


def create_features(df):
    df['Social_Exhaustion'] = (df['Drained_after_socializing'] == 'Yes').astype(int) * df['Social_event_attendance']
    df['Recovery_Need'] = df['Time_spent_Alone'] / (df['Friends_circle_size'] + 0.1)
    df['Online_Offline_Ratio'] = (df['Post_frequency'] + 1) / (df['Going_outside'] + 1)
    df['Social_Consistency'] = df['Social_event_attendance'] - df['Social_event_attendance'].mean()
    df['Engagement_Capacity'] = df['Friends_circle_size'] * (
        1 - df['Stage_fear'].map({'High': 0.8, 'Medium': 0.5, 'Low': 0.2})
    )
    
    df['Alone_log'] = np.log1p(df['Time_spent_Alone'])
    df['Post_log'] = np.log1p(df['Post_frequency'])
    df['Alone_per_event'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    df['Friend_Post_Density'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1)
    df['Outside_Engagement'] = df['Going_outside'] * df['Social_event_attendance']
    
    return df


train = create_features(df_train)
test = create_features(df_test)


for col in ['Time_spent_Alone', 'Friends_circle_size', 'Post_frequency']:
    train[col] = np.where(train[col] > train[col].quantile(0.99), 
                         train[col].quantile(0.99), train[col])
    test[col] = np.where(test[col] > test[col].quantile(0.99),
                        test[col].quantile(0.99), test[col])


cat_cols = ['Stage_fear', 'Drained_after_socializing']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = encoder.fit_transform(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])


X = train.drop(['id', 'Personality'], axis=1)
y = (train['Personality'] == 'Extrovert').astype(int)
test_data = test.drop('id', axis=1)


model=HistGradientBoostingClassifier(
    max_iter=2000,           
    learning_rate=0.01,     
    max_depth=8,              
    min_samples_leaf=20,      
    l2_regularization=2.0,   
    categorical_features=[X.columns.get_loc(c) for c in cat_cols],
    early_stopping=True,
    random_state=42,
    scoring='f1'            
)


oof_preds = cross_val_predict(model, X, y, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
thresholds = np.linspace(0.25, 0.75, 101)
scores = [f1_score(y, (oof_preds > t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(scores)]


model.fit(X, y)
test_probs = model.predict_proba(test_data)[:, 1]
test_preds = (test_probs > best_threshold).astype(int)


submission = pd.DataFrame({
    'id': test['id'],
    'Personality': ['Extrovert' if p == 1 else 'Introvert' for p in test_preds]
})

print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"OOF F1-Score: {max(scores):.5f}")
submission.to_csv('submission.csv', index=False)




