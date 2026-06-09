from ydata_profiling import ProfileReport

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer, FunctionTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import optuna

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

import pickle

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df.sample(10)


df.shape


df.describe(include = 'all')


df.info()


df.isnull().sum().sort_values(ascending = False)


df.nunique()


df.duplicated().sum()


num_cols = df.select_dtypes(exclude = ['object']).columns.tolist()
cat_cols = df.select_dtypes(include = ['object']).columns.tolist()

print("Numerical Variables are ", num_cols)
print("Categorical Variables are ", cat_cols)


profile = ProfileReport(df, title = "Bank Subscription Report", explorative = True)
profile.to_file("bank_subscription_report.html")


sns.countplot(data = df, x = 'y')


for col in num_cols[1:-1]:
    plt.figure(figsize = (10, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(x = col, data = df, kde = True, bins = 30, hue = 'y')
    plt.title(f'{col} Distribution by y')

    plt.subplot(1, 2, 2)
    sns.boxplot(x = 'y', y = col, data = df)
    plt.title(f'Boxplot for {col} by y')

    plt.tight_layout()
    plt.show()


for col in cat_cols:
    plt.figure(figsize = (10, 6))
    plt.show()
    sns.countplot(data = df, x = col, hue = 'y')
    plt.title(f'{col} by y')
    plt.xticks(rotation = 30)
    plt.tight_layout()


class CapOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, columns, multiplier = 1.5):
        self.columns = columns
        self.multiplier = multiplier

    def fit(self, X, y = None):
        self.bounds = {}
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.multiplier * IQR
            upper = Q3 + self.multiplier * IQR
            self.bounds[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds.items():
            X[col] = X[col].clip(lower = lower, upper = upper)
        return X


power_transformer_pipeline = Pipeline([
    ('power_transformer', PowerTransformer())
])

standard_scaler_pipeline = Pipeline([
    ('standard_scaler', StandardScaler())
])

ordinal_encoder_pipeline = Pipeline([
    ('ordinal_encoder', OrdinalEncoder()),
    ('scaler', StandardScaler())
])

one_hot_encoder_pipeline = Pipeline([
    ('one_hot_encoder', OneHotEncoder(drop = 'first'))
])


preprocessor = ColumnTransformer(transformers = [
    ('ordinal_encoder', ordinal_encoder_pipeline, ['marital', 'education', 'month']),
    ('one_hot_encoder', one_hot_encoder_pipeline, ['job', 'default', 'housing', 'loan', 'contact', 'poutcome']),
    ('passthrough', 'passthrough', num_cols[1:-1])
], remainder = 'drop')


X = df.drop('y', axis = 1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



from catboost import CatBoostClassifier

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(
    verbose=0,
    random_seed=42,
    thread_count=-1,
    loss_function='Logloss',
    task_type='GPU',
    devices='0'  
))
])


set_config(display = 'diagram')
pipeline


scores = cross_val_score(pipeline, X, y, cv = 5, scoring = 'roc_auc')
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")


with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved successfully.")


pipeline.fit(X_train, y_train)

df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
y_pred_probs = pipeline.predict_proba(df_test)[:, 1]

submission = pd.DataFrame({
    'id': df_test['id'],
    'y': y_pred_probs
})

submission.to_csv('submission_cat.csv', index = False)
print("Submission file 'submission_cat.csv' created successfully.")

