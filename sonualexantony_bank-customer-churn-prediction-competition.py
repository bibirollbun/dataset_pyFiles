#Necessary Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Prprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

#Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


df = pd.read_csv('/kaggle/input/testtestbambooddd/train.csv')
df.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)
df.head()


#Feature extraction
numerical_features = []
categorical_features = []
features = []
target = 'Exited'

for name in df.columns:
    if df[name].dtype == 'object':
        categorical_features.append(name)
    else:
        numerical_features.append(name)
numerical_features.remove(target)
features = numerical_features + categorical_features
print(features)


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier()
}


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)


test_df = pd.read_csv('/kaggle/input/testtestbambooddd/test.csv')
X_test = test_df[features]
test_df.head()


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X_test)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'Exited': y_pred.ravel()
    })
    submission.to_csv(name[:5]+'_submission.csv', index=False)

