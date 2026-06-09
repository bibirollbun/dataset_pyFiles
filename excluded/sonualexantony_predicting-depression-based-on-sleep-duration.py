#Required Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

#Classifier models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


X = pd.read_csv('/kaggle/input/study-w5p2/trainX.csv')
y = pd.read_csv('/kaggle/input/study-w5p2/trainY.csv')
X.head()


features = list(X.columns)
print(features)


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers =[
        ('num', numeric_transformer, features)
    ]
)


models = {
    'Logistic Regression': LogisticRegression(),
    'SGD Classifier': SGDClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'LightBGM': LGBMClassifier()
}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print detailed results
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)


#Using XGBoost
test_X = pd.read_csv('/kaggle/input/study-w5p2/testX.csv')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier())
])

pipeline.fit(X, y)
y_pred = pipeline.predict(test_X)

submission = pd.DataFrame({
    'id': list(range(1,24)),
    'label': y_pred
})

submission.to_csv('xgb_submission.csv', index=False)


#Using LightGBM
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier())
])

pipeline.fit(X, y)
y_pred = pipeline.predict(test_X)

submission = pd.DataFrame({
    'id': list(range(1,24)),
    'label': y_pred
})

submission.to_csv('lgbm_submission.csv', index=False)

