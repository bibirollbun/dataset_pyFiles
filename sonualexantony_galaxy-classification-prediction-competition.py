#Necessary Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report


#Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


X = pd.read_csv('/kaggle/input/study-w5p1/trainX.csv')
y = pd.read_csv('/kaggle/input/study-w5p1/trainY.csv')
X.head()


features = list(X.columns)
preprocessor = ColumnTransformer(
    transformers = [
        ('pre', StandardScaler(), features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(X, y['Category'], test_size=0.2, random_state=3)


models = {
    'Logistic Regression': LogisticRegression(multi_class='multinomial'),
    'Random Foreset': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'LightBGM': LGBMClassifier(),
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


testX = pd.read_csv('/kaggle/input/study-w5p1/testX.csv')
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X, y['Category'])

    y_pred = pipeline.predict(testX)
    submission = pd.DataFrame({
        'Id': list(range(len(y_pred))),
        'Category': y_pred.ravel()
    })
    submission.to_csv(name[:5]+'_submission.csv', index=False)

