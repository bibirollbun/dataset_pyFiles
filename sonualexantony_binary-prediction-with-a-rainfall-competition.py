import pandas as pd
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train = train.drop('day', axis=1)
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train.head()


features = list(train.columns)
target = 'rainfall'
features.remove(target)
print(features)


transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers = [
        ('pre', transformer, features)
    ]
)


X = train[features]
y = train[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


models = {
    'Logistic Regression': LogisticRegression(),
    'Random Foreset': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'LightBGM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(logging_level='Silent'),
    'VotingClassifier': VotingClassifier(estimators=[
        ('Logistic Regression', LogisticRegression()),
        ('GB', GradientBoostingClassifier()),
        ('SVM', SVC(probability=True)),
        ('XGB', XGBClassifier()),
        ('LGBM', LGBMClassifier()),
        ('CatBoost', CatBoostClassifier(logging_level='Silent'))
    ], voting='soft')
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


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X, y)

    y_pred = pipeline.predict(test)
    submission = pd.DataFrame({
        'id': test['id'],
        target : y_pred.ravel()
    })
    submission.to_csv(name[:5]+'_submission.csv', index=False)

