import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train = train.drop(columns=['id'])


train.info()


target = 'diagnosed_diabetes'
train[target] = train[target].astype(int)


cat_cols = train.select_dtypes(exclude=['number']).columns
num_cols = train.select_dtypes(include=['number']).columns


print(f"Categorcal Columns are:\n{cat_cols}\n")
print(f"Numericsl Columns are:\n{num_cols.to_list}")


train[num_cols].corr()


train.describe().T


train.head()


for col in num_cols:
    print(f"Maximum of {col} is {train[col].max()}")
    print(f"Miinimum of {col} is {train[col].min()}")
    print(f'unique valuse in {col} is: {train[col].unique()}\n')


for col in cat_cols:
    print(f'unique valuse in {col} is: {train[col].unique()}')


from ydata_profiling import ProfileReport

profile = ProfileReport(train, title="EDA Report", explorative=True)
profile


X = train.drop(columns=[target])
y = train[target]


print(f'Shape of X is: {X.shape}')
print(f'Shape of y is: {y.shape}')


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Shape of X train is: {X_train.shape}')
print(f'Shape of y train is: {y_train.shape}')
print(f'Shape of X test is: {X_test.shape}')
print(f'Shape of y test is: {y_test.shape}')


scaler = StandardScaler()

cat_cols = X_train.select_dtypes(exclude=['number']).columns
num_cols = X_train.select_dtypes(include=['number']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)


preprocessor


models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        max_features='sqrt',
        min_samples_leaf=2,
        n_jobs=-1,         
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        tree_method='hist',   
        predictor='gpu_predictor',
        device='cuda',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500,
        learning_rate=0.07,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        random_state=42,
        device_type='gpu'
    )
}


cv_results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring='roc_auc',
        n_jobs=1
    )
    
    cv_results[name] = scores.mean()
    print(f"{name} CV ROC-AUC: {scores.mean():.4f}")


best_model_name = max(cv_results, key=cv_results.get)
best_model_name


best_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', models[best_model_name])
])

best_pipeline.fit(X_train, y_train)


y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


test_predictions = best_pipeline.predict(test)


submission = pd.DataFrame({
    "id": test["id"],
    'diagnosed_diabetes': test_predictions
})

submission.to_csv("submission.csv", index=False)


submission.head()


submission = pd.DataFrame({
    "id": test["id"],
    'diagnosed_diabetes': best_pipeline.predict_proba(test)[:, 1]
})
submission.to_csv("submission_pred.csv", index=False)


submission.head()




