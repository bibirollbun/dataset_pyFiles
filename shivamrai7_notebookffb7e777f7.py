import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,log_loss
from xgboost import XGBClassifier



train=pd.read_csv("/kaggle/input/mle-ese-mock/train (5).csv")
test=pd.read_csv("/kaggle/input/mle-ese-mock/test (4).csv")


train.isnull().sum()


test.isnull().sum()


test_id=test['id']
test=test.drop(columns=['id'])


train=train.dropna(subset=['quality_grade'])


train.isnull().sum()


X=train.drop(columns=['id','quality_grade'])
y=train['quality_grade']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


numeric_features=X.select_dtypes(include=['int64','float64']).columns
categorical_features=X.select_dtypes(include=['object']).columns


numerical_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])
categorical_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])


preprocessing=ColumnTransformer(transformers=[
    ('num',numerical_pipeline,numeric_features),
    ('cat',categorical_pipeline,categorical_features)
])


from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    loss="log_loss",        # IMPORTANT
    max_iter=100,
    learning_rate=0.05,

    max_depth=6,
    min_samples_leaf=30,

    l2_regularization=0.1,
    max_bins=255,

    random_state=42
)



pipeline=Pipeline(steps=[
    ('preprocessing',preprocessing),
    ('model',model)
])


le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)  # fit on train
y_test_enc = le.transform(y_test)        # transform test


pipeline.fit(X_train,y_train_enc)


y_proba=pipeline.predict_proba(X_test)


loss = log_loss(y_test_enc, y_proba)
print("Log Loss:", loss)


y_final=pipeline.predict_proba(test)


class_names = le.classes_  # use label encoder mapping
submission = pd.DataFrame(y_final, columns=[f"Status_{cls}" for cls in class_names])
submission.insert(0, 'id', test_id)
submission.to_csv("submission4.csv", index=False)
print("\n✅ Submission file created successfully!")
print(submission.head())

