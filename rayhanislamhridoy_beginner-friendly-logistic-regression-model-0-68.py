import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_sub= pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
train.drop(columns="id",inplace=True)
test.drop(columns="id",inplace=True)


cat_cols= train.select_dtypes(include="object").columns
num_cols=test.select_dtypes(include="number").columns
# Feature and Target
x=train.drop(columns="diagnosed_diabetes")
y=train["diagnosed_diabetes"]
## Preprocessing: One-hot encode categoricals, scale numeric features
preprocess= ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_cols),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
    ]
)



#Logistic Regression Pipeline
model=Pipeline(steps=[
    ('preprocess',preprocess),
    ("logreg",LogisticRegression())
])

#Fit model
model.fit(x,y)
#predict probabilities for ROC-AUC
y_pred= model.predict_proba(test)[:, 1]


# creat csv file for submission

sample_sub["diagnosed_diabetes"]= y_pred
sample_sub.to_csv("submission.csv",index=False)


sample_sub




