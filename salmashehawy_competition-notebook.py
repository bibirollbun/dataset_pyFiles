import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score


train =pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train.isnull().sum()


train.duplicated().sum()


print("train data shape: ",train.shape)
print("test data shape: ",test.shape)
train.head()


x=train.drop(columns=["id","y"])
y=train["y"]
x_test=test.drop(columns=["id"])


x["contact"].unique()


ohe_cols=["marital"]
ordinal_cols=["job","education","default","housing","loan","month","poutcome","contact"]
numerical_cols=x.select_dtypes(include=["int64","float64"]).columns.tolist()


preprocessor=ColumnTransformer(transformers=[
    ("onehot",OneHotEncoder(handle_unknown="ignore",sparse=False),ohe_cols),
    ("ordinal",OrdinalEncoder(),ordinal_cols),
    ("num","passthrough",numerical_cols)
])


x_encoded=preprocessor.fit_transform(x)
x_test_encoded=preprocessor.transform(x_test)










