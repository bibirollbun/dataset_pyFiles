import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


train = pd.read_csv("/kaggle/input/mle-ese-mock/train (5).csv")
test = pd.read_csv("/kaggle/input/mle-ese-mock/test (4).csv")


train.sample(5)


train.isnull().mean()


train["quality_grade"].mode()


target_imputer = SimpleImputer(strategy="most_frequent")
train["quality_grade"] = target_imputer.fit_transform(
    train[["quality_grade"]]
).ravel()



train.isnull().sum()


test.sample(5)


test.isnull().sum()


x_train = train.drop(columns=['quality_grade'])
y_train = train['quality_grade']


x_train = x_train.drop(columns=['id'])
y_train = y_train.drop(columns=['id'])


x_train.head(4)


class_names = ["Q1_premium_fresh","Q2_fresh","Q3_export_grade","Q4_dessert","Q5_juice_high",
    "Q6_juice_low","Q7_processing", "Q8_local_sale", "Q9_feed","Q10_waste"]



le = LabelEncoder()
le.fit(class_names)     
y_encoded = le.transform(y_train)


cat_cols = x_train.select_dtypes(include="object").columns
num_cols = x_train.select_dtypes(exclude="object").columns


preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_cols),
        ('cat',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),cat_cols)   
])


Rand_f_cls = Pipeline(
    steps=[
        ('preprocessing', preprocessor),
        ('RandomForestClassifier',RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1))
])


Rand_f_cls.fit(x_train, y_train)


test_prob = Rand_f_cls.predict_proba(test)


sample = pd.read_csv("/kaggle/input/mle-ese-mock/submission (6).csv")
submission = sample.copy()

submission.iloc[:, 1:] = test_prob

submission.to_csv("/kaggle/working/submission (6).csv", index=False)





