import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn import metrics
from xgboost import XGBClassifier

train_dir = "/kaggle/input/playground-series-s5e3/train.csv"
test_csv = "/kaggle/input/playground-series-s5e3/test.csv"
submit_dir = "/kaggle/input/playground-series-s5e3/sample_submission.csv"
from warnings import filterwarnings
filterwarnings('ignore')


df = pd.read_csv(train_dir)
df.head(10)


df.info()


df = df.sample(frac=1, random_state=42).reset_index(drop=True)


X = df.drop(['id','rainfall'],axis=1).values
Y = df.rainfall.values
X_train,X_val,y_train,y_val = train_test_split(X,Y,test_size=0.2,random_state=42)


 model = XGBClassifier(
        device="cuda",
        max_depth=6,  
        colsample_bytree=0.9, 
        subsample=0.9, 
        n_estimators=10_000,  
        learning_rate=0.1, 
        eval_metric="auc",
        early_stopping_rounds=100,
        alpha=1,
    )


model.fit(X_train,y_train,eval_set = [(X_val,y_val)],verbose=100)
model.score(X_val,y_val)


expected_y  = y_val
predicted_y = model.predict(X_val)
print(metrics.classification_report(expected_y, predicted_y))
print(metrics.confusion_matrix(expected_y, predicted_y))


df_test = pd.read_csv(test_csv)
df_test.info()


df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)


Test = df_test.drop("id",axis=1).values
ids = df_test.id.values


predict = model.predict(Test)



sample = pd.read_csv(submit_dir)
sample


submission = pd.DataFrame({
    'id' : ids,
    "rainfall" : predict
})
submission.to_csv("submission.csv",index=False)

