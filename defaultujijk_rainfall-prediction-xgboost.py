
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb


train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')  # Update with your actual file path
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')    # Update with your actual file path


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df = pd.DataFrame(train_data)
df.head()


X = df[["id","day","pressure","maxtemp","temparature","mintemp","dewpoint","humidity","cloud","sunshine","winddirection","windspeed"]] 
y = df['rainfall']  # Target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


models = Pipeline(steps=[
    ('standarization', StandardScaler()),
    # ('MinScaler', MinMaxScaler()),
    ('XBoostClassifier', xgb.XGBClassifier(eval_metric="auc",learning_rate=0.1,n_estimators=100,max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=4))
])

models.fit(X_train, y_train)
models


y_pred = models.predict(X_test)
y_pred_probs = models.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_probs)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC Score: {auc_score:.4f}")


evaluate = pd.DataFrame(test_data)

last_models = models.predict(evaluate)


submision = pd.DataFrame({'id': test_data['id'], 'rainfall': last_models})
submision.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'.")

