import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test =pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sam = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train.head(3)


train.isna().sum()


X = train.drop(columns=['id','rainfall'])
y = train['rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = GradientBoostingClassifier(n_estimators=10000, learning_rate=0.03,max_depth=1)#0.02 0.7638#, max_depth=80, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
auc = roc_auc_score(y_test, y_pred)
print(f'Area Under the Curve (AUC): {auc:.4f}')


test = test.drop(columns=['id'])
test.isna().sum()


imputer = KNNImputer(n_neighbors=2)
test1 = imputer.fit_transform(test)
test = pd.DataFrame(test1, columns=test.columns)
test.isna().sum()


Predictions = model.predict_proba(test)[:, 1]
sam['rainfall']= Predictions
sam['rainfall'] = sam['rainfall'].round(1)
sam.to_csv('submission.csv',index=False)


sam.head()




