import pandas as pd
selected_features = ['TransactionID', 'isFraud']
selected_features_df =  pd.read_csv("../../data/fraud_detection/train_transaction.csv", usecols=selected_features)

train_identity_df = pd.read_csv("../../data/fraud_detection/train_identity.csv")
merged_df = train_identity_df.merge(selected_features_df, on='TransactionID', how='left')
missing_ratio = merged_df.isnull().mean()
valid_columns = missing_ratio[missing_ratio < 0.6].index.tolist()
invalid_columns = missing_ratio[missing_ratio >= 0.6].index.tolist()
valid_col_df =  merged_df.loc[:, valid_columns]
print(valid_columns)



from utils import handle_missing_val
handle_missing_val(valid_col_df)

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

X = valid_col_df.loc[:, valid_col_df.columns.difference(["isFraud"])] 
y = valid_col_df.loc[:, "isFraud"]

print(X.shape)
selector = SelectKBest(k=5).fit(X, y)
print(selector.get_feature_names_out())
X_train, X_test, y_train, y_test = train_test_split(X.loc[:, selector.get_feature_names_out()], y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))


invalid_columns.append("isFraud")
invalid_col_df= merged_df.loc[:, invalid_columns]
handle_missing_val(invalid_col_df)


from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

X = invalid_col_df.loc[:, invalid_col_df.columns.difference(["isFraud"])] 
y = invalid_col_df.loc[:, "isFraud"]
print(X.shape)
selector = SelectKBest(k=5).fit(X, y)
print(selector.get_feature_names_out())
X_train, X_test, y_train, y_test = train_test_split(X.loc[:, selector.get_feature_names_out()], y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
# X，y就是缺失率>=0.6的数据集

over = SMOTE()
under = RandomUnderSampler()

X_resampled, y_resampled = Pipeline(steps=[("over", over), ("under", under)]).fit_resample(X,y)

selector = SelectKBest(k=5).fit(X_resampled, y_resampled)

key_features = selector.get_feature_names_out()

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled.loc[:, key_features], y_resampled, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
#print(classification_report(y_test, pred))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)


