import pandas as pd
cols = pd.read_csv("../../data/fraud_detection/train_transaction.csv", nrows=1).columns.tolist()
cols.remove("TransactionID")
train_transaction = pd.read_csv("../../data/fraud_detection/train_transaction.csv", usecols=cols)
print(train_transaction.columns.tolist())
print(f"Len = {len(train_transaction.columns.tolist())}")
# 运行时间1min


from utils import handle_missing_val
missing_ratio = train_transaction.isnull().mean()
valid_columns = missing_ratio[missing_ratio < 0.6].index.tolist() # FIXME 如果某个特征对判断isFraud很关键，那么该方法就需要优化
invalid_columns = missing_ratio[missing_ratio >= 0.6].index.tolist()
new_train_transaction = train_transaction.loc[:, valid_columns]

handle_missing_val(new_train_transaction)




from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
X = new_train_transaction.loc[:, new_train_transaction.columns.difference(["isFraud"])] 
y = new_train_transaction.loc[:, "isFraud"]

selector = SelectKBest(k=5).fit(X, y)
print(selector.get_feature_names_out())
# ['V40' 'V45' 'V51' 'V52' 'V79']
X_train, X_test, y_train, y_test = train_test_split(X.loc[:, selector.get_feature_names_out()], y, test_size=0.2, random_state=0)




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))





invalid_columns.append("isFraud")
invalid_col_df= train_transaction.loc[:, invalid_columns]
# 填补invalid_col_df中的缺失值
for col in invalid_columns:
    if invalid_col_df[col].dtype == 'object':
        invalid_col_df[col].fillna('unknown', inplace=True)
    else:
        invalid_col_df[col].fillna(0, inplace=True)

# 将invalid_col_df中字符串类型的特征变为数值型
for col in invalid_col_df.select_dtypes(include=['object']).columns:
    invalid_col_df[col] = invalid_col_df[col].astype('category').cat.codes


from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

X = invalid_col_df.loc[:, invalid_col_df.columns.difference(["isFraud"])] 
y = invalid_col_df.loc[:, "isFraud"]

selector = SelectKBest(k=5).fit(X, y)
print(selector.get_feature_names_out())


X_train, X_test, y_train, y_test = train_test_split(X.loc[:, selector.get_feature_names_out()], y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))


selected_features = ['isFraud', 'V40', 'V45', 'V51', 'V52', 'V79', 'V200', 'V201', 'V244', 'V246', 'V257']
train_transaction_df = pd.read_csv("../../data/fraud_detection/train_transaction.csv", usecols=selected_features)
handle_missing_val(train_transaction_df)


from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

X = train_transaction_df.loc[:, train_transaction_df.columns.difference(["isFraud"])] 
y = train_transaction_df.loc[:, "isFraud"]

selector = SelectKBest(k="all").fit(X, y)

print(selector.get_feature_names_out())

X_train, X_test, y_train, y_test = train_test_split(X.loc[:, selector.get_feature_names_out()], y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))


from sklearn.metrics import roc_auc_score

y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])

print(auc)


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# 'V200' 'V201' 'V244' 'V246' 'V257' 'V40' 'V45' 'V51' 'V52' 'V79'
over = SMOTE()
under = RandomUnderSampler()
X = X.loc[:, ['V200', 'V201','V244' ,'V246', 'V257', 'V40' ,'V45', 'V51', 'V52', 'V79']]
X_resampled, y_resampled = over.fit_resample(X,y)

X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))



from sklearn.metrics import roc_auc_score

y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])

print(auc)

