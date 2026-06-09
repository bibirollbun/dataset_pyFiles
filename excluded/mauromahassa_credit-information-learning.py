import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import seaborn as sns
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def Encoder(df):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df



# Problem 2: Learning and verification
# ------------------------------------

# data preparation
application_train = pd.read_csv('application_train.csv')
application_test = pd.read_csv('application_test.csv')
print("Number of samples in application_train:{}".format(len(application_train)))
print("Number of features in application_train:{}".format(len(application_train.columns)))
print("Number of samples in application_test:{}".format(len(application_test)))
print("Number of features in application_test:{}".format(len(application_test.columns)))

print(application_test.columns)




application_train.head()


application_test.head()


application_train.dropna(axis = 1, how='any', inplace=True)
y_train = application_train['TARGET']
application_train = application_train.drop("TARGET", axis=1)
print("after drop features with NA or null values from application_train:{}".format(len(application_train)))
print("features used in train:{}".format(application_train.columns))
print('Feature data types:{}'.format(application_train.dtypes))

application_test = application_test[application_train.columns]
print(application_train.head(10))



plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');




application_train = Encoder(application_train)
application_test = Encoder(application_test)

application_train = pd.get_dummies(application_train)
application_test = pd.get_dummies(application_test)
# training 
scaler = StandardScaler()
scaler.fit(application_train)
X_train = scaler.transform(application_train)
X_test = scaler.transform(application_test)

x_tr, x_val, y_tr, y_val = train_test_split(X_train, y_train, random_state=123)

lr = LogisticRegression(C=0.0001)
lr.fit(x_tr, y_tr)




# Problem 3: Estimate for test data
# ---------------------------------

y_test_pred = lr.predict_proba(x_val)[:, 1]
print("roc auc score", roc_auc_score(y_val, y_test_pred))

y_test_pred = lr.predict_proba(X_test)[:, 1]
data = application_test[["SK_ID_CURR"]].copy()
data['TARGET'] = y_test_pred.tolist()

data.to_csv('submission.csv', index=False)
print(data)





# Problem 4: Feature engineering
# ------------------------------

application_train = pd.read_csv('application_train.csv')
application_test = pd.read_csv('application_test.csv')
print("Number of samples in application_train:{}".format(len(application_train)))
print("Number of features in application_train:{}".format(len(application_train.columns)))
print("Number of samples in application_test:{}".format(len(application_test)))
print("Number of features in application_test:{}".format(len(application_test.columns)))


encoder_application_train=Encoder(application_train)
correlations = encoder_application_train.corr()["TARGET"].sort_values()

print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))



ext_data = application_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
print(ext_data_corrs)


y_ext = ext_data['TARGET']
X_train = ext_data.drop(columns=['TARGET'])
X_test = application_test[X_train.columns]

print(X_train.columns)




imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)

scaler = StandardScaler()
scaler.fit(X_train)
train = scaler.transform(X_train)
test = scaler.transform(X_test)


x_tr, x_val, y_tr, y_val = train_test_split(train, y_ext, random_state=123)

log_reg = LogisticRegression(C=0.0001)
log_reg.fit(x_tr, y_tr)

log_reg_pred = log_reg.predict_proba(x_val)[:, 1]
print("roc auc score", roc_auc_score(y_val, log_reg_pred))




y_test_pred = log_reg.predict_proba(X_test)[:, 1]

data = application_test[["SK_ID_CURR"]].copy()
data['TARGET'] = y_test_pred.tolist()


data.to_csv('submission.csv', index=False)
print(data)


