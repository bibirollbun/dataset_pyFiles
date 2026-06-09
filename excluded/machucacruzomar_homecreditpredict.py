import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# loading the csv of the dataset
df = pd.read_csv('/kaggle/input/homecredit/application_train.csv')

#cleaning the dataset by removing the empy data(null)
cleaned_df = df.dropna()

categorical_feats = df.select_dtypes('object').columns.tolist()

#separating them into variables
X = df.drop(columns=['TARGET'])
y = df['TARGET']


!pip install category_encoders


from category_encoders import CountEncoder
#Encodings values
X = CountEncoder(cols=categorical_feats).fit_transform(X)


!pip install lightgbm


# splitting  the data into trainig and testing data using train_test_split from sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# standardizing tha data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_trans = scaler.transform(X_train)
X_test_trans = scaler.transform(X_test)

# fitting the data
from lightgbm import LGBMClassifier

reg = LGBMClassifier(random_state=5).fit(X_train_trans, y_train)

#predicting
reg_pred = reg.predict(X_test_trans)

print('Acc: ', accuracy_score(y_true=y_test, y_pred=reg_pred))
print('ROC: ', roc_auc_score(y_test, reg_pred))


# loading the csv of the test dataset
test_df = pd.read_csv('/kaggle/input/homecredit/application_test.csv')

# cleaning the datasets by removing the empy data(null)
test_cleaned_df = test_df.dropna(axis=0)

# separating them into variables
test_X = CountEncoder(cols=categorical_feats).fit_transform(test_df)

# standaring the data
test_scaler = StandardScaler()
test_X_trans = scaler.fit_transform(test_X)

# predicting
test_reg_pred = reg.predict(test_X_trans)

kgl_submision = pd.concat([test_df['SK_ID_CURR'], pd.Series(test_reg_pred, name='TARGET')], axis=1)
kgl_submision.to_csv('kgl_submission.csv', index=False)


kgl_submision


# cleaning the dataset removing the empy data (null)
cleaned_df = df.dropna()

# separating the intro variables
X = cleaned_df.drop(columns=['TARGET'])
y = cleaned_df['TARGET']
print(X.shape, y.shape)


# imputation
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

#X = df.drop(columns=['TARGET'])
#y = df['TARGET']

# pattern 1
imp_mean = SimpleImputer(strategy='mean')

# Select only numerical columns for mean imputation
X_numerical = X.select_dtypes(include=np.number)

# drop the missing values - apply imputer to numerical data
# Note: This only imputes numerical columns. You will need to handle categoriacal columns separately.
imp_X_numerical = imp_mean.fit_transform(X_numerical)

# spliting the data into trainingand testing data using train_test_split from sklearn
# use the imputed numerical data

from sklearn.preprocessing import OneHotEncoder

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(imp_X_numerical, y, test_size=0.25, random_state=42)

# standarizing the data
scaler1 = StandardScaler()
scaler1.fit(X_train_1)
X_train_trans_1 = scaler1.transform(X_train_1)
X_test_trans_1 = scaler1.transform(X_test_1)

# fitting the data
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(random_state=5)
lgb = lgbm.fit(X_train_trans_1, y_train_1)

# predicing
reg_pred_1 = lgb.predict(X_test_trans_1)

print('Accuracy: ', accuracy_score(y_test_1, reg_pred_1))


# imputation
from sklearn.preprocessing import OneHotEncoder

#X = df.drop(columns=['TARGET'])
#y = df['TARGET']

#separate numerical and categorical columns
X_numerical = X.select_dtypes(include=np.number)
X_categorical = X.select_dtypes(exclude=np.number)

#impute numerical columns using the median strategy
imp_median_numerical = SimpleImputer(strategy='median')
imp_X_numerical = imp_median_numerical.fit_transform(X_numerical)


#Impute  categorical columns using the most frenquent strategy (or another suitable strategy for categorical data)
imp_mf_categorical = SimpleImputer(strategy='most_frequent')
imp_X_categorical = imp_mf_categorical.fit_transform(X_categorical)

# one hot encode the imputed categorical data
enc_1 = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Use sparse_output=False for dense array output
enc_imp_X_categorical = enc_1.fit_transform(imp_X_categorical)

# combine the imputed numerical data and the one-hot encoded categorical data
imp_X_1 =np.hstack((imp_X_numerical, enc_imp_X_categorical))

#splitting the data into training and testing using train_test_Split form sklearn
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(imp_X_1, y, test_size=0.25, random_state=42)

# standarizing the data
scaler2 = StandardScaler()
scaler2.fit(X_train_2)
X_train_trans_2 = scaler2.transform(X_train_2)
X_test_trans_2 = scaler2.transform(X_test_2)

# fitting the data
from lightgbm import LGBMClassifier
lgbm_1 = LGBMClassifier(random_state=5)
lgb_1 = lgbm_1.fit(X_train_trans_2, y_train_2)

# predicting
reg_pred_2 = lgb_1.predict(X_test_trans_2)

print('Accuracy: ', accuracy_score(y_test_2, reg_pred_2))
print(X.shape)


imp_mf = SimpleImputer(strategy='most_frequent')

#drop the missing values
imp_X_2 = imp_mf.fit_transform(X)

# One hot encoding
enc_2 = OneHotEncoder(handle_unknown='ignore')
enc_imp_X_2 = enc_2.fit_transform(imp_X_2).toarray()

# splitting the data into training and testing data
X_train_3, X_test_3 , y_train_3, y_test_3 = train_test_split(enc_imp_X_2, y, test_size=0.25, random_state=42)

#standardizing the data
scaler = StandardScaler()
scaler.fit(X_train_3)
X_train_trans_3 = scaler.transform(X_train_3)
X_test_trans_3 = scaler.transform(X_test_3)

# fitting the data
lgbm_2 = LGBMClassifier(random_state=5)
lgb_2 = lgbm_2.fit(X_train_trans_3, y_train_3)

# predicting
reg_pred_3 = lgb_2.predict(X_test_trans_3)

print(reg_pred_3.shape)
print('Accuracy: ', accuracy_score(y_test_3, reg_pred_3))

