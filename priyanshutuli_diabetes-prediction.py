# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%matplotlib inline


train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


pd.set_option('display.max_columns', None)


train_df.head()


train_df.shape


train_df['diagnosed_diabetes'].value_counts(normalize=True) * 100


train_df.isna().mean() * 100


train_df.describe()


train_df.select_dtypes(include=['object']).head()


cat_cols = train_df.select_dtypes(include=['object']).columns


for col in cat_cols:
    print(train_df[col].value_counts(normalize=True) * 100)
    print()


numerical_cols = list(train_df.select_dtypes(include=['float', 'int']).columns)


bool_cols = list(train_df.select_dtypes(include=['bool']).columns)


bool_cols


numerical_cols


numerical_cols.remove('diagnosed_diabetes')


numerical_cols.remove('id')


numerical_cols


import seaborn as sns


for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(train_df[col], kde=True)
    plt.title(f"{col} Distrbution Plot")
    plt.xlabel(f"{col}")
    plt.ylabel("Frequency")
    plt.show()


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(handle_unknown='ignore', drop='first')  # sparse=True returns sparse matrix
encoded = encoder.fit_transform(train_df[cat_cols])


encoded.shape, encoder.get_feature_names_out(cat_cols)


df_encoded = pd.DataFrame.sparse.from_spmatrix(encoded, 
                                               columns=encoder.get_feature_names_out(cat_cols))


df_encoded.head()


X = pd.concat([train_df.drop(columns=list(cat_cols) + ['diagnosed_diabetes']).reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)
Y = train_df['diagnosed_diabetes']


X.head()


X.drop(columns=['id'], inplace=True)


X.shape, train_df.shape


from sklearn.model_selection import train_test_split


X_train,X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
X_val_scaled = scaler.transform(X_val[numerical_cols])


X_train_scaled= pd.DataFrame(X_train_scaled, columns=numerical_cols)
X_val_scaled=pd.DataFrame(X_val_scaled, columns=numerical_cols)


X_train_scaled = pd.concat([X_train_scaled.reset_index(drop=True), X_train[list(df_encoded.columns)].reset_index(drop=True)], axis=1)
X_val_scaled = pd.concat([X_val_scaled.reset_index(drop=True), X_val[list(df_encoded.columns)].reset_index(drop=True)], axis=1)


X_train_scaled.shape, X_val_scaled.shape


X_train_scaled.isna().mean(), X_val_scaled.isna().mean()


X_train_scaled.head()


from statsmodels.stats.outliers_influence import variance_inflation_factor


vif = pd.DataFrame()
vif["features"] = X_train_scaled.columns


vif["VIF"] = [variance_inflation_factor(X_train_scaled.values, i) for i in range(X_train_scaled.shape[1])]


vif


from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(class_weight='balanced', solver='lbfgs',
    max_iter=500, random_state=42, C=1.0)


lr.fit(X_train_scaled, Y_train)


y_train_pred = lr.predict(X_train_scaled)
y_val_pred = lr.predict(X_val_scaled)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print("Accuracy Train:", accuracy_score(Y_train, y_train_pred))
print("Precision Train:", precision_score(Y_train, y_train_pred))
print("Recall Train:", recall_score(Y_train, y_train_pred))
print("F1 Score Train:", f1_score(Y_train, y_train_pred))


print("Accuracy Val:", accuracy_score(Y_val, y_val_pred))
print("Precision Val:", precision_score(Y_val, y_val_pred))
print("Recall Val:", recall_score(Y_val, y_val_pred))
print("F1 Score Val:", f1_score(Y_val, y_val_pred))


print("\nIntercept:\n", lr.intercept_)


print("\nCoefficients corresponding to each feature:\n")

for feat, coef in zip(X_train_scaled.columns, lr.coef_[0]):
    print(f"{feat:<30} : {coef:.4f}")


importance = np.abs(lr.coef_[0])
top_indices = np.argsort(importance)[-2:]
top_features = X.columns[top_indices]


top_features


X2 = X_train[top_features].values
Y2 = Y_train.values


X_train[top_features].head()


X_train['family_history_diabetes'].value_counts()


x_min,x_max = X2[:, 0].min() -1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1


x_min, x_max


y_min, y_max


xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 900),
    np.linspace(y_min, y_max, 900)
)


xx.shape


lr2 = LogisticRegression(class_weight='balanced', solver='lbfgs',
    max_iter=500, random_state=42, C=1.0)
lr2.fit(X2, Y2)


Z = lr2.predict(np.c_[xx.ravel(), yy.ravel()])


Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X2[:,0], X2[:,1], c=Y2, edgecolors='k')
plt.show()


X_test = test_df.drop(columns='id')


test_encoded = encoder.transform(X_test[cat_cols])


test_encoded_df = pd.DataFrame.sparse.from_spmatrix(test_encoded, 
                                               columns=encoder.get_feature_names_out(cat_cols))


X_test = pd.concat([X_test.drop(columns=cat_cols), test_encoded_df], axis=1)


X_test_scaled = scaler.transform(X_test[numerical_cols])


X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_cols)


X_test_scaled = pd.concat([X_test_scaled_df, test_encoded_df], axis=1)


y_test_pred = lr.predict(X_test_scaled)


test_df.loc[:,'diagnosed_diabetes'] = y_test_pred.astype(int)


submission_df = test_df[['id', 'diagnosed_diabetes']]


submission_df.to_csv("/kaggle/working/submission.csv", index=False)




