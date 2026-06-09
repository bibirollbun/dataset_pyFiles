# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_t=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df.sample(5)


df['poutcome'].unique()


df['job'].unique()


df.info()
df.describe()


import matplotlib.pyplot as plt
import seaborn as sns


num_cols=df.select_dtypes(include=['int64']).columns.tolist()


corr_matrix=df[num_cols].corr()
np.fill_diagonal(corr_matrix.values, 0)
sns.heatmap(corr_matrix,annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


#unique count
uc=df.nunique()
print(uc)



X=df.drop(["id","y"],axis=1)
y=df['y']



cat_col=X.select_dtypes(include=['object']).columns.tolist()
num_cols=X.select_dtypes(include=['int64']).columns.tolist()
cat_col.remove('education')


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder

ohe=OneHotEncoder(drop='first',sparse=False)

encode=ColumnTransformer(
    transformers=[
        ('cat', ohe, cat_col),
        ('num', 'passthrough', num_cols),
        ('edu',OrdinalEncoder(), ['education'])
    ]
)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)


X_train=encode.fit_transform(X_train)
X_train=pd.DataFrame(
    X_train,
    columns = encode.get_feature_names_out()
)
X_train.keys()


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

model = LGBMClassifier(
    objective='binary', 
    metric='auc', 
    n_estimators=1000, 
    learning_rate=0.05,
    n_jobs=-1,   
    random_state=42
)
from lightgbm import early_stopping
model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[early_stopping(stopping_rounds=50, verbose=True)]
)




x_test_trans=encode.transform(X_test)
x_test_trans=pd.DataFrame(
    x_test_trans,
    columns = encode.get_feature_names_out()
)
x_test_trans.keys()


from sklearn.metrics import roc_auc_score, roc_curve
y_proba = model.predict_proba(x_test_trans)[:,1]
print("ROC-AUC:",roc_auc_score(y_test, y_proba))
fpr,tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],linestyle='--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


#fitting on entire dataset
xt=encode.fit_transform(X)
X=pd.DataFrame(
    xt,
    columns=encode.get_feature_names_out()
)
model.fit(X,y)


test_df=encode.transform(df_t)
test_df=pd.DataFrame(
    test_df,
    columns = encode.get_feature_names_out()
)


y_pred_test=model.predict_proba(test_df)[:,1]
submission = pd.DataFrame({
    "id":range(750000,750000+len(y_pred_test)),
    "y":y_pred_test
})
submission.to_csv("submission.csv", index=False)


submission

