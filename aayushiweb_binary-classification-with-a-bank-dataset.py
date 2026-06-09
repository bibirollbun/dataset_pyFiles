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


train= pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train.head()


test= pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

test.head()


train.shape


test.shape


concat=pd.concat([train,test],axis=0)
concat.shape


numeric_cols = concat.select_dtypes(include=['int64', 'float64'])
print(numeric_cols)


cat_df = concat.select_dtypes(include='object')
print(cat_df)


n = concat.nunique()
na = concat.isna().sum()
print(n)
print(na)


print(train.isna().sum())
print()
print(test.isna().sum())


for df in [train, test]:
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_squared'] = df['duration'] ** 2
    df['duration_log'] = np.log1p(df['duration'])
    df['duration_sqrt'] = np.sqrt(df['duration'])


train.head()


X = train.drop(['id', 'y'], axis=1)
y = train['y']

# Dropping id from test dataset
test.drop(['id'], axis=1, inplace=True)


# printing the columns of 'object' datatype
object_cols = X.select_dtypes(include="object").columns.tolist()
print(f"The object columns are: \n{object_cols}")


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


encoder= LabelEncoder()
for obj in object_cols:
    X[obj]=encoder.fit_transform(X[obj])
    test[obj]=encoder.fit_transform(test[obj])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


target_variance = 0.99
pca = PCA(target_variance)
principalComponents = pca.fit(X_scaled)
print(f"The number of components to achieve {target_variance} variance is \
{principalComponents.n_components_}")


plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title(f'PCA: {principalComponents.n_components_} \
Components to Explain {target_variance:.0%} Variance')
plt.axhline(y=target_variance, color='r', linestyle='--')
plt.axvline(x=principalComponents.n_components_, color='g', linestyle='--',
            label=f'n_components={principalComponents.n_components_}')
plt.grid(True)
plt.legend()
plt.show()


mi_scores = mutual_info_classif(X_scaled, y, random_state=42)


mi_series = pd.Series(mi_scores, index=X.columns)
mi_series = mi_series.sort_values(ascending=True)

print(mi_series)


mi_series.plot(kind='barh', figsize=(10, 6), color='green')
plt.title('Mutual Information Scores')
plt.ylabel('MI Score')
plt.xlabel('Features')
plt.tight_layout()
plt.show()


X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df.head()


test_scaled_df = pd.DataFrame(test_scaled, columns=test.columns)
test_scaled_df.head()


import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold

def train_lightgbm(train, test, target):
    X = train
    y = target
    
    X_test = test.copy()
    
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_probs = np.zeros(len(X_test))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n<== Training fold {fold + 1}/{n_splits} ==>")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=30000,
            class_weight='balanced',
            learning_rate=0.055,
            num_leaves=100,
            max_depth=10,
            min_child_samples=8,
            subsample=0.85,
            colsample_bytree=0.5,
            reg_alpha=0.8,
            reg_lambda=0.3,
            max_bin=4851,
            random_state=2003,
            verbosity=-1,
            boosting_type='gbdt'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(300),
                lgb.log_evaluation(500)
            ]
        )
        
        models.append(model)
        y_probs += model.predict_proba(X_test)[:, 1] / n_splits
    
    print("\n✅ LightGBM model training complete.")
    return y_probs, models



y_probs, models = train_lightgbm(X_scaled_df, test_scaled_df, y)


sub_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

submission = pd.DataFrame({
    'id': sub_df['id'],
    'target': y_probs 
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




