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


import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = "/kaggle/input/santander-customer-transaction-prediction"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train.shape, test.shape
                    


train.head(4)


print("Shape:", train.shape)
print("\nColumns:", train.columns[:10], "...")


#missing value check
train.isnull().sum().sort_values(ascending = False)


train.duplicated().sum()


train['target'].value_counts(normalize = True)


train.describe().T.head()


train.dtypes.value_counts()


#Feature Distribution Visualization
feature = "var_0"
plt.figure(figsize = (10,5))
sns.histplot(train[feature], kde =True,bins = 50)
plt.title('Distribution of {feature}')
plt.show()


#Target vs Feature Distribution
plt.figure(figsize=(10,5))
sns.kdeplot(data=train, x= 'var_0', hue = 'target', common_norm =  False)
plt.title("var_0 distribution for target=0 vs target=1")
plt.show()



#Correlation Heatmap
sample = train.iloc[:, 2:52]  # just 50 features to keep heatmap readable

plt.figure(figsize=(12,8))
sns.heatmap(sample.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap (first 50 features)")
plt.show()


#Outlier Check (Simple)

plt.figure(figsize=(10,5))
sns.boxplot(x=train["var_0"])
plt.title("Outlier check for var_0")
plt.show()


#Train vs Test Distribution Shift
feature = "var_0"

plt.figure(figsize=(10,5))
sns.kdeplot(train[feature], label="train")
sns.kdeplot(test[feature], label="test")
plt.title(f"Train vs Test Distribution for {feature}")
plt.legend()
plt.show()


from sklearn.model_selection import train_test_split

X = train.drop(["target", "ID_code"], axis=1)
y = train["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape, X_val.shape



from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

log_reg_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=200))
])

log_reg_clf.fit(X_train, y_train)

preds = log_reg_clf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, preds)

print("Logistic Regression ROC-AUC:", roc_auc)



from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict_proba(X_val)[:, 1]

lgbm_auc = roc_auc_score(y_val, lgbm_preds)
print("LightGBM ROC-AUC:", lgbm_auc)



#Row Statistical Features

fe_train = X.copy()
fe_test  = test.drop("ID_code", axis=1).copy()

# Row statistics
fe_train["row_mean"] = X.mean(axis=1)
fe_train["row_std"]  = X.std(axis=1)
fe_train["row_min"]  = X.min(axis=1)
fe_train["row_max"]  = X.max(axis=1)
fe_train["row_skew"] = X.skew(axis=1)
fe_train["row_kurt"] = X.kurtosis(axis=1)

fe_test["row_mean"] = fe_test.mean(axis=1)
fe_test["row_std"]  = fe_test.std(axis=1)
fe_test["row_min"]  = fe_test.min(axis=1)
fe_test["row_max"]  = fe_test.max(axis=1)
fe_test["row_skew"] = fe_test.skew(axis=1)
fe_test["row_kurt"] = fe_test.kurtosis(axis=1)

fe_train.head()



#LIGHTGBM WITH ROW STAT FEATURES
#from sklearn.model_selection import train_test_split
#from lightgbm import LGBMClassifier
#from sklearn.metrics import roc_auc_score

y = train["target"]

X_train_fe, X_val_fe, y_train_fe, y_val_fe = train_test_split(
    fe_train, y, test_size=0.2, random_state=42, stratify=y
)

lgbm2 = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

lgbm2.fit(X_train_fe, y_train_fe)
preds_fe = lgbm2.predict_proba(X_val_fe)[:, 1]

auc_fe = roc_auc_score(y_val_fe, preds_fe)
print("LightGBM + Row Features AUC:", auc_fe)



#PCA features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale before PCA
scaler = StandardScaler()
scaled = scaler.fit_transform(X)

pca = PCA(n_components=20, random_state=42)
pca_components = pca.fit_transform(scaled)

pca_df = pd.DataFrame(pca_components, columns=[f"pca_{i}" for i in range(20)])

# Add to feature set
fe_train_pca = pd.concat([fe_train.reset_index(drop=True), pca_df], axis=1)



#retrain with PCA 

X_train_pca, X_val_pca, y_train_pca, y_val_pca = train_test_split(
    fe_train_pca, y, test_size=0.2, random_state=42, stratify=y
)

lgbm3 = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

lgbm3.fit(X_train_pca, y_train_pca)
preds_pca = lgbm3.predict_proba(X_val_pca)[:, 1]

auc_pca = roc_auc_score(y_val_pca, preds_pca)
print("LightGBM + Row Stats + PCA AUC:", auc_pca)



#interaction features
# fe_train["var_0_1"] = X["var_0"] * X["var_1"]
# fe_train["var_10_20"] = X["var_10"] + X["var_20"]
# fe_train["var_50_div_100"] = X["var_50"] / (X["var_100"] + 1)

# fe_train.head()



import matplotlib.pyplot as plt
import pandas as pd

# Get importance
importance = lgbm3.feature_importances_
feature_names = X_train_pca.columns

# Build dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values(by='importance', ascending=False)

# Plot top 30
plt.figure(figsize=(10, 12))
plt.barh(importance_df['feature'][:30], importance_df['importance'][:30])
plt.gca().invert_yaxis()
plt.title("LightGBM Feature Importance (Top 30)")
plt.show()



!pip install shap



import shap

# 1. Create SHAP explainer for LightGBM
explainer = shap.TreeExplainer(lgbm3)

# 2. Compute SHAP values
shap_values = explainer.shap_values(X_val_pca)

# 3. Summary plot (shows top features)
shap.summary_plot(shap_values[1], X_val_pca, feature_names=X_val_pca.columns)








