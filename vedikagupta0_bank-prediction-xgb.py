!pip uninstall -y scikit-learn imbalanced-learn category_encoders
!pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1 category_encoders==2.6.0
import math
import numpy as np 
import pandas as pd 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
tf = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')


df.head()


df.info()


# Identify column types
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns


# Histograms for Numeric
n = len(num_cols)
cols = 3  
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Barplots for Categorical
n = len(cat_cols)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    df[col].value_counts().plot(kind="bar", ax=axes[i], color="skyblue", edgecolor="black")
    axes[i].set_title(f"Counts of {col}")

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42, )
X = df.drop(columns=['y'])  
y = df['y']

X_df,y_df = ros.fit_resample(X, y)


X_df.shape, X.shape


# Target encode categorical columns in df
cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
encoder = TargetEncoder(cols=cat_cols)
X_df_enc = encoder.fit_transform(X_df, y_df)
tf1 = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
tfz = tf1.drop(columns='id')
X_tf_enc = encoder.transform(tfz)


X_train, X_test, y_train, y_test = train_test_split(
    X_df_enc, y_df, test_size=0.1, random_state=42, stratify=y_df
)

xgb = XGBClassifier(
    n_estimators=300,   
    learning_rate=0.115,     
    max_depth=80,           
    subsample=1.0,            
    colsample_bytree=0.5,     
    gamma=0.0,               
    reg_alpha=0.001,            
    reg_lambda=0.0084,           
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train
xgb.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb.predict(X_test)
y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]


# Evaluation
acc_xgb = accuracy_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print("\n--- XGBoost ---")
print(f"Accuracy: {acc_xgb:.4f}")
print(f"ROC AUC: {roc_auc_xgb:.4f}")
print(classification_report(y_test, y_pred_xgb))




# Predictions
y_pred_proba = xgb.predict_proba(X_tf_enc)[:, 1]
submission = pd.DataFrame({
    "id": tf1["id"],
    "y": y_pred_proba
})
submission.to_csv("submission.csv", index=False, float_format="%.6f")

