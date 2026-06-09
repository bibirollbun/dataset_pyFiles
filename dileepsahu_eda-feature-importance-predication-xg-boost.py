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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv') 
submission= pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


df = df.drop('id', axis=1)


eda = df.copy()
eda.head()



eda.info()


eda.describe()


## Checking missing values
eda.isnull().sum()

# Target variable Distribution
sns.countplot(x=eda['diagnosed_diabetes'])
plt.title('Diabetes Distribution')
plt.show()


plt.figure(figsize=(14, 8))
sns.heatmap(eda.select_dtypes(include=np.number).corr(), cmap='coolwarm',annot=False)
plt.title('Correlation heatmap')
plt.show()


numeric_features = df.select_dtypes(include=np.number).columns.tolist()
categorical_features = df.select_dtypes(include="object").columns.tolist()



# Remove target variable
numeric_features.remove("diagnosed_diabetes") 

preprocessor = ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)



X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]
x_processed= preprocessor.fit_transform(X)
x_test_processed = preprocessor.transform(df_test)



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)


rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))


# Extract feature names after One-Hot Encoding
ohe = rf_model.named_steps["preprocess"].named_transformers_["cat"]
ohe_features = ohe.get_feature_names_out(categorical_features)

all_features = np.concatenate([ohe_features, numeric_features])

# Get importance from model
importances = rf_model.named_steps["model"].feature_importances_
fi = pd.DataFrame({"feature": all_features, "importance": importances})
fi = fi.sort_values("importance", ascending=False)

fi.head(20)


nfolds = 10

folds = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state=42)

off_preds = np.zeros(X.shape[0])
test_preds = np.zeros(x_test_processed.shape[0])


scores = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"Fold {fold_+1}/{nfolds}")
    
    x_train_fold, x_val_fold = x_processed[trn_idx], x_processed[val_idx]
    y_train_fold, y_val_fold = y.iloc[trn_idx], y.iloc[val_idx]

    #model = lgb.LGBMClassifier(**params)
    model = xgb.XGBClassifier()
    
    model.fit(
        x_train_fold, y_train_fold,
        eval_set=[(x_val_fold, y_val_fold)],
        eval_metric='auc',
        #callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    off_preds[val_idx] = model.predict_proba(x_val_fold)[:, 1]
    
    fold_test_pred = model.predict_proba(x_test_processed)[:, 1]
    test_preds += fold_test_pred / nfolds
    
    fold_auc = roc_auc_score(y_val_fold, off_preds[val_idx])
    scores.append(fold_auc)

print(f"\nModel Training Complete. Average Cross-Validation AUC: **{np.mean(scores):.5f}**")


# n_estimators=400,
# max_depth=6,
# learning_rate=0.05,
# subsample=0.8,
# colsample_bytree=0.8,
# eval_metric="logloss",
# random_state=42




submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': test_preds   
})


submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")


!kaggle competitions submit -c playground-series-s5e12 -f './submission.csv' -m "Message"


!pwd




