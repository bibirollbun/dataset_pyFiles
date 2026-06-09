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


# 1. Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


# 2. Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_format = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# 3. Fix column typo
train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
test.rename(columns={'Temparature': 'Temperature'}, inplace=True)


train.head()


test.head()



sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# 3. Fix column typo
# train.rename(columns={'Temparature': 'Temperature'}, inplace=True)

# 4. Initial Checks
print("ğŸ”� Train Shape:", train.shape)
print("ğŸ§ª Test Shape:", test.shape)
print("\nğŸ“Œ Column Data Types:\n", train.dtypes)
print("\nâ�“ Missing Values:\n", train.isnull().sum())


# 5. Target Distribution
plt.figure()
sns.countplot(data=train, x="Fertilizer Name", order=train["Fertilizer Name"].value_counts().index)
plt.title("Target Variable Distribution (Fertilizer Name)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 6. Categorical Features Summary
cat_cols = [col for col in train.columns if train[col].dtype == 'object' and col != 'Fertilizer Name']
for col in cat_cols:
    print(f"\nğŸ§© {col} value counts:\n{train[col].value_counts()}")
    sns.countplot(data=train, x=col, hue="Fertilizer Name", order=train[col].value_counts().index)
    plt.title(f"{col} vs Fertilizer Name")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



# 7. Numerical Features Summary
num_cols = train.select_dtypes(include=np.number).columns.tolist()
num_cols = [col for col in num_cols if col not in ['id']]  # exclude ID

train[num_cols].describe().T.style.background_gradient(cmap="YlGnBu")



# 8. Distribution Plots
for col in num_cols:
    plt.figure()
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()


# 9. Boxplots vs Target
for col in num_cols:
    plt.figure()
    sns.boxplot(x="Fertilizer Name", y=col, data=train)
    plt.title(f"{col} by Fertilizer Name")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 10. Correlation Heatmap
corr_matrix = train[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()


# 11. Pairwise Interactions (Optional: heavy)
# sns.pairplot(train[num_cols + ['Fertilizer Name']], hue='Fertilizer Name', diag_kind='kde')

# 12. Feature Interaction
train['NPK'] = train['Nitrogen'] + train['Phosphorous'] + train['Potassium']
train['Temp_Humidity'] = train['Temperature'] * train['Humidity']

sns.scatterplot(data=train, x="NPK", y="Moisture", hue="Fertilizer Name")
plt.title("NPK vs Moisture")
plt.show()

sns.scatterplot(data=train, x="Temperature", y="Humidity", hue="Fertilizer Name")
plt.title("Temperature vs Humidity")
plt.show()


# 13. Encode categorical columns using OrdinalEncoder
cat_cols = [col for col in train.columns if train[col].dtype == 'object' and col != 'Fertilizer Name']
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = oe.fit_transform(train[cat_cols])
test[cat_cols] = oe.transform(test[cat_cols])


# 14. Add interaction features
train["Soil_Crop"] = train["Soil Type"].astype(int).astype(str) + "_" + train["Crop Type"].astype(int).astype(str)
test["Soil_Crop"] = test["Soil Type"].astype(int).astype(str) + "_" + test["Crop Type"].astype(int).astype(str)

le_interact = LabelEncoder()
train["Soil_Crop"] = le_interact.fit_transform(train["Soil_Crop"])
test["Soil_Crop"] = le_interact.transform(test["Soil_Crop"])

# Numeric interaction features
train['Temp_Humidity'] = train['Temperature'] * train['Humidity']
test['Temp_Humidity'] = test['Temperature'] * test['Humidity']

train['NPK'] = train['Nitrogen'] + train['Phosphorous'] + train['Potassium']
test['NPK'] = test['Nitrogen'] + test['Phosphorous'] + test['Potassium']



# 15. Encode target
target_le = LabelEncoder()
train["Fertilizer Name"] = target_le.fit_transform(train["Fertilizer Name"])

# 16. Features and labels
X = train.drop(columns=["id", "Fertilizer Name"])
y = train["Fertilizer Name"]
X_test = test.drop(columns=["id"])


X_test = X_test[X.columns]



X.head()


X_test.head()


# 17. Define MAP@3
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits = 0.0, 0
        for i, pred in enumerate(p):
            if pred in a:
                hits += 1
                score += hits / (i + 1.0)
        return score / min(len(a), k)
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])


# 18. Stratified K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((X_test.shape[0], y.nunique()))
val_map3_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ§ª Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    
    
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=y.nunique(),
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.7,
        tree_method='gpu_hist',
        eval_metric='mlogloss',
        n_estimators=3000,
        random_state=fold,
        verbosity=0
    )

    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )
    
    val_probs = model.predict_proba(X_val)
    val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]
    map3 = mapk(y_val, val_top3)
    val_map3_scores.append(map3)
    print(f"âœ… Fold {fold + 1} MAP@3: {map3:.5f}")
    
    test_preds += model.predict_proba(X_test) / skf.n_splits

print(f"\nğŸ�¯ Average CV MAP@3: {np.mean(val_map3_scores):.5f}")





from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation
# 18. Stratified K-Fold CV with LightGBM
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((X_test.shape[0], y.nunique()))
val_map3_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ§ª Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(
        objective='multiclass',
        num_class=y.nunique(),
        max_depth=12,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.7,
        n_estimators=5000,
        random_state=fold,
        importance_type='gain'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)],
        # verbose=100
    )
    
    val_probs = model.predict_proba(X_val)
    val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]
    map3 = mapk(y_val, val_top3)
    val_map3_scores.append(map3)
    print(f"âœ… Fold {fold + 1} MAP@3: {map3:.5f}")
    
    test_preds += model.predict_proba(X_test) / skf.n_splits

print(f"\nğŸ�¯ Average CV MAP@3: {np.mean(val_map3_scores):.5f}")



top_3_preds = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission = pd.DataFrame({
    'id': submission_format['id'],
    'Fertilizer Name': [' '.join(map(str, row)) for row in top_3_labels]  # âœ… Fix here
})

submission.to_csv("submission.csv", index=False)
print("âœ… Submission saved as 'submission.csv'")





