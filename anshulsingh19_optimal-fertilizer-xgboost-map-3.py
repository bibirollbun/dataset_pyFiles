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


# Ignore any future warnings -

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# training datsset -
trn_dset = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")

# testing dataset -
tst_dset = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


trn_dset.head(3)


tst_dset.head(3)


trn_dset.info()


# description of categorical columns -

trn_dset.describe(include = 'object')


# checking for duplicate values -

trn_dset.duplicated().sum()


# checking for null values -

print(trn_dset.isna().sum().sum())

print(tst_dset.isna().sum().sum())


# Separating numeric and categorical columns -

num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
cat_cols = ['Soil Type', 'Crop Type']
target_col = 'Fertilizer Name'


# 1. Soil Type -

print(trn_dset['Soil Type'].unique())


trn_dset['Soil Type'].value_counts().to_frame().T


# Count-plot of Soil Type -

plt.figure(figsize =(12, 5))

plot = sns.countplot(data=trn_dset, x='Soil Type', palette='Set2')
for container in plot.containers:
    plot.bar_label(container, fmt='%d', label_type='center', color='black')
    
plt.title("Count-plot of Soil Type")
plt.xlabel('Soil Type')
plt.ylabel('Count')

plt.show()


# 2. Crop Type -

print(trn_dset['Crop Type'].unique())


trn_dset['Crop Type'].value_counts().to_frame().T


# Count-plot of Crop Type -

plt.figure(figsize=(12, 5))

plot = sns.countplot(data=trn_dset, x='Crop Type', palette='Set2')
for container in plot.containers:
    plot.bar_label(container, fmt='%d', label_type='center', color='black')

plt.title("Count-plot of Crop Type")
plt.xlabel("Crop Type")
plt.ylabel("Count")

plt.show()


# Fertilizer name -

print(trn_dset['Fertilizer Name'].unique())


trn_dset['Fertilizer Name'].value_counts()


# Count-plot of Fertilizer Name -

plt.figure(figsize=(12, 5))

plot = sns.countplot(data=trn_dset, x='Fertilizer Name', palette='Set2')
for container in plot.containers:
    plot.bar_label(container, fmt='%d', label_type='center', color='black')
    
plt.title("Count-plot of Fertilizer Name")
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")

plt.show()


# Comparing Fertilizer name for each type of Soil -

soil_types = trn_dset['Soil Type'].unique()
n = len(soil_types)

fig, axes = plt.subplots(nrows=(n + 1) // 2, ncols=2, figsize=(16, n * 3))

for ax, soil in zip(axes.flat, soil_types):
    filtered = trn_dset[trn_dset['Soil Type'] == soil]
    plot = sns.countplot(data=filtered, x='Fertilizer Name', palette='Set2', ax=ax)

    for container in plot.containers:
        plot.bar_label(container, fmt='%d', label_type='edge', color='black')

    ax.set_title(f"{soil} Soil", fontsize=12)
    ax.set_xlabel("Fertilizer")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Comparing Fertilizer name for each type of Soil -

crop_types = trn_dset['Crop Type'].unique()
n = len(crop_types)

fig, axes = plt.subplots(nrows=(n + 1) // 2, ncols=2, figsize=(16, n * 3))

for ax, crop in zip(axes.flat, crop_types):
    filtered = trn_dset[trn_dset['Crop Type'] == crop]
    plot = sns.countplot(data=filtered, x='Fertilizer Name', palette='Set2', ax=ax)

    for container in plot.containers:
        plot.bar_label(container, fmt='%d', label_type='edge', color='black')

    ax.set_title(f"{crop} Crop", fontsize=12)
    ax.set_xlabel("Fertilizer")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 15))

for i, col in enumerate(num_cols ,1):
    plt.subplot(3, 2, i)
    sns.histplot(trn_dset[col], kde=True, color='skyblue')
    plt.title(f"Histplot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    
plt.tight_layout()
plt.show()


# Cramér’s V -

from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

for col in cat_cols:
    v = cramers_v(trn_dset[col], trn_dset['Fertilizer Name'])
    print(f"Cramér’s V between {col} and {'Fertilizer Name'}: {v:.3f}")


cramer_scores = {col: cramers_v(trn_dset[col], trn_dset['Fertilizer Name']) for col in cat_cols}
cramer_df = pd.DataFrame.from_dict(cramer_scores, orient='index', columns=["Cramér’s V"])
cramer_df.sort_values(by="Cramér’s V", ascending=False, inplace=True)


plt.figure(figsize=(8, 4))
sns.heatmap(cramer_df, annot=True, cmap="YlGnBu", cbar=False)
plt.title("Cramér’s V - Association with Fertilizer Name")
plt.show()


# ANOVA F-test (Linear class separation (parametric))-

from sklearn.feature_selection import f_classif

X = trn_dset[num_cols]
y = trn_dset['Fertilizer Name']

f_scores, p_values = f_classif(X, y)

anova_result = pd.Series(f_scores, index=num_cols).sort_values(ascending=False)
print("ANOVA F-scores with Target:")
print(anova_result)


# temporariy encoding our Traget column for EDA -

from sklearn.preprocessing import LabelEncoder

df_copy = trn_dset.copy()
le = LabelEncoder()

df_copy['Fertilizer Name'] = le.fit_transform(df_copy['Fertilizer Name'])


# Mutual Information (Any kind of dependency (nonlinear))-

from sklearn.feature_selection import mutual_info_classif

X_num = df_copy[num_cols]
y_tar = df_copy['Fertilizer Name'] 

# Compute MI
mi_scores = mutual_info_classif(X_num, y_tar, random_state=42)
mi_series = pd.Series(mi_scores, index=num_cols).sort_values(ascending=False)

print("Mutual Information Scores:\n")
print(mi_series)


# Pearson Correlation with encoded target (Linear relationship)-

corr_cont = df_copy[num_cols + ['Fertilizer Name']].corr(numeric_only = True)
corr_with_target = pd.Series(corr_cont['Fertilizer Name']).sort_values(ascending = False)

print("Pearson correlation :\n")
print(corr_with_target)


# Combining Anova score, MI score and Pearson correlation together -

combined_df = pd.DataFrame({'F_score': anova_result,
                           'Mutual_info': mi_series,
                           'Pearson_corr': corr_with_target}).sort_values(by='Mutual_info', ascending=False)

print("Combined Feature Selection Metrics:\n")
print(combined_df)


combined_df = combined_df.drop(index='Fertilizer Name', errors='ignore')  # Drop target if still present

combined_df.plot(kind='bar', figsize=(12, 6))
plt.title("Feature Selection Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Box-plot. -

for col in num_cols:
    plt.figure(figsize = (8, 5))
    sns.boxplot(x=target_col, y=col, data=trn_dset, palette='Set2')
    plt.title(f'{col} Distribution by Fertilizer Type')
    plt.tight_layout()
    plt.show()


df_copy[target_col] = pd.to_numeric(df_copy[target_col], errors='coerce')

corr_cols = num_cols + [target_col]
corr_matrix = df_copy[corr_cols].corr(numeric_only=True)

plt.figure(figsize=(10, 6), dpi=100)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": .75}, linewidths=0.5, linecolor='white')
plt.title('Correlation Heatmap with Fertilizer Name', fontsize=14, fontweight='bold')
plt.show()


# label encoding -

# Dictionary to hold encoders
label_encoders = {}
from sklearn.preprocessing import LabelEncoder
# Step 1: Encode categorical features in training data
for col in cat_cols:
    le = LabelEncoder()
    trn_dset[col + '_Encoded'] = le.fit_transform(trn_dset[col])
    label_encoders[col] = le  

# Step 2: Apply the same encoding to test data
for col in cat_cols:
    le = label_encoders[col]  
    tst_dset[col + '_Encoded'] = tst_dset[col].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    ).astype(int)

# Step 3: Encode the target variable (only for training data)
target_encoder = LabelEncoder()
trn_dset['Fertilizer_Label'] = target_encoder.fit_transform(trn_dset[target_col])


trn_dset['Soil Type_Encoded'].unique()


trn_dset['Crop Type_Encoded'].unique()


trn_dset['Fertilizer_Label'].unique()


trn_dset.head(2)


tst_dset.head(2)


feature_col = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type_Encoded', 'Crop Type_Encoded']
target_col = 'Fertilizer_Label'

X_trn = trn_dset[feature_col]
y_trn = trn_dset[target_col]

X_tst = tst_dset[feature_col]


X_trn.dtypes.to_frame().T


X_tst.dtypes.to_frame().T


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        for i, pred in enumerate(p):
            if pred == a:
                score += 1.0 / (i + 1.0)
                break  
        return score

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_trn, y_trn, test_size=0.2, stratify=y_trn, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_val)
rf_top3 = np.argsort(rf_probs, axis=1)[:, -3:][:, ::-1]

rf_map3 = mapk(y_val.tolist(), [list(p) for p in rf_top3], k=3)
print(f"Random Forest MAP@3: {rf_map3:.4f}")


#gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
#gb_model.fit(X_train, y_train)

#gb_probs = gb_model.predict_proba(X_val)
#gb_top3 = np.argsort(gb_probs, axis=1)[:, -3:][:, ::-1]

#gb_map3 = mapk(y_val.tolist(), [list(p) for p in gb_top3], k=3)
#print(f"Gradient Boosting MAP@3: {gb_map3:.4f}")


xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_probs = xgb_model.predict_proba(X_val)
xgb_top3 = np.argsort(xgb_probs, axis=1)[:, -3:][:, ::-1]

xgb_map3 = mapk(y_val.tolist(), [list(p) for p in xgb_top3], k=3)
print(f"XGBoost MAP@3: {xgb_map3:.4f}")


# Best XGBoost Parameters from Optuna run -
best_params = {
    'max_depth': 9,
    'learning_rate': 0.012867370073126772,
    'subsample': 0.8086449227141107,
    'colsample_bytree': 0.6384061218670892,
    'min_child_weight': 4,
    'gamma': 0.08789920829906737,
    'reg_alpha': 0.4801264972851718,
    'reg_lambda': 0.21088173494552948
}

best_params.update({
    'n_estimators': 1000,  
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'tree_method': 'hist', 
    'num_class': len(np.unique(y)),  
    'use_label_encoder': False
})


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from copy import deepcopy

# Initialize K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Track true values and predictions
oof_true = []
oof_top3_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_trn, y_trn)):
    print(f"Fold {fold+1}")

    X_train, X_val = X_trn.iloc[train_idx], X_trn.iloc[val_idx]
    y_train, y_val = y_trn.iloc[train_idx], y_trn.iloc[val_idx]

    # Set model params
    params = deepcopy(best_params)
    params['eval_metric'] = 'mlogloss'
    params['use_label_encoder'] = False

    # Train model
    xgb_model = XGBClassifier(**params, n_jobs=-1, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Predict probabilities
    val_probs = xgb_model.predict_proba(X_val)

    # Top 3 predictions
    val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]

    # Store for overall MAP
    oof_true.extend(y_val.tolist())
    oof_top3_preds.extend(val_top3)

    # Fold metrics
    fold_logloss = log_loss(y_val, val_probs)
    fold_map3 = mapk(y_val.tolist(), [list(p) for p in val_top3], k=3)

    print(f"  Fold {fold+1} Log Loss: {fold_logloss:.5f}")
    print(f"  Fold {fold+1} MAP@3:    {fold_map3:.5f}")

# Overall MAP@3
overall_map3 = mapk(oof_true, [list(p) for p in oof_top3_preds], k=3)
print(f"\n Overall K-Fold MAP@3: {overall_map3:.4f}")


final_model = XGBClassifier(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_trn, y_trn)


test_probs = final_model.predict_proba(X_tst)

# Get top 3 predicted class indices
top3_test_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]


# Convert Class Indices Back to Original Labels

top3_fertilizers = []
for row in top3_test_preds:
    top3_labels = target_encoder.inverse_transform(row)
    top3_fertilizers.append(" ".join(top3_labels))


submission_df = pd.DataFrame({
    'id': tst_dset.index,  
    'Fertilizer Name': top3_fertilizers
})


submission_df.to_csv("submission.csv", index=False)
print("✅ Submission file saved as 'submission.csv'")

