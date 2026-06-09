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


!nvidia-smi


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import optuna
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# Read training file
train_df = pd.read_csv("/kaggle/input/dataset/train.csv")
train_df.head(10)


# CSV information
print(f"DATA SHAPE: {train_df.shape}")
print("\nDATA INFO: ")
train_df.info()


# Feature summmny
print("Numerical Feature Summny:")
display(train_df.describe())


# Check Nan values
missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Missing Values')
    plt.title('Missing Values per Feature')
    plt.tight_layout()
    plt.show()
else:
    print("âœ… No missing values found in the dataset.")


# Target features
numerical_feature = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]

# Create the hisplot
for feature in numerical_feature:
    print(f"\nStatisitics for {feature}: ")
    print("\n")
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature], kde=True, bins=30, color="green")
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[feature], color="yellow")
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()
    


# numerical distributions
numerical_cols = ['Temparature', 
                  'Humidity',
                  'Moisture',
                  'Nitrogen',
                  'Potassium',
                  'Phosphorous']
train_df[numerical_cols].hist(bins=20, figsize=(15, 10), color='skyblue')
plt.suptitle('Distributions of Numerical Features')
plt.show()


# Soil Type & Crop Type distirbution
categorical_feature = ["Soil Type", "Crop Type"]
for feature in categorical_feature:
    counts = train_df[feature].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"Distirbution of {feature}")
    plt.axis("equal")
    plt.show()

    print(f"Type Counts: {counts}")
    print(f"Totla Counts: {counts.values.sum()}")


sns.countplot(x='Soil Type', data=train_df)
pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name']).plot(kind='bar', stacked=True)



# Fertilizer distributon
plt.figure(figsize=(12, 6))
f_counts = train_df['Fertilizer Name'].value_counts()
train_df['Fertilizer Name'].value_counts().plot(kind='bar')
plt.title('Distirbution of Fertilizer Labels')
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Choose a colormap (you can try others: 'tab20c', 'Pastel1', etc.)
cmap = plt.get_cmap('tab20c')
colors = cmap(np.linspace(0, 1, len(f_counts)))

#  Draw a pie chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    f_counts,
    labels=f_counts.index,
    autopct='%1.1f%%',
    pctdistance=0.85,      # how far from center to put the pct labels
    startangle=90,         # rotate so first slice starts at 12 o'clock
    colors=colors,
    wedgeprops=dict(width=0.3, edgecolor='white')  # width<1 makes it a donut
)

# Style the texts
plt.setp(texts,     size=12, weight='bold')
plt.setp(autotexts, size=12, weight='bold', color='black')

#  Add a central white circle to finish the donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

plt.tight_layout()
plt.show()

print(f"Fertilizer Counts: {f_counts}")


# List of numerical features
num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Plot boxplots
for feature in num_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=train_df, x='Fertilizer Name', y=feature)
    plt.title(f'{feature} Distribution by Fertilizer Name')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



train_df['NPK_total'] = train_df['Nitrogen'] + train_df['Phosphorous'] + train_df['Potassium']

sns.boxplot(x='Fertilizer Name', y='NPK_total', data=train_df)
plt.xticks(rotation=45)
plt.title("Distribution of NPK_total by Fertilizer")
plt.tight_layout()
plt.show()


# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train_df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']].corr(), annot=True, cmap='coolwarm')



# Fertilizer used by Crop Type
fert_crop_ct = pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name'])

plt.figure(figsize=(14, 8))
sns.heatmap(fert_crop_ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Fertilizer Use by Crop Type')
plt.ylabel('Crop Type')
plt.xlabel('Fertilizer Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Fertilizer used by Soil Type
fert_soil_ct = pd.crosstab(train_df['Soil Type'], train_df['Fertilizer Name'])

plt.figure(figsize=(12, 6))
sns.heatmap(fert_soil_ct, annot=True, fmt='d', cmap='Blues')
plt.title('Fertilizer Use by Soil Type')
plt.ylabel('Soil Type')
plt.xlabel('Fertilizer Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Soil Type vs. Fertilizer
plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x='Soil Type', hue='Fertilizer Name')
plt.title("Fertilizer Distribution by Soil Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Crop Type vs. Fertilizer
plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='Crop Type', hue='Fertilizer Name')
plt.title("Fertilizer Distribution by Crop Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ========== Step 0: Feature Engineering ==========

# Basic Ratios and Totals
train_df["NPK_total"] = train_df["Nitrogen"] + train_df["Phosphorous"] + train_df["Potassium"]
train_df["N_P_ratio"] = train_df["Nitrogen"] / (train_df["Phosphorous"] + 1e-5)
train_df["P_K_ratio"] = train_df["Phosphorous"] / (train_df["Potassium"] + 1e-5)
train_df["N_K_ratio"] = train_df["Nitrogen"] / (train_df["Potassium"] + 1e-5)
train_df["NPK_variance"] = train_df[["Nitrogen", "Phosphorous", "Potassium"]].var(axis=1)

# Climate Interaction
train_df["temp_moist_diff"] = train_df["Temparature"] - train_df["Moisture"]
train_df["humid_temp_ratio"] = train_df["Humidity"] / (train_df["Temparature"] + 1e-5)
train_df["climate_index"] = (train_df["Temparature"] + train_df["Humidity"] + train_df["Moisture"]) / 3

# Categorical cleanup
soil_cols = [col for col in train_df.columns if col.startswith("Soil Type_")]
crop_cols = [col for col in train_df.columns if col.startswith("Crop Type_")]

if soil_cols:
    train_df["Soil_Type"] = train_df[soil_cols].idxmax(axis=1).str.replace("Soil Type_", "")
else:
    train_df["Soil_Type"] = train_df["Soil Type"]

if crop_cols:
    train_df["Crop_Type"] = train_df[crop_cols].idxmax(axis=1).str.replace("Crop Type_", "")
else:
    train_df["Crop_Type"] = train_df["Crop Type"]

train_df["soil_crop_combo"] = train_df["Soil_Type"] + "_" + train_df["Crop_Type"]

# Frequency Encoding
soil_freq = train_df["Soil_Type"].value_counts(normalize=True).to_dict()
crop_freq = train_df["Crop_Type"].value_counts(normalize=True).to_dict()
train_df["soil_freq"] = train_df["Soil_Type"].map(soil_freq)
train_df["crop_freq"] = train_df["Crop_Type"].map(crop_freq)

# Quantile Binning
train_df["temp_bin"] = pd.qcut(train_df["Temparature"], q=4, labels=False, duplicates='drop')
train_df["moisture_bin"] = pd.qcut(train_df["Moisture"], q=4, labels=False, duplicates='drop')
train_df["N_bin"] = pd.qcut(train_df["Nitrogen"], q=4, labels=False, duplicates='drop')

# Domain-Inspired
train_df["dry_and_low_k"] = ((train_df["Moisture"] < 30) & (train_df["Potassium"] < 10)).astype(int)
train_df["humid_and_high_n"] = ((train_df["Humidity"] > 60) & (train_df["Nitrogen"] > 40)).astype(int)



# ========== Step 1: Prepare Features & Target ==========
categorical_cols = ["Soil_Type", "Crop_Type", "soil_crop_combo"]
X = train_df.drop(columns=["Fertilizer Name", "Soil Type", "Crop Type"])
X_encoded = pd.get_dummies(X, columns=categorical_cols)

y = train_df["Fertilizer Name"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# ========== Step 2: MAP@3 Function ==========
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits = 0.0, 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



# ========== Step 0: Feature Engineering ==========
train_df["N_to_P"] = train_df["Nitrogen"] / (train_df["Phosphorous"] + 1e-5)
train_df["N_to_K"] = train_df["Nitrogen"] / (train_df["Potassium"] + 1e-5)
train_df["P_to_K"] = train_df["Phosphorous"] / (train_df["Potassium"] + 1e-5)
train_df["NP_total"] = train_df["Nitrogen"] + train_df["Phosphorous"]
train_df["climate_index"] = (train_df["Temparature"] + train_df["Humidity"] + train_df["Moisture"]) / 3

soil_cols = [col for col in train_df.columns if col.startswith("Soil Type_")]
crop_cols = [col for col in train_df.columns if col.startswith("Crop Type_")]

if soil_cols:
    train_df["Soil_Type"] = train_df[soil_cols].idxmax(axis=1).str.replace("Soil Type_", "")
else:
    train_df["Soil_Type"] = train_df["Soil Type"]

if crop_cols:
    train_df["Crop_Type"] = train_df[crop_cols].idxmax(axis=1).str.replace("Crop Type_", "")
else:
    train_df["Crop_Type"] = train_df["Crop Type"]

train_df["soil_crop_combo"] = train_df["Soil_Type"] + "_" + train_df["Crop_Type"]

# ========== Step 1: Prepare Features & Target ==========
categorical_cols = ["Soil_Type", "Crop_Type", "soil_crop_combo"]
X = train_df.drop(columns=["Fertilizer Name", "Soil Type", "Crop Type"])

# ========== Step 3: Optuna Objective Function ==========
def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_encoded)),
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'gamma': trial.suggest_float("gamma", 0, 5),
        'lambda': trial.suggest_float("lambda", 0, 5),
        'alpha': trial.suggest_float("alpha", 0, 5),
        'n_estimators': 300,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'use_label_encoder': False
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    for train_idx, val_idx in skf.split(X_encoded, y_encoded):
        X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )

        y_pred_probs = model.predict_proba(X_val)
        top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y_val]
        map3_scores.append(mapk(actual, top_3_preds, k=3))

    return np.mean(map3_scores)

# ========== Step 4: Run Optimization ==========
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

# ========== Best Result ==========
print("\nâœ… Best MAP@3:", study.best_value)
print("ğŸ�¯ Best hyperparameters:\n", study.best_params)


# ========== Step 0: Feature Engineering ==========

# Basic Ratios and Totals
train_df["NPK_total"] = train_df["Nitrogen"] + train_df["Phosphorous"] + train_df["Potassium"]
train_df["N_P_ratio"] = train_df["Nitrogen"] / (train_df["Phosphorous"] + 1e-5)
train_df["P_K_ratio"] = train_df["Phosphorous"] / (train_df["Potassium"] + 1e-5)
train_df["N_K_ratio"] = train_df["Nitrogen"] / (train_df["Potassium"] + 1e-5)
train_df["NPK_variance"] = train_df[["Nitrogen", "Phosphorous", "Potassium"]].var(axis=1)

# Climate Interaction
train_df["temp_moist_diff"] = train_df["Temparature"] - train_df["Moisture"]
train_df["humid_temp_ratio"] = train_df["Humidity"] / (train_df["Temparature"] + 1e-5)
train_df["climate_index"] = (train_df["Temparature"] + train_df["Humidity"] + train_df["Moisture"]) / 3

# Categorical cleanup
soil_cols = [col for col in train_df.columns if col.startswith("Soil Type_")]
crop_cols = [col for col in train_df.columns if col.startswith("Crop Type_")]

if soil_cols:
    train_df["Soil_Type"] = train_df[soil_cols].idxmax(axis=1).str.replace("Soil Type_", "")
else:
    train_df["Soil_Type"] = train_df["Soil Type"]

if crop_cols:
    train_df["Crop_Type"] = train_df[crop_cols].idxmax(axis=1).str.replace("Crop Type_", "")
else:
    train_df["Crop_Type"] = train_df["Crop Type"]

train_df["soil_crop_combo"] = train_df["Soil_Type"] + "_" + train_df["Crop_Type"]

# Frequency Encoding
soil_freq = train_df["Soil_Type"].value_counts(normalize=True).to_dict()
crop_freq = train_df["Crop_Type"].value_counts(normalize=True).to_dict()
train_df["soil_freq"] = train_df["Soil_Type"].map(soil_freq)
train_df["crop_freq"] = train_df["Crop_Type"].map(crop_freq)

# Quantile Binning
train_df["temp_bin"] = pd.qcut(train_df["Temparature"], q=4, labels=False, duplicates='drop')
train_df["moisture_bin"] = pd.qcut(train_df["Moisture"], q=4, labels=False, duplicates='drop')
train_df["N_bin"] = pd.qcut(train_df["Nitrogen"], q=4, labels=False, duplicates='drop')

# Domain-Inspired
train_df["dry_and_low_k"] = ((train_df["Moisture"] < 30) & (train_df["Potassium"] < 10)).astype(int)
train_df["humid_and_high_n"] = ((train_df["Humidity"] > 60) & (train_df["Nitrogen"] > 40)).astype(int)

# ========== Step 1: Prepare Features & Target ==========
categorical_cols = ["Soil_Type", "Crop_Type", "soil_crop_combo"]
X = train_df.drop(columns=["Fertilizer Name", "Soil Type", "Crop Type"])
X_encoded = pd.get_dummies(X, columns=categorical_cols)

y = train_df["Fertilizer Name"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ========== Step 2: MAP@3 Function ==========
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits = 0.0, 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# ========== Step 3: Optuna Objective Function ==========
def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_encoded)),
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'gamma': trial.suggest_float("gamma", 0, 5),
        'lambda': trial.suggest_float("lambda", 0, 5),
        'alpha': trial.suggest_float("alpha", 0, 5),
        'n_estimators': 300,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'use_label_encoder': False
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    for train_idx, val_idx in skf.split(X_encoded, y_encoded):
        X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )

        y_pred_probs = model.predict_proba(X_val)
        top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y_val]
        map3_scores.append(mapk(actual, top_3_preds, k=3))

    return np.mean(map3_scores)

# ========== Step 4: Run Optimization ==========
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

# ========== Best Result ==========
print("\nâœ… Best MAP@3:", study.best_value)
print("ğŸ�¯ Best hyperparameters:\n", study.best_params)


# ========== Step 0: Feature Engineering ==========

# Basic Ratios and Totals
train_df["NPK_total"] = train_df["Nitrogen"] + train_df["Phosphorous"] + train_df["Potassium"]
train_df["N_P_ratio"] = train_df["Nitrogen"] / (train_df["Phosphorous"] + 1e-5)
train_df["P_K_ratio"] = train_df["Phosphorous"] / (train_df["Potassium"] + 1e-5)
train_df["N_K_ratio"] = train_df["Nitrogen"] / (train_df["Potassium"] + 1e-5)
train_df["NPK_variance"] = train_df[["Nitrogen", "Phosphorous", "Potassium"]].var(axis=1)

# Climate Interaction
train_df["temp_moist_diff"] = train_df["Temparature"] - train_df["Moisture"]
train_df["humid_temp_ratio"] = train_df["Humidity"] / (train_df["Temparature"] + 1e-5)
train_df["climate_index"] = (train_df["Temparature"] + train_df["Humidity"] + train_df["Moisture"]) / 3

# Categorical cleanup
soil_cols = [col for col in train_df.columns if col.startswith("Soil Type_")]
crop_cols = [col for col in train_df.columns if col.startswith("Crop Type_")]

if soil_cols:
    train_df["Soil_Type"] = train_df[soil_cols].idxmax(axis=1).str.replace("Soil Type_", "")
else:
    train_df["Soil_Type"] = train_df["Soil Type"]

if crop_cols:
    train_df["Crop_Type"] = train_df[crop_cols].idxmax(axis=1).str.replace("Crop Type_", "")
else:
    train_df["Crop_Type"] = train_df["Crop Type"]

train_df["soil_crop_combo"] = train_df["Soil_Type"] + "_" + train_df["Crop_Type"]

# Frequency Encoding
soil_freq = train_df["Soil_Type"].value_counts(normalize=True).to_dict()
crop_freq = train_df["Crop_Type"].value_counts(normalize=True).to_dict()
train_df["soil_freq"] = train_df["Soil_Type"].map(soil_freq)
train_df["crop_freq"] = train_df["Crop_Type"].map(crop_freq)

# Quantile Binning
train_df["temp_bin"] = pd.qcut(train_df["Temparature"], q=4, labels=False, duplicates='drop')
train_df["moisture_bin"] = pd.qcut(train_df["Moisture"], q=4, labels=False, duplicates='drop')
train_df["N_bin"] = pd.qcut(train_df["Nitrogen"], q=4, labels=False, duplicates='drop')

# Domain-Inspired
train_df["dry_and_low_k"] = ((train_df["Moisture"] < 30) & (train_df["Potassium"] < 10)).astype(int)
train_df["humid_and_high_n"] = ((train_df["Humidity"] > 60) & (train_df["Nitrogen"] > 40)).astype(int)

# ========== Step 1: Prepare Features & Target ==========
categorical_cols = ["Soil_Type", "Crop_Type", "soil_crop_combo"]
X = train_df.drop(columns=["Fertilizer Name", "Soil Type", "Crop Type"])
X[categorical_cols] = X[categorical_cols].astype("category")  # CatBoost handles this internally

y = train_df["Fertilizer Name"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ========== Step 2: MAP@3 Function ==========
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits = 0.0, 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# ========== Step 3: Optuna Objective Function ==========
def objective(trial):
    params = {
        'loss_function': 'MultiClass',
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'depth': trial.suggest_int("depth", 4, 10),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1, 10),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0, 1),
        'random_strength': trial.suggest_float("random_strength", 1e-9, 10),
        'iterations': 300,
        'random_seed': 42,
        'verbose': False,
        'task_type': 'GPU'
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    for train_idx, val_idx in skf.split(X, y_encoded):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
        val_pool = Pool(X_val, y_val, cat_features=categorical_cols)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30)

        y_pred_probs = model.predict_proba(val_pool)
        top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y_val]
        map3_scores.append(mapk(actual, top_3_preds, k=3))

    return np.mean(map3_scores)

# ========== Step 4: Run Optimization ==========
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

# ========== Best Result ==========
print("\nâœ… Best MAP@3:", study.best_value)
print("ğŸ�¯ Best hyperparameters:\n", study.best_params)


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ========== Step 0: Feature Engineering ==========
train_df["NPK_total"] = train_df["Nitrogen"] + train_df["Phosphorous"] + train_df["Potassium"]
train_df["N_P_ratio"] = train_df["Nitrogen"] / (train_df["Phosphorous"] + 1e-5)
train_df["P_K_ratio"] = train_df["Phosphorous"] / (train_df["Potassium"] + 1e-5)
train_df["N_K_ratio"] = train_df["Nitrogen"] / (train_df["Potassium"] + 1e-5)
train_df["NPK_variance"] = train_df[["Nitrogen", "Phosphorous", "Potassium"]].var(axis=1)

train_df["temp_moist_diff"] = train_df["Temparature"] - train_df["Moisture"]
train_df["humid_temp_ratio"] = train_df["Humidity"] / (train_df["Temparature"] + 1e-5)
train_df["climate_index"] = (train_df["Temparature"] + train_df["Humidity"] + train_df["Moisture"]) / 3

soil_cols = [col for col in train_df.columns if col.startswith("Soil Type_")]
crop_cols = [col for col in train_df.columns if col.startswith("Crop Type_")]

if soil_cols:
    train_df["Soil_Type"] = train_df[soil_cols].idxmax(axis=1).str.replace("Soil Type_", "")
else:
    train_df["Soil_Type"] = train_df["Soil Type"]

if crop_cols:
    train_df["Crop_Type"] = train_df[crop_cols].idxmax(axis=1).str.replace("Crop Type_", "")
else:
    train_df["Crop_Type"] = train_df["Crop Type"]

train_df["soil_crop_combo"] = train_df["Soil_Type"] + "_" + train_df["Crop_Type"]

# Frequency Encoding
soil_freq = train_df["Soil_Type"].value_counts(normalize=True).to_dict()
crop_freq = train_df["Crop_Type"].value_counts(normalize=True).to_dict()
train_df["soil_freq"] = train_df["Soil_Type"].map(soil_freq)
train_df["crop_freq"] = train_df["Crop_Type"].map(crop_freq)

# Quantile Binning
train_df["temp_bin"] = pd.qcut(train_df["Temparature"], q=4, labels=False, duplicates='drop')
train_df["moisture_bin"] = pd.qcut(train_df["Moisture"], q=4, labels=False, duplicates='drop')
train_df["N_bin"] = pd.qcut(train_df["Nitrogen"], q=4, labels=False, duplicates='drop')

# Domain-Inspired Features
train_df["dry_and_low_k"] = ((train_df["Moisture"] < 30) & (train_df["Potassium"] < 10)).astype(int)
train_df["humid_and_high_n"] = ((train_df["Humidity"] > 60) & (train_df["Nitrogen"] > 40)).astype(int)

# ========== Step 1: Prepare Features & Target ==========
features = [
    "Temparature", "Humidity", "Moisture", "Nitrogen", "Phosphorous", "Potassium",
    "NPK_total", "N_P_ratio", "P_K_ratio", "N_K_ratio", "NPK_variance",
    "climate_index", "humid_temp_ratio", "temp_moist_diff",
    "soil_freq", "crop_freq",
    "soil_crop_combo", "dry_and_low_k", "humid_and_high_n",
    "temp_bin", "moisture_bin", "N_bin"
]
categorical_features = ["soil_crop_combo"]

X = train_df[features].copy()
y = train_df["Fertilizer Name"]

for col in categorical_features:
    X[col] = X[col].astype("category")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

X_xgb = pd.get_dummies(X, columns=categorical_features)

# ========== Step 2: MAP@3 Function ==========
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits = 0.0, 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# ========== Step 3: Define Model Parameters ==========
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': num_classes,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'learning_rate': 0.1898,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.9250,
    'colsample_bytree': 0.9027,
    'gamma': 0.6413,
    'reg_lambda': 0.4691,
    'reg_alpha': 4.9767,
    'n_estimators': 300,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'use_label_encoder': False
}

cat_params = {
    'loss_function': 'MultiClass',
    'learning_rate': 0.2779,
    'depth': 5,
    'l2_leaf_reg': 6.4477,
    'bagging_temperature': 0.0304,
    'random_strength': 3.4592,
    'iterations': 300,
    'task_type': 'GPU',
    'devices': '0',
    'random_seed': 42,
    'verbose': False
}

# ========== Step 4: Cross-Validation Ensemble ==========
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies, map3_scores = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n========== Fold {fold + 1} ==========")

    X_train_cb, X_val_cb = X.iloc[train_idx], X.iloc[val_idx]
    train_pool = Pool(X_train_cb, y_encoded[train_idx], cat_features=categorical_features)
    val_pool = Pool(X_val_cb, y_encoded[val_idx], cat_features=categorical_features)

    X_train_xgb, X_val_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]

    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_xgb, y_encoded[train_idx])

    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(train_pool)

    xgb_probs = xgb_model.predict_proba(X_val_xgb)
    cat_probs = cat_model.predict_proba(val_pool)
    blended_probs = (xgb_probs + cat_probs) / 2.0

    y_pred = np.argmax(blended_probs, axis=1)
    acc = accuracy_score(y_encoded[val_idx], y_pred)
    accuracies.append(acc)

    top_3_preds = np.argsort(blended_probs, axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_encoded[val_idx]]
    map3 = mapk(actual, top_3_preds)
    map3_scores.append(map3)

    print(f"Accuracy: {acc:.4f}")
    print(f"MAP@3: {map3:.4f}")
    print(classification_report(y_encoded[val_idx], y_pred, target_names=le.classes_))

# ========== Step 5: Summary ==========
print("\n========== Ensemble Cross-Validation Summary ==========")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average MAP@3: {np.mean(map3_scores):.4f}")



# ========== Step 0: Import Libraries ==========
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ========== Step 1: Load Train Dataset ==========
train_df = pd.read_csv("/kaggle/input/dataset/train.csv")
test_df = pd.read_csv("/kaggle/input/dataset/test.csv")

# ========== Step 2: Feature Engineering ==========
train_df["NPK_total"] = train_df["Nitrogen"] + train_df["Phosphorous"] + train_df["Potassium"]
train_df["N_P_ratio"] = train_df["Nitrogen"] / (train_df["Phosphorous"] + 1e-5)
train_df["P_K_ratio"] = train_df["Phosphorous"] / (train_df["Potassium"] + 1e-5)
train_df["N_K_ratio"] = train_df["Nitrogen"] / (train_df["Potassium"] + 1e-5)
train_df["NPK_variance"] = train_df[["Nitrogen", "Phosphorous", "Potassium"]].var(axis=1)
train_df["temp_moist_diff"] = train_df["Temparature"] - train_df["Moisture"]
train_df["humid_temp_ratio"] = train_df["Humidity"] / (train_df["Temparature"] + 1e-5)
train_df["climate_index"] = (train_df["Temparature"] + train_df["Humidity"] + train_df["Moisture"]) / 3

soil_cols = [col for col in train_df.columns if col.startswith("Soil Type_")]
crop_cols = [col for col in train_df.columns if col.startswith("Crop Type_")]
if soil_cols:
    train_df["Soil_Type"] = train_df[soil_cols].idxmax(axis=1).str.replace("Soil Type_", "")
else:
    train_df["Soil_Type"] = train_df["Soil Type"]
if crop_cols:
    train_df["Crop_Type"] = train_df[crop_cols].idxmax(axis=1).str.replace("Crop Type_", "")
else:
    train_df["Crop_Type"] = train_df["Crop Type"]

train_df["soil_crop_combo"] = train_df["Soil_Type"] + "_" + train_df["Crop_Type"]
soil_freq = train_df["Soil_Type"].value_counts(normalize=True).to_dict()
crop_freq = train_df["Crop_Type"].value_counts(normalize=True).to_dict()
train_df["soil_freq"] = train_df["Soil_Type"].map(soil_freq)
train_df["crop_freq"] = train_df["Crop_Type"].map(crop_freq)
train_df["temp_bin"] = pd.qcut(train_df["Temparature"], q=4, labels=False, duplicates='drop')
train_df["moisture_bin"] = pd.qcut(train_df["Moisture"], q=4, labels=False, duplicates='drop')
train_df["N_bin"] = pd.qcut(train_df["Nitrogen"], q=4, labels=False, duplicates='drop')
train_df["dry_and_low_k"] = ((train_df["Moisture"] < 30) & (train_df["Potassium"] < 10)).astype(int)
train_df["humid_and_high_n"] = ((train_df["Humidity"] > 60) & (train_df["Nitrogen"] > 40)).astype(int)

# ========== Step 3: Feature Preparation ==========
features = [
    "Temparature", "Humidity", "Moisture", "Nitrogen", "Phosphorous", "Potassium",
    "NPK_total", "N_P_ratio", "P_K_ratio", "N_K_ratio", "NPK_variance",
    "climate_index", "humid_temp_ratio", "temp_moist_diff",
    "soil_freq", "crop_freq",
    "soil_crop_combo", "dry_and_low_k", "humid_and_high_n",
    "temp_bin", "moisture_bin", "N_bin"
]
categorical_features = ["soil_crop_combo"]

X = train_df[features].copy()
y = train_df["Fertilizer Name"]
for col in categorical_features:
    X[col] = X[col].astype("category")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))
X_xgb = pd.get_dummies(X, columns=categorical_features)

# ========== Step 4: MAP@3 Function ==========
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits = 0.0, 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# ========== Step 5: Model Parameters ==========
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': num_classes,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'learning_rate': 0.1898,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.9250,
    'colsample_bytree': 0.9027,
    'gamma': 0.6413,
    'reg_lambda': 0.4691,
    'reg_alpha': 4.9767,
    'n_estimators': 300,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'use_label_encoder': False
}

cat_params = {
    'loss_function': 'MultiClass',
    'learning_rate': 0.2779,
    'depth': 5,
    'l2_leaf_reg': 6.4477,
    'bagging_temperature': 0.0304,
    'random_strength': 3.4592,
    'iterations': 300,
    'task_type': 'GPU',
    'devices': '0',
    'random_seed': 42,
    'verbose': False
}

# ========== Step 6: Cross-Validation Ensemble ==========
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies, map3_scores = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n========== Fold {fold + 1} ==========")
    X_train_cb, X_val_cb = X.iloc[train_idx], X.iloc[val_idx]
    train_pool = Pool(X_train_cb, y_encoded[train_idx], cat_features=categorical_features)
    val_pool = Pool(X_val_cb, y_encoded[val_idx], cat_features=categorical_features)
    X_train_xgb, X_val_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]

    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_xgb, y_encoded[train_idx])

    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(train_pool)

    xgb_probs = xgb_model.predict_proba(X_val_xgb)
    cat_probs = cat_model.predict_proba(val_pool)
    blended_probs = (xgb_probs + cat_probs) / 2.0

    y_pred = np.argmax(blended_probs, axis=1)
    acc = accuracy_score(y_encoded[val_idx], y_pred)
    accuracies.append(acc)

    top_3_preds = np.argsort(blended_probs, axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_encoded[val_idx]]
    map3 = mapk(actual, top_3_preds)
    map3_scores.append(map3)

    print(f"Accuracy: {acc:.4f}")
    print(f"MAP@3: {map3:.4f}")

print("\n========== Ensemble Cross-Validation Summary ==========")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average MAP@3: {np.mean(map3_scores):.4f}")

# ========== Step 7: Predict on test.csv ==========
test_df["NPK_total"] = test_df["Nitrogen"] + test_df["Phosphorous"] + test_df["Potassium"]
test_df["N_P_ratio"] = test_df["Nitrogen"] / (test_df["Phosphorous"] + 1e-5)
test_df["P_K_ratio"] = test_df["Phosphorous"] / (test_df["Potassium"] + 1e-5)
test_df["N_K_ratio"] = test_df["Nitrogen"] / (test_df["Potassium"] + 1e-5)
test_df["NPK_variance"] = test_df[["Nitrogen", "Phosphorous", "Potassium"]].var(axis=1)
test_df["temp_moist_diff"] = test_df["Temparature"] - test_df["Moisture"]
test_df["humid_temp_ratio"] = test_df["Humidity"] / (test_df["Temparature"] + 1e-5)
test_df["climate_index"] = (test_df["Temparature"] + test_df["Humidity"] + test_df["Moisture"]) / 3
test_df["Soil_Type"] = test_df["Soil Type"]
test_df["Crop_Type"] = test_df["Crop Type"]
test_df["soil_crop_combo"] = test_df["Soil_Type"] + "_" + test_df["Crop_Type"]
test_df["soil_freq"] = test_df["Soil_Type"].map(soil_freq)
test_df["crop_freq"] = test_df["Crop_Type"].map(crop_freq)
test_df["temp_bin"] = pd.qcut(test_df["Temparature"], q=4, labels=False, duplicates='drop')
test_df["moisture_bin"] = pd.qcut(test_df["Moisture"], q=4, labels=False, duplicates='drop')
test_df["N_bin"] = pd.qcut(test_df["Nitrogen"], q=4, labels=False, duplicates='drop')
test_df["dry_and_low_k"] = ((test_df["Moisture"] < 30) & (test_df["Potassium"] < 10)).astype(int)
test_df["humid_and_high_n"] = ((test_df["Humidity"] > 60) & (test_df["Nitrogen"] > 40)).astype(int)

X_test = test_df[features].copy()
for col in categorical_features:
    X_test[col] = X_test[col].astype("category")

X_test_xgb = pd.get_dummies(X_test, columns=categorical_features)
X_test_xgb = X_test_xgb.reindex(columns=X_xgb.columns, fill_value=0)

# ========== Step 8: Final Training & Prediction ==========
final_xgb = XGBClassifier(**xgb_params)
final_xgb.fit(X_xgb, y_encoded)
final_cat = CatBoostClassifier(**cat_params)
final_cat.fit(Pool(X, y_encoded, cat_features=categorical_features))

xgb_probs = final_xgb.predict_proba(X_test_xgb)
cat_probs = final_cat.predict_proba(Pool(X_test, cat_features=categorical_features))
blended_probs = (xgb_probs + cat_probs) / 2.0
top3_preds = np.argsort(blended_probs, axis=1)[:, -3:][:, ::-1]
top3_labels = np.stack([
    le.inverse_transform(top3_preds[:, 0]),
    le.inverse_transform(top3_preds[:, 1]),
    le.inverse_transform(top3_preds[:, 2])
], axis=1)  

# ========== Step 9: Export Submission ==========
submission = pd.DataFrame({
    "id": test_df["id"],
    "Fertilizer Name": top3_labels[:, 0]  # only the top-1 prediction
})
submission.to_csv("submission.csv", index=False)
print("âœ… Predicted completeï¼�submission.csv has saved")




