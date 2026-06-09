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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Drop 'maxtemp' and 'mintemp' columns (redundant)
train.drop(columns=['maxtemp', 'mintemp'], inplace=True)
test.drop(columns=['maxtemp', 'mintemp'], inplace=True)

# Fill missing values with mean
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# Splitting features and target
X = train.drop(columns=['rainfall'])
y = train['rainfall']
X_test = test.copy()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# **ANOVA Feature Selection** - Select top K best features
k = 7  # Adjust as needed
anova_selector = SelectKBest(score_func=f_classif, k=k)
X_selected = anova_selector.fit_transform(X_scaled, y)
X_test_selected = anova_selector.transform(X_test_scaled)

# **PCA for Dimensionality Reduction** - Keep 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_selected)
X_test_pca = pca.transform(X_test_selected)

# Train-test split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Define models with hyperparameter tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5, 10]
    },
    "ExtraTrees": {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5, 10]
    },
    "GradientBoosting": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7]
    },
    "XGBoost": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7]
    },
    "LightGBM": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 40]
    },
    "CatBoost": {
        "iterations": [500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 5, 7]
    }
}

best_model = None
best_val_accuracy = 0

for name, params in param_grids.items():
    print(f"Tuning {name}...")
    if name == "RandomForest":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    elif name == "ExtraTrees":
        model = ExtraTreesClassifier(random_state=42, class_weight="balanced")
    elif name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    elif name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif name == "LightGBM":
        model = LGBMClassifier(random_state=42)
    elif name == "CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42)
    
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model_instance = grid_search.best_estimator_
    
    train_acc = accuracy_score(y_train, best_model_instance.predict(X_train))
    val_acc = accuracy_score(y_val, best_model_instance.predict(X_val))
    
    print(f"{name} Best Params: {grid_search.best_params_}")
    print(f"{name} Train Accuracy: {train_acc}")
    print(f"{name} Validation Accuracy: {val_acc}\n")
    print(classification_report(y_val, best_model_instance.predict(X_val)))
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model = best_model_instance

# Predict probabilities on test set using best model
rainfall_prob = best_model.predict_proba(X_test_pca)[:, 1]

# Prepare submission file
submission = pd.DataFrame({'id': test.index, 'rainfall': rainfall_prob})
submission.to_csv("submission_optimized.csv", index=False)

print(f"\nBest Model: {best_model}")









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Drop irrelevant columns
train.drop(columns=['maxtemp', 'mintemp'], inplace=True)
test.drop(columns=['maxtemp', 'mintemp'], inplace=True)

# Fill missing values with mean
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# Splitting features and target
X = train.drop(columns=['rainfall'])
y = train['rainfall']
X_test = test.copy()

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# **PCA for Dimensionality Reduction** (Keeping 95% Variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models with hyperparameter tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [300, 500],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5]
    },
    "ExtraTrees": {
        "n_estimators": [300, 500],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5]
    },
    "GradientBoosting": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7]
    },
    "XGBoost": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7]
    },
    "LightGBM": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 40]
    },
    "CatBoost": {
        "iterations": [500, 1000],
        "learning_rate": [0.05, 0.1],
        "depth": [5, 7]
    }
}

best_model = None
best_avg_accuracy = 0

for name, params in param_grids.items():
    print(f"\nTuning {name} with K-Fold Cross Validation...")

    if name == "RandomForest":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    elif name == "ExtraTrees":
        model = ExtraTreesClassifier(random_state=42, class_weight="balanced")
    elif name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    elif name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif name == "LightGBM":
        model = LGBMClassifier(random_state=42)
    elif name == "CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42)

    grid_search = GridSearchCV(model, params, cv=kf, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_pca, y)
    
    best_model_instance = grid_search.best_estimator_
    avg_accuracy = grid_search.best_score_

    print(f"{name} Best Params: {grid_search.best_params_}")
    print(f"{name} Average Cross-Validation Accuracy: {avg_accuracy:.4f}\n")

    if avg_accuracy > best_avg_accuracy:
        best_avg_accuracy = avg_accuracy
        best_model = best_model_instance

# Predict probabilities on test set using the best model
rainfall_prob = best_model.predict_proba(X_test_pca)[:, 1]

# Prepare submission file
submission = pd.DataFrame({'id': test.index, 'rainfall': rainfall_prob})
submission.to_csv("submission_kfold_optimized.csv", index=False)

print(f"\nBest Model: {best_model}")









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Drop irrelevant columns
train.drop(columns=['maxtemp', 'mintemp'], inplace=True)
test.drop(columns=['maxtemp', 'mintemp'], inplace=True)

# Fill missing values with mean
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# Splitting features and target
X = train.drop(columns=['rainfall'])
y = train['rainfall']
X_test = test.copy()

# Scaling using RobustScaler (handles outliers better than StandardScaler)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Feature Engineering: Polynomial Features (2nd degree interaction terms)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Dimensionality Reduction: PCA (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_poly)
X_test_pca = pca.transform(X_test_poly)

# Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models with hyperparameter tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [300, 500],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5]
    },
    "ExtraTrees": {
        "n_estimators": [300, 500],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5]
    },
    "GradientBoosting": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7]
    },
    "XGBoost": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7],
        "gamma": [0.1, 0.2],
        "reg_alpha": [0.1, 0.5]
    },
    "LightGBM": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 40]
    },
    "CatBoost": {
        "iterations": [500, 1000],
        "learning_rate": [0.05, 0.1],
        "depth": [5, 7]
    }
}

best_model = None
best_avg_accuracy = 0

for name, params in param_grids.items():
    print(f"\nTuning {name} with K-Fold Cross Validation...")

    if name == "RandomForest":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    elif name == "ExtraTrees":
        model = ExtraTreesClassifier(random_state=42, class_weight="balanced")
    elif name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    elif name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif name == "LightGBM":
        model = LGBMClassifier(random_state=42)
    elif name == "CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42)

    grid_search = RandomizedSearchCV(model, params, cv=kf, n_iter=20, scoring='accuracy', n_jobs=-1, random_state=42)
    grid_search.fit(X_resampled, y_resampled)
    
    best_model_instance = grid_search.best_estimator_
    avg_accuracy = grid_search.best_score_

    print(f"{name} Best Params: {grid_search.best_params_}")
    print(f"{name} Average Cross-Validation Accuracy: {avg_accuracy:.4f}\n")

    if avg_accuracy > best_avg_accuracy:
        best_avg_accuracy = avg_accuracy
        best_model = best_model_instance

# **Stacking Classifier**: Combining the best models for better predictions
stack_model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(**grid_search.best_params_)),
        ('lgbm', LGBMClassifier(**grid_search.best_params_)),
        ('rf', RandomForestClassifier(**grid_search.best_params_))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# Train the stacked model
stack_model.fit(X_resampled, y_resampled)

# Predict probabilities on test set using Stacked Model
rainfall_prob = stack_model.predict_proba(X_test_pca)[:, 1]

# Prepare submission file
submission = pd.DataFrame({'id': test.index, 'rainfall': rainfall_prob})
submission.to_csv("submission_stacking_optimized.csv", index=False)

print(f"\nBest Model (Stacked): {stack_model}")









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Drop irrelevant columns
train.drop(columns=['maxtemp', 'mintemp'], inplace=True)
test.drop(columns=['maxtemp', 'mintemp'], inplace=True)

# Fill missing values with mean
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# Splitting features and target
X = train.drop(columns=['rainfall'])
y = train['rainfall']
X_test = test.copy()

# Scaling using RobustScaler (handles outliers better than StandardScaler)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Feature Engineering: Polynomial Features (2nd degree interaction terms)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Dimensionality Reduction: PCA (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_poly)
X_test_pca = pca.transform(X_test_poly)

# Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models with hyperparameter tuning
param_grids = {
    "RandomForest": {
        "n_estimators": [300, 500],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5]
    },
    "ExtraTrees": {
        "n_estimators": [300, 500],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5]
    },
    "GradientBoosting": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7]
    },
    "XGBoost": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7],
        "gamma": [0.1, 0.2],
        "reg_alpha": [0.1, 0.5]
    },
    "LightGBM": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 40]
    },
    "CatBoost": {
        "iterations": [500, 1000],
        "learning_rate": [0.05, 0.1],
        "depth": [5, 7]
    }
}

best_model = None
best_avg_accuracy = 0
best_params = {}

for name, params in param_grids.items():
    print(f"\nTuning {name} with K-Fold Cross Validation...")

    if name == "RandomForest":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    elif name == "ExtraTrees":
        model = ExtraTreesClassifier(random_state=42, class_weight="balanced")
    elif name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    elif name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif name == "LightGBM":
        model = LGBMClassifier(random_state=42)
    elif name == "CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42)

    grid_search = RandomizedSearchCV(model, params, cv=kf, n_iter=20, scoring='accuracy', n_jobs=-1, random_state=42)
    grid_search.fit(X_resampled, y_resampled)
    
    best_model_instance = grid_search.best_estimator_
    avg_accuracy = grid_search.best_score_

    print(f"{name} Best Params: {grid_search.best_params_}")
    print(f"{name} Average Cross-Validation Accuracy: {avg_accuracy:.4f}\n")

    if avg_accuracy > best_avg_accuracy:
        best_avg_accuracy = avg_accuracy
        best_model = best_model_instance

    best_params[name] = grid_search.best_params_

# **Stacking Classifier**: Combining the best models for better predictions
stack_model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(**{k: v for k, v in best_params["XGBoost"].items() if k in XGBClassifier().get_params()},
                              use_label_encoder=False, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(**{k: v for k, v in best_params["LightGBM"].items() if k in LGBMClassifier().get_params()})),
        ('rf', RandomForestClassifier(**{k: v for k, v in best_params["RandomForest"].items() if k in RandomForestClassifier().get_params()}))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# Train the stacked model
stack_model.fit(X_resampled, y_resampled)

# Predict probabilities on test set using Stacked Model
rainfall_prob = stack_model.predict_proba(X_test_pca)[:, 1]

# Prepare submission file
submission = pd.DataFrame({'id': test.index, 'rainfall': rainfall_prob})
submission.to_csv("submission_stacking_optimized.csv", index=False)

print(f"\nBest Model (Stacked): {stack_model}")









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

# ==============================
# Step 1: Load and Preprocess Data
# ==============================

# Load the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Drop irrelevant columns ('maxtemp' and 'mintemp')
train.drop(columns=['maxtemp', 'mintemp'], inplace=True)
test.drop(columns=['maxtemp', 'mintemp'], inplace=True)

# Fill missing values with mean
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# Separate features and target
X = train.drop(columns=['rainfall'])
y = train['rainfall']
X_test = test.copy()

# ==============================
# Step 2: Scaling and Feature Engineering
# ==============================

# Use RobustScaler to handle outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Generate Polynomial Features (2nd degree interactions only)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Apply PCA to reduce dimensionality while retaining 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_poly)
X_test_pca = pca.transform(X_test_poly)

# ==============================
# Step 3: Handle Class Imbalance with SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# ==============================
# Step 4: Hyperparameter Tuning with K-Fold CV for Multiple Models
# ==============================

# Define K-Fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define parameter grids for different models
param_grids = {
    "RandomForest": {
        "n_estimators": [300, 500],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5]
    },
    "ExtraTrees": {
        "n_estimators": [300, 500],
        "max_depth": [None, 30],
        "min_samples_split": [2, 5]
    },
    "GradientBoosting": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7]
    },
    "XGBoost": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7],
        "gamma": [0.1, 0.2],
        "reg_alpha": [0.1, 0.5]
    },
    "LightGBM": {
        "n_estimators": [300, 500],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 40]
    },
    "CatBoost": {
        "iterations": [500, 1000],
        "learning_rate": [0.05, 0.1],
        "depth": [5, 7]
    }
}

best_model = None
best_avg_accuracy = 0
# Dictionary to store best parameters for each model
best_params_dict = {}

# Tune each model using RandomizedSearchCV
for name, params in param_grids.items():
    print(f"\nTuning {name} with K-Fold Cross Validation...")
    
    if name == "RandomForest":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    elif name == "ExtraTrees":
        model = ExtraTreesClassifier(random_state=42, class_weight="balanced")
    elif name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    elif name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif name == "LightGBM":
        model = LGBMClassifier(random_state=42)
    elif name == "CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42)
    
    grid_search = RandomizedSearchCV(model, params, cv=kf, n_iter=20, scoring='accuracy', n_jobs=-1, random_state=42)
    grid_search.fit(X_resampled, y_resampled)
    
    best_model_instance = grid_search.best_estimator_
    avg_accuracy = grid_search.best_score_
    
    best_params_dict[name] = grid_search.best_params_
    
    print(f"{name} Best Params: {grid_search.best_params_}")
    print(f"{name} Average CV Accuracy: {avg_accuracy:.4f}\n")
    print(classification_report(y_resampled, best_model_instance.predict(X_resampled)))
    
    if avg_accuracy > best_avg_accuracy:
        best_avg_accuracy = avg_accuracy
        best_model = best_model_instance

print("\nOverall Best Model from Tuning:", best_model)
    
# ==============================
# Step 5: Build a Stacking Classifier Using Filtered Parameters
# ==============================

# For stacking, we choose base models: XGBoost, LightGBM, and RandomForest.
# Filter only the parameters that are valid for each model.
best_xgb_params = {k: v for k, v in best_params_dict.get("XGBoost", {}).items() if k in XGBClassifier().get_params()}
best_lgbm_params = {k: v for k, v in best_params_dict.get("LightGBM", {}).items() if k in LGBMClassifier().get_params()}
best_rf_params = {k: v for k, v in best_params_dict.get("RandomForest", {}).items() if k in RandomForestClassifier().get_params()}

# Instantiate base models with their filtered best parameters
xgb_model = XGBClassifier(**best_xgb_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
lgbm_model = LGBMClassifier(**best_lgbm_params, random_state=42)
rf_model = RandomForestClassifier(**best_rf_params, random_state=42, class_weight="balanced")

# Build stacking classifier with a Logistic Regression final estimator
stack_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('rf', rf_model)
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# Train the stacking classifier on the resampled data
stack_model.fit(X_resampled, y_resampled)

# ==============================
# Step 6: Predict on Test Set and Prepare Submission
# ==============================

# Predict probabilities on the test set using the stacked model
rainfall_prob = stack_model.predict_proba(X_test_pca)[:, 1]

# Prepare submission file (use test index as id)
submission = pd.DataFrame({'id': test.index, 'rainfall': rainfall_prob})
submission.to_csv("submission_stacking_optimized.csv", index=False)

print("\nStacking Model Trained Successfully!")
print("Submission file 'submission_stacking_optimized.csv' created.")
print("\nBest Stacked Model Details:")
print(stack_model)








