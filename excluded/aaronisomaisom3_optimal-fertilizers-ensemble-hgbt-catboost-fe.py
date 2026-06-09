#!pip install optuna --quiet


# Author: Aaron Isom
# Kaggle Playground-Series-S5e6 - Predicting Optimal Fertilizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings

from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from optuna.samplers import TPESampler
from sklearn.dummy import DummyClassifier

warnings.filterwarnings('ignore')
tune = False # Toggle for Optuna tuning and Final Submission


# Mean Average Precision @ 3 (MAP@3)
def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    actual: array of true labels (as strings)
    predicted: list of lists, each list is the k predicted labels for each sample
    """
    def apk(a, p, k):
        """Average Precision at k for a single sample"""
        if a in p[:k]:
            return 1.0 / (p.index(a) + 1)
        else:
            return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Optuna Tuning
def objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 128),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 10.0, log=True),
        'max_bins': trial.suggest_int('max_bins', 64, 255)
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    X = train[features]
    y = train[target]

    print(f"-----------HistGradientBoostingClassifier by MAP@3 Tuning on 5 k-Folds -----------")
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = HistGradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)
        top_3 = np.argsort(preds, axis=1)[:, -3:][:, ::-1]  # get top 3
        top_3_labels = le_target.inverse_transform(top_3.ravel()).reshape(top_3.shape)
        y_val_labels = le_target.inverse_transform(y_val)
        pred_list = [list(row) for row in top_3_labels] # convert to lists for mapk
        map3_score = mapk(y_val_labels, pred_list, k=3)
        print(f"-----------Fold {i+1}: MAP@3 Score: {map3_score:.5f}-----------")
        scores.append(map3_score)
        
    return np.mean(scores)


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


# Basic EDA
print("Train shape:", train.shape)
print("\nTest shape:", test.shape)
print(train.info())
print("\nNulls:\n", train.isnull().sum())
print("\nDuplicates:", train.duplicated().sum())
print("\nData Types:\n", train.dtypes)

# Correlation Matrix & Basic EDA Plots
num_features = train.select_dtypes(include=[np.number]).columns.tolist()
corr = train[num_features].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(10, 4))
sns.countplot(y=train['Fertilizer Name'], order=train['Fertilizer Name'].value_counts().index)
plt.title("Fertilizer Name Class Distribution")
plt.show()

top_classes = train['Fertilizer Name'].value_counts().index[:3]
for col in num_features:
    plt.figure(figsize=(10, 4))
    for c in top_classes:
        sns.kdeplot(train[train['Fertilizer Name']==c][col], label=f"{c}")
    plt.title(f"{col} by Top 3 Fertilizers")
    plt.legend()
    plt.show()

# Categorical variable distributions
for col in train.select_dtypes('object'):
    plt.figure(figsize=(10, 4))
    train[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()

# Basic Feature Engineering
# Combine two categorical columns
train['Soil_Crop'] = train['Soil Type'] + "_" + train['Crop Type']
test['Soil_Crop'] = test['Soil Type'] + "_" + test['Crop Type']

# Build features list
target = 'Fertilizer Name'
features = [col for col in train.columns if col not in [target]]
if 'Soil_Crop' not in features:
    features.append('Soil_Crop')

# Bin numericals into a continuous variable
for col in ['Temparature', 'Humidity', 'Moisture']:
    train_bins, bin_edges = pd.qcut(train[col], 4, labels=False, retbins=True, duplicates='drop')
    train[f'{col}_bin'] = train_bins
    test[f'{col}_bin'] = pd.cut(test[col], bins=bin_edges, labels=False, include_lowest=True)
    features.append(f'{col}_bin')


features = list(dict.fromkeys(features))

# Label Encode categorical features
cat_feature_names = train[features].select_dtypes('object').columns.tolist()
cat_features = [features.index(col) for col in cat_feature_names]

le_dict = {}
for col in cat_feature_names:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    le_dict[col] = le

# Encode target for classification
le_target = LabelEncoder()
train[target] = le_target.fit_transform(train[target])
n_classes = train[target].nunique()

# Quick test to establish baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(train[features], train[target])
probs = dummy.predict_proba(train[features])
top3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
top3_labels = le_target.inverse_transform(top3.ravel()).reshape(top3.shape)
y_true = le_target.inverse_transform(train[target])
dummy_score = mapk(y_true, [list(row) for row in top3_labels], k=3)
print("Dummy Classifier MAP@3 Score:", dummy_score)

if tune:
    # Optuna Study (25 trials)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=25, show_progress_bar=True)
    best_params = study.best_trial.params
    print("Best trial:", study.best_trial)

else:
    # Best tuned params
    best_params = {'max_iter': 713, 'max_leaf_nodes': 86, 'learning_rate': 0.1696077178349903, 'max_depth': 4, 'l2_regularization': 1.9554295990246628, 'max_bins': 148}
    
hgbt_model = HistGradientBoostingClassifier(**best_params, random_state=42)
hgbt_model.fit(train[features], train[target])

catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.15, depth=7, verbose=0, random_seed=42)
catboost_model.fit(train[features], train[target], cat_features=cat_features)

# Predict on Test Data
hgbt_probs = hgbt_model.predict_proba(test[features])
catboost_probs = catboost_model.predict_proba(test[features])

blend_weight = 0.7  # 70% HGBT, 30% CatBoost

ensemble_probs = blend_weight * hgbt_probs + (1 - blend_weight) * catboost_probs
top_3 = ensemble_probs.argsort(axis=1)[:, -3:][:, ::-1] # Get top 3 predictions per row
top_3_labels = le_target.inverse_transform(top_3.ravel()).reshape(top_3.shape)


# Format predictions for submission
submission['Fertilizer Name'] = [' '.join(row) for row in top_3_labels]
submission.to_csv('submission.csv', index=False)
display(submission)
print('Submission file saved.')

