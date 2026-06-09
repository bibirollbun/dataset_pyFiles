import pandas as pd

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
train.head()


test.head()


print(train.isna().sum())
print(test.isna().sum())


print("\nTrain Data Statistical Summary:")
# Dropping 'id' as it's just an identifier
print(train.drop('id', axis=1).describe())


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 6))
sns.countplot(data=train, y='Personality', order=train['Personality'].value_counts().index, palette='viridis', hue='Personality')
plt.title('Distribution of Personality Types')
plt.xlabel('Count')
plt.ylabel('Personality Type')
plt.show()

print("\nPersonality Counts:")
print(train['Personality'].value_counts())


features = train.columns.drop(['id', 'Personality'])

train[features].hist(figsize=(16, 12), bins=20, color='skyblue', edgecolor='black', grid=False)
plt.suptitle('Distribution of All Features', y=1.02, fontsize=20)
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np

# --- Step 1: Define the mapping for 'Yes'/'No' columns ---
binary_map = {'Yes': 1, 'No': 0}

# List of columns to apply the mapping to
categorical_cols = ['Stage_fear', 'Drained_after_socializing']


# --- Step 2: Apply mapping to both train and test data ---
for col in categorical_cols:
    train[col] = train[col].map(binary_map)
    # Use .get() for test data in case a column doesn't exist, though it should
    if col in test.columns:
        test[col] = test[col].map(binary_map)

        
# --- Step 3: Impute missing values with the median ---
# Get all columns that are not the 'id' or the target 'Personality'
features_to_impute = train.columns.drop(['id', 'Personality'])

# Calculate medians from the TRAINING data only
medians = train[features_to_impute].median()

# Fill NaN in both train and test sets using the calculated medians
for col in features_to_impute:
    train[col].fillna(medians[col], inplace=True)
    if col in test.columns:
        test[col].fillna(medians[col], inplace=True)


# --- Step 4: Verify the changes ---
print("--- Cleaned Train Data Info ---")
train.info()

print("\n--- Missing Values After Cleaning ---")
print(train.isnull().sum())

print("\n--- Data Head After Cleaning ---")
print(train.head())


# --- Bivariate Analysis: Features vs. Target ---
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 22))
axes = axes.flatten() # Flatten the 2D array of axes to a 1D array

# Define the features to plot
features = train.columns.drop(['id', 'Personality'])

for i, col in enumerate(features):
    # CORRECTED LINE: Removed the redundant 'hue' parameter
    sns.boxplot(x='Personality', y=col, data=train, ax=axes[i], palette='plasma', hue='Personality')
    axes[i].set_title(f'{col} by Personality', fontsize=14)
    axes[i].tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability

# Remove the last unused subplot if the number of features is odd
if len(features) % 2 != 0:
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.suptitle('Feature Distributions Across Personality Types', fontsize=20, y=1.03)
plt.show()


# --- Correlation Heatmap ---
# First, let's convert the 'Personality' target to a numerical format for the correlation calculation
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_corr = train.copy()
train_corr['Personality'] = label_encoder.fit_transform(train_corr['Personality'])

plt.figure(figsize=(12, 10))
correlation_matrix = train_corr.drop('id', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Features and Target')
plt.show()


# Visualizing pairplot of 'df_train'
sns.pairplot(train.drop(['Stage_fear', 'Drained_after_socializing'], axis=1), hue='Personality', corner=True)
plt.suptitle('Pairplot of Training Data')
plt.show()


# Feature engineering, creating new features (columns) that represent interactions
def feature_engineering(df):

    # Interactions between extroverted activities
    df['Social_x_Outside'] = df['Social_event_attendance'] * df['Going_outside']
    df['Social_x_Friends'] = df['Social_event_attendance'] * df['Friends_circle_size']
    df['Social_x_Posts'] = df['Social_event_attendance'] * df['Post_frequency']
    df['Outside_x_Friends'] = df['Going_outside'] * df['Friends_circle_size']
    df['Outside_x_Posts'] = df['Going_outside'] * df['Post_frequency']
    df['Friends_x_Posts'] = df['Friends_circle_size'] * df['Post_frequency']

    # Interactions between alone time and extroverted activites
    df['Alone_x_Social'] = df['Time_spent_Alone'] * df['Social_event_attendance']
    df['Alone_x_Outside'] = df['Time_spent_Alone'] * df['Going_outside']
    df['Alone_x_Friends'] = df['Time_spent_Alone'] * df['Friends_circle_size']
    df['Alone_x_Posts'] = df['Time_spent_Alone'] * df['Post_frequency']
    
# Applying function to training and testing sets
feature_engineering(train)
feature_engineering(test)


train.info()


run = 0

if run == 1:
    # Reading in a clean version of the original dataset
    df_original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")

    # Applying encoding steps to the original dataset
    df_original['Personality'].replace({'Extrovert': 1, 'Introvert': 0}, inplace=True)
    df_original = pd.get_dummies(df_original, columns=['Stage_fear', 'Drained_after_socializing'])

    # Applying feature engineering to original dataset
    feature_engineering(df_original)
    
    # Combining training data and original data
    df_combined = pd.concat([train, df_original], axis=0, ignore_index=True)
    
    # Creating final versions of each cleaned dataset
    train = df_combined.copy()
    test = test.copy()


run = 1

if run == 1:
    # Creating final versions of each cleaned dataset (NOT INCLUDING ORIGINAL)
    train = train.copy()
    test = test.copy()


train.head()


train['Personality'].replace({'Extrovert': 1, 'Introvert': 0}, inplace=True)


Introvert = (train['Personality'] == 0).sum()
Extrovert = (train['Personality'] == 1).sum()
scale_pos_weight = Introvert / Extrovert
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


ID = test['id']
X_train = train.drop(['Personality', 'id'], axis=1)
y_train = train['Personality']
X_test = test.drop(['id'], axis=1)


# LightGBM
import lightgbm
from lightgbm import LGBMClassifier, plot_importance

# Optuna
import optuna
from optuna.samplers import TPESampler

# XGBoost
from xgboost import XGBClassifier

# Tqdm
from tqdm import tqdm

# Sklearn
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier



def objective(trial):

    imbalance_strategy = trial.suggest_categorical("imbalance_strategy", ["scale_pos_weight", "is_unbalance"])
    
    params = {
        'objective': 'binary',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 142,
        'num_leaves': trial.suggest_int('num_leaves', 20, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'subsample': trial.suggest_float('subsample', 0.25, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0)
    }

    # Applying imbalance handling
    if imbalance_strategy == "scale_pos_weight":
        params['scale_pos_weight'] = scale_pos_weight
    else:
        params['is_unbalance'] = True
    
    # Fitting LGBM model with parameters from the trials
    model = LGBMClassifier(**params)
    # Stratified sampling 
    cv = StratifiedKFold(5, shuffle=True, random_state=142)
    cv_splits = cv.split(X_train, y_train)
    
    # Creating empty scores list to hold AUC scores from each trialed model
    scores = []
    for train_idx, val_idx in cv_splits:
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold)
        y_pred_acc = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred_acc)
        scores.append(score)
    
    # Printing and returning mean AUC scores
    mean_score = np.mean(scores)
    print(f"Mean Accuracy Score = {mean_score:.5f}")
    return mean_score

# When set to 1, optuna will create a study to find the optimal parameters
run = 1

if run == 1:
    
    # Each optuna study uses an independent sampler with a TPE algorithm
    # For each trial, the TPE essentially uses Gaussian Mixture Models to identify the optimal parameter value
    study = optuna.create_study(sampler=TPESampler(n_startup_trials=30, multivariate=True, seed=42), direction="maximize")
    study.optimize(objective, n_trials=200)
    print('Best value:', study.best_value)
    print('Best trial:', study.best_trial.params)


best_params = {'objective': 'binary',
                 'verbosity': -1,
                 'boosting_type': 'gbdt',
                 'random_state': 42,
                 **study.best_trial.params}


final_model = LGBMClassifier(**best_params)
final_model.fit(X_train, y_train)


cv = StratifiedKFold(7, shuffle=True, random_state=142)
cv_splits = tqdm(cv.split(X_train, y_train), total=cv.get_n_splits(), desc='CV Progress')

scores = []
for train_idx, val_idx in cv_splits:
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    final_model.fit(X_train_fold, y_train_fold)
    y_pred = final_model.predict(X_val_fold)
    score = accuracy_score(y_val_fold, y_pred)
    scores.append(score)
    
    print(f'score: {score:.5f}')

print(f"Mean Score ＝ {np.mean(scores):.5f}") 


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='r')
plt.title("Accuracy Scores Across Folds", fontsize=14)
plt.xlabel("Fold", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.grid(True)
plt.xticks(range(1, len(scores) + 1))
plt.show()


y_pred = final_model.predict(X_val_fold)
cm = confusion_matrix(y_val_fold, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


plot_importance(final_model)


# Catboost
import catboost
from catboost import Pool, CatBoostClassifier
from catboost.utils import eval_metric


# Splitting 'train' dataset into features (X) and target (y)
X = train.drop(['Personality', 'id'], axis=1)
y = train['Personality']
# Splitting data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Setting up ensemble of models (XGBoost, CatBoost, LGBM)
xgb = XGBClassifier(
    max_depth=4,         
    learning_rate=0.01,   
    n_estimators=1000,    
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)

cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    class_weights=[scale_pos_weight, 1],
    random_seed=42,
    verbose=0)

lgbm = LGBMClassifier(**best_params)

# Creating ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lgbm', lgbm)],
    voting='soft')

# Fitting ensemble model to training data
ensemble.fit(X_train, y_train)


val_probs = ensemble.predict_proba(X_valid)[:, 1]
best_threshold = 0.5
best_acc = 0

for threshold in np.arange(0.4, 0.6, 0.01):
    preds = (val_probs >= threshold).astype(int)


print(preds)


probs = ensemble.predict_proba(X_test)[:, 1]

predictions = (probs >= best_threshold).astype(int)

submission = pd.DataFrame({'id': ID, "Personality": predictions})

submission['Personality'].replace({1: 'Extrovert', 0: 'Introvert'}, inplace=True)


submission.to_csv('submission.csv', index=False)

