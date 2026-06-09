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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')


# ----------------------------
# Load data
# ----------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

train.head()


#Checking Correlation among columns
# Calculate the correlation matrix for all numerical columns
correlation_matrix = train.select_dtypes(include = ['float64','int64']).corr()

# Create the heatmap using seaborn
# Define the plot size for better readability
plt.figure(figsize=(10, 8))

# Use sns.heatmap() to create the visualization
sns.heatmap(
    correlation_matrix,
    annot=True,          # Display the correlation values on the heatmap
    cmap='coolwarm',     # Choose a divergent colormap for better visual distinction
    vmin=-1,             # Set the minimum color value to -1
    vmax=1,              # Set the maximum color value to 1
    fmt=".2f",           # Format the annotations to 2 decimal places
    linewidths=.5        # Add lines between cells for clarity
)

# 4. Add titles and show the plot
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.show()


y = train['loan_paid_back']
X = train.drop(columns=["id", "loan_paid_back"]).copy()
X_test = test.drop(columns=["id"]).copy()


# Select only numerical columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Plot histograms for each numerical column
train[numerical_cols].hist(figsize=(15, 12), bins=20, edgecolor='black')
plt.tight_layout() # Adjusts subplot params so that subplots are nicely fit
plt.show()


# Assuming 'train' is your DataFrame
# Define your target variable name
target_col = 'loan_paid_back' # Replace with your actual target column name

# Identify all numerical columns, excluding the target column itself if it's numerical
numerical_cols = train.select_dtypes(include=['number']).columns.tolist()
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

# Set up the plotting area (adjust figsize and layout as needed)
# Determine grid size dynamically based on the number of features
num_features = len(numerical_cols)
if num_features == 0:
    print("No numerical columns found to plot.")
else:
    # Calculate rows and columns for subplot grid
    num_cols_plot = 3
    num_rows_plot = int(np.ceil(num_features / num_cols_plot))

    fig, axes = plt.subplots(num_rows_plot, num_cols_plot, figsize=(18, 5 * num_rows_plot))
    # Flatten axes array for easier iteration if we have more than 1 row
    axes = axes.flatten()

    # Loop through each numerical column and plot
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        sns.histplot(
            data=train,
            x=col,
            hue=target_col,
            multiple="layer",
            alpha=0.6,
            bins=30,
            kde=True,
            ax=ax # Assigns the plot to the correct subplot axis
        )
        ax.set_title(f'Distribution of {col} by {target_col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend(title=target_col)

    # Hide any unused subplots if the number of features doesn't fill the grid perfectly
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


#Feature Engineering

X["loan_to_income"] = X["loan_amount"] /X['annual_income']
X_test["loan_to_income"] = X_test["loan_amount"] /X_test['annual_income']


X["income_per_credit"] = X["annual_income"] /X['credit_score']
X_test["income_per_credit"] = X_test["annual_income"] /X_test['credit_score']


X["loan_to_credit_ratio"] = X["loan_amount"] /X['credit_score']
X_test["loan_to_credit_ratio"] = X_test["loan_amount"] /X_test['credit_score']


# Apply log1p transformation to create a new column
X['annual_income_LogTransformed'] = np.log1p(X['annual_income'])
X['loan_amount_LogTransformed'] = np.log1p(X['loan_amount'])

X_test['annual_income_LogTransformed'] = np.log1p(X_test['annual_income'])
X_test['loan_amount_LogTransformed'] = np.log1p(X_test['loan_amount'])

X['debt_to_income_ratio'] = X['debt_to_income_ratio'].clip(upper=0.5)
X_test['debt_to_income_ratio'] = X_test['debt_to_income_ratio'].clip(upper=0.5)





# We maintain the same class proportion of the target (loan_paid_back) in both splits.
# Since the target is imbalanced (~80% paid back, ~20% not), stratification ensures the test set reflects this 
# same 80/20 distribution.

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# --- 2. Simple One-Hot Encoding ---
# Identify categorical features
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Apply one-hot encoding using pandas get_dummies
    # drop_first=True helps avoid multi-collinearity in linear models
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)


# Align columns to ensure the test set has the same columns as the training set
X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)


#Dropping columns whose correlation is more that 0.9
X_train_encoded_dropped = X_train_encoded.drop(columns=['annual_income','loan_amount'])
X_val_encoded_dropped = X_val_encoded.drop(columns=['annual_income','loan_amount'])
X_test_encoded_dropped = X_test_encoded.drop(columns=['annual_income','loan_amount'])


#Identifying multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 2. Calculate VIF for each feature ---
# The function requires adding a constant term for the intercept
#X_train_encoded = sm.add_constant(X)
X_vif = X_train_encoded_dropped.select_dtypes(include=[np.number])

vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_data.sort_values(by="VIF", ascending=False).head(10)


drop_cols = [
    'annual_income_LogTransformed',
    'loan_amount_LogTransformed',
    'income_per_credit',
    'loan_to_credit_ratio'
]
X_train_vif_cleaned = X_train_encoded_dropped.drop(columns=drop_cols, errors='ignore')
X_val_vif_cleaned = X_val_encoded_dropped.drop(columns=drop_cols, errors='ignore')
X_test_vif_cleaned = X_test_encoded_dropped.drop(columns=drop_cols, errors='ignore')

X_test_vif_cleaned = X_test_encoded.reindex(columns=X_train_vif_cleaned.columns, fill_value=0)






# Combine train + val for full training after CV
X_full = pd.concat([X_train_vif_cleaned,X_val_vif_cleaned], axis=0).reset_index(drop=True)
y_full = pd.concat([y_train,y_val], axis=0).reset_index(drop=True)

# Test set
X_test = X_test_vif_cleaned.copy()

# -----------------------
# CONFIG 
# -----------------------
CV_FOLDS = 5        # set to 50 for your final runs, 5 for fast testing
OPTUNA_TRIALS = 50   # set to e.g. 50 or 100 for real runs; 2 for quick dev
RANDOM_STATE = 42
EPOCHS = 50

# make randomness reproducible (best-effort)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)



# -------------------------------
# Optuna Objective Function
# -------------------------------
def objective(trial, model_name, X,y,n_splits = CV_FOLDS):
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state= RANDOM_STATE)
    auc_scores=[]

   
    params = {
        'iterations': trial.suggest_int('iterations', 850, 950),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 2.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'verbose': False,
        'thread_count': -1
    }
    model = ctb.CatBoostClassifier(**params)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
       
        model = ctb.CatBoostClassifier(**params, random_state=RANDOM_STATE)

        #fit
        model.fit(X_tr, y_tr)
        pred_proba = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, pred_proba)
        auc_scores.append(auc)
   
    return float(np.mean(auc_scores))


# -------------------------------
# Tune Each Model with Optuna
# -------------------------------
def tune_model(model_name, X, y, n_trials = OPTUNA_TRIALS): 
    print(f"\nTuning {model_name.upper()} (CV folds={CV_FOLDS}) with {n_trials} Optuna trials...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed = RANDOM_STATE))
    objective_func = lambda trial: objective(trial, model_name, X, y, n_splits=CV_FOLDS)
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)

    print(f"Best AUC for {model_name}: {study.best_value:.5f}")
    return study.best_params




# Generate OOF predictions
def generate_oof_predictions(model_name, best_params, X,y,X_test,n_splits = CV_FOLDS):
    skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state = RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds_folds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X,y)):
        X_tr,X_va = X.iloc[train_idx],X.iloc[val_idx]
        y_tr,y_va = y.iloc[train_idx],y.iloc[val_idx]

        if model_name == 'cat':
            # catboost prefers 'random_seed' param; ensure we pass that if available
            bp = best_params.copy()
            if 'random_state' in bp:
                bp['random_seed'] = bp.pop('random_state')
            model = ctb.CatBoostClassifier(**bp, verbose=False)

        model.fit(X_tr,y_tr)
        oof_preds[val_idx] = model.predict_proba(X_va)[:,1]
        test_preds_folds.append(model.predict_proba(X_test)[:,1])

    avg_test_preds = np.mean(test_preds_folds,axis=0)

    return oof_preds, avg_test_preds




# -----------------------
# Neural net OOF generator (Keras MLP)
# -----------------------

def generate_oof_nn(X, y, X_test, n_splits=CV_FOLDS, epochs= EPOCHS, batch_size=2048):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X))
    test_fold_preds = []

    # scale features (use same scaling per fold fitted on train)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, X_va_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)
        X_test_scaled = scaler.transform(X_test)

        # build model
        inp_dim = X_tr.shape[1]
        model = keras.Sequential([
            layers.Input(shape=(inp_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss='binary_crossentropy')

        # early stopping to avoid long train on noisy residuals
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

        model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                  epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

        oof[val_idx] = model.predict(X_va).reshape(-1)
        test_fold_preds.append(model.predict(X_test_scaled).reshape(-1))

    test_preds = np.mean(test_fold_preds, axis=0)
    return oof, test_preds
        



# Tune all three models
best_params_cat = tune_model('cat', X_full, y_full, n_trials=OPTUNA_TRIALS)


# Run the OOF generation function for all models


oof_cat, test_cat = generate_oof_predictions('cat', best_params_cat, X_full, y_full, X_test, n_splits=CV_FOLDS)


# 3) Neural Net OOF (Keras)
oof_nn, test_nn = generate_oof_nn(X_full, y_full, X_test, n_splits=CV_FOLDS, epochs= EPOCHS)


# quick sanity prints
print("OOF shapes:",  oof_cat.shape, oof_nn.shape)
print("Test pred shapes:",  test_cat.shape, test_nn.shape)



#Learn dynamic ensemble weights using Logistic Regression (Level-1 blender)

meta_X_train = np.vstack([oof_cat, oof_nn]).T
meta_X_test = np.vstack([test_cat, test_nn]).T


from sklearn.linear_model import LogisticRegression

meta_model = LogisticRegression()
meta_model.fit(meta_X_train,y_full)

pred0_train = meta_model.predict_proba(meta_X_train)[:,1]
pred0_test = meta_model.predict_proba(meta_X_test)[:,1]



# -------------------------------
# Step 2: Boost over Residuals
# new_target = pred0 - original_target
# But for classification: we train on residual = y - pred0 (for probability calibration)
# -------------------------------
residual = y_full - pred0_train  # Note: y_full is 0/1, pred0_train is [0,1]

# Train a booster on residuals (e.g., another XGBoost regressor or classifier)
print("\nBoosting over residuals...")

# We'll use XGBoost Regressor on residuals (common in gradient boosting over probs)
residual_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    tree_method='hist'
)

residual_model.fit(X_full, residual)

# Correction on train and test
residual_correction_train = residual_model.predict(X_full)
residual_correction_test = residual_model.predict(X_test)

# Final prediction = pred0 + correction
final_pred_train = pred0_train + residual_correction_train
final_pred_test = pred0_test + residual_correction_test

# Clip to [0,1]
final_pred_train = np.clip(final_pred_train, 0, 1)
final_pred_test = np.clip(final_pred_test, 0, 1)


# -------------------------------
# Final Evals & Output
# -------------------------------
print(f"\nFinal Model AUC (after residual boost): {roc_auc_score(y_full, final_pred_train):.5f}")

# Save predictions
pred0 = pred0_test  # as requested
final_predictions = final_pred_test

# Example: Save to CSV
submission = pd.DataFrame({
    'id': test["id"],
    'loan_paid_back': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nPredictions saved to 'submission.csv'")


# joblib.dump(xgb_model, "xgb_final.pkl")
# joblib.dump(lgb_model, "lgbm_final.pkl")
# joblib.dump(cat_model, "catboost_final.pkl")
# joblib.dump(residual_model, "residual_corrector.pkl")

# np.save("pred0_train.npy", pred0_train)
# np.save("final_pred_train.npy", final_pred_train)




