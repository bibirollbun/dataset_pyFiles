import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import optuna


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
train_df


train_df.info()


train_df.describe()


numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()[1:]
print(numerical_cols)


categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_cols)


def plot_features_automatically(df, target_col='loan_paid_back'):
    # Identify feature types
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Remove target from numerical cols
    numerical_cols = [col for col in numerical_cols if col != target_col]
    
    print(f"Plotting {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features")
    
    # Plot numerical features
    for col in numerical_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.histplot(data=df, x=col, kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {col}')
        
        sns.boxplot(data=df, x=target_col, y=col, ax=ax2)
        ax2.set_title(f'{col} vs {target_col}')
        
        plt.tight_layout()
        plt.show()
    
    # Plot categorical features
    for col in categorical_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.countplot(data=df, x=col, ax=ax1)
        ax1.set_title(f'Distribution of {col}')
        ax1.tick_params(axis='x', rotation=45)
        
        sns.countplot(data=df, x=target_col, hue=col, ax=ax2)
        ax2.set_title(f'{target_col} by {col}')
        ax2.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

# Usage
plot_features_automatically(train_df, 'loan_paid_back')


train_df_2 = train_df[(train_df['annual_income'] < 200000) & 
        (train_df['debt_to_income_ratio'] < 0.4) &
        (train_df['credit_score'] < 500) &
        (train_df['loan_amount'] < 35000) &
        (train_df['interest_rate'] > 6) & (train_df['interest_rate'] < 19)] 


# Usage
plot_features_automatically(train_df_2, 'loan_paid_back')


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'n_estimators': 10000,
    'early_stopping_rounds': 50,
    'learning_rate': 0.01,
    'random_state': 42,
    'n_jobs': -1,
    'enable_categorical': True,
}


train_df.columns


y = train_df['loan_paid_back']
X = train_df.drop(columns=['id', 'loan_paid_back'])


# 1. Define an objective function to be maximized.
def objective(trial):

    X[categorical_cols] = X[categorical_cols].astype('category')
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2)
    dtrain = xgb.DMatrix(train_x, label=train_y, enable_categorical=True)
    dvalid = xgb.DMatrix(valid_x, label=valid_y, enable_categorical=True)


    # 2. Suggest values of the hyperparameters using a trial object.
    param = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "early_stopping_rounds": 50,
        "learning_rate":trial.suggest_float("colsample_bytree", 0.001, 0.1),
        'random_state': 42,
        'n_jobs': -1,
        'enable_categorical': True,
    }

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    score = roc_auc_score(valid_y, pred_labels)
    return score

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10000)

# Get the best parameters in a more structured way
best_param = study.best_trial.params


test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test = test_df.copy()
test.drop(columns='id', inplace=True)


test


# Assuming you have these defined
# X, y, test, CATS (list of categorical columns), params

N_SPLITS = 5
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

# Initialize KFold
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- Fold {fold}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Convert categorical columns - do this BEFORE splitting to avoid data leakage
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    test_cat = test.copy()
    
    for col in categorical_cols:
        # Ensure all category values are seen in training data
        X_train_cat[col] = X_train_cat[col].astype('category')
        
        # For validation and test, use categories from training
        all_categories = X_train_cat[col].cat.categories
        X_val_cat[col] = pd.Categorical(X_val_cat[col], categories=all_categories)
        test_cat[col] = pd.Categorical(test_cat[col], categories=all_categories)

    # Initialize model with your parameters
    model = xgb.XGBClassifier(**best_param)
    
    # Train with early stopping
    model.fit(
        X_train_cat, y_train,
        eval_set=[(X_val_cat, y_val)],
        verbose=100  # Reduced from 1000 to see more frequent updates
    )

    # Get predictions
    val_preds = model.predict_proba(X_val_cat)[:, 1]
    oof_preds[val_idx] = val_preds
    
    # Test predictions for this fold
    fold_test_preds = model.predict_proba(test_cat)[:, 1]
    test_preds += fold_test_preds / N_SPLITS  # Average across folds
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'Fold {fold} AUC: {fold_score:.4f}')
    print(f'Best iteration: {model.best_iteration}')
    print()

# Calculate overall OOF score
overall_auc = roc_auc_score(y, oof_preds)
print(f'====================')
print(f'Overall OOF AUC: {overall_auc:.4f}')
print(f'====================')

# Optional: Save OOF predictions and test predictions
oof_df = pd.DataFrame({
    'true_target': y,
    'oof_preds': oof_preds
})

test_predictions_df = pd.DataFrame({
    'preds': test_preds
})


test_predictions_df = pd.concat([test_df['id'], test_predictions_df], axis=1)


test_predictions_df


test_predictions_df.to_csv('submission.csv', index=False)


feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
    'feature': X_train.columns, 
    'importance': feature_importances
})

importance_df = importance_df.sort_values('importance', ascending=False)


importance_df




