import os

import pandas as pd
import numpy as  np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original_df = pd.read_csv('/kaggle/input/extrovert-introvert-dataset/personality_datasert.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


num_col = [col for col in train_df.select_dtypes(['number']).columns if col!='id']
cat_col = [col for col in train_df.select_dtypes(['object']).columns if col!='Personality']
target_col = "Personality"
id_col = 'id'


combined_df = pd.concat([train_df.drop(columns=id_col, axis=1),original_df], ignore_index=True).drop_duplicates()


le = LabelEncoder()
tgt_le = LabelEncoder()


non_impute_columns = [target_col]
features = combined_df.drop(columns=non_impute_columns)


encoders = {}

for col in cat_col:
    temp_col = combined_df[col].fillna("NaN_Placeholder")
    combined_df[col] = le.fit_transform(temp_col)  # Encode the column
    combined_df[col] = combined_df[col].where(temp_col != "NaN_Placeholder", np.nan) # Restore NaN values
    encoders[col] = {cls: le.transform([cls])[0] for cls in le.classes_ if cls != "NaN_Placeholder"}


numerical_columns = list(combined_df.select_dtypes('number').columns)


# Impute numerical columns
imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), random_state=42)
combined_df[numerical_columns] = imputer.fit_transform(combined_df[numerical_columns])


target_col = 'Personality'
tgt_le = LabelEncoder()

X = combined_df.drop(target_col, axis=1)
y = combined_df[target_col]
y_encoded = tgt_le.fit_transform(y)


params = {
    "n_estimators": 100,
    "learning_rate": 0.01,
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "subsample": 0.8,
    "random_state": 42
}


gbc = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)



def oof_gradient_boosting_with_best_model(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    feature_importances = np.zeros(X.shape[1])
    fold_scores = []
    fold_models = []  # To store models for each fold

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold+1}/{n_splits}...")
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        oof_preds[valid_idx] = preds
        fold_score = np.sqrt(mean_squared_error(y_valid, preds))
        fold_scores.append(fold_score)
        fold_models.append(model)  # Save the trained model for the fold

        # Accumulate feature importance
        feature_importances += model.feature_importances_ / n_splits

    best_fold_idx = np.argmin(fold_scores)  # Find the fold with the lowest RMSE
    best_model = fold_models[best_fold_idx]  # Retrieve the best model

    print(f"Average CV RMSE: {np.mean(fold_scores):.4f}")
    print(f"Best Fold: {best_fold_idx+1}, RMSE: {fold_scores[best_fold_idx]:.4f}")
    return oof_preds, feature_importances, best_model


oof_preds, feature_importances, best_model = oof_gradient_boosting_with_best_model(gbc, X.values, y_encoded)


final_predictions = best_model.predict(X)


accuracy_score(y_true=y_encoded,y_pred=final_predictions)

