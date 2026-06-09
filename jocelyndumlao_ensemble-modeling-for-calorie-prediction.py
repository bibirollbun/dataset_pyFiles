import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train.head().style.background_gradient(cmap='gist_rainbow')


test.head().style.background_gradient(cmap='gist_rainbow_r')


print("Train Shape:", train.shape)
print("Test Shape:", test.shape)


print("\nTrain Info:")
train.info()
print("\nTest Info:")
test.info()


print("\nTrain Describe:")
train.describe().style.background_gradient(cmap='GnBu_r')


# Preprocessing
def preprocess(df):
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    return df

train = preprocess(train.copy())
test = preprocess(test.copy())


# Feature Engineering
train['BMI'] = train['Weight'] / (train['Height'] / 100) ** 2
test['BMI'] = test['Weight'] / (test['Height'] / 100) ** 2



# Target Transformation
train['Calories'] = np.log1p(train['Calories'])


# Define Features and Target
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
categorical_features = ['Sex']
features = numerical_features + categorical_features
TARGET = 'Calories'

X = train[features]
y = train[TARGET]
X_test = test[features]



# Define RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))



# Model Training Function
def train_models(X, y, X_test, FOLDS=5):
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    lgb_predictions = np.zeros(len(X_test))
    xgb_predictions = np.zeros(len(X_test))
    cat_predictions = np.zeros(len(X_test))

    lgb_oof_predictions = np.zeros(len(X))
    xgb_oof_predictions = np.zeros(len(X))
    cat_oof_predictions = np.zeros(len(X))

    lgb_rmsle_scores = []
    xgb_rmsle_scores = []
    cat_rmsle_scores = []

    lgb_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # LightGBM
        lgb_model = lgb.LGBMRegressor(objective='regression',
                                      metric='rmse',
                                      n_estimators=500,
                                      learning_rate=0.05,
                                      random_state=42,
                                      n_jobs=-1,
                                      colsample_bytree=0.7,
                                      subsample=0.7,
                                      num_leaves=31)

        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        lgb_val_preds = lgb_model.predict(X_val)
        lgb_oof_predictions[val_idx] = lgb_val_preds
        lgb_rmsle_scores.append(rmsle(y_val, lgb_val_preds))
        lgb_predictions += lgb_model.predict(X_test) / FOLDS

        # XGBoost
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                     eval_metric='rmse',
                                     n_estimators=500,
                                     learning_rate=0.05,
                                     random_state=42,
                                     n_jobs=-1,
                                     colsample_bytree=0.7,
                                     subsample=0.7,
                                     max_depth=5)

        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        xgb_val_preds = xgb_model.predict(X_val)
        xgb_oof_predictions[val_idx] = xgb_val_preds
        xgb_rmsle_scores.append(rmsle(y_val, xgb_val_preds))
        xgb_predictions += xgb_model.predict(X_test) / FOLDS

        # CatBoost
        cat_model = CatBoostRegressor(iterations=500,
                                      learning_rate=0.05,
                                      depth=6,
                                      l2_leaf_reg=3,
                                      loss_function='RMSE',
                                      eval_metric='RMSE',
                                      random_state=42,
                                      verbose=0)

        cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
        cat_val_preds = cat_model.predict(X_val)
        cat_oof_predictions[val_idx] = cat_val_preds
        cat_rmsle_scores.append(rmsle(y_val, cat_val_preds))
        cat_predictions += cat_model.predict(X_test) / FOLDS

    # Print Scores
    print(f"LightGBM RMSLE: {np.mean(lgb_rmsle_scores):.4f}")
    print(f"XGBoost RMSLE: {np.mean(xgb_rmsle_scores):.4f}")
    print(f"CatBoost RMSLE: {np.mean(cat_rmsle_scores):.4f}")
    print(f"OOF LightGBM RMSLE: {rmsle(y, lgb_oof_predictions):.4f}")
    print(f"OOF XGBoost RMSLE: {rmsle(y, xgb_oof_predictions):.4f}")
    print(f"OOF CatBoost RMSLE: {rmsle(y, cat_oof_predictions):.4f}")

    return lgb_predictions, xgb_predictions, cat_predictions, lgb_oof_predictions, xgb_oof_predictions, cat_oof_predictions, lgb_model



# Train and Predict
lgb_preds, xgb_preds, cat_preds, lgb_oof_preds, xgb_oof_preds, cat_oof_preds, lgb_model = train_models(X, y, X_test)


# Ensemble
ensemble_predictions = (lgb_preds + xgb_preds + cat_preds) / 3
ensemble_predictions = np.expm1(ensemble_predictions)
lgb_oof_preds = np.expm1(lgb_oof_preds)
xgb_oof_preds = np.expm1(xgb_oof_preds)
cat_oof_preds = np.expm1(cat_oof_preds)



# Visualization
def plot_feature_importance(model, features, top_n=10):
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance (LightGBM)',fontsize = 14, fontweight = 'bold', color = 'darkblue')
    plt.xlabel('Importance Score',fontsize = 12, fontweight = 'bold', color = 'darkred')
    plt.tight_layout()
    plt.gca().set_facecolor('#f5e4d0')
    plt.show()

plot_feature_importance(lgb.LGBMRegressor(**lgb_model.get_params()).fit(X, y), features)

def plot_oof_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('Actual Calories', fontsize=12, fontweight='bold', color='deeppink')
    plt.ylabel('Predicted Calories',fontsize=12, fontweight='bold', color='crimson')
    plt.title(f'OOF Predictions vs. Actuals ({model_name})',fontsize = 14, fontweight = 'bold', color = 'darkgreen')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.tight_layout()
    plt.gca().set_facecolor('#f5e4d0')
    plt.show()

plot_oof_predictions(np.expm1(y), lgb_oof_preds, 'LightGBM')
plot_oof_predictions(np.expm1(y), xgb_oof_preds, 'XGBoost')
plot_oof_predictions(np.expm1(y), cat_oof_preds, 'CatBoost')

def plot_prediction_distribution(predictions, model_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(predictions, kde=True)
    plt.title(f'Distribution of Predictions ({model_name})', fontsize=14, fontweight='bold', color='red')
    plt.xlabel('Predicted Calories',fontsize=12, fontweight='bold', color='saddlebrown')
    plt.ylabel('Frequency',fontsize=12, fontweight='bold', color='saddlebrown')
    plt.tight_layout()
    plt.gca().set_facecolor('#f5e4d0')
    plt.show()

plot_prediction_distribution(ensemble_predictions, 'Ensemble')



# Final Submission
submission['Calories'] = ensemble_predictions
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")
submission.head()


print(f"\nPredict Mean: {ensemble_predictions.mean():.2f}")
print(f"Predict Median: {np.median(ensemble_predictions):.2f}")




