import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")



# Outlier capping using IQR
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

for col in ['Duration', 'Heart_Rate', 'Calories']:
    train = cap_outliers(train, col)


# Feature Engineering
train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
test['BMI'] = test['Weight'] / ((test['Height'] / 100) ** 2)
train['Duration_Heart_Rate'] = train['Duration'] * train['Heart_Rate']
test['Duration_Heart_Rate'] = test['Duration'] * test['Heart_Rate']
train['Exercise_Intensity'] = train['Heart_Rate'] / train['Age']
test['Exercise_Intensity'] = test['Heart_Rate'] / test['Age']
train['Log_Body_Temp'] = np.log1p(train['Body_Temp'])
test['Log_Body_Temp'] = np.log1p(test['Body_Temp'])
train['Log_Duration'] = np.log1p(train['Duration'])
test['Log_Duration'] = np.log1p(test['Duration'])
train['Log_Heart_Rate'] = np.log1p(train['Heart_Rate'])
test['Log_Heart_Rate'] = np.log1p(test['Heart_Rate'])
train['Weight_Body_Temp'] = train['Weight'] * train['Body_Temp']
test['Weight_Body_Temp'] = test['Weight'] * test['Body_Temp']
train['Heart_Rate_Zone'] = pd.cut(train['Heart_Rate'], bins=[0, 100, 130, 160, 220], labels=[0, 1, 2, 3])
test['Heart_Rate_Zone'] = pd.cut(test['Heart_Rate'], bins=[0, 100, 130, 160, 220], labels=[0, 1, 2, 3])
train['Heart_Rate_Zone'] = train['Heart_Rate_Zone'].astype(float)
test['Heart_Rate_Zone'] = test['Heart_Rate_Zone'].astype(float)
train['BMI_Exercise_Intensity'] = train['BMI'] * train['Exercise_Intensity']
test['BMI_Exercise_Intensity'] = test['BMI'] * test['Exercise_Intensity']
# Additional features
train['Age_Body_Temp'] = train['Age'] * train['Body_Temp']
test['Age_Body_Temp'] = test['Age'] * test['Body_Temp']
train['Heart_Rate_Variability_Proxy'] = train['Heart_Rate'] / (train['Duration'] + 1)  # Proxy for variability
test['Heart_Rate_Variability_Proxy'] = test['Heart_Rate'] / (test['Duration'] + 1)
train['Log_BMI'] = np.log1p(train['BMI'])
test['Log_BMI'] = np.log1p(test['BMI'])

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(train[['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'BMI']])
poly_columns = [f'poly_{i}' for i in range(poly_features.shape[1])]
train[poly_columns] = poly_features
poly_features_test = poly.transform(test[['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'BMI']])
test[poly_columns] = poly_features_test




# Label Encoding
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

# PCA for Height and Weight
pca = PCA(n_components=1)
train['PCA_Height_Weight'] = pca.fit_transform(train[['Height', 'Weight']])
test['PCA_Height_Weight'] = pca.transform(test[['Height', 'Weight']])

# Prepare features and target
X = train.drop(columns=["id", "Calories", "Height", "Weight"])
y = np.log1p(train["Calories"])  # Log-transform target
X_test = test.drop(columns=["id", "Height", "Weight"])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Bin Calories for StratifiedKFold
y_bins = pd.qcut(train['Calories'], q=5, labels=False)

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)




# Define an improved neural network model
def create_model(input_dim):
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model


# Train neural network
oof_preds_nn = np.zeros(len(X))
test_preds_nn = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins)):
    print(f"\nFold {fold + 1} (Neural Network)")
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = create_model(X_scaled.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=1024,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    val_preds = model.predict(X_val, verbose=0).flatten()
    oof_preds_nn[val_idx] = val_preds
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold + 1} RMSE (NN): {fold_rmse:.4f}")
    
    test_preds_nn += model.predict(X_test_scaled, verbose=0).flatten() / kf.n_splits


# Train gradient boosting models
def train_model(ModelClass, model_name, X, y, X_test, kf, **params):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    print(f"\nTraining {model_name}...")
    feature_importances = np.zeros(X.shape[1])
    
    for fold, (train_idx, valid_idx) in tqdm(enumerate(kf.split(X, y_bins)), total=kf.get_n_splits(), desc=f"{model_name} Folds"):
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        
        if model_name == "LightGBM":
            model = ModelClass(device='gpu', **params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse", callbacks=[lgb.early_stopping(100)])
        elif model_name == "CatBoost":
            model = ModelClass(task_type='GPU', **params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        else:  # XGBoost
            model = ModelClass(tree_method='gpu_hist', eval_metric="rmse", **params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        oof_preds[valid_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / kf.n_splits
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[valid_idx]))
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")
        
        # Collect feature importances
        if model_name == "LightGBM":
            feature_importances += model.feature_importances_ / kf.n_splits
        elif model_name == "CatBoost":
            feature_importances += model.get_feature_importance() / kf.n_splits
        else:
            feature_importances += model.feature_importances_ / kf.n_splits
    
    full_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"{model_name} CV RMSE: {full_rmse:.4f}")
    return oof_preds, test_preds, feature_importances


# Optimized parameters for gradient boosting models (fine-tuned)
lgb_params = {'n_estimators': 1500, 'learning_rate': 0.008, 'num_leaves': 50, 'max_depth': 12, 'subsample': 0.85, 'colsample_bytree': 0.75, 'random_state': 42}
cat_params = {'iterations': 1500, 'learning_rate': 0.008, 'depth': 10, 'l2_leaf_reg': 3, 'random_seed': 42, 'verbose': 0}
xgb_params = {'n_estimators': 1500, 'learning_rate': 0.008, 'max_depth': 12, 'subsample': 0.85, 'colsample_bytree': 0.75, 'random_state': 42}

oof_lgb, test_lgb, lgb_importances = train_model(LGBMRegressor, "LightGBM", X_scaled, y, X_test_scaled, kf, **lgb_params)
oof_cat, test_cat, cat_importances = train_model(CatBoostRegressor, "CatBoost", X_scaled, y, X_test_scaled, kf, **cat_params)
oof_xgb, test_xgb, xgb_importances = train_model(XGBRegressor, "XGBoost", X_scaled, y, X_test_scaled, kf, **xgb_params)





# Feature Selection based on average importance
feature_names = X.columns
avg_importances = (lgb_importances + cat_importances + xgb_importances) / 3
importance_df = pd.DataFrame({'feature': feature_names, 'importance': avg_importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
top_features = importance_df['feature'].head(25).values  # Select top 25 features



# Rebuild datasets with selected features
X_selected = X[top_features]
X_test_selected = X_test[top_features]
X_scaled_selected = scaler.fit_transform(X_selected)
X_test_scaled_selected = scaler.transform(X_test_selected)




# Retrain models with selected features
oof_lgb, test_lgb, _ = train_model(LGBMRegressor, "LightGBM", X_scaled_selected, y, X_test_scaled_selected, kf, **lgb_params)
oof_cat, test_cat, _ = train_model(CatBoostRegressor, "CatBoost", X_scaled_selected, y, X_test_scaled_selected, kf, **cat_params)
oof_xgb, test_xgb, _ = train_model(XGBRegressor, "XGBoost", X_scaled_selected, y, X_test_scaled_selected, kf, **xgb_params)



# Stacking ensemble for gradient boosting models
base_models = [
    ('lgb', LGBMRegressor(device='gpu', **lgb_params)),
    ('cat', CatBoostRegressor(task_type='GPU', **cat_params)),
    ('xgb', XGBRegressor(tree_method='gpu_hist', eval_metric="rmse", **xgb_params))
]
stacking_model = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0), cv=5)
stacking_model.fit(X_scaled_selected, y)
test_preds_stacking = stacking_model.predict(X_test_scaled_selected)



# Meta-blending: Use a simple model to learn optimal weights
meta_X = np.column_stack((oof_lgb, oof_cat, oof_xgb, oof_preds_nn))
meta_X_test = np.column_stack((test_lgb, test_cat, test_xgb, test_preds_nn))
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X, y)
test_preds_final = meta_model.predict(meta_X_test)



# Clip predictions to avoid extreme values
test_preds_final = np.clip(test_preds_final, y.min(), y.max())
submission['Calories'] = np.expm1(test_preds_final)
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nSubmission saved.")

