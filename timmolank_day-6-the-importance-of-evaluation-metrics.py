# scikit-learn core tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

# scikit-learn models 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

#Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader


# XGBoost model
from xgboost import XGBClassifier
import xgboost as xgb

# Lightgbm model
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Catboost model
from catboost import CatBoostClassifier, Pool


import numpy as np 
import pandas as pd 

import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings("ignore")

import re
import string
import optuna
from scipy.sparse import hstack
import shap



train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train_cat = train.copy()
test_cat = test.copy()

cat_features = ['Soil Type','Crop Type']
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


# Check unique values in categorical features
for feature in cat_features:
    print(f"\n{feature} unique values:")
    print(f"Train: {train[feature].unique()}")
    print(f"Test: {test[feature].unique()}")
    print(f"Train count: {train[feature].nunique()}, Test count: {test[feature].nunique()}")


# Check for missing values 
def null_table(training, testing):
    print("Training Data Frame")
    print(pd.isnull(train).sum()) 
    print(" ")
    print("Testing Data Frame")
    print(pd.isnull(test).sum())

null_table(train, test)


def create_engineered_features(df):
    
    df = df.copy()
    
    # Ratio is probably not the right word but i was trying to capture interactions 
    df['Temp_Humid_Ratio'] = df['Temparature'] * (df['Humidity'] + 1e-8)  # Avoid division by zero
    df['Humid_Moist_Ratio'] = df['Humidity'] * (df['Moisture'] + 1e-8)
    df['Temp_Moist_Ratio'] = df['Temparature'] * (df['Moisture'] + 1e-8) 
    df['Temp_Moist_Humid_Ratio'] = df['Temparature'] * df['Humidity'] * (df['Moisture'] + 1e-8) 

    df['Nitro_Potas_Ratio'] = df['Nitrogen'] * (df['Potassium'] + 1e-8)
    df['Nitro_Phosp_Ratio'] = df['Nitrogen'] * (df['Phosphorous'] + 1e-8)
    df['Potas_Phosp_Ratio'] = df['Potassium'] * (df['Phosphorous'] + 1e-8)
    df['Nitro_Potas_Phosp_Ratio'] = df['Potassium'] * df['Nitrogen'] * (df['Phosphorous'] + 1e-8)

    # Extracting avg of feature per Soil/Crop type
    soil_moisture_avg = df.groupby('Soil Type')['Moisture'].mean()
    soil_temp_avg = df.groupby('Soil Type')['Temparature'].mean()
    soil_humid_avg = df.groupby('Soil Type')['Humidity'].mean()
    crop_temp_avg = df.groupby('Crop Type')['Temparature'].mean()
    crop_humid_avg = df.groupby('Crop Type')['Humidity'].mean()
    crop_moist_avg = df.groupby('Crop Type')['Moisture'].mean()
    
    # Comparing feature vs it's average per Soil/Crop type 
    df['moisture_vs_soil_avg'] = df['Moisture'] / df['Soil Type'].map(soil_moisture_avg + 1e-8)
    df['temp_vs_soil_avg'] = df['Temparature'] / df['Soil Type'].map(soil_temp_avg + 1e-8)
    df['humid_vs_soil_avg'] = df['Humidity'] / df['Soil Type'].map(soil_humid_avg + 1e-8)
    df['temp_vs_crop_avg'] = df['Temparature'] / df['Crop Type'].map(crop_temp_avg + 1e-8)
    df['humid_crop_avg'] = df['Humidity'] / df['Crop Type'].map(crop_humid_avg + 1e-8)
    df['moist_crop_avg'] = df['Moisture'] / df['Crop Type'].map(crop_moist_avg + 1e-8)


    #Arbitrary feature made out of desperation
    texture_map = {
    'Clayey': 'fine',
    'Sandy': 'coarse',
    'Loamy': 'balanced',
    'Red': 'acidic',
    'Black': 'fertile'
    }
    df['Soil_Texture_Class'] = df['Soil Type'].map(texture_map)

    #Turned out to be a really good feature with high shap value
    retention_map = {
    'Sandy': 1,
    'Red': 1,
    'Loamy': 2,
    'Black': 3,
    'Clayey': 3
    }
    df['Soil_Moisture_Retention'] = df['Soil Type'].map(retention_map)


    df['Moisture_vs_Soil'] = df['Moisture'] / df.groupby('Soil Type')['Moisture'].transform('mean')

    return df

    



# Apply feature engineering to both datasets
def apply_feature_engineering(train, test, train_cat, test_cat):
    
    # Create features for one-hot encoded datasets
    train_engineered = create_engineered_features(train)
    test_engineered = create_engineered_features(test)
    
    # Create features for categorical datasets
    train_cat_engineered = create_engineered_features(train_cat)
    test_cat_engineered = create_engineered_features(test_cat)
    
    print(f"Original features: {train.shape[1]}")
    print(f"After feature engineering: {train_engineered.shape[1]}")
    print(f"New features added: {train_engineered.shape[1] - train.shape[1]}")
    
    return train_engineered, test_engineered, train_cat_engineered, test_cat_engineered

train_eng, test_eng, train_cat_eng, test_cat_eng = apply_feature_engineering(train, test, train_cat, test_cat)


# One-hot-encode categorical features for the models that need it, LightGBM and XGBOOST
train_eng = pd.get_dummies(train_eng, columns=['Soil Type', 'Crop Type','Soil_Texture_Class'], prefix=['Soil Type', 'Crop Type','Soil_Texture_Class'])
test_eng = pd.get_dummies(test_eng, columns=['Soil Type', 'Crop Type','Soil_Texture_Class' ], prefix=['Soil Type', 'Crop Type','Soil_Texture_Class'])


# Ensure cat_features are all string type
for col in cat_features:
    if col in train_cat_eng.columns:
        train_cat_eng[col] = train_cat_eng[col].astype(str)
    if col in test_cat_eng.columns:
        test_cat_eng[col] = test_cat_eng[col].astype(str)



print(train_eng.columns)


print(train_cat_eng.columns)


features = [
    'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium',
       'Phosphorous', 'Temp_Humid_Ratio',
       'Humid_Moist_Ratio', 'Temp_Moist_Ratio', 'Temp_Moist_Humid_Ratio',
       'Nitro_Potas_Ratio', 'Nitro_Phosp_Ratio', 'Potas_Phosp_Ratio',
       'Nitro_Potas_Phosp_Ratio', 'moisture_vs_soil_avg', 'temp_vs_soil_avg',
       'humid_vs_soil_avg', 'temp_vs_crop_avg', 'humid_crop_avg',
       'moist_crop_avg', 'Soil_Moisture_Retention',
       'Moisture_vs_Soil', 'Soil Type_Black', 'Soil Type_Clayey',
       'Soil Type_Loamy', 'Soil Type_Red', 'Soil Type_Sandy',
       'Crop Type_Barley', 'Crop Type_Cotton', 'Crop Type_Ground Nuts',
       'Crop Type_Maize', 'Crop Type_Millets', 'Crop Type_Oil seeds',
       'Crop Type_Paddy', 'Crop Type_Pulses', 'Crop Type_Sugarcane',
       'Crop Type_Tobacco', 'Crop Type_Wheat']

X_train = train_eng[features] #define training features set
y_train = train_eng["Fertilizer Name"] #define training label set
X_test = test_eng[features] #define testing features set
#we don't have y_test, that is what we're trying to predict with our model



# Categorical features for Catboost
cat_features = ['Soil Type','Crop Type','Soil_Texture_Class']

# Complete list for Catboost
cat_feature_names = [
  'Temparature', 'Humidity', 'Moisture',
       'Nitrogen', 'Potassium', 'Phosphorous',
       'Temp_Humid_Ratio', 'Humid_Moist_Ratio', 'Temp_Moist_Ratio',
       'Temp_Moist_Humid_Ratio', 'Nitro_Potas_Ratio', 'Nitro_Phosp_Ratio',
       'Potas_Phosp_Ratio', 'Nitro_Potas_Phosp_Ratio', 'moisture_vs_soil_avg',
       'temp_vs_soil_avg', 'humid_vs_soil_avg', 'temp_vs_crop_avg',
       'humid_crop_avg', 'moist_crop_avg', 'Soil_Texture_Class',
       'Soil_Moisture_Retention', 'Moisture_vs_Soil','Soil Type','Crop Type'
]

X_train_cat = train_cat_eng[cat_feature_names]
y_train_cat = train_cat_eng["Fertilizer Name"]
X_test_cat = test_cat_eng[cat_feature_names]


#Encode target to make it single label
le = LabelEncoder()
y_encoded = le.fit_transform(y_train) 


#Train feature sample (for faster optuna hyperparamter search)
sample_idx = np.random.choice(len(X_train), size=150000, replace=False)
X_sample = X_train.iloc[sample_idx]
y_sample = y_encoded[sample_idx]

#Cat Feature Sample
sample_idx_cat = np.random.choice(len(X_train_cat), size=150000, replace=False)
X_sample_cat = X_train_cat.iloc[sample_idx_cat]
y_sample_cat = y_encoded[sample_idx_cat]




#StratifiedKFold to cross validate while keeping the percentages for each target class
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


print(X_train.shape)
print(X_test.shape)

print(y_encoded)


# Fixed MAP@K function for single-label classification
def mapk(actual, predicted, k=3):
    """
    MAP@K for single-label classification
    actual: list of true labels (integers)
    predicted: list of lists containing top-k predictions for each sample
    """
    def apk(a, p, k):
        p = p[:k]
        if a in p:
            # Find the position of the true label in predictions
            return 1.0 / (p.index(a) + 1)
        return 0.0
    
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def get_top_k_predictions(y_probs, k=3):
    """
    Convert probability matrix to top-k predictions
    """
    top_k_preds = []
    for probs in y_probs:
        # Get indices of top k probabilities
        top_k_indices = np.argsort(probs)[::-1][:k]
        top_k_preds.append(top_k_indices.tolist())
    return top_k_preds


# XGBoost Optuna
def objective_xgb(trial, k=3):
    params = {
        "device": "cuda",
        "tree_method": "gpu_hist",
        "verbosity": 0,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.045),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "n_estimators": 1500,  # Adjust for faster optimization
        "random_state": 44,
        'early_stopping_rounds': 50,
    }
    
    mapk_scores = []
    for train_idx, val_idx in skf.split(X_sample, y_sample):
        X_tr, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        y_tr, y_val = y_sample[train_idx], y_sample[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, 
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        
        # Get probability predictions
        y_probs = model.predict_proba(X_val)
        
        # Convert to top-k predictions
        top_k_preds = get_top_k_predictions(y_probs, k=k)
        
        # Calculate MAP@K (y_val are single integer labels)
        mapk_score = mapk(y_val.tolist(), top_k_preds, k=k)
        mapk_scores.append(mapk_score)
    
    return np.mean(mapk_scores)


# LightGBM Optuna
def objective_lgb(trial, k=3):
    params = {
        "device": "gpu",
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "n_estimators": 1000, # Adjust for faster optimization
        "verbosity": -1,
        "objective": "multiclass",
        "num_class": 7,  # Make sure this matches your actual number of classes
        "random_state": 42
    }
    
    mapk_scores = []
    for train_idx, val_idx in skf.split(X_sample, y_sample):
        X_tr, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        y_tr, y_val = y_sample[train_idx], y_sample[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_val, y_val)], 
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
        )
        
        y_probs = model.predict_proba(X_val)
        top_k_preds = get_top_k_predictions(y_probs, k=k)
        
        mapk_score = mapk(y_val.tolist(), top_k_preds, k=k)
        mapk_scores.append(mapk_score)
    
    return np.mean(mapk_scores)



# CatBoost optuna
def objective_cat(trial, k=3):
    params = {
        "iterations": 1000,  # Adjust for faster optimization
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "od_type": "Iter",
        "od_wait": 20, 
        "random_seed": 44,
        "task_type": "GPU",
        "verbose": False
    }

    mapk_scores = []
    for train_idx, val_idx in skf.split(X_sample_cat, y_sample_cat):
        X_tr, X_val = X_sample_cat.iloc[train_idx], X_sample_cat.iloc[val_idx]
        y_tr, y_val = y_sample_cat[train_idx], y_sample_cat[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            verbose=False,
        )
        
        # Get probability predictions
        y_probs = model.predict_proba(X_val)
        
        # Convert to top-k predictions
        top_k_preds = get_top_k_predictions(y_probs, k=k)
        
        # Calculate MAP@K
        mapk_score = mapk(y_val.tolist(), top_k_preds, k=k)
        mapk_scores.append(mapk_score)

    return np.mean(mapk_scores)



#Optuna study (Tests for different hyperparameter combinations, trying to maximize MAPK)

print("Optimizing XGBoost...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=50)

print("Optimizing LightGBM...")
study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=50)

print("Optimizing CatBoost...")
study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=50)

print("Best XGB params:", study_xgb.best_params, "Score:", study_xgb.best_value)
print("Best LGB params:", study_lgb.best_params, "Score:", study_lgb.best_value)
print("Best CAT params:", study_cat.best_params, "Score:", study_cat.best_value)



# Train final models with best parameters
final_xgb = xgb.XGBClassifier(**study_xgb.best_params, n_estimators=1500, random_state=43, tree_method='gpu_hist', predictor='gpu_predictor')
final_xgb.fit(X_train, y_encoded)

final_lgb = lgb.LGBMClassifier(**study_lgb.best_params, n_estimators=3000, device='gpu')
final_lgb.fit(X_train, y_encoded)

final_cat = CatBoostClassifier(**study_cat.best_params, iterations=1200, random_seed=42, verbose=False,  task_type='GPU') # This is the key for CatBoost
final_cat.fit(X_train_cat, y_encoded, cat_features=cat_features)



# The evaluation metric works on probability confidence so i used predict_proba
pred_xgb = final_xgb.predict_proba(X_test)
pred_lgb = final_lgb.predict_proba(X_test)
pred_cat = final_cat.predict_proba(X_test_cat)



# Average the probabilities
ensemble_proba = (pred_xgb + pred_lgb + pred_cat) / 3



# Get TOP-3 predictions for each sample
def get_top3_predictions(probabilities, label_encoder):
    top3_preds = []
    for prob_row in probabilities:
        
        # Get indices of top 3 probabilities
        top3_indices = np.argsort(prob_row)[-3:][::-1]  # Top 3 in descending order

        # Convert back from encoded to fertilizer names
        top3_names = label_encoder.inverse_transform(top3_indices)
        
        # Join with spaces
        top3_string = ' '.join(top3_names)
        top3_preds.append(top3_string)
    return top3_preds

# Get top-3 predictions
test_predictions_top3 = get_top3_predictions(ensemble_proba, le)



# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': test_predictions_top3
})

#Make sure it's the right shape and it looks correct
print(f"\nSubmission shape: {submission.shape}")
print("Submission head:")
print(submission.head())

# Save submission
submission.to_csv('fertilizer_predictions.csv', index=False)
print("\nSubmission saved as 'fertilizer_predictions.csv'")


explainer_xgb = shap.Explainer(final_xgb)
shap_values_xgb = explainer_xgb(X_test)

# Get the total number of features
num_features = X_test.shape[1]

# Plot SHAP values for class 0, showing all features
shap.plots.beeswarm(shap_values_xgb[..., 0], max_display=num_features)



explainer_lgb = shap.Explainer(final_lgb)
shap_values_lgb = explainer_lgb(X_test)

# Get the total number of features
num_features = X_test.shape[1]

# Plot SHAP values for class 0, showing all features
shap.plots.beeswarm(shap_values_lgb[..., 0], max_display=num_features)



# Didn't manage to plot shap values for Catboost yet

