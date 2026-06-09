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


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col = 0)
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col = 0)


print(train.shape)
train.head()


train["Fertilizer Name"].value_counts()


print(test.shape)
test.head()


# read fertilizer_prediction 
fer_prediction_df = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


print(fer_prediction_df.shape)
fer_prediction_df.head()


union_df = pd.concat([train, fer_prediction_df], ignore_index=True)


print(union_df.shape)
union_df.head()


from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


train.columns


feature_columns = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type','Nitrogen', 'Potassium', 'Phosphorous']
target_column = 'Fertilizer Name'


# Separate features and target
X = union_df.drop(columns=['Fertilizer Name'])
y = union_df['Fertilizer Name']


X['Soil Type'].value_counts()


X['Soil Type'].unique()


SoilRetention = {
    'Clayey': 0.9,
    'Sandy': 0.3,
    'Red': 0.6,
    'Loamy': 0.8,
    'Black': 0.7
}


SoilpHPreference = {'Clayey': 6.0, 'Sandy': 6.5, 'Red': 6.2, 'Loamy': 6.8, 'Black': 7.0}


SoilDrainageMap = {
    'Sandy': 0.9,
    'Clayey': 0.2,
    'Loamy': 0.5,
    'Black': 0.5,
    'Red': 0.5
}


SoilOrganicMatterMap = {
    'Sandy': 1.0,
    'Red': 1.5,
    'Clayey': 2.8,
    'Loamy': 3.8,
    'Black': 5.5
}


SoilCecMap = {
    'Sandy': 5,
    'Clayey': 30,
    'Loamy': 20,
    'Black': 25,
    'Red': 15
}


X['Crop Type'].value_counts()


X['Crop Type'].unique()


#Approximate Nutrient Demands (kg/ha)
CropNutrientDemands = {
    'Sugarcane': {'N': 200, 'P': 80, 'K': 160},
    'Millets': {'N': 50, 'P': 25, 'K': 30},
    'Barley': {'N': 50, 'P': 25, 'K': 35},
    'Paddy': {'N': 100, 'P': 50, 'K': 50},
    'Pulses': {'N': 30, 'P': 50, 'K': 25},
    'Tobacco': {'N': 100, 'P': 50, 'K': 90},
    'Ground Nuts': {'N': 30, 'P': 50, 'K': 50},
    'Maize': {'N': 125, 'P': 60, 'K': 50},
    'Cotton': {'N': 80, 'P': 40, 'K': 40},
    'Wheat': {'N': 100, 'P': 50, 'K': 50},
    'Oil seeds': {'N': 40, 'P': 30, 'K': 30}
}
def get_nutrient_demand(crop, nutrient):
    return CropNutrientDemands.get(crop).get(nutrient)


GrowthCycleDaysMap = {
    'Sugarcane': 330,
    'Millets': 100,
    'Barley': 100,
    'Paddy': 130,
    'Pulses': 100,
    'Tobacco': 120,
    'Ground Nuts': 110,
    'Maize': 110,
    'Cotton': 170,
    'Wheat': 130,
    'Oil seeds': 110
}


YieldPotentialMap = {
    'Sugarcane': 36.3,
    'Maize': 5.5,
    'Wheat': 3.5,
    'Paddy': 3.46,
    'Cotton': 2.2,
    'Pulses': 2.0,
    'Ground Nuts': 3.7,
    'Barley': 2.8,
    'Millets': 1.0,
    'Oil seeds': 3.57,
    'Tobacco': 2.6
}


RootDepthMap = {
    'Sugarcane': 150,
    'Millets': 60,
    'Barley': 70,
    'Paddy': 50,
    'Pulses': 40,
    'Tobacco': 120,
    'Ground Nuts': 70,
    'Maize': 100,
    'Cotton': 150,
    'Wheat': 90,
    'Oil seeds': 70
}


CropWaterDemandMap = {
    'Sugarcane': 2000,
    'Millets': 500,
    'Barley': 550,
    'Paddy': 1350,
    'Pulses': 400,
    'Tobacco': 600,
    'Ground Nuts': 600,
    'Maize': 650,
    'Cotton': 800,
    'Wheat': 550,
    'Oil seeds': 500
}


CropNutrientUptakeEfficiency = {
    'Sugarcane': 70,     # very efficient
    'Millets': 25,       # low
    'Barley': 45,        # medium
    'Paddy': 65,         # high
    'Pulses': 25,        # low
    'Tobacco': 45,       # medium
    'Ground Nuts': 45,   # medium
    'Maize': 70,         # high
    'Cotton': 45,        # medium
    'Wheat': 45,         # medium
    'Oil seeds': 25      # low
}



def engineer_features(df):
    df = df.copy()
    
    # Nutrient ratios
    df['N_to_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['N_to_K'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['P_to_K'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)

    # Interaction features
    df['Temp_Moisture'] = df['Temparature'] * df['Moisture']
    df['Temp_Humidity'] = df['Temparature'] * df['Humidity']

    # Crop 
    df['Crop_Water_Demand'] = df['Crop Type'].map(CropWaterDemandMap).fillna(600)
    df['Nutrient_Uptake_Efficiency'] = df['Crop Type'].map(CropNutrientUptakeEfficiency).fillna(2)
    
    #Nutrient demands 
    df['Nitrogen_Demand'] = df['Crop Type'].apply(lambda crop: get_nutrient_demand(crop, 'N'))
    df['Phosphorous_Demand'] = df['Crop Type'].apply(lambda crop: get_nutrient_demand(crop, 'P'))
    df['Potassium_Demand'] = df['Crop Type'].apply(lambda crop: get_nutrient_demand(crop, 'K'))
    
    #Nutrient deficit
    df['Nitrogen_Deficit'] = df['Nitrogen_Demand'] - df['Nitrogen']
    df['Phosphorous_Deficit'] = df['Phosphorous_Demand'] - df['Phosphorous']
    df['Potassium_Deficit'] = df['Potassium_Demand'] - df['Potassium']


    # Apply mappings
    df['Yield_Potential'] = df['Crop Type'].map(YieldPotentialMap).fillna(2)
    df['Growth_Cycle'] = df['Crop Type'].map(GrowthCycleDaysMap).fillna(2)
    df['Root_Depth'] = df['Crop Type'].map(RootDepthMap).fillna(2)

    # Soil  
    df['Soil_Retention'] = df['Soil Type'].map(SoilRetention)
    df['Soil_pH_Preference'] = df['Soil Type'].map(SoilpHPreference).fillna(6.5)
    df['Soil_Drainage_Potential'] = df['Soil Type'].map(SoilDrainageMap).fillna(0.5)
    df['Soil_Organic_Matter'] = df['Soil Type'].map(SoilOrganicMatterMap).fillna(3)
    df['Soil_CEC'] = df['Soil Type'].map(SoilCecMap).fillna(15)
    
    
    #Advanced features
    df['Soil_Fertility_Index'] = (
        df['Soil_Retention'] * 0.4 +
        df['Soil_Organic_Matter'] * 0.2 +
        df['Soil_CEC'] * 0.4
    )
    df['Stress_Index'] = (df['Temparature'] * df['Humidity']) / (df['Moisture'] + 1e-6)
    df['Soil_Temp_Interaction'] = df['Soil_Retention'] * df['Temparature']
    df['Water_Retention_Index'] = df['Moisture'] * df['Soil_Retention']

    df.drop(columns=['Crop Type', 'Soil Type'], inplace=True)
    
    return df


X_updated = engineer_features(X)
print(X_updated.shape)
print(len(X_updated.columns))
X_updated.head()
X_updated.columns


X_updated.head(15)


all_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium',
       'Phosphorous', 'N_to_P', 'N_to_K', 'P_to_K', 'Temp_Moisture',
       'Temp_Humidity', 'Crop_Water_Demand', 'Nutrient_Uptake_Efficiency',
       'Nitrogen_Demand', 'Phosphorous_Demand', 'Potassium_Demand',
       'Nitrogen_Deficit', 'Phosphorous_Deficit', 'Potassium_Deficit',
       'Yield_Potential', 'Growth_Cycle', 'Root_Depth', 'Soil_Retention',
       'Soil_pH_Preference', 'Soil_Drainage_Potential', 'Soil_Organic_Matter',
       'Soil_CEC', 'Soil_Fertility_Index', 'Stress_Index',
       'Soil_Temp_Interaction', 'Water_Retention_Index']


# ===Scale numeric features ===
scaler = StandardScaler()
X_updated = scaler.fit_transform(X_updated)

# ===Encode target variable ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)


y_encoded


print(X_updated.shape)
X_updated


test_updated = engineer_features(test)
print(test_updated.shape)
print(len(test_updated.columns))
test_columns = test_updated.columns
test_updated.head()


print(len(test_columns))
test_columns


X_test_processed = scaler.transform(test_updated)


X_test_processed


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


def mapk_from_probs(y_true, y_probs, k=3):
    """
    Compute MAP@k when predictions are probability distributions.

    Parameters:
    y_true : list or array of true class indices (length = n_samples)
    y_probs : 2D array (n_samples x n_classes), predicted probabilities
    k : cutoff for top-k predictions

    Returns:
    MAP@k score
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    n_samples = y_probs.shape[0]
    topk_preds = np.argsort(-y_probs, axis=1)[:, :k]  # Get top-k indices per row

    score = 0.0
    for i in range(n_samples):
        if y_true[i] in topk_preds[i]:
            rank = np.where(topk_preds[i] == y_true[i])[0][0] + 1  # 1-based index
            score += 1.0 / rank
    return score / n_samples


# from sklearn.metrics import make_scorer

# def map3_scorer(estimator, X, y):
#     y_proba = estimator.predict_proba(X)
#     return mapk_from_probs(y, y_proba, k= 3 )

# # Wrap it for scikit-learn
# scorer = make_scorer(map3_scorer, greater_is_better=True)


# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.model_selection import RandomizedSearchCV, train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from scipy.stats import uniform, randint

# # Train-test split (replace with your own dataset)
# X_train, X_test, y_train, y_test = train_test_split(X_updated, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)

# # Base model
# xgb = XGBClassifier(
#     objective='multi:softprob',
#     num_class=7,
#     tree_method='gpu_hist',
#     use_label_encoder=False,
#     eval_metric='mlogloss',
#     verbosity=0,
#     random_state=42
# )

# # Parameter distributions
# param_dist = {
#     'n_estimators': randint(500, 3000),
#     'max_depth': randint(4, 12),
#     'learning_rate': uniform(0.01, 0.2),
#     'min_child_weight': randint(1, 6),
#     'subsample': uniform(0.6, 0.4),
#     'colsample_bytree': uniform(0.5, 0.5),
#     'colsample_bylevel': uniform(0.7, 0.3),
#     'colsample_bynode': uniform(0.7, 0.3),
#     'alpha': uniform(0.0, 2.0),
#     'reg_lambda': uniform(0.5, 5.0),
#     'max_bin': randint(64, 256),
# }

# # Randomized search
# search = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_dist,
#     n_iter=10,  # try 50 combinations
#     scoring=map3_scorer,
#     cv=5,
#     verbose=1,
#     random_state=42
# )

# search.fit(X_train, y_train)

# # Best model and performance
# best_model = search.best_estimator_
# y_pred = best_model.predict(X_test)

# print("Best Parameters:", search.best_params_)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# {'tree_method': 'hist',
#  'n_estimators': 5000,
#  'objective': 'multi:softprob',
#  'random_state': 42,
#  'enable_categorical': True,
#  'verbosity': 0,
#  'early_stopping_rounds': 100,
#  'eval_metric': 'mlogloss',
#  'booster': 'gbtree',
#  "device": "cuda",
#  'n_jobs': -1,
#  'learning_rate': 0.1,
#  'num_class': 7,
#  'lambda': 0.05656209749983576,
#  'alpha': 5.620898657099113,
#  'colsample_bytree': 0.2587327850345624, 
#  'subsample': 0.8276149323901826,
#  'max_depth': 20,
#  'min_child_weight': 10
# }


params = {
        'objective': 'multi:softprob', 
        'num_class': 7,  
        'max_depth': 7,
        'learning_rate': 0.02953442280127678,
        'min_child_weight' : 4,
        'n_estimators': 1978,
        'alpha': 1.8977710745066665, 
        'reg_lambda':  3.5499832889131047, 
        'colsample_bytree': 0.5,
        'subsample': 0.9332779646944658,
        'max_bin': 147,
        'colsample_bylevel': 0.9896896099223678, 
        'colsample_bynode': 0.9425192044349383, 
        'colsample_bytree': 0.6523068845866853, 
        'verbosity': 0,
        'tree_method': 'gpu_hist',  
        'random_state': 42,
        'eval_metric': 'mlogloss',
               
    }


# # === K-Fold Setup ===
# FOLDS = 5
# skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# # === Placeholders for OOF and Test predictions ===
# oof_preds_xgb = np.zeros((X_updated.shape[0], len(le.classes_)))
# test_preds = np.zeros((len(X_test_processed), len(le.classes_)))

# # === Training XGBoost ===

# for fold, (train_idx, val_idx) in enumerate(skf.split(X_updated, y_encoded)):
#     print(f"\n{'='*15} Fold {fold+1}/{FOLDS} {'='*15}")
    
#     X_train, y_train = X_updated[train_idx], y_encoded[train_idx]
#     X_val, y_val = X_updated[val_idx], y_encoded[val_idx]
    
#     # === XGBoost ===
#     xgb_model = xgb.XGBClassifier(**params)
#     xgb_model.set_params(early_stopping_rounds=30)
#     xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
#     oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)
#     test_preds += xgb_model.predict_proba(X_test_processed)
    
#     map3 = mapk_from_probs(y_val, oof_preds_xgb[val_idx])
#     print(f"XGBoost Fold {fold+1} - MAP@3: {map3:.4f}")
    
# # === Overall Results ===
# overall_map3 = mapk_from_probs(y_encoded, oof_preds_xgb)
# print(f"\nOverall XGBoost - MAP@3: {overall_map3:.4f}")


# # === Average test predictions ===
# test_preds /= FOLDS


# # === Overall Results ===
# overall_map3 = mapk_from_probs(y_encoded, oof_preds_xgb)
# print(f"\nOverall XGBoost - MAP@3: {overall_map3:.4f}")


# # === Average test predictions ===
# test_preds /= FOLDS


# # === Get top-3 fertilizer names for test set ===
# top3_indices_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
# top3_labels_test = le.inverse_transform(top3_indices_test.flatten()).reshape(top3_indices_test.shape)


# top3_labels_test


# importance = xgb_model.feature_importances_ 


# feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importance})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# print(feature_importance_df.head(20))


# feature_importance_df


X_df = pd.DataFrame(X_updated, columns = all_features)
print(X_df.shape)
X_df.head()


X_test_df = pd.DataFrame(X_test_processed, columns = all_features)
print(X_test_df.shape)
X_test_df.head()


# important_columns = feature_importance_df.head(20).Feature.to_list()
# important_columns


important_columns = ['Nitrogen_Demand',
 'Moisture',
 'Phosphorous',
 'Crop_Water_Demand',
 'Nitrogen',
 'Root_Depth',
 'Potassium_Demand',
 'Phosphorous_Demand',
 'Potassium',
 'Soil_Retention',
 'Yield_Potential',
 'Nutrient_Uptake_Efficiency',
 'Soil_Drainage_Potential',
 'Nitrogen_Deficit',
 'Temparature',
 'Growth_Cycle',
 'Phosphorous_Deficit',
 'Potassium_Deficit',
 'Humidity',
 'Soil_Organic_Matter']


X_shorter = X_df[important_columns]
X_test_shorter = X_test_df[important_columns]


X_shorter.shape, X_test_shorter.shape


# === K-Fold Setup ===
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# === Placeholders for OOF and Test predictions ===
oof_preds_xgb = np.zeros((X_shorter.shape[0], len(le.classes_)))
test_preds = np.zeros((len(X_test_shorter), len(le.classes_)))

# === Training XGBoost ===

for fold, (train_idx, val_idx) in enumerate(skf.split(X_shorter, y_encoded)):
    print(f"\n{'='*15} Fold {fold+1}/{FOLDS} {'='*15}")
    
    X_train, y_train = X_shorter.iloc[train_idx], y_encoded[train_idx]
    X_val, y_val = X_shorter.iloc[val_idx], y_encoded[val_idx]
    
    # === XGBoost ===
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.set_params(early_stopping_rounds=30)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)
    test_preds += xgb_model.predict_proba(X_test_shorter)
    
    map3 = mapk_from_probs(y_val, oof_preds_xgb[val_idx])
    print(f"XGBoost Fold {fold+1} - MAP@3: {map3:.4f}")
    
# === Overall Results ===
overall_map3 = mapk_from_probs(y_encoded, oof_preds_xgb)
print(f"\nOverall XGBoost - MAP@3: {overall_map3:.4f}")


# === Average test predictions ===
test_preds /= FOLDS


# === Overall Results ===
overall_map3 = mapk_from_probs(y_encoded, oof_preds_xgb)
print(f"\nOverall XGBoost - MAP@3: {overall_map3:.4f}")


# === Average test predictions ===
test_preds /= FOLDS


# === Get top-3 fertilizer names for test set ===
top3_indices_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top3_labels_test = le.inverse_transform(top3_indices_test.flatten()).reshape(top3_indices_test.shape)


top3_labels_test


submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
print(submission.shape)
submission.head()


submission['Fertilizer Name'] = [" ".join(row) for row in top3_labels_test]


submission.head()


submission.to_csv('submission.csv', index=False)




