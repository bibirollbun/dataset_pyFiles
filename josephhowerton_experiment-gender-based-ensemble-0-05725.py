# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Machine Learning Libraries

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ML Models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')


def rmse(y_true, y_pred):
   return np.sqrt(mean_squared_error(y_true, y_pred))

def find_best_ensemble_weights(predictions_list, y_true, num_iterations=100):
    num_models = len(predictions_list)
    best_rmse = float('inf')
    best_weights = None

    for _ in range(num_iterations):
        weights = np.random.dirichlet(np.ones(num_models))
        rounded_weights = tuple(np.round(weights, 2))

        if len(rounded_weights) != num_models:
            raise ValueError("Number of weights does not match the number of prediction arrays.")

        ensemble_preds = np.zeros_like(predictions_list[0])
        for i in range(num_models):
            ensemble_preds += rounded_weights[i] * predictions_list[i]

        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))

        if ensemble_rmse < best_rmse:
            best_rmse = ensemble_rmse
            best_weights = rounded_weights

    print(f"\nBest weights: {', '.join([f'Model {i+1}: {w}' for i, w in enumerate(best_weights)])}")
    print(f"Best ensemble RMSE: {best_rmse:.4f}")

    return best_weights, best_rmse

def extract_feature_importances(models, feature_names):
    all_importances = pd.DataFrame(index=feature_names)
    
    if 'XGBoost' in models:
        all_importances['XGBoost Weight'] = models['XGBoost'].feature_importances_
        xgb_gain = models['XGBoost'].get_booster().get_score(importance_type='gain')
        all_importances['XGBoost Gain'] = _map_importance_to_features(xgb_gain, feature_names)
    
    if 'LightGBM' in models:
        all_importances['LightGBM Split'] = models['LightGBM'].feature_importances_
        lgbm_gain = models['LightGBM'].booster_.feature_importance(importance_type='gain')
        all_importances['LightGBM Gain'] = lgbm_gain
    
    if 'CatBoost' in models:
        all_importances['CatBoost'] = models['CatBoost'].feature_importances_
    
    if 'RandomForest' in models:
        all_importances['RandomForest'] = models['RandomForest'].feature_importances_
    
    _normalize_importances(all_importances)
    
    return all_importances

def _map_importance_to_features(importance_dict, feature_names):
    importance_values = np.zeros(len(feature_names))
    
    if list(importance_dict.keys())[0].startswith('f') and list(importance_dict.keys())[0][1:].isdigit():
        for key, value in importance_dict.items():
            idx = int(key.replace('f', ''))
            importance_values[idx] = value
    else:
        for feature_name in feature_names:
            if feature_name in importance_dict:
                idx = list(feature_names).index(feature_name)
                importance_values[idx] = importance_dict[feature_name]
                
    return importance_values

def _normalize_importances(df):
    for col in df.columns:
        if df[col].sum() > 0:
            df[col] = df[col] / df[col].sum()
    
    return df


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.info()


train.head()


train.describe()


display(train.corr(numeric_only=True))


def bmi_category(bmi):
    """Determine BMI category based on BMI value."""
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal Weight'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


def intensity_zone(intensity):
    """Determine exercise intensity zone based on heart rate intensity."""
    if intensity < 0.5:
        return 'Very Light'
    elif intensity < 0.6:
        return 'Light'
    elif intensity < 0.7:
        return 'Moderate'
    elif intensity < 0.8:
        return 'Hard'
    elif intensity < 0.9:
        return 'Very Hard'
    else:
        return 'Maximum'


def prepare_data(data):
    """
    Prepare and enrich dataset with health-related metrics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input DataFrame containing health data
        
    Returns:
    --------
    pandas.DataFrame
        Enriched DataFrame with additional health metrics
    """
    copy = data.copy()
    
    # Basic BMI calculation
    copy['Height_m'] = copy['Height'] / 100  # Convert height from cm to meters
    copy['BMI'] = copy['Weight'] / (copy['Height_m'] ** 2)
    
    # Basal Metabolic Rate (BMR) using Mifflin-St Jeor equation
    copy['BMR'] = np.where(copy['Sex'] == 'Male',
                          (10 * copy['Weight']) + (6.25 * copy['Height']) - (5 * copy['Age']) + 5,
                          (10 * copy['Weight']) + (6.25 * copy['Height']) - (5 * copy['Age']) - 161)
    
    # Maximum Heart Rate based on age
    copy['Max_Heart_Rate'] = 220 - copy['Age']
    
    # Relative Heart Rate Intensity
    copy['HR_Intensity'] = copy['Heart_Rate'] / copy['Max_Heart_Rate']
    
    # Body Surface Area (BSA)
    copy['BSA'] = np.sqrt((copy['Height'] * copy['Weight']) / 3600)
    
    # Cardiovascular metrics
    copy['CV_Load'] = copy['Heart_Rate'] * copy['Duration']
    copy['CV_Demand'] = copy['Weight'] * copy['Duration']
    copy['Thermal_Stress'] = copy['Body_Temp'] * copy['Duration']
    
    # Categorization
    copy['BMI_Category'] = copy['BMI'].apply(bmi_category)
    copy['Intensity_Zone'] = copy['HR_Intensity'].apply(intensity_zone)
    
    return copy


train = prepare_data(train)
test = prepare_data(test)


# Load both datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Apply data preparation to both datasets (assuming this doesn't modify categorical data types)
train = prepare_data(train)
test = prepare_data(test)

# Make sure categorical columns are string type before encoding
categorical_cols = ['Sex', 'BMI_Category', 'Intensity_Zone']

for col in categorical_cols:
    # Convert to string type to ensure consistent handling
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)

# Create separate label encoders for each category
sex_encoder = LabelEncoder()
bmi_encoder = LabelEncoder()
intensity_encoder = LabelEncoder()

# A safer approach: fit on combined unique values to handle all possible values
# Sex encoding
all_sex_values = pd.concat([train['Sex'], test['Sex']]).unique()
sex_encoder.fit(all_sex_values)
train['Sex'] = sex_encoder.transform(train['Sex'])
test['Sex'] = sex_encoder.transform(test['Sex'])
train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')

# BMI_Category encoding
all_bmi_values = pd.concat([train['BMI_Category'], test['BMI_Category']]).unique()
bmi_encoder.fit(all_bmi_values)
train['BMI_Category'] = bmi_encoder.transform(train['BMI_Category'])
test['BMI_Category'] = bmi_encoder.transform(test['BMI_Category'])
train['BMI_Category'] = train['BMI_Category'].astype('category')
test['BMI_Category'] = test['BMI_Category'].astype('category')

# Intensity_Zone encoding
all_intensity_values = pd.concat([train['Intensity_Zone'], test['Intensity_Zone']]).unique()
intensity_encoder.fit(all_intensity_values)
train['Intensity_Zone'] = intensity_encoder.transform(train['Intensity_Zone'])
test['Intensity_Zone'] = intensity_encoder.transform(test['Intensity_Zone'])
train['Intensity_Zone'] = train['Intensity_Zone'].astype('category')
test['Intensity_Zone'] = test['Intensity_Zone'].astype('category')

# Store the mapping for future reference if needed
sex_mapping = dict(zip(sex_encoder.classes_, range(len(sex_encoder.classes_))))
bmi_mapping = dict(zip(bmi_encoder.classes_, range(len(bmi_encoder.classes_))))
intensity_mapping = dict(zip(intensity_encoder.classes_, range(len(intensity_encoder.classes_))))

print("Sex mapping:", sex_mapping)
print("BMI Category mapping:", bmi_mapping)
print("Intensity Zone mapping:", intensity_mapping)

# Prepare labels and drop unnecessary columns
labels = np.log1p(train["Calories"])
train = train.drop(columns=["id", "Calories"])
test = test.drop(columns=["id"])


corr = train.corr()

plt.figure(figsize=(10, 8))

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1, vmax=1, 
            linewidths=0.5,
            annot_kws={"size": 7},
            square=True)

plt.title('Feature Correlation Heatmap', fontsize=18, pad=20)
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train_male = X_train[X_train['Sex'] == 1]
y_train_male = y_train[X_train['Sex'] == 1]

X_train_female = X_train[X_train['Sex'] == 0]
y_train_female = y_train[X_train['Sex'] == 0]

X_val_male = X_val[X_val['Sex'] == 1]
y_val_male = y_val[X_val['Sex'] == 1]

X_val_female = X_val[X_val['Sex'] == 0]
y_val_female = y_val[X_val['Sex'] == 0]

X_test_male = X_test[X_test['Sex'] == 1]
y_test_male = y_test[X_test['Sex'] == 1]

X_test_female = X_test[X_test['Sex'] == 0]
y_test_female = y_test[X_test['Sex'] == 0]


xgb_config = {
    'n_estimators': 1100,
    'learning_rate': 0.013716909843542555,
    'max_depth': 11,
    'min_child_weight': 3,
    'subsample': 0.6474544779422277,
    'colsample_bytree': 0.6756263234265003,
    'gamma': 2.9760049606720922e-05,
    'reg_lambda': 0.7456837622464084,
    'reg_alpha': 7.130044917976876e-06,
    'objective': 'reg:squaredlogerror',
    'enable_categorical': True,
    'eval_set': [(X_val_male, y_val_male)],
}


xgb_model_male = xgb.XGBRegressor(**xgb_config)
xgb_model_male.fit(X=X_train_male, y=y_train_male)

xgb1_predictions_male = xgb_model_male.predict(X_test_male)
print(f"{np.sqrt(mean_squared_error(y_test_male, xgb1_predictions_male)):.6f}")


xgb_config['eval_set'] = [(X_val_female, y_val_female)]
xgb_model_female = xgb.XGBRegressor(**xgb_config)
xgb_model_female.fit(X=X_train_female, y=y_train_female)

xgb1_predictions_female = xgb_model_female.predict(X_test_female)
print(f"{np.sqrt(mean_squared_error(y_test_female, xgb1_predictions_female)):.6f}")


lgb_config = {
    'learning_rate': 0.011974973795927277,
    'n_estimators': 900,
    'num_leaves': 191,
    'max_depth': 27,
    'min_child_weight': 0.3723518241236667,
    'min_child_samples': 7,
    'subsample': 0.8654774420162681,
    'subsample_freq': 6,
    'colsample_bytree': 0.8400402205580921,
    'reg_alpha': 0.002468873033000459,
    'reg_lambda': 4.6462056410378543e-07,
    'objective': 'regression',
    'random_state': 42,
    'force_col_wise': True,
    'n_jobs': -1,
    'verbose': -1,
}


lgb_model_male = lgb.LGBMRegressor(**lgb_config)
lgb_model_male.fit(
    X=X_train_male,
    y=y_train_male,
    eval_set=[(X_val_male, y_val_male)],
    callbacks=[lgb.early_stopping(25)]
)

lgb_preds_male = lgb_model_male.predict(X_test_male)
rmse_male = np.sqrt(mean_squared_error(y_test_male, lgb_preds_male))
print(f"Male RMSE: {rmse_male:.6f}")


lgb_model_female = lgb.LGBMRegressor(**lgb_config)
lgb_model_female.fit(
    X=X_train_female,
    y=y_train_female,
    eval_set=[(X_val_female, y_val_female)],
    callbacks=[lgb.early_stopping(25)]
)

lgb_preds_female = lgb_model_female.predict(X_test_female)
rmse_female = np.sqrt(mean_squared_error(y_test_female, lgb_preds_female))
print(f"Female RMSE: {rmse_female:.6f}")


cat_config = {
    'learning_rate': 0.01928353485682738, 
    'depth': 9, 
    'l2_leaf_reg': 0.0009825798445035239, 
    'random_strength': 1.4800703979936343e-08, 
    'bagging_temperature': 2.9976035010300635, 
    'border_count': 201, 
    'grow_policy': 'SymmetricTree', 
    'min_data_in_leaf': 70, 
    'subsample': 0.7665975430759543, 
    'max_ctr_complexity': 6
}


cat_model_male = CatBoostRegressor(**cat_config)
cat_model_male.fit(
    X=X_train_male,
    y=y_train_male,
    cat_features=['Sex', 'BMI_Category', 'Intensity_Zone'],
    eval_set=(X_val_male, y_val_male),
    early_stopping_rounds=50,
    verbose=False
)

cat_preds_male = cat_model_male.predict(X_test_male)
rmse_male = np.sqrt(mean_squared_error(y_test_male, cat_preds_male))
print(f"Male RMSE: {rmse_male:.6f}")


cat_model_female = CatBoostRegressor(**cat_config)
cat_model_female.fit(
    X=X_train_female,
    y=y_train_female,
    cat_features=['Sex', 'BMI_Category', 'Intensity_Zone'],
    eval_set=(X_val_female, y_val_female),
    early_stopping_rounds=50,
    verbose=False
)

cat_preds_female = cat_model_female.predict(X_test_female)
rmse_female = np.sqrt(mean_squared_error(y_test_female, cat_preds_female))
print(f"Female RMSE: {rmse_female:.6f}")


rf_config_male = {
    'n_estimators': 1000,
    'max_depth': 44,
    'min_samples_split': 18,
    'min_samples_leaf': 1,
    'max_features': 0.5,
    'bootstrap': True
}

rf_model_male = RandomForestRegressor(**rf_config_male)
rf_model_male.fit(X_train_male, y_train_male)
rf_predictions_male = rf_model_male.predict(X_test_male)
print(f"{np.sqrt(mean_squared_error(y_test_male, rf_predictions_male)):.6f}")


rf_config_female = {
    'n_estimators': 1000,
    'max_depth': 44,
    'min_samples_split': 18,
    'min_samples_leaf': 1,
    'max_features': 0.5,
    'bootstrap': True
}

rf_model_female = RandomForestRegressor(**rf_config_female)
rf_model_female.fit(X_train_female, y_train_female)
rf_predictions_female = rf_model_female.predict(X_test_female)
print(f"{np.sqrt(mean_squared_error(y_test_female, rf_predictions_female)):.6f}")


male_preds = [xgb1_predictions_male, lgb_preds_male, cat_preds_male, rf_predictions_male]
female_preds = [xgb1_predictions_female, lgb_preds_female, cat_preds_female, rf_predictions_female]

male_weights, male_rmsle = find_best_ensemble_weights(male_preds, y_test_male)
ensemble_male = np.average(male_preds, axis=0, weights=male_weights)

female_weights, female_rmsle = find_best_ensemble_weights(female_preds, y_test_female)
ensemble_female = np.average(female_preds, axis=0, weights=female_weights)

final_preds = np.zeros(len(X_test))

final_preds[X_test['Sex'] == 1] = ensemble_male
final_preds[X_test['Sex'] == 0] = ensemble_female

print(f"{np.sqrt(mean_squared_error(y_test, final_preds)):.6f}")


test_male_mask = test['Sex'] == 1
test_female_mask = test['Sex'] == 0

X_test_male = test[test_male_mask]
X_test_female = test[test_female_mask]


xgb_test_preds_male = xgb_model_male.predict(X_test_male)
lgb_test_preds_male = lgb_model_male.predict(X_test_male)
cat_test_preds_male = cat_model_male.predict(X_test_male)
rf_test_preds_male = rf_model_male.predict(X_test_male)

xgb_test_preds_female = xgb_model_female.predict(X_test_female)
lgb_test_preds_female = lgb_model_female.predict(X_test_female)
cat_test_preds_female = cat_model_female.predict(X_test_female)
rf_test_preds_female = rf_model_female.predict(X_test_female)

test_male_preds = [xgb_test_preds_male, lgb_test_preds_male, cat_test_preds_male, rf_test_preds_male]
test_female_preds = [xgb_test_preds_female, lgb_test_preds_female, cat_test_preds_female, rf_test_preds_female]

ensemble_test_male = np.average(test_male_preds, axis=0, weights=male_weights)
ensemble_test_female = np.average(test_female_preds, axis=0, weights=female_weights)

final_test_preds = np.zeros(len(test))

final_test_preds[test_male_mask] = ensemble_test_male
final_test_preds[test_female_mask] = ensemble_test_female

original_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_ids = original_test['id']

submission = pd.DataFrame({'id': test_ids,'target': np.expm1(final_test_preds)})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


display(submission.describe())


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error

def analyze_model_errors(y_true, y_pred, model_name="Model"):
    residuals = y_true - y_pred

    # Metrics
    rmse = np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))
    rmsle = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"ğŸ”� {model_name}")
    print(f"   RMSE : {rmse:.6f}")
    print(f"   RMSLE: {rmsle:.6f}")
    print(f"   Mean Error        : {residuals.mean():.4f}")
    print(f"   Std Dev of Error  : {residuals.std():.4f}")
    print(f"   Max Overprediction: {residuals.min():.4f}")
    print(f"   Max Underprediction: {residuals.max():.4f}")
    
    # Residual histogram
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.title(f"{model_name}")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Scatter plot: True vs Predicted
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Calories")
    plt.ylabel("Predicted Calories")
    plt.title(f"{model_name} True vs Predicted")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

analyze_model_errors(y_test_male, xgb1_predictions_male, model_name="XGB")
analyze_model_errors(y_test_male, lgb_preds_male, model_name="LGB")
analyze_model_errors(y_test_male, cat_preds_male, model_name="CAT")


analyze_model_errors(y_test_female, xgb1_predictions_female, model_name="XGB")
analyze_model_errors(y_test_female, lgb_preds_female, model_name="LGB")
analyze_model_errors(y_test_female, cat_preds_female, model_name="CAT")

