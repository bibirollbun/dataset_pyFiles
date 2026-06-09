import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()



import os
df_input = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')



df_input.head()




df_input['Crop Type'].value_counts().plot(kind='bar')


test_data['Crop Type'].value_counts().plot(kind='bar')


# df_input['N_P_ratio'] = df_input['Nitrogen'] / (df_input['Phosphorous'] + 0.0001)
# df_input['N_K_ratio'] = df_input['Nitrogen'] / (df_input['Potassium'] + 0.0001)
# df_input['P_K_ratio'] = df_input['Phosphorous'] / (df_input['Potassium'] + 0.0001)

df_input['Total_NPK'] = df_input['Nitrogen'] + df_input['Phosphorous'] + df_input['Potassium']
# df_input['NPK_std'] = df_input[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1)



# test_data['N_P_ratio'] = test_data['Nitrogen'] / (test_data['Phosphorous'] + 0.0001)
# test_data['N_K_ratio'] = test_data['Nitrogen'] / (test_data['Potassium'] + 0.0001)
# test_data['P_K_ratio'] = test_data['Phosphorous'] / (test_data['Potassium'] + 0.0001)

test_data['Total_NPK'] = test_data['Nitrogen'] + test_data['Phosphorous'] + test_data['Potassium']
# test_data['NPK_std'] = test_data[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1)




df_input['Moisture_Temp'] = df_input['Moisture'] * df_input['Temparature']
df_input['Humidity_Temp'] = df_input['Humidity'] * df_input['Temparature']
df_input['Moisture_Humidity'] = df_input['Moisture'] * df_input['Humidity']
# df_input['Weather_Index'] = df_input[['Temparature', 'Humidity', 'Moisture']].mean(axis=1)



test_data['Moisture_Temp'] = test_data['Moisture'] * test_data['Temparature']
test_data['Humidity_Temp'] = test_data['Humidity'] * test_data['Temparature']
test_data['Moisture_Humidity'] = test_data['Moisture'] * test_data['Humidity']
# test_data['Weather_Index'] = test_data[['Temparature', 'Humidity', 'Moisture']].mean(axis=1)



df_input.set_index('id' ,inplace= True)



test_data.set_index('id' ,inplace= True)



df_input['Crop Type'].value_counts()


ideal_npk = {
    'Paddy': [120, 35, 50],         # N: 120 kg/ha, P2O5 ~80 â†’ P â‰ˆ 35, K2O ~60 â†’ K â‰ˆ 50
    'Pulses': [25, 26, 25],         # N: 20â€“40, P2O5: 60 â†’ P â‰ˆ 26, K2O: 30â€“50 â†’ K â‰ˆ 25
    'Cotton': [100, 35, 50],        # N: 80â€“120, P â‰ˆ 35, K â‰ˆ 50
    'Tobacco': [130, 35, 80],       # Higher K for leaf quality
    'Wheat': [100, 50, 50],
    'Millets': [50, 17, 25],        # Low input crop; P â‰ˆ 17 (from P2O5 ~40), K â‰ˆ 25
    'Barley': [90, 26, 40],
    'Sugarcane': [180, 35, 80],     # Very high nutrient demand
    'Oil seeds': [50, 26, 25],      # E.g., mustard, sunflower
    'Maize': [120, 35, 50],
    'Ground Nuts': [30, 26, 25]
}


def compute_deficiency(row):
    ideal = ideal_npk.get(row['Crop Type'], [90, 40, 40])
    return pd.Series({
        'N_def': ideal[0] - row['Nitrogen'],
        'P_def': ideal[1] - row['Phosphorous'],
        'K_def': ideal[2] - row['Potassium']
    })

df_input[['N_def', 'P_def', 'K_def']] = df_input.apply(compute_deficiency, axis=1)

5. Binned and z-score versions of environmental variables
for col in ['Temparature', 'Humidity', 'Moisture']:
    df_input[f'{col}_bin'] = pd.qcut(df_input[col], q=5, labels=False, duplicates='drop')
    df_input[f'{col}_zscore'] = (df_input[col] - df_input[col].mean()) / df_input[col].std()



test_data[['N_def', 'P_def', 'K_def']] = test_data.apply(compute_deficiency, axis=1)



for col in ['Temparature', 'Humidity', 'Moisture']:
    test_data[f'{col}_bin'] = pd.qcut(test_data[col], q=5, labels=False, duplicates='drop')
    test_data[f'{col}_zscore'] = (test_data[col] - test_data[col].mean()) / test_data[col].std()



from sklearn.preprocessing import LabelEncoder
# le_soil = LabelEncoder()
# le_crop = LabelEncoder()
# le_fert = LabelEncoder()

test_data['Soil Type'] = le_soil.fit_transform(test_data['Soil Type'])
test_data['Crop Type'] = le_crop.fit_transform(test_data['Crop Type'])




df_input['Soil Type'] = le_soil.fit_transform(df_input['Soil Type'])
df_input['Crop Type'] = le_crop.fit_transform(df_input['Crop Type'])
df_input['Fertilizer Label'] = le_fert.fit_transform(df_input['Fertilizer Name'])



test_data.head()



df_input['Fertilizer Label'].value_counts()


df_names.shape


# df_names['name'] = df_input['Fertilizer Name']
df_names['label'] = df_input['Fertilizer Label']


df_names.head()


df_input.dtypes


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Replace this with your actual DataFrame
# df_input = pd.read_csv("your_data.csv")

# Step 1: Define categorical and numerical columns
categorical_cols = [
    'Soil Type', 'Crop Type', 'Fertilizer Name', 'Fertilizer Label', 
    'Temparature_bin', 'Humidity_bin', 'Moisture_bin'
]

# Step 2: Automatically get numerical columns to scale (exclude categorical and bin)
columns_to_exclude = set(categorical_cols)
numerical_cols = [
    col for col in df_input.columns 
    if df_input[col].dtype in ['int64', 'float64'] and col not in columns_to_exclude
]

# Step 3: Apply StandardScaler to numerical columns
# scaler = StandardScaler()
# df_input[numerical_cols] = scaler.fit_transform(df_input[numerical_cols])
# test_data[numerical_cols] = scaler.fit_transform(test_data[numerical_cols])

# Step 4 (Optional): Convert categorical columns to string if you plan to one-hot encode later
for col in categorical_cols:
    df_input[col] = df_input[col].astype('category')


for col in categorical_cols:
    if col.startswith('Fertilizer'):
        continue  # Skip fertilizer-related columns
    test_data[col] = test_data[col].astype('category')


# Final check
print("Scaled numerical columns:")
print(numerical_cols)

print("\nCategorical columns (as string):")
print(categorical_cols)


test_data.shape


df_input.describe()


df_input.info()


df_input.drop(columns =['Fertilizer Name'] , inplace =True)


from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(
    df_input.drop(columns=['Fertilizer Label']),
    df_input['Fertilizer Label'],
    discrete_features='auto',
    random_state=42
)

# Create the DataFrame using columns from the dropped DataFrame
mi_df = pd.DataFrame({
    'Feature': df_input.drop(columns=['Fertilizer Label']).columns,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)



mi_df = mi_df[mi_df['Feature'] != 'Fertilizer Label']


mi_df['Cumulative'] = mi_df['MI_Score'].cumsum() / mi_df['MI_Score'].sum()
selected = mi_df[mi_df['Cumulative'] <= 0.90]



selected.shape


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.barh(selected['Feature'], selected['MI_Score'])
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.title('Feature Importance via Mutual Information')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



df_input.shape



def remove_outliers_iqr(df, cols=None, factor=1.5):
    """
    Removes rows with outliers in specified numeric columns using the IQR method.
    - cols: list of columns to check (default: all numeric columns)
    - factor: multiplier for IQR (default: 1.5)
    """
    # âœ… Ensure only numeric columns are selected
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if cols is None:
        cols = numeric_cols
    else:
        # Filter the user-passed list to include only numeric columns
        cols = [col for col in cols if col in numeric_cols]

    df_clean = df.copy()
    for col in cols:
        if col == 'Fertilizer Label':
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        before = df_clean.shape[0]
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        after = df_clean.shape[0]
        print(f"{col}: Removed {before - after} outliers")

    return df_clean


df_input = remove_outliers_iqr(df_input, cols=df_input.columns)


y = df_input['Fertilizer Label']


X = df_input.drop(columns = [ 'Fertilizer Label'])


y.value_counts()


import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 ,stratify = y)



X.head()


selected


X_train.shape


X_train = X_train[selected['Feature'].to_list()]



X_test = X_test[selected['Feature'].to_list()]



X_test.shape


y_train.value_counts()


test_data = test_data[selected['Feature'].to_list()]



xgb_results = {k: [] for k in [
    'trial', 'map3_scores', 'avg_map3'
]}
lgbm_results = {k: [] for k in [
    'trial', 'map3_scores', 'avg_map3'
]}
catboost_results = {k: [] for k in [
    'trial', 'map3_scores', 'avg_map3'
]}




df_input.to_feather('/kaggle/working/payload_for_model_29june.feather')


df_input = pd.read_feather('/kaggle/working/payload_for_model_29june.feather')


from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
n_classes = len(np.unique(y))



def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



# List of object-type columns that are categorical
categorical_cols = ['Soil Type', 'Crop Type', 'Temparature_bin', 'Humidity_bin', 'Moisture_bin']

# Convert them to pandas 'category' dtype
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')
    test_data[col] = test_data[col].astype('category')
    


X_train.dtypes


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import numpy as np


test_data.head()



def objective(trial, model_type='xgb' ,results_dict= None):
    cat_cols_lgbm = X_train.select_dtypes(include='category').columns.tolist()
    if model_type == 'xgb':
        params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': 7,
            'max_depth': trial.suggest_int('max_depth', 1, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'verbosity': 0,
            'enable_categorical': True,
            'max_bin': 128,
        }
        model_func = lambda: xgb.XGBClassifier(**params, use_label_encoder=False)

    elif model_type == 'lgbm':
        params = {
            'objective': 'multiclass',
            'num_class': 7,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'device_type': 'gpu',         # âœ… GPU
            'gpu_platform_id': 0,
            'gpu_device_id': 0,           # âœ… Use GPU 0
            'verbosity': -1
        }
        model_func = lambda: LGBMClassifier(**params)

    elif model_type == 'catboost':
        params = {
            'loss_function': 'MultiClass',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 10),
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'task_type': 'GPU',           # âœ… GPU
            'devices': '1',               # âœ… Use GPU 1
            'verbose': False
        }
        model_func = lambda: CatBoostClassifier(**params)

    elif model_type == 'svm':
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'probability': True,
        }
        model_func = lambda: SVC(**params)

    elif model_type == 'lr':
        params = {
            'C': trial.suggest_float('C', 0.01, 10.0),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
            'multi_class': 'multinomial',
            'max_iter': 1000
        }
        model_func = lambda: LogisticRegression(**params)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((X_train.shape[0], 7))
    map3_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        x_tr, x_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        y_tr = y_tr.astype(int)
        y_val = y_val.astype(int)
        model = model_func()
        if model_type == 'catboost':
            print('x')
            model.fit(x_tr,y_tr,cat_features=cat_features)
        
        elif  model_type =='lgbm' :
            model.fit(x_tr, y_tr ,categorical_feature=cat_cols_lgbm )
        else:
            model.fit(x_tr,y_tr)

        val_probs = model.predict_proba(x_val)
        oof_preds[valid_idx] = val_probs

        top_3_preds = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y_val]
        fold_map3 = mapk(actual, top_3_preds, k=3)
        map3_scores.append(fold_map3)

    oof_top3 = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
    actual_all = [[label] for label in y_train]
    avg_map3 = mapk(actual_all, oof_top3, k=3)
    trial.set_user_attr("oof_preds", oof_preds)
    trial.set_user_attr("model", model)
    
    if results_dict is not None:
        results_dict['trial'].append(trial.number)
        results_dict['map3_scores'].append(map3_scores)
        results_dict['avg_map3'].append(avg_map3)


    return avg_map3


cat_features = X_train.select_dtypes(include=['category', 'object']).columns.tolist()


# model.fit(x_tr, y_tr, cat_features=cat_features)
cat_features


model_types = ['xgb']


best_trials = {}
oof_dict = []
oof_dict = {}
model_dict = {}



for model_type in model_types:
    print(f" Running Optuna for {model_type}")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_type=model_type), n_trials=15)
    
    best_trials[model_type] = study.best_trial
    oof_dict[model_type] = study.best_trial.user_attrs["oof_preds"]
    model_dict[model_type] = study.best_trial.user_attrs["model"]

    print(f"âœ… Best {model_type} MAP@3: {study.best_value:.5f}")


# study


ensemble_oof = (
    oof_dict['xgb'] + oof_dict['lgbm'] + oof_dict['catboost']
) / 3

ensemble_top3 = np.argsort(ensemble_oof, axis=1)[:, -3:][:, ::-1]
ensemble_map3 = mapk([[y] for y in y_train], ensemble_top3, k=3)

print(f"ðŸ“ˆ Ensemble MAP@3: {ensemble_map3:.5f}")



selected


  params = {
            'objective': 'multi:softprob',
            'num_class': 7,
            'max_depth': 7,
            'learning_rate': 0.01,
            'n_estimators': 100_000,
            'reg_alpha': 7,
            'reg_lambda': 5.3,
            'gamma': 0.3,
            'max_delta_step': 4,
            'subsample': 0.86,
            'colsample_bytree': 0.4,
            'min_child_weight': 5,
            'random_state': 43,
            'eval_metric': 'mlogloss',
            'enable_categorical': True,
            'device': "cuda"
            }


X_test = test_data[selected['Feature'].to_list()]


X_test.head()


import joblib

for model_type in model_types:
    print(f"ðŸ’¾ Training and saving final {model_type.upper()} model...")

    # best_params = best_trials[model_type].params

    # Construct final model using best params
    if model_type == 'xgb':
        final_model = xgb.XGBClassifier(
            max_depth = 7,
            learning_rate =0.01,
            n_estimators = 1000 ,
            reg_alpha = 7,
            reg_lambda = 5.3,
            gamma = 0.3,
            max_delta_step = 4,
            subsample =  0.86,
            colsample_bytree = 0.4,
            min_child_weight =  5,
            random_state = 42 ,
            enable_categorical = True,
            objective='multi:softprob',
            eval_metric='mlogloss',
            device = "cuda",
            # tree_method='gpu_hist',
            # predictor='gpu_predictor',
            num_class=7,
        )

    elif model_type == 'lgbm':
        final_model = LGBMClassifier(
            **best_params,
            objective='multiclass',
            num_class=7,
            device_type='gpu',
            gpu_platform_id = 0,
            gpu_device_id =0,           # âœ… Use GPU 0
            verbosity = -1
        )

    elif model_type == 'catboost':
        final_model = CatBoostClassifier(
            **best_params,
            loss_function='MultiClass',
            task_type='GPU',
            devices = '1',               # âœ… Use GPU 1
            verbose = False
        )

    else:
        continue  # Skip unsupported

    # âœ… Fit on full training data
    final_model.fit(X_train, y_train )

    # âœ… Save model to file
    model_path = f"final_model_{model_type}.pkl"
    joblib.dump(final_model, model_path)
    print(f"âœ… Saved {model_type} model to: {model_path}")



    # âœ… Store in model_dict for reuse
    # model_dict[model_type] = final_model



test_data = test_data[selected['Feature'].to_list()]


test_data.head()


import pickle
import numpy as np

# 1. Load models correctly
with open('/kaggle/working/final_model_xgb.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# with open('/kaggle/working/final_model_catboost.pkl', 'rb') as f:
#     catboost_model = pickle.load(f)

# 2. Predict probabilities
xgb_test_probs = xgb_model.predict_proba(test_data)
# catboost_test_probs = catboost_model.predict_proba(X_test)

# 1. Average the probabilities
ensemble_probs = xgb_test_probs

# 2. Get top 3 class indices
ensemble_top3 = np.argsort(ensemble_probs, axis=1)[:, -3:][:, ::-1]





test_data


ensemble_top3


# 2. Build submission DataFrame
ensemble_preds_df = pd.DataFrame({
    'id': test_data.index,
    'pred1': ensemble_top3[:, 0],
    'pred2': ensemble_top3[:, 1],
    'pred3': ensemble_top3[:, 2]
})



best_trials['catboost'].params
model_type = ['xgb']
# cat_features



xgb_results_df = pd.DataFrame(xgb_results)

xgb_results_df.to_csv('xgb_results.csv', index = False)



import json

trial_info = {
    "trial_number": 18,
    "value": 0.329209,
    "params": {
        "max_depth": 7,
        "learning_rate": 0.04115985547,
        "n_estimators": 1035,
        "subsample": 0.922033,
        "colsample_bytree": 0.672724
    }
}

# Save to JSON
with open("xgb_best_trial_info.json", "w") as f:
    json.dump(trial_info, f, indent=4)



import json

# Save best params to JSON
with open("best_xgb_params.json", "w") as f:
    json.dump(best_params, f, indent=4)



test_data.head()


X_test.rename(columns ={'Temperature':'Temparature'} ,inplace =True)


X_test.columns


label_to_name = dict(zip(df_input['Fertilizer Label'], df_input['Fertilizer Name']))



# Apply mapping to each prediction column
ensemble_preds_df['pred1'] = ensemble_preds_df['pred1'].map(label_to_name)
ensemble_preds_df['pred2'] = ensemble_preds_df['pred2'].map(label_to_name)
ensemble_preds_df['pred3'] = ensemble_preds_df['pred3'].map(label_to_name)



ensemble_preds_df


submission = pd.DataFrame({
    'id': ensemble_preds_df['id'],  # your test IDs
    'Fertilizer Name': ensemble_preds_df['pred1'] + " " + ensemble_preds_df['pred2'] + " " + ensemble_preds_df['pred3']
})



submission.head()


submission.to_csv('/kaggle/working/submission.csv' ,index =False)


test_data.head()


submission.shape




