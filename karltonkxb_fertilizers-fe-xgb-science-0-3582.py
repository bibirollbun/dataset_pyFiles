import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict, KFold

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import lightgbm as lgb
import catboost as cat
import xgboost as xgb


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
source_data = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

train["dataset"] = "train"
source_data['dataset'] = 'train'
test["dataset"] = "test"

df = pd.concat([train.copy(), test.copy(), source_data.copy()], ignore_index = True).drop(columns=['id'])

df


# downcast float columns
def downcast_float(df):
    for col in df.select_dtypes(include=np.floating).columns:
        min_val = df[col].min()
        max_val = df[col].max()
    
        if min_val >= np.finfo(np.float16).min and max_val <= np.finfo(np.float16).max:
            df[col] = df[col].astype(np.float16)
        
        elif min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)
    
    return df

# downcast int columns
def downcast_int(df):
    for col in df.select_dtypes(include=np.integer).columns:
        min_val = df[col].min()
        max_val = df[col].max()
    
        if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    
    return df

# check the memory usage before and after
print("Memory usage before downcasting in MB:")
print(df.memory_usage(deep=True).sum() / (1024 ** 2))


df = downcast_int(df)

# after downcasting
print("Memory usage after downcasting in MB:")
print(df.memory_usage(deep=True).sum() / (1024 ** 2))

print()
df.info()


# convert target to categorical codes
train['Fertilizer_Code'] = train['Fertilizer Name'].astype('category').cat.codes

# numerical features
num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
X_num = train[num_features]
y = train['Fertilizer_Code']

# mutual Information
mi_scores = mutual_info_classif(X_num, y, discrete_features=False, random_state=42)
mi_series = pd.Series(mi_scores, index=num_features).sort_values(ascending=False)

print("Mutual Information Scores:\n\n", mi_series)



from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

# Compute for each categorical feature
cat_features = ['Soil Type', 'Crop Type']
for feature in cat_features:
    score = cramers_v(train[feature], train['Fertilizer Name'])
    print(f"Cramerâ€™s V for {feature} <--> Fertilizer Name: {score:.4f}")


target_col = 'Fertilizer Name'

chi2_results = {}
for col in cat_features:
    contingency = pd.crosstab(train[col], train[target_col])
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency)
    chi2_results[col] = p_val

pd.Series(chi2_results).sort_values()


n_cols = 3
n_rows = (len(num_features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
axes = axes.flatten()

# barplots
for idx, col in enumerate(num_features):
    sns.barplot(x=target_col, y=col, data=train, ci='sd', ax=axes[idx])
    axes[idx].set_title(f'{col} by {target_col}')
    axes[idx].tick_params(axis='x', rotation=45)

# remove unused subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(h_pad=3)
plt.show()


train_num = train[num_features + ['Fertilizer_Code']]  # use target as code
corr = train_num.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap (Numerical Features + Target Code)")
plt.tight_layout()
plt.show()



# feature interaction with - 'Nitrogen', 'Potassium', 'Phosphorous'

nutrient_lookup = {
    "28-28":     [28,  28,   0],
    "17-17-17":  [17,  17,  17],
    "10-26-26":  [10,  26,  26],
    "DAP":       [18,  46,   0],
    "20-20":     [20,  20,   0],
    "14-35-14":  [14,  35,  14],
    "Urea":      [46,   0,   0]
}

mean_nitrogen = np.mean([i[0] for i in nutrient_lookup.values()])
mean_potassium = np.mean([i[1] for i in nutrient_lookup.values()])
mean_phosphorous = np.mean([i[2] for i in nutrient_lookup.values()])

median_nitrogen = np.median([i[0] for i in nutrient_lookup.values()])
median_potassium = np.median([i[1] for i in nutrient_lookup.values()])
median_phosphorous = np.median([i[2] for i in nutrient_lookup.values()])

std_nitrogen = np.std([i[0] for i in nutrient_lookup.values()])
std_potassium = np.std([i[1] for i in nutrient_lookup.values()])
std_phosphorous = np.std([i[2] for i in nutrient_lookup.values()])


# std_levels = [1, 2, 3]
# for level in std_levels:
#     # Standard deviation based calculations
#     df[f"chemical_mixer_std_{level}"] = (
#         (df['Nitrogen'] * (level * std_nitrogen) * 0.01) +
#         (df['Phosphorous'] * (level * std_phosphorous) * 0.01) +
#         (df['Potassium'] * (level * std_potassium) * 0.01)
#     )

# chemical_cols = ['chemical_mixer_std_1', 'chemical_mixer_std_2', 'chemical_mixer_std_3']


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def add_season_by_clustering(df):
    df = df.copy()
    features = ['Temparature', 'Humidity', 'Moisture']
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
    df['season_cluster'] = kmeans.fit_predict(X_scaled)

    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)

    centroid_df = pd.DataFrame(
        centroids,
        columns=features
    )
    centroid_df['cluster_label'] = centroid_df.index
    centroid_df = centroid_df.sort_values(by='Temparature').reset_index(drop=True)

    # corresponds to "Winter", "Spring", "Fall", "Summer", "Rainy".

    season_names = ['Winter', 'Spring', 'Fall', 'Summer', "Rainy"]
    cluster_to_season = {
        int(row.cluster_label): season_names[i]
        for i, row in enumerate(centroid_df.itertuples(index=False))
    }

    df['season'] = df['season_cluster'].map(cluster_to_season)
    df = df.drop(columns=['season_cluster'])
    return df


df = add_season_by_clustering(df)


crop_type_train = set(df.loc[df.dataset == 'train', 'Crop Type'].unique())
crop_type_test  = set(df.loc[df.dataset == 'test', 'Crop Type'].unique())

if len(crop_type_train ^ crop_type_test) > 1:
    print(f"There is new Crop type in test set: {crop_type_train ^ crop_type_test}")
else:
    print("There is not any new Crop type in test set")    

soil_type_train = set(df.loc[df.dataset == 'train', 'Soil Type'].unique())
soil_type_test  = set(df.loc[df.dataset == 'test', 'Soil Type'].unique())

if len(soil_type_train ^ soil_type_test) > 1:
    print(f"There is new Soil type in test set: {soil_type_train ^ soil_type_test}")
else:
    print("There is not any new Soil type in test set")


# encoders
df['Soil_Crop'] = df['Soil Type'] + "_" + df['Crop Type']

# Initialize label encoders
label_encoder_crop_type = LabelEncoder()
label_encoder_soil_type = LabelEncoder()
label_encoder_soil_crop_type = LabelEncoder()
label_encoder_season = LabelEncoder()
label_encoder_chemical_cluster = LabelEncoder()

# Fit and transform
df['Encoded_Crop_Type'] = label_encoder_crop_type.fit_transform(df['Crop Type'])
df['Encoded_Soil_Type'] = label_encoder_soil_type.fit_transform(df['Soil Type'])
df['Encoded_Soil_Crop'] = label_encoder_soil_crop_type.fit_transform(df['Soil_Crop'])
df['Encoded_Season'] = label_encoder_season.fit_transform(df['season'])

# Drop original columns and keep encoded columns
df_final = df.drop(columns=['Crop Type', 'Soil Type', 'Soil_Crop', 'season'])

# Optionally, you can rename the encoded columns for clarity
df_final.rename(columns={
    'Encoded_Crop_Type': 'Crop_Type_Encoded',
    'Encoded_Soil_Type': 'Soil_Type_Encoded',
    'Encoded_Soil_Crop': 'Soil_Crop_Encoded',
    'Encoded_Soil_Season': 'Season_Encoded',
}, inplace=True)

df_final


from sklearn.preprocessing import PolynomialFeatures

def feature_engineer(df, fe_columns: list[str] = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']):
    df_poly = df.copy()

    # base_feats = df[fe_columns]

    # poly = PolynomialFeatures(degree=2, include_bias=False)
    # poly_array = poly.fit_transform(base_feats)
    # poly_feature_names = poly.get_feature_names_out(base_feats.columns)

    # df_poly = pd.DataFrame(
    #     poly_array,
    #     columns=poly_feature_names,
    #     index=df.index
    # )

    df_poly['Temp_Humidity_Interaction'] = df_poly['Temparature'] * df_poly['Humidity']
    df_poly['N_P_Ratio'] = df_poly['Nitrogen'] / (df_poly['Phosphorous'].replace(0, 1e-6))
    df_poly['K_P_Ratio'] = df_poly['Potassium'] / (df_poly['Phosphorous'].replace(0, 1e-6))
    df_poly['N_K_Ratio'] = df_poly['Nitrogen'] / (df_poly['Potassium'].replace(0, 1e-6))
    df_poly['P_K_Ratio'] = df_poly['Phosphorous'].replace(0, 1e-6) / (df_poly['Potassium'].replace(0, 1e-6))
    df_poly['N_P_K_interaction'] = df_poly['Phosphorous'] + df_poly['Potassium'] + df_poly['Nitrogen']

    # # keep only the new features (exclude the original base columns)
    # new_cols = [c for c in df_poly.columns if c not in base_feats.columns]
    # # df = pd.concat([df, df_poly[new_cols]], axis=1)

    # col_to_return = [c for c in new_cols if "^2" not in c]
    col_to_return = [
                    'Temp_Humidity_Interaction',
                     'N_P_Ratio',
                     # 'K_P_Ratio',
                     # 'N_K_Ratio',
                     # 'P_K_Ratio',
                     'N_P_K_interaction'
                    ]
    
    return df_poly[col_to_return]

# apply feature engineering
columns_feature_engineering = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
df_fe = feature_engineer(df, columns_feature_engineering)

df_fe


# # apply standard scaler to numerical columns
# from sklearn.preprocessing import StandardScaler

# num_cols = columns_feature_engineering + chemical_cols + list(df_fe.columns)

# concatenate dataframes
df_final1 = pd.concat([df_final, df_fe], ignore_index=False, axis=1)

# scaler = StandardScaler()

# df_final1[num_cols] = scaler.fit_transform(df_final1[num_cols])
# df_final1


print("Memory usage before downcasting in MB:")
print(df_final1.memory_usage(deep=True).sum() / (1024 ** 2))

df_final1 = downcast_float(df_final1)

print("\nMemory usage after downcasting in MB:")
print(df_final1.memory_usage(deep=True).sum() / (1024 ** 2))


df_train = df_final1[df_final1.dataset == 'train'].drop(columns=['dataset'])
df_test  = df_final1[df_final1.dataset == 'test'].drop(columns=['dataset', 'Fertilizer Name'])

X = df_train.drop(columns=['Fertilizer Name'])
y = df_train['Fertilizer Name']

# encode target column
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # fit and transform the target column


corr_extended = X.corrwith(pd.Series(y_encoded), numeric_only=True)

top_corr = corr_extended.sort_values(ascending=False).head(50)
top_corr = corr_extended[top_corr.index]

# plot
plt.figure(figsize=(15, 10))
sns.barplot(x=top_corr.values, y=top_corr.index, palette='viridis')
plt.title('Top 30 Features Correlated with Fertilizer')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Feature')
plt.axvline(0, color='gray', linewidth=0.8)
plt.tight_layout()
plt.show()





def mapk(y_true, y_pred, k=3):
    score = 0.0
    for true, pred in zip(y_true, y_pred):
        if true in pred:
            rank = pred.index(true) + 1
            score += 1.0 / rank

    return score / len(y_true)


def map3(y_true, y_pred_probs):
    y_true = [[x] for x in y_true]
    y_pred_probs = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1].tolist()
    
    def ap3(y_true, y_pred_probs):
        y_pred_probs = y_pred_probs[:3]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(y_pred_probs):
            if p in y_true and p not in y_pred_probs[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not y_true:
            return 0.0

        return score
    
    return np.mean([ap3(a,p) for a,p in zip(y_true, y_pred_probs)])


from sklearn.preprocessing import label_binarize
import numpy as np


def map3_score(predicted_top3: np.ndarray,   # shape = (n_val, 3), dtype = object or int
               y_true_fold: np.ndarray,      # shape = (n_val,)
              ) -> float:
    """
    predicted_top5[i] is a lengthâ€�3 array of labels (strings/ints) that your model thinks
    are most likely for sample i, ordered from most confident 3rd most confident.
    y_true_fold[i] is the single true label for sample i.
    We give credit = 1/rank if the true label is at position 'rank' in that topâ€�3 list;
    otherwise 0. Then we average over all i.
    """
    print(type(predicted_top3), type(y_true_fold))
    
    n_val = y_true_fold.shape[0]
    total_score = 0.0

    for i in range(n_val):
        true_label = y_true_fold[i]
        top3_preds = predicted_top3[i].tolist()  # convert row to a Python list

        try:
            # .index(...) returns 0-based position. Add +1 to get 1-based rank.
            rank = top3_preds.index(true_label) + 1
            if rank <= 3:
                total_score += 1.0 / rank
            # If rank > 3, that cannot happen here, because top3_preds has exactly 3 items.
        except ValueError:
            # true_label not in top-3  score += 0
            pass

    return total_score / n_val


import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


X_train, X_valid, y_train, y_valid = train_test_split(
    X, pd.Series(y_encoded), test_size=0.2, stratify=y_encoded, random_state=42)


def xgboost_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 31),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_float('subsample', 0, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
        'eta': trial.suggest_float('eta', 0, 1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    map3_scores = []

    for train_index, val_index in cv.split(X_train, y_train):

        x_train_fold = X_train.iloc[train_index]
        x_val_fold = X_train.iloc[val_index]
        
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]
        
        model = XGBClassifier(
            **params,
            verbosity=0,
            objective='multi:softprob',
            enable_categorical=True,
            tree_method='hist',
            # tree_method="gpu_hist",
            # gpu_id=0, 
            # predictor="gpu_predictor",
            n_jobs=-1,
            random_seed=42,
            early_stopping_rounds=75,
        )
        
        model.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
               verbose=False)

        pred_proba = model.predict_proba(x_val_fold)
        top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
        class_labs = model.classes_
        top3_labs = class_labs[top3_index]

        fold_map3 = map3_score(top3_labs, y_val_fold.to_numpy())
        map3_scores.append(fold_map3)
        mean_map3 = np.mean(map3_scores)

    return mean_map3


optimize_xgb = False

if optimize_xgb:
    # run study
    study = optuna.create_study(direction="maximize", study_name="xgb_optimization", sampler=TPESampler(n_startup_trials=30, seed=42, multivariate=True))
    study.optimize(xgboost_objective, n_trials=100, n_jobs=1)
    print("Best trial:")
    print(study.best_trial.params)
    best_trial_params = study.best_trial.params


params_xgb ={
         'learning_rate': 0.03188950451801693, 
        'max_depth': 13, 
        'min_child_weight': 5.290774921382717, 
        'gamma': 0.370024188161062, 
        'alpha': 0.24220634641035022, 
        'subsample': 0.47612738905078533, 
        'colsample_bytree': 0.3200756289396693, 
        'eta': 0.9252726435636992, 
        'n_estimators': 1189, 
        'lambda': 0.106343712713482,
        'objective': 'multi:softprob', 
        'eval_metric': 'mlogloss', 
        'num_class': 7,  
        'device': 'cuda', 
        'tree_method': 'gpu_hist',
        # 'tree_method': 'hist',
        'predictor': 'gpu_predictor',
        'use_label_encoder': False, 
        'n_jobs': -1, 
        'random_state': 42,
    }


# Catboost classifier

X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.1, random_state=42)


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.03, log=True),
        'depth': trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'iterations': trial.suggest_int('iterations', 700, 1500),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'verbose': 0,
        'task_type': 'GPU',
        'devices': '0',
        'random_seed': 42
    }

    model = CatBoostClassifier(**params)

    model.fit(X_train, y_train, 
              eval_set=(X_valid, y_valid), 
              early_stopping_rounds=300, 
              use_best_model=True)

    pred_probs = model.predict_proba(X_valid)
    top3 = np.argsort(-pred_probs, axis=1)[:, :3]
    top3_list = [list(line) for line in top3]

    map_score = mapk(y_valid, top3_list, k=3)

    return map_score

optimize_catboost = False

if optimize_catboost:
    # Optimize CatBoost
    study = optuna.create_study(direction='maximize', study_name='catboost_map3_opt')
    study.optimize(objective, n_trials=15)
    
    # Best result
    print("Best MAP@3 score: {:.4f}".format(study.best_value))
    print("Best parameters:", study.best_params)



params_cat = {
        'learning_rate': 0.029117889991646665,
        'depth': 8,
        'l2_leaf_reg': 0.04492566030061831,
        'iterations': 1142,
        'bagging_temperature': 0.37734640187254076,
        'random_strength': 0.593183564121753,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'verbose': 0,
        'task_type': 'GPU',
        'devices': '0',
        'random_seed': 42,
    }


if True:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.1, random_state=42, shuffle=True)
    
    model_xgb = XGBClassifier(**params_xgb)
        
    model_xgb.fit(X_train, y_train, eval_set=[(X_valid,y_valid)], verbose=0)
        
    pred_probs = model_xgb.predict_proba(X_valid)
    
    top3 = np.argsort(-pred_probs, axis=1)[:, :3]
    top3_list = [list(line) for line in top3]
        
    map_score = mapk(y_valid, top3_list, k=3)
    print(f"XGBBoost - mapk: {map_score:.4f}")
    
    model_xgb.fit(X, y_encoded,  eval_set=[(X_valid,y_valid)], verbose=0)


# Extract the feature importances and feature names from the model
importances = model_xgb.feature_importances_
feature_names = model_xgb.feature_names_in_

# Create a DataFrame for easier plotting and sorting
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='importance', y='feature', data=feat_imp_df, palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('XGB Feature Importances')
plt.show()


# probs = model_xgb.predict_proba(df_test)

# top3_preds = np.argsort(-probs, axis=1)[:, :3]
# top3_list = [list(line) for line in top3_preds]

# decoded_top3_string = [" ".join(label_encoder.inverse_transform(line)) for line in top3_list]

# submission = pd.DataFrame({
#     "id": test['id'],
#     "Fertilizer Name" : decoded_top3_string
# })

# submission.to_csv("submission.csv", index = False)

# submission


def map_calculator(y_valid, pred_probs):
    top3 = np.argsort(-pred_probs, axis=1)[:, :3]
    top3_list = [list(line) for line in top3]

    map_score = mapk(y_valid, top3_list, k=3)

    return map_score


import time
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold

X = X.copy()
y = pd.Series(y_encoded)
test_features = df_test

n_classes = len(np.unique(y_encoded))

# Stratified K-Fold setup
FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_cat = np.zeros((len(X), n_classes), dtype=int)
oof_xgb = np.zeros((len(X), n_classes), dtype=int)

pred_cat_proba_folds = np.zeros((FOLDS, len(test_features), n_classes))
pred_xgb_proba_folds = np.zeros((FOLDS, len(test_features), n_classes))

# cat_model = CatBoostClassifier(**params_cat)
xgb_model = xgb.XGBClassifier(**params_xgb)

#K-Fold loop: train and collect proba

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    start_time = time.time()
    print(f"\n{'--'*12} FOLD {fold+1} {'--'*12}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # # --- CatBoostClassifier ---
    # cat_model.fit(
    #     X_train, y_train,
    #     eval_set=(X_valid, y_valid),
    #     early_stopping_rounds=150,
    #     verbose=False
    # )

    # proba_valid_cat = cat_model.predict_proba(X_valid)
    # oof_cat[valid_idx] = proba_valid_cat
    # pred_cat_proba_folds[fold] = cat_model.predict_proba(test_features)

    # map_3_cat  = map_calculator(y_valid, proba_valid_cat)

    # --- XGBClassifier ---
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=150,
        verbose=False
    )

    proba_valid_xgb = xgb_model.predict_proba(X_valid)
    oof_xgb[valid_idx] = proba_valid_xgb

    pred_xgb_proba_folds[fold] = xgb_model.predict_proba(test_features)

    map_3_xgb  = map_calculator(y_valid, proba_valid_xgb)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f" CatBoost - map3: {map_3_cat:.4f}, XGBoost - map3: {map_3_xgb:.4f}, time taken for fold {fold + 1}: {elapsed_time:.2f} seconds ~ {elapsed_time/60:.2f} minutes")
    print(f"XGBoost - map3: {map_3_xgb:.4f}, time taken for fold {fold + 1}: {elapsed_time:.2f} seconds ~ {elapsed_time/60:.2f} minutes")



# average probabilities over folds (axis=0 gives shape (n_test, n_classes))
# cat_avg_proba = pred_cat_proba_folds.mean(axis=0)
xgb_avg_proba = pred_xgb_proba_folds.mean(axis=0)


cat_weight = 0.20
xgb_weight = 0.80
# ensemble_avg_proba = (cat_weight * cat_avg_proba) + (xgb_weight * xgb_avg_proba)


# evaluate OOF performance on the whole training set
# map_3_cat  = map_calculator(y, oof_cat) 
# map_3_xgb  = map_calculator(y, oof_xgb)

# print(f"Overall CatBoost - map3: {map_3_cat:.4f}, XGBoost - map3: {map_3_xgb:.4f}")



# top3_indices = np.argsort(ensemble_avg_proba, axis=1)[:, -3:][:, ::-1]
top3_indices = np.argsort(xgb_avg_proba, axis=1)[:, -3:][:, ::-1]

top3_labels = label_encoder.inverse_transform(top3_indices.ravel())

top3_labels_reshaped = top3_labels.reshape(len(test), 3)
decoded_top3_string = [" ".join(row) for row in top3_labels_reshaped]

submission = pd.DataFrame({
    "id": test['id'],
    "Fertilizer Name": decoded_top3_string
})

submission.to_csv("submission.csv", index=False)

submission.head()




