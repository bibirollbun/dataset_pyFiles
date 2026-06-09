import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


# 1 | Data Loading
train_path = "/kaggle/input/playground-series-s5e4/train.csv"  
test_path = "/kaggle/input/playground-series-s5e4/test.csv"   
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("Train Data:", train_df.shape)
print("Test Data:", test_df.shape)


# 2 | Advanced Preprocessing & Group-Level Imputation
def group_impute(df, group_col, target_col):
    grp_mean = df.groupby(group_col)[target_col].transform("mean")
    return df[target_col].fillna(grp_mean)

for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    train_df[col] = group_impute(train_df, 'Podcast_Name', col)
    test_df[col] = group_impute(test_df, 'Podcast_Name', col)

# Fill other missing numerical values with overall mean
num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
for col in num_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mean())
    test_df[col] = test_df[col].fillna(test_df[col].mean())

# Fill missing categorical values with mode
cat_cols = ['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment']
for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])


# 3 | Feature Engineering
# Extract numeric part from Episode_Title if available
train_df["Episode_Title_Num"] = train_df["Episode_Title"].str.extract(r'(\d+)').astype(float)
test_df["Episode_Title_Num"] = test_df["Episode_Title"].str.extract(r'(\d+)').astype(float)

# Create Publication feature
train_df["Publication"] = train_df["Publication_Day"].astype(str) + "_" + train_df["Publication_Time"].astype(str)
test_df["Publication"] = test_df["Publication_Day"].astype(str) + "_" + test_df["Publication_Time"].astype(str)

# New Interaction Features
train_df['Host_Guest_Avg_Popularity'] = (train_df['Host_Popularity_percentage'] + train_df['Guest_Popularity_percentage']) / 2
test_df['Host_Guest_Avg_Popularity'] = (test_df['Host_Popularity_percentage'] + test_df['Guest_Popularity_percentage']) / 2

train_df['Popularity_Diff'] = train_df['Host_Popularity_percentage'] - train_df['Guest_Popularity_percentage']
test_df['Popularity_Diff'] = test_df['Host_Popularity_percentage'] - test_df['Guest_Popularity_percentage']

# Binary feature for ads
train_df['Has_Ads'] = (train_df['Number_of_Ads'] > 0).astype(int)
test_df['Has_Ads'] = (test_df['Number_of_Ads'] > 0).astype(int)

# TF-IDF on Podcast_Name
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=10)
train_podcast_tfidf = tfidf.fit_transform(train_df['Podcast_Name'])
test_podcast_tfidf = tfidf.transform(test_df['Podcast_Name'])
train_tfidf_df = pd.DataFrame(train_podcast_tfidf.toarray(), columns=[f'Podcast_TFIDF_{i}' for i in range(train_podcast_tfidf.shape[1])])
test_tfidf_df = pd.DataFrame(test_podcast_tfidf.toarray(), columns=[f'Podcast_TFIDF_{i}' for i in range(test_podcast_tfidf.shape[1])])
train_df = pd.concat([train_df.reset_index(drop=True), train_tfidf_df], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_tfidf_df], axis=1)
# Drop original Podcast_Name as we now have TF-IDF features
train_df.drop(columns=['Podcast_Name'], inplace=True)
test_df.drop(columns=['Podcast_Name'], inplace=True)

# Encode remaining categorical features with LabelEncoder
for col in ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Publication']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Create polynomial features for numerical columns (interaction terms)
poly_cols = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Episode_Title_Num']
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
train_poly = poly.fit_transform(train_df[poly_cols])
test_poly = poly.transform(test_df[poly_cols])
poly_features = [f'poly_{i}' for i in range(train_poly.shape[1])]
train_poly_df = pd.DataFrame(train_poly, columns=poly_features)
test_poly_df = pd.DataFrame(test_poly, columns=poly_features)
train_df = pd.concat([train_df.reset_index(drop=True), train_poly_df], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_poly_df], axis=1)


# 4 | Define Features and Target
TARGET = 'Listening_Time_minutes'
# Select features from categorical, numerical, TF-IDF, and polynomial features
FEATURES = ['Episode_Title_Num','Episode_Length_minutes','Host_Popularity_percentage',
            'Guest_Popularity_percentage','Number_of_Ads','Genre','Publication_Day',
            'Publication_Time','Episode_Sentiment','Publication','Host_Guest_Avg_Popularity',
            'Popularity_Diff','Has_Ads'] + list(train_tfidf_df.columns) + poly_features
print("Total features used:", len(FEATURES))

# Standardize numerical features
scaler = StandardScaler()
for col in FEATURES:
    if train_df[col].dtype in ['float64','int64']:
        train_df[col] = scaler.fit_transform(train_df[[col]])
        test_df[col] = scaler.transform(test_df[[col]])


# 5 | Model Training & Stacking Ensemble
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Initialize out-of-fold predictions for base models
oof_cat = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_lgbm = np.zeros(len(train_df))
pred_cat = np.zeros(len(test_df))
pred_xgb = np.zeros(len(test_df))
pred_lgbm = np.zeros(len(test_df))

# Define categorical features list (using original names before label encoding)
CATS = ['Genre','Publication_Day','Publication_Time','Episode_Sentiment','Publication']

print("Training base models with stacking...")

for i, (train_index, val_index) in enumerate(kf.split(train_df), 1):
    print(f"\n### Fold {i}")
    # For all models, extract training and validation splits.
    X_tr = train_df.loc[train_index, FEATURES].copy()
    y_tr = train_df.loc[train_index, TARGET].copy()
    X_val = train_df.loc[val_index, FEATURES].copy()
    y_val = train_df.loc[val_index, TARGET].copy()
    
    # --- CatBoost Branch (convert categorical features to strings) ---
    X_tr_cat = X_tr.copy()
    X_val_cat = X_val.copy()
    X_test_cat = test_df[FEATURES].copy()
    for col in CATS:
        X_tr_cat[col] = X_tr_cat[col].astype(str)
        X_val_cat[col] = X_val_cat[col].astype(str)
        X_test_cat[col] = X_test_cat[col].astype(str)
        
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.0877,
        depth=10,
        l2_leaf_reg=0.126,
        bootstrap_type='Bayesian',
        random_strength=4e-08,
        bagging_temperature=0.36,
        od_type='Iter',
        od_wait=39,
        verbose=200,
        allow_writing_files=False,
        task_type='GPU',
        cat_features=CATS,
        random_seed=42
    )
    model_cat.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), early_stopping_rounds=500, verbose=200)
    oof_cat[val_index] = model_cat.predict(X_val_cat)
    pred_cat += model_cat.predict(X_test_cat) / FOLDS

    # --- XGBoost Branch (convert categorical features to 'category' dtype) ---
    X_tr_xgb = X_tr.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = test_df[FEATURES].copy()
    for col in CATS:
        X_tr_xgb[col] = X_tr_xgb[col].astype('category')
        X_val_xgb[col] = X_val_xgb[col].astype('category')
        X_test_xgb[col] = X_test_xgb[col].astype('category')
    
    model_xgb = XGBRegressor(
        n_estimators=2800,
        eta=0.00946,
        gamma=0.2866,
        max_depth=31,
        min_child_weight=47,
        subsample=0.6956,
        colsample_bytree=0.3671,
        grow_policy='lossguide',
        max_leaves=73,
        enable_categorical=True,
        n_jobs=-1,
        tree_method='hist',
        random_state=42
    )
    model_xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], early_stopping_rounds=500, verbose=200)
    oof_xgb[val_index] = model_xgb.predict(X_val_xgb)
    pred_xgb += model_xgb.predict(X_test_xgb) / FOLDS

    # --- LightGBM Branch (categorical features already numeric) ---
    model_lgbm = LGBMRegressor(
        n_estimators=3500,
        random_state=42,
        max_bin=1024,
        colsample_bytree=0.6,
        reg_lambda=80,
        verbosity=-1,
        num_leaves=64,  
        max_depth=15,  
        learning_rate=0.05,  
        feature_fraction=0.8,  
        bagging_fraction=0.8,  
        lambda_l1=0.1, 
        lambda_l2=0.1 
    )
    # Remove verbose parameter here as LGBMRegressor.fit() does not support it.
    model_lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    oof_lgbm[val_index] = model_lgbm.predict(X_val)
    pred_lgbm += model_lgbm.predict(test_df[FEATURES]) / FOLDS

# Compute base model CV RMSEs
rmse_cat = np.sqrt(mean_squared_error(train_df[TARGET], oof_cat))
rmse_xgb = np.sqrt(mean_squared_error(train_df[TARGET], oof_xgb))
rmse_lgbm = np.sqrt(mean_squared_error(train_df[TARGET], oof_lgbm))
print("CV RMSEs: CatBoost: {:.4f}, XGBoost: {:.4f}, LGBM: {:.4f}".format(rmse_cat, rmse_xgb, rmse_lgbm))


# 6 | Stacking Meta-Model
# Create new DataFrames with base model predictions as features
stacked_train = pd.DataFrame({
    'cat_pred': oof_cat,
    'xgb_pred': oof_xgb,
    'lgbm_pred': oof_lgbm
})
stacked_test = pd.DataFrame({
    'cat_pred': pred_cat,
    'xgb_pred': pred_xgb,
    'lgbm_pred': pred_lgbm
})

meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(stacked_train, train_df[TARGET])
stacked_pred = meta_model.predict(stacked_test)


# 7 | Final Prediction and Submission
sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub.Listening_Time_minutes = stacked_pred
sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
sub.head()

