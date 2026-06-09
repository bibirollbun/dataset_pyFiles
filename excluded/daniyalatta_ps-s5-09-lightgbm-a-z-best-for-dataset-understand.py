# Dependencies
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import gc
import warnings
warnings.filterwarnings("ignore")


# Free Memory
def free():
    gc.collect()


# Calculate Mode
def calc_mode(x):
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return pd.Series(x).mode()[0]


# RMSE Metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Data Loading
PATH = "/kaggle/input/playground-series-s5e9/"
dt = pd.read_csv(f"{PATH}train.csv")
dtest = pd.read_csv(f"{PATH}test.csv")
sub = pd.read_csv(f"{PATH}sample_submission.csv")
sub_temp = sub.iloc[:, 1].copy()
# Debug: Inspect raw data
print("Raw train data shape:", dt.shape)
print("Raw train data non-NaN counts:", dt.notna().sum())
print("Raw train data types:", dt.dtypes)


# Identify Target
target = [col for col in dt.columns if col not in dtest.columns]
Y = dt[target].max(axis=1)
dt = dt.drop(columns=target)


# Merge Train and Test
dt['fuente'] = 'train'
dtest['fuente'] = 'test'
dt_total = pd.concat([dt, dtest], ignore_index=True)


# Remove Duplicated Columns
cols_duplicated = dt_total.columns[dt_total.T.duplicated()].tolist()
if cols_duplicated:
    dt_total = dt_total.drop(columns=cols_duplicated)
    print(f"Removed duplicated columns: {cols_duplicated}")


# Remove High-Null Columns
null_cols = dt_total.columns[dt_total.isnull().mean() >= 0.95].tolist()
if null_cols:
    dt_total = dt_total.drop(columns=null_cols)
    print(f"Removed columns with >= 95% nulls: {null_cols}")


# Remove Single-Value Columns
n_unique = dt_total.nunique()
one_value_cols = n_unique[n_unique == 1].index.tolist()
if one_value_cols:
    dt_total = dt_total.drop(columns=one_value_cols)
    print(f"Removed columns with one unique value: {one_value_cols}")


# Add Null Count Feature
if dt_total.isnull().sum().sum() > 0:
    dt_total['FilasNulas'] = dt_total.isnull().sum(axis=1)


# Convert potential numeric columns to numeric type
for col in dt_total.columns:
    if col not in ['id', 'fuente']:
        try:
            dt_total[col] = pd.to_numeric(dt_total[col], errors='coerce')
        except:
            pass


# Encode Categorical Columns
cat_cols = dt_total.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col != 'fuente']  # Exclude 'fuente' from encoding
for col in cat_cols:
    dt_total[col] = LabelEncoder().fit_transform(dt_total[col].astype(str))


# Remove Low-Variance Features
cols_total = dt_total.columns[3:]
i_cols_borradas = 0
for c in cols_total:
    mode_val = calc_mode(dt_total[c])
    prc_repetido = (dt_total[c] == mode_val).mean()
    if prc_repetido >= 0.95:
        print(f"Removing {c} with repeated value proportion {prc_repetido:.3f}")
        dt_total = dt_total.drop(columns=c)
        i_cols_borradas += 1
print(f"Total columns removed: {i_cols_borradas}")


# Frequency Encoding
n_categories = dt_total[cols_total].nunique()
cat_features = n_categories[n_categories <= 15].index.tolist()
for c in cat_features:
    freq = dt_total[c].value_counts().to_dict()
    dt_total[f"{c}_FreqEnc"] = dt_total[c].map(freq)


# Handle Missing Values
null_cols = dt_total.columns[dt_total.isnull().sum() > 0]
for c in null_cols:
    if dt_total[c].notna().sum() > 0:  # If column has at least one non-NaN value
        median_val = dt_total[c].median()
        dt_total[c].fillna(median_val, inplace=True)
    else:  # If column is entirely NaN
        print(f"Column {c} is entirely NaN, imputing with 0")
        dt_total[c].fillna(0, inplace=True)


# Split Train and Test
dt = dt_total[dt_total['fuente'] == 'train'].drop(columns='fuente')
dtest = dt_total[dt_total['fuente'] == 'test'].drop(columns='fuente')


# Preprocessing Pipeline
numeric_cols = dt.select_dtypes(include=[np.number]).columns.drop(['id'], errors='ignore')
numeric_cols = [col for col in numeric_cols if dt[col].notna().sum() > 0]
if len(numeric_cols) > 0:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )
    dt_processed = preprocessor.fit_transform(dt)
    dtest_processed = preprocessor.transform(dtest)
    all_cols = numeric_cols + [col for col in dt.columns if col not in numeric_cols]
    dt = pd.DataFrame(dt_processed, columns=all_cols)
    dtest = pd.DataFrame(dtest_processed, columns=all_cols)
else:
    print("No valid numeric columns to scale. Using all columns as-is.")
    dt = dt.copy()
    dtest = dtest.copy()
    if dt.drop(columns=['id'], errors='ignore').shape[1] == 0:
        raise ValueError("No features available for modeling after preprocessing.")
dt['id'] = dt['id'].astype(int)
dtest['id'] = dtest['id'].astype(int)


# Setup
oof = pd.DataFrame({'target': Y})
sub_seeds = sub.copy()
sub_seeds.iloc[:, 1:] = 0
scores = []
SEEDS = [1975, 2000, 2503, 1511, 2604]


for s in SEEDS:
    print(f"\nSeed: {s}")
    np.random.seed(s)
    name_seed = f"Seed_{s}"
    sub_seeds[name_seed] = 0
    if dt.shape[0] == 0:
        raise ValueError("Training dataset (dt) is empty. Check data loading or preprocessing steps.")
    kf = KFold(n_splits=10, shuffle=True, random_state=s)
    pred_lgbm_total = np.zeros(len(dtest))
    sub_temp = pd.DataFrame({'id': sub['id']})
    auc_lgbm_total = 0

    for i_fold, (train_idx, val_idx) in enumerate(kf.split(dt), 1):
        print(f"\nLGBM Fold {i_fold} - 10")
        X_train, y_train = dt.iloc[train_idx].drop(columns='id'), Y[train_idx]
        X_val, y_val = dt.iloc[val_idx].drop(columns='id'), Y[val_idx]

        # LightGBM model
        lgb_params = {
            'random_state': 0,
            'n_estimators': 4500,
            'learning_rate': 0.005,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse'
        }

        model = LGBMRegressor(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='rmse',
            callbacks=[early_stopping(stopping_rounds=500), log_evaluation(period=250)]
        )

        # Predictions
        test_lgbm = model.predict(X_val)
        oof.loc[val_idx, f"LGBM_Seed_{s}"] = test_lgbm
        auc_lgbm = rmse(y_val, test_lgbm)
        auc_lgbm_total += auc_lgbm
        scores.append(auc_lgbm)
        print(f"LGBM Fold {i_fold} RMSE: {auc_lgbm:.4f}, Avg RMSE: {np.mean(scores):.4f}")

        # Test predictions
        pred_lgbm = model.predict(dtest.drop(columns='id'))
        pred_lgbm_total += pred_lgbm
        sub_temp[f"Seed_s{s}_Fold_{i_fold}"] = pred_lgbm
        sub_seeds[name_seed] += pred_lgbm / 10

        # Feature importance
        importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        })
        importance.to_csv("ImportanciaLGBM_v1.csv", index=False)
        free()

    pred_lgbm_total /= 10
    print(f"AUC LGBM Seed {s}: {np.mean(scores):.4f}")


# Save Outputs
sub.iloc[:, 1] = sub_temp.iloc[:, 1:].mean(axis=1)
sub.to_csv("subLGBMFit_v1.csv", index=False)

sub.iloc[:, 1] = sub_temp.iloc[:, 1:].median(axis=1)
sub.to_csv("subLGBMFitMedian_v1.csv", index=False)

sub_temp.to_csv("subFoldLGBMFit_v1.csv", index=False)
oof.to_csv("OOF_LGBMFit_v1.csv", index=False)
sub_seeds.to_csv("subSeedsLGBM_v1.csv", index=False)


# Evaluate Performance
for s in SEEDS:
    print(f"RMSE Seed {s}: {rmse(oof['target'], oof[f'LGBM_Seed_{s}']):.4f}")

auc_mean = np.mean([rmse(oof['target'], oof[f'LGBM_Seed_{s}']) for s in SEEDS])
print(f"Mean RMSE: {auc_mean:.4f}")




