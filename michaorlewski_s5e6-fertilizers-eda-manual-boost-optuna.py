import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import optuna

import itertools
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', index_col='id')

train_df.shape, test_df.shape


train_df.info()


train_df.describe()


print(f'Missing values: {train_df.isna().sum().sum() + test_df.isna().sum().sum()}')


X_train = train_df.drop('Fertilizer Name', axis=1)
y_train = train_df['Fertilizer Name']
X_test = test_df

num_columns = X_train.select_dtypes(exclude=['object']).columns
cat_columns = X_train.select_dtypes(include=['object']).columns

print(f'Available fertilizers = {y_train.unique()}')
y_train.value_counts()


plt.figure(figsize=(12, 8))
for i, col in enumerate(num_columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(X_train[col], bins=12)
    sns.histplot(X_test[col], bins=12)

plt.tight_layout()
plt.show()


sns.heatmap(train_df.corr(numeric_only=True), annot=True)


plt.figure(figsize=(12, 8))
for i, col in enumerate(num_columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=X_train[col], y=y_train)

plt.tight_layout()
plt.show()


for i, col in enumerate(num_columns):
    groups = [group[col].values for name, group in train_df.groupby('Fertilizer Name')]
    anova, pvalue = stats.f_oneway(*groups)
    print(f'{col}: Anova = {anova}, p_value = {pvalue}')


plt.figure(figsize=(12, 12))
for i, col in enumerate(cat_columns):
    plt.subplot(2, 1, i+1)
    sns.countplot(x=X_train[col], hue=y_train)

plt.tight_layout()
plt.show()


def calc_chi2_pvalue(df, feature, target):
    counts = df[[feature, target]].groupby([feature, target]).size().reset_index(name='Count')
    counts_pivoted = counts.pivot(index=target, values='Count', columns=feature)
    chi2, pvalue, _, _ = stats.chi2_contingency(counts_pivoted)
    return chi2, pvalue

for col in cat_columns:
    chi2, pvalue = calc_chi2_pvalue(train_df, col, 'Fertilizer Name')
    print(f'{col}: chi2 = {chi2:.4f}, p-value = {pvalue}')


cat_only = train_df[['Soil Type', 'Crop Type', 'Fertilizer Name']]
fertilizer_count = cat_only.groupby(['Soil Type', 'Crop Type', 'Fertilizer Name']).size().reset_index(name='count')
fertilizer_count = fertilizer_count.sort_values(['Soil Type', 'Crop Type', 'count'], ascending=[True, True, False])
top3_fertilizers = fertilizer_count.groupby(['Soil Type', 'Crop Type']).head(3)
top3_fertilizers


soil_crop_to_fert = {}

for p in itertools.product(train_df['Soil Type'].unique(), train_df['Crop Type'].unique()):
    soil_cond = top3_fertilizers['Soil Type'] == p[0]
    crop_cond = top3_fertilizers['Crop Type'] == p[1]
    fert_list = top3_fertilizers[soil_cond & crop_cond]['Fertilizer Name'].values.tolist()
    soil_crop_to_fert[p] = fert_list


train_predictions_manual = train_df.apply(
    lambda r: soil_crop_to_fert.get((r['Soil Type'], r['Crop Type']), 'Unknown'),
    axis=1
)

test_predictions_manual = test_df.apply(
    lambda r: soil_crop_to_fert.get((r['Soil Type'], r['Crop Type']), 'Unknown'),
    axis=1
)


def mapk(actual, predicted, k=3):
    s = 0.0
    for a, p in zip(actual, predicted):
        if len(p) > k:
            p = p[:k]
        s += 1/(p.index(a)+1) if a in p else 0
    return s / len(actual)

mapk(y_train, train_predictions_manual)


test_predictions = test_predictions_manual.apply(lambda x : ' '.join(x))
test_predictions

submission_manual = pd.DataFrame({'id': X_test.index, 'Fertilizer Name': test_predictions})
submission_manual.to_csv('/kaggle/working/manual.csv', index=False)


for i in range(3):
    X_train[f'SoilCropToFert_{i+1}'] = X_train.apply(lambda r: soil_crop_to_fert.get((r['Soil Type'], r['Crop Type']))[i], axis=1)
    X_test[f'SoilCropToFert_{i+1}'] = X_test.apply(lambda r: soil_crop_to_fert.get((r['Soil Type'], r['Crop Type']))[i], axis=1)


num_columns = X_train.select_dtypes(exclude=['object']).columns
cat_columns = X_train.select_dtypes(include=['object']).columns

cat_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(sparse_output=False, drop='first'))
])

num_pipeline = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_columns),
    # ('num', num_pipeline, num_columns)
], remainder='passthrough')


preprocessor.fit(X_train)
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

le = LabelEncoder()
y_train_preprocessed = le.fit_transform(y_train)


def get_predictions(model, X):
    probs = model.predict_proba(X)
    probs_sorted = np.argsort(probs, axis=1)[:, ::-1][:,:3]                 
    predictions = [[le.classes_[i] for i in row] for row in probs_sorted]
    return predictions

def cross_validate(model, X, y, y_raw, verbose=True):
    train_scores = []
    valid_scores = []
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    for i, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, y_train, y_train_raw = X[train_idx, :], y[train_idx], y_raw[train_idx]
        X_valid, y_valid, y_valid_raw = X[valid_idx, :], y[valid_idx], y_raw[valid_idx]

        model.fit(X_train, y_train)
        train_preds = get_predictions(model, X_train)
        valid_preds = get_predictions(model, X_valid)

        train_scores.append(mapk(y_train_raw, train_preds))
        valid_scores.append(mapk(y_valid_raw, valid_preds))

        if verbose:
            print(f'Split {i+1} scores: train = {train_scores[-1]:.4f}, valid = {valid_scores[-1]:.4f}')
    if verbose:
        print(f'Mean scores: train = {np.mean(train_scores):.4f}, valid = {np.mean(valid_scores)}')
    return np.mean(valid_scores)


def objective(trial):
    xgb = XGBClassifier(
        n_estimators=trial.suggest_int('n_estimators', 200, 1000),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3),
        max_depth=trial.suggest_int('max_depth', 3, 12),
        gamma=trial.suggest_float('gamma', 0, 5),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        n_jobs=-1,
    )

    score = cross_validate(xgb, X_train_preprocessed[:250_000], y_train_preprocessed[:250_000], y_train[:250_000], verbose=False)
    return score

study_xgb = optuna.create_study(direction='maximize', study_name='XGBClassifier')
# study_xgb.optimize(objective, n_trials=20) # Commented out to reduce run time


# cross validate
best_params = {'n_estimators': 940,
               'learning_rate': 0.18142786897335514,
               'max_depth': 3,
               'gamma': 0.02112342723616445,
               'min_child_weight': 2,
               'subsample': 0.8526829794242452,
               'colsample_bytree': 0.7574380767842364,
               'reg_alpha': 3.9806598604001158,
               'reg_lambda': 1.032473452258038}
xgb = XGBClassifier(**best_params, n_jobs=-1)
cross_validate(xgb, X_train_preprocessed, y_train_preprocessed, y_train)

# predict
xgb.fit(X_train_preprocessed, y_train_preprocessed)
test_predictions_xgb = get_predictions(xgb, X_test_preprocessed)
test_predictions = pd.Series(test_predictions_xgb).apply(lambda x : ' '.join(x))

# save predictions
submission_xgb = pd.DataFrame({'id': X_test.index, 'Fertilizer Name': test_predictions})
submission_xgb.to_csv('/kaggle/working/xgb.csv', index=False)


def objective(trial):
    lgbm = LGBMClassifier(
        n_estimators=trial.suggest_int('n_estimators', 200, 1000),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3),
        num_leaves=trial.suggest_int('num_leaves', 31, 4096),
        max_depth=trial.suggest_int('max_depth', 3, 12),
        min_split_gain=trial.suggest_float('min_split_gain', 0, 5),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        n_jobs=-1,
        verbose=-1,
    )

    score = cross_validate(lgbm, X_train_preprocessed[:250_000], y_train_preprocessed[:250_000], y_train[:250_000], verbose=False)
    return score

study_lgbm = optuna.create_study(direction='maximize', study_name='LGBMClassifier')
# study_lgbm.optimize(objective, n_trials=20) # Commented out to reduce run time


# cross validate
best_params = {'n_estimators': 204,
              'learning_rate': 0.28205510888149266,
              'num_leaves': 142,
              'max_depth': 8,
              'min_split_gain': 0.09342509846548624,
              'min_child_weight': 4,
              'subsample': 0.912533394254343,
              'colsample_bytree': 0.9801067074046347,
              'reg_alpha': 4.746196445006296,
              'reg_lambda': 4.798381661293112}
lgbm = LGBMClassifier(**best_params, n_jobs=-1, verbose=-1)
cross_validate(lgbm, X_train_preprocessed, y_train_preprocessed, y_train)

# predict
lgbm.fit(X_train_preprocessed, y_train_preprocessed)
test_predictions_lgbm = get_predictions(lgbm, X_test_preprocessed)
test_predictions = pd.Series(test_predictions_lgbm).apply(lambda x : ' '.join(x))

# save predictions
submission_lgbm = pd.DataFrame({'id': X_test.index, 'Fertilizer Name': test_predictions})
submission_lgbm.to_csv('/kaggle/working/lgbm.csv', index=False)


def objective(trial):
    hist = HistGradientBoostingClassifier(
        max_iter=trial.suggest_int('max_iter', 50, 500),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.3),
        max_depth=trial.suggest_int('max_depth', 3, 12),
        l2_regularization=trial.suggest_float('l2_regularization', 0, 10),
        max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 31, 4096),
    )

    score = cross_validate(hist, X_train_preprocessed[:250_000], y_train_preprocessed[:250_000], y_train[:250_000], verbose=False)
    return score

study_hist = optuna.create_study(direction='maximize', study_name='HistGradientBoostingClassifier')
# study_hist.optimize(objective, n_trials=20) # Commented out to reduce run time


# cross validate
best_params = {'max_iter': 180,
               'learning_rate': 0.14097560479193186,
               'max_depth': 4,
               'l2_regularization': 5.771982192954164,
               'max_leaf_nodes': 835}
hist = HistGradientBoostingClassifier(**best_params)
cross_validate(hist, X_train_preprocessed, y_train_preprocessed, y_train)

# predict
hist.fit(X_train_preprocessed, y_train_preprocessed)
test_predictions_hist = get_predictions(hist, X_test_preprocessed)
test_predictions = pd.Series(test_predictions_hist).apply(lambda x : ' '.join(x))

# save predictions
submission_hist = pd.DataFrame({'id': X_test.index, 'Fertilizer Name': test_predictions})
submission_hist.to_csv('/kaggle/working/hist.csv', index=False)


from collections import defaultdict

test_predictions_combined = []
for test_predictions in zip(test_predictions_xgb, test_predictions_lgbm, test_predictions_hist):
    scores = defaultdict(int)
    for preds in test_predictions:
        scores[preds[0]] += 3
        scores[preds[1]] += 2
        scores[preds[2]] += 1

    preds_combined = ' '.join(np.array(sorted(scores.items(), key=lambda x: x[1], reverse=True))[:, 0])
    test_predictions_combined.append(preds_combined)


submission_combined = pd.DataFrame({'id': X_test.index, 'Fertilizer Name': test_predictions_combined})
submission_combined.to_csv('/kaggle/working/combined.csv', index=False)

