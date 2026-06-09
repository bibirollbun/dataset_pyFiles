import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import shap
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
train.sample(5)


shap_columns = pd.read_csv('/kaggle/input/predicting-fertilizers/shap_cv_importance.csv')


plt.figure(figsize=(10, 6))
sns.barplot(x='mean_abs_shap', y='feature', data=shap_columns, palette='viridis')
plt.xlabel('Mean |SHAP value|')
plt.ylabel('Feature')
plt.title('Feature Importance by Mean Absolute SHAP Value')
plt.tight_layout()
plt.show()


shap_columns = shap_columns.loc[:shap_columns[shap_columns['feature'] == 'random'].index[0] - 1, :]


features = shap_columns['feature']


x, y = train.drop(columns = ['Fertilizer Name']), train.loc[:, 'Fertilizer Name']

x['soil_crop_type'] = x['Soil Type'] + '_' + x['Crop Type']

x['Nitrogen_Potassium'] = x['Nitrogen'] * x['Potassium']
x['Phosphorous_Potassium'] = x['Phosphorous'] * x['Potassium']
x['Nitrogen_Phosphorous'] = x['Nitrogen'] * x['Phosphorous']
x['N_to_K'] = x['Nitrogen'] / (x['Potassium'] + 1e-5)
x['N_to_P'] = x['Nitrogen'] / (x['Phosphorous'] + 1e-5)
x['Moisture_Temp_Interaction'] = x['Moisture'] * x['Temparature']
x['K_to_P'] = x['Potassium'] / (x['Phosphorous'] + 1e-5)
x['Humidity_Temp_Interaction'] = x['Humidity'] * x['Temparature']
x['P_minus_K'] = x['Phosphorous'] - x['Potassium']
x['NPK_sum'] = x['Nitrogen'] + x['Phosphorous'] + x['Potassium']
x['Humidity_to_Temp'] = x['Humidity'] / (x['Temparature'] + 1e-5)
x['N_minus_P'] = x['Nitrogen'] - x['Phosphorous']
x['Moisture_to_Temp'] = x['Moisture'] / (x['Temparature'] + 1e-5)
x['N_minus_K'] = x['Nitrogen'] - x['Potassium']



x = x[features]


objects = [col for col in x.columns if x[col].dtype == 'O']


x[objects] = x[objects].astype('category')


cats = [col for col in x.columns if x[col].dtype == 'category']
nums = [col for col in x.columns if col not in cats]
del objects


def reduce_numeric_memory(df, nums, verbose=True):
    df = df[nums].copy()
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            elif pd.api.types.is_float_dtype(col_type):
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df


x[nums] = reduce_numeric_memory(x, nums)


correlation_matrix = x[nums].corr()
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


correlation_matrix = correlation_matrix.abs()
high_corr_pairs = (
    correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
high_corr_pairs.columns = ['feature_1', 'feature_2', 'correlation']
high_corr_pairs = high_corr_pairs[high_corr_pairs['correlation'] > 0.7]


high_corr_pairs


shap_map = shap_columns.set_index('feature')['mean_abs_shap']
high_corr_pairs['shap_1'] = high_corr_pairs['feature_1'].map(shap_map)
high_corr_pairs['shap_2'] = high_corr_pairs['feature_2'].map(shap_map)

high_corr_pairs['drop'] = np.where(high_corr_pairs['shap_1'] > high_corr_pairs['shap_2'],
                                   high_corr_pairs['feature_2'],
                                   high_corr_pairs['feature_1'])

dropping_features = high_corr_pairs['drop'].unique().tolist()


x = x.drop(columns = dropping_features)


cats = [col for col in x.columns if x[col].dtype == 'category']
nums = [col for col in x.columns if col not in cats]

correlation_matrix = x[nums].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Define MAP@K

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Encode target
y_encoded = LabelEncoder().fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Optuna objective

def objective(trial):
    FOLDS = 5
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 6, 13),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
        "reg_lambda": trial.suggest_float("lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("alpha", 0.0, 5.0),
    }

    oof_preds = np.zeros((len(x), num_classes))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y_encoded)):
        X_train, X_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y_encoded[train_idx], y_encoded[valid_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=4000,
            evals=[(dval, "validation")],
            early_stopping_rounds=250,
            verbose_eval=False
        )

        proba = model.predict(dval)
        oof_preds[valid_idx] = proba

    top3_preds = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
    map3_final = mapk(y_encoded, top3_preds, k=3)
    print("ðŸ“Š Final OOF MAP@3:", map3_final)

    return map3_final


# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best trial:")
print(f"  MAP@3 Score: {study.best_value:.5f}")
print("  Params:", study.best_params)



import json

with open("best_trial.json", "w") as f:
    json.dump({
        "MAP@3 Score": round(study.best_value, 5),
        "Best Params": study.best_params
    }, f, indent=4)


