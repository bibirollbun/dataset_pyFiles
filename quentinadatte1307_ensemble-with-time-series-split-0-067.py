!pip install scikit-learn==1.5.2


from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from tqdm import tqdm
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import optuna
import gc

warnings.filterwarnings("ignore")


class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    target = "label"
    n_folds = 5
    seed = 42

    run_optuna = True
    n_optuna_trials = 250


def reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe


train = pd.read_parquet(CFG.train_path).reset_index(drop=True)
test = pd.read_parquet(CFG.test_path).reset_index(drop=True)


cols_to_drop = [
    'X697', 'X698', 'X699', 'X700', 'X701', 'X702', 'X703', 'X704', 'X705', 'X706', 
    'X707', 'X708', 'X709', 'X710', 'X711', 'X712', 'X713', 'X714', 'X715', 'X716',
    'X717', 'X864', 'X867', 'X869', 'X870', 'X871', 'X872', 'X104', 'X110', 'X116',
    'X122', 'X128', 'X134', 'X140', 'X146', 'X152', 'X158', 'X164', 'X170', 'X176',
    'X182', 'X351', 'X357', 'X363', 'X369', 'X375', 'X381', 'X387', 'X393', 'X399',
    'X405', 'X411', 'X417', 'X423', 'X429'
]


train = train.drop(columns=cols_to_drop)
test = test.drop(columns=["label"] + cols_to_drop)

def clip_outliers_to_bounds(df, lower_q=0.05, upper_q=0.95):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Compute lower and upper quantiles
    lower_bounds = df[numeric_cols].quantile(lower_q)
    upper_bounds = df[numeric_cols].quantile(upper_q)

    # Clip values to within bounds
    for col in numeric_cols:
        df[col] = df[col].clip(lower=lower_bounds[col], upper=upper_bounds[col])

    print(f"Outliers beyond {lower_q:.3f}–{upper_q:.3f} quantiles clipped to boundary values.")
    return df

# train = clip_outliers_to_bounds(train)
# test = clip_outliers_to_bounds(test)

train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test


class Trainer:
    def __init__(self, model):
        self.model = model

    def fit_predict(self, X, y, X_test, split_mode="kfold"):
        print(f"Training {self.model.__class__.__name__}\n")

        fold_scores = []
        oof_preds   = np.zeros(X.shape[0])
        valid_mask  = np.zeros((CFG.n_folds, X.shape[0]), dtype=bool)
        test_preds = np.zeros(X_test.shape[0])

        fold_oof_preds = []
        fold_valid_mask = []
        fold_test_preds = []

        if split_mode == "kfold":
            split = KFold(n_splits=CFG.n_folds, shuffle=False).split(X, y)
        elif split_mode == "timefold":
            split = TimeSeriesSplit(n_splits=CFG.n_folds).split(X, y)
            
        for fold_idx, (train_idx, val_idx) in enumerate(split):

            # if val_idx[-1] < X.shape[0]-1:
            #    val_idx = np.concatenate((val_idx, np.array(range(val_idx[-1]+1, X.shape[0]))))

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            print(f"Training with train idx: {train_idx[0]}-{train_idx[-1]}")
            print(f"Training size: {X_train.shape[0]}")
            print(f"Validating with val idx: {val_idx[0]}-{val_idx[-1]}")
            print(f"Validation size: {X_val.shape[0]}")

            model = clone(self.model)

            model.fit(
                X_train, 
                y_train, 
                # Prevent overfitting
                eval_set=[(X_val, y_val)]
            )

            y_preds = model.predict(X_val)
            fold_score = pearsonr(y_val, y_preds)[0]
            fold_scores.append(fold_score)

            oof_preds[val_idx] = y_preds
            valid_mask[fold_idx, val_idx] = True

            # Additional validation slicing analysis:
            print("Validation time-slice PearsonR scores:")
            num_slices = 5
            val_size = len(val_idx)
            step = val_size // num_slices
            for i in range(num_slices):
                start = i * step
                end = val_size if i == num_slices - 1 else (i + 1) * step
                y_slice_true = y_val[start:end]
                y_slice_pred = y_preds[start:end]
                slice_score = pearsonr(y_slice_true, y_slice_pred)[0] if len(y_slice_true) > 1 else float('nan')
                print(f"  Slice {i+1}: {slice_score:.6f} (idx {val_idx[start]}–{val_idx[end-1]})")

            temp_test_preds = model.predict(X_test)
            test_preds += temp_test_preds / CFG.n_folds
            
            print(f"--- Fold {fold_idx} - Score: {fold_score:.6f}")

            del X_train, y_train, X_val, y_val, y_preds, model, temp_test_preds
            gc.collect()

        # Filter to rows where all models gave OOF prediction
        
        overall_score = pearsonr(y, oof_preds)[0]
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"\n------ Overall Score: {overall_score:.6f} - Mean Score: {mean_score:.6f} ± {std_score:.6f}")
        return oof_preds, test_preds, fold_scores, valid_mask 
        
    def tune(self, X, y):
        fold_scores = []
        
        if split_mode == "kfold":
            split = KFold(n_splits=CFG.n_folds, shuffle=False).split(X, y)
        elif split_mode == "timefold":
            split = TimeSeriesSplit(n_splits=CFG.n_folds).split(X, y)
            
        for train_idx, val_idx in split:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = clone(self.model)
            
            model.fit(
                X_train, y_train
            )

            y_preds = model.predict(X_val)
            fold_score = pearsonr(y_val, y_preds)[0]
            fold_scores.append(fold_score)

            del X_train, y_train, X_val, y_val, y_preds, model
            gc.collect()

        return np.mean(fold_scores)


lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.5242042724303907,
    "learning_rate": 0.014470794293130388,
    "min_child_samples": 47,
    "min_child_weight": 0.1936457311991661,
    "n_estimators": 441,
    "n_jobs": -1,
    "num_leaves": 65,
    "random_state": 42,
    "reg_alpha": 76.69015407123774,
    "reg_lambda": 78.57981723239948,
    "subsample": 0.35497610282716086,
    "verbose": -1,
    "early_stopping_rounds": 100
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.32266516869045214,
    "learning_rate": 0.013684657681610528,
    "min_child_samples": 47,
    "min_child_weight": 0.652800548618323,
    "n_estimators": 268,
    "n_jobs": -1,
    "num_leaves": 25,
    "random_state": 42,
    "reg_alpha": 24.43093150663448,
    "reg_lambda": 39.81794248056326,
    "subsample": 0.21026644887863555,
    "verbose": -1,
    "early_stopping_rounds": 100
}

xgb_params = {
    "colsample_bylevel": 0.4634967322919854,
    "colsample_bynode": 0.6046331585629835,
    "colsample_bytree": 0.11495541333509408,
    "gamma": 1.0397769239502863,
    "learning_rate": 0.09622196913585954,
    "max_depth": 40,
    "max_leaves": 19,
    "min_child_weight": 76,
    "n_estimators": 679,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 65.41659225037377,
    "reg_lambda": 19.907991015311545,
    "subsample": 0.014465324175810368,
    "verbosity": 0,
    "early_stopping_rounds": 100
}


scores = {}
oof_preds = {}
test_preds = {}
valid_mask = {}

split_mode = "timefold"


lgbm_trainer = Trainer(
    LGBMRegressor(**lgbm_params)
)

oof_preds["LightGBM (gbdt)"], test_preds["LightGBM (gbdt)"], scores["LightGBM (gbdt)"], valid_mask["LightGBM (gbdt)"] = lgbm_trainer.fit_predict(X, y, X_test, split_mode)


lgbm_goss_trainer = Trainer(LGBMRegressor(**lgbm_goss_params))

oof_preds["LightGBM (goss)"], test_preds["LightGBM (goss)"], scores["LightGBM (goss)"], valid_mask["LightGBM (goss)"] = lgbm_goss_trainer.fit_predict(X, y, X_test, split_mode)


xgb_trainer = Trainer(XGBRegressor(**xgb_params))

oof_preds["XGBoost"], test_preds["XGBoost"], scores["XGBoost"], valid_mask["XGBoost"] = xgb_trainer.fit_predict(X, y, X_test, split_mode)


# After training the base models
X = pd.DataFrame(oof_preds)
X_test = pd.DataFrame(test_preds)

# Filter to rows where all models gave OOF predictions
meta_mask = (
    valid_mask["LightGBM (gbdt)"].any(axis=0) &
    valid_mask["LightGBM (goss)"].any(axis=0) &
    valid_mask["XGBoost"].any(axis=0) 
)

# These are your clean stacking datasets
X_meta = X.loc[meta_mask].reset_index(drop=True)
y_meta = y.loc[meta_mask].reset_index(drop=True)


# Optuna objective for tuning Ridge
def objective(trial):    
    params = {
        "random_state": CFG.seed,
        "alpha": trial.suggest_float("alpha", 0, 1000),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "positive": trial.suggest_categorical("positive", [True, False])
    }
    
    trainer = Trainer(Ridge(**params))
    # Use the filtered meta data
    return trainer.tune(X_meta, y_meta)

# Run Optuna tuning
if CFG.run_optuna:
    sampler = optuna.samplers.TPESampler(seed=CFG.seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=CFG.n_optuna_trials, n_jobs=-1, catch=(ValueError,))
    best_params = study.best_params

    ridge_params = {
        "random_state": CFG.seed,
        "alpha": best_params["alpha"],
        "tol": best_params["tol"],
        "fit_intercept": best_params["fit_intercept"],
        "positive": best_params["positive"]
    }
else:
    ridge_params = {
        "random_state": CFG.seed
    }



# Final training and test prediction
ridge_trainer = Trainer(Ridge(**ridge_params))
oof_ridge_preds, ridge_test_preds, ridge_scores, _ = ridge_trainer.fit_predict(X_meta, y_meta, X_test)

scores["Ridge (ensemble)"] = ridge_scores


sub = pd.read_csv(CFG.sample_sub_path)
sub["prediction"] = ridge_test_preds
sub.to_csv(f"sub_ridge_{np.mean(scores['Ridge (ensemble)']):.6f}.csv", index=False)
sub.head()


scores = pd.DataFrame(scores)
mean_scores = scores.mean().sort_values(ascending=False)
order = scores.mean().sort_values(ascending=False).index.tolist()

min_score = mean_scores.min()
max_score = mean_scores.max()
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.5))

boxplot = sns.boxplot(data=scores, order=order, ax=axs[0], orient="h", color="grey")
axs[0].set_title(f"Fold Score")
axs[0].set_xlabel("")
axs[0].set_ylabel("")

barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], color="grey")
axs[1].set_title(f"Average Score")
axs[1].set_xlabel("")
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel("")

for i, (score, model) in enumerate(zip(mean_scores.values, mean_scores.index)):
    color = "cyan" if "ensemble" in model.lower() else "grey"
    barplot.patches[i].set_facecolor(color)
    boxplot.patches[i].set_facecolor(color)
    barplot.text(score, i, round(score, 6), va="center")

plt.tight_layout()
plt.show()

