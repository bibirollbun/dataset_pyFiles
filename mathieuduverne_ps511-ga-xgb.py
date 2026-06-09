
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from deap import base, creator, tools, algorithms
from tqdm import tqdm


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
target = df.columns.tolist()[-1]
print(df.shape)
df.head()


def create_frequency_features(df, df_test):
    """
    Add frequency and binning features efficiently.

    - For each categorical column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5, 10, 15 quantile bins.
    """
    # Pre-allocate DataFrames for new features to avoid fragmentation
    freq_features_train = pd.DataFrame(index=df.index)
    freq_features_test = pd.DataFrame(index=df_test.index)
    bin_features_train = pd.DataFrame(index=df.index)
    bin_features_test = pd.DataFrame(index=df_test.index)

    for col in cols:
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # --- Quantile binning for numeric columns ---
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0

    # Concatenate all new features at once
    df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
    df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

    return df, df_test


def target_encoding(train, predict, n_splits=5,g=True):
    """
    Add K-Fold target mean encoded features to train and predict datasets.
    
    Parameters:
    - train: training DataFrame
    - predict: prediction/test DataFrame
    - target: name of the target column
    - n_splits: number of folds for K-Fold encoding
    
    Returns:
    - train and predict DataFrames with new mean encoded features
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features_train = pd.DataFrame(index=train.index)
    mean_features_test = pd.DataFrame(index=predict.index)

    for col in cols:
        # --- K-Fold Target Mean Encoding ---
        if (col == "annual_income") and (g==True) :
            # print("here it is")
            mean_encoded = np.zeros(len(train))
            # print(f"mean_encoded {mean_encoded}")
            for tr_idx, val_idx in kf.split(train):
                tr_fold = train.iloc[tr_idx]
                # print(f"tr_fold {tr_fold}")
                val_fold = train.iloc[val_idx]
                mean_map = tr_fold.groupby(col)[target].mean()
                # print(f"mean_map {mean_map}")
                mean_encoded[val_idx] = val_fold[col].map(mean_map)
                # print(f"mean_encoded {mean_encoded}")
    
            mean_features_train[f'mean_{col}'] = mean_encoded
    
            # --- Apply global mean mapping to prediction/test data ---
            global_mean = train.groupby(col)[target].mean()
            mean_features_test[f'mean_{col}'] = predict[col].map(global_mean)
            g= False
        
        else:
            # print("it works")
            mean_encoded = np.zeros(len(train))
            for tr_idx, val_idx in kf.split(train):
                tr_fold = train.iloc[tr_idx]
                val_fold = train.iloc[val_idx]
                mean_map = tr_fold.groupby(col)[target].mean()
                mean_encoded[val_idx] = val_fold[col].map(mean_map)
    
            mean_features_train[f'mean_{col}'] = mean_encoded
    
            # --- Apply global mean mapping to prediction/test data ---
            global_mean = train.groupby(col)[target].mean()
            mean_features_test[f'mean_{col}'] = predict[col].map(global_mean)

    # --- Concatenate new features at once to avoid fragmentation ---
    train = pd.concat([train, mean_features_train], axis=1)
    predict = pd.concat([predict, mean_features_test], axis=1)

    # Defragment
    train = train.copy()
    predict = predict.copy()
    return train, predict


# Rounding the values
for c in ['annual_income', 'loan_amount']:
    for s, l in {'1s': 0, '10s': -1}.items():
        for g in [df, df_test]:
            g[f'{c}_ROUND_{s}'] = g[c].round(l).astype(int)

# Specific feature engineering
for gf in [df, df_test]:
    gf['subgrade'] = gf['grade_subgrade'].str[1:].astype(int)
    gf['grade'] = gf['grade_subgrade'].str[0]
    gf['total_debt_burden'] = (gf['loan_amount'] * gf['interest_rate'] / 100) / (gf['annual_income'] + 1)


cols = df.drop(columns=[target,"id"]).columns.tolist()
cat = [c for c in cols if df[c].dtype in ["object","category"]]
num = [c for c in cols if df[c].dtype not in ["object","category","bool"]]

# Creating new features based on the frequency of numerical features
df, df_test = target_encoding(df, df_test, 10)
df, df_test = create_frequency_features(df, df_test)

# Preparing categorical features
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")


remove = [
    'annual_income_ROUND_10s_bin10','annual_income_ROUND_1s_bin10','annual_income_ROUND_1s_bin15','annual_income_ROUND_1s_bin5',
    'annual_income_bin10','annual_income_bin5','credit_score_bin10','credit_score_bin5','debt_to_income_ratio_bin15','debt_to_income_ratio_bin5',
    'education_level_freq','gender_freq','interest_rate_bin10','interest_rate_bin5','loan_amount_ROUND_10s_bin5','loan_amount_ROUND_1s_bin10',
    'loan_amount_ROUND_1s_bin15','loan_amount_ROUND_1s_bin5','loan_amount_bin10','loan_amount_bin15','loan_amount_bin5','marital_status_freq',
    'subgrade','subgrade_bin10','subgrade_bin15','subgrade_bin5','subgrade_freq'
]

df, df_test = df.drop(columns = remove+["id"]), df_test.drop(columns = remove)


print(f"Number of columns {len(df.columns.tolist())}")
print(df.columns.tolist())


n_features = df.columns.tolist()
n_features = len(df.drop(columns=[target]).columns)


# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", lambda: np.random.choice([0,1]))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_individual(individual):
    selected = [i for i, bit in enumerate(individual) if bit == 1]

    if len(selected) == 0:
        return 0.,

    X_sel = df.drop(columns=target).iloc[:, selected]
    y_sel = df[target]

    dtrain = xgb.DMatrix(
        X_sel,
        label=y_sel,
        enable_categorical=True
    )

    xgb_params = {
        'tree_method': 'hist', 
        'device': 'cuda',
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'random_state': 42,
        'min_child_weight': 89,
        "max_leaves": 4,
        "reg_alpha": 3.2,
        "reg_lambda": 5,
        "eta": 0.1,
    }

    cv_results = xgb.cv(
        params=xgb_params,
        dtrain=dtrain,
        nfold=7,
        num_boost_round=1500,
        metrics='auc',
        early_stopping_rounds=50,
        verbose_eval=False
    )

    best_round = cv_results['test-auc-mean'].idxmax()
    best_auc = cv_results['test-auc-mean'][best_round]

    return best_auc,

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=15)
NGEN = 12

best_ind = None
print("Starting GA...")

for gen in range(NGEN):
    print(f"Generation {gen+1}/{NGEN}")
    offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)

    fits = []
    for ind in tqdm(offspring, desc="Evaluating individuals"):
        fits.append(toolbox.evaluate(ind))

    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    pop = toolbox.select(offspring, k=len(pop))
    best_ind = tools.selBest(pop, k=1)[0]
    print("Best AUC:", best_ind.fitness.values[0])

selected_features = [X.columns[i] for i, bit in enumerate(best_ind) if bit == 1]
print("Final selected features:", selected_features)

