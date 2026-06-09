%pip install statsmodels -q
%pip install fancyimpute -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import time
import pickle

from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from xgboost import XGBRegressor
from fancyimpute import SoftImpute

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


DATA_PATH = "/kaggle/input/playground-series-s5e4/"
RSEED = 42
RESULTS_PATH = "/kaggle/working/results/"
os.makedirs(RESULTS_PATH, exist_ok=True)

np.random.seed(RSEED)
target = 'listening_time_minutes'
global_results = pd.DataFrame()


def get_current_time():
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d_%H-%M-%S")

def read_data(fileName, dataPath=DATA_PATH):
    path = dataPath + fileName
    return pd.read_csv(path)

def root_mean_squared_error(ytrue, yhat):
    return np.sqrt(mean_squared_error(ytrue, yhat))


def plot_histplot(df, column, bins="auto", title=None, figsize=(10, 6), kde=True, hist=True,
                 color='steelblue', stat='count', log_scale=False, ax=None, ylabel="", label=None):
    """
    Creates a histogram plot for numerical data.
    """
    # Don't create a new figure if ax is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if kde and not hist:
        sns.kdeplot(data=df, x=column, ax=ax, color=color, label=label)
    else:
        sns.histplot(data=df, x=column, bins=bins, kde=kde, 
                    element="step" if not hist else "bars",
                    color=color, stat=stat, ax=ax, label=label)
    
    # Set title and labels on the provided axis
    if title: ax.set_title(title, fontsize=14)
    if ylabel: ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(column, fontsize=12)
    
    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)
    
    # Only do tight_layout if we created the figure
    if ax is None:
        plt.tight_layout()
        
    return ax


def print_metrics(ytrue, yhat):
    rmse = root_mean_squared_error(ytrue, yhat)
    r2 = r2_score(ytrue, yhat)

    print("RMSE:", rmse)
    print("R2:", r2)

    return {"r2": r2, "rmse": rmse}


def save_submission(id, yhat, column=target, filename="", path="results/"):
    if not filename:
        filename = get_current_time() + ".csv"

    res = pd.DataFrame({
        "id": id,
        column: yhat
    })

    path += filename
    res.to_csv(path, index=False)


def handle_prediction(X, y, Xval, yval, model, storing="", print_yhat=False):
    model.fit(X, y)

    yhat = model.predict(X)
    train_res = print_metrics(y, yhat)

    print()

    yhat = model.predict(Xval)
    val_res = print_metrics(yval, yhat)

    # cross_val
    X = pd.concat([X, Xval])
    y = pd.concat([y, yval])
    cross_res = abs(cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring="neg_root_mean_squared_error"))
    print("\nCross_val mean RMSE:", cross_res.mean())
    print("Cross_val RMSE:", cross_res)

    if print_yhat:
        print(yhat)

    if storing:
        cur = []
        for res in [train_res, val_res]:
            for name, metric in res.items():
                cur.append(f"{name}_{metric:.4f}")
        cur.append(f"cross_rmse_{cross_res.mean()}")
        global_results[storing] = cur

    return model


train = read_data("train.csv")
train.columns = train.columns.str.lower()
train = train.drop("id", axis=1)

test = read_data("test.csv")
test.columns = test.columns.str.lower()
test_id = test["id"]
test = test.drop("id", axis=1)


train.shape


train.describe()


print(train.isnull().sum())


test.isnull().sum()


columns_to_encode = ["podcast_name", "genre", "publication_day", "publication_time", "episode_sentiment"]
num_cols = ['episode_title', 'episode_length_minutes', 'host_popularity_percentage', 'guest_popularity_percentage', 'listening_time_minutes']


train["episode_title"] = train["episode_title"].apply(lambda x: int(x.split(" ")[1]))
test["episode_title"] = test["episode_title"].apply(lambda x: int(x.split(" ")[1]))


for col in columns_to_encode:
    train[col] = train[col].astype('category') 
    test[col] = test[col].astype('category')


train.columns


def encode_columns(df, columns, to_drop=True):
    result_df = df.copy()
    
    # Encode publication_time as ordinal (representing time of day)
    if "publication_time" in columns:
        time_mapping = {
            "Morning": 0,
            "Afternoon": 1, 
            "Evening": 2,
            "Night": 3
        }
        result_df["publication_time"] = result_df["publication_time"].map(time_mapping)
    
    # Encode publication_day as ordinal (representing day of week)
    if "publication_day" in columns:
        day_mapping = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }
        result_df["publication_day"] = result_df["publication_day"].map(day_mapping)
    
    # Encode episode_sentiment
    if "episode_sentiment" in columns:
        sentiment_mapping = {
            "Negative": -1,
            "Neutral": 0,
            "Positive": 1
        }
        result_df["episode_sentiment"] = result_df["episode_sentiment"].map(sentiment_mapping)
    
    # One-hot encode nominal categories
    for col in columns:
        if col in ["genre", "podcast_name"] and col in result_df.columns:
            dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=False)
            result_df = pd.concat([result_df, dummies], axis=1)
    
    if to_drop: result_df = result_df.drop(columns, axis=1)
    result_df.columns = result_df.columns.str.replace(" ", "_").str.lower()
    return result_df 


plot_histplot(train, column='episode_length_minutes', bins=100);


print(train[train["episode_length_minutes"] > 120].shape)
print(train[train["episode_length_minutes"] < 5].shape)


plot_histplot(train, column="guest_popularity_percentage", bins=100);


print(train[train["guest_popularity_percentage"] > 100].shape)


def cap_values(df, column, min_val, max_val):
    df[column] = df[column].apply(lambda x: min(max(x, min_val), max_val))
    return df


train = cap_values(train, column="episode_length_minutes", min_val=5, max_val=120)
train = cap_values(train, column="guest_popularity_percentage", min_val=0, max_val=100)


train["number_of_ads"] = train["number_of_ads"].fillna(train["number_of_ads"].mode()[0])


missing_columns_orig = ["episode_length_minutes", "guest_popularity_percentage"]
missing_columns = ["episode_length_minutes_missing", "guest_popularity_percentage_missing"]
train["episode_length_minutes_missing"] = train["episode_length_minutes"].isna().astype(int)
train["guest_popularity_percentage_missing"] = train["guest_popularity_percentage"].isna().astype(int)


print("Missingness ratio in the variables: ")
for col in missing_columns:
    ratio = train[col].sum() / train.shape[0] * 100
    print(f"Ratio of {col}: {ratio:.2f}%")


print("Number of missing samples in episode_length", 
      train[train["episode_length_minutes_missing"] == 1].shape[0]) 

print("Number of missing samples in episode_length and guest_popularity", 
      train[(train["episode_length_minutes_missing"] == 1) & (train["guest_popularity_percentage_missing"] == 1)].shape[0])


# Calculate actual probabilities
p_episode_missing = train["episode_length_minutes_missing"].mean()
p_guest_missing = train["guest_popularity_percentage_missing"].mean()
p_both_missing = (train["episode_length_minutes_missing"] & 
                 train["guest_popularity_percentage_missing"]).mean()

# If MCAR, these should be approximately equal
expected_p_both = p_episode_missing * p_guest_missing
expected_both_missing_samples = train.shape[0] * expected_p_both
actual_p_both = p_both_missing
actual_both_missing_samples = train.shape[0] * actual_p_both

print(f"Expected probability if MCAR: {expected_p_both:.4f}.",
      f"Expected number of samples: {round(expected_both_missing_samples)}. ")

print(f"Actual probability: {actual_p_both:.4f}.",
      f"Actual number of samples: {round(actual_both_missing_samples)}. ")

print(f"Ratio (actual/expected): {actual_p_both/expected_p_both:.4f}")


encoded_train = encode_columns(train, columns=columns_to_encode)
encoded_test = encode_columns(test, columns=columns_to_encode)


missing_corr = encoded_train.corr()[missing_columns]
for row, values in missing_corr.iterrows():
    for col, corr in values.items():
        if corr >= 0.05 and col != row:
            print(row, col, corr)


%%time

log_reg = LogisticRegression(max_iter=10000, random_state=RSEED, verbose=0)
temp = encoded_train.drop(missing_columns_orig, axis=1)

for col in missing_columns:
    X = temp.drop(col, axis=1)
    y = temp[col]
    score = cross_val_score(log_reg, X, y, scoring="roc_auc", verbose=0, n_jobs=-1)
    print(f"The AUC score for {col}: {score.mean()}")
    


def compare_imputation_methods(train_df, num_cols, columns_to_encode, cv=5, target_col='listening_time_minutes'):
    train = train_df.copy()
    
    start_time = time.time()
    print("Starting imputation method comparison...")

    missing_cols = train.columns[train.isna().any()].tolist()
    X_train = train.drop(target_col, axis=1)
    y_train = train[target_col]
    
    print(f"Columns with missing values: {missing_cols}")

    results = {
        'method': [],
        'rmse_cv_mean': [],
        'rmse_cv_std': [],
        'runtime': []
    }

    sample_size = min(50000, len(train))
    eval_model = LGBMRegressor(n_estimators=100, random_state=RSEED, verbose=0)
    rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
    kf = KFold(n_splits=cv, shuffle=True, random_state=RSEED)
    
    imputers = {}
    imputed_dfs = {}
    
    X_train_encoded = encode_columns(X_train.copy(), columns_to_encode)

    # ======================= Listwise deletion ======================
    print("\nImplementing Listwise Deletion...")
    method_start = time.time()

    X_train_del = X_train.copy()
    y_train_del = y_train.copy()
    nan_mask = X_train_del.isna().any(axis=1)

    X_train_del = X_train_del[~nan_mask]
    y_train_del = y_train_del[~nan_mask]

    X_train_del_model = X_train_del.copy()
    X_train_del_model = encode_columns(X_train_del_model, columns_to_encode)
    
    scores = cross_val_score(
        eval_model, X_train_del_model, y_train_del, 
        cv=kf, scoring=rmse_scorer, n_jobs=-1
    )
    
    method_time = time.time() - method_start
    results['method'].append('Listwise Deletion')
    results['rmse_cv_mean'].append(abs(scores.mean()))
    results['rmse_cv_std'].append(scores.std())
    results['runtime'].append(method_time)
    
    print(f"Listwise Deletion - RMSE: {abs(scores.mean()):.4f} ± {scores.std():.4f}, Time: {method_time:.2f}s")
    X_train_del[target_col] = y_train_del
    imputed_dfs['listwise'] = X_train_del

    # ======================= Simple Mean/Mode Imputation ======================
    print("\nImplementing Simple Mean/Mode Imputation...")
    method_start = time.time()

    simple_imputer = SimpleImputer(strategy="mean")

    X_train_simple = X_train.copy()
    X_train_simple[missing_cols] = simple_imputer.fit_transform(X_train_simple[missing_cols])
    
    scores = cross_val_score(
        eval_model, X_train_simple, y_train, 
        cv=kf, scoring=rmse_scorer, n_jobs=-1
    )
    
    method_time = time.time() - method_start
    results['method'].append('Simple (Mean/Mode)')
    results['rmse_cv_mean'].append(abs(scores.mean()))
    results['rmse_cv_std'].append(scores.std())
    results['runtime'].append(method_time)
    
    print(f"Simple imputation - RMSE: {abs(scores.mean()):.4f} ± {scores.std():.4f}, Time: {method_time:.2f}s")
    imputers["simple"] = simple_imputer
    X_train_simple[target_col] = y_train
    imputed_dfs['simple'] = X_train_simple

    # ======================= MICE Imputation ======================
    try:
        print("\nImplementing MICE Imputation...")
        method_start = time.time()

        mice_imputer = IterativeImputer(max_iter=20, verbose=0, random_state=RSEED)

        # Using sampling because MICE takes a moderate amount of compute power. 
        X_train_mice_sampled = X_train_encoded.sample(sample_size, random_state=RSEED)

        mice_imputer.fit(X_train_mice_sampled)
        X_train_mice = mice_imputer.transform(X_train_encoded)
        
        # Convert back to DataFrame with proper column names and preserve original index
        X_train_mice = pd.DataFrame(X_train_mice, columns=X_train_encoded.columns, index=X_train_encoded.index)
        
        scores = cross_val_score(
            eval_model, X_train_mice, y_train, 
            cv=kf, scoring=rmse_scorer, n_jobs=-1
        )
        
        method_time = time.time() - method_start
        results['method'].append('MICE')
        results['rmse_cv_mean'].append(abs(scores.mean()))
        results['rmse_cv_std'].append(scores.std())
        results['runtime'].append(method_time)
        
        print(f"MICE imputation - RMSE: {abs(scores.mean()):.4f} ± {scores.std():.4f}, Time: {method_time:.2f}s")
        
        imputers['mice'] = {'imputer': mice_imputer}
        X_train_mice[target_col] = y_train
        imputed_dfs['mice'] = X_train_mice
        
    except Exception as e:
        print(f"Error with MICE imputation: {e}")

    # ======================= Soft Impute Imputation ======================
    try:
        print("\nImplementing Soft Impute Imputation...")
        method_start = time.time()
        X_train_mtc = X_train_encoded.copy()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train_mtc[num_cols])
        X_train_mtc[num_cols] = pd.DataFrame(X_scaled, columns=num_cols)
        
        X_imputed_mtc = SoftImpute(
            max_iters=10, # more iters is mostly useless, because the model converges 
                          # much faster according to verbose messages
            verbose=0
        ).fit_transform(X_train_mtc)
        X_imputed_mtc = pd.DataFrame(X_imputed_mtc, columns=X_train_mtc.columns, index=X_train_mtc.index)
        
        scores = cross_val_score(
            eval_model, X_imputed_mtc, y_train, 
            cv=kf, scoring=rmse_scorer, n_jobs=-1
        )
        
        method_time = time.time() - method_start
        results['method'].append('Matrix Completion')
        results['rmse_cv_mean'].append(abs(scores.mean()))
        results['rmse_cv_std'].append(scores.std())
        results['runtime'].append(method_time)
        
        print(f"Soft Impute imputation - RMSE: {abs(scores.mean()):.4f} ± {scores.std():.4f}, Time: {method_time:.2f}s")
        
        imputers['matrix'] = {'imputer': 'svd_completion'}
        X_imputed_orig = X_imputed_mtc.copy()
        X_imputed_orig[num_cols] = scaler.inverse_transform(X_imputed_mtc[num_cols])
        X_imputed_orig[target_col] = y_train
        imputed_dfs['matrix'] = X_imputed_orig
        
    except Exception as e:
        print(f"Error with Soft Impute imputation: {e}")

    # ======================= KNN Imputation (SINGLE K) ======================
    print("\nImplementing KNN Imputation (k=5)...")
    method_start = time.time()
    
    try:
        k = 5
        train_sample = X_train_encoded.sample(n=sample_size, random_state=RSEED)
        X_train_knn = X_train_encoded.copy()
        knn_imputer = KNNImputer(n_neighbors=k)

        knn_imputer.fit(train_sample)
        X_train_knn_imputed = knn_imputer.transform(X_train_knn)
        
        X_train_knn = pd.DataFrame(X_train_knn_imputed, columns=X_train_knn.columns, index=X_train_knn.index)
        
        scores = cross_val_score(
            eval_model, X_train_knn, y_train, 
            cv=kf, scoring=rmse_scorer, n_jobs=-1
        )
        
        method_time = time.time() - method_start
        results['method'].append(f'KNN (k={k})')
        results['rmse_cv_mean'].append(abs(scores.mean()))
        results['rmse_cv_std'].append(scores.std())
        results['runtime'].append(method_time)
        
        print(f"KNN imputation - RMSE: {abs(scores.mean()):.4f} ± {scores.std():.4f}, Time: {method_time:.2f}s")
        
        imputers['knn'] = {'imputer': knn_imputer}
        X_train_knn[target_col] = y_train
        imputed_dfs['knn'] = X_train_knn
    
    except Exception as e:
        print(f"Error with KNN imputation: {e}")

    
    # ======================= Results ======================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse_cv_mean')
    
    best_method = results_df.iloc[0]['method']
    print(f"\nBest imputation method: {best_method} with RMSE: {results_df.iloc[0]['rmse_cv_mean']:.4f}")

    best_method_key = best_method.split(' ')[0].lower()
    best_imputer = next((imputers[k] for k in imputers.keys() if k in best_method_key.lower()), None)
    
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")
    
    return results_df, best_imputer, imputed_dfs, imputers


temp = train.drop(missing_columns, axis=1)
mis_train, mis_val = train_test_split(temp, test_size=0.1, shuffle=True, random_state=RSEED)


%%time

missing_results_path = RESULTS_PATH + "missing_imputer.pkl"
to_save = False

if not os.path.exists(missing_results_path):
    cols_to_scale = [c for c in num_cols if c != target]
    results_df, best_imputer, imputed_dfs, imputers = compare_imputation_methods(
        mis_train, cols_to_scale, 
        columns_to_encode, target_col='listening_time_minutes', cv=5  
    )

    display(results_df)

    # It will save only while running computing. 
    # Next time the cell will be actived, to_save will become False.
    to_save = True


with open(missing_results_path, "wb" if to_save else "rb") as f:
    if to_save:
        obj = {
            "results_df": results_df, 
            "best_imputer": best_imputer, 
            "imputed_dfs": imputed_dfs, 
            "imputers": imputers
        }
        pickle.dump(obj, f)

    else:
        data = pickle.load(f)
        results_df, best_imputer, imputed_dfs, imputers = data.values()



train_imp_listwise = imputed_dfs['listwise']
train_imp_simple = imputed_dfs['simple']

train_imp_lw_enc = encode_columns(train_imp_listwise, columns_to_encode, to_drop=True)
train_imp_sim_enc = encode_columns(train_imp_simple, columns_to_encode, to_drop=True)

X_imp_lw_enc, y_lw_train = train_imp_lw_enc.drop(target, axis=1), train_imp_lw_enc[target]
X_imp_sim_enc, y_train = train_imp_sim_enc.drop(target, axis=1), train_imp_sim_enc[target]


encoded_mis_val = encode_columns(mis_val, columns_to_encode, to_drop=True)
mis_val_sim_enc = encoded_mis_val.copy()
mis_val_sim_enc[missing_columns_orig] = imputers["simple"].transform(mis_val_sim_enc[missing_columns_orig])
X_val_sim_enc, y_val = mis_val_sim_enc.drop(target, axis=1), mis_val_sim_enc[target]

test_imp_sim_enc = encoded_test.copy()
test_imp_sim_enc[missing_columns_orig] = imputers["simple"].transform(test_imp_sim_enc[missing_columns_orig])


xgb = XGBRegressor(random_state=RSEED)
handle_prediction(X_imp_lw_enc, y_lw_train, X_val_sim_enc, y_val, xgb, storing="xgb_enc_lw")


handle_prediction(X_imp_sim_enc, y_train, X_val_sim_enc, y_val, xgb, storing="xgb_enc_sim")


global_results


# Minimalist visualization with values
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 7))
methods = results_df.sort_values('rmse_cv_mean')

# Create barplot
bars = sns.barplot(x='method', y='rmse_cv_mean', data=methods, ax=axes[0])
bars2 = sns.barplot(x='method', y='runtime', data=methods, ax=axes[1])

# Add value labels
for i, bar in enumerate(bars.patches):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{methods.iloc[i]['rmse_cv_mean']:.3f}", 
            ha='center', fontsize=9)
    
for i, bar in enumerate(bars2.patches):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7,
            f"{methods.iloc[i]['runtime']:.3f}", 
            ha='center', fontsize=9)


axes[0].set_title('Imputation Method Performance (RMSE)', fontsize=13)
axes[1].set_title('Runtime Comparison', fontsize=13)
axes[0].set_ylabel('RMSE (lower is better)', fontsize=11)
axes[0].set_xlabel('')
axes[1].set_ylabel('Runtime (seconds)', fontsize=11)
axes[1].set_xlabel('Imputation Method', fontsize=11)
plt.tight_layout()
plt.show()


X_final = pd.concat([X_imp_sim_enc, X_val_sim_enc], axis=0)
y_final = pd.concat([y_train, y_val], axis=0)

xgb.fit(X_final, y_final)
yhat = xgb.predict(test_imp_sim_enc)
save_submission(test_id, yhat, target, "xgb_enc_sim.csv")

