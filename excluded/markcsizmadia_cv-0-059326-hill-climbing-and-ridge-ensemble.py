import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn import set_config
set_config(transform_output="pandas")


dir_data = Path("/kaggle/input/playground-series-s5e5")

data = pd.read_csv(
    filepath_or_buffer=dir_data / "train.csv",
    dtype={
        'id': 'int32',
        'Sex': 'string',
        'Age': 'int32',
        'Height': 'float32',
        'Weight': 'float32',
        'Duration': 'float32',
        'Heart_Rate': 'float32',
        'Body_Temp': 'float32',
        'Calories': 'float32',
    },
)

data_sub = pd.read_csv(
    filepath_or_buffer=dir_data / "test.csv",
    dtype={
        'id': 'int32',
        'Sex': 'string',
        'Age': 'int32',
        'Height': 'float32',
        'Weight': 'float32',
        'Duration': 'float32',
        'Heart_Rate': 'float32',
        'Body_Temp': 'float32',
    },
)


# drop duplicates
print(data.shape)
data = data.drop_duplicates(subset=data.columns.drop("id"), keep="first")
data.reset_index(drop=True, inplace=True)
print(data.shape)


models = ["xgb", "cat", "h2o"]
oof_df_dict = {
    "xgb": pd.read_csv("/kaggle/input/cv-0-05992-xgb-feature-eng-eda/oof.csv"),
    "h2o": pd.read_csv("/kaggle/input/cv-0-06071-feature-eng-h2o-automl/18_oof.csv"),
    "cat": pd.read_csv("/kaggle/input/cv-0-05966-catboost-feature-eng-eda/oof.csv")
}
y_test_hat_ave_df_dict = {
    "xgb": pd.read_csv("/kaggle/input/cv-0-05992-xgb-feature-eng-eda/submission.csv"),
    "h2o": pd.read_csv("/kaggle/input/cv-0-06071-feature-eng-h2o-automl/18.csv"),
    "cat": pd.read_csv("/kaggle/input/cv-0-05966-catboost-feature-eng-eda/submission.csv"),
}


oof_dict = {k: v["Calories_Pred"].to_numpy() for k, v in oof_df_dict.items()}
y_test_hat_ave_dict = {k: v["Calories"].to_numpy() for k, v in y_test_hat_ave_df_dict.items()}


oof_dict


y_test_hat_ave_dict


# seed numpy
random_state = 42
np.random.seed(random_state)
oofs = [oof_dict[model] for model in models]
y_test_hat_aves = [y_test_hat_ave_dict[model] for model in models]
# Shape: (n_samples_train, n_models)
oof_preds = np.column_stack(oofs)
oof_preds_df = pd.DataFrame(oof_preds, columns=models)
# Shape: (n_samples_test, n_models)
test_preds = np.column_stack(y_test_hat_aves)
test_preds_df = pd.DataFrame(test_preds, columns=models)
oof_preds_df.head()
test_preds_df.head()



def compute_metric(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


# best individual model
best_score = 1e10
best_model = None
for model, oof_pred in oof_preds_df.items():
    s = compute_metric(y_true=np.log1p(data["Calories"]), y_pred=np.log1p(oof_pred))
    if s < best_score:
        best_score = s
        best_model = model
    print(f'RMSE {s:0.5f} {model}') 
print()
print(f'Best single model is {best_model} with RMSE = {best_score:0.5f}')


import numpy as np
import pandas as pd
import gc

# === INPUTS ===
# df_preds: DataFrame with columns as model names and rows as OOF predictions
# y_true: 1D NumPy array of true target values

USE_NEGATIVE_WGT = True
MAX_MODELS = 20
TOL = 1e-6

# Convert DataFrame to NumPy array for fast computation
x_train_np = np.log1p(oof_preds_df.values)
model_names = oof_preds_df.columns.tolist()
truth_np = np.array(np.log1p(data["Calories"]))

# === START WITH BEST SINGLE MODEL ===
rmse_scores = np.sqrt(np.mean((x_train_np - truth_np[:, None]) ** 2, axis=0))
best_index = np.argmin(rmse_scores)
best_score = rmse_scores[best_index]

print(f'0 Starting with RMSE {best_score:.5f} from "{model_names[best_index]}"')

indices = [best_index]
models = [best_index]
weights = []
metrics = [best_score]
ensemble = x_train_np[:, best_index]
old_best_score = best_score

# === WEIGHT GRID ===
start_weight = -0.50 if USE_NEGATIVE_WGT else 0.01
weight_grid = np.arange(start_weight, 0.51, 0.01)

# === HILL CLIMBING LOOP ===
for step in range(1_000_000):
    best_score = float('inf')
    best_candidate = -1
    best_weight = 0

    for k in range(x_train_np.shape[1]):
        if k in models:
            continue  # Skip already used models

        candidate = x_train_np[:, k]

        blend_matrix = np.outer(ensemble, 1 - weight_grid) + np.outer(candidate, weight_grid)
        
        errors = blend_matrix - truth_np[:, None]
        rmse_scores = np.sqrt(np.mean(errors ** 2, axis=0))

        min_idx = np.argmin(rmse_scores)
        score = rmse_scores[min_idx]

        if score < best_score:
            best_score = score
            best_candidate = k
            best_weight = weight_grid[min_idx]
            best_blend = blend_matrix[:, min_idx]

    # del candidate, blend_matrix, errors, rmse_scores
    # gc.collect()

    # === STOPPING CRITERIA ===
    indices.append(best_candidate)
    indices = list(np.unique(indices))

    if len(indices) > MAX_MODELS:
        print(f'=> Stopped: Reached MAX_MODELS = {MAX_MODELS}')
        indices = indices[:-1]
        break
    if old_best_score - best_score < TOL:
        print(f'=> Stopped: Improvement < TOL = {TOL}')
        break

    # === RECORD NEW BEST ===
    print(f'{step + 1} New best RMSE {best_score:.5f} by adding "{model_names[best_candidate]}" with weight {best_weight:.3f}')
    models.append(best_candidate)
    weights.append(best_weight)
    metrics.append(best_score)
    ensemble = best_blend
    old_best_score = best_score



wgt = np.array([1])
for w in weights:
    wgt = wgt*(1-w)
    wgt = np.concatenate([wgt,np.array([w])])
    
rows = []
t = 0
for m,w,s in zip(models,wgt,metrics):
    name = model_names[m]
    dd = {}
    dd['weight'] = w
    dd['model'] = name
    rows.append(dd)
    t += float( f'{w:.3f}' )

# DISPLAY WEIGHT PER MODEL
df = pd.DataFrame(rows)
df = df.groupby('model').agg('sum').reset_index().sort_values('weight',ascending=False)
df = df.reset_index(drop=True)
df


print('Ensemble weights sum to',df.weight.sum())


# COMBINE OOF PREDITIONS (using weights from hill climbing)
x_map = {x:y for x,y in zip(model_names,np.arange(len(model_names)))}

ensemble = x_train_np[:, x_map[df.model.iloc[0]] ] * df.weight.iloc[0]
for k in range(1,len(df)):
    ensemble += x_train_np[:, x_map[df.model.iloc[k]] ] * df.weight.iloc[k]
m = compute_metric(y_true=np.log1p(data["Calories"]), y_pred=ensemble)
print(f'Overall Hill climbing RMSE = {m:0.6f}')

np.save(f'oof_hill_climb',ensemble)


# COMBINE TEST PREDITIONS (using weights from hill climbing)
x_test_np = np.log1p(test_preds_df.values)
x_map = {x:y for x,y in zip(model_names,np.arange(len(model_names)))}
pred = x_test_np[:, x_map[df.model.iloc[0]] ] * df.weight.iloc[0]
for k in range(1,len(df)):
    pred += x_test_np[:, x_map[df.model.iloc[k]] ] * df.weight.iloc[k]


# CLIP TO TRAIN MIN AND MAX
mn = data["Calories"].min(); mx = data["Calories"].max()
data_sub["Calories"] = np.clip( np.expm1( pred ),mn,mx )

print("Test shape", data_sub.shape )
print("Test target mean is", data_sub.Calories.mean())
data_sub[["id", "Calories"]].to_csv(f"submission_hill_climb.csv",index=False)
data_sub.head()


from sklearn.linear_model import Ridge

def ridge_ensemble(oof_preds, y_true, test_preds, alpha=1.0):
    """
    oof_preds: (n_samples, n_models)
    y_true: (n_samples,)
    test_preds: (n_test_samples, n_models)
    alpha: Ridge regularization strength
    """
    # Transform to log1p space
    log_oof_preds = np.log1p(np.clip(oof_preds, a_min=0, a_max=None))
    log_y_true = np.log1p(np.clip(y_true, a_min=0, a_max=None))
    log_test_preds = np.log1p(np.clip(test_preds, a_min=0, a_max=None))

    # Fit ridge regression
    ridge = Ridge(alpha=alpha, fit_intercept=False, positive=True)
    ridge.fit(log_oof_preds, log_y_true)
    weights = ridge.coef_

    # Apply weights
    final_log_oof = log_oof_preds @ weights
    final_log_test = log_test_preds @ weights

    # Transform back
    final_oof = np.expm1(final_log_oof)
    final_test = np.expm1(final_log_test)

    return weights, final_oof, final_test


weights_ridge, final_oof_ridge, final_test_ridge = ridge_ensemble(oof_preds=oof_preds, y_true=data["Calories"], test_preds=test_preds)
print(f"Ensemble Ridge model-weights: {list(zip(models, weights_ridge))}")
local_cv_score_ridge = compute_metric(y_true=np.log1p(data["Calories"]), y_pred=np.log1p(final_oof_ridge))
print(f"Overall CV Score Ridge = {local_cv_score_ridge:.8f}")

mn = data["Calories"].min(); mx = data["Calories"].max()
data_sub["Calories"] = np.clip( final_test_ridge ,mn,mx )
print("Test shape", data_sub.shape )
print("Test target mean is", data_sub.Calories.mean())
data_sub[["id", "Calories"]].to_csv(f"submission_ridge.csv",index=False)
data_sub.head()




