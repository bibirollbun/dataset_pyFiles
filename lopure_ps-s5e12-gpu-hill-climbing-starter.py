VER=1

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
y_true = train['diagnosed_diabetes'].values

train.head()


import pandas as pd
import numpy as np
import glob
import os
import re

# -----------------------------------------------------
# 1. Setup Path and Model List
# -----------------------------------------------------
PATH = "/kaggle/input/dec2025-playground-s5e12-oofs-testpreds/Hill Climbing"
folders = sorted([
    d for d in os.listdir(PATH)
    if os.path.isdir(os.path.join(PATH, d))
])

x_train = [] 
x_test = []  
cv_scores = [] # NEW: To store the scores extracted from filenames
model_names = []

print("Loading files and extracting CV scores...")

for fold_name in folders:
    dir_path = os.path.join(PATH, fold_name)
    
    # Patterns to find the files
    oof_pattern = os.path.join(dir_path, "*_OOF_Preds.csv")
    test_pattern = os.path.join(dir_path, "*_Test_Preds.csv")
    
    oof_files = glob.glob(oof_pattern)
    test_files = glob.glob(test_pattern)
    
    if oof_files and test_files:
        filename = os.path.basename(oof_files[0])
        
        # --- NEW: Extract CV Score using Regex ---
        # This looks for a decimal number (like 0.73075) in the filename
        score_match = re.search(r"(\d+\.\d+)", filename)
        if score_match:
            score = float(score_match.group(1))
        else:
            score = 0.0 # Default if no score found
            
        print(f"=> Found {fold_name} | Extracted CV: {score:.5f}")
        
        # Load Data
        df_oof = pd.read_csv(oof_files[0])
        df_test = pd.read_csv(test_files[0])
        
        x_train.append(df_oof['pred'].values)
        x_test.append(df_test['diagnosed_diabetes'].values)
        cv_scores.append(score)
        model_names.append(fold_name)
    else:
        print(f"!! Missing files in folder: {fold_name}")

# Convert to Matrix
x_train = np.array(x_train)
x_test = np.array(x_test)
cv_scores = np.array(cv_scores)

print("-" * 30)
print(f"Final Count: {len(model_names)} models loaded.")
print(f"Average CV Score: {np.mean(cv_scores):.5f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Convert to DataFrame for easier handling
oof_df = pd.DataFrame(
    x_train.T,
    columns=model_names
)

print(oof_df.shape)
corr = oof_df.corr(method="pearson")
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5
)
plt.title("OOF Prediction Correlation Between Models", fontsize=16)
plt.tight_layout()
plt.show()




x_train = np.stack(x_train).T
print("Our combined OOF have shape:",x_train.shape)

x_test = np.stack(x_test).T
print("Our combined PRED have shape:",x_test.shape)


from sklearn.metrics import roc_auc_score
# 4. Corrected Metric Calculation
def compute_metric_auc(p):
    # Use the renamed y_true_labels to ensure no None errors
    return roc_auc_score(y_true, p)

best_score = -1.0
best_index = -1

for k, name in enumerate(model_names):
    # Slice as [:, k] because shape is (Samples, Models)
    s = compute_metric_auc(x_train[:, k])
    
    if s > best_score:
        best_score = s
        best_index = k
    
    print(f'AUC {s:0.5f} {name}') 

print("-" * 30)
print(f'Best single model is {model_names[best_index]} with AUC = {best_score:0.5f}')


import cupy as cp
import gc

def multiple_auc_scores(actual, predicted):
    """
    Computes multiple exact AUC scores simultaneously using GPU.
    
    Parameters:
    ----------
    actual : cupy.ndarray
        1D GPU array of shape (N,) containing binary true labels (0 or 1).
    
    predicted : cupy.ndarray
        2D GPU array of shape (N, K) containing predicted probabilities.
        K is the number of weight combinations/models.

    Returns:
    -------
    cupy.ndarray
        1D GPU array of shape (K,) containing AUC scores for each column.
    """
    # Ensure actual is a 1D array for ranking
    if len(actual.shape) > 1:
        actual = actual.flatten()
        
    n_samples = predicted.shape[0]
    n_models = predicted.shape[1]
    
    # Count positives and negatives
    n_pos = cp.sum(actual)
    n_neg = n_samples - n_pos
    
    # 1. Get ranks for each column (0 to N-1)
    # cp.argsort returns the indices that would sort each column
    # We then argsort again to get the rank of each original index
    ranks = cp.argsort(predicted, axis=0).argsort(axis=0) + 1.0
    
    # 2. Sum ranks of positive samples only
    # We multiply by actual (0s and 1s) to zero out ranks of negative samples
    # Resulting shape: (n_models,)
    pos_rank_sum = cp.sum(ranks * actual[:, cp.newaxis], axis=0)
    
    # 3. Calculate AUC using the Mann-Whitney U formula
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    return auc


print(f'0 Starting with best model AUC {best_score:0.5f} from "{model_names[best_index]}"')


import cupy as cp, gc
import plotly.graph_objects as go

# 1. Configuration 
USE_NEGATIVE_WGT = False 
MAX_MODELS = 1000
TOL = 1e-6
BATCH_SIZE = 100 

indices = [best_index]
old_best_score = best_score

# 2. Variable Movement to GPU
x_train2 = cp.array(x_train)
best_ensemble = x_train2[:, best_index]
truth = cp.array(y_true)

# 3. Search Space Setup
start = -0.20 if USE_NEGATIVE_WGT else 0.001
ww_all = cp.arange(start, 0.501, 0.001)

# ---  Lists for plotting ---
models = [best_index]
weights = [0] # Starting model has no "blending weight" relative to itself
metrics = [best_score]
step_labels = [f"Start: {model_names[best_index]}"] # For the x-axis labels

for kk in range(MAX_MODELS):
    best_step_score = -1.0
    best_step_index = -1
    best_step_weight = 0

    for k in range(len(model_names)):
        new_model = x_train2[:, k]
        for i in range(0, len(ww_all), BATCH_SIZE):
            ww = ww_all[i : i + BATCH_SIZE]
            nn = len(ww)
            m1 = cp.repeat(best_ensemble[:, cp.newaxis], nn, axis=1) * (1-ww)
            m2 = cp.repeat(new_model[:, cp.newaxis], nn, axis=1) * ww
            mm = m1 + m2 
            new_aucs = multiple_auc_scores(truth, mm)
            current_batch_max = cp.max(new_aucs).item()
            
            if current_batch_max > best_step_score:
                best_step_score = current_batch_max 
                best_step_index = k 
                ii = cp.argmax(new_aucs).item() 
                best_step_weight = ww[ii].item() 
                potential_ensemble = mm[:, ii]
            
            del m1, m2, mm, new_aucs
            cp.get_default_memory_pool().free_all_blocks()

    if (best_step_score - old_best_score) < TOL: 
        print(f'=> Reached tolerance {TOL}')
        break

    # Update lists for plotting
    models.append(best_step_index)
    weights.append(best_step_weight)
    metrics.append(best_step_score)
    step_labels.append(f"Step {kk+1}: {model_names[best_step_index]}")
    
    # Print status as backup
    print(f'{kk+1} New best AUC {best_step_score:0.6f} adding "{model_names[best_step_index]}"')
    
    best_ensemble = potential_ensemble
    old_best_score = best_step_score
    gc.collect()

# --- NEW: PLOTLY PROGRESS GRAPH ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(len(metrics))),
    y=metrics,
    mode='lines+markers',
    text=step_labels,
    hoverinfo='text+y',
    line=dict(color='#00CC96', width=3),
    marker=dict(size=10, symbol='diamond')
))

fig.update_layout(
    title="Hill Climbing Ensemble Progress (AUC Optimization)",
    xaxis_title="Iteration (Models Added)",
    yaxis_title="OOF AUC Score",
    template="plotly_white",
    hovermode="x unified"
)

fig.show(renderer='iframe')


import numpy as np
import pandas as pd

# 1. Initialize global weights starting with the first model (weight 1.0)
wgt = np.array([1.0])

# 2. Iteratively update weights based on the Hill Climbing history
# If we added a model with weight 'w', the existing ensemble is scaled by (1-w)
for w in weights:
    wgt = wgt * (1 - w)
    wgt = np.concatenate([wgt, np.array([w])])
    
rows = []
# 3. Zip the model indices, calculated weights, and the AUC metrics together
# Note: 'models' contains the index, 'wgt' the absolute weight, 'metrics' the AUC history
for m, w, s in zip(models, wgt, metrics):
    name = model_names[m] # Using your model_names list
    dd = {}
    dd['model'] = name
    dd['weight'] = w
    dd['auc_at_step'] = s
    rows.append(dd)

# 4. Aggregate weights by model 
# (Since a model can be added multiple times during Hill Climbing)
df_weights = pd.DataFrame(rows)
df_weights = df_weights.groupby('model').agg({'weight': 'sum', 'auc_at_step': 'max'}).reset_index()
df_weights = df_weights.sort_values('weight', ascending=False).reset_index(drop=True)

# 5. Sanity Check
print(f"Ensemble weights sum to: {df_weights.weight.sum():.4f}")
print("-" * 30)
display(df_weights)


# 1. Create a mapping of model names to their column index in x_train
# x_train was stacked as (Samples, Models), so XGB might be index 0, CatBoost index 1, etc.
x_map = {name: i for i, name in enumerate(model_names)}

# 2. Bring the OOF predictions back from GPU to CPU for final storage
# x_train2 was our CuPy array; .get() converts it back to a NumPy array
x_train_cpu = x_train2.get()

# 3. Initialize the ensemble with the first model's weighted predictions
first_model = df_weights.model.iloc[0]
first_weight = df_weights.weight.iloc[0]
ensemble_oof = x_train_cpu[:, x_map[first_model]] * first_weight

# 4. Add the remaining models scaled by their respective weights
for k in range(1, len(df_weights)):
    model_name = df_weights.model.iloc[k]
    weight = df_weights.weight.iloc[k]
    ensemble_oof += x_train_cpu[:, x_map[model_name]] * weight

# 5. Compute the final Overall CV Score (AUC)
# Using your compute_metric_auc function defined earlier
final_cv_auc = compute_metric_auc(ensemble_oof)

print(f'ğŸ�� Overall Hill Climbing Ensemble AUC = {final_cv_auc:0.6f}')

# 6. Save the OOF predictions for future stacking or analysis
np.save(f'oof_hill_climb_v{VER}.npy', ensemble_oof)


# 1. Create mapping for test columns
x_map = {name: i for i, name in enumerate(model_names)}

# 2. Initialize test predictions with the first weighted model
# x_test shape is (300000, 6)
first_model = df_weights.model.iloc[0]
first_weight = df_weights.weight.iloc[0]
final_preds = x_test[:, x_map[first_model]] * first_weight

# 3. Sum the rest of the weighted models
for k in range(1, len(df_weights)):
    model_name = df_weights.model.iloc[k]
    weight = df_weights.weight.iloc[k]
    final_preds += x_test[:, x_map[model_name]] * weight

# 4. Prepare Submission CSV
# Ensure the path points to the correct competition sample submission
sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

# 5. Assign predictions
# Since this is a probability-based competition (AUC), we do NOT clip or expm1
sub['diagnosed_diabetes'] = final_preds

print("Test shape:", sub.shape)
print("Prediction Mean:", sub['diagnosed_diabetes'].mean())

# 6. Final Export
sub.to_csv(f"submission_hill_climb_v{VER}.csv", index=False)
sub.head()


import plotly.express as px

# Create the interactive histogram
fig = px.histogram(
    sub, 
    x="diagnosed_diabetes", 
    nbins=100,
    title=f"Distribution of Ensemble Predictions (v{VER})",
    labels={'diagnosed_diabetes': 'Predicted Probability'},
    color_discrete_sequence=['#636EFA'], # Standard Plotly blue
    opacity=0.7
)

# Customize layout for better readability
fig.update_layout(
    xaxis_title="Predicted Probability (diagnosed_diabetes)",
    yaxis_title="Count",
    showlegend=False,
    template="plotly_white",
    bargap=0.1
)

fig.show(renderer='iframe')




