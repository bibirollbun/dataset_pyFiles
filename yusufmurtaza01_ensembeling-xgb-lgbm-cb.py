import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import os

import warnings

warnings.filterwarnings("ignore")


TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
ORIGINAL_PATH = "/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"

SEED = 42
N_ESTIMATORS = 3000
EARLY_STOPPING = 100  # Stop if no improvement for 100 rounds
TARGET = 'diagnosed_diabetes'

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
original_df = pd.read_csv(ORIGINAL_PATH)

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")
print(f"Original Shape: {original_df.shape}")


# Handle ID columns
if 'id' in train_df.columns:
    train_df = train_df.drop(columns=['id'])
if 'id' in test_df.columns:
    submission_id = test_df['id']
    test_df = test_df.drop(columns=['id'])
else:
    submission_id = test_df.index

# Align columns
common_cols = list(set(train_df.columns).intersection(set(original_df.columns)))
original_df = original_df[common_cols]

# Concatenate (Hybrid Data Loading)
train_full = pd.concat([train_df, original_df], axis=0).reset_index(drop=True)
print(f"Combined Training Data shape: {train_full.shape}")


# Prepare X, y
X = train_full.drop(columns=[TARGET])
y = train_full[TARGET]
X_test = test_df[X.columns] # Ensure column alignment


# Ordinal Encoding (Safe for Tree Models)
cat_cols = X.select_dtypes(include=['object']).columns
# handle_unknown='use_encoded_value' prevents errors if Test data has new categories
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X[cat_cols] = enc.fit_transform(X[cat_cols])
X_test[cat_cols] = enc.transform(X_test[cat_cols])


# --- 2. TRAINING FUNCTION (Fixed) ---
def train_and_submit(model_name, params, X, y, X_test, submission_filename, N_FOLDS=1):
    
    # N_FOLDS=2 # for dry run/ check
    
    print(f"\nTraining {model_name} ({N_FOLDS} folds)...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # --- XGBOOST ---
        if model_name == 'XGBoost':
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)], 
                verbose=False
            )
            
        # --- LIGHTGBM ---
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(**params)
            # Early stopping is passed via callbacks in newer sklearn API
            callbacks = [
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(0)
            ]
            model.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)], 
                eval_metric='auc',
                callbacks=callbacks
            )
            
        # --- CATBOOST ---
        elif model_name == 'CatBoost':
            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train, 
                eval_set=(X_val, y_val), 
                verbose=False,
                early_stopping_rounds=EARLY_STOPPING
            )
        
        # Prediction & Scoring
        # Use best_iteration_ if available (XGB/LGBM), usually handled automatically by predict_proba
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        score = roc_auc_score(y_val, val_pred)
        scores.append(score)
        
        # Test Prediction (Average over folds)
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"Fold {fold+1} AUC: {score:.5f}")
        
    avg_auc = np.mean(scores)
    print(f"  -> {model_name} ({N_FOLDS}) Avg AUC: {avg_auc:.5f}")
    
    # Save Submission
    sub = pd.DataFrame({'id': submission_id, TARGET: test_preds})
    sub.to_csv(submission_filename, index=False)
    print(f"Saved: {submission_filename}")
    
    return avg_auc, test_preds


#  MODEL PARAMETERS (Optimized) ---
# Note: Added early_stopping_rounds to constructors where applicable as a backup

xgb_params = {
    'n_estimators': N_ESTIMATORS,
    'learning_rate': 0.015,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': SEED,
    'early_stopping_rounds': EARLY_STOPPING, # Added for XGB constructor
    
    'tree_method': 'gpu_hist', # Uncomment if using GPU
    'predictor': 'gpu_predictor'   # GPU ENABLED
}

lgb_params = {
    'n_estimators': N_ESTIMATORS,
    'learning_rate': 0.015,
    'num_leaves': 64,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'random_state': SEED,
    'verbosity': -1,
    'device': 'cpu' # Use 'gpu' only if you are sure images are available
}

cat_params = {
    'iterations': N_ESTIMATORS,
    'learning_rate': 0.015,
    'depth': 8,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': SEED,
    'verbose': False,
    'allow_writing_files': False,

    'task_type': 'GPU',            # GPU ENABLED
    'devices': '0'                 # Use first GPU}
}


#  EXECUTION ---
results = []
ensemble_preds = {}

models_config = [
    ('XGBoost', xgb_params, 10, 'submission_xgb_10.csv'),
    ('XGBoost', xgb_params, 5, 'submission_xgb_5.csv'),
    ('LightGBM', lgb_params, 10, 'submission_lgb_10.csv'),
    ('LightGBM', lgb_params, 5, 'submission_lgb_5.csv'),
    ('CatBoost', cat_params, 10, 'submission_cb10.csv'),
    ('CatBoost', cat_params, 5, 'submission_cb5.csv'),
]

for name, params, folds, filename in models_config:
    auc, preds = train_and_submit(name, params, X, y, X_test, filename, folds)
    results.append({'Model': name, 'N_Folds': folds, 'AUC': auc, 'Filename': filename})
    ensemble_preds[filename] = preds


# Result. TABLE DISPLAY ---
results_df = pd.DataFrame(results)
print("\n" + "="*40)
print(" MODEL RESULTS TABLE")
print("="*40)
try:
    print(results_df[['Model', 'N_Folds', 'AUC']].to_markdown(index=False))
except:
    print(results_df[['Model', 'N_Folds', 'AUC']]) # Fallback if markdown not installed


# WEIGHTED ENSEMBLE WITH BEST SUBMISSION ---

best_sub_path = '/kaggle/input/submission-best2/0.70373_submission.csv'
# taken from the result of my best score(0.70368):
# s5-e12-diabetes prediction ensemble

# Add External Best Submission if it exists
if os.path.exists(best_sub_path):
    print(f"Found {best_sub_path}!")
    best_df = pd.read_csv(best_sub_path)
    
    # We assign a theoretical AUC slightly higher than our best model
    # to ensure the formula gives it the highest weight.
    current_max_auc = results_df['AUC'].max()
    best_auc = current_max_auc + 0.002 
    
    results.append({
        'Model': 'Best_External', 
        'N_Folds': '-', 
        'AUC': best_auc, 
        'Filename': 'submission_best.csv'
    })
    ensemble_preds['submission_best.csv'] = best_df[TARGET].values
else:
    print(f"Warning: {best_sub_path} not found. Proceeding without it.")


# Calculate Weights: (AUC - Min_AUC)^2
valid_aucs = [r['AUC'] for r in results]
min_auc = min(valid_aucs)
weights = {}
total_weight = 0

print("\nComputed Weights:")
for r in results:
    raw_auc = r['AUC']
    fname = r['Filename']
    
    # Weight Calculation
    weight = (raw_auc - min_auc + 0.0001) ** 2
    
    # Multiplier for Best External (as requested)
    if r['Model'] == 'Best_External':
        weight *= 3.0  # Give it 3x influence relative to its score position
        
    weights[fname] = weight
    total_weight += weight
    print(f"  {r['Model']} ({r['N_Folds']}): AUC={raw_auc:.5f}, Norm Weight={weight:.5f}")


# Final Blend
final_preds = np.zeros(len(submission_id))
for fname, w in weights.items():
    norm_weight = w / total_weight
    final_preds += ensemble_preds[fname] * norm_weight

# Save Ensemble
ensemble_sub = pd.DataFrame({'id': submission_id, TARGET: final_preds})
ensemble_sub.to_csv('submission.csv', index=False)
print(f"\nFinal Ensemble Saved: submission_ensemble_weighted.csv")


# import numpy as np
# import pandas as pd
# import os, ast, shutil, copy
# from bokeh.plotting import figure, gridplot 
# from bokeh.io import output_file, show, output_notebook

# # Initialize Bokeh for notebook
# output_notebook()


# def arr_colors(color):
#     sg = ['silver','gainsboro']
#     if color=='red'   or color=='r': return ['red','crimson','firebrick'] + sg
#     if color=='Red'   or color=='R': return ['red','tomato','crimson'] + sg
#     if color=='Green' or color=='G': return ['forestgreen','limegreen', 'darkgreen'] + sg
#     if color=='Blue'  or color=='B': return ['blue','royalblue','mediumblue'] + sg
#     if color=='RGB'   or color=='S': return ['mediumblue','darkgreen','crimson'] + sg
#     return ['black','dimgray','gray'] + sg

# def convert(schema):
#     colors = arr_colors(schema[2])
#     dicts  = [{'name': schema[0][i],'weight':schema[1][i],'color':colors[i]} for i in range(len(schema[0]))]
#     return {'subm':dicts}

# def matrix_vs(path,fs_names):
#     def load(path,fs_names):
#         dfs = [pd.read_csv(os.path.join(path, name_subm +'.csv')) for name_subm in fs_names]
#         for i in range(len(dfs)):
#             dfs[i] = dfs[i].rename(columns={"diagnosed_diabetes": f'{fs_names[i]}'}) # Note: Ensure target matches
#         dfsm = pd.merge(dfs[0], dfs[1], on="id")
#         for i in range(2,len(dfs)):
#             dfsm = pd.merge(dfsm,dfs[i],on='id')
#         return dfsm   
#     def make_list_vs(fs_names):
#         list_vs = []
#         for i in range(0,len(fs_names)-1):
#             for j in range(i+1,len(fs_names)):
#                 list_vs.append(fs_names[i] + "_vs_" + fs_names[j])
#         return list_vs
#     def get_mvs(dfs, list_vs):
#         def get_abs_distance(x,t1,t2): return abs(x[t1]-x[t2])
#         for vs in list_vs:
#             t = vs.split('_vs_')
#             dfs[vs] = dfs.apply(lambda x: get_abs_distance(x,t[0],t[1]), axis=1)
#         return dfs   
#     def distance_vs(name, st_names, list_vs, dfs):
#         distances = []
#         for st in st_names:
#             vs_between = name + "_vs_" + st
#             if vs_between not in list_vs: distances.append(0)
#             else: distances.append(round(dfs[vs_between].sum()))
#         return distances
#     dfs = load(path,fs_names)
#     list_vs = make_list_vs(fs_names)
#     mvs = get_mvs(dfs, list_vs)
#     m1 = pd.DataFrame({'subm':fs_names})
#     m2 = pd.DataFrame({ name :distance_vs(name, fs_names, list_vs, mvs) for name in fs_names})
#     return pd.concat([m1,m2],axis=1)

# def display_distances(params):
#     files = [subm['name'] for subm in params['subm']]
#     distances = matrix_vs ( params['path'], files )            
#     display(distances)


# # PLOTTING FUNCTION 
# def bokeh_show(params, df_cross, show_figures1, show_figures2, wps_fig2, color_cross):
#     colors = [subm['color'] for subm in params['subm']]
    
#     # Logic to prepare data for the first set of plots (Bar charts of contributions)
#     def dossier(js, subms, cols):
#         def quant(i, js, subms, cols):
#             # Count how often subms[js] appears in position i across all rows
#             return {"c": i, "q": sum([1 for subm in cols[i] if subm == subms[js]])}
#         return {
#             'name': subms[js],
#             'q_in': [quant(i, js, subms, cols) for i in range(len(subms))]
#         }

#     # Load the description file generated by h_blend
#     # We use the 'tida_desc.csv' that h_blend saves locally
#     if not os.path.exists('tida_desc.csv'):
#         print("Warning: 'tida_desc.csv' not found. Skipping bar charts.")
#         return

#     alls = pd.read_csv('tida_desc.csv')
    
#     # Parse the stringified lists in 'alls' column
#     # The original notebook uses ast.literal_eval to turn "[0.701, ...]" string back to list
#     matrix = [ast.literal_eval(str(row.alls)) for row in alls.itertuples()]
    
#     # Get unique submission names from the first row of matrix
#     subms = sorted(matrix[0]) 
    
#     # Transpose logic: cols[i] is the list of files that appeared at rank i
#     cols = [[data[i] for data in matrix] for i in range(len(subms))]
    
#     dossiers = [dossier(js, subms, cols) for js in range(len(subms))]
#     subm_names = [one_dossier['name'] for one_dossier in dossiers]
    
#     figures1, qss, i = [], [], 0
    
#     # Dynamic height adjustment
#     n_files = len(colors)
#     height = 100 + (n_files - 2) * 20 if n_files > 2 else 100
    
#     # --- Plot 1: Bar charts per rank ---
#     for one_dossier in dossiers: 
#         i_col = 'Rank ' + str(one_dossier['q_in'][i]['c'] + 1)
#         qs = [one['q'] for one in one_dossier['q_in']]
        
#         # Clean up names for x-axis
#         x_names = [str(name).replace("_submission", "") for name in subm_names]
        
#         # Dynamic width
#         width = 130 + (n_files * 5)
        
#         f = figure(x_range=x_names, width=width, height=height, title=i_col)
#         f.vbar(x=x_names, width=0.585, top=qs, color=colors)
#         figures1.append(f)
#         qss.append(qs)
#         i += 1
    
#     if show_figures1:
#         grid = gridplot([figures1])
#         show(grid)

#     # --- Plot 2: Mass relations ---
#     sub_wts = params['subwts']
#     main_wts = [subm['weight'] for subm in params['subm']]
#     mms, acc_mass = [], []
    
#     for j in range(len(dossiers)):
#         one_dossier = dossiers[j]
#         qs = [one['q'] for one in one_dossier['q_in']]
#         # Calculate weighted mass contribution
#         # Note: We safeguard against index errors if sub_wts is shorter than qs
#         mm = [qs[h] * (main_wts[j] + sub_wts[h]) for h in range(len(qs))]
#         mass = sum(mm)
#         mms.append(mm)
#         acc_mass.append(round(mass))

#     y_names = [name + " - " + str(mass) for name, mass in zip(subm_names, acc_mass)]
    
#     f1 = figure(y_range=y_names, width=270 + (n_files*10), height=height, title='Relations of general masses')
#     f1.hbar(y=y_names, height=0.555, right=acc_mass, left=0, color=colors)
    
#     # Stacked bars
#     alls_labels = [f'Rank {i+1}' for i in range(len(dossiers))]
#     subm_keys = [f'sub{i}' for i in range(len(dossiers))]
    
#     mmsT = np.asarray(mms).T
#     data_mass = {'cols': alls_labels}
#     for i in range(len(dossiers)): 
#         data_mass[f'sub{i}'] = mmsT[i,:]
        
#     f2 = figure(y_range=alls_labels, height=height, width=270, title="Relations of columns masses")
#     f2.hbar_stack(subm_keys, y='cols', height=0.555, color=colors, source=data_mass)

#     qssT = np.asarray(qss).T
#     data_ratios = {'cols': alls_labels}
#     for i in range(len(dossiers)): 
#         data_ratios[f'sub{i}'] = qssT[i,:]
        
#     f3 = figure(y_range=alls_labels, height=height, width=245, title="Ratios in columns")
#     f3.hbar_stack(subm_keys, y='cols', height=0.555, color=colors, source=data_ratios)
    
#     grid2 = gridplot([[f3, f2, f1]])
#     show(grid2)

#     # --- Plot 3: Line Chart (Predictions) ---
#     if show_figures2:
#         def read_local(params, i):
#             # Reconstruct filename: path + name + .csv
#             fname = params["subm"][i]["name"] + ".csv"
#             fpath = os.path.join(params["path"], fname)
            
#             # Rename target col to the submission name for plotting legend
#             target_name_back = {
#                 params["target"]: params["subm"][i]["name"],
#                 'target': params["subm"][i]["name"] # fallback
#             }
#             return pd.read_csv(fpath).rename(columns=target_name_back)

#         # Read all files + the blended result (df_cross)
#         dfs = [read_local(params, i) for i in range(len(params["subm"]))] + [df_cross]
        
#         _height = 350
#         f_line = figure(width=785, height=_height, title='Click on legend entries to mute lines')
        
#         # Plot a slice of data (e.g., 100 points) to avoid browser lag
#         b, e = 0, 150 
        
#         # Prepare data lines
#         # We assume 'id' is the x-axis and the target is y
#         id_col = params['id_target'][0]
#         # For the original files, the column name is the file name
#         # For the cross file, it is params['target']
        
#         legend_labels = subm_names + ['Ensemble']
#         plot_colors = colors + [color_cross]
        
#         for i in range(len(dfs)):
#             current_df = dfs[i].iloc[b:e]
            
#             # Determine Y column name
#             if i < len(params["subm"]):
#                 y_col = params["subm"][i]["name"]
#             else:
#                 y_col = params['target'] # The ensemble dataframe
            
#             f_line.line(
#                 current_df[id_col], current_df[y_col], 
#                 line_width=2 if i == len(dfs)-1 else 1, 
#                 color=plot_colors[i], 
#                 alpha=0.8,
#                 muted_color='white',
#                 legend_label=legend_labels[i]
#             )
            
#         f_line.legend.location = "top_left"
#         f_line.legend.click_policy = "mute"
#         show(f_line)


# #  BLENDING FUNCTION 
# def h_blend(params, _update={}, cross='silver', details=False, fig1=False, fig2=False, wf2=555, dtls=False, dist=False, subm=''):
#     if 'path' in _update: params.update(_update)
#     color_cross, dk = cross, copy.deepcopy(params)
    
#     # Setup params
#     type_sort    = params['type_sort'][0]
#     dk['asc']    = params['type_sort'][1]
#     dk['desc']   = params['type_sort'][2]
#     dk['id']     = params['id_target'][0]
#     dk['target'] = params['id_target'][1]

#     def read(dk, i):
#         tnm = dk["subm"][i]["name"]
#         FiN = os.path.join(dk["path"], tnm + ".csv")
#         return pd.read_csv(FiN).rename(columns={'target':tnm, 'pred':tnm, dk["target"]:tnm})
        
#     def merge(dfs_subm):
#         df_subms = pd.merge(dfs_subm[0], dfs_subm[1], on=[dk['id']])
#         for i in range(2, len(dk["subm"])): 
#             df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])
#         return df_subms
        
#     def da(dk, sorting_direction, show_details):
#         df_subms = merge([read(dk,i) for i in range(len(dk["subm"]))])
#         cols = [col for col in df_subms.columns if col != dk['id']]
#         short_name_cols = [c for c in cols]
        
#         def alls1(x, sd=sorting_direction, cs=cols):
#             reverse = True if sd=='desc' else False
#             tes = {c: x[c] for c in cs}.items()
#             return [t[0] for t in sorted(tes, key=lambda k:k[1], reverse=reverse)]

#         def alls2(x, sd=sorting_direction, cs=cols):
#             import random
#             tes = {c: x[c] for c in cs}.items()
#             subms_random = [t[0] for t in tes]
#             random.shuffle(subms_random)
#             return subms_random

#         alls = alls1 if type_sort == 'asc/desc' else alls2
            
#         wts = [[[e['weight'] for e in dk["subm"]], [w for w in dk["subwts"]]]]

#         # NOISE BLENDING
#         if len(wts) == 1:
#             correct_sub_weights = [wt for wt in dk["subwts"]]
#             weights = [subm['weight'] for subm in dk["subm"]]
            
#             noise_strength = 0.05  
#             weights = [w * (1 + np.random.uniform(-noise_strength, noise_strength)) for w in weights]
#             correct_sub_weights = [w * (1 + np.random.uniform(-noise_strength, noise_strength)) for w in correct_sub_weights]

#             def correct(x, cs=cols, w=weights, cw=correct_sub_weights):
#                 ic = [x['alls'].index(c) for c in short_name_cols]
#                 cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
#                 return sum(cS)

#         df_subms['alls'] = df_subms.apply(lambda x: alls(x), axis=1)
#         df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        
#         # Formatting
#         df_subms = df_subms.rename(columns={"ensemble": dk["target"]})
        
#         # SAVE INTERMEDIATE FILE FOR PLOTS
#         if sorting_direction == 'desc': 
#             df_subms.to_csv(f'tida_{sorting_direction}.csv', index=False)
            
#         return df_subms[[dk['id'], dk['target']]]
   
#     def ensemble_da(dk, show_details): 
#         # We run both desc and asc sort logic
#         dfD = da(dk, 'desc', show_details) # This saves tida_desc.csv used by plots
#         dfA = da(dk, 'asc', show_details)
#         # Combine them
#         dfA[dk['target']] = dk['desc']*dfD[dk['target']] + dfA[dk['target']]*dk['asc']
#         return dfA

#     # Run the blending
#     da_result = ensemble_da(dk, details)
    
#     # Run the plotting
#     # We pass 'fig1=True' and 'fig2=True' to ensure plots generate
#     bokeh_show(dk, da_result, show_figures1=True, show_figures2=True, wps_fig2=wf2, color_cross=color_cross)
    
#     if subm != '': da_result.to_csv(subm, index=False)
    
#     return da_result


# # CONSOLIDATE FILES
# # We want all 7 files in the same directory for the blend function to work smoothly.
# # The 6 trained models are already in the current directory ('.').
# # We copy the best external submission here as well.

# external_best_path = '/kaggle/input/submission-best1/submission_best1.csv'
# local_best_name = 'submission_best.csv'

# if os.path.exists(external_best_path):
#     shutil.copy(external_best_path, local_best_name)
#     print(f"Copied {external_best_path} to current directory.")
# else:
#     # If not found, we will just proceed with the 6 we have, but warn the user.
#     print(f"Warning: {external_best_path} not found. Blending only trained models.")
#     local_best_name = None


# # Get AUCs from the training results_df we created earlier
# # Convert results to a list of dicts: {'name': 'filename', 'auc': 0.7xxx}
# file_candidates = []

# # Add the 6 trained models
# for i, row in results_df.iterrows():
#     file_candidates.append({
#         'name': row['Filename'],
#         'auc': row['AUC']
#     })

# # Add the external best model
# # We assign it a "theoretical" AUC higher than the best trained model 
# # to ensure it takes the top spot (Rank 1).
# if local_best_name:
#     max_trained_auc = max([x['auc'] for x in file_candidates]) if file_candidates else 0
#     file_candidates.append({
#         'name': local_best_name,
#         'auc': max_trained_auc + 0.005 # Ensure it stays on top
#     })

# # SORT descending (Best AUC first)
# file_candidates.sort(key=lambda x: x['auc'], reverse=True)

# print(f"\nFinal Sorted Order for Blending (Best to Worst):")
# for i, item in enumerate(file_candidates):
#     print(f"  Rank {i+1}: {item['name']} (Score: {item['auc']:.5f})")


# #  PREPARE PARAMS FOR h_blend
# # Define the Aggressive Weights as requested
# # [Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7]
# base_subwts = [20, 5, 2, -1, -4, -8, -14]

# # Handle edge case: if fewer than 7 files exist (e.g. external missing), slice the weights
# current_subwts = base_subwts[:len(file_candidates)]

# # Colors for plotting
# colors_pal = ['tomato', 'darkmagenta', 'limegreen', 'royalblue', 'orange', 'teal', 'gold']

# subm_list = []
# for i, item in enumerate(file_candidates):
#     # Remove .csv extension for the 'name' parameter required by h_blend logic
#     clean_name = item['name'].replace('.csv', '')
    
#     subm_list.append({
#         'name': clean_name,
#         'weight': 1.0 / len(file_candidates), # Equal base weight (h_blend applies subwts on top)
#         'color': colors_pal[i % len(colors_pal)]
#     })

# params = {
#     'path': './', # All files are now in the current directory
#     'id_target': ['id', 'diagnosed_diabetes'], 
#     'type_sort': ['asc/desc', 0.30, 0.70],     
#     'subwts': [w/200 for w in current_subwts], # Apply scaling /200
#     'subm': subm_list
# }


# #  RUN BLENDING & PLOTTING
# # This will generate 'submission.csv' and show the Bokeh plots
# print("\nRunning h_blend...")
# df_final = h_blend(
#     params, 
#     details=True,        # Show dataframe head
#     subm='submission.csv' # Save final output
# )

# print(f"\nFinal Blended Submission Shape: {df_final.shape}")

