# ======================================================
# ðŸš€ Super-Powered Blending for 3 Submissions (Optuna)
# ======================================================

import pandas as pd
import numpy as np
import optuna

# --- Step 1: define your files and their public RMSEs ---
files = [
    '/kaggle/input/accident-prediction-blending/submission_0.05538.csv',
    '/kaggle/input/accident-prediction-blending/submission_0.05538.csv',
    '/kaggle/input/accident-prediction-blending/submission_0.05538.csv',
]
public_rmse = [0.05538, 0.05538, 0.05538]

# --- Step 2: read all submissions ---
subs = [pd.read_csv(f) for f in files]
ids = subs[0]['id']
preds = np.stack([s['accident_risk'].values for s in subs], axis=1)

# --- Step 3: use public RMSEs to get initial confidence weights ---
conf = np.exp(-np.array(public_rmse))
conf = conf / conf.sum()
print("Initial weights (from RMSE):", conf.round(4))

# --- Step 4: define Optuna objective for fine-tuning ---
def objective(trial):
    w = np.array([
        trial.suggest_float('w1', 0, 1),
        trial.suggest_float('w2', 0, 1),
        trial.suggest_float('w3', 0, 1)
    ])
    w /= w.sum() + 1e-9
    blended = np.dot(preds, w)
    # pseudo score: penalize distance from best RMSE + encourage smoothness
    pseudo_score = np.sqrt(np.sum((w - conf)**2)) + 0.1 * np.std(w)
    return pseudo_score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=214, show_progress_bar=False)

best_weights = np.array([
    study.best_params['w1'],
    study.best_params['w2'],
    study.best_params['w3']
])
best_weights /= best_weights.sum()

print("Optimized Weights:", best_weights.round(4))

# --- Step 5: create final blended predictions ---
final_preds = np.dot(preds, best_weights)
final = pd.DataFrame({'id': ids, 'accident_risk': final_preds})
final.to_csv('submission.csv', index=False)

print("âœ… Final blended file saved as super_blend.csv")





