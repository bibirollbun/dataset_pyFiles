!pip install optuna --quiet


# Author: Aaron Isom
# Kaggle Playground-Series-S5e6 - Predicting Optimal Fertilizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
import cupy as cp

from scipy.stats import rankdata
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
tune = False # Toggle for Optuna tuning and Final Submission


# Mean Average Precision @ 3 (MAP@3)
def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    actual: array of true label indices (ints)
    predicted: list of lists, each list is the k predicted indices for each sample
    """
    def apk(a, p, k):
        if a in p[:k]:
            return 1.0 / (p[:k].index(a) + 1)
        else:
            return 0.0
    # Ensure predicted is a list of lists
    return np.mean([apk(a, list(p), k) for a, p in zip(actual, predicted)])

# Optuna objective for XGBoost
def objective(trial):
    params = {
        "num_class": len(np.unique(y)),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        #"min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000, step=100),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)
    }

    model = XGBClassifier(**params, objective='multi:softprob', eval_metric='mlogloss', random_state=42, device='cuda', 
                          enable_categorical=True, tree_method='hist')
    model.fit(X, y)
    probs = model.predict_proba(X)
    
    top3 = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    top3 = top3.tolist()  # Convert to list of lists

    return mapk(y, top3, k=3)


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

train = pd.concat([train, original], axis=0, ignore_index=True)
train['Fertilizer Name'].value_counts(normalize=True)

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


print("Train shape:", train.shape)
print(train.info())

print("\nTest shape:", test.shape)
print(test.info())

print("\nSubmission shape:", test.shape)
print(submission.info())

print("\nNulls:\n", train.isnull().sum())
print("\nDuplicates:", train.duplicated().sum())
print("\nData Types:\n", train.dtypes)

X = train.drop(columns=['Fertilizer Name'])
y = train['Fertilizer Name']

# Label Encoding for all categorical features
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
# num_features = X.select_dtypes(include=[np.number]).columns.tolist()

for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])

le = LabelEncoder()
y = le.fit_transform(y)

X = cp.asarray(X) # CuPy (for GPU) or NumPy (for CPU)


if tune:
    # Optuna Study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=5400, show_progress_bar=True)
    best_params = study.best_trial.params
    print('Best Parameters:', best_params)
    print('Best Trial:', study.best_trial)

else:
    # Best tuned params
    #best_params =  {'max_depth': 9, 'learning_rate': 0.04756857658387681, 'subsample': 0.6838006702792171, 'colsample_bytree': 0.32432445096116524, 
    #                'min_child_weight': 7, 'max_delta_step': 3, 'n_estimators': 4500, 
    #                'gamma': 0.9424141997972995, 'reg_alpha': 2.53361391876808, 'reg_lambda': 1.7064302359889767}
    # Score: 0.349
    
    best_params = {'max_depth': 7, 'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.4, 
                   'max_delta_step': 4, 'n_estimators': 15000, 'gamma': 0.26, 
                   'reg_alpha': 2.7, 'reg_lambda': 1.4}
    # Score: 0.359
    
# Final XGBoost model
final_model = XGBClassifier(**best_params, objective='multi:softprob', eval_metric='mlogloss', random_state=42, device='cuda', enable_categorical=True, tree_method='hist')
#final_cv_score = cross_val_score(final_model, X, y, cv=3, scoring='neg_log_loss')
final_model.fit(X, y)

#print(f"Final Model CV=5 Score: {final_cv_score}")
#print(f"Final Model Mean CV=5 Score: {final_cv_score.mean():.4f} ± {final_cv_score.std():.4f}")


#Try Rank Averaging
probs = final_model.predict_proba(test)
classes = sorted(train['Fertilizer Name'].unique())
priors = train['Fertilizer Name'].value_counts(normalize=True).reindex(classes).values
priors_matrix = np.tile(priors, (probs.shape[0], 1))
model_ranks = np.apply_along_axis(rankdata, 1, probs)
prior_ranks = np.apply_along_axis(rankdata, 1, priors_matrix)
model_ranks = model_ranks / model_ranks.max(axis=1, keepdims=True)
prior_ranks = prior_ranks / prior_ranks.max(axis=1, keepdims=True)
weight = 0.8
rank_avg = weight * model_ranks + (1 - weight) * prior_ranks
rank_avg = rank_avg / rank_avg.sum(axis=1, keepdims=True)
top3_idx = np.argsort(-rank_avg, axis=1)[:, :3]
top3_names = le.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_names]
})
submission.to_csv('submission.csv', index=False)
display(submission.head())
print('Submission file saved.')




