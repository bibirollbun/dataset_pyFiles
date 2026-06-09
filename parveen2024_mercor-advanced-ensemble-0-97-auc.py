# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Final Pipeline (v10.1 - With Optuna Hyperparameter Tuning, Fully Online)

# STEP 1: CONFIG & SETUP
# =======================================
import os
import gc
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("Installing required libraries...")
!pip install sentence-transformers catboost xgboost optuna --quiet

import optuna
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
INPUT_DIR = "/kaggle/input/mercor-ai-detection"
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("Setup complete.")

# STEP 2: DATA LOADING & PREP
# =======================================
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
train['text'] = (train['topic'].fillna("") + " " + train['answer'].fillna("")).str.replace(r'\s+', ' ', regex=True).str.strip()
test['text']  = (test['topic'].fillna("") + " " + test['answer'].fillna("")).str.replace(r'\s+', ' ', regex=True).str.strip()
y = train['is_cheating'].values
print("Data loaded.")

# STEP 3: FEATURE ENGINEERING
# =========================================================
print("Generating features...")
embedding_model = SentenceTransformer(MODEL_NAME, device='cuda')
X_embed_train = embedding_model.encode(train['text'].tolist(), show_progress_bar=True)
X_embed_test = embedding_model.encode(test['text'].tolist(), show_progress_bar=True)

word_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=100000, sublinear_tf=True)
svd_word = TruncatedSVD(n_components=128, random_state=SEED)
X_word_svd_train = svd_word.fit_transform(word_vectorizer.fit_transform(train['text']))
X_word_svd_test = svd_word.transform(word_vectorizer.transform(test['text']))

char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,6), max_features=100000, sublinear_tf=True)
svd_char = TruncatedSVD(n_components=128, random_state=SEED)
X_char_svd_train = svd_char.fit_transform(char_vectorizer.fit_transform(train['text']))
X_char_svd_test = svd_char.transform(char_vectorizer.transform(test['text']))

X_dense_full_train = np.hstack([X_embed_train, X_word_svd_train, X_char_svd_train])
X_dense_full_test  = np.hstack([X_embed_test, X_word_svd_test, X_char_svd_test])
scaler = StandardScaler()
X_dense_full_train = scaler.fit_transform(X_dense_full_train)
X_dense_full_test = scaler.transform(X_dense_full_test)
print("Features generated and scaled.")

# STEP 4: HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# =========================================================
print("--- Starting Optuna hyperparameter search for CatBoost ---")

def objective(trial):
    params = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bootstrap_type': 'Bernoulli',
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'random_seed': SEED, 'verbose': 0, 'eval_metric': 'AUC', 'task_type': 'GPU'
    }
    model = CatBoostClassifier(**params)
    skf_optuna = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    score = cross_val_score(model, X_dense_full_train, y, cv=skf_optuna, scoring='roc_auc').mean()
    return score

# For a full search to get the best score, increase n_trials to 100+ (this will take several hours)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=60)

best_cat_params = study.best_params
print(f"Optuna found best AUC: {study.best_value:.5f}")
print("Best CatBoost parameters found:", best_cat_params)

# STEP 5: FINAL MODEL TRAINING
# =========================================================
print("--- Training final model with tuned parameters ---")
final_cat_params = {
    'iterations': 4000, 'random_seed': SEED, 'verbose': 0, 
    'eval_metric': 'AUC', 'task_type': 'GPU'
}
final_cat_params.update(best_cat_params)
if 'subsample' in final_cat_params:
    final_cat_params['bootstrap_type'] = 'Bernoulli'

cat_tuned = CatBoostClassifier(**final_cat_params)

# We will train on all data for the final submission model
cat_tuned.fit(X_dense_full_train, y, early_stopping_rounds=50, verbose=0)
final_predictions = cat_tuned.predict_proba(X_dense_full_test)[:, 1]

# STEP 6: SUBMISSION
# ======================================
submission = pd.DataFrame({"id": test['id'], "is_cheating": final_predictions})
sub_path = os.path.join(OUT_DIR, "submission.csv") # Name the final file submission.csv
submission.to_csv(sub_path, index=False)
print(f"\n Submission file saved to: {sub_path}")

