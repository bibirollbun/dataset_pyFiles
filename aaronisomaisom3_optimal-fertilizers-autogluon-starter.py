!pip install -U autogluon --quiet


# Author: Aaron Isom
# Kaggle Playground-Series-S5e6 - Predicting Optimal Fertilizers
# AutoGluon Tabular Baseline for Optimal Fertilizers (MAP@3)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from autogluon.core.metrics import make_scorer
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# Mean Average Precision @3 (MAP@3)
def map3(y_true, y_pred_proba):
    if not isinstance(y_pred_proba, np.ndarray):
        y_pred_proba = np.asarray(y_pred_proba)
    
    top3 = np.argsort(-y_pred_proba, axis=1)[:, :3]
    y_true = np.array(y_true)
    
    def apk(true_label, pred_labels, k=3):
        return int(true_label in pred_labels[:k]) / min(1, k)
        
    scores = [apk(t, p, k=3) for t, p in zip(y_true, top3)]
    return np.mean(scores)

map3_score = make_scorer(name='map3', score_func=map3, optimum=1, needs_proba=True, greater_is_better=True)


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

train = pd.concat([train, original], axis=0, ignore_index=True)
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


target = 'Fertilizer Name'
train_data, val_data = train_test_split(train, test_size=0.25, stratify=train[target], random_state=42)

# Train AutoGluon TabularPredictor
predictor = TabularPredictor(label=target, eval_metric=map3_score, problem_type='multiclass').fit(train_data=train_data, time_limit=3600, 
                                                                                                  presets='best_quality', verbosity=1, 
                                                                                                  ag_args_fit={'num_gpus': 1}, auto_stack=True)
predictor.leaderboard(val_data, silent=False).style.background_gradient(subset=['score_val'], cmap='RdYlGn')

# Get the best model
best_model = predictor.model_best
print('Best model:', best_model)

importance = predictor.feature_importance(train_data)
importance_sorted = importance.sort_values('importance', ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(importance_sorted.index[::-1], importance_sorted['importance'][::-1]) 
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('AutoGluon Feature Importance')
plt.tight_layout()
plt.show()

# Final prediction on test set
pred_proba_test = predictor.predict_proba(test, model=best_model)
class_labels = list(pred_proba_test.columns)
top_3_test = np.argsort(-pred_proba_test.values, axis=1)[:, :3]
top_3_labels_test = [[class_labels[idx] for idx in row] for row in top_3_test]

submission[target] = [' '.join(row) for row in top_3_labels_test]
submission.to_csv('submission.csv', index=False)
print('Submission file saved.')
display(submission)

