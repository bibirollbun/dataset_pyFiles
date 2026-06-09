import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance, cv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# The target
target = 'Fertilizer Name'
# Load the training set
X = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
y = X.pop(target)
# T=Load the testing set
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', index_col='id')


# Encode labels if necessary
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

for cat_feat in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    X[cat_feat] = le.fit_transform(X[cat_feat])
    test_data[cat_feat] = le.transform(test_data[cat_feat])


# Split into train and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.3, random_state=42)


# Define the classifier
xgb_best_params = {
   'n_estimators': 3500,
    'max_depth':12,
    'subsample': 0.9,
    'colsample_bytree':0.5,
    'learning_rate':0.03,
    'gamma':0.5,
    'max_delta_step': 5,
    'early_stopping_rounds':50,
    # 'objective':'multi:softprob',
    # 'objective':'rank:map',
    'objective': 'multi:softmax',
    'enable_categorical':True,
    'tree_method':'hist',
    'device':'cuda',
    'reg_alpha':2.7,
    'reg_lambda':1.4,
    'num_parallel_tree': 5,
    # 'disable_default_eval_metric': True,    
    # 'eval_metrics': 'accuracy',
    # 'verbose': 100
}

clf = XGBClassifier(**xgb_best_params)


# Train a classifier
clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=50)


# Predict probabilities
y_probs = clf.predict_proba(X_valid)
y_probs[-5:]


# Get top-k predictions
def get_top_k_predictions(probs, k):
    return np.argsort(probs, axis=1)[:, -k:][:, ::-1]


# Single-label MAP@K
def mapk_single_label(y_true, y_pred, k=3):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)[:, :k]
    matches = (y_true.reshape(-1, 1) == y_pred)
    ranks = np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, np.inf)
    return np.mean(ranks ** -1)


# Multi-label MAP@K (each instance has one label in a list)
def apk(actual, predicted, k=10):
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score = 0.0
    num_hits = 0
    seen = set()
    actual_set = set(actual)
    for i, p in enumerate(predicted):
        if p in actual_set and p not in seen:
            num_hits += 1
            score += num_hits / (i + 1)
            seen.add(p)
    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])


# Evaluate MAP@K for k = 1 to k_max
k_max = 8
k_values = range(1, k_max)
mapk_single_scores = []
mapk_multi_scores = []

for k in k_values:
    top_k_preds = get_top_k_predictions(y_probs, k)
    mapk_single_scores.append(mapk_single_label(y_valid, top_k_preds, k))
    mapk_multi_scores.append(mapk(y_valid, top_k_preds, k))


# Plot the results
plt.figure(figsize=(8, 5))
ax = plt.plot(k_values, mapk_single_scores, marker='o', label='Single-label MAP@K')
ax = plt.plot(k_values, mapk_multi_scores, marker='s', label='Multi-label MAP@K', linestyle='--')
plt.title(f'MAP@k Comparison from 1 to {k_max}')
plt.xlabel('K')
plt.ylabel('MAP@K Score')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




test_proba = clf.predict_proba(test_data)
test_proba


test_proba[:1]


test_proba.shape


preds = np.argsort(test_proba, axis=1)[:, ::-1]
preds


test_top_3 = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
test_top_3


test_top_3_names = label_encoder.inverse_transform(test_top_3.ravel())
test_3_picks = test_top_3_names.reshape(test_top_3.shape)

test_3_picks


preds_df = pd.DataFrame({
    'id': test_data.index,
    'Fertilizer Name': [' '.join(preds) for preds in test_3_picks]
})

preds_df.head(10)


preds_df.to_csv('submission.csv', index=False)
print("Let's submit to the competition.")

