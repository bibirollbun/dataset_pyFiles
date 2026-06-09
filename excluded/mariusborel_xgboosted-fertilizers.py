import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, cv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# verify the versions of my tools
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'xgboost version: {xgb.__version__}')

include_ext_data = True


# The target
target = 'Fertilizer Name'
# Load the training set
X = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
# Load external data
X_ext = pd.read_csv('/kaggle/input/d/irakozekelly/fertilizer-prediction/Fertilizer Prediction.csv')
# Load external 2 data
X_ext_2 = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
# Load the testing set
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', index_col='id')


[d.shape for d in [X, X_ext, X_ext_2]]


# Multiply and shuffle the external dataset
X_ext = pd.concat([X_ext]*2, axis=0).sample(frac=1)
# Name the columns in the external datasets
X_ext.columns = X.columns
# Load external 2 data
X_ext_2.columns = X.columns

# Decide if external data should be included
if include_ext_data:
    X = pd.concat([X, X_ext], ignore_index=True)
    y = X.pop(target)
else:
    X = X
    y = X.pop(target)
# Get the ext_target
y_ext = X_ext.pop(target)
y_ext_2 = X_ext_2.pop(target)


y_ext_2.value_counts()


X.tail(4)


test_data.head(4)


pd.pivot_table(data=X, index=['Crop Type'], columns=y, aggfunc='count')['Soil Type'].style.format('{:}').background_gradient(cmap='Blues', axis=1)


pd.pivot_table(data=X, index=['Soil Type'], columns=y, aggfunc='count')['Crop Type'].style.format('{:}').background_gradient(cmap='Blues', axis=1)


pd.pivot_table(data=X, index=['Crop Type'], columns=['Soil Type'], aggfunc='count')['Nitrogen'].style.format('{:}').background_gradient(cmap='Blues', axis=1)


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


# # Encode labels if necessary
# target_encoder = LabelEncoder()
# y_encoded = target_encoder.fit_transform(y)
# y_ext_2_enc = target_encoder.transform(y_ext_2)

# # Covert num_features from int64 to lower memory int8
# for col in X.select_dtypes('int64').columns:
#     # train data
#     X[col] = X[col].astype('category')
#     # external data
#     X_ext[col] = X_ext[col].astype('category')
#     # external 2 data
#     X_ext_2[col] = X_ext_2[col].astype('category')
#     # test data
#     test_data[col] = test_data[col].astype('category')
    
# # Encode the cat_features   
# for cat_feat in ['Soil Type', 'Crop Type']:
#     cat_le = LabelEncoder()
#     # train data
#     X[cat_feat] = cat_le.fit_transform(X[cat_feat])
#     X[cat_feat] = X[cat_feat].astype('category')
#     # external data
#     X_ext[cat_feat] = cat_le.fit_transform(X_ext[cat_feat])
#     X_ext[cat_feat] = X_ext[cat_feat].astype('category')
#     # external 2 data
#     X_ext_2[cat_feat] = cat_le.fit_transform(X_ext_2[cat_feat])
#     X_ext_2[cat_feat] = X_ext_2[cat_feat].astype('category')
#     # test data
#     test_data[cat_feat] = cat_le.transform(test_data[cat_feat])
#     test_data[cat_feat] = test_data[cat_feat].astype('category')


# Encode labels if necessary
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
y_ext_2_encoded = target_encoder.transform(y_ext_2)

# Covert num_features from int64 to lower memory int8
for col in X.select_dtypes('int64').columns:
    # train data
    X[col] = X[col].astype('int8')
    # external data
    X_ext[col] = X_ext[col].astype('int8')
    # external 2 data
    X_ext_2[col] = X_ext_2[col].astype('int8')
    # test data
    test_data[col] = test_data[col].astype('int8')
    
# Encode the cat_features   
for cat_feat in ['Soil Type', 'Crop Type']:
    cat_le = LabelEncoder()
    # train data
    X[cat_feat] = cat_le.fit_transform(X[cat_feat])
    X[cat_feat] = X[cat_feat].astype('category')
    # external data
    X_ext[cat_feat] = cat_le.fit_transform(X_ext[cat_feat])
    X_ext[cat_feat] = X_ext[cat_feat].astype('category')
    # external 2 data
    X_ext_2[cat_feat] = cat_le.fit_transform(X_ext_2[cat_feat])
    X_ext_2[cat_feat] = X_ext_2[cat_feat].astype('category')
    # test data
    test_data[cat_feat] = cat_le.transform(test_data[cat_feat])
    test_data[cat_feat] = test_data[cat_feat].astype('category')


# Split into train and test sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded,
                                                      test_size=0.3,
                                                      random_state=4)

[d.shape for d in [X_train, X_valid, y_train, y_valid]]


# Define the classifier
xgb_best_params = {
    'num_class': 7,
    'n_estimators': 5000,
    'max_depth': 16,
    'subsample': 0.7,
    'colsample_bytree': 0.4,
    # 'colsample_bynode': 0.5,
    'min_child_weight': 5,
    'learning_rate': 0.03, 
    'gamma': 0.26,
    'max_delta_step': 5, # read more on 
    'max_bins': 16, # optima 32
    'early_stopping_rounds': 100,
    'objective': 'multi:softprob',
    'enable_categorical': True,
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'reg_alpha': 3,
    'reg_lambda': 1.4,
    'n_jobs': -1,
    'num_parallel_tree': 5,
    'enable_categorical': True
}


# n_splits= 3
# my_spliter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# color = 90
# for f, (tr_ind, va_ind) in enumerate(my_spliter.split(X, y_encoded), 1):
#     color+=1
#     print(38*f'\033[{color}m:\033[0m')
#     clf = XGBClassifier(**xgb_best_params)
#     X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
#     y_tr, y_va = y_encoded[tr_ind], y_encoded[va_ind]
#     print(f'\n\033[{color}mğŸ�° Fitting Fold_{f}\033[0m\n')
#     clf.fit(X_tr, y_tr, 
#             eval_set=[(X_va, y_va)],
#             verbose=200)
#     clf.save_model(f'my_xgb_fold_{f}.json')
#     y_probs = clf.predict_proba(X_va)
#     top_3_preds = get_top_k_predictions(y_probs, 3)
#     map3 = mapk_single_label(y_va, top_3_preds, 3)
#     print('\n\033[{}mâš–ï¸� map3: {:.5}\n\033[0m'.format(color, map3))


# n_splits= 10
# my_spliter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# color = 90
# for f, (tr_ind, va_ind) in enumerate(my_spliter.split(X, y_encoded), 1):
#     if f == 5: # Limit the oof to odd fold to reduce run time
#         color+=1
#         print(38*f'\033[{color}m:\033[0m')
#         clf = XGBClassifier(**xgb_best_params)
#         X_tr, X_va = X.iloc[tr_ind], X.iloc[va_ind]
#         y_tr, y_va = y_encoded[tr_ind], y_encoded[va_ind]
#         print(f'\n\033[{color}mğŸ�° Fitting Fold_{f}\033[0m\n')
#         clf.fit(X_tr, y_tr, 
#                 eval_set=[(X_va, y_va)],
#                 verbose=200)
#         clf.save_model(f'my_xgb_fold_{f}.json')
#         y_probs = clf.predict_proba(X_va)
#         top_3_preds = get_top_k_predictions(y_probs, 3)
#         map3 = mapk_single_label(y_va, top_3_preds, 3)
#         print('\n\033[{}mâš–ï¸� map3: {:.5}\n\033[0m'.format(color, map3))


# try:
#     loaded_Clf_Fold_1 = xgb.XGBClassifier()
#     loaded_Clf_Fold_1.load_model('my_xgb_fold_1.json')
# except:
#     pass

# try:
#     loaded_Clf_Fold_2 = xgb.XGBClassifier()
#     loaded_Clf_Fold_2.load_model('my_xgb_fold_2.json')
# except:
#     pass

# try:
#     loaded_Clf_Fold_3 = xgb.XGBClassifier()
#     loaded_Clf_Fold_3.load_model('my_xgb_fold_3.json')
# except:
#     pass

# try:
#     loaded_Clf_Fold_4 = xgb.XGBClassifier()
#     loaded_Clf_Fold_4.load_model('my_xgb_fold_4.json')
# except:
#     pass

# try:
#     loaded_Clf_Fold_5 = xgb.XGBClassifier()
#     loaded_Clf_Fold_5.load_model('my_xgb_fold_5.json')
# except:
#     pass


# n_splits=5
# import xgboost as xgb

# loaded_models = {}

# for i in range(1, n_splits + 1):
#     try:
#         model = xgb.XGBClassifier()
#         model.load_model(f'my_xgb_fold_{i}.json')
#         loaded_models[f'Fold_{i}'] = model
#     except:
#         pass  # Optionally, log the exception for debugging


# Define the classifier
clf_final = XGBClassifier(**xgb_best_params)

# Train a classifier
clf_final.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)],
              verbose=200)


# Predict probabilities
y_probs = clf_final.predict_proba(X_valid)
y_probs[:3]


# Evaluate MAP@K for k = 1 to k_max
k_max = y.nunique() + 1
k_values = range(1, k_max)
mapk_scores = []

for k in k_values:
    top_k_preds = get_top_k_predictions(y_probs, k)
    mapk_scores.append(mapk(y_valid, top_k_preds, k))


# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(k_values, mapk_scores, marker='o', linestyle='--')
# Annotate each point with its value
for k, score in zip(k_values, mapk_scores):
    if k!=3:
        plt.text(k, score, f'@{k}: {score:.4f}', 
                 ha='right', va='bottom', color='steelblue', size=12)
    else:
        plt.text(k, score, f'@{k}: {score:.4f}', ha='right', 
                 va='bottom', color='maroon', weight='bold', size=16)
# Add titles and labels
plt.title(f'MAP@k for k in range 1 to {k_max-1}', color='maroon', size=16, weight='bold')
plt.xlabel('@K')
plt.ylabel('MAP@K Score')
plt.xticks(k_values)
# plt.grid(True)
plt.tight_layout()
plt.show()


test_proba = clf.predict_proba(test_data)
test_proba[:1]


preds = np.argsort(test_proba, axis=1)[:, ::-1]
preds[:10]


test_top_3 = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
test_top_3


test_top_3_names = target_encoder.inverse_transform(test_top_3.ravel())
test_3_picks = test_top_3_names.reshape(test_top_3.shape)

test_3_picks


picks = pd.DataFrame(test_3_picks, columns=['First', 'Second', 'Thirth']).apply(lambda x: x)
picks


colors = ['grey', 'red', 'blue', 'orange']

plt.figure(figsize=(12,3))
for n, pick_rank in enumerate(picks.columns, start=1):
    plt.subplot(1, 3, n)
    sns.countplot(picks.sort_values(by=pick_rank), x=pick_rank, dodge=False)
    plt.xticks(rotation=90)
    plt.title(f'Count as {pick_rank} pick')
    if n!=1:
        plt.yticks([])
    plt.ylabel('')
plt.show()


# prep the submission dataframe
preds_df = pd.DataFrame({
    'id': test_data.index,
    'Fertilizer Name': [' '.join(preds) for preds in test_3_picks]
})

preds_df.head(10)


preds_df.to_csv('submission.csv', index=False)
print("Let's submit to the competition.")


# Create the figure and GridSpec layout
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, width_ratios=[3, 2])

# Decode the classes
target_classes = target_encoder.classes_

# First plot: Heatmap spanning both columns
ax0 = fig.add_subplot(gs[:, 0])
sns.heatmap(test_proba, cmap='copper_r', ax=ax0)
ax0.set_xticks(ticks=np.arange(7) + 0.5)
ax0.set_xticklabels(target_classes, rotation=-45)
ax0.set_title('Predicted probabilities across the test set', color='maroon')

# Second plot: Sorted probabilities
ax1 = fig.add_subplot(gs[0, 1])
for i in range(1, 7):
    pd.Series(np.sort(test_proba, axis=1)[:, i]).plot(ax=ax1)
ax1.set_title('Sorted predicted probabilities for test set', color='maroon')

# Third plot: Max, Min, Mean probabilities
ax2 = fig.add_subplot(gs[1, 1])
pd.Series(np.max(test_proba, axis=1)).plot(ax=ax2, label='Max')
pd.Series(np.min(test_proba, axis=1)).plot(ax=ax2, label='Min')
pd.Series(np.mean(test_proba, axis=1)).plot(ax=ax2, color='darkgreen', label='Mean')
ax2.set_title('Scope of Max, Min, Mean probabilities', color='maroon')
ax2.legend()

plt.tight_layout()
plt.show()


# original = pd.concat([original] * 7, axis=0)


# original.reset_index(drop=True, inplace=True)


# shuffled_df = original.sample(frac=1).reset_index(drop=True)


