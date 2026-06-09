# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold # Key change for multi-class CV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score # Not directly MAP@K, but for individual AP
# import gc

# warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.model_selection._split') # Ignore UserWarning about stratification for KFold

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


train['const'] = 1
test['const'] = 1
original['const'] = 1

train = train.astype(str)
test = test.astype(str)
original = original.astype(str)

train = train.drop(columns=['id'])
# frames = [train, original]

# train = pd.concat(frames, ignore_index=False)


train.info()


X = train.drop(columns=['Fertilizer Name'])
y = train['Fertilizer Name']

# X.head()

le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Store the classes for later mapping: le.classes_ will be like ['dengue', 'flu', 'malaria', 'typhoid', 'zika']
# You can access the prognosis names later using le.inverse_transform([0, 1, 2]) etc.
all_fertilizer = le.classes_
num_classes = len(all_fertilizer)
print(f"Encoded fertilizers: {all_fertilizer}")
print(f"All possible fertilizers (and their encoded values): {list(zip(all_fertilizer, range(num_classes)))}")
print(f"Shape of y_encoded: {y_encoded.shape}")


# num_feats = ['const']
cat_feats = ['Soil Type','Crop Type','Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous','const']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])

xgb_classifier = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    eval_metric='mlogloss',
    # use_label_encoder=False,
    random_state=42,
    n_estimators=100,
    max_depth=16,
    subsample=0.61,
    colsample_bytree=0.3,
    gamma=0.26,
    reg_alpha=.15,
    reg_lambda=.67,
    learning_rate=0.07,
    # tree_method='hist',  # Use 'gpu_hist' for GPU acceleration
    # device='cuda'  # Use 'gpu' if you have a compatible GPU and want to use it
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_classifier)
])

n_splits = 5

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)



map_at_3_scores = []

print("\nStarting cross-validation...")
for fold, (train_index, val_index) in enumerate(kf.split(X, y_encoded)):
    print(f"Fold {fold + 1}/{n_splits}")
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    # augmentation here
    # Octuple the original data to give it more weight

    X_orig = original.drop(columns=['Fertilizer Name'])
    y_orig = le.transform(original['Fertilizer Name'])

    X_orig_oct = pd.concat([X_orig] * 2, ignore_index=True)
    y_orig_oct = np.concatenate([y_orig] * 2)

    X_train_aug = pd.concat([X_train, X_orig_oct], ignore_index=True)
    y_train_aug = np.concatenate([y_train, y_orig_oct])
    
    pipeline.fit(X_train_aug, y_train_aug)
    
    # Get predicted probabilities
    y_pred_proba = pipeline.predict_proba(X_val)

    current_fold_map_at_3 = 0
    num_val_samples = y_val.shape[0]
    
    if num_val_samples > 0:
        ap_scores_for_fold = []
        for i in range(num_val_samples):
            true_label_idx = y_val[i]

            ranked_pred_indeces = np.argsort(y_pred_proba[i])[::-1]
            top_3_pred_indeces = ranked_pred_indeces[:3]

            relevant_hits = 0
            sum_precision = 0

            if true_label_idx in top_3_pred_indeces:
                rank_of_true_label = np.where(top_3_pred_indeces == true_label_idx)[0][0]
                sum_precision = 1.0 / (rank_of_true_label + 1)
                ap_at_k = sum_precision
            else:
                ap_at_k = 0

            ap_scores_for_fold.append(ap_at_k)

        current_fold_map_at_3 = np.mean(ap_scores_for_fold)
        map_at_3_scores.append(current_fold_map_at_3)
        print(f"Fold {fold + 1} MAP@3: {current_fold_map_at_3:.4f}")
    else:
        print(f"Fold {fold + 1} has no validation samples, skipping MAP@3 calculation.")

print(f"\nAverage MAP@3 across {n_splits} folds: {np.mean(map_at_3_scores):.4f}")
print(f"Standard deviation of MAP@3 across folds: {np.std(map_at_3_scores):.4f}")

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_classifier)
])

final_pipeline.fit(X, y_encoded)

# X_new_test = test[[col for col in X.columns if col != 'const']]
X_new_test = test[X.columns]
test_ids = test['id']

final_test_probas = final_pipeline.predict_proba(X_new_test)

final_output = pd.DataFrame({'id': test_ids})

def get_top_k_fert_string_multi_class(intance_probas, k, fertilizer_names_map):
    sorted_indices = np.argsort(intance_probas)[::-1]
    top_k_indices = sorted_indices[:k]

    return ' '.join(fertilizer_names_map[i] for i in top_k_indices)

final_output['Fertilizer Name'] = [get_top_k_fert_string_multi_class(row, 3, all_fertilizer) for row in final_test_probas]
print("\nSample final Output for Submission:")
print(final_output.head())


final_output.to_csv('submission.csv',index = False)

