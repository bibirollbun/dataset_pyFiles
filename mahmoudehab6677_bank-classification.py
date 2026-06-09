import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')


class Config:
    RANDOM_STATE = 42
    N_SPLITS = 5
    N_REPEATS = 2
    TARGET_COL = 'y'


df_train_comp = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
df_orig = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')
df_orig[Config.TARGET_COL] = df_orig[Config.TARGET_COL].map({'no':0,'yes':1})
df_train = pd.concat([df_train_comp, df_orig], ignore_index=True).drop_duplicates()


def feature_engineer(df):
    df_copy = df.copy()
    # Logs
    df_copy['balance_log'] = np.sign(df_copy['balance']) * np.log1p(np.abs(df_copy['balance']))
    df_copy['duration_log'] = np.log1p(df_copy['duration'])
    # Group stats
    df_copy['avg_balance_by_job'] = df_copy.groupby('job')['balance'].transform('mean')
    df_copy['median_duration_by_education'] = df_copy.groupby('education')['duration'].transform('median')
    # Interactions
    df_copy['job_education_interaction'] = df_copy['job'].astype(str) + '_' + df_copy['education'].astype(str)
    df_copy['poutcome_contact_interaction'] = df_copy['poutcome'].astype(str) + '_' + df_copy['contact'].astype(str)
    df_copy['duration_per_campaign'] = df_copy['duration'] / (df_copy['campaign'] + 1e-6)
    df_copy['was_not_contacted'] = (df_copy['pdays'] == -1).astype(int)
    # Numeric interactions
    df_copy['balance_per_age'] = df_copy['balance'] / (df_copy['age'] + 1e-6)
    df_copy['duration_x_age'] = df_copy['duration'] * df_copy['age']
    df_copy['campaign_x_previous'] = df_copy['campaign'] * (df_copy['previous'] + 1)
    return df_copy

df_train = feature_engineer(df_train)
df_test = feature_engineer(df_test)

y = df_train[Config.TARGET_COL]
X = df_train.drop(Config.TARGET_COL, axis=1)
X_test = df_test.copy()
for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X.columns]


# CatBoost
X_cat = X.astype(str)
X_test_cat = X_test.astype(str)
cat_features_indices_cat = list(range(X_cat.shape[1]))

# LightGBM & XGBoost
X_tree = X.copy()
X_test_tree = X_test.copy()
cat_features_tree = X_tree.select_dtypes(include=['object']).columns.tolist()
for col in cat_features_tree:
    le = LabelEncoder()
    X_tree[col] = le.fit_transform(X_tree[col])
    X_test_tree[col] = X_test_tree[col].map(lambda s: s if s in le.classes_ else '<unknown>')
    le.classes_ = np.append(le.classes_, '<unknown>')
    X_test_tree[col] = le.transform(X_test_tree[col])

# Neural Network
cat_features_nn = X.select_dtypes(include=['object','category']).columns.tolist()
num_features_nn = X.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X_num_nn = scaler.fit_transform(X[num_features_nn])
X_test_num_nn = scaler.transform(X_test[num_features_nn])

X_cat_nn, X_test_cat_nn, cat_dims = pd.DataFrame(), pd.DataFrame(), []
for col in cat_features_nn:
    le = LabelEncoder()
    X_cat_nn[col] = le.fit_transform(X[col])
    test_col_mapped = X_test[col].map(lambda s: s if s in le.classes_ else '<unknown>')
    le.classes_ = np.append(le.classes_, '<unknown>')
    X_test_cat_nn[col] = le.transform(test_col_mapped)
    cat_dims.append(len(le.classes_))


cat_model = CatBoostClassifier(
    n_estimators=4000, learning_rate=0.025, depth=8, l2_leaf_reg=3.5,
    loss_function='CrossEntropy', eval_metric='AUC', task_type='GPU',
    random_seed=Config.RANDOM_STATE, cat_features=cat_features_indices_cat,
    verbose=0, allow_writing_files=False
)
lgb_model = lgb.LGBMClassifier(
    random_state=Config.RANDOM_STATE, objective='binary', metric='auc',
    n_estimators=4000, learning_rate=0.02, num_leaves=80, max_depth=12,
    subsample=0.85, colsample_bytree=0.45, reg_alpha=0.15, reg_lambda=0.15, n_jobs=-1
)
xgb_model = XGBClassifier(
    n_estimators=4000, learning_rate=0.02, max_depth=10, subsample=0.8, colsample_bytree=0.5,
    objective='binary:logistic', eval_metric='auc', random_state=Config.RANDOM_STATE,
    tree_method='hist', enable_categorical=True, n_jobs=-1
)

def build_nn_model(cat_dims, num_features_len):
    cat_inputs, embeddings = [], []
    for i, dim in enumerate(cat_dims):
        input_layer = layers.Input(shape=(1,), name=f'cat_input_{i}')
        embedding_size = min(50, (dim + 1)//2)
        embedding_layer = layers.Embedding(dim, embedding_size)(input_layer)
        embedding_layer = layers.Flatten()(embedding_layer)
        cat_inputs.append(input_layer)
        embeddings.append(embedding_layer)
    num_input = layers.Input(shape=(num_features_len,), name='num_input')
    x = layers.Concatenate()(embeddings + [num_input])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=cat_inputs + [num_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-4),
                  loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return model


rskf = RepeatedStratifiedKFold(n_splits=Config.N_SPLITS, n_repeats=Config.N_REPEATS, random_state=Config.RANDOM_STATE)

oof_cat, test_preds_cat = np.zeros(len(X)), np.zeros(len(X_test))
oof_lgb, test_preds_lgb = np.zeros(len(X)), np.zeros(len(X_test))
oof_xgb, test_preds_xgb = np.zeros(len(X)), np.zeros(len(X_test))
oof_nn, test_preds_nn = np.zeros(len(X)), np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
    print(f"--- FOLD {fold+1}/{Config.N_SPLITS*Config.N_REPEATS} ---")
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost
    cb = clone(cat_model)
    cb.fit(X_cat.iloc[train_idx], y_train, eval_set=[(X_cat.iloc[val_idx], y_val)],
           early_stopping_rounds=200, verbose=0)
    oof_cat[val_idx] += cb.predict_proba(X_cat.iloc[val_idx])[:,1]
    test_preds_cat += cb.predict_proba(X_test_cat)[:,1]

    # LightGBM
    lgbm = clone(lgb_model)
    lgbm.fit(X_tree.iloc[train_idx], y_train, eval_set=[(X_tree.iloc[val_idx], y_val)],
             callbacks=[lgb.early_stopping(200, verbose=False)],
             categorical_feature=cat_features_tree)
    oof_lgb[val_idx] += lgbm.predict_proba(X_tree.iloc[val_idx])[:,1]
    test_preds_lgb += lgbm.predict_proba(X_test_tree)[:,1]

    # XGBoost
    xgb = clone(xgb_model)
    xgb.fit(X_tree.iloc[train_idx], y_train, eval_set=[(X_tree.iloc[val_idx], y_val)],
            early_stopping_rounds=200, verbose=False)
    oof_xgb[val_idx] += xgb.predict_proba(X_tree.iloc[val_idx])[:,1]
    test_preds_xgb += xgb.predict_proba(X_test_tree)[:,1]

    # Neural Network
    X_train_cat_nn_fold = [X_cat_nn.iloc[train_idx,i] for i in range(X_cat_nn.shape[1])]
    X_val_cat_nn_fold = [X_cat_nn.iloc[val_idx,i] for i in range(X_cat_nn.shape[1])]
    X_train_num_nn_fold, X_val_num_nn_fold = X_num_nn[train_idx], X_num_nn[val_idx]
    nn_model = build_nn_model(cat_dims, len(num_features_nn))
    es = callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_auc', mode='max')
    lr = callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    nn_model.fit([*X_train_cat_nn_fold, X_train_num_nn_fold], y_train,
                 validation_data=([*X_val_cat_nn_fold, X_val_num_nn_fold], y_val),
                 epochs=100, batch_size=1024, verbose=0, callbacks=[es, lr])
    oof_nn[val_idx] += nn_model.predict([*X_val_cat_nn_fold, X_val_num_nn_fold]).flatten()
    test_preds_nn += nn_model.predict([*[X_test_cat_nn.iloc[:,i] for i in range(X_test_cat_nn.shape[1])], X_test_num_nn]).flatten()

# Average predictions
total_folds = Config.N_SPLITS * Config.N_REPEATS
for oof in [oof_cat, oof_lgb, oof_xgb, oof_nn]:
    oof /= Config.N_REPEATS
for test_preds in [test_preds_cat, test_preds_lgb, test_preds_xgb, test_preds_nn]:
    test_preds /= total_folds


oof_stack = np.vstack([oof_cat, oof_lgb, oof_xgb, oof_nn]).T
test_stack = np.vstack([test_preds_cat, test_preds_lgb, test_preds_xgb, test_preds_nn]).T

meta_model = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=Config.RANDOM_STATE)
meta_model.fit(oof_stack, y)
final_preds_stack = meta_model.predict_proba(test_stack)[:,1]
print(f"Stacked model OOF AUC: {roc_auc_score(y, meta_model.predict_proba(oof_stack)[:,1]):.6f}")



# Stacked submission
df_submission['y'] = final_preds_stack
df_submission.to_csv("submission_4model_stacked.csv", index=False)

# Rank averaging as alternative
final_preds_ranked = (rankdata(test_preds_cat) + rankdata(test_preds_lgb) + rankdata(test_preds_xgb) + rankdata(test_preds_nn))
df_submission['y'] = (final_preds_ranked - final_preds_ranked.min()) / (final_preds_ranked.max() - final_preds_ranked.min())
df_submission.to_csv("submission_4model_ranked.csv", index=False)

print("\nStacked and rank average submissions created.")
df_submission.head()

