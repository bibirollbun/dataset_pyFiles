# Standard library
import gc
import os

# Data handling
import cudf
import cupy
import numpy as np
import pandas as pd

# Machine learning & preprocessing
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Deep learning (TensorFlow / Keras)
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model


# Reading Data
def read_data(path):
    df = pd.read_parquet(path)
    df['customer_ID'] = df['customer_ID'].str[-16:].apply(lambda x: int(x, 16)).astype('int64')
    df['S_2'] = pd.to_datetime(df['S_2'])
    df = df.fillna(0)
    print(f"Data shape: {df.shape}")
    return df


def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    # ————————————————————————————
    # 1) Prep
    all_f = [c for c in df.columns if c not in ['customer_ID','S_2']]
    cat   = [
        "B_30","B_38","D_114","D_116","D_117",
        "D_120","D_126","D_63","D_64","D_66","D_68"
    ]
    num   = [c for c in all_f if c not in cat]
    
    # Sort once to enable group.tail(k) without re-sorting
    df.sort_values(['customer_ID','S_2'], inplace=True)
    
    # GroupBy object
    g = df.groupby('customer_ID')
    
    # ————————————————————————————
    # 2a) numeric aggregations
    print('2a')
    num_agg = g[num].agg(['mean','std','min','max','last'])
    num_agg.columns = [f'{c}_{agg}' for c, agg in num_agg.columns]
    num_agg = num_agg.astype('float32')
    full_feats = num_agg
    del num_agg; gc.collect()

    # ————————————————————————————
    # 2b) categorical aggregations
    print('2b')
    cat_agg = g[cat].agg(['count','last','nunique'])
    cat_agg.columns = [f'{c}_{agg}' for c, agg in cat_agg.columns]
    cat_agg = cat_agg.astype('float32')
    full_feats = full_feats.join(cat_agg, how='left')
    del cat_agg; gc.collect()

    # ————————————————————————————
    # 2c) first/last differences and percent change
    print('2c')
    first_vals = g[num].first().astype('float32')
    last_vals  = g[num].last().astype('float32')
    
    diff = last_vals - first_vals
    diff.columns = [f'diff_{c}' for c in diff.columns]
    diff = diff.astype('float32')
    full_feats = full_feats.join(diff, how='left')
    del diff; gc.collect()

    pctchg = (last_vals - first_vals).div(first_vals.replace(0, np.nan))
    pctchg.columns = [f'pctchg_{c}' for c in pctchg.columns]
    pctchg = pctchg.fillna(0).astype('float32')
    full_feats = full_feats.join(pctchg, how='left')
    del first_vals, last_vals, pctchg; gc.collect()

    # ————————————————————————————
    # 2d) time span & recency
    print('2d')
    max_time = g['S_2'].max()
    min_time = g['S_2'].min()
    span     = (max_time - min_time).dt.days.to_frame('days_span').astype('int16')
    recency  = (pd.Timestamp('today') - max_time).dt.days.to_frame('days_since_last').astype('int16')
    del max_time, min_time; gc.collect()

    full_feats = full_feats.join(span, how='left')
    del span; gc.collect()
    full_feats = full_feats.join(recency, how='left')
    del recency; gc.collect()
    # 3) Memory-friendly Last-k windows by customer batches
    ids = df['customer_ID'].unique()
    batch_size = 50_000    # tune this to taste
    count = 0

    for batch_ids in np.array_split(ids, len(ids) // batch_size + 1):
        print(f'batch {count}')
        count += 1
        sub = df[df['customer_ID'].isin(batch_ids)]
        # since df is already sorted, cumcount(descending) gives “reverse rank”
        rev_rank = sub.groupby('customer_ID').cumcount(ascending=False)

        # ---- k=3 aggregates ----
        sub3 = sub[rev_rank < 3]
        # existing numeric & categorical aggs
        num3 = (
            sub3.groupby('customer_ID')[num]
                .agg(['mean','std','min','max'])
                .astype('float32')
        )
        num3.columns = [f'last3_{agg}_{c}' for c, agg in num3.columns]
    
        cat3 = (
            sub3.groupby('customer_ID')[cat]
                .agg(['count','nunique'])
                .astype('float32')
        )
        cat3.columns = [f'last3_{agg}_{c}' for c, agg in cat3.columns]
    
        # —— new: first/last diff for k=3 —— 
        first3 = sub3.groupby('customer_ID')[num].first().astype('float32')
        last3  = sub3.groupby('customer_ID')[num].last().astype('float32')
        diff3  = (last3 - first3)
        diff3.columns = [f'last3_diff_{c}' for c in diff3.columns]
    
        # combine all last-3 features
        part3 = pd.concat([num3, cat3, diff3], axis=1)

        # inject into full_feats
        full_feats.loc[part3.index, part3.columns] = part3
    
        # cleanup for this batch
        del sub3, num3, cat3, first3, last3, diff3, part3
        gc.collect()
    
    
        # # —— repeat the same for k=6 —— (gave up due to RAM limitation)
    
        # sub6 = sub[rev_rank < 6]
        # num6 = (
        #     sub6.groupby('customer_ID')[num]
        #         .agg(['mean','std','min','max'])
        #         .astype('float32')
        # )
        # num6.columns = [f'last6_{agg}_{c}' for c, agg in num6.columns]
    
        # cat6 = (
        #     sub6.groupby('customer_ID')[cat]
        #         .agg(['count','nunique'])
        #         .astype('float32')
        # )
        # cat6.columns = [f'last6_{agg}_{c}' for c, agg in cat6.columns]
    
        # # —— first/last diff for k=6 —— 
        # first6 = sub6.groupby('customer_ID')[num].first().astype('float32')
        # last6  = sub6.groupby('customer_ID')[num].last().astype('float32')
        # diff6  = (last6 - first6)
        # diff6.columns = [f'last6_diff_{c}' for c in diff6.columns]
    
        # part6 = pd.concat([num6, cat6, diff6], axis=1)

        # full_feats.loc[part6.index, part6.columns] = part6
    
        # del sub6, num6, cat6, first6, last6, diff6, part6, sub, rev_rank
        del sub, rev_rank
        gc.collect()


    # ————————————————————————————
    # 4) Final clean and return
    full_feats = full_feats.fillna(0)
    print('Feature Engineering Done — shape', full_feats.shape)
    return full_feats



train = read_data('../input/amex-data-integer-dtypes-parquet-format/train.parquet')


train = feature_eng(train)
features = train.columns[1:-1]
train.head()


labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv')
labels['customer_ID'] = labels['customer_ID'].str[-16:].apply(lambda x: int(x, 16)).astype('int64')
labels = labels.set_index('customer_ID')
train = train.merge(labels, left_index=True, right_index=True, how='left')
train.target = train.target.astype('int8')
del labels
gc.collect()

train = train.sort_index().reset_index()


# Evaluation for Valid
def amex_metric_mod(y_true, y_pred):

    y_pred = np.squeeze(np.array(y_pred))
    y_true = np.squeeze(np.array(y_true))

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def lgb_amex_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    score  = amex_metric_mod(y_true, y_pred)
    # name must be unique among metrics
    return 'amex_metric', score, True


# Transform the training data
X = train.drop(columns=['customer_ID', 'target']).astype('float32').to_numpy()
y = train['target'].astype('float32').to_numpy()
del train
gc.collect()


# Stratified K-Fold for training
NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)


# params = {
#     # task and device
#     'objective': 'binary',
#     'metric': ['binary_logloss', 'auc'],
#     'boosting': 'gbdt',
#     'device': 'gpu',
#     'seed': 42,
#     'verbosity': -1,

#     # learning
#     'learning_rate': 0.03,
#     'lambda_l1': 0.1,
#     'lambda_l2': 30,

#     # tree complexity
#     'num_leaves': 64,
#     'max_depth': -1,
#     'min_data_in_leaf': 256,
#     'min_data_in_bin': 256,
#     'max_bin': 63,

#     # sampling
#     'feature_fraction': 0.10,
#     'bagging_fraction': 0.75,
#     'bagging_freq': 5,

#     # disable boosting from global average
#     'boost_from_average': False,
# }



# # Train for LightGBM

# oof_preds = np.zeros(X.shape[0], dtype=np.float32)

# for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
#     print(f"Starting LGB Fold #{fold}")
#     train_data = lgb.Dataset(X[train_index], label=y[train_index])
#     valid_data = lgb.Dataset(X[valid_index], label=y[valid_index])
    
#     model = lgb.train(
#         params,
#         train_data,
#         num_boost_round=5000,
#         valid_sets=[valid_data],
#         callbacks=[
#             lgb.early_stopping(100),
#             lgb.log_evaluation(50),
#         ]
#     )
#     model.save_model(f'lgb_model_{fold}.txt', num_iteration=model.best_iteration)


#     # predict on the validation fold
#     y_preds = model.predict(
#         X[valid_index],
#         num_iteration=model.best_iteration
#     )
#     oof_preds[valid_index] = y_preds
    
#     acc = amex_metric_mod(y[valid_index], y_preds)
#     print('Kaggle Metric =',acc,'\n')

#     # clean up
#     del train_data, valid_data, y_preds, model
#     gc.collect()

# np.save('oof_preds_train.npy', oof_preds)
# del oof_preds
# gc.collect()


# oof_preds = np.load('oof_preds_train.npy').astype('float32')
# X = np.hstack([X, oof_preds.reshape(-1, 1)])
# del oof_preds


# params = {
#     # task and device
#     'objective': 'binary',
#     'metric': ['binary_logloss', 'auc'],
#     'boosting': 'gbdt',
#     'device': 'gpu',
#     'seed': 42,
#     'verbosity': -1,

#     # learning
#     'learning_rate': 0.03,
#     'lambda_l1': 0.1,
#     'lambda_l2': 30,
    

#     # tree complexity
#     'num_leaves': 64,
#     'max_depth': -1,
#     'min_data_in_leaf': 256,
#     'min_data_in_bin': 256,
#     'max_bin': 63,

#     # sampling
#     'feature_fraction': 0.10,
#     'bagging_fraction': 0.75,
#     'bagging_freq': 5,

#     # disable boosting from global average
#     'boost_from_average': False,
# }



# # Train for LightGBM 2
# for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
#     print(f"Starting LGB-2 Fold #{fold}")
#     train_data = lgb.Dataset(X[train_index], label=y[train_index])
#     valid_data = lgb.Dataset(X[valid_index], label=y[valid_index])
    
#     model = lgb.train(
#         params,
#         train_data,
#         num_boost_round=5000,
#         valid_sets=[valid_data],
#         callbacks=[
#             lgb.early_stopping(100),
#             lgb.log_evaluation(50),
#         ]
#     )
#     model.save_model(f'lgb_model2_{fold}.txt', num_iteration=model.best_iteration)


#     # predict on the validation fold
#     y_preds = model.predict(
#         X[valid_index],
#         num_iteration=model.best_iteration
#     )
    
#     acc = amex_metric_mod(y[valid_index], y_preds)
#     print('Kaggle Metric =',acc,'\n')

#     # clean up
#     del train_data, valid_data, y_preds, model
#     gc.collect()

# del params
# gc.collect()


# Build NN Model
def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(2*input_dim, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(input_dim, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    # Use a lower learning rate to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Transforming Test Data
scaler = StandardScaler()
X = X.astype('float32')   
chunk = 20_000                           # ~20 k rows at a time
for i in range(0, X.shape[0], chunk):
    print(f"Chunk #{i//chunk}")
    j = min(i+chunk, X.shape[0])
    X[i:j] = scaler.fit_transform(X[i:j])
    gc.collect()
gc.collect()


def make_dataset(X, y, batch_size=128, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        # shuffle buffer can be much smaller than full dataset
        ds = ds.shuffle(buffer_size=10_000)  
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"Starting NN Fold #{fold}")
    
    K.clear_session()
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    
    # Build the model using the number of features in X_train
    model = build_model(X_train.shape[1])
    
    # Set up early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    ds_train = make_dataset(X_train, y_train, batch_size=128, shuffle=True)
    ds_val   = make_dataset(X_valid, y_valid, batch_size=128, shuffle=False)
    
    # Train the model
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=64,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )
    model.save(f'NN_model_{fold}.h5')

    preds = model.predict(X_valid)
    preds = preds.flatten()  
    
    acc = amex_metric_mod(y_valid, preds)
    print('Kaggle Metric =',acc,'\n')

    del X_train, X_valid
    del y_train, y_valid
    del model
    gc.collect()


# Delete training data to free RAM
del X, y
gc.collect()


# for name in list(globals().keys()):
#     if name.startswith('_'):
#         continue
#     del globals()[name]


# # Standard library
# import gc

# import numpy as np
# import pandas as pd

# # Machine learning & preprocessing
# import lightgbm as lgb
# from sklearn.preprocessing import StandardScaler

# # Deep learning (TensorFlow / Keras)
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# gc.collect()

# # Reading Data
# def read_data(path):
#     df = pd.read_parquet(path)
#     df['customer_ID'] = df['customer_ID'].str[-16:].apply(lambda x: int(x, 16)).astype('int64')
#     df['S_2'] = pd.to_datetime(df['S_2'])
#     df = df.fillna(0)
#     print(f"Data shape: {df.shape}")
#     return df

# def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
#     # ————————————————————————————
#     # 1) Prep
#     all_f = [c for c in df.columns if c not in ['customer_ID','S_2']]
#     cat   = [
#         "B_30","B_38","D_114","D_116","D_117",
#         "D_120","D_126","D_63","D_64","D_66","D_68"
#     ]
#     num   = [c for c in all_f if c not in cat]
    
#     # Sort once to enable group.tail(k) without re-sorting
#     df.sort_values(['customer_ID','S_2'], inplace=True)
    
#     # GroupBy object
#     g = df.groupby('customer_ID')
    
#     # ————————————————————————————
#     # 2a) numeric aggregations
#     print('2a')
#     num_agg = g[num].agg(['mean','std','min','max','last'])
#     num_agg.columns = [f'{c}_{agg}' for c, agg in num_agg.columns]
#     num_agg = num_agg.astype('float32')
#     full_feats = num_agg
#     del num_agg; gc.collect()

#     # ————————————————————————————
#     # 2b) categorical aggregations
#     print('2b')
#     cat_agg = g[cat].agg(['count','last','nunique'])
#     cat_agg.columns = [f'{c}_{agg}' for c, agg in cat_agg.columns]
#     cat_agg = cat_agg.astype('float32')
#     full_feats = full_feats.join(cat_agg, how='left')
#     del cat_agg; gc.collect()

#     # ————————————————————————————
#     # 2c) first/last differences and percent change
#     print('2c')
#     first_vals = g[num].first().astype('float32')
#     last_vals  = g[num].last().astype('float32')
    
#     diff = last_vals - first_vals
#     diff.columns = [f'diff_{c}' for c in diff.columns]
#     diff = diff.astype('float32')
#     full_feats = full_feats.join(diff, how='left')
#     del diff; gc.collect()

#     pctchg = (last_vals - first_vals).div(first_vals.replace(0, np.nan))
#     pctchg.columns = [f'pctchg_{c}' for c in pctchg.columns]
#     pctchg = pctchg.fillna(0).astype('float32')
#     full_feats = full_feats.join(pctchg, how='left')
#     del first_vals, last_vals, pctchg; gc.collect()

#     # ————————————————————————————
#     # 2d) time span & recency
#     print('2d')
#     max_time = g['S_2'].max()
#     min_time = g['S_2'].min()
#     span     = (max_time - min_time).dt.days.to_frame('days_span').astype('int16')
#     recency  = (pd.Timestamp('today') - max_time).dt.days.to_frame('days_since_last').astype('int16')
#     del max_time, min_time; gc.collect()

#     full_feats = full_feats.join(span, how='left')
#     del span; gc.collect()
#     full_feats = full_feats.join(recency, how='left')
#     del recency; gc.collect()
#     # 3) Memory-friendly Last-k windows by customer batches
#     ids = df['customer_ID'].unique()
#     batch_size = 50_000    # tune this to taste

#     for batch_ids in np.array_split(ids, len(ids) // batch_size + 1):
#         sub = df[df['customer_ID'].isin(batch_ids)]
#         # since df is already sorted, cumcount(descending) gives “reverse rank”
#         rev_rank = sub.groupby('customer_ID').cumcount(ascending=False)
    
#         # pick last 3, last 6
#         sub3 = sub[rev_rank < 3]
    
#         # ---- k=3 aggregates ----
#         num3 = (
#             sub3.groupby('customer_ID')[num]
#                 .agg(['mean','std','min','max'])
#                 .astype('float32')
#         )
#         num3.columns = [f'last3_{agg}_{c}' for c, agg in num3.columns]
    
#         cat3 = (
#             sub3.groupby('customer_ID')[cat]
#                 .agg(['count','nunique'])
#                 .astype('float32')
#         )
#         cat3.columns = [f'last3_{agg}_{c}' for c, agg in cat3.columns]
    
#         part3 = pd.concat([num3, cat3], axis=1)
    
#         # inject into full_feats **in place**
#         # this will create the columns if they don’t exist yet,
#         # but only update rows in `batch_ids`
#         full_feats.loc[part3.index, part3.columns] = part3
    
#         # cleanup for this batch
#         del sub, rev_rank, sub3, num3, cat3, part3
#         gc.collect()

#     # ————————————————————————————
#     # 4) Final clean and return
#     full_feats = full_feats.fillna(0)
#     print('Feature Engineering Done — shape', full_feats.shape)
#     return full_feats


# test = read_data('../input/amex-data-integer-dtypes-parquet-format/test.parquet')
# test = feature_eng(test)


# NUM_FOLDS = 5
# # Initialize an array to accumulate predictions
# SHAPE = test.shape
# preds_sum = np.zeros(SHAPE[0], dtype=np.float32)

# for fold in range(NUM_FOLDS):
#     print(f"Fold #{fold}")
#     model = lgb.Booster(model_file=f'lgb_model_{fold}.txt')
    
#     preds = model.predict(test, num_iteration=model.best_iteration)
    
#     preds_sum += preds
#     del model, preds
#     gc.collect()

# preds_sum /= NUM_FOLDS
# np.save('lgb_preds.npy', preds_sum)
# test['oof_pred'] = preds_sum
# del preds_sum
# gc.collect()


# # Initialize an array to accumulate predictions
# preds_sum = np.zeros(SHAPE[0], dtype=np.float32)

# for fold in range(NUM_FOLDS):
#     print(f"Fold #{fold}")
#     model = lgb.Booster(model_file=f'lgb_model2_{fold}.txt')
    
#     preds = model.predict(test, num_iteration=model.best_iteration)
    
#     preds_sum += preds
#     del model, preds
#     # gc.collect()

# preds_sum /= NUM_FOLDS
# np.save('lgb_preds2.npy', preds_sum)
# del preds_sum
# gc.collect()


# # Transforming Test Data
# scaler = StandardScaler()
# test = test.astype('float32')   
# chunk = 20_000                           # ~20 k rows at a time
# for i in range(0, test.shape[0], chunk):
#     print(f"Chunk #{i/chunk}")
#     j = min(i+chunk, test.shape[0])
#     test[i:j] = scaler.fit_transform(test[i:j])
#     gc.collect()
# gc.collect()


# preds_sum = np.zeros(SHAPE[0], dtype=np.float32)

# for fold in range(NUM_FOLDS):
#     print(f"Fold #{fold}")
#     model = load_model(f'NN_model_{fold}.h5')
    
#     for start in range(0, SHAPE[0], chunk):
#         end = min(start + chunk, SHAPE[0])
#         chunk_preds = model.predict(
#             test[start:end],
#             verbose=0
#         ).ravel()
        
#         preds_sum[start:end] += chunk_preds
#         del chunk_preds
#         gc.collect()
    
#     del model
#     gc.collect()

# preds_sum /= NUM_FOLDS
# np.save('nn_preds.npy', preds_sum)
# del test, preds_sum
# gc.collect()


# # Average the predictions across all folds (Soft Voting)
# lgb_preds = np.load('lgb_preds.npy')
# lgb_preds2 = np.load('lgb_preds2.npy')
# nn_preds  = np.load('nn_preds.npy')

# final_pred = (0.4 * nn_preds + 0.5 * lgb_preds + 0.1 * lgb_preds2)
# del nn_preds
# # gc.collect()


# # WRITE SUBMISSION FILE
# sub = pd.read_csv('../input/amex-default-prediction/sample_submission.csv')[['customer_ID']]
# sub['customer_ID_hash'] = sub['customer_ID'].str[-16:].apply(lambda x: int(x, 16)).astype('int64')
# sub = sub.set_index('customer_ID_hash')
# sub = sub.sort_index()
# sub['prediction'] = final_pred  # test_preds should be a 1-D array or list of predictions

# sub[['customer_ID', 'prediction']].to_csv('submission.csv', index=False)
# sub = sub.reset_index(drop=True)

# sub.to_csv(f'submission.csv', index=False)
# print('Submission file shape is', sub.shape )
# sub.head()

