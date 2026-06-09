!pip install -U catboost lightgbm xgboost scikit-learn


import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm_notebook, tqdm
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
%matplotlib inline

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")

seed = 52


def fix_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds fixed to {seed} for random, numpy, torch")


fix_all_seeds(seed)


def map_at_3(y_true, y_pred):
    map_scores = []
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        pred_labels = pred_labels[:3]
        
        average_precision = 0
        num_correct = 0
        correct_labels = set(true_labels)
        
        for k, pred in enumerate(pred_labels, 1):
            if pred in correct_labels:
                num_correct += 1
                precision_at_k = num_correct / k
                average_precision += precision_at_k
                correct_labels.remove(pred)
        
        if len(true_labels) > 0:
            average_precision /= min(len(true_labels), 3)
        
        map_scores.append(average_precision)
    
    return np.mean(map_scores) if map_scores else 0


def evaluate_models(models, model_names, X, Y) -> pd.DataFrame:
    results = []
    for model, name in zip(models, model_names):
        y_pred = model.predict(X)
        accuracy = accuracy_score(Y, y_pred)
        f1 = f1_score(Y, y_pred, average='macro')
        recall = recall_score(Y, y_pred, average='macro')
        precision = precision_score(Y, y_pred, average='macro')
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            mapK = map_at_3([[i] for i in Y], np.argsort(-y_proba, axis=1)[:, :3])
        else:
            mapK = None

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Recall': recall,
            'Precision': precision,
            'MAP@3': mapK

        })
    results_df = pd.DataFrame(results)
    return results_df


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv').drop(columns=['id'])
data = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
train = pd.concat([train, data]).drop_duplicates().reset_index(drop=True)
del data


train


fer_name_to_num = {
    '14-35-14': 0,
    '10-26-26': 1,
    '17-17-17': 2,
    '28-28': 3,
    '20-20': 4,
    'DAP': 5,
    'Urea': 6
}
num_to_fer_name = {fer_name_to_num[i]: i for i in fer_name_to_num}


sns.set(rc={'figure.figsize': (15, 8)})
colors = sns.color_palette('pastel')[0:5]
labels = sorted(train['Fertilizer Name'].unique())
dt = np.array([len(train[train['Fertilizer Name'] == i])for i in labels])
plt.pie(dt, labels=labels, colors = colors, autopct='%.0f%%')
plt.show()


train.hist(figsize=(12, 8));


sns.set(rc={'figure.figsize': (12, 6)})
colors = sns.color_palette('pastel')[0:5]
labels = sorted(train['Soil Type'].unique())
dt = np.array([len(train[train['Soil Type'] == i])for i in labels])
plt.pie(dt, labels=labels, colors = colors, autopct='%.0f%%')
plt.show()


sns.set(rc={'figure.figsize': (12, 6)})
colors = sns.color_palette('pastel')[0:5]
labels = sorted(train['Crop Type'].unique())
dt = np.array([len(train[train['Crop Type'] == i])for i in labels])
plt.pie(dt, labels=labels, colors = colors, autopct='%.0f%%')
plt.show()


train['Fertilizer Name'] = train['Fertilizer Name'].apply(lambda x: fer_name_to_num[x])
train


train['Soil Type'] = train['Soil Type'].astype('category')
train['Crop Type'] = train['Crop Type'].astype('category')


train.hist(figsize=(12, 8));


sns.set(rc={'figure.figsize': (11, 8)})
correlation_matrix = pd.get_dummies(train, columns=train.drop('Fertilizer Name', axis=1).select_dtypes(include=['object', 'bool', 'category']).columns.tolist()).corr()
sns.heatmap(correlation_matrix);


# sns.set(rc={'figure.figsize': (16, 16)})
# sns.pairplot(train, hue='Fertilizer Name');


train


scale = StandardScaler()
train[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']] = scale.fit_transform(
    train[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']]
)
train


train = reduce_mem_usage(train)
train


X_train, X_valid, Y_train, Y_valid = train_test_split(
    train.drop(columns=['Fertilizer Name']),
    train['Fertilizer Name'],
    test_size=0.1,
    shuffle=True,
    stratify=train['Fertilizer Name'],
    random_state=seed
)
X_train.shape, X_valid.shape


cbc = CatBoostClassifier(random_state=seed, task_type='GPU')
cbc.fit(X_train, Y_train, cat_features=['Soil Type', 'Crop Type'], eval_set=(X_valid, Y_valid), verbose=1000)

xgb = XGBClassifier(random_state=seed, enable_categorical=True, device="cuda")
xgb.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)], verbose=128)

lgbm = LGBMClassifier(random_state=seed, categorical_feature=['Soil Type', 'Crop Type'], device='gpu', verbose=-1)
lgbm.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)])

hgbc = HistGradientBoostingClassifier(random_state=seed, categorical_features="from_dtype")
hgbc.fit(X_train, Y_train)


evaluate_models(
    [cbc, xgb, lgbm, hgbc],
    [
        'CatBoostClassifier', 'XGBClassifier', 'LGBMClassifier', 'HistGradientBoostingClassifier'
    ],
    X_valid, Y_valid
).sort_values(by=['MAP@3'], ascending=[False])


cat = CatBoostClassifier(
    iterations=4096 * 8,
    learning_rate=0.02,
    l2_leaf_reg=0.16,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    depth=8,
    use_best_model=True,
    random_state=seed,
    bagging_temperature=0.24,
    border_count=128,
    task_type='GPU',
    verbose=1000
)
cat.fit(
    X_train,
    Y_train,
    cat_features=['Soil Type', 'Crop Type'],
    eval_set=(X_valid, Y_valid),
    early_stopping_rounds=100
)


xgb = XGBClassifier(
    # booster='dart',
    enable_categorical=True,
    tree_method='hist',
    eval_metric='mlogloss',
    objective='multi:softprob',
    n_jobs=-1,
    n_estimators=4096*8,
    early_stopping_rounds=100,
    max_depth=10,
    min_child_weight=4,
    alpha=5.7,
    reg_lambda=5,
    learning_rate=0.01,
    gamma=0.26,
    subsample=0.8,
    colsample_bytree=0.35,
    random_state=seed,
    device="cuda"
)

xgb.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)], verbose=128)


lgbm = LGBMClassifier(
    boosting_type='gbdt',
    categorical_feature=['Soil Type', 'Crop Type'],
    learning_rate=0.04,
    n_estimators=4096*8,
    num_class=train['Fertilizer Name'].nunique(),
    reg_lambda=64,
    max_depth=8,
    eval_metric='multi_logloss',
    objective='multiclass',
    random_state=seed,
    device='gpu',
    min_child_weight=4,
    min_child_samples=13,
    subsample=0.4,
    colsample_bytree=0.4,
    early_stopping_round=100,
    verbose=-1,
    n_jobs=-1
)

lgbm.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)])


hgbc = HistGradientBoostingClassifier(
    random_state=seed,
    max_iter=4096*8,
    early_stopping=True,
    learning_rate=0.04,
    loss='log_loss',
    l2_regularization=2.56,
    max_depth=4,
    max_leaf_nodes=32,
    min_samples_leaf=64,
    categorical_features="from_dtype"
)
hgbc.fit(X_train, Y_train)


evaluate_models(
    models=[cat, xgb, lgbm, hgbc],
    model_names=['CatBoostClassifier', 'XGBClassifier', 'LGBMClassifier', 'HistGradientBoostingClassifier'],
    X=X_valid,
    Y=Y_valid
).sort_values(by=['MAP@3'], ascending=[False])


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
score_mean = 0
cat_models = []

for train_idx, valid_idx in tqdm(list(skf.split(train.drop(columns=['Fertilizer Name']), train['Fertilizer Name']))):
    X_tr, X_val = train.drop(columns=['Fertilizer Name']).iloc[train_idx], train.drop(columns=['Fertilizer Name']).iloc[valid_idx]
    y_tr, y_val = train['Fertilizer Name'].iloc[train_idx], train['Fertilizer Name'].iloc[valid_idx]
    print('Train size:', len(y_tr), 'Valid size:', len(y_val))
    cat = CatBoostClassifier(
        iterations=4096 * 8,
        learning_rate=0.02,
        l2_leaf_reg=0.16,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        depth=8,
        use_best_model=True,
        random_state=seed,
        bagging_temperature=0.24,
        border_count=128,
        task_type='GPU',
        verbose=1000
    )
    cat.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        cat_features=['Soil Type', 'Crop Type'],
        early_stopping_rounds=100
    )
    map3 = map_at_3([[i] for i in y_val], np.argsort(-cat.predict_proba(X_val), axis=1)[:, :3])
    print('MAP@3:', map3)
    print()
    score_mean += map3
    cat_models.append(cat)
print('Mean MAP@3:', score_mean / 10)


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
score_mean = 0
xgb_models = []

for train_idx, valid_idx in tqdm(list(skf.split(train.drop(columns=['Fertilizer Name']), train['Fertilizer Name']))):
    X_tr, X_val = train.drop(columns=['Fertilizer Name']).iloc[train_idx], train.drop(columns=['Fertilizer Name']).iloc[valid_idx]
    y_tr, y_val = train['Fertilizer Name'].iloc[train_idx], train['Fertilizer Name'].iloc[valid_idx]
    print('Train size:', len(y_tr), 'Valid size:', len(y_val))
    xgb = XGBClassifier(
        # booster='dart',
        enable_categorical=True,
        tree_method='hist',
        eval_metric='mlogloss',
        objective='multi:softprob',
        n_jobs=-1,
        n_estimators=4096*8,
        early_stopping_rounds=100,
        max_depth=10,
        min_child_weight=4,
        alpha=5.7,
        reg_lambda=5,
        learning_rate=0.01,
        gamma=0.26,
        subsample=0.8,
        colsample_bytree=0.35,
        random_state=seed,
        device="cuda"
    )
    
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=128)
    map3 = map_at_3([[i] for i in y_val], np.argsort(-xgb.predict_proba(X_val), axis=1)[:, :3])
    print('MAP@3:', map3)
    print()
    score_mean += map3
    xgb_models.append(xgb)
print('Mean MAP@3:', score_mean / 10)


# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# score_mean = 0
# lgbm_models = []

# for train_idx, valid_idx in tqdm(list(skf.split(train.drop(columns=['Fertilizer Name']), train['Fertilizer Name']))):
#     X_tr, X_val = train.drop(columns=['Fertilizer Name']).iloc[train_idx], train.drop(columns=['Fertilizer Name']).iloc[valid_idx]
#     y_tr, y_val = train['Fertilizer Name'].iloc[train_idx], train['Fertilizer Name'].iloc[valid_idx]
#     print('Train size:', len(y_tr), 'Valid size:', len(y_val))
#     lgbm = LGBMClassifier(
#         boosting_type='gbdt',
#         categorical_feature=['Soil Type', 'Crop Type'],
#         learning_rate=0.04,
#         n_estimators=4096*8,
#         num_class=train['Fertilizer Name'].nunique(),
#         reg_lambda=64,
#         max_depth=8,
#         eval_metric='multi_logloss',
#         objective='multiclass',
#         random_state=seed,
#         device='gpu',
#         min_child_weight=4,
#         min_child_samples=13,
#         subsample=0.4,
#         colsample_bytree=0.4,
#         early_stopping_round=100,
#         verbose=-1,
#         n_jobs=-1
#     )
    
#     lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    
#     map3 = map_at_3([[i] for i in y_val], np.argsort(-lgbm.predict_proba(X_val), axis=1)[:, :3])
#     print('MAP@3:', map3)
#     print()
#     score_mean += map3
#     lgbm_models.append(lgbm)
# print('Mean MAP@3:', score_mean / 10)


test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test


test.isna().sum()


test['Soil Type'] = test['Soil Type'].astype('category')
test['Crop Type'] = test['Crop Type'].astype('category')
test[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']] = scale.transform(
    test[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']]
)
test = test.drop(columns=['id'])
test = reduce_mem_usage(test)


test


submit = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
submit


pred_xgb = sum([m.predict_proba(test) for m in tqdm(xgb_models)]) / 10 #xgb.predict_proba(test)
pred = np.argsort(-pred_xgb, axis=1)[:, :3]
pred = [' '.join([num_to_fer_name[j] for j in i]) for i in pred]
submit['Fertilizer Name'] = pred
submit


submit.to_csv('xgb.csv', index=False)


# pred_lgbm = sum([m.predict_proba(test) for m in tqdm(lgbm_models)]) / 10 # lgbm.predict_proba(test)
# pred = np.argsort(-pred_lgbm, axis=1)[:, :3]
# pred = [' '.join([num_to_fer_name[j] for j in i]) for i in pred]
# submit['Fertilizer Name'] = pred
# submit


# submit.to_csv('lgbm.csv', index=False)


pred_cat = sum([m.predict_proba(test) for m in tqdm(cat_models)]) / 10 # cat.predict_proba(test)
pred = np.argsort(-pred_cat, axis=1)[:, :3]
pred = [' '.join([num_to_fer_name[j] for j in i]) for i in pred]
submit['Fertilizer Name'] = pred
submit


submit.to_csv('cat.csv', index=False)


pred = pred_xgb + pred_cat
pred /= 2
pred = np.argsort(-pred, axis=1)[:, :3]
pred = [' '.join([num_to_fer_name[j] for j in i]) for i in pred]
submit['Fertilizer Name'] = pred
submit


submit.to_csv('Ensemble.csv', index=False)


pred = pred_xgb * 0.75 + pred_cat * 0.25
pred = np.argsort(-pred, axis=1)[:, :3]
pred = [' '.join([num_to_fer_name[j] for j in i]) for i in pred]
submit['Fertilizer Name'] = pred
submit


submit.to_csv('Blending_solo.csv', index=False)


# rfr = RandomForestClassifier(random_state=seed)
lr = LogisticRegression(random_state=seed)
knc = KNeighborsClassifier()
supvc = SVC()


center_models = []
for train_idx, valid_idx in tqdm(list(skf.split(train.drop(columns=['Fertilizer Name']), train['Fertilizer Name']))):
    X_tr, X_val = train.drop(columns=['Fertilizer Name']).iloc[train_idx], train.drop(columns=['Fertilizer Name']).iloc[valid_idx]
    y_tr, y_val = train['Fertilizer Name'].iloc[train_idx], train['Fertilizer Name'].iloc[valid_idx]
    
    xgb_tr, cat_tr = sum([m.predict_proba(X_tr) for m in tqdm(xgb_models)]) / 10, sum([m.predict_proba(X_tr) for m in tqdm(cat_models)]) / 10
    X_tr = result = np.concatenate((xgb_tr, cat_tr), axis=1)

    xgb_val, cat_val = sum([m.predict_proba(X_val) for m in tqdm(xgb_models)]) / 10, sum([m.predict_proba(X_val) for m in tqdm(cat_models)]) / 10
    X_val = result = np.concatenate((xgb_val, cat_val), axis=1)
    
    # rfr.fit(X_tr, y_tr)
    lr.fit(X_tr, y_tr)
    knc.fit(X_tr, y_tr)
    supvc.fit(X_tr, y_tr)

    # y_pred_val_rfr = rfr.predict_proba(X_val)
    y_pred_val_lr = lr.predict_proba(X_val)
    y_pred_val_knc = knc.predict_proba(X_val)
    y_pred_val_supvc = supvc.predict_proba(X_val)

    # map3_rfr = map_at_3([[i] for i in y_val], np.argsort(-y_pred_val_rfr, axis=1)[:, :3])
    map3_lr = map_at_3([[i] for i in y_val], np.argsort(-y_pred_val_lr, axis=1)[:, :3])
    map3_knc = map_at_3([[i] for i in y_val], np.argsort(-y_pred_val_knc, axis=1)[:, :3])
    map3_supvc = map_at_3([[i] for i in y_val], np.argsort(-y_pred_val_supvc, axis=1)[:, :3])
    print('RFR:', map3_rfr, 'LR:', map3_lr, 'KNC:', map3_knc, 'SVC:', map3_supvc)
    # if max(map3_rfr, map3_lr, map3_knc, map3_supvc) == map3_rfr:
    #     center_models.append(rfr)
    if max(map3_rfr, map3_lr, map3_knc, map3_supvc) == map3_lr:
        center_models.append(lr)
    elif max(map3_rfr, map3_lr, map3_knc, map3_supvc) == map3_knc:
        center_models.append(knc)
    elif max(map3_rfr, map3_lr, map3_knc, map3_supvc) == map3_supvc:
        center_models.append(supvc)


test_es = np.concatenate((pred_xgb, pred_cat), axis=1)


pred = sum([m.predict_proba(test_es) for m in tqdm(center_models)]) / 10 # cat.predict_proba(test)
pred = np.argsort(-pred, axis=1)[:, :3]
pred = [' '.join([num_to_fer_name[j] for j in i]) for i in pred]
submit['Fertilizer Name'] = pred
submit


submit.to_csv('Ensemble2.csv', index=False)

