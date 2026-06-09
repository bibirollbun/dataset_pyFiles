import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import xgboost as xgb 
import matplotlib.pyplot as plt 
import seaborn as sns 
import cupy as cp
from tqdm import tqdm
import gc


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv',index_col='id')
origin = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


target = 'Fertilizer Name'
cat_columns = [i for i in train.columns if train[i].dtype == np.object_][:-1]
num_columns = [i for i in train.columns if i not in cat_columns][:-1]
num_columns


label_enc = LabelEncoder()
ordinal_enc = OrdinalEncoder(handle_unknown='error')

train[cat_columns] = ordinal_enc.fit_transform(train[cat_columns])
test[cat_columns] = ordinal_enc.transform(test[cat_columns])
origin[cat_columns] = ordinal_enc.transform(origin[cat_columns])

train['Fertilizer Name'] = label_enc.fit_transform(train['Fertilizer Name'])
origin['Fertilizer Name'] = label_enc.fit_transform(origin['Fertilizer Name'])

train['const'] = 1
test['const'] =1
origin['const'] =1

train = train.astype('category')
test = test.astype('category')
origin = origin.astype('category')




def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



X = train.drop(['Fertilizer Name'],axis = 1)
y = train["Fertilizer Name"]
X_origin = origin.drop(['Fertilizer Name'],axis = 1)
y_origin = origin["Fertilizer Name"]


FOLDS =5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_1 = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob_1 = np.zeros(shape = (len(test),y.nunique()))
oof_2 = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob_2 = np.zeros(shape = (len(test),y.nunique()))

params_1 = {
    'max_depth': 10,
    'colsample_bytree': 0.3483397750785222,
    'subsample': 0.7190580486691939,
    'learning_rate': 0.01103266768429434,
    'alpha': 5.741670127816571,
    'reg_lambda': 5.091745473693504,
    'min_child_weight': 3,
    'max_bin': 128,
    'random_state': 42,
    'device': "cuda",
    'tree_method': 'hist',
    'eval_metric': "mlogloss",
    'objective': 'multi:softprob',
    'num_class': 7,
}

params_2 = {
    'max_depth': 8,
    'colsample_bytree': 0.5575975512767724,
    'subsample': 0.4449395275047807,
    'learning_rate': 0.005143173885294387,
    'gamma': 0.1126854190246728, 'max_delta_step': 1,
    'min_child_weight': 4,
    'reg_alpha': 1.4348838556239878,
    'reg_lambda': 1.7927746643723705,
    'device': "cuda",
    'tree_method': 'hist',
    'eval_metric': "mlogloss",
    'objective': 'multi:softprob',
    'num_class': 7,
    
}

print('Starting training XGB models...')
for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15,'FOLD:', i+1, '#' *15)

    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]
    x_test = test.copy()
    
    ## EXTRA DATA
    X_origin_expanded = X_origin.copy()
    y_origin_expanded = y_origin.copy()
    
    for _ in range(4):  
        X_origin_expanded = pd.concat([X_origin_expanded, X_origin.copy()], axis=0, ignore_index=True)
        y_origin_expanded = pd.concat([y_origin_expanded, y_origin.copy()], axis=0, ignore_index=True)

    #
    x_train = pd.concat([x_train, X_origin_expanded], axis=0, ignore_index=True)
    y_train = pd.concat([y_train, y_origin_expanded], axis=0, ignore_index=True)

    print(x_train.shape)
    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(x_valid, label=y_valid, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, enable_categorical=True)

    ## MODEL 1
    print('âš™ï¸� MODEL_1')
    model_1 = xgb.train(
        params_1,
        dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=100,
        verbose_eval=500
    )

    actual = [[label] for label in y_valid]
    oof_1[valid_idx] = model_1.predict(dval, iteration_range=(0, model_1.best_iteration + 1))
    pred_prob_1 += model_1.predict(dtest, iteration_range=(0, model_1.best_iteration + 1))/ FOLDS

    top_3_preds_1 = np.argsort(oof_1[valid_idx], axis=1)[:, -3:][:, ::-1]  
    map3_score_1 = mapk(actual, top_3_preds_1)
    print(f"âœ… FOLD {i+1}: MAP@3 MODEL_1 Score: {map3_score_1:.5f}")

    ## MODEL_2
    print('âš™ï¸� MODEL_2')
    model_2 = xgb.train(
        params_2,
        dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=100,
        verbose_eval=500
    )

    actual = [[label] for label in y_valid]
    oof_2[valid_idx] = model_2.predict(dval, iteration_range=(0, model_2.best_iteration + 1))
    pred_prob_2 += model_2.predict(dtest, iteration_range=(0, model_2.best_iteration + 1))/ FOLDS

    top_3_preds_2 = np.argsort(oof_2[valid_idx], axis=1)[:, -3:][:, ::-1]  
    map3_score_2 = mapk(actual, top_3_preds_2)
    print(f"âœ… FOLD {i+1}: MAP@3 MODEL_2 Score: {map3_score_2:.5f}")
    

actual = [[label] for label in y]
# MODEL_1
top_3_preds = np.argsort(oof_1, axis=1)[:, -3:][:, ::-1]  
map3_score = mapk(actual, top_3_preds)
print(f'âœ… Final MODEL_1 MAP@3 Score: {map3_score:.5f}')
# MODEL_2
top_3_preds = np.argsort(oof_2, axis=1)[:, -3:][:, ::-1]  
map3_score = mapk(actual, top_3_preds)
print(f'âœ… Final MODEL_2 MAP@3 Score: {map3_score:.5f}')


pred_prob_1 = np.load('/kaggle/input/models-xgb-250625/pred_250625_1.npy')
pred_prob_2 = np.load('/kaggle/input/models-xgb-250625/pred_250625_2.npy')
pred_prob_ori_1 = np.load('/kaggle/input/model-xgb-240625/pred_240625_1.npy')
pred_prob_ori_2 = np.load('/kaggle/input/xgb-2-8-re-2406/pred_240625_2.npy')
pred_prob_lgb_5 = np.load('/kaggle/input/models-lgb/lgb_test.npy')
pred_prob_lgbgoss_6 = np.load('/kaggle/input/models-lgb/lgb_goss_test.npy')


oof_1 = cp.asarray(np.load('/kaggle/input/models-xgb-250625/oof_250625_1.npy'))
oof_2 = cp.asarray(np.load('/kaggle/input/models-xgb-250625/oof_250625_2.npy'))
oof_ori_1 = cp.asarray(np.load('/kaggle/input/model-xgb-240625/oof_240625_1.npy'))
oof_ori_2 = cp.asarray(np.load('/kaggle/input/xgb-2-8-re-2406/oof_240625_2.npy'))
oof_lgb_5 = cp.asarray(np.load('/kaggle/input/models-lgb/lgb_oof.npy'))
oof_lgbgoss_6 = cp.asarray(np.load('/kaggle/input/models-lgb/lgb_goss_oof.npy'))

true = cp.asarray(y)
#  MAP@3 on GPU
def map3_cupy(y_true, y_score):
    top3 = cp.argsort(y_score, axis=1)[:, -3:][:, ::-1]   # (n, 3)
    y_true = y_true.reshape(-1, 1)                        # (n, 1)
    matches = (top3 == y_true).astype(cp.float32)         # (n, 3)
    weights = 1.0 / cp.arange(1, 4, dtype=cp.float32)     # (3,)
    score = (matches * weights).sum(axis=1)               # (n,)
    return score.mean().item()


USE_NEGATIVE_WGT = True
MAX_MODELS = 1000
TOL = 1e-5
files = ['model_1','model_2','model_ori_1', 'model_ori_2','model_lgb_5','model_lgbgoss_6']

#  (n_samples, n_classes, n_models)
oof_all = cp.stack([
    oof_1, oof_2, oof_ori_1,oof_ori_2, oof_lgb_5,oof_lgbgoss_6
], axis=-1)
n_models = oof_all.shape[2]


map3_scores = []
for i in range(n_models):
    score = map3_cupy(true, oof_all[:, :, i])
    map3_scores.append(score)

best_index =int(cp.argmax(cp.array(map3_scores)))

best_score = map3_scores[best_index]
print(f'0 We begin with best single model MAP@3 {best_score:0.5f} from "{files[best_index]}"')


start = -0.5 if USE_NEGATIVE_WGT else 0.02
ww = cp.arange(start, 0.51, 0.005)
models = [best_index]
weights = []
metrics = [best_score]
ensemble = oof_all[:, :, best_index].copy()
old_best_score = best_score

# Hill Climbing Ensemble
for kk in range(1_000_000):
    best_candidate_score = 0
    best_candidate_index = -1
    best_weight = 0
    best_combination = None

    candidates = list(range(n_models))

    with tqdm(total=len(candidates) * len(ww), desc=f"Iteracja {kk+1}") as pbar:
        for i in candidates:
            for w in ww:
                combined = (1 - w) * ensemble + w * oof_all[:, :, i]
                score = map3_cupy(true, combined)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate_index = i
                    best_weight = float(w)
                    best_combination = combined
                pbar.update(1)

    if best_candidate_index == -1 or best_candidate_score - old_best_score < TOL:
        print(f'=> We reached tolerance {TOL}')
        break

    print(f'{kk+1} New best MAP@3 {best_candidate_score:.5f} adding "{files[best_candidate_index]}" with weight {best_weight:.3f}')
    ensemble = best_combination
    old_best_score = best_candidate_score
    models.append(best_candidate_index)
    weights.append(best_weight)
    metrics.append(best_candidate_score)

    if len(models) >= MAX_MODELS:
        print(f'=> We reached {MAX_MODELS} models')
        break

    del combined, best_combination
    gc.collect()
    cp._default_memory_pool.free_all_blocks()

# Summary
print("\nğŸ“Œ Final ensemble:")
for i, idx in enumerate(models[1:], 1):
    print(f"{i}. {files[idx]} with weight {weights[i-1]:.3f}")
print(f"\nğŸ“ˆ Final MAP@3 score: {metrics[-1]:.5f}")


wgt = np.array([1])
for w in weights:
    wgt = wgt*(1-w)
    wgt = np.concatenate([wgt,np.array([w])])
    
rows = []
t = 0
for m,w,s in zip(models,wgt,metrics):
    name = files[m]
    dd = {}
    dd['weight'] = w
    dd['model'] = name
    rows.append(dd)
    t += float( f'{w:.3f}' )

# DISPLAY WEIGHT PER MODEL
df = pd.DataFrame(rows)
df = df.groupby('model').agg('sum').reset_index().sort_values('weight',ascending=False)
df = df.reset_index(drop=True)
df


oof_dict = {}

for name in files:
    try:
        var_name = name.replace('model_', 'oof_')
        var = eval(var_name)
        oof_dict[name] = var.get() if hasattr(var, "get") else var
    except NameError:
        print(f"âš ï¸� Variable '{var_name}' not found â€” skipped.")

preds_oof = sum(w * oof_dict[m] for m, w in zip(df['model'], df['weight']))

actual = [[label] for label in y]
top_3_preds = np.argsort(preds_oof, axis=1)[:, -3:][:, ::-1]  
map3_score = mapk(actual, top_3_preds)
print(f'âœ… Final  MAP@3 Score: {map3_score:.5f}')


df['model'] = df['model'].str.replace('model_', '', regex=False)
pred_dict = {}
for model_name in df['model'].unique():
    var_name = f'pred_prob_{model_name}'
    try:
        var = eval(var_name)
        pred_dict[model_name] = var.get() if hasattr(var, "get") else var
    except NameError:
        print(f"âš ï¸� Variable '{var_name}' not found â€” skipped.")


pred_prob = sum(w * pred_dict[m] for m, w in zip(df['model'], df['weight']))


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")


np.save("oof_3006_1.npy", oof_1)
np.save("pred_prob_3006_1.npy_1", pred_prob_1)
np.save("oof_3006_2.npy", oof_2)
np.save("pred_prob_3006_2.npy", pred_prob_2)

