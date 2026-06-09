import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import cupy as cp
from tqdm import tqdm
import gc


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv",index_col = 'id')
origin = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


train = pd.concat([train,origin], axis= 0 )


target = 'Fertilizer Name'
cat_columns = [i for i in train.columns if train[i].dtype == np.object_][:-1]
num_columns = [i for i in train.columns if i not in cat_columns]


for i in train.columns:
    print(f'Unique values in column: {i} - {train[i].nunique()}')
for i in train.columns[:-1]:
    print(f'Unique values in column: {i} - {test[i].nunique()}')


target = 'Fertilizer Name'
cat_columns = [i for i in train.columns if train[i].dtype == np.object_][:-1]
num_columns = [i for i in train.columns if i not in cat_columns]

label_enc = LabelEncoder()
ordinal_enc = OrdinalEncoder(handle_unknown='error')

train[cat_columns] = ordinal_enc.fit_transform(train[cat_columns])
test[cat_columns] = ordinal_enc.transform(test[cat_columns])
train[cat_columns] = train[cat_columns].astype('category')
test[cat_columns] = test[cat_columns].astype('category')

train['Fertilizer Name'] = label_enc.fit_transform(train['Fertilizer Name'])

train['const'] = 0
test['const'] =0


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


X = train.drop(target, axis = 1)
y = train[target]


FOLDS =5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
##  1
oof_xgb_1 = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob_xgb_1 = np.zeros(shape = (len(test),y.nunique()))
##  2 
oof_xgb_2 = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob_xgb_2 = np.zeros(shape = (len(test),y.nunique()))
# # ##  3
oof_xgb_3 = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob_xgb_3 = np.zeros(shape = (len(test),y.nunique()))
## 4 

oof_xgb_4 = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob_xgb_4 = np.zeros(shape = (len(test),y.nunique()))


params = {
        'objective': 'multi:softprob',
        'num_class': 7,
        'max_depth': 16,
        'learning_rate': 0.01,
        'n_estimators': 100_000,
        'reg_alpha': 3,
        'reg_lambda': 1.4,
        'gamma': 0.26,
        'max_delta_step': 5,
        'subsample': 0.86,
        'colsample_bytree': 0.4,
        'min_child_weight': 5,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss',
        'enable_categorical': True,
        'device': "cuda"   
}


xgb_model_1 = XGBClassifier(
   **params, early_stopping_rounds=30
)

xgb_model_2 = XGBClassifier(
    colsample_bytree= 0.5027610585618012,
    subsample= 0.9077307142274171,
    n_estimators = 5000,
    reg_lambda= 7.068573438476172,
    reg_alpha= 7.2900716804098735,
    max_depth= 11,
    gamma= 0.03702232586704518,
    learning_rate = 0.03,
    early_stopping_rounds=100,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state = 13,
    enable_categorical=True,
    device = 'cuda')

xgb_model_3 = XGBClassifier(
    max_depth= 13,
    colsample_bytree=0.30440196038980377,
    subsample= 0.5302363702993608,
    n_estimators= 5000,
    learning_rate=0.043509813901570604,
    gamma= 0.34649185501450364,
    max_delta_step=8,
    reg_alpha=2.0136709028472195,
    reg_lambda= 3.131760778539737,
    early_stopping_rounds= 100,
    random_state=13,
    enable_categorical= True,
    tree_method = 'hist',
    device = 'cuda',
    objective='multi:softprob')


params_2 = {'max_depth': 15, 
            'colsample_bytree': 0.47614479274606314,
            'subsample': 0.6514918955129614,
            'learning_rate': 0.0538335985032927,
            'gamma': 0.3749966216959073,
            'max_delta_step': 5,
            'reg_alpha': 0.7849738554509411,
            'reg_lambda': 1.1104143560242528,
            'n_estimators': 5000,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'enable_categorical': True,
            'device': "cuda",
            'objective': 'multi:softprob',
            'num_class': 7,
           }

xgb_model_4 = XGBClassifier(
   **params_2, early_stopping_rounds=30
)

correlations = []

for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15,'FOLD:', i+1, '#' *15)
    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    actual = [[label] for label in y_valid]
    ##XGB 1 
    xgb_model_1.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 1000)
    oof_xgb_1[valid_idx] = xgb_model_1.predict_proba(x_valid)
    pred_prob_xgb_1 +=xgb_model_1.predict_proba(test)/ FOLDS

    top_3_preds_xgb_1 = np.argsort(oof_xgb_1[valid_idx], axis=1)[:, -3:][:, ::-1]  
    map3_score_xgb_1 = mapk(actual, top_3_preds_xgb_1)
    print(f"âœ… FOLD {i+1}: MAP@3 XGB_1 Score: {map3_score_xgb_1:.5f}")

    ##XGB 2 
    xgb_model_2.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 1000)
    oof_xgb_2[valid_idx] = xgb_model_2.predict_proba(x_valid)
    pred_prob_xgb_2 +=xgb_model_2.predict_proba(test)/ FOLDS

    top_3_preds_xgb_2 = np.argsort(oof_xgb_2[valid_idx], axis=1)[:, -3:][:, ::-1]  
    map3_score_xgb_2 = mapk(actual, top_3_preds_xgb_2)
    print(f"â˜‘ï¸� FOLD {i+1}: MAP@3 XGB_2 Score: {map3_score_xgb_2:.5f}")

    ##XGB 3
    xgb_model_3.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 1000)
    oof_xgb_3[valid_idx] = xgb_model_3.predict_proba(x_valid)
    pred_prob_xgb_3 +=xgb_model_3.predict_proba(test) / FOLDS

    top_3_preds_xgb_3 = np.argsort(oof_xgb_3[valid_idx], axis=1)[:, -3:][:, ::-1]  
    map3_score_xgb_3 = mapk(actual, top_3_preds_xgb_3)
    print(f"âœ”ï¸� FOLD {i+1}: MAP@3 XGB_3 Score: {map3_score_xgb_3:.5f}")

    # XGB 4
    xgb_model_4.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=1000)
    oof_xgb_4[valid_idx] = xgb_model_4.predict_proba(x_valid)
    pred_prob_xgb_4 += xgb_model_4.predict_proba(test) /FOLDS

    top_3_preds_xgb_4 = np.argsort(oof_xgb_4[valid_idx], axis=1)[:, -3:][:, ::-1]
    map3_score_xgb_4 = mapk(actual, top_3_preds_xgb_4)
    print(f"ğŸŸ© FOLD {i+1}: MAP@3 XGB_4 Score: {map3_score_xgb_4:.5f}")


    
    # Correlations
    corr_12 = np.corrcoef(oof_xgb_1[valid_idx].ravel(), oof_xgb_2[valid_idx].ravel())[0, 1]
    corr_13 = np.corrcoef(oof_xgb_1[valid_idx].ravel(), oof_xgb_3[valid_idx].ravel())[0, 1]
    corr_14 = np.corrcoef(oof_xgb_1[valid_idx].ravel(), oof_xgb_4[valid_idx].ravel())[0, 1]
    corr_23 = np.corrcoef(oof_xgb_2[valid_idx].ravel(), oof_xgb_3[valid_idx].ravel())[0, 1]
    corr_24 = np.corrcoef(oof_xgb_2[valid_idx].ravel(), oof_xgb_4[valid_idx].ravel())[0, 1]
    corr_34 = np.corrcoef(oof_xgb_3[valid_idx].ravel(), oof_xgb_4[valid_idx].ravel())[0, 1]

    print(f"ğŸ”— FOLD {i+1}:")
    print(f'Corr XGB_1 vs XGB_2: {corr_12:.5f}')
    print(f'Corr XGB_1 vs XGB_3: {corr_13:.5f}')
    print(f'Corr XGB_1 vs XGB_4: {corr_14:.5f}')
    print(f'Corr XGB_2 vs XGB_3: {corr_23:.5f}')
    print(f'Corr XGB_2 vs XGB_4: {corr_24:.5f}')
    print(f'Corr XGB_3 vs XGB_4: {corr_34:.5f}')

    avg_corr = np.mean([corr_12, corr_13, corr_14, corr_23, corr_24, corr_34])
    correlations.append(avg_corr)

print(f"\nğŸ“Š Mean correlation after {skf.get_n_splits()} Folds: {np.mean(correlations):.5f}")


actual = [[label] for label in y]

# XGB_1
top_3_preds_xgb_1 = np.argsort(oof_xgb_1, axis=1)[:, -3:][:, ::-1]  
map3_score_xgb_1 = mapk(actual, top_3_preds_xgb_1)
print(f'âœ… Final XGB_1 MAP@3 Score: {map3_score_xgb_1:.5f}')

# XGB_2
top_3_preds_xgb_2 = np.argsort(oof_xgb_2, axis=1)[:, -3:][:, ::-1]  
map3_score_xgb_2 = mapk(actual, top_3_preds_xgb_2)
print(f'âœ… Final XGB_2 MAP@3 Score: {map3_score_xgb_2:.5f}')

# XGB_3
top_3_preds_xgb_3 = np.argsort(oof_xgb_3, axis=1)[:, -3:][:, ::-1]  
map3_score_xgb_3 = mapk(actual, top_3_preds_xgb_3)
print(f'âœ… Final XGB_3 MAP@3 Score: {map3_score_xgb_3:.5f}')

# XGB 4
top_3_preds_xgb_4 = np.argsort(oof_xgb_4, axis=1)[:, -3:][:, ::-1]  
map3_score_xgb_4 = mapk(actual, top_3_preds_xgb_4)
print(f'âœ… Final XGB_3 MAP@3 Score: {map3_score_xgb_4:.5f}')



oof_xgb_1 = cp.asarray(oof_xgb_1)
oof_xgb_2 = cp.asarray(oof_xgb_2)
oof_xgb_3 = cp.asarray(oof_xgb_3)
oof_xgb_4 = cp.asarray(oof_xgb_4)

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
files = ['model_xgb_1', 'model_xgb_2','model_xgb_3','model_xgb_4']

#  (n_samples, n_classes, n_models)
oof_all = cp.stack([
    oof_xgb_1,oof_xgb_2,oof_xgb_3, oof_xgb_4
], axis=-1)
n_models = oof_all.shape[2]


map3_scores = []
for i in range(n_models):
    score = map3_cupy(true, oof_all[:, :, i])
    map3_scores.append(score)

best_index = int(cp.argmax(cp.array(map3_scores)))
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


submission.head()

