import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import optuna


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


label_enc = LabelEncoder()
for i in cat_columns:
    train[i] = label_enc.fit_transform(train[i])
    test[i] = label_enc.transform(test[i])
    train[i] = train[i].astype('category')
    test[i] = test[i].astype('category')
train['Fertilizer Name'] = label_enc.fit_transform(train['Fertilizer Name'])


train['max_env'] = train[['Temparature','Humidity','Moisture']].max(axis= 1)
test['max_env'] = test[['Temparature','Humidity','Moisture']].max(axis= 1)


def get_crop_optimal_temp(crop):
    crop_temp_map = {
    'Sugarcane': (26, 35), 'Maize': (25, 32), 'Wheat': (20, 30),
    'Paddy': (25, 35), 'Cotton': (25, 35), 'Tobacco': (20, 30),
    'Barley': (15, 25), 'Millets': (25, 35), 'Pulses': (20, 30),
    'Oil seeds': (20, 30), 'Ground Nuts': (25, 32)
    }
    return crop_temp_map.get(crop, (25, 32))

train['temp_suitability'] = train.apply(lambda row: 1 if get_crop_optimal_temp(row['Crop Type'])[0] <= row['Temparature'] <= get_crop_optimal_temp(row['Crop Type'])[1] else 0, axis=1)
test['temp_suitability'] = test.apply(lambda row: 1 if get_crop_optimal_temp(row['Crop Type'])[0] <= row['Temparature'] <= get_crop_optimal_temp(row['Crop Type'])[1] else 0, axis= 1)


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


xgb_model_1 = XGBClassifier(
    max_depth= 17,
    colsample_bytree= 0.390,
    subsample=0.552,
    n_estimators= 1364,
    learning_rate= 0.0278,
    gamma= 0.296,
    max_delta_step= 2,
    reg_alpha= 1.8967, 
    reg_lambda= 1.458,
    early_stopping_rounds=50,
    objective='multi:softprob',
    random_state = 42,
    eval_metric='mlogloss',
    enable_categorical=True,
    device = 'cuda'
)

xgb_model_2 = XGBClassifier(
    max_depth=14,
    colsample_bytree=0.3012,
    subsample=0.5735,
    n_estimators=3500,
    learning_rate=0.0339,
    gamma=0.2942,
    max_delta_step=5,
    reg_alpha= 2.4983,
    reg_lambda= 1.3456,
    early_stopping_rounds=200,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state = 13,
    enable_categorical=True,
    device = 'cuda')

xgb_model_3 = XGBClassifier(
    max_depth=11,
    colsample_bytree=0.4845,
    subsample=0.6902,
    n_estimators=1321,
    learning_rate=0.0409,
    gamma=0.2925,
    max_delta_step=1,
    reg_alpha= 2.5228,
    reg_lambda= 1.6911,
    early_stopping_rounds=100,
    objective='multi:softprob',
    random_state = 13,
    enable_categorical=True,
    device = 'cuda')

correlations = []

for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15,'FOLD:', i+1, '#' *15)
    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    actual = [[label] for label in y_valid]
    ##XGB 1 
    xgb_model_1.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 1000)
    oof_xgb_1[valid_idx] = xgb_model_1.predict_proba(x_valid)
    pred_prob_xgb_1 +=xgb_model_1.predict_proba(test)

    top_3_preds_xgb_1 = np.argsort(oof_xgb_1[valid_idx], axis=1)[:, -3:][:, ::-1]  
    map3_score_xgb_1 = mapk(actual, top_3_preds_xgb_1)
    print(f"âœ… FOLD {i+1}: MAP@3 XGB_1 Score: {map3_score_xgb_1:.5f}")

    ##XGB 2 
    xgb_model_2.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 1000)
    oof_xgb_2[valid_idx] = xgb_model_2.predict_proba(x_valid)
    pred_prob_xgb_2 +=xgb_model_2.predict_proba(test)

    top_3_preds_xgb_2 = np.argsort(oof_xgb_2[valid_idx], axis=1)[:, -3:][:, ::-1]  
    map3_score_xgb_2 = mapk(actual, top_3_preds_xgb_2)
    print(f"â˜‘ï¸� FOLD {i+1}: MAP@3 XGB_2 Score: {map3_score_xgb_2:.5f}")

    ##XGB 3
    xgb_model_3.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 1000)
    oof_xgb_3[valid_idx] = xgb_model_3.predict_proba(x_valid)
    pred_prob_xgb_3 +=xgb_model_3.predict_proba(test)

    top_3_preds_xgb_3 = np.argsort(oof_xgb_3[valid_idx], axis=1)[:, -3:][:, ::-1]  
    map3_score_xgb_3 = mapk(actual, top_3_preds_xgb_3)
    print(f"âœ”ï¸� FOLD {i+1}: MAP@3 XGB_3 Score: {map3_score_xgb_3:.5f}")

    
    corr_12 = np.corrcoef(oof_xgb_1[valid_idx].ravel(), oof_xgb_2[valid_idx].ravel())[0, 1]
    corr_13 = np.corrcoef(oof_xgb_1[valid_idx].ravel(), oof_xgb_3[valid_idx].ravel())[0, 1]
    corr_23 = np.corrcoef(oof_xgb_2[valid_idx].ravel(), oof_xgb_3[valid_idx].ravel())[0, 1]

    
    print(f"ğŸ”— FOLD {i+1}:")
    print(f'Corr XGB_1 vs XGB_2: {corr_12:.5f}')
    print(f'Corr XGB_1 vs XGB_3: {corr_13:.5f}')
    print(f'Corr XGB_2 vs XGB_3: {corr_23:.5f}')
    
    avg_corr = np.mean([corr_12, corr_13, corr_23])
    correlations.append(avg_corr)


print(f"\nğŸ“Š Mean after 5 Folds: {np.mean(correlations):.5f}")


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



pred_prob_xgb_1 /= FOLDS
pred_prob_xgb_2 /=FOLDS
pred_prob_xgb_3 /= FOLDS



optuna.logging.set_verbosity(optuna.logging.WARNING)
def objective(trial):
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = trial.suggest_float("w2", 0.0, 1.0 - w1)
    w3 = 1.0 - w1 - w2 

    if not (0.0 <= w3 <= 1.0):
        return 0.0

    combined = (
        w1 * oof_xgb_1 +
        w2 * oof_xgb_2 +
        w3 * oof_xgb_3
    )

    # Top-3 predict
    top_3 = np.argsort(combined, axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y]

    return mapk(actual, top_3)

# Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

# best weights
best_w1 = study.best_params["w1"]
best_w2 = study.best_params["w2"]
best_w3 = 1.0 - best_w1 -best_w2

print("\nğŸ“Œ Best-found weights (Optuna, normalized to 1):")
print(f"XGB_1 = {best_w1:.4f}, XGB_2 = {best_w2:.4f}, XGB_3 = {best_w3:.4f}")
print(f"ğŸ“ˆ MAP@3 score after optimization: {study.best_value:.5f}")


w1 = study.best_params["w1"]
w2 = study.best_params["w2"]
w3 = 1.0 - w1 - w2 


print(f'weight 1 : {w1:.4f},\nweight 2 : {w2:.4f},\nweight 3 : {w3:.4f}')


pred_prob = w1 * pred_prob_xgb_1 + w2 *pred_prob_xgb_2 + w3 * pred_prob_xgb_3

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

