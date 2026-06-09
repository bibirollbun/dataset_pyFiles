import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
# Target Encoding iÃ§in bu kÃ¼tÃ¼phaneyi yÃ¼klemeniz gerekebilir: !pip install category_encoders
import category_encoders as ce
import warnings

# Ayarlar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')
pd.options.mode.copy_on_write = True

# Veri YÃ¼kleme
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_orginal = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer-Prediction.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# ID ve SÃ¼tun AdÄ± TemizliÄŸi
df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)
df_train.columns = df_train.columns.str.strip()
df_orginal.columns = df_orginal.columns.str.strip()
df_test.columns = df_test.columns.str.strip()
df_orginal.rename(columns={'Soil Moisture': 'Moisture'}, inplace=True)

# ğŸ§  1. ADIM: GELÄ°Å�MÄ°Å� Ã–ZNÄ°TELÄ°K MÃœHENDÄ°SLÄ°Ä�Ä°
def create_advanced_features(df):
    epsilon = 1e-6
    # YazarÄ±n Ã¶nerdiÄŸi N/P/K oranlarÄ± ve toplamÄ±
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + epsilon)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + epsilon)
    df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + epsilon)
    df['N_plus_P_plus_K'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    
    # YazarÄ±n Ã¶nerdiÄŸi iklim etkileÅŸimi
    df['Temp_x_Humidity'] = df['Temparature'] * df['Humidity']
    
    # YazarÄ±n en gÃ¼Ã§lÃ¼ sinyal olarak belirttiÄŸi Soil-Crop etkileÅŸimi
    df['Soil_Crop_Interaction'] = df['Soil Type'] + '_' + df['Crop Type']
    
    return df

df_train = create_advanced_features(df_train)
df_orginal = create_advanced_features(df_orginal)
df_test = create_advanced_features(df_test)

# Kategorik sÃ¼tunlarÄ± dÃ¶nÃ¼ÅŸtÃ¼r
for col in df_test.select_dtypes(include=['object']).columns:
    df_train[col] = df_train[col].astype('category')
    df_orginal[col] = df_orginal[col].astype('category')
    df_test[col] = df_test[col].astype('category')

# Hedef deÄŸiÅŸkeni ve Label Encoding
target = df_train.pop('Fertilizer Name')
target_org = df_orginal.pop('Fertilizer Name')
le = LabelEncoder()
target = le.fit_transform(target)
target_org = le.transform(target_org)

# fast_map_k fonksiyonu (zaten mevcut)
def fast_map_k(actual: list, predicted: list, k: int = 3) -> float:
    # ... (kodunuzdaki fonksiyonun aynÄ±sÄ±)
    total_score = 0.0
    for true_items, pred_items in zip(actual, predicted):
        if not true_items: continue
        pred_items = pred_items[:k]
        true_set = set(true_items)
        hits = np.array([item in true_set for item in pred_items])
        if not hits.any(): continue
        cumulative_hits = np.cumsum(hits)
        positions = np.arange(1, len(pred_items) + 1)
        precisions = cumulative_hits[hits] / positions[hits]
        score = np.sum(precisions) / min(len(true_items), k)
        total_score += score
    return total_score / len(actual)

# ğŸš€ 2. ADIM: ENSEMBLE MODEL EÄ�Ä°TÄ°MÄ°
FOLDS = 5
sk_fold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Her model iÃ§in ayrÄ± tahmin dizileri
oof_lgb, pred_test_lgb = np.zeros((len(df_train), 7)), np.zeros((len(df_test), 7))
oof_xgb, pred_test_xgb = np.zeros((len(df_train), 7)), np.zeros((len(df_test), 7))

# Modeller iÃ§in parametreler (GPU destekli)
params_lgb = {'objective': 'multiclass','num_class': 7,'metric': 'multi_logloss','boosting_type': 'gbdt','device': 'gpu','n_estimators': 5000,'learning_rate': 0.05,'num_leaves': 40,'max_depth': 8,'subsample': 0.8,'colsample_bytree': 0.8,'seed': 42,'n_jobs': -1,'verbose': -1}
params_xgb = {'objective': 'multi:softprob','num_class': 7,'eval_metric': 'mlogloss','device': 'cuda','n_estimators': 5000,'learning_rate': 0.05,'max_depth': 8,'subsample': 0.8,'colsample_bytree': 0.8,'gamma': 0.2,'reg_alpha': 0.1,'reg_lambda': 0.1,'random_state': 42}

for i, (indx_train, indx_valid) in enumerate(sk_fold.split(df_train, target)):
    print(f"========== FOLD {i+1} ==========")
    
    # Veri ayÄ±rma
    X_train, y_train_fold = df_train.iloc[indx_train], target[indx_train]
    X_valid, y_valid_fold = df_train.iloc[indx_valid], target[indx_valid]

    # Target Encoding (Veri sÄ±zÄ±ntÄ±sÄ±nÄ± Ã¶nlemek iÃ§in sadece train fold'unda fit edilir)
    te = ce.TargetEncoder(cols=['Soil_Crop_Interaction'], handle_unknown='value', smoothing=10)
    X_train = te.fit_transform(X_train, y_train_fold)
    X_valid = te.transform(X_valid)
    X_test_fold = te.transform(df_test.copy()) # Her fold iÃ§in test verisini de dÃ¶nÃ¼ÅŸtÃ¼r
    
    # Orijinal veriyi ekleme (target encoding sonrasÄ±)
    X_train_full = pd.concat([X_train, te.transform(df_orginal)], axis=0)
    y_train_full = np.concatenate([y_train_fold, target_org], axis=0)
    
    # --- LightGBM EÄŸitimi ---
    print("--- Training LightGBM ---")
    lgb_model = lgb.LGBMClassifier(**params_lgb)
    lgb_model.fit(X_train_full.drop(columns=['Soil Type', 'Crop Type']), y_train_full,
                  eval_set=[(X_valid.drop(columns=['Soil Type', 'Crop Type']), y_valid_fold)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_lgb[indx_valid] = lgb_model.predict_proba(X_valid.drop(columns=['Soil Type', 'Crop Type']))
    pred_test_lgb += lgb_model.predict_proba(X_test_fold.drop(columns=['Soil Type', 'Crop Type'])) / FOLDS

    # --- XGBoost EÄŸitimi ---
    print("--- Training XGBoost ---")
    # XGBoost kategorik sÃ¼tunlarÄ± sevdiÄŸi iÃ§in onlarÄ± atmÄ±yoruz
    X_train_full_xgb = X_train_full.copy()
    X_valid_xgb = X_valid.copy()
    X_test_fold_xgb = X_test_fold.copy()
    for col in X_train_full_xgb.select_dtypes(include=['category']).columns:
        X_train_full_xgb[col] = X_train_full_xgb[col].cat.codes
        X_valid_xgb[col] = X_valid_xgb[col].cat.codes
        X_test_fold_xgb[col] = X_test_fold_xgb[col].cat.codes

    xgb_model = xgb.XGBClassifier(**params_xgb)
    xgb_model.fit(X_train_full_xgb, y_train_full,
                  eval_set=[(X_valid_xgb, y_valid_fold)],
                  callbacks=[xgb.callback.EarlyStopping(rounds=150, save_best=True)],
                  verbose=False)
    oof_xgb[indx_valid] = xgb_model.predict_proba(X_valid_xgb)
    pred_test_xgb += xgb_model.predict_proba(X_test_fold_xgb) / FOLDS


# 3. ADIM: DEÄ�ERLENDÄ°RME VE ENSEMBLE
print("\n========== MODEL PERFORMANCE ==========")
score_lgb = fast_map_k([[l] for l in target], np.argsort(oof_lgb, axis=1)[:, -3:][:, ::-1])
print(f"ğŸ“Š LightGBM OOF Score: {score_lgb:.5f}")
score_xgb = fast_map_k([[l] for l in target], np.argsort(oof_xgb, axis=1)[:, -3:][:, ::-1])
print(f"ğŸ“Š XGBoost OOF Score: {score_xgb:.5f}")

# Ensemble (Basit Ortalama)
oof_ensemble = (oof_lgb + oof_xgb) / 2
score_ensemble = fast_map_k([[l] for l in target], np.argsort(oof_ensemble, axis=1)[:, -3:][:, ::-1])
print(f"ğŸ�† ENSEMBLE OOF Score: {score_ensemble:.5f}")

# Submission DosyasÄ± OluÅŸturma
pred_test_ensemble = (pred_test_lgb + pred_test_xgb) / 2
top_preds = np.argsort(pred_test_ensemble, axis=1)[:, -3:][:, ::-1]
top_labels = le.inverse_transform(top_preds.ravel()).reshape(top_preds.shape)

df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
df_sub['Fertilizer Name'] = [' '.join(row) for row in top_labels]
df_sub.to_csv('submission_ensemble.csv', index=False)
print("\n'submission_ensemble.csv' dosyasÄ± baÅŸarÄ±yla oluÅŸturuldu.")




