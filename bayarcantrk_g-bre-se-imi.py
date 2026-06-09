# 1. Gerekli kÃ¼tÃ¼phaneler
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import xgboost
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

print(f"âœ… XGBoost sÃ¼rÃ¼mÃ¼: {xgboost.__version__}")

# 2. Veri yolunu bul
data_dir = None
for dirname, _, filenames in os.walk('/kaggle/input'):
    if 'train.csv' in filenames and 'test.csv' in filenames:
        data_dir = dirname
        break

if data_dir is None:
    raise FileNotFoundError("train.csv ve test.csv bulunamadÄ±.")

print(f"ðŸ“‚ Veriler bulundu: {data_dir}")

# 3. Veriyi oku
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# 4. Etiketleyiciler
label_encoders = {}
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

target_encoder = LabelEncoder()
train['Fertilizer Name'] = target_encoder.fit_transform(train['Fertilizer Name'])

# 5. Ã–zellik mÃ¼hendisliÄŸi
def feature_engineering(df):
    df['NPK_Ratio'] = df['Nitrogen'] / (df['Potassium'] + df['Phosphorous'] + 1e-6)
    df['N_Moisture'] = df['Nitrogen'] * df['Moisture']
    df['P_Humidity'] = df['Phosphorous'] * df['Humidity']
    df['K_Temp'] = df['Potassium'] * df['Temparature']
    df['NP_sum'] = df['Nitrogen'] + df['Phosphorous']
    df['Temp_Humidity'] = df['Temparature'] * df['Humidity']
    df['SoilCrop_Interaction'] = df['Soil Type'] * 100 + df['Crop Type'] * 10
    df['Moisture_log'] = np.log1p(df['Moisture'])
    df['N_to_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['K_to_N'] = df['Potassium'] / (df['Nitrogen'] + 1e-6)
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# 6. Ã–zellik listesi
features = [
    'Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
    'Nitrogen', 'Potassium', 'Phosphorous',
    'NPK_Ratio', 'N_Moisture', 'P_Humidity', 'K_Temp',
    'NP_sum', 'Temp_Humidity', 'SoilCrop_Interaction',
    'Moisture_log', 'N_to_P', 'K_to_N'
]

X = train[features]
y = train['Fertilizer Name']
X_test = test[features]

# 7. Model eÄŸitimi - Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_preds = np.zeros((len(train), len(np.unique(y))))
test_preds = np.zeros((len(test), len(np.unique(y))))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"ðŸ“˜ Fold {fold + 1} eÄŸitiliyor...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(target_encoder.classes_),
        random_state=42,
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=1.0,
        tree_method='hist',
        eval_metric='mlogloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )

    val_preds[val_idx] = model.predict_proba(X_val)
    test_preds += model.predict_proba(X_test) / skf.n_splits

# 8. Top-3 tahmin fonksiyonu
def get_top_k_predictions(probs, encoder, k=3):
    top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    top_k_labels = [encoder.inverse_transform(indices) for indices in top_k_indices]
    return [' '.join(str(label) for label in labels) for labels in top_k_labels]

# 9. Submission dosyasÄ±
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': get_top_k_predictions(test_preds, target_encoder, k=3)
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("âœ… 'submission.csv' baÅŸarÄ±yla oluÅŸturuldu!")

# 10. MAP@3 metriÄŸi
def map_at_k(y_true, y_pred_probs, k=3):
    score = 0.0
    for i in range(len(y_true)):
        true_label = target_encoder.inverse_transform([y_true[i]])[0]
        top_k_preds = target_encoder.inverse_transform(np.argsort(y_pred_probs[i])[::-1][:k])
        for j in range(k):
            if top_k_preds[j] == true_label:
                score += 1.0 / (j + 1)
                break
    return score / len(y_true)

val_score = map_at_k(y, val_preds, k=3)
print(f"ðŸ“Š Validation MAP@3: {val_score:.4f}")

# 11. Ã–zellik Ã¶nemi gÃ¶rselleÅŸtirme
model.get_booster().feature_names = features
xgboost.plot_importance(model, max_num_features=15, height=0.5)
plt.title('Ã–zellik Ã–nemi')
plt.show()

