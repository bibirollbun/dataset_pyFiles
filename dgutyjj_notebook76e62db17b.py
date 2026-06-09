import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ========== 1. 数据加载 ==========
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# ========== 2. 特征预处理 ==========
numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
scaler = StandardScaler()
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
test_data[numeric_features] = scaler.transform(test_data[numeric_features])

# Soil Type One-Hot
soil_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
soil_encoded = soil_encoder.fit_transform(train_data[['Soil Type']])
soil_encoded_df = pd.DataFrame(soil_encoded, columns=soil_encoder.get_feature_names_out(['Soil Type']))
soil_encoded_test = soil_encoder.transform(test_data[['Soil Type']])
soil_test_df = pd.DataFrame(soil_encoded_test, columns=soil_encoder.get_feature_names_out(['Soil Type']))

# Crop Type One-Hot
crop_encoded = pd.get_dummies(train_data['Crop Type'], prefix='Crop')
crop_encoded_test = pd.get_dummies(test_data['Crop Type'], prefix='Crop')

# Align训练集和测试集的列
crop_encoded, crop_encoded_test = crop_encoded.align(crop_encoded_test, join='left', axis=1, fill_value=0)

# ========== 3. 标签编码 ==========
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_data['Fertilizer Name'])
y_categories = label_encoder.classes_

# ========== 4. 合并特征 ==========
X = pd.concat([train_data[numeric_features], soil_encoded_df, crop_encoded], axis=1)
X_test = pd.concat([pd.DataFrame(test_data[numeric_features], columns=numeric_features), soil_test_df, crop_encoded_test], axis=1)

# ========== 5. 划分训练/验证集 ==========
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ========== 6. LightGBM ==========
lgb_params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(y_categories),
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 8,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'random_state': 42,
    'verbose': -1
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=50)
    ]
)

val_probs_lgb = lgb_model.predict(X_val)
val_pred_lgb = np.argmax(val_probs_lgb, axis=1)

# ========== 7. XGBoost ==========
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(y_categories),
    learning_rate=0.03,
    n_estimators=1000,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_val, y_val)], verbose=50)
xgb_probs = xgb_model.predict_proba(X_val)
xgb_pred = np.argmax(xgb_probs, axis=1)

# ========== 8. 集成融合（加权平均） ==========
def ensemble_predict(probs1, probs2, weight1=0.6, weight2=0.4):
    return weight1 * probs1 + weight2 * probs2

ensemble_probs = ensemble_predict(val_probs_lgb, xgb_probs)
ensemble_pred = np.argmax(ensemble_probs, axis=1)

# ========== 9. MAP@5 ==========
def get_topk_predictions(probs, k=5):
    topk_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    return [[y_categories[i] for i in row] for row in topk_indices]

def mapk(y_true, y_pred, k=5):
    score = 0.0
    for true, preds in zip(y_true, y_pred):
        try:
            index = preds.index(true)
            if index < k:
                score += 1.0 / (index + 1)
        except ValueError:
            continue
    return score / len(y_true)

val_true_labels = [y_categories[label] for label in y_val]
val_top5 = get_topk_predictions(ensemble_probs, k=5)
val_map5 = mapk(val_true_labels, val_top5)

print(f"LightGBM Accuracy: {accuracy_score(y_val, val_pred_lgb):.4f}")
print(f"XGBoost Accuracy: {accuracy_score(y_val, xgb_pred):.4f}")
print(f"Ensemble Accuracy: {accuracy_score(y_val, ensemble_pred):.4f}")
print(f"Validation MAP@5: {val_map5:.4f}")

# ========== 10. 测试集预测 ==========
test_probs_lgb = lgb_model.predict(X_test)
test_probs_xgb = xgb_model.predict_proba(X_test)
test_probs_ensemble = ensemble_predict(test_probs_lgb, test_probs_xgb)
test_top5 = get_topk_predictions(test_probs_ensemble, k=5)

# ========== 11. 保存提交结果 ==========
submission = pd.DataFrame({
    'Id': test_data['id'],
    'Fertilizer Name': [' '.join(preds) for preds in test_top5]
})
submission.to_csv('fertilizer_predictions_optimized.csv', index=False)

# ========== 12. 可视化特征重要性 ==========
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_model.feature_importance()
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp['Feature'], feature_imp['Importance'])
plt.xlabel('Feature Importance')
plt.title('LightGBM Feature Importance')
plt.gca().invert_yaxis()
plt.show()


