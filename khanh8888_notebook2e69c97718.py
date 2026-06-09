# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/flight-delays-spring-2018/flight_delays_train.csv')
test = pd.read_csv('/kaggle/input/flight-delays-spring-2018/flight_delays_test.csv')


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# 1) One-hot encode các cột phân loại
cols_to_encode = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
Train = pd.get_dummies(train, columns=cols_to_encode)
Test  = pd.get_dummies(test,  columns=cols_to_encode)

# 2) Tách X, y từ bảng đã được encode
y = Train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
X = Train.drop('dep_delayed_15min', axis=1)

# 3) Đồng bộ cột giữa Train/Test
Test = Test.reindex(columns=X.columns, fill_value=0)

# 4) Chia train/valid
X_train_part, X_valid, y_train_part, y_valid = train_test_split(
    X.values, y, test_size=0.3, random_state=17
)

# 5) Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=3,
    random_state=17,
    n_jobs=-1
)



print (X_valid[:5])


# print (type(train))
# feature_names = Train.columns[:-1].tolist()

X = Train.drop('dep_delayed_15min', axis=1)
feature_names = X.columns.tolist() 

print (feature_names)
class_names =list( train['dep_delayed_15min'].unique())
print(class_names)


xgb_model.fit(X_train_part, y_train_part)

# 6) Đánh giá AUC
xgb_valid_pred = xgb_model.predict_proba(X_valid)[:, 1]
print("Valid AUC:", roc_auc_score(y_valid, xgb_valid_pred))

# 7) (tuỳ chọn) dự đoán trên test đồng bộ cột
# xgb_test_pred = xgb_model.predict_proba(Test.values)[:, 1


def predict_proba_fn(x):
    return xgb_model.predict_proba(x)


idx = 5
x0 =  X_valid[idx]
y0 = y_valid[idx]
proba = predict_proba_fn(x0.reshape(1, -1))[0]
pred = int(np.argmax(proba))
print(f"True label: {class_names[y0]} | Predicted: {class_names[pred]} | proba={proba}")


import numpy as np
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(
    training_data=X_train_part,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    discretize_continuous=False,    # rời rạc hoá để giải thích dễ hiểu
    kernel_width=0.5              # “độ cục bộ” của Gaussian kernel
)


exp = explainer.explain_instance(
    data_row=x0,
    predict_fn=predict_proba_fn,
    num_features=8,      # số feature hiển thị
    top_labels=1,        # nhãn có xác suất cao nhất
    num_samples=5000     # số lượng mẫu perturbation (mặc định ~5000)
)


label_to_explain = pred
print("\nLIME explanation (feature, weight):")
for feat, weight in exp.as_list(label=label_to_explain):
    print(f"{feat} : {weight:.4f}")


exp.save_to_file('lime_xgb_explanation.html')
print("\nĐã lưu: lime_xgb_explanation.html (mở bằng trình duyệt để xem biểu đồ & bảng chi tiết).")


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from xgboost import XGBClassifier
# from sklearn.metrics import roc_auc_score
# import optuna
# import warnings
# warnings.filterwarnings('ignore')

# # Load data (giữ nguyên tên biến của bạn)
# train = pd.read_csv('/kaggle/input/flight-delays-spring-2018/flight_delays_train.csv')
# test = pd.read_csv('/kaggle/input/flight-delays-spring-2018/flight_delays_test.csv')

# print(f"Train shape: {train.shape}, Test shape: {test.shape}")
# print(f"Target distribution: {train['dep_delayed_15min'].value_counts(normalize=True)}")

# # =====================================================
# # DATA PREPROCESSING (giữ nguyên logic của bạn)
# # =====================================================

# # 1) One-hot encode các cột phân loại (giữ nguyên)
# cols_to_encode = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# Train = pd.get_dummies(train, columns=cols_to_encode)
# Test = pd.get_dummies(test, columns=cols_to_encode)

# # 2) Tách X, y từ bảng đã được encode (giữ nguyên)
# y = Train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
# X = Train.drop('dep_delayed_15min', axis=1)

# # 3) Đồng bộ cột giữa Train/Test (giữ nguyên)
# Test = Test.reindex(columns=X.columns, fill_value=0)

# print(f"Feature shape after encoding: {X.shape}")
# print(f"Features: {list(X.columns)}")

# # =====================================================
# # HYPERPARAMETER OPTIMIZATION
# # =====================================================

# def objective(trial):
#     """Tối ưu hyperparameters cho XGBoost"""
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 200, 800),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
#         'subsample': trial.suggest_float('subsample', 0.7, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1, 5),
#         'random_state': 17,
#         'n_jobs': -1
#     }
    
#     # Cross-validation để đánh giá
#     model = XGBClassifier(**params)
#     cv_scores = cross_val_score(
#         model, X.values, y, 
#         cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=17),
#         scoring='roc_auc',
#         n_jobs=-1
#     )
    
#     return cv_scores.mean()

# print("Starting hyperparameter optimization...")
# study = optuna.create_study(direction='maximize', 
#                            sampler=optuna.samplers.TPESampler(seed=17))
# study.optimize(objective, n_trials=30, timeout=900)  # 15 minutes max

# best_params = study.best_params
# print(f"Best parameters: {best_params}")
# print(f"Best CV AUC: {study.best_value:.6f}")

# # =====================================================
# # MODEL TRAINING (giữ nguyên cấu trúc của bạn)
# # =====================================================

# # 4) Chia train/valid (giữ nguyên)
# X_train_part, X_valid, y_train_part, y_valid = train_test_split(
#     X.values, y, test_size=0.3, random_state=17
# )

# # 5) Train XGBoost với best parameters
# xgb_model = XGBClassifier(**best_params)

# # Train với early stopping để tránh overfitting
# xgb_model.fit(
#     X_train_part, y_train_part,
#     eval_set=[(X_valid, y_valid)],
#     early_stopping_rounds=50,
#     verbose=False
# )

# # =====================================================
# # MODEL EVALUATION
# # =====================================================

# # 6) Đánh giá AUC (giữ nguyên)
# xgb_valid_pred = xgb_model.predict_proba(X_valid)[:, 1]
# valid_auc = roc_auc_score(y_valid, xgb_valid_pred)
# print(f"Valid AUC: {valid_auc:.6f}")

# # Cross-validation score trên toàn bộ data để đánh giá chính xác hơn
# final_cv_scores = cross_val_score(
#     xgb_model, X.values, y,
#     cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=17),
#     scoring='roc_auc',
#     n_jobs=-1
# )
# print(f"Final CV AUC: {final_cv_scores.mean():.6f} (+/- {final_cv_scores.std() * 2:.6f})")

# # =====================================================
# # FEATURE IMPORTANCE ANALYSIS
# # =====================================================

# # Phân tích feature importance
# feature_importance = pd.DataFrame({
#     'feature': X.columns,
#     'importance': xgb_model.feature_importances_
# }).sort_values('importance', ascending=False)

# print(f"\nTop 10 Most Important Features:")
# print(feature_importance.head(10))

# print(f"\nTop 5 Carriers by importance:")
# carrier_features = feature_importance[feature_importance['feature'].str.contains('UniqueCarrier')]
# print(carrier_features.head())

# print(f"\nTop 5 Origins by importance:")
# origin_features = feature_importance[feature_importance['feature'].str.contains('Origin')]
# print(origin_features.head())

# # =====================================================
# # TEST PREDICTIONS
# # =====================================================

# # 7) Dự đoán trên test (giữ nguyên cấu trúc)
# xgb_test_pred = xgb_model.predict_proba(Test.values)[:, 1]

# print(f"\nTest predictions statistics:")
# print(f"Min: {xgb_test_pred.min():.6f}")
# print(f"Max: {xgb_test_pred.max():.6f}")
# print(f"Mean: {xgb_test_pred.mean():.6f}")
# print(f"Std: {xgb_test_pred.std():.6f}")

# # Tạo submission file
# submission = pd.DataFrame({
#     'id': range(len(xgb_test_pred)),
#     'dep_delayed_15min': xgb_test_pred
# })
# submission.to_csv('improved_flight_delay_predictions.csv', index=False)

# # =====================================================
# # MODEL SUMMARY
# # =====================================================

# print(f"\n=== MODEL SUMMARY ===")
# print(f"Model: XGBoost with optimized hyperparameters")
# print(f"Features: {X.shape[1]} (after one-hot encoding)")
# print(f"Training samples: {len(y_train_part)}")
# print(f"Validation samples: {len(y_valid)}")
# print(f"Best validation AUC: {valid_auc:.6f}")
# print(f"Cross-validation AUC: {final_cv_scores.mean():.6f}")
# print(f"Final model parameters:")
# for param, value in best_params.items():
#     print(f"  {param}: {value}")

# # =====================================================
# # ADDITIONAL IMPROVEMENTS (optional)
# # =====================================================

# # So sánh với model gốc của bạn
# print(f"\n=== COMPARISON WITH ORIGINAL MODEL ===")
# original_model = XGBClassifier(
#     n_estimators=500,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=17,
#     n_jobs=-1
# )
# original_model.fit(X_train_part, y_train_part)
# original_pred = original_model.predict_proba(X_valid)[:, 1]
# original_auc = roc_auc_score(y_valid, original_pred)

# print(f"Original model AUC: {original_auc:.6f}")
# print(f"Optimized model AUC: {valid_auc:.6f}")
# print(f"Improvement: {((valid_auc - original_auc) / original_auc * 100):.2f}%")


# xgb_model.fit(X, y)
# xgb_test_pred = xgb_model.predict_proba(Test)[:,1]

# pd.Series(xgb_test_pred,
#           name='dep_delayed_15min').to_csv('xgb_2feat.csv',
#                                            index_label='id', header=True)

