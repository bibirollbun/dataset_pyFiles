import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score
)
pd.set_option("display.max_columns",None)

import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTGBM_VERBOSE"] = "0"
warnings.filterwarnings('ignore')
import os
os.environ['LGBM_VERBOSITY'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Optional: Silence XGBoost and CatBoost specific logs
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)

pd.set_option("display.max_columns",None)
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns",None)

%matplotlib inline


df=pd.read_csv("/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_train.csv")


df.drop(columns=["label_source"],axis=1,inplace=True)


df.head()



# 1. Ratio Features
# Speed-to-Design-Speed Ratio
df['speed_to_design_speed'] = df['speed'] / df['design_speed']

# Engine Load-to-RPM Ratio (avoid division by zero)
df['engine_load_to_rpm'] = df['engine_load_value'] / (df['rpm'] + 1)

# 2. Difference Features
# Speed Deviation from Design Speed
df['speed_deviation'] = abs(df['speed'] - df['design_speed'])

# Heart Rate Deviation from Mean
heart_rate_mean = df['heart_rate'].mean()
df['heart_rate_deviation'] = df['heart_rate'] - heart_rate_mean

# 3. Normalization
# Normalize heart_rate to 0-1 scale
heart_rate_min = df['heart_rate'].min()
heart_rate_max = df['heart_rate'].max()
df['normalized_heart_rate'] = (df['heart_rate'] - heart_rate_min) / (heart_rate_max - heart_rate_min)

# 4. Interaction Features
# Weather and Visibility Interaction
df['weather_visibility'] = df['current_weather'] * df['visibility']

# Heart Rate and Engine Load Interaction
df['heart_rate_engine_load'] = df['heart_rate'] * df['engine_load_value']

# 5. Aggregate Features
# Accidents Onsite Ratio (avoid division by zero)
df['accidents_onsite_ratio'] = df['accidents_onsite'] / (df['accidents_time'] + 1)

# 6. Categorical Binning
# Bin heart_rate into low, medium, high
bins = [0, 75, 80, float('inf')]
labels = ['low', 'medium', 'high']
df['heart_rate_bin'] = pd.cut(df['heart_rate'], bins=bins, labels=labels, include_lowest=True)

# 7. Anomaly Flag
# Flag high acceleration with zero throttle_position
df['accel_anomaly'] = np.where((df['acceleration'] > 800) & (df['throttle_position'] == 0), 1, 0)

# 8. Environmental Risk Score
# Weighted combination of weather, visibility, and precipitation
w1, w2, w3 = 0.4, 0.3, 0.3
df['env_risk'] = (w1 * df['current_weather'] + w2 * (1 / df['visibility']) + w3 * df['precipitation'])

# 9. Driver Stress Indicator
df['high_stress'] = np.where(df['heart_rate'] > 80, 1, 0)


df["heart_rate_bin"].unique()


mapping = {'low': 0, 'medium': 1, 'high': 2}
df['heart_rate_bin'] = df['heart_rate_bin'].map(mapping)
df['heart_rate_bin'] = df['heart_rate_bin'].astype(int)


df.shape


df.head()


df.info()


X=df.drop(columns=["risk_level"],axis=1)
y=df["risk_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = CatBoostClassifier(iterations=100,learning_rate=0.1,depth=6,eval_metric='Accuracy',verbose=100,random_seed=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))


test=pd.read_csv("/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_test.csv")


test.head()


test.drop(columns=["label_source"],axis=1,inplace=True)


# 1. Ratio Features
test['speed_to_design_speed'] = test['speed'] / test['design_speed']
test['engine_load_to_rpm'] = test['engine_load_value'] / (test['rpm'] + 1)  # avoid division by zero

# 2. Difference Features
test['speed_deviation'] = abs(test['speed'] - test['design_speed'])
test['heart_rate_deviation'] = test['heart_rate'] - test['heart_rate'].mean()

# 3. Normalization
test['normalized_heart_rate'] = (test['heart_rate'] - test['heart_rate'].min()) / \
                                (test['heart_rate'].max() - test['heart_rate'].min())

# 4. Interaction Features
test['weather_visibility'] = test['current_weather'] * test['visibility']
test['heart_rate_engine_load'] = test['heart_rate'] * test['engine_load_value']

# 5. Aggregate Features
test['accidents_onsite_ratio'] = test['accidents_onsite'] / (test['accidents_time'] + 1)

# 6. Categorical Binning for heart_rate
bins = [0, 75, 80, float('inf')]
labels = ['low', 'medium', 'high']
test['heart_rate_bin'] = pd.cut(test['heart_rate'], bins=bins, labels=labels, include_lowest=True)

# 7. Anomaly Flag
test['accel_anomaly'] = np.where((test['acceleration'] > 800) & (test['throttle_position'] == 0), 1, 0)

# 8. Environmental Risk Score
w1, w2, w3 = 0.4, 0.3, 0.3
test['env_risk'] = (w1 * test['current_weather'] + w2 * (1 / test['visibility']) + w3 * test['precipitation'])

# 9. Driver Stress Indicator
test['high_stress'] = np.where(test['heart_rate'] > 80, 1, 0)

# Optional: Encode heart_rate_bin numerically
mapping = {'low': 0, 'medium': 1, 'high': 2}
test['heart_rate_bin'] = test['heart_rate_bin'].map(mapping)
test['heart_rate_bin'] = test['heart_rate_bin'].astype(int)


test.shape


y_pred=model.predict(test)
y_pred = y_pred.flatten() if hasattr(y_pred, "flatten") else np.ravel(y_pred)

submission = pd.DataFrame({'id': range(len(y_pred)),'risk_level': y_pred})
submission.to_csv('sec_submission.csv', index=False)
submission.head()


!rm -rf /kaggle/working/*


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=2000,random_state=42, n_jobs=-1,max_depth=12)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")


y_pred=rf_model.predict(test)
y_pred = y_pred.flatten() if hasattr(y_pred, "flatten") else np.ravel(y_pred)

submission = pd.DataFrame({'id': range(len(y_pred)),'risk_level': y_pred})
submission.to_csv('three_submission.csv', index=False)
submission.head()


# !rm -rf /kaggle/working/*


# # -------------------------------------------------
# # 1. Import Libraries
# # -------------------------------------------------
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # -------------------------------------------------
# # 2. Data
# # -------------------------------------------------
# # Assuming df and test are already loaded
# X = df.drop(columns=['risk_level'])
# y = df['risk_level']  # classes 1,2,3,4

# # -------------------------------------------------
# # 3. Cross-Validation for Performance Estimate
# # -------------------------------------------------
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# fold_scores = []
# final_rf = None  # Placeholder for last trained model

# print("=== 10-FOLD CV (Random Forest Only) ===")
# for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
#     X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#     y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

#     rf = RandomForestClassifier(
#         n_estimators=1000,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         max_features='sqrt',
#         n_jobs=-1,
#         random_state=42,
#         class_weight='balanced',
#         max_depth=None
#     )

#     rf.fit(X_tr, y_tr)
#     val_pred = rf.predict(X_val)
#     acc = accuracy_score(y_val, val_pred)
#     fold_scores.append(acc)

#     print(f"Fold {fold:2d} | Accuracy: {acc:.5f}")
    
#     final_rf = rf  # Save last fold model for later use

# # -------------------------------------------------
# # 4. CV Summary
# # -------------------------------------------------
# print("\n=== FINAL CV RESULT ===")
# print(f"Random Forest Mean Accuracy: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")

# # -------------------------------------------------
# # 5. Predict on Test Set using last fold model
# # -------------------------------------------------
# test_pred = final_rf.predict(test)
# test_pred_labels = test_pred  # already 1,2,3,4

# # -------------------------------------------------
# # 6. Create Submission
# # -------------------------------------------------
# submission = pd.DataFrame({'id': range(len(test_pred_labels)),'risk_level': test_pred_labels})
# submission.to_csv('rf_only_full_train.csv', index=False)
# print("\nSUBMISSION READY: rf_only_full_train.csv")



submission.head()







