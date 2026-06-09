!ls /kaggle/input/playground-series-s5e9


import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor,VotingRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve


train=pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample=pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


y = train["BeatsPerMinute"]

x = train.drop(["id", "BeatsPerMinute","VocalContent","TrackDurationMs","LivePerformanceLikelihood"], axis=1)
# from data test
x_test_kaggle= test.drop(["id","VocalContent","LivePerformanceLikelihood", "TrackDurationMs"], axis=1)
test_ids= test["id"]



x_train, x_valid, y_train, y_valid = train_test_split(
    x, y, test_size=0.2, random_state=42
)


 rf= RandomForestRegressor(
    n_estimators=500,       # banyak pohon, lebih stabil
    max_depth=15,           # batasi kedalaman biar ga overfit
    min_samples_split=10,   # minimal data untuk split
    min_samples_leaf=4,     # minimal data di leaf
    max_features='sqrt',    # ambil sqrt fitur tiap split, biar random
    n_jobs=-1,
    random_state=55
    )



xgb= XGBRegressor(
    n_estimators=1000,       # banyak iterasi
    learning_rate=0.05,      # lambat tapi stabil
    max_depth=13,             # ga terlalu dalam
    min_child_weight=5,      # minimal data di leaf
    subsample=0.8,           # pakai 80% data tiap tree
    colsample_bytree=0.8,    # pakai 80% fitur tiap tree
    gamma=1,                 # regularisasi
    tree_method='hist',      # lebih cepat di dataset besar
    n_jobs=-1,
    random_state=55
    ) 


lgbm= LGBMRegressor(
   n_estimators=1000,
    learning_rate=0.05,
    num_leaves=50,            # jumlah leaf per tree
    min_child_samples=30,     # minimal data tiap leaf
    subsample=0.8,            # pakai sebagian data
    colsample_bytree=0.8,     # pakai sebagian fitur
    reg_alpha=0.5,            # regularisasi L1
    reg_lambda=0.5,           # regularisasi L2
    n_jobs=-1,
    random_state=42
    )


stack_model = StackingRegressor(
    estimators=[
        ('rf', rf), 
        ('xgb', xgb),
        ('lgbm', lgbm)
        ],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1
)

stack_model.fit(x_train,y_train)


vote_model = VotingRegressor([
    ('rf', rf),
    ('xgb', xgb),
    ('lgbm', lgbm)
], n_jobs=-1)


val_pred = stack_model.predict(x_valid)
mae = mean_absolute_error(y_valid, val_pred)


mape = np.mean(np.abs((y_valid - val_pred) / y_valid)) * 100
accuracy = 100 - mape

print("\nModel Performance Metrics:")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(8,5))
plt.scatter(y_valid, val_pred, alpha=0.6)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
plt.xlabel("Nilai Aktual")
plt.ylabel("Nilai Prediksi")
plt.title("Perbandingan Nilai Aktual vs Prediksi")
plt.show()


# train_sizes, train_scores, val_scores = learning_curve(
#     stack_model, x_train, y_train,
#     cv=5,
#     scoring='r2',
#     n_jobs=-1,
#     train_sizes=np.linspace(0.1, 1.0, 10)
# )

# plt.figure(figsize=(8,5))
# plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train R²')
# plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation R²')
# plt.xlabel('Jumlah Data Latih')
# plt.ylabel('R² Score')
# plt.title('Learning Curve Ensemble (Stacking)')
# plt.legend()
# plt.show()


# xgb.fit(
#     x_train, y_train,
#     eval_set=[(x_train, y_train), (x_test, y_test)],
#     eval_metric='rmse',
#     verbose=False
# )

# results = xgb.evals_result()

# plt.figure(figsize=(8,5))
# plt.plot(results['validation_0']['rmse'], label='Train RMSE')
# plt.plot(results['validation_1']['rmse'], label='Validation RMSE')
# plt.xlabel('Jumlah Pohon (Iteration)')
# plt.ylabel('RMSE')
# plt.title('XGBoost: Train vs Validation')
# plt.legend()
# plt.show()


# lgbm.fit(
#     x_train, y_train,
#     eval_set=[(x_train, y_train), (x_test, y_test)],
#     eval_metric='rmse',
#     verbose=False
# )

# results = lgbm.evals_result_

# plt.figure(figsize=(8,5))
# plt.plot(results['training']['rmse'], label='Train RMSE')
# plt.plot(results['valid_1']['rmse'], label='Validation RMSE')
# plt.xlabel('Iteration (n_estimators)')
# plt.ylabel('RMSE')
# plt.title('LightGBM: Train vs Validation')
# plt.legend()
# plt.show()


test_pred = stack_model.predict(x_test_kaggle)


submission = pd.DataFrame({
    "id": test_ids,
    "BeatsPerMinute": test_pred
})

submission.to_csv("submission.csv", index=False)
print("✅ File submission.csv ready to submit in Kaggle")

