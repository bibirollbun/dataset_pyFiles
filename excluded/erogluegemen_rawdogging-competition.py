import os

import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

import optuna

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score

import warnings
warnings.simplefilter('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample Submission shape:", sample_submission.shape)
print("\nTrain columns:\n", train.columns)


print("Missing Values:\n", train.isnull().sum())

print("\nDescriptive Statistics:\n", train.describe())

plt.figure(figsize=(10, 5))
sns.histplot(train["Calories"], bins=50, kde=True)
plt.title("Distribution of Calories")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=train, x="Sex", y="Calories")
plt.title("Calories Burned by Sex")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
corr = train.drop(columns=["id"]).corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()



gender_counts = train['Sex'].value_counts()
labels = gender_counts.index
sizes = gender_counts.values
colors = ['#66b3ff', '#ff9999']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.05, 0))
plt.title("Gender Distribution")
plt.axis('equal')  
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=train, x="Age", y="Calories", hue="Sex", alpha=0.3)
plt.title("Calories vs Age by Sex")
plt.tight_layout()
plt.show()

features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
sns.pairplot(train.sample(5000), vars=features + ["Calories"], hue="Sex", corner=True)
plt.suptitle("Pairplot of Numerical Features", y=1.02)
plt.show()

print("\nGrouped Means by Sex:\n")
print(train.groupby("Sex")[features + ["Calories"]].mean())

plt.figure(figsize=(8, 5))
sns.regplot(data=train, x="Duration", y="Calories", scatter_kws={'alpha':0.1})
plt.title("Calories vs Duration")
plt.tight_layout()
plt.show()


train_fe = train.copy()
test_fe = test.copy()

le = LabelEncoder()
train_fe['Sex'] = le.fit_transform(train_fe['Sex'])  # male = 1, female = 0
test_fe['Sex'] = le.transform(test_fe['Sex'])

train_fe['BMI'] = train_fe['Weight'] / ((train_fe['Height'] / 100) ** 2)
test_fe['BMI'] = test_fe['Weight'] / ((test_fe['Height'] / 100) ** 2)

train_fe['Dur_Weight'] = train_fe['Duration'] * train_fe['Weight']
test_fe['Dur_Weight'] = test_fe['Duration'] * test_fe['Weight']

train_fe['HR_Duration'] = train_fe['Heart_Rate'] * train_fe['Duration']
test_fe['HR_Duration'] = test_fe['Heart_Rate'] * test_fe['Duration']

train_fe['Temp_Duration'] = train_fe['Body_Temp'] * train_fe['Duration']
test_fe['Temp_Duration'] = test_fe['Body_Temp'] * test_fe['Duration']

train_fe['Age_BMI'] = train_fe['Age'] * train_fe['BMI']
test_fe['Age_BMI'] = test_fe['Age'] * test_fe['BMI']

train_fe['Speed'] = train_fe['Height'] / train_fe['Duration']
test_fe['Speed'] = test_fe['Height'] / test_fe['Duration']

train_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
test_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
train_fe.fillna(0, inplace=True)
test_fe.fillna(0, inplace=True)

print(train_fe.head())


features = [
    'Sex', 'Age', 'Height', 'Weight', 'Duration',
    'Heart_Rate', 'Body_Temp', 'BMI', 'Dur_Weight',
    'HR_Duration', 'Temp_Duration', 'Age_BMI', 'Speed'
]

target = 'Calories'

X = train_fe[features]
y = train_fe[target]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training shape:", X_train.shape)
print("Validation shape:", X_valid.shape)


rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
val_preds = rf.predict(X_valid)

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

rmsle_score = rmsle(y_valid, val_preds)
print(f"Validation RMSLE: {rmsle_score:.4f}")


lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=70,
    min_child_samples=30,
    feature_fraction=0.8,
    random_state=42,
    n_jobs=-1,
    device='gpu'
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

val_preds_lgb = lgb_model.predict(X_valid)
print("Validation RMSLE (LGBM):", rmsle(y_valid, val_preds_lgb))


X = train_fe[features]
y = train_fe["Calories"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmsle_scores = []
test_preds = np.zeros(len(test_fe))


for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n Foldieeee {fold + 1}")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=70,
        min_child_samples=30,
        feature_fraction=0.8,
        random_state=42,
        n_jobs=-1,
        device='gpu'
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    val_pred = model.predict(X_val)
    rmsle_score = np.sqrt(np.mean((np.log1p(val_pred) - np.log1p(y_val))**2))
    fold_rmsle_scores.append(rmsle_score)


    test_preds += model.predict(test_fe[features]) / kf.n_splits

print("\n Average RMSLE across folds fr!:", np.mean(fold_rmsle_scores))


max_depths = [8, 10, 12]
num_leaves_list = [31, 50, 70]
min_child_samples_list = [10, 20, 30]
feature_fractions = [0.8, 0.9, 1.0]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score = float("inf")
best_params = None

for max_depth in max_depths:
    for num_leaves in num_leaves_list:
        for min_child_samples in min_child_samples_list:
            for feature_fraction in feature_fractions:
                
                fold_scores = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr = np.log1p(y.iloc[train_idx])
                    y_val = np.log1p(y.iloc[val_idx])

                    model = lgb.LGBMRegressor(
                        n_estimators=1000,
                        learning_rate=0.05,
                        max_depth=max_depth,
                        num_leaves=num_leaves,
                        min_child_samples=min_child_samples,
                        feature_fraction=feature_fraction,
                        random_state=42,
                        n_jobs=-1
                    )

                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric='rmse',
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
                    )

                    preds_val = np.expm1(model.predict(X_val))
                    rmsle_score = np.sqrt(np.mean((np.log1p(preds_val) - np.log1p(np.expm1(y_val)))**2))
                    fold_scores.append(rmsle_score)

                avg_score = np.mean(fold_scores)
                print(f"Depth: {max_depth}, Leaves: {num_leaves}, MinChild: {min_child_samples}, FF: {feature_fraction} -> RMSLE: {avg_score:.5f}")

                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        "max_depth": max_depth,
                        "num_leaves": num_leaves,
                        "min_child_samples": min_child_samples,
                        "feature_fraction": feature_fraction
                    }
                    print("New best score!")

print(f"\nBest LGBM Params: {best_params}")
print(f"Best RMSLE: {best_score:.5f}")


X = train_fe[features]
y = train_fe["Calories"] 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmsle_scores = []
test_preds_lgb_tuned = np.zeros(len(test_fe))  

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1} â€” Tuned LGBM")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=70,
        min_child_samples=30,
        feature_fraction=0.8,
        random_state=42,
        n_jobs=-1,
        device='gpu'  
    )

    model.fit(
        X_tr, y_tr_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    val_pred_log = model.predict(X_val)
    val_pred = np.expm1(val_pred_log)

    rmsle_score = np.sqrt(np.mean((np.log1p(val_pred) - np.log1p(y_val))**2))
    fold_rmsle_scores.append(rmsle_score)

    test_preds_lgb_tuned += np.expm1(model.predict(test_fe[features])) / kf.n_splits

print("\nTuned LGBM Log-Target Average RMSLE:", np.mean(fold_rmsle_scores))


submission = pd.DataFrame({
    "id": test_fe["id"],
    "Calories": test_preds_lgb_tuned
})
submission.to_csv("submission_log_target_final.csv", index=False)


max_depths = [6, 8, 10]
min_child_weights = [1, 3, 5]
subsamples = [0.8, 0.9, 1.0]
colsample_bytrees = [0.8, 0.9, 1.0]

best_score = float("inf")
best_params = None

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for max_depth in max_depths:
    for min_child_weight in min_child_weights:
        for subsample in subsamples:
            for colsample_bytree in colsample_bytrees:

                fold_scores = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr = np.log1p(y.iloc[train_idx])
                    y_val = np.log1p(y.iloc[val_idx])

                    model = XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.05,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=42,
                        n_jobs=-1,
                        tree_method='gpu_hist' 
                    )

                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric='rmse',
                        early_stopping_rounds=50,
                        verbose=0
                    )

                    preds_val = np.expm1(model.predict(X_val))
                    rmsle_score = np.sqrt(np.mean((np.log1p(preds_val) - np.log1p(np.expm1(y_val)))**2))
                    fold_scores.append(rmsle_score)

                avg_score = np.mean(fold_scores)
                print(f"Depth: {max_depth}, ChildWeight: {min_child_weight}, Subsample: {subsample}, Colsample: {colsample_bytree} -> RMSLE: {avg_score:.5f}")

                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        "max_depth": max_depth,
                        "min_child_weight": min_child_weight,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree
                    }
                    print("New best score!")

print(f"\nBest XGBoost Params: {best_params}")
print(f"Best RMSLE: {best_score:.5f}")


X = train_fe[features]
y = train_fe["Calories"] 

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmsle_scores = []
test_preds_xgb = np.zeros(len(test_fe))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n Fold {fold + 1} - XGBoost")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    y_tr_log = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        tree_method='gpu_hist'
    )

    model.fit(
        X_tr, y_tr_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric='rmse',
        early_stopping_rounds=50,
        verbose=100
    )

    val_pred_log = model.predict(X_val)
    val_pred = np.expm1(val_pred_log)

    rmsle_score = np.sqrt(np.mean((np.log1p(val_pred) - np.log1p(y_val))**2))
    fold_rmsle_scores.append(rmsle_score)

    test_preds_xgb += np.expm1(model.predict(test_fe[features])) / kf.n_splits

print("\n XGBoost Log-Target Average RMSLE across folds:", np.mean(fold_rmsle_scores))


final_preds = 0.5 * test_preds_lgb_tuned + 0.5 * test_preds_xgb

submission = pd.DataFrame({
    "id": test_fe["id"],
    "Calories": final_preds
})
submission.to_csv("submission_lgbm_xgb_blend.csv", index=False)


depths = [8, 9, 10]
learning_rates = [0.03, 0.05, 0.07]
l2_leaf_regs = [1, 3, 5]
bagging_temps = [0.5, 1.0]

best_score = float("inf")
best_params = None

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for depth in depths:
    for lr in learning_rates:
        for l2 in l2_leaf_regs:
            for temp in bagging_temps:
                
                fold_scores = []

                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr = np.log1p(y.iloc[train_idx])
                    y_val = np.log1p(y.iloc[val_idx])

                    model = CatBoostRegressor(
                        iterations=1000,
                        learning_rate=lr,
                        depth=depth,
                        l2_leaf_reg=l2,
                        bagging_temperature=temp,
                        loss_function='RMSE',
                        verbose=0,
                        random_seed=42,
                        task_type="GPU"
                    )

                    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
                    preds_val = np.expm1(model.predict(X_val))

                    rmsle = np.sqrt(np.mean((np.log1p(preds_val) - np.log1p(np.expm1(y_val)))**2))
                    fold_scores.append(rmsle)

                avg_score = np.mean(fold_scores)
                print(f"Depth: {depth}, LR: {lr}, L2: {l2}, Temp: {temp} -> RMSLE: {avg_score:.5f}")

                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        "depth": depth,
                        "learning_rate": lr,
                        "l2_leaf_reg": l2,
                        "bagging_temperature": temp
                    }
                    print("New best score!")

print(f"\nğŸ�† Best CatBoost Params: {best_params}")
print(f"âœ… Best RMSLE: {best_score:.5f}")


cat_preds_tuned = np.zeros(len(test_fe))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmsle_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1} â€” Tuned CatBoost")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr = np.log1p(y.iloc[train_idx])
    y_val = np.log1p(y.iloc[val_idx])

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=10,
        l2_leaf_reg=1,
        bagging_temperature=0.5,
        loss_function='RMSE',
        verbose=100,
        random_seed=42,
        task_type='GPU'
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )

    val_pred_log = model.predict(X_val)
    val_pred = np.expm1(val_pred_log)

    rmsle = np.sqrt(np.mean((np.log1p(val_pred) - np.log1p(np.expm1(y_val)))**2))
    fold_rmsle_scores.append(rmsle)

    cat_preds_tuned += np.expm1(model.predict(test_fe[features])) / kf.n_splits

print("\nTuned CatBoost Log-Target Average RMSLE:", np.mean(fold_rmsle_scores))


final_preds = (
    0.3 * test_preds +
    0.2 * test_preds_xgb +
    0.5 * cat_preds_tuned
)

submission = pd.DataFrame({
    "id": test_fe["id"],
    "Calories": final_preds
})

submission.to_csv("submission_all_tuned_final2.csv", index=False)


final_preds = (
    0.3 * test_preds +
    0.2 * test_preds_xgb +
    0.5 * cat_preds_tuned
)

submission = pd.DataFrame({
    "id": test_fe["id"],
    "Calories": final_preds
})

submission.to_csv("submission_b.csv", index=False)

final_preds = (
    0.25 * test_preds +
    0.25 * test_preds_xgb +
    0.5 * cat_preds_tuned
)

submission = pd.DataFrame({
    "id": test_fe["id"],
    "Calories": final_preds
})

submission.to_csv("submission_c.csv", index=False)

