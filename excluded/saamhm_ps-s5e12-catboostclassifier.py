import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier


train_df=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


target = "diagnosed_diabetes"

x = train_df.drop([target, "id"], axis=1)
y = train_df[target]

test_x = test_df.drop("id", axis=1)



train_df.columns


cat_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history"
]

cat_features = [x.columns.get_loc(col) for col in cat_cols]



x


def add_engineered_features(train_df, test_df):
    # 1. Comorbidity count
    risk_cols = ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]
    train_df["comorbidity_count"] = train_df[risk_cols].sum(axis=1)
    test_df["comorbidity_count"] = test_df[risk_cols].sum(axis=1)

    # 2. Age × BMI interaction
    train_df["age_bmi_interaction"] = train_df["age"] * train_df["bmi"]
    test_df["age_bmi_interaction"] = test_df["age"] * test_df["bmi"]

    # 3. Triglycerides-to-BMI ratio
    train_df["triglycerides_bmi_ratio"] = train_df["triglycerides"] / (train_df["bmi"] + 1e-5)
    test_df["triglycerides_bmi_ratio"] = test_df["triglycerides"] / (test_df["bmi"] + 1e-5)

    # 4. Pulse pressure (systolic - diastolic)
    train_df["pulse_pressure"] = train_df["systolic_bp"] - train_df["diastolic_bp"]
    test_df["pulse_pressure"] = test_df["systolic_bp"] - test_df["diastolic_bp"]

    return train_df, test_df




x,test_x=add_engineered_features(x,test_x)


print(f'train shape : {x.shape,y.shape}\n test shape : {test_x.shape}')


test_x.info()


x.info()


cat_model = CatBoostClassifier(
    iterations=1800,
    learning_rate=0.005,
    depth=7,              # slightly smaller → less overfit
    l2_leaf_reg=8,
    bagging_temperature=0.6,
    random_strength=1.2,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=0
)


# lgb_model=LGBMClassifier(
#         n_estimators=1800,
#         learning_rate=0.03,
#         max_depth=7,
#         num_leaves=31,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         class_weight="balanced",
#         random_state=42
#     )


# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cat_oof = np.zeros(len(x))
# lgb_oof = np.zeros(len(x))

# cat_test_preds = np.zeros(len(test_x))
# lgb_test_preds = np.zeros(len(test_x))

# for fold, (train_idx, val_idx) in enumerate(cv.split(x, y)):
#     print(f"Fold {fold+1}")

#     x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
#     y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
# # -------- CatBoost --------
    
#     cat_model.fit(
#         x_tr, y_tr,
#         cat_features=cat_features,
#         eval_set=(x_val, y_val),
#         use_best_model=True
#     )

#     cat_oof[val_idx] = cat_model.predict_proba(x_val)[:, 1]
#     cat_test_preds += cat_model.predict_proba(test_x)[:, 1]

# # -------- LightGBM --------
#     lgb_model.fit(x_tr, y_tr)

#     lgb_oof[val_idx] = lgb_model.predict_proba(x_val)[:, 1]
#     lgb_test_preds += lgb_model.predict_proba(test_x)[:, 1]

# cat_test_preds /= cv.n_splits
# lgb_test_preds /= cv.n_splits



# final_oof = 0.6 * cat_oof + 0.4 * lgb_oof
# final_test_preds = 0.6 * cat_test_preds + 0.4 * lgb_test_preds



# print("Ensemble CV AUC:", roc_auc_score(y, final_oof))



cat_model.fit(
    x,y,cat_features=cat_features,verbose=0
)


test_proba =cat_model.predict_proba(test_x)[:, 1]

submission = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": test_proba
})

submission.to_csv("submission.csv", index=False)




