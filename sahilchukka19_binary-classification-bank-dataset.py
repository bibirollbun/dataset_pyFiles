import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
import optuna


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


#  =================Checking missing values for both the datasets========================
missing_train = train_df.isnull().mean() * 100
missing_test = test_df.isnull().mean() * 100
print(f"\nMissing Values in Training Dataset : {missing_train}")
print(f"\nMissing Values in Testing Dataset : {missing_test}")


#================= Plot Distributions of Numerical Columns of Training Dataset =======================
num_cols_train = train_df.select_dtypes(include=['int64','float64']).columns

for col in num_cols_train:
    # Clean column: replace inf/-inf with NaN, then drop
    col_data = train_df[col].replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(6,4))
    sns.histplot(col_data, kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()



#================= Correaltions of Numerical Columns of Training & Testing Datasets=======================

plt.figure(figsize=(10,8))
corr_train = train_df[num_cols_train].corr()
sns.heatmap(corr_train, annot=False,cmap="coolwarm",center=0)
plt.title("Correlation Heatmap of Training Dataset")
plt.show()


#================Show top categories and counts for categorical features==================
cat_cols_train = train_df.select_dtypes(include=['object']).columns
for cols in cat_cols_train:
    print(f"\n{cols} - unique: {train_df[cols].nunique()}")
    print(train_df[cols].value_counts().head(10))


X_train = train_df.drop(["y"], axis = 1)
y_train = train_df["y"]
X_test = test_df


cat_cols = X_train.select_dtypes(include=['object']).columns
num_cols = X_train.select_dtypes(include=['int64','float64']).columns


X_train_encoded = pd.get_dummies(X_train[cat_cols])
X_test_encoded = pd.get_dummies(X_test[cat_cols])


X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded, join="left", axis=1, fill_value=0)


scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[num_cols])
X_test_num_scaled = scaler.fit_transform(X_test[num_cols])

# Convert scaled arrays back to DataFrame with column names
X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=num_cols, index=X_train.index)
X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=num_cols, index=X_test.index)



X_train_final = pd.concat([X_train_num_scaled, X_train_encoded], axis=1)
X_test_final = pd.concat([X_test_num_scaled, X_test_encoded], axis=1)


model = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
            random_state=42, use_label_encoder=False, n_jobs=-1
        )
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scores = cross_val_score(model, X_train_final, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.4f} ± {scores.std():.4f}")




# CV splitter
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    # Define hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0)
    }

    # Define model
    model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        **params
    )

    # Cross-validation AUC
    scores = cross_val_score(
        model,
        X_train_final, 
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    return scores.mean()


# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best parameters:", study.best_params)
print("Best CV AUC:", study.best_value)



best_params = study.best_params
final_model = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric="auc",
    **best_params
)

final_model.fit(X_train_final, y_train)


y_test_pred = final_model.predict_proba(X_test_final)[:, 1]


submission = pd.DataFrame({
    "id": test_df["id"],       
    "y": y_test_pred
})

submission.to_csv("submission.csv", index=False)

