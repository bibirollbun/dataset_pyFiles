import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb




from sklearn.metrics import mean_squared_error
import numpy as np

root_mean_squared_error = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))



train = pd.read_csv("/kaggle/input/molecular-machine-learning/train.csv")

test = pd.read_csv("/kaggle/input/molecular-machine-learning/test.csv")


plt.hist(train["T80"], bins=20)
plt.title("Original T80 Distribution")
plt.xlabel("T80")
plt.ylabel("Frequency")
plt.show()

print("Skewness:", train["T80"].skew())



train["T80_log"] = np.log1p(train["T80"])



# Choose numeric features (excluding T80 and Batch_ID)
feature_cols = [col for col in train.columns if col not in ["Batch_ID", "T80", "T80_log", "Smiles"]]

# Groupby stats
agg_funcs = ['mean', 'std', 'min', 'max']
group_stats = train.groupby("Batch_ID")[feature_cols].agg(agg_funcs)

# Flatten MultiIndex
group_stats.columns = [f"{feat}_{stat}" for feat, stat in group_stats.columns]

# Merge back into train
train = train.merge(group_stats, on="Batch_ID", how="left")
test = test.merge(group_stats, on="Batch_ID", how="left")



train["gap"] = train["LUMO(eV)"] - train["HOMO(eV)"]
test["gap"] = test["LUMO(eV)"] - test["HOMO(eV)"]



train["gap_p1"] = train["LUMOp1(eV)"] - train["HOMO(eV)"]
test["gap_p1"] = test["LUMOp1(eV)"] - test["HOMO(eV)"]



train["lumo_width"] = train["LUMOp1(eV)"] - train["LUMO(eV)"]
test["lumo_width"] = test["LUMOp1(eV)"] - test["LUMO(eV)"]

train["homo_width"] = train["HOMO(eV)"] - train["HOMOm1(eV)"]
test["homo_width"] = test["HOMO(eV)"] - test["HOMOm1(eV)"]



train["lumo_mean"] = 0.5 * (train["LUMO(eV)"] + train["LUMOp1(eV)"])
test["lumo_mean"] = 0.5 * (test["LUMO(eV)"] + test["LUMOp1(eV)"])



train["smiles_len"] = train["Smiles"].str.len()
test["smiles_len"] = test["Smiles"].str.len()

train["num_branches"] = train["Smiles"].str.count("\(")
test["num_branches"] = test["Smiles"].str.count("\(")



for atom in ['C', 'N', 'O', 'F', 'S']:
    train[f"count_{atom}"] = train["Smiles"].str.count(atom)
    test[f"count_{atom}"] = test["Smiles"].str.count(atom)



molecule_size = train.groupby("Batch_ID").size().rename("molecule_size")
train = train.merge(molecule_size, on="Batch_ID", how="left")
test = test.merge(molecule_size, on="Batch_ID", how="left")



non_features = ["Batch_ID", "T80", "T80_log", "Smiles", "T80_pred"]  # <-- added T80_pred
features = [col for col in train.columns if col not in non_features]



X = train[features]
X_test = test[features]
y = train["T80_log"]  
groups = train["Batch_ID"]


from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
import warnings
warnings.filterwarnings("ignore")

logo = LeaveOneGroupOut()

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for train_idx, val_idx in logo.split(X, y, groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_child_samples=1,
        colsample_bytree=1.0,
        subsample=1.0,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],)
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / logo.get_n_splits(groups=groups)
rmse = root_mean_squared_error(y, oof_preds)
print(f"LOG CV RMSE: {rmse:.4f}")



train["T80_pred"] = np.expm1(oof_preds)
test["T80"] = np.expm1(test_preds)



sample_submission=pd.read_csv('/kaggle/input/molecular-machine-learning/sample_submission.csv')
submission = sample_submission.copy()
submission["T80"] = test["T80"]
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved.")





