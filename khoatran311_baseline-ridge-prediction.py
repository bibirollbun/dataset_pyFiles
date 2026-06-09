import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import optuna
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)


sample_submission.head()


train.head()


test.head()


plt.figure(figsize=(20,8))
for i,col in enumerate(train.columns):
    plt.subplot(2, 5, i+1)
    sea.histplot(train[col])
    plt.title(col)


plt.figure(figsize=(16,4))

M_corr = train.corr(method="spearman")
U_mask = np.triu(M_corr)
plt.subplot(121)
sea.heatmap(M_corr, mask=U_mask, 
            vmin=-.5, vmax=.3, 
            linecolor="black", linewidths=.25,
            square=True
           )
plt.title("Spearman Correlation Matrix")



M_corr = train.corr(method="pearson")
U_mask = np.triu(M_corr)
plt.subplot(122)
sea.heatmap(M_corr, mask=U_mask, 
            vmin=-.5, vmax=.3, 
            linecolor="black", linewidths=.25,
            square=True
           )
plt.title("Pearson Correlation Matrix")

plt.show()


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


## RMSE Metric
RMSE = lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred)**2))

## Splitted dataset
X = train.copy()
y = X.pop("BeatsPerMinute")

standard_scaler = StandardScaler()


def cross_validate(model, X, y, n_splits=10, scaler=None):
    ## Converts to numpy arrays
    X_npy = np.asarray(X)
    y_npy = np.asarray(y)
        
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=3126)
    y_oof = np.zeros_like(y_npy, dtype=float)
    rmse  = []
    history = {}
        
    for train_idx,val_idx in kfold.split(X_npy):
        X_train, y_train = X_npy[train_idx], y_npy[train_idx]
        X_val,   y_val   = X_npy[val_idx],   y_npy[val_idx]

        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
        
        ## Fits cloned model
        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        ## Gets OOF prediction and stores it
        y_pred = cloned_model.predict(X_val)
        y_oof[val_idx] = y_pred
        ## Stores metric
        rmse.append(RMSE(y_val, y_pred))

    history["y_oof"] = y_oof
    history["rmse"]  = np.array(rmse)
    return history


lr = LinearRegression(n_jobs=-1)
history_lr = cross_validate(lr, X, y, n_splits=10, scaler=standard_scaler)

ridge = Ridge(alpha=1, random_state=3126)
history_ridge = cross_validate(ridge, X, y, n_splits=10, scaler=standard_scaler)

lasso = Lasso(alpha=1, random_state=3126)
history_lasso = cross_validate(lasso, X, y, n_splits=10, scaler=standard_scaler)


sea.scatterplot(x=np.arange(1,11), y=history_lr["rmse"])
sea.lineplot(x=np.arange(1,11), y=history_lr["rmse"], label="LinearRegression")

sea.scatterplot(x=np.arange(1,11), y=history_ridge["rmse"])
sea.lineplot(x=np.arange(1,11), y=history_ridge["rmse"], label="Ridge")

sea.scatterplot(x=np.arange(1,11), y=history_lasso["rmse"])
sea.lineplot(x=np.arange(1,11), y=history_lasso["rmse"], label="Lasso")

plt.xlabel("K-Fold Iteration")
plt.ylabel("RMSE Score")
plt.title("Cross Validated RMSE Score of Linear Models")
plt.show()


print(f"Linear Regression RMSE: {history_lr['rmse'].mean():.3f} ± {history_lr['rmse'].std():.3f}")
print(f"Ridge RMSE: {history_ridge['rmse'].mean():.3f} ± {history_ridge['rmse'].std():.3f}")
print(f"Lasso RMSE: {history_lasso['rmse'].mean():.3f} ± {history_lasso['rmse'].std():.3f}")


ridge = Ridge(alpha=1.0, random_state=3126)

X_copy = X.copy()
X_copy = standard_scaler.fit_transform(X_copy)

ridge.fit(X_copy, y)


features = X.columns
coefs = ridge.coef_

ridge_coefs = pd.DataFrame({"feature":features, "ridge_coef":coefs})
ridge_coefs.sort_values(by="ridge_coef", ascending=False)


oof_data = pd.DataFrame(
    {"lr_oof":history_lr["y_oof"], 
     "ridge_oof":history_ridge["y_oof"], 
     "lasso_oof":history_lasso["y_oof"]}
)

oof_data.corr(method="pearson")


# def objective(trial):
#     ## Parameter to tune
#     alpha = trial.suggest_float("alpha", .005, 100, log=True)

#     ## Cross validates model
#     ridge = Ridge(alpha=alpha, random_state=3126)
#     standard_scaler = StandardScaler()
#     history_ridge = cross_validate(ridge, X, y, n_splits=10, scaler=standard_scaler)
#     return history_ridge["rmse"].mean()
    
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)

# # Best parameters
# print("Best trial:")
# trial = study.best_trial
# print("RMSE:", trial.value)
# print("Params:", trial.params)


# RMSE: 26.466253127110697
# Params: {'alpha': 99.82924504600531}


## Uses ridge with alpha= 1.0 instead since no significant changes in score occurred
sample_submission["BeatsPerMinute"] = ridge.predict(standard_scaler.transform(test))


sample_submission.to_csv('submission.csv', index=False)

