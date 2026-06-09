import pandas as pd
import numpy as np


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

import shap
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


train=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
print("Train Shape =",train.shape)
test=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
print("Test Shape =",test.shape)
sample=pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
print("Sample Submission Shape =",sample.shape)


numerical_features = ['num_reported_accidents', 'speed_limit','num_lanes' , 'curvature']
categorical_features = ['school_season', 'holiday',"time_of_day","public_road","road_signs_present","weather","lighting","road_type"]

for col in numerical_features:
    plt.figure(figsize=(7, 3))
    plt.hist(train[col].dropna(), bins=10, alpha=0.5, label='Train', density=True)
    plt.hist(test[col].dropna(), bins=10, alpha=0.5, label='Test', density=True)
    plt.title(f'{col} distribution (Train/Test)')
    plt.xlabel(col)
    plt.legend()
    plt.show()

for col in categorical_features:
    plt.figure(figsize=(7, 3))
    train_counts = train[col].value_counts(normalize=True)
    test_counts = test[col].value_counts(normalize=True)
    
    all_categories = set(train_counts.index).union(set(test_counts.index))
    train_plot = [train_counts.get(cat, 0) for cat in all_categories]
    test_plot = [test_counts.get(cat, 0) for cat in all_categories]

    width = 0.35
    idx = range(len(all_categories))
    plt.bar(idx, train_plot, width=width, label='Train', alpha=0.5)
    plt.bar([i + width for i in idx], test_plot, width=width, label='Test', alpha=0.5)
    plt.xticks([i + width/2 for i in idx], list(all_categories))
    plt.title(f'{col} distribution (Train/Test)')
    plt.xlabel(col)
    plt.legend()
    plt.show()


df = train.copy()
def plot_3d_surface(x_col, y_col, z_col="accident_risk", sample_size=20000):
    df_small = df.sample(sample_size, random_state=42)

    x = df_small[x_col]
    y = df_small[y_col]
    z = df_small[z_col]

    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method="linear")

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xi, yi, zi, cmap="viridis", edgecolor="none", alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    plt.title(f"3D Surface: {x_col} vs {y_col} vs {z_col}")
    plt.colorbar(surf, shrink=0.5, aspect=10, label=z_col)
    plt.show()

plot_3d_surface("num_lanes", "curvature", "accident_risk")
plot_3d_surface("num_lanes", "speed_limit", "accident_risk")
plot_3d_surface("num_lanes", "num_reported_accidents", "accident_risk")
plot_3d_surface("curvature", "num_reported_accidents", "accident_risk")
plot_3d_surface("curvature", "speed_limit", "accident_risk")
plot_3d_surface("speed_limit", "num_reported_accidents", "accident_risk")


RMV = ["id","accident_risk"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train,test],axis=0,ignore_index=True) 
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)

    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "accident_risk"]
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = train.loc[test_index, "accident_risk"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        n_estimators=1535,
        max_depth=8,
        learning_rate=0.009391286763566058,
        subsample=0.8964030011147961,
        colsample_bytree=0.7892025840865715,
        reg_alpha=0.1786345577524364,
        reg_lambda=0.028911509885374744,
        min_child_weight=3,
        tree_method="hist",
        enable_categorical=True,
        device="cuda",
        early_stopping_rounds=25,
        verbosity=1
    )

    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=200
    )

    oof_xgb[test_index] = model_xgb.predict(x_valid)
    pred_xgb += model_xgb.predict(x_test)

pred_xgb /= FOLDS
rmse = np.sqrt(mean_squared_error(train['accident_risk'], oof_xgb))
print(f"\nFinal CV RMSE: {rmse:.4f}")


feature_importance = model_xgb.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(8, 4))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.show()



explainer = shap.TreeExplainer(model_xgb, feature_perturbation="tree_path_dependent", model_output="raw")
shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values, x_test)


sample["accident_risk"] = pred_xgb
sample.to_csv("submission.csv", index=False)
sample.head()

