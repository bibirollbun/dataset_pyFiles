import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')



# Load and Combine Data
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test["accident_risk"] = 0.5  # Initialize with placeholder value


train.head().style.background_gradient(cmap='gist_rainbow_r')


test.head().style.background_gradient(cmap='gist_rainbow_r')


# Load synthetic road accident datasets (multiple sizes)
synthetic = []
for size in [2, 10, 100]:
    df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{size}k.csv")
    synthetic.append(df)
synthetic = pd.concat(synthetic, axis=0)

# Align columns and add new IDs for synthetic data
synthetic["id"] = np.arange(len(synthetic)) + test["id"].max() + 1
synthetic = synthetic[train.columns]

# Combine all data for unified preprocessing
combined = pd.concat([train, test, synthetic], axis=0, ignore_index=True)



df.head().style.background_gradient(cmap='gist_rainbow_r')


df.describe().style.background_gradient(cmap='tab20c')


# Feature Engineering

FEATURES = list(synthetic.columns[1:-1])
TARGET = synthetic.columns[-1]

# Custom feature 'y' based on road & weather conditions
def road_risk(X):
    return (
        0.3 * X["curvature"] +
        0.2 * (X["lighting"] == "night").astype(int) +
        0.1 * (X["weather"] != "clear").astype(int) +
        0.2 * (X["speed_limit"] >= 60).astype(int) +
        0.1 * (X["num_reported_accidents"] > 2).astype(int)
    )

# Smoothed clipping using normal distribution
def clipped(func):
    def clip_f(X):
        mu = func(X)
        sigma = 0.05
        a, b = -mu / sigma, (1 - mu) / sigma
        Phi_a, Phi_b = stats.norm.cdf(a), stats.norm.cdf(b)
        phi_a, phi_b = stats.norm.pdf(a), stats.norm.pdf(b)
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
    return clip_f

combined["y"] = clipped(road_risk)(combined)

# Add feature interaction: curvature * speed_limit
combined["curv_speed_interaction"] = combined["curvature"] * combined["speed_limit"]
FEATURES.append("y")
FEATURES.append("curv_speed_interaction")



# Handle Categorical Data

CATS, NUMS = [], []
for col in FEATURES:
    if combined[col].dtype == "object":
        CATS.append(col)
    else:
        NUMS.append(col)

# Factorize (encode) categorical columns
for col in CATS:
    combined[col], _ = combined[col].factorize()
    combined[col] = combined[col].astype("int32")



# Split Data Back

train = combined.iloc[:len(train)]
test = combined.iloc[len(train):len(train) + len(test)]
synthetic = combined.iloc[-len(synthetic):]



# Target Encoding

TE_features = []
for col in FEATURES:
    te_map = synthetic.groupby(col)[TARGET].mean()
    te_name = f"TE_{col}"
    train = train.merge(te_map.rename(te_name), on=col, how="left")
    test = test.merge(te_map.rename(te_name), on=col, how="left")
    TE_features.append(te_name)


# Model Training (XGBoost)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.0075,
    "max_depth": 7,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "lambda": 1.5,
    "alpha": 0.3,
    "n_jobs": -1,
    "seed": 42,
    "tree_method": "hist",
    "device": "cuda",
}

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=2025)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

print("\nğŸš€ Training model with 7-Fold Cross Validation...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print(f"ğŸ“‚ Fold {fold+1}/{FOLDS}")
    
    X_train = train.iloc[train_idx][FEATURES + TE_features]
    y_train = train.iloc[train_idx][TARGET] - train.iloc[train_idx]["y"]

    X_val = train.iloc[val_idx][FEATURES + TE_features]
    y_val = train.iloc[val_idx][TARGET] - train.iloc[val_idx]["y"]
    y_val_base = train.iloc[val_idx]["y"].values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(test[FEATURES + TE_features])

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100_000,
        evals=[(dtrain, "Train"), (dval, "Valid")],
        early_stopping_rounds=200,
        verbose_eval=False
    )

    oof_preds[val_idx] = model.predict(dval) + y_val_base
    test_preds += (model.predict(dtest) + test["y"].values) / FOLDS



# Evaluate Model

rmse_model = np.sqrt(mean_squared_error(train[TARGET], oof_preds))
rmse_baseline = np.sqrt(mean_squared_error(train[TARGET], train["y"]))

print(f"\nâœ… Model RMSE: {rmse_model:.5f}")
print(f"ğŸ“‰ Baseline RMSE: {rmse_baseline:.5f}\n")


# Visualize Results: Comprehensive Dashboard

#plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
fig.suptitle("ğŸŒ� Accident Risk Model Performance Dashboard", fontsize=18, fontweight="bold")

# Plot 1 â€” True vs Predicted: Enhanced Scatter Plot with Density
ax1 = fig.add_subplot(gs[0, 0])

# Use a hexbin plot to show density
hb = ax1.hexbin(train[TARGET], oof_preds, gridsize=50, cmap='viridis') # 'viridis' is often a good default
ax1.plot([0, 1], [0, 1], "--", color="white", linewidth=1)  # Changed to white for contrast
ax1.set_title("ğŸ”� True vs Predicted Values (Density)", fontsize=13, fontweight="bold")
ax1.set_xlabel("True Accident Risk")
ax1.set_ylabel("Predicted Accident Risk")
cb = fig.colorbar(hb, ax=ax1) # Add a colorbar to the hexbin plot
cb.set_label('Density')

# Plot 2 â€” Prediction Error Distribution: Enhanced with QQ-Plot
ax2 = fig.add_subplot(gs[0, 1])

# Calculate errors
errors = oof_preds - train[TARGET]

# Create a combined plot: histogram + QQ plot
sns.histplot(errors, bins=40, kde=True, color="#FF595E", ax=ax2, label="Error Distribution")
ax2.axvline(errors.mean(), color="black", linestyle="--", linewidth=1, label=f"Mean Error: {errors.mean():.3f}")

# Add a QQ-plot (probability plot) on a secondary y-axis
ax2_qq = ax2.twinx() # Create a secondary axis that shares the x-axis.
stats.probplot(errors, dist="norm", plot=ax2_qq)  # "norm" for normal distribution

# Customize the QQ plot appearance
ax2_qq.lines[0].set_markerfacecolor('green') # Color the points
ax2_qq.lines[0].set_markersize(4) # Reduce marker size
ax2_qq.lines[1].set_color('blue') # Color the line

ax2_qq.set_ylabel("Theoretical Quantiles (Normal Distribution)") # set label
ax2_qq.yaxis.label.set_color('blue') #set label color.
ax2_qq.tick_params(axis='y', colors='blue') #set tick color
ax2_qq.set_title("QQ Plot of Error Distribution")

ax2.set_title("ğŸ“Š Prediction Error Distribution & QQ-Plot", fontsize=13, fontweight="bold")
ax2.set_xlabel("Error")
ax2.set_ylabel("Frequency") # This will be on the left y axis
ax2.legend() # Show the legend

# Plot 3 â€” Correlation Heatmap
ax3 = fig.add_subplot(gs[1, 0])
corr = train[NUMS + ["y", TARGET]].corr()
sns.heatmap(corr, cmap="YlGnBu", annot=False, cbar=True, ax=ax3)
ax3.set_title("ğŸ”— Correlation Heatmap", fontsize=13, fontweight="bold")

# Plot 4 â€” Feature Importance:  Interactive Scatter Plot
ax4 = fig.add_subplot(gs[1, 1])

# Get feature importances from the model
importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15) # Select top 15

# Create a scatter plot
sns.scatterplot(x='Importance', y='Feature', data=importance_df, ax=ax4, s=100, color="#2A9D8F") #Adjusted size and color

ax4.set_title("ğŸ”¥ Top Feature Importances (Scatter)", fontsize=13, fontweight="bold")
ax4.set_xlabel("Importance (Gain)")
ax4.set_ylabel("Feature")
ax4.grid(axis='x')  # Add gridlines for easier readability

plt.gca().set_facecolor('#fdf6ce')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# Submission

sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
sub["accident_risk"] = test_preds
sub.to_csv("submission.csv", index=False)

print("\nğŸ“¤ Submission File Preview:")
sub.head()




