# import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor


# Load the data
train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train_data


test_data
# This is the data we will use for submitting our predictions to the competition
# 'accident_risk' is not given for this reason.


data_ratio = len(test_data) / (len(test_data) + len(train_data))
print(f"The training data has {len(train_data)} rows.\nThe test data has {len(test_data)} rows")
print(f"\nTest data is {data_ratio:.2%} of the sum of train and test data")


# helper function to compare proportions of our features for training and testing data
def plot_proportion_compare(feature, df1, df2, name1='Dataset A', name2='Dataset B', bins=None, figsize=(10,5)):
    s1 = df1[feature]
    s2 = df2[feature]
    if bins is not None:
        if isinstance(bins, int):
            combined = pd.concat([s1, s2], ignore_index=True)
            combined_nonnull = combined.dropna()
            edges = np.histogram_bin_edges(combined_nonnull, bins=bins)
        else:
            edges = bins
        s1 = pd.cut(s1, bins=edges, include_lowest=True)
        s2 = pd.cut(s2, bins=edges, include_lowest=True)
    p1 = s1.value_counts(normalize=True, dropna=False)
    p2 = s2.value_counts(normalize=True, dropna=False)
    cats = sorted(set(p1.index).union(p2.index), key=lambda x: str(x))
    p1 = p1.reindex(cats, fill_value=0.0)
    p2 = p2.reindex(cats, fill_value=0.0)
    x = np.arange(len(cats))
    w = 0.45

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - w/2, p1.values, width=w, label=name1, alpha=0.9, edgecolor="black")
    bars2 = ax.bar(x + w/2, p2.values, width=w, label=name2, alpha=0.9, edgecolor="black")

    # Add value labels
    ax.bar_label(bars1, labels=[f"{v:.2%}" for v in p1.values], padding=2)
    ax.bar_label(bars2, labels=[f"{v:.2%}" for v in p2.values], padding=2)

    xtick_labels = [str(c) for c in cats]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=30, ha='right')
    ax.set_ylabel('Proportion (%)')
    ax.set_title(f'Proportion of {feature}: {name1} vs {name2}')
    ax.legend(title='Dataset', loc='best')
    ymax = max(p1.max() if len(p1) else 0, p2.max() if len(p2) else 0)
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
    plt.tight_layout()
    plt.show()


plot_proportion_compare("road_type",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("num_lanes",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("speed_limit",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("lighting",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("weather",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("road_signs_present",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("public_road",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("time_of_day",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("holiday",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("school_season",train_data, test_data,name1="Train",name2="Test")


plot_proportion_compare("num_reported_accidents",train_data, test_data,name1="Train",name2="Test")


plt.figure(figsize=(8, 5))
plt.boxplot(
    [train_data['curvature'], test_data['curvature']],
    labels=['Generated', 'Comp Test'],
    patch_artist=True  # Optional: for colored boxes
)
plt.title('Curvature: Generated vs Comp Test')
plt.ylabel('Curvature')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()


sns.histplot(train_data['accident_risk'], bins=50, kde=True, edgecolor='black')
plt.title('Accident Risk Histogram')
plt.xlabel('Accident Risk')
plt.ylabel('Density')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x='time_of_day', y='accident_risk', data=train_data)
plt.title('Accident Risk by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Accident Risk')
plt.show()


target_col = "accident_risk"

X_train = train_data.drop(columns=[target_col]).copy()
y_train = train_data[target_col].astype(float).values
X_test  = test_data.copy()

# Keep id for submission
test_id = X_test["id"]

# Separate features by data type for the pipeline
feature_cols = [c for c in X_train.columns if c != "id"]
X_train = X_train[feature_cols].copy()
X_test  = X_test[feature_cols].copy()

cat_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Encode and scale
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("num", StandardScaler(with_mean=False), num_cols)
    ],
    remainder="drop"
)

# XGBoost Regression
xgb = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    grow_policy="lossguide",
    max_depth=6,
    n_estimators=600,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    reg_alpha=0.01,
    min_child_weight=2,
    max_bin=256,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", xgb)])

# 5-fold CV RMSE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
for tr_idx, va_idx in kf.split(X_train, y_train):
    pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
    pred = pipe.predict(X_train.iloc[va_idx])
    rmse = mean_squared_error(y_train[va_idx], pred, squared=False)
    rmse_scores.append(rmse)

print(f"5-Fold CV RMSE: {np.mean(rmse_scores):.6f} ± {np.std(rmse_scores):.6f}")
print("Fold RMSEs:", [round(s, 6) for s in rmse_scores])

# Fit and predict test submission
pipe.fit(X_train, y_train)



# Get the competition predictions
test_pred = pipe.predict(X_test)

# Submission DataFrame
submission_df = pd.DataFrame({
    "id": test_data["id"],
    "accident_risk": test_pred
})[["id", "accident_risk"]]

print(submission_df.head())


# Competition submission
submission_df.to_csv("submission.csv", index=False)


# Extract the trained pieces
prep = pipe.named_steps["prep"]
xgb_model = pipe.named_steps["model"]

# Get transformed feature names (OHE expanded + numeric passthrough)
ohe = prep.named_transformers_["cat"]
num_names = prep.transformers_[1][2]  # numeric column list
ohe_names = ohe.get_feature_names_out(input_features=prep.transformers_[0][2])
feature_names = np.concatenate([ohe_names, np.array(num_names)])

# Get importance values aligned to transformed features
imp = xgb_model.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)

# aggregate OHE columns back to their base feature for interpretability
# OneHotEncoder names look like 'weather_clear'
def base_name(col):
    # For OneHotEncoder's get_feature_names_out, pattern is '<col>_<category>'
    # Numeric columns pass through as-is
    if "_" in col and col.split("_")[0] in prep.transformers_[0][2]:
        return col.split("_")[0]
    return col

fi_df["base_feature"] = fi_df["feature"].map(base_name)
agg_df = fi_df.groupby("base_feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)

# top k one-hot features
top_k = 10 
fig, ax = plt.subplots(figsize=(10, max(6, 0.3*top_k)))
plot_df = fi_df.head(top_k).iloc[::-1]  # reverse for horizontal bar ascending
ax.barh(plot_df["feature"], plot_df["importance"], color="#4C78A8")
ax.set_title(f"XGBoost Feature Importance (top {top_k} transformed features)")
ax.set_xlabel("Importance (gain)")
ax.set_ylabel("Transformed feature")
plt.tight_layout()
plt.show()

# aggregated by original feature name plot
top_m = 5
fig, ax = plt.subplots(figsize=(9, max(5, 0.35*top_m)))
agg_plot = agg_df.head(top_m).iloc[::-1]
ax.barh(agg_plot["base_feature"], agg_plot["importance"], color="#72B7B2")
ax.set_title(f"XGBoost Feature Importance (aggregated, top {top_m})")
ax.set_xlabel("Total importance (sum over OHE levels)")
ax.set_ylabel("Original feature")
plt.tight_layout()
plt.show()

# fig.savefig("feature_importance_top_transformed.png", dpi=200, bbox_inches="tight")
# fig.savefig("feature_importance_aggregated.png", dpi=200, bbox_inches="tight")

print("Top transformed features:")
print(fi_df.head(20))

print("\nTop aggregated features:")
print(agg_df.head(20))

