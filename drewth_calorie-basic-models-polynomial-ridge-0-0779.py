%matplotlib inline


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import linregress, pearsonr, ttest_rel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_squared_log_error,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, RobustScaler
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from tabulate import tabulate


# Read a CSV file with header in the first row (default behavior)
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


try:
    import catppuccin
    from catppuccin.extras.matplotlib import load_color
    # style preference
    plt.style.use(["ggplot", catppuccin.PALETTE.mocha.identifier])
    
    blue = load_color(catppuccin.PALETTE.mocha.identifier, "blue")
    orange = load_color(catppuccin.PALETTE.mocha.identifier, "peach")
    red = load_color(catppuccin.PALETTE.mocha.identifier, "red")
    text = load_color(catppuccin.PALETTE.mocha.identifier, "text")
except ModuleNotFoundError:
    # Fallback or skip using catppuccin
    print("catppuccin not installed, skipping palette setup.")
    plt.style.use(["ggplot", "dark_background"])
    blue = "#89b4fa"
    orange = "#fab387"
    red = "#f38ba8"
    text = "#cdd6f4"


try:
    from sklearn.metrics import root_mean_squared_log_error

    def rmsle_calc(y_true, y_pred):
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        return root_mean_squared_log_error(y_true, y_pred)
except ImportError:
    def rmsle_calc(y_true, y_pred):
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


rmsle_scorer = make_scorer(rmsle_calc, greater_is_better=False)


print(train.describe())
print(train.head())


print(test.describe())
print(test.head())


# confirm nothing missing
missing_counts = train.isna().sum()
missing_percent = (missing_counts / len(train)) * 100

missing_df = (
    pd.concat([missing_counts, missing_percent], axis=1)
    .rename(columns={0: "Missing Count", 1: "Missing %"})
    .sort_values(by="Missing %", ascending=False)
)

print(missing_df[missing_df["Missing Count"] > 0])


print(train["Calories"].describe())
print("\nSkewness:", train["Calories"].skew())
print("Kurtosis:", train["Calories"].kurtosis())


num_zeros = (train["Calories"] == 0).sum()
num_negatives = (train["Calories"] < 0).sum()
print(f"Zeros: {num_zeros}, Negatives: {num_negatives}")


plt.figure(figsize=(14, 4))

# Histogram + KDE
plt.subplot(1, 3, 1)
sns.histplot(train["Calories"], bins=50, kde=True)
plt.title("Calories Distribution (Histogram + KDE)")
plt.xlabel("Calories")

# Boxplot for outliers
plt.subplot(1, 3, 2)
sns.boxplot(x=train["Calories"])
plt.title("Calories Boxplot")

# Log-transform histogram (if skewed)
plt.subplot(1, 3, 3)
sns.histplot(np.log1p(train["Calories"]), bins=50, kde=True, color=orange)
plt.title("Log(1 + Calories) Distribution")
plt.xlabel("Log(1 + Calories)")

plt.tight_layout()
plt.show()


# subset the data for easy analysis
sampled = train.sample(n=50000)

target = "Calories"

numeric_cols = train.select_dtypes(include="number").columns.drop("id")
features = numeric_cols.drop(target)


g = sns.PairGrid(sampled[numeric_cols], diag_sharey=False)
# Scatter + regression off-diagonal


def scatter_reg(x, y, **kwargs):
    r, _ = pearsonr(x, y)
    # Determine color
    if x.name == target or y.name == target:
        color = red
    else:
        color = blue
    alpha = 0.6
    if r < 0.3:
        alpha = 0.05
    sns.regplot(
        x=x,
        y=y,
        scatter_kws={"s": 5, "alpha": alpha, "color": color},
        line_kws={"color": "black", "linewidth": 2},
        truncate=False,
        ci=None,
    )
    ax = plt.gca()
    if abs(r) < 0.3:
        font_size = 10
    elif abs(r) < 0.5:
        font_size = 12
    else:
        font_size = 14
    font_weight = "bold" if abs(r) > 0.8 else "normal"
    ax.annotate(
        f"r = {r:.2f}",
        xy=(0.05, 0.85),
        xycoords="axes fraction",
        fontsize=font_size,
        fontweight=font_weight,
        color=text,
    )


g.map_lower(scatter_reg)

# Histograms on the diagonal


def diag_hist(x, **kwargs):
    if x.name == target:
        sns.histplot(x, color=red, kde=True)
    else:
        sns.histplot(x, color="gray", kde=True)


g.map_diag(diag_hist)

# Hide the upper triangle
for i in range(len(g.axes)):
    for j in range(len(g.axes)):
        if j > i:
            g.axes[i, j].set_visible(False)


# Adjust subplot spacing
g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
plt.tight_layout()

g.fig.text(
    0.5,
    0.93,
    "Feature Correlations",
    ha="center",
    va="center",
    fontsize=20,
    weight="bold",
)
g.fig.text(
    0.5,
    0.91,
    "Target shown in red.  Plots with R<0.3 are more opaque.",
    ha="center",
    va="center",
    fontsize=15,
)
plt.show()


features = ["Heart_Rate", "Body_Temp", "Duration", "Weight", "Age"]
X = train[features].values
y = train[target].values
X_test = test[features].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


X_vif = train[["Duration", "Heart_Rate", "Body_Temp"]]
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
]

print(vif_data)


def detect_outliers_zscore(df, cols, threshold=3):
    outlier_indices = set()
    for col in cols:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = df.index[np.abs(z_scores) > threshold]
        outlier_indices.update(outliers)
    return list(outlier_indices)


outliers = detect_outliers_zscore(train, features, threshold=3)
print(f"Detected {len(outliers)} outliers")

# View some outlier rows
print(train.loc[outliers].head())


# Standardize features (important for PCA)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)

# Fit PCA without reducing components yet (to inspect all)
pca = PCA()
pca.fit(X_scaled)

# Explained variance ratio per component
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Create a DataFrame for clarity
variance_df = pd.DataFrame(
    {
        "PC": np.arange(1, len(explained_variance) + 1),
        "Explained Variance Ratio": explained_variance,
        "Cumulative Variance": cumulative_variance,
    }
)

print(variance_df)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(variance_df["PC"], variance_df["Cumulative Variance"], marker="o")
plt.axhline(y=0.95, color=red, linestyle="--", label="95% Variance Threshold")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.legend()
plt.grid(True)
plt.show()


# After fitting PCA (e.g., pca = PCA().fit(X_scaled))
loadings = pca.components_  # shape: (n_components, n_features)

# Create DataFrame of loadings for first few components
loading_df = pd.DataFrame(
    loadings.T,  # transpose: features x components
    columns=[f"PC{i+1}" for i in range(loadings.shape[0])],
    index=features,
)
print("Loadings:")
print(loading_df.head())  # Show loadings for first few features


def plot_pca_loadings(loading_df, pc_num=1):
    """
    Plot the loadings for the specified principal component number (1-indexed).
    """
    pc_name = f"PC{pc_num}"
    if pc_name not in loading_df.columns:
        print(f"{pc_name} not found in loadings.")
        return

    loadings = loading_df[pc_name].sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    loadings.plot(kind="barh")
    plt.title(f"PCA Loadings for {pc_name}")
    plt.xlabel("Loading Value")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.show()


# Example: plot loadings for first 3 PCs
for i in range(1, 4):
    plot_pca_loadings(loading_df, pc_num=i)


# Fit simple Linear Regression on raw data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on validation set
y_pred = lr.predict(X_val)

# Compute RMSLE manually
rmsle = rmsle_calc(y_val, y_pred)

print(f"Linear Regression Validation RMSLE (raw data): {rmsle:.4f}")

# Optional: coefficients
coef_df = pd.DataFrame({"Feature": features, "Coefficient": lr.coef_})
best_rmsle = rmsle
best_pipeline = lr
current_best = "Linear Regression"
print(coef_df)


# Ridge regression with built-in cross-validation over alpha
ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), scoring=rmsle_scorer, cv=5)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_val)

# Compute RMSLE
ridge_rmsle = rmsle_calc(y_val, ridge_pred)

print(f"Ridge best alpha: {ridge.alpha_}")
print(f"Ridge Validation RMSLE: {ridge_rmsle:.4f}")
if ridge_rmsle < best_rmsle:
    best_rmsle = ridge_rmsle
    best_pipeline = ridge
    current_best = "Ridge"
print(f"\nThe current best is {current_best} with {best_rmsle}")


# Lasso regression with built-in CV
lasso = LassoCV(alphas=None, cv=5, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_val)

# Compute RMSLE
lasso_rmsle = rmsle_calc(y_val, lasso_pred)

print(f"Lasso best alpha: {lasso.alpha_}")
print(f"Lasso Validation RMSLE: {lasso_rmsle:.4f}")
if lasso_rmsle < best_rmsle:
    best_rmsle = lasso_rmsle
    best_pipeline = lasso
    current_best = "Lasso"
print(f"\nThe current best is {current_best} with {best_rmsle}")


#  coefficients overview
coef_df = pd.DataFrame(
    {"Feature": features, "Ridge_coef": ridge.coef_, "Lasso_coef": lasso.coef_}
)

print(coef_df)


# Build pipeline with scaling and ElasticNetCV
elasticnet_pipeline = make_pipeline(
    RobustScaler(),
    ElasticNetCV(
        l1_ratio=np.linspace(
            0.1, 1.0, 10
        ),  # Avoid 0 to prevent pure ridge which causes warnings
        alphas=np.logspace(-4, 1, 30),
        cv=5,
        max_iter=100000,
        n_jobs=-1,
    ),
)

# Fit model
elasticnet_pipeline.fit(X_train, y_train)

# Predict and evaluate RMSLE
y_pred = elasticnet_pipeline.predict(X_val)
elastic_rmsle = rmsle_calc(y_val, y_pred)

print(
    f'ElasticNet best alpha: {elasticnet_pipeline.named_steps["elasticnetcv"].alpha_}'
)
print(
    f'ElasticNet best l1_ratio: {elasticnet_pipeline.named_steps["elasticnetcv"].l1_ratio_}'
)
print(f"ElasticNet Validation RMSLE: {elastic_rmsle:.4f}")


if elastic_rmsle < best_rmsle:
    best_rmsle = elastic_rmsle
    best_pipeline = elasticnet_pipeline
    current_best = "ElasticNet"
print(f"\nThe current best is {current_best} with {best_rmsle}")


# Coefficients
coef_df = pd.DataFrame(
    {
        "Feature": features,
        "ElasticNet_coef": elasticnet_pipeline.named_steps["elasticnetcv"].coef_,
    }
)

print(coef_df)


# Ridge + PCA pipeline
ridge_pca_pipeline = make_pipeline(
    RobustScaler(),
    PCA(n_components=0.95),
    RidgeCV(alphas=np.logspace(-3, 3, 13), scoring=rmsle_scorer, cv=5),
)

ridge_pca_pipeline.fit(X_train, y_train)
ridge_pca_preds = ridge_pca_pipeline.predict(X_val)
ridge_pca_rmsle = rmsle_calc(y_val, ridge_pca_preds)

print(f"Ridge + PCA Validation RMSLE: {ridge_pca_rmsle:.4f}")
print(
    f'Number of PCA components used: {ridge_pca_pipeline.named_steps["pca"].n_components_}'
)

if ridge_pca_rmsle < best_rmsle:
    best_rmsle = ridge_pca_rmsle
    best_pipeline = ridge_pca_pipeline
    current_best = "Ridge + PCA"
print(f"\nThe current best is {current_best} with {best_rmsle}")


# Lasso + PCA pipeline
lasso_pca_pipeline = make_pipeline(
    RobustScaler(), PCA(n_components=0.95), LassoCV(cv=5, max_iter=10000)
)

lasso_pca_pipeline.fit(X_train, y_train)
lasso_pca_preds = lasso_pca_pipeline.predict(X_val)
lasso_pca_rmsle = rmsle_calc(y_val, lasso_pca_preds)

print(f"Lasso + PCA Validation RMSLE: {lasso_pca_rmsle:.4f}")
print(
    f'Number of PCA components used: {lasso_pca_pipeline.named_steps["pca"].n_components_}'
)

if lasso_pca_rmsle < best_rmsle:
    best_rmsle = lasso_pca_rmsle
    best_pipeline = lasso_pca_pipeline
    current_best = "Lasso + PCA"


print(f"\nThe current best is {current_best} with {best_rmsle}")


def plot_predictions_and_residuals(
    y_true,
    y_pred,
    title="Predicted vs Actual",
    xlabel="Actual",
    ylabel="Predicted",
    outlier_thresh=None,
    outlier_multiplier=8,
    hex_gridsize=None,
    separate_outliers=True,
):

    df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    df["Residual"] = df["Actual"] - df["Predicted"]

    if separate_outliers and outlier_thresh is None:
        outlier_thresh = df["Residual"].std() * outlier_multiplier
        print(f"Setting outlier threshold to {outlier_thresh}")
    if hex_gridsize is None:
        hex_gridsize = int(np.cbrt(df.shape[0]))
        print(f"Setting grid size to {hex_gridsize}")

    # Define limits for plots (square range)
    lim_min = min(df[["Actual", "Predicted"]].min().min(), 0)
    lim_max = df[["Actual", "Predicted"]].max().max()
    lims = [lim_min, lim_max]

    # Identify outliers by residuals threshold (absolute residual)
    inliers = df
    if separate_outliers:
        outliers = df[np.abs(df["Residual"]) > outlier_thresh]
        inliers = df[np.abs(df["Residual"]) <= outlier_thresh]

    plt.close("all")
    fig, axs = plt.subplots(
        2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 2]}
    )

    # Top plot: Actual vs Predicted - hexbin for dense + scatter for outliers
    hb = axs[0].hexbin(
        inliers["Actual"],
        inliers["Predicted"],
        gridsize=hex_gridsize,
        cmap="viridis",
        mincnt=1,
    )
    cb = fig.colorbar(hb, ax=axs[0])
    cb.set_label("Counts")

    if separate_outliers:
        # Overlay outliers
        axs[0].scatter(
            outliers["Actual"],
            outliers["Predicted"],
            color=text,
            s=10,
            label=f"Outliers (residual > {outlier_thresh:.1f})",
            alpha=0.7,
        )

    # Regression line (using seaborn regplot with scatter=False)
    sns.regplot(
        x="Actual",
        y="Predicted",
        data=df,
        scatter=False,
        ax=axs[0],
        color="orange",
        line_kws={"linewidth": 2},
    )
    # Perfect prediction line
    axs[0].plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")

    axs[0].set_xlim(lims)
    axs[0].set_ylim(lims)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].grid(True, zorder=0)

    # Bottom plot: Residuals - hexbin + outliers
    hb2 = axs[1].hexbin(
        inliers["Predicted"],
        inliers["Residual"],
        gridsize=hex_gridsize,
        cmap="viridis",
        mincnt=1,
    )
    cb2 = fig.colorbar(hb2, ax=axs[1])
    cb2.set_label("Counts")
    if separate_outliers:
        # Overlay outliers
        axs[1].scatter(
            outliers["Predicted"], outliers["Residual"], color=text, s=10, alpha=0.7
        )

    axs[1].set_xlim(lims)
    axs[1].axhline(0, color=text, linestyle="--", linewidth=2)
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("Residual (Actual - Predicted)")
    axs[1].set_title("Residuals Plot")
    axs[1].grid(True, zorder=0)

    plt.tight_layout()
    plt.show()


y_pred = best_pipeline.predict(X_val)
plot_predictions_and_residuals(
    y_val,
    y_pred,
    title=current_best + " Predictions vs Actual with Regression Fit",
    xlabel="Actual Calories",
    ylabel="Predicted Calories",
    separate_outliers=True,
)
plt.show()


# Build pipeline: PolynomialFeatures -> RobustScaler -> Ridge Regression
poly_ridge_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    RobustScaler(),
    Ridge(alpha=1.0),  # You can tune alpha later
)

# Fit on training data
poly_ridge_pipeline.fit(X_train, y_train)

# Predict on validation data
y_val_pred = poly_ridge_pipeline.predict(X_val)

# Calculate RMSLE
poly_ridge_rmsle = rmsle_calc(y_val, y_val_pred)

print(f"Validation RMSLE (Polynomial Ridge): {poly_ridge_rmsle:.4f}")

if poly_ridge_rmsle < best_rmsle:
    best_rmsle = poly_ridge_rmsle
    best_pipeline = poly_ridge_pipeline
    current_best = "Poly RMSLE"
print(f"\nThe current best is {current_best} with {best_rmsle}")


y_pred = poly_ridge_pipeline.predict(X_val)
plot_predictions_and_residuals(
    y_val,
    y_pred,
    title="Polynomial Ridge Predictions vs Actual with Regression Fit",
    xlabel="Actual Calories",
    ylabel="Predicted Calories",
)


def plot_residual_histogram(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color="skyblue")
    plt.axvline(0, color=red, linestyle="--")
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.show()


plot_residual_histogram(y_true=y_val, y_pred=y_pred)


def calc_cooks_distance(X, y, model):
    # Add constant for intercept
    X_const = sm.add_constant(X)

    # Fit OLS to get influence measures (we do this as ElasticNet doesn't provide these)
    ols_model = sm.OLS(y, X_const).fit()
    influence = OLSInfluence(ols_model)

    cooks_d = influence.cooks_distance[0]

    # Optional: print indices of influential points above a threshold
    threshold = 4 / len(y)
    influential_points = np.where(cooks_d > threshold)[0]
    print(
        f"Number of influential points (Cook's D > {threshold:.4f}): {len(influential_points)}"
    )
    print(f"Indices of top influential points: {influential_points[:10]}")
    return influential_points, cooks_d


influential_points, cooks_d = calc_cooks_distance(X_train, y_train, poly_ridge_pipeline)


def plot_cooks_distance(
    X, y, cooks_d, model=None, features=None, size=5, palette="inferno"
):
    """
    Plot Actual vs Predicted colored by log10 Cook's distance.

    Parameters:
        X (array-like): Feature matrix
        y (array-like): True target values
        cooks_d (np.ndarray): Cook's distance values (same length as y)
        model (optional): Fitted model with a .predict() method
        features (list, optional): List of feature names for dataframe columns
        size (int): Marker size for scatter plot
        palette (str): Color palette for scatter plot
    """
    df = pd.DataFrame(X, columns=features) if features is not None else pd.DataFrame(X)
    df["Actual"] = y

    if model is not None:
        df["Predicted"] = model.predict(X)
    else:
        # If no model given, use Actual for Predicted (or raise error)
        df["Predicted"] = y

    df["Residual"] = df["Actual"] - df["Predicted"]
    df["cooks_d"] = cooks_d
    df_sorted = df.sort_values(by="cooks_d", ascending=True)

    # Add small offset to avoid log(0)
    log_cooks = np.log10(df_sorted["cooks_d"] + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True, zorder=0)  # grid behind points

    sns.scatterplot(
        data=df_sorted,
        x="Actual",
        y="Predicted",
        hue=log_cooks,
        palette=palette,
        size=size,
        alpha=0.5,
        edgecolor=None,
        legend=False,
        ax=ax,
    )

    norm = plt.Normalize(log_cooks.min(), log_cooks.max())
    smap = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    smap.set_array([])
    fig.colorbar(smap, ax=ax, label="Log10 Cook's Distance")

    ax.plot(
        [0, df_sorted["Actual"].max()],
        [0, df_sorted["Actual"].max()],
        "w--",
        alpha=0.5,
        label="Perfect Prediction",
    )

    ax.set_xlabel("Actual Calories")
    ax.set_ylabel("Predicted Calories")
    ax.set_title("Actual vs Predicted Colored by Cook's Distance")
    ax.legend(loc="upper left")

    plt.show()


plot_cooks_distance(
    X_train,
    y_train,
    cooks_d,
    model=poly_ridge_pipeline,
    features=features,
    size=5,
    palette="inferno",
)


pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False), RobustScaler(), Ridge()
)

param_grid = {
    "polynomialfeatures__interaction_only": [False, True],
    "ridge__alpha": np.logspace(-5, 1, 10),
    "polynomialfeatures__degree": [2, 3, 4],
    "polynomialfeatures__include_bias": [False, True],
}

grid = GridSearchCV(
    pipeline, param_grid, cv=5, scoring=rmsle_scorer, n_jobs=-1, verbose=1
)

grid.fit(X, y)

print("Best params:", grid.best_params_)


results_df = pd.DataFrame(grid.cv_results_)
results_df = results_df.sort_values(by="mean_test_score", ascending=False)
display(
    results_df[
        [
            "param_polynomialfeatures__degree",
            "param_polynomialfeatures__include_bias",
            "param_polynomialfeatures__interaction_only",
            "param_ridge__alpha",
            "mean_test_score",
            "std_test_score",
        ]
    ].head(20)
)


best_rows = results_df.head(10)
print(
    f"- The best mean test score is around {best_rows['mean_test_score'].iloc[0]:.3f} for degree {best_rows['param_polynomialfeatures__degree'].iloc[0]:.0f}"
)
if (
    best_rows["param_polynomialfeatures__include_bias"].iloc[0]
    == best_rows["param_polynomialfeatures__include_bias"].iloc[1]
):
    if best_rows["param_polynomialfeatures__include_bias"].iloc[0]:
        print("- It is best to include bias")
    else:
        print("- It is best not to include bias")
else:
    print("- Including bias does not seem to have much of an effect either way")
if best_rows["param_ridge__alpha"].iloc[0] > 0.05:
    print(
        f"- It is good to have some amount of regularization, with an alpha of {best_rows['param_ridge__alpha'].iloc[0]:.3f}"
    )
else:
    print(
        f"- It doesn't seem helpful to have regularization, with an alpha of {best_rows['param_ridge__alpha'].iloc[0]:.3f} "
    )


# Use seaborn lineplot, grouping by degree and plotting each on the same axes
custom_palette = {
    2: sns.color_palette("pastel")[3],
    3: sns.color_palette("pastel")[1],
    4: sns.color_palette("pastel")[0],
}

best_row = results_df.loc[
    results_df["mean_test_score"].idxmax()
]  # or idxmin() if lower is better
best_alpha = best_row["param_ridge__alpha"]

sns.lineplot(
    data=results_df,
    x="param_ridge__alpha",
    y="mean_test_score",
    hue="param_polynomialfeatures__degree",
    marker=".",
    alpha=0.8,
    palette=custom_palette,
)

plt.xscale("log")
plt.xlabel("Ridge Alpha (log scale)")
plt.ylabel("Mean Test Score")
plt.title("Mean Test Score vs Ridge Alpha by Polynomial Degree")
plt.axvline(
    best_alpha,
    color="white",
    linestyle="--",
    label=f"Best alpha: {best_alpha:.1e}",
    alpha=0.8,
)
plt.legend(title="Polynomial Degree", loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()


sns.violinplot(
    data=results_df,
    x="param_polynomialfeatures__degree",
    y="mean_test_score",
    hue="param_polynomialfeatures__degree",
    palette=custom_palette,
    legend=False,
)
plt.title("Mean Test Score vs Polynomial Degree")
plt.show()


kf = KFold(n_splits=10, shuffle=True)

rmsle_deg3 = []
rmsle_deg4 = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Degree 3 model
    model3 = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        RobustScaler(),
        Ridge(alpha=best_alpha),  # use best alpha from grid search
    )
    model3.fit(X_tr, y_tr)
    preds3 = model3.predict(X_val)
    rmsle3 = rmsle_calc(y_val, preds3)
    rmsle_deg3.append(rmsle3)

    # Degree 4 model
    model4 = make_pipeline(
        PolynomialFeatures(degree=4, include_bias=False),
        RobustScaler(),
        Ridge(alpha=best_alpha),
    )
    model4.fit(X_tr, y_tr)
    preds4 = model4.predict(X_val)
    rmsle4 = rmsle_calc(y_val, preds4)
    rmsle_deg4.append(rmsle4)

# Paired t-test
stat, p = ttest_rel(rmsle_deg3, rmsle_deg4)
print(f"Mean RMSLE degree 3: {np.mean(rmsle_deg3):.4f}")
print(f"Mean RMSLE degree 4: {np.mean(rmsle_deg4):.4f}")
print(f"Paired t-test p-value: {p:.4f}")

if p < 0.05:
    print("Degree 4 polynomial model has significantly different RMSLE from degree 3")
else:
    print("No significant difference in RMSLE between degree 3 and degree 4")


# Create DataFrame for plotting
degree_df = pd.DataFrame(
    {
        "degree": [3] * len(rmsle_deg3) + [4] * len(rmsle_deg4),
        "rmsle": rmsle_deg3 + rmsle_deg4,
    }
)
degree_df["rmsle"] = -degree_df["rmsle"]
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=degree_df, x="degree", y="rmsle", hue="degree", palette=custom_palette
)
plt.title("RMSLE Distribution by Polynomial Degree (CV folds)")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Test Score")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


pipeline = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=True),
    RobustScaler(),
    Ridge(alpha=best_alpha),
)

# Fit on training data
pipeline.fit(X_train, y_train)

# Predict on validation data
y_val_pred = pipeline.predict(X_val)

# Calculate RMSLE
rmsle = rmsle_calc(y_val, y_val_pred)

print(f"Validation RMSLE: {rmsle:.4f}")

if rmsle < best_rmsle:
    best_rmsle = rmsle
    best_pipeline = pipeline
    current_best = "Poly Ridge"
print(f"\nThe current best is {current_best} with {best_rmsle}")


plot_predictions_and_residuals(
    y_val,
    y_val_pred,
    title="Polynomial Predictions vs Actual with Regression Fit",
    xlabel="Actual Calories",
    ylabel="Predicted Calories",
)


plot_residual_histogram(y_true=y_val, y_pred=y_val_pred)


influential_points, cooks_d = calc_cooks_distance(X_train, y_train, pipeline)

plot_cooks_distance(
    X_train,
    y_train,
    cooks_d,
    model=poly_ridge_pipeline,
    features=features,
    size=5,
    palette="inferno",
)


# Fit pipeline on training data
pipeline.fit(X, y)

# Predict on test data
test_preds = pipeline.predict(X_test)

# Build submission DataFrame
submission_df = pd.DataFrame(
    {"id": test["id"], "Calories": np.clip(test_preds, a_min=0, a_max=None)}
)

# Save CSV without index
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())

