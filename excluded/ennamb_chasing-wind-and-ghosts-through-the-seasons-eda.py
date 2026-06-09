# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load in the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.head()


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load train data (assuming it's already defined in the notebook)
df = train.copy()

# === 1. Compute Pearson & Spearman Correlation ===
pearson_corr, _ = pearsonr(df["winddirection"], df["rainfall"])
spearman_corr, _ = spearmanr(df["winddirection"], df["rainfall"])

print("\n=== Correlation Analysis ===")
print(f"Pearson Correlation (Linear): {pearson_corr:.4f}")
print(f"Spearman Correlation (Monotonic): {spearman_corr:.4f}")

# === 2. Scatter Plot with Linear Fit ===
plt.figure(figsize=(8, 5))
sns.regplot(x=df["winddirection"], y=df["rainfall"], scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
plt.xlabel("Wind Direction (Degrees)")
plt.ylabel("Rainfall (Binary)")
plt.title("Scatter Plot: Wind Direction vs. Rainfall")
plt.grid()
plt.show()

# === 3. Fit Linear & Quadratic Polynomial Models ===
X = df[["winddirection"]]
y = df["rainfall"]

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
r2_linear = r2_score(y, y_pred_linear)

# Quadratic (Polynomial) Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)
r2_poly = r2_score(y, y_pred_poly)

print("\n=== Polynomial Fit Comparison ===")
print(f"R² (Linear Model): {r2_linear:.4f}")
print(f"R² (Quadratic Model): {r2_poly:.4f}")

# === 4. Plot Polynomial Fit ===
plt.figure(figsize=(8, 5))
plt.scatter(df["winddirection"], df["rainfall"], alpha=0.2, label="Actual Data")
plt.scatter(df["winddirection"], y_pred_linear, color="red", s=10, label="Linear Fit")
plt.scatter(df["winddirection"], y_pred_poly, color="green", s=10, label="Quadratic Fit")
plt.xlabel("Wind Direction (Degrees)")
plt.ylabel("Rainfall (Binary)")
plt.title("Polynomial Fit: Wind Direction vs. Rainfall")
plt.legend()
plt.grid()
plt.show()


from scipy.stats import circmean, circstd, rayleigh
import statsmodels.api as sm
import statsmodels.formula.api as smf

# === Convert Wind Direction to Sin & Cos Features ===
def transform_wind_direction(df):
    df["wind_sin"] = np.sin(np.radians(df["winddirection"]))
    df["wind_cos"] = np.cos(np.radians(df["winddirection"]))
    return df

train = transform_wind_direction(train)

# === Circular Statistics & Rayleigh Test ===
circ_mean = circmean(train["winddirection"], high=360, low=0)
circ_std = circstd(train["winddirection"], high=360, low=0)

# Rayleigh test
# Convert wind direction to radians and pass it to the Rayleigh test
wind_radians = np.radians(train["winddirection"])
rayleigh_result = rayleigh(wind_radians)  # Rayleigh test doesn't need the `test=` argument
rayleigh_p = rayleigh_result.sf(np.abs(rayleigh_result.mean()))[0]  # Extract the p-value from the result array

print("\n=== Circular Statistics ===")
print(f"Mean Wind Direction: {circ_mean:.2f}°")
print(f"Circular Standard Deviation: {circ_std:.2f}°")
print(f"Rayleigh p-value: {rayleigh_p:.4f}")

if rayleigh_p < 0.05:
    print("Wind direction is **not uniform**—preferred orientations exist.")
else:
    print("Wind direction appears **uniformly distributed**—no strong pattern detected.")

# === Circular Regression with Wind Sin & Cos Features ===
# We use logistic regression with transformed sin/cos features instead of raw wind direction
model = smf.logit("rainfall ~ wind_sin + wind_cos", data=train).fit()

print("\n=== Circular Logistic Regression Summary ===")
print(model.summary())

# === Visualize Sin & Cos Features ===
plt.figure(figsize=(12, 5))
sns.scatterplot(x=train["wind_sin"], y=train["wind_cos"], hue=train["rainfall"], palette="coolwarm", alpha=0.6)
plt.xlabel("Sin(Wind Direction)")
plt.ylabel("Cos(Wind Direction)")
plt.title("Wind Direction (Sin/Cos) vs Rainfall")
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.grid()
plt.show()


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# === Convert Day to Month Function ===
def day_to_month(day):
    """Converts day of year (1-365) to month (1-12)."""
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cumulative_days = np.cumsum(months)
    for month, days_in_month in enumerate(cumulative_days, 1):
        if day <= days_in_month:
            return month
    return 12

# === Convert 'day' to 'month' in train & test data ===
train['month'] = train['day'].apply(day_to_month)
test['month'] = test['day'].apply(day_to_month)

# === Compute Monthly Average Rainfall for TRAINING ONLY ===
train_monthly_rainfall = train.groupby("month")["rainfall"].mean().reset_index()

# === Standardize Rainfall Data for Clustering ===
scaler = StandardScaler()
train_monthly_rainfall["rainfall_scaled"] = scaler.fit_transform(train_monthly_rainfall[["rainfall"]])

# === Automatic Cluster Selection with Elbow & Silhouette ===
def find_optimal_clusters(data, max_k=10):
    distortions, silhouette_scores = [], []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, cluster_labels))

    # Find best k: Prioritize silhouette but use elbow if close
    elbow_k = k_range[np.argmin(np.gradient(distortions))]
    silhouette_k = k_range[np.argmax(silhouette_scores)]
    optimal_k = silhouette_k if abs(silhouette_k - elbow_k) <= 1 else elbow_k

    print(f"Optimal K: {optimal_k} (Elbow: {elbow_k}, Silhouette: {silhouette_k})")
    return optimal_k

# === Find Optimal Clusters Using ONLY TRAINING DATA ===
optimal_clusters = find_optimal_clusters(train_monthly_rainfall[["rainfall_scaled"]], max_k=10)

# === Apply K-Means Clustering Using ONLY TRAINING DATA ===
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
train_monthly_rainfall["season_cluster"] = kmeans.fit_predict(train_monthly_rainfall[["rainfall_scaled"]])

# === Assign Seasonal Labels (Sorted by Rainfall) ===
cluster_means = train_monthly_rainfall.groupby("season_cluster")["rainfall"].mean().sort_values()
cluster_mapping = {cluster: f"Season_{i+1}" for i, cluster in enumerate(cluster_means.index, 1)}
train_monthly_rainfall["season"] = train_monthly_rainfall["season_cluster"].map(cluster_mapping)

# === Assign New Season Labels to Train & Test Data WITHOUT Using Test Rainfall ===
season_dict = dict(zip(train_monthly_rainfall["month"], train_monthly_rainfall["season"]))
train["data_season"] = train["month"].map(season_dict)
test["data_season"] = test["month"].map(season_dict)  # Assign based on TRAINING clusters

# === Plot Monthly Rainfall with Clustered Seasons ===
plt.figure(figsize=(10, 5))
sns.barplot(x="month", y="rainfall", hue="season", data=train_monthly_rainfall, palette="coolwarm")
plt.xlabel("Month")
plt.ylabel("Avg Rainfall")
plt.title("Rainfall Trend-Based Seasonal Clustering")
plt.legend(title="Season", loc="upper right")
plt.show()

# === Map Wind Direction to Cardinal Directions ===
def wind_to_cardinal(degrees):
    """Maps wind direction degrees to cardinal directions."""
    if 337.5 <= degrees or degrees < 22.5:
        return "N"
    elif 22.5 <= degrees < 67.5:
        return "NE"
    elif 67.5 <= degrees < 112.5:
        return "E"
    elif 112.5 <= degrees < 157.5:
        return "SE"
    elif 157.5 <= degrees < 202.5:
        return "S"
    elif 202.5 <= degrees < 247.5:
        return "SW"
    elif 247.5 <= degrees < 292.5:
        return "W"
    elif 292.5 <= degrees < 337.5:
        return "NW"

train["wind_cardinal"] = train["winddirection"].apply(wind_to_cardinal)
test["wind_cardinal"] = test["winddirection"].apply(wind_to_cardinal)

# === Group by Data-Driven Season and Month for Wind Analysis ===
wind_by_data_season = train.groupby(['data_season', 'wind_cardinal']).size().unstack(fill_value=0)
wind_by_month = train.groupby(['month', 'wind_cardinal']).size().unstack(fill_value=0)

# === Visualize Wind Direction by Data-Driven Season ===
plt.figure(figsize=(12, 6))
sns.heatmap(wind_by_data_season.T, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Wind Direction Distribution by Data-Driven Season')
plt.ylabel('Wind Cardinal')
plt.xlabel('Season')
plt.show()

# === Visualize Wind Direction by Month ===
plt.figure(figsize=(12, 6))
sns.heatmap(wind_by_month.T, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Wind Direction Distribution by Month')
plt.ylabel('Wind Cardinal')
plt.xlabel('Month')
plt.show()


from scipy.stats import chi2_contingency

# === Apply sin/cos transformation to wind direction ===
def transform_wind_direction(df):
    df["wind_sin"] = np.sin(np.radians(df["winddirection"]))
    df["wind_cos"] = np.cos(np.radians(df["winddirection"]))
    return df

# Apply sin/cos transformation to the train dataset
train = transform_wind_direction(train)

# === Group by wind direction (cardinals) and calculate the mean rainfall per cardinal direction ===
rainfall_by_wind = train.groupby("wind_cardinal")["rainfall"].mean().reset_index()

# === Chi-Square Test of Independence (if categorical data) ===
contingency_table = pd.crosstab(train["wind_cardinal"], train["rainfall"])
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# === Logistic Regression Model with sin/cos transformation ===
# Using sin and cos features instead of cardinal wind direction
train_dummies = pd.concat([train[['wind_sin', 'wind_cos']], train['rainfall']], axis=1)

# Fit logistic regression model
model = smf.logit("rainfall ~ wind_sin + wind_cos", data=train_dummies).fit()

# === Printouts ===
print("\n=== Mean Rainfall by Wind Direction ===")
print(rainfall_by_wind)

print("\n=== Chi-Square Test Result ===")
print(f"Chi2 Value: {chi2:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("There is a significant relationship between wind direction and rainfall.")
else:
    print("No significant relationship between wind direction and rainfall.")

print("\n=== Logistic Regression Summary ===")
print(model.summary())

# === Rainfall Distribution Description by Wind Cardinal ===
print("\n=== Rainfall Distribution by Wind Cardinal Direction ===")
most_rain_wind = rainfall_by_wind.loc[rainfall_by_wind["rainfall"].idxmax()]
least_rain_wind = rainfall_by_wind.loc[rainfall_by_wind["rainfall"].idxmin()]

print(f"  - **Highest Rainfall Direction:** {most_rain_wind['wind_cardinal']} ({most_rain_wind['rainfall']:.3f})")
print(f"  - **Lowest Rainfall Direction:** {least_rain_wind['wind_cardinal']} ({least_rain_wind['rainfall']:.3f})")

print("\nGeneral Observations:")
if most_rain_wind["rainfall"] - least_rain_wind["rainfall"] > 0.10:
    print(f"  - Wind direction has a **notable impact** on rainfall (difference of {most_rain_wind['rainfall'] - least_rain_wind['rainfall']:.3f}).")
else:
    print("  - Wind direction impact on rainfall is **relatively minor**.")

# === Visualizing the Distribution of Rainfall by Wind Sin & Cos ===
plt.figure(figsize=(12, 6))
sns.scatterplot(x="wind_sin", y="wind_cos", hue="rainfall", data=train, palette="coolwarm", alpha=0.6)
plt.title("Wind Direction (Sin/Cos) vs Rainfall")
plt.xlabel("Sin(Wind Direction)")
plt.ylabel("Cos(Wind Direction)")
plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.grid()
plt.show()

# === Visualize the Distribution of Rainfall by Wind Cardinal Direction ===
plt.figure(figsize=(12, 6))
sns.boxplot(x="wind_cardinal", y="rainfall", data=train, palette="Set2")
plt.title("Rainfall Distribution by Wind Cardinal Direction")
plt.xlabel("Wind Cardinal Direction")
plt.ylabel("Rainfall")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# === Visualizing the distribution of 'wind_sin' and 'wind_cos' ===
print("=== Visualizing the Distribution of 'wind_sin' and 'wind_cos' ===")
print("Creating scatter plot to visualize the relationship between wind direction (sin/cos) and rainfall...")
plt.figure(figsize=(14, 6))

# Plot for 'wind_sin' vs 'wind_cos' with rainfall as hue
plt.subplot(1, 2, 1)
sns.scatterplot(x=train["wind_sin"], y=train["wind_cos"], hue=train["rainfall"], palette="coolwarm", alpha=0.6)
plt.title("Wind Direction (Sin/Cos) vs Rainfall")
plt.xlabel("Sin(Wind Direction)")
plt.ylabel("Cos(Wind Direction)")
plt.grid(True)

# Plot histograms of 'wind_sin' and 'wind_cos'
plt.subplot(1, 2, 2)
sns.histplot(train["wind_sin"], kde=True, color='blue', label="Wind Sin", stat="density", linewidth=0)
sns.histplot(train["wind_cos"], kde=True, color='red', label="Wind Cos", stat="density", linewidth=0)
plt.title("Distribution of Wind Sin and Cos")
plt.xlabel("Value")
plt.legend()

plt.tight_layout()
plt.show()

# === Visualizing the relationship between 'wind_sin', 'wind_cos' and Rainfall ===
print("\n=== Visualizing the Relationship Between 'wind_sin', 'wind_cos' and Rainfall ===")
print("Creating boxplots to understand how wind direction (sin/cos) influences rainfall...")
plt.figure(figsize=(12, 6))

# Plot for 'wind_sin' vs rainfall
plt.subplot(1, 2, 1)
sns.boxplot(x=train["wind_sin"], y=train["rainfall"], palette="coolwarm")
plt.title("Rainfall vs Wind Sin")
plt.xlabel("Sin(Wind Direction)")
plt.ylabel("Rainfall")
plt.grid(True)

# Plot for 'wind_cos' vs rainfall
plt.subplot(1, 2, 2)
sns.boxplot(x=train["wind_cos"], y=train["rainfall"], palette="coolwarm")
plt.title("Rainfall vs Wind Cos")
plt.xlabel("Cos(Wind Direction)")
plt.ylabel("Rainfall")
plt.grid(True)

plt.tight_layout()
plt.show()

# === Correlation Matrix for Sin and Cos features ===
print("\n=== Correlation Matrix for Wind Sin, Wind Cos, and Rainfall ===")
print("Calculating correlation between sin, cos features and rainfall...")
corr_matrix = train[["wind_sin", "wind_cos", "rainfall"]].corr()

# Plot heatmap for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix: Wind Sin, Wind Cos and Rainfall")
plt.show()

print("\n=== Correlation Results ===")
print(f"Pearson Correlation between 'wind_sin' and 'rainfall': {corr_matrix.loc['wind_sin', 'rainfall']:.4f}")
print(f"Pearson Correlation between 'wind_cos' and 'rainfall': {corr_matrix.loc['wind_cos', 'rainfall']:.4f}")
print(f"Pearson Correlation between 'wind_sin' and 'wind_cos': {corr_matrix.loc['wind_sin', 'wind_cos']:.4f}")


!pip install pygam


from sklearn.neighbors import KernelDensity
from pygam import LogisticGAM, s
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# === Apply KDE to estimate conditional probability of rainfall given wind direction ===
def kde_density_estimation(feature, target, feature_name):
    feature_df = pd.DataFrame(feature, columns=[feature_name])  # Ensure named DataFrame
    
    kde_rain = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(feature_df[target == 1])
    kde_dry = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(feature_df[target == 0])

    x_vals = np.linspace(-1, 1, 100).reshape(-1, 1)
    x_vals_df = pd.DataFrame(x_vals, columns=[feature_name])

    rain_density = np.exp(kde_rain.score_samples(x_vals_df))
    dry_density = np.exp(kde_dry.score_samples(x_vals_df))

    # **Fix: Ensure scalar values are extracted before printing**
    peak_rainfall = float(x_vals[np.argmax(rain_density)][0])  # Extract scalar
    peak_dry = float(x_vals[np.argmax(dry_density)][0])  # Extract scalar

    # Print KDE summary statistics
    print(f"\n=== KDE Density Estimation for {feature_name} ===")
    print(f" - Estimated density range: [{x_vals.min():.2f}, {x_vals.max():.2f}]")
    print(f" - Peak density for Rainfall: {peak_rainfall:.2f}")
    print(f" - Peak density for No Rainfall: {peak_dry:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, rain_density, label="Rainfall", color="blue")
    plt.plot(x_vals, dry_density, label="No Rainfall", color="red")
    plt.xlabel(feature_name)
    plt.ylabel("Density")
    plt.title(f"KDE for {feature_name} vs Rainfall")
    plt.legend()
    plt.grid()
    plt.show()

# === Generalized Additive Model (GAM) with smoothing splines ===
def gam_modeling(X, y):
    gam = LogisticGAM(s(0) + s(1)).fit(X, y)

    XX_sin = np.linspace(X["wind_sin"].min(), X["wind_sin"].max(), 100)
    XX_cos = np.linspace(X["wind_cos"].min(), X["wind_cos"].max(), 100)
    XX = np.column_stack((XX_sin, XX_cos))  

    preds_sin = gam.partial_dependence(term=0, X=XX)
    preds_cos = gam.partial_dependence(term=1, X=XX)

    plt.figure(figsize=(10, 5))
    plt.plot(XX_sin, preds_sin, label="Effect of Wind Sin", color="blue")
    plt.plot(XX_cos, preds_cos, label="Effect of Wind Cos", color="red")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Transformed Wind Direction")
    plt.ylabel("Log Odds of Rainfall")
    plt.title("Generalized Additive Model (GAM) - Wind Direction Effect")
    plt.legend()
    plt.grid()
    plt.show()

    print("\n=== GAM Model Evaluation ===")
    print(f"AIC: {gam.statistics_['AIC']:.4f}")

    if "BIC" in gam.statistics_:
        print(f"BIC: {gam.statistics_['BIC']:.4f}")
    else:
        print("BIC not available for this model configuration.")

    if "deviance" in gam.statistics_:
        print(f"Deviance: {gam.statistics_['deviance']:.4f}")
    else:
        print("Deviance not available for this model configuration.")

# === Random Forest Feature Importance ===
def random_forest_eval(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
    feature_importances = feature_importances.sort_values("Importance", ascending=False)

    print("\n=== Random Forest Feature Importance ===")
    for index, row in feature_importances.iterrows():
        print(f" - {row['Feature']}: {row['Importance']:.4f}")

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feature_importances, palette="coolwarm")
    plt.title("Random Forest Feature Importance for Wind Direction")
    plt.grid()
    plt.show()

    y_pred_proba = rf.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, y_pred_proba)

    print(f"\n=== Random Forest AUC Score: {auc_score:.4f}")
    if auc_score > 0.75:
        print(" - **Good predictive power detected!**")
    elif auc_score > 0.60:
        print(" - **Moderate predictive power detected. More features may help.**")
    else:
        print(" - **Weak predictive power. Consider additional feature engineering.**")

# === Prepare Data ===
X = train[["wind_sin", "wind_cos"]]
y = train["rainfall"]

# Run all three analyses
print("\n=== Running KDE Density Estimation ===")
kde_density_estimation(X["wind_sin"], y, "wind_sin")
kde_density_estimation(X["wind_cos"], y, "wind_cos")

print("\n=== Running GAM Smoothing Splines ===")
gam_modeling(X, y)

print("\n=== Running Random Forest Feature Importance ===")
random_forest_eval(X, y)


# Compute Cramér's V
def cramers_v(chi2, n, k):
    return np.sqrt(chi2 / (n * (k - 1)))

# Create contingency table
contingency_table = pd.crosstab(train["wind_cardinal"], train["rainfall"])

# Compute Chi-Square
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Compute Cramér's V
n = contingency_table.sum().sum()  # Total sample size
k = min(contingency_table.shape)   # Min rows or columns
cramers_v_value = cramers_v(chi2, n, k)

print(f"Chi-Square Value: {chi2:.4f}")
print(f"P-Value: {p:.4f}")
print(f"Cramér's V: {cramers_v_value:.4f}")

if cramers_v_value < 0.1:
    print("Weak relationship (likely a red herring).")
elif cramers_v_value < 0.3:
    print("Moderate relationship (may be worth investigating).")
else:
    print("Strong relationship (should be predictive).")


from sklearn.utils import shuffle
from joblib import Parallel, delayed  # Multiprocessing

# === Prepare Data ===
X_wind = train[["wind_sin", "wind_cos"]]
y = train["rainfall"]

# === Compute Baseline AUC ===
rf_wind = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_wind.fit(X_wind, y)
auc_wind = roc_auc_score(y, rf_wind.predict_proba(X_wind)[:, 1])

# === Define Permutation Test Function ===
def permuted_auc(trial):
    """Shuffles y (not X_wind) to test for feature importance."""
    permuted_y = shuffle(y, random_state=None)  # Shuffle labels instead
    rf_permuted = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, warm_start=False)
    rf_permuted.fit(X_wind, permuted_y)  # Keep X_wind the same, shuffle target
    auc = roc_auc_score(permuted_y, rf_permuted.predict_proba(X_wind)[:, 1])
    
    if (trial + 1) % 50 == 0:
        print(f"Completed {trial + 1}/{n_trials} trials...")
    
    return auc

# === Run Parallelized Permutation Test ===
n_trials = 1000
permuted_aucs = Parallel(n_jobs=-1, backend="loky")(delayed(permuted_auc)(i) for i in range(n_trials))

# === Compute Results ===
mean_permuted_auc = np.mean(permuted_aucs)

# === Print Final Results ===
print("\n=== Permutation Test Results ===")
print(f"Original AUC With Wind: {auc_wind:.4f}")
print(f"Mean AUC With Randomized Wind: {mean_permuted_auc:.4f}")

if auc_wind - mean_permuted_auc < 0.02:
    print("Wind direction is **not a real predictor** (random chance).")
else:
    print("Wind direction **has a real predictive effect**.")


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_wind, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

auc_xgb = roc_auc_score(y_val, xgb.predict_proba(X_val)[:, 1])
print(f"XGBoost AUC: {auc_xgb:.4f}")


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

seasons = train["data_season"].unique()  # Fix: Use data-driven season clusters
seasonal_models = {}

for season in seasons:
    subset = train[train["data_season"] == season]  # Fix: Use dynamic season clusters
    
    if len(subset) < 50:  # Too small to model
        continue
    
    # Proper train-test split (80% train, 20% test)
    X_season = subset[["wind_sin", "wind_cos"]]
    y_season = subset["rainfall"]
    
    X_train, X_test, y_train, y_test = train_test_split(X_season, y_season, test_size=0.2, random_state=42, stratify=y_season)
    
    # Train on TRAINING data only
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)
    
    # Evaluate on TEST data only
    auc_season = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    
    print(f"Season: {season} → AUC: {auc_season:.4f}")
    
    # Store model for later use
    seasonal_models[season] = xgb



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Dictionary to store trained models
seasonal_models = {}

# Train a model for each season
for season in train["data_season"].unique():
    subset = train[train["data_season"] == season]
    
    if len(subset) < 50:  # Skip if too few samples
        continue
    
    # Proper train-test split (80% train, 20% test)
    X_season = subset[["wind_sin", "wind_cos"]]
    y_season = subset["rainfall"]
    
    X_train, X_test, y_train, y_test = train_test_split(X_season, y_season, test_size=0.2, random_state=42)
    
    # Train on TRAINING data only
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Store trained model
    seasonal_models[season] = model  

print("Seasonal models trained on separate training data.")



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# === Function to Predict Rainfall Probability for All Seasons (Optimized) ===
def predict_rainfall_for_seasons(wind_sin, wind_cos):
    """Predicts rainfall probability for all seasons given wind_sin and wind_cos (vectorized)."""
    X_input = pd.DataFrame([[wind_sin, wind_cos]], columns=["wind_sin", "wind_cos"])
    return {season: model.predict_proba(X_input)[:, 1][0] for season, model in seasonal_models.items()}

# === Generate a Grid of Wind Directions (Vectorized) ===
wind_sin_values, wind_cos_values = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
wind_sin_values = wind_sin_values.flatten()
wind_cos_values = wind_cos_values.flatten()

def compute_probabilities_batch(start_idx, end_idx, batch_num, total_batches):
    """Process a batch of wind_sin, wind_cos values to reduce joblib overhead and print progress."""
    batch_results = []
    for i in range(start_idx, end_idx):
        batch_results.append((wind_sin_values[i], wind_cos_values[i], predict_rainfall_for_seasons(wind_sin_values[i], wind_cos_values[i])))
    
    # Print progress update at fixed intervals
    print(f"Completed batch {batch_num}/{total_batches} ({(batch_num / total_batches) * 100:.1f}%)")
    return batch_results

# Process in batches of 250 to reduce job overhead (10 batches of 250 instead of 2500 separate jobs)
n_samples = len(wind_sin_values)
batch_size = 250
n_batches = n_samples // batch_size

# Run computations in parallel (Using threading backend for faster model inference)
print("Running parallel computation for rainfall prediction...")
results = Parallel(n_jobs=-1, backend="threading")(
    delayed(compute_probabilities_batch)(i * batch_size, (i + 1) * batch_size, i + 1, n_batches) for i in range(n_batches)
)
print("All batches completed!")

# === Store Results in Dictionary (Efficient Processing) ===
rainfall_probs = {season: np.zeros((50, 50)) for season in seasonal_models.keys()}
sin_bins = np.linspace(-1, 1, 50)
cos_bins = np.linspace(-1, 1, 50)

for batch in results:
    for wind_sin, wind_cos, probs in batch:
        i = np.digitize(wind_sin, sin_bins) - 1
        j = np.digitize(wind_cos, cos_bins) - 1
        for season, prob in probs.items():
            rainfall_probs[season][i, j] = prob

# === Handle NaN Values in Predictions (Ensured Fix) ===
for season in rainfall_probs.keys():
    matrix = rainfall_probs[season]
    if np.isnan(matrix).any():
        print(f"Warning: NaN values detected in {season} predictions. Replacing with mean.")
        matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix))  # Replace NaN with season's mean probability
        rainfall_probs[season] = matrix  # Store fixed matrix

# === Plot Heatmaps for Predicted Rainfall Probability by Season ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (season, matrix) in zip(axes.flat, rainfall_probs.items()):
    sns.heatmap(matrix, ax=ax, cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False, center=0.5)
    ax.set_title(f"Predicted Rainfall Probability - {season}")

plt.tight_layout()
plt.show()

# === Compare Predicted vs. Actual Rainfall Data ===
seasonal_actual_rainfall = train.groupby("data_season")["rainfall"].mean()  # Actual frequency of rain per season
seasonal_predicted_rainfall = {season: np.mean(matrix) for season, matrix in rainfall_probs.items()}

# Convert to DataFrame
comparison_df = pd.DataFrame({
    "Actual Rainfall Frequency": seasonal_actual_rainfall,
    "Predicted Rainfall Probability": pd.Series(seasonal_predicted_rainfall)
})

# === Bar Chart for Comparison ===
comparison_df.plot(kind="bar", figsize=(10, 6), color=["blue", "red"], alpha=0.7)
plt.title("Actual vs. Predicted Rainfall Probability by Season")
plt.ylabel("Rainfall Probability")
plt.xlabel("Season")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--")
plt.legend(["Actual Rainfall Frequency", "Predicted Probability"])
plt.show()

# === Statistical Summary ===
print("\n=== Statistical Comparison: Actual vs Predicted Rainfall ===")
print(comparison_df)
print("\nCorrelation between actual and predicted: ", comparison_df.corr().iloc[0,1])


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Convert season to categorical
train["data_season"] = train["data_season"].astype("category").cat.codes  # Encode seasons as numbers

# Define features (Ensure there's no leakage)
X = train[["data_season", "wind_sin", "wind_cos"]]  # Only features
y = train["rainfall"]  # Target variable

# Fix: Split into train & test sets (Prevents model from seeing test rainfall)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # Train only on training data

# Evaluate model on test set (Ensures no leakage)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
auc_new_model = roc_auc_score(y_test, y_pred_proba)  # Uses unseen test set

print(f"New Model AUC (With Season, No Leakage): {auc_new_model:.4f}")



from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Encode season labels numerically
season_encoder = LabelEncoder()
train["season_encoded"] = season_encoder.fit_transform(train["data_season"])

X_season_only = train[["season_encoded"]]
y = train["rainfall"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_season_only, y, test_size=0.2, random_state=42, stratify=y)

# Train model
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

# Evaluate model
auc_season_only = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(f"XGBoost AUC (Only Season): {auc_season_only:.4f}")



import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# === Prepare Features and Target ===
X = train[["data_season", "wind_sin", "wind_cos"]]  # Features (Season + Wind)
y = train["rainfall"]  # Target variable

# === Split Data into Training and Testing Sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Train XGBoost Model ===
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

# === Evaluate Model Performance ===
y_pred_proba = xgb.predict_proba(X_test)[:, 1]
auc_season_and_wind = roc_auc_score(y_test, y_pred_proba)
print(f"XGBoost AUC (Season + Wind): {auc_season_and_wind:.4f}")

# === Plot Feature Importance ===
plt.figure(figsize=(10, 5))
plot_importance(xgb, importance_type='weight', max_num_features=10, height=0.8)
plt.title("XGBoost Feature Importance (Season + Wind)")
plt.show()


X_wind_only = train[["wind_sin", "wind_cos"]]
y = train["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(X_wind_only, y, test_size=0.2, random_state=42, stratify=y)

xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

auc_wind_only = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
print(f"XGBoost AUC (Only Wind Features): {auc_wind_only:.4f}")

