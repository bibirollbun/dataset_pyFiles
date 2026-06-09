from IPython.display import display, HTML

img_url = "https://www.kaggle.com/competitions/91720/images/header"

display(HTML(f'''
<div style="text-align: center;">
    <img src="{img_url}" width="800">
</div>
'''))


from IPython.display import display, HTML

display(HTML("""
<style>
.toc a {
    text-decoration: none;
    color: #0077cc;
    font-weight: bold;
    padding: 5px;
    display: block;
}
.toc a:hover {
    background: #e0f0ff;
    cursor: pointer;
}
</style>

<div class="toc">
    <h3>Table of Contents</h3>
    <a href="#dataloading">1. Data Loading</a>
    <a href="#eda">2. Exploratory Data Analysis</a>
    <a href="#preprocessing">3. Data Preprocessing</a>
    <a href="#feature-engineering">4. Feature Engineering</a>
    <a href="#modeling">5. Modeling</a>
    <a href="#evaluation">6. Evaluation</a>
    <a href="#conclusion">7. Conclusion</a>
</div>
"""))



# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

from xgboost import XGBRegressor



warnings.filterwarnings("ignore")

# Settings
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)


test  = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")  


# Quick look
train.head()


print("Shape of dataset:", train.shape)



train.info()




train.describe(include='all').T



# Missing values
print("Missing Values:", train.isnull().sum())


#Duplicate rows
print("Duplicate rows:", train.duplicated().sum())


import matplotlib.pyplot as plt
import seaborn as sns

target = "BeatsPerMinute"

def plot_target_distribution(train, target, bins=50):
    """
    Generates a comprehensive plot for analyzing a regression target variable.
    
    The plot includes a histogram, KDE, boxplot, and key statistical annotations.
    
    Parameters:
    - train (pd.DataFrame): The input dataframe.
    - target (str): The name of the target column.
    - bins (int): The number of bins for the histogram.
    """
    # --- Calculate Statistics ---
    mean_val = train[target].mean()
    
    median_val = train[target].median()
    std_val = train[target].std()
    skew_val = train[target].skew()
    kurt_val = train[target].kurt()

    # --- Create the plot ---
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, 
        gridspec_kw={'height_ratios': (0.8, 0.2)}
    )
    
    # --- Histogram and KDE (Top Plot) ---
    sns.histplot(train[target], ax=ax_hist, kde=True, bins=bins, line_kws={'linewidth': 2})
    
    # Add vertical lines for mean and median
    ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax_hist.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
    
    ax_hist.set_title(f'Distribution of {target}', fontsize=16, weight='bold')
    ax_hist.set_ylabel('Frequency', fontsize=12)
    ax_hist.legend(loc='upper right')
    ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
    ax_hist.set_xlabel('')  # Hide x-label for the top plot

    # --- Statistical Annotations ---
    stats_text = (
        f"Std. Dev: {std_val:.2f}\n"
        f"Skewness: {skew_val:.2f}\n"
        f"Kurtosis: {kurt_val:.2f}"
    )
    ax_hist.text(0.97, 0.97, stats_text, transform=ax_hist.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))

    # --- Boxplot (Bottom Plot) ---
    sns.boxplot(x=train[target], ax=ax_box, color='skyblue')
    ax_box.set_xlabel(target, fontsize=12)
    ax_box.set_ylabel(' ', fontsize=12)

    # --- Final Touches ---
    plt.suptitle(f'Detailed Analysis of Target Variable: {target}', fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()
plot_target_distribution(train, target)



# ========================
# 4. Feature Distributions
# ========================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Select numeric features (excluding ID if present)
num_features = train.select_dtypes(include=[np.number]).columns.tolist()
# Columns to exclude
exclude_cols = ["id", "BeatsPerMinute"]
num_features = [col for col in num_features if col not in exclude_cols]

# Define grid size automatically (rows & cols)

n_features = len(num_features)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

# Plot each feature
for i, col in enumerate(num_features):
    sns.histplot(train[col], bins=30, kde=True, ax=axes[i], color="skyblue")
    axes[i].set_title(f"Distribution of {col}", fontsize=12, weight="bold")
    axes[i].set_xlabel("")
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)

# Remove empty subplots (if any)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Feature Distributions", fontsize=16, weight="bold", y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# Select numeric features (excluding ID if present)
num_features = train.select_dtypes(include=[np.number]).columns.tolist()
# Columns to exclude
exclude_cols = ["id", "BeatsPerMinute"]
num_features = [col for col in num_features if col not in exclude_cols]

# Define grid size automaticid", "BeatsPerMinute"
n_features = len(num_features)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

# Plot each feature as boxplot
for i, col in enumerate(num_features):
    sns.boxplot(y=train[col], ax=axes[i], color="skyblue", showfliers=True, whis=1.5)
    axes[i].set_title(f"Boxplot of {col}", fontsize=12, weight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)

# Remove empty subplots (if any)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Feature Boxplots", fontsize=16, weight="bold", y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Select numeric columns, excluding 'id' if present
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = [c for c in numerical_cols if c.lower() != "id"]

# Define color palette
palette = sns.color_palette("husl", len(numerical_cols))

# Grid layout: e.g., 4 columns
ncols = 4
nrows = -(-len(numerical_cols) // ncols)  # ceiling division

plt.figure(figsize=(5*ncols, 4*nrows))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(nrows, ncols, i)
    sns.boxplot(y=train[col], color=palette[i-1], showfliers=True, whis=1.5)
    plt.title(f'Boxplot of {col}', fontsize=12)
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.suptitle("Boxplots of Numerical Features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()



num_features = train.select_dtypes(include=[np.number]).columns.tolist()

# Columns to exclude

exclude_cols = ["id"]
num_features = [col for col in num_features if col not in exclude_cols]

# ==================================
# 5. Enhanced Correlation Analysis
# ==================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- You already have this from the previous step ---
# Assuming 'num_features' is your list of numeric columns
# ---------------------------------------------------

# 1. Calculate the correlation matrix once
corr_matrix = train[num_features].corr()

# 2. Create a mask to hide the redundant upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 3. Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# 4. **Expert Tip**: Choose a better diverging colormap and center it
# 'vlag' is a great blue-white-red palette. 'icefire' is another good one.
# Centering at 0 ensures that 0 correlation is neutral (white).
sns.heatmap(corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap="vlag", # A better color palette for correlations
            vmin=-1, vmax=1, # Lock the color scale
            center=0,
            linewidths=.5, # Add lines between cells
            cbar_kws={"shrink": .8}) # Shrink the color bar a bit

plt.title("Correlation Heatmap of Numeric Features", fontsize=16, weight="bold")
plt.xticks(rotation=45, ha='right') # Rotate labels for readability
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ====================================================
# 6. Correlation with Target Variable (Visualized)
# ====================================================

# Assuming 'target_col' is the name of your target variable string
target_col = 'BeatsPerMinute' # Replace with your actual target name

# We can reuse the corr_matrix from before
if 'corr_matrix' not in locals():
    corr_matrix = train[num_features].corr()

# Get correlations with the target, drop the target's self-correlation, and sort
target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)

# Plotting
plt.figure(figsize=(10, 8))
sns.barplot(x=target_corr.values, y=target_corr.index, palette="vlag", orient='h')

plt.title(f"Feature Correlation with {target_col}", fontsize=16, weight="bold")
plt.xlabel("Correlation Coefficient", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# You can still print the sorted values for exact numbers
print(f"\n--- Correlation with {target_col} ---")
print(target_corr)


# ==========================================================
# 7. FAST Targeted Pairplot via Random Sampling
# ==========================================================
import seaborn as sns
import matplotlib.pyplot as plt

# --- Define your features and target ---
num_features = train.select_dtypes(include=[np.number]).columns.tolist()

# Columns to exclude

exclude_cols = ["id"]
num_features = [col for col in num_features if col not in exclude_cols]
target_col = 'BeatsPerMinute'
predictor_features = [col for col in num_features if col != target_col]
# ----------------------------------------

# **THE KEY IMPROVEMENT: Create a smaller, random sample for plotting**
n_samples = 3000 # Start with 1k, increase if needed
if len(train) > n_samples:
    plot_df = train.sample(n=n_samples, random_state=42)
else:
    plot_df = train

print(f"Generating targeted pairplot on a sample of {len(plot_df)} data points for speed...")

# Now, run the EXACT SAME plotting code, but on the smaller 'plot_df'
g = sns.pairplot(
    plot_df, # Use the sample, not the full 'train' dataframe
    x_vars=predictor_features,
    y_vars=[target_col],
    kind='scatter',
    height=4,
    aspect=1.2,
    plot_kws={'alpha': 0.4, 's': 20, 'edgecolor': None} # Can use slightly less alpha
)

g.fig.suptitle(f"Relationships Between Features and {target_col} (on Sampled Data)", y=1.02, fontsize=16, weight='bold')
plt.show()


# Load train data
df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")

# Features & target
X = df.drop(columns=["id", "BeatsPerMinute"])
y = df["BeatsPerMinute"]


# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Baseline XGBoost
model = XGBRegressor(
    n_estimators=200,       # keep it small for quick testing
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


# Train
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50, early_stopping_rounds=20)


# Predict on validation
preds = model.predict(X_val)
rmse = mean_squared_error(y_val, preds, squared=False)
print("Validation RMSE:", rmse)


# Feature importance
importances = model.feature_importances_
feature_names = X.columns
fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
fi_df = fi_df.sort_values(by="Importance", ascending=False)


# Plot
plt.figure(figsize=(10,6))
plt.barh(fi_df["Feature"], fi_df["Importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")
plt.show()


# Optional: display sorted importance table
print(fi_df)


import numpy as np
import pandas as pd

# --- Duration features ---
train["TrackDurationMin"] = train["TrackDurationMs"] / 60000.0
train["TrackDurationSec"] = train["TrackDurationMs"] / 1000.0
train["LogDuration"] = np.log1p(train["TrackDurationMs"])

test["TrackDurationMin"] = test["TrackDurationMs"] / 60000.0
test["TrackDurationSec"] = test["TrackDurationMs"] / 1000.0
test["LogDuration"] = np.log1p(test["TrackDurationMs"])

# --- Interaction features ---
train["Rhythm_Energy"] = train["RhythmScore"] * train["Energy"]
train["Acoustic_Vocal"] = train["AcousticQuality"] * train["VocalContent"]
train["Mood_Live"] = train["MoodScore"] * train["LivePerformanceLikelihood"]

test["Rhythm_Energy"] = test["RhythmScore"] * test["Energy"]
test["Acoustic_Vocal"] = test["AcousticQuality"] * test["VocalContent"]
test["Mood_Live"] = test["MoodScore"] * test["LivePerformanceLikelihood"]

# --- Pairwise ratios (safe division) ---
eps = 1e-6
train["Energy_over_Rhythm"] = train["Energy"] / (train["RhythmScore"] + eps)
train["Vocal_over_Acoustic"] = train["VocalContent"] / (train["AcousticQuality"] + eps)

test["Energy_over_Rhythm"] = test["Energy"] / (test["RhythmScore"] + eps)
test["Vocal_over_Acoustic"] = test["VocalContent"] / (test["AcousticQuality"] + eps)

# --- Polynomial features ---
for col in ["RhythmScore","Energy","MoodScore","AudioLoudness","AcousticQuality","VocalContent"]:
    train[f"{col}_sq"] = train[col] ** 2
    test[f"{col}_sq"] = test[col] ** 2

# --- Duration binning ---
train["DurationBin"] = pd.qcut(train["TrackDurationMin"], q=10, duplicates='drop').cat.codes
test["DurationBin"] = pd.qcut(test["TrackDurationMin"], q=10, duplicates='drop').cat.codes


print("Train columns after FE:", len(train.columns))
display(train.head())


# Prepare features & target (drop id and target column)
X_fe = train.drop(columns=["id", "BeatsPerMinute"])
y_fe = train["BeatsPerMinute"]


# Train/validation split
X_train_fe, X_val_fe, y_train_fe, y_val_fe = train_test_split(
    X_fe, y_fe, test_size=0.2, random_state=42
)


# XGBoost model
model_fe = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


# Train
model_fe.fit(
    X_train_fe, y_train_fe,
    eval_set=[(X_val_fe, y_val_fe)],
    verbose=50,
    early_stopping_rounds=20
)


# Predict on validation
preds_fe = model_fe.predict(X_val_fe)
rmse_fe = mean_squared_error(y_val_fe, preds_fe, squared=False)
print("Validation RMSE:", rmse_fe)


# Approximate Â±10 BPM coverage
tolerance = 10
accuracy_like_fe = np.mean(np.abs(preds_fe - y_val_fe) <= tolerance)
print("Approx. Accuracy within Â±10 BPM:", accuracy_like_fe)


# Prepare test features (drop id column)
X_test = test.drop(columns=["id"])

# Predict BPM
test_preds = model_fe.predict(X_test)

# Prepare submission DataFrame
submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": test_preds
})

# Save to CSV
submission_file = "submission.csv"
submission.to_csv(submission_file, index=False)
print(f"Submission saved to {submission_file}")





