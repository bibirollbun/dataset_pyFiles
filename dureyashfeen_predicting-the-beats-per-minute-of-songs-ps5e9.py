# Import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
import joblib  # for saving the model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Correct way to load CSV files
df_tr = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df_ts = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
samp_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


df_tr.head()


df_tr.tail()


df_tr.shape


df_tr.max()


df_tr.min()


df_tr.describe()


df_tr.dtypes


df_tr.isnull().sum()


df_tr.duplicated().sum()


df_tr["BeatsPerMinute"].value_counts().sum()


# Correlation with target
corr = df_tr.corr(numeric_only=True)['BeatsPerMinute'].sort_values(ascending=False)
print(corr)

plt.figure(figsize=(10,6))
sns.barplot(x=corr.values, y=corr.index, palette="viridis")
plt.title("Correlation of Features with BPM")


# Scatterplots with regression line
features = [col for col in df_tr.columns if col not in ['id','BeatsPerMinute']]
for feat in features:
    plt.figure(figsize=(6,4))
    sns.regplot(data=df_tr, x=feat, y="BeatsPerMinute", scatter_kws={'alpha':0.3}, line_kws={"color":"red"})
    plt.title(f"{feat} vs BPM")
    plt.show()


# --- 1. Correlation Heatmap ---
plt.figure(figsize=(10,8))
sns.heatmap(df_tr.drop(columns=['id']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# --- 2. Variance Inflation Factor (VIF) ---
X = df_tr.drop(columns=['id','BeatsPerMinute'])
X_const = add_constant(X)

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i+1)  # +1 to skip constant
                   for i in range(len(X.columns))]

print("\nVariance Inflation Factors:")
print(vif_data.sort_values(by="VIF", ascending=False))

# --- 3. Pairplot for Top Features ---
corr = df_tr.corr(numeric_only=True)['BeatsPerMinute'].sort_values(ascending=False)
top_feats = corr.index[1:5].tolist() + ['BeatsPerMinute']

sns.pairplot(df_tr[top_feats], diag_kind='kde', corner=True)
plt.suptitle("Pairplot of Top Correlated Features with BPM", y=1.02)
plt.show()

# --- 4. PCA (2D) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['BeatsPerMinute'] = df_tr['BeatsPerMinute'].values

plt.figure(figsize=(10,6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="BeatsPerMinute",
                palette="viridis", alpha=0.6)
plt.title("PCA (2D) of Song Features Colored by BPM")
plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), label="BeatsPerMinute")
plt.show()

explained_var = pca.explained_variance_ratio_
print("Explained variance by PC1 and PC2:", explained_var)
print("Total variance captured:", explained_var.sum())

# --- 5. PCA (3D) ---
from mpl_toolkits.mplot3d import Axes3D

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2],
                c=df_tr['BeatsPerMinute'], cmap='viridis', alpha=0.6)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("PCA (3D) of Song Features Colored by BPM")
plt.colorbar(sc, label="BeatsPerMinute")
plt.show()

explained_var3 = pca3.explained_variance_ratio_
print("Explained variance by PC1, PC2, PC3:", explained_var3)
print("Total variance captured (3 PCs):", explained_var3.sum())


# Select numerical columns only
num_cols = df_tr.select_dtypes(include=['int64', 'float64']).columns

# Set figure size
plt.figure(figsize=(15, 10))

# Create box plots for each numerical column
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 4, i)  # Adjust grid size based on number of features
    sns.boxplot(y=df_tr[col], color="skyblue")
    plt.title(f"Boxplot of {col}", fontsize=10)

plt.tight_layout()
plt.show()


print(df_tr.columns)


from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Drop id column since it's not useful
df_fe = df_tr.drop(columns=['id']).copy()

# Separate features and target
X = df_fe.drop(columns=['RhythmScore'])  # Assuming RhythmScore is the target
y = df_fe['RhythmScore']

# 1. Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for readability
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 2. Polynomial Features (Optional â€“ captures interactions/non-linear relations)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Convert back to DataFrame
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

print("Original features shape:", X_scaled.shape)
print("After polynomial features:", X_poly.shape)

# Final dataset for modeling
X_final = X_poly
y_final = y


# Copy dataframe
df_out = df_tr.copy()

# Check if TrackDurationMs_log exists, if not, create it
if "TrackDurationMs_log" not in df_out.columns and "TrackDurationMs" in df_out.columns:
    df_out["TrackDurationMs_log"] = np.log1p(df_out["TrackDurationMs"])

# Select numerical columns again
num_cols = df_out.select_dtypes(include=['int64', 'float64']).columns

# Outlier removal using IQR
for col in num_cols:
    Q1 = df_out[col].quantile(0.25)
    Q3 = df_out[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]

print("âœ… Shape after outlier removal:", df_out.shape)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
# 3. Features & Target
# ==========================
X = df_tr.drop(columns=['BeatsPerMinute', 'id'])  # features
y = df_tr['BeatsPerMinute']                       # target

X_test = df_ts.drop(columns=['id'])

# ==========================
# 4. Train-Test Split
# ==========================
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# 5. Model Training
# ==========================
# LightGBM
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

# Predictions
y_pred = lgb_model.predict(X_valid)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation RÂ²: {r2:.4f}")

# ==========================
# 6. Save Model
# ==========================
joblib.dump(lgb_model, "best_model_lgb.pkl")
print("âœ… Model saved as best_model_lgb.pkl")

# ==========================
# 7. Final Submission
# ==========================
final_preds = lgb_model.predict(X_test)
samp_sub['BeatsPerMinute'] = final_preds
samp_sub.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")

