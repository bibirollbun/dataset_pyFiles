import numpy as np
import pandas as pd
SEED = 42
np.random.seed(SEED)

df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


import matplotlib.pyplot as plt
import seaborn as sns

print("===== Dataset Overview =====")
print(df.info())
print("\n===== First 5 Rows =====")
print(df.head())


categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical Features:", categorical_cols)
print("Numerical Features:", numerical_cols)

# Check target variable type
target_col = "accident_risk"
if np.issubdtype(df[target_col].dtype, np.number):
    print(f"\nâœ… '{target_col}' is numerical â€” suitable for regression.")
else:
    print(f"\nâš ï¸� '{target_col}' is categorical â€” consider classification instead.")



print("\n===== Descriptive Statistics (Numerical) =====")
display(df[numerical_cols].describe().T)

# Additional insights
print("\nMissing Values per Column:")
print(df.isna().sum())



print("\n===== Numerical Feature Distributions =====")
for col in numerical_cols:
    print(f"\nFeature: {col}")
    print("-" * (10 + len(col)))
    print(f"Min Value     : {df[col].min():.3f}")
    print(f"Max Value     : {df[col].max():.3f}")
    print(f"Mean          : {df[col].mean():.3f}")
    print(f"Median        : {df[col].median():.3f}")
    print(f"Std Deviation : {df[col].std():.3f}")

    # Skewness and Kurtosis â€” optional shape indicators
    print(f"Skewness      : {df[col].skew():.3f}")
    print(f"Kurtosis      : {df[col].kurt():.3f}")

    # Simple bin-based frequency summary
    counts, bins = np.histogram(df[col], bins=5)
    print("\nApproximate Frequency Distribution:")
    for i in range(len(counts)):
        print(f"  {bins[i]:.2f} â€“ {bins[i+1]:.2f} : {counts[i]} samples")
    print("-" * 40)

print("\n===== Categorical Feature Summaries =====")
for col in categorical_cols:
    print(f"\nFeature: {col}")
    print("-" * (10 + len(col)))

    value_counts = df[col].value_counts(dropna=False)
    proportions = df[col].value_counts(normalize=True, dropna=False) * 100

    summary_df = pd.DataFrame({
        "Count": value_counts,
        "Percentage": proportions.round(2)
    })

    print(summary_df)
    print("-" * 40)



from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder


# Encode categorical features temporarily for correlation analysis
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Compute correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Correlation with the target variable 'accident_risk'
target_corr = corr_matrix['accident_risk'].sort_values(ascending=False)
print("Correlation with accident_risk:")
print(target_corr)

# Visualize correlation matrix (heatmap)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()


# Select only numerical features for VIF
numerical_features = df.select_dtypes(include=[np.number]).drop(columns=['id', 'accident_risk'])

# Compute VIF for each numerical feature
vif_data = pd.DataFrame()
vif_data["feature"] = numerical_features.columns
vif_data["VIF"] = [variance_inflation_factor(numerical_features.values, i)
                   for i in range(numerical_features.shape[1])]

print("\nVariance Inflation Factor (VIF) Results:")
print(vif_data.sort_values(by="VIF", ascending=False))

# Identify potential multicollinearity
high_vif = vif_data[vif_data['VIF'] > 10]
if not high_vif.empty:
    print("\nâš ï¸� Features with potential multicollinearity:")
    print(high_vif)


# Summary statistics for numerical columns
desc_stats = df_test.describe().T
desc_stats['missing_values'] = df_test.isna().sum()
desc_stats['dtype'] = df_test.dtypes
print("\nDescriptive Statistics Summary:")
print(desc_stats)

# Compare scale ranges
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_test.select_dtypes(include=[np.number]))
plt.title("Scale Comparison of Numerical Features")
plt.xticks(rotation=45)
plt.show()


from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


target_col = "accident_risk"

# Drop non-informative ID column
df_prep = df.drop(columns=["id"], errors='ignore')

# Separate features and target
X = df_prep.drop(columns=[target_col])
y = df_prep[target_col]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Interaction Feature: curvature Ã— speed_limit
        X["curvature_speed_interaction"] = X["curvature"] * X["speed_limit"]
        
        # Binning speed_limit into categories
        X["speed_category"] = pd.cut(
            X["speed_limit"],
            bins=[0, 40, 60, np.inf],
            labels=["low", "medium", "high"]
        )
        
        # Aggregated feature: avg accidents per road_type
        mean_accidents_per_type = X.groupby("road_type")["num_reported_accidents"].transform("mean")
        X["avg_accidents_by_type"] = mean_accidents_per_type
        
        return X


categorical_cols = ["road_type", "lighting", "weather", "time_of_day", 
                    "holiday", "school_season", "road_signs_present", "public_road"]

numeric_cols = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]

# Derived from Feature Engineering
one_hot_cols = ["road_type", "lighting", "weather", "time_of_day", "speed_category"]
label_cols   = ["holiday", "school_season", "road_signs_present", "public_road"]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_cols),
    ('label', 'passthrough', label_cols)
], remainder='drop')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, one_hot_cols + label_cols)
    ],
    remainder='drop'
)


data_prep_pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineer()),
    ('preprocessing', preprocessor)
])


X_prepared_array = data_prep_pipeline.fit_transform(X)

# Get feature names from the preprocessor
feature_names = data_prep_pipeline.named_steps['preprocessing'].get_feature_names_out()

# Convert to DataFrame
X_train_prepared_final = pd.DataFrame(X_prepared_array, columns=feature_names)

print("Shape after preprocessing:", X_train_prepared_final.shape)
print("Columns preview:", X_train_prepared_final.columns[:10].tolist())


X_test = df_test.drop(columns=["id"], errors='ignore').copy()


X_prepared_array = data_prep_pipeline.fit_transform(X_test)

# Get feature names from the preprocessor
feature_names = data_prep_pipeline.named_steps['preprocessing'].get_feature_names_out()

# Convert to DataFrame
X_test_prepared_final = pd.DataFrame(X_prepared_array, columns=feature_names)

print("Shape after preprocessing:", X_test_prepared_final.shape)
print("Columns preview:", X_test_prepared_final.columns[:10].tolist())


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Train directly using DataFrame
xgb_model.fit(X_train_prepared_final, y)


importance = (
    pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    })
    .sort_values(by='Importance', ascending=False)
    .reset_index(drop=True)
)



print("Top 15 Most Important Features (XGBoost Quick Evaluation):\n")
for idx, row in importance.head(15).iterrows():
    print(f"{idx+1:02d}. {row['Feature']:<45} Importance: {row['Importance']:.6f}")



import pandas as pd
import numpy as np

def add_enhanced_features(X: pd.DataFrame) -> pd.DataFrame:
    X_enhanced = X.copy()

    # Lighting Ã— Weather â†’ combine visual & weather conditions
    X_enhanced["lighting_weather_risk"] = (
        X_enhanced["cat__onehot__lighting_2"] *
        (X_enhanced["cat__onehot__weather_1"] + X_enhanced["cat__onehot__weather_2"])
    )

    # Speed limit Ã— Curvature â†’ risky road segments
    X_enhanced["speed_curvature_risk"] = (
        X_enhanced["num__speed_limit"] * X_enhanced["num__curvature"]
    )

    # Speed limit Ã— Lighting (night)
    X_enhanced["speed_night_risk"] = (
        X_enhanced["num__speed_limit"] * X_enhanced["cat__onehot__lighting_2"]
    )

    # Curvature Ã— Weather (rainy)
    X_enhanced["curvature_rain_risk"] = (
        X_enhanced["num__curvature"] * X_enhanced["cat__onehot__weather_1"]
    )
    
    # Combine time_of_day_evening + holiday + public_road â†’ high-risk behavior feature
    X_enhanced["high_risk_period"] = (
        (X_enhanced["cat__onehot__time_of_day_2"] == 1).astype(int) &
        (X_enhanced["cat__label__holiday"] == 1).astype(int) &
        (X_enhanced["cat__label__public_road"] == 1).astype(int)
    ).astype(int)

    # Average accidents by lighting + weather combination
    X_enhanced["avg_accidents_by_lighting_weather"] = (
        X_enhanced
        .groupby(["cat__onehot__lighting_0", "cat__onehot__weather_0"])["num__num_reported_accidents"]
        .transform("mean")
    )

    # Average accidents per lighting condition
    X_enhanced["avg_accidents_by_lighting"] = (
        X_enhanced
        .groupby("cat__onehot__lighting_0")["num__num_reported_accidents"]
        .transform("mean")
    )

    print(f"Columns before: {X.shape[1]} | after: {X_enhanced.shape[1]}")

    return X_enhanced


X_train_prepared_final_enhanced = add_enhanced_features(X_train_prepared_final)


# Pastikan X_test_prepared_final sudah ada di memori
X_test_prepared_final = X_test_prepared_final.rename(columns={
    # Road type
    "cat__onehot__road_type_highway": "cat__onehot__road_type_0",
    "cat__onehot__road_type_rural": "cat__onehot__road_type_1",
    "cat__onehot__road_type_urban": "cat__onehot__road_type_2",

    # Lighting
    "cat__onehot__lighting_daylight": "cat__onehot__lighting_0",
    "cat__onehot__lighting_dim": "cat__onehot__lighting_1",
    "cat__onehot__lighting_night": "cat__onehot__lighting_2",

    # Weather
    "cat__onehot__weather_clear": "cat__onehot__weather_0",
    "cat__onehot__weather_foggy": "cat__onehot__weather_1",
    "cat__onehot__weather_rainy": "cat__onehot__weather_2",

    # Time of day
    "cat__onehot__time_of_day_morning": "cat__onehot__time_of_day_0",
    "cat__onehot__time_of_day_afternoon": "cat__onehot__time_of_day_1",
    "cat__onehot__time_of_day_evening": "cat__onehot__time_of_day_2"
})


X_test_prepared_final_enhanced = add_enhanced_features(X_test_prepared_final)


model_xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model_xgb.fit(X_train_prepared_final_enhanced, y)


feature_importances = pd.DataFrame({
    "Feature": X_train_prepared_final_enhanced.columns,
    "Importance": model_xgb.feature_importances_
}).sort_values(by="Importance", ascending=False)


plt.figure(figsize=(10, 12))
plt.barh(feature_importances["Feature"][:30][::-1],  # top 30 teratas
         feature_importances["Importance"][:30][::-1])
plt.title("Top 30 Feature Importances - XGBoost")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\nğŸ”¥ Top 15 Most Important Features:")
print(feature_importances.head(15))


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor


def enhance_features_with_scaling(X):
    X = X.copy()

    interaction_features = [
        "lighting_weather_risk",
        "speed_curvature_risk",
        "speed_night_risk",
        "curvature_rain_risk",
        "high_risk_period",
        "avg_accidents_by_lighting_weather",
        "avg_accidents_by_lighting"
    ]

    existing_interactions = [f for f in interaction_features if f in X.columns]
    
    if existing_interactions:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X[existing_interactions]),
            columns=[f"{col}_scaled" for col in existing_interactions],
            index=X.index
        )
        X = pd.concat([X, X_scaled], axis=1)

    def get_safe(col):
        """Safely access columns (return zeros if missing)."""
        return X[col] if col in X.columns else 0

    X["weather_speed_risk"] = get_safe("cat__onehot__weather_1") * get_safe("num__speed_limit")
    X["lighting_curvature_risk"] = get_safe("cat__onehot__lighting_2") * get_safe("num__curvature")
    X["speed_weather_night"] = (
        get_safe("num__speed_limit") *
        get_safe("cat__onehot__lighting_2") *
        get_safe("cat__onehot__weather_1")
    )

    for prefix in ["cat__onehot__weather_", "cat__onehot__lighting_"]:
        rare_cols = [c for c in X.columns if c.startswith(prefix) and X[c].sum() < 5]
        if rare_cols:
            X[f"{prefix}others"] = X[rare_cols].sum(axis=1)
            X.drop(columns=rare_cols, inplace=True, errors="ignore")

    print(f"âœ… Enhanced features added: {X.shape[1]} columns total.")
    return X

# Run the feature enhancement
X_enhanced = enhance_features_with_scaling(X_train_prepared_final_enhanced)


X_test_enhanced = enhance_features_with_scaling(X_test_prepared_final_enhanced)


# Train XGBoost on the enhanced dataset
model_xgb = XGBRegressor(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model_xgb.fit(X_enhanced, y)

importance_df = pd.DataFrame({
    "Feature": X_enhanced.columns,
    "Importance": model_xgb.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Remove features with zero or near-zero importance
importance_threshold = 0.005 
important_features = importance_df[importance_df["Importance"] > importance_threshold]["Feature"].tolist()

X_filtered = X_enhanced[important_features]
print(f"âœ… Features retained after removing low-importance (<{importance_threshold}): {X_filtered.shape[1]}")



corr_matrix = X_filtered.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
X_final = X_filtered.drop(columns=high_corr_features, errors="ignore")

print(f"âœ… After correlation pruning (>0.9): {X_final.shape[1]} features remain.")


model_final = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model_final.fit(X_final, y)

final_importance = pd.DataFrame({
    "Feature": X_final.columns,
    "Importance": model_final.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 15 Most Important Features (After Robust Feature Selection):\n")
for i, row in final_importance.head(15).iterrows():
    print(f"{i+1:02d}. {row['Feature']:<40} Importance: {row['Importance']:.6f}")


selected_features = X_final.columns.tolist()


X_test_aligned = X_test_enhanced.reindex(columns=selected_features, fill_value=0)
print(f"Test data aligned: {X_test_aligned.shape[1]} features (same as training)")


import numpy as np
import pandas as pd

def create_advanced_math_features(X):
    X_new = X.copy()
    
    for col in [
        "num__speed_limit",
        "num__curvature",
        "lighting_curvature_risk",
        "speed_night_risk_scaled",
        "speed_curvature_risk_scaled",
    ]:
        Î¼ = X_new[col].mean()
        Ïƒ = X_new[col].std() + 1e-6
        X_new[f"{col}_hat"] = (X_new[col] - Î¼) / Ïƒ  # xÌ‚ = (x - Î¼)/Ïƒ

    X_new["log_speed_limit"] = np.log1p(X_new["num__speed_limit"])
    X_new["exp_curvature"] = np.exp(np.abs(X_new["num__curvature"]) / 10)
    X_new["sqrt_accidents"] = np.sqrt(np.abs(X_new["num__num_reported_accidents"]) + 1)
    
    X_new["interaction_speed_curvature"] = (
        X_new["num__speed_limit_hat"] * X_new["num__curvature_hat"]
    )

    X_new["interaction_lighting_speed"] = (
        X_new["lighting_curvature_risk_hat"] * X_new["num__speed_limit_hat"]
    )

    X_new["interaction_weather_risk"] = (
        X_new["cat__onehot__weather_0"] * X_new["lighting_weather_risk"]
    )

    X_new["interaction_night_risk"] = (
        X_new["speed_night_risk_scaled_hat"] * X_new["cat__onehot__lighting_2"]
    )

    X_new["curvature_per_speed"] = (
        np.abs(X_new["num__curvature"]) / (X_new["num__speed_limit"] + 1e-6)
    )

    X_new["risk_ratio"] = (
        np.abs(X_new["lighting_curvature_risk"]) /
        (np.abs(X_new["speed_curvature_risk_scaled"]) + 1e-6)
    )

    X_new["accident_density"] = (
        X_new["num__num_reported_accidents"] /
        (X_new["avg_accidents_by_lighting_weather_scaled"] + 1e-6)
    )

    X_new["geo_mean_speed_risk"] = np.sqrt(
        np.abs(X_new["num__speed_limit"] * X_new["speed_night_risk_scaled"]) + 1e-6
    )

    X_new["harmonic_mean_speed_risk"] = (
        2 * X_new["num__speed_limit"] * X_new["speed_night_risk_scaled"]
    ) / (X_new["num__speed_limit"] + X_new["speed_night_risk_scaled"] + 1e-6)

    X_new["entropy_risk"] = -(
        X_new["lighting_curvature_risk"] *
        np.log(np.abs(X_new["lighting_curvature_risk"]) + 1e-6)
    )

    X_new["gradient_speed_curvature"] = (
        (X_new["num__speed_limit"] - X_new["num__curvature"]) /
        (np.abs(X_new["num__speed_limit"] + X_new["num__curvature"]) + 1e-6)
    )

    X_new["delta_risk_balance"] = (
        X_new["speed_night_risk_scaled"] - X_new["speed_curvature_risk_scaled"]
    )

    X_new["weighted_safety_index"] = (
        0.4 * X_new["cat__onehot__lighting_2"]
        + 0.3 * X_new["cat__onehot__weather_0"]
        + 0.2 * X_new["num__curvature_hat"]
        - 0.1 * X_new["risk_ratio"]
    )

    X_new["advanced_risk_intensity"] = (
        X_new["lighting_curvature_risk_hat"]**2 +
        X_new["speed_night_risk_scaled_hat"]**2 +
        X_new["num__curvature_hat"]**2
    ) ** 0.5  # âˆš(Î£ xÌ‚Â²)

    print(f"âœ… Advanced features successfully generated: {X_new.shape[1]} columns total.")
    return X_new

X_train_advanced = create_advanced_math_features(X_final)
print(X_train_advanced.head())
print("Total columns after enhancement:", X_train_advanced.shape[1])



X_test_advanced = create_advanced_math_features(X_test_aligned)
print(X_test_advanced.head())
print("Total columns after enhancement:", X_test_advanced.shape[1])


from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_feature_importance(X, y, top_n=20, random_state=42):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        gamma=0.2,           # minimum loss reduction for split
        min_child_weight=5,  # require min sum of instance weight
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(
        X, y,
        eval_set=[(X, y)],
        verbose=False
    )

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print(f"\nâœ… XGBoost Model Trained Successfully!")
    print(f"ğŸ“Š Total Features Evaluated: {X.shape[1]}")
    print(f"â­� Top {top_n} Most Important Features:\n")

    for i, row in importance_df.head(top_n).iterrows():
        print(f"{i+1:02d}. {row['Feature']:<45} Importance: {row['Importance']:.6f}")

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df.head(top_n)["Feature"][::-1],
        importance_df.head(top_n)["Importance"][::-1]
    )
    plt.title(f"Top {top_n} Feature Importances (XGBoost)", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return importance_df, model

importance_df, model_xgb = evaluate_feature_importance(X_train_advanced, y)


import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

def balance_feature_influence(X, noise_strength=0.01, variance_ratio=0.95, random_state=42):
    X = X.copy()
    np.random.seed(random_state)

    # Step 1: Fill missing values
    X = X.fillna(X.mean(numeric_only=True))

    # Step 2: Drop constant columns
    constant_cols = [c for c in X.columns if X[c].std() == 0]
    if constant_cols:
        print(f"âš ï¸� Dropping constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    # Step 3: Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Step 4: Advanced statistical transformation
    X_transformed = pd.DataFrame(index=X.index)
    for col in X_scaled.columns:
        x = X_scaled[col].values.reshape(-1, 1)
        try:
            pt = PowerTransformer(method='yeo-johnson')
            x_hat = pt.fit_transform(x)
        except Exception:
            qt = QuantileTransformer(output_distribution='normal', random_state=random_state)
            x_hat = qt.fit_transform(x)

        # Rumus matematis tingkat lanjut
        # xÌ‚_i = log(1 + |x|) * sign(x) + sin(x^3) / (1 + x^2)
        x_hat = np.log1p(np.abs(x_hat)) * np.sign(x_hat) + np.sin(x_hat ** 3) / (1 + x_hat ** 2)
        X_transformed[col] = x_hat.flatten()

    # Step 5: Add slight noise (stochastic regularization)
    X_hat = X_transformed + np.random.normal(0, noise_strength, X_transformed.shape)

    # Step 6: PCA smoothing (tanpa ubah nama kolom)
    pca = PCA(n_components=variance_ratio, random_state=random_state)
    X_pca = pca.fit_transform(X_hat)

    # Proyeksi balik agar fitur lama tetap ada (approximation)
    X_final = pd.DataFrame(
        np.dot(X_pca, pca.components_),
        columns=X.columns,
        index=X.index
    )

    # Step 7: Normalize output
    X_final = (X_final - X_final.mean()) / (X_final.std() + 1e-8)

    print(f"âœ… Advanced balancing done: {X_final.shape}, Variance preserved = {np.sum(pca.explained_variance_ratio_):.3f}")
    return X_final


X_advanced_balanced = balance_feature_influence(X_train_advanced)


importance_df, model_xgb = evaluate_feature_importance(X_advanced_balanced, y)


X_test_advanced_balanced = balance_feature_influence(X_test_advanced)


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel


X_train, X_valid, y_train, y_valid = train_test_split(
    X_advanced_balanced, y, test_size=0.2, random_state=42
)


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


xgb_fast = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=150,    
    learning_rate=0.07,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.2,
    min_child_weight=3,
    tree_method="hist",    
    random_state=42,
    n_jobs=2                   
)

lgb_fast = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.07,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.2,
    boosting_type="gbdt",
    random_state=42,
    n_jobs=2               
)

rf_fast = RandomForestRegressor(
    n_estimators=100,          
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=2
)

ensemble_fast = VotingRegressor(
    estimators=[
        ("lgb", lgb_fast),
        ("xgb", xgb_fast),
        ("rf", rf_fast)
    ],
    weights=[0.6, 0.3, 0.1]  
)

ensemble_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=False)),  
    ("feature_selection", SelectFromModel(
        LGBMRegressor(
            n_estimators=80,         
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=2
        ),
        threshold="median"
    )),
    ("ensemble", ensemble_fast)
])

param_distributions = {
    "ensemble__lgb__num_leaves": [31],
    "ensemble__lgb__subsample": [0.8],
    "ensemble__lgb__colsample_bytree": [0.8],
    "ensemble__xgb__max_depth": [4],
    "ensemble__xgb__subsample": [0.8],
    "ensemble__rf__max_depth": [8],
}

random_search = RandomizedSearchCV(
    estimator=ensemble_pipeline,
    param_distributions=param_distributions,
    n_iter=3,                   
    cv=2,                   
    scoring="neg_root_mean_squared_error",
    verbose=1,
    random_state=42,
    n_jobs=2                    
)

random_search.fit(X_train, y_train)

best_pipeline = random_search.best_estimator_
best_ensemble = best_pipeline.named_steps["ensemble"]


y_pred = best_pipeline.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print("\nâœ… Best Parameters:")
print(random_search.best_params_)
print(f"âœ… Validation RMSE: {rmse:.4f}")


ensemble = best_pipeline.named_steps["ensemble"]

lgb_model = ensemble.estimators[0][1]
xgb_model = ensemble.estimators[1][1]

lgb_model.set_params(reg_lambda=2.0, reg_alpha=0.3, num_leaves=25)
xgb_model.set_params(reg_lambda=2.0, reg_alpha=0.3)

best_pipeline.fit(X_train, y_train)

y_test_pred = best_pipeline.predict(X_test_advanced_balanced)
df_test['accident_risk'] = y_test_pred


df_submission = df_test[['id', 'accident_risk']]
df_submission.to_csv('submission.csv', index=False, sep=',')

