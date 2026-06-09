import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Define global variables for features and target
FEATURES = [
    'RhythmScore',
    'AudioLoudness',
    'VocalContent',
    'AcousticQuality',
    'InstrumentalScore',
    'LivePerformanceLikelihood',
    'MoodScore',
    'TrackDurationMs',
    'Energy'
]
TARGET = 'BeatsPerMinute'
OPTIMAL_K = 5


print("Loading and preparing data...")

# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Handle missing values using median imputation
for col in FEATURES:
    if train_df[col].isnull().any():
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
        test_df[col].fillna(median_val, inplace=True)

# Extract features and target
X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].copy()
X_test = test_df[FEATURES].copy()

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data loaded, imputed, and scaled.")


print("Starting MiniBatchKMeans Clustering...")

# Elbow Method to find optimal K (saved as elbow_curve.png)
inertias = []
K_range = range(2, 11)

for k in K_range:
    mbk = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=1024,
        n_init=3,
        max_iter=100
    )
    mbk.fit(X_train_scaled)
    inertias.append(mbk.inertia_)

# Plot elbow curve and save
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WSS)')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)
plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')

# Train final clustering model with OPTIMAL_K = 5
print(f"Fitting final cluster model with k={OPTIMAL_K}...")
mbk_final = MiniBatchKMeans(
    n_clusters=OPTIMAL_K,
    random_state=42,
    batch_size=1024,
    n_init=10,
    max_iter=300,
    verbose=0
)

train_df['MusicType'] = mbk_final.fit_predict(X_train_scaled)
test_df['MusicType'] = mbk_final.predict(X_test_scaled)

print("Clustering complete. 'MusicType' feature added.")
print("\nCluster Distribution:")
print(train_df['MusicType'].value_counts().sort_index())


print("Preparing enhanced features for regression...")

# Create feature sets with MusicType
X_train_enhanced = train_df[FEATURES + ['MusicType']].copy()
X_test_enhanced = test_df[FEATURES + ['MusicType']].copy()

print(f"Enhanced feature matrix shape: {X_train_enhanced.shape}")

# Define scalable regression models
models = {
    'Linear Regression': LinearRegression(n_jobs=-1),

    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        verbose=0
    ),

    'LightGBM': LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
}

print(f"Defined {len(models)} regression models.")


print("Performing 5-fold cross-validation (RMSE)...")
results = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")

    # Cross-validation with negative MSE
    cv_scores = cross_val_score(
        model,
        X_train_enhanced,
        y_train,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    # Convert to RMSE
    rmse_scores = np.sqrt(-cv_scores)
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()

    results[name] = {
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'cv_scores': rmse_scores
    }

    print(f"Mean RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")

# Sort by performance
sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_rmse'])
best_model_name = sorted_results[0][0]

print("\nModel Rankings (by Mean RMSE):")
for rank, (name, metrics) in enumerate(sorted_results, 1):
    print(f"{rank}. {name:20s} RMSE: {metrics['mean_rmse']:.4f} ± {metrics['std_rmse']:.4f}")

# Plot comparison and save
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
mean_rmses = [results[m]['mean_rmse'] for m in model_names]
std_rmses = [results[m]['std_rmse'] for m in model_names]

plt.bar(model_names, mean_rmses, yerr=std_rmses, capsize=10, alpha=0.7)
plt.ylabel('RMSE (Beats Per Minute)')
plt.title('Model Performance Comparison\n5-Fold Cross-Validation')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\nModel comparison chart saved as 'model_comparison.png'")


print(f"Training best model: {best_model_name} and predicting test data...")

# Train best model on full data
best_model = models[best_model_name]
best_model.fit(X_train_enhanced, y_train)

# Make predictions on test set
test_predictions = best_model.predict(X_test_enhanced)
test_df['BeatsPerMinute'] = test_predictions # Use 'BeatsPerMinute' for submission consistency

print("Predictions generated.")

# Prepare submission file
submission = test_df[['id', 'BeatsPerMinute']].copy()

# Save all final outputs
train_df.to_csv('train_with_clusters.csv', index=False)
test_df.to_csv('test_with_predictions.csv', index=False)
submission.to_csv('submission.csv', index=False)

print("\nPipeline Complete. Outputs saved:")
print(" - submission.csv")
print(" - train_with_clusters.csv")
print(" - test_with_predictions.csv")
print(" - elbow_curve.png")
print(" - model_comparison.png")

print("\nSample Submission (First 10):")
print(submission.head(10).to_string(index=False))

