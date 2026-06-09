import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# -----------------------------
# 1. Detect target automatically
# -----------------------------
target_candidates = [c for c in train.columns if c not in test.columns and c != "id"]
if len(target_candidates) != 1:
    raise ValueError("Cannot detect target column. Found: " + str(target_candidates))
TARGET = target_candidates[0]
print("Detected target:", TARGET)
print(f"Target distribution:\n{train[TARGET].value_counts()}\n")

# -----------------------------
# 2. Advanced Feature Engineering
# -----------------------------
def engineer_features(df):
    """Create additional features to improve matching"""
    df_feat = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    categorical_cols = df_feat.select_dtypes(include=['object']).columns.tolist()
    
    # Statistical features for numeric columns
    if len(numeric_cols) > 0:
        df_feat['num_mean'] = df_feat[numeric_cols].mean(axis=1)
        df_feat['num_std'] = df_feat[numeric_cols].std(axis=1)
        df_feat['num_min'] = df_feat[numeric_cols].min(axis=1)
        df_feat['num_max'] = df_feat[numeric_cols].max(axis=1)
        df_feat['num_median'] = df_feat[numeric_cols].median(axis=1)
        df_feat['num_range'] = df_feat['num_max'] - df_feat['num_min']
        
        # Interaction features (top correlated pairs)
        for i, col1 in enumerate(numeric_cols[:5]):  # Limit to avoid explosion
            for col2 in numeric_cols[i+1:6]:
                df_feat[f'{col1}_x_{col2}'] = df_feat[col1] * df_feat[col2]
    
    return df_feat

# -----------------------------
# 3. Combine and encode
# -----------------------------
train_target = train[TARGET].copy()
train_feat = engineer_features(train.drop(columns=[TARGET]))
test_feat = engineer_features(test)

combined = pd.concat([train_feat, test_feat], axis=0)

# Encode categorical columns
categorical_cols = combined.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    combined[col], _ = pd.factorize(combined[col])

# Split back
train_enc = combined.iloc[:len(train)].reset_index(drop=True)
test_enc = combined.iloc[len(train):].reset_index(drop=True)

# -----------------------------
# 4. Feature columns
# -----------------------------
features = [c for c in train_enc.columns if c != "id"]
print(f"Using {len(features)} features for matching\n")

# -----------------------------
# 5. Multi-Model Ensemble Approach
# -----------------------------
predictions_list = []

# Model 1: KNN with Euclidean distance (scaled)
print("Model 1: KNN with StandardScaler + Euclidean...")
scaler1 = StandardScaler()
X_train_scaled1 = scaler1.fit_transform(train_enc[features].fillna(0))
X_test_scaled1 = scaler1.transform(test_enc[features].fillna(0))

nn1 = NearestNeighbors(n_neighbors=5, metric='euclidean', n_jobs=-1)
nn1.fit(X_train_scaled1)
distances1, indices1 = nn1.kneighbors(X_test_scaled1)

# Weighted voting based on distance
weights1 = 1 / (distances1 + 1e-6)
weights1 = weights1 / weights1.sum(axis=1, keepdims=True)
pred1 = []
for i in range(len(test_enc)):
    neighbors = train_target.iloc[indices1[i]].values
    weighted_votes = {}
    for neighbor, weight in zip(neighbors, weights1[i]):
        weighted_votes[neighbor] = weighted_votes.get(neighbor, 0) + weight
    pred1.append(max(weighted_votes, key=weighted_votes.get))
predictions_list.append(pred1)

# Model 2: KNN with Manhattan distance (robust scaling)
print("Model 2: KNN with RobustScaler + Manhattan...")
scaler2 = RobustScaler()
X_train_scaled2 = scaler2.fit_transform(train_enc[features].fillna(0))
X_test_scaled2 = scaler2.transform(test_enc[features].fillna(0))

nn2 = NearestNeighbors(n_neighbors=5, metric='manhattan', n_jobs=-1)
nn2.fit(X_train_scaled2)
distances2, indices2 = nn2.kneighbors(X_test_scaled2)

weights2 = 1 / (distances2 + 1e-6)
weights2 = weights2 / weights2.sum(axis=1, keepdims=True)
pred2 = []
for i in range(len(test_enc)):
    neighbors = train_target.iloc[indices2[i]].values
    weighted_votes = {}
    for neighbor, weight in zip(neighbors, weights2[i]):
        weighted_votes[neighbor] = weighted_votes.get(neighbor, 0) + weight
    pred2.append(max(weighted_votes, key=weighted_votes.get))
predictions_list.append(pred2)

# Model 3: KNN with Cosine similarity
print("Model 3: KNN with Cosine similarity...")
from sklearn.preprocessing import normalize
X_train_norm = normalize(X_train_scaled1, norm='l2')
X_test_norm = normalize(X_test_scaled1, norm='l2')

nn3 = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
nn3.fit(X_train_norm)
distances3, indices3 = nn3.kneighbors(X_test_norm)

weights3 = 1 / (distances3 + 1e-6)
weights3 = weights3 / weights3.sum(axis=1, keepdims=True)
pred3 = []
for i in range(len(test_enc)):
    neighbors = train_target.iloc[indices3[i]].values
    weighted_votes = {}
    for neighbor, weight in zip(neighbors, weights3[i]):
        weighted_votes[neighbor] = weighted_votes.get(neighbor, 0) + weight
    pred3.append(max(weighted_votes, key=weighted_votes.get))
predictions_list.append(pred3)

# Model 4: KNN with PCA (dimensionality reduction)
print("Model 4: KNN with PCA + Euclidean...")
n_components = min(50, len(features))
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled1)
X_test_pca = pca.transform(X_test_scaled1)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

nn4 = NearestNeighbors(n_neighbors=5, metric='euclidean', n_jobs=-1)
nn4.fit(X_train_pca)
distances4, indices4 = nn4.kneighbors(X_test_pca)

weights4 = 1 / (distances4 + 1e-6)
weights4 = weights4 / weights4.sum(axis=1, keepdims=True)
pred4 = []
for i in range(len(test_enc)):
    neighbors = train_target.iloc[indices4[i]].values
    weighted_votes = {}
    for neighbor, weight in zip(neighbors, weights4[i]):
        weighted_votes[neighbor] = weighted_votes.get(neighbor, 0) + weight
    pred4.append(max(weighted_votes, key=weighted_votes.get))
predictions_list.append(pred4)

# -----------------------------
# 6. Ensemble: Majority Voting
# -----------------------------
print("\nCombining predictions with majority voting...")
final_predictions = []
for i in range(len(test_enc)):
    votes = [pred[i] for pred in predictions_list]
    final_predictions.append(stats.mode(votes, keepdims=True)[0][0])

# -----------------------------
# 7. Save submission
# -----------------------------
submission = pd.DataFrame({
    'id': test['id'],
    TARGET: final_predictions
})
submission.to_csv("submission.csv", index=False)

print("\n" + "="*50)
print("✓ submission.csv generated successfully!")
print("="*50)
print(f"\nPrediction distribution:\n{submission[TARGET].value_counts()}")
print(f"\nFirst few predictions:\n{submission.head(10)}")




