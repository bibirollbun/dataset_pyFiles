import pandas as pd 
import numpy as np 
import os 
import time
import logging 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from category_encoders import TargetEncoder

from tqdm.auto import tqdm
import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


# Train + Test ko combine kar lo sirf check ke liye
combined = pd.concat([train, test], axis=0)

# 1. Unique special values dhoondo
special_values = set()

for col in combined.columns:
    if combined[col].dtype != "object":  # sirf numeric
        # Unique values nikal lo jo bohot chhoti ya bohot badi hain
        vals = combined[col].unique()
        for v in vals:
            if pd.notna(v) and (abs(v) < 1e-5 or abs(v) > 1e10):
                special_values.add(v)

print("âš¡ Special numeric values found:", special_values)



# âœ… In values ko replace karo 0.0 se
for df in [train, test]:
    df.replace(list(special_values), 0.0, inplace=True)  # exact match
    df.replace([np.inf, -np.inf], 0.0, inplace=True)     # inf handle
    df.fillna(0.0, inplace=True)                         # NaN handl


# âœ… Check karo ke replace hua ya nahi
for val in special_values:
    print(val, "-> train count:", (train == val).sum().sum(),
          "test count:", (test == val).sum().sum())
# Train + Test ko combine kar lo sirf check ke liye
combined = pd.concat([train, test], axis=0)

# 1. Unique special values dhoondo
special_values = set()

for col in combined.columns:
    if combined[col].dtype != "object":  # sirf numeric
        # Unique values nikal lo jo bohot chhoti ya bohot badi hain
        vals = combined[col].unique()
        for v in vals:
            if pd.notna(v) and (abs(v) < 1e-5 or abs(v) > 1e10):
                special_values.add(v)

print("âš¡ Special numeric values found:", special_values)



import matplotlib.pyplot as plt
import seaborn as sns

# Sirf numeric columns lo
corr_matrix = train.corr()

plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()
target_col = "BeatsPerMinute"
corr_with_target = train.corr()[target_col].sort_values(ascending=False)

print("Top correlations with target:\n", corr_with_target.head(11))





import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from itertools import combinations

def create_extensive_features(df):
    """
    Create maximum features for linear model behavior
    """
    df_new = df.copy()
    
    # ===== 1. BASIC TRANSFORMATIONS =====
    numeric_cols = ['MoodScore', 'TrackDurationMs', 'RhythmScore', 'VocalContent', 
                   'LivePerformanceLikelihood','Energy', 'InstrumentalScore', 'AcousticQuality', 'AudioLoudness']
    
    for col in numeric_cols:
        # Square features
        df_new[f'{col}_squared'] = df_new[col] ** 2
        
        # Cube features  
        df_new[f'{col}_cubed'] = df_new[col] ** 3
        
        # Square root features
        df_new[f'{col}_sqrt'] = np.sqrt(np.abs(df_new[col]))
        
        # Log features (add small constant to avoid log(0))
        df_new[f'{col}_log'] = np.log(np.abs(df_new[col]) + 1e-6)
        
        # Reciprocal features
        df_new[f'{col}_reciprocal'] = 1 / (df_new[col] + 1e-6)
        
        # Exponential features (scaled down)
        df_new[f'{col}_exp'] = np.exp(df_new[col] / 10)
        
        # Sine and Cosine features
        df_new[f'{col}_sin'] = np.sin(df_new[col] * np.pi)
        df_new[f'{col}_cos'] = np.cos(df_new[col] * np.pi)
        
        # Binning features
        df_new[f'{col}_bin_low'] = (df_new[col] < df_new[col].quantile(0.33)).astype(int)
        df_new[f'{col}_bin_mid'] = ((df_new[col] >= df_new[col].quantile(0.33)) & 
                                   (df_new[col] < df_new[col].quantile(0.67))).astype(int)
        df_new[f'{col}_bin_high'] = (df_new[col] >= df_new[col].quantile(0.67)).astype(int)
    
    # ===== 2. INTERACTION FEATURES (ALL PAIRS) =====
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols[i+1:], i+1):
            # Multiplication interactions
            df_new[f'{col1}_x_{col2}'] = df_new[col1] * df_new[col2]
            
            # Division interactions
            df_new[f'{col1}_div_{col2}'] = df_new[col1] / (df_new[col2] + 1e-6)
            df_new[f'{col2}_div_{col1}'] = df_new[col2] / (df_new[col1] + 1e-6)
            
            # Addition interactions
            df_new[f'{col1}_plus_{col2}'] = df_new[col1] + df_new[col2]
            
            # Subtraction interactions
            df_new[f'{col1}_minus_{col2}'] = df_new[col1] - df_new[col2]
            df_new[f'{col2}_minus_{col1}'] = df_new[col2] - df_new[col1]
            
            # Power interactions
            df_new[f'{col1}_power_{col2}'] = np.power(np.abs(df_new[col1]) + 1e-6, 
                                                     np.abs(df_new[col2]) + 1e-6)

    # ===== 3. TRIPLE INTERACTIONS (SELECTED) =====
    important_combos = [
        ('MoodScore', 'RhythmScore', 'AudioLoudness'),
        ('VocalContent', 'InstrumentalScore', 'AcousticQuality'),
        ('TrackDurationMs', 'MoodScore', 'RhythmScore'),
        ('LivePerformanceLikelihood', 'AcousticQuality', 'VocalContent')
    ]
    
    for col1, col2, col3 in important_combos:
        df_new[f'{col1}_x_{col2}_x_{col3}'] = df_new[col1] * df_new[col2] * df_new[col3]
        df_new[f'{col1}_plus_{col2}_plus_{col3}'] = df_new[col1] + df_new[col2] + df_new[col3]

    # ===== 4. STATISTICAL FEATURES =====
    # Row-wise statistics
    score_cols = [col for col in numeric_cols if 'Score' in col]
    df_new['ScoreSum'] = df_new[score_cols].sum(axis=1)
    df_new['ScoreMean'] = df_new[score_cols].mean(axis=1)
    df_new['ScoreStd'] = df_new[score_cols].std(axis=1)
    df_new['ScoreMax'] = df_new[score_cols].max(axis=1)
    df_new['ScoreMin'] = df_new[score_cols].min(axis=1)
    df_new['ScoreRange'] = df_new['ScoreMax'] - df_new['ScoreMin']
    
    # All numeric columns statistics
    df_new['AllSum'] = df_new[numeric_cols].sum(axis=1)
    df_new['AllMean'] = df_new[numeric_cols].mean(axis=1)
    df_new['AllStd'] = df_new[numeric_cols].std(axis=1)
    df_new['AllMax'] = df_new[numeric_cols].max(axis=1)
    df_new['AllMin'] = df_new[numeric_cols].min(axis=1)
    df_new['AllRange'] = df_new['AllMax'] - df_new['AllMin']

    # ===== 5. DOMAIN-SPECIFIC FEATURES =====
    # Music-specific combinations
    df_new['EnergyLevel'] = df_new['MoodScore'] * df_new['RhythmScore'] * df_new['AudioLoudness']
    df_new['VocalInstrumental'] = df_new['VocalContent'] / (df_new['InstrumentalScore'] + 1e-6)
    df_new['AcousticElectronic'] = df_new['AcousticQuality'] - df_new['InstrumentalScore']
    df_new['LiveStudio'] = df_new['LivePerformanceLikelihood'] - df_new['AcousticQuality']
    
    # Duration features
    df_new['DurationMinutes'] = df_new['TrackDurationMs'] / 60000
    df_new['DurationSeconds'] = df_new['TrackDurationMs'] / 1000
    df_new['IsShortTrack'] = (df_new['TrackDurationMs'] < 180000).astype(int)
    df_new['IsLongTrack'] = (df_new['TrackDurationMs'] > 300000).astype(int)
    df_new['IsMediumTrack'] = ((df_new['TrackDurationMs'] >= 180000) & 
                               (df_new['TrackDurationMs'] <= 300000)).astype(int)

    # ===== 6. PERCENTILE FEATURES =====
    for col in numeric_cols:
        df_new[f'{col}_percentile'] = df_new[col].rank(pct=True)
        df_new[f'{col}_zscore'] = (df_new[col] - df_new[col].mean()) / (df_new[col].std() + 1e-6)

    # ===== 7. CLUSTERING-BASED FEATURES =====
    # High/Low categorizations
    for col in numeric_cols:
        median_val = df_new[col].median()
        df_new[f'{col}_above_median'] = (df_new[col] > median_val).astype(int)
        df_new[f'{col}_below_median'] = (df_new[col] < median_val).astype(int)
        
        # Quartile features
        q1 = df_new[col].quantile(0.25)
        q3 = df_new[col].quantile(0.75)
        df_new[f'{col}_Q1'] = (df_new[col] <= q1).astype(int)
        df_new[f'{col}_Q2'] = ((df_new[col] > q1) & (df_new[col] <= median_val)).astype(int)
        df_new[f'{col}_Q3'] = ((df_new[col] > median_val) & (df_new[col] <= q3)).astype(int)
        df_new[f'{col}_Q4'] = (df_new[col] > q3).astype(int)

    print(f"Original features: {len(df.columns)}")
    print(f"Total features after engineering: {len(df_new.columns)}")
    print(f"New features created: {len(df_new.columns) - len(df.columns)}")
    
    return df_new

def add_polynomial_features(df, degree=2):
    """
    Add polynomial features using PolynomialFeatures
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Select important columns for polynomial features (to avoid memory issues)
    important_cols = ['MoodScore', 'RhythmScore', 'AudioLoudness', 'VocalContent', 'InstrumentalScore']
    important_cols = [col for col in important_cols if col in numeric_cols]
    
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(df[important_cols])
    
    # Get feature names
    feature_names = poly.get_feature_names_out(important_cols)
    
    # Create DataFrame with polynomial features
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Remove original columns (they're already in df)
    original_cols = [col for col in feature_names if col in important_cols]
    poly_df = poly_df.drop(columns=original_cols)
    
    # Concatenate with original dataframe
    df_combined = pd.concat([df, poly_df], axis=1)
    
    print(f"Polynomial features added: {len(poly_df.columns)}")
    return df_combined

# Usage example:

# Apply to train and test datasets
train_featured = create_extensive_features(train)
test_featured = create_extensive_features(test)

# Optional: Add polynomial features (be careful with memory)
train_featured = add_polynomial_features(train_featured, degree=2)
test_featured = add_polynomial_features(test_featured, degree=2)

# Check final feature count
print(f"Final feature count: {len(train_featured.columns)}")



display(train_featured.shape)
display(test_featured.shape)



import pandas as pd
import numpy as np

def select_best_features_per_original(df, target_col, original_cols, top_k_positive=2):
    """
    Feature selection rules:
    - 'minus' wale features skip karo
    - Agar group ke sabhi correlations NEGATIVE hain -> sirf ek (abs correlation highest) lo
    - Agar positive correlation bhi hai -> max 2 hi features lo (base + top positive ya sirf top positives)
    """
    corr_full = df.corr()[target_col].drop(target_col)
    
    selected_features = []
    
    for orig in original_cols:
        group_features = []
        
        # Related engineered features (skip 'minus')
        related = [
            col for col in corr_full.index 
            if col.startswith(orig) and col != orig and "minus" not in col
        ]
        
        if not related and orig not in corr_full.index:
            continue
        
        corr_related = corr_full[related]
        
        # Case 1: All negative correlations
        if not corr_related.empty and all(corr_related < 0):
            best_one = corr_related.abs().nlargest(1).index.tolist()
            group_features.extend(best_one)
        
        else:
            # Case 2: Positive correlations exist
            positives = corr_related[corr_related > 0].sort_values(ascending=False)
            
            if orig in corr_full.index and corr_full[orig] > 0:
                # Base feature positive hai â†’ base + top engineered positive (max 2 total)
                top_one = positives.head(1).index.tolist() if not positives.empty else []
                group_features = [orig] + top_one
            elif not positives.empty:
                # Base feature absent ya negative hai â†’ top 2 positives lo
                group_features = positives.head(top_k_positive).index.tolist()
        
        # Only keep max 2 per group
        group_features = group_features[:2]
        
        selected_features.extend(group_features)
    
    selected_features = list(dict.fromkeys(selected_features))
    
    print(f"âœ… Total selected features: {len(selected_features)}")
    return selected_features


# ====== USE FUNCTION ======
original_cols = [
    'MoodScore', 'TrackDurationMs', 'RhythmScore', 
    'VocalContent', 'LivePerformanceLikelihood', 
    'InstrumentalScore','Energy', 'AcousticQuality', 'AudioLoudness'
]

# Step 1: Best features select karo
best_features = select_best_features_per_original(
    train_featured, target_col='BeatsPerMinute', 
    original_cols=original_cols, top_k_positive=2
)

# Step 2: Filter only valid features
valid_best_features = [col for col in best_features if col in train_featured.columns]

print(f"Total selected features before filtering: {len(best_features)}")
print(f"Valid features available in train_filtered: {len(valid_best_features)}")

# Step 3: Train/Test subsets banao
train_best = train_featured[valid_best_features + ['BeatsPerMinute']]
# test_best = test_featured[valid_best_features]



# Remove highly correlated features (multicollinearity)
from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns

# Correlation matrix banao
corr_matrix = train_best.corr()

# High correlation wale features remove karo (>0.90)
def remove_highly_correlated(df, threshold=0.90):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(columns=to_drop)

train_filtered = remove_highly_correlated(train_best)


display(test_featured.shape)
display(train_filtered.shape)


# Train ke columns list banao (target column hata ke)
cols_to_keep = [c for c in train_filtered.columns if c != "BeatsPerMinute"]

# Test se sirf wahi columns lo
test_filtered = test_featured[cols_to_keep]

print("Train shape:", train_filtered.shape)
print("Test shape:", test_filtered.shape)



target_col = "BeatsPerMinute"
corr_with_target = train_filtered.corr()[target_col].sort_values(ascending=False)

print("Top correlations with target:\n", corr_with_target.head(55))


# Train aur Test ke columns sets banao
train_cols = set(train_filtered.columns)
test_cols = set(test_filtered.columns)

# Common aur different columns nikaalo
common_cols = train_cols.intersection(test_cols)
train_only = train_cols - test_cols
test_only = test_cols - train_cols

print(f"âœ… Total common features: {len(common_cols)}")
# print(f"ğŸ“Š Common features list: {list(common_cols)[:10]} ...")  # sirf first 10 dikhaye

print(f"ğŸ”¹ Features only in train ({len(train_only)}): {list(train_only)}")
print(f"ğŸ”¹ Features only in test  ({len(test_only)}): {list(test_only)}")



# 4. NaN check
print("Step 4 ğŸ•µ NaN in train:", np.isnan(train_filtered).sum().sum())
print("Step 4 ğŸ•µ NaN in test:", np.isnan(test_filtered).sum().sum())


from sklearn.preprocessing import StandardScaler, RobustScaler

# Standard scaling (Linear models ke liye zaruri)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_filtered.drop('BeatsPerMinute', axis=1))
X_test_scaled = scaler.transform(test_filtered)

# Convert back to DataFrame
feature_names = train_filtered.drop('BeatsPerMinute', axis=1).columns
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)


# Train aur Test ke columns sets banao
train_cols = set(X_train_scaled.columns)
test_cols = set(X_train_scaled.columns)

# Common aur different columns nikaalo
common_cols = train_cols.intersection(test_cols)
train_only = train_cols - test_cols
test_only = test_cols - train_cols

print(f"âœ… Total common features: {len(common_cols)}")
# print(f"ğŸ“Š Common features list: {list(common_cols)[:10]} ...")  # sirf first 10 dikhaye

print(f"ğŸ”¹ Features only in train ({len(train_only)}): {list(train_only)}")
print(f"ğŸ”¹ Features only in test  ({len(test_only)}): {list(test_only)}")



import matplotlib.pyplot as plt
import seaborn as sns

# Sirf numeric columns lo
corr_matrix = train_filtered.corr()

plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()





X = X_train_scaled
y = train_filtered['BeatsPerMinute']
X_test = X_test_scaled


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import time

FOLDS = 5
FEATURES = X.columns.tolist()

# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof = np.zeros(len(train))
pred = np.zeros(len(test))

# Start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = X_test.copy()

    start = time.time()

    # Train model with GPU
    model = XGBRegressor(
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    max_depth=2,                # balanced depth
    learning_rate=0.01,         # slightly faster but safe
    n_estimators=3000,          # enough boosting rounds
    gamma=1,                    # mild regularization
    subsample=0.8,              # more randomness
    colsample_bytree=0.8,       # more randomness
    reg_alpha=0.5,              # L1 regularization (sparse features)
    reg_lambda=1.5,             # L2 regularization
    early_stopping_rounds=100,
    eval_metric="rmse",
    enable_categorical=True,
    random_state=42
)

    

    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    # Predict OOF and test
    oof[valid_idx] = model.predict(x_valid)
    pred += model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Feature engineering & training time: {time.time() - start:.1f} sec")

# Average test predictions
pred /= FOLDS

# Final RMSE
full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nFinal CV RMSE: {full_rmse:.4f}")
pred


y_pred_after = np.clip(pred, 46.718, 206.037)
print('predict mean after clip:',y_pred_after.mean())
print('predict median after clip:',np.median(y_pred_after))

submission["BeatsPerMinute"] = y_pred_after
submission.to_csv("submission_Xgb.csv", index=False)
submission.head()


catsub=pd.read_csv('/kaggle/input/musicsubmission2-0/CatSubmission.csv')
catsub
BeatsPerMinute_global_avg = train['BeatsPerMinute'].mean()
print(BeatsPerMinute_global_avg)
pred1 = (pred + BeatsPerMinute_global_avg)/2
pred2 = 0.7 * catsub["BeatsPerMinute"] + 0.3 * pred1
display(pred2)
print('predict mean :',pred2.mean())
print('predict median :',np.median(pred2))

y_pred_after = np.clip(pred2, 46.718, 206.037)
print('predict mean after clip:',y_pred_after.mean())
print('predict median after clip:',np.median(y_pred_after))
submission["BeatsPerMinute"] = y_pred_after
submission.to_csv("submission.csv", index=False)
submission.head()

