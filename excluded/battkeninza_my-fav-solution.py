


# Enhanced Personality Prediction with Neural Network, GBDT, Hill Climbing and Weighted Average Ensemble

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Neural Network imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.decomposition import PCA



# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
datasert_df = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# Data preprocessing
datasert_df = (
    datasert_df
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']

train_df = train_df.merge(datasert_df, how='left', on=merge_cols)
test_df = test_df.merge(datasert_df, how='left', on=merge_cols)

# Store IDs and prepare data
train_ID = train_df['id']
test_ID = test_df['id']

train_df.drop("id", axis=1, inplace=True)
test_df.drop("id", axis=1, inplace=True)

ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values

# Combine datasets for preprocessing
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)


# Missing value imputation functions
def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    """Fill missing values in target_col by grouping based on quantiles of group_source_col"""
    if labels is None:
        labels = [f'Q{i+1}' for i in range(len(quantiles)-1)]
    
    temp_bin_col = f'{group_source_col}_bin'
    df[temp_bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)
    df[target_col] = df[target_col].fillna(df.groupby(temp_bin_col)[target_col].transform('median'))
    df.drop(columns=[temp_bin_col], inplace=True)
    return df

# Missing value imputation
# 1. Time_spent_Alone
all_data['social_attend_bin'] = pd.qcut(all_data['Social_event_attendance'], q=[0, 0.25, 0.5, 0.75, 1.0], labels=['Q1', 'Q2', 'Q3', 'Q4'])
all_data['Time_spent_Alone'] = all_data['Time_spent_Alone'].fillna(all_data.groupby('social_attend_bin')['Time_spent_Alone'].transform('median'))
all_data.drop(columns=['social_attend_bin'], inplace=True)

# Continue with Going_outside for remaining missing values
all_data['Going_outside_bin'] = pd.qcut(all_data['Going_outside'], q=[0, 0.25, 0.5, 0.75, 1.0], labels=['Q1', 'Q2', 'Q3', 'Q4'])
all_data['Time_spent_Alone'] = all_data['Time_spent_Alone'].fillna(all_data.groupby('Going_outside_bin')['Time_spent_Alone'].transform('median'))
all_data.drop(columns=['Going_outside_bin'], inplace=True)

# 2. Social_event_attendance
all_data = fill_missing_by_quantile_group(all_data, 'Going_outside', 'Social_event_attendance')
all_data = fill_missing_by_quantile_group(all_data, 'Friends_circle_size', 'Social_event_attendance')
all_data = fill_missing_by_quantile_group(all_data, 'Post_frequency', 'Social_event_attendance')

# 3. Other features
all_data = fill_missing_by_quantile_group(all_data, 'Social_event_attendance', 'Going_outside')
all_data = fill_missing_by_quantile_group(all_data, 'Post_frequency', 'Friends_circle_size')
all_data = fill_missing_by_quantile_group(all_data, 'Going_outside', 'Friends_circle_size')
all_data = fill_missing_by_quantile_group(all_data, 'Friends_circle_size', 'Post_frequency')

# 4. Categorical features
all_data.fillna({
    'Stage_fear': 'UnKnow',
    'Drained_after_socializing': 'UnKnow'
}, inplace=True)

# =============================================================================
# COMPREHENSIVE FEATURE ENGINEERING
# =============================================================================

print("Starting Feature Engineering...")

# Store original numerical columns before engineering
original_numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                          'Friends_circle_size', 'Post_frequency']

# 1. RATIO AND INTERACTION FEATURES
print("Creating ratio and interaction features...")

# Social engagement ratios
all_data['social_to_alone_ratio'] = all_data['Social_event_attendance'] / (all_data['Time_spent_Alone'] + 1e-6)
all_data['outside_to_alone_ratio'] = all_data['Going_outside'] / (all_data['Time_spent_Alone'] + 1e-6)
all_data['friends_to_alone_ratio'] = all_data['Friends_circle_size'] / (all_data['Time_spent_Alone'] + 1e-6)
all_data['post_to_friends_ratio'] = all_data['Post_frequency'] / (all_data['Friends_circle_size'] + 1e-6)

# Social activity combinations
all_data['social_activity_score'] = (all_data['Social_event_attendance'] + all_data['Going_outside'] + 
                                    all_data['Friends_circle_size'] + all_data['Post_frequency']) / 4
all_data['social_minus_alone'] = all_data['social_activity_score'] - all_data['Time_spent_Alone']

# Interaction features
all_data['social_going_interaction'] = all_data['Social_event_attendance'] * all_data['Going_outside']
all_data['friends_post_interaction'] = all_data['Friends_circle_size'] * all_data['Post_frequency']
all_data['alone_social_interaction'] = all_data['Time_spent_Alone'] * all_data['Social_event_attendance']

# 2. POLYNOMIAL FEATURES (degree 2 for key features)
print("Creating polynomial features...")
poly_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside']
for feat in poly_features:
    all_data[f'{feat}_squared'] = all_data[feat] ** 2
    all_data[f'{feat}_sqrt'] = np.sqrt(all_data[feat] + 1e-6)

# 3. BINNING AND DISCRETIZATION
print("Creating binned features...")

# Quantile-based binning
for col in original_numerical_cols:
    all_data[f'{col}_bin'] = pd.qcut(all_data[col], q=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    all_data[f'{col}_bin_encoded'] = pd.qcut(all_data[col], q=5, labels=[1, 2, 3, 4, 5])

# Custom binning based on domain knowledge
all_data['time_alone_category'] = pd.cut(all_data['Time_spent_Alone'], 
                                        bins=[0, 3, 6, 8, 10], 
                                        labels=['Low', 'Medium', 'High', 'Very_High'])

all_data['social_frequency_category'] = pd.cut(all_data['Social_event_attendance'], 
                                              bins=[0, 2, 5, 8, 10], 
                                              labels=['Rare', 'Occasional', 'Regular', 'Frequent'])

# 4. STATISTICAL FEATURES
print("Creating statistical aggregation features...")

# Rolling statistics (treating as time series-like)
numerical_cols = [col for col in all_data.columns if all_data[col].dtype in ['int64', 'float64']]
for col in original_numerical_cols:
    all_data[f'{col}_rank'] = all_data[col].rank(pct=True)
    all_data[f'{col}_zscore'] = (all_data[col] - all_data[col].mean()) / all_data[col].std()

# 5. CLUSTERING-BASED FEATURES
print("Creating clustering-based features...")
from sklearn.cluster import KMeans

# Cluster social behavior patterns
social_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']
kmeans = KMeans(n_clusters=5, random_state=42)
all_data['social_cluster'] = kmeans.fit_predict(all_data[social_features])

# Distance to cluster centers
cluster_centers = kmeans.cluster_centers_
for i in range(5):
    distances = np.sqrt(np.sum((all_data[social_features].values - cluster_centers[i])**2, axis=1))
    all_data[f'distance_to_cluster_{i}'] = distances

# 6. PRINCIPAL COMPONENT ANALYSIS
print("Creating PCA features...")
pca = PCA(n_components=3)
pca_features = pca.fit_transform(all_data[social_features])
for i in range(3):
    all_data[f'pca_component_{i}'] = pca_features[:, i]

# 7. AGGREGATION FEATURES
print("Creating aggregation features...")

# Min/Max/Mean across features
all_data['social_features_mean'] = all_data[social_features].mean(axis=1)
all_data['social_features_std'] = all_data[social_features].std(axis=1)
all_data['social_features_min'] = all_data[social_features].min(axis=1)
all_data['social_features_max'] = all_data[social_features].max(axis=1)
all_data['social_features_range'] = all_data['social_features_max'] - all_data['social_features_min']

# Skewness and kurtosis of individual's behavior pattern
from scipy import stats
all_data['social_features_skew'] = all_data[social_features].apply(lambda x: stats.skew(x), axis=1)
all_data['social_features_kurtosis'] = all_data[social_features].apply(lambda x: stats.kurtosis(x), axis=1)

# 8. BOOLEAN/BINARY FEATURES
print("Creating boolean features...")

# Threshold-based binary features
all_data['is_highly_social'] = (all_data['Social_event_attendance'] > 7).astype(int)
all_data['is_highly_alone'] = (all_data['Time_spent_Alone'] > 7).astype(int)
all_data['is_large_friend_circle'] = (all_data['Friends_circle_size'] > 5).astype(int)
all_data['is_frequent_poster'] = (all_data['Post_frequency'] > 5).astype(int)
all_data['is_outgoing'] = (all_data['Going_outside'] > 7).astype(int)

# Combined boolean features
all_data['is_social_extrovert'] = ((all_data['Social_event_attendance'] > 6) & 
                                  (all_data['Going_outside'] > 6) & 
                                  (all_data['Time_spent_Alone'] < 5)).astype(int)

all_data['is_social_introvert'] = ((all_data['Social_event_attendance'] < 4) & 
                                  (all_data['Going_outside'] < 4) & 
                                  (all_data['Time_spent_Alone'] > 6)).astype(int)

# 9. LOGARITHMIC AND EXPONENTIAL TRANSFORMATIONS
print("Creating log/exp transformations...")

for col in original_numerical_cols:
    all_data[f'{col}_log'] = np.log1p(all_data[col])
    all_data[f'{col}_exp'] = np.exp(all_data[col] / 10)  # Scaled to prevent overflow

# 10. PERCENTILE RANKS
print("Creating percentile rank features...")

for col in original_numerical_cols:
    all_data[f'{col}_percentile'] = all_data[col].rank(pct=True) * 100

# One-hot encoding for categorical features
print("Performing one-hot encoding...")
categorical_cols = ['Stage_fear', 'Drained_after_socializing', 'match_p'] + \
                  [col for col in all_data.columns if col.endswith('_bin') or col.endswith('_category')]

all_data = pd.get_dummies(all_data, columns=categorical_cols, 
                         prefix=[col.replace('_bin', '').replace('_category', '') for col in categorical_cols])

print(f"Feature engineering complete! Total features: {all_data.shape[1]}")

# Split data back
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

# 11. FEATURE SELECTION
print("Performing feature selection...")

# Remove features with zero variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(X_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()].tolist()
X_train = pd.DataFrame(X_train_selected, columns=selected_features)
X_test = pd.DataFrame(X_test_selected, columns=selected_features)

# Correlation-based feature selection
print("Removing highly correlated features...")
corr_matrix = X_train.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
X_train = X_train.drop(columns=high_corr_features)
X_test = X_test.drop(columns=high_corr_features)

print(f"Final feature count after selection: {X_train.shape[1]}")

# Calculate class weights for imbalanced data
class_0 = y_train.sum()
class_1 = len(y_train) - class_0
scale_pos_weight = class_1 / class_0

# Feature scaling for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Neural Network Model Definition
def create_neural_network(input_dim, dropout_rate=0.3, l1_reg=0.01, l2_reg=0.01):
    """Create a neural network model with regularization"""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,), 
              kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='sigmoid', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        Dropout(dropout_rate),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

# Custom Neural Network Classifier for sklearn compatibility
class NeuralNetworkClassifier:
    def __init__(self, input_dim, epochs=1000, batch_size=32, dropout_rate=0.3, l1_reg=0.01, l2_reg=0.01):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model
        self.model = create_neural_network(
            self.input_dim, self.dropout_rate, self.l1_reg, self.l2_reg
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001
        )
        
        # Calculate class weights
        class_weight = {0: scale_pos_weight, 1: 1.0}
        
        # Train model
        self.model.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weight,
            verbose=0
        )
        
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        pred_proba = self.model.predict(X_scaled, verbose=0)
        # Return probabilities for both classes
        return np.column_stack([1 - pred_proba.ravel(), pred_proba.ravel()])
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)



cb_params = {
    "border_count": 39,
    "colsample_bylevel": 0.19459088572914465,
    "depth": 2,
    "iterations": 1467,
    "l2_leaf_reg": 31.236169478676036,
    "learning_rate": 0.06852669420904771,
    "min_child_samples": 160,
    "random_state": 42,
    "random_strength": 0.8517786189616939,
    "scale_pos_weight": 1.1691394390533685,
    "subsample": 0.3192330024411618,
    "verbose": False,
}

xgb_params = {
    "colsample_bylevel": 0.8168489864941239,
    "colsample_bynode": 0.8850485490950061,
    "colsample_bytree": 0.8379339940113913,
    "gamma": 2.3977359439809276,
    "learning_rate": 0.0616974880921061,
    "max_depth": 344,
    "max_leaves": 89,
    "min_child_weight": 10,
    "n_estimators": 696,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 1.849084818346014,
    "reg_lambda": 29.680324563362227,
    "subsample": 0.5902901569391961,
    "verbosity": 0,
    "enable_categorical": True
}

hgb_params = {
    "l2_regularization": 28.13576008319012,
    "learning_rate": 0.1543598086529694,
    "max_depth": 325,
    "max_features": 0.323620656779567,
    "max_iter": 2490,
    "max_leaf_nodes": 216,
    "min_samples_leaf": 12,
    "random_state": 42,
    "categorical_features": "from_dtype"
}

lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.6467443250209886,
    "learning_rate": 0.06547186748153115,
    "min_child_samples": 34,
    "min_child_weight": 0.24399244943904663,
    "n_estimators": 498,
    "n_jobs": -1,
    "num_leaves": 158,
    "random_state": 42,
    "reg_alpha": 6.568921253574134,
    "reg_lambda": 62.66165355751099,
    "subsample": 0.0011019938618584968,
    "verbose": -1
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.8384834064170148,
    "learning_rate": 0.07006829797238343,
    "min_child_samples": 46,
    "min_child_weight": 0.7625394962666617,
    "n_estimators": 1887,
    "n_jobs": -1,
    "num_leaves": 341,
    "random_state": 42,
    "reg_alpha": 10.53082019937197,
    "reg_lambda": 67.44600065144685,
    "subsample": 0.4925008305336127,
    "verbose": -1
}

lgbm_dart_params = {
    "boosting_type": "dart",
    "colsample_bytree": 0.7592971191793424,
    "learning_rate": 0.046141766106846074,
    "min_child_samples": 18,
    "min_child_weight": 0.4740109054323218,
    "n_estimators": 4035,
    "n_jobs": -1,
    "num_leaves": 393,
    "random_state": 42,
    "reg_alpha": 48.016799341666605,
    "reg_lambda": 89.12860300833658,
    "subsample": 0.016333358901112538,
    "verbose": -1
}


# Initialize models (including Neural Network and additional GBDT models)
models = {
    'logistic': LogisticRegression(max_iter=1000, C=1.0, solver='liblinear'),
    'svm': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True),
    'rf': RandomForestClassifier(n_estimators=1000, max_depth=5, min_samples_leaf=16, random_state=42),
    'gb': GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'dt': DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=42),
    'xgb': XGBClassifier(**xgb_params),
    'cat': CatBoostClassifier(**cb_params),
    'lgbm_goss': LGBMClassifier(**lgbm_goss_params),
    'lgbm_gdbt': LGBMClassifier(**lgbm_params),
    'lgbm_dart': LGBMClassifier(**lgbm_dart_params),
    # Additional GBDT models with different configurations
    'gb_deep': GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42),
    'gb_wide': GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, max_depth=3, random_state=42),
    # Neural Network
    'nn': NeuralNetworkClassifier(input_dim=X_train.shape[1], epochs=150, batch_size=64)
}


# Split for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# For neural network, we need scaled data
X_train_split_scaled, X_val_scaled, y_train_split_scaled, y_val_scaled = train_test_split(
    X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Train individual models and collect predictions
val_preds_dict = {}
test_preds_dict = {}

print("Training individual models...")
for name, model in models.items():
    print(f"Training {name}...")
    
    if name == 'nn':
        # Use scaled data for neural network
        model.fit(X_train_split_scaled, y_train_split_scaled)
        val_preds_dict[name] = model.predict_proba(X_val_scaled)[:, 1]
        test_preds_dict[name] = model.predict_proba(X_test_scaled)[:, 1]
    else:
        # Use original data for other models
        model.fit(X_train_split, y_train_split)
        val_preds_dict[name] = model.predict_proba(X_val)[:, 1]
        test_preds_dict[name] = model.predict_proba(X_test)[:, 1]
    
    # Print individual model performance
    val_auc = roc_auc_score(y_val, val_preds_dict[name])
    print(f"{name} validation AUC: {val_auc:.5f}")

print("\n" + "="*50)


# Hill Climbing Ensemble
def hill_climb_ensemble(preds_dict, y_val, metric_fn=roc_auc_score, max_iters=None):
    """Hill climbing ensemble builder"""
    model_names = list(preds_dict.keys())
    selected_models = []
    best_score = 0
    final_preds = None

    for iteration in range(max_iters or len(model_names)):
        improved = False
        best_candidate = None
        best_candidate_preds = None

        for model in model_names:
            if model in selected_models:
                continue
            
            candidate_models = selected_models + [model]
            preds_stack = np.mean([preds_dict[m] for m in candidate_models], axis=0)
            score = metric_fn(y_val, preds_stack)

            if score > best_score:
                best_score = score
                best_candidate = model
                best_candidate_preds = preds_stack
                improved = True

        if improved:
            selected_models.append(best_candidate)
            final_preds = best_candidate_preds
            print(f"Hill Climb Iteration {iteration+1}: Added {best_candidate}, Score = {best_score:.5f}")
        else:
            print("Hill Climb: No improvement. Stopping.")
            break

    return selected_models, best_score, final_preds

# Weighted Average Ensemble
def weighted_average_ensemble(preds_dict, y_val, metric_fn=roc_auc_score):
    """Create weighted average ensemble based on individual model performance"""
    model_names = list(preds_dict.keys())
    weights = {}
    
    # Calculate weights based on individual performance
    for name in model_names:
        score = metric_fn(y_val, preds_dict[name])
        weights[name] = score
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {name: w/total_weight for name, w in weights.items()}
    
    # Create weighted average
    weighted_preds = np.zeros_like(preds_dict[model_names[0]])
    for name, weight in weights.items():
        weighted_preds += weight * preds_dict[name]
    
    weighted_score = metric_fn(y_val, weighted_preds)
    
    return weights, weighted_score, weighted_preds



# Apply Hill Climbing
print("Applying Hill Climbing Ensemble...")
selected_models, hill_climb_auc, hill_climb_preds = hill_climb_ensemble(val_preds_dict, y_val)
print(f"Hill Climb Final models: {selected_models}")
print(f"Hill Climb Final AUC: {hill_climb_auc:.5f}")

print("\n" + "="*50)

# Apply Weighted Average
print("Applying Weighted Average Ensemble...")
weights, weighted_auc, weighted_preds = weighted_average_ensemble(val_preds_dict, y_val)
print("Weighted Average model weights:")
for name, weight in weights.items():
    print(f"  {name}: {weight:.4f}")
print(f"Weighted Average AUC: {weighted_auc:.5f}")

print("\n" + "="*50)

# Generate test predictions for both methods
print("Generating test predictions...")

# Hill Climb test predictions
hill_climb_test_preds = np.mean([test_preds_dict[m] for m in selected_models], axis=0)
hill_climb_test_labels = (hill_climb_test_preds >= 0.5).astype(int)

# Weighted Average test predictions
weighted_test_preds = np.zeros_like(test_preds_dict[list(test_preds_dict.keys())[0]])
for name, weight in weights.items():
    weighted_test_preds += weight * test_preds_dict[name]
weighted_test_labels = (weighted_test_preds >= 0.5).astype(int)


# Create submissions
# Hill Climb submission
hill_climb_submission = pd.DataFrame({
    'id': test_ID,
    'Personality': hill_climb_test_labels
})
hill_climb_submission['Personality'] = hill_climb_submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
hill_climb_submission.to_csv('hill_climb_submission.csv', index=False)

# Weighted Average submission
weighted_submission = pd.DataFrame({
    'id': test_ID,
    'Personality': weighted_test_labels
})
weighted_submission['Personality'] = weighted_submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
weighted_submission.to_csv('weighted_average_submission.csv', index=False)

print("Submissions created successfully!")
print(f"Hill Climb submission saved as 'hill_climb_submission.csv'")
print(f"Weighted Average submission saved as 'weighted_average_submission.csv'")


# Comparison of methods
print("\n" + "="*50)
print("ENSEMBLE COMPARISON:")
print(f"Hill Climbing AUC: {hill_climb_auc:.5f}")
print(f"Weighted Average AUC: {weighted_auc:.5f}")

# Validation confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Hill Climb confusion matrix
hill_climb_val_labels = (hill_climb_preds >= 0.5).astype(int)
cm1 = confusion_matrix(y_val, hill_climb_val_labels)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f"Hill Climb Confusion Matrix\n(AUC: {hill_climb_auc:.5f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# Weighted Average confusion matrix
weighted_val_labels = (weighted_preds >= 0.5).astype(int)
cm2 = confusion_matrix(y_val, weighted_val_labels)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
axes[1].set_title(f"Weighted Average Confusion Matrix\n(AUC: {weighted_auc:.5f})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()

print("\nAnalysis complete!")


# Display submission distributions
print(f"\nHill Climb Submission distribution:")
print(hill_climb_submission['Personality'].value_counts())

print(f"\nWeighted Average Submission distribution:")
print(weighted_submission['Personality'].value_counts())

# Model performance summary
print("\n" + "="*50)
print("INDIVIDUAL MODEL PERFORMANCE SUMMARY:")
performance_summary = []
for name in models.keys():
    auc = roc_auc_score(y_val, val_preds_dict[name])
    performance_summary.append({'Model': name, 'Validation_AUC': auc})

performance_df = pd.DataFrame(performance_summary).sort_values('Validation_AUC', ascending=False)
print(performance_df.to_string(index=False))

print("\nAnalysis complete!")

