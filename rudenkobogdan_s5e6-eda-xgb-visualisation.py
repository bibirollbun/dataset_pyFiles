import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up visualization style
sns.set(style="whitegrid")
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)
ALPHA = 0.75


test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')


train.info()


train = train.drop(['id'], axis=1)
test_id = test['id']
test = test.drop(['id'], axis=1)


num = test.select_dtypes(include=['int64', 'float64']).columns
cat = test.select_dtypes(include=['object']).columns
target_col = 'Fertilizer Name'


print("\n--- Target Variable Analysis ---")
target_counts = train[target_col].value_counts()
plt.figure(figsize=(15, 6))
ax = sns.barplot(x=target_counts.index, y=target_counts.values, alpha=ALPHA)
plt.title('Distribution of Target Variable: Fertilizer Name')
plt.ylabel('Count')
plt.xlabel('Fertilizer')
plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(f'{p.get_height()} ({p.get_height()/len(train):.1%})', 
               (p.get_x() + p.get_width()/2., p.get_height()), 
               ha='center', va='bottom')
plt.tight_layout()
plt.show()


print("\n--- Numerical Features Analysis ---")
num_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target_col in num_features:
    num_features.remove(target_col)

# Statistical summary
print("Statistical Summary of Numerical Features:")
display(train[num_features].describe().T.sort_values('mean', ascending=False))

# Distribution of numerical features
if len(num_features) > 0:
    # Create histograms for numerical features
    plt.figure(figsize=(15, len(num_features)*3))
    for i, feature in enumerate(num_features):
        plt.subplot(len(num_features), 2, i*2+1)
        sns.histplot(train[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        
        plt.subplot(len(num_features), 2, i*2+2)
        sns.boxplot(x=train[feature])
        plt.title(f'Boxplot of {feature}')
    plt.tight_layout()
    plt.show()
    
    # Check for outliers using IQR method
    print("\nOutlier Analysis (IQR method):")
    outliers = {}
    for feature in num_features:
        Q1 = train[feature].quantile(0.25)
        Q3 = train[feature].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((train[feature] < (Q1 - 1.5 * IQR)) | (train[feature] > (Q3 + 1.5 * IQR))).sum()
        outlier_percent = outlier_count / len(train) * 100
        if outlier_count > 0:
            outliers[feature] = (outlier_count, outlier_percent)
    
    if outliers:
        outliers_df = pd.DataFrame.from_dict(outliers, orient='index', columns=['Count', 'Percentage'])
        outliers_df['Percentage'] = outliers_df['Percentage'].round(2)
        display(outliers_df.sort_values('Percentage', ascending=False))
    else:
        print("No outliers found using IQR method.")



print("\n--- Correlation Analysis ---")
plt.figure(figsize=(12, 10))
correlation_matrix = train[num_features].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Get top correlations
print("\nTop Feature Correlations:")
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

corr_df = pd.DataFrame(corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
display(corr_df.sort_values('Correlation', key=abs, ascending=False).head(10))



print("\n--- Categorical Features Analysis ---")
cat_features = train.select_dtypes(include=['object']).columns.tolist()
if target_col in cat_features:
    cat_features.remove(target_col)

if len(cat_features) > 0:
    # Count unique values in each categorical feature
    cat_unique = {col: train[col].nunique() for col in cat_features}
    cat_unique_df = pd.DataFrame.from_dict(cat_unique, orient='index', columns=['Unique Values'])
    cat_unique_df = cat_unique_df.sort_values('Unique Values', ascending=False)
    display(cat_unique_df)
    
    # Show value counts for categorical features
    for feature in cat_features:
        plt.figure(figsize=(12, 6))
        value_counts = train[feature].value_counts()
        ax = sns.countplot(x=feature, data=train, order=value_counts.index, alpha=ALPHA)
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45)
        
        # Add count labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height()} ({p.get_height()/len(train):.1%})', 
                       (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
        
        # Relationship with target variable
        plt.figure(figsize=(14, 8))
        sns.countplot(x=feature, hue=target_col, data=train)
        plt.title(f'Relationship between {feature} and {target_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Normalized distribution
        plt.figure(figsize=(14, 8))
        pd.crosstab(train[feature], train[target_col], normalize='index').plot(kind='bar', stacked=True)
        plt.title(f'Normalized Distribution of {target_col} by {feature}')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



print("\n--- Data Distribution Analysis ---")
if len(num_features) > 0:
    # Calculate skewness
    skewness = train[num_features].skew().sort_values(ascending=False)
    print("Skewness of numerical features:")
    display(skewness)
    
    # Visualize features with high skewness
    high_skew = skewness[abs(skewness) > 1].index.tolist()
    if high_skew:
        print(f"\nFeatures with high skewness (|skew| > 1): {len(high_skew)}")
        plt.figure(figsize=(15, 5 * (len(high_skew) + 1) // 2))
        for i, feature in enumerate(high_skew):
            plt.subplot((len(high_skew) + 1) // 2, 2, i+1)
            sns.histplot(train[feature], kde=True)
            plt.title(f'{feature} (Skew: {skewness[feature]:.2f})')
            
            # Add log-transformed version
            if skewness[feature] > 0:  # Only for positive skew
                non_zeros = train[feature] > 0
                if non_zeros.all():
                    log_data = np.log1p(train[feature])
                    log_skew = stats.skew(log_data)
                    plt.subplot((len(high_skew) + 1) // 2, 2, i+2)
                    sns.histplot(log_data, kde=True, color='green')
                    plt.title(f'Log({feature}) (Skew: {log_skew:.2f})')
        plt.tight_layout()
        plt.show()


print("\n--- Feature Relationships with Target ---")
# For numerical features vs categorical target
for feature in num_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=target_col, y=feature, data=train)
    plt.title(f'Distribution of {feature} by {target_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # ANOVA test to check if the means are significantly different
    groups = [train[train[target_col] == fertilizer][feature].values for fertilizer in train[target_col].unique()]
    f_val, p_val = stats.f_oneway(*groups)
    print(f"ANOVA test for {feature} by {target_col}: F={f_val:.2f}, p={p_val:.4f}")
    if p_val < 0.05:
        print(f"The means of {feature} are significantly different across fertilizer types (p<0.05)")
    else:
        print(f"No significant difference in means of {feature} across fertilizer types (p≥0.05)")
    print("-" * 50)


print("\n--- Principal Component Analysis ---")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

if len(num_features) > 2:
    # Standardize the data
    X = train[num_features]
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df[target_col] = train[target_col].values
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue=target_col, data=pca_df, palette='viridis', s=100, alpha=0.7)
    plt.title('PCA: Principal Component Analysis')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.show()
    
    # Print explained variance ratio
    print(f"Explained variance ratio by the first two components: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Print feature loadings
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'], 
        index=num_features
    )
    display(loadings)



print("\n--- Summary Statistics by Target ---")
for feature in num_features:
    display(train.groupby(target_col)[feature].describe().T)

# Check for data imbalance
print("\n--- Data Balance Analysis ---")
target_balance = train[target_col].value_counts(normalize=True) * 100
imbalance = target_balance.max() - target_balance.min()
print(f"Class imbalance: {imbalance:.2f}%")
if imbalance > 10:
    print("The dataset shows some class imbalance (>10% difference between most and least common classes)")
    if imbalance > 20:
        print("The imbalance is substantial (>20%) and may require addressing during modeling")



print("\n--- Feature Engineering ---")

# Create a copy of the datasets to avoid modifying originals
train_fe = train.copy()
test_fe = test.copy()


# These can capture important relationships between soil components
# N:P:K ratios (common in fertilizer analysis)
if 'N' in train_fe.columns and 'P' in train_fe.columns and 'K' in train_fe.columns:
    train_fe['N_P_ratio'] = train_fe['N'] / (train_fe['P'] + 0.1)  # Adding small constant to avoid division by zero
    train_fe['N_K_ratio'] = train_fe['N'] / (train_fe['K'] + 0.1)
    train_fe['P_K_ratio'] = train_fe['P'] / (train_fe['K'] + 0.1)
    train_fe['NPK_sum'] = train_fe['N'] + train_fe['P'] + train_fe['K']
    
    test_fe['N_P_ratio'] = test_fe['N'] / (test_fe['P'] + 0.1)
    test_fe['N_K_ratio'] = test_fe['N'] / (test_fe['K'] + 0.1)
    test_fe['P_K_ratio'] = test_fe['P'] / (test_fe['K'] + 0.1)
    test_fe['NPK_sum'] = test_fe['N'] + test_fe['P'] + test_fe['K']


# Moisture to temperature ratio (can indicate evaporation potential)
if 'Moisture' in train_fe.columns and 'Temperature' in train_fe.columns:
    train_fe['moisture_temp_ratio'] = train_fe['Moisture'] / (train_fe['Temperature'] + 0.1)
    test_fe['moisture_temp_ratio'] = test_fe['Moisture'] / (test_fe['Temperature'] + 0.1)



# Soil pH interactions
if 'pH' in train_fe.columns:
    # pH ranges (acidic, neutral, alkaline)
    train_fe['is_acidic'] = (train_fe['pH'] < 6.5).astype(int)
    train_fe['is_neutral'] = ((train_fe['pH'] >= 6.5) & (train_fe['pH'] <= 7.5)).astype(int)
    train_fe['is_alkaline'] = (train_fe['pH'] > 7.5).astype(int)
    
    test_fe['is_acidic'] = (test_fe['pH'] < 6.5).astype(int)
    test_fe['is_neutral'] = ((test_fe['pH'] >= 6.5) & (test_fe['pH'] <= 7.5)).astype(int)
    test_fe['is_alkaline'] = (test_fe['pH'] > 7.5).astype(int)
    
    # pH interactions with nutrients
    for nutrient in ['N', 'P', 'K', 'Potassium', 'Nitrogen', 'Phosphorous']:
        if nutrient in train_fe.columns:
            train_fe[f'{nutrient}_pH_interaction'] = train_fe[nutrient] * train_fe['pH']
            test_fe[f'{nutrient}_pH_interaction'] = test_fe[nutrient] * test_fe['pH']



# Polynomial features for key soil properties
for feature in ['N', 'P', 'K', 'pH', 'Moisture', 'Temperature']:
    if feature in train_fe.columns:
        train_fe[f'{feature}_squared'] = train_fe[feature] ** 2
        test_fe[f'{feature}_squared'] = test_fe[feature] ** 2


# Soil texture and quality indices
# Clay-silt-sand ratio if available
soil_components = [col for col in train_fe.columns if col in ['Clay', 'Silt', 'Sand']]
if len(soil_components) >= 2:
    for i in range(len(soil_components)):
        for j in range(i+1, len(soil_components)):
            comp1 = soil_components[i]
            comp2 = soil_components[j]
            train_fe[f'{comp1}_{comp2}_ratio'] = train_fe[comp1] / (train_fe[comp2] + 0.1)
            test_fe[f'{comp1}_{comp2}_ratio'] = test_fe[comp1] / (test_fe[comp2] + 0.1)



# Composite features that might represent overall soil fertility
fertility_features = [col for col in train_fe.columns if col in 
                      ['N', 'P', 'K', 'Nitrogen', 'Phosphorous', 'Potassium', 'Organic_Matter']]
if len(fertility_features) >= 3:
    train_fe['fertility_index'] = train_fe[fertility_features].mean(axis=1)
    test_fe['fertility_index'] = test_fe[fertility_features].mean(axis=1)



# Statistical interactions between key features
from itertools import combinations
key_features = [col for col in train_fe.columns if col in 
               ['N', 'P', 'K', 'pH', 'Moisture', 'Temperature', 'Rainfall']]
if len(key_features) >= 2:
    for feat1, feat2 in list(combinations(key_features, 2))[:10]:  # Limit to first 10 combinations
        train_fe[f'{feat1}_{feat2}_interaction'] = train_fe[feat1] * train_fe[feat2]
        test_fe[f'{feat1}_{feat2}_interaction'] = test_fe[feat1] * test_fe[feat2]


# Clustering as a feature
from sklearn.cluster import KMeans
if len(num_features) >= 3:
    # Select numerical features for clustering
    X_cluster = train_fe[num_features].copy()
    X_test_cluster = test_fe[num_features].copy()
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    X_test_cluster_scaled = scaler.transform(X_test_cluster)
    
    # Create clusters
    for n_clusters in [3, 5, 7]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        train_fe[f'cluster_{n_clusters}'] = kmeans.fit_predict(X_cluster_scaled)
        test_fe[f'cluster_{n_clusters}'] = kmeans.predict(X_test_cluster_scaled)



# Log transformations for skewed features
skewed_features = []
for feature in num_features:
    if abs(train_fe[feature].skew()) > 1:
        skewed_features.append(feature)

for feature in skewed_features:
    # Apply log transformation for positive skewed features
    if train_fe[feature].min() >= 0:  # Ensure no negative values
        train_fe[f'log_{feature}'] = np.log1p(train_fe[feature])
        test_fe[f'log_{feature}'] = np.log1p(test_fe[feature])


# Binning important numerical features
for feature in ['pH', 'Moisture', 'Temperature']:
    if feature in train_fe.columns:
        train_fe[f'{feature}_bin'] = pd.qcut(train_fe[feature], 4, labels=False, duplicates='drop')
        
        # For test set, use the same bins as train
        bins = pd.qcut(train_fe[feature], 4, retbins=True, duplicates='drop')[1]
        test_fe[f'{feature}_bin'] = pd.cut(test_fe[feature], bins=bins, labels=False, include_lowest=True)
        # Handle potential out-of-bounds values
        test_fe[f'{feature}_bin'] = test_fe[f'{feature}_bin'].fillna(0)



categorical_cols = [col for col in train_fe.columns if col in cat_features]
if categorical_cols:
    print("One-hot encoding categorical features...")
    train_fe = pd.get_dummies(train_fe, columns=categorical_cols, drop_first=True)
    test_fe = pd.get_dummies(test_fe, columns=categorical_cols, drop_first=True)
    
    # Ensure test has all columns from train
    for col in train_fe.columns:
        if col not in test_fe.columns and col != target_col:
            test_fe[col] = 0


print(f"Original train shape: {train.shape}")
print(f"Engineered train shape: {train_fe.shape}")
print(f"Original test shape: {test.shape}")
print(f"Engineered test shape: {test_fe.shape}")


new_features = [col for col in train_fe.columns if col not in train.columns and col != target_col]
print(f"\nNumber of new features created: {len(new_features)}")
print("New features:", new_features[:10], "..." if len(new_features) > 10 else "")


# --- XGBoost Model Training with CUDA & Optuna ---
print("\n--- XGBoost Model Training with CUDA & Optuna ---")

import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Separate features and target
X_train = train_fe.drop(target_col, axis=1)
y_train = train_fe[target_col]

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
print(f"Original classes: {label_encoder.classes_}")
print(f"Encoded as: {np.unique(y_train_encoded)}")

# Check for GPU availability - safer approach
try:
    # Try to create a small DMatrix and set device to GPU
    dtrain_check = xgb.DMatrix(np.random.randn(10, 10))
    param_check = {'tree_method': 'gpu_hist'}
    # Just check if the setting works without training
    handle = xgb.training.train(param_check, dtrain_check, num_boost_round=0)
    del dtrain_check, handle
    gpu_available = True
    print("GPU is available for XGBoost")
except Exception as e:
    gpu_available = False
    print(f"GPU is not available for XGBoost: {e}")
    print("Using CPU instead")

import warnings
warnings.filterwarnings("ignore")

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to optimize
    params = {
        'objective': 'multi:softmax',  # Multi-class classification
        'eval_metric': 'mlogloss',     # Multiclass logloss
        'num_class': len(np.unique(y_train_encoded)),
        'tree_method': 'hist',  # Use GPU if available
        'device': 'cuda',
        
        # Hyperparameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # Fix the log distribution by using a small positive value instead of 0
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        xgb.XGBClassifier(**params, random_state=42),
        X_train, y_train_encoded,  # Use encoded target
        scoring='accuracy', 
        cv=cv, 
        n_jobs=-1
    )
    
    # Return the mean cross-validation score
    return scores.mean()

# Create and run the Optuna study
print("Starting hyperparameter optimization with Optuna...")
start_time = time.time()

optuna.logging.disable_default_handler()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25) 

optimization_time = time.time() - start_time
print(f"Optimization completed in {optimization_time:.2f} seconds")

# Get the best parameters
best_params = study.best_params
best_score = study.best_value
print(f"Best cross-validation accuracy: {best_score:.4f}")
print("Best parameters:", best_params)

# Add fixed parameters to best_params
best_params['objective'] = 'multi:softmax'
best_params['eval_metric'] = 'mlogloss'
best_params['num_class'] = len(np.unique(y_train_encoded))
best_params['tree_method'] = 'gpu_hist' if gpu_available else 'hist'

# Make predictions on the test set
print("\nMaking predictions on test data...")
X_test = test_fe.copy()

# Ensure X_test has the same columns as X_train
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0  # Fill missing columns with zeros

# Remove any extra columns in X_test that aren't in X_train
X_test = X_test[X_train.columns]

# Predict

model = xgb.XGBClassifier(**best_params, random_state=42)
model.fit(X_train, y_train_encoded)

y_pred_encoded = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Convert numerical predictions back to original class labels
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded.astype(int))

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_id,
    'Fertilizer Name': y_pred_labels
})

# Save submission
submission.to_csv('xgboost_submission.csv', index=False)
print("Submission file created: xgboost_submission.csv")


# Train the model with the best parameters on the full training set
print("\nTraining final model with best parameters...")
model = xgb.XGBClassifier(**best_params, random_state=42)
model.fit(X_train, y_train_encoded)  # Use encoded target

# Plot feature importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(model, max_num_features=20, height=0.5)
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.show()

# Optuna visualization
plt.figure(figsize=(12, 8))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title('Hyperparameter Importance')
plt.tight_layout()
plt.show()

# Plot optimization history
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.tight_layout()
plt.show()

# Examine prediction distribution
plt.figure(figsize=(12, 6))
sns.countplot(x=y_pred_labels)
plt.title('Distribution of Predicted Fertilizer Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate prediction probabilities statistics
pred_proba_df = pd.DataFrame(y_pred_proba, columns=range(len(label_encoder.classes_)))
# Rename columns to actual class names for clarity
pred_proba_df.columns = label_encoder.classes_
avg_confidence = pred_proba_df.max(axis=1).mean()
min_confidence = pred_proba_df.max(axis=1).min()
print(f"Average prediction confidence: {avg_confidence:.4f}")
print(f"Minimum prediction confidence: {min_confidence:.4f}")

# Save the model for future use
model.save_model('xgboost_fertilizer_model.json')
print("Model saved to xgboost_fertilizer_model.json")

# Cross-validation evaluation on training data
print("\n--- Cross-validation evaluation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=cv, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Additional model insights
print("\n--- Model insights ---")
# Check performance on each class with cross-validation
from sklearn.model_selection import cross_val_predict
y_train_pred_encoded = cross_val_predict(model, X_train, y_train_encoded, cv=5)

# Convert back to original labels for reporting
y_train_true_labels = label_encoder.inverse_transform(y_train_encoded)
y_train_pred_labels = label_encoder.inverse_transform(y_train_pred_encoded)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_train_true_labels, y_train_pred_labels))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_train_true_labels, y_train_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Save the label encoder for future use
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder saved to label_encoder.pkl")

