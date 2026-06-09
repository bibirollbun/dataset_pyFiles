!pip install catboost
!pip install xgboost lightgbm catboost scikit-learn pandas matplotlib seaborn


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# 1. Basic Information
print("="*50)
print("Basic Dataset Information")
print("="*50)
print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())
print("\nMissing values in train set:", train.isnull().sum().sum())
print("Missing values in test set:", test.isnull().sum().sum())

# 2. Data Description
print("\n" + "="*50)
print("Numerical Features Description")
print("="*50)
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
print(train[num_cols].describe().transpose())

print("\n" + "="*50)
print("Categorical Features Description")
print("="*50)
cat_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
for col in cat_cols:
    print(f"\n{col} unique values:", train[col].nunique())
    print(train[col].value_counts(normalize=True).head(10))


# Numerical features to analyze
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# First show individual train/test distributions
print("="*60)
print("Individual Train/Test Distributions")
print("="*60)
for col in num_cols:
    plt.figure(figsize=(12, 5))

    # Train distribution
    plt.subplot(1, 2, 1)
    sns.histplot(train[col], kde=True, color='blue')
    plt.title(f'Train {col} Distribution')
    plt.xlabel('')

    # Test distribution
    plt.subplot(1, 2, 2)
    sns.histplot(test[col], kde=True, color='orange')
    plt.title(f'Test {col} Distribution')
    plt.xlabel('')

    plt.tight_layout()
    plt.show()

# Then show combined comparison
print("\n" + "="*60)
print("Combined Distribution Comparison")
print("="*60)
plt.figure(figsize=(15, 20))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(train[col], color='blue', label='Train', kde=True, alpha=0.6)
    sns.histplot(test[col], color='orange', label='Test', kde=True, alpha=0.6)
    plt.title(f'{col} Distribution', fontsize=12)
    plt.legend()

plt.tight_layout()
plt.show()


# Set style and color palette
plt.style.use('seaborn-v0_8')
sns.set_style("whitegrid")
custom_palette = sns.color_palette("husl", len(train['Fertilizer Name'].unique()))

# 1. Box Plot of Numerical Features by Fertilizer Type
for col in num_cols:
    plt.figure(figsize=(10, 5))  # Smaller figure size
    ax = sns.boxplot(x='Fertilizer Name', y=col, data=train,
                    palette=custom_palette,
                    width=0.6,  # Narrower boxes
                    fliersize=3)  # Smaller outlier markers

    plt.title(f'{col} by Fertilizer Type', fontsize=12, pad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylabel(col, fontsize=10)
    plt.xlabel('Fertilizer Name', fontsize=10)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

# 2. Combined Nutrient Box Plot
plt.figure(figsize=(12, 6))
melted_df = train.melt(id_vars=['Fertilizer Name'],
                      value_vars=['Nitrogen', 'Potassium', 'Phosphorous'],
                      var_name='Nutrient',
                      value_name='Value')

ax = sns.boxplot(x='Nutrient', y='Value', hue='Fertilizer Name',
                data=melted_df, palette=custom_palette,
                width=0.7, linewidth=1)

plt.title('Nutrient Levels by Fertilizer Type', fontsize=12, pad=10)
plt.xlabel('Nutrient', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# legend
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1),
           loc='upper left', fontsize=8, title='Fertilizer',
           title_fontsize=9)

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Target Distribution Trio (Improved)
plt.figure(figsize=(15, 5))

# Count Plot
plt.subplot(1, 3, 1)
sns.countplot(y='Fertilizer Name', data=train,
             order=train['Fertilizer Name'].value_counts().index,
             palette=custom_palette)
plt.title('Fertilizer Distribution (Count)', fontsize=12)
plt.xlabel('Count', fontsize=10)
plt.ylabel('', fontsize=10)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# Pie Chart
plt.subplot(1, 3, 2)
train['Fertilizer Name'].value_counts().plot.pie(
    autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
    startangle=90,
    colors=custom_palette,
    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
    textprops={'fontsize': 9}
)
plt.title('Percentage Distribution', fontsize=12)
plt.ylabel('')

# Box Plot of Counts
plt.subplot(1, 3, 3)
sns.boxplot(x=train['Fertilizer Name'].value_counts().values,
           color='skyblue', width=0.4)
plt.title('Fertilizer Count Distribution', fontsize=12)
plt.xlabel('Records per Type', fontsize=10)
plt.xticks(fontsize=9)

plt.tight_layout()
plt.show()


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Pair Plot for Train Set
print("\n" + "="*60)
print("Pair Plot for Train Set")
print("="*60 + "\n")

sns.pairplot(train)

# 2. Pair Plot for Train Set
print("\n" + "="*60)
print("Train Set: Pairwise Relationships")
print("="*60 + "\n")

plt.figure(figsize=(15, 15))
train_pair = sns.pairplot(
    train[num_cols + ['Fertilizer Name']],
    hue='Fertilizer Name',
    palette='viridis',
    plot_kws={'alpha': 0.7, 's': 20, 'edgecolor': 'k', 'linewidth': 0.3},
    diag_kind='kde',
    corner=True,
    height=2.5
)
train_pair.fig.suptitle(
    'Train Set: Pairwise Feature Relationships by Fertilizer Type',
    y=1.02,
    fontsize=14
)
plt.show()

# 3. Correlation Heatmap (Train Set)
print("\n" + "="*60)
print("Train Set Correlation Analysis")
print("="*60 + "\n")

plt.figure(figsize=(10, 8))
corr_matrix = train[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    mask=mask,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    annot_kws={"size": 10}
)
plt.title('Train Set: Numerical Features Correlation', fontsize=14)
plt.xticks(rotation=45)
plt.show()


cat_cols = ['Soil Type', 'Crop Type']

## Categorical Feature Analysis
print("\n" + "="*60)
print("Individual Categorical Feature Distributions")
print("="*60 + "\n")

# Custom color palette
palette = sns.color_palette('pastel')

for col in cat_cols:
    # Train Distribution - Pie Chart
    plt.figure(figsize=(5, 5))
    train[col].value_counts().plot.pie(autopct='%1.1f%%',
                                     startangle=90,
                                     colors=palette,
                                     wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
                                     textprops={'fontsize': 8})
    plt.title(f'Train {col}\n', fontsize=10)
    plt.ylabel('')
    plt.show()

    # Train Distribution - Bar Chart
    plt.figure(figsize=(6, 3))
    sns.countplot(y=col, data=train, order=train[col].value_counts().index, palette=palette)
    plt.title(f'Train {col}\n', fontsize=10)
    plt.xlabel('Count', fontsize=8)
    plt.ylabel('', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.show()

    # Test Distribution - Pie Chart
    plt.figure(figsize=(5, 5))
    test[col].value_counts().plot.pie(autopct='%1.1f%%',
                                    startangle=90,
                                    colors=palette,
                                    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
                                    textprops={'fontsize': 8})
    plt.title(f'Test {col}\n', fontsize=10)
    plt.ylabel('')
    plt.show()

    # Test Distribution - Bar Chart
    plt.figure(figsize=(6, 3))
    sns.countplot(y=col, data=test, order=test[col].value_counts().index, palette=palette)
    plt.title(f'Test {col}\n', fontsize=10)
    plt.xlabel('Count', fontsize=8)
    plt.ylabel('', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.show()

## Comparison Between Train and Test
print("\n" + "="*60)
print("Train-Test Distribution Comparison")
print("="*60 + "\n")

for col in cat_cols:
    # Prepare data
    train_counts = train[col].value_counts(normalize=True).reset_index()
    train_counts.columns = [col, 'Train']
    test_counts = test[col].value_counts(normalize=True).reset_index()
    test_counts.columns = [col, 'Test']
    merged = pd.merge(train_counts, test_counts, on=col, how='outer').fillna(0)

    # Plot
    plt.figure(figsize=(8, 4))
    merged.set_index(col).plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title(f'{col} Comparison\n', fontsize=10)
    plt.xlabel('', fontsize=8)
    plt.ylabel('Percentage', fontsize=8)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(title='Dataset', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


print("\n" + "="*60)
print("Target-Focused Correlation Analysis")
print("="*60 + "\n")

# 1. Prepare encoded data for correlation analysis
# Encode categorical features for correlation calculation
encoded_data = train.copy()
le = LabelEncoder()
for col in cat_cols:
    encoded_data[col] = le.fit_transform(encoded_data[col])
encoded_data['Fertilizer Name'] = le.fit_transform(encoded_data['Fertilizer Name'])

# 2. Calculate correlations with target
corr_with_target = encoded_data.corr()[['Fertilizer Name']].drop('Fertilizer Name')
corr_with_target.columns = ['Correlation']
corr_with_target['Absolute_Correlation'] = corr_with_target['Correlation'].abs()
corr_with_target = corr_with_target.sort_values('Absolute_Correlation', ascending=False)

# 3. Top 10 Features Correlated with Target
plt.figure(figsize=(10, 6))
top_features = corr_with_target.head(10).sort_values('Correlation', ascending=True)
colors = ['red' if x < 0 else 'green' for x in top_features['Correlation']]
top_features['Correlation'].plot(kind='barh', color=colors)
plt.title('Top 10 Features Correlated with Fertilizer Type', fontsize=14)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add correlation values on bars
for i, v in enumerate(top_features['Correlation']):
    plt.text(v, i, f"{v:.2f}", color='black', ha='left' if v < 0 else 'right', va='center')

plt.tight_layout()
plt.show()

# 4. Detailed Correlation Table
print("\nDetailed Correlation with Fertilizer Name:")
display(corr_with_target.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1)
                           .format("{:.2f}")
                           .set_caption("Feature Correlations with Target"))

# 5. Feature-Target Relationship Visualization
top_3_features = corr_with_target.index[:3]
plt.figure(figsize=(15, 5))
for i, feature in enumerate(top_3_features, 1):
    plt.subplot(1, 3, i)
    if feature in num_cols:
        sns.boxplot(x='Fertilizer Name', y=feature, data=train, palette='viridis')
    else:
        sns.countplot(x='Fertilizer Name', hue=feature, data=train, palette='viridis')
    plt.title(f"{feature} by Fertilizer Type", fontsize=12)
    plt.xticks(rotation=45)
    if i != 1:
        plt.ylabel('')
plt.tight_layout()
plt.show()

# 6. Correlation Interpretation
print("\n" + "="*60)
print("Key Insights from Correlation Analysis")
print("="*60)
print(f"\nMost Positively Correlated: {corr_with_target.index[0]} (r = {corr_with_target.iloc[0,0]:.2f})")
print(f"Most Negatively Correlated: {corr_with_target.index[-1]} (r = {corr_with_target.iloc[-1,0]:.2f})")
print("\nRecommendations:")
print("- Prioritize features with |r| > 0.3 in modeling")
print("- Investigate strongly negative correlations for potential inverse relationships")
print("- Consider feature interactions for top correlated pairs")


# Preprocessing
def preprocess_data(df):
    # Make a copy
    df = df.copy()

    # Drop ID column
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    return df

# Prepare data
X = preprocess_data(train.drop('Fertilizer Name', axis=1))
y = train['Fertilizer Name']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import numpy as np
import lightgbm as lgb

# Encode the target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(iterations=200, random_state=42, verbose=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

# Function to calculate MAP@3
def mapk(y_true, y_pred_proba, k=3):
    top_k = np.argsort(-y_pred_proba, axis=1)[:, :k]
    score = 0.0
    for i in range(len(y_true)):
        if y_true[i] in top_k[i]:
            rank = np.where(top_k[i] == y_true[i])[0][0]
            score += 1 / (rank + 1)
    return score / len(y_true)

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train_encoded)

    # Get predictions
    preds = model.predict(X_val)

    # Calculate F1 score
    f1 = f1_score(y_val_encoded, preds, average='macro')

    # Calculate MAP@3 (requires class probabilities)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)
        map3 = mapk(y_val_encoded, y_proba)
    else:
        map3 = np.nan

    results.append({
        'Model': name,
        'F1 (macro)': f1,
        'MAP@3': map3
    })
    print(f"{name}: F1 (macro) = {f1:.4f}, MAP@3 = {map3:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results).set_index('Model')

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# F1 Score Plot
results_df['F1 (macro)'].sort_values().plot(kind='barh', ax=ax1, color='skyblue')
ax1.set_title('Model Comparison by F1 (macro)')
ax1.set_xlabel('F1 Score')
ax1.grid(axis='x', linestyle='--', alpha=0.6)

# MAP@3 Plot
results_df['MAP@3'].sort_values().plot(kind='barh', ax=ax2, color='salmon')
ax2.set_title('Model Comparison by MAP@3')
ax2.set_xlabel('MAP@3 Score')
ax2.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Display results table
print("\nPerformance Summary:")
display(results_df.sort_values(by='F1 (macro)', ascending=False))


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np

# 1. Encode the target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

# 2. Reduced hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],  
    'max_depth': [3, 5],         
    'learning_rate': [0.1, 0.2], 
    'subsample': [0.9],          
    'colsample_bytree': [0.9]   
}

# 3. Faster XGBoost configuration
xgb = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    objective='multi:softmax',
    num_class=len(le.classes_),
    tree_method='hist',         
    n_jobs=-1,                  
    early_stopping_rounds=10  
)

# 4. Reduced GridSearchCV configuration
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,                  
    verbose=1,
    refit=False                 
)

# 5. Fit with validation set for early stopping
grid_search.fit(
    X_train, y_train_encoded,
    eval_set=[(X_val, y_val_encoded)],
    verbose=0
)

# 6. Train best model with optimal parameters
best_params = grid_search.best_params_
final_model = XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='mlogloss',
    objective='multi:softmax',
    num_class=len(le.classes_),
    tree_method='hist',
    n_jobs=-1
)

final_model.fit(
    X_train, y_train_encoded,
    eval_set=[(X_val, y_val_encoded)],
    verbose=1
)

# 7. Evaluate
val_preds = final_model.predict(X_val)
print("\nValidation F1 (macro):", f1_score(y_val_encoded, val_preds, average='macro'))


# Get the top 3 predicted probabilities for each sample
y_val_proba = final_model.predict_proba(X_val)

# For each sample, get the indices of top 3 predicted classes
top3_pred_indices = np.argsort(-y_val_proba, axis=1)[:, :3]

# Function to calculate MAP@3
def map_at_3(y_true, y_pred_top3):
    ap_sum = 0.0
    for true, pred in zip(y_true, y_pred_top3):
        hits = 0
        precision_sum = 0.0
        for i, p in enumerate(pred, 1):
            if p == true:
                hits += 1
                precision_sum += hits / i
        ap_sum += precision_sum / min(3, 1)  # Divide by min(3, number of relevant items)
    return ap_sum / len(y_true)

# Calculate MAP@3
map3_score = map_at_3(y_val_encoded, top3_pred_indices)
print(f"Validation MAP@3: {map3_score:.4f}")

# Alternatively, you can use sklearn's implementation if available
try:
    from sklearn.metrics import label_ranking_average_precision_score
    # Need to convert true labels to binary indicators for sklearn's version
    y_true_binary = np.zeros_like(y_val_proba)
    y_true_binary[np.arange(len(y_val_encoded)), y_val_encoded] = 1
    map3_score_sklearn = label_ranking_average_precision_score(y_true_binary, y_val_proba)
    print(f"Validation MAP@3 (sklearn): {map3_score_sklearn:.4f}")
except ImportError:
    pass


import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 1. Preprocess and encode target
X_full = preprocess_data(train.drop('Fertilizer Name', axis=1))
y_full = train['Fertilizer Name']
test_processed = preprocess_data(test)

# Encode target variable
le = LabelEncoder()
y_full_encoded = le.fit_transform(y_full)

# 2. Scale numerical features
X_full[num_cols] = scaler.fit_transform(X_full[num_cols])
test_processed[num_cols] = scaler.transform(test_processed[num_cols])

# 3. Train final model with proper class specification
final_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric='mlogloss',
    objective='multi:softmax',
    num_class=len(le.classes_)  # Critical for multi-class
)

final_model.fit(X_full, y_full_encoded)

# 4. MAP@3 Implementation
def mapk(y_true, y_pred_proba, k=3):
    top_k = np.argsort(-y_pred_proba, axis=1)[:, :k]  # Note: - for descending
    score = 0.0
    for i in range(len(y_true)):
        if y_true[i] in top_k[i]:
            rank = np.where(top_k[i] == y_true[i])[0][0]
            score += 1 / (rank + 1)
    return score / len(y_true)

# 5. Validate (if you have validation set)
if 'X_val' in locals():
    y_val_proba = final_model.predict_proba(X_val)
    y_val_encoded = le.transform(y_val)  # Use same encoder!
    map3_score = mapk(y_val_encoded, y_val_proba)
    print(f"Validation MAP@3 Score: {map3_score:.4f}")

# 6. Generate predictions
test_preds_encoded = final_model.predict(test_processed)
test_preds = le.inverse_transform(test_preds_encoded)  # Convert back to original labels

# 7. Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file created!")

