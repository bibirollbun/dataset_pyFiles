from IPython.display import display, HTML

display(HTML("""
<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/ABUALHUSSEIN/Predicting-Optimal-Fertilizers/main/Predicting_Optimal_Fertilizers.jpeg
" width="1000">
</div>
"""))


# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import classification_report


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_train.head()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df_test.head()


column_names = df_train.columns.tolist()
print(column_names)


print(df_train.shape)


# Check the data types and memory usage
df_train.info()


# Save test IDs before dropping
test_ids = df_test['id'].copy()


# Drop 'id' column from both
df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


df_train.head()


# Check for missing values in the dataset
missing_values = df_train.isnull().sum()

print(missing_values)


# Display only features with missing values (if any)
missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    print("âœ… No missing values found in the training dataset.")
else:
    print("âš ï¸� Missing values detected:\n", missing_values)



# Count duplicate rows in the entire DataFrame
duplicate_count = df_train.duplicated().sum()
print("Number of duplicate records:", duplicate_count)



df_train.describe()


df_train.describe(include='O')


df_train['Fertilizer Name'].value_counts()



# Create a value count DataFrame so we can assign categories
fertilizer_counts = df_train['Fertilizer Name'].value_counts().reset_index()
fertilizer_counts.columns = ['Fertilizer Name', 'Count']

# Plot using hue for coloring
sns.barplot(data=fertilizer_counts, x='Fertilizer Name', y='Count', hue='Fertilizer Name', palette='pastel', dodge=False)
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.title('Distribution of Fertilizer Name')
plt.legend([],[], frameon=False)  # Remove redundant legend
plt.show()



# Calculate class distribution
class_dist = df_train['Fertilizer Name'].value_counts(normalize=True).mul(100).round(2)

# Convert to a DataFrame for table output
table = pd.DataFrame({
    'Fertilizer Name': class_dist.index,
    'Proportion (%)': class_dist.values
})

# Print the table nicely
print(table.to_string(index=False))



# Check unique Soil types
df_train['Soil Type'].value_counts()



# Create a value count DataFrame so we can assign categories
fertilizer_counts = df_train['Soil Type'].value_counts().reset_index()
fertilizer_counts.columns = ['Soil Type', 'Count']

# Plot using hue for coloring
sns.barplot(data=fertilizer_counts, x='Soil Type', y='Count', hue='Soil Type', palette='pastel', dodge=False)
plt.xlabel('Soil Type')
plt.ylabel('Count')
plt.title('Distribution of Soil Type')
plt.legend([],[], frameon=False)  # Remove redundant legend
plt.show()



# Check unique Soil types
df_train['Crop Type'].value_counts()



# Count crop types and sort them by frequency
crop_counts = df_train['Crop Type'].value_counts().reset_index()
crop_counts.columns = ['Crop Type', 'Count']

# Sort for better visual order
crop_counts = crop_counts.sort_values(by='Count', ascending=False)

# Plot without hue, sorted and rotated
plt.figure(figsize=(10, 6))
sns.barplot(data=crop_counts, x='Crop Type', y='Count', palette='pastel')

# Rotate x-axis labels to avoid clutter
plt.xticks(rotation=45, ha='right')

# Add title and labels
plt.xlabel('Crop Type')
plt.ylabel('Count')
plt.title('Distribution of Crop Type')
plt.tight_layout()
plt.show()




# Sample only a subset if dataset is large
sample_df = df_train.sample(n=1000, random_state=42)

# Plot pairplot of selected numerical features
sns.pairplot(sample_df,
             vars=[
                 'Temparature',
                 'Humidity',
                 'Moisture',
                 'Nitrogen',
                 'Phosphorous',
                 'Potassium'
             ],
             hue='Fertilizer Name',
             palette='husl',
             plot_kws={'alpha': 0.6, 's': 25},
             diag_kind='kde')

plt.suptitle("Pairplot of Selected Features by Fertilizer Type", y=1.02)
plt.tight_layout()
plt.show()



numeric_features = [
    'Temparature',  # assuming it's spelled like this in your dataset
    'Humidity',
    'Moisture',
    'Nitrogen',
    'Phosphorous',
    'Potassium'
]



pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_train[numeric_features])



# Add PCA results back to the DataFrame
pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
pca_df['Fertilizer Name'] = df_train['Fertilizer Name'].values

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Fertilizer Name', palette='tab10', alpha=0.7)
plt.title('PCA Projection of Fertilizer Data')
plt.tight_layout()
plt.show()



# List of numeric features
numeric_features = [
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Phosphorous', 'Potassium'
]

# Plot
plt.figure(figsize=(18, 22))

for i, feature in enumerate(numeric_features):
    plt.subplot(3, 2, i + 1)

    sns.violinplot(
        data=df_train,
        x='Fertilizer Name',
        y=feature,
        hue='Fertilizer Name',   
        palette='Spectral',
        inner='point',        
        linewidth=1,
        legend=False            
    )

    plt.title(f'{feature} Distribution by Fertilizer Type', fontsize=12)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.suptitle("Violin Plots of Numeric Features by Fertilizer Type", fontsize=16, y=1.02)
plt.show()



# Compute the correlation matrix (numeric columns only)
corr = df_train.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.show()


# Encode Fertilizer Name to numeric codes
df_encoded = df_train.copy()
df_encoded['Fertilizer_Code'] = LabelEncoder().fit_transform(df_encoded['Fertilizer Name'])

# Get correlation of features with target
cor_with_target = df_encoded.corr(numeric_only=True)['Fertilizer_Code'].drop('Fertilizer_Code')

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=cor_with_target.index, y=cor_with_target.values, palette='viridis')
plt.title("Correlation of Numeric Features with Fertilizer Type")
plt.ylabel("Correlation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



X = df_train.drop(columns=['Fertilizer Name'])
y_raw = df_train['Fertilizer Name']  # keep original strings for encoding

# 2. Label encode the target
le = LabelEncoder()
y = le.fit_transform(y_raw)


df_train.head()


import pandas as pd

numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Calculate skewness for each numeric column in your train data
skew_values = df_train[numeric_features].skew()

print("Skewness of numeric features:")
print(skew_values)



# 3. Split train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature types
categorical_features = ['Soil Type', 'Crop Type']
numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# For Logistic Regression (scaling + encoding)
logistic_transformer = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# For tree models (encoding only, no scaling)
tree_transformer = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])


# Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ('preprocessor', logistic_transformer),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Random Forest pipeline
rf_pipeline = Pipeline([
    ('preprocessor', tree_transformer),
    ('classifier', RandomForestClassifier(random_state=42))
])

# XGBoost pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', tree_transformer),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])



models = {
    'Logistic Regression': logistic_pipeline,
    'Random Forest': rf_pipeline,
    'XGBoost': xgb_pipeline
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))



# !pip install lightgbm catboost


import lightgbm as lgb
from catboost import CatBoostClassifier



# LightGBM pipeline
lgb_pipeline = Pipeline([
    ('preprocessor', tree_transformer),
    ('classifier', lgb.LGBMClassifier(random_state=42))
])

# CatBoost pipeline
cat_pipeline = Pipeline([
    ('preprocessor', tree_transformer),
    ('classifier', CatBoostClassifier(random_seed=42, verbose=0))
])


models = {
    'lightgbm': lgb_pipeline,
    # 'CatBoost': cat_pipeline
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))




scores = cross_val_score(xgb_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)

print("5-fold CV accuracy scores:", scores)
print("Mean 5-fold CV accuracy:", np.mean(scores))


# Choose the final model (e.g., XGBoost here)
final_model = xgb_pipeline  # you can switch to rf_pipeline or logistic_pipeline

# Train on full data
final_model.fit(X_train, y_train)



# 8. Prepare test data (make sure df_test has all needed columns except 'id')
X_test = df_test.copy()  # assuming df_test is loaded

# 9. Predict probabilities on test set
test_proba = final_model.predict_proba(X_test)

# 10. Get top 3 fertilizer class indices per sample
top_3_indices = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]  # descending order

# 11. Map back to original fertilizer names using LabelEncoder.classes_
top_3_labels = le.classes_[top_3_indices]

# 12. Join top 3 labels per row with space
joined_preds = [' '.join(row) for row in top_3_labels]
# joined_preds = [' '.join(map(str, row)) for row in top_3_labels]
# 13. Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,  # make sure you have saved test IDs earlier from original test dataframe
    'Fertilizer Name': joined_preds
})

# 14. Save submission file
submission.to_csv('submission.csv', index=False)

# 15. Preview submission head
print(submission.head())




