import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')


print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nColumns:", train_df.columns.tolist())
print("\nTarget distribution:")
print(train_df['Cover_Type'].value_counts().sort_index())


quantitative_features = [
    'Elevation', 'Aspect', 'Slope', 
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
    'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
]

wilderness_features = [col for col in train_df.columns if 'Wilderness_Area' in col]
soil_features = [col for col in train_df.columns if 'Soil_Type' in col]

print(f"Quantitative features: {len(quantitative_features)}")
print(f"Wilderness area features: {len(wilderness_features)}")
print(f"Soil type features: {len(soil_features)}")

corr_matrix = train_df[quantitative_features + ['Cover_Type']].corr()
target_correlations = corr_matrix['Cover_Type'].sort_values(ascending=False)

print("\nTop features correlated with Cover_Type:")
print(target_correlations.head(10))


wilderness_counts = train_df[wilderness_features].sum().sort_values(ascending=False)
print("Wilderness Area Distribution:")
print(wilderness_counts)

plt.figure(figsize=(12, 6))
wilderness_counts.plot(kind='bar')
plt.title('Distribution of Samples Across Wilderness Areas')
plt.xlabel('Wilderness Area')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.show()


soil_counts = train_df[soil_features].sum().sort_values(ascending=False)
print("Top 10 Most Common Soil Types:")
print(soil_counts.head(10))

plt.figure(figsize=(15, 6))
soil_counts.head(15).plot(kind='bar')
plt.title('Top 15 Soil Types by Sample Count')
plt.xlabel('Soil Type')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.show()


top_features = target_correlations.index[1:6]  

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, feature in enumerate(top_features[:6]):
    for cover_type in sorted(train_df['Cover_Type'].unique()):
        data = train_df[train_df['Cover_Type'] == cover_type][feature]
        axes[i].hist(data, alpha=0.6, label=f'Type {cover_type}', bins=30)
    axes[i].set_title(f'{feature} by Cover Type')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

plt.tight_layout()
plt.show()


rf_preview = RandomForestClassifier(n_estimators=100, random_state=42)
rf_preview.fit(train_df[quantitative_features], train_df['Cover_Type'])

feature_importance = pd.DataFrame({
    'feature': quantitative_features,
    'importance': rf_preview.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Preliminary Feature Importance (Quantitative Features Only)')
plt.tight_layout()
plt.show()

print("Top 10 Most Important Quantitative Features:")
print(feature_importance.head(10))


outlier_info = []
for feature in quantitative_features:
    Q1 = train_df[feature].quantile(0.25)
    Q3 = train_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = train_df[(train_df[feature] < lower_bound) | (train_df[feature] > upper_bound)]
    outlier_info.append({
        'feature': feature,
        'outliers_count': len(outliers),
        'outliers_percentage': len(outliers) / len(train_df) * 100
    })

outlier_df = pd.DataFrame(outlier_info)
print("Outlier Analysis:")
print(outlier_df.sort_values('outliers_percentage', ascending=False))


X = train_df.drop(['Id', 'Cover_Type'], axis=1)
y = train_df['Cover_Type']

y = y - 1

X_test = test_df.drop('Id', axis=1)
test_ids = test_df['Id']


def create_features(df):
    """Create additional features to improve model performance"""
    df_copy = df.copy()
    
    df_copy['elevation_squared'] = df_copy['Elevation'] ** 2
    df_copy['elevation_log'] = np.log1p(df_copy['Elevation'])
    
    df_copy['total_distance_hydrology'] = df_copy['Horizontal_Distance_To_Hydrology'] + df_copy['Vertical_Distance_To_Hydrology']
    df_copy['euclidean_distance_hydrology'] = np.sqrt(
        df_copy['Horizontal_Distance_To_Hydrology']**2 + 
        df_copy['Vertical_Distance_To_Hydrology']**2
    )
    
    df_copy['hillshade_mean'] = df_copy[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].mean(axis=1)
    df_copy['hillshade_range'] = df_copy['Hillshade_3pm'] - df_copy['Hillshade_9am']
    
    
    df_copy['distance_ratio_road_fire'] = df_copy['Horizontal_Distance_To_Roadways'] / (df_copy['Horizontal_Distance_To_Fire_Points'] + 1)
    df_copy['distance_ratio_hydrology_fire'] = df_copy['Horizontal_Distance_To_Hydrology'] / (df_copy['Horizontal_Distance_To_Fire_Points'] + 1)
    
    
    wilderness_cols = [col for col in df_copy.columns if 'Wilderness_Area' in col]
    soil_cols = [col for col in df_copy.columns if 'Soil_Type' in col]
    
    df_copy['wilderness_area_count'] = df_copy[wilderness_cols].sum(axis=1)
    df_copy['soil_type_count'] = df_copy[soil_cols].sum(axis=1)
    
    return df_copy



X_enhanced = create_features(X)
X_test_enhanced = create_features(X_test)

print(f"Original features: {X.shape[1]}")
print(f"Enhanced features: {X_enhanced.shape[1]}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enhanced)
X_test_scaled = scaler.transform(X_test_enhanced)


X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")


models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}


results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    
    print(f"{name} Validation Accuracy: {accuracy:.4f}")
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")


print(f"\nTraining {best_model_name} on full dataset...")
best_model.fit(X_scaled, y)

test_predictions = best_model.predict(X_test_scaled)

test_predictions_original = test_predictions + 1

print("Predicted class distribution:")
print(pd.Series(test_predictions_original).value_counts().sort_index())


submission = pd.DataFrame({
    'Id': test_ids,
    'Cover_Type': test_predictions_original
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created!")

print("\nSubmission preview:")
print(submission.head())
print(f"\nCover_Type values in submission: {sorted(submission['Cover_Type'].unique())}")

