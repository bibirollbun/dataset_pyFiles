# First, install correct versions (run this once)
!pip install --upgrade scikit-learn==1.3.0 imbalanced-learn
# then restart



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head()


test.head()


train.shape


train.info()


test.info()


# Target distribution
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Personality', data=train, palette='viridis')
plt.title('Personality Distribution', fontsize=16)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)


# Add percentages
total = len(train)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    x = p.get_x() + p.get_width()/2
    y = p.get_height() + 50
    ax.annotate(percentage, (x, y), ha='center', fontsize=12)
    plt.show()


# Numerical features
num_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']


plt.figure(figsize=(20, 15))
for i, feature in enumerate(num_features, 1):
    plt.subplot(3, 2, i)
    sns.kdeplot(data=train, x=feature, hue='Personality', fill=True, common_norm=False, 
                palette={'Introvert': 'blue', 'Extrovert': 'orange'})
    plt.title(f'{feature} Distribution by Personality', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Density', fontsize=12)
plt.tight_layout()
plt.show()



# Categorical features
cat_features = ['Stage_fear', 'Drained_after_socializing']

plt.figure(figsize=(15, 6))
for i, feature in enumerate(cat_features, 1):
    plt.subplot(1, 2, i)
    sns.countplot(data=train, x=feature, hue='Personality', palette='viridis')
    plt.title(f'{feature} Distribution by Personality', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Personality')
plt.tight_layout()
plt.show()


# Encode target for correlation
train_corr = train.copy()


# Convert categorical columns to numerical values
train_corr['Stage_fear'] = train_corr['Stage_fear'].map({'Yes': 1, 'No': 0})
train_corr['Drained_after_socializing'] = train_corr['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
train_corr['Personality'] = train_corr['Personality'].map({'Introvert': 0, 'Extrovert': 1})


# Drop ID column as it's not useful for correlation
train_corr = train_corr.drop('id', axis=1)


corr = train_corr.corr()


# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
            annot_kws={'size': 10}, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=16)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10, rotation=0)
plt.tight_layout()
plt.show()


# Correlation with target
target_corr = corr['Personality'].sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr.index, y=target_corr.values, palette='viridis')
plt.title('Correlation with Personality', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Create new features
def create_features(df):
    df = df.copy()
    # Social interaction ratio
    df['Social_Interaction_Ratio'] = df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1e-5)
    
    # Energy balance feature
    df['Energy_Balance'] = df['Drained_after_socializing'].map({'Yes': -1, 'No': 1}) * df['Social_event_attendance']
    
    # Social activity index
    df['Social_Activity_Index'] = (df['Social_event_attendance'] + df['Going_outside'] + 
                                   df['Friends_circle_size'] / 10)
    
    # Post frequency adjusted for social circle
    df['Post_Frequency_Adjusted'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1)
    
    return df


# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)


# Show new features
print("Engineered features preview:")
print(train_fe[['Social_Interaction_Ratio', 'Energy_Balance', 
                'Social_Activity_Index', 'Post_Frequency_Adjusted']].head())


# Handle missing values
def handle_missing(df):
    df = df.copy()
    # Numerical columns: fill with median
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency', 'Social_Interaction_Ratio',
                'Social_Activity_Index', 'Post_Frequency_Adjusted']
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical columns: fill with mode
    cat_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Energy_Balance might have NaNs from categorical mapping
    df['Energy_Balance'].fillna(df['Energy_Balance'].median(), inplace=True)
    
    return df


# Apply missing value handling
train_clean = handle_missing(train_fe)
test_clean = handle_missing(test_fe)


# Verify no missing values
print("\nMissing values in training data after handling:")
print(train_clean.isnull().sum().sum())
print("Missing values in test data after handling:")
print(test_clean.isnull().sum().sum())


# Separate features and target
X = train_clean.drop(['id', 'Personality'], axis=1)
y = train_clean['Personality']


# Apply undersampling
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)


# Check new distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=y_res, palette='viridis')
plt.title('Personality Distribution After Undersampling', fontsize=16)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# Define preprocessing
categorical_features = ['Stage_fear', 'Drained_after_socializing']
numerical_features = [col for col in X_res.columns if col not in categorical_features]


# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)]
)


# Create full pipeline with feature selection
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', SVC(probability=True, random_state=42))
])


# Parameter grid for tuning
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'feature_selector__k': [8, 10, 12]
}


# Setup GridSearchCV with stratified k-fold
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)


# Train model
grid_search.fit(X_res, y_res)


# Best parameters
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")


# Get best model
best_model = grid_search.best_estimator_


# Cross-validated metrics
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_model, X_res, y_res, cv=cv, scoring='accuracy')
print(f"\nCross-validated Accuracy: {np.mean(scores):.4f} (± {np.std(scores):.4f})")


# Train-test split evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)


best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)


# Classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))


# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Introvert', 'Extrovert'], 
            yticklabels=['Introvert', 'Extrovert'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.show()


# Get selected features
feature_selector = best_model.named_steps['feature_selector']
preprocessor = best_model.named_steps['preprocessor']


# Get feature names after preprocessing
cat_features_transformed = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = np.concatenate([numerical_features, cat_features_transformed])



# Get selected feature mask
selected_mask = feature_selector.get_support()
selected_features = all_features[selected_mask]




# Get SVM coefficients for linear kernel
if best_model.named_steps['classifier'].kernel == 'linear':
    coefficients = best_model.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': coefficients
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
    plt.title('Feature Importance (SVM Coefficients)', fontsize=16)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.show()


# Prepare test data
test_final = test_clean.drop('id', axis=1)

# Make predictions
test_preds = best_model.predict(test_final)




# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_preds
})

# Save submission
submission.to_csv('submission444.csv', index=False)
print("\nSubmission file created successfully!")
print(submission.head())




# Class distribution in predictions
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=submission, palette='viridis')
plt.title('Predicted Personality Distribution', fontsize=16)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()




