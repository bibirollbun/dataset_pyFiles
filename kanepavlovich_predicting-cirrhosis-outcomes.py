import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.stats import spearmanr, skew
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


train=pd.read_csv('../Data/train.csv')
test=pd.read_csv('../Data/test.csv')
sample=pd.read_csv('../Data/sample_submission.csv')
train.head()


train.describe()


train.dtypes


# Seperate numerical and categorical columns for exploratory analysis
categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']
numerical_cols = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 
                  'Tryglicerides', 'Platelets', 'Prothrombin']
all_features = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage', 'N_Days', 'Age', 
                'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 
                'Tryglicerides', 'Platelets', 'Prothrombin']


# Plot histogram for each numerical column
fig, axes = plt.subplots(nrows=len(numerical_cols), ncols=1, figsize=(12, 3*len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    sns.histplot(data=train, x=col, ax=axes[i]) 
    axes[i].set_title(col, fontsize=14)
    axes[i].set_ylabel('Frequency', fontsize=12)  
    axes[i].set_xlabel(col, fontsize=12)

plt.tight_layout()
plt.show()


# Relationship between features (assessing similar features for redundancy)

correlation_matrix = train[numerical_cols].corr()

# Correlation Heatmap
correlation_matrix = train[numerical_cols].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
heatmap = sns.heatmap(correlation_matrix, 
                      mask=mask,
                      annot=True, 
                      cmap='RdBu_r', 
                      center=0,
                      square=True, 
                      fmt='.2f',
                      cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Numerical Features', fontsize=16, pad=20)

plt.tight_layout()
plt.show()



# Visualise the relationship between each numerical variable and the predictor
fig, axes = plt.subplots(nrows=len(numerical_cols), ncols=1, figsize=(12, 3*len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    sns.boxenplot(data=train, x='Status', y=col, ax=axes[i]) 
    axes[i].set_title(f'{col} by Status', fontsize=14)
    axes[i].set_xlabel('Status', fontsize=12)
    axes[i].set_ylabel(col, fontsize=12)

plt.tight_layout()
plt.show()


# Plot countplot for each categorical column
fig, axes = plt.subplots(nrows=len(categorical_cols), ncols=1, figsize=(12, 3*len(categorical_cols)))

for i, col in enumerate(categorical_cols):
    sns.countplot(data=train, x=col, ax=axes[i]) 
    axes[i].set_title(col, fontsize=14)
    axes[i].set_ylabel('count', fontsize=12)  
    axes[i].set_xlabel(col, fontsize=12)

plt.tight_layout()
plt.show()


# Visualise the relationship between each categorical variable and the predictor
for col in categorical_cols:
    g = sns.catplot(data=train, x=col, hue='Status', kind='count', 
                   height=4, aspect=1.5)
    g.fig.suptitle(f'{col} distribution by Status', fontsize=16, y=1.02)
    g.set_axis_labels(col, 'Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Check percentage of missing values
missing_percent = train.isnull().mean() * 100
print("Missing values percentage:")
print(missing_percent.sort_values(ascending=False))


# Remove features with >50% missing
columns_to_drop = ['Tryglicerides', 'Cholesterol']
train_cleaned = train.drop(columns=columns_to_drop)

# Create Missing column indicators
missing_cols = train_cleaned.columns[train_cleaned.isnull().any()].tolist()

categorical_missing = [col for col in missing_cols if col in ['Spiders', 'Hepatomegaly', 'Ascites', 'Drug']]
numerical_missing = [col for col in missing_cols if col not in categorical_missing]

# Impute missing categorical values with the mode
cat_imputer = SimpleImputer(strategy='most_frequent')
train_cleaned[categorical_missing] = cat_imputer.fit_transform(train_cleaned[categorical_missing])

# Impute missing numerical with median
num_imputer = SimpleImputer(strategy='median')
train_cleaned[numerical_missing] = num_imputer.fit_transform(train_cleaned[numerical_missing])

# Check all have been imputed successfully
print("\nMissing values after Imputation:")
print(train_cleaned.isnull().sum().sort_values(ascending=False))


X = train_cleaned.drop(columns=['Status']) 
y = train_cleaned['Status']

# As there are low amounts of the CL predictor, stratify the split so that we have enough to both train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def create_skewness_preprocessor(X_train, numerical_cols, categorical_cols):
    """
    Create a fitted preprocessor that can be reused on new data
    """    
    skewness = X_train[numerical_cols].apply(lambda x: skew(x.dropna()))
    
    # Separate columns by skewness level
    highly_skewed = skewness[abs(skewness) > 1].index.tolist()
    moderately_skewed = skewness[abs(skewness) <= 1].index.tolist()
    
    # Identify different transformers for different data
    transformers = []
    if highly_skewed:
        transformers.append(('power_transform', PowerTransformer(), highly_skewed))
    
    if moderately_skewed:
        transformers.append(('standard_scale', StandardScaler(), moderately_skewed))
    
    if categorical_cols:
        transformers.append(('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'), categorical_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'  # Keep other columns unchanged
    )
    
    preprocessor.fit(X_train)
    return preprocessor




# re-initialise numerical and categorical columns without the removed columns
categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']
numerical_cols = ['N_Days', 'Age', 'Bilirubin', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 
                  'Platelets', 'Prothrombin']

skewness_processor = create_skewness_preprocessor(X_train, numerical_cols, categorical_cols)

X_train_transformed = skewness_processor.transform(X_train)
X_test_transformed = skewness_processor.transform(X_test)


feature_names = skewness_processor.get_feature_names_out()

# Convert to DataFrame for better readability
X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

print("Transformed feature names:", feature_names)
print("Transformed training data shape:", X_train_transformed.shape)
print("Transformed test data shape:", X_test_transformed.shape)


# Define models and their parameter grids
models_param_grids = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=42, verbose=-1),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
}

# Perform GridSearchCV for each model
best_models = {}
cv_results = {}

for model_name, config in models_param_grids.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name} ")
    print(f"{'='*50}")
    
    grid_search = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['param_grid'],
        n_iter=20,  
        cv=5,      
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    grid_search.fit(X_train_transformed, y_train)
    
    best_models[model_name] = grid_search.best_estimator_
    cv_results[model_name] = grid_search.best_score_
    
    print(f"Best {model_name} parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    
    y_pred = grid_search.best_estimator_.predict(X_test_transformed)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")

# Compare all models
print(f"\n{'='*60}")
print("MODEL COMPARISON RESULTS")
print(f"{'='*60}")

results_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'CV_Accuracy': list(cv_results.values()),
    'Test_Accuracy': [accuracy_score(y_test, best_models[model].predict(X_test_transformed)) 
                     for model in cv_results.keys()]
})

results_df = results_df.sort_values('CV_Accuracy', ascending=False)
print(results_df)

# Evaluate the best model
best_model_name = results_df.iloc[0]['Model']
best_model = best_models[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"Cross-validation accuracy: {results_df.iloc[0]['CV_Accuracy']:.4f}")
print(f"Test accuracy: {results_df.iloc[0]['Test_Accuracy']:.4f}")

# Detailed evaluation of best model
y_pred_best = best_model.predict(X_test_transformed)
print(f"\nDetailed classification report for {best_model_name}:")
print(classification_report(y_test, y_pred_best))



# Feature importance from best model
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train_transformed.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop features for {best_model_name}:")
    print(feature_importance)



# Train final model on full data
X_full = pd.concat([X_train_transformed, X_test_transformed])
y_full = pd.concat([y_train, y_test])

final_model = best_model.__class__(**best_model.get_params())
final_model.fit(X_full, y_full)


# Remove features with >50% missing
columns_to_drop = ['Tryglicerides', 'Cholesterol']
test_cleaned = test.drop(columns=columns_to_drop)

# Create Missing column indicators
missing_cols = test_cleaned.columns[test_cleaned.isnull().any()].tolist()

categorical_missing = [col for col in missing_cols if col in ['Spiders', 'Hepatomegaly', 'Ascites', 'Drug']]
numerical_missing = [col for col in missing_cols if col not in categorical_missing]

# Impute missing categorical values with the mode
cat_imputer = SimpleImputer(strategy='most_frequent')
test_cleaned[categorical_missing] = cat_imputer.fit_transform(test_cleaned[categorical_missing])

# Impute missing numerical with median
num_imputer = SimpleImputer(strategy='median')
test_cleaned[numerical_missing] = num_imputer.fit_transform(test_cleaned[numerical_missing])

# Check all have been imputed successfully
print("\nMissing values after Imputation:")
print(test_cleaned.isnull().sum().sort_values(ascending=False))


# Adjust skew and normalise
categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']
numerical_cols = ['N_Days', 'Age', 'Bilirubin', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 
                  'Platelets', 'Prothrombin']

test_transformed = skewness_processor.transform(test_cleaned)
feature_names = skewness_processor.get_feature_names_out()

# Convert to DataFrame for better readability
test_transformed = pd.DataFrame(test_transformed, columns=feature_names, index=test_cleaned.index)

print("Transformed feature names:", feature_names)
print("Transformed data shape:", test_transformed.shape)


y_pred_probabilities = final_model.predict_proba(test_transformed)
class_order = final_model.classes_
class_order


# prepare dataframe for submission
results = pd.DataFrame({
    'id': test_transformed['remainder__id'].astype(int),
    'Status_C': y_pred_probabilities[:, 0],
    'Status_CL': y_pred_probabilities[:, 1], 
    'Status_D': y_pred_probabilities[:, 2]
})
results


results.to_csv('../Data/submission.csv', index=False)




