import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb
import time
import warnings
warnings.filterwarnings('ignore')
import os
import joblib
import re


train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')


train.head()



train.shape


train.columns


test.columns


train.dtypes


cat_cols = ['balcony_exist', 'elevator', 'cellar_exist', 'furnished', 'barrier_free', 'lodge_exist', 'terace_exist'
           , 'garage_exist']

num_cols = ['price', 'area', 'floor', 'area_floor', 'balcony', 'cellar', 'lodge', 'terace', 'area_built_up', 
            'area_garden', 'garage', 'latitude', 'longitude']

cat_for_encoding = ['flat_type', 'county', 'ownership']


# Visualise numerical columns
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.ravel()  # Flatten the 2D array of axes

# Create histograms for each numerical column
for i, col in enumerate(num_cols):
    if col in train.columns:
        data = train[col]
        if len(data) > 0:
            axes[i].hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[i].set_title(f'Histogram of {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
                
            # Add basic statistics as text
            stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nN: {len(data)}'
            axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes, 
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           fontsize=9)
        else:
            axes[i].text(0.5, 0.5, f'No data for {col}', 
                        ha='center', va='center', transform=axes[i].transAxes)
    else:
        axes[i].text(0.5, 0.5, f'Column {col} not found', 
                   ha='center', va='center', transform=axes[i].transAxes)


# Visualise categorical columns
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()  # Flatten the 2D array of axes

# Define colors for the pie charts
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0', '#ffb3e6', '#c4e17f']

# Create pie charts for each categorical column
for i, col in enumerate(cat_cols):
    if col in train.columns:
        value_counts = train[col].value_counts()
        
        if len(value_counts) > 0:
            labels = [f'{idx} ({val})' for idx, val in value_counts.items()]
            values = value_counts.values
            
            # Create pie chart
            wedges, texts, autotexts = axes[i].pie(values, labels=labels, autopct='%1.1f%%', 
                                                  colors=colors[:len(values)], startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            axes[i].set_title(f'{col}\n(Total: {value_counts.sum()})', fontsize=12, fontweight='bold', pad=20)
            
            # Add legend if there are many categories
            if len(value_counts) > 2:
                axes[i].legend(wedges, [f'{k}: {v}' for k, v in value_counts.items()],
                              title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                
        else:
            axes[i].text(0.5, 0.5, f'No data for {col}', 
                       ha='center', va='center', transform=axes[i].transAxes,
                       fontsize=12, fontweight='bold')
    else:
        axes[i].text(0.5, 0.5, f'Column {col} not found', 
                   ha='center', va='center', transform=axes[i].transAxes,
                   fontsize=12, fontweight='bold')

plt.tight_layout()
plt.suptitle('Distribution of Binary/Categorical Features', fontsize=16, fontweight='bold', y=1.02)
plt.show()


# Examine the relationship between location and price
required_cols = ['latitude', 'longitude', 'price']
plot_data = train[required_cols]

plt.figure(figsize=(12, 8))
        
# Create scatter plot 
scatter = plt.scatter(plot_data['longitude'], plot_data['latitude'], 
                        c=plot_data['price'], 
                        cmap='Greys', 
                        alpha=0.6, 
                        s=30,  # point size
                        edgecolors='w', 
                        linewidth=0.5)
        
# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Price', fontsize=12)
        
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Geographic Distribution of Properties Colored by Price', fontsize=14, fontweight='bold')
        
plt.grid(True, alpha=0.3) # add grid
        
# Add some statistics to the plot
plt.text(0.02, 0.98, f'N = {len(plot_data)} properties\n'
                        f'Price range: {plot_data["price"].min():,.0f} - {plot_data["price"].max():,.0f}',
        transform=plt.gca().transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10)
        
plt.tight_layout()
plt.show()


# examine the qualitative categories of interest to see if they show any major differences in price.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Create boxplot for each categorical column
for i, col in enumerate(cat_for_encoding):
    if col in train.columns and 'price' in train.columns:
        plot_data = train[[col, 'price']]
        
        if len(plot_data) > 0:
            # Get value counts s
            value_counts = plot_data[col].value_counts()
            
            # If too many categories, show only top N
            if len(value_counts) > 10:
                top_categories = value_counts.head(10).index
                plot_data = plot_data[plot_data[col].isin(top_categories)]
                title_suffix = f"\n(Top 10 categories only)"
            else:
                title_suffix = ""
            
            # Create boxplot
            sns.boxplot(x=col, y='price', data=plot_data, ax=axes[i])
            
            # Customize the plot
            axes[i].set_title(f'Price by {col}{title_suffix}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Price', fontsize=12)
            
            # Rotate x-axis labels if they're long
            if plot_data[col].nunique() > 3:
                axes[i].tick_params(axis='x', rotation=45)
            
            # Add number of observations to each box
            for j, category in enumerate(plot_data[col].unique()):
                count = len(plot_data[plot_data[col] == category])
                axes[i].text(j, plot_data['price'].max() * 1.02, f'n={count}', 
                           ha='center', va='bottom', fontsize=9)
            
        else:
            axes[i].text(0.5, 0.5, f'No data for {col}', 
                       ha='center', va='center', transform=axes[i].transAxes,
                       fontsize=12)
    else:
        axes[i].text(0.5, 0.5, f'Column {col} or price not found', 
                   ha='center', va='center', transform=axes[i].transAxes,
                   fontsize=12)

plt.tight_layout()
plt.suptitle('Price Distribution by Categorical Variables', fontsize=16, fontweight='bold', y=1.05)
plt.show()


# Run a correlation analysis with the numerical variables to see if there are any highly associated variables

correlation_data = train[num_cols].copy()
correlation_data['price'] = train['price'].copy()

# Calculate correlation matrix
corr_matrix = correlation_data.corr()

price_correlations = corr_matrix['price'].sort_values(ascending=False)

print("\n" + "="*60)
print("CORRELATION WITH PRICE (Sorted by strength)")
print("="*60)
for feature, correlation in price_correlations.items():
    if feature != 'price':  # Skip self-correlation
        print(f"{feature:15}: {correlation:7.3f}")

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Create bar plot of correlations with price
plt.figure(figsize=(12, 6))
price_correlations_without_self = price_correlations.drop('price', errors='ignore')
price_correlations_without_self.sort_values(ascending=False).plot(kind='bar', color='steelblue')
plt.title('Correlation Coefficients with Price', fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()



# Examine percent of missing values for train data
missing_summary = pd.DataFrame({
    'Missing_Count': train.isnull().sum(),
    'Missing_Percent': (train.isnull().sum() / len(train)) * 100,
    'Data_Type': train.dtypes
})

missing_summary = missing_summary.sort_values('Missing_Percent', ascending=False)

print("Missing Values Summary:")
print(missing_summary)


# Examine percent of missing values for test data
missing_summary = pd.DataFrame({
    'Missing_Count': test.isnull().sum(),
    'Missing_Percent': (test.isnull().sum() / len(test)) * 100,
    'Data_Type': test.dtypes
})

missing_summary = missing_summary.sort_values('Missing_Percent', ascending=False)

print("Missing Values Summary:")
print(missing_summary)


# Remove columns with a high amount of missingness (70% or more)

columns_to_drop = ['starting_price', 'area_built_up',  'flat_type', 'area_garden', 'terace', 'lodge', 'garage',
    'garage_exist', 'terace_exist', 'balcony', 'lodge_exist', 'balcony_exist', 'barrier_free']

train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)



# Remove columns that are not necessary or helpful
train = train.drop(columns=['price_note', 'download_date', 'address'])
test = test.drop(columns=['price_note', 'download_date', 'address'])


# Update Numerical and categorical columns
cat_cols = ['elevator', 'cellar_exist', 'furnished']

num_cols = ['price', 'area', 'floor', 'area_floor', 'cellar', 'latitude', 'longitude']

cat_for_encoding = ['county', 'ownership']





# Re show missing values after removal and encoding
missing_summary = pd.DataFrame({
    'Missing_Count': train.isnull().sum(),
    'Missing_Percent': (train.isnull().sum() / len(train)) * 100,
    'Data_Type': train.dtypes
})

missing_summary = missing_summary.sort_values('Missing_Percent', ascending=False)

print("Missing Values Summary:")
print(missing_summary)


# Impute missing values - for categorical impute with mode, for numerical impute with median

categorical_cols = train.select_dtypes(include=['object']).columns
categorical_missing = [col for col in categorical_cols if train[col].isnull().sum() > 0]

for col in categorical_missing:
    # For energy_performance and street use 'Unknown' category
    if col == 'energy_performance' or col == 'street':
        train[col] = train[col].fillna('Unknown')
        test[col] = test[col].fillna('Unknown')
        print(f"Filled '{col}' missing values with 'Unknown'")
        
    # For other categorical columns with low missingness - use mode
    else:
        mode_val = train[col].mode()[0] if not train[col].mode().empty else 'Unknown'
        train[col] = train[col].fillna(mode_val)
        test[col] = test[col].fillna(mode_val)
        print(f"Imputed '{col}' with mode: {mode_val}")
            
numerical_cols = train.select_dtypes(include=[np.number]).columns
numerical_missing = [col for col in numerical_cols if train[col].isnull().sum() > 0]
    
for col in numerical_missing:
    # Use median for numerical columns
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)
    print(f"Imputed '{col}' with median: {median_val:.2f}")


# Re show missing values after filling
missing_summary = pd.DataFrame({
    'Missing_Count': train.isnull().sum(),
    'Missing_Percent': (train.isnull().sum() / len(train)) * 100,
    'Data_Type': train.dtypes
})

missing_summary = missing_summary.sort_values('Missing_Percent', ascending=False)

print("Missing Values Summary:")
print(missing_summary)


# Encode categorical features with text values so we can make predictions
def encode_categorical_features(train_df, test_df=None):
    """
    Encode categorical features with appropriate strategies for each column
    Fits ONLY on training data to avoid data leakage
    """
    # Make copies to avoid modifying original data
    train = train_df.copy()
    test = test_df.copy()
    
    print("Encoding categorical features (fitting only on training data)...")
    
    # First, simplify the energy_performance values
    def simplify_energy_performance(value):
        if value == 'Unknown':
            return 'Unknown'
        
        # Extract the energy class (A, B, C, D, E, F, G)
        match = re.search(r'Třída ([A-G])', value)
        if match:
            energy_class = match.group(1)
            return f'Class {energy_class}'
        return 'Unknown'
    
    # Apply simplification
    train['energy_performance_simple'] = train['energy_performance'].apply(simplify_energy_performance)
    test['energy_performance_simple'] = test['energy_performance'].apply(simplify_energy_performance)
    
    print("Simplified energy performance values:")
    print(train['energy_performance_simple'].value_counts())
    
    # Define the natural order for ordinal columns
    ordinal_orders = {
        'energy_performance_simple': [
            'Unknown',     # Unknown (lowest priority)
            'Class G',     # Worst - Mimořádně nehospodárná
            'Class F',     # Velmi nehospodárná
            'Class E',     # Nehospodárná
            'Class D',     # Méně úsporná
            'Class C',     # Úsporná
            'Class B',     # Velmi úsporná
            'Class A'      # Best - Mimořádně úsporná
        ],
        'building_state': [
            'K demolici',      # Worst - For demolition
            'Špatný',          # Bad
            'Před rekonstrukcí', # Before reconstruction
            'Dobrý',           # Good
            'Velmi dobrý',     # Very good
            'Po rekonstrukci', # After reconstruction
            'Projekt',         # Project
            'Ve výstavbě',     # Under construction
            'Novostavba'       # Best - New building
        ]
    }
    
    # Store encoders for future use
    encoders = {}
    
    # Encode each categorical column - FIT ONLY ON TRAINING DATA
    categorical_columns = ['county', 'ownership', 'district', 'energy_performance', 'building_state']
    
    for col in categorical_columns:
        if col in train.columns or (col == 'energy_performance' and 'energy_performance_simple' in train.columns):
            print(f"\nEncoding '{col}'...")
            
            # Use simplified version for energy performance
            encode_col = 'energy_performance_simple' if col == 'energy_performance' else col
            
            if col in ['county', 'ownership', 'district']:
                # Label encoding for nominal categories - FIT ON TRAIN ONLY
                encoder = LabelEncoder()
                encoder.fit(train[encode_col])
                
                # Transform both datasets
                train[f'{col}_encoded'] = encoder.transform(train[encode_col])
                # For test data, handle unseen categories by mapping to -1
                test[f'{col}_encoded'] = test[encode_col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
                
                encoders[col] = encoder
                print(f"  Classes: {list(encoder.classes_)}")
                
            elif col in ['energy_performance', 'building_state']:
                # Ordinal encoding for categories with natural order - FIT ON TRAIN ONLY
                if encode_col in ordinal_orders:
                    custom_order = ordinal_orders[encode_col]
                    encoder = OrdinalEncoder(categories=[custom_order], dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1)
                    
                    # Fit on training data only
                    encoder.fit(train[[encode_col]])
                    
                    # Transform both datasets
                    train[f'{col}_encoded'] = encoder.transform(train[[encode_col]]).flatten()
                    if test_df is not None and encode_col in test.columns:
                        test[f'{col}_encoded'] = encoder.transform(test[[encode_col]]).flatten()
                    
                    encoders[col] = encoder
                    print(f"  Order: {custom_order}")
            
            # Show encoding results
            value_counts = train[encode_col].value_counts()
            print(f"  Training value counts: {len(value_counts)} categories")
            print(f"  Sample mapping:")
            for i, (category, count) in enumerate(value_counts.head(5).items()):
                encoded_val = train.loc[train[encode_col] == category, f'{col}_encoded'].iloc[0]
                print(f"    {category} -> {encoded_val} (count: {count})")
            
            # Check for unseen categories in test data
            if test_df is not None and encode_col in test.columns:
                unseen_categories = set(test[encode_col].unique()) - set(train[encode_col].unique())
                if unseen_categories:
                    print(f"  Warning: {len(unseen_categories)} unseen categories in test data: {list(unseen_categories)}")
    
    # Create one-hot encoding for nominal features - FIT ON TRAIN ONLY
    print("\nCreating one-hot encoding for nominal features...")
    nominal_cols = ['county', 'ownership', 'district']
    
    for col in nominal_cols:
        if col in train.columns:
            # Get all possible categories from TRAINING data only
            all_categories = sorted(train[col].unique())
            
            # Create one-hot encoded columns for training data
            train_dummies = pd.get_dummies(train[col], prefix=col)
            train = pd.concat([train, train_dummies], axis=1)
            
            if col in test.columns:
                # For test data, use the same categories as training
                test_dummies = pd.get_dummies(test[col], prefix=col)
                
                # Ensure test data has the same columns as training data
                missing_cols = set([f'{col}_{cat}' for cat in all_categories]) - set(test_dummies.columns)
                for missing_col in missing_cols:
                    test_dummies[missing_col] = 0
                
                # Reorder columns to match training data
                test_dummies = test_dummies[[f'{col}_{cat}' for cat in all_categories]]
                test = pd.concat([test, test_dummies], axis=1)
            
            print(f"  Created {len(all_categories)} one-hot columns for '{col}'")
    
    # Save encoders for future use
    joblib.dump(encoders, 'categorical_encoders.pkl')
    print(f"\nEncoders saved to 'categorical_encoders.pkl'")
    
    return (train, test, encoders) if test_df is not None else (train, encoders)

# Apply encoding to both datasets
train_encoded, test_encoded, encoders = encode_categorical_features(train, test)



# Create a location feature via clustering
n_clusters = min(20, len(train_encoded) // 100)

# Create location clusters
coords = train_encoded[['latitude', 'longitude']].values
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
train_encoded['location_cluster'] = kmeans.fit_predict(coords)

test_coords = test_encoded[['latitude', 'longitude']].values
test_encoded['location_cluster'] = kmeans.predict(test_coords)

print(f"Created {n_clusters} location clusters")


X['furnished'].unique()





# define columns to use for prediction and perform train-test split on the train data
features = ['floor', 'area', 'elevator', 'cellar_exist', 'county_encoded', 'district_encoded', 
            'energy_performance_encoded', 'building_state_encoded', 'location_cluster']

target = 'price'

X = train_encoded[features]
y = train_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


# Scale the features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for better readability
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

# Define evaluation metrics
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model performance with multiple metrics"""
    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    # Train and test predictions
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'CV_RMSE_Mean': cv_rmse.mean(),
        'CV_RMSE_Std': cv_rmse.std(),
        'Train_R2': r2_score(y_train, y_pred_train),
        'Test_R2': r2_score(y_test, y_pred_test),
        'Train_MAE': mean_absolute_error(y_train, y_pred_train),
        'Test_MAE': mean_absolute_error(y_test, y_pred_test),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    }
    
    return metrics, model

# Define individual models with their parameter grids
models = {}

# 1. Random Forest (doesn't need scaling)
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
models['RandomForest'] = {
    'model': RandomForestRegressor(random_state=42),
    'param_grid': rf_param_grid,
    'use_scaled': False
}

# 2. Gradient Boosting (doesn't need scaling)
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10]
}
models['GradientBoosting'] = {
    'model': GradientBoostingRegressor(random_state=42),
    'param_grid': gb_param_grid,
    'use_scaled': False
}

# 3. XGBoost (doesn't need scaling)
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
models['XGBoost'] = {
    'model': xgb.XGBRegressor(random_state=42, verbosity=0),
    'param_grid': xgb_param_grid,
    'use_scaled': False
}

# 4. LightGBM (doesn't need scaling)
lgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
models['LightGBM'] = {
    'model': lgb.LGBMRegressor(random_state=42, verbose=-1),
    'param_grid': lgb_param_grid,
    'use_scaled': False
}

# 5. Ridge Regression (needs scaling)
ridge_param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
}
models['Ridge'] = {
    'model': Pipeline([('scaler', StandardScaler()), ('model', Ridge(random_state=42))]),
    'param_grid': {'model__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]},
    'use_scaled': True
}

# 6. Lasso Regression (needs scaling)
lasso_param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}
models['Lasso'] = {
    'model': Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=42, max_iter=10000))]),
    'param_grid': {'model__alpha': [0.1, 1.0, 10.0, 100.0]},
    'use_scaled': True
}

# Hyperparameter tuning and evaluation
results = []
best_models = {}

print("Performing hyperparameter tuning for individual models...")
print("=" * 60)

for model_name, model_info in models.items():
    print(f"\nTuning {model_name}...")
    start_time = time.time()
    
    # Select appropriate data
    if model_info['use_scaled']:
        X_train_data = X_train
        X_test_data = X_test
    else:
        X_train_data = X_train
        X_test_data = X_test
    
    # Randomized search for hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=model_info['model'],
        param_distributions=model_info['param_grid'],
        n_iter=15,  # Reduced for faster execution
        scoring='neg_mean_squared_error',
        cv=3,  # Reduced for faster execution
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    # Fit the random search
    random_search.fit(X_train_data, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = np.sqrt(-random_search.best_score_)
    
    # Evaluate the best model
    metrics, trained_model = evaluate_model(
        best_model, X_train_data, y_train, X_test_data, y_test, model_name
    )
    
    metrics['Best_Params'] = best_params
    metrics['Best_CV_Score'] = best_score
    metrics['Training_Time'] = time.time() - start_time
    
    results.append(metrics)
    best_models[model_name] = trained_model
    
    print(f"  Best CV RMSE: {best_score:.2f}")
    print(f"  Test R2: {metrics['Test_R2']:.3f}")
    print(f"  Time: {metrics['Training_Time']:.1f}s")

# Create results dataframe
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("INDIVIDUAL MODEL PERFORMANCE")
print("=" * 60)
print(results_df[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'CV_RMSE_Mean']].round(3))

# Build ensemble models
print("\n" + "=" * 60)
print("BUILDING ENSEMBLE MODELS")
print("=" * 60)

# 1. Voting Regressor
top_models = {name: model for name, model in best_models.items() 
             if name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']}

voting_regressor = VotingRegressor(
    estimators=[(name, model) for name, model in top_models.items()],
    n_jobs=-1
)

voting_metrics, voting_model = evaluate_model(
    voting_regressor, X_train, y_train, X_test, y_test, "Voting_Ensemble"
)
results.append(voting_metrics)

# 2. Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=[(name, model) for name, model in top_models.items()],
    final_estimator=Ridge(),
    n_jobs=-1
)

stacking_metrics, stacking_model = evaluate_model(
    stacking_regressor, X_train, y_train, X_test, y_test, "Stacking_Ensemble"
)
results.append(stacking_metrics)

# Final results
final_results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

# Sort by Test_R2 score
final_results_df = final_results_df.sort_values('Test_R2', ascending=False)
print(final_results_df[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'CV_RMSE_Mean']].round(3))

# Feature importance from the best tree-based model
best_tree_model = best_models[final_results_df.iloc[0]['Model']]
if hasattr(best_tree_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_tree_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (from best model)")
    print("=" * 60)
    print(feature_importance)

# Save the best model
best_model_name = final_results_df.iloc[0]['Model']
best_model = best_models[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"Test R²: {final_results_df.iloc[0]['Test_R2']:.3f}")
print(f"Test RMSE: {final_results_df.iloc[0]['Test_RMSE']:.2f}")

# Predict on test set for analysis
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

print("\n" + "=" * 60)
print("PREDICTION ANALYSIS")
print("=" * 60)
print(f"Prediction range: {y_pred.min():.0f} - {y_pred.max():.0f}")
print(f"Actual range: {y_test.min():.0f} - {y_test.max():.0f}")

# Create prediction comparison
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Residual': residuals,
    'Error_Percentage': np.abs(residuals) / y_test * 100
})

print(f"\nAverage error percentage: {comparison_df['Error_Percentage'].mean():.1f}%")
print(f"Median error percentage: {comparison_df['Error_Percentage'].median():.1f}%")

# Return everything for further analysis
ensemble_results = {
    'results': final_results_df,
    'best_model': best_model,
    'best_model_name': best_model_name,
    'scaler': scaler,
    'predictions': comparison_df,
    'feature_importance': feature_importance if 'feature_importance' in locals() else None
}

print("\nEnsemble modeling completed successfully!")


# Plot the predicted vs actual price
plt.style.use('default')
sns.set_palette("husl")

# Create the scatter plot
plt.figure(figsize=(12, 10))

# Main scatter plot
plt.subplot(1, 2, 1)
scatter = plt.scatter(y_test, y_pred, alpha=0.6, s=50, c='steelblue', edgecolors='white', linewidth=0.5)

# Perfect prediction line
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

# Add statistics to the plot
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

plt.xlabel('Actual Price', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Price', fontsize=12, fontweight='bold')
plt.title(f'Actual vs Predicted Prices\nR² = {r2:.3f}, MAE = {mae:,.0f}, RMSE = {rmse:,.0f}', 
          fontsize=14, fontweight='bold')

plt.grid(True, alpha=0.3)
plt.legend()

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, s=50, c='coral', edgecolors='white', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Price', fontsize=12, fontweight='bold')
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add some statistics to residual plot
mean_residual = residuals.mean()
std_residual = residuals.std()
plt.text(0.05, 0.95, f'Mean residual: {mean_residual:,.0f}\nStd residual: {std_residual:,.0f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))




X_test_holdout = test_encoded[features].copy()

predictions = best_model.predict(X_test_holdout)

# Create the output dataframe
output_df = pd.DataFrame({
    'id': test_encoded['id'],
    'price': predictions})
if (output_df['price'] < 0).any():
    print("Warning: Some predicted prices are negative. Clipping to zero...")
    output_df['price'] = output_df['price'].clip(lower=0)

output_file = '../Data/test_predictions.csv'
output_df.to_csv(output_file, index=False)
print(f"\nPredictions saved to '{output_file}'")


output_df




