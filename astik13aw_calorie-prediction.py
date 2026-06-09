import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.stats as stats
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_log_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.decomposition import PCA


df_trains=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


df_train=df_trains.copy()


df_train


df_train.describe()


df_train.info()


df_train=df_train.drop('id',axis=1)


numerical_columns=df_train.select_dtypes(include=['int','float']).columns
for i in numerical_columns:

    sns.boxplot(df_train[i])
    plt.title(f'Boxplot of {i}')
    plt.show()
    


correlation= df_train.select_dtypes(include=['int','float']).corr()
print(correlation)


sns.heatmap(correlation,annot=True)


plt.figure(figsize=(10,6))
sns.histplot(data=df_train['Calories'],bins=30,kde=True)
plt.title('Histogram of Calories Burned During Workouts',fontsize=14,pad=15)
plt.xlabel('Calories Burned Kcal')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10,6))
sns.histplot(data=df_train['Duration'],bins=30,kde=True)
plt.title('Histogram of Duratio During Workouts',fontsize=14,pad=15)
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10,6))
sns.histplot(data=df_train['Age'],bins=30,kde=True)
plt.title('Histogram of Age During Workouts',fontsize=14,pad=15)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


indipendi_column=['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',]
fig,axes=plt.subplots(3,2,figsize=(12,18))
axes=axes.flatten()
for i ,feature in enumerate(indipendi_column):
    sns.scatterplot(data=df_train,x=feature ,y='Calories',hue='Sex',palette='deep',ax=axes[i])
    axes[i].set_title(f'Calories vs. {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Calories Bured (kcal)')
    axes[i].grid(True,alpha=0.3)
    axes[i].legend(title='Sex')
plt.tight_layout()
plt.show()


df=df_train.copy()
# Convert Age to categorical (Young, Middle, Old)
bins = [0, 30, 50, float('inf')]  # Age ranges: <30, 30-50, >50
labels = ['Young', 'Middle', 'Old']
df['Age_Category'] = pd.cut(df_train['Age'], bins=bins, labels=labels, include_lowest=True).astype(str)


# Verify the new column
print("Age Category Distribution:")
print(df['Age_Category'].value_counts())


# Define numerical features to plot against Calories (including original Age)
numerical_features = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'Weight', 'Height']

# Set up the plot grid (3 rows, 2 columns for 6 features)
fig, axes = plt.subplots(3, 2, figsize=(12, 18))
axes = axes.flatten()  # Flatten to iterate over axes

# Create scatter plots for each numerical feature vs. Calories, colored by Sex
for i, feature in enumerate(numerical_features):
    sns.scatterplot(data=df, x=feature, y='Calories', hue='Sex', palette={'male': 'blue', 'female': 'orange'}, 
                    ax=axes[i], alpha=0.6, s=50)
    axes[i].set_title(f'Calories vs. {feature} (Corr: {df[feature].corr(df["Calories"]):.3f})', fontsize=12)
    axes[i].set_xlabel(feature, fontsize=10)
    axes[i].set_ylabel('Calories Burned (kcal)', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(title='Sex')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()


indipendi_column=['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',]
fig,axes=plt.subplots(3,2,figsize=(12,18))
axes=axes.flatten()
for i ,feature in enumerate(indipendi_column):
    sns.scatterplot(data=df,x=feature ,y='Calories',hue='Age_Category',palette='deep',ax=axes[i])
    axes[i].set_title(f'Calories vs. {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Calories Bured (kcal)')
    axes[i].grid(True,alpha=0.3)
    axes[i].legend(title='Age_Category')
plt.tight_layout()
plt.show()


# Preprocessing
# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # 'M'/'F' to 0/1
#test_df['Sex'] = le.transform(test_df['Sex'])

# One-hot encode Age_Category
df = pd.get_dummies(df, columns=['Age_Category'], prefix='Age')
#test_df = pd.get_dummies(test_df, columns=['Age_Category'], prefix='Age')


# Define features and target
features = ['Sex',  'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 
            'Age_Young', 'Age_Middle', 'Age_Old']
X = df[features]
y = df['Calories']
#X_test = test_df[features]


# Scale numerical features
scaler = StandardScaler()
numerical_cols = [ 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
#X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])





# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Custom RMSLE scorer
def rmsle_scorer(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)  # Clip predictions to avoid negative/zero values
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


import time
# Initialize models
# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': lgb.LGBMRegressor(
        objective='regression',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        device_type='cpu',  # Fallback to CPU
        random_state=42,
        n_jobs=-1
    )
}

# Ensemble: Voting Regressor
ensemble = VotingRegressor(
    estimators=[
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting']),
        ('lgb', models['LightGBM'])
    ]
)
models['Ensemble (RF+GB+LGBM)'] = ensemble

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model_start_time = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, None)
    
    rmsle = rmsle_scorer(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    results[name] = {'RMSLE': rmsle, 'R2': r2}
    print(f"{name} RMSLE: {rmsle:.4f}")
    print(f"{name} R²: {r2:.4f}")
    print(f"{name} Time: {time.time() - model_start_time:.2f} seconds")

# Print results
print("\nModel Comparison:")
print("Model\t\t\tRMSLE\t\tR²")
for name, metrics in results.items():
    print(f"{name:<20}\t{metrics['RMSLE']:.4f}\t\t{metrics['R2']:.4f}")

# Best model (based on RMSLE)
best_model = min(results, key=lambda x: results[x]['RMSLE'])
print(f"\nBest Model (by RMSLE): {best_model} (RMSLE: {results[best_model]['RMSLE']:.4f}, R²: {results[best_model]['R2']:.4f})")

# Feature importance for tree-based models
for name in ['Random Forest', 'Gradient Boosting', 'LightGBM']:
    if name in models:
        print(f"\nFeature Importance for {name}:")
        importance = pd.Series(models[name].feature_importances_, index=features).sort_values(ascending=False)
        print(importance)



train_df=df.copy()



# Check for negative values in Calories
if (train_df['Calories'] < 0).any():
    print("Warning: Negative values found in Calories. Clipping to 0.")
    train_df['Calories'] = train_df['Calories'].clip(lower=0)


# Outlier capping
train_df = train_df[train_df['Heart_Rate'].between(40, 200)]
train_df = train_df[train_df['Body_Temp'].between(36, 42)]
train_df = train_df[train_df['Calories'] < 1000]


# Feature engineering
train_df['BMI'] = train_df['Weight'] / (train_df['Height'] / 100) ** 2
train_df['Duration_Heart_Rate'] = train_df['Duration'] * train_df['Heart_Rate']
train_df['Weight_Duration'] = train_df['Weight'] * train_df['Duration']
train_df['Age_Duration'] = train_df['Age'] * train_df['Duration']
train_df['Age_Body_Temp'] = train_df['Age'] * train_df['Body_Temp']
train_df['Body_Temp_Duration'] = train_df['Body_Temp'] * train_df['Duration']
train_df['Body_Temp_Heart_Rate'] = train_df['Body_Temp'] * train_df['Heart_Rate']
train_df['Age_Heart_Rate'] = train_df['Age'] * train_df['Heart_Rate']
train_df['Body_Temp_Heart_Rate_Duration'] = train_df['Body_Temp'] * train_df['Heart_Rate'] * train_df['Duration']


# Create Age_Category
bins = [0, 30, 50, float('inf')]
labels = ['Young', 'Middle', 'Old']
train_df['Age_Category'] = pd.cut(train_df['Age'], bins=bins, labels=labels, include_lowest=True).astype(str)
train_df['Age_Old_Duration'] = train_df['Age_Category'].apply(lambda x: 1 if x == 'Old' else 0) * train_df['Duration']


# Preprocessing
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df['Age_Middle'] = le.fit_transform(train_df['Age_Middle'])
train_df[ 'Age_Old'] = le.fit_transform(train_df['Age_Old'])
train_df[ 'Age_Young'] = le.fit_transform(train_df[ 'Age_Young'])



train_df.columns



# Define base features
base_features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
                     'Age_Middle', 'Age_Old', 'Age_Young', 'BMI',
                   'Duration_Heart_Rate', 'Weight_Duration', 'Age_Duration',
                   'Age_Body_Temp', 'Body_Temp_Duration', 'Body_Temp_Heart_Rate',
                   'Age_Heart_Rate', 'Body_Temp_Heart_Rate_Duration']
X = train_df[base_features]
y = train_df['Calories']


# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import RobustScaler, LabelEncoder
# Scale numerical features with RobustScaler
scaler = RobustScaler(quantile_range=(10.0, 90.0))
numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 
                  'BMI', 'Duration_Heart_Rate', 'Weight_Duration', 'Age_Duration', 
                  'Age_Body_Temp', 'Body_Temp_Duration', 'Body_Temp_Heart_Rate', 
                  'Age_Heart_Rate', 'Body_Temp_Heart_Rate_Duration']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])


from sklearn.metrics import mean_squared_log_error, r2_score, make_scorer
# Custom RMSLE scorer
def rmsle_scorer(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

custom_rmsle = make_scorer(rmsle_scorer, greater_is_better=False)


# Apply PCA on numerical features
pca = PCA(n_components=5)  # Fixed 5 components for simplicity
X_train_pca = pca.fit_transform(X_train[numerical_cols])
X_val_pca = pca.transform(X_val[numerical_cols])
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")


# Add PCA components to feature set
pca_features = [f'PCA_{i}' for i in range(X_train_pca.shape[1])]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_features, index=X_train.index)
X_val_pca_df = pd.DataFrame(X_val_pca, columns=pca_features, index=X_val.index)
X_train = pd.concat([X_train, X_train_pca_df], axis=1)
X_val = pd.concat([X_val, X_val_pca_df], axis=1)
features = base_features + pca_features


X_train=X_train.drop(['Age','Age_Body_Temp','BMI','Age_Middle','Age_Old','Age_Young'],axis=1)
X_val=X_val.drop(['Age','Age_Body_Temp','BMI','Age_Middle','Age_Old','Age_Young'],axis=1)




models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': lgb.LGBMRegressor(
        objective='regression',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        device_type='cpu',  # Fallback to CPU
        random_state=42,
        n_jobs=-1
    )
}

# Ensemble: Voting Regressor
ensemble = VotingRegressor(
    estimators=[
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting']),
        ('lgb', models['LightGBM'])
    ]
)
models['Ensemble (RF+GB+LGBM)'] = ensemble

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model_start_time = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, None)
    
    rmsle = rmsle_scorer(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    results[name] = {'RMSLE': rmsle, 'R2': r2}
    print(f"{name} RMSLE: {rmsle:.4f}")
    print(f"{name} R²: {r2:.4f}")
    print(f"{name} Time: {time.time() - model_start_time:.2f} seconds")

# Print results
print("\nModel Comparison:")
print("Model\t\t\tRMSLE\t\tR²")
for name, metrics in results.items():
    print(f"{name:<12}\t{metrics['RMSLE']:.4f}\t\t{metrics['R2']:.4f}")

# Best model (based on RMSLE)
best_model = min(results, key=lambda x: results[x]['RMSLE'])
print(f"\nBest Model (by RMSLE): {best_model} (RMSLE: {results[best_model]['RMSLE']:.4f}, R²: {results[best_model]['R2']:.4f})")

# Feature importance for tree-based models
for name in ['Random Forest', 'Gradient Boosting', 'LightGBM']:
    if name in models:
        print(f"\nFeature Importance for {name}:")
        importance = pd.Series(models[name].feature_importances_, index=features).sort_values(ascending=False)
        print(importance)


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
from itertools import combinations
# Greedy Sequential Feature Elimination
print("\nGreedy Sequential Feature Elimination...")
current_features = base_features.copy()
best_rmse = float('inf')
best_r2 = -float('inf')
best_subset = current_features.copy()
iteration = 0

while len(current_features) > 1:
    iteration += 1
    print(f"\nIteration {iteration}: Testing {len(current_features)} features")
    subset_results = []
    
    # Test dropping each feature
    for feature in current_features:
        test_features = [f for f in current_features if f != feature]
        model = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=100,learning_rate=0.1,max_depth=6,tree_method='hist',device='cuda', random_state=42,n_jobs=-1)
        model.fit(X_train[test_features], y_train)
        y_pred = model.predict(X_val[test_features])
        subset_rmse = rmsle_scorer(y_val, y_pred)
        subset_r2 = r2_score(y_val, y_pred)
        subset_results.append((test_features, subset_rmse, subset_r2))
    
    # Find best subset in this iteration
    subset_results.sort(key=lambda x: x[1])  # Sort by RMSE
    best_iter_features, best_iter_rmse, best_iter_r2 = subset_results[0]
    
    if best_iter_rmse < best_rmse:
        best_rmse = best_iter_rmse
        best_r2 = best_iter_r2
        best_subset = best_iter_features.copy()
        current_features = best_iter_features.copy()
        print(f"Dropped feature: RMSE = {best_rmse:.4f}, R² = {best_r2:.4f}, Features: {best_subset}")
    else:
        print(f"No improvement. Stopping at {len(current_features)} features.")
        break

# Random Subset Sampling (1,000 random subsets)
print("\nRandom Subset Sampling (1,000 subsets)...")
n_samples = 1000
random_results = []
all_features = base_features.copy()

for _ in range(n_samples):
    k = random.randint(1, len(all_features))  # Random subset size
    subset = random.sample(all_features, k)
    model = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=100,learning_rate=0.1,max_depth=6,tree_method='hist',device='cuda', random_state=42,n_jobs=-1)
    model.fit(X_train[subset], y_train)
    y_pred = model.predict(X_val[subset])
    subset_rmse = rmsle_scorer(y_val, y_pred)
    subset_r2 = r2_score(y_val, y_pred)
    random_results.append((subset, subset_rmse, subset_r2))
    print(f'Rmse- ',subset_rmse,' R2- ',subset_r2)

# Find best random subset
random_results.sort(key=lambda x: x[1])  # Sort by RMSE
best_random_subset, best_random_rmse, best_random_r2 = random_results[0]

# Compare greedy and random results
if best_random_rmse < best_rmse:
    best_subset = best_random_subset
    best_rmse = best_random_rmse
    best_r2 = best_random_r2
    print(f"\nRandom Subset Outperformed Greedy:")
    print(f"Best Subset: {best_subset}")
    print(f"RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}")
else:
    print(f"\nGreedy Subset Outperformed Random:")
    print(f"Best Subset: {best_subset}")
    print(f"RMSE: {best_rmse:.4f}, R²: {best_r2:.4f}")

# Final model evaluation
print("\nFinal Linear Regression Results:")
model = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=100,learning_rate=0.1,max_depth=6,tree_method='hist',device='cuda', random_state=42,n_jobs=-1)
model.fit(X_train[best_subset], y_train)
y_pred = model.predict(X_val[best_subset])
final_rmse = rmsle_scorer(y_val, y_pred)
final_r2 = r2_score(y_val, y_pred)
print(f"RMSE: {final_rmse:.4f}")
print(f"R²: {final_r2:.4f}")

# Feature coefficients
print("\nFeature Coefficients:")
coefficients = pd.Series(model.coef_, index=best_subset).sort_values(ascending=False)
print(coefficients)

# Summary statistics for Age_Category
print("\nAverage Calories Burned by Age Category:")
print(train_df.groupby('Age_Category')['Calories'].mean())
print("\nAverage Calorie-Burning Rate (kcal/min) by Age Category:")
train_df['Calorie_Rate'] = train_df['Calories'] / train_df['Duration'].replace(0, float('nan'))
print(train_df.groupby('Age_Category')['Calorie_Rate'].mean())
print("\nAverage Heart Rate by Age Category:")
print(train_df.groupby('Age_Category')['Heart_Rate'].mean())

# Print total time
print(f"\nTotal Time: {time.time() - start_time:.2f} seconds")


best_feture=['Sex', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Age_Middle', 'Age_Old', 'Age_Young', 'Duration_Heart_Rate', 'Weight_Duration', 'Age_Duration', 'Body_Temp_Duration', 'Body_Temp_Heart_Rate', 'Age_Heart_Rate', 'Body_Temp_Heart_Rate_Duration']
X = train_df[best_feture]
y = train_df['Calories']


# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import RobustScaler, LabelEncoder
# Scale numerical features with RobustScaler
scaler = RobustScaler(quantile_range=(10.0, 90.0))
numerical_cols =[  'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Age_Middle', 'Age_Old', 'Age_Young', 'Duration_Heart_Rate', 'Weight_Duration', 'Age_Duration', 'Body_Temp_Duration', 'Body_Temp_Heart_Rate', 'Age_Heart_Rate', 'Body_Temp_Heart_Rate_Duration']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])


# Apply PCA on numerical features
pca = PCA(n_components=5)  # Fixed 5 components for simplicity
X_train_pca = pca.fit_transform(X_train[numerical_cols])
X_val_pca = pca.transform(X_val[numerical_cols])
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")


# Add PCA components to feature set
pca_features = [f'PCA_{i}' for i in range(X_train_pca.shape[1])]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_features, index=X_train.index)
X_val_pca_df = pd.DataFrame(X_val_pca, columns=pca_features, index=X_val.index)
X_train = pd.concat([X_train, X_train_pca_df], axis=1)
X_val = pd.concat([X_val, X_val_pca_df], axis=1)
features = base_features + pca_features




# Configure XGBoost for GPU
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    tree_method='hist',  # Use histogram-based method, optimized for GPU
    device='cuda',       # Explicitly set to GPU (CUDA)
    random_state=42,
    n_jobs=-1
)


# Train the model
xgb_model.fit(X_train, y_train)

# Predict on validation set
y_pred = xgb_model.predict(X_val)
y_pred = np.clip(y_pred, 0, None)  # Ensure non-negative predictions for RMSLE


# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
print(f"Validation RMSLE: {rmsle:.4f}")


# Configure XGBoost base model for GPU
base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',  # Histogram-based method for GPU
    device='cuda',       # Use CUDA for P100
    random_state=42,
    n_jobs=1             # Set to 1 to avoid GPU conflicts
)

# Define comprehensive parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    
   
}

# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_log_error',
    cv=3,  # 3-fold cross-validation
    verbose=2,
    n_jobs=1  # Set to 1 to avoid GPU conflicts
)

# Train the model with GridSearchCV
grid_search.fit(X_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation RMSLE:", np.sqrt(-grid_search.best_score_))

# Predict on validation set
y_pred = best_model.predict(X_val)
y_pred = np.clip(y_pred, 0, None)  # Ensure non-negative predictions for RMSLE

# Calculate validation RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
print(f"Validation RMSLE: {rmsle:.4f}")


df_test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


test_df=df_test.copy()


test_df


test_df['BMI'] = test_df['Weight'] / (test_df['Height'] / 100) ** 2
test_df['Duration_Heart_Rate'] = test_df['Duration'] * test_df['Heart_Rate']
test_df['Weight_Duration'] = test_df['Weight'] * test_df['Duration']
test_df['Age_Duration'] = test_df['Age'] * test_df['Duration']
test_df['Age_Body_Temp'] = test_df['Age'] * test_df['Body_Temp']
test_df['Body_Temp_Duration'] = test_df['Body_Temp'] * test_df['Duration']
test_df['Body_Temp_Heart_Rate'] = test_df['Body_Temp'] * test_df['Heart_Rate']
test_df['Age_Heart_Rate'] = test_df['Age'] * test_df['Heart_Rate']
test_df['Body_Temp_Heart_Rate_Duration'] = test_df['Body_Temp'] * test_df['Heart_Rate'] * test_df['Duration']


# Create Age_Category
bins = [0, 30, 50, float('inf')]
labels = ['Young', 'Middle', 'Old']
test_df['Age_Category'] = pd.cut(test_df['Age'], bins=bins, labels=labels, include_lowest=True).astype(str)
test_df['Age_Old_Duration'] = test_df['Age_Category'].apply(lambda x: 1 if x == 'Old' else 0) * test_df['Duration']


test_df = pd.get_dummies(test_df, columns=['Age_Category'], prefix='Age')


# Preprocessing
le = LabelEncoder()
test_df['Sex'] = le.fit_transform(test_df['Sex'])
test_df['Age_Middle'] = le.fit_transform(test_df['Age_Middle'])
test_df[ 'Age_Old'] = le.fit_transform(test_df['Age_Old'])
test_df[ 'Age_Young'] = le.fit_transform(test_df[ 'Age_Young'])


best_feture=['Sex', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Age_Middle', 'Age_Old', 'Age_Young', 'Duration_Heart_Rate', 'Weight_Duration', 'Age_Duration', 'Body_Temp_Duration', 'Body_Temp_Heart_Rate', 'Age_Heart_Rate', 'Body_Temp_Heart_Rate_Duration']
test_df= test_df[best_feture]



from sklearn.preprocessing import RobustScaler, LabelEncoder
# Scale numerical features with RobustScaler
scaler = RobustScaler(quantile_range=(10.0, 90.0))
numerical_cols =[  'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Age_Middle', 'Age_Old', 'Age_Young', 'Duration_Heart_Rate', 'Weight_Duration', 'Age_Duration', 'Body_Temp_Duration', 'Body_Temp_Heart_Rate', 'Age_Heart_Rate', 'Body_Temp_Heart_Rate_Duration']
test_df[numerical_cols] = scaler.fit_transform(test_df[numerical_cols])



# Apply PCA on numerical features
pca = PCA(n_components=5)  # Fixed 5 components for simplicity
test_df_pca = pca.fit_transform(test_df[numerical_cols])

print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")


test_df_pca


# Add PCA components to feature set
pca_features = [f'PCA_{i}' for i in range(test_df_pca.shape[1])]
test_df_pca_df = pd.DataFrame(test_df_pca, columns=pca_features, index=test_df.index)

test_df = pd.concat([test_df, test_df_pca_df], axis=1)

features = base_features + pca_features


# Configure XGBoost for GPU
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    tree_method='hist',  # Use histogram-based method, optimized for GPU
    device='cuda',       # Explicitly set to GPU (CUDA)
    random_state=42,
    n_jobs=-1
)
# Train the model
xgb_model.fit(X_train, y_train)

# Predict on validation set
y_pred = xgb_model.predict(X_val)
y_pred = np.clip(y_pred, 0, None)  # Ensure non-negative predictions for RMSLE
final_rmse = rmsle_scorer(y_val, y_pred)
final_r2 = r2_score(y_val, y_pred)
print(f"RMSE: {final_rmse:.4f}")
print(f"R²: {final_r2:.4f}")


test_pred = xgb_model.predict(test_df)


test_pred


df = pd.DataFrame(test_pred)


submission = pd.DataFrame({'id': df_test['id'], 'Calories': test_pred})
submission.to_csv('submission_lgbm.csv', index=False)


df.to_excel('test_pred.xlsx', index=False)







