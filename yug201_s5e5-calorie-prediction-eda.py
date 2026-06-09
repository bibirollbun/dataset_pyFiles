import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df1 = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv') # Example if loading needed

print("--- Descriptive Statistics for Numerical Columns ---")
print(df1.describe())
print("\n")
print("--- Value Counts for 'Sex' Column ---")
if 'Sex' in df1.columns:
    print(df1['Sex'].value_counts())
else:
    print("'Sex' column not found in the DataFrame.")
print("\n") 

print("--- Missing Value Counts per Column ---")
print(df1.isnull().sum())
print("\n") 



numerical_cols = df1.select_dtypes(include=np.number).columns.tolist()

numerical_cols.remove('id')

print(f"Numerical columns selected for plotting: {numerical_cols}")

# Determine grid size for subplots
n_cols = len(numerical_cols)
n_rows = (n_cols + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
axes = axes.flatten()

# Generate histograms for each numerical column
for i, col in enumerate(numerical_cols):
    sns.histplot(df1[col], kde=True, ax=axes[i], bins=50)
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
image_filename = 'numerical_distributions.png'
plt.savefig(image_filename)
plt.show() 
print(f"Image saved as: {image_filename}")



numerical_cols.remove('Calories') 
print(f"Numerical features selected for scatter plots against Calories: {numerical_cols}")

# Determine grid size for subplots
n_cols_plot = len(numerical_cols)
n_rows_plot = (n_cols_plot + 2) // 3
fig, axes = plt.subplots(n_rows_plot, 3, figsize=(18, n_rows_plot * 5))
axes = axes.flatten()

# Generate scatter plots for each numerical feature against Calories
# Use a smaller sample or alpha blending for large datasets if performance/visibility is an issue
# Using alpha blending here for better visualization with 750k points
plot_alpha = 0.1 
for i, col in enumerate(numerical_cols):
    sns.scatterplot(data=df1, x=col, y='Calories', ax=axes[i], alpha=plot_alpha, s=5) # Added alpha and reduced marker size
    axes[i].set_title(f'Calories vs. {col}', fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Calories')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
image_filename = 'scatter_plots_vs_calories.png'
plt.savefig(image_filename)
plt.show() 
print(f"Image saved as: {image_filename}")



print(f"Numerical columns for correlation matrix: {numerical_cols}")

    # Calculate the correlation matrix
correlation_matrix = df1[numerical_cols].corr()

    # Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black') # annot=True displays values, fmt formats them, added lines for clarity
plt.title('Correlation Heatmap of Numerical Features', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

    # Save the heatmap image
image_filename = 'correlation_heatmap.png'
plt.savefig(image_filename)
plt.show()
print(f"Image saved as: {image_filename}")



image_filename = None

        # Plot the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df1, x='Sex', y='Calories')
plt.title('Calories Burned Distribution by Sex', fontsize=14)
plt.xlabel('Sex')
plt.ylabel('Calories Burned')
plt.tight_layout() # Adjust layout

        # Save the boxplot image
image_filename = 'calories_by_sex_boxplot.png'
plt.savefig(image_filename)
plt.show()
print(f"Image saved as: {image_filename}")



key_cols = ['Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Calories']

sample_size = 50000
if len(df1) > sample_size:
    print(f"Dataset is large ({len(df1)} rows). Creating pairplot based on a random sample of {sample_size} rows.")
    df1_sample = df1.sample(n=sample_size, random_state=42)
else:
    print(f"Dataset size ({len(df1)} rows) is manageable. Using full dataset for pairplot.")
    df1_sample = df1

print(f"Generating pairplot for columns: {key_cols}")

# Generate the pairplot
pair_plot_fig = sns.pairplot(df1_sample[key_cols], plot_kws={'alpha': 0.3, 's': 10})
pair_plot_fig.fig.suptitle(f'Pairwise Relationships of Key Numerical Features (Sample: {len(df1_sample)} rows)', y=1.02)

# Save the pairplot image
image_filename = 'pairplot_sample_key_features.png'
plt.savefig(image_filename)
plt.show() 
print(f"Image saved as: {image_filename}")


key_cols = ['Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Calories']
print(f"Selected columns for conditional distribution plots: {key_cols}")
n_cols_plot = len(key_cols)
# Arrange plots logically, e.g., 3 columns, 2 rows
n_plot_rows = (n_cols_plot + 2) // 3
n_plot_cols = 3
fig, axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=(15, n_plot_rows * 4))
axes = axes.flatten() # Flatten the axes array for easy iteration

# Generate KDE plots for each key numerical column conditional on Sex
for i, col in enumerate(key_cols):
    sns.kdeplot(data=df1, x=col, hue='Sex', ax=axes[i], fill=True, common_norm=False)
    axes[i].set_title(f'Distribution of {col} by Sex', fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Distributions of Key Numerical Features and Calories by Sex', fontsize=16, y=1.03)
plt.tight_layout()

# Save the combined plot to a file
image_filename = 'conditional_distributions_by_sex.png'
plt.savefig(image_filename)
plt.show()
print(f"Image saved as: {image_filename}")


key_cols_outlier = ['Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Calories']
print(f"Selected columns for outlier box plots: {key_cols_outlier}")
n_rows_outlier = 2
n_cols_outlier = 3
fig, axes = plt.subplots(n_rows_outlier, n_cols_outlier, figsize=(15, 8))
axes = axes.flatten() # Flatten the axes array for easy iteration

# Generate box plots for each key numerical column
for i, col in enumerate(key_cols_outlier):
    sns.boxplot(y=df1[col], ax=axes[i]) # Using y= for vertical box plots
    axes[i].set_title(f'Box plot of {col}', fontsize=12)
    axes[i].set_xlabel('') # Remove x-label as it's redundant for vertical boxplot
    axes[i].set_ylabel(col) # Set y-label to column name

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Box Plots for Outlier Detection in Key Numerical Features', fontsize=16, y=1.02) # Add overall title
plt.tight_layout()

# Save the combined plot to a file
image_filename = 'outlier_boxplots_key_features.png'
plt.savefig(image_filename)
plt.show()
print(f"Image saved as: {image_filename}")


# Apply log(1+x) transformation
df1['log_Calories'] = np.log1p(df1['Calories'])
print("Created 'log_Calories' column.")

# Display descriptive statistics for the new column
print("\n--- Descriptive Statistics for 'log_Calories' ---")
print(df1['log_Calories'].describe())
print("\n")

# Visualize the distribution of the transformed column
plt.figure(figsize=(10, 6))
sns.histplot(df1['log_Calories'], kde=True, bins=50)
plt.title('Distribution of log(1 + Calories)', fontsize=14)
plt.xlabel('log(1 + Calories)')
plt.ylabel('Frequency')
plt.tight_layout()

# Save the plot
image_filename = 'log_calories_distribution.png'
plt.savefig(image_filename)
plt.show()
print(f"Image saved as: {image_filename}")



df1 = pd.get_dummies(df1, columns=['Sex'], drop_first=True, dtype=int)
print(df1.head())
print("\n")
print("--- Updated DataFrame Columns ---")
print(df1.columns)




# Define columns to cap
columns_to_cap = ['Duration', 'Heart_Rate', 'Body_Temp', 'Weight']
print(f"Columns identified for capping: {columns_to_cap}")

# Apply capping based on the 99th percentile
for col in columns_to_cap:
    if col in df1.columns:
        percentile_99 = df1[col].quantile(0.99)
        print(f"Capping '{col}' at 99th percentile: {percentile_99:.4f}")
        # Using clip for concise capping
        df1[col] = df1[col].clip(upper=percentile_99)
    else:
        print(f"Warning: Column '{col}' not found in DataFrame. Skipping capping for this column.")

print("\n--- Descriptive Statistics After Capping ---")
# Display descriptive statistics for the modified columns only
print(df1[columns_to_cap].describe())
print("\n")
print("Outlier capping applied successfully. Descriptive statistics for capped columns printed above.")



# Define independent variables for VIF calculation
# Exclude id, target (Calories), and transformed target (log_Calories)
# Include the encoded 'Sex_male' feature
independent_vars = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex_male']

# Verify all expected columns exist
missing_cols = [col for col in independent_vars if col not in df1.columns]
if missing_cols:
     print(f"Error: The following required columns for VIF calculation are missing: {missing_cols}")
else:
    print(f"Independent variables selected for VIF calculation: {independent_vars}")

    # Create a DataFrame with only the independent variables
    X = df1[independent_vars].copy()

    # Add a constant term for VIF calculation
    # Ensure no NaN/inf values which can cause errors in VIF
    X = X.dropna() # Drop rows with NaNs in selected columns if any (shouldn't be based on Step 1.1)
    X = X[~X.isin([np.inf, -np.inf]).any(axis=1)]

    if X.empty:
         print("Error: DataFrame is empty after removing NaN/Inf values. Cannot calculate VIF.")
    else:
         X_with_const = sm.add_constant(X)
         print("Added constant for VIF calculation.")

         # Calculate VIF for each feature
         vif_data = pd.DataFrame()
         vif_data["feature"] = X_with_const.columns
         # Calculate VIF - list comprehension handles potential errors if a column has zero variance
         vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

         # Filter out the constant's VIF as it's not typically interpreted
         vif_results = vif_data[vif_data["feature"] != "const"]

         print("\n--- Variance Inflation Factor (VIF) Results ---")
         print(vif_results)
         print("\n")
         print("VIF calculation successful. Results printed above.")



df= pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')



from scipy.stats import zscore


# Drop the target column and select only numeric features
features = df.drop(columns=["Calories"]).select_dtypes(include=[np.number])

# Calculate z-scores for numeric features
z_scores = np.abs(zscore(features))

# Find rows with any feature having z-score > 3
feature_outliers = (z_scores > 3).any(axis=1)

# IQR for Calories (target outliers)
Q1 = df["Calories"].quantile(0.25)
Q3 = df["Calories"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
target_outliers = (df["Calories"] < lower) | (df["Calories"] > upper)

# Combine both feature and target outlier condition/s
combined_outliers = feature_outliers | target_outliers

# Remove rows with outliers
df= df[~combined_outliers].copy()

print(f"Original rows: {df.shape[0]}, After removing outliers: {df.shape[0]}")



from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score
)
import lightgbm as lgb

# FEATURE ENGINEERING (basic) 
# df=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
# df = df.drop(columns=['id'])  

# 1.2 Target‐encode Sex by mean Calories
sex_mean = df.groupby('Sex')['Calories'].mean()
df['Sex_enc'] = df['Sex'].map(sex_mean)
df = df.drop(columns=['Sex'])

# 1.3 Create BMI feature (Weight in kg / Height in m²)
df['Height_m'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)

# 1.4 (Optional) Interaction: Duration × Heart_Rate
df['Dur_x_HR'] = df['Duration'] * df['Heart_Rate']

# 1.5 Drop intermediate columns if desired
df = df.drop(columns=['Height_m'])

# TRAIN/TEST SPLIT
X = df.drop(columns=['Calories'])
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.005, random_state=42
)

# LOG-TRANSFORM TARGET (for RMSLE)
y_train_log = np.log1p(y_train)
y_test_log  = np.log1p(y_test)

#  LIGHTGBM DATASETS 
lgb_train = lgb.Dataset(X_train, label=y_train_log)
lgb_test  = lgb.Dataset(X_test,  label=y_test_log, reference=lgb_train)

# TRAIN MODEL
params = {
    'objective': 'regression',
    'metric':    'rmse',
    'learning_rate': 0.02074751088357589,
    'max_depth':     8,
    'num_leaves':    64,
    'feature_fraction': 0.9,
    'bagging_fraction':  0.9,
    'bagging_freq':      1,
    'device_type': 'gpu',      # use GPI
    'gpu_platform_id': 0,
    'gpu_device_id':   0,
    'verbose':  -1
}
# params = {
#     "learning_rate": 0.02074751088357589,
#     "max_depth": 7,
#     "min_child_weight": 1,
#     "subsample": 0.8566606013469481,
#     "colsample_bytree": 0.8219016576232071,
#     "reg_alpha": 0.0002542371757814725,
#     "reg_lambda": 0.0063623742088803024,
#     "objective": "regression",
#     "tree_method": "gpu_hist",
#     # ✅ GPU enabled
#     "random_state": 42
# }
# 2. Prepare datasets as before
lgb_train = lgb.Dataset(X_train, label=y_train_log)
lgb_test  = lgb.Dataset(X_test,  label=y_test_log, reference=lgb_train)

# 3. Train with callbacks for early stopping and logging
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=3500,
    valid_sets=[lgb_test],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=50)
    ]
)

# 4. Predict & inverse log-transform
y_pred_log = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(y_pred, 0)

# 5. Evaluate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score
)
import numpy as np

mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
mae   = mean_absolute_error(y_test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
r2    = r2_score(y_test, y_pred)

print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"RMSLE: {rmsle:.4f}")
print(f"R²:    {r2:.4f}")


# MSE:   12.6843
# RMSE:  3.5615
# MAE:   2.1510
# RMSLE: 0.0576
# R²:    0.9967


dft=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
dft_orig = dft.copy()

dft = dft.drop(columns=['id'])  # drop raw id
dft['Sex_enc'] = dft['Sex'].map(sex_mean)
dft = dft.drop(columns=['Sex'])
dft['Height_m'] = dft['Height'] / 100
dft['BMI']      = dft['Weight'] / (dft['Height_m'] ** 2)
dft['Dur_x_HR'] = dft['Duration'] * dft['Heart_Rate']
dft = dft.drop(columns=['Height_m'])
y_pred_log = model.predict(dft, num_iteration=model.best_iteration)
y_pred     = np.expm1(y_pred_log).clip(min=0)
submission = pd.DataFrame({
    'id':       dft_orig['id'],
    'Calories': y_pred
})
submission.to_csv('submission.csv', index=False)
print("Wrote submission.csv with", len(submission), "rows")





