from IPython.display import HTML, display

url = "https://media.istockphoto.com/id/872085152/photo/together-we-are-stronger.jpg?s=612x612&w=0&k=20&c=ytdjdtlRKouyoBPrhFYrAlFb_1-zCi5FUh2bwLqZaUg="

html_code = f'''
<img src="{url}" style="width:100%; height:300px; object-fit: cover; border-radius: 20px;">
'''
display(HTML(html_code))


#ğŸ”� Ah-ha! You found the secret sauce! ğŸ�”


# Importing Libraries

import warnings
warnings.filterwarnings("ignore")

import optuna
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import catboost as cb
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer


# Reading .csv data file
train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
original_data = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')


train_data.sample(5)


test_data.sample(5)


original_data.sample(5)


# Checking the number of rows and columns

num_train_rows, num_train_columns = train_data.shape

num_test_rows, num_test_columns = test_data.shape

num_original_rows, num_original_columns = original_data.shape

print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")

print("Original Data:")
print(f"Number of Rows: {num_original_rows}")
print(f"Number of Columns: {num_original_columns}")


# Creating a table for missing values, unique values and data types of the features

missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

missing_values_original = pd.DataFrame({'Feature': original_data.columns,
                             '[ORIGINAL] No.of Missing Values': original_data.isnull().sum().values,
                             '[ORIGINAL] % of Missing Values': ((original_data.isnull().sum().values)/len(original_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, missing_values_original, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df.style.background_gradient(cmap='viridis')


# Count duplicate rows in train_data
train_duplicates = train_data.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = test_data.duplicated().sum()

# Count duplicate rows in original_data
original_duplicates = original_data.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")
print(f"Number of duplicate rows in original_data: {original_duplicates}")


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the train dataset')
train_data.describe().T.style.background_gradient(cmap='viridis')


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the test dataset')
test_data.describe().T.style.background_gradient(cmap='viridis')


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the original dataset')
original_data.describe().T.style.background_gradient(cmap='viridis')


numerical_variables = ['Age','Height', 'Weight', 'Duration', 'Heart_Rate','Body_Temp']
target_variable = 'Calories'
categorical_variables =['Sex'] 


# Analysis of all NUMERICAL features

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'
original_data['Dataset'] = 'Original'

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_data, test_data,original_data.dropna()]), x=variable, y="Dataset", palette=custom_palette)
    plt.xlabel(variable)
    plt.title(f"Box Plot for {variable}")

    # Separate Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train")
    sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test")
    sns.histplot(data=original_data.dropna(), x=variable, color=custom_palette[2], kde=True, bins=30, label="Original")
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {variable} [TRAIN, TEST & ORIGINAL]")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each variable
for variable in numerical_variables:
    create_variable_plots(variable)

# Drop the 'Dataset' column after analysis
train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)
original_data.drop('Dataset', axis=1, inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap

# Define color palettes
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Only first two used for train and test respectively
countplot_color = '#5C67A3'

# Add a 'dataset' column to differentiate train and test data
train_data = train_data.copy()
test_data = test_data.copy()
train_data['dataset'] = 'train'
test_data['dataset'] = 'test'

# Function to create and display a row of plots for a single categorical variable
def create_categorical_plots(variable):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie Chart - Handling many categories
    plt.subplot(1, 2, 1)

    combined = pd.concat([train_data, test_data])
    value_counts = combined[variable].value_counts()

    # Combine small categories into "Other" if they contribute less than 5%
    threshold = 0.05 * value_counts.sum()
    filtered_values = value_counts[value_counts >= threshold]

    wedges, texts, autotexts = plt.pie(
        filtered_values,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',  # Hide labels < 5%
        colors=pie_chart_palette[:len(filtered_values)],
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=[0.05 if p > 5 else 0 for p in filtered_values],  # Slightly separate larger slices
        textprops={'fontsize': 10}  # Adjust font size
    )

    plt.title("\n".join(textwrap.wrap(f"Pie Chart for {variable} [TRAIN & TEST Combined]", width=50)))
    plt.legend(filtered_values.index, loc="upper left", bbox_to_anchor=(1, 1))

    # Bar Graph: Use hue for dataset (train and test)
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=combined,
        x=variable,
        hue='dataset',
        palette=custom_palette[:2],
        alpha=0.8
    )
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title("\n".join(textwrap.wrap(f"Bar Graph for {variable}  [TRAIN & TEST Combined]", width=50)))
    plt.xticks(rotation=30)  # Rotate labels for readability

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each categorical variable
for variable in categorical_variables:
    create_categorical_plots(variable)

# Drop the 'Dataset' column after analysis
train_data.drop('dataset', axis=1, inplace=True)
test_data.drop('dataset', axis=1, inplace=True)


# Custom palette for datasets
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

# Tag datasets for comparison
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'
original_data['Dataset'] = 'Original'

# Function to visualize a continuous target variable
def plot_continuous_target(variable):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Boxplot across datasets
    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=pd.concat([train_data, original_data.dropna()]),
        x=variable,
        y='Dataset',
        palette=custom_palette
    )
    plt.title(f"Box Plot for {variable}")
    plt.xlabel(variable)

    # --- Histogram with KDE overlay
    plt.subplot(1, 2, 2)
    sns.histplot(train_data[variable], color=custom_palette[0], kde=True, bins=30, label='Train')
    sns.histplot(original_data.dropna()[variable], color=custom_palette[2], kde=True, bins=30, label='Original')
    plt.title(f"Distribution of {variable} [Train/Original]")
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ğŸ“Œ Call for your continuous target variable (e.g., 'Listening_Time_minutes')
plot_continuous_target(target_variable)

# Drop 'Dataset' column after use
train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)
original_data.drop('Dataset', axis=1, inplace=True)


variables = [col for col in train_data.columns if col in numerical_variables]

# Adding variables to the existing list
test_variables = variables
train_variables = variables+ [target_variable]

# Calculate correlation matrices for train_data and test_data
corr_train = train_data[train_variables].corr()
corr_test = test_data[test_variables].corr()

# Create masks for the upper triangle
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Set the text size and rotation
annot_kws = {"size": 8, "rotation": 45}

# Generate heatmaps for train_data
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
ax_train = sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
                      square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data')

# Generate heatmaps for test_data
plt.subplot(1, 2, 2)
ax_test = sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
                     square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# Selecting numerical features + target variable
variables = [col for col in train_data.columns if col in numerical_variables]
train_variables = variables + [target_variable]

# Compute correlation with 'rainfall' and transpose for horizontal display
corr_train = train_data[train_variables].corr()[[target_variable]].T  # Transpose for horizontal orientation

# Set the text size and rotation
annot_kws = {"size": 10}  # Increased size for better visibility

# Generate horizontal heatmap without color bar
plt.figure(figsize=(10, 2))  # Adjusted for a horizontal layout
ax_train = sns.heatmap(corr_train, cmap='viridis', annot=True,
                      square=False, linewidths=0.5, annot_kws=annot_kws,
                      cbar=False)  # **Removed color bar**

# Formatting
plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
plt.title('Correlation Heatmap - Train Data (ONLY TARGET)')
plt.yticks(rotation=0)  # Keep y-labels horizontal

# Show plot
plt.show()


import numpy as np
import pandas as pd

def create_features(df):
    """
    Given a dataframe with workout data, create new features for modeling.

    Engineered features:
    - BMI: weight (kg) divided by squared height (m^2).
    - Weight_Height_Ratio: weight divided by height (cm).
    - Log_{col}: natural log of Duration, Heart_Rate, Body_Temp, Weight, Age (addresses skew).
    - Duration_x_HR: Duration * Heart_Rate (overall workload).
    - Duration_x_Temp: Duration * Body_Temp (thermal work).
    - HR_x_Temp: Heart_Rate * Body_Temp (intensity Ã— heat).
    - Age_x_Duration: Age * Duration (age-modulated effort).
    - HR_per_Duration: Heart_Rate / Duration (beats per minute per minute).
    - Temp_Anomaly: Body_Temp minus 37Â°C (deviation from baseline).
    - HR_Ratio_Max: Heart_Rate divided by estimated max HR (220 âˆ’ Age).
    - Is_Male: binary indicator (1 if male, 0 if female).
    """
    df = df.copy()

    # 1. Anthropometrics
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Weight_Height_Ratio'] = df['Weight'] / df['Height']

    # 2. Log-transforms to tame skew
    for col in ['Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Age']:
        df[f'Log_{col}'] = np.log(df[col] + 1)

    # 3. Interaction features
    df['Duration_x_HR']   = df['Duration']   * df['Heart_Rate']
    df['Duration_x_Temp'] = df['Duration']   * df['Body_Temp']
    df['HR_x_Temp']       = df['Heart_Rate'] * df['Body_Temp']
    df['Age_x_Duration']  = df['Age']        * df['Duration']

    # 4. Intensity & anomaly metrics
    df['HR_per_Duration'] = df['Heart_Rate'] / df['Duration'].replace(0, np.nan)
    df['Temp_Anomaly']    = df['Body_Temp'] - 37
    df['HR_Ratio_Max']    = df['Heart_Rate'] / (220 - df['Age'])

    # 5. Demographic encoding
    df['Is_Male'] = (df['Sex'] == 'male').astype(int)

    return df

# Apply to train and test
train_data = create_features(train_data)
test_data  = create_features(test_data)


# Drop columns from both train and test datasets
columns_to_drop = [ 'Weight', 'Height', 'Sex', 'Duration', 'Heart_Rate', 'Body_Temp']

train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical variables
columns_to_check = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove unwanted variables
columns_to_check = [col for col in columns_to_check if col not in ['id']]

# Function to remove outliers using IQR and visualize only affected features
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.01)
    Q3 = data[column].quantile(0.99)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)

    # Only proceed if outliers were detected (i.e., rows were deleted)
    if rows_deleted > 0:
        # Create a 1x2 plot for before & after visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Original Data Boxplot
        sns.boxplot(x=data[column], color='lightblue', ax=axes[0],
                    flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
        axes[0].set_title(f'Before Outlier Removal: {column}')

        # Highlight Q1, Q3, and Bounds in the first plot
        axes[0].axvline(Q1, color='green', linestyle='--', label='Q1 (1st Percentile)')
        axes[0].axvline(Q3, color='blue', linestyle='--', label='Q3 (99th Percentile)')
        axes[0].axvline(lower_bound, color='red', linestyle='-', label='Lower Bound')
        axes[0].axvline(upper_bound, color='red', linestyle='-', label='Upper Bound')
        axes[0].legend()

        # Boxplot after outlier removal
        sns.boxplot(x=filtered_data[column], color='lightgreen', ax=axes[1],
                    flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
        axes[1].set_title(f'After Outlier Removal: {column}')

        plt.suptitle(f'Outlier Detection & Removal for {column}')
        plt.tight_layout()
        plt.show()

        print(f"âœ… Outliers detected and removed for {column} â†’ {rows_deleted} rows deleted")

    return filtered_data, rows_deleted

# Apply function to each numerical column and visualize only affected features
rows_deleted_total = 0
features_with_outliers = []

for column in columns_to_check:
    train_data_filtered, rows_deleted = remove_outliers_iqr_with_plot(train_data, column)

    # Only update train_data if outliers were removed
    if rows_deleted > 0:
        train_data = train_data_filtered
        rows_deleted_total += rows_deleted
        features_with_outliers.append(column)

# Summary
print("\nğŸ“Š Summary of Outlier Removal:")
if features_with_outliers:
    print(f"Total rows deleted: {rows_deleted_total}")
    print(f"Features with outliers removed: {features_with_outliers}")
else:
    print("No significant outliers detected. No rows removed.")


y = train_data[target_variable].reset_index(drop=True).values
id_test = test_data['id']
id_train = train_data['id']
target = [target_variable]
train_data.drop(columns=['id'], inplace=True)
test_data.drop(columns=['id'], inplace=True)


from sklearn.preprocessing import MinMaxScaler

train_data_to_scale = train_data
test_data_to_scale = test_data

# Initialize MinMaxScaler
minmax_scaler = MinMaxScaler()

# Fit the scaler on the training data
minmax_scaler.fit(train_data_to_scale.drop(target, axis=1))

# Scale the training data
scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(target, axis=1))
scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(target, axis=1).columns)

# Scale the test data using the parameters from the training data
scaled_data_test = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)


# Concatenate train datasets
train_data_combined = scaled_train_df.reset_index(drop=True)

# Concatenate test datasets
test_data_combined = scaled_test_df.reset_index(drop=True)


nb_type = 'Submission'


print("ğŸ§© Features used for modeling:\n")
print(train_data_combined.columns.tolist())


if nb_type == 'Train':
    import warnings
    warnings.filterwarnings("ignore")
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_log_error
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import (
        RandomForestRegressor, ExtraTreesRegressor,
        HistGradientBoostingRegressor, GradientBoostingRegressor
    )
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.base import clone

    # competition RMSLE
    def competition_rmsle(y_true, y_pred):
        y_pred = np.clip(y_pred, 0, None)
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    # â”€â”€â”€ 2) Define base regressors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    base_models = {
        'Ridge'        : Ridge(alpha=1.0, random_state=42),
        'ElasticNet'   : ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
        'RandomForest' : RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        'ExtraTrees'   : ExtraTreesRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        'HistGB'       : HistGradientBoostingRegressor(max_iter=100, max_depth=10, random_state=42),
        'GBM'          : GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42),
        'LightGBM'     : LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=8, n_jobs=-1, random_state=42, verbose=-1),
        'XGBoost'      : XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=6, n_jobs=-1, verbosity=0, random_state=42),
        'CatBoost'     : CatBoostRegressor(iterations=50, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    }

    # â”€â”€â”€ 3) Cross-validate (5-fold) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {name: [] for name in base_models}

    print("Running 5-fold CV for Calories regressionâ€¦\n")
    for name, model in base_models.items():
        for tr_idx, va_idx in cv.split(X):
            m = clone(model)
            m.fit(X[tr_idx], y[tr_idx])
            preds = m.predict(X[va_idx])
            results[name].append(competition_rmsle(y[va_idx], preds))
        print(f"{name:12s} â†’ folds: {np.round(results[name], 4)} | mean: {np.mean(results[name]):.4f}")

    # â”€â”€â”€ 4) Plot comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [results[n] for n in results],
        labels=list(results.keys()),
        showmeans=True
    )
    plt.title("Calories Regression: RMSLE by Model (5-Fold CV)")
    plt.ylabel("RMSLE")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if nb_type == 'Train':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_log_error

    # Prepare data
    X = train_data_combined

    # Initialize model
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)

    # Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmsle = []

    print("Training HistGradientBoostingRegressor with 5-Fold CV...\n")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        preds = np.clip(preds, 0, None)

        rmsle = np.sqrt(mean_squared_log_error(y_val, preds))
        fold_rmsle.append(rmsle)
        print(f"Fold {fold+1}: RMSLE = {rmsle:.5f}")

    # Plot RMSLE across folds
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), fold_rmsle, marker='o', linestyle='-', color='#2980B9')
    plt.title("RMSLE per Fold - HistGradientBoostingRegressor")
    plt.xlabel("Fold")
    plt.ylabel("RMSLE")
    plt.xticks(range(1, 6))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
import pandas as pd
import numpy as np

# 1. Prepare data
X_train = train_data_combined
X_test = test_data_combined
y_train = y

# 2. Initialize models
et_model = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
histgb_model = HistGradientBoostingRegressor(max_iter=100, max_depth=10, random_state=42)

# 3. Train all models
et_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
histgb_model.fit(X_train, y_train)

# 4. Predict on test set
pred_et = et_model.predict(X_test)
pred_rf = rf_model.predict(X_test)
pred_histgb = histgb_model.predict(X_test)

# 5. Average predictions (simple mean ensemble)
ensemble_preds = (pred_et + pred_rf + pred_histgb) / 3
ensemble_preds = np.clip(ensemble_preds, 0, None)  # avoid negative predictions

# 6. Format submission
submission = pd.DataFrame({
    'id': id_test,
    target_variable: ensemble_preds
})

# 7. Save to CSV
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file created: 'submission_ensemble_ET_RF_HistGB.csv'")

