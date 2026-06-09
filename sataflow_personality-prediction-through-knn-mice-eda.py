import os
import pandas as pd
import warnings

# Suppress all warnings.
# While useful for cleaner output during development, be cautious in production
# as it might hide important information.
warnings.filterwarnings("ignore")

print("Starting data loading process...")

# Define the base directory where Kaggle datasets are mounted.
# For the 'playground-series-s5e7' competition, your files will be here.
input_directory = '/kaggle/input/playground-series-s5e7/'

# Verify if the input directory exists. This helps catch issues if the dataset
# hasn't been added to the notebook or if the path is incorrect.
if not os.path.exists(input_directory):
    print(f"Error: The expected input directory '{input_directory}' was not found.")
    print("Please ensure you have added the 'playground-series-s5e7' dataset to your Kaggle Notebook.")
else:
    # Construct the full paths to your train and test CSV files.
    train_file_path = os.path.join(input_directory, 'train.csv')
    test_file_path = os.path.join(input_directory, 'test.csv')

    # Load the datasets using pandas.
    try:
        train_df = pd.read_csv(train_file_path)
        test_df = pd.read_csv(test_file_path)
        
        print("Data loading complete.")
        print(f"Train data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")

    except FileNotFoundError:
        print(f"Error: One or both files ('train.csv', 'test.csv') not found in '{input_directory}'.")
        print("Please verify the file names within the dataset.")
    except Exception as e:
        # Catch any other unexpected errors during the loading process.
        print(f"An unexpected error occurred during data loading: {e}")


print("=== Basic information of test_df ===")
print(f"Rows: {test_df.shape[0]}, Columns: {test_df.shape[1]}")
print("\nPreview of the first 5 rows:")
print(test_df.head())

print("\n\n=== Basic information of train_df ===")
print(f"Rows: {train_df.shape[0]}, Columns: {train_df.shape[1]}")
print("\nPreview of the first 5 rows:")
print(train_df.head())

print("\n\n=== Data structure information of train_df ===")
train_df.info()

print("\n\n=== Missing value percentage (%) for each column in test_df ===")
print((test_df.isnull().mean() * 100).sort_values(ascending=False))

print("\n=== Missing value percentage (%) for each column in train_df ===")
print((train_df.isnull().mean() * 100).sort_values(ascending=False))


import numpy as np
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'id']

print("\n【Numeric columns for correlation analysis】:")
print(numeric_cols)

from scipy.stats import pearsonr, spearmanr, kendalltau

corr_results = {
    'Pearson': {},
    'Spearman': {},
    'Kendall': {}
}

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col1 = numeric_cols[i]
        col2 = numeric_cols[j]

        valid_data = train_df[[col1, col2]].dropna()
        x = valid_data[col1]
        y = valid_data[col2]

        pearson_corr, _ = pearsonr(x, y)
        corr_results['Pearson'][f"{col1} & {col2}"] = round(pearson_corr, 4)

        spearman_corr, _ = spearmanr(x, y)
        corr_results['Spearman'][f"{col1} & {col2}"] = round(spearman_corr, 4)

        kendall_corr, _ = kendalltau(x, y)
        corr_results['Kendall'][f"{col1} & {col2}"] = round(kendall_corr, 4)

print("\n【Pearson Correlation Coefficient】")
for pair, value in corr_results['Pearson'].items():
    print(f"{pair}: {value}")

print("\n【Spearman Rank Correlation Coefficient】")
for pair, value in corr_results['Spearman'].items():
    print(f"{pair}: {value}")

print("\n【Kendall Tau Correlation Coefficient】")
for pair, value in corr_results['Kendall'].items():
    print(f"{pair}: {value}")


import pandas as pd
from scipy.stats import skew, kurtosis

numeric_cols = ['Time_spent_Alone', 'Social_event_attendance',
                'Going_outside', 'Friends_circle_size', 'Post_frequency']

print("=== Descriptive Statistics for Numerical Variables ===")
for col in numeric_cols:
    desc_stats = train_df[col].describe()
    print(f"\n【{col}】Descriptive Statistics:")
    print(desc_stats)

print("\n\n=== Distribution Shape Analysis (Skewness & Kurtosis) ===")
for col in numeric_cols:
    s = skew(train_df[col].dropna())
    k = kurtosis(train_df[col].dropna())
    print(f"\n【{col}】")
    print(f"Skewness: {s:.4f} -> {'Right-skewed' if s > 0.5 else 'Left-skewed' if s < -0.5 else 'Approximately Symmetric'}")
    print(f"Kurtosis: {k:.4f} -> {'Leptokurtic' if k > 0.5 else 'Platykurtic' if k < -0.5 else 'Mesokurtic'}")

print("\n\n=== Outlier Detection (IQR Method) ===")
for col in numeric_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)][col]

    print(f"\n【{col}】")
    print(f"IQR Lower Bound: {lower_bound:.4f}")
    print(f"IQR Upper Bound: {upper_bound:.4f}")
    print(f"Number of Outliers: {len(outliers)}")


from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
working_directory = r'/kaggle/working/'
os.chdir(working_directory)

train_processed_df = train_df.copy(deep=True)
print("\n--- Step 2: Create deep copy of train_processed_df ---")
print("✅ train_df deeply copied to train_processed_df for subsequent imputation.")

print("\n--- Step 3: Perform KNN imputation on binary missing variables (Stage_fear, Drained_after_socializing) ---")

mapping_binary_cols = {'No': 0, 'Yes': 1}
mapping_personality = {'Extrovert': 0, 'Introvert': 1}

train_processed_df['Stage_fear_encoded'] = train_processed_df['Stage_fear'].map(mapping_binary_cols)
train_processed_df['Drained_after_socializing_encoded'] = train_processed_df['Drained_after_socializing'].map(mapping_binary_cols)
train_processed_df['Personality_encoded'] = train_processed_df['Personality'].map(mapping_personality)

cols_to_impute_knn_encoded = ['Stage_fear_encoded', 'Drained_after_socializing_encoded']
features_for_knn = cols_to_impute_knn_encoded + ['Personality_encoded']

knn_data = train_processed_df[features_for_knn].copy()

imputer_knn = KNNImputer(n_neighbors=5, weights='uniform')

imputed_knn_array = imputer_knn.fit_transform(knn_data)

imputed_knn_df_temp = pd.DataFrame(imputed_knn_array, columns=features_for_knn, index=train_processed_df.index)

for col_encoded in cols_to_impute_knn_encoded:
    train_processed_df[col_encoded] = imputed_knn_df_temp[col_encoded].round().astype(int)

train_processed_df['Stage_fear'] = train_processed_df['Stage_fear_encoded'].map({0: 'No', 1: 'Yes'})
train_processed_df['Drained_after_socializing'] = train_processed_df['Drained_after_socializing_encoded'].map({0: 'No', 1: 'Yes'})

train_processed_df.drop(columns=['Stage_fear_encoded', 'Drained_after_socializing_encoded', 'Personality_encoded'], inplace=True)

print("✅ Binary variables (Stage_fear, Drained_after_socializing) missing values filled using KNN.")
print("    Missing values in Stage_fear after imputation:", train_processed_df['Stage_fear'].isnull().sum())
print("    Missing values in Drained_after_socializing after imputation:", train_processed_df['Drained_after_socializing'].isnull().sum())

print("\n--- Step 4: Perform MICE imputation on numerical missing variables (initial_strategy='median') ---")

numerical_cols_to_impute = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

imputer_mice = IterativeImputer(
    initial_strategy='median',
    max_iter=10,
    random_state=42,
    estimator=RandomForestRegressor(n_jobs=-1, random_state=42)
)

data_for_mice = train_processed_df[numerical_cols_to_impute]

print("    Applying MICE imputation. This might take some time...")
imputed_numerical_array = imputer_mice.fit_transform(data_for_mice)

train_processed_df[numerical_cols_to_impute] = imputed_numerical_array

for col in numerical_cols_to_impute:
    train_processed_df[col] = train_processed_df[col].round().astype(int)

print("✅ Numerical variables missing values filled using MICE (initial_strategy='median').")

print("\n--- Step 5: Final check for missing values in train_processed_df ---")
final_missing_percentages = (train_processed_df.isnull().mean() * 100).sort_values(ascending=False)
print("Missing value percentage (%) for each column in train_processed_df after imputation:")
print(final_missing_percentages)

if final_missing_percentages.drop(['id', 'Personality']).sum() == 0:
    print("✅ All target missing values successfully imputed, train_processed_df has no missing values (id and Personality had no missing values initially).")
else:
    print("⚠️ Warning: Missing values still exist after imputation. Please check the missing percentages above.")

print("\n--- Preview of the first 5 rows of the imputed train_processed_df dataset ---")
print(train_processed_df.head())

print("\n--- Step 6: Save the imputed dataframe ---")
output_filename = os.path.join(working_directory, 'fill.csv')
train_processed_df.to_csv(output_filename, index=False)

print(f"✅ Imputed dataframe saved to: {output_filename}")
print("--- All operations complete ---")


import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.metrics import r2_score 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["axes.unicode_minus"] = False

# --- Data Loading for Kaggle Notebook ---
KAGGLE_INPUT_DIR = '/kaggle/input/playground-series-s5e7/'
KAGGLE_WORKING_DIR = '/kaggle/working/'

fill_file_path = os.path.join(KAGGLE_WORKING_DIR, 'fill.csv')
train_file_path = os.path.join(KAGGLE_INPUT_DIR, 'train.csv')
test_file_path = os.path.join(KAGGLE_INPUT_DIR, 'test.csv')

Fill, test_df, train_df = None, None, None

try:
    if os.path.exists(fill_file_path):
        Fill = pd.read_csv(fill_file_path)
        print(f"✅ 'Fill' DataFrame loaded from {fill_file_path}.")
    else:
        alt_fill_path = os.path.join(KAGGLE_INPUT_DIR, 'fill.csv')
        if os.path.exists(alt_fill_path):
            Fill = pd.read_csv(alt_fill_path)
            print(f"✅ 'Fill' DataFrame loaded from {alt_fill_path}.")
        else:
            raise FileNotFoundError(f"'fill.csv' not found at {fill_file_path} or {alt_fill_path}.")

    if os.path.exists(test_file_path):
        test_df = pd.read_csv(test_file_path)
        print(f"✅ 'test_df' DataFrame loaded from {test_file_path}.")
    else:
        raise FileNotFoundError(f"'test.csv' not found at {test_file_path}.")

    if os.path.exists(train_file_path):
        train_df = pd.read_csv(train_file_path)
        print(f"✅ 'train_df' DataFrame loaded from {train_file_path}.")
    else:
        raise FileNotFoundError(f"'train.csv' not found at {train_file_path}.")

except FileNotFoundError as e:
    print(f"❌ Error: Required CSV file not found. {e}")
    raise
except Exception as e:
    print(f"❌ An unexpected error occurred during data loading: {e}")
    raise


print(f"\nFill DataFrame shape: {Fill.shape}")
print(f"test_df original shape: {test_df.shape}")
print(f"train_df original shape: {train_df.shape}")

numeric_cols = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

binary_cols = ['Stage_fear', 'Drained_after_socializing']

print("\n" + "="*50)
print("=== Imputation Results Evaluation (Fill DataFrame vs Original Data) ===")
print("="*50)

print("\n--- 1. Missing Value Check ---")
print((Fill.isnull().sum() / len(Fill) * 100).sort_values(ascending=False))
print("Expected result: Missing value percentage for all columns except 'id' and 'Personality' should be 0%.")

print("\n--- 2. Descriptive Statistics Comparison ---")
print("\n--- Fill DataFrame (after imputation) Descriptive Statistics ---")
print(Fill[numeric_cols].describe())

print("\n--- train_df (original) Descriptive Statistics (non-missing part only) ---")
print(train_df[numeric_cols].describe())

print("\n--- test_df (original) Descriptive Statistics (non-missing part only) ---")
print(test_df[numeric_cols].describe())

print("\n--- 3. Distribution Shape Analysis (Skewness & Kurtosis) Comparison ---")

skew_data = []
kurt_data = []

for col in numeric_cols:
    print(f"\n--- {col} ---")
    s_fill = skew(Fill[col])
    k_fill = kurtosis(Fill[col])
    print(f"    Fill (after imputation) - Skewness: {s_fill:.4f} ({'Right-skewed' if s_fill > 0.5 else 'Left-skewed' if s_fill < -0.5 else 'Approximately Symmetric'})")
    print(f"    Fill (after imputation) - Kurtosis: {k_fill:.4f} ({'Leptokurtic' if k_fill > 0.5 else 'Platykurtic' if k_fill < -0.5 else 'Mesokurtic'})")

    s_train_orig = skew(train_df[col].dropna())
    k_train_orig = kurtosis(train_df[col].dropna())
    print(f"    train_df (original non-missing) - Skewness: {s_train_orig:.4f} ({'Right-skewed' if s_train_orig > 0.5 else 'Left-skewed' if s_train_orig < -0.5 else 'Approximately Symmetric'})")
    print(f"    train_df (original non-missing) - Kurtosis: {k_train_orig:.4f} ({'Leptokurtic' if k_train_orig > 0.5 else 'Platykurtic' if k_train_orig < -0.5 else 'Mesokurtic'})")

    s_test_orig = skew(test_df[col].dropna())
    k_test_orig = kurtosis(test_df[col].dropna())
    print(f"    test_df (original non-missing) - Skewness: {s_test_orig:.4f} ({'Right-skewed' if s_test_orig > 0.5 else 'Left-skewed' if s_test_orig < -0.5 else 'Approximately Symmetric'})")
    print(f"    test_df (original non-missing) - Kurtosis: {k_test_orig:.4f} ({'Leptokurtic' if k_test_orig > 0.5 else 'Platykurtic' if k_test_orig < -0.5 else 'Mesokurtic'})")

    skew_diff = s_fill - s_test_orig
    kurt_diff = k_fill - k_test_orig

    skew_data.append({
        'Column Name': col,
        'Fill Skewness': s_fill,
        'test_df Skewness': s_test_orig,
        'Skewness Difference': skew_diff,
        'Difference Extent': 'Significant' if abs(skew_diff) > 0.5 else 'Moderate' if abs(skew_diff) > 0.2 else 'Minor'
    })

    kurt_data.append({
        'Column Name': col,
        'Fill Kurtosis': k_fill,
        'test_df Kurtosis': k_test_orig,
        'Kurtosis Difference': kurt_diff,
        'Difference Extent': 'Significant' if abs(kurt_diff) > 0.5 else 'Moderate' if abs(kurt_diff) > 0.2 else 'Minor'
    })

print("\n--- Skewness Difference Report ---")
skew_df = pd.DataFrame(skew_data)
print(skew_df)

print("\n--- Kurtosis Difference Report ---")
kurt_df = pd.DataFrame(kurt_data)
print(kurt_df)


import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.metrics import r2_score # This is imported but not used in the provided snippet
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings("ignore")

# --- Font Configuration for Chinese Characters ---
# This block attempts to set a Chinese font for matplotlib.
# In Kaggle Notebooks, 'SimHei' or 'WenQuanYi Micro Hei' are often available.
plt.rcParams["axes.unicode_minus"] = False

# Try common Kaggle/Linux fonts first, then Windows specific.
font_set = False
for font_name_option in ["WenQuanYi Micro Hei", "SimHei", "HarmonyOS Sans SC", "Heiti TC", "Arial Unicode MS"]:
    try:
        # Check if the font is truly available on the system
        fm.findfont(font_name_option, fallback_to_default=False)
        plt.rcParams["font.family"] = [font_name_option]
        print(f"✅ Found and set font: '{font_name_option}' for matplotlib.")
        font_set = True
        break
    except Exception:
        continue

if not font_set:
    print("⚠️ Warning: Failed to load a common Chinese font. Chinese characters in charts may not display correctly.")
    print("Consider adding a font file to /kaggle/working/ and referencing it directly if needed.")

# --- Data Loading for Kaggle Notebook ---
# Define the standard Kaggle input directory for datasets
KAGGLE_INPUT_DIR = '/kaggle/input/playground-series-s5e7/'
# Define the working directory for generated files (like 'fill.csv' if it's an output)
KAGGLE_WORKING_DIR = '/kaggle/working/'

# Ensure 'fill.csv' is treated correctly.
# If 'fill.csv' is an imputed version generated by a *previous* step and saved to /kaggle/working/,
# or if it's part of the input data, adjust its path accordingly.
# For this example, assuming 'fill.csv' is either in input or working, preferring working.
# Adjust 'fill_file_path' if your 'fill.csv' is part of the /kaggle/input/ data.

fill_file_path = os.path.join(KAGGLE_WORKING_DIR, 'fill.csv') # Assuming fill.csv is an output from a previous step
train_file_path = os.path.join(KAGGLE_INPUT_DIR, 'train.csv')
test_file_path = os.path.join(KAGGLE_INPUT_DIR, 'test.csv')

# Initialize DataFrames outside the try block
Fill, test_df, train_df = None, None, None

try:
    # Load Fill DataFrame
    if os.path.exists(fill_file_path):
        Fill = pd.read_csv(fill_file_path)
        print(f"✅ 'Fill' DataFrame loaded from {fill_file_path}.")
    else:
        # If fill.csv is not in /kaggle/working/, check /kaggle/input/
        alt_fill_path = os.path.join(KAGGLE_INPUT_DIR, 'fill.csv')
        if os.path.exists(alt_fill_path):
            Fill = pd.read_csv(alt_fill_path)
            print(f"✅ 'Fill' DataFrame loaded from {alt_fill_path}.")
        else:
            raise FileNotFoundError(f"'fill.csv' not found at {fill_file_path} or {alt_fill_path}.")

    # Load test_df
    if os.path.exists(test_file_path):
        test_df = pd.read_csv(test_file_path)
        print(f"✅ 'test_df' DataFrame loaded from {test_file_path}.")
    else:
        raise FileNotFoundError(f"'test.csv' not found at {test_file_path}.")

    # Load train_df
    if os.path.exists(train_file_path):
        train_df = pd.read_csv(train_file_path)
        print(f"✅ 'train_df' DataFrame loaded from {train_file_path}.")
    else:
        raise FileNotFoundError(f"'train.csv' not found at {train_file_path}.")

except FileNotFoundError as e:
    print(f"❌ Error: Required CSV file not found. {e}")
    # Re-raise the exception to stop execution
    raise
except Exception as e:
    print(f"❌ An unexpected error occurred during data loading: {e}")
    raise


print(f"\nFill DataFrame shape: {Fill.shape}")
print(f"test_df original shape: {test_df.shape}")
print(f"train_df original shape: {train_df.shape}")

# Continue with your existing analysis logic...
numeric_cols = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

binary_cols = ['Stage_fear', 'Drained_after_socializing'] # This is defined but not used in the provided snippet

print("\n" + "="*50)
print("=== Imputation Results Evaluation (Fill DataFrame vs Original Data) ===")
print("="*50)

print("\n--- 1. Missing Value Check ---")
print((Fill.isnull().sum() / len(Fill) * 100).sort_values(ascending=False))
# 'id' and 'Personality' should have 0% missing if they are never missing in original or imputation handles them
print("Expected result: Missing value percentage for all columns except 'id' and 'Personality' should be 0%.")

print("\n--- 2. Descriptive Statistics Comparison ---")
print("\n--- Fill DataFrame (after imputation) Descriptive Statistics ---")
print(Fill[numeric_cols].describe())

print("\n--- train_df (original) Descriptive Statistics (non-missing part only) ---")
print(train_df[numeric_cols].describe())

print("\n--- test_df (original) Descriptive Statistics (non-missing part only) ---")
print(test_df[numeric_cols].describe())

print("\n--- 3. Distribution Shape Analysis (Skewness & Kurtosis) Comparison ---")

skew_data = []
kurt_data = []

for col in numeric_cols:
    print(f"\n【{col}】")
    s_fill = skew(Fill[col])
    k_fill = kurtosis(Fill[col])
    print(f"    Fill (after imputation) - Skewness: {s_fill:.4f} ({'Right-skewed' if s_fill > 0.5 else 'Left-skewed' if s_fill < -0.5 else 'Approximately Symmetric'})")
    print(f"    Fill (after imputation) - Kurtosis: {k_fill:.4f} ({'Leptokurtic' if k_fill > 0.5 else 'Platykurtic' if k_fill < -0.5 else 'Mesokurtic'})")

    s_train_orig = skew(train_df[col].dropna())
    k_train_orig = kurtosis(train_df[col].dropna())
    print(f"    train_df (original non-missing) - Skewness: {s_train_orig:.4f} ({'Right-skewed' if s_train_orig > 0.5 else 'Left-skewed' if s_train_orig < -0.5 else 'Approximately Symmetric'})")
    print(f"    train_df (original non-missing) - Kurtosis: {k_train_orig:.4f} ({'Leptokurtic' if k_train_orig > 0.5 else 'Platykurtic' if k_train_orig < -0.5 else 'Mesokurtic'})")

    s_test_orig = skew(test_df[col].dropna())
    k_test_orig = kurtosis(test_df[col].dropna())
    print(f"    test_df (original non-missing) - Skewness: {s_test_orig:.4f} ({'Right-skewed' if s_test_orig > 0.5 else 'Left-skewed' if s_test_orig < -0.5 else 'Approximately Symmetric'})")
    print(f"    test_df (original non-missing) - Kurtosis: {k_test_orig:.4f} ({'Leptokurtic' if k_test_orig > 0.5 else 'Platykurtic' if k_test_orig < -0.5 else 'Mesokurtic'})")

    skew_diff = s_fill - s_test_orig
    kurt_diff = k_fill - k_test_orig

    skew_data.append({
        'Column Name': col,
        'Fill Skewness': s_fill,
        'test_df Skewness': s_test_orig,
        'Skewness Difference': skew_diff,
        'Difference Extent': 'Significant' if abs(skew_diff) > 0.5 else 'Moderate' if abs(skew_diff) > 0.2 else 'Minor'
    })

    kurt_data.append({
        'Column Name': col,
        'Fill Kurtosis': k_fill,
        'test_df Kurtosis': k_test_orig,
        'Kurtosis Difference': kurt_diff,
        'Difference Extent': 'Significant' if abs(kurt_diff) > 0.5 else 'Moderate' if abs(kurt_diff) > 0.2 else 'Minor'
    })

print("\n--- Skewness Difference Report ---")
skew_df = pd.DataFrame(skew_data)
print(skew_df)

print("\n--- Kurtosis Difference Report ---")
kurt_df = pd.DataFrame(kurt_data)
print(kurt_df)


class PlotConfig:
    # Use standard Kaggle directories
    kaggle_input_dir = r'/kaggle/input/playground-series-s5e7/'
    kaggle_working_dir = r'/kaggle/working/'

    # File names relative to their respective directories
    file_names = {
        'fill': 'fill.csv',  # Assuming fill.csv is saved to /kaggle/working/
        'test': 'test.csv',  # These are from the competition input
        'train': 'train.csv' # These are from the competition input
    }

    # Ordered list of common Chinese fonts, prioritizing those often available in Kaggle/Linux
    font_options = [
        'WenQuanYi Micro Hei',
        'SimHei',
        'HarmonyOS Sans SC', # Less common but good to try if you have it
        'Heiti TC',
        'Microsoft YaHei', # Windows specific, less likely on Kaggle
        'FangSong',
        'STSong',
        'Arial Unicode MS'
    ]

    numeric_cols = [
        'Time_spent_Alone',
        'Social_event_attendance',
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]

    plot_style = "whitegrid"

    figure_sizes = {
        'distribution': (18, 12),
        'other': (12, 8)
    }

def load_data_from_kaggle(config):
    # This function is now specifically for Kaggle Notebooks
    print("Starting data loading for plotting...")

    # Define full paths based on Kaggle conventions
    fill_path = os.path.join(config.kaggle_working_dir, config.file_names['fill'])
    train_path = os.path.join(config.kaggle_input_dir, config.file_names['train'])
    test_path = os.path.join(config.kaggle_input_dir, config.file_names['test'])

    Fill, test_df, train_df = None, None, None # Initialize to None

    try:
        # Load Fill DataFrame (assumed to be an output generated by a previous step)
        if os.path.exists(fill_path):
            Fill = pd.read_csv(fill_path)
            print(f"✅ 'Fill' DataFrame loaded from {fill_path}.")
        else:
            # Fallback: check if 'fill.csv' is part of the original input dataset
            alt_fill_path = os.path.join(config.kaggle_input_dir, config.file_names['fill'])
            if os.path.exists(alt_fill_path):
                Fill = pd.read_csv(alt_fill_path)
                print(f"✅ 'Fill' DataFrame loaded from {alt_fill_path}.")
            else:
                raise FileNotFoundError(f"'fill.csv' not found at {fill_path} or {alt_fill_path}.")


        # Load test_df
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            print(f"✅ 'test_df' DataFrame loaded from {test_path}.")
        else:
            raise FileNotFoundError(f"'test.csv' not found at {test_path}.")

        # Load train_df
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            print(f"✅ 'train_df' DataFrame loaded from {train_path}.")
        else:
            raise FileNotFoundError(f"'train.csv' not found at {train_path}.")

    except FileNotFoundError as e:
        print(f"❌ Error: Required CSV file not found for plotting. {e}")
        raise # Re-raise to stop execution if essential files are missing
    except Exception as e:
        print(f"❌ An unexpected error occurred during data loading for plotting: {e}")
        raise # Re-raise to stop execution

    print(f"\nFill DataFrame shape: {Fill.shape}")
    print(f"test_df original shape: {test_df.shape}")
    print(f"train_df original shape: {train_df.shape}")

    return Fill, test_df, train_df

def detect_font(config):
    # More robust font detection for Kaggle environment
    plt.rcParams['axes.unicode_minus'] = False # Ensure minus sign displays correctly

    for font_name_option in config.font_options:
        try:
            # Check if the font is available
            fm.findfont(font_name_option, fallback_to_default=False)
            print(f"✅ Found available Chinese font: '{font_name_option}'")
            return font_name_option
        except Exception:
            continue
    print("⚠️ Warning: No common Chinese font found. Chinese characters in charts may not display correctly.")
    return 'DejaVu Sans' # Fallback to a common English font

def plot_distribution(config, font_name, Fill, test_df, train_df):
    plt.rcParams['font.family'] = [font_name] # Set font for plotting
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=config.figure_sizes['distribution'])
    sns.set_style(config.plot_style)

    for i, col in enumerate(config.numeric_cols):
        plt.subplot(2, 3, i + 1)

        # Plot KDE for Fill data
        sns.kdeplot(data=Fill[col], color='blue', linestyle='-',
                    label='Fill (Imputed) - KDE', linewidth=2)

        # Plot KDE for original non-missing train_df
        train_non_missing = train_df[col].dropna()
        sns.kdeplot(data=train_non_missing, color='green', linestyle='--',
                    label='train_df (Original Non-Missing) - KDE', linewidth=2)

        # Plot KDE for original non-missing test_df
        test_non_missing = test_df[col].dropna()
        sns.kdeplot(data=test_non_missing, color='red', linestyle=':',
                    label='test_df (Original Non-Missing) - KDE', linewidth=2)

        plt.title(f'{col} Distribution', fontsize=16)
        plt.xlabel(f'{col} Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('Numerical Variable Distribution Comparison: Imputed Data (Fill) vs Original Data',
                 fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

    print("\n--- Distribution visualization complete ---")

def plot_correlation(config, font_name, data):
    plt.rcParams['font.family'] = [font_name] # Set font for plotting
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=config.figure_sizes['other'])
    sns.set_style(config.plot_style) # Ensure style is applied

    corr_matrix = data[config.numeric_cols].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                square=True, linewidths=.5)

    plt.title('Numerical Variable Correlation Matrix (Imputed Data)', fontsize=18)
    plt.tight_layout()
    plt.show()

    print("\n--- Correlation visualization complete ---")


if __name__ == "__main__":
    config = PlotConfig()

    font_name = detect_font(config) # Detect font before setting style globally

    sns.set_style(config.plot_style)
    print(f"✅ Seaborn plot style set to '{config.plot_style}'.")

    # Load data using the Kaggle-specific loader
    Fill, test_df, train_df = load_data_from_kaggle(config) # Changed function name

    # Proceed with plotting only if data was loaded successfully
    if Fill is not None and test_df is not None and train_df is not None:
        plot_distribution(config, font_name, Fill, test_df, train_df)
        plot_correlation(config, font_name, Fill) # Plot correlation on the imputed data
        # plot_boxplot(config, font_name, Fill, test_df, train_df) # Removed this line


import os
import pandas as pd
from scipy.stats import skew, kurtosis

try:
    Fill = pd.read_csv('fill.csv')
    print("✅ File 'fill.csv' successfully read and named 'Fill'.")
    print(f"Fill DataFrame shape: {Fill.shape}")
except FileNotFoundError:
    print("❌ Error: 'fill.csv' file not found. Please ensure the file exists in the correct working directory.")
    exit()

print("\n\n=== Basic information of Fill DataFrame ===")
print(f"Rows: {Fill.shape[0]}, Columns: {Fill.shape[1]}")
print("\nPreview of the first 5 rows:")
print(Fill.head())

print("\n\n=== Data structure information of Fill DataFrame ===")
Fill.info()

print("\n\n=== Missing value percentage (%) for each column in Fill DataFrame ===")
print((Fill.isnull().mean() * 100).sort_values(ascending=False))

numeric_cols = ['Time_spent_Alone', 'Social_event_attendance',
                'Going_outside', 'Friends_circle_size', 'Post_frequency']

print("\n\n=== Descriptive statistics for numerical variables (based on Fill DataFrame) ===")
for col in numeric_cols:
    desc_stats = Fill[col].describe()
    print(f"\n【{col}】Descriptive Statistics:")
    print(desc_stats)

print("\n\n=== Distribution shape analysis (Skewness & Kurtosis) (based on Fill DataFrame) ===")
for col in numeric_cols:
    s = skew(Fill[col])
    k = kurtosis(Fill[col])
    print(f"\n【{col}】")
    print(f"Skewness: {s:.4f} -> {'Right-skewed' if s > 0.5 else 'Left-skewed' if s < -0.5 else 'Approximately Symmetric'}")
    print(f"Kurtosis: {k:.4f} -> {'Leptokurtic' if k > 0.5 else 'Platykurtic' if k < -0.5 else 'Mesokurtic'}")

print("\n\n=== Outlier detection (IQR method) (based on Fill DataFrame) ===")
for col in numeric_cols:
    Q1 = Fill[col].quantile(0.25)
    Q3 = Fill[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = Fill[(Fill[col] < lower_bound) | (Fill[col] > upper_bound)][col]

    print(f"\n【{col}】")
    print(f"IQR Lower Bound: {lower_bound:.4f}")
    print(f"IQR Upper Bound: {upper_bound:.4f}")
    print(f"Number of Outliers: {len(outliers)}")

print("\n--- Fill DataFrame analysis complete ---")


import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


print("Data loading complete.")

print("--- Step 1: Set working directory and load data ---")
working_directory = r'/kaggle/working/'
os.chdir(working_directory)
print(f"✅ Working directory set to: {working_directory}")

test_processed_df = test_df.copy(deep=True)
print("✅ test_df deeply copied to test_processed_df for subsequent imputation.")

print("\n--- Step 2: Perform KNN imputation on binary missing variables (Stage_fear, Drained_after_socializing) ---")

mapping_binary_cols = {'No': 0, 'Yes': 1}

test_processed_df['Stage_fear_encoded'] = test_processed_df['Stage_fear'].map(mapping_binary_cols)
test_processed_df['Drained_after_socializing_encoded'] = test_processed_df['Drained_after_socializing'].map(mapping_binary_cols)

cols_to_impute_knn_encoded = ['Stage_fear_encoded', 'Drained_after_socializing_encoded']

features_for_knn = cols_to_impute_knn_encoded

knn_data_test = test_processed_df[features_for_knn].copy()

imputer_knn = KNNImputer(n_neighbors=5, weights='uniform')
imputed_knn_array = imputer_knn.fit_transform(knn_data_test)

imputed_knn_df_temp = pd.DataFrame(imputed_knn_array, columns=features_for_knn, index=test_processed_df.index)

for col_encoded in cols_to_impute_knn_encoded:
    test_processed_df[col_encoded] = imputed_knn_df_temp[col_encoded].round().astype(int)

test_processed_df['Stage_fear'] = test_processed_df['Stage_fear_encoded'].map({0: 'No', 1: 'Yes'})
test_processed_df['Drained_after_socializing'] = test_processed_df['Drained_after_socializing_encoded'].map({0: 'No', 1: 'Yes'})

test_processed_df.drop(columns=['Stage_fear_encoded', 'Drained_after_socializing_encoded'], inplace=True)

print("✅ Binary variables (Stage_fear, Drained_after_socializing) missing values filled using KNN.")
print("    Missing values in Stage_fear after imputation:", test_processed_df['Stage_fear'].isnull().sum())
print("    Missing values in Drained_after_socializing after imputation:", test_processed_df['Drained_after_socializing'].isnull().sum())

print("\n--- Step 3: Perform MICE imputation on numerical missing variables ---")

numerical_cols_to_impute = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

imputer_mice = IterativeImputer(
    initial_strategy='median',
    max_iter=10,
    random_state=42,
    estimator=RandomForestRegressor(n_jobs=-1, random_state=42)
)

data_for_mice_test = test_processed_df[numerical_cols_to_impute]

imputed_numerical_array = imputer_mice.fit_transform(data_for_mice_test)

test_processed_df[numerical_cols_to_impute] = imputed_numerical_array

for col in numerical_cols_to_impute:
    test_processed_df[col] = test_processed_df[col].round().astype(int)

print("✅ Numerical variables missing values filled using MICE.")

print("\n--- Step 4: Final check for missing values in test_processed_df ---")
final_missing_percentages = (test_processed_df.isnull().mean() * 100).sort_values(ascending=False)
print("Missing value percentage (%) for each column in test_processed_df after imputation:")
print(final_missing_percentages)

if final_missing_percentages.sum() == 0:
    print("✅ All missing values successfully imputed, test_processed_df has no missing values.")
else:
    print("⚠️ Warning: Missing values still exist after imputation. Please check the missing percentages above.")

print("\n--- Step 5: Save the imputed dataframe ---")
output_filename = os.path.join(working_directory, 'fill_test.csv')
test_processed_df.to_csv(output_filename, index=False)

print(f"✅ Imputed test_processed_df saved to: {output_filename}")
print("--- All operations complete ---")


import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

plt.rcParams["axes.unicode_minus"] = False

try:
    font = fm.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
    plt.rcParams["font.family"] = font.get_name()
except:
    try:
        font = fm.FontProperties(fname="C:/Windows/Fonts/simhei.ttf")
        plt.rcParams["font.family"] = font.get_name()
    except:
        print("Warning: Failed to load specified Chinese font, Chinese characters in charts may not display correctly.")

working_directory = r'/kaggle/working/'
os.chdir(working_directory)
print(f"Current working directory set to: {os.getcwd()}")

def load_data(filename):
    try:
        df = pd.read_csv(filename)
        print(f"'{filename}' successfully loaded with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: '{filename}' file not found, please check path or if the file exists.")
        exit()

Fill_train = load_data('fill.csv')
Fill_test = load_data('fill_test.csv')

print("\n\nBasic Information of Fill_train DataFrame")
print(f"Rows: {Fill_train.shape[0]}, Columns: {Fill_train.shape[1]}")
print("Preview of the first 5 rows:")
print(Fill_train.head())

print("\n\nBasic Information of Fill_test DataFrame")
print(f"Rows: {Fill_test.shape[0]}, Columns: {Fill_test.shape[1]}")
print("Preview of the first 5 rows:")
print(Fill_test.head())

print("\n\nMissing Value Percentage (%) for Fill_train")
print((Fill_train.isnull().mean() * 100).sort_values(ascending=False))

print("\n\nMissing Value Percentage (%) for Fill_test")
print((Fill_test.isnull().mean() * 100).sort_values(ascending=False))

numeric_cols = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

binary_cols = ['Stage_fear', 'Drained_after_socializing']

print("\n\nDescriptive Statistics Comparison (Fill_train vs Fill_test)")
print("\nFill_train")
print(Fill_train[numeric_cols].describe())
print("\nFill_test")
print(Fill_test[numeric_cols].describe())

print("\n\nDistribution Shape Analysis (Skewness & Kurtosis)")

skew_comparison = []
kurt_comparison = []

for col in numeric_cols:
    s_train = skew(Fill_train[col])
    k_train = kurtosis(Fill_train[col])
    s_test = skew(Fill_test[col])
    k_test = kurtosis(Fill_test[col])

    skew_diff = abs(s_train - s_test)
    kurt_diff = abs(k_train - k_test)

    skew_comparison.append({
        'Column Name': col,
        'Fill_train Skewness': s_train,
        'Fill_test Skewness': s_test,
        'Skewness Difference': skew_diff,
        'Difference Extent': 'Significant' if skew_diff > 0.5 else 'Moderate' if skew_diff > 0.2 else 'Minor'
    })

    kurt_comparison.append({
        'Column Name': col,
        'Fill_train Kurtosis': k_train,
        'Fill_test Kurtosis': k_test,
        'Kurtosis Difference': kurt_diff,
        'Difference Extent': 'Significant' if kurt_diff > 0.5 else 'Moderate' if kurt_diff > 0.2 else 'Minor'
    })

print("\n--- Skewness Difference Report ---")
print(pd.DataFrame(skew_comparison))

print("\n--- Kurtosis Difference Report ---")
print(pd.DataFrame(kurt_comparison))

def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] < lower) | (df[col] > upper)].shape[0]

print("\n\nOutlier Detection (IQR Method)")
for col in numeric_cols:
    outliers_train = detect_outliers(Fill_train, col)
    outliers_test = detect_outliers(Fill_test, col)
    print(f"{col} - Fill_train Outlier Count: {outliers_train}, Fill_test Outlier Count: {outliers_test}")

print("\n\nDistribution Visualization Comparison")
for col in numeric_cols:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(Fill_train[col], kde=True, color='blue')
    plt.title(f'{col} - Fill_train')

    plt.subplot(1, 2, 2)
    sns.histplot(Fill_test[col], kde=True, color='green')
    plt.title(f'{col} - Fill_test')

    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import os # Import os for path manipulation

# Define the standard Kaggle input directory for datasets
KAGGLE_INPUT_DIR = '/kaggle/input/playground-series-s5e7/'
# Define the Kaggle working directory for files generated by your notebook
KAGGLE_WORKING_DIR = '/kaggle/working/'

# Initialize DataFrames to None to prevent UnboundLocalError if loading fails
fill_test = None
test_df = None

try:
    # Attempt to load fill_test.csv.
    # It's common for 'fill_test.csv' to be an output from a previous imputation step,
    # so we first check the working directory.
    fill_test_path = os.path.join(KAGGLE_WORKING_DIR, 'fill_test.csv')
    if os.path.exists(fill_test_path):
        fill_test = pd.read_csv(fill_test_path)
        print(f"✅ 'fill_test' DataFrame loaded from {fill_test_path}.")
    else:
        # If not in working, check if it's part of the input dataset
        alt_fill_test_path = os.path.join(KAGGLE_INPUT_DIR, 'fill_test.csv')
        if os.path.exists(alt_fill_test_path):
            fill_test = pd.read_csv(alt_fill_test_path)
            print(f"✅ 'fill_test' DataFrame loaded from {alt_fill_test_path}.")
        else:
            raise FileNotFoundError(f"'fill_test.csv' not found at {fill_test_path} or {alt_fill_test_path}.")

    # Load test.csv from the competition input directory
    test_df_path = os.path.join(KAGGLE_INPUT_DIR, 'test.csv')
    if os.path.exists(test_df_path):
        test_df = pd.read_csv(test_df_path)
        print(f"✅ 'test_df' DataFrame loaded from {test_df_path}.")
    else:
        raise FileNotFoundError(f"'test.csv' not found at {test_df_path}.")

except FileNotFoundError as e:
    print(f"❌ Error: Required CSV file not found. Please ensure the dataset is added to your notebook and files exist. {e}")
    # Re-raise the exception to stop execution if critical files are missing
    raise
except Exception as e:
    print(f"❌ An unexpected error occurred during data loading: {e}")
    raise

# Ensure both dataframes are loaded before proceeding
if fill_test is None or test_df is None:
    print("Data loading failed. Exiting analysis.")
else:
    numeric_cols = [
        'Time_spent_Alone',
        'Social_event_attendance',
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]

    print("\n\n=== Distribution Shape Difference Analysis (Fill_test vs test_df) ===")

    skew_comparison = []
    kurt_comparison = []

    for col in numeric_cols:
        # Check if column exists in both dataframes before calculating skew/kurtosis
        if col in fill_test.columns and col in test_df.columns:
            s_fill = skew(fill_test[col])
            k_fill = kurtosis(fill_test[col])

            s_test = skew(test_df[col].dropna())
            k_test = kurtosis(test_df[col].dropna())

            skew_diff = abs(s_fill - s_test)
            kurt_diff = abs(k_fill - k_test)

            def diff_level(val):
                if val > 0.5:
                    return 'Significant'
                elif val > 0.2:
                    return 'Moderate'
                else:
                    return 'Minor'

            skew_comparison.append({
                'Column Name': col,
                'Fill_test Skewness': s_fill,
                'test_df Skewness': s_test,
                'Skewness Difference': skew_diff,
                'Difference Extent': diff_level(skew_diff)
            })

            kurt_comparison.append({
                'Column Name': col,
                'Fill_test Kurtosis': k_fill,
                'test_df Kurtosis': k_test,
                'Kurtosis Difference': kurt_diff,
                'Difference Extent': diff_level(kurt_diff)
            })
        else:
            print(f"⚠️ Warning: Column '{col}' not found in one or both DataFrames. Skipping comparison for this column.")

    skew_df = pd.DataFrame(skew_comparison)
    kurt_df = pd.DataFrame(kurt_comparison)

    print("\n--- Skewness Difference Report (Fill_test vs test_df) ---")
    print(skew_df)

    print("\n--- Kurtosis Difference Report (Fill_test vs test_df) ---")
    print(kurt_df)


import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer # Or any other imputation method you use

# Define Kaggle paths
KAGGLE_INPUT_DIR = '/kaggle/input/playground-series-s5e7/'
KAGGLE_WORKING_DIR = '/kaggle/working/'

print("--- Starting Imputation and Saving Filled Files ---")

# Load original train and test data
try:
    original_train_df = pd.read_csv(os.path.join(KAGGLE_INPUT_DIR, 'train.csv'))
    original_test_df = pd.read_csv(os.path.join(KAGGLE_INPUT_DIR, 'test.csv'))
    print(f"✅ Original 'train.csv' and 'test.csv' loaded from {KAGGLE_INPUT_DIR}.")
except FileNotFoundError as e:
    print(f"❌ Error loading original files: {e}. Ensure 'playground-series-s5e7' dataset is added.")
    raise # Stop if original files can't be loaded

# Identify numerical features (adjust this based on your actual data)
# For simplicity, using all numeric columns for imputation here
# You might have specific columns for imputation
numeric_cols_for_imputation = original_train_df.select_dtypes(include=np.number).columns.tolist()
# Exclude 'id' and 'Personality' from imputation if they don't have missing values
# (or if Personality is your target and shouldn't be imputed)
if 'id' in numeric_cols_for_imputation:
    numeric_cols_for_imputation.remove('id')
if 'Personality' in numeric_cols_for_imputation: # If Personality is numerical and needs imputation
    numeric_cols_for_imputation.remove('Personality')


# --- Perform Imputation (Example: Mean Imputation) ---
# Use SimpleImputer for numerical columns
imputer_numeric = SimpleImputer(strategy='mean')

# Fit on training data and transform both train and test
original_train_df[numeric_cols_for_imputation] = imputer_numeric.fit_transform(original_train_df[numeric_cols_for_imputation])
original_test_df[numeric_cols_for_imputation] = imputer_numeric.transform(original_test_df[numeric_cols_for_imputation])

print(f"✅ Numerical columns imputed using mean strategy.")

# You might also have categorical columns that need imputation.
# Example for categorical (using mode imputation):
categorical_cols_for_imputation = original_train_df.select_dtypes(include='object').columns.tolist()
if 'Personality' in categorical_cols_for_imputation: # If 'Personality' is categorical target, exclude from imputation
    categorical_cols_for_imputation.remove('Personality')

if categorical_cols_for_imputation:
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    original_train_df[categorical_cols_for_imputation] = imputer_categorical.fit_transform(original_train_df[categorical_cols_for_imputation])
    original_test_df[categorical_cols_for_imputation] = imputer_categorical.transform(original_test_df[categorical_cols_for_imputation])
    print(f"✅ Categorical columns imputed using most_frequent strategy.")
else:
    print("No categorical columns identified for imputation (or already imputed).")


# --- Save the imputed DataFrames to /kaggle/working/ ---
fill_train_file = os.path.join(KAGGLE_WORKING_DIR, 'fill_train.csv')
fill_test_file = os.path.join(KAGGLE_WORKING_DIR, 'fill_test.csv')

original_train_df.to_csv(fill_train_file, index=False)
original_test_df.to_csv(fill_test_file, index=False)

print(f"✅ 'fill_train.csv' saved to {fill_train_file}")
print(f"✅ 'fill_test.csv' saved to {fill_test_file}")

print("--- Imputation and File Saving Complete ---")

# You can add a quick check here:
print("\nMissing values in fill_train.csv after imputation:")
print(original_train_df.isnull().sum().sum()) # Should be 0 if all relevant columns are imputed
print("\nMissing values in fill_test.csv after imputation:")
print(original_test_df.isnull().sum().sum()) # Should be 0


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os

# --- Define Kaggle specific paths ---
# Input directory for original competition data (if you needed train.csv/test.csv directly)
KAGGLE_INPUT_DIR = '/kaggle/input/playground-series-s5e7/'
# Working directory for generated files (like imputed data or submission files)
KAGGLE_WORKING_DIR = '/kaggle/working/'

# Paths for the filled (imputed) datasets
# Assuming 'fill_train.csv' and 'fill_test.csv' have been saved to /kaggle/working/
fill_train_file = os.path.join(KAGGLE_WORKING_DIR, 'fill_train.csv')
fill_test_file = os.path.join(KAGGLE_WORKING_DIR, 'fill_test.csv')

# Initialize DataFrames to None to prevent UnboundLocalError if loading fails
fill_train_df = None
fill_test_df = None

try:
    # Attempt to load fill_train_df
    if os.path.exists(fill_train_file):
        fill_train_df = pd.read_csv(fill_train_file)
        print(f"✅ 'fill_train_df' loaded successfully from {fill_train_file}!")
    else:
        # Fallback: if 'fill_train.csv' is somehow in the input directory (less common for filled data)
        alt_fill_train_file = os.path.join(KAGGLE_INPUT_DIR, 'fill_train.csv')
        if os.path.exists(alt_fill_train_file):
            fill_train_df = pd.read_csv(alt_fill_train_file)
            print(f"✅ 'fill_train_df' loaded successfully from {alt_fill_train_file} (fallback)!")
        else:
            raise FileNotFoundError(f"Neither '{fill_train_file}' nor '{alt_fill_train_file}' found. "
                                    "Please ensure your imputation script saved them to /kaggle/working/.")

    # Attempt to load fill_test_df
    if os.path.exists(fill_test_file):
        fill_test_df = pd.read_csv(fill_test_file)
        print(f"✅ 'fill_test_df' loaded successfully from {fill_test_file}!")
    else:
        # Fallback: if 'fill_test.csv' is somehow in the input directory
        alt_fill_test_file = os.path.join(KAGGLE_INPUT_DIR, 'fill_test.csv')
        if os.path.exists(alt_fill_test_file):
            fill_test_df = pd.read_csv(alt_fill_test_file)
            print(f"✅ 'fill_test_df' loaded successfully from {alt_fill_test_file} (fallback)!")
        else:
            raise FileNotFoundError(f"Neither '{fill_test_file}' nor '{alt_fill_test_file}' found. "
                                    "Please ensure your imputation script saved them to /kaggle/working/.")

except FileNotFoundError as e:
    print(f"❌ Error: Required file not found. {e}")
    # In a notebook, raising the error is usually preferred over exit()
    raise
except Exception as e:
    print(f"❌ An unexpected error occurred during data loading: {e}")
    raise

# Proceed only if dataframes are successfully loaded
if fill_train_df is not None and fill_test_df is not None:
    print("\n--- fill_train_df first five rows ---")
    print(fill_train_df.head())
    print("\n--- fill_test_df first five rows ---")
    print(fill_test_df.head())

    print("\n--- fill_train_df column information ---")
    print(fill_train_df.info())
    print("\n--- fill_test_df column information ---")
    print(fill_test_df.info())

    train_ids = fill_train_df['id']
    test_ids = fill_test_df['id']

    X = fill_train_df.drop(['id', 'Personality'], axis=1)
    y = fill_train_df['Personality']
    X_test_final = fill_test_df.drop('id', axis=1)

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    print(f"\nNumerical Features: {numerical_features}")
    print(f"Categorical Features: {categorical_features}")

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"\nPersonality Original Classes: {label_encoder.classes_}")
    print(f"Personality Encoded Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # For XGBoost, 'use_label_encoder=False' is preferred and 'eval_metric' should be set.
    # The warning about use_label_encoder is standard in newer versions.
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', model)])

    param_grid = {
        'classifier__n_estimators': [100, 500], # Adjusted common search space values
        'classifier__learning_rate': [0.005, 0.1], # Adjusted common search space values
        'classifier__max_depth': [3,  9], # Adjusted common search space values
    }

    print("\n--- Starting Hyperparameter Tuning (GridSearchCV) ---")
    # Reduced n_splits for a quicker example run; increase for more robust tuning
    kf = KFold(n_splits=15, shuffle=True, random_state=42)
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y_encoded)

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-validation Accuracy: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    print("\n--- Evaluating Best Model on Full Training Set ---")
    y_pred_train_encoded = best_model.predict(X)
    y_pred_train = label_encoder.inverse_transform(y_pred_train_encoded)

    print("\nTraining Set Classification Report:")
    print(classification_report(y, y_pred_train))
    print("\nTraining Set Confusion Matrix:")
    print(confusion_matrix(y, y_pred_train))

    cv_scores = cross_val_score(best_model, X, y_encoded, cv=kf, scoring='accuracy', n_jobs=-1)
    print(f"\nCross-validation Accuracy (KFold={kf.n_splits}): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("\n--- Making Predictions on fill_test.csv ---")
    test_predictions_encoded = best_model.predict(X_test_final)
    test_predictions_personality = label_encoder.inverse_transform(test_predictions_encoded)

    submission_df = pd.DataFrame({'id': test_ids, 'Personality': test_predictions_personality})

    # Save the submission file to the Kaggle working directory
    output_file = os.path.join(KAGGLE_WORKING_DIR, 'NEW2_predictions_personality.csv')
    submission_df.to_csv(output_file, index=False)

    print(f"\nPrediction complete! Results saved to: {output_file}")
    print("\n--- First five rows of predictions ---")
    print(submission_df.head())
else:
    print("Skipping model training and prediction due to data loading errors.")

