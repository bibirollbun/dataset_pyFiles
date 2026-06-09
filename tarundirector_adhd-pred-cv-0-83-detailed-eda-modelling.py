# Importing Libraries

import warnings
warnings.filterwarnings("ignore")

import textwrap
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
from imblearn.over_sampling import RandomOverSampler
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
import networkx as nx
import time
import warnings
import seaborn as sns
from matplotlib.lines import Line2D
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'iframe'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


pip install pycirclize


nb_type = "Submission"


#ğŸ”� Ah-ha! You found the secret sauce! ğŸ�”


train_df_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
train_df_fcm= pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
train_df_Q = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
train_df_sol = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')

test_df_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
test_df_fcm = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
test_df_Q = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')


dict_df = pd.read_excel('/kaggle/input/widsdatathon2025/Data Dictionary.xlsx')

# Load data
dict_APQP_df = pd.read_excel('/kaggle/input/full-data-dictionaries/APQ_P.xlsx', header=None)
dict_ColorVision_df = pd.read_excel('/kaggle/input/full-data-dictionaries/ColorVision.xlsx', header=None)
dict_SDQ_df = pd.read_excel('/kaggle/input/full-data-dictionaries/SDQ.xlsx', header=None)

# Function to get the first value of the second row before setting header
def get_first_value_before_header(df, var_name):
    first_value = df.iloc[0, 0]  # Second row, first column (before setting header)
    print(f"{var_name}: {first_value}")

# Print first values
get_first_value_before_header(dict_APQP_df, "dict_APQP_df")
get_first_value_before_header(dict_ColorVision_df, "dict_ColorVision_df")
get_first_value_before_header(dict_SDQ_df, "dict_SDQ_df")

# Set second row as the header
dict_APQP_df.columns = dict_APQP_df.iloc[1]
dict_ColorVision_df.columns = dict_ColorVision_df.iloc[1]
dict_SDQ_df.columns = dict_SDQ_df.iloc[1]

# Drop the first two rows as they are now redundant
dict_APQP_df = dict_APQP_df[2:].reset_index(drop=True)
dict_ColorVision_df = dict_ColorVision_df[2:].reset_index(drop=True)
dict_SDQ_df = dict_SDQ_df[2:].reset_index(drop=True)


train_data = train_df_cat.merge(train_df_Q, on="participant_id", how="inner") \
                        .merge(train_df_sol, on="participant_id", how="inner")

test_data = test_df_cat.merge(test_df_Q, on="participant_id", how="inner")


train_data.head()


test_data.head()


dict_df['Field'] = dict_df['Field'].replace({'MRI_Track,Age_at_Scan': 'MRI_Track_Age_at_Scan'})


# Checking the number of rows and columns

num_train_rows, num_train_columns = train_data.shape

num_test_rows, num_test_columns = test_data.shape

print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")


# Count duplicate rows in train_data
train_duplicates = train_data.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = test_data.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")


# Creating a table for missing values, unique values and data types of the features

missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')

merged_df.style.background_gradient(cmap='viridis')


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the train dataset')
train_data.describe().T.style.background_gradient(cmap='viridis')


# Having a look at the description of all the numerical columns present in the dataset
print('Description of all the numerical columns present in the test dataset')
test_data.describe().T.style.background_gradient(cmap='viridis')


categorical_variables = ['Basic_Demos_Enroll_Year', 'Basic_Demos_Study_Site', 'PreInt_Demos_Fam_Child_Ethnicity', 'PreInt_Demos_Fam_Child_Race',
'MRI_Track_Scan_Location', 'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ',
'ColorVision_CV_Score', 'APQ_P_APQ_P_CP', 'SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Generating_Impact',
'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial']

numerical_variables = ['EHQ_EHQ_Total', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD', 'APQ_P_APQ_P_PM',
'APQ_P_APQ_P_PP', 'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Internalizing', 'MRI_Track_Age_at_Scan']

target_variables = ['ADHD_Outcome', 'Sex_F']


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# Define custom color palettes
box_palette = {'Train': '#F46D43', 'Test': '#66C2A5'}   # Dark green for Train, red for Test
hist_train_color = '#F46D43'  # Darkish green for Train histogram
hist_test_color = '#66C2A5'   # Use same color as before for Test histogram

# Palettes for the additional KDE plots
gender_palette = {"Male": "lightblue", "Female": "lightpink"}
adhd_palette = {"Non-ADHD": "grey", "ADHD": "#FFDB58"}

# Add 'Dataset' column to distinguish between train and test data
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'

# Ensure we only analyze the numerical variables
variables = [col for col in train_data.columns if col in numerical_variables]

# Create new columns for gender and ADHD status if not already present.
# Assuming train_data has a binary "Sex_F" column where 1 represents Female.
if 'Gender' not in train_data.columns:
    train_data['Gender'] = train_data['Sex_F'].apply(lambda x: "Female" if x == 1 else "Male")

# Assuming "ADHD_Outcome" is binary with 1 meaning ADHD and 0 meaning Non-ADHD.
if 'ADHD_Status' not in train_data.columns:
    train_data['ADHD_Status'] = train_data['ADHD_Outcome'].apply(lambda x: "ADHD" if x == 1 else "Non-ADHD")

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')

    # Create a 1x4 subplot: box plot, histogram, gender KDE, ADHD KDE
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # ---------------------
    # 1. Box plot (Train & Test Combined)
    # ---------------------
    combined_data = pd.concat([train_data, test_data])
    sns.boxplot(ax=axes[0], data=combined_data, x=variable, y="Dataset", palette=box_palette)
    axes[0].set_xlabel(variable)
    title_box = f"Box Plot for {dict_df.loc[dict_df['Field'] == variable, 'Description'].values[0]}  [TRAIN & TEST Combined]"
    axes[0].set_title("\n".join(textwrap.wrap(title_box, width=50)))

    # ---------------------
    # 2. Histogram (Countplot) for Train vs Test
    # ---------------------
    sns.histplot(ax=axes[1], data=train_data, x=variable, color=hist_train_color, kde=True, bins=30, label="Train")
    sns.histplot(ax=axes[1], data=test_data, x=variable, color=hist_test_color, kde=True, bins=30, label="Test")
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("Frequency")
    title_hist = f"Histogram for {variable}:  {dict_df.loc[dict_df['Field'] == variable, 'Description'].values[0]} [TRAIN & TEST]"
    axes[1].set_title("\n".join(textwrap.wrap(title_hist, width=50)))
    axes[1].legend()

    # ---------------------
    # 3. KDE Plot by Gender (Male vs Female)
    # ---------------------
    sns.kdeplot(ax=axes[2], data=train_data, x=variable, hue="Gender", fill=True, common_norm=False,
                palette=gender_palette, alpha=0.4, linewidth=2)
    axes[2].set_xlabel(variable)
    axes[2].set_title(f"KDE by Gender for {variable}")

    # ---------------------
    # 4. KDE Plot by ADHD Status (ADHD vs Non-ADHD)
    # ---------------------
    sns.kdeplot(ax=axes[3], data=train_data, x=variable, hue="ADHD_Status", fill=True, common_norm=False,
                palette=adhd_palette, alpha=0.4, linewidth=2)
    axes[3].set_xlabel(variable)
    axes[3].set_title(f"KDE by ADHD Status for {variable}")

    # Adjust spacing and show the plots
    plt.tight_layout()
    plt.show()

# Perform univariate analysis for each variable in the list
for variable in variables:
    create_variable_plots(variable)

# Clean up: Drop the 'Dataset' column after analysis if desired
train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# -----------------------
# Merge Features with Solutions
# -----------------------
train_features_df = pd.read_csv('/kaggle/input/full-data-dictionaries/train_connectome_features.csv')
train_merged = pd.merge(train_features_df, train_df_sol, on='participant_id')

# -----------------------
# Define Colors for ADHD and Sex outcomes
# -----------------------
# For ADHD: 0 = grey, 1 = yellow
adhd_colors = {0: '#808080', 1: '#f1c40f'}
# For Sex: 0 = blue (male), 1 = pink (female)
sex_colors = {0: '#3498db', 1: '#e91e63'}

# -----------------------
# Define List of Feature Columns to Plot
# -----------------------
plot_columns = [
    'mean_degree', 'std_degree', 'mean_strength', 'std_strength',
    'mean_betweenness', 'std_betweenness', 'avg_clustering',
    'characteristic_path_length', 'global_efficiency', 'modularity',
    'small_worldness', 'num_connected_components'
]

# -----------------------
# Plotting Loop: For each feature, create 4 subplots (2x2)
# -----------------------
for col in plot_columns:
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    # Use participant_id as x-axis if numeric; otherwise, use the index.
    if pd.api.types.is_numeric_dtype(train_merged['participant_id']):
        x_vals = train_merged['participant_id']
    else:
        x_vals = train_merged.index

    # --- Top Left: Scatter Plot for ADHD Outcome ---
    axs[0, 0].scatter(
        x_vals,
        train_merged[col],
        c=train_merged['ADHD_Outcome'].map(adhd_colors),
        alpha=0.7
    )
    axs[0, 0].set_title(f"Scatter: {col} vs Participant ID (ADHD)", fontsize=16, fontweight='bold')
    axs[0, 0].set_xlabel("Participant ID", fontsize=14)
    axs[0, 0].set_ylabel(col, fontsize=14)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    legend_elements_adhd = [
        Line2D([0], [0], marker='o', color='w', label='Non-ADHD',
               markerfacecolor=adhd_colors[0], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='ADHD',
               markerfacecolor=adhd_colors[1], markersize=10)
    ]
    axs[0, 0].legend(handles=legend_elements_adhd, title="ADHD Outcome", fontsize=12, title_fontsize=12)

    # --- Top Right: KDE Plot for ADHD Outcome ---
    sns.kdeplot(
        data=train_merged,
        x=col,
        hue='ADHD_Outcome',
        palette=adhd_colors,
        fill=True,
        common_norm=False,
        alpha=0.6,
        ax=axs[0, 1]
    )
    axs[0, 1].set_title(f"KDE: {col} by ADHD Outcome", fontsize=16, fontweight='bold')
    axs[0, 1].set_xlabel(col, fontsize=14)
    axs[0, 1].set_ylabel("Density", fontsize=14)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # --- Bottom Left: Scatter Plot for Sex Outcome ---
    axs[1, 0].scatter(
        x_vals,
        train_merged[col],
        c=train_merged['Sex_F'].map(sex_colors),
        alpha=0.7
    )
    axs[1, 0].set_title(f"Scatter: {col} vs Participant ID (Sex)", fontsize=16, fontweight='bold')
    axs[1, 0].set_xlabel("Participant ID", fontsize=14)
    axs[1, 0].set_ylabel(col, fontsize=14)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    legend_elements_sex = [
        Line2D([0], [0], marker='o', color='w', label='Male',
               markerfacecolor=sex_colors[0], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Female',
               markerfacecolor=sex_colors[1], markersize=10)
    ]
    axs[1, 0].legend(handles=legend_elements_sex, title="Sex", fontsize=12, title_fontsize=12)

    # --- Bottom Right: KDE Plot for Sex Outcome ---
    sns.kdeplot(
        data=train_merged,
        x=col,
        hue='Sex_F',
        palette=sex_colors,
        fill=True,
        common_norm=False,
        alpha=0.6,
        ax=axs[1, 1]
    )
    axs[1, 1].set_title(f"KDE: {col} by Sex", fontsize=16, fontweight='bold')
    axs[1, 1].set_xlabel(col, fontsize=14)
    axs[1, 1].set_ylabel("Density", fontsize=14)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# -----------------------------
# 1. Merge and Aggregate Data
# -----------------------------
# (Assuming train_df_fcm and train_df_sol are already loaded)

# Merge the connectome data with the solution data on 'participant_id'
merged_df = pd.merge(train_df_fcm, train_df_sol, on='participant_id')

# Identify connectivity columns (exclude 'participant_id')
conn_cols = [col for col in train_df_fcm.columns if col != 'participant_id']

# Function to sort connectome columns based on node indices extracted from names.
def sort_connectome_columns(columns):
    def parse_col(col):
        # Expected pattern: 'ithrow_jthcolumn' where i and j are integers
        m = re.match(r"(\d+)throw_(\d+)thcolumn", col)
        if m:
            i = int(m.group(1))
            j = int(m.group(2))
            return (i, j)
        else:
            return (float('inf'), float('inf'))
    return sorted(columns, key=parse_col)

sorted_conn_cols = sort_connectome_columns(conn_cols)

# Aggregate connectivity data by ADHD_Outcome and Sex_F (using mean)
# (Assuming in train_df_sol: ADHD_Outcome (0/1) and Sex_F (0=Male, 1=Female))
adhd_groups = merged_df.groupby('ADHD_Outcome')[sorted_conn_cols].mean()
sex_groups  = merged_df.groupby('Sex_F')[sorted_conn_cols].mean()

# Function to convert a connectivity vector (of length n*(n-1)/2) into a symmetric matrix.
def vector_to_symmetric_matrix(vector, n=200):
    mat = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = vector[idx]
            mat[j, i] = vector[idx]
            idx += 1
    return mat

# Create aggregated connectivity matrices:
# For ADHD groups: index 1 means ADHD-positive, index 0 means nonâ€‘ADHD.
adhd_positive_matrix = vector_to_symmetric_matrix(adhd_groups.loc[1].values, n=200)
adhd_negative_matrix = vector_to_symmetric_matrix(adhd_groups.loc[0].values, n=200)
# For Sex groups: index 1 means Female, index 0 means Male.
female_matrix = vector_to_symmetric_matrix(sex_groups.loc[1].values, n=200)
male_matrix   = vector_to_symmetric_matrix(sex_groups.loc[0].values, n=200)

# Compute difference matrices
adhd_diff_matrix = adhd_positive_matrix - adhd_negative_matrix
sex_diff_matrix  = female_matrix - male_matrix

# -----------------------------
# 2. Continuous Heatmap Plots (3 per row)
# -----------------------------
# We'll create 2 rows (one for ADHD groups and one for Sex groups) and 3 columns per row.
fig, axes = plt.subplots(2, 3, figsize=(24, 12))

# Define a function to plot a matrix with a lower-triangular mask.
def plot_lower_triangle(ax, matrix, title):
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, mask=mask, cmap="coolwarm", square=True, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_title(title)

# ADHD row: column 0: ADHD Positive, 1: ADHD Negative, 2: Difference
plot_lower_triangle(axes[0, 0], adhd_positive_matrix, "ADHD Positive Aggregated Connectome")
plot_lower_triangle(axes[0, 1], adhd_negative_matrix, "ADHD Negative Aggregated Connectome")
plot_lower_triangle(axes[0, 2], adhd_diff_matrix, "Difference (ADHD Positive - Negative)")

# Sex row: column 0: Female, 1: Male, 2: Difference
plot_lower_triangle(axes[1, 0], female_matrix, "Female Aggregated Connectome")
plot_lower_triangle(axes[1, 1], male_matrix, "Male Aggregated Connectome")
plot_lower_triangle(axes[1, 2], sex_diff_matrix, "Difference (Female - Male)")

plt.tight_layout()
plt.show()

# -----------------------------
# 3. Thresholding and Binary Graphs
# -----------------------------
# For binary graphs, we threshold the matrices (e.g. keep only connections above the 75th percentile)
def threshold_binary(matrix, percentile=75):
    threshold_value = np.percentile(matrix, percentile)
    binary_matrix = (matrix > threshold_value).astype(int)
    return binary_matrix

# Compute binary versions for ADHD groups
binary_adhd_positive = threshold_binary(adhd_positive_matrix, percentile=75)
binary_adhd_negative = threshold_binary(adhd_negative_matrix, percentile=75)
# Difference as binary: subtracting yields -1, 0, or 1
binary_adhd_diff = binary_adhd_positive - binary_adhd_negative

# Compute binary versions for Sex groups
binary_female = threshold_binary(female_matrix, percentile=75)
binary_male   = threshold_binary(male_matrix, percentile=75)
binary_sex_diff = binary_female - binary_male

# Now, create binary heatmaps in a similar layout (2 rows x 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(24, 12))

# Define a function to plot a binary matrix.
def plot_binary_heatmap(ax, matrix, title):
    # Using a diverging palette so that -1, 0, 1 can be seen.
    # Here, we'll use a custom discrete colormap: -1 (blue), 0 (white), 1 (red)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(matrix, cmap=cmap, square=True, cbar=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title(title)

# ADHD row: binary plots
plot_binary_heatmap(axes[0, 0], binary_adhd_positive, "Binary: ADHD Positive")
plot_binary_heatmap(axes[0, 1], binary_adhd_negative, "Binary: ADHD Negative")
plot_binary_heatmap(axes[0, 2], binary_adhd_diff, "Binary Difference (ADHD)")

# Sex row: binary plots
plot_binary_heatmap(axes[1, 0], binary_female, "Binary: Female")
plot_binary_heatmap(axes[1, 1], binary_male, "Binary: Male")
plot_binary_heatmap(axes[1, 2], binary_sex_diff, "Binary Difference (Sex)")

plt.tight_layout()
plt.show()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pycirclize import Circos

# -----------------------------
# Helper Functions
# -----------------------------
def aggregate_matrix_by_bins(matrix, num_bins=10):
    """
    Aggregates a full connectome matrix (e.g., 200x200) into a smaller
    num_bins x num_bins matrix by averaging over blocks.
    """
    n = matrix.shape[0]
    bin_size = n // num_bins
    agg_matrix = np.zeros((num_bins, num_bins))
    for i in range(num_bins):
        for j in range(num_bins):
            block = matrix[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size]
            agg_matrix[i, j] = block.mean()  # Use mean connection strength
    return agg_matrix

def save_chord_diagram(matrix, title, filename, cmap='coolwarm', r_lim=(93, 100)):
    """
    Generates a chord diagram for the (binned) connectome matrix using pycirclize,
    then saves the resulting figure to a file.
    """
    # Aggregate the full matrix into bins
    agg_matrix = aggregate_matrix_by_bins(matrix, num_bins=10)
    bin_labels = [f'Bin {i+1}' for i in range(10)]
    agg_df = pd.DataFrame(agg_matrix, index=bin_labels, columns=bin_labels)

    # Create the chord diagram with pycirclize
    circos = Circos.chord_diagram(
        agg_df,
        start=-265,
        end=95,
        space=5,
        r_lim=r_lim,
        cmap=cmap,
        label_kws=dict(r=r_lim[0]-1, size=10, color="black"),
        link_kws=dict(ec="black", lw=0.5),
    )
    # (Optional) You can set the title here on the circos figure if needed:
    # fig = circos.plotfig(suptitle=title)
    fig = circos.plotfig()
    fig.savefig(filename)
    plt.close(fig)

# -----------------------------
# Example Aggregated Matrices
# -----------------------------
# (Assume adhd_positive_matrix, adhd_negative_matrix, female_matrix, male_matrix
#  have been previously defined, for example using vector_to_symmetric_matrix)

# -----------------------------
# Save Chord Diagrams as Images
# -----------------------------
image_files = {
    "Chord Diagram: ADHD Positive": "chord_adhd_positive.png",
    "Chord Diagram: ADHD Negative": "chord_adhd_negative.png",
    "Chord Diagram: Female": "chord_female.png",
    "Chord Diagram: Male": "chord_male.png",
}

save_chord_diagram(adhd_positive_matrix, "Chord Diagram: ADHD Positive", image_files["Chord Diagram: ADHD Positive"])
save_chord_diagram(adhd_negative_matrix, "Chord Diagram: ADHD Negative", image_files["Chord Diagram: ADHD Negative"])
save_chord_diagram(female_matrix, "Chord Diagram: Female", image_files["Chord Diagram: Female"])
save_chord_diagram(male_matrix, "Chord Diagram: Male", image_files["Chord Diagram: Male"])

# Close any lingering figures before creating the subplot grid
plt.close('all')

# -----------------------------
# Load Images into a 2x2 Subplot Grid
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(16, 16))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

titles = list(image_files.keys())
for ax, title in zip(axs.flatten(), titles):
    img = plt.imread(image_files[title])
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Optionally, delete the temporary image files after display:
for file in image_files.values():
    if os.path.exists(file):
        os.remove(file)


import plotly.io as pio
pio.renderers.default = 'iframe'  # Use iframe renderer for published notebooks

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.decomposition import PCA

# Create a copy of merged_df so the original remains unchanged.
df_pca = merged_df.copy()

# --- Assume sorted_conn_cols is already defined ---
X = df_pca[sorted_conn_cols].values  # shape: (n_subjects, 19900)

# Perform PCA with 3 components for the 3D plot
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Add the components to our copy (df_pca) without modifying the original merged_df
df_pca = df_pca.copy()  # Ensure we're working on our local copy
df_pca['pca1'] = X_pca[:, 0]
df_pca['pca2'] = X_pca[:, 1]
df_pca['pca3'] = X_pca[:, 2]

# Convert target labels to string-based categorical columns
df_pca['ADHD_Label'] = df_pca['ADHD_Outcome'].map({0: "Non-ADHD", 1: "ADHD"})
df_pca['Sex_Label'] = df_pca['Sex_F'].map({0: "Male", 1: "Female"})

# Define color palettes for the targets
gender_palette = {"Male": "lightblue", "Female": "lightpink"}
adhd_palette = {"Non-ADHD": "grey", "ADHD": "#FFDB58"}

# -----------------------
# 3D PCA Plot: Two Subplots
# -----------------------
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=("3D PCA: Colored by ADHD", "3D PCA: Colored by Sex")
)

# Plot 1: Colored by ADHD Outcome
for label in df_pca['ADHD_Label'].unique():
    df_subset = df_pca[df_pca['ADHD_Label'] == label]
    fig.add_trace(
        go.Scatter3d(
            x=df_subset['pca1'],
            y=df_subset['pca2'],
            z=df_subset['pca3'],
            mode='markers',
            name=label,
            marker=dict(size=4, color=adhd_palette[label]),
            legendgroup="ADHD"
        ),
        row=1, col=1
    )

# Plot 2: Colored by Sex
for label in df_pca['Sex_Label'].unique():
    df_subset = df_pca[df_pca['Sex_Label'] == label]
    fig.add_trace(
        go.Scatter3d(
            x=df_subset['pca1'],
            y=df_subset['pca2'],
            z=df_subset['pca3'],
            mode='markers',
            name=label,
            marker=dict(size=4, color=gender_palette[label]),
            legendgroup="Sex"
        ),
        row=1, col=2
    )

# Layout adjustments for the 3D plot
fig.update_layout(
    height=600,
    width=900,
    legend_title_text="Groups",
    scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
    scene2=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
    margin=dict(l=10, r=10, t=40, b=10)
)

fig.show()  # Ensure the plot is displayed


# -----------------------
# 2D PCA Plot: Two Subplots
# -----------------------
# Perform PCA with 2 components for the 2D plot on a fresh copy to avoid altering previous columns.
df_pca2 = merged_df.copy()
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(df_pca2[sorted_conn_cols].values)
df_pca2['pca2_1'] = X_pca2[:, 0]
df_pca2['pca2_2'] = X_pca2[:, 1]

fig_2d = make_subplots(
    rows=1, cols=2,
    subplot_titles=("2D PCA Colored by ADHD", "2D PCA Colored by Sex")
)

# Plot 1: Colored by ADHD Outcome in 2D
for label in df_pca2['ADHD_Outcome'].map({0: "Non-ADHD", 1: "ADHD"}).unique():
    subset = df_pca2[df_pca2['ADHD_Outcome'].map({0: "Non-ADHD", 1: "ADHD"}) == label]
    fig_2d.add_trace(
        go.Scatter(
            x=subset['pca2_1'],
            y=subset['pca2_2'],
            mode='markers',
            marker=dict(color=adhd_palette[label], size=5),
            name=label,
            legendgroup="ADHD"
        ),
        row=1, col=1
    )

# Plot 2: Colored by Sex in 2D
for label in df_pca2['Sex_F'].map({0: "Male", 1: "Female"}).unique():
    subset = df_pca2[df_pca2['Sex_F'].map({0: "Male", 1: "Female"}) == label]
    fig_2d.add_trace(
        go.Scatter(
            x=subset['pca2_1'],
            y=subset['pca2_2'],
            mode='markers',
            marker=dict(color=gender_palette[label], size=5),
            name=label,
            legendgroup="Sex"
        ),
        row=1, col=2
    )

fig_2d.update_layout(
    height=600,
    width=900,
    legend_title_text="Groups",
    xaxis_title="PC1",
    yaxis_title="PC2",
    xaxis2_title="PC1",
    yaxis2_title="PC2",
    margin=dict(l=10, r=10, t=40, b=10)
)
fig_2d.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap

# Define color palettes for target groups and for datasets
adhd_palette = {"Non-ADHD": "grey", "ADHD": "#FFDB58"}
gender_palette = {"Male": "lightblue", "Female": "lightpink"}
dataset_palette = ['#33638d', '#28ae80']  # For train and test respectively

# Ensure you have copies of train_data and test_data
train_data = train_data.copy()
test_data = test_data.copy()

# Add a 'dataset' column to differentiate train and test data
train_data['dataset'] = 'train'
test_data['dataset'] = 'test'

# Combine train and test for the dataset-specific plot
combined = pd.concat([train_data, test_data])

def create_categorical_plots(feature):
    sns.set_style('whitegrid')

    # Create a figure with 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # ---------------------
    # Plot 1: Overall Pie Chart
    # ---------------------
    value_counts = combined[feature].value_counts()
    threshold = 0.05 * value_counts.sum()
    filtered_values = value_counts[value_counts >= threshold]
    if value_counts[value_counts < threshold].sum() > 0:
        filtered_values['Other'] = value_counts[value_counts < threshold].sum()

    wedges, texts, autotexts = axes[0].pie(
        filtered_values,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        colors=sns.color_palette("viridis", len(filtered_values)),
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=[0.05 if count > threshold else 0 for count in filtered_values],
        textprops={'fontsize': 10}
    )
    axes[0].set_title("\n".join(textwrap.wrap(
        f"Pie Chart for {feature}: {dict_df.loc[dict_df['Field'] == feature, 'Description'].values[0]}", width=50)))

    # ---------------------
    # Plot 2: Countplot by Dataset (Train vs Test)
    # ---------------------
    sns.countplot(
        data=combined,
        x=feature,
        hue='dataset',
        palette=dataset_palette,
        ax=axes[1]
    )
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Count")
    axes[1].set_title("\n".join(textwrap.wrap(f"Countplot for {feature} by Dataset", width=50)))
    axes[1].tick_params(axis='x', rotation=30)

    # For the following plots, ensure the feature remains categorical (do not convert to numeric)
    
    # ---------------------
    # Plot 3: Countplot for ADHD Outcome (Train only)
    # ---------------------
    train_data['ADHD_Label'] = train_data['ADHD_Outcome'].map({0: "Non-ADHD", 1: "ADHD"})
    sns.countplot(
        data=train_data,
        x=feature,
        hue='ADHD_Label',
        palette=adhd_palette,
        ax=axes[2]
    )
    axes[2].set_xlabel(feature)
    axes[2].set_ylabel("Count")
    axes[2].set_title("\n".join(textwrap.wrap(f"Distribution of {feature} by ADHD Outcome", width=50)))
    axes[2].legend()

    # ---------------------
    # Plot 4: Countplot for Sex (Train only)
    # ---------------------
    train_data['Sex_Label'] = train_data['Sex_F'].map({0: "Male", 1: "Female"})
    sns.countplot(
        data=train_data,
        x=feature,
        hue='Sex_Label',
        palette=gender_palette,
        ax=axes[3]
    )
    axes[3].set_xlabel(feature)
    axes[3].set_ylabel("Count")
    axes[3].set_title("\n".join(textwrap.wrap(f"Distribution of {feature} by Sex", width=50)))
    axes[3].legend()

    plt.tight_layout()
    plt.show()

# Perform univariate analysis for each categorical variable
for feature in categorical_variables:
    create_categorical_plots(feature)

# Cleanup: Drop temporary columns
train_data.drop(['dataset', 'ADHD_Label', 'Sex_Label'], axis=1, inplace=True)
test_data.drop(['dataset'], axis=1, inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap

# Define default color palettes for non-target variables
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779',
                       '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']
countplot_color = '#5C67A3'

# Define custom palettes for the target variables
sex_color_map = {0: 'lightblue', 1: 'lightpink'}
adhd_color_map = {0: 'grey', 1: '#FFDB58'}

# Function to create and display a row of plots for a single categorical variable
def create_categorical_plots(variable):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---------------------
    # Pie Chart - Handling many categories
    # ---------------------
    plt.subplot(1, 2, 1)

    # Get combined counts from train and test
    combined = pd.concat([train_data, test_data])
    value_counts = combined[variable].value_counts()
    total = value_counts.sum()

    # For target variables, enforce an order and custom palette
    if variable == 'Sex_F':
        order = [0, 1]  # Male then Female
        value_counts = value_counts.reindex(order).dropna()
        custom_pie_palette = [sex_color_map[val] for val in order if val in value_counts.index]
    elif variable == 'ADHD_Outcome':
        order = [0, 1]  # Non-ADHD then ADHD
        value_counts = value_counts.reindex(order).dropna()
        custom_pie_palette = [adhd_color_map[val] for val in order if val in value_counts.index]
    else:
        custom_pie_palette = pie_chart_palette[:len(value_counts)]

    # Combine small categories (<5%) into "Other" (only for non-target variables)
    threshold = 0.05 * total
    if variable not in ['Sex_F', 'ADHD_Outcome']:
        filtered_values = value_counts[value_counts >= threshold]
        other_total = value_counts[value_counts < threshold].sum()
        if other_total > 0:
            filtered_values['Other'] = other_total
        value_counts = filtered_values
        # Adjust palette and explode for the filtered categories
        custom_pie_palette = pie_chart_palette[:len(value_counts)]
        explode = [0.05 if count >= threshold else 0 for count in value_counts]
    else:
        explode = [0.05] * len(value_counts)  # Slight explode for both bars

    wedges, texts, autotexts = plt.pie(
        value_counts,
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',  # Hide labels < 5%
        colors=custom_pie_palette,
        startangle=140,
        wedgeprops=dict(width=0.3),
        explode=explode,
        textprops={'fontsize': 10}
    )

    title_text = dict_df.loc[dict_df['Field'] == variable, 'Description'].values[0] \
                    if variable in dict_df['Field'].values else variable
    plt.title("\n".join(textwrap.wrap(f"Pie Chart for {title_text}  [TRAIN]", width=50)))
    plt.legend(value_counts.index, loc="upper left", bbox_to_anchor=(1, 1))

    # ---------------------
    # Bar Graph (Countplot)
    # ---------------------
    plt.subplot(1, 2, 2)
    # For target variables, use a custom palette; otherwise, use default color.
    if variable == 'Sex_F':
        order = [0, 1]
        sns.countplot(
            data=combined,
            x=variable,
            palette=sex_color_map,
            order=order
        )
    elif variable == 'ADHD_Outcome':
        order = [0, 1]
        sns.countplot(
            data=combined,
            x=variable,
            palette=adhd_color_map,
            order=order
        )
    else:
        sns.countplot(
            data=combined,
            x=variable,
            color=countplot_color,
            alpha=0.8
        )

    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title("\n".join(textwrap.wrap(f"Bar Graph for {title_text}  [TRAIN]", width=50)))
    plt.xticks(rotation=30)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Perform univariate analysis for each categorical variable in your target_variables list
for variable in target_variables:
    create_categorical_plots(variable)


# Adding variables to the existing list
test_variables = categorical_variables+numerical_variables
train_variables = categorical_variables+numerical_variables+ target_variables

# Calculate correlation matrices for train_data and test_data
corr_train = train_data[train_variables].corr()
corr_test = test_data[test_variables].corr()

# Create masks for the upper triangle
mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

# Set the text size and rotation
annot_kws = {"size": 6, "rotation": 45}

# Generate heatmaps for train_data
plt.figure(figsize=(10, 20))
plt.subplot(2, 1, 1)
ax_train = sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
                      square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data')

# Generate heatmaps for test_data
plt.subplot(2, 1, 2)
ax_test = sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
                     square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Create a contingency table (crosstab)
crosstab = pd.crosstab(train_data['Sex_F'], train_data['ADHD_Outcome'])
# Optionally, you can convert counts to percentages if desired:
crosstab_percent = crosstab.apply(lambda r: r / r.sum() * 100, axis=1)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, fmt="d", cmap="viridis")
plt.xlabel("ADHD Outcome (0: Non-ADHD, 1: ADHD)")
plt.ylabel("Sex (0: Male, 1: Female)")
plt.title("Crosstab of Sex vs. ADHD Outcome")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix
test_variables = categorical_variables + numerical_variables
train_variables = categorical_variables + numerical_variables + target_variables
corr_train = train_data[train_variables].corr()[target_variables]

# Setup for vertical bar plots (features on x-axis)
num_targets = len(target_variables)
fig, axs = plt.subplots(num_targets, 1, figsize=(len(train_variables) * 0.3 + 2, 8 * num_targets), constrained_layout=True)

if num_targets == 1:
    axs = [axs]

for i, target in enumerate(target_variables):
    sorted_corr = corr_train[target].drop(target).sort_values(ascending=False)
    colors = sns.color_palette("viridis", n_colors=len(sorted_corr))

    sns.barplot(y=sorted_corr.values, x=sorted_corr.index, palette=colors, ax=axs[i])
    axs[i].set_title(f"Correlation with {target}", fontsize=14)
    axs[i].set_ylabel("Correlation", fontsize=12)
    axs[i].set_xlabel("Features", fontsize=12)
    axs[i].tick_params(axis='x', rotation=90)
    axs[i].tick_params(axis='y', labelsize=10)

plt.suptitle("Feature Correlations with Target Variables", fontsize=16, y=1.02)
plt.show()


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LOG_VARS = [
    "EHQ_EHQ_Total",
    "APQ_P_APQ_P_ID", "APQ_P_APQ_P_INV", "APQ_P_APQ_P_OPD",
    "APQ_P_APQ_P_PM", "APQ_P_APQ_P_PP",
    "SDQ_SDQ_Difficulties_Total",
    "SDQ_SDQ_Externalizing", "SDQ_SDQ_Internalizing",
    "SDQ_SDQ_Conduct_Problems", "SDQ_SDQ_Emotional_Problems",
    "SDQ_SDQ_Hyperactivity", "SDQ_SDQ_Peer_Problems",
    "SDQ_SDQ_Generating_Impact", "SDQ_SDQ_Prosocial"
]

PARENT_EDU    = ["Barratt_Barratt_P1_Edu", "Barratt_Barratt_P2_Edu"]
PARENT_OCC    = ["Barratt_Barratt_P1_Occ", "Barratt_Barratt_P2_Occ"]
PARENT_SCORES = ["APQ_P_APQ_P_CP", "APQ_P_APQ_P_PM", "APQ_P_APQ_P_PP"]
SDQ_EXTERNAL  = ["SDQ_SDQ_Conduct_Problems", "SDQ_SDQ_Hyperactivity"]
SDQ_INTERNAL  = ["SDQ_SDQ_Emotional_Problems", "SDQ_SDQ_Peer_Problems"]

def add_engineered_features(train_df: pd.DataFrame,
                            test_df:  pd.DataFrame):
    # SES composites
    ses_cols   = PARENT_EDU + PARENT_OCC
    ses_scaler = StandardScaler().fit(train_df[ses_cols].fillna(0))
    for df in (train_df, test_df):
        df["SES_zmean"]        = ses_scaler.transform(df[ses_cols].fillna(0)).mean(axis=1)
        df["SES_gap"]          = (df[PARENT_EDU[0]] - df[PARENT_EDU[1]]).abs() \
                                + (df[PARENT_OCC[0]] - df[PARENT_OCC[1]]).abs()
        df["SES_missing_cnt"]  = df[ses_cols].isna().sum(axis=1)

    # Parentingâ€�style axis
    ps_scaler = StandardScaler().fit(train_df[PARENT_SCORES].fillna(0))
    for df in (train_df, test_df):
        z      = ps_scaler.transform(df[PARENT_SCORES].fillna(0))
        zdf    = pd.DataFrame(z, columns=[c + "_z" for c in PARENT_SCORES], index=df.index)
        df[zdf.columns] = zdf
        df["Parenting_harsh_vs_pos"] = df["APQ_P_APQ_P_CP_z"] - df["APQ_P_APQ_P_PP_z"]

    # SDQ aggregates & ratio
    for df in (train_df, test_df):
        df["SDQ_external_sum"] = df[SDQ_EXTERNAL].sum(axis=1)
        df["SDQ_internal_sum"] = df[SDQ_INTERNAL].sum(axis=1)
        df["SDQ_ext_int_ratio"] = df["SDQ_external_sum"] / (df["SDQ_internal_sum"] + 1e-3)

    # Temporal & domainâ€�shift guards
    med_year = train_df["Basic_Demos_Enroll_Year"].median()
    for df in (train_df, test_df):
        df["Enroll_recency"]  = df["Basic_Demos_Enroll_Year"] - med_year
        df["Enroll_post2020"] = (df["Basic_Demos_Enroll_Year"] >= 2020).astype(int)
    seen_sites = train_df["Basic_Demos_Study_Site"].unique()
    seen_locs  = train_df["MRI_Track_Scan_Location"].unique()
    test_df["Unseen_site"]     = (~test_df["Basic_Demos_Study_Site"].isin(seen_sites)).astype(int)
    test_df["Unseen_scan_loc"] = (~test_df["MRI_Track_Scan_Location"].isin(seen_locs)).astype(int)
    train_df["Unseen_site"]    = 0
    train_df["Unseen_scan_loc"]= 0

    # Age standardisation
    age_mean = train_df["MRI_Track_Age_at_Scan"].mean()
    age_std  = train_df["MRI_Track_Age_at_Scan"].std()
    for df in (train_df, test_df):
        df["Age_z"] = (df["MRI_Track_Age_at_Scan"] - age_mean) / age_std

    # Log transforms
    for col in LOG_VARS:
        for df in (train_df, test_df):
            if col in df.columns:
                df[col + "_log"] = np.log1p(df[col].clip(lower=0))

    # --- ensure no NaNs anywhere ---
    train_df.fillna(0, inplace=True)
    test_df.fillna(0,  inplace=True)

    return train_df, test_df


train_df, test_df = add_engineered_features(train_data, test_data)


# Define the features for median and mode imputations
median_features = ['MRI_Track_Age_at_Scan', 'EHQ_EHQ_Total']
mode_features = ['PreInt_Demos_Fam_Child_Ethnicity', 'PreInt_Demos_Fam_Child_Race', 'MRI_Track_Scan_Location', 'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ', 'ColorVision_CV_Score', 'APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD', 'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP', 'SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Generating_Impact', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing', 'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial']

# Impute missing values in the training data
for col in median_features:
    median_val = train_data[col].median()
    train_data[col] = train_data[col].fillna(median_val)

for col in mode_features:
    mode_val = train_data[col].mode()[0]
    train_data[col] = train_data[col].fillna(mode_val)

# Impute missing values in the test data using values computed from the training set
for col in median_features:
    median_val = train_data[col].median()
    test_data[col] = test_data[col].fillna(median_val)

for col in mode_features:
    mode_val = train_data[col].mode()[0]
    test_data[col] = test_data[col].fillna(mode_val)


import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical variables
columns_to_check = ['MRI_Track_Age_at_Scan', 'EHQ_EHQ_Total']

# Function to remove outliers using IQR and visualize only affected features
def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.10)
    Q3 = data[column].quantile(0.90)
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
        axes[0].axvline(Q1, color='green', linestyle='--', label='Q1 (10th Percentile)')
        axes[0].axvline(Q3, color='blue', linestyle='--', label='Q3 (90th Percentile)')
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


y_sexf = train_data['Sex_F']
y_adhd = train_data ['ADHD_Outcome']

id_test = test_data['participant_id']
id_train = train_data['participant_id']

train_data.drop(columns=['participant_id'], inplace=True)
test_data.drop(columns=['participant_id'], inplace=True)


train_data.drop(columns = ['Gender','ADHD_Status'], inplace=True)


from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Target columns only present in training data
target_cols = ['Sex_F', 'ADHD_Outcome']

# Separate features and target variables in training data
features_train = train_data.drop(columns=target_cols)
targets_train = train_data[target_cols]

# No need to drop anything from test data
features_test = test_data

# Initialize MinMaxScaler
minmax_scaler = MinMaxScaler()

# Fit the scaler only on the training features
minmax_scaler.fit(features_train)

# Scale the training features
scaled_data_train = minmax_scaler.transform(features_train)
scaled_train_df = pd.DataFrame(scaled_data_train, columns=features_train.columns)

# Scale the entire test data
scaled_data_test = minmax_scaler.transform(features_test)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=features_test.columns)

# Concatenate the target columns back to the scaled training data
scaled_train_df = pd.concat([scaled_train_df, targets_train.reset_index(drop=True)], axis=1)


import pandas as pd
from sklearn.utils import resample

# Assuming train_df_sol is already loaded

# Split the dataset into female and male
female_df = train_df_sol[train_df_sol['Sex_F'] == 1]
male_df = train_df_sol[train_df_sol['Sex_F'] == 0]

# Find the minimum count to balance
min_count = min(len(female_df), len(male_df))

# Downsample both to min_count
female_downsampled = resample(female_df, replace=False, n_samples=min_count, random_state=42)
male_downsampled = resample(male_df, replace=False, n_samples=min_count, random_state=42)

# Combine the downsampled data
balanced_df = pd.concat([female_downsampled, male_downsampled])

# Get participant IDs
balanced_sex_participant_ids = balanced_df['participant_id'].tolist()

# Get counts for ADHD and non-ADHD
adhd_count = balanced_df['ADHD_Outcome'].sum()
non_adhd_count = len(balanced_df) - adhd_count

print(f"Number of participants after downsampling: {len(balanced_df)}")
print(f"Number of ADHD cases: {adhd_count}")
print(f"Number of non-ADHD cases: {non_adhd_count}")


nb_type='Submission'


if nb_type == 'Train':
    # ----------------------------------------------------------
    #  WiDS 2025  â€”  fMRI Graphâ€‘Encoder (TransformerConv) TRAIN
    # ----------------------------------------------------------
    # (dependencies) -------------------------------------------
    # pip install torch-scatter torch-sparse torch-geometric -f \
    #     https://data.pyg.org/whl/torch-2.0.0+cpu.html
    # ----------------------------------------------------------
    import os, random, numpy as np, pandas as pd, torch, torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Dataset, DataLoader
    from torch_geometric.nn import TransformerConv, global_mean_pool
    from torch_geometric.utils import degree

    # reproducibility -----------------------------------------
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"â�¡ï¸�  Using device: {device}")

    # ---------------------------------------------------------
    # 1.  Merge targets  â€”  NO DOWNSAMPLING ANY MORE
    # ---------------------------------------------------------
    targets_df = train_df_sol.copy()
    targets_df["Sex_F"]        = targets_df["Sex_F"].astype(int)
    targets_df["ADHD_Outcome"] = targets_df["ADHD_Outcome"].astype(int)

    df_full = train_df_fcm.merge(targets_df, on="participant_id")
    print(f"âœ… Training on full dataset: {len(df_full)} rows")

    y_full = df_full[["ADHD_Outcome", "Sex_F"]].values.astype(np.float32)

    # ---------------------------------------------------------
    # 2. Helper â€” vector â�œ sparse graph (topâ€‘k edges)
    # ---------------------------------------------------------
    NUM_NODES = 200
    TOP_K     = 12
    tri_u     = np.triu_indices(NUM_NODES, k=1)

    def vec_to_graph(vec: np.ndarray) -> Data:
        adj = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
        adj[tri_u] = vec
        adj += adj.T
        keep = np.zeros_like(adj, bool)
        for i in range(NUM_NODES):
            idx = np.argsort(adj[i])[-TOP_K:]
            keep[i, idx] = True
        keep = np.logical_or(keep, keep.T)
        row, col = np.where(keep & (adj != 0))
        edge_w   = adj[row, col]
        edge_idx = torch.tensor(np.vstack([row, col]), dtype=torch.long)
        edge_attr= torch.tensor(edge_w, dtype=torch.float32)
        deg      = degree(edge_idx[0], NUM_NODES).unsqueeze(1)
        x        = deg.float()
        return Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)

    # ---------------------------------------------------------
    # 3. Build PyG Dataset objects â€” FULL TRAIN + TEST
    # ---------------------------------------------------------
    class ConnectomeDataset(Dataset):
        def __init__(self, df_fcm: pd.DataFrame, y: np.ndarray = None):
            super().__init__()
            self.vecs = df_fcm.drop(columns=["participant_id"]).values.astype(np.float32)
            self.ids  = df_fcm["participant_id"].values
            self.y    = y

        def len(self):
            return len(self.vecs)

        def get(self, idx):
            g = vec_to_graph(self.vecs[idx])
            if self.y is not None:
                g.y = torch.tensor(self.y[idx], dtype=torch.float32)
            g.participant_id = self.ids[idx]
            return g

    train_ds = ConnectomeDataset(
        df_full.drop(columns=["ADHD_Outcome", "Sex_F"]),
        y_full
    )
    test_ds  = ConnectomeDataset(test_df_fcm)

    # ---------------------------------------------------------
    # 4. Graph Transformer Encoder
    # ---------------------------------------------------------
    class GraphTransformer(nn.Module):
        def __init__(self, d_model=64, heads=4, dropout=0.25):
            super().__init__()
            self.edge_encoder = nn.Linear(1, d_model)
            self.conv1 = TransformerConv(1,       d_model // heads, heads=heads,
                                         dropout=dropout, edge_dim=d_model)
            self.conv2 = TransformerConv(d_model, d_model // heads, heads=heads,
                                         dropout=dropout, edge_dim=d_model)
            self.conv3 = TransformerConv(d_model, d_model // heads, heads=heads,
                                         dropout=dropout, edge_dim=d_model)
            self.lin_rescale = nn.Linear(d_model, 128)
            self.classifier  = nn.Linear(128, 2)
            self.dp = dropout

        def forward(self, data):
            x, ei, ew, batch = data.x, data.edge_index, data.edge_attr, data.batch
            ew_emb = self.edge_encoder(ew.view(-1, 1))

            x = F.elu(self.conv1(x, ei, edge_attr=ew_emb))
            x = F.dropout(x, p=self.dp, training=self.training)
            x = F.elu(self.conv2(x, ei, edge_attr=ew_emb))
            x = F.dropout(x, p=self.dp, training=self.training)
            x = F.elu(self.conv3(x, ei, edge_attr=ew_emb))

            g   = global_mean_pool(x, batch)          # [B, 64]
            emb = F.relu(self.lin_rescale(g))         # [B, 128]
            logits = self.classifier(emb)             # [B, 2]
            return logits, emb

    model = GraphTransformer().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---------------------------------------------------------
    # 5.  Loss â€” POS_WEIGHT for ADHD & SEX
    # ---------------------------------------------------------
    adhd_pos_w = (df_full["ADHD_Outcome"] == 0).sum() / (df_full["ADHD_Outcome"] == 1).sum()
    sex_pos_w  = (df_full["Sex_F"]        == 0).sum() / (df_full["Sex_F"]        == 1).sum()
    print(f"â�© pos_weight ADHD = {adhd_pos_w:.2f},  Sex_F = {sex_pos_w:.2f}")

    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([adhd_pos_w, sex_pos_w], device=device)
    )

    optimizer     = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loader  = DataLoader(train_ds, batch_size=32, shuffle=True)

    # ---------------------------------------------------------
    # 6. Training (FULL data)
    # ---------------------------------------------------------
    EPOCHS = 40
    model.train()
    for epoch in range(1, EPOCHS + 1):
        cum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            out, _ = model(batch)
            y_true = batch.y.view(-1, 2)
            loss   = bce(out, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum += loss.item() * batch.num_graphs
        print(f"Epoch {epoch:02}/{EPOCHS} â€¢ loss = {cum/len(train_ds):.4f}")

    torch.save(model.state_dict(), "graph_transformer_fmri_full.pt")
    print("âœ… saved model  â�œ  graph_transformer_fmri_full.pt")

    # ---------------------------------------------------------
    # 7. Embedding extraction â€” FULL TRAIN + TEST
    # ---------------------------------------------------------
    def write_embeddings(dataset, csv_name):
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        model.eval()
        embs, ids = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                _, e = model(batch)
                embs.append(e.cpu().numpy())
                ids.extend(batch.participant_id)
        embs = np.vstack(embs)
        df_out = pd.DataFrame(
            embs,
            columns=[f"gt_emb_{i}" for i in range(embs.shape[1])]
        )
        df_out.insert(0, "participant_id", ids)
        df_out.to_csv(csv_name, index=False)
        print(f"ğŸ’¾ wrote {csv_name}")

    write_embeddings(train_ds, "train_fmri_graph_embeddings_full.csv")
    write_embeddings(test_ds,  "test_fmri_graph_embeddings_full.csv")



if nb_type == 'Train':
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.linear_model import RidgeClassifier, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.multioutput import MultiOutputClassifier

    # â”€â”€â”€ 1) Data Prep â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    survey_cols = [c for c in scaled_train_df.columns if c not in ['participant_id','Sex_F','ADHD_Outcome']]
    participants = id_train  # assumed aligned with scaled_train_df

    X_full  = scaled_train_df[survey_cols].values
    y_full  = np.vstack([scaled_train_df['ADHD_Outcome'].values,
                         scaled_train_df['Sex_F'].values]).T

    mask    = np.isin(participants, balanced_sex_participant_ids)
    X_train = X_full[mask]
    y_train = y_full[mask]

    # â”€â”€â”€ 2) Competition F1 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def competition_f1(y_true, y_pred):
        f1_sex = f1_score(y_true[:, 1], y_pred[:, 1])
        w      = np.where((y_true[:, 1] == 1) & (y_true[:, 0] == 1), 2, 1)
        f1_adhd = f1_score(y_true[:, 0], y_pred[:, 0], sample_weight=w)
        return (f1_sex + f1_adhd) / 2

    # â”€â”€â”€ 3) Models to Compare â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "Ridge": RidgeClassifier(alpha=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42),
        "HistGB": HistGradientBoostingClassifier(max_iter=100, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, n_jobs=-1,
                                   random_state=42, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                                 use_label_encoder=False, eval_metric="logloss", n_jobs=-1,
                                 random_state=42)
    }

    # â”€â”€â”€ 4) CV Eval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("ğŸš€ Running 5-Fold CV for Survey-Only Models...\n")

    for name, model in models.items():
        fold_scores = []
        clf = MultiOutputClassifier(model)

        for fold, (tr, va) in enumerate(cv.split(X_train, y_train[:, 0]), 1):
            clf.fit(X_train[tr], y_train[tr])
            preds = clf.predict(X_full)
            score = competition_f1(y_full, preds)
            fold_scores.append(score)

        results[name] = fold_scores
        print(f"{name:18s} â†’ {np.round(fold_scores, 4).tolist()} | Mean: {np.round(np.mean(fold_scores), 4)}")

    # â”€â”€â”€ 5) Boxplot Comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plt.figure(figsize=(12,6))
    plt.boxplot([results[name] for name in results], labels=list(results.keys()), showmeans=True)
    plt.xticks(rotation=45)
    plt.ylabel("Competition F1 Score")
    plt.title("ğŸ“¦ 5-Fold CV Scores (Survey Models)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if nb_type == 'Train':
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.multioutput import MultiOutputClassifier

    # â”€â”€â”€ 1) Data Prep â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fcm_cols   = [c for c in train_df_fcm.columns if c != "participant_id"]
    X_fmri_raw = train_df_fcm[fcm_cols].values
    y_full     = np.vstack([train_df_sol["ADHD_Outcome"].values,
                            train_df_sol["Sex_F"].values]).T
    participants = id_train

    mask     = np.isin(participants, balanced_sex_participant_ids)
    X_fmri   = X_fmri_raw[mask]
    y_train  = y_full[mask]

    # â”€â”€â”€ 2) PCA Transform (0.95) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    Xpca_full = pca.fit_transform(X_fmri_raw)
    Xpca_train = Xpca_full[mask]

    # â”€â”€â”€ 3) Competition Metric â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def competition_f1(y_true, y_pred):
        f1_sex = f1_score(y_true[:, 1], y_pred[:, 1])
        w      = np.where((y_true[:, 1] == 1) & (y_true[:, 0] == 1), 2, 1)
        f1_adhd = f1_score(y_true[:, 0], y_pred[:, 0], sample_weight=w)
        return (f1_sex + f1_adhd) / 2

    # â”€â”€â”€ 4) Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "Ridge": RidgeClassifier(alpha=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42),
        "HistGB": HistGradientBoostingClassifier(max_iter=100, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                                 use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=42)
    }

    # â”€â”€â”€ 5) CV Loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("ğŸš€ Running 5-Fold CV for fMRI PCA(0.95)-Only Models...\n")

    for name, model in models.items():
        clf = MultiOutputClassifier(model)
        fold_scores = []

        for fold, (tr, va) in enumerate(cv.split(Xpca_train, y_train[:, 0]), 1):
            clf.fit(Xpca_train[tr], y_train[tr])
            preds = clf.predict(Xpca_full)
            score = competition_f1(y_full, preds)
            fold_scores.append(score)

        results[name] = fold_scores
        print(f"{name:18s} â†’ {np.round(fold_scores, 4).tolist()} | Mean: {np.round(np.mean(fold_scores), 4)}")

    # â”€â”€â”€ 6) Boxplot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plt.figure(figsize=(12,6))
    plt.boxplot([results[m] for m in results], labels=list(results.keys()), showmeans=True)
    plt.title("ğŸ“¦ 5-Fold CV Scores â€“ fMRI PCA(0.95) Only")
    plt.ylabel("Competition F1")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if nb_type == 'Train':
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.linear_model import RidgeClassifier, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.multioutput import MultiOutputClassifier

    # â”€â”€â”€ 1) Load Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    emb_df = pd.read_csv("/kaggle/input/full-data-dictionaries/train_fmri_gat_embeddings.csv")
    emb_df['participant_id'] = emb_df['participant_id'].astype(str)

    participants = id_train
    y_full = np.vstack([
        scaled_train_df["ADHD_Outcome"].values,
        scaled_train_df["Sex_F"].values
    ]).T

    # Align embeddings with full train
    X_emb_full = emb_df.set_index("participant_id").loc[participants].values

    # Downsample
    mask = participants.isin(balanced_sex_participant_ids)
    X_train = X_emb_full[mask]
    y_train = y_full[mask]

    # â”€â”€â”€ 2) Competition Metric â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def competition_f1(y_true, y_pred):
        f1_sex = f1_score(y_true[:, 1], y_pred[:, 1])
        w = np.where((y_true[:, 1] == 1) & (y_true[:, 0] == 1), 2, 1)
        f1_adhd = f1_score(y_true[:, 0], y_pred[:, 0], sample_weight=w)
        return (f1_sex + f1_adhd) / 2

    # â”€â”€â”€ 3) Models to Compare â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "Ridge": RidgeClassifier(alpha=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42),
        "HistGB": HistGradientBoostingClassifier(max_iter=100, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                                 use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=42)
    }

    # â”€â”€â”€ 4) CV Eval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("ğŸš€ Running 5-Fold CV for fMRI Embeddings Only Models...\n")

    for name, model in models.items():
        fold_scores = []
        clf = MultiOutputClassifier(model)

        for fold, (tr, va) in enumerate(cv.split(X_train, y_train[:, 0]), 1):
            clf.fit(X_train[tr], y_train[tr])
            preds = clf.predict(X_emb_full)
            score = competition_f1(y_full, preds)
            fold_scores.append(score)

        results[name] = fold_scores
        print(f"{name:18s} â†’ {np.round(fold_scores, 4).tolist()} | Mean: {np.round(np.mean(fold_scores), 4)}")

    # â”€â”€â”€ 5) Boxplot Comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plt.figure(figsize=(12,6))
    plt.boxplot([results[m] for m in models], labels=list(models.keys()), showmeans=True)
    plt.title("ğŸ“¦ 5-Fold CV Scores â€“ fMRI Embeddings Only")
    plt.ylabel("Competition F1")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if nb_type == 'Train':
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier

    # â”€â”€â”€ 1) Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fcm_cols    = [c for c in train_df_fcm.columns if c != "participant_id"]
    survey_cols = [c for c in scaled_train_df.columns if c not in ['participant_id','Sex_F','ADHD_Outcome']]
    pca_vars    = [0.90, 0.95, 0.99]

    # â”€â”€â”€ 2) Load & mask data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_fmri_full    = train_df_fcm[fcm_cols].values
    X_survey_full  = scaled_train_df[survey_cols].values
    y_full         = np.vstack([
        scaled_train_df["ADHD_Outcome"].values,
        scaled_train_df["Sex_F"].values
    ]).T
    participants   = train_df_sol["participant_id"].astype(str).values

    mask           = np.isin(participants, balanced_sex_participant_ids)
    X_fmri         = X_fmri_full[mask]
    X_survey       = X_survey_full[mask]
    y_train        = y_full[mask]

    # â”€â”€â”€ 3) Metric â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def competition_f1(y_true, y_pred):
        f1_sex = f1_score(y_true[:,1], y_pred[:,1])
        w      = np.where((y_true[:,1]==1)&(y_true[:,0]==1), 2, 1)
        f1_adhd = f1_score(y_true[:,0], y_pred[:,0], sample_weight=w)
        return (f1_sex + f1_adhd) / 2

    # â”€â”€â”€ 4) 10â€‘Fold CV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_survey, y_train[:,0]), 1):
        # Branch 1: XGB on survey
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
            random_state=42
        )
        clf_xgb = MultiOutputClassifier(xgb)
        clf_xgb.fit(X_survey[tr_idx], y_train[tr_idx])
        proba = clf_xgb.predict_proba(X_survey_full)
        branch1 = np.stack([p[:,1] for p in proba], axis=1).astype(np.float32)
        
        # Branch 2: Ridge on fMRI PCA (0.90, 0.95, 0.99)
        ridge_preds = []
        for var in pca_vars:
            pca = PCA(n_components=var, svd_solver="full", random_state=42)
            Xp_tr = pca.fit_transform(X_fmri[tr_idx])
            Xp_te = pca.transform(X_fmri_full)
            clf_ridge = MultiOutputClassifier(RidgeClassifier(alpha=1.0))
            clf_ridge.fit(Xp_tr, y_train[tr_idx])
            ridge_preds.append(clf_ridge.predict(Xp_te).astype(np.float32))
        branch2 = np.mean(ridge_preds, axis=0)

        # Blend and threshold
        final_soft = 0.5 * branch1 + 0.5 * branch2
        final_pred = np.zeros_like(final_soft, dtype=int)
        final_pred[:,0] = (final_soft[:,0] >= 0.5).astype(int)
        final_pred[:,1] = (final_soft[:,1] >= 0.5).astype(int)

        score = competition_f1(y_full, final_pred)
        fold_scores.append(score)

    # â”€â”€â”€ 5) Print results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    scores_rounded = [round(s, 4) for s in fold_scores]
    mean_score = np.mean(fold_scores)
    print(f"Fold scores: {scores_rounded}")
    print(f"Mean 10â€‘Fold CV Score: {mean_score:.4f}")

    # â”€â”€â”€ 6) Line Plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 11), fold_scores, marker='o', linestyle='-', linewidth=2)
    plt.xticks(range(1, 11))
    plt.xlabel("Fold Number")
    plt.ylabel("Competition F1 Score")
    plt.title("10â€‘Fold CV Competition F1 Scores (XGB + PCAâ€‘Ridge)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

# â”€â”€â”€ 1) Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fcm_cols    = [c for c in train_df_fcm.columns if c != "participant_id"]
survey_cols = [c for c in scaled_train_df.columns if c not in ['participant_id','Sex_F','ADHD_Outcome']]

# PCA configs for fMRI branch
pca_vars = [0.90, 0.95,0.99]

# â”€â”€â”€ 2) Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Train (downsampled by sex)
X_fmri_full   = train_df_fcm[fcm_cols].values
X_survey_full = scaled_train_df[survey_cols].values
y_full        = np.vstack([
    scaled_train_df["ADHD_Outcome"].values,
    scaled_train_df["Sex_F"].values
]).T
participants  = id_train

mask    = np.isin(participants, balanced_sex_participant_ids)
X_fmri  = X_fmri_full[mask]
X_survey= X_survey_full[mask]
y_train = y_full[mask]

# Test
X_survey_test = scaled_test_df[survey_cols].values
X_fmri_test   = test_df_fcm[fcm_cols].values
test_ids      = test_df_fcm["participant_id"].values

# â”€â”€â”€ 3) Branch 1: XGB on Survey â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1,
    verbosity=0,
    random_state=42
)
clf_xgb = MultiOutputClassifier(xgb)
clf_xgb.fit(X_survey, y_train)
xgb_proba = clf_xgb.predict_proba(X_survey_test)
branch1 = np.stack([p[:, 1] for p in xgb_proba], axis=1).astype(np.float32)

# â”€â”€â”€ 4) Branch 2: Ridge on fMRI PCA (0.90 & 0.95 ensemble) â”€
ridge_preds = []
for var in pca_vars:
    pca = PCA(n_components=var, svd_solver="full", random_state=42)
    Xp_tr = pca.fit_transform(X_fmri)
    Xp_te = pca.transform(X_fmri_test)

    clf_ridge = MultiOutputClassifier(RidgeClassifier(alpha=1.0))
    clf_ridge.fit(Xp_tr, y_train)
    preds = clf_ridge.predict(Xp_te).astype(np.float32)
    ridge_preds.append(preds)

branch2 = np.mean(ridge_preds, axis=0)

# â”€â”€â”€ 5) Combine & Threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
final_soft = 0.5 * branch1 + 0.3 * branch2
final_pred = np.zeros_like(final_soft, dtype=int)
final_pred[:, 0] = (final_soft[:, 0] >= 0.5).astype(int)
final_pred[:, 1] = (final_soft[:, 1] >= 0.5).astype(int)

# â”€â”€â”€ 6) Save Submission â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
submission = pd.DataFrame({
    "participant_id": test_ids,
    "ADHD_Outcome": final_pred[:, 0],
    "Sex_F": final_pred[:, 1]
})
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv created: 50% XGB(survey) + 30% Ridge(fMRI PCA 0.90,0.95 & 0.99) ensemble")

