import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

!pip install skimpy
!pip install sweetviz
!pip install jupyter-summarytools

from summarytools import dfSummary
import sweetviz as sv
from IPython.core.display import display, HTML
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
print(train.shape)
print(test.shape)
print(train_extra.shape)


### Checking missing values for 3 dataframe

def checking_quality(df, df_name):
    print(f'Checking {df_name} dataframe') 
    print('Number of duplicated rows: ', df.duplicated().sum())
    
    missing_col = [col for col in df.columns if df[col].isnull().sum() > 0]
    
    print('Columns with missing values: ', missing_col)

# Example Usage
checking_quality(train, "train") 
print('_'*50)
checking_quality(test, "test")   
print('_'*50)
checking_quality(train_extra, "train_extra")  


# Create subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(25, 10))

# Plot missing value matrix for each DataFrame
msno.matrix(train, ax=axes[0])
axes[0].set_title("Missing Value of Train Dataset",fontsize = 20)

msno.matrix(test, ax=axes[1])
axes[1].set_title("Missing Value of Test Dataset",fontsize = 20)

msno.matrix(train_extra, ax=axes[2])
axes[2].set_title("Missing Value of Train Extra Dataset",fontsize = 20)

# Adjust layout
plt.tight_layout()
plt.show()


def check_missing_values(df):
    """
    Function to calculate missing values and their percentage in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: A summary DataFrame with missing values count and percentage.
    """
    missing_values = df.isnull().sum()  # Count missing values
    missing_percentage = round((missing_values / len(df)) * 100,2)  # Calculate percentage

    # Create DataFrame
    missing_summary = pd.DataFrame({
        'Column': missing_values.index,
        'Missing_Value': missing_values.values,
        'Percentage_Missing_Value': missing_percentage.values
    })

    # Filter out columns with no missing values
    missing_summary = missing_summary[missing_summary['Missing_Value'] > 0]

    # Round percentage to 2 decimal places
    # missing_summary['Percentage_Missing_Value'] = missing_summary['Percentage_Missing_Value'].round(2)

    # Reset index
    missing_summary.reset_index(drop=True, inplace=True)

    return missing_summary  # Return the DataFrame

train_missing = check_missing_values(train)
test_missing = check_missing_values(test)
train_extra_missing = check_missing_values(train_extra)

# Create subplots (1 row, 3 columns)
fig, ax = plt.subplots(1, 3, figsize=(30, 10))

# Function to add annotations on bars
def add_annotations(axis, data):
    for index, value in enumerate(data['Percentage_Missing_Value']):
        axis.text(value+0.1, index, f"{value:.2f}%", fontsize=14, ha='center', va='center', color='black')  # Position annotation

# Plot missing value bar charts for each DataFrame
ax[0].barh(train_missing['Column'], train_missing['Percentage_Missing_Value'], color='skyblue')
ax[0].set_title("Missing Values - Train Data", fontsize=20)
ax[0].set_xlabel("Percentage Missing", fontsize=16)
ax[0].set_ylabel("Columns", fontsize=16)
ax[0].tick_params(axis='both', labelsize=14)
add_annotations(ax[0], train_missing)  # Add annotations

ax[1].barh(test_missing['Column'], test_missing['Percentage_Missing_Value'], color='salmon')
ax[1].set_title("Missing Values - Test Data", fontsize=20)
ax[1].set_xlabel("Percentage Missing", fontsize=16)
ax[1].tick_params(axis='both', labelsize=14)
add_annotations(ax[1], test_missing)  # Add annotations

ax[2].barh(train_extra_missing['Column'], train_extra_missing['Percentage_Missing_Value'], color='lightgreen')
ax[2].set_title("Missing Values - Train Extra Data", fontsize=20)
ax[2].set_xlabel("Percentage Missing", fontsize=16)
ax[2].tick_params(axis='both', labelsize=14)
add_annotations(ax[2], train_extra_missing)  # Add annotations

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


dfSummary(train[[col for col in train.columns if train[col].isnull().sum() > 0]])


def fill_missing_values(df):
    """
    Fills missing values in the DataFrame based on specified rules:
    
    - For 'Laptop Compartment' and 'Waterproof': fill missing with 'No'
    - For 'Weight Capacity (kg)': fill missing with the mean of the column
    - For 'Brand', 'Material', 'Size', 'Style', and 'Color': fill missing with the mode of the column
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    
    # Define columns for each strategy
    fill_with_no = ['Laptop Compartment', 'Waterproof']
    fill_with_mean = ['Weight Capacity (kg)']
    fill_with_mode = ['Brand', 'Material', 'Size', 'Style', 'Color']
    
    # Fill with "No"
    for col in fill_with_no:
        if col in df.columns:
            df[col] = df[col].fillna("No")
    
    # Fill with mean (for numeric column)
    for col in fill_with_mean:
        if col in df.columns:
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
    
    # Fill with mode (for categorical columns)
    for col in fill_with_mode:
        if col in df.columns:
            if not df[col].mode().empty:
                mode_value = df[col].mode()[0]  # use the first mode
                df[col] = df[col].fillna(mode_value)
    
    return df

train = fill_missing_values(train)
test = fill_missing_values(test)
train_extra = fill_missing_values(train_extra)


import sweetviz as sv
from IPython.core.display import display, HTML

feature_config = sv.FeatureConfig(
    skip=['id']  # Exclude ID column from analysis
)
train_vs_test = sv.compare(
    source=[train, "Train Data"],
    compare=[test, "Test Data"],
    target_feat='Price',
    feat_cfg=feature_config,
    pairwise_analysis='on'  # Show feature correlations
)

# Display in notebook
train_vs_test.show_notebook(
    w=1000,
    # h=700,
    layout='vertical',
    scale=0.85
)


from sklearn.feature_selection import mutual_info_regression

# Prepare the data for train_extra
X_extra = train_extra.copy()
y_extra = X_extra.pop("Price")

# Label encoding for categoricals in train_extra
for colname in X_extra.select_dtypes("object"):
    X_extra[colname], _ = X_extra[colname].factorize()

X_extra = X_extra.drop('id', axis=1)
discrete_features_extra = X_extra.dtypes == int

# Prepare the data for train
X = train.copy()
y = X.pop("Price")

# Label encoding for categoricals in train
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

X = X.drop('id', axis=1)
discrete_features = X.dtypes == int


# Mutual Information function
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores_extra = make_mi_scores(X_extra, y_extra, discrete_features_extra)
mi_scores_train = make_mi_scores(X, y, discrete_features)
    
# Create subplots with 1 row and 3 columns
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Function to plot MI scores
def plot_mi_scores(ax, scores, title):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    ax.barh(width, scores)
    ax.set_yticks(width)
    ax.set_yticklabels(ticks)
    ax.set_title(title)

# Plot MI scores for train_extra, train, and combined dataset
plot_mi_scores(ax[0], mi_scores_extra, "Mutual Information Scores of Train Extra")
plot_mi_scores(ax[1], mi_scores_train, "Mutual Information Scores of Train")

# Adjust layout
plt.tight_layout()
plt.show()


### Combine train vs train_extra, then exclude duplicated (with remove id) and drop duplicates
train_extra = train_extra.drop_duplicates(subset=train_extra.columns[1:])
full = pd.concat([train,train_extra],axis=0)
full = full.drop_duplicates(subset=full.columns[1:])
test = test.drop_duplicates(subset = test.columns[1:10])


fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(x='Price', data=full, bins=10, ax=ax[0])
ax[0].set_title('Histogram of Price with 10 Bins')

sns.histplot(x='Price', data=full, ax=ax[1])
ax[1].set_title('Histogram of Price with Default Bins')

fig.suptitle('Comparison of Price Distributions with Different Bin Settings', fontsize=14)

plt.tight_layout()
plt.show()


check = full.columns[1:10]

# Filter columns in `check` that are of type object and not 'Material'
cols_to_plot = [col for col in check if full[col].dtype == object and col != 'Material']

# Create a 2x3 grid of subplots (total of 6 subplots)
fig, axes = plt.subplots(2, 3, figsize=(20,10))
axes = axes.flatten()  # Flatten to iterate easily

# Plot each selected column on its respective subplot
for i, col in enumerate(cols_to_plot):
    # Group by the current column and 'Material', then calculate the median of 'Price'
    grouped = full.groupby([col, 'Material'])['Price'].median().unstack()

    # Plot on the corresponding subplot axis
    grouped.plot(kind='line', marker='o', ax=axes[i])
    axes[i].set_title(f"Median Price by {col} and Material")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Median Price")
    axes[i].legend(title="Material")

# Remove any unused subplot axes if less than 6 plots were created
for j in range(len(cols_to_plot), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Insight Values from Material and other features', fontsize=20)
plt.tight_layout()
plt.show()


material_mapping = {
    'Canvas': 2,
    'Polyester': 1,
    'Nylon': 3,
    'Leather': 4
}

# Create a new column 'Material_Price_Category' using the mapping
full['Rank_Material_Price'] = full['Material'].map(material_mapping)
test['Rank_Material_Price'] = test['Material'].map(material_mapping)

def material_type(material):
    # Define synthetic vs natural
    if material in ["Polyester", "Nylon"]:
        return "Synthetic"
    else:
        return "Natural"

def material_quality(material):
    if material == "Leather":
        return "Highest"  # Highest quality
    elif material == "Canvas":
        return "Medium"  # Mid quality
    else:
        return "Lowest"  # Lower quality

full["Material_Quality"] = full["Material"].apply(material_quality)
full["Material_Type"] = full["Material"].apply(material_type)
test["Material_Quality"] = test["Material"].apply(material_quality)
test["Material_Type"] = test["Material"].apply(material_type)


# Define mosaic layout
mosaic = """
ABCD
EFGH
IIJJ
"""

# Create the figure and axis dictionary
fig, axes = plt.subplot_mosaic(mosaic, figsize=(20, 12))

# ---------------- Row 1: Using Material_Type ----------------

# 1. Countplot of Material_Type
sns.countplot(data=full, x='Material_Type', ax=axes['A'])
axes['A'].set_title('Count of Material Types')

# 2. Countplot of Material_Type with hue as Brand
sns.countplot(data=full, x='Material_Type', hue='Brand', ax=axes['B'])
axes['B'].set_title('Count by Material Type and Brand')

# 3. Bar plot showing median Price by Material_Type
sns.barplot(data=full, x='Material_Type', y='Price', estimator=np.median, ci=None, ax=axes['C'])
axes['C'].set_title('Median Price by Material Type')
axes['C'].set_ylim(79, 83)

# 4. Bar plot showing median Price by Material_Type with hue as Brand
sns.barplot(data=full, x='Material_Type', y='Price', hue='Brand', estimator=np.median, ci=None, ax=axes['D'])
axes['D'].set_title('Median Price by Material Type & Brand')
axes['D'].set_ylim(79, 83)

# ---------------- Row 2: Using Material_Quality ----------------

# 5. Countplot of Material_Quality
sns.countplot(data=full, x='Material_Quality', ax=axes['E'])
axes['E'].set_title('Count of Material Quality')

# 6. Countplot of Material_Quality with hue as Brand
sns.countplot(data=full, x='Material_Quality', hue='Brand', ax=axes['F'])
axes['F'].set_title('Count by Material Quality and Brand')

# 7. Bar plot showing median Price by Material_Quality
sns.barplot(data=full, x='Material_Quality', y='Price', estimator=np.median, ci=None, ax=axes['G'])
axes['G'].set_title('Median Price by Material Quality')
axes['G'].set_ylim(79, 83)

# 8. Bar plot showing median Price by Material_Quality with hue as Brand
sns.barplot(data=full, x='Material_Quality', y='Price', hue='Brand', estimator=np.median, ci=None, ax=axes['H'])
axes['H'].set_title('Median Price by Material Quality & Brand')
axes['H'].set_ylim(77, 83)

# ---------------- Row 3: Additional Analysis ----------------

# 9. Countplot of Material_Quality with hue as Material_Type
sns.countplot(x='Material_Quality', hue='Material_Type', data=full, ax=axes['I'])
axes['I'].set_title('Count by Material Quality and Material Type')

# 10. Bar plot of median Price by Material_Quality and Material_Type
median_prices = full.groupby(['Material_Quality', 'Material_Type'])['Price'].median().reset_index()
sns.barplot(data=median_prices, x='Material_Quality', y='Price', hue='Material_Type', ax=axes['J'])
axes['J'].set_title('Median Price by Material Quality and Material Type')
axes['J'].set_ylim(75, 85)


# Set an overall title for the entire figure
fig.suptitle('Material Type/Quality and Price Analysis', fontsize=16)

# Adjust layout so that all titles and labels fit well within the figure area
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the plots
plt.show()



# Define the last mosaic layout
mosaic = """
AAA
BCD
EFG
"""

check = full.columns[1:10]

# Create the figure and axes using subplot_mosaic
fig, axes = plt.subplot_mosaic(mosaic, figsize=(21, 18))

# First row: Add a countplot for `Color`
sns.countplot(x='Color', data=full, ax=axes['A'])
axes['A'].set_title('Count of Colors')
axes['A'].set_xlabel('Color')
axes['A'].set_ylabel('Count')

# Get the columns of type object (excluding 'Color')
cols_to_plot = [col for col in check if full[col].dtype == object and col != 'Color']

color_mapping = {
    'Black': 'Black',
    'Green': 'Green',
    'Red': 'Red',
    'Blue': 'Blue',
    'Gray': 'Gray',
    'Pink': 'Pink'
}

# Plot each selected column in the remaining subplots
keys = list("BCDEFG")  # Subplot keys for remaining plots
for i, col in enumerate(cols_to_plot[:6]):  # Limit to 6 plots for the mosaic layout
    grouped = full.groupby([col, 'Color'])['Price'].median().unstack()

    # Map the colors in the plot to the predefined color_mapping
    plot_colors = [color_mapping.get(color, None) for color in grouped.columns]

    # Plot the data on the corresponding subplot
    grouped.plot(kind='line', marker='o', ax=axes[keys[i]], color=plot_colors)
    axes[keys[i]].set_title(f"Median Price by {col} and Color")
    axes[keys[i]].set_xlabel(col)
    axes[keys[i]].set_ylabel('Median Price')
    axes[keys[i]].legend(title='Color')

# Set a figure-wide title
fig.suptitle('Insights from Color and Other Features', fontsize=20)

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()



color_price_mapping = {
    'Blue': 1,
    'Green': 1,
    'Red': 3,
    'Pink': 2,
    'Black': 5,
    'Gray': 4
}

# Create a new column in the DataFrame that categorizes the price based on color
full['Rank_Color_Price'] = full['Color'].map(color_price_mapping)
test['Rank_Color_Price'] = test['Color'].map(color_price_mapping)


bins = [0, 3, 7, 10]  # Note: lower bound set to 0 to include 1 in the first bin
labels = ['Low', 'Medium', 'High']

## number of compartments
### from 1-3: Low Compartment, 4-7: Medium, 7-10: High
# Create a new categorical column 'Compartment_Bin'
full['Category_Compartment'] = pd.cut(full['Compartments'], bins=bins, labels=labels, include_lowest=True)
test['Category_Compartment'] = pd.cut(test['Compartments'], bins=bins, labels=labels, include_lowest=True)


# Define the bin edges and corresponding labels
bins = [5, 8, 12, 18, 30]  
labels = ["Lightweight", "Medium", "Heavy", "Extra-Heavy"]

### Classify weight 5-8kg: Lightweight, 8-12 Medium, 12-18 Heavy, 18-30: Extra-Heavy
# Use pd.cut() to bin the "Weight Capacity (kg)" column and create a new "Category" column
full['Category_Weight'] = pd.cut(full['Weight Capacity (kg)'],
                          bins=bins,
                          labels=labels,
                          include_lowest=True)

test['Category_Weight'] = pd.cut(test['Weight Capacity (kg)'],
                          bins=bins,
                          labels=labels,
                          include_lowest=True)

median_prices = full.groupby(['Laptop Compartment', 'Waterproof'])['Price'].median().reset_index()
median_prices_style = full.groupby('Style')['Price'].median().reset_index()
median_weight = full.groupby('Category_Weight')['Price'].median().reset_index()

fig, ax = plt.subplots(1, 5, figsize=(24, 6))

# First subplot (Category Compartment countplot)
sns.countplot(x='Category_Compartment', data=full, ax=ax[0])
ax[0].set_title('Count of Category Compartment')
ax[0].set_xlabel('Category Compartment')
ax[0].set_ylabel('Count')

# Second subplot (Category Weight countplot)
sns.countplot(x='Category_Weight', data=full, ax=ax[1])
ax[1].set_title('Count of Category Weight')
ax[1].set_xlabel('Category Weight')
ax[1].set_ylabel('Count')

sns.lineplot(x='Category_Weight', y='Price',data=median_weight, ax=ax[2])
ax[2].set_title('Median Price by Category Weight')
ax[2].set_xlabel('Category Weight')
ax[2].set_ylabel('Median Price')

# Third subplot (Line plot for median prices)
sns.lineplot(data=median_prices, x='Laptop Compartment', y='Price', hue='Waterproof', ax=ax[3])
ax[3].set_title("Median Price by Laptop Compartment and Waterproof")
ax[3].set_xlabel("Laptop Compartment")
ax[3].set_ylabel("Median Price")
ax[3].legend(title="Waterproof")


sns.lineplot(data=median_prices_style, x='Style', y='Price', ax=ax[4])
ax[4].set_title("Median Price by Style")
ax[4].set_xlabel("Style")
ax[4].set_ylabel("Median Price")

plt.tight_layout()
plt.show()

def feature_combo_label(row):
    if row['Laptop Compartment'] == 'Yes' and row['Waterproof'] == 'Yes':
        return 2
    elif row['Laptop Compartment'] == 'Yes' or row['Waterproof'] == 'Yes':
        return 1
    else:
        return 0

full['Rank_Laptop_Waterproof'] = full.apply(feature_combo_label, axis=1)
test['Rank_Laptop_Waterproof'] = test.apply(feature_combo_label, axis=1)
full["Style_Weight"] = full["Style"].astype(str) + "_" + full["Category_Weight"].astype(str)


full


price_max = full[full['Price'] == full['Price'].max()]
price_min = full[full['Price'] == full['Price'].min()]
price_max = price_max[price_max.columns[1:10]]
price_min = price_min[price_min.columns[1:10]]
new_test = test[test.columns[1:10]]

