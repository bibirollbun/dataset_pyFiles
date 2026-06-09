# Load the cuDF pandas extension
#%load_ext cudf.pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import plotly.express as px  
import plotly.io as pio  
pio.renderers.default = 'iframe'  
from IPython.display import display

import cudf
from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, plot_importance 

import warnings
warnings.filterwarnings("ignore")


# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
sample_data = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
train_ex_data = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv',index_col='id')

# Verify shapes
print("Train Data Shape:", train_data.shape)
print("Train Extra Data Shape:", train_ex_data.shape)
print("Test Data Shape:", test_data.shape)


# Display sample data
print("\nTrain Data Sample:")
display(train_data.head())

print("\nTrain Extra Data Sample:")
display(train_ex_data.head())

print("\nTest Data Sample:")
display(test_data.head())


# Display information about the DataFrames
print("\nTrain Data Info:")
train_data.info()

print("\nTrain Extra Data Info:")
train_ex_data.info()

print("\nTest Data Info:")
test_data.info()


# Descriptive statistics for numerical columns
print("\nTrain Data Describe:")
display(train_data.describe())

print("\nTrain Extra Data Describe:")
display(train_ex_data.describe())

print("\nTest Data Describe:")
display(test_data.describe())


# Descriptive statistics for object (categorical) columns
print("\nTrain Data Describe (Categorical):")
display(train_data.describe(include='object'))

print("\nTrain Extra Data Describe (Categorical):")
display(train_ex_data.describe(include='object'))

print("\nTest Data Describe (Categorical):")
display(test_data.describe(include='object'))


# Function to create a heatmap of missing values
def plot_missing_values_heatmap(df, title):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cmap='PuBuGn_r', cbar=False, yticklabels=False)
    plt.title(f'Missing Values Heatmap - {title}', fontsize=14)
    plt.xlabel("Features")
    plt.ylabel("Index")
    plt.show()

# Generate heatmaps for each dataset
plot_missing_values_heatmap(train_data, "Training Dataset")
plot_missing_values_heatmap(train_ex_data, "Training Extra Dataset")
plot_missing_values_heatmap(test_data, "Test Dataset")


# Function to calculate missing values, percentages, and data types
def missing_values_table(df):
    missing_count = df.isnull().sum()
    missing_percentage = 100 * missing_count / len(df)
    data_types = df.dtypes
    return pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage (%)': missing_percentage,
        'Data Type': data_types
    })

# Create tables for train, train_extra, and test datasets
train_missing_table = missing_values_table(train_data)
train_extra_missing_table = missing_values_table(train_ex_data)
test_missing_table = missing_values_table(test_data)

# Display the tables (only features with missing values)
print("\nMissing Values Table - Training Dataset:\n")
display(train_missing_table[train_missing_table['Missing Values'] > 0])
print("\n")

print("\nMissing Values Table - Training Extra Dataset:\n")
display(train_extra_missing_table[train_extra_missing_table['Missing Values'] > 0])
print("\n")

print("\nMissing Values Table - Test Dataset:\n")
display(test_missing_table[test_missing_table['Missing Values'] > 0])


# Function to calculate missing values
def missing_values_table(df):
    missing_count = df.isnull().sum()
    return missing_count[missing_count > 0]

# Get missing values counts
train_null = missing_values_table(train_data)
test_null = missing_values_table(test_data)
train_ex_null = missing_values_table(train_ex_data)

fig, axes = plt.subplots(3, 2, figsize=(12, 15))

def create_donut_chart(data, ax, title):
    wedges, texts, autotexts = ax.pie(
        data,
        labels=data.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("PuBuGn", len(data))
    )
    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title(title)

sns.barplot(x=train_null.values, y=train_null.index, ax=axes[0, 0], palette='PuBuGn')
axes[0, 0].set_title('Count of Missing Values - Train Dataset')
axes[0, 0].set_xlabel('Count')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', color='gray', linestyle='--', linewidth=0.7)

sns.barplot(x=test_null.values, y=test_null.index, ax=axes[1, 0], palette='PuBuGn')
axes[1, 0].set_title('Count of Missing Values - Test Dataset')
axes[1, 0].set_xlabel('Count')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', color='gray', linestyle='--', linewidth=0.7)

sns.barplot(x=train_ex_null.values, y=train_ex_null.index, ax=axes[2, 0], palette='PuBuGn')
axes[2, 0].set_title('Count of Missing Values - Train Extra Dataset')
axes[2, 0].set_xlabel('Count')
axes[2, 0].invert_yaxis()
axes[2, 0].grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  

create_donut_chart(train_null, axes[0, 1], 'Missing Values in Train Dataset')
create_donut_chart(test_null, axes[1, 1], 'Missing Values in Test Dataset')
create_donut_chart(train_ex_null, axes[2, 1], 'Missing Values in Train Extra Dataset')

plt.tight_layout()
plt.show()


# Check for duplicated rows
print("\nDuplicate Rows in Train Data:", train_data.duplicated().sum())
print("\nDuplicate Rows in Train Extra Data:", train_ex_data.duplicated().sum())
print("\nDuplicate Rows in Test Data:", test_data.duplicated().sum())


# Plotting Price Distribution for Train Data
plt.figure(figsize=(12, 4))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(train_data['Price'], bins=30, kde=True, color=sns.color_palette("PuBuGn", 1)[0])  
plt.title('Price Distribution - Train Data')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)  

# Box Plot
plt.subplot(1, 2, 2)
sns.boxplot(x=train_data['Price'], color=sns.color_palette("PuBuGn", 1)[0])  
plt.title('Price Box Plot - Train Data')
plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  

plt.tight_layout()
plt.show()

# Repeat for Train Extra Data
plt.figure(figsize=(12, 4))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(train_ex_data['Price'], bins=30, kde=True, color=sns.color_palette("PuBuGn", 1)[0])  
plt.title('Price Distribution - Train Extra Data')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)  

# Box Plot
plt.subplot(1, 2, 2)
sns.boxplot(x=train_ex_data['Price'], color=sns.color_palette("PuBuGn", 1)[0])  
plt.title('Price Box Plot - Train Extra Data')
plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  

plt.tight_layout()
plt.show()


# Identify numerical columns (excluding 'Price' if it exists)
numerical_cols_train = train_data.select_dtypes(include=['number']).columns.tolist()
if 'Price' in numerical_cols_train:
    numerical_cols_train.remove('Price')

numerical_cols_test = test_data.select_dtypes(include=['number']).columns.tolist()
if 'Price' in numerical_cols_test:
    numerical_cols_test.remove('Price')
    
numerical_cols_train_ex = train_ex_data.select_dtypes(include=['number']).columns.tolist()
if 'Price' in numerical_cols_train_ex:
    numerical_cols_train_ex.remove('Price')

# Function to plot histograms for each numerical column
def plot_numerical_distributions(data, numerical_cols, title):
    num_cols = len(numerical_cols)
    num_rows = (num_cols // 3) + (num_cols % 3 > 0)  
    
    plt.figure(figsize=(15, 5 * num_rows))  

    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(num_rows, 3, i)
        sns.histplot(data[col], kde=True, color=sns.color_palette("PuBuGn", 1)[0])  
        plt.title(f'Distribution of {col} - {title}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)  

    plt.tight_layout()
    plt.show()

# Plotting the distributions
plot_numerical_distributions(train_data, numerical_cols_train, 'Train Data')
plot_numerical_distributions(test_data, numerical_cols_test, 'Test Data')
plot_numerical_distributions(train_ex_data, numerical_cols_train_ex, 'Train Extra Data')


# Identify categorical columns
cat_columns_train = train_data.select_dtypes(include=['object']).columns.tolist()
cat_columns_test = test_data.select_dtypes(include=['object']).columns.tolist()
cat_columns_train_ex = train_ex_data.select_dtypes(include=['object']).columns.tolist()


def plot_categorical_distributions_grid(cat_columns, datasets, titles):
    num_features = len(cat_columns)
    num_datasets = len(datasets)

    fig, axes = plt.subplots(num_features, num_datasets, figsize=(15, 5 * num_features))

    custom_palette = sns.color_palette("PuBuGn", 8)  

    for i, col in enumerate(cat_columns):
        for j, (data, title) in enumerate(zip(datasets, titles)):
            sns.countplot(y=col, data=data, ax=axes[i, j], palette=custom_palette)
            axes[i, j].set_title(f'Count of {col} - {title}')
            axes[i, j].set_xlabel('Count')
            axes[i, j].set_ylabel(col)

            for p in axes[i, j].patches:
                axes[i, j].annotate(f'{int(p.get_width())}', 
                                    (p.get_width(), p.get_y() + p.get_height() / 2), 
                                    ha='left', va='center', 
                                    color='black', fontsize=12)

            axes[i, j].set_axisbelow(True)  
            axes[i, j].grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  
            sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()

datasets = [train_data, test_data, train_ex_data]
titles = ['Train Data', 'Test Data', 'Train Extra Data']

plot_categorical_distributions_grid(cat_columns_train, datasets, titles)


def plot_categorical_distributions_grid(cat_columns, datasets, titles):
    num_features = len(cat_columns)
    num_datasets = len(datasets)

    fig, axes = plt.subplots(num_features, num_datasets, figsize=(15, 5 * num_features))

    custom_palette = sns.color_palette("PuBuGn", 8) 

    for i, col in enumerate(cat_columns):
        for j, (data, title) in enumerate(zip(datasets, titles)):
            counts = data[col].value_counts()

            wedges, texts, autotexts = axes[i, j].pie(
                counts,
                labels=counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=custom_palette[:len(counts)]
            )

            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            axes[i, j].add_artist(centre_circle)

            axes[i, j].set_title(f'{col} Distribution - {title}')
            axes[i, j].axis('equal')  

    plt.tight_layout()
    plt.show()

datasets = [train_data, test_data, train_ex_data]
titles = ['Train Data', 'Test Data', 'Train Extra Data']

plot_categorical_distributions_grid(cat_columns_train, datasets, titles)


# Identify categorical columns
cat_columns_train = train_data.select_dtypes(include=['object']).columns.tolist()

def plot_grouped_bar(data1, data2, data3, cat_columns):
    sns.set(style="whitegrid", palette="PuBuGn")  
    
    for col in cat_columns:
        plt.figure(figsize=(12, 6))
        
        # Count occurrences in each dataset
        count_train = data1[col].value_counts().reset_index()
        count_train.columns = [col, 'Train']
        
        count_test = data2[col].value_counts().reset_index()
        count_test.columns = [col, 'Test']
        
        count_train_ex = data3[col].value_counts().reset_index()
        count_train_ex.columns = [col, 'Train Extra']
        
        # Merge counts into a single DataFrame
        merged_counts = count_train.merge(count_test, on=col, how='outer').merge(count_train_ex, on=col, how='outer').fillna(0)
        
        # Melt the DataFrame for easier plotting
        melted_counts = merged_counts.melt(id_vars=col, var_name='Dataset', value_name='Count')

        ax = sns.barplot(data=melted_counts, x=col, y='Count', hue='Dataset', palette="PuBuGn")
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=10)

        plt.title(f'Comparison of {col} Across Datasets', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Dataset', fontsize=10)

        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)  

        plt.tight_layout()
        plt.show()

plot_grouped_bar(train_data, test_data, train_ex_data, cat_columns_train)


# Create a scatter plot of Price vs Weight Capacity (kg) colored by Brand
plt.figure(figsize=(12, 6))
sns.scatterplot(data=train_data, x='Weight Capacity (kg)', y='Price', hue='Brand', palette='PuBuGn')

plt.title('Price Distribution by Weight Capacity and Brand', fontsize=14)
plt.xlabel('Weight Capacity (kg)', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(title='Brand', fontsize=10)

plt.grid(axis='both', color='gray', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()


# Create a scatter plot of Price vs Compartments colored by Brand
# plt.figure(figsize=(12, 6))
# sns.scatterplot(data=train_data, x='Compartments', y='Price', hue='Brand', palette='PuBuGn')

plt.figure(figsize=(12, 6))
sns.stripplot(data=train_data, x='Compartments', y='Price', hue='Brand', palette='PuBuGn', jitter=True)

plt.title('Price Distribution by Compartments and Brand', fontsize=14)
plt.xlabel('Compartments', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(title='Brand', fontsize=10)

plt.grid(axis='both', color='gray', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()


# Convert counts into a hierarchical structure
df = train_data.groupby(['Brand', 'Style']).size().reset_index(name='Count')

custom_palette = sns.color_palette("PuBuGn", 6)
custom_palette_hex = custom_palette.as_hex()

fig = px.sunburst(df, path=['Brand', 'Style'], values='Count',
                  title='Brand vs. Style Distribution',
                  color_discrete_sequence=custom_palette_hex)
fig.show()


custom_palette = sns.color_palette("PuBuGn", len(train_data['Material'].unique()))

plt.figure(figsize=(12, 6))

sns.countplot(x='Size', hue='Material', data=train_data, palette=custom_palette)

plt.title('Count of Materials by Size')
plt.xlabel('Size')
plt.ylabel('Count')
plt.xticks(rotation=45)  
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)
plt.legend(title='Material')

for p in plt.gca().patches:
    plt.annotate(f'{int(p.get_height())}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='baseline', 
                 color='black', fontsize=10, 
                 xytext=(0, 5), 
                 textcoords='offset points')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))

# Create the count plot
sns.countplot(x='Waterproof', hue='Material', data=train_data, palette=custom_palette)

plt.title('Count of Materials by Waterproof Status')
plt.xlabel('Waterproof Status')
plt.ylabel('Count')
plt.xticks(rotation=0)  
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)
plt.legend(title='Material')

for p in plt.gca().patches:
    plt.annotate(f'{int(p.get_height())}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='baseline', 
                 color='black', fontsize=10, 
                 xytext=(0, 5), 
                 textcoords='offset points')

plt.tight_layout()
plt.show()


df = train_data.groupby(['Brand', 'Size', 'Waterproof']).size().reset_index(name='Count')

custom_palette = sns.color_palette("PuBuGn", 6)
custom_palette_hex = custom_palette.as_hex()

fig = px.sunburst(df, path=['Brand', 'Size', 'Waterproof'], values='Count',
                  title='Brand vs. Size vs. Waterproof Distribution',
                  color_discrete_sequence=custom_palette_hex)
fig.show()


# Define the custom color palette
custom_palette = sns.color_palette("PuBuGn", len(train_data['Color'].unique()))

def plot_color_by_brand(ax, df, brand):
    filtered_data = df[df['Brand'] == brand].dropna(subset=['Color'])  # Drop rows with NaN in 'Color'
    
    sns.countplot(y='Color', data=filtered_data, ax=ax[0],
                 palette=custom_palette[:len(filtered_data['Color'].unique())])
    ax[0].set_title(f'Color Count for {brand}')
    ax[0].set_xlabel('Count')
    ax[0].set_ylabel('Color')
    ax[0].tick_params(axis='y', labelsize=8)

    for p in ax[0].patches:
        ax[0].annotate(f'{int(p.get_width())}',
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center',
                        color='black', fontsize=8)

    ax[0].set_axisbelow(True)
    ax[0].grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  
    sns.despine(left=True, bottom=True)

    color_counts = filtered_data['Color'].value_counts()
    ax[1].pie(color_counts, labels=color_counts.index,
               autopct='%1.1f%%', startangle=90,
               colors=custom_palette[:len(color_counts)])
    ax[1].set_title(f'Color Distribution for {brand}')
    ax[1].axis('equal')
  
valid_brands = [brand for brand in train_data['Brand'].unique() 
               if train_data[train_data['Brand'] == brand].dropna(subset=['Color']).shape[0] > 0]

num_brands = len(valid_brands)

fig, axes = plt.subplots(num_brands, 2, figsize=(14, 4*num_brands))

for i, brand in enumerate(valid_brands):
    plot_color_by_brand(axes[i], train_data, brand)

plt.tight_layout()
plt.show()


custom_palette = sns.color_palette("PuBuGn", len(train_data['Material'].unique()))

def plot_material_by_compartment(ax, df, compartment):
    filtered_data = df[df['Laptop Compartment'] == compartment].dropna(subset=['Material'])  # Drop rows with NaN in 'Material'
    
    if filtered_data.empty:  # Check if filtered data is empty
        ax[0].set_title(f'No Data for {compartment}')
        ax[1].set_title(f'No Data for {compartment}')
        return
    
    sns.countplot(y='Material', data=filtered_data, ax=ax[0],
                 palette=custom_palette[:len(filtered_data['Material'].unique())])
    ax[0].set_title(f'Material Count for Laptop Compartment ({compartment})')
    ax[0].set_xlabel('Count')
    ax[0].set_ylabel('Material')
    ax[0].tick_params(axis='y', labelsize=8)

    for p in ax[0].patches:
        ax[0].annotate(f'{int(p.get_width())}',
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center',
                        color='black', fontsize=8)

    ax[0].set_axisbelow(True)
    ax[0].grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  
    sns.despine(left=True, bottom=True)

    material_counts = filtered_data['Material'].value_counts()
    ax[1].pie(material_counts, labels=material_counts.index,
               autopct='%1.1f%%', startangle=90,
               colors=custom_palette[:len(material_counts)])
    ax[1].set_title(f'Material Distribution for Laptop Compartment ({compartment})')
    ax[1].axis('equal')
  
# Filter compartments with data
valid_compartments = [compartment for compartment in train_data['Laptop Compartment'].unique() 
                     if train_data[train_data['Laptop Compartment'] == compartment].dropna(subset=['Material']).shape[0] > 0]

num_compartments = len(valid_compartments)

fig, axes = plt.subplots(num_compartments, 2, figsize=(14, 4*num_compartments))

for i, compartment in enumerate(valid_compartments):
    plot_material_by_compartment(axes[i], train_data, compartment)

plt.tight_layout()
plt.show()


custom_palette = sns.color_palette("PuBuGn", len(train_data['Material'].unique()))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter Plot
sns.histplot(data=train_data, x='Weight Capacity (kg)', hue='Material',
             bins=10, palette=custom_palette,
             ax=axes[0], kde=True)
axes[0].set_title('Weight Capacity Distribution by Material')
axes[0].set_xlabel('Weight Capacity')
axes[0].set_ylabel('Frequency')
axes[0].set_axisbelow(True)
axes[0].grid(axis='y', color='gray', linestyle='--')

# Box Plot
sns.boxplot(x='Material', y='Weight Capacity (kg)', data=train_data, palette=custom_palette, ax=axes[1])
axes[1].set_title('Weight Capacity Distribution by Material (Box Plot)')
axes[1].set_xlabel('Material')
axes[1].set_ylabel('Weight Capacity (kg)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', color='gray', linestyle='--')

plt.tight_layout()
plt.show()


# Define the custom color palette
custom_palette = sns.color_palette("PuBuGn", len(train_data['Material'].unique()))

plt.figure(figsize=(12, 6))

# Create the scatter plot
sns.scatterplot(data=train_data, x='Weight Capacity (kg)', y='Price', hue='Material', palette=custom_palette)

plt.title('Price Distribution by Weight Capacity and Material', fontsize=14)
plt.xlabel('Weight Capacity (kg)', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(title='Material', fontsize=10)

plt.grid(axis='both', color='gray', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()


merged_train_data = pd.concat([train_data, train_ex_data], ignore_index=True)

# Verify the shapes of the datasets
print("Shape of Train Data:", train_data.shape)
print("Shape of Train Extra Data:", train_ex_data.shape)
print("Shape of Merged Train Data:", merged_train_data.shape)

# Display the merged dataset
print("\nMerged Data Sample:")
display(merged_train_data.head())


# Display data types before merging
print("Data types before merging:")
print("Train Data Types:\n", train_data.dtypes)
print("\nTrain Extra Data Types:\n", train_ex_data.dtypes)

# Display data types after merging
print("\nData Types of Merged Train Data:")
print(merged_train_data.dtypes)


# Identify missing values in the merged dataset
missing_values = merged_train_data.isnull().sum()

# Display missing values and their counts
print("Missing Values in Merged Dataset:")
print(missing_values[missing_values > 0])  


# Identify missing values in each dataset
missing_merged_train = merged_train_data.isnull().sum()
missing_test = test_data.isnull().sum()

def create_donut_chart(data, ax, title):
    missing_values = data[data > 0]
    
    wedges, texts, autotexts = ax.pie(
        missing_values,
        labels=missing_values.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("PuBuGn", len(missing_values))
    )
    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

create_donut_chart(missing_merged_train, axes[0], 'Missing Values in Merged Dataset')
create_donut_chart(missing_test, axes[1], 'Missing Values in Test Dataset')

plt.tight_layout()
plt.show()


# Impute missing numerical data with median values
for col in test_data.select_dtypes(include=['number']).columns:
    median_value = merged_train_data[col].median()
    merged_train_data[col].fillna(median_value, inplace=True)
    test_data[col].fillna(median_value, inplace=True)


# Function to create a summary table for missing values and data types
def missing_values_summary(df):
    missing_count = df.isnull().sum()
    data_types = df.dtypes
    return pd.DataFrame({
        'Data Type': data_types,
        'Missing Values Count': missing_count
    })

# Create summary tables for merged train and test datasets
merged_train_summary = missing_values_summary(merged_train_data)
test_summary = missing_values_summary(test_data)

# Filter to show only columns with missing values
print("Merged Train Dataset Summary (Post-Imputation):")
print(merged_train_summary[merged_train_summary['Missing Values Count'] > 0])

print("\nTest Dataset Summary (Post-Imputation):")
print(test_summary[test_summary['Missing Values Count'] > 0])


# Impute missing object data with 'None'
for col in merged_train_data.select_dtypes(include=['object']).columns:
    merged_train_data[col].fillna('None', inplace=True)
    test_data[col].fillna('None', inplace=True)


# Function to create a summary table for missing values and data types
def missing_values_summary(df):
    missing_count = df.isnull().sum()
    data_types = df.dtypes
    return pd.DataFrame({
        'Data Type': data_types,
        'Missing Values Count': missing_count
    })

# Create summary tables for merged train and test datasets
merged_train_summary = missing_values_summary(merged_train_data)
test_summary = missing_values_summary(test_data)

print("Merged Train Dataset Summary (Post-Imputation):")
print(merged_train_summary)

print("\nTest Dataset Summary (Post-Imputation):")
print(test_summary)


non_numerical_columns = merged_train_data.select_dtypes(include=['object']).columns.tolist()

# Display unique values for each categorical column
for col in non_numerical_columns:
    print(f"\nColumn: {col}")
    print(f"Unique Values: {merged_train_data[col].unique()}")


non_numerical_columns = test_data.select_dtypes(include=['object']).columns.tolist()

# Display unique values for each categorical column
for col in non_numerical_columns:
    print(f"\nColumn: {col}")
    print(f"Unique Values: {test_data[col].unique()}")


# Define the target variable
target_variable = 'Price'


# Analyze each categorical column with respect to the target variable 'Price'
for col in non_numerical_columns:
    print(f"\nColumn: {col}")
    
    # Calculate average price for each unique value in the categorical column
    average_price = merged_train_data.groupby(col)[target_variable].mean().sort_values(ascending=False)
    print(f"Average Price by {col}:\n{average_price}\n")
    
    plt.figure(figsize=(12, 4))
    sns.barplot(x=average_price.index, y=average_price.values, palette='PuBuGn')
    plt.title(f'Average Price by {col}')
    plt.xlabel(col)
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.7) 
    plt.tight_layout()
    plt.show()


# Define the column of interest
column_of_interest = 'Weight Capacity (kg)'

# List of categorical columns
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color', 'Compartments']

# Analyze each categorical column
for col in categorical_columns:
    print(f"\nColumn: {col}")
    
    # Calculate average weight capacity for each unique value in the categorical column
    average_weight_capacity = merged_train_data.groupby(col)[column_of_interest].mean().sort_values(ascending=False)
    print(f"Average Weight Capacity by {col}:\n{average_weight_capacity}\n")
    
    plt.figure(figsize=(12, 4))
    sns.barplot(x=average_weight_capacity.index, y=average_weight_capacity.values, palette='PuBuGn')
    plt.title(f'Average Weight Capacity by {col}')
    plt.xlabel(col)
    plt.ylabel('Average Weight Capacity (kg)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.7) 
    plt.tight_layout()
    plt.show()


# Instantiate TargetEncoder
TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

features = test_data.columns.tolist()

for col in features:
    TE.fit(merged_train_data[col], merged_train_data[target_variable])  # Fit on training data
    merged_train_data[col] = TE.transform(merged_train_data[col])  # Transform training data
    test_data[col] = TE.transform(test_data[col])  # Transform test data

# Function to display summary
def display_summary(df, name):
    print(f"\n{name} Summary:")
    print("-" * 30)
    print("\nData Info:")
    df.info()
    print("\nFirst Rows:")
    display(df.head().T)

# Display summary of the transformed datasets
display_summary(merged_train_data, "Merged Train Dataset")
display_summary(test_data, "Test Dataset")


# List of numerical features to plot (excluding 'Price')
numerical_features = merged_train_data.select_dtypes(include=['number']).columns.tolist()
if 'Price' in numerical_features:
    numerical_features.remove('Price')

def plot_individual_boxplot(data, columns, ax, title):
    sns.boxplot(data=data[columns], ax=ax, palette="PuBuGn", showfliers=False)
    ax.set_title(title)
    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.grid(color='gray', linestyle='--', linewidth=0.7)  
    ax.tick_params(axis='x', rotation=45)  

def plot_boxplot(train, test, columns):
    num_features = len(columns)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6 * 2), sharex=False)
    plot_individual_boxplot(train, columns, axes[0], 'Boxplot of Merged Train Data')
    plot_individual_boxplot(test, columns, axes[1], 'Boxplot of Test Data')
    plt.tight_layout()
    plt.show()

plot_boxplot(merged_train_data, test_data, numerical_features)


# Calculate the correlation matrix (including 'Price')
correlation_matrix = merged_train_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='PuBuGn', fmt=".2f", linewidths=0.5, square=True)
plt.title('Correlation Heatmap of Merged Train Dataset')
plt.xticks()  
plt.yticks() 
plt.tight_layout() 
plt.show()


# Create a correlation table for the target variable with other features
target_correlation_table = merged_train_data.corr()[[target_variable]].sort_values(by=target_variable, ascending=False)

# Displaying the table
print("Correlation of 'Price' with Other Features:")
display(target_correlation_table)


# Define the target variable and features
target_variable = 'Price'
features = merged_train_data.select_dtypes(include=['number']).columns.tolist()
features.remove(target_variable)  # Exclude target variable from features


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    merged_train_data[features], 
    merged_train_data[target_variable], 
    test_size=0.2, 
    random_state=42
)


model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    random_state=42,
    num_leaves=31, 
    learning_rate=0.1, 
    n_estimators=1000, 
    objective='regression',
    device='gpu',
    verbose=-1
)


# Define dictionary to store evaluation results
evals_result = {}

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[
        lgb.record_evaluation(evals_result),
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ]
)


# Extract validation RMSE values
val_rmse = evals_result['valid_0']['rmse']
min_rmse = min(val_rmse)
min_index = val_rmse.index(min_rmse)

# Displaying Minimum Validation RMSE
print(f"Validation RMSE(min): {min_rmse:.4f}")


importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="PuBuGn_r")
plt.title('Feature Importance')
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()


# Create subplots for Actual vs Predicted and Residual Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
palette = sns.color_palette("PuBuGn", 2)

# Actual vs Predicted Scatter Plot
sns.scatterplot(x=y_val, y=model.predict(X_val), alpha=0.6, color=palette[1], ax=axes[0])
axes[0].plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], '--', color="red")  
axes[0].set_xlabel("Actual Price")
axes[0].set_ylabel("Predicted Price")
axes[0].set_title("Actual vs Predicted Scatter Plot")
axes[0].grid(color='gray', linestyle='--', linewidth=0.7)

# Residual Plot
residuals = y_val - model.predict(X_val)
sns.scatterplot(x=model.predict(X_val), y=residuals, alpha=0.6, color=palette[1], ax=axes[1])
axes[1].axhline(y=0, color="red", linestyle="--")  
axes[1].set_xlabel("Predicted Price")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")
axes[1].grid(color='gray', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()


# Predict on Test Data
y_test_pred = model.predict(test_data[features])

# Box Plot and Histogram as Subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 3]})

sns.boxplot(x=y_test_pred, ax=axes[0], color=sns.color_palette("PuBuGn", 1)[0])
axes[0].set_title("Box Plot of Predicted Prices")

sns.histplot(y_test_pred, bins=50, kde=True, ax=axes[1], color=sns.color_palette("PuBuGn", 1)[0])
axes[1].set_title("Histogram of Predicted Prices")
axes[1].set_xlabel("Price")

axes[0].grid(color='gray', linestyle='--', linewidth=0.7)
axes[1].grid(color='gray', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()



# Save Predictions to CSV
submission = pd.DataFrame({'id': test_data.index, 'Price': y_test_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

# Display first 10 predictions
display(submission.head(10))

