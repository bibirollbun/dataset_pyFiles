# table manipulation, calculating
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100) # increase the maximum number of columns

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


display(df_train)

# 欠損している行を出す。
df_train_nan_index = df_train[df_train.isnull().any(axis=1)].index # 欠損値を含む行のインデックスを保存
df_train_nan_rows = df_train.loc[df_train_nan_index] # 後でインデックスを使用して行を呼び出す
display(df_train_nan_index)
display(df_train_nan_rows)


display(df_test)

# 欠損している行を出す。
df_test_nan_index = df_test[df_test.isnull().any(axis=1)].index # 欠損値を含む行のインデックスを保存
df_test_nan_rows = df_test.loc[df_test_nan_index] # 後でインデックスを使用して行を呼び出す
display(df_test_nan_index)
display(df_test_nan_rows)


# Check if each column has a unique value of 0 or 1, and assign 1 or 0
def unique_to_binary(column):
    unique_values = column.unique()
    if len(unique_values) == 2 and set(unique_values) == {0, 1}:
        return 1
    else:
        return 0

def find_outliers(df):
    outliers_rate_dict = {}

    for column_name in df.columns:
        column = df[column_name]

        # Skip if not a numeric column
        if not pd.api.types.is_numeric_dtype(column):
            print(f"列 '{column_name}' is skipped because it is not a numeric type.")
            continue

        # Calculate the mean and standard deviation of the data
        mean = column.mean()
        std = column.std()

        # Set outlier threshold
        threshold = 2  # Adjust this value to change the outlier criteria

        # Set conditions for detecting outliers
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        # Detect outliers
        outliers = (column < lower_bound) | (column > upper_bound)

        # Calculate the percentage of outliers
        outliers_rate = outliers.sum() / len(column)

        # Save results to dictionary
        outliers_rate_dict[column_name] = outliers_rate

    return outliers_rate_dict


def create_data_inspection(df):
    """データフレームのデータ検査情報を生成する関数"""

    # process:1
    data_inspection = pd.DataFrame({
        'column_name': df.columns,
        'data_type': df.dtypes,
        'num_of_rows': len(df),
        'num_of_uniques': df.nunique(),
        'num_of_duplicated_rows': len(df) - df.nunique(),
        'non-null_cnt': df.count().values,
        'null_cnt': df.isnull().sum(),
        'null_rate': (df.isnull().sum() / len(df)),
    })

    # process:2
    description = df.describe().T.reset_index().rename(columns={'index': 'column_name'})
    median = df.median(numeric_only=True).reset_index().rename(columns={'index': 'column_name', 0: 'median'})

    # Calculate the most frequent value for each column
    mode_values = df.mode(numeric_only=True).iloc[0]

    # Calculate the percentage of most frequent values
    mode_ratios = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            mode_value = mode_values[column]
            mode_ratio = (df[column] == mode_value).sum() / len(df)
            mode_ratios[column] = mode_ratio

    # Convert the most common value and its percentage into a data frame
    mode = mode_values.reset_index().rename(columns={'index': 'column_name', 0: 'mode'})
    mode['rate mode'] = mode['column_name'].map(mode_ratios)

    # process:3
    data_inspection = pd.merge(data_inspection, description, how = 'left', on = 'column_name')
    data_inspection = pd.merge(data_inspection, mode, how = 'left', on = 'column_name')

    outliers_rate = find_outliers(df)
    outliers_rate = pd.DataFrame(list(outliers_rate.items()), columns=['column_name', 'outliers_rate']).reset_index(drop=True)
    data_inspection = pd.merge(data_inspection, outliers_rate, how = 'left', on = 'column_name')

    # ★Add new criteria
    skew = df.skew().reset_index().rename(columns={'index': 'column_name', 0: 'skewness'})
    kurt = df.kurt().reset_index().rename(columns={'index': 'column_name', 0: 'kurtosis'})

    data_inspection = pd.merge(data_inspection, skew, how = 'left', on = 'column_name')
    data_inspection = pd.merge(data_inspection, kurt, how = 'left', on = 'column_name')

    # ★Add new criteria
    # Calculating correlation coefficients (numeric columns only), where errors may occur.
    correlation_matrix = df.corr(numeric_only=True)
    target_corr = pd.DataFrame({'column_name': correlation_matrix.columns, 'target correlation': correlation_matrix.iloc[:, -1]})

    data_inspection = pd.merge(data_inspection, target_corr, how = 'left', on = 'column_name')

    # process:4
    data_inspection_else = pd.DataFrame({
        'column_name': df.columns,
        'flag_or_not': df.apply(unique_to_binary),
        'columns_details': None,
        'remarks': None,
        'trigger': None,
        'dataset_name': None,
        'existence_of_table_definition': None,
        # 'data_exmaple': df.head(1).T[0] # or df.dropna().T.iloc[0]
        'data_exmaple': df.head(1).T.iloc[:, 0]
    })

    data_inspection = pd.merge(data_inspection, data_inspection_else, how = 'left', on = 'column_name')

    # visualization `data_inspection`
    # blue → green → yellow
    data_inspection_styled = data_inspection.style.background_gradient(cmap='viridis', subset=pd.IndexSlice[:, data_inspection.select_dtypes(include=['number']).columns])

    return data_inspection_styled


data_inspection_styled = create_data_inspection(df_train)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false')


data_inspection_styled = create_data_inspection(df_test)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false')


# Create a 1-by-2 subplot
fig, axes = plt.subplots(2, 1, figsize=(8, 30))

# Missing value heatmap of training data
sns.heatmap(df_train.isnull(), cbar=False, cmap='viridis', ax=axes[0])
axes[0].set_title('Missing Data in Train')

# Missing value heatmap for test data
sns.heatmap(df_test.isnull(), cbar=False, cmap='viridis', ax=axes[1])
axes[1].set_title('Missing Data in Test')

# Adjust layout
plt.tight_layout()
plt.show()


# # EDA for both of numeric and category
# # custom palette of colors
# custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# # Create a list of variables (both numerical and categorical data)
# numerical_variables = df_train.select_dtypes(include=['number']).columns
# categorical_variables = df_train.select_dtypes(include=['object']).columns

# # A function to create plots for each variable
# def create_variable_plots(variable, data_type='numerical'):
#     sns.set_style('whitegrid')
    
#     # For numeric data
#     if data_type == 'numerical':
#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#         # Box plot
#         plt.subplot(1, 2, 1)
#         sns.boxplot(data=pd.concat([df_train, df_test]), x=variable, y="dataset", palette=custom_palette)
#         plt.xlabel(variable)
#         plt.title(f"Box plot for {variable}")

#         # histgram
#         plt.subplot(1, 2, 2)
#         sns.histplot(data=df_train, x=variable, color=custom_palette[0], kde=True, bins=30, label="train")
#         if variable in df_test.columns:
#             sns.histplot(data=df_test, x=variable, color=custom_palette[1], kde=True, bins=30, label="test")
        
#         plt.xlabel(variable)
#         plt.ylabel("Frequency")
#         plt.title(f"Histogram for {variable} [train & test]" if variable in df_test.columns else f"Histogram for {variable} [train]")
#         plt.legend()

#     # For categorical data
#     elif data_type == 'categorical':
#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#         # pie chart
#         plt.subplot(1, 2, 1)
#         gender_counts = pd.concat([df_train[variable], df_test[variable]]).value_counts(normalize=True)
#         gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=custom_palette, startangle=90, ax=plt.gca())
#         plt.title(f"Pie chart for {variable}")

#         # countplat
#         plt.subplot(1, 2, 2)
#         sns.countplot(data=pd.concat([df_train, df_test]), x=variable, hue="dataset", palette=custom_palette)
#         plt.title(f"Count plot for {variable}")

#     plt.tight_layout()
#     plt.show()

# # Create plots for numerical data
# for variable in numerical_variables:
#     create_variable_plots(variable, data_type='numerical')

# # Create plots for categorical data
# for variable in categorical_variables:
#     create_variable_plots(variable, data_type='categorical')



# box plot and histogram

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
df_train['dataset'] = 'train'
df_test['dataset'] = 'test'

# Get only numeric variables
numerical_variables = df_train.select_dtypes(include=['number']).columns

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    
    # Check if variable exists in both df_train and df_test
    if variable in df_train.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Box plot
        plt.subplot(1, 2, 1)
        if variable in df_test.columns:
            sns.boxplot(data=pd.concat([df_train, df_test]), x=variable, y="dataset", palette=custom_palette)
        else:
            sns.boxplot(data=df_train, x=variable, y="dataset", palette=custom_palette)
        plt.xlabel(variable)
        plt.title(f"box plot for {variable}")

        # Separate Histograms
        plt.subplot(1, 2, 2)
        sns.histplot(data=df_train, x=variable, color=custom_palette[0], kde=True, bins=30, label="train")
        if variable in df_test.columns:
            sns.histplot(data=df_test, x=variable, color=custom_palette[1], kde=True, bins=30, label="test")
        
        plt.xlabel(variable)
        plt.ylabel("frequency")
        plt.title(f"histogram for {variable} [train & test]" if variable in df_test.columns else f"histogram for {variable} [train]")
        plt.legend()

        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the plots
        plt.show()

# Perform univariate analysis for each variable
for variable in numerical_variables:
    create_variable_plots(variable)

# delete unnecessary column
del df_train['dataset']
del df_test['dataset']


# Define custom color palette for Train, Test, and Original datasets
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

# Function to create Wind Rose plot in a subplot
def create_wind_rose(ax, data, dataset_name, color):
    # Convert wind direction to radians
    wind_direction_radians = np.radians(data['winddirection'].dropna())

    # Create histogram bins (every 10°)
    bins = np.linspace(0, 2*np.pi, 37)  # 36 bins (every 10°)
    counts, bin_edges = np.histogram(wind_direction_radians, bins=bins)

    # Plot on the polar axis with improved style
    bars = ax.bar(bin_edges[:-1], counts, width=np.radians(10), color=color, edgecolor='black', alpha=0.8)

    # Formatting for professional appearance
    ax.set_theta_zero_location("N")  # North is at 0°
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))  # Tick labels every 45°
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10, fontweight='bold')

    # Add grid and labels for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_yticklabels([])  # Remove radial labels to avoid clutter
    ax.set_title(f"Wind Direction ({dataset_name})", fontsize=12, fontweight='bold', pad=10)

# Create a single row with three wind rose plots
fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': 'polar'})

# Generate wind rose plots for Train, Test, and Original datasets
create_wind_rose(axes[0], df_train, "Train Data", custom_palette[0])  # Blue
create_wind_rose(axes[1], df_test, "Test Data", custom_palette[1])    # Red

# Adjust layout for better spacing
plt.tight_layout()
plt.show()





# Create a subplot (1 column, 2 rows)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Correlation matrix of df_train
# (if necessary, extract only highly correlated variables.ex：mask=(corr < 0.8)))
sns.heatmap(df_train[numerical_variables].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[0])
axes[0].set_title('Train Data Feature Correlation')

# # Correlation matrix of df_test
# (if necessary, extract only highly correlated variables.ex：mask=(corr < 0.8)))
sns.heatmap(df_test.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title('Test Data Feature Correlation')

# Layout adjustment
plt.tight_layout()
plt.show()


# pairplot を作成
print('Pairplot of Train Data')
sns.pairplot(df_train)
plt.show()

print('------------------------------------------------------------------------------------------------------')

print('Pairplot of Test Data')
sns.pairplot(df_test)
plt.show()


# # Cross-tabulation table (combination of category 1 and category 2)
# contingency_table = pd.crosstab(df_test['category1'], df_test['category2'])

# # Heatmap drawing
# plt.figure(figsize=(8, 6))
# sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title('Contingency Table Heatmap')
# plt.show()





# from statsmodels.tsa.seasonal import seasonal_decompose

# # 時系列データを設定（'date'をインデックスにして'sales'をターゲット）
# ts = df.set_index('date')['sales']

# # 時系列の分解（加法モデルで分解、周期は365日と仮定）
# decomposition = seasonal_decompose(ts, model='additive', period=365)

# # 分解結果から各成分を取得
# trend_component = decomposition.trend
# seasonal_component = decomposition.seasonal
# residual_component = decomposition.resid

# # トレンド + 季節性 + ノイズを合成
# reconstructed_series = trend_component + seasonal_component + residual_component

# # 4行に分けてプロットを作成
# plt.figure(figsize=(10, 12))

# # 1行目：再構成された時系列（トレンド + 季節性 + 残差）
# plt.subplot(4, 1, 1)
# plt.plot(reconstructed_series, label='Reconstructed Series (Trend + Seasonal + Residual)')
# plt.title('Reconstructed Time Series (Trend + Seasonal + Residual)')

# # 2行目：トレンド成分
# plt.subplot(4, 1, 2)
# plt.plot(trend_component)
# plt.title('Trend Component')

# # 3行目：季節性成分
# plt.subplot(4, 1, 3)
# plt.plot(seasonal_component)
# plt.title('Seasonal Component')

# # 4行目：残差成分
# plt.subplot(4, 1, 4)
# plt.plot(residual_component)
# plt.title('Residual (Noise) Component')

# # レイアウトの調整
# plt.tight_layout()
# plt.show()


# Define colors for Train and Test data
train_color = '#3498db'  # Blue
test_color = '#e74c3c'   # Red

# Create the plot
plt.figure(figsize=(12, 5))

# Plot Train Data
plt.plot(df_train['id'], df_train['day'], linestyle='-', color=train_color, label='Train Data', alpha=0.7)

# Plot Test Data
plt.plot(df_test['id'], df_test['day'], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

# Formatting
plt.xlabel('ID')
plt.ylabel('Day')
plt.title('Trend Plot: Day vs ID')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.show()


# Generate the expected repeating pattern (1-365 for 6 years)
expected_pattern = np.tile(np.arange(1, 366), 6)  # Repeats 1-365 exactly 6 times

# Check for incorrect labels
df_train['expected_day'] = expected_pattern[:len(df_train)]  # Assign expected pattern
df_train['day_mismatch'] = df_train['day'] != df_train['expected_day']  # Flag mismatches

flag_color = '#8B0000'   # Dark Red (for mismatched days)

# Generate expected repeating pattern (1-365 for 6 years)
expected_pattern = np.tile(np.arange(1, 366), 6)  # Repeats 1-365 exactly 6 times

# Assign expected pattern and flag mismatches
df_train['expected_day'] = expected_pattern[:len(df_train)]
df_train['day_mismatch'] = df_train['day'] != df_train['expected_day']  # Boolean flag

# Create the plot
plt.figure(figsize=(12, 5))

# Plot Train Data
plt.plot(df_train['id'], df_train['day'], linestyle='-', color=train_color, label='Train Data', alpha=0.7)

# Plot Test Data
plt.plot(df_test['id'], df_test['day'], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

# Flag mismatched days using red markers
plt.scatter(
    df_train.loc[df_train['day_mismatch'], 'id'],  # X-axis: IDs of mismatched days
    df_train.loc[df_train['day_mismatch'], 'day'], # Y-axis: Corresponding incorrect days
    color=flag_color, marker='X', s=80, label='Mismatched Days', alpha=0.9
)

# Formatting
plt.xlabel('ID')
plt.ylabel('Day')
plt.title('Trend Plot: Day vs ID (Flagging Mismatches)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.show()


# Define colors
train_color = '#3498db'  # Blue
test_color = '#e74c3c'   # Red
rainfall_colors = {0: '#f1c40f', 1: '#2980b9'}  # Dark Yellow (no rainfall), Blue (rainfall)

# Numerical columns to plot
numerical_columns = df_test.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in ['id', 'day', 'rainfall']:
    if col in numerical_columns:
        numerical_columns.remove(col)

# Plotting loop for each numerical variable
for column in numerical_columns:
    # Create figure with specific layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # ---- Trend Plot (ID vs Variable) ----
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(df_train['id'], df_train[column], linestyle='-', color=train_color, label='Train Data', alpha=0.7)
    ax0.plot(df_test['id'], df_test[column], linestyle='-', color=test_color, label='Test Data', alpha=0.7)

    ax0.set_xlabel('ID', fontsize=14)
    ax0.set_ylabel(column, fontsize=14)
    ax0.set_title(f'Trend Plot: {column} vs ID', fontsize=16, fontweight='bold')  # ✅ Fix applied
    ax0.legend(fontsize=12)
    ax0.grid(True, linestyle='--', alpha=0.5)

    # ---- Scatter Plot (Day vs Variable) ----
    ax1 = fig.add_subplot(gs[1, 0])
    scatter = ax1.scatter(
        df_train['day'], df_train[column],
        c=df_train['rainfall'].map(rainfall_colors), alpha=0.7
    )
    ax1.set_xlabel('Day', fontsize=14)
    ax1.set_ylabel(column, fontsize=14)
    ax1.set_title(f'Scatter Plot: {column} vs Day (by Rainfall)', fontsize=16, fontweight='bold')  # ✅ Fix applied

    # Custom legend for rainfall
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='No Rainfall',
               markersize=10, markerfacecolor=rainfall_colors[0]),
        Line2D([0], [0], marker='o', color='w', label='Rainfall',
               markersize=10, markerfacecolor=rainfall_colors[1])
    ]
    ax1.legend(handles=legend_elements, title="Rainfall", fontsize=12, title_fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- KDE Plot (Variable distribution by Rainfall) ----
    ax2 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=df_train, x=column, hue='rainfall', palette=rainfall_colors, ax=ax2, fill=True, common_norm=False, alpha=0.6)

    ax2.set_xlabel(column, fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_title(f'Distribution (KDE) of {column} by Rainfall', fontsize=16, fontweight='bold')  
    ax2.legend(title='Rainfall', fontsize=12, title_fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout spacing
    plt.tight_layout(pad=3.0)
    plt.show()

    # ---- Add clear separation after each variable ----
    plt.figure(figsize=(16, 0.3))  # Adjust spacing
    plt.axhline(y=0, color='gray', linewidth=5, linestyle='-') 
    plt.axis('off')
    plt.show()





df_train_rainfall_1 = df_train[df_train['rainfall'] == 1]
df_train_rainfall_1


data_inspection_styled = create_data_inspection(df_train_rainfall_1)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false')


df_train_rainfall_0 = df_train[df_train['rainfall'] == 0]
df_train_rainfall_0


data_inspection_styled = create_data_inspection(df_train_rainfall_0)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false')


# box plot and histogram

# Define a custom color palette
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
df_train_rainfall_1['dataset'] = 'rainfall_1'
df_train_rainfall_0['dataset'] = 'rainfall_0'

# Get only numeric variables
numerical_variables = df_train_rainfall_1.select_dtypes(include=['number']).columns

# Function to create and display a row of plots for a single variable
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    
    # Check if variable exists in both df_train_rainfall_1 and df_train_rainfall_0
    if variable in df_train_rainfall_1.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Box plot
        plt.subplot(1, 2, 1)
        if variable in df_train_rainfall_0.columns:
            sns.boxplot(data=pd.concat([df_train_rainfall_1, df_train_rainfall_0]), x=variable, y="dataset", palette=custom_palette)
        else:
            sns.boxplot(data=df_train_rainfall_1, x=variable, y="dataset", palette=custom_palette)
        plt.xlabel(variable)
        plt.title(f"box plot for {variable}")

        # Separate Histograms
        plt.subplot(1, 2, 2)
        sns.histplot(data=df_train_rainfall_1, x=variable, color=custom_palette[0], kde=True, bins=30, label="rainfall_1")
        if variable in df_train_rainfall_0.columns:
            sns.histplot(data=df_train_rainfall_0, x=variable, color=custom_palette[1], kde=True, bins=30, label="rainfall_0")
        
        plt.xlabel(variable)
        plt.ylabel("frequency")
        plt.title(f"histogram for {variable} [rainfall_1 & rainfall_0]" if variable in df_train_rainfall_0.columns else f"histogram for {variable} [train]")
        plt.legend()

        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the plots
        plt.show()

# Perform univariate analysis for each variable
for variable in numerical_variables:
    create_variable_plots(variable)

# delete unnecessary column
del df_train_rainfall_1['dataset']
del df_train_rainfall_0['dataset']


# Define custom color palette for Train, Test, and Original datasets
custom_palette = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

# Function to create Wind Rose plot in a subplot
def create_wind_rose(ax, data, dataset_name, color):
    # Convert wind direction to radians
    wind_direction_radians = np.radians(data['winddirection'].dropna())

    # Create histogram bins (every 10°)
    bins = np.linspace(0, 2*np.pi, 37)  # 36 bins (every 10°)
    counts, bin_edges = np.histogram(wind_direction_radians, bins=bins)

    # Plot on the polar axis with improved style
    bars = ax.bar(bin_edges[:-1], counts, width=np.radians(10), color=color, edgecolor='black', alpha=0.8)

    # Formatting for professional appearance
    ax.set_theta_zero_location("N")  # North is at 0°
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))  # Tick labels every 45°
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10, fontweight='bold')

    # Add grid and labels for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_yticklabels([])  # Remove radial labels to avoid clutter
    ax.set_title(f"Wind Direction ({dataset_name})", fontsize=12, fontweight='bold', pad=10)

# Create a single row with three wind rose plots
fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': 'polar'})

# Generate wind rose plots for Train, Test, and Original datasets
create_wind_rose(axes[0], df_train_rainfall_1, "rainfall_1", custom_palette[0])  # Blue
create_wind_rose(axes[1], df_train_rainfall_0, "rainfall_0", custom_palette[1])    # Red

# Adjust layout for better spacing
plt.tight_layout()
plt.show()










