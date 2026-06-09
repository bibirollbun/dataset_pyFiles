# Import Basis
import pandas as pd 
import optuna
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 
from colorama import Fore, Style, init;
# Import necessary libraries
from IPython.display import display, HTML
from scipy.stats import skew  # Import the skew function
# Import Plotly.go
import plotly.graph_objects as go
# import Subplots
from plotly.subplots import make_subplots
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Model Classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler , StandardScaler , QuantileTransformer
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn.metrics import *
# Paellete
palette = ["#00B1D2FF", "#FDDB27FF"]
color_palette = sns.color_palette(palette)
# Remove Warnings
import warnings 
warnings.filterwarnings("ignore")
# Set the option to display all columns
pd.set_option('display.max_columns', None)


# Load Submission Data 
d_s = pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')
# Load test Data 
te_d = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
#Train Data 
tr_d = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
# Orignal Data 
O_D = pd.read_csv('/kaggle/input/obesity-levels/ObesityDataSet_raw_and_data_sinthetic.csv')


# Dropping Id from  Train 
tr_d.drop(columns=['id'], inplace=True)
te_d.drop(columns=['id'], inplace=True)


# Concat Train and Original Data 
tr_d = pd.concat([tr_d, O_D], ignore_index=True)


# Text Color
def PrintColor(text: str, color=Fore.CYAN, style=Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)

# Text For Main Heading
def print_blue_large(text):
    PrintColor(text, Fore.BLUE + Style.BRIGHT)

# Main Heading
def print_boxed_blue_heading(text):
    length = len(text) + 4
    print(f"\n{Style.BRIGHT}{Fore.BLUE}{'='*length}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.BLUE}| {text} |{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.BLUE}{'='*length}{Style.RESET_ALL}")

# Function to Overview Data
def data_overview(tr_d, te_d):

    # Display head of the training dataset nicely
    print_boxed_blue_heading("The Head Of Train Dataset is:")
    display(HTML(tr_d.head(5).to_html(index=False).replace('<table border="1" class="dataframe">', '<table style="border: 2px solid blue;">')))

    print('\n')

    # Display head of the test dataset nicely
    print_boxed_blue_heading("The Head Of Test Dataset is:")
    display(HTML(te_d.head(5).to_html(index=False).replace('<table border="1" class="dataframe">', '<table style="border: 2px solid blue;">')))

    print('\n')

    # Shapes of Train and Test
    print_boxed_blue_heading("Shape Information:")
    PrintColor(f"The Shape Of Train Data is {tr_d.shape} || No of Rows is : {tr_d.shape[0]} and Columns is {tr_d.shape[1]}", Fore.CYAN)
    print('\n')
    PrintColor(f"The Shape Of Test Data is {te_d.shape}  || No of Rows is : {te_d.shape[0]} and Columns is {te_d.shape[1]}", Fore.CYAN)
    print('\n')

    # Info of Both Datasets
    print_boxed_blue_heading("Dataset Information:")
    PrintColor(f"\nThe Info Of Train Dataset", Fore.CYAN)
    tr_d.info()
    PrintColor(f"\nThe Info Of Test Dataset is", Fore.CYAN)
    te_d.info()
    print('\n')

    # Describe Both
    print_boxed_blue_heading("Numerical Summary:")
    PrintColor(f"\nThe Numerical Summary of Train is", Fore.CYAN)
    display(tr_d.describe().style.set_caption("Train Data Summary").set_table_styles([{'selector': 'caption', 'props': [('color', 'blue')]}]))
    PrintColor(f"\nThe Numerical Summary of Test is", Fore.CYAN)
    display(te_d.describe().style.set_caption("Test Data Summary").set_table_styles([{'selector': 'caption', 'props': [('color', 'blue')]}]))
    print('\n')

    # Null Values in Train and Test
    print_boxed_blue_heading("Null Values:")
    PrintColor("\nNull Values in Train", Fore.CYAN)
    print(tr_d.isnull().sum())
    PrintColor("\nNull Values in Test", Fore.CYAN)
    print(te_d.isnull().sum())
    print('\n')

    # Duplicates Values in Train and Test
    print_boxed_blue_heading("Duplicate Values:")
    PrintColor("\nDuplicates Values in Train", Fore.CYAN)
    print(tr_d.duplicated().sum())
    PrintColor("\nDuplicates Values in Test", Fore.CYAN)
    print(te_d.duplicated().sum())


# Data Overview
data_overview(tr_d,te_d)


# Drop Duplicates 
tr_d.drop_duplicates(inplace=True)


def Feature_E(df):
    # Feature 1: BMI (Body Mass Index)
    df['BMI'] = df['Weight'] / (df['Height'] / 100)**2
    
    # Feature 2: Number of meals per day
    df['Meals_Per_Day'] = df['FCVC'] + df['NCP']
    
    # Feature 3: Total physical activity score
    df['Total_Activity_Score'] = df['FAF'] * df['TUE']
    
    # Feature 5: Age category (e.g., young, adult, elderly)
    df['Age_Category'] = pd.cut(df['Age'], bins=[0, 18, 60, float('inf')], labels=['Young', 'Adult', 'Elderly'])
    
    # Feature 6: Water intake per kg of body weight
    df['Water_Intake_Per_Kg'] = df['CH2O'] / df['Weight']

    return df

# Apply the feature engineering function to your training and testing datasets
tr_d = Feature_E(tr_d)
te_d = Feature_E(te_d)


# # Function to Plot Single Pie and Bar Plot
def single_plot_distribution(column_name, dataframe):
    # Get the value counts of the specified column
    value_counts = dataframe[column_name].value_counts()

    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]}) 

    # Donut pie chart
    pie_colors = palette[0:3]
    ax1.pie(value_counts, autopct='%0.001f%%', startangle=90, pctdistance=0.85, colors=pie_colors, labels=None)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title(f'Distribution of {column_name}', fontsize=16)

    # Bar chart
    bar_colors = palette[0:3]
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax2, palette=bar_colors,) 
    ax2.set_title(f'Count of {column_name}', fontsize=16)
    ax2.set_xlabel(column_name, fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)

    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)

    # Show the plots
    plt.tight_layout()
    plt.show()


# NObeyesdad Ditribution
single_plot_distribution('NObeyesdad',tr_d)


# Gender Ditribution
single_plot_distribution('Gender',tr_d)


# SMOKE Ditribution
single_plot_distribution('SMOKE',tr_d)


# SCC Ditribution
single_plot_distribution('SCC',tr_d)


# CAEC Ditribution
single_plot_distribution('CAEC',tr_d)


# family_history_with_overweight	 Ditribution
single_plot_distribution('family_history_with_overweight',tr_d)


# Age_Category Ditribution
single_plot_distribution('Age_Category',tr_d)


# FAVC Ditribution
single_plot_distribution('FAVC',tr_d)


# MTRANS Ditribution
single_plot_distribution('MTRANS',tr_d)


# Scatter Plot to Show Realationship Bw 2 Cols
def advanced_scatter_plot(x_column, y_column, target_column, dataframe):
    plt.figure(figsize=(15, 6))
    sns.scatterplot(x=x_column, y=y_column, hue=target_column, data=dataframe, palette=palette[0:3])
    plt.title(f'Scatter Plot of {x_column} vs {y_column} Hue by {target_column}', fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.legend(title=target_column)
    plt.grid(True)
    plt.show()


# Scatter Plot | to Show Age vs Weight RealtionShip
advanced_scatter_plot('Age', 'Weight', 'Gender', tr_d)


# Scatter Plot | to Show Age vs Weight RealtionShip
advanced_scatter_plot('Age', 'Weight', 'SMOKE', tr_d)


# Scatter Plot | to Show Age vs Height RealtionShip
advanced_scatter_plot('Age', 'Height', 'Gender', tr_d)


# # Scatter Plot | to Show Weight vs Water_Intake_Per_Kg RealtionShip
advanced_scatter_plot('Weight', 'Water_Intake_Per_Kg', 'Gender', tr_d)


# # Scatter Plot | to Show Age vs BMI RealtionShip
advanced_scatter_plot('Age', 'BMI', 'Gender', tr_d)


# # Scatter Plot | to Show Age vs Meals_Per_Day RealtionShip
advanced_scatter_plot('Age', 'Meals_Per_Day', 'Gender', tr_d)


# # Scatter Plot | to Show Height vs TUE RealtionShip
advanced_scatter_plot('Height', 'TUE', 'Age_Category', tr_d)


# Cols to Plot 
columns_to_plot = ['Gender', 'Age', 'Height', 'Weight',
       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'SCC', 'FAF',
       'CALC', 'MTRANS', 'NObeyesdad',
       'Age_Category', ]

# Data Columns
data_to_plot = tr_d[columns_to_plot]

# Create a dictionary to map colors to unique values of the 'Quality' column
Q_colors = {'Male': palette[0], 'Female': palette[1], 'other': 'gray'}  

# Creating the pairplot with the specified palette for categorical variables
sns.pairplot(data_to_plot, hue='Gender', palette=Q_colors)
plt.show()


# Num _COLS 
NUM_COLS_F = [col for col in tr_d.columns if tr_d[col].dtype == 'float']

# Define the number of rows and columns for subplots
num_rows = 4  # 4 rows
num_cols = 4  # 4 columns

# Create subplots with appropriate titles
fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 17))

# Paellet 
palettes = ["rgb(0, 177, 210)", "rgb(253, 219, 39)"]

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through each numerical column and create a box plot
for i, col in enumerate(NUM_COLS_F[:num_rows * num_cols]):
    sns.boxplot(x=tr_d[col], ax=axes[i], color=palette[i % len(palette)])
    axes[i].set_title(col)

# Hide empty subplots
for i in range(len(NUM_COLS_F), num_rows * num_cols):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# Function to Plot Numerical Distribution 
def plot_numerical_distribution_with_hue(data, num_cols, hue_col='Gender', figsize=(25, 25), dpi=100):
    # Create subplots
    rows = (len(num_cols) + 1) // 2 
    fig, ax = plt.subplots(rows, 2, figsize=figsize, dpi=dpi)
    ax = ax.flatten() 
        # Loop through each column and plot the distribution with hue
    for i, column in enumerate(num_cols):  
        sns.histplot(data=data, x=column, hue=hue_col, ax=ax[i], kde=True, palette=palette)
        ax[i].set_title(f'{column} Distribution', size=14)
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(None)
        
        # Calculate skewness
        skewness = skew(data[column].dropna())
        skew_label = f'Skewness: {skewness:.2f}'
        
        # Add skewness annotation
        ax[i].annotate(skew_label, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='red')
    
    # Remove any extra subplots
    for j in range(len(num_cols), len(ax)):
        fig.delaxes(ax[j])
    
    # Set Tight Layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


# Cols to Plot
NUM_COLS_F = [col for col in tr_d.columns if tr_d[col].dtype == 'float']
# Numerical Distribution of Age Vs Fare
plot_numerical_distribution_with_hue(tr_d,NUM_COLS_F,'Gender')


# Function to Scale Data
def apply_scaling(data, columns, scaler_type):
    # Check the type of scaler and initialize the appropriate scaler object
    if scaler_type == 'S':
        scaler = StandardScaler()  # Initialize StandardScaler
    elif scaler_type == 'M':
        scaler = MinMaxScaler()  # Initialize MinMaxScaler
    elif scaler_type == 'Q':
        scaler = QuantileTransformer(output_distribution='normal')  # Initialize QuantileTransformer
    else:
        raise ValueError("Invalid scaler type. Choose 'S' for StandardScaler, 'M' for MinMaxScaler, or 'Q' for QuantileTransformer.")

    # Create a copy of the input data to avoid modifying the original data
    scaled_data = data.copy()

    # Loop through each column to be scaled
    for col in columns:
        # Apply the scaler to the current column and update the data with the scaled values
        scaled_data[col] = scaler.fit_transform(scaled_data[[col]])

    # Return the scaled data
    return scaled_data

# Specify columns and scaler type
columns_to_scale = [col for col in tr_d.columns if tr_d[col].dtype == 'float']
scaler_type = 'Q' 

# Apply scaling to training data
tr_d = apply_scaling(tr_d, columns_to_scale, scaler_type)
# Apply the same scaling to testing data
te_d = apply_scaling(te_d, columns_to_scale, scaler_type)
PrintColor('Data Scaled Done')


# Select only numeric columns
N_d = tr_d.select_dtypes(include='number')

# Compute the correlation matrix
correlation_matrix = N_d.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(25, 15))
sns.heatmap(correlation_matrix, annot=True, cmap=palette, fmt=".1f", linewidths=0.5)
plt.title('Correlation Plot', fontsize=22)  
plt.tight_layout()  
plt.show()


# Defining the categorical columns to encode
CAT_COL_E = [
 'Gender',
 'family_history_with_overweight',
 'FAVC',
 'CAEC',
 'SMOKE',
 'SCC',
 'CALC',
 'MTRANS',
]
# Function to Encode Data 
def E_D(data, columns, method='L'):
    encoded_data = data.copy()  # Make a copy of the input data
    
    if method == 'L':
        # Initialize LabelEncoder
        L_E = LabelEncoder()
        
        # Encode categorical columns using LabelEncoder
        for col in columns:
            encoded_data[col] = L_E.fit_transform(encoded_data[col])
    
    elif method == 'D':
        # Create dummy variables for categorical columns
        dummy_cols = pd.get_dummies(encoded_data[columns], prefix=columns)
        
        # Concatenate dummy variables with original data
        encoded_data = pd.concat([encoded_data, dummy_cols], axis=1)
        
        # Drop the original categorical columns
        encoded_data = encoded_data.drop(columns, axis=1)
    
    else:
        raise ValueError("Invalid method! Please choose either 'L' or 'D'.")
    
    return encoded_data


# Encoder Train and test
tr_d = E_D(tr_d, CAT_COL_E , 'D')
te_d = E_D(te_d, CAT_COL_E, "D")
PrintColor('Data is Encoded Successfully')


# # # =================================================================================================================
# # #                         X < y 
# # #================================================================================================================== 
X_T = tr_d.drop('NObeyesdad', axis=1)
y_T = tr_d['NObeyesdad']

# # # =================================================================================================================
# # #                         Train < Test Split
# # #================================================================================================================== 
X_TR, X_TE, Y_TR, Y_TE = train_test_split(X_T, y_T, test_size=0.1, random_state=42)

# # # =================================================================================================================
# # #                         Shapes < 
# # #================================================================================================================== 
PrintColor(f"Training set shape - X: {X_TR.shape}, y: {Y_TR.shape}")
PrintColor(f"Testing set shape - X: {X_TE.shape}, y: {Y_TE.shape}")


# # # =================================================================================================================
# # #                         Params < LGB Classifier
# # #================================================================================================================== 
lgb_params = {
 'n_estimators': 899,
 'learning_rate': 0.013003893032117776,
 'max_depth': 18,
 'reg_alpha': 0.9218377389528793,
 'reg_lambda': 0.020694654173173645,
 'num_leaves': 24,
 'subsample': 0.7402011916024158,
 'colsample_bytree': 0.25484261764678784,
 'verbose' : -1
}
# # # =================================================================================================================
# # #                        Train < LGB Classifier
# # #================================================================================================================== 
L_BASE = lgb.LGBMClassifier(**lgb_params)
V_CV = cross_val_score(L_BASE,
                       X_T, 
                       y_T, 
                       scoring='accuracy',
                       cv=15, 
                       n_jobs=-1)
# # # =================================================================================================================
# # #                        ROC AUC < LGB Classifier
# #================================================================================================================== 
print_boxed_blue_heading(f"The AUCCURACY Of LGB Classifier is : {V_CV.mean()}")
# # # =================================================================================================================
# # #                        Submission < LGB Classifier
# #================================================================================================================== 
# Fit Again 
L_BASE.fit(X_T,y_T)
# Pred 
T_P = L_BASE.predict(te_d)

# Submission File 
SUB = pd.DataFrame({'id': d_s['id'], 'NObeyesdad': T_P})
# Save Submission File
SUB.to_csv('submission_O_R_0.9000.csv', index=False)
PrintColor('Submission File Saved ! Hurray')


# # # =================================================================================================================
# # #                        Params < CAT Classifier
# # #================================================================================================================== 
cat_params = {
    'n_estimators': 853,
    'learning_rate': 0.10899577626375372,
    'depth': 7,
    'colsample_bylevel': 0.7340962061535496,
    'random_strength': 6.262882561405091,
    'min_data_in_leaf': 92,
    'verbose': 0,
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
}  

# Specify categorical features
cat_features = ['Age_Category']  # Add the categorical features here

# # # =================================================================================================================
# # #                        Train < CAT Classifier
# # #================================================================================================================== 
C_BASE = CatBoostClassifier(**cat_params, cat_features=cat_features)
C_CV = cross_val_score(C_BASE,
                       X_T, 
                       y_T, 
                       scoring='accuracy',
                       cv=5, 
                       n_jobs=-1)
# # # =================================================================================================================
# # #                        ROC AUC < CAT Classifier
# # #================================================================================================================== 
print_boxed_blue_heading(f"The Accuracy Of CatBoost Classifier is: {C_CV.mean()}")
















