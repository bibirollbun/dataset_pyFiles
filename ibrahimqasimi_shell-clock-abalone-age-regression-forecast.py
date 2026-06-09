# Import MODULES
import pandas as pd 
import optuna
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 
from colorama import Fore, Style, init;
from scipy.stats import skew  # Import the skew function
# Import Plotly.go
import plotly.graph_objects as go
# import Subplots
from plotly.subplots import make_subplots
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Iterative Imputer 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer , SimpleImputer
# Normalization 
from sklearn.preprocessing import QuantileTransformer , PowerTransformer , LabelEncoder,MinMaxScaler
# Model Classifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier, VotingRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import *
# Model Regression 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
# Paellete
palette = ['#FBBACD', '#F99DB9', '#F881A6', '#D68F6F', '#BC8B69']

color_palette = sns.color_palette(palette)
# Remove Warnings
import warnings 
warnings.filterwarnings("ignore")

# Set the option to display all columns
pd.set_option('display.max_columns', None)


# Load Data 
tr_d = pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
te_d = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')
s_d = pd.read_csv('/kaggle/input/playground-series-s4e4/sample_submission.csv')
O_D = pd.read_csv('/kaggle/input/playgrounds4e04originaldata/Original.csv')


# Test Id's
test_id = te_d['id']


# Original column names
original_columns = ['id', 'Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

# New column names
new_columns = ['id', 'Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
               'Whole weight.1', 'Whole weight.2', 'Shell weight', 'Rings']

# Rename columns
O_D.columns = new_columns


# Concating df_org and df_train DataFrames
tr_d = pd.concat([tr_d, O_D], ignore_index=True)
tr_d.reset_index(inplace=True,drop=True)


# Dropping Id from Test and Train 
tr_d.drop(columns=['id'], inplace=True)
te_d.drop(columns=['id'], inplace=True)


from colorama import Fore, Style
import plotly.express as px

def PrintColor(text: str, color=Fore.GREEN, style=Style.NORMAL):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)

def print_yellow_large(text):
    PrintColor(text, Fore.YELLOW + Style.BRIGHT)

def print_separator(symbol='-', length=60):
    separator = symbol * length
    PrintColor(separator, Fore.YELLOW + Style.BRIGHT)

# Main Heading
def print_heading(text):
    print_separator()
    PrintColor('╔══════════════════════════════════════════════════════════╗', Fore.YELLOW + Style.BRIGHT)
    print_yellow_large(f" {text.upper():^42} ")
    PrintColor('╚══════════════════════════════════════════════════════════╝', Fore.YELLOW + Style.BRIGHT)
    print_separator()
def data_overview(tr_d, te_d):
    # Display head of the training dataset nicely
    print_heading("The Head Of Training Dataset")
    print(tr_d.head(5).to_string(index=False))
    print()

    # Display head of the test dataset nicely
    print_heading("The Head Of Test Dataset")
    print(te_d.head(5).to_string(index=False))
    print()

    # Shapes of Train
    print_heading("Shape Information")
    PrintColor(f"The Shape Of Train Data: Rows - {tr_d.shape[0]}, Columns - {tr_d.shape[1]}", Fore.MAGENTA)
    PrintColor(f"The Shape Of Test Data: Rows - {te_d.shape[0]}, Columns - {te_d.shape[1]}", Fore.MAGENTA)
    print()

    # Info of Train Dataset
    print_heading("Dataset Information")
    PrintColor(f"The Info Of Train Dataset\n{tr_d.info()}", Fore.MAGENTA)
    PrintColor(f"The Info Of Test Dataset\n{te_d.info()}", Fore.MAGENTA)
    print()

    # Describe Train
    print_heading("Numerical Summary")
    PrintColor(f"The Numerical Summary of Train\n{tr_d.describe()}", Fore.MAGENTA)
    PrintColor(f"The Numerical Summary of Test\n{te_d.describe()}", Fore.MAGENTA)
    print()

    # Null Values in Train
    print_heading("Null Values")
    PrintColor("Null Values in Train\n" + str(tr_d.isnull().sum()), Fore.MAGENTA)
    PrintColor("Null Values in Test\n" + str(te_d.isnull().sum()), Fore.MAGENTA)
    print()

    # Duplicates Values in Train
    print_heading("Duplicate Values")
    PrintColor(f"Duplicates Values in Train: {tr_d.duplicated().sum()}", Fore.MAGENTA)
    PrintColor(f"Duplicates Values in Test: {te_d.duplicated().sum()}", Fore.MAGENTA)
    print()




# Data Overview
data_overview(tr_d, te_d)


# # Lets Do Some Feature Enginnering 

def Feature_Engineering(df):

    # 1. Shell Volume
    df['Shell Volume'] = df['Length'] * df['Diameter'] * df['Height']

    # 2. Meat Ratio
    df['Meat Ratio'] = df['Whole weight'] / df['Shell weight']

    # 3. Body Mass Index (BMI)
    df['BMI'] = df['Whole weight'] / (df['Length'] ** 2)

    # 4. Shell Surface Area (assuming cylindrical shell)
    df['Shell Surface Area'] = 2 * (df['Length'] * df['Diameter'] + df['Length'] * df['Height'] + df['Diameter'] * df['Height'])

    # 5. Volume to Weight Ratio
    df['Volume to Weight Ratio'] = df['Shell Volume'] / df['Whole weight.1']

    
    # 7. Shell Weight to Length Ratio
    df['Shell Weight to Length Ratio'] = df['Shell weight'] / df['Length']

    # 8. Meat Weight to Length Ratio
    df['Meat Weight to Length Ratio'] = df['Whole weight'] / df['Length']
    
    # Return Data 
    return df

# Train And test 
tr_d = Feature_Engineering(tr_d)
te_d = Feature_Engineering(te_d)


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


# Sex Ditribution
single_plot_distribution('Sex',tr_d)


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


# Scatter Plot | to Show Lenght vs Whole Weight RealtionShip
advanced_scatter_plot('Length', 'Whole weight', 'Sex', tr_d)


# Scatter Plot | to Show Diameter vs Whole Weight RealtionShip
advanced_scatter_plot('Diameter', 'Whole weight', 'Sex', tr_d)


# Scatter Plot | to Show Height vs Whole Weight RealtionShip
advanced_scatter_plot('Height', 'Whole weight', 'Sex', tr_d)


# Scatter Plot | to Show Height vs Shell weight RealtionShip
advanced_scatter_plot('Length', 'Shell weight', 'Sex', tr_d)


# Scatter Plot | to Show Shell Volume vs Shell weight RealtionShip
advanced_scatter_plot('Shell Volume', 'Shell weight', 'Sex', tr_d)


# Scatter Plot | to Show Diameter vs Meat Ratio RealtionShip
advanced_scatter_plot('Diameter', 'Meat Ratio', 'Sex', tr_d)


# Scatter Plot | to Show Diameter vs Shell weight RealtionShip
advanced_scatter_plot('Diameter', 'Shell weight', 'Sex', tr_d)


# Some Plotly Visualization to Show Relations More Deeper
def advanced_scatter_plot(x_column, y_column, hue_column, dataframe):
    fig = px.scatter(dataframe, x=x_column, y=y_column, color=hue_column, 
                     title=f'Scatter Plot: {x_column} vs {y_column} (Hue by {hue_column})',
                     labels={x_column: x_column, y_column: y_column, hue_column: hue_column},
                     color_discrete_map={'F': palette[0], 'I': palette[1], 'M':palette[2]})
    fig.show()


# Scatter Plot | to Show Sex  vs BMI RealtionShip
advanced_scatter_plot('Sex', 'BMI', 'Sex', tr_d)


# Scatter Plot | to Show Length  vs BMI RealtionShip
advanced_scatter_plot('BMI', 'Length', 'Sex', tr_d)


# Scatter Plot | to Show Height  vs BMI RealtionShip
advanced_scatter_plot('BMI', 'Height', 'Sex', tr_d)


# Scatter Plot | to Show Length  vs RING RealtionShip
advanced_scatter_plot('Length', 'Rings', 'Sex', tr_d)


# Scatter Plot | to Show BMI  vs RING RealtionShip
advanced_scatter_plot('BMI', 'Rings', 'Sex', tr_d)


# Cols to Plot 
columns_to_plot = ['Sex','Length', 'Diameter', 'Height', 'Whole weight',
                   'Shell weight',
                   'Shell Volume','BMI',
                   ]

# Data Columns
data_to_plot = tr_d[columns_to_plot]

# Create a dictionary to map colors to unique values of the 'Sex' column
sex_colors = {'M': palette[0] , 'F': palette[1], 'I': palette[2]}  

# Creating the pairplot with the specified palette for categorical variables
sns.pairplot(data_to_plot, hue='Sex', palette=sex_colors)
plt.show()


# Num _COLS 
NUM_COLS_F = [col for col in tr_d.columns if tr_d[col].dtype == 'float']

# BoxPLot To Identify Outliers
fig = go.Figure()

# Define the number of rows and columns for subplots
num_rows = 4  # 4 rows
num_cols = 4  # 4 columns

# Create subplots with appropriate titles
fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=NUM_COLS_F[:num_rows * num_cols])

# Loop through each row of subplots
for i in range(num_rows):
    # Loop through each subplot in the row
    for j in range(num_cols):
        # Calculate the index of the current numerical column
        index = i * num_cols + j
        # Check if the index is within the range of available numerical columns
        if index < len(NUM_COLS_F):
            # Add a box plot for the current numerical column to the subplot
            fig.add_trace(go.Box(x=tr_d[NUM_COLS_F[index]], name=NUM_COLS_F[index], marker_color=palette[index % len(palette)]), row=i + 1, col=j + 1)

# Update layout
fig.update_layout(height=800, width=2000, title_text="Boxplot of Numerical Columns")

# Show the plot
fig.show()


# Function to Plot Numerical Distribution 
def plot_numerical_distribution_with_hue(data, num_cols, hue_col='Sex', figsize=(25, 25), dpi=100):
    # Create subplots
    rows = (len(num_cols) + 1) // 2 
    fig, ax = plt.subplots(rows, 2, figsize=figsize, dpi=dpi)
    ax = ax.flatten() 
    
    # Define the palette
    palette = ['#328ca9', '#0e6ea9', '#2c4ea3', '#193882', '#102446']
    
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
plot_numerical_distribution_with_hue(tr_d,NUM_COLS_F,'Sex')


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
scaler_type = 'M' 

# Apply scaling to training data
tr_d_scaled = apply_scaling(tr_d, columns_to_scale, scaler_type)
# Apply the same scaling to testing data
te_d_scaled = apply_scaling(te_d, columns_to_scale, scaler_type)
PrintColor('Data Scaled Done')


# Select only numeric columns
N_d = tr_d.select_dtypes(include='number')

# Compute the correlation matrix
correlation_matrix = N_d.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(25, 15))
sns.heatmap(correlation_matrix, annot=True, cmap=palette[0:3], fmt=".2f", linewidths=0.5)
plt.title('Correlation Plot', fontsize=22)  
plt.tight_layout()  
plt.show()


# Defining the categorical columns to encode
CAT_COL_E = ['Sex']

# Function to encode columns using One-Hot Encoding
def Encode(data, columns):
    # Perform One-Hot Encoding
    encoded_data = pd.get_dummies(data, columns=columns)
    
    # Return the encoded data
    return encoded_data

# Encoder Train Test 
tr_d = Encode(tr_d, CAT_COL_E)
te_d = Encode(te_d, CAT_COL_E)
PrintColor('Data is Encoded Successfully')


# X and y
X_T = tr_d.drop('Rings', axis=1)
y_T = tr_d['Rings']

# Train Test Split
X_TR, X_TE, Y_TR, Y_TE = train_test_split(X_T, y_T, test_size=0.1, random_state=42)

# Print Shapes
PrintColor(f"Training set shape - X: {X_TR.shape}, y: {Y_TR.shape}")
PrintColor(f"Testing set shape - X: {X_TE.shape}, y: {Y_TE.shape}")


# # # ╔══════════════════════════════════════════════════════════╗
# # #                         Params < LGB Regressor
# # # ╚══════════════════════════════════════════════════════════╝
lgb_params = {
     'n_estimators': 855,
     'learning_rate': 0.03188929865038832,
     'max_depth': 9,
     'reg_alpha': 0.05242740495804349,
     'reg_lambda': 0.3389170890195228,
     'num_leaves': 54,
     'subsample': 0.6251285203728641,
     'colsample_bytree': 0.5205780443860558,
    'verbose' : -1
}
# # # ╔══════════════════════════════════════════════════════════╗
# # #                         Train < LGB Regressor
# # # ╚══════════════════════════════════════════════════════════╝
L_BASE = lgb.LGBMRegressor(**lgb_params)
V_CV = cross_val_score(L_BASE,
                         X_T, 
                         y_T, 
                         scoring='neg_mean_squared_log_error',
                         cv=15, 
                         n_jobs=-1)
# # # ╔══════════════════════════════════════════════════════════╗
# # #                         RMSE < LGB Regressor
# # # ╚══════════════════════════════════════════════════════════╝
print_heading(f"The Average RMSLE Of Voting Regressor is : {-V_CV.mean()}")
# # # ╔══════════════════════════════════════════════════════════╗
# # #                         Submission < LGB Regressor
# # # ╚══════════════════════════════════════════════════════════╝ 
# # # Fit the LGB
L_BASE.fit(X_T, y_T)
# Test Pred
test_pred_L_BASE = L_BASE.predict(te_d)
# Submission DF
submission_df = pd.DataFrame({
    'id': test_id,  
    'Rings': test_pred_L_BASE  
})
# Head
submission_df.head()
# Save 
submission_df.to_csv('submission_L_CV.csv', index=False)
PrintColor('Submission File Saved')


# # # ╔══════════════════════════════════════════════════════════╗
# # #                         Params < Cat Regressor
# # # ╚══════════════════════════════════════════════════════════╝
cat_params = {
     'n_estimators': 853,
     'learning_rate': 0.10899577626375372,
     'depth': 7,
     'subsample': 0.998357427917925,
     'colsample_bylevel': 0.7340962061535496,
     'random_strength': 6.262882561405091,
     'min_data_in_leaf': 92,
    'verbose' : 0
             } 
# # # ╔══════════════════════════════════════════════════════════╗
# # #                         Train < CatRegressor
# # # ╚══════════════════════════════════════════════════════════╝
C_BASE = CatBoostRegressor(**cat_params)
C_CV = cross_val_score(C_BASE,
                       X_T, 
                       y_T, 
                       scoring='neg_mean_squared_log_error',
                       cv=15, 
                       n_jobs=-1)
# # # ╔══════════════════════════════════════════════════════════╗
# # #                         Params < RMSLE Regressor
# # # ╚══════════════════════════════════════════════════════════╝
print_heading(f"The Average RMSLE Of CatBoost Regressor is: {-C_CV.mean()}")
# # # ╔══════════════════════════════════════════════════════════╗
# # #                         Submiison < Cat Regressor
# # # ╚══════════════════════════════════════════════════════════╝
C_BASE.fit(X_T, y_T)
# Test Pred
test_pred_C_BASE = C_BASE.predict(te_d)
# Submission DF
submission_df = pd.DataFrame({
    'id': test_id,  
    'Rings': test_pred_C_BASE  
})
# Head
submission_df.head()
# Save
submission_df.to_csv('submission_C_CV.csv', index=False)
PrintColor('Submission File Saved')

