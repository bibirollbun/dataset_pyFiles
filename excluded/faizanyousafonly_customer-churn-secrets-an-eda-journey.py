# import libraries:

# 1. to handel the data:
import numpy as np
import pandas as pd

# 2. to visualize the data:
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# 3. to preprocess the data:
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 4. to build the model:
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

# 5. for classification task:
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

# 6. Metrics:

from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, f1_score , classification_report, mean_absolute_error, mean_absolute_percentage_error

# 7. to ignore the warnings:
import warnings
warnings.filterwarnings("ignore")

print("Libraries have been loaded successfully")

# # 8. Display all rows and columns: (uncomment if you want the whole output in the cells, I don't prefer it while uploading my notebook on Kaggle or Github)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


# Setting a style for the plots
sns.set_theme(style="whitegrid")


# ğŸš€ Step 1: Loading the Data
print("Loading the dataset... ğŸ•µï¸�â€�â™‚ï¸�")
df_train = pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s4e1/sample_submission.csv")
print("Dataset loaded successfully!")




print("Training dataset:")
print("--------------------------------------------------------")
df_train.head()



print("Testing dataset:")
print("--------------------------------------------------------")
df_test.head()


print("Submission dataset:")
print("--------------------------------------------------------")
submission.head()


# first of all we will make a copy of our df_train to keep the original data

df = df_train.copy()


# Displaying the first few rows of the dataset
print("\nLet's take a peek at the dataset: ğŸ‘€")
df.head()


# Basic information about the dataset
print("\nğŸ“� Quick summary of our dataset:")
print(df.info())


print("\nUnderstanding the dataset structure and dimensions: ğŸ§±")

print()

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")




print("\nğŸ§® Let's crunch some numbers!")
df.describe().T


# find the min,max and mean values of all the numerical columns

print("\nğŸ“Š Let's find the min, max, and mean values of all the numerical columns:")

for col in df.select_dtypes(include=np.number).columns:
    print(f"\n{col}:")
    print(f"Minimum: {df[col].min()}")
    print(f"Maximum: {df[col].max()}")
    print(f"Mean: {df[col].mean()}")
    print(f"Standard Deviation: {df[col].std()}")
    print(f"25th Percentile: {df[col].quantile(0.25)}")
    print(f"Median: {df[col].median()}")
    


df['Exited'].value_counts()




# # make a pie chart of the 'IsActiveMember' column

# print("\nğŸ¥§ Let's visualize the distribution of the 'IsActiveMember' column:")

# fig = px.pie(df, names='IsActiveMember', title='Distribution of Active Members')
# fig.show()



print("\nğŸ”� Are there any missing pieces in our puzzle?")
print(df.isnull().sum())



print("\nğŸŒˆ Time to explore our categorical variables!")


# Plotly for categorical columns except the 'Surname' column

categorical_cols = df[['Geography', 'Gender']]

# we have excluded the "Surname" column as it contains unique values for each customer


for col in categorical_cols:
    print(f"\nğŸ“Š Distribution of {col}:")
    print(df[col].value_counts(normalize=True))
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=df[col].value_counts().index, 
                y=df[col].value_counts().values,
                marker_color="#FF6F61"  # Red for categorical data (e.g., churned customers)
            )
        ],
        layout_title_text=f'Distribution of {col}'
    )
    fig.update_layout(
        xaxis_title=col,
        yaxis_title='Count',
        title_x=0.5,
        title_font=dict(size=18, color="#1F2937"),
        paper_bgcolor="#F9FAFB",
        plot_bgcolor="#FFFFFF"
    )
    fig.show()




# Plotly for numerical columns
numerical_cols = df[['CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary']]

for col in numerical_cols:
    fig = px.histogram(
        df, 
        x=col, 
        nbins=30, 
        marginal="box", 
        title=f'Distribution of {col}',
        color_discrete_sequence=["#1565C0"]  # Blue theme for numerical data
    )
    fig.update_layout(
        xaxis_title=col,
        yaxis_title='Count',
        title_x=0.5,
        title_font=dict(size=18, color="#1F2937"),  # Dark gray for title text
        paper_bgcolor="#F9FAFB",  # Light gray background
        plot_bgcolor="#FFFFFF"  # White plot area
    )
    fig.show()


# Let's see how the "Boolean Columns" are acting in our dataset: âš›

boolean_cols = ['HasCrCard', 'IsActiveMember', 'Gender']

for col in boolean_cols:
    fig = px.pie(
        df, 
        names=col, 
        title=f'Proportion of {col}',
        color=col,
        color_discrete_map={0: "#4CAF50", 1: "#FF6F61"}  # Green for No, Red for Yes
    )
    fig.update_layout(
        title_x=0.5,
        paper_bgcolor="#F9FAFB"
    )
    fig.show()


# distribution of the target column 'Exited' using a 

print("\nğŸ�¯ Let's focus on our target: Churn!")

# Customized Pie Chart for 'Exited'
fig = px.pie(
    df, 
    names='Exited',  # Column for labels (0 = No, 1 = Yes)
    title='Distribution of Churn', 
    color='Exited',  # Color by 'Exited' values
    color_discrete_map={0: "#4CAF50", 1: "#FF6F61"}  # Green for No, Red for Yes
)

# Layout Customization
fig.update_layout(
    title_x=0.5,  # Center the title
    paper_bgcolor="#F9FAFB",  # Light background
    margin=dict(t=50, l=50, r=50, b=50)  # Add margins for spacing
)

fig.show()



import plotly.express as px

# Sunburst Plot: Geography -> Gender -> Exited
fig = px.sunburst(
    df, 
    path=['Geography', 'Gender', 'Exited'],  # Hierarchy of levels
    values='CreditScore',  # Aggregate by CreditScore or use another metric (e.g., Count)
    color='Exited',  # Color by Exited (0 = No, 1 = Yes)
    color_continuous_scale=[[0, "#4CAF50"], [1, "#FF6F61"]],  # Green for No, Red for Yes
    title='Customer Distribution by Geography, Gender, and Exit Status',
)

# Layout customization
fig.update_layout(
    title_x=0.5,
    paper_bgcolor="#F9FAFB",  # Background color
    margin=dict(t=50, l=50, r=50, b=50)  # Adjust margins for better visibility
)

fig.show()



# Geography vs. Churn
geography_churn = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
print("ğŸŒ� Geography vs. Churn")
print(geography_churn)

# Gender vs. Churn
gender_churn = df.groupby('Gender')['Exited'].mean().sort_values(ascending=False)
print("\nğŸ‘¥ Gender vs. Churn")
print(gender_churn)

# Group by age bins and calculate churn rate
age_bins = [0, 12, 19, 35, 50, 100]
age_labels = ['Child', 'Teen', 'Young Adult', 'Middle-Aged Adult', 'Senior']
df['AgeCategory'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
age_churn = df.groupby('AgeCategory')['Exited'].mean().sort_values(ascending=False)
print("\nğŸ�‚ Age vs. Churn")
print(age_churn)

# Tenure vs. Churn
tenure_churn = df.groupby('Tenure')['Exited'].mean().sort_values(ascending=False)
print("\nğŸ“… Tenure vs. Churn")
print(tenure_churn)

# Balance vs. Churn
balance_churn = df.groupby(pd.cut(df['Balance'], bins=[0, 50000, 100000, 150000, 200000]))['Exited'].mean()
print("\nğŸ’° Balance vs. Churn")
print(balance_churn)

# Number of Products vs. Churn
products_churn = df.groupby('NumOfProducts')['Exited'].mean().sort_values(ascending=False)
print("\nğŸ›’ Number of Products vs. Churn")
print(products_churn)

# IsActiveMember vs. Churn
active_member_churn = df.groupby('IsActiveMember')['Exited'].mean().sort_values(ascending=False)
print("\nğŸŸ¢ IsActiveMember vs. Churn")
print(active_member_churn)




print("\nğŸ”— Time to find the connections in our data!")

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Custom colormap for your color theme
custom_cmap = sns.diverging_palette(133, 10, as_cmap=True)  # Green to red with white in the middle

plt.figure(figsize=(12, 10))
sns.heatmap(
    numeric_df.corr(), 
    annot=True, 
    cmap=custom_cmap,  # Use custom colormap
    linewidths=0.5, 
    center=0,  # Center the colormap at 0 for a diverging effect
    fmt=".2f"  # Format the correlation values
)
plt.title('Correlation Heatmap', fontsize=16, pad=15)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()




print("\nğŸ�­ How do our features relate to churn?")

for col in df.columns:
    if col != 'Exited' and df[col].dtype != 'object':
        # Create boxplot using Plotly
        fig = go.Figure()

        # Add the box plot for 'Exited' vs the feature column
        fig.add_trace(go.Box(
            y=df[df['Exited'] == 0][col], 
            name=f'Active ({col})', 
            marker_color='#66BB6A',  # Green for positive outcomes (Exited = 0)
            boxmean='sd'
        ))
        fig.add_trace(go.Box(
            y=df[df['Exited'] == 1][col], 
            name=f'Churned ({col})', 
            marker_color='#EF5350',  # Red for churn (Exited = 1)
            boxmean='sd'
        ))

        # Update layout
        fig.update_layout(
            title=f'{col} vs Churn',
            title_x=0.5,
            plot_bgcolor='#F5F5F5',  # Light Grey background
            paper_bgcolor='#F5F5F5',  # Light Grey background
            font=dict(
                color='#424242'  # Dark Grey for text
            ),
            xaxis=dict(
                title='Churn Status',
                tickvals=[0, 1],
                ticktext=['Active', 'Churned'],
                showgrid=True,
                gridcolor='#E0E0E0'
            ),
            yaxis=dict(
                title=col,
                showgrid=True,
                gridcolor='#E0E0E0'
            ),
            boxmode='group'  # Makes the box plots appear side by side
        )

        # Show the figure
        fig.show()



# Make categories of Age column for both df_train and df_test

# Define age bins and labels
bins = [0, 12, 19, 35, 50, 100]  # Bin edges
labels = ['Child', 'Teen', 'Young Adult', 'Middle-Aged Adult', 'Senior']  # Categories

# Create age categories for main df
df['AgeCategory'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

# Create age categories for df_train
df_train['AgeCategory'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=True)

# Create age categories for df_test
df_test['AgeCategory'] = pd.cut(df_test['Age'], bins=bins, labels=labels, right=True)

# Show the results
print("Main DataFrame:")
print(df[['Age', 'AgeCategory']].head())
print("\nTraining DataFrame:")
print(df_train[['Age', 'AgeCategory']].head())
print("\nTest DataFrame:")
print(df_test[['Age', 'AgeCategory']].head())



# Define credit score bins and labels
bins = [300, 579, 669, 739, 799, 850]  # Bin edges
labels = ['Low', 'Fair', 'Good', 'High', 'Exceptional']  # Categories

# Create categories for main df
df['CreditScoreCategory'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, right=True)

# Create categories for df_train
df_train['CreditScoreCategory'] = pd.cut(df_train['CreditScore'], bins=bins, labels=labels, right=True)

# Create categories for df_test
df_test['CreditScoreCategory'] = pd.cut(df_test['CreditScore'], bins=bins, labels=labels, right=True)

# Display the results
print("Main DataFrame:")
print(df[['CreditScore', 'CreditScoreCategory']].head())
print("\nTraining DataFrame:")
print(df_train[['CreditScore', 'CreditScoreCategory']].head())
print("\nTest DataFrame:")
print(df_test[['CreditScore', 'CreditScoreCategory']].head())


# Define balance bins and labels
bins = [-1, 0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
labels = ['No Balance', '0-100K', '100K-200K', '200K-300K', '300K-400K', '400K-500K', 
          '500K-600K', '600K-700K', '700K-800K', '800K-900K', '900K-1M']

# Create balance categories for main df
df['BalanceCategory'] = pd.cut(df['Balance'], bins=bins, labels=labels, right=True)

# Create balance categories for df_train
df_train['BalanceCategory'] = pd.cut(df_train['Balance'], bins=bins, labels=labels, right=True)

# Create balance categories for df_test
df_test['BalanceCategory'] = pd.cut(df_test['Balance'], bins=bins, labels=labels, right=True)

# Display the results
print("Main DataFrame:")
print(df[['Balance', 'BalanceCategory']].head())
print("\nTraining DataFrame:")
print(df_train[['Balance', 'BalanceCategory']].head())
print("\nTest DataFrame:")
print(df_test[['Balance', 'BalanceCategory']].head())



# Define salary bins and labels
bins = [0, 30000, 60000, 90000, 120000, 150000, 180000, 200000]  
labels = ['Zero Income', 'Low Income', 'Lower Middle Class', 'Middle Class', 
          'Upper Middle Class', 'High Income', 'Very High Income']

# Create categories for main df
df['SalaryCategory'] = pd.cut(df['EstimatedSalary'], bins=bins, labels=labels, right=True)

# Create categories for df_train
df_train['SalaryCategory'] = pd.cut(df_train['EstimatedSalary'], bins=bins, labels=labels, right=True)

# Create categories for df_test
df_test['SalaryCategory'] = pd.cut(df_test['EstimatedSalary'], bins=bins, labels=labels, right=True)

# Display the results
print("Main DataFrame:")
print(df[['EstimatedSalary', 'SalaryCategory']].head())
print("\nTraining DataFrame:")
print(df_train[['EstimatedSalary', 'SalaryCategory']].head())
print("\nTest DataFrame:")
print(df_test[['EstimatedSalary', 'SalaryCategory']].head())


# Get columns of each dataset
df_cols = set(df.columns)
train_cols = set(df_train.columns)
test_cols = set(df_test.columns)

# Print the column names for each dataset
print("Main DataFrame columns:")
print(sorted(df_cols))
print("\nTraining DataFrame columns:")
print(sorted(train_cols))
print("\nTesting DataFrame columns:")
print(sorted(test_cols))

# Check if all datasets have the same columns
print("\nDo all datasets have the same columns?")
print(f"df and df_train match: {df_cols == train_cols}")
print(f"df and df_test match: {df_cols == test_cols}")
print(f"df_train and df_test match: {train_cols == test_cols}")

# If there are differences, show them
if df_cols != train_cols or df_cols != test_cols or train_cols != test_cols:
    print("\nColumns present in one dataset but not others:")
    print(f"In df but not in train: {df_cols - train_cols}")
    print(f"In df but not in test: {df_cols - test_cols}")
    print(f"In train but not in df: {train_cols - df_cols}")
    print(f"In test but not in df: {test_cols - df_cols}")



# Encode the age column using ordinal encoding from sklearn:

# Define the age categories in order
labels = ['Child', 'Teen', 'Young Adult', 'Middle-Aged Adult', 'Senior']

# Initialize the encoder
age_encoder = OrdinalEncoder(categories=[labels], dtype=int)

# Fit and transform the 'AgeCategory' column in the training df
df_train['AgeCategory'] = age_encoder.fit_transform(df_train[['AgeCategory']])
# Transform the 'AgeCategory' column in the testing df using the fitted encoder
df_test['AgeCategory'] = age_encoder.transform(df_test[['AgeCategory']])

# Display the results
print("\nTraining DataFrame:")
print(df_train[['Age', 'AgeCategory']].head())
print("\nTest DataFrame:")
print(df_test[['Age', 'AgeCategory']].head())



# encode the Geography column using One-Hot Encoding on both df_train and df_test:

# Ensure Geography column is of type string
df_train['Geography'] = df_train['Geography'].astype(str)
df_test['Geography'] = df_test['Geography'].astype(str)

# Initialize the encoder
geo_encoder = OneHotEncoder(sparse_output=False, drop='first')

# Print original unique values to check the data
print("Original Geography values in training set:", df_train['Geography'].unique())
print("Original Geography values in test set:", df_test['Geography'].unique())

# First, get unique values from both training and test sets after ensuring they're strings
unique_geographies = set(df_train['Geography'].astype(str).unique()) | set(df_test['Geography'].astype(str).unique())
print("\nUnique geography values after conversion:", unique_geographies)

# Fit and transform the 'Geography' column in the training df
geo_encoded_train = geo_encoder.fit_transform(df_train[['Geography']])

# Get feature names
feature_names = geo_encoder.get_feature_names_out(['Geography'])

# Create DataFrames with encoded values
geo_encoded_train_df = pd.DataFrame(geo_encoded_train, columns=feature_names, index=df_train.index)

# Transform the test data
geo_encoded_test = geo_encoder.transform(df_test[['Geography']])
geo_encoded_test_df = pd.DataFrame(geo_encoded_test, columns=feature_names, index=df_test.index)

# Add encoded columns to original dataframes
for col in feature_names:
	df_train[col] = geo_encoded_train_df[col]
	df_test[col] = geo_encoded_test_df[col]

# Drop original Geography column
df_train.drop('Geography', axis=1, inplace=True)
df_test.drop('Geography', axis=1, inplace=True)

# Display the results
print("\nTraining DataFrame encoded geography columns:")
print(df_train[feature_names].head())
print("\nTest DataFrame encoded geography columns:")
print(df_test[feature_names].head())



#  encode 'SalaryCategory' and 'CreditScoreCategory'in both df_train and df_test :

# Initialize the encoder
ordinal_encoder = OrdinalEncoder()

# Fit and transform the 'SalaryCategory' column in the training df
df_train['SalaryCategory'] = ordinal_encoder.fit_transform(df_train[['SalaryCategory']])
# Transform the 'SalaryCategory' column in the testing df using the fitted encoder
df_test['SalaryCategory'] = ordinal_encoder.transform(df_test[['SalaryCategory']])

# Fit and transform the 'CreditScoreCategory' column in the training df     
df_train['CreditScoreCategory'] = ordinal_encoder.fit_transform(df_train[['CreditScoreCategory']])
# Transform the 'CreditScoreCategory' column in the testing df using the fitted encoder
df_test['CreditScoreCategory'] = ordinal_encoder.transform(df_test[['CreditScoreCategory']])

# Display the results   

print("\nTraining DataFrame:")
print(df_train[['EstimatedSalary', 'SalaryCategory']].head())
print("\nTest DataFrame:")
print(df_test[['EstimatedSalary', 'SalaryCategory']].head())


print("\nTraining DataFrame:")
print(df_train[['CreditScore', 'CreditScoreCategory']].head())
print("\nTest DataFrame:")
print(df_test[['CreditScore', 'CreditScoreCategory']].head())



#  encode 'BalanceCategory' in both df_train and df_test :

# Initialize the encoder
ordinal_encoder = OrdinalEncoder()

# Fit and transform the 'BalanceCategory' column in the training df
df_train['BalanceCategory'] = ordinal_encoder.fit_transform(df_train[['BalanceCategory']])
# Transform the 'BalanceCategory' column in the testing df using the fitted encoder
df_test['BalanceCategory'] = ordinal_encoder.transform(df_test[['BalanceCategory']])
                                                       
# Display the results
print("\nTraining DataFrame:")
print(df_train[['Balance', 'BalanceCategory']].head())
print("\nTest DataFrame:")
print(df_test[['Balance', 'BalanceCategory']].head())




# Initialize the encoder
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

# Fit on df_train and transform both df_train and df_test
df_train_encoded = pd.DataFrame(one_hot_encoder.fit_transform(df_train[['Gender']]), columns=one_hot_encoder.get_feature_names_out(['Gender']))
df_test_encoded = pd.DataFrame(one_hot_encoder.transform(df_test[['Gender']]), columns=one_hot_encoder.get_feature_names_out(['Gender']))

# Drop the original Gender column from df_train and df_test
df_train = df_train.drop(columns=['Gender']).reset_index(drop=True)
df_test = df_test.drop(columns=['Gender']).reset_index(drop=True)

# Concatenate the encoded columns back to the original datasets
df_train = pd.concat([df_train, df_train_encoded], axis=1)
df_test = pd.concat([df_test, df_test_encoded], axis=1)

# Display the first few rows
df_train.head(), df_test.head()



import matplotlib.pyplot as plt
import seaborn as sns

# Define the numerical columns
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']


# Create a boxplot for each numerical column with the updated color theme
plt.figure(figsize=(12, 10), facecolor='#F5F5F5')  # Background color

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[col], color='#42A5F5', flierprops=dict(markerfacecolor='#EF5350', marker='o'))
    plt.title(f'Boxplot of {col}', color='#424242', fontsize=12)
    plt.xlabel(col, color='#424242')
    plt.grid(True, linestyle='--', linewidth=0.5, color='#F5F5F5')  # Light grid lines

plt.tight_layout()
plt.show()

# Define the columns to check for outliers
cols = numerical_cols

# Check for outliers in each column and print results
for col in cols:    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    # Color-coded print statements
    outlier_count = len(outliers)
    outlier_percentage = round((outlier_count / len(df)) * 100, 2)
    
    color = "\033[91m" if outlier_count > 50 else "\033[92m"  # Red if too many outliers, Green otherwise
    reset_color = "\033[0m"
    
    print(f"{color}Outliers in {col}: {outlier_count}{reset_color}")
    print(f"{color}Percentage of outliers in {col}: {outlier_percentage}%{reset_color}\n")



# checking the duplicates in the dataset:

# Check for duplicates in the dataset
duplicates = df.duplicated().sum()
print(f"Number of duplicates: {duplicates}")



# # save the df_tain and df_test to csv files:

# df_train.to_csv("train_preprocessed.csv", index=False)
# df_test.to_csv("test_preprocessed.csv", index=False)

# print("Preprocessed datasets saved successfully!")


print("\nğŸ§  What have we learned? What's next?")

