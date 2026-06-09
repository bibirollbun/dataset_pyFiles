import sys
print(sys.version)


!pip install ray==2.10.0 autogluon.tabular  > /dev/null 2>&1
!pip install optuna-integration[sklearn] > /dev/null 2>&1
!pip install langchain-core > /dev/null 2>&1
!pip install langchain-openai  > /dev/null 2>&1
!pip install sweetviz > /dev/null 2>&1
!pip install numba==0.58.1 visions==0.7.5 pandas==1.5.3 ydata-profiling==4.7.0 > /dev/null 2>&1


# General Purpose Libraries
import logging
import tempfile
import json
import re

# Data analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

# Auto EDA
import sweetviz as sv
from ydata_profiling import ProfileReport

# LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from IPython.display import Markdown, display, IFrame

# Feature engineering
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from featuretools import dfs, EntitySet

# Auto ML
from autogluon.tabular import TabularPredictor

# supress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import pkg_resources
import importlib

# List of packages from your imports
packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "sweetviz",
    "ydata-profiling",  # for ProfileReport
    "langchain-core",
    "langchain-openai",
    "ipython",
    "scikit-learn",     # for sklearn imports
    "featuretools",
    "autogluon.tabular"
]

# Create requirements.txt content
requirements = []
for package in packages:
    try:
        # Handle special cases
        if package == "autogluon.tabular":
            package = "autogluon"
        
        # Get the package version
        version = pkg_resources.get_distribution(package).version
        requirements.append(f"{package}=={version}")
    except pkg_resources.DistributionNotFound:
        print(f"Warning: Package {package} not found")
        requirements.append(package)

# Write requirements to file
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements))


TIME_LIMIT = 3600 * 0.1


MODEL = 'o3-mini'


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
OPENAI_API_KEY = user_secrets.get_secret("openai_key")


# Define the LLM model using LangChain
model = ChatOpenAI(
    model=MODEL,
    api_key=OPENAI_API_KEY
)


# Define the prompt template for LangChain
template_eda = """Provide an analysis of the following EDA summary:
{context}

Key insights and observations:
"""


# Function to classify columns into continuous and categorical
def classify_columns(df):
    continuous_cols = []
    categorical_cols = []
    for column in df.columns:
        if df[column].dtypes == 'object':
            categorical_cols.append(column)
        else:
            unique_values = df[column].nunique()
            if unique_values < 15:
                categorical_cols.append(column)
            else:
                continuous_cols.append(column)
    return continuous_cols, categorical_cols

# Function to perform basic visualizations for continuous and categorical features
def eda_visualizations(df, target=None):
    continuous_cols, categorical_cols = classify_columns(df)
    
    # Plotting continuous columns
    for col in continuous_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
    
    # Plotting categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x=col, hue=target)
        plt.title(f'Count plot for {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

# Function to compare train and test datasets
def compare_train_test(train, test):
    continuous_cols, categorical_cols = classify_columns(train)
    
    # Compare continuous columns
    for col in continuous_cols:
        plt.figure(figsize=(10, 4))
        sns.kdeplot(train[col], label='Train', shade=True)
        sns.kdeplot(test[col], label='Test', shade=True)
        plt.title(f'Comparison of {col} Distribution in Train vs Test')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.show()
    
    # Compare categorical columns
    for col in categorical_cols:
        if col in test.columns:  # Ensure the column exists in the test dataset
            plt.figure(figsize=(10, 4))
            train_counts = train[col].value_counts(normalize=True)
            test_counts = test[col].value_counts(normalize=True)
            train_counts.plot(kind='bar', alpha=0.5, label='Train', color='blue')
            test_counts.plot(kind='bar', alpha=0.5, label='Test', color='red')
            plt.title(f'Comparison of {col} Proportions in Train vs Test')
            plt.xlabel(col)
            plt.ylabel('Proportion')
            plt.legend()
            plt.xticks(rotation=45)
            plt.show()
            
# Function to create key statistics for a dataset
def eda_summary(df):
    summary = {}
    
    # General Info
    summary['general'] = {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'num_missing_values': df.isnull().sum().sum(),
        'percent_missing_values': df.isnull().mean().mean() * 100
    }
    
    # Column Data Types
    summary['data_types'] = df.dtypes.to_dict()
    
    # Missing Value Summary (per column)
    summary['missing_values'] = (
        df.isnull()
        .sum()
        .to_frame(name='missing_count')
        .assign(percent_missing=lambda x: (x['missing_count'] / df.shape[0]) * 100)
        .to_dict(orient='index')
    )
    
    # Numerical Summary (Mean, Median, Std, Min, Max)
    describe_df = df.describe()
    numerical_columns = ['mean', '50%', 'std', 'min', 'max']
    available_columns = [col for col in numerical_columns if col in describe_df.columns]
    summary['numerical_summary'] = (
        describe_df[available_columns]
        .rename(columns={'50%': 'median'})
        .to_dict(orient='index')
    )
    
    # Unique Counts for Categorical Columns
    summary['categorical_summary'] = (
        df.select_dtypes(include=['object', 'category'])
        .nunique()
        .to_frame(name='unique_counts')
        .to_dict(orient='index')
    )
    
    # Skewness and Kurtosis
    summary['skewness_kurtosis'] = {
        column: {
            'skewness': df[column].skew(),
            'kurtosis': df[column].kurt()
        } for column in df.select_dtypes(include=[np.number]).columns
    }
    
    # Correlations
    try:
        summary['correlations'] = df.corr(numeric_only=True).to_dict()
    except ValueError:
        summary['correlations'] = "Unable to calculate correlations due to data type issues."
    
    # Outlier Count based on IQR
    outlier_summary = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
        outlier_summary[column] = {
            'outlier_count': outliers.shape[0],
            'percent_outliers': (outliers.shape[0] / df.shape[0]) * 100
        }
    summary['outlier_summary'] = outlier_summary

    return summary


# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_data.head()


test_data.head()


sample_submission.head()


train_data.shape


test_data.shape


train_data.describe()


test_data.describe()


train_columns = set(train_data.columns)
test_columns = set(test_data.columns)

target_vars = list(train_columns - test_columns)

if len(target_vars) == 1:
    target = target_vars[0]
    print(f"Identified target variable: {target}")
else:
    raise ValueError(f"Expected a single target variable, but found: {target_variable}")


# Generate the report with Sweetviz
target_variable = target_vars[0]
report = sv.compare([train_data, "Train"], [test_data, "Test"], target_feat=target_variable)
report_path = 'Comparative_EDA_Report.html'
report.show_html(filepath=report_path, open_browser=False)

# Display the report inline in Kaggle
IFrame(src=report_path, width=1000, height=600)


# Generate Pandas Profiling report for the training data
profile_train = ProfileReport(train_data, title="Train Data Profile Report", explorative=True)
profile_train_path = 'Train_Data_Profile_Report.html'
profile_train.to_file(profile_train_path)

# Display the training report inline in Kaggle
display(IFrame(src=profile_train_path, width=1000, height=600))


# Generate Pandas Profiling report for the test data
profile_test = ProfileReport(test_data, title="Test Data Profile Report", explorative=True)
profile_test_path = 'Test_Data_Profile_Report.html'
profile_test.to_file(profile_test_path)

# Display the test report inline in Kaggle
display(IFrame(src=profile_test_path, width=1000, height=600))


summary = eda_summary(train_data)
summary_json = json.dumps(summary, indent=4, default=str)


# Define the prompt template for LangChain
template = """Provide an analysis of the following EDA summary, The aim of this dataset and EDA is to understand how several variables influence depression.
Ultimately the aim is to build a classification model to predict depression:
{context}

Key insights and observations:
"""

prompt = ChatPromptTemplate.from_template(template)

# Create a chain to pass the summary to the model
chain = prompt | model | StrOutputParser()

# Invoke the chain to analyze the EDA summary
result = chain.invoke(summary_json)

# Print the result
display(Markdown(result))


# Define the prompt template for LangChain
template_features = """Provide an analysis of the following EDA summary and offer advice on feature engineering to improve predictions of Price:
{context}

Feature Engineering Recommendations for tree-based models:
"""

prompt = ChatPromptTemplate.from_template(template_features)
chain = prompt | model | StrOutputParser()
result = chain.invoke(summary_json)
display(Markdown(result))


# warning when adding code template to the prompt you have to be careful about the formating. If the code is place as is in the template some might be
# interpreted as variables by the langchain prompt parser

code_template = """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming train_data is your DataFrame and 'id' is one of its columns.
cols_to_plot = [col for col in train_data.columns if col != 'id']

# Set a style for consistency
sns.set(style="whitegrid")

for col in cols_to_plot:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(train_data[col]):
        # Left subplot: Histogram (deep blue)
        axes[0].hist(train_data[col].dropna(), bins=30, color='darkblue', edgecolor='black')
        axes[0].set_title(f'Histogram of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')

        # Right subplot: Boxplot (orange)
        sns.boxplot(x=train_data[col].dropna(), ax=axes[1],
                    color='orange', fliersize=3,
                    boxprops={{'facecolor': 'orange'}})
        axes[1].set_title(f'Boxplot of {col}')
        axes[1].set_xlabel(col)
    else:
        # For categorical variables:
        # Left subplot: Count plot (bar plot)
        sns.countplot(x=train_data[col], ax=axes[0], color='darkblue')
        axes[0].set_title(f'Count Plot of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Count')

        # Right subplot: Pie chart
        value_counts = train_data[col].value_counts()
        axes[1].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                    startangle=90, colors=sns.color_palette('pastel'))
        axes[1].set_title(f'Pie Chart of {col}')

    plt.tight_layout()
    plt.show()
"""

template_features = """The aim of this EDA is to understand the impact of several variables on Price. See below the EDA summary:
{context}

Please create Python code to perform the univariate analyse using seaborn and matplotlib. The dataset is stored in memory in train_data.
- Ignore the id column. 
- For each variable plot histogram and boxplot side by side. I would like the boxplot to be orange and the histogram to be deep blue. 
Put your python code inside a ```python``` block.

example code template
{code_template}
"""

prompt = ChatPromptTemplate.from_template(template_features)
chain = prompt | model | StrOutputParser()
result = chain.invoke({"context": summary_json, "code_template": code_template})

display(Markdown(result))

match = re.search(r"```python\s*(.*?)\s*```", result, re.DOTALL)
if match:
    python_code = match.group(1)
    # Execute the extracted Python code
    exec(python_code)
else:
    print("No Python code block found.")


import pandas as pd
import numpy as np

# Assuming train_data is already loaded as a DataFrame

# --- Preprocessing: Drop the 'id' column if it exists ---
data = train_data.drop(columns=['id'], errors='ignore')

# --- Dictionary to store univariate tables for each variable ---
# For continuous variables, two entries will be stored:
# one for overall stats and another for outlier details.
univariate_tables = {}

# Loop over each variable in the dataset
for col in data.columns:
    # --- Continuous Variables ---
    if pd.api.types.is_numeric_dtype(data[col]):
        series = data[col]
        desc = series.describe()
        missing_count = series.isna().sum()

        # Compute quartiles and IQR for boxplot-like info
        q1 = series.quantile(0.25)
        median = series.quantile(0.50)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        # Calculate lower and upper bounds using the 1.5*IQR rule
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers based on these bounds
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = outliers.count()

        # Build the continuous summary table (mimicking boxplot information)
        cont_table = pd.DataFrame({
            'Statistic': ['count', 'missing', 'mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max',
                          'lower_bound', 'upper_bound', 'outlier_count'],
            col: [desc['count'], missing_count, desc['mean'], desc['std'], desc['min'],
                  q1, median, q3, desc['max'], lower_bound, upper_bound, outlier_count]
        })
        univariate_tables[col] = cont_table

        # Build an outlier details table for this variable
        if outlier_count > 0:
            outlier_summary = outliers.describe().to_frame(name='Value').reset_index()\
                                  .rename(columns={'index': 'Statistic'})
        else:
            outlier_summary = pd.DataFrame({'Message': ['No outliers detected']})
        # Store the outlier table with a modified key
        univariate_tables[col + '_outliers'] = outlier_summary

    # --- Categorical Variables ---
    else:
        # Calculate frequency counts (including NaNs)
        vc = data[col].value_counts(dropna=False)
        if vc.shape[0] > 20:
            # Keep the top 20 and aggregate the remaining categories as 'Others'
            top_20 = vc.iloc[:20].reset_index()
            top_20.columns = ['Category', 'Frequency']
            others_count = vc.iloc[20:].sum()
            others_row = pd.DataFrame([{'Category': 'Others', 'Frequency': others_count}])
            cat_table = pd.concat([top_20, others_row], ignore_index=True)
        else:
            cat_table = vc.reset_index()
            cat_table.columns = ['Category', 'Frequency']
        univariate_tables[col] = cat_table

# --- Create an Overall Missing Values Table ---
missing_values_table = data.isna().sum().reset_index()
missing_values_table.columns = ['Variable', 'Missing Count']

# --- Example: Display the univariate table for each variable ---
for var, table in univariate_tables.items():
    print(f"--- {var} ---")
    print(table, "\n")

print("=== Overall Missing Values Summary ===")
print(missing_values_table)


# warning when adding code template to the prompt you have to be careful about the formating. If the code is place as is in the template some might be
# interpreted as variables by the langchain prompt parser

code_template = """
"""

template_features = """The aim of this EDA is to understand the impact of several variables on Price. See below the EDA summary:
{context}

Please create Python code to perform the univariate analyse using tables. The dataset is stored in memory in train_data.
- Ignore the id column. 
- For each variable create tables those will be stored in a dataframe
Put your python code inside a ```python``` block.

example code template
{code_template}
"""

prompt = ChatPromptTemplate.from_template(template_features)
chain = prompt | model | StrOutputParser()
result = chain.invoke({"context": summary_json, "code_template": code_template})

display(Markdown(result))

match = re.search(r"```python\s*(.*?)\s*```", result, re.DOTALL)
if match:
    python_code = match.group(1)
    # Execute the extracted Python code
    exec(python_code)
else:
    print("No Python code block found.")


# warning when adding code template to the prompt you have to be careful about the formating. If the code is place as is in the template some might be
# interpreted as variables by the langchain prompt parser

code_template = """

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming the dataset is already loaded in train_data
# Drop the id column
data = train_data.drop(columns=['id'])

# -------------------------------
# 1. Correlation Analysis for Numerical Features
# -------------------------------
# Select numerical columns (Price, Compartments, Weight Capacity (kg))
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = data[numeric_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# -------------------------------
# 2. Scatter Plots for Continuous Features vs. Price
# -------------------------------
# Scatter plot: Price vs Weight Capacity (kg)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Weight Capacity (kg)', y='Price', data=data, alpha=0.5)
plt.title("Price vs. Weight Capacity (kg)")
plt.xlabel("Weight Capacity (kg)")
plt.ylabel("Price")
plt.show()

# Scatter plot: Price vs Compartments
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Compartments', y='Price', data=data, alpha=0.5)
plt.title("Price vs. Compartments")
plt.xlabel("Compartments")
plt.ylabel("Price")
plt.show()

# -------------------------------
# 3. Boxplots for Categorical Features vs. Price
# -------------------------------
# List of relevant categorical features
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

# Iterate over each categorical variable and create boxplots
for col in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=col, y="Price", data=data)
    plt.title(f"Price Distribution by {col}")
    plt.xlabel(col)
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 4. Pairplot to Get an Overall View of Relationships (including Price)
# -------------------------------
# For a multivariate perspective, select a subset of features
# (If plotting all 11 columns becomes cluttered, consider reducing to select features)
subset_features = ['Price', 'Compartments', 'Weight Capacity (kg)'] + categorical_features[:2]  # using only two of the categorical for clarity
sns.pairplot(data[subset_features], hue="Brand", diag_kind="kde")
plt.suptitle("Pairplot for Selected Features", y=1.02)
plt.show()
"""



template_features = """The aim of this EDA is to understand the impact of several variables on Price. See below the EDA summary:
{context}

Please create Python code to perform the multivariate analyse using seaborn and matplotlib. The dataset is stored in memory in train_data.
- The target variable is Price
- Ignore the id column. 
Put your python code inside a ```python``` block.

example code template
{code_template}
"""

prompt = ChatPromptTemplate.from_template(template_features)
chain = prompt | model | StrOutputParser()
result = chain.invoke({"context": summary_json, "code_template": code_template})

display(Markdown(result))

match = re.search(r"```python\s*(.*?)\s*```", result, re.DOTALL)
if match:
    python_code = match.group(1)
    # Execute the extracted Python code
    exec(python_code)
else:
    print("No Python code block found.")


# Generate EDA summary and format it as a JSON string
summary = eda_summary(train_data)
summary_json = json.dumps(summary, indent=4, default=str)  # summary_json is now a single string

# Prepare the first few rows of train and test data as strings
train_data_head = train_data.head().to_string(index=False)
test_data_head = test_data.head().to_string(index=False)

# Define the prompt template with placeholders for dynamic values
template_features = """The aim of this exploratory data analysis (EDA) is to understand the impact of various variables on Price. Below is the EDA summary:
{context}

Please generate Python code for feature engineering to enhance the model’s performance based on the provided EDA summary.
Place your code into a ```python``` block.
Before the Python code create a few bullet points in the mardown format to explain your appraoch. Keep it short. (do not us any # symbols just bullet points -)
When handling missing values, take into consideration how much of the data is missing for each column and adapt the code based on this.
The datasets are preloaded in memory as `train_data` and `test_data`.
The feature engineering needs to be applied to both datasets in a way that avoids data leakage.
Be mindful of the fact that the data distribution in the train anb test data could be different and some categorical values present in the test data
could be abscent in the train data. The code will need to that into consideration.

The final transformed datasets should be called: train_data_processed and test_data_processed.

First rows of the train data:
{train_data_head}

First rows of the test data:
{test_data_head}
"""

# Create the ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(template_features)

# Define the LLM model and output parser
output_parser = StrOutputParser()

# Chain components together
chain = prompt | model | output_parser

# Prepare the input as a dictionary with `context`, `train_data_head`, and `test_data_head`
input_data = {
    "context": summary_json,
    "train_data_head": train_data_head,
    "test_data_head": test_data_head
}

# Format the prompt with the input data for troubleshooting
formatted_prompt = prompt.format(
    context=summary_json,
    train_data_head=train_data_head,
    test_data_head=test_data_head
)

# Print the formatted prompt for troubleshooting
#print("Formatted Prompt for Troubleshooting:\n")
#print(formatted_prompt)

# Now proceed with invoking the chain
result = chain.invoke(input_data)

# Display the result as Markdown
display(Markdown(result))


MAX_ITERATIONS = 2
TARGET_SCORE = 45


import traceback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


metric = "RMSE"
train_data_path = "/kaggle/input/playground-series-s5e2/train.csv"
test_data_path = "/kaggle/input/playground-series-s5e2/test.csv"
submission_example_path = "/kaggle/input/playground-series-s5e2/sample_submission.csv"
submission_path = "/kaggle/working/submission.csv"
target_variable = 'Price'

train_data_summary_json = summary_json

# --------------------------------------------------------------------
# 1. Prepare your "initial" prompt
# --------------------------------------------------------------------
system_instructions = f"""
You are a coding assistant specialized in tabular data analysis. Here you must predict the Price. 
Output valid Python code that runs end to end.
"""

user_instructions = f"""
You are given the following dataset information:
- Train data path: {train_data_path}
- Test data path: {test_data_path}
- Submission example path: {submission_example_path}
- Train data summary: {train_data_summary_json}
- Target variable: {target_variable}
- Path to the final submission file: {submission_path}

**Task**:
2. Train a model to predict {target_variable}.
3. Generate a valid Kaggle submission at {submission_path}.
4. Compute the '{metric}' on a validation split and store it in 'val_rmse'.
5. Return *only* valid Python code, with no triple backticks.

Begin now.
"""

initial_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_instructions),
    HumanMessage(content=user_instructions),
])

# --------------------------------------------------------------------
# 2. Prepare a "repair" prompt template
# --------------------------------------------------------------------
repair_prompt_template = """
The previous code caused an error or had unsatisfactory results. Below is the code that was generated:

--- CODE START ---
{previous_code}
--- CODE END ---

Here is the traceback or error message:

--- ERROR START ---
{error_trace}
--- ERROR END ---

**Task**:
- Train a model to predict Price.
- Generate a valid Kaggle submission at '/kaggle/working/submission.csv'.
- Compute the 'RMSE' on a validation split and store it in 'val_rmse'.
- Return *only* valid Python code, with no triple backticks.

Begin now.
"""

repair_chain_prompt = ChatPromptTemplate.from_template(repair_prompt_template)

# --------------------------------------------------------------------
# 3. Create your LLM & output parser
# --------------------------------------------------------------------
model_params = {
    "model": MODEL,
    "openai_api_key": OPENAI_API_KEY,
}
    
llm = ChatOpenAI(**model_params)
parser = StrOutputParser()

# --------------------------------------------------------------------
# 4. Helper Function to remove triple backticks
# --------------------------------------------------------------------
def remove_markdown_code_fences(code_str: str) -> str:
    """
    Remove triple-backtick fences from code.
    Also removes lines that contain them.
    """
    lines = code_str.splitlines()
    cleaned = []
    for line in lines:
        if "```" not in line:
            cleaned.append(line)
    return "\n".join(cleaned).strip()

# --------------------------------------------------------------------
# 5. Iterative generation logic
# --------------------------------------------------------------------
iteration = 0
success = False
current_code = None

while iteration < MAX_ITERATIONS and not success:
    iteration += 1
    print(f"\n--- Attempt #{iteration} ---")

    if iteration == 1:
        # Use the initial chain
        chain = initial_prompt_template | llm | parser
        result_code = chain.invoke({
            "train_path": train_data_path,
            "test_path": test_data_path,
            "submission_example_path": submission_example_path,
            "train_summary": train_data_summary_json,
            "target_variable": target_variable,
            "submission_path": submission_path,
            "metric": metric
        })
    else:
        # Use the repair chain with previous_code & error_trace
        # That chain is basically the repair_prompt_template + system_instructions if needed
        repair_chain = repair_chain_prompt | llm | parser
        result_code = repair_chain.invoke({
            "previous_code": current_code,
            "error_trace": error_message,
            "metric": metric
        })
    # Clean out triple backticks
    cleaned_code = remove_markdown_code_fences(result_code)
    current_code = cleaned_code  # store for next iteration if needed

    print("--- Generated/Corrected Code Start ---")
    print(cleaned_code)
    print("--- Generated/Corrected Code End ---\n")

    # Attempt to exec the code
    local_namespace = {}
    try:
        exec(cleaned_code, {}, local_namespace)

        # Check if there's a val_mape and if it is numeric
        if "val_rmse" in local_namespace:
            val_rmse = local_namespace["val_rmse"]
            print(f"val_rmse from script: {val_rmse}")
            if val_rmse < TARGET_SCORE:
                success = True
            else:
                error_message = f"RMSE {val_rmse} is above threshold"
                print(error_message)
                continue
        success = True
    except Exception as e:
        error_trace = traceback.format_exc()
        print("Error encountered while running the generated code:")
        print(error_trace)
        error_message = error_trace  # store for next iteration
        # Not success, continue the loop

if not success:
    print("\nMax iterations reached. Still failing. Exiting.\n")
else:
    print("\nProcess completed successfully!")


import datetime
timesampt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
!cp /kaggle/working/submission.csv /kaggle/working/m02_o3_mini_$timesampt.csv

