# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# data modeling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# configure Gemini API key
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# import ADK packages
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.tools import AgentTool, FunctionTool, google_search, ToolContext
from google.genai import types

# import ADK tool packages
from google.adk.sessions import InMemorySessionService
from google.adk.code_executors import BuiltInCodeExecutor

# import ADK session packages
from typing import Any, Dict, List
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import DatabaseSessionService
from google.adk.tools.tool_context import ToolContext

print("âœ… ADK components imported successfully.")


# configure retry options
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)
print("âœ… Retry options configured sucessfully.")


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Returns a DataFrame with irrelevant columns removed
        Parameters:
            df (pd.DataFrame): DataFrame with irrelevant columns
        Returns: 
            df (pd.DataFrame): DataFrame with relevant columns
    """
    colnames = [colname for colname in df.columns]
    for colname in colnames:
        if colname.startswith("Unnamed"):
            df = df.drop(colname, axis = 1)
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Returns a DataFrame with appropriate column names
        Parameters:
            df (pd.DataFrame): unprocessed DataFrame
        Returns:
            df (pd.DataFrame): DataFrame with cleaned column names
    """
    number_map = {"1": "one", "2":"two", "3":"three", "4": "four", "5":"five",
             "6":"six", "7":"seven", "8": "eight", "9":"nine", "0":"zero"}
    numbers = "0123456789"
    colnames = [colname for colname in df.columns]
    new_colnames = {} # hold new column names
    for colname in colnames:
        # remove trailing whitespace
        colname_s = colname.strip()
        
        # check if colname starts with a number 
        if colname_s[0] in numbers:
            replacement = number_map[colname_s[0]]
            new_colname = replacement + colname_s[1:]
            new_colnames[colname] = new_colname
        else:
            # keep original column name otherwise
            new_colnames[colname] = colname_s
    df = df.rename(columns = new_colnames)
    print("All columns are appropriate for modeling.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate row entries from DataFrame"""
    df = df.drop_duplicates()
    print("Dataset does not contain duplicates.")
    return df


def impute_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Imputes missing features values
        Parameters:
            df (pd.DataFrame): unprocessed DataFrame
        Returns:
            df (pd.DataFrame): DataFrame with missing values imputed (if any)
    """
    # list of columns with missing values
    missing_cols = [
        col 
        for col in df.columns 
        if df[col].isna().sum() > 0
    ]
    
    for col in missing_cols:      
        # impute using average value (continuous)
        if df[col].dtype in ["int64", "float64"]:
            col_mean = df[col].mean()
            df[col] = df[col].replace(np.nan, col_mean)
            
        # impute using most frequent value (categorical)
        if df[col].dtype in ["object"]:
            col_mfreq = df[col].mode()
            df[col] = df[col].replace(np.nan, col_mfreq)
    print('Dataset has no missing features') 
    return df
        

def remove_missing_targets(df: pd.DataFrame, target="price") -> pd.DataFrame:
    """ 
    Drops rows that do not contain the target variable
        Parameters:
            df (pd.DataFrame): full dataset
            target (pd.Series): target column
        Returns:
            df (pd.DataFrame): dataset with missing target values removed
    """
    try:
        df = df.dropna(subset=target, axis = 0)
        print("Target column has no missing values.")
        return df
    except:
        print("Please choose an appropriate target.")
    


def get_data(file_path: str) -> pd.DataFrame:
    """ 
    Creates a pandas DataFrame from a file path
        Parameters:
            file_path (str): file path to .csv file of dataset
        Returns:
            df (pd.DataFrame): pandas DataFrame of dataset
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Cleans up erroneous or irrelevant data in the DataFrame
        Parameters:
            df (pd.DataFrame): unprocessed pandas DataFrame
        Returns:
            
    """
    df = drop_columns(df)
    df = rename_columns(df)
    df = remove_duplicates(df)
    df = impute_missing_features(df)
    df = remove_missing_targets(df)
    return df
    


file_path = "/kaggle/input/kc-house-data-nan/kc_house_data_NaN.csv"
df0 = get_data(file_path)
df0.head()


df0.info()


# create a dictionary of missing columns for the agent to understand
missing_cols = {
    col: missing_val 
    for col, missing_val in df0.isna().sum().items() 
    if missing_val > 0
}
missing_cols


# create a dictionary of the column data types for the agent to understand
col_dtypes = {
    col: str(dtype)
    for col, dtype in df0.dtypes.items()
}
col_dtypes


# target col and features
feature_cols = [col for col in df0.columns if col != "price"]
target_col = "price"
print(feature_cols, target_col)


# unique values, max, min, average for continuous columns
continuous_cols = [col for col in df0.columns if df0[col].dtype in ["int64", "float64"]]

cont_col_attributes = {}
for col in continuous_cols:
    unique_vals = {"unique_values": df0[col].nunique()}
    min_val = {"min_value": df0[col].min()}
    max_val = {"max_value": df0[col].max()}
    avg_val = {"average_value": round(df0[col].mean(), 2)}

    cont_col_attributes[col] = [unique_vals]
    cont_col_attributes[col].append(min_val)
    cont_col_attributes[col].append(max_val)
    cont_col_attributes[col].append(avg_val)

cont_col_attributes['bedrooms']


def describe_initial_data(file_path: str) -> dict:
    """ 
    Describes properties of the initial dataset
        Parameters:
            file_path (str): directory to dataset
        Returns:
            Dictionary containg status and dataset information
                Success: DataFrame properties
                Error: exception
    """
    try:
        df = pd.read_csv(file_path)
        # initial columns
        total_cols = list(df.columns)

        # feature columns
        feature_cols = [col for col in df.columns if col != "price"]

        target_col = "price"

        # column datatypes
        col_dtypes = {
            col: str(dtype)
            for col, dtype in df0.dtypes.items()
        }

        # rows and columns
        shape = df.shape

        # unique values, max, min, average for continuous columns
        continuous_cols = [
            col 
            for col in df.columns 
            if df[col].dtype in ["int64", "float64"]
        ]
        
        cont_col_attributes = {}
        for col in continuous_cols:
            unique_vals = {"unique_values": df0[col].nunique()}
            min_val = {"min_value": df0[col].min()}
            max_val = {"max_value": df0[col].max()}
            avg_val = {"average_value": round(df0[col].mean(), 2)}
        
            cont_col_attributes[col] = [unique_vals]
            cont_col_attributes[col].append(min_val)
            cont_col_attributes[col].append(max_val)
            cont_col_attributes[col].append(avg_val)
        
        # columns with missing values
        missing_cols = {
            col: missing_val 
            for col, missing_val in df.isna().sum().items() 
            if missing_val > 0
        }

        # choose impute method
        impute_methods = {}
        for col in missing_cols:
            if df[col].dtype in ["int64", "float64"]:
                impute_methods[col] = "average_value"
            else:
                impute_methods[col] = "most_frequent_value"
        
        return {
            "status": "success", 
            "total_cols": total_cols,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "column_datatypes": col_dtypes,
            "shape": shape,
            "continuous_column_attributes": cont_col_attributes,
            "missing_cols": missing_cols,
            "impute_methods": impute_methods,
        }
        
    except Exception as e:
        return {"status": "error",
               "message": str(e)}



preprocessor_agent = Agent(
    name = "preprocessor_agent",
    model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction = """ You are a thoughtful dataset preprocessor.
    
    For data preprocessing:
    1. Answer any questions about the original dataset using `describe_initial_data()`.
    2. If the user does not provide the file path, ask for the file path and consider 
       the query an error.
    3. The column datatypes are stored as values within the `col_dtypes` dictionary.
    4. For continuous columns, the number of unique values, maximum value, minimum value, 
       and average value are stored in `cont_col_attributes` for each continuous column. 
       Use proper units where available (i.e. $ for price assumming USD and sqft for area).

    If any status returns as error, explain the error to the user.
    """,
    tools = [describe_initial_data],
    output_key = "preprocessing",
)

print("âœ… Preprocessor agent created with custom function tools")
print("ðŸ”§ Available tools:")
print("  â€¢ describe_initial_data - Describes properties of the original dataset")



APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session
MODEL_NAME = "gemini-2.5-flash-lite"

# Step 2: Switch to DatabaseSessionService
# SQLite database will be created automatically
db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

# Create the Preprocessor Runner
preprocessor_runner = Runner(
    agent=preprocessor_agent, 
    app_name=APP_NAME, 
    session_service=session_service
)

print("âœ… Upgraded to persistent sessions!")
print(f"   - Database: my_agent_data.db")
print(f"   - Sessions will survive restarts!")


# Run a conversation
await run_session(
    preprocessor_runner,
    ["The file path is /kaggle/input/kc-house-data-nan/kc_house_data_NaN.csv",
    "How many rows are in the dataset?"],
    "stateful-agentic-session",
)


await run_session(
    preprocessor_runner,
    ["Are there any missing values in the dataset?"],
    "stateful-agentic-session",
)


await run_session(
    preprocessor_runner,
    ["How should I handle the missing values?"],
    "stateful-agentic-session",
)


await run_session(
    preprocessor_runner,
    ["What is the average price of a house?"],
    "stateful-agentic-session",
)


df1 = clean_data(df0)
df1.head()


# Verify that functions work
print(df1.isna().sum() > 0)
print(df1.duplicated())


# Create a Series object of price correlations
price_corrs_s = df1.corr(numeric_only=True)['price'].sort_values(ascending=False)
price_corrs_s


# Create a dictionary for the agent to understand
price_corrs = {
    feature: corr 
    for feature, corr in price_corrs_s.items()
    if feature != "price"
}
price_corrs


def show_correlations(features: List[str], target: str) -> dict:
    """ 
    Describes correlations between features and target variables
        Parameters:
            features (list[str]): list of column feature names
            target (str): target variable
        Returns:
            Dictionary of status and numeric feature correlations
                Success: {"status": "success","correlations": price_corrs,}
                Error: {"status": "error","message": e,}
    """
    try:
        df = pd.read_csv(file_path)
        df = clean_data(df)
        price_corrs_s = df.corr(numeric_only=True)['price'].sort_values(ascending=False)
        price_corrs = {
            feature: corr 
            for feature, corr in price_corrs_s.items()
            if feature != "price"
        }
        return {
            "status": "success",
            "correlations": price_corrs,
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
    
def generate_boxplot(colnames: List[str]) -> dict:
    """ 
    Generates a boxplot based on columns names in DataFrame
        Parameters:
            features (list[str]): list of column names
        Returns:
            Dictionary of status and dataframe properties
                Success: {"status": "success","column_names": colnames,}
                Error: {"status": "error","message": e,}
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        df = pd.read_csv(file_path)
        df = clean_data(df)
        sns.set_theme()
        # comparision between two variables
        if len(colnames) > 1:
            sns.boxplot(x = colnames[0], y = colnames[1], data=df[colnames])
            plt.title(f"{colnames[1]} vs {colnames[0]}")
        # plot of one variable
        else:
            sns.boxplot(x = colnames[0], data=df[colnames])
            plt.title(f"{colnames[0]}")
        plt.show()
        return {
            "status": "success",
            "column_names": colnames,
        }
    except:
        return {
            "status": "error",
            "message": "Cannot generate boxplot",
        }

def generate_regplot(colnames: List[str]) -> dict:
    """ 
    Generates a regression plot based on columns names in DataFrame
        Parameters:
            features (list[str]): list of column names
        Returns:
            Dictionary of status and dataframe properties
                Success: {"status": "success","column_names": colnames,}
                Error: {"status": "error","message": e,}
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        df = pd.read_csv(file_path)
        df = clean_data(df)
        sns.set_theme()
        # price is always y
        sns.regplot(x = colnames[0], y = colnames[1], data=df[colnames])
        plt.title(f"{colnames[1]} vs {colnames[0]}")
        plt.show()
        return {
            "status": "success",
            "column_names": colnames,
        }
    except:
        return {
            "status": "error",
            "message": "Cannot generate regplot",
        }


eda_agent = Agent(
    name = "eda_agent",
    model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction = """ You are a thoughtful exploratory analysis processor.
    
    For data exploration:
    1. Assume the data has been been preprocessed. Use the file path specified for preprocessing.
    2. Feature correlations to the target "price" are stored in correlations for each numeric feature.
       If the user asks for a correlation for a nonnumeric feature (data type not in int64 or float64),
       tell them it is not possible and suggest investigating the top two feature correlations. Use
       `show_correlations()` to find the correlations.
    3. Treat colnames as column names passed from the user to make boxplots with `generate_boxplot()`.
       If price is given, treat price as features[1]. Categorical features are x and continuous features
       are y in sns.boxplot(). If only one continuous variable is given, make a boxplot for that 
       variable. Use `generate_replot()` if regression is mentioned.
    4. Generate a regression plot with `generate_regplot()`. If price is mentioned by the user,
       then price is y in sns.regplot(). Provide a brief explanation of the plot by describing 
       the correlation between the column names provided by the user.
    5. If a plot cannot be made with either `generate_boxplot()` or `generate_regplot()`, provide
       a breakdown of the plot constraints.

    If any status returns as error, explain the error to the user.
    """,
    tools = [show_correlations, generate_boxplot, generate_regplot],
    output_key = "eda",
)

print("âœ… EDA agent created with custom function tools")
print("ðŸ”§ Available tools:")
print("  â€¢ show_correlations() - shows correlations between the features and target")
print("  â€¢ generate_boxplot() - generates a boxplot based on one or more columns")
print("  â€¢ generate_regplot() - generates a regression plot between the target and a feature")



APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session
MODEL_NAME = "gemini-2.5-flash-lite"

# Step 2: Switch to DatabaseSessionService
# SQLite database will be created automatically
db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

# Create the Preprocessor Runner
preprocessor_runner = Runner(
    agent=preprocessor_agent, 
    app_name=APP_NAME, 
    session_service=session_service
)

# Create the EDA Runner
eda_runner = Runner(
    agent=eda_agent, 
    app_name=APP_NAME, 
    session_service=session_service
)

print("âœ… Upgraded to persistent sessions!")
print(f"   - Database: my_agent_data.db")
print(f"   - Sessions will survive restarts!")


await run_session(
    preprocessor_runner,
    ["The file path is '/kaggle/input/kc-house-data-nan/kc_house_data_NaN.csv' What is the average price for a house?"],
    "stateful-agentic-session",
)


await run_session(
    eda_runner,
    ["Can you generate a boxplot for waterfront and price?"],
    "stateful-agentic-session",
)


await run_session(
    eda_runner,
    ["What feature is most correlated with price?"],
    "stateful-agentic-session",
)


await run_session(
    eda_runner,
    ["Can you show a boxplot of sqft_living?",
    "What does a correlation of 0.70 mean?"],
    "stateful-agentic-session",
)


await run_session(
    eda_runner,
    ["Can you show a regression plot between sqft_above and price?",
    "What is the relationship between sqft_above and the target?"],
    "stateful-agentic-session",
)


root_agent = SequentialAgent(
    name="DataAnalysisPipeline",
    sub_agents=[preprocessor_agent, eda_agent],
)

print("âœ… Sequential Agent created.")


APP_NAME = "root_app"  # Application
USER_ID = "root_user"  # User
SESSION = "root_session"  # Session
MODEL_NAME = "gemini-2.5-flash-lite"

# Step 2: Switch to DatabaseSessionService
# SQLite database will be created automatically
db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

# Create the Root Runner
root_runner = Runner(
    agent=root_agent, 
    app_name=APP_NAME, 
    session_service=session_service
)


print("âœ… Upgraded to persistent sessions!")
print(f"   - Database: my_agent_data.db")
print(f"   - Sessions will survive restarts!")


await run_session(
    root_runner,
    ["The file path is '/kaggle/input/kc-house-data-nan/kc_house_data_NaN.csv'. Can you make a boxplot for waterfront and price?"],
     "stateful-agentic-session",
)


# Clean up any existing database to start fresh (if Notebook is restarted)
import os

if os.path.exists("my_agent_data.db"):
    os.remove("my_agent_data.db")
print("âœ… Cleaned up old database files")

