# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install google-genai pandas


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")


# ==========================================
# STEP 1: INSTALL & SETUP
# ==========================================

import os
import pandas as pd
import json
import numpy as np
from google import genai
from google.genai import types

try:
    if 'GOOGLE_API_KEY' in os.environ:
        client = genai.Client()
        print("Gemini client successfully initialized.")
    else:
        client = None
        print("ERROR: GOOGLE_API_KEY not found in Secrets.")
except Exception as e:
    client = None
    print(f"Error: {e}")

def load_data():
    try:
        df = pd.read_csv("/kaggle/input/titanic/train.csv")
        # Drop irrelevant columns for this demo
        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        return df
    except FileNotFoundError:
        print("Error: Titanic dataset not found. Please add it to the notebook.")
        return pd.DataFrame()

df_global = load_data()
print("Data loaded. Shape:", df_global.shape)

# ==========================================
# STEP 2: TOOL DEFINITIONS
# ==========================================

def identify_missing_values(data: pd.DataFrame) -> str:
    """Identifies columns with nan values."""
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    
    if missing.empty:
        return json.dumps({"status": "Clean", "message": "No missing values found."})

    report = {col: {"count": int(count), "dtype": str(data[col].dtype)} 
              for col, count in missing.items()}
    return json.dumps({"status": "Issues Found", "missing_columns": report}, indent=2)

def impute_median(data: pd.DataFrame, column_name: str) -> str:
    """Imputes missing values in numeric columns with the median."""
    if column_name not in data.columns:
        return json.dumps({"error": f"Column {column_name} not found."})
    
    if data[column_name].dtype not in ['int64', 'float64']:
        return json.dumps({"error": f"Column {column_name} is not numeric."})

    median_val = data[column_name].median()
    data[column_name].fillna(median_val, inplace=True)
    
    global df_global
    df_global[column_name] = data[column_name]

    return json.dumps({"status": "Success", "action": "Impute", "value": median_val})

def one_hot_encode(data: pd.DataFrame, column_name: str) -> str:
    """Performs One-Hot Encoding on categorical columns."""
    if column_name not in data.columns:
        return json.dumps({"error": f"Column {column_name} not found."})

    # Simple fill for categorical nan before encoding
    if data[column_name].isnull().any():
        data[column_name].fillna(data[column_name].mode()[0], inplace=True)

    original_shape = data.shape
    encoded_df = pd.get_dummies(data, columns=[column_name], prefix=column_name, drop_first=True)
    
    global df_global
    df_global = encoded_df
    
    return json.dumps({
        "status": "Success", 
        "new_columns": encoded_df.shape[1] - original_shape[1] + 1
    })

TOOLS = {
    "identify_missing_values": identify_missing_values,
    "impute_median": impute_median,
    "one_hot_encode": one_hot_encode
}

# ==========================================
# STEP 3: AGENT ORCHESTRATION
# ==========================================

def run_agent(task: str, client: genai.Client) -> str:
    if not client: return "Agent not active."

    tools_list = [identify_missing_values, impute_median, one_hot_encode]
    
    messages = [
        types.Content(role="user", parts=[
            types.Part.from_text(f"Task: {task}. Use the provided tools to analyze/modify the dataframe 'df_global'. Be concise.")
        ])
    ]
    
    for _ in range(5):
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=messages,
            config=types.GenerateContentConfig(tools=tools_list)
        )

        if response.function_calls:
            fc = response.function_calls[0]
            func_name, args = fc.name, dict(fc.args)
            
            print(f"Agent calling: {func_name} | Args: {args}")
            if func_name in TOOLS:
                result = TOOLS[func_name](df_global, **{k:v for k,v in args.items() if k!='data'})
            else:
                result = "Error: Tool not found."
            
            messages.append(response.candidates[0].content)
            messages.append(types.Content(
                role="tool", 
                parts=[types.Part.from_function_response(name=func_name, response={"result": result})]
            ))
        else:
            return response.text
            
    return "Task incomplete after max steps."

# ==========================================
# STEP 4: EXECUTION & DEMO
# ==========================================

if client:
    print("\n--- 1. ANALYSIS ---")
    print(run_agent("Identify all missing values in the dataset.", client))
    
    print("\n--- 2. ACTION: IMPUTATION ---")
    print(run_agent("Impute missing values in 'Age' using the median.", client))
    
    print("\n--- 3. ACTION: ENCODING ---")
    print(run_agent("Encode the 'Sex' column using One-Hot Encoding.", client))
    
    print("\n--- FINAL DATA STATUS ---")
    print(df_global.head().to_markdown(index=False))
    print("\nMissing Values:", df_global.isnull().sum().sum())

