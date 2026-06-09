import google.generativeai as genai
import pandas as pd
import io

# Configure your API Key
#genai.configure(api_key="AIzaSyAFj33QK_PJD**********")


import pandas as pd

# Change this path to match the dataset you added
df = pd.read_csv("/kaggle/input/income/train.csv")

df.head()



def generate_kaggle_starter(input_type, input_data):
    """
    input_type: 'url_text' or 'csv'
    input_data: The actual text string or the pandas DataFrame
    """
    context_str = ""

    if input_type == 'csv':
        buffer = io.StringIO()
        input_data.info(buf=buffer)
        info_str = buffer.getvalue()
        head_str = input_data.head(10).to_markdown()

        context_str = f"""
**Data Schema Info:**
{info_str}

**First 10 Rows of Data:**
{head_str}
"""
    elif input_type == 'url_text':
        context_str = f"**Competition Description & Data Overview:**\n{input_data}"

    # We ONLY build the prompt here, no API call
    prompt = f"""
You are an expert Kaggle AI Teammate.

### INPUT CONTEXT
{context_str}

### TASK
1. Parse the input to define the problem type and target variable.
2. Create a step-by-step EDA and Baseline plan.
3. Generate Python code (pandas, seaborn, sklearn) for the user to copy.
4. Explain the code simply for a beginner audience.

Ensure the code includes data cleaning (handling NaNs) and encoding before modeling.
"""
    return prompt



df = pd.read_csv("/kaggle/input/income/train.csv")
prompt = generate_kaggle_starter('csv', df)
print(prompt[:2000])



import pandas as pd
import numpy as np

# Already loaded:
# df = pd.read_csv("/kaggle/input/income/train.csv")

# Shape and data types
print(df.shape)
print(df.info())

# Quick stats for numeric columns
df.describe()

# Count missing values
df.isna().sum()
df.isna().mean() * 100  # percentage



import seaborn as sns
import matplotlib.pyplot as plt

target_col = "income_>50K"  # change if your column name is different

df[target_col].value_counts(normalize=True).plot(kind="bar")
plt.title("Target distribution")
plt.ylabel("Proportion")
plt.show()



num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [c for c in num_cols if c != target_col]

for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()



cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    plt.figure(figsize=(8,4))
    df[col].value_counts(normalize=True).head(10).plot(kind="bar")
    plt.title(f"Top categories in {col}")
    plt.ylabel("Proportion")
    plt.show()



for col in num_cols:
    plt.figure()
    sns.boxplot(x=target_col, y=col, data=df)
    plt.title(f"{col} vs {target_col}")
    plt.show()



# categorical vs target
for col in cat_cols:
    plt.figure(figsize=(8,4))
    prop = (df
            .groupby(col)[target_col]
            .mean()
            .sort_values(ascending=False)
            .head(10))
    prop.plot(kind="bar")
    plt.title(f"Mean {target_col} by {col}")
    plt.ylabel(f"Proportion with {target_col}=1")
    plt.show()





