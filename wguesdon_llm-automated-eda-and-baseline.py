!pip install openai==1.58.1 langchain-core langchain-openai


BASE_LLM = 'gpt-4o-2024-05-13'
ADVANCED_LLM = 'o1-preview'
SELECTED_LLM = BASE_LLM
TEMPERATURE = 0
MAX_TOKENS=3000


# Standard Library Imports
import os
import datetime
import json
import traceback

## LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from IPython.display import display, Markdown


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
OPENAI_API_KEY = user_secrets.get_secret("openai_key")


context = """ 
"general": {
"num_rows": 230130,
"num_columns": 6,
"num_missing_values": "8871",
"percent_missing_values": 0.6424629557206796
},
"data_types": {
"date": "object",
"country": "object",
"store": "object",
"product": "object",
"num_sold": "float64"
},
"missing_values": {
"date": {
"missing_count": 0,
"percent_missing": 0.0
},
"country": {
"missing_count": 0,
"percent_missing": 0.0
},
"store": {
"missing_count": 0,
"percent_missing": 0.0
},
"product": {
"missing_count": 0,
"percent_missing": 0.0
},
"num_sold": {
"missing_count": 8871,
"percent_missing": 3.8547777343240774
}
},
"numerical_summary": {
"count": {},
"mean": {},
"std": {},
"min": {},
"25%": {},
"50%": {},
"75%": {},
"max": {}
},
"categorical_summary": {
"date": {
"unique_counts": 2557
},
"country": {
"unique_counts": 6
},
"store": {
"unique_counts": 3
},
"product": {
"unique_counts": 5
}
},
"skewness_kurtosis": {
"num_sold": {
"skewness": 1.415373452498392,
"kurtosis": 2.6123350629213618
}
},
"correlations": {
"num_sold": {
"num_sold": 1.0
}
},
"outlier_summary": {
"num_sold": {
"outlier_count": 6630,
"percent_outliers": 2.8809803154738627
}
}
}
"""


template = """

Provide an analysis of the following EDA summary: The target variable is num_sold.
{context}

Add a comment about the missing values in the target variable: num_sold. And the implications if those are missing at random or not. 
Key Insights and Observations

"""


# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(template)

# Prepare parameters for ChatOpenAI
model_params = {
    "model": SELECTED_LLM,
    "api_key": OPENAI_API_KEY
}


display(Markdown(f'**selected model: {SELECTED_LLM}**'))

# Conditionally set temperature if supported
if SELECTED_LLM != ADVANCED_LLM:
    model_params["temperature"] = TEMPERATURE 
    model_params["max_tokens"] = MAX_TOKENS

if SELECTED_LLM == ADVANCED_LLM:
    model_params["max_completion_tokens"] = MAX_COMPLETION_TOKENS

# Initialize the model with the appropriate parameters
model = ChatOpenAI(**model_params)

# Create the processing chain
chain = prompt | model | StrOutputParser()

try:
    # Invoke the chain to get the result
    result = chain.invoke(context)

    # Save both the prompt and the result to a Markdown file
    file_path = '/kaggle/working/output_base_model.md'
    with open(file_path, 'w') as f:
        f.write("# EDA Report\n\n")
        f.write("## Prompt\n")
        f.write(template.format(context=context))
        f.write("\n\n## Response\n")
        f.write(result)

    # Display the result as Markdown in the notebook
    display(Markdown(result))

    display(Markdown(f"**Markdown report saved to: {file_path}**"))

except BadRequestError as e:
    print(f"An error occurred: {e}")


TEMPERATURE = 0.5
MAX_TOKENS=3500
MAX_ITERATIONS = 10
TARGET_SCORE = 0.07


best_model_script = """
# =========================================
# 1. LIBRARIES & CONFIGURATION
# =========================================
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# For reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =========================================
# 2. DATA LOADING
# =========================================
# Paths to the datasets
TRAIN_PATH = "/kaggle/input/playground-series-s5e1/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e1/test.csv"
GDP_PATH = "/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv"
SUBMISSION_PATH = "/kaggle/input/playground-series-s5e1/sample_submission.csv"

# Load datasets
train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
test_df = pd.read_csv(TEST_PATH, parse_dates=["date"])
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# =========================================
# 3. PREPROCESSING & IMPUTING MISSING VALUES
# =========================================

# Read GDP per capita data
gdp_per_capita_df = pd.read_csv(GDP_PATH)
years = [str(year) for year in range(2010, 2021)]  # 2010 to 2020 inclusive

# Prepare GDP ratios per country per year
relevant_countries = train_df["country"].unique()
gdp_per_capita_filtered_df = gdp_per_capita_df.loc[gdp_per_capita_df["Country Name"].isin(relevant_countries), ["Country Name"] + years].set_index("Country Name")
for year in years:
    gdp_per_capita_filtered_df[f"{year}_ratio"] = gdp_per_capita_filtered_df[year] / gdp_per_capita_filtered_df[year].sum()
gdp_per_capita_filtered_ratios_df = gdp_per_capita_filtered_df[[f"{year}_ratio" for year in years]]
gdp_per_capita_filtered_ratios_df.columns = [int(year) for year in years]
gdp_per_capita_filtered_ratios_df = gdp_per_capita_filtered_ratios_df.stack().reset_index().rename(columns={"level_1": "year", 0: "ratio"})
gdp_per_capita_filtered_ratios_df['year'] = gdp_per_capita_filtered_ratios_df['year'].astype(int)
gdp_per_capita_filtered_ratios_df.rename(columns={"Country Name": "country"}, inplace=True)

# Impute missing values in train_df
train_df_imputed = train_df.copy()
train_df_imputed['year'] = train_df_imputed['date'].dt.year
print(f"Missing values before imputation: {train_df_imputed['num_sold'].isna().sum()}")

for year in train_df_imputed['year'].unique():
    target_ratio = gdp_per_capita_filtered_ratios_df.loc[(gdp_per_capita_filtered_ratios_df['year'] == year) & (gdp_per_capita_filtered_ratios_df['country'] == 'Norway'), 'ratio'].values[0]
    # For Canada
    current_ratio_canada = gdp_per_capita_filtered_ratios_df.loc[(gdp_per_capita_filtered_ratios_df['year'] == year) & (gdp_per_capita_filtered_ratios_df['country'] == 'Canada'), 'ratio'].values[0]
    ratio_can = current_ratio_canada / target_ratio
    # Impute for Canada
    combinations_canada = [
        ('Discount Stickers', 'Holographic Goose'),
        ('Premium Sticker Mart', 'Holographic Goose'),
        ('Stickers for Less', 'Holographic Goose')
    ]
    for store, product in combinations_canada:
        mask_missing = (train_df_imputed['country'] == 'Canada') & \
                       (train_df_imputed['store'] == store) & \
                       (train_df_imputed['product'] == product) & \
                       (train_df_imputed['year'] == year) & \
                       (train_df_imputed['num_sold'].isna())
        if not mask_missing.any():
            continue
        corresponding_dates = train_df_imputed.loc[mask_missing, 'date']
        mask_norway = (train_df_imputed['country'] == 'Norway') & \
                      (train_df_imputed['store'] == store) & \
                      (train_df_imputed['product'] == product) & \
                      (train_df_imputed['year'] == year) & \
                      (train_df_imputed['date'].isin(corresponding_dates))
        norway_num_sold = train_df_imputed.loc[mask_norway, 'num_sold']
        train_df_imputed.loc[mask_missing, 'num_sold'] = norway_num_sold.values * ratio_can

    # For Kenya
    current_ratio_kenya = gdp_per_capita_filtered_ratios_df.loc[(gdp_per_capita_filtered_ratios_df['year'] == year) & (gdp_per_capita_filtered_ratios_df['country'] == 'Kenya'), 'ratio'].values[0]
    ratio_ken = current_ratio_kenya / target_ratio
    combinations_kenya = [
        ('Discount Stickers', 'Holographic Goose'),
        ('Premium Sticker Mart', 'Holographic Goose'),
        ('Stickers for Less', 'Holographic Goose'),
        ('Discount Stickers', 'Kerneler')
    ]
    for store, product in combinations_kenya:
        mask_missing = (train_df_imputed['country'] == 'Kenya') & \
                       (train_df_imputed['store'] == store) & \
                       (train_df_imputed['product'] == product) & \
                       (train_df_imputed['year'] == year) & \
                       (train_df_imputed['num_sold'].isna())
        if not mask_missing.any():
            continue
        corresponding_dates = train_df_imputed.loc[mask_missing, 'date']
        mask_norway = (train_df_imputed['country'] == 'Norway') & \
                      (train_df_imputed['store'] == store) & \
                      (train_df_imputed['product'] == product) & \
                      (train_df_imputed['year'] == year) & \
                      (train_df_imputed['date'].isin(corresponding_dates))
        norway_num_sold = train_df_imputed.loc[mask_norway, 'num_sold']
        train_df_imputed.loc[mask_missing, 'num_sold'] = norway_num_sold.values * ratio_ken

print(f"Missing values after imputation: {train_df_imputed['num_sold'].isna().sum()}")

# Handle any remaining missing values manually (if any)
remaining_missing = train_df_imputed[train_df_imputed['num_sold'].isna()]
if not remaining_missing.empty:
    # Assign specific values if necessary
    train_df_imputed.loc[train_df_imputed["id"] == 23719, "num_sold"] = 4
    train_df_imputed.loc[train_df_imputed["id"] == 207003, "num_sold"] = 195
    print(f"Missing values after manual assignment: {train_df_imputed['num_sold'].isna().sum()}")

# =========================================
# 4. CALCULATE STORE WEIGHTS
# =========================================
store_weights = train_df_imputed.groupby("store")["num_sold"].sum() / train_df_imputed["num_sold"].sum()
store_weights_df = store_weights.reset_index().rename(columns={"num_sold": "store_ratio"})

# =========================================
# 5. CALCULATE PRODUCT RATIOS
# =========================================
# Calculate daily product ratios
product_df = train_df_imputed.groupby(["date", "product"])["num_sold"].sum().reset_index()
product_pivot_df = product_df.pivot(index='date', columns='product', values='num_sold')
product_ratio_df = product_pivot_df.apply(lambda x: x / x.sum(), axis=1).stack().reset_index().rename(columns={0: "product_ratio"})
product_ratio_df['year'] = product_ratio_df['date'].dt.year

# Prepare forecasted product ratios by shifting previous years
product_ratio_2017_df = product_ratio_df[product_ratio_df['year'] == 2015].copy()
product_ratio_2018_df = product_ratio_df[product_ratio_df['year'] == 2016].copy()
product_ratio_2019_df = product_ratio_df[product_ratio_df['year'] == 2015].copy()

product_ratio_2017_df['date'] = product_ratio_2017_df['date'] + pd.DateOffset(years=2)
product_ratio_2018_df['date'] = product_ratio_2018_df['date'] + pd.DateOffset(years=2)
product_ratio_2019_df['date'] = product_ratio_2019_df['date'] + pd.DateOffset(years=4)

forecasted_ratios_df = pd.concat([product_ratio_2017_df, product_ratio_2018_df, product_ratio_2019_df], ignore_index=True)

# =========================================
# 6. AGGREGATE TIME SERIES
# =========================================
train_df_imputed = train_df_imputed.groupby(["date"])["num_sold"].sum().reset_index()
train_df_imputed["year"] = train_df_imputed["date"].dt.year
train_df_imputed["month"] = train_df_imputed["date"].dt.month
train_df_imputed["day"] = train_df_imputed["date"].dt.day
train_df_imputed["day_of_week"] = train_df_imputed["date"].dt.dayofweek

# =========================================
# 7. ADJUST FOR DAY OF WEEK EFFECTS
# =========================================
day_of_week_ratio = (train_df_imputed.groupby("day_of_week")["num_sold"].mean() / train_df_imputed["num_sold"].mean()).rename("day_of_week_ratios")
train_df_imputed = train_df_imputed.merge(day_of_week_ratio, on='day_of_week', how='left')
train_df_imputed["adjusted_num_sold"] = train_df_imputed["num_sold"] / train_df_imputed["day_of_week_ratios"]

# =========================================
# 8. MAKE FORECAST
# =========================================
train_day_mean_df = train_df_imputed.groupby(["month", "day"])["adjusted_num_sold"].mean().reset_index()

# Prepare test_total_sales_df
test_total_sales_df = test_df[['date']].drop_duplicates()
test_total_sales_df['month'] = test_total_sales_df['date'].dt.month
test_total_sales_df['day'] = test_total_sales_df['date'].dt.day
test_total_sales_df['day_of_week'] = test_total_sales_df['date'].dt.dayofweek

test_total_sales_df = test_total_sales_df.merge(train_day_mean_df, on=['month', 'day'], how='left')
test_total_sales_df = test_total_sales_df.merge(day_of_week_ratio.reset_index(), on='day_of_week', how='left')
test_total_sales_df["day_num_sold"] = test_total_sales_df["adjusted_num_sold"] * test_total_sales_df["day_of_week_ratios"]

# =========================================
# 9. DISAGGREGATE TOTAL SALES FORECAST
# =========================================
# Merge test_df with test_total_sales_df
test_sub_df = test_df.merge(test_total_sales_df[['date', 'day_num_sold']], on='date', how='left')

# Add store ratios
test_sub_df = test_sub_df.merge(store_weights_df, on='store', how='left')

# Add country ratios
test_sub_df['year'] = test_sub_df['date'].dt.year
test_sub_df = test_sub_df.merge(gdp_per_capita_filtered_ratios_df.rename(columns={'ratio': 'country_ratio'}), on=['country', 'year'], how='left')

# Add product ratios
test_sub_df = test_sub_df.merge(forecasted_ratios_df[['date', 'product', 'product_ratio']], on=['date', 'product'], how='left')

# Disaggregate to get num_sold
test_sub_df["num_sold"] = test_sub_df["day_num_sold"] * test_sub_df["store_ratio"] * test_sub_df["country_ratio"] * test_sub_df["product_ratio"]
test_sub_df["num_sold"] = test_sub_df["num_sold"].round()

# Ensure predictions are non-negative
test_sub_df["num_sold"] = test_sub_df["num_sold"].clip(lower=0)

# =========================================
# 10. SUBMISSION GENERATION
# =========================================
submission = pd.read_csv(SUBMISSION_PATH)
submission['num_sold'] = test_sub_df['num_sold']
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
submission.to_csv(f"sub_m13_{timestamp}.csv", index=False)
print("Submission saved!")
"""


metric = "Mean Absolute Percentage Error (MAPE)"
train_data_path = "/kaggle/input/playground-series-s5e1/train.csv"
test_data_path = "/kaggle/input/playground-series-s5e1/test.csv"
submission_example_path = "/kaggle/input/playground-series-s5e1/sample_submission.csv"
gdp_path = "/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv"
submission_path = "/kaggle/working/submission.csv"
target_variable = "num_sold"

train_data_summary_json = json.dumps(context, indent=2)

# --------------------------------------------------------------------
# 1. Prepare your "initial" prompt
# --------------------------------------------------------------------
system_instructions = f"""
You are a coding assistant. 
**IMPORTANT**: You must use the following MANDATORY CODE snippet as a baseline to improve. 
Do not remove the part using an external GDP dataset for features enginering. 

**IMPORTANT**: You must generate ready to run Python code. Do not add additional plain text and do not use delimiters like ```python ```
the code must be ready to run via exec() do not add anything that would make this fails. 

--- BEGIN MANDATORY CODE ---

{best_model_script}

--- END MANDATORY CODE ---

Only fix minor syntax or logic details if needed, but do not remove the code that reads and merges GDP data. 
Output valid Python code that runs end to end.
"""

user_instructions = f"""
You are given the following dataset information:
- Train data path: {train_data_path}
- GDP path: {gdp_path}
- Test data path: {test_data_path}
- Submission example path: {submission_example_path}
- Train data summary: {train_data_summary_json}
- Target variable: {target_variable}
- Path to the final submission file: {submission_path}

**Task**:
1. Incorporate the mandatory GDP snippet from the system instructions (already included above).
2. Train a model to predict {target_variable}.
3. Generate a valid Kaggle submission at {submission_path}.
4. Compute the '{metric}' on a validation split and store it in 'val_mape'.
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
1. Provide a corrected Python script that STILL meets all original requirements:
   - Must incorporate the mandatory GDP snippet (already provided in the system instructions).
   - Must generate a submission file and compute '{metric}' in a variable named 'val_mape'.
2. Do **not** remove or ignore the GDP logic from the snippet, only fix the necessary parts.
3. Return only valid Python code, with no triple backticks or markdown.

Begin now.
"""

repair_chain_prompt = ChatPromptTemplate.from_template(repair_prompt_template)

# --------------------------------------------------------------------
# 3. Create your LLM & output parser
# --------------------------------------------------------------------
model_params = {
    "model": BASE_LLM,
    "openai_api_key": OPENAI_API_KEY,
}

if SELECTED_LLM != ADVANCED_LLM:
    model_params["temperature"] = TEMPERATURE 
    model_params["max_tokens"] = MAX_TOKENS
    
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
            "best_model_script" : best_model_script,
            "train_path": train_data_path,
            "gdp_path": gdp_path,
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
        if "val_mape" in local_namespace:
            val_mape = local_namespace["val_mape"]
            print(f"val_mape from script: {val_mape}")
            if val_mape < 0.07:
                success = True
            else:
                error_message = f"MAPE {val_mape} is above threshold"
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

