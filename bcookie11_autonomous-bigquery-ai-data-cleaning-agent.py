import os 
import pandas as pd
from dataclasses import dataclass 
from google.cloud import bigquery
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

OLLAMA_HOST_URL = user_secrets.get_secret("OLLAMA_HOST_URL")

BQ_PROJECT_ID = "genial-motif-472804-s1"
# IN TESTING -- CURRENTLY USED IN THIS WORKSPACE
BQ_DATASET_ID = "cleaning_service" # Database to store unclean and workspace for now 
BQ_TABLE_ID = "sample_dataset" # actual data table named as
SAMPLE_FULL_ID = "genial-motif-472804-s1.cleaning_service.sample_dataset"
SAMPLE_SUMMARY_FULL_ID = "genial-motif-472804-s1.cleaning_service.field_summary"

# TRUE DB CONFIGURATION
# DATASET NAMES
BQ_DATASET_OBSERVATION_ID = "observations"
BQ_DATASET_ACTION_ID = "cognitive"
BQ_DATASET_OUTPUT_ID = "cleaned_data"

# TABLE NAMES
BQ_TABLE_SUMMARY_ID = "field_summary" # stores in the summary of the uploaded dataset should be prefixed with the episodes id 

# GATEKEEPER GENERATIVE MODEL 
BQ_MODEL_CONNECTION = "projects/481034637222/locations/australia-southeast1/connections/__default_cloudresource_connection__"
BQ_MODEL_ENDPOINT = "projects/genial-motif-472804-s1/locations/australia-southeast1/publishers/google/models/gemini-2.5-flash"
BQ_MODEL_ID = "genial-motif-472804-s1.cleaning_service.gatekeeper"
DEFAULT_MODEL_TYPE = "gemini-2.5-flash"
# LOCAL FILE PATHS
LOCAL_SAMPLE_PATH = "/kaggle/input/cafe-sales-dirty-data-for-cleaning-training/dirty_cafe_sales.csv"


client = bigquery.Client(BQ_PROJECT_ID)


EPISODE_WINDOW_PROMPT = """
You are a senior data analyst and currently performing data cleaning tasks.
{current_workspace}
Given the following knowledge of your current dataset, respond with your next action from one of the following options:
{action_space}
Your response must be one of the action options.
"""


"""

src.gaby_agent.core.agent._core
Core classes and functions for the Gaby Agent system.
"""

import os
from abc import ABC
from pathlib import Path
from threading import Lock
from dotenv import load_dotenv
from dataclasses import dataclass

#try:
#    os.chdir(Path(__file__).resolve().parents[4])
#except Exception:
#    os.chdir(Path.cwd().root if hasattr(Path.cwd(), "root") else Path.cwd())

#load_dotenv(".env.local")

import ollama

@dataclass
class Instructor:
    prompt: str
    input_template: str | None = None

    def input_validator(self, **kwargs) -> str:
        """Fill in the input template with kwargs if provided, otherwise return kwargs as str."""

        if self.input_template:
            return self.input_template.format(**kwargs)

        return str(kwargs)

class GabyBasement(ABC):
    """ Prompt Base Constructor. """

    _instance = None
    _lock = Lock()
    client = None
    base_model_id = os.getenv("BASE_GUFF_LLM_MODEL", "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q3_K_XL")

    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'client'):
            raise AttributeError("Ollama client is not initialized. Revise subclass / Base class structure design.")

    def __new__(cls, *args, **kwargs):
        # Double-checked locking for thread safety
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls, *args, **kwargs)
                try:
                    cls._instance.client = ollama.Client(os.getenv("OLLAMA_HOST_URL", "http://localhost:11434"))
                except Exception as e:
                    raise RuntimeError(f"Failed to init GabyBasement client: {e}")

            if len(cls._instance.client.list().models) == 0 or cls.base_model_id not in cls._instance.client.list().models  :
                cls._instance.client.pull(model=cls.base_model_id)
                print(f"Pulled model: {cls.base_model_id}")

        return cls._instance

    def __init_subclass__(cls, prompt: Instructor, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.prompt = getattr(cls, "prompt", prompt)  # don’t overwrite if re-init
        cls.name = cls.__qualname__

    @property
    def system_prompt(self):# -> list[dict[str, Any]]:
        text = self.prompt.prompt if isinstance(self.prompt, Instructor) else str(self.prompt)
        return [{"role": "system", "content": text}]

    def post_process(self, response) -> str:
        """ Post-processes the response from the LLM before returning to the user. Subclass can override this method to implement custom post-processing logic. """

        return response.message.get('content', None).strip()

    def pre_process(self, **kwargs) -> dict:
        """ Pre-processes the input arguments before sending to the LLM. Subclass can override this method to implement custom pre-processing logic. """

        return kwargs

    def run(self, **kwargs) -> str:
        """ Main method to execute the thought chain. """

        if not hasattr(self, "client") or self.client is None:
            raise RuntimeError("Ollama client is not initialized."
                               " Ensure Ollama is running and OLLAMA_HOST_URL is correct.")

        print(f"Running thought Chain: {self.name}")

        kwargs = self.pre_process(**kwargs)
        user_inputs = self.prompt.input_validator(**kwargs)

        response = self.client.chat(
            model=self.base_model_id,
            messages=self.system_prompt + [{"role": "user", "content": user_inputs}],
            stream=False,
            options={
                "max_tokens": 100,
                "num_ctx": 100
            }
        )

        return self.post_process(response)

if __name__ == "__main__":
    pass


# Wrapper / Utilities functions

import functools


def pandas_gatekeeper(func):
    """
    A decorator that executes a SQL query generated by a function and returns a DataFrame.

    This decorator intercepts the SQL string returned by the wrapped function,
    executes it using the provided BigQuery client, and returns the result
    as a pandas DataFrame.

    The decorated function must be called with a 'client' keyword argument
    of type `google.cloud.bigquery.Client`.

    Args:
        func: The function to be decorated, which should return a SQL query string.

    Returns:
        A wrapper function that executes the query and returns a DataFrame.

    Raises:
        TypeError: If the 'client' keyword argument is not provided or is not a
                   `bigquery.Client` instance.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # client = bigquery.Client() -- Since running in notebook.

        # Call the original function to get the SQL query string
        sql_query = func(*args, **kwargs)

        print(f"--- Executing SQL from '{func.__name__}' ---")
        print(sql_query)

        # Execute the query and return the result as a pandas DataFrame
        job = client.query(sql_query)
        return job.to_dataframe()

    return wrapper


# HELPER FUNCTIONS 

def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        total_count = len(df)
        missing_count = df[col].isna().sum()
        data_type = df[col].dtype

        # Check if continuous: numeric with many unique values
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20:
            unique_vals = "continuous"
        else:
            unique_vals = df[col].nunique()

        yield {
            "data_field_name": col,
            "missing_count": missing_count,
            "total_count": total_count,
            "data_type": str(data_type),
            "unique_values": unique_vals
        }

def upload_dataframe_to_bq(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str):
    """ Uploads a pandas DataFrame to a specified BigQuery table. """
    
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, 
        autodetect=True
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete.
    print(f"✅ Data uploaded to {table_ref}")


# Data Reports & Profilers for updating databases in BigQuery between sessions etc. 

# PROFILERS 
from uuid import uuid4
from datetime import datetime 

@dataclass 
class EpisodeConfig:
    input_id: str = SAMPLE_FULL_ID
    summary_id: str = SAMPLE_SUMMARY_FULL_ID
    # BIGQUERY MODEL STORED LOCS
    bq_model_connection: str = BQ_MODEL_CONNECTION
    bq_model_endpoint: str = BQ_MODEL_ENDPOINT
    bq_model_id: str = BQ_MODEL_ID
    default_model_type: str = DEFAULT_MODEL_TYPE
    # DEFAULT NAMES
    default_project_id: str = BQ_PROJECT_ID
    default_dataset_id: str = BQ_DATASET_ID
    default_table_id: str = BQ_TABLE_ID
    
@dataclass 
class EntryReport:
    description: str | None = None 
    data_field_summary: pd.DataFrame | None = None
    data_field_description: pd.DataFrame | None = None
    numeric_table: pd.DataFrame | None = None
    
@dataclass 
class MissingDataReport:
    missing_pattern: str
    missing_count: float 
    missing_perc: float
    data_field_type: str
    
@dataclass
class DataProfiler:
    # User Inputs
    data: pd.DataFrame
    user_input_tags: str | list | None = None

    # Defining the dataset
    description: str | None = None # Describing the dataset in natural language
    data_field_summary: pd.DataFrame | None = None # Data field summary table
    data_field_description: pd.DataFrame | None = None # Data field summary table with description columns 
    numeric_table: pd.DataFrame | None = None # Data field summary table with description columns
    
    _send_to_gatekeeper: bool = False
    # Episode ID & Configuration
    episode_id: str = uuid4().hex
    timestamp: str = datetime.now().isoformat()
    config: EpisodeConfig | None = None 
    
    def __post_init__(self):
        self.define_dataset(self._send_to_gatekeeper)
        # the EpisodeConfig stored dataset id for the input dataset is set to self.episode_id-self.tiemstamp in prod
        # self.config = EpisodeConfig(
        #    input_dataset_id=f"{self.episode_id}-{self.timestamp}",
        #    summary_table_id=f"{self.episode_id}-{BQ_TABLE_SUMMARY_ID}"
        #)
        self.config = EpisodeConfig()
        
    def define_dataset(self, upload_summary: bool = False):
        if self.data.shape[0] == 0 or self.user_input_tags is None:
            raise ValueError("The provided DataFrame is empty or data origin is not specified. Both these are required to start the workflow.")
        
        # assuming have loaded the model and returned it 
        self.data_field_summary = pd.DataFrame.from_records(list(summarize_dataframe(self.data)))
        
        print(f"✅ Dataset defined with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
        
        if upload_summary is True:
            upload_dataframe_to_bq(self.data, BQ_PROJECT_ID, BQ_DATASET_ID, BQ_TABLE_ID)
            upload_dataframe_to_bq(self.data_field_summary, BQ_PROJECT_ID, BQ_DATASET_ID, BQ_TABLE_SUMMARY_ID)

        print(f"Completed profiling for dataset id: {self.episode_id} and uploaded to BQ.")
        
    @property 
    def end_cleaning_report(self) -> EntryReport:
        return EntryReport(
            description=self.description,
            data_field_summary=self.data_field_summary,
            data_field_description=self.data_field_description
        )

    def episode_recap(self):
        context_prompt = f"--- Data Summary ---"
        
        end_cleaning_report = self.end_cleaning_report

        context_prompt += f"\nDataset Description: {end_cleaning_report.description}\n" if end_cleaning_report.description is not None else "\nNo dataset description available.\n"
        context_prompt += f"\nData Field Summary:\n{end_cleaning_report.data_field_summary.to_markdown(index=False)}\n" if end_cleaning_report.data_field_summary is not None else ""
        context_prompt += f"\nData Field Description:\n{end_cleaning_report.data_field_description.to_markdown(index=False)}\n" if end_cleaning_report.data_field_description is not None else ""
            
        return context_prompt



# load the datasets into workspace
data = pd.read_csv(LOCAL_SAMPLE_PATH, header=0)
gb = DataProfiler(data=data, user_input_tags=["sales", "beverages", "cafe"])


""" clean_stage_a.py """

class DatasetSummarizer(
    GabyBasement,
    prompt = Instructor(
        prompt="""
        You are a senior data analyst. Based on the dataset’s fields and descriptive metadata, provide a concise summary (no more than 2 sentences) that highlights:
        •	the dataset’s key characteristics and notable features or patterns,
        •	potential modeling or analytical objectives it may support, and
        •	whether the dataset contains any unique identifiers.
        """,
        input_template="""
        Dataset descriptive labels: {user_inputs},
        Dataset Subset:
        {data_table}
        """
    )
):
    pass 

class DataFieldMetaDescription(
    GabyBasement,
    prompt = Instructor(
        prompt="You are a data analyst. Given the dataset description and a specific data field label, return a concise description of what the data field possibly means in the context of the dataset. Return your response in at most 1 sentence.",
        input_template="""
        Dataset Description: {data_description}
        Data Field Label: {data_label}
        Data Sample: {data_sample}
        """
    )
):
    def run_loop(self, data: pd.DataFrame, data_description: str) -> dict:
        """ Run the description for each data field in the dataframe. """

        descriptions = {}

        for column in data.columns:
            sample = data[column].dropna().unique()[:3].tolist()
            sample_str = ", ".join(map(str, sample))

            description = self.run(
                data_description=data_description,
                data_label=column,
                data_sample=sample_str
            )

            descriptions[column] = [
                {
                    'name': self.post_process(description),
                    'data_type': str(data[column].dtype),
                    'description': description
                }
            ]

        return descriptions



SQL_DESCRIBE_DATA_FIELD_LABEL= """
SELECT
  data_field_name,
  AI.GENERATE( ('The data field name ',
      data_field_name,
      'with values of data type,',
      data_type,
      'is one of the dataset column labels of a dataset with description: cafe sale logs. In a sentence, define what each data field represent.'),
    connection_id => '{connection_id}',
    endpoint => '{endpoint}',
    output_schema => 'data_field_name STRING, description STRING').description
FROM
  {data_summary_id};
"""

# 2 methods in this file to distinguish numeric values.
# Ordinal: categories have a natural order but no consistent distance between them (e.g., low, medium, high)
SQL_DETECT_NUMERIC_FIELD = """
SELECT
  data_field_name,
  AI.GENERATE( ('The data field name ',
      data_field_name,
      'with values of data type,',
      data_type,
      'is one of the dataset column labels of a dataset with description: cafe sale logs.'
      'Classify the dataset field into one of: Nominal, Ordinal, Continuous, Unknown, if the data type is numerical and if not, return Unknown. Return the result strictly as: "<Nominal|Ordinal|Continuous|Unknown>"'
      ),
    connection_id => '{connection_id}',
    endpoint => '{endpoint}',
    output_schema => 'data_field_name STRING, numeric_type STRING').numeric_type
FROM
  {data_summary_id};
"""

@pandas_gatekeeper
def describe_data_field(
    data_summary_id: str,
    connection_id: str = BQ_MODEL_CONNECTION,
    endpoint: str = DEFAULT_MODEL_TYPE
): 
    return SQL_DESCRIBE_DATA_FIELD_LABEL.format(
        data_summary_id=data_summary_id,
        connection_id=connection_id,
        endpoint=endpoint
    )

@pandas_gatekeeper
def detect_numeric_field(
    data_summary_id: str,
    connection_id: str = BQ_MODEL_CONNECTION,
    endpoint: str = DEFAULT_MODEL_TYPE
): 
    return SQL_DETECT_NUMERIC_FIELD.format(
        data_summary_id=data_summary_id,
        connection_id=connection_id,
        endpoint=endpoint
    )
    
#gb.data_field_description = describe_data_field(
#    data_summary_id=SAMPLE_SUMMARY_FULL_ID,
#    connection_id=BQ_MODEL_CONNECTION,
#    endpoint=DEFAULT_MODEL_TYPE
#)


detect_numeric_field(data_summary_id=SAMPLE_SUMMARY_FULL_ID)


def data_cleaning_pipeline(report: DataProfiler) -> DataProfiler:
    """ Main function to run the data cleaning pipeline. """
    
    report.description = DatasetSummarizer().run(
        user_inputs=report.user_input,
        data_table=report.data.head(3).to_string(index=False)
    )
    
    try:
        report.data_field_description = describe_data_field(
            connection_id=report.config.bq_model_connection,
            model_endpoint=report.config.default_model_type,
        )

        report.numeric_table = detect_numeric_field(
            connection_id=report.config.bq_model_connection,
            endpoint=report.config.default_model_type,
            data_summary_id=report.config.summary_id
        )
    except Exception as e:
        print("Error using GCP model, falling back to local model (takes longer to run):", e)
        
        report.data_field_description = DataFieldMetaDescription().run_loop(
            data=report.data,
            data_description=report.description
        )
        report.numeric_table = None 
        
    return report

gb = data_cleaning_pipeline(gb)


gb.data_field_summary


print(gb.description)


gb.data_field_description


SQL_MISSING_DATA_PATTERN_METHOD = """
SELECT
  data_field_name,
  data_type,
  AI.GENERATE( ('The data field name ',
      data_field_name,
      'is sourced from dataset with description: car sale logs.',
      
      'Data field is of data type,',
      data_type,
      'and has a total',
      missing_count,
      'number of missing values. Suggest the most common method in dealing with missing datasets of this data type and context. Your response must only contain one of the values fromthe following: [\'imputation\', \'drop_missing\', ].'),
    connection_id => '{connection_id}',
    endpoint => '{endpoint}',
    output_schema => 'data_field_name STRING, description STRING').description
FROM
  {data_summary_id};
"""

@pandas_gatekeeper
def suggest_missing_data_pattern_method(
    data_summary_id: str,
    connection_id: str = BQ_MODEL_CONNECTION,
    endpoint: str = DEFAULT_MODEL_TYPE
): 
    return SQL_MISSING_DATA_PATTERN_METHOD.format(
        data_summary_id=data_summary_id,
        connection_id=connection_id,
        endpoint=endpoint
    )


def get_continuous_fields(df: pd.DataFrame,
                          numeric_types=("INT64", "FLOAT64", "NUMERIC"),
                          min_unique=20,
                          min_unique_ratio=0.05):
    """
    Return only the continuous fields from a dataset summary DataFrame.

    Args:
        df (pd.DataFrame): Must have columns [data_field_name, data_type, unique_values, total_count].
        numeric_types (tuple): BigQuery numeric types considered numeric.
        min_unique (int): Minimum unique values threshold.
        min_unique_ratio (float): Minimum ratio (unique / total_count).

    Returns:
        pd.DataFrame: Filtered DataFrame containing only continuous fields.
    """
    mask = (
        df["data_type"].isin(numeric_types)
        & (
            (df["unique_values"] > min_unique) |
            (df["unique_values"] / df["total_count"] > min_unique_ratio)
        )
        & (df["unique_values"] < df["total_count"])  # exclude pure IDs
    )
    return df.loc[mask].reset_index(drop=True)


BQ_DATA_OBSERVATION_ID = "observations"
BQ_DATA_INSIGHT_ID = "insights"


from functools import wraps

N_STEPS = 10 # Number of steps  

def every_n_steps(n: int):
    """Decorator to ensure a function only runs every n steps."""
    def decorator(func):
        counter = {"step": 0}  # mutable closure

        @wraps(func)
        def wrapper(*args, **kwargs):
            counter["step"] += 1
            if counter["step"] % n == 0:
                return func(*args, **kwargs)
            else:
                # Optional: return None or a placeholder when skipped
                return None
        return wrapper
    return decorator

@every_n_steps(N_STEPS)
@pandas_gatekeeper
def fetch_recent_observations(days: int = 7) -> str:
    """ Fetch recent observations from the last number of days. """
    return f"""
    SELECT *
    FROM `{BQ_PROJECT_ID}.{BQ_DATASET_OBSERVATION_ID}.observations`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
    ORDER BY timestamp DESC
    """
    
@every_n_steps(N_STEPS)
@pandas_gatekeeper
def store_agent_insight(action: str,
                        reward: float,
                        metadata: dict) -> str:
    """ Log an agent's decision, reward, and metadata into BigQuery. """
    return f"""
    INSERT INTO `{BQ_PROJECT_ID}.{BQ_DATA_INSIGHT_ID}.rewards`
    (timestamp, action, reward, metadata)
    VALUES (
        CURRENT_TIMESTAMP(),
        '{action}',
        {reward},
        TO_JSON_STRING({metadata})
    )
    """

@every_n_steps(N_STEPS)
@pandas_gatekeeper
def forecast_agent_rewards(horizon: int = 10) -> str:
    """ Forecast expected rewards for next N steps using BigQuery ML. """
    return f"""
    SELECT *
    FROM ML.FORECAST(
      MODEL `{BQ_PROJECT_ID}.{BQ_DATA_INSIGHT_ID}.reward_forecaster`,
      STRUCT({horizon} AS horizon, 0.9 AS confidence_level)
    )
    """

