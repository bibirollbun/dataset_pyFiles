!pip install --quiet google-cloud-bigquery-storage #Reduces future errors and increases readability


from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
PROJECT_ID = user_secrets.get_secret("GCP_PROJECT_ID")
DATASET_ID = 'concept_data'
DATASET_TABLE = 'concept_content'
DATASET_LOCATION = 'US' 

PATIENTS_TO_PROCESS = 10 #How many patients to grab from the public dataset
REPORTS_PER_PATIENT = 50 #How many reports do you want to generate and embed per person?

print('Project ID and Database settings have been succesfully set.')


from google.cloud import bigquery
from google.api_core import exceptions as api_exceptions

#BigQuery Client
client = bigquery.Client(project=PROJECT_ID)

try:
    #Basic Dataset params
    dataset_ref = client.dataset(DATASET_ID)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = DATASET_LOCATION
    
    #Send the dataset to the API for creation.
    dataset = client.create_dataset(dataset)
    print(f"Dataset '{DATASET_ID}' created successfully in {DATASET_LOCATION}")
except api_exceptions.Conflict:
    #Handle the case where the dataset already exists.
    print(f"Dataset '{DATASET_ID}' already exists. Skipping creation.")
except Exception as e:
    #Handle other potential API errors.
    print(f"An error occurred: {e}")


#Import necessary libraries for BigQuery and data handling
from google.cloud import bigquery
import pandas as pd

#BigQuery Client
client = bigquery.Client(project=PROJECT_ID)

#Query setup, joining tables together from the public dataset to create our own. Lacks a unique identifier, so we use FARM_FINGERPRINT to make one.
# This query will create or replace the 'DATASET_TABLE' table in your dataset.
query = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{DATASET_TABLE}` AS
WITH RankedConditions AS (
    SELECT
        t1.person_id,
        t1.condition_concept_id,
        t3.ancestor_concept_id,
        ANY_VALUE(t1.condition_start_date) AS condition_start_date,
        ANY_VALUE(t1.condition_end_date) AS condition_end_date,
        FORMAT('Condition Name: %s\\nAncestor Name: %s',
               ANY_VALUE(t2.concept_name),
               ANY_VALUE(t4.concept_name)) AS concept_details,
        CAST(NULL AS STRING) AS generated_report,
        CAST(NULL AS ARRAY<FLOAT64>) AS report_embeddings,
        ROW_NUMBER() OVER(PARTITION BY t1.person_id ORDER BY ANY_VALUE(t1.condition_start_date)) AS row_num
    FROM
        `bigquery-public-data.cms_synthetic_patient_data_omop.condition_occurrence` AS t1
    JOIN
        (SELECT DISTINCT person_id FROM `bigquery-public-data.cms_synthetic_patient_data_omop.person` ORDER BY person_id LIMIT {PATIENTS_TO_PROCESS}) AS limited_persons
        ON t1.person_id = limited_persons.person_id
    JOIN
        `bigquery-public-data.cms_synthetic_patient_data_omop.concept` AS t2
        ON t1.condition_concept_id = t2.concept_id
    JOIN
        `bigquery-public-data.cms_synthetic_patient_data_omop.concept_ancestor` AS t3
        ON t1.condition_concept_id = t3.descendant_concept_id
    JOIN
        `bigquery-public-data.cms_synthetic_patient_data_omop.concept` AS t4
        ON t3.ancestor_concept_id = t4.concept_id
    GROUP BY
        t1.person_id,
        t1.condition_concept_id,
        t3.ancestor_concept_id
)
SELECT
    FARM_FINGERPRINT(TO_JSON_STRING(STRUCT(person_id, condition_concept_id, ancestor_concept_id))) AS unique_id,
    person_id,
    condition_concept_id,
    ancestor_concept_id,
    condition_start_date,
    condition_end_date,
    concept_details,
    generated_report,
    report_embeddings
FROM
    RankedConditions
WHERE
    row_num <= {REPORTS_PER_PATIENT};
"""

print(f"Executing BigQuery query to create/replace table '{DATASET_TABLE}'...")
# Run the query
query_job = client.query(query)
# Wait for the query to complete
query_job.result()
print(f"Table '{DATASET_TABLE}' created/replaced successfully.")

#Querying the resulting table to show the data we are working with.
print("\nFetching first 5 rows from the new table for verification...")
query_results_sql = f"""
SELECT *
FROM `{PROJECT_ID}.{DATASET_ID}.{DATASET_TABLE}`
LIMIT 5;
"""
df_results = client.query(query_results_sql).to_dataframe()

print("First 5 rows of `{DATASET_TABLE}`:")
print(df_results)


GENERATION_MODEL_NAME = "report_generation" #Set this to what you want the ML model to be named.
GENERATION_MODEL_TYPE = "gemini-2.5-flash-lite" #What model do you want to use.


full_model_name = f"`{PROJECT_ID}.{DATASET_ID}.{GENERATION_MODEL_NAME}`"

model_generation_query = f"""
CREATE OR REPLACE MODEL {full_model_name}
REMOTE WITH CONNECTION DEFAULT
OPTIONS(ENDPOINT = '{GENERATION_MODEL_TYPE}')
"""

try:
    # Execute the query
    job = client.query(model_generation_query)
    job.result()  # Wait for the job to complete
    print("Model created successfully.")
except Exception as e:
    print(f"An error occurred: {e}")


from google.cloud import bigquery
import time

# Initialize the BigQuery client
client = bigquery.Client(project=PROJECT_ID)

#The update query to call the Gemini model for each row.
#We'll use a `WHERE` clause to process only rows where the report is still NULL.
#This prevents reprocessing the same data if the script is run again.
update_query = f"""
UPDATE `{PROJECT_ID}.{DATASET_ID}.{DATASET_TABLE}` AS t
SET
    t.generated_report = model_output.ml_generate_text_llm_result
FROM
    ML.GENERATE_TEXT(
        MODEL `{PROJECT_ID}.{DATASET_ID}.{GENERATION_MODEL_NAME }`,
        (
            SELECT
                
                t.unique_id AS unique_row_id,
                CONCAT(
                    "Generate a detailed and fictional patient progress note using the SOAP (Subjective, Objective, Assessment, Plan) format for the following patient data. ",
                    "Do not include any personal identifying information. Keep the tone informative and formal. Do not include date or time, they are noted elsewhere - only report the details of SOAP, and nothing else. ",
                    "Patient Data: ", t.concept_details,
                    "\\nSubjective Based on the patient's condition, describe the patient's chief complaint or subjective narrative of their symptoms.",
                    "\\nObjective Based on the patient's condition, provide an objective description of the findings, such as physical examination results or lab work (create fictional but plausible details).",
                    "\\nAssessment Formulate a differential diagnosis or an assessment of the patient's condition based on the subjective and objective information.",
                    "\\nPlan Outline a plan for the patient's treatment, follow-up, and further testing."
                ) AS prompt
            FROM
                `{PROJECT_ID}.{DATASET_ID}.{DATASET_TABLE}` AS t
            WHERE
                t.generated_report IS NULL
            LIMIT 100
        ),
        STRUCT(
            1024 AS max_output_tokens,
            0.5 AS temperature,
            TRUE AS flatten_json_output
        )
    ) AS model_output
WHERE
    t.unique_id = model_output.unique_row_id;
"""

print(f"Executing update query using BigQuery ML model '{GENERATION_MODEL_NAME }'...")

#Loop until all rows have been updated
while True:
    #Run the update query
    query_job = client.query(update_query)

    #Wait for the query to complete
    query_job.result()

    #Get the number of rows affected by the UPDATE statement
    rows_affected = query_job.num_dml_affected_rows

    print(f"\nUpdate completed. {rows_affected} reports generated and saved.")

    #If no rows were affected, we have processed all of them.
    if rows_affected == 0:
        print("All records have been updated. The process is complete.")
        break

    #Pause briefly between loops to avoid rate-limiting issues.
    time.sleep(1)

print("\nFetching first 5 rows with the new generated reports for verification...")

verification_query = f"""
SELECT
    person_id,
    concept_details,
    generated_report
FROM `{PROJECT_ID}.{DATASET_ID}.{DATASET_TABLE}`
WHERE generated_report IS NOT NULL
LIMIT 5;
"""

df_results = client.query(verification_query).to_dataframe()

print("First 5 rows of `concept_content` with generated reports:")
print(df_results)


EMBEDDING_MODEL_NAME = "embedding_generation" #Set this to what you want the ML model to be named.
EMBEDDING_MODEL_TYPE = "text-embedding-005" #What model do you want to use.


full_model_name = f"`{PROJECT_ID}.{DATASET_ID}.{EMBEDDING_MODEL_NAME}`"

model_generation_query = f"""
CREATE OR REPLACE MODEL {full_model_name}
REMOTE WITH CONNECTION DEFAULT
OPTIONS(ENDPOINT = '{EMBEDDING_MODEL_TYPE}')
"""

try:
    # Execute the query
    job = client.query(model_generation_query)
    job.result()  # Wait for the job to complete
    
    print("Model created successfully.")
except Exception as e:
    print(f"An error occurred: {e}")


from google.cloud import bigquery
import time

# Initialize the BigQuery client
client = bigquery.Client(project=PROJECT_ID)

merge_query = f"""
MERGE `{PROJECT_ID}.{DATASET_ID}.{DATASET_TABLE}` AS T
USING (
  WITH ReportsToEmbed AS (
    SELECT
      t.unique_id,
      t.generated_report AS content,
      ROW_NUMBER() OVER() AS row_num
    FROM
      `{PROJECT_ID}.{DATASET_ID}.{DATASET_TABLE}` AS t
    WHERE
      t.generated_report IS NOT NULL AND (t.report_embeddings IS NULL OR ARRAY_LENGTH(t.report_embeddings) = 0)
    LIMIT 100
  ),
  EmbeddingsWithId AS (
    SELECT
      ml_generate_embedding_result,
      ROW_NUMBER() OVER() AS row_num
    FROM
      ML.GENERATE_EMBEDDING(
        MODEL `{PROJECT_ID}.{DATASET_ID}.{EMBEDDING_MODEL_NAME}`,
        (SELECT content FROM ReportsToEmbed ORDER BY row_num)
      )
  )
  SELECT
    r.unique_id,
    e.ml_generate_embedding_result
  FROM
    ReportsToEmbed r
  JOIN
    EmbeddingsWithId e
  ON
    r.row_num = e.row_num
) AS S
ON T.unique_id = S.unique_id
WHEN MATCHED THEN
  UPDATE SET T.report_embeddings = S.ml_generate_embedding_result;
"""

print(f"Embedding reports using the BigQuery ML model  and updating the '{DATASET_TABLE}' table...")

# Loop until all reports have been embedded
while True:
    #Run the MERGE query
    query_job = client.query(merge_query)

    #Wait for the query to complete
    query_job.result()

    #Get the number of rows affected by the MERGE statement
    rows_affected = query_job.num_dml_affected_rows

    print(f"\nMerge completed. {rows_affected} reports embedded and saved.")

    # If no rows were affected, all reports have been processed.
    if rows_affected == 0:
        print("All records have been embedded. The process is complete.")
        break

    # Optional: Pause briefly between loops to avoid rate-limiting issues.
    time.sleep(1)

print("\nFetching first 5 rows with the new embeddings for verification...")

verification_query = f"""
SELECT
    person_id,
    generated_report,
    report_embeddings
FROM `{PROJECT_ID}.{DATASET_ID}.{DATASET_TABLE}`
WHERE report_embeddings IS NOT NULL
LIMIT 5;
"""

df_results = client.query(verification_query).to_dataframe()

print("First 5 rows of `concept_content` with generated embeddings:")
print(df_results)


query = "Lower back problems"
patient_id = 1
start_date = "2009-01-01"
end_date = "2010-01-01"


import numpy as np
import pandas as pd
from google.cloud import bigquery

TOP_K_RESULTS = 3 #How many results we want returned
MINIMAL_SIMILARITY = 0.6 #How good results have to be in order to be returned. 

def perform_similarity_search_bigquery_ml(query, patient_id, start_date, end_date):
    client = bigquery.Client(project=PROJECT_ID)

    try:
        print(f"Performing similarity search using BigQuery ML for query: '{query}'")

        # The SQL query to generate the query embedding and perform the search
        sql_query = f"""
        SELECT
          base.concept_details,
          base.generated_report,
          1 - distance AS cosine_similarity -- Convert cosine distance to similarity
        FROM
          VECTOR_SEARCH(
            (
              SELECT
                *
              FROM
                `{PROJECT_ID}.{DATASET_ID}.{DATASET_TABLE}`
              WHERE
                person_id = @patient_id
                AND condition_start_date >= @start_date
                AND condition_end_date < @end_date
            ),
            'report_embeddings', -- The column to search, as a string literal
            (
              SELECT
                ml_generate_embedding_result AS embedding
              FROM
                ML.GENERATE_EMBEDDING(
                  MODEL `{PROJECT_ID}.{DATASET_ID}.{EMBEDDING_MODEL_NAME}`,
                  (SELECT @query AS content)
                )
            ),
            query_column_to_search => 'embedding', -- Named argument for the query column
            top_k => @top_k_results, -- Named argument for top_k
            distance_type => 'COSINE' -- Named argument for distance_type
          )
        WHERE
          1 - distance >= @minimal_similarity 
        ORDER BY
          cosine_similarity DESC;
        """
        
        #Configure the query with parameters to prevent SQL injection and ensure type safety
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("query", "STRING", query),
                bigquery.ScalarQueryParameter("patient_id", "INT64", patient_id),
                bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
                bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
                bigquery.ScalarQueryParameter("top_k_results", "INT64", TOP_K_RESULTS),
                bigquery.ScalarQueryParameter("minimal_similarity", "FLOAT64", MINIMAL_SIMILARITY),

            ]
        )

        #Run the query
        query_job = client.query(sql_query, job_config=job_config)

        #Load the results directly into a DataFrame
        results_df = query_job.to_dataframe()

        if results_df.empty:
            print("No matching concepts found for the given criteria.")
            return

        # Display the results
        print(f"\n--- Top Matching Concepts for query: '{query}' ---")
        for rank, row in results_df.iterrows():
            print(f"\nRank {rank + 1}: Score = {row['cosine_similarity']:.4f}")
            print(f"  Concept: {row['concept_details']}")
            print(f"  Details: {row['generated_report']}")
            print("------------")

    except Exception as e:
        print(f"An error occurred: {e}")

perform_similarity_search_bigquery_ml(query=query, patient_id=patient_id, start_date=start_date, end_date=end_date)


MODEL_NAME = "gemini-2.5-flash-lite"
PROMPT = "Does this patient have a history with lower back problems?"
PATIENT_ID = 1


import time
import json
from google import genai
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    Tool,
)

from google.cloud import bigquery
import time

from kaggle_gcp import KaggleKernelCredentials

try:
    # Get the credentials object from the Kaggle environment
    credentials = KaggleKernelCredentials()

    # Pass the credentials object directly to the genai.Client
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="us-east1",
        credentials=credentials 
    )
    print("GenAI client initialized successfully with Kaggle credentials.")
except Exception as e:
    print(f"Failed to initialize client with credentials: {e}")
    raise SystemExit("Exiting due to authentication failure.")

def function_calling_agent():
    #Define the tool for the model using the FunctionDeclaration and Tool classes.
    get_similarity_search = FunctionDeclaration(
        name='perform_similarity_search_bigquery_ml',
        description='Performs a semantic similarity search on the medical report data to find patient reports that are similar to the question of the medical professional interacting with you.',
        
        #Function parameters are specified in JSON schema format
        parameters={
            "type": "OBJECT",
            "properties": {
                'query': {"type": "STRING", "description": 'The user\'s search query.'},
                'patient_id': {"type": "STRING", "description": 'The ID of the patient.'},
                'start_date': {"type": "STRING", "description": 'The start date of the search range in YYYY-MM-DD format. If not given, assume 1970-01-01'},
                'end_date': {"type": "STRING", "description": 'The end date of the search range in YYYY-MM-DD format. If not given, assume today'},
            },
            "required": ['query', 'patient_id', 'start_date', 'end_date'],
        },
    )

    #Defining the tool itself to use the function
    search_tool = Tool(function_declarations=[get_similarity_search])

    #Create a user prompt that would naturally trigger the function call.
    final_prompt = PROMPT + f"\nPatient ID: {PATIENT_ID}, current date: {time.strftime('%Y-%m-%d')}" #Adding our other variables to the prompt.
    print("Sending prompt to the model...")
    print(f'''As a reminder, prompt was "{PROMPT}"''')

    #Generating the content now with all of our input. Temperature lets Gemini be a bit 'creative' with its output.
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=final_prompt,
        config=GenerateContentConfig(
            tools=[search_tool],
            temperature=0.3,
        ),
    )

    #Check the response for a function call and execute it.
    if response.function_calls:
        print("\nModel response contains a function call:")
        call = response.function_calls[0]
        print(call)

        #Get the Python functio to call
        function_to_call = globals()[call.name]

        #Execute the function with the arguments provided by the model.
        function_result = function_to_call(**call.args)
    else:
        print("\nModel did not return a function call.")
        print(f"Model response: {response.text}")


function_calling_agent()


PROMPT = "Did this patient have issues with diabetes through 2009?"
PATIENT_ID = 3

function_calling_agent()


PROMPT = "Have there been dementia related symptoms since september 2009?"
PATIENT_ID = 4

function_calling_agent()

