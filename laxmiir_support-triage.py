#Installation
!pip install datasets faker google-cloud-bigquery-storage


# Imports
from google.cloud import bigquery
import pandas as pd
from datasets import load_dataset
from faker import Faker
import random
from google.api_core.exceptions import NotFound
import os


# 1.  Load dataset from Hugging Face
dataset = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
df = dataset.to_pandas()

# 2. Keep relevant fields
df = df.rename(columns={"answer": "resolution", "queue":"category"})
df = df[["subject", "body", "resolution", "type", "category", "priority"]]

#3. Clean dataset

#Drop duplicates
df= df.drop_duplicates(subset=['subject', 'body'], keep='first')
## Fill null subject
df.loc[:, 'subject'] = df['subject'].fillna('')
df.loc[:, 'body'] = df['body'].fillna('')

#4. Simulate Data

faker = Faker()
Faker.seed(42)

#Add ticket ID
df.loc[:, "ticket_id"] = range(1, len(df)+1)
#Add body + subject field
df.loc[:,"ticket_text"] =  (
    df["subject"].fillna('') + " " + df["body"].fillna('')
)
#Simulate Created date, Resolved date and Status
created_dates = []
resolved_dates = []
statuses = []

for _, row in df.iterrows():
    # Determine status
    if pd.isnull(row["resolution"]) or row["resolution"].strip() == "":
        # Open ticket
        status = "open"
        created = faker.date_time_between(start_date="-7d", end_date="now")
        resolved = None
    else:
        # Closed ticket
        status = "closed"
        created = faker.date_time_between(start_date="-2y", end_date="-7d")
        
        # Resolution time based on priority
        row_priority = row["priority"].strip()
        
        # Set resolution delay (in minutes)
        if row_priority == "high":
            delay = random.randint(1, 6*60)            # 1 min to 6 hours
        elif row_priority == "medium":
            delay = random.randint(6*60, 24*60)       # 6h to 1 day
        else:  # Low
            delay = random.randint(24*60, 7*24*60)    # 1 day to 7 days
        
        resolved = created + pd.to_timedelta(delay, unit="m")
    
    created_dates.append(created)
    resolved_dates.append(resolved)
    statuses.append(status)

df.loc[:, "created_at"] = created_dates
df.loc[:, "resolved_at"] = resolved_dates
df.loc[:, "status"] = statuses

# 7. Reorder columns
df = df[["ticket_id", "created_at", "resolved_at", "status",
         "subject", "body", "resolution", "type", "category", "priority","ticket_text"]]

output_file="support_tickets_data.csv"
# Only save CSV if it does not exist
if not os.path.exists(output_file):
    df.to_csv(output_file, index=False)
    print(f"✅ Dataset saved: {output_file}")
else:
    print(f"⚠️ File already exists: {output_file}")

df.head(5)


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
DATASET_ID = user_secrets.get_secret("GCP_DATASET_ID")
LOCATION = user_secrets.get_secret("GCP_LOCATION")
PROJECT_ID = user_secrets.get_secret("GCP_PROJECT_ID")
CONNECTION_ID = user_secrets.get_secret("GCP_CONNECTION_ID")



#BigQuery Connection
from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID)

# Table reference
table_id = f"{PROJECT_ID}.{DATASET_ID}.raw_support_ticket_data"

try:
    # Try to get the table
    table = client.get_table(table_id)
    print(f"Table {table_id} already exists with {table.num_rows} rows. Skipping load.")
except NotFound:
    # Table does not exist, so load CSV
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True
    )

    with open("/kaggle/working/support_tickets_data.csv", "rb") as source_file:
        job = client.load_table_from_file(source_file, table_id, job_config=job_config)
    
    job.result()  # wait for the load to finish
    table = client.get_table(table_id)
    print(f"Loaded {table.num_rows} rows into {table_id}.")



# Create a Embedding remote model
model_id = f"{PROJECT_ID}.{DATASET_ID}.embedding_model"

try:
    # Check if model already exists
    model = client.get_model(model_id)
    print(f"Model {model_id} already exists. Skipping creation.")
except NotFound:
    # Creating Model
    query = f"""
    CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.embedding_model`
    REMOTE WITH CONNECTION `projects/{PROJECT_ID}/locations/{LOCATION}/connections/{CONNECTION_ID}`
    OPTIONS (ENDPOINT = 'text-multilingual-embedding-002');
    """

    job = client.query(query)
    job.result()
    print("Embedding model created ✅")

#Generate Embeddings table
table_id = f"{PROJECT_ID}.{DATASET_ID}.ticket_embeddings"
try:
    # Check if table already exists
    table = client.get_table(table_id)
    print(f"Table {table_id} already exists with {table.num_rows} rows. Skipping load.")
except NotFound:
    # Creatinh Embedding Table
    embedding_query = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.ticket_embeddings` AS
    SELECT * from `{PROJECT_ID}.{DATASET_ID}.raw_support_ticket_data` t1
    left join 
    (SELECT * FROM
     ML.GENERATE_EMBEDDING(
        MODEL `{PROJECT_ID}.{DATASET_ID}.embedding_model`,
        (SELECT ticket_text as content FROM `{PROJECT_ID}.{DATASET_ID}.raw_support_ticket_data`),
        STRUCT(TRUE AS flatten_json_output, 'SEMANTIC_SIMILARITY' as task_type)
      ) AS embedding) t2
      on t2.content = t1.ticket_text 
    """
    client.query(embedding_query).result()
    print("Embeddings table created.")

# Create vector index
query_create = f"""
    CREATE VECTOR INDEX IF NOT EXISTS support_tickets_embedding_idx
    ON `{PROJECT_ID}.{DATASET_ID}.ticket_embeddings`(ml_generate_embedding_result)
    OPTIONS(
        INDEX_TYPE = 'TREE_AH',
        DISTANCE_TYPE = 'COSINE'
    )
"""
job = client.query(query_create)
job.result()  # wait for completion
print(f"✅ Vector index 'support_tickets_embedding_idx' created on bq-kaggle-hackathon.Support_Triage_Bot.ticket_embeddings(ml_generate_embedding_result).")


#Function to finds the most similar support tickets based on subject + body text.
def find_similar_tickets(subject: str, body: str, PROJECT_ID: str, DATASET_ID: str, top_k: int = 3):
    """
    Args:
        subject (str): The subject of the new ticket
        body (str): The body/description of the new ticket
        top_k (int): Number of similar tickets to retrieve (default = 3)
    
    Returns:
        pandas.DataFrame: Similar tickets with similarity scores

    #Step 1: Generate embedding for query ticket
    #Step 2: Search in vector index
    """
    client = bigquery.Client(project=PROJECT_ID)

    # Concatenate subject and body
    ticket_text = f"{subject} {body}"

    query = f"""CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.top_n_similar_tickets` AS
   
    WITH query_embedding AS (
      SELECT ml_generate_embedding_result AS embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{PROJECT_ID}.{DATASET_ID}.embedding_model`,
        (SELECT @ticket_text AS content),
        STRUCT(TRUE AS flatten_json_output, 'SEMANTIC_SIMILARITY' as task_type)
      )
    ),
    similar_ticketS AS (
    SELECT
      s.ticket_id,
      s.subject,
      s.body,
      s.resolution,
      (1 - distance) as similarity_score
    FROM VECTOR_SEARCH(
          TABLE `{PROJECT_ID}.{DATASET_ID}.ticket_embeddings` ,
           'ml_generate_embedding_result',
           TABLE query_embedding,
           'embedding',
           top_k => 3
         ) AS vs
    JOIN `{PROJECT_ID}.{DATASET_ID}.raw_support_ticket_data` s
    ON s.ticket_id = base.ticket_id),
    new_ticket AS (
      SELECT
        0 AS ticket_id,
        @new_subject AS subject,
        @new_body AS body,
        '' AS resolution,
        1.0 AS similarity_score
    )
    SELECT * FROM similar_tickets  
    UNION ALL
    SELECT * FROM new_ticket;
    """

    # Run query with parameters
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticket_text", "STRING", ticket_text),
            bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
            bigquery.ScalarQueryParameter("new_subject", "STRING", subject),
            bigquery.ScalarQueryParameter("new_body", "STRING", body)
        ]
    )

    query_job=client.query(query, job_config=job_config)
    print(f"✅ Table `{PROJECT_ID}.{DATASET_ID}.top_n_similar_tickets` created/updated")

    result_query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.top_n_similar_tickets` where ticket_id != 0"
    results = client.query(result_query).to_dataframe()

    return results



from google.cloud import bigquery

def get_summary_and_recommendation(project_id: str, dataset_id: str,connection_id: str ):
    """
    Executes a BigQuery query to generate summary and recommended action 
    for a new support ticket using AI.GENERATE.

    Args:
        project_id (str): GCP project ID
        dataset_id (str): BigQuery dataset ID

    Returns:
        dict: Contains 'summary' and 'recommendation'
    """

    client = bigquery.Client(project=project_id)

    query = f"""
    WITH combined_resolutions AS (
        SELECT STRING_AGG(resolution, '\\n') AS combined_resolution
        FROM `{project_id}.{dataset_id}.top_n_similar_tickets`
        WHERE ticket_id != 0
    ),
    new_ticket_content AS (
        SELECT CONCAT(
            'Subject: ', COALESCE(subject, ''),
            ' Body: ', COALESCE(body, '')
            ) AS new_ticket
        FROM `{project_id}.{dataset_id}.top_n_similar_tickets`
        WHERE ticket_id = 0
    ),
    ai_input AS (
        SELECT
            combined_resolution,
            new_ticket
        FROM combined_resolutions CROSS JOIN new_ticket_content
    ),
    ai_output AS (
        SELECT 
            AI.GENERATE(
                prompt => ('Summarize the historical resolutions:', ai_input.combined_resolution),
                connection_id => 'projects/{project_id}/locations/us/connections/kaggle_connection',
                output_schema => 'summary STRING'
            ).summary AS ai_summary,
            AI.GENERATE(
                prompt => ('Based on the historical resolutions:', ai_input.combined_resolution,
                          'Generate recommended next actions for the new ticket:', ai_input.new_ticket),
                connection_id => 'projects/{project_id}/locations/us/connections/{connection_id}',
                output_schema => 'recommended_action STRING'
            ).recommended_action AS ai_recommendation
        FROM ai_input
    )
    SELECT ai_summary, ai_recommendation FROM ai_output
    """

    query_job = client.query(query)
    result = query_job.result()

    # Extract results
    for row in result:
        return {
            "summary": row["ai_summary"],
            "recommendation": row["ai_recommendation"]
        }



new_ticket_subject = "My printer is not working"

new_ticket_body = "The printer shows a paper jam error even though there is no paper stuck"


find_similar_tickets(new_ticket_subject,new_ticket_body,PROJECT_ID,DATASET_ID )


response = get_summary_and_recommendation(PROJECT_ID,DATASET_ID,CONNECTION_ID)
print("Summary:", response["summary"])
print("Recommendation:", response["recommendation"])

