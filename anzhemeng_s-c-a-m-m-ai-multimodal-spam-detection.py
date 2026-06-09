import warnings
warnings.filterwarnings("ignore")
from google.cloud import storage
import os
from kaggle_secrets import UserSecretsClient
from pathlib import Path
import pandas as pd

user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)
# ⚠️ MAKE SURE TO ADD YOUR GCP CONFIG TO KAGGLE SECRETS
BUCKET_NAME = user_secrets.get_secret("BUCKET_NAME")
DATASET = user_secrets.get_secret("DATASET")
PROJECT_ID = user_secrets.get_secret("PROJECT_ID")
bucket = None
storage_client = storage.Client(project=PROJECT_ID)


def upload_blob(source_file_name, destination_blob_name):
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def create_bucket(bucket_name):
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        """Creates a new bucket if it doesn't exist. """
        bucket = storage_client.create_bucket(bucket_name)
        print('Bucket {} created'.format(bucket.name))

    return bucket

def check_folder_exists(folder):
    # Ensure the folder ends with a slash for accurate prefix matching
    if not folder.endswith('/'):
        folder += '/'

    # List objects with the given prefix. If any objects are found,
    # the "folder" is considered to exist.
    blobs = list(bucket.list_blobs(prefix=folder, max_results=1))

    return len(blobs) > 0
    
def batch_upload(folder):
    for dirname, _, filenames in os.walk(os.path.join(root_dir, folder)):
        relative_dir = str(Path(dirname).relative_to(Path(root_dir)))
        for filename in filenames:
            upload_blob( 
                os.path.join(dirname, filename), 
                f"{relative_dir}/{filename}")
                    
root_dir = '/kaggle/input'
bucket = create_bucket(BUCKET_NAME)
for folder in os.listdir(root_dir):
    # only upload if the folder does not exist
    if not check_folder_exists(folder):
        if folder == 'spam-misleading-images-dataset':
            batch_upload(folder)
        elif folder == 'sms-spam-collection-dataset':
            for dirname, _, filenames in os.walk(os.path.join(root_dir, folder)):
                relative_dir = str(Path(dirname).relative_to(Path(root_dir)))
                for filename in filenames:
                    # the raw data is messy, it needs to be cleaned before being ingested
                    sms_df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding="latin1")
                    # data cleansing
                    ## combine all content from the 2nd to 5th column
                    sms_df['message'] = None
                    sms_df['message'].loc[sms_df['Unnamed: 4'].notna()] = \
                      sms_df['v2'] + ',' + sms_df['Unnamed: 2'].astype(str) + ',' + sms_df['Unnamed: 3'].astype(str) + ',' + sms_df['Unnamed: 4'].astype(str)
                    sms_df['message'].loc[(sms_df['Unnamed: 4'].isna()) & (sms_df['Unnamed: 3'].notna())] = \
                      sms_df['v2'] + ',' + sms_df['Unnamed: 2'].astype(str) + ',' + sms_df['Unnamed: 3'].astype(str)
                    sms_df['message'].loc[(sms_df['Unnamed: 3'].isna()) & (sms_df['Unnamed: 2'].notna())] = \
                      sms_df['v2'] + ',' + sms_df['Unnamed: 2'].astype(str)
                    sms_df['message'].loc[sms_df['Unnamed: 2'].isna()] = sms_df['v2']
                    
                    ## rename the column to make it meaningful
                    sms_df.rename(columns={'v1': 'label'}, inplace=True)
    
                    ## remove unwanted characters
                    sms_df['message'] = sms_df['message'].str.replace('\r', ' ')
                    
                    ## take the necessary columns
                    output_file = f"/kaggle/working/{filename}"
                    sms_df[['message', 'label']].to_csv(output_file, index=False)
    
                    # upload
                    destination = f"{relative_dir}/{filename}"
                    upload_blob(output_file, destination)


# initiate a BQ instance
from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)


CREATE_TABLE_DDL = f'''
    CREATE OR REPLACE EXTERNAL TABLE `{PROJECT_ID}.{DATASET}.spam_messages` (
      message STRING,
      label STRING
    )
    OPTIONS (
      format = 'CSV',
      skip_leading_rows = 1,
      uris = ['gs://{BUCKET_NAME}/sms-spam-collection-dataset/spam.csv'],
      ignore_unknown_values = TRUE
    )
'''
try:
    bigquery_client.query(CREATE_TABLE_DDL).result()
    print("Spam Text Table is successfully created.")
except Exception as e:
    print(e)


RAW_IMAGE_DDL = f'''
    CREATE OR REPLACE EXTERNAL TABLE `{PROJECT_ID}.{DATASET}.images`
    WITH CONNECTION DEFAULT
    OPTIONS(
      object_metadata = 'SIMPLE',
      uris = [
        'gs://{BUCKET_NAME}/spam-misleading-images-dataset/test-images/*',
        'gs://{BUCKET_NAME}/spam-misleading-images-dataset/train-Images/*',
        'gs://{BUCKET_NAME}/spam-misleading-images-dataset/validation-images/*'
        ]
    )
'''

try:
    bigquery_client.query(CREATE_TABLE_DDL).result()
    print("Spam Image Table is successfully created.")
except Exception as e:
    print(e)


PREDICTING_LABEL_DQL = f'''
      SELECT message,
      AI.GENERATE(
        ('Is the message spam or ham? Please only answer "spam" or "ham" in ONE word.', message),
        connection_id => 'projects/{PROJECT_ID}/locations/us/connections/__default_cloudresource_connection__',
        endpoint => 'gemini-2.5-flash'
      ).result AS predicted_label
      FROM `{PROJECT_ID}.{DATASET}.generated_spam_messages`
'''

try:
    predicted_label_df = bigquery_client.query(PREDICTING_LABEL_DQL).to_dataframe()
except Exception as e:
    print(e)


predicted_label_df.head()


ACCURACY_DQL = f'''
  SELECT 1.0 * SUM(
  CASE WHEN LOWER(TRIM(predicted_label)) = LOWER(TRIM(actual_label)) THEN 1 ELSE 0 END) / COUNT(actual_label) 
  AS accuracy
FROM (
  SELECT label as actual_label,
  AI.GENERATE(
    ('Is the message spam or ham? Please only answer "spam" or "ham" in ONE word.', message),
    connection_id => 'projects/{PROJECT_ID}/locations/us/connections/__default_cloudresource_connection__',
    endpoint => 'gemini-2.5-flash'
  ).result AS predicted_label
  FROM `{PROJECT_ID}.{DATASET}.generated_spam_messages`
) AS tmp
'''

try:
    result_df = bigquery_client.query(ACCURACY_DQL).to_dataframe()
    print(f"The AI model labeling accuracy is {(100 * result_df['accuracy'].iloc[0]):.2f}%.")
except Exception as e:
    print(e)


# MODEL_NAME = 'my_gemini'
# CREATE_MODEL_DDL = f'''
# CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET}.{MODEL_NAME}`
# REMOTE WITH CONNECTION DEFAULT
# OPTIONS(
#   ENDPOINT = 'gemini-2.0-flash-001',
#   max_iterations = 500,
#   prompt_col = 'prompt',
#   input_label_cols = ['label'])
# AS
# SELECT
#   CONCAT(
#     'Is the text spam or ham? Answer me in one all-lowercase word. \\nText: \\n', message) AS prompt,
#   label
# FROM `{PROJECT_ID}.{DATASET}.spam_messages`
# '''

# try:
#     bigquery_client.query(CREATE_MODEL_DDL).result()
#     print("Fine-tuned LLM for spam classification is successfully trained and built.")
# except Exception as e:
#     print(e)


# CREATE_TEST_SET_DDL = f'''
#     CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.generated_spam_messages` AS
#     SELECT *
#     FROM (
#       SELECT
#         AI.GENERATE(
#           ('Generate one and ONLY one SMS text message someone might receive. Make sure it is ham(legitimate). It doesn\'t have to be an ice breaker, instead, something that usually only appear in the middle of a conversation works. Just output the message for me. Don\'t put any other text before or after the message, for example, "Here are a few distinct examples:". Make sure the message is eligible to English speakers, not truncated and distinct. You may add some emotional elements to the tone, perhaps emojis would help.'),
#           connection_id => 'projects/{PROJECT_ID}/locations/us/connections/__default_cloudresource_connection__',
#           endpoint => 'gemini-2.0-flash',
#           output_schema => 'message STRING, label STRING').message,
#         'ham' AS label
#       FROM (
#         SELECT
#           NULL AS dummy
#         FROM
#           UNNEST(GENERATE_ARRAY(1, 1000))
#       )
#       UNION ALL
#       SELECT
#         AI.GENERATE(
#           ('Generate one and ONLY one SMS spam message someone might receive. Just output the message for me. Don\'t put any other text before or after the message, for example, "Here are a few distinct examples:". '),
#           connection_id => 'projects/{PROJECT_ID}/locations/us/connections/__default_cloudresource_connection__',
#           endpoint => 'gemini-2.0-flash',
#           output_schema => 'message STRING, label STRING').message,
#         'spam' AS label
#       FROM (
#         SELECT
#           NULL AS dummy
#         FROM
#           UNNEST(GENERATE_ARRAY(1, 1000))
#       )
#      ) gen_messages
#     ORDER BY RAND()
# '''

# try:
#     bigquery_client.query(CREATE_MODEL_DDL).result()
#     test_set = bigquery_client.query(f'SELECT * FROM `{PROJECT_ID}.{DATASET}.generated_spam_messages` LIMIT 10').to_dataframe()
# except Exception as e:
#     print(e)


# ACCURACY_DQL = f'''
#   SELECT 1.0 * SUM(
#   CASE WHEN LOWER(TRIM(predicted_label)) = LOWER(TRIM(actual_label)) THEN 1 ELSE 0 END) / COUNT(actual_label) 
#   AS accuracy
# FROM (
#   SELECT label as actual_label,
#   AI.GENERATE(
#     ('Is the message spam or ham? Please only answer "spam" or "ham" in ONE word.', message),
#     connection_id => 'projects/{PROJECT_ID}/locations/us/connections/__default_cloudresource_connection__',
#     endpoint => 'gemini-2.0-flash'
#   ).result AS predicted_label
#   FROM `{PROJECT_ID}.{DATASET}.generated_spam_messages`
# ) AS tmp
# '''

# try:
#     result_df = bigquery_client.query(ACCURACY_DQL).to_dataframe()
#     print(f"The AI model labeling accuracy is {(100 * result_df['accuracy'].iloc[0]):.2f}%.")
# except Exception as e:
#     print(e)


# # introduce a text embedding model from Vertex AI
# EMBEDDER_DDL = f'''
#     CREATE MODEL IF NOT EXISTS`{PROJECT_ID}.{DATASET}.embedder`
#     REMOTE WITH CONNECTION DEFAULT
#     OPTIONS(
#       ENDPOINT = 'text-embedding-005'
#     )
# '''
# try:
#     bigquery_client.query(EMBEDDER_DDL).result()
#     print("Text Embedding Model is imported.")
# except Exception as e:
#     print(e)

# # Use it to embed our corpus
# EMBED_TRAIN_SET_DDL = f'''
#     CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.embedded_train_set` AS
#     SELECT ml_generate_embedding_result as embedding,
#     label
#     FROM
#     ML.GENERATE_EMBEDDING(
#       MODEL `{PROJECT_ID}.{DATASET}.embedder`,
#       (SELECT message AS content, label FROM `{PROJECT_ID}.{DATASET}.spam_messages`),
#       STRUCT('RETRIEVAL_DOCUMENT' as task_type)
#     )
# '''
# EMBED_TEST_SET_DDL = f'''
#     CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.embedded_test_set` AS
#     SELECT ml_generate_embedding_result as embedding,
#     label
#     FROM
#     ML.GENERATE_EMBEDDING(
#       MODEL `{PROJECT_ID}.{DATASET}.embedder`,
#       (SELECT message AS content, label FROM `{PROJECT_ID}.{DATASET}.generated_spam_messages`),
#       STRUCT('RETRIEVAL_DOCUMENT' as task_type)
#     )
# '''

# try:
#     bigquery_client.query(EMBED_TRAIN_SET_DDL).result()
#     bigquery_client.query(EMBED_TEST_SET_DDL).result()
#     embedded_train_set = bigquery_client.query(f'SELECT * FROM `{PROJECT_ID}.{DATASET}.embedded_train_set` LIMIT 10').to_dataframe()
#     embedded_test_set = bigquery_client.query(f'SELECT * FROM `{PROJECT_ID}.{DATASET}.embedded_test_set` LIMIT 10').to_dataframe()
# except Exception as e:
#     print(e)


# CLASSIFIER_MODEL_DDL = f'''
#     CREATE OR REPLACE MODEL
#       `{PROJECT_ID}.{DATASET}.xgboost_spam_classifier`
#     OPTIONS (
#         MODEL_TYPE = 'BOOSTED_TREE_CLASSIFIER',
#         BOOSTER_TYPE = 'GBTREE'
#     ) AS
#     SELECT
#       embedding,
#       label
#     FROM
#       `{PROJECT_ID}.{DATASET}.embedded_train_set`
# '''

# try:
#     bigquery_client.query(CLASSIFIER_MODEL_DDL).result()
#     print("GBT Classifier is trained!")
# except Exception as e:
#     print(e)


# EVAL_DQL = f'''
#     SELECT accuracy
#     FROM ML.EVALUATE(
#       MODEL `{PROJECT_ID}.{DATASET}.xgboost_spam_classifier`, 
#       TABLE `{PROJECT_ID}.{DATASET}.embedded_test_set`
#     )
# '''

# try:
#     eval_accuracy = bigquery_client.query(EVAL_DQL).to_dataframe()
#     print(f"The accuracy on the test set is {(100 * eval_accuracy['accuracy'].iloc[0]):.2f}%.")
# except Exception as e:
#     print(e)


IMAGE_ACCURACY_DQL = f'''
  SELECT 1.0 * SUM(
  CASE WHEN LOWER(TRIM(predicted_label)) = LOWER(TRIM(actual_label)) THEN 1 ELSE 0 END) / COUNT(actual_label) 
  AS accuracy
FROM (
  SELECT
  SPLIT(i.uri, '/')[OFFSET(6)] as actual_label,
  AI.GENERATE(
    ('Is the image spam or ham? Please only answer "spam" or "ham" in ONE word.', i.ref),
    connection_id => 'projects/{PROJECT_ID}/locations/us/connections/__default_cloudresource_connection__',
    endpoint => 'gemini-2.5-flash',
    output_schema => 'predicted_label STRING').predicted_label
  FROM `{PROJECT_ID}.{DATASET}.images` i
  WHERE SPLIT(i.uri, '/')[OFFSET(5)] = 'train'
)
'''

try:
    image_result_df = bigquery_client.query(IMAGE_ACCURACY_DQL).to_dataframe()
    print(f"The AI model labeling accuracy is {(100 * image_result_df['accuracy'].iloc[0]):.2f}%.")
except Exception as e:
    print(e)

