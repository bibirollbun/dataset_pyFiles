import time, re, subprocess
import pandas as pd
from google.cloud import bigquery
from google.colab import auth, drive, files


from kaggle_secrets import UserSecretsClient
import json
import os

user_secrets = UserSecretsClient()
secret = user_secrets.get_secret("__gcloud_sdk_auth__")

with open("gcloud-key.json", "w") as f:
    f.write(secret)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcloud-key.json"


# Config
PROJECT_ID = "claimwise-ai-472323"  # @param {type:"string"}
DATASET = "claimwise_db"
REGION = "US"
BUCKET = "gs://claim_media"



# Set the project id
! gcloud config set project {PROJECT_ID}


!bq mk --connection --location=us \
    --connection_type=CLOUD_RESOURCE test_connection


SERVICE_ACCT = !bq show --format=prettyjson --connection us.test_connection | grep "serviceAccountId" | cut -d '"' -f 4
SERVICE_ACCT_EMAIL = SERVICE_ACCT[-1]
print(SERVICE_ACCT_EMAIL)


# Assign the service account to roles that allow the codes below to be executed in BigQuery, AIplatform and GCStorage.
!gcloud projects add-iam-policy-binding --format=none $PROJECT_ID --member=serviceAccount:$SERVICE_ACCT_EMAIL --role='roles/bigquery.connectionUser'
!gcloud projects add-iam-policy-binding --format=none $PROJECT_ID --member=serviceAccount:$SERVICE_ACCT_EMAIL --role='roles/aiplatform.user'
!gcloud projects add-iam-policy-binding --format=none $PROJECT_ID --member=serviceAccount:$SERVICE_ACCT_EMAIL --role='roles/storage.objectViewer'

# wait 60 seconds, give IAM updates time to propagate, otherwise, following cells will fail
time.sleep(60)


%load_ext google.colab.data_table


# drive.mount('/content/drive')


# !gsutil mb -l us-east1 gs://claim_medias


# !gsutil -m cp "/content/drive/MyDrive/BigQuery AI Team/Dataset/Image/*" gs://claim_medias/


# remove truncation for long text fields
pd.set_option("display.max_colwidth", None)


%%bigquery --project {PROJECT_ID}

CREATE OR REPLACE EXTERNAL TABLE `claimwise_db.claim_medias`
WITH CONNECTION `us.test_connection`
OPTIONS (
  object_metadata = 'SIMPLE',
  uris = ['gs://claim_medias/*']
);


# Sanity Check to confirm the table is successfully created.
%%bigquery --project {PROJECT_ID}

SELECT * FROM `claimwise_db.claim_medias` LIMIT 5;


%%bigquery --project {PROJECT_ID}

CREATE OR REPLACE MODEL `claimwise_db.text_embedding_model`
  REMOTE WITH CONNECTION `us.test_connection`
  OPTIONS (ENDPOINT = 'gemini-embedding-001');


%%bigquery --project {PROJECT_ID}

CREATE OR REPLACE MODEL `claimwise_db.mm_embedding_model`
  REMOTE WITH CONNECTION `us.test_connection`
  OPTIONS (ENDPOINT = 'multimodalembedding@001');


%%bigquery --project {PROJECT_ID}

ALTER TABLE claimwise_db.claims
ADD COLUMN text_embedding ARRAY<FLOAT64>;


# Concat all fields defined by devloper as Content, then convert it to numeric vectors called ml_generate_embedding_result
# This field is set and updated in the field text_embedding in table claims that has claim_id correspondingly.
%%bigquery --project {PROJECT_ID}

UPDATE `claimwise_db.claims` AS t
SET t.text_embedding = s.ml_generate_embedding_result
FROM (
  SELECT
    claim_id,
    ml_generate_embedding_result
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `claimwise_db.text_embedding_model`,
      (
        SELECT
          claim_id,
          CONCAT(auto_make, ' ', auto_model, ' ', auto_year, ' ', incident_type, ' ', collision_type, ' ', incident_severity) AS content
        FROM `claimwise_db.claims`
      ),
      STRUCT(TRUE AS flatten_json_output)
    )
) AS s

WHERE t.claim_id = s.claim_id;


# Sanity check 5 rows that have new field text_embedding populated.
%%bigquery --project {PROJECT_ID}

SELECT * FROM  `claimwise_db.claims` LIMIT 5;


%%bigquery --project {PROJECT_ID}

CREATE OR REPLACE TABLE `claimwise_db.claim_medias_embeddings` AS
SELECT
  uri,
  -- Extract CLxxxx from URI
  REGEXP_EXTRACT(uri, r'gs://claim_medias/(CL[0-9]+)') AS claim_id,
  size,
  content_type,
  metadata,
  ref,
  CAST(NULL AS STRING) AS description,
  CAST(NULL AS ARRAY<FLOAT64>) AS description_embedding,
  CAST(NULL AS ARRAY<FLOAT64>) AS mm_embedding
FROM `claimwise_db.claim_medias`;


# Update the field mm_embedding in table  claim_medias_embeddings
# by generating from ML.GENERATE_EMBEDDING of the media file
%%bigquery --project {PROJECT_ID}

UPDATE `claimwise_db.claim_medias_embeddings` AS t
SET t.mm_embedding = s.ml_generate_embedding_result
FROM (
  SELECT
    uri,
    ml_generate_embedding_result
  FROM ML.GENERATE_EMBEDDING(
    MODEL `claimwise_db.mm_embedding_model`,
    (
      SELECT
        uri,
        content_type,
        OBJ.GET_ACCESS_URL(
          OBJ.FETCH_METADATA(OBJ.MAKE_REF(uri, 'us.test_connection')),
          'r'
        ) AS content
      FROM `claimwise_db.claim_medias_embeddings`
    ),
    STRUCT(TRUE AS flatten_json_output)
  )
) AS s
WHERE t.uri = s.uri;


# Generate description from image

%%bigquery --project {PROJECT_ID}

UPDATE `claimwise_db.claim_medias_embeddings` AS t
SET t.description = s.generated_description
FROM (
  SELECT
    uri,
    AI.GENERATE(
      STRUCT(
        '''
          Look at this image and classify the accident in EXACTLY this format:
          "<Colour> <Type>, <collision_type>"

          Where <collision_type> must be one of:
          - Collision Incidents:
            "rollover collision", "front impact collision",
            "rear impact collision", "side impact collision", "object collision"
          - Comprehensive Incidents:
            "theft/vandalism", "fire damage", "weather damage",
            "animal strike", "falling object damage"
          - Or: "No accident"

          Rules:
          - If vehicle damage is at the front → "front impact collision"
          - If vehicle damage is at the back → "rear impact collision"
          - If damage is on the side → "side impact collision"
          - If the vehicle rolled over → "rollover collision"
          - If hit another vehicle → "multi-vehicle collision"
          - If only one vehicle hit an object (tree, pole, guardrail) → "object collision"
          - If theft, broken window, spray paint, etc. → "theft/vandalism"
          - If fire or burning visible → "fire damage"
          - If storm, hail, flooding, or other natural cause → "weather damage"
          - If caused by animal → "animal strike"
          - If hit by falling object (tree, rock, etc.) → "falling object damage"
          - If no accident is visible → "No accident"
          - Only if you can't recognize car type, leave <Type> as Car

          Examples:
          "Black Sedan, front impact collision"
          "White SUV, rear impact collision"
          "Blue Pickup, weather damage"
          "Gray Sedan, theft/vandalism"
          "Red Car, No accident"
          ''' AS prompt,
        ref AS input
      ),
      connection_id => 'us.test_connection',
      endpoint => 'gemini-2.5-flash'
    ).result AS generated_description
  FROM `claimwise_db.claim_medias_embeddings`
) AS s
WHERE t.uri = s.uri;


# Turn AI generated description to numeric embedding and populate the field description_embedding.
%%bigquery --project {PROJECT_ID}

UPDATE `claimwise_db.claim_medias_embeddings` AS t
SET t.description_embedding = s.ml_generate_embedding_result
FROM (
  SELECT
    uri,
    ml_generate_embedding_result
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `claimwise_db.text_embedding_model`,
      (
        SELECT
         uri,
         description AS content
        FROM `claimwise_db.claim_medias_embeddings`
      ),
      STRUCT(TRUE AS flatten_json_output)
    )
) AS s
WHERE t.uri = s.uri;



client = bigquery.Client(project=PROJECT_ID)
bucket_name = "gs://claim_medias/"




# claim_id Format is as followed CLXXXX
# Query latest claim_id by numeric part
query = """
SELECT MAX(CAST(SUBSTR(claim_id, 3) AS INT64)) AS last_num
FROM `claimwise_db.claims`
WHERE claim_id LIKE 'CL%'
"""
df = client.query(query).to_dataframe()

# Get current claim_id
if df["last_num"][0]:
    current_num = (df["last_num"][0])
    current_claim_id = f"CL{current_num}"
    new_num = int(current_num) + 1
else:
    current_claim_id = None
    new_num = 1001

# Generate new claim_id
claim_id = f"CL{new_num}"

# Print results
print("Current Claim ID in table:", current_claim_id)
print("New Claim ID to be inserted:", claim_id)


# uploaded = files.upload()  # Upload one or more images from local machine to colab
# bucket_name = "gs://claim_medias"  # Target GCS bucket



new_record = {
  "meta": {
    "claim_status": "Pending",
    "customer_id": "CUST654321",
    "customer_birth_year": 1985,
    "policy_id": "POL123789",
    "insured_zip": "54321",
    "auto_make": "Toyota",
    "auto_model": "Corolla",
    "auto_year": 2014,
    "incident_date": "2025-09-20",
    "incident_type": "Collision",
    "collision_type": "Front Impact",
    "incident_severity": "Moderate",
    "incident_state": "ON",
    "incident_city": "Toronto",
    "incident_location": "Bay St & King St",
    "total_claim_amount": 4800.00,
    "fraud_reported": "N"
  }
}



def q(s):
    # minimal SQL quoting for strings
    return "'" + str(s).replace("'", "''") + "'"

struct_data = new_record['meta']

fraud_value = struct_data['fraud_reported']
fraud_bool = "TRUE" if fraud_value.upper() == "Y" else "FALSE"

insert_claims_query = f"""
INSERT INTO `claimwise_db.claims`(
  claim_id, claim_filing_date, claim_status, customer_id,
  customer_birth_year, policy_id, insured_zip, auto_make,
  auto_model, auto_year, incident_date, incident_type,
  collision_type, incident_severity, incident_state,
  incident_city, incident_location, total_claim_amount, fraud_reported
)
VALUES
(
  {q(claim_id)},
  CAST(CURRENT_DATE() AS STRING),
  {q(struct_data['claim_status'])},
  {q(struct_data['customer_id'])},
  {int(struct_data['customer_birth_year'])},
  {q(struct_data['policy_id'])},
  {int(struct_data['insured_zip'])},
  {q(struct_data['auto_make'])},
  {q(struct_data['auto_model'])},
  {int(struct_data['auto_year'])},
  {q(struct_data['incident_date'])},
  {q(struct_data['incident_type'])},
  {q(struct_data['collision_type'])},
  {q(struct_data['incident_severity'])},
  {q(struct_data['incident_state'])},
  {q(struct_data['incident_city'])},
  {q(struct_data['incident_location'])},
  {int(struct_data['total_claim_amount'])},
  {fraud_bool}
)
"""

try:
    job = client.query(insert_claims_query)
    job.result()  # wait for completion
except Exception as e:
    print(f"Error: {e}")
else:
    print("Successfully inserted row.")



# Create a text_embedding field
struct_data = new_record['meta']

# assumes struct_data = new_record['meta'] and claim_id defined
update_embedding_query = f"""
UPDATE `claimwise_db.claims` AS c
SET text_embedding = (
  SELECT ml_generate_embedding_result
  FROM ML.GENERATE_EMBEDDING(
    MODEL `claimwise_db.text_embedding_model`,
    (
      SELECT CONCAT(
        {q(struct_data["auto_make"])}, ' ',
        {q(struct_data["auto_model"])}, ' ',
        {q(struct_data["auto_year"])}, ' ',
        {q(struct_data["incident_type"])}, ' ',
        {q(struct_data["collision_type"])}, ' ',
        {q(struct_data["incident_severity"])}
      ) AS content
    ),
    STRUCT(TRUE AS flatten_json_output)
  )
)
WHERE c.claim_id = {q(claim_id)};
"""

try:
    job = client.query(update_embedding_query)
    job.result()  # force wait for job completion
except Exception as e:
    print(f"Error while inserting {claim_id}: {e}")
else:
    print(f"Successfully updated text_embedding for claim {claim_id}")



# Perform a vector similarity search to find the top 10 most similar historical claims.
search_query = f"""
SELECT
  base.claim_id,
  base.auto_make,
  base.auto_model,
  base.auto_year,
  base.incident_type,
  base.collision_type,
  base.incident_severity,
  base.total_claim_amount,
  distance
FROM VECTOR_SEARCH(
    (
      SELECT *
      FROM `claimwise_db.claims`
      WHERE text_embedding IS NOT NULL
        AND ARRAY_LENGTH(text_embedding) > 0
    ),
    'text_embedding',
    (
      SELECT
        ml_generate_embedding_result,
        content AS query
      FROM ML.GENERATE_EMBEDDING(
        MODEL `claimwise_db.text_embedding_model`,
        (
          SELECT CONCAT(
            {q(struct_data["auto_make"])}, ' ',
            {q(struct_data["auto_model"])}, ' ',
            {q(struct_data["auto_year"])}, ' ',
            {q(struct_data["incident_type"])}, ' ',
            {q(struct_data["collision_type"])}, ' ',
            {q(struct_data["incident_severity"])}) AS content
        )
      )
    ),
    top_k => 10
)
WHERE base.claim_id != '{claim_id}'
ORDER BY distance ASC;
"""

# Execute the query and load results into a DataFrame
df = client.query(search_query).to_dataframe()
df.head()
df.to_csv("/kaggle/working/submission.csv", index=False)



"""
“Upload files to GCS with standardized claim_id-based names:
– If a single file is uploaded, we rename to claim_id.jpg
– If multiple files are uploaded, we rename sequentially (claim_id_1.jpg, claim_id_2.jpg, …)”.
"""

if not uploaded:
    print("No files selected for upload.")  # Early exit if nothing uploaded
else:
    try:
        if len(uploaded) == 1:
            # Single file case, we rename to claim_id.jpg
            for filename in uploaded.keys():
                new_name = f"{claim_id}.jpg"
                subprocess.run(["mv", filename, new_name], check=True)
                subprocess.run(["gsutil", "cp", new_name, bucket_name], check=True)
                print(f"Uploaded {new_name} to {bucket_name}")
        else:
            # Multiple files case, we add sequential suffixes
            for i, filename in enumerate(uploaded.keys(), start=1):
                new_name = f"{claim_id}_{i}.jpg"
                subprocess.run(["mv", filename, new_name], check=True)
                subprocess.run(["gsutil", "cp", new_name, bucket_name], check=True)
                print(f"Uploaded {new_name} to {bucket_name}")
    except Exception as e:
        print(f"Upload failed: {e}")  # Basic error logging



# Query to recreate the external table pointing to claim images in GCS
refresh_sql = """
    CREATE OR REPLACE EXTERNAL TABLE `claimwise_db.claim_medias`
    WITH CONNECTION `us.test_connection`
    OPTIONS (object_metadata = 'SIMPLE', uris = ['gs://claim_medias/*']);
"""

try:
    # Execute the DDL and block until the table is created/refreshed
    client.query(refresh_sql).result()
    print("External table refreshed for claim_medias")
except Exception as e:
    # Catch errors like invalid connection ID or bad bucket URI
    print(f"Failed to refresh external table: {e}")



# Insert new records into claim_medias_embeddings
insert_query = f"""
INSERT INTO `claimwise_db.claim_medias_embeddings`
(uri, claim_id, size, content_type, metadata, ref, description, description_embedding, mm_embedding)
SELECT
  uri,
  REGEXP_EXTRACT(uri, r'gs://claim_medias/(CL[0-9]+)') AS claim_id,
  size,
  content_type,
  metadata,
  ref,
  NULL, NULL, NULL
FROM `claimwise_db.claim_medias` m
WHERE NOT EXISTS (
  SELECT 1 FROM `claimwise_db.claim_medias_embeddings` e WHERE e.uri = m.uri
);

"""

try:
    # Run the insert query and wait for completion
    job = client.query(insert_query)
    result = job.result()
    # Report number of new rows successfully inserted
    print(f"Inserted {job.num_dml_affected_rows} new records")
except Exception as e:
    # Handle errors such as schema mismatches or invalid URIs
    print(f"Insert failed: {e}")


# Use Gemini to generate structured descriptions for claim images
# in the format "<Colour> <Type>, <collision_type>" for later embedding
image_description_query = f"""
UPDATE `claimwise_db.claim_medias_embeddings` AS t
SET t.description = s.generated_description
FROM (
    SELECT
        uri,
        AI.GENERATE(
            STRUCT(
                '''
                Look at this image and classify the accident in EXACTLY this format:
                "<Colour> <Type>, <collision_type>"

                Where <collision_type> must be one of:
                - Collision Incidents:
                  "rollover collision", "front impact collision",
                  "rear impact collision", "side impact collision", "object collision"
                - Comprehensive Incidents:
                  "theft/vandalism", "fire damage", "weather damage",
                  "animal strike", "falling object damage"
                - Or: "No accident"

                Rules:
                - If vehicle damage is at the front → "front impact collision"
                - If vehicle damage is at the back → "rear impact collision"
                - If damage is on the side → "side impact collision"
                - If the vehicle rolled over → "rollover collision"
                - If hit another vehicle → "multi-vehicle collision"
                - If only one vehicle hit an object (tree, pole, guardrail) → "object collision"
                - If theft, broken window, spray paint, etc. → "theft/vandalism"
                - If fire or burning visible → "fire damage"
                - If storm, hail, flooding, or other natural cause → "weather damage"
                - If caused by animal → "animal strike"
                - If hit by falling object (tree, rock, etc.) → "falling object damage"
                - If no accident is visible → "No accident"
                - Only if you can't recognize car type, leave <Type> as Car

                Examples:
                "Black Sedan, front impact collision"
                "White SUV, rear impact collision"
                "Blue Pickup, weather damage"
                "Gray Sedan, theft/vandalism"
                "Red Car, No accident"
                ''' AS prompt,
                ref AS input
            ),
            connection_id => 'us.test_connection',
            endpoint => 'gemini-2.0-flash-exp'
        ).result AS generated_description
    FROM `claimwise_db.claim_medias_embeddings`
    WHERE claim_id = '{claim_id}'
) AS s
WHERE t.uri = s.uri;
"""

# Run the update query; UPDATEs don’t return rows but this confirms execution
result = client.query(image_description_query).to_dataframe()
result  # Displays query response, though UPDATEs typically don’t return rows


# Update description_embedding column with new embeddings generated from AI-generated descriptions
description_embedding_update_query = f"""
UPDATE `claimwise_db.claim_medias_embeddings` AS t
SET t.description_embedding = s.ml_generate_embedding_result
FROM (
  SELECT
    uri,
    ml_generate_embedding_result
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `claimwise_db.text_embedding_model`,
      (
        SELECT
          uri,
          description AS content
        FROM `claimwise_db.claim_medias_embeddings`
        WHERE claim_id = '{claim_id}'
      ),
      STRUCT(TRUE AS flatten_json_output)
    )
) AS s
WHERE t.uri = s.uri;
"""
# Execute query in BigQuery and wait for job completion
job = client.query(description_embedding_update_query)
job.result()


# Update 'mm_embedding' in claim_medias_embeddings
# by generating multimodal embeddings directly from image content.
mm_embedding_update_query = f"""
UPDATE `claimwise_db.claim_medias_embeddings` AS t
SET t.mm_embedding = s.ml_generate_embedding_result
FROM (
  SELECT
    uri,
    ml_generate_embedding_result
  FROM ML.GENERATE_EMBEDDING(
    MODEL `claimwise_db.mm_embedding_model`,
    (
      SELECT
        uri,
        content_type,
        OBJ.GET_ACCESS_URL(
          OBJ.FETCH_METADATA(OBJ.MAKE_REF(uri, 'us.test_connection')),
          'r'
        ) AS content
      FROM `claimwise_db.claim_medias_embeddings`
      WHERE claim_id = '{claim_id}'
    ),
    STRUCT(TRUE AS flatten_json_output)
  )
) AS s
WHERE t.uri = s.uri;
"""

# We run the update query in BigQuery and wait for completion
job = client.query(mm_embedding_update_query)
job.result()


current_claim_media_query = f"""
SELECT *
FROM `claimwise_db.claim_medias_embeddings`
WHERE claim_id = '{claim_id}';
"""

job = client.query(current_claim_media_query)
df = job.result().to_dataframe()
df


# Identify duplicate and similar claim images by comparing image embeddings and text description embeddings.
# Exact duplicates are listed first, followed by the most semantically similar candidates.

query = f"""
WITH query_image AS (
  SELECT
    uri AS query_uri,
    mm_embedding AS query_embedding,
    description_embedding AS query_desc_embedding,
    ARRAY_LENGTH(mm_embedding) AS query_len
  FROM `claimwise_db.claim_medias_embeddings`
  WHERE claim_id = '{claim_id}'
),

duplicates AS (
  SELECT
    e.claim_id,
    e.uri,
    e.content_type,
    ML.DISTANCE(e.mm_embedding, q.query_embedding, 'COSINE') AS mm_similarity,
    0.0 AS desc_similarity,
    TRUE AS is_duplicate,
    JSON_VALUE(
      OBJ.GET_ACCESS_URL(OBJ.MAKE_REF(e.uri, "us.test_connection"), "r"),
      "$.access_urls.read_url"
    ) AS read_url
  FROM `claimwise_db.claim_medias_embeddings` e, query_image q
  WHERE e.uri != q.query_uri
    AND ARRAY_LENGTH(e.mm_embedding) = q.query_len
    AND ML.DISTANCE(e.mm_embedding, q.query_embedding, 'COSINE') < 1e-2
),


desc_candidates AS (
  SELECT
    e.claim_id,
    e.uri,
    e.content_type,
    ML.DISTANCE(e.mm_embedding, q.query_embedding, 'COSINE') AS mm_similarity,
    ML.DISTANCE(e.description_embedding, q.query_desc_embedding, 'COSINE') AS desc_similarity,
    FALSE AS is_duplicate,
    JSON_VALUE(
      OBJ.GET_ACCESS_URL(OBJ.MAKE_REF(e.uri, "us.test_connection"), "r"),
      "$.access_urls.read_url"
    ) AS read_url
  FROM `claimwise_db.claim_medias_embeddings` e, query_image q
  WHERE e.uri != q.query_uri
    AND ARRAY_LENGTH(e.mm_embedding) = q.query_len
    AND ML.DISTANCE(e.mm_embedding, q.query_embedding, 'COSINE') > 1e-2
  ORDER BY desc_similarity ASC
  LIMIT 20
),

combined AS (
  SELECT * FROM duplicates
  UNION ALL
  SELECT * FROM desc_candidates
)

SELECT
  claim_id,
  uri,
  content_type,
  is_duplicate,
  mm_similarity,
  desc_similarity,
  read_url
FROM combined
ORDER BY
  is_duplicate DESC,
  desc_similarity ASC,
  mm_similarity ASC
LIMIT 5;
"""
# Execute the query and fetch results into a pandas DataFrame
df_image = client.query(query).to_dataframe()
df_image.head()



import os
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

# Copy the most recently uploaded .jpg file from the GCS bucket (gs://claim_medias) to local runtime
!gsutil cp $(gsutil ls -l gs://claim_medias/*.jpg | sort -k2 | tail -n 1 | awk '{print $3}') .

# Detect downloaded image file(s) in the current directory
files = [f for f in os.listdir(".") if f.endswith(".jpg", ".jpeg", ".png")]

# Pick the most recently modified file (latest downloaded image)
latest_file = max(files, key=os.path.getmtime)

# Open and display the image using PIL and matplotlib
img = Image.open(latest_file)
plt.figure(figsize=(6,6))    # set display size
plt.imshow(img)              # render the image
plt.axis("off")              # hide axes for clean display
plt.show()


n = len(df_image)
fig, axes = plt.subplots(1, n, figsize=(4*n, 5))   # create 1 row of subplots, one for each image

if n == 1:
    axes = [axes]  # ensure axes is iterable when only 1 image

# Loop over results and subplot axes
for ax, (_, row) in zip(axes, df_image.iterrows()):
    url = str(row["read_url"]).strip('"')          # signed URL of the image
    response = requests.get(url)                   # fetch image from URL
    img = Image.open(BytesIO(response.content))    # load image into memory

    ax.imshow(img)                                 # display image
    ax.axis("off")                                 # remove axis ticks and labels
    ax.set_title(                                  # set subplot title with metadata
        f"Claim {row['claim_id']}\n"
        f"Dup={row['is_duplicate']}\n"
        f"mm={row['mm_similarity']:.2f}\n"
        f"desc={row['desc_similarity']:.2f}",
        fontsize=9
    )

plt.suptitle("Top 5 Similar / Duplicate Images", fontsize=14, weight="bold")  # overall figure title
plt.tight_layout()   # adjust spacing between subplots
plt.show()           # render the final plot



# Check if any duplicate/similar images exist with mm_similarity <= 0.06 as a threshold
threshold = 0.06
similar_images = df_image[df_image["mm_similarity"] <= threshold]

if not similar_images.empty:
    similar_claim_ids = similar_images["claim_id"].tolist()
    all_claim_ids = list(set(similar_claim_ids + [claim_id]))

    update_query = f"""
    UPDATE `claimwise_db.claims`
    SET fraud_reported = TRUE
    WHERE claim_id IN UNNEST({all_claim_ids})
    """
    client.query(update_query).result()
    print("Fraud flags updated for:", all_claim_ids)
else:
    print("No duplicate/similar images found.")


# Query fraud vs non-fraud counts
fraud_query = """
SELECT
  fraud_reported,
  COUNT(*) AS count
FROM `claimwise_db.claims`
GROUP BY fraud_reported
"""

# Run query and load results into DataFrame
fraud_df = client.query(fraud_query).to_dataframe()

# Prepare labels and values
labels = ["Fraud Reported", "No Fraud"]
values = [
    int(fraud_df.loc[fraud_df["fraud_reported"] == True, "count"].sum()),
    int(fraud_df.loc[fraud_df["fraud_reported"] == False, "count"].sum())
]

# Plot fraud vs non-fraud distribution
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmap = cm.get_cmap("Set2")              # pastel colormap
colors = [cmap(0), cmap(1)]             # select two colors

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    values,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops=dict(edgecolor="white")
)

plt.title("Fraud vs Non-Fraud Claims", fontsize=14, weight="bold")
plt.savefig("fraud_distribution.png", dpi=300, bbox_inches="tight")
plt.show()



import matplotlib.pyplot as plt
import random

# Fraud by Incident Type
fraud_by_type_query = """
SELECT
  incident_type,
  COUNTIF(fraud_reported = TRUE) AS fraud_cases,
  COUNT(*) AS total_cases,
  ROUND(COUNTIF(fraud_reported = TRUE) / COUNT(*), 2) AS fraud_rate
FROM `claimwise_db.claims`
GROUP BY incident_type
ORDER BY fraud_rate DESC
"""
fraud_by_type_df = client.query(fraud_by_type_query).to_dataframe()

# Fraud by Auto Year
fraud_by_year_query = """
SELECT
  auto_year,
  COUNTIF(fraud_reported = TRUE) AS fraud_cases,
  COUNT(*) AS total_cases,
  ROUND(COUNTIF(fraud_reported = TRUE) / COUNT(*), 2) AS fraud_rate
FROM `claimwise_db.claims`
GROUP BY auto_year
ORDER BY auto_year
"""
fraud_by_year_df = client.query(fraud_by_year_query).to_dataframe()

# Fraud by Incident Severity
fraud_by_severity_query = """
SELECT
  incident_severity,
  COUNTIF(fraud_reported = TRUE) AS fraud_cases,
  COUNT(*) AS total_cases,
  ROUND(COUNTIF(fraud_reported = TRUE) / COUNT(*), 2) AS fraud_rate
FROM `claimwise_db.claims`
GROUP BY incident_severity
ORDER BY fraud_rate DESC
"""
fraud_by_severity_df = client.query(fraud_by_severity_query).to_dataframe()

# Colors
colors = list(plt.cm.tab20.colors)
random.shuffle(colors)

# Create Multi-Panel Figure
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Helper function for annotations
def annotate_bars(ax, rects, horizontal=False, y_offset=0.005, x_offset=0.01, percentage=False):
    for rect in rects:
        if horizontal:
            width = rect.get_width()
            ax.text(width + x_offset, rect.get_y() + rect.get_height()/2,
                    f"{width:.2f}", va='center', fontsize=9)
        else:
            height = rect.get_height()
            label = f"{height*100:.0f}%" if percentage else f"{height:.2f}"
            ax.text(rect.get_x() + rect.get_width()/2, height + y_offset,
                    label, ha='center', va='bottom', fontsize=9, color="black")

# Fraud by Incident Type
bars = axes[0].barh(
    fraud_by_type_df["incident_type"],
    fraud_by_type_df["fraud_rate"],
    color=colors[:len(fraud_by_type_df)]
)
axes[0].set_title("Fraud Rate by Incident Type", fontsize=13, weight="bold")
axes[0].set_xlabel("Fraud Rate")
axes[0].invert_yaxis()
annotate_bars(axes[0], bars, horizontal=True)

# Fraud by Auto Year (with ALL years, including 0 fraud rate)
bars = axes[1].bar(
    fraud_by_year_df["auto_year"].astype(str),
    fraud_by_year_df["fraud_rate"],
    color=colors[:len(fraud_by_year_df)]
)
axes[1].set_title("Fraud Rate by Auto Year", fontsize=13, weight="bold")
axes[1].set_xlabel("Auto Year")
axes[1].set_ylabel("Fraud Rate")
axes[1].tick_params(axis='x', rotation=45)
annotate_bars(axes[1], bars, percentage=True, y_offset=0.01)  # percentage labels

# Fraud by Incident Severity
bars = axes[2].bar(
    fraud_by_severity_df["incident_severity"],
    fraud_by_severity_df["fraud_rate"],
    color=colors[:len(fraud_by_severity_df)]
)
axes[2].set_title("Fraud Rate by Severity", fontsize=13, weight="bold")
axes[2].set_xlabel("Incident Severity")
axes[2].set_ylabel("Fraud Rate")
axes[2].tick_params(axis='x', rotation=30)
annotate_bars(axes[2], bars, y_offset=0.005)

plt.suptitle("Fraud Rate Analysis by Type, Year, and Severity", fontsize=16, weight="bold")
plt.tight_layout()
plt.show()



# Plot similarity scores for each candidate image
plt.figure(figsize=(6, 4))

# Horizontal bars for multimodal (image) similarity
plt.barh(
    df_image["uri"],
    df_image["mm_similarity"],
    color="#1f77b4",
    label="Image (mm)"
)

# Horizontal bars for description (text) similarity, semi-transparent overlay
plt.barh(
    df_image["uri"],
    df_image["desc_similarity"],
    color="#ff7f0e",
    alpha=0.7,
    label="Description (text)"
)

# Axis labels and title
plt.xlabel("Cosine Distance (smaller = more similar)")
plt.ylabel("Claim Image")
plt.title("Similarity Scores for Top Matches")

# Add legend and flip y-axis so best matches appear at top
plt.legend()
plt.gca().invert_yaxis()

# Show the plot
plt.show()


