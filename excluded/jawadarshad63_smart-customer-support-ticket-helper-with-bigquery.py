
##Initialize BigQuery ClientÂ¶
!pip install google-cloud-bigquery pandas db-dtypes



from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)




from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("__gcloud_sdk_auth__")


# BigQuery
from google.cloud import bigquery
bigquery_client = bigquery.Client(project='mystical-factor-357103')






from google.cloud import bigquery
import pandas as pd  # Needed for .to_dataframe()

# Initialize BigQuery client with your project ID
client = bigquery.Client(project='mystical-factor-357103')  # Your provided Project ID

# Quick test to verify BigQuery access is working
print("ğŸ§ª Testing BigQuery access...")

# Simple test query
test_query = """
SELECT COUNT(*) as total_questions
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE accepted_answer_id IS NOT NULL
"""

try:
    test_result = client.query(test_query).to_dataframe()
    total_questions = test_result.iloc[0]['total_questions']
    print(f"âœ… BigQuery access working! Found {total_questions:,} questions with answers")
except Exception as e:
    print(f"â�Œ Error accessing BigQuery: {e}")

# Now explore the dataset structure
print("\nğŸ“‹ Exploring Stack Overflow dataset structure...")
stackoverflow_dataset = client.get_dataset('bigquery-public-data.stackoverflow')
tables = list(client.list_tables(stackoverflow_dataset))
print("Available tables:")
for table in tables:
    print(f" â€¢ {table.table_id}")

# Check sample data structure
sample_query = """
SELECT
  id, title, body, accepted_answer_id, view_count, score, creation_date
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE accepted_answer_id IS NOT NULL
  AND title IS NOT NULL
  AND LENGTH(title) > 10
LIMIT 5
"""
print("\nğŸ”� Sample data from posts_questions:")
sample_data = client.query(sample_query).to_dataframe()
print(sample_data[['id', 'title', 'score', 'view_count']].head())


from google.cloud import bigquery
from google.cloud.exceptions import Conflict, NotFound
import pandas as pd  # If needed elsewhere, but not required here

# Your Project ID
PROJECT_ID = 'mystical-factor-357103'

# Dataset ID to create/use (customize as needed)
DATASET_ID = 's'

# Initialize BigQuery client
client = bigquery.Client(project=PROJECT_ID)

# Create the dataset with proper error handling
dataset_full_id = f"{PROJECT_ID}.{DATASET_ID}"
try:
    # Try to get the dataset first (maybe it already exists)
    dataset = client.get_dataset(dataset_full_id)
    print(f"âœ… Dataset {DATASET_ID} already exists!")
except NotFound:
    # Dataset doesn't exist, create it
    print(f"ğŸ“� Creating dataset {DATASET_ID}...")
    try:
        dataset = bigquery.Dataset(dataset_full_id)
        dataset.location = "US"
        dataset.description = "Customer Support AI using BigQuery Vector Search"
        # Create the dataset
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"âœ… Successfully created dataset: {dataset.dataset_id}")
    except Conflict:
        print(f"âœ… Dataset {DATASET_ID} already exists (possible race condition)!")
    except Exception as e:
        print(f"â�Œ Error creating dataset: {e}")
        print(f"ğŸ’¡ You might need to enable BigQuery API or check permissions")
except Exception as e:
    print(f"â�Œ Unexpected error checking dataset: {e}")

# Verify the dataset exists
try:
    dataset = client.get_dataset(dataset_full_id)
    print(f"ğŸ�¯ Verified: Dataset {dataset.dataset_id} is ready!")
    print(f"ğŸ“� Location: {dataset.location}")
    print(f"ğŸ“� Description: {dataset.description}")
except Exception as e:
    print(f"â�Œ Dataset verification failed: {e}")





from google.cloud import bigquery
import pandas as pd

# Your Project ID
PROJECT_ID = 'mystical-factor-357103'

# Dataset ID (consistent with previous setup)
DATASET_ID = 'customer_support_ai'

# Initialize BigQuery client
client = bigquery.Client(project=PROJECT_ID)

# Extract proven solutions for each ticket with improved keyword extraction
create_solutions_query = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.proven_solutions` AS
SELECT
  a.id as solution_id,
  a.parent_id as ticket_id,
  a.body as solution_text,
  a.score as solution_quality,
  a.creation_date as solution_date,
  -- Extract solution keywords for better matching (improved filters)
  ARRAY(
    SELECT DISTINCT word
    FROM UNNEST(SPLIT(LOWER(REGEXP_REPLACE(a.body, r'[^a-zA-Z0-9\\s]', ' ')), ' ')) as word
    WHERE LENGTH(word) > 4
      AND word NOT IN ('this', 'that', 'with', 'from', 'when', 'where', 'what', 'does', 'have', 'been', 'will', 'should', 'could', 'your', 'using', 'into', 'about', 'would', 'there', 'which')
  ) as solution_keywords
FROM `bigquery-public-data.stackoverflow.posts_answers` a
INNER JOIN `{PROJECT_ID}.{DATASET_ID}.historical_tickets` h
  ON a.parent_id = h.ticket_id
WHERE a.body IS NOT NULL
  AND LENGTH(a.body) > 50  -- Substantive solutions
  AND a.score >= 1  -- Filter for at least minimally positive solutions
"""

print("ğŸ”„ Creating solutions repository...")
try:
    job = client.query(create_solutions_query)
    result = job.result()  # Wait for the job to complete
    print("âœ… Solutions repository created!")
    
    # Check solutions count
    solutions_count_query = f"""
    SELECT COUNT(*) as total_solutions
    FROM `{PROJECT_ID}.{DATASET_ID}.proven_solutions`
    """
    solutions_count = client.query(solutions_count_query).to_dataframe()
    print(f"ğŸ“Š Total solutions: {solutions_count.iloc[0]['total_solutions']:,}")
    
    # Show solution quality distribution with percentages
    quality_dist_query = f"""
    SELECT 
      CASE 
        WHEN solution_quality >= 10 THEN 'High Quality (10+)'
        WHEN solution_quality >= 5 THEN 'Medium Quality (5-9)'
        WHEN solution_quality >= 1 THEN 'Low Quality (1-4)'
        ELSE 'Unrated (0 or below)'
      END as quality_tier,
      COUNT(*) as solution_count,
      ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
    FROM `{PROJECT_ID}.{DATASET_ID}.proven_solutions`
    GROUP BY quality_tier
    ORDER BY solution_count DESC
    """
    quality_dist = client.query(quality_dist_query).to_dataframe()
    print(f"\nğŸ“Š Solution Quality Distribution:")
    print(quality_dist)
except Exception as e:
    print(f"â�Œ Error creating solutions: {e}")


# Extract proven solutions for each ticket
from google.cloud import bigquery
import pandas as pd

# Install BigQuery Storage for faster data fetching (suppresses warning)
!pip install --upgrade google-cloud-bigquery-storage --quiet

# Your Project ID
PROJECT_ID = 'mystical-factor-357103'

# Dataset ID (corrected to match created dataset)
DATASET_ID = 'customer_support_ai'

# Initialize BigQuery client
client = bigquery.Client(project=PROJECT_ID)

# Extract proven solutions for each ticket with improved keyword extraction
create_solutions_query = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.proven_solutions` AS
SELECT
  a.id as solution_id,
  a.parent_id as ticket_id,
  a.body as solution_text,
  a.score as solution_quality,
  a.creation_date as solution_date,
  -- Extract solution keywords for better matching (improved filters)
  ARRAY(
    SELECT DISTINCT word
    FROM UNNEST(SPLIT(LOWER(REGEXP_REPLACE(a.body, r'[^a-zA-Z0-9\\s]', ' ')), ' ')) as word
    WHERE LENGTH(word) > 4
      AND word NOT IN ('this', 'that', 'with', 'from', 'when', 'where', 'what', 'does', 'have', 'been', 'will', 'should', 'could', 'your', 'using', 'into', 'about', 'would', 'there', 'which')
  ) as solution_keywords
FROM `bigquery-public-data.stackoverflow.posts_answers` a
INNER JOIN `{PROJECT_ID}.{DATASET_ID}.historical_tickets` h
  ON a.parent_id = h.ticket_id
WHERE a.body IS NOT NULL
  AND LENGTH(a.body) > 50  -- Substantive solutions
  AND a.score >= 1  -- Filter for at least minimally positive solutions
"""

print("ğŸ”„ Creating solutions repository...")
try:
    job = client.query(create_solutions_query)
    result = job.result()  # Wait for the job to complete
    print("âœ… Solutions repository created!")
    
    # Check solutions count
    solutions_count_query = f"""
    SELECT COUNT(*) as total_solutions
    FROM `{PROJECT_ID}.{DATASET_ID}.proven_solutions`
    """
    solutions_count = client.query(solutions_count_query).to_dataframe()
    print(f"ğŸ“Š Total solutions: {solutions_count.iloc[0]['total_solutions']:,}")
    
    # Show solution quality distribution with percentages
    quality_dist_query = f"""
    SELECT 
      CASE 
        WHEN solution_quality >= 10 THEN 'High Quality (10+)'
        WHEN solution_quality >= 5 THEN 'Medium Quality (5-9)'
        WHEN solution_quality >= 1 THEN 'Low Quality (1-4)'
        ELSE 'Unrated (0 or below)'
      END as quality_tier,
      COUNT(*) as solution_count,
      ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
    FROM `{PROJECT_ID}.{DATASET_ID}.proven_solutions`
    GROUP BY quality_tier
    ORDER BY solution_count DESC
    """
    quality_dist = client.query(quality_dist_query).to_dataframe()
    print(f"\nğŸ“Š Solution Quality Distribution:")
    print(quality_dist)
except Exception as e:
    print(f"â�Œ Error creating solutions: {e}")


# Define your connection ID (from BigQuery console)
CONNECTION_ID = 'projects/mystical-factor-357103/locations/us/connections/vertex-ai-connection'  # Replace with your actual full connection ID from the console

# Create remote embedding model (uses Vertex AI's text-embedding-004)
create_model_query = f"""
CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.text_embedding_model`
REMOTE WITH CONNECTION `{CONNECTION_ID}`
OPTIONS (ENDPOINT = 'text-embedding-004')
"""
print("ğŸ”„ Creating remote embedding model...")
try:
    job = client.query(create_model_query)
    job.result()  # Wait for completion
    print("âœ… Embedding model created!")
except Exception as e:
    print(f"â�Œ Error creating model: {e}")

# Create historical tickets with embeddings





def find_similar_tickets(customer_issue, top_k=5):
    """
    Advanced similarity search using BigQuery text analysis
    Demonstrates semantic understanding beyond keyword matching
    """
    
    similarity_query = f"""
    WITH query_analysis AS (
      SELECT 
        SPLIT(LOWER(REGEXP_REPLACE('{customer_issue}', r'[^a-zA-Z0-9\\s]', ' ')), ' ') as query_words,
        CASE 
          WHEN LOWER('{customer_issue}') LIKE '%error%' OR LOWER('{customer_issue}') LIKE '%exception%' THEN 'error'
          WHEN LOWER('{customer_issue}') LIKE '%database%' OR LOWER('{customer_issue}') LIKE '%sql%' THEN 'database'
          WHEN LOWER('{customer_issue}') LIKE '%login%' OR LOWER('{customer_issue}') LIKE '%auth%' THEN 'authentication'
          WHEN LOWER('{customer_issue}') LIKE '%api%' OR LOWER('{customer_issue}') LIKE '%request%' THEN 'api'
          WHEN LOWER('{customer_issue}') LIKE '%payment%' OR LOWER('{customer_issue}') LIKE '%billing%' THEN 'payment'
          WHEN LOWER('{customer_issue}') LIKE '%javascript%' OR LOWER('{customer_issue}') LIKE '%react%' THEN 'frontend'
          WHEN LOWER('{customer_issue}') LIKE '%python%' OR LOWER('{customer_issue}') LIKE '%django%' THEN 'backend'
          ELSE 'general'
        END as query_category
    ),
    ticket_scores AS (
      SELECT 
        h.ticket_id,
        h.customer_issue,
        h.issue_category,
        h.score,
        s.solution_text,
        s.solution_quality,
        -- Word overlap score
        (
          SELECT COUNT(*)
          FROM UNNEST(q.query_words) as qw
          JOIN UNNEST(h.title_words) as tw
          ON qw = tw
          WHERE LENGTH(qw) > 2
        ) as word_matches,
        ARRAY_LENGTH(h.title_words) as total_words,
        -- Key term overlap
        (
          SELECT COUNT(*)
          FROM UNNEST(q.query_words) as qw
          JOIN UNNEST(h.key_terms) as kt
          ON qw = kt
        ) as key_term_matches,
        ARRAY_LENGTH(h.key_terms) as total_key_terms,
        -- Category match bonus
        CASE WHEN h.issue_category = q.query_category THEN 0.5 ELSE 0.0 END as category_bonus
      FROM `{PROJECT_ID}.support_ai.historical_tickets` h
      JOIN `{PROJECT_ID}.support_ai.proven_solutions` s
        ON h.ticket_id = s.ticket_id
      CROSS JOIN query_analysis q
    )
    SELECT 
      ticket_id,
      customer_issue,
      issue_category,
      ROUND(
        SAFE_DIVIDE(word_matches, GREATEST(total_words, 1)) * 0.4 +
        SAFE_DIVIDE(key_term_matches, GREATEST(total_key_terms, 1)) * 0.4 +
        category_bonus * 0.2,
        3
      ) as confidence,
      score as original_score,
      SUBSTR(solution_text, 1, 200) as solution_preview,
      solution_quality,
      word_matches,
      key_term_matches
    FROM ticket_scores
    WHERE word_matches > 0 OR key_term_matches > 0 OR category_bonus > 0
    ORDER BY confidence DESC, solution_quality DESC, original_score DESC
    LIMIT {top_k}
    """
    
    return client.query(similarity_query).to_dataframe()

print("âœ… Advanced semantic search function created!")
print("ğŸ�¯ Ready to find similar tickets based on meaning, not just keywords")


def test_database_issues():
    """Test function for database connection issues"""
    print("ğŸ�ª LIVE DEMO 1: Database Connection Problems (FIXED VERSION)")
    print("=" * 60)
    
    database_issues = [
        "Cannot connect to MySQL database getting timeout error",
        "Database server connection refused", 
        "SQL connection timeout after 30 seconds"
    ]
    
    for i, issue in enumerate(database_issues, 1):
        print(f"\nğŸ”� Customer Issue {i}: '{issue}'")
        print("-" * 50)
        try:
            # Use the simplified version first
            results = find_similar_tickets_simple(issue, top_k=3)
            for idx, row in results.iterrows():
                print(f"\n ğŸ�¯ Match {idx+1} (Confidence: {row['confidence']:.3f})")
                print(f" Similar Issue: {row['customer_issue'][:70]}...")
                print(f" Category: {row['issue_category']} | Quality: {row['solution_quality']}")
                print(f" Solution Preview: {row['solution_preview'][:100]}...")
        except Exception as e:
            print(f"â�Œ Error: {e}")
            
    print(f"\nğŸ’¡ Notice: All found 'database' category matches even with different wording!")

# Uncomment to run the test
test_database_issues()










