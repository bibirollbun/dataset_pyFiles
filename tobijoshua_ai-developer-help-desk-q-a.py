                                    # --- Parameters ---
PROJECT_ID   = "ai-dev-helpdesk"     # GCP project ID
LOCATION     = "US"                  # BigQuery location for dataset & jobs
DATASET      = "stackoverflow_demo"  # Dataset to create/use
CONNECTION_ID = "projects/ai-dev-helpdesk/locations/us/connections/vertex-conn-us-1"  

                                # --- BigQuery connection to Vertex AI --- 

# LLM + Embedding endpoints (BigQuery AI over Vertex)
GENERATION_ENDPOINT = "gemini-2.0-flash"        # For AI.GENERATE()
EMBEDDING_MODEL     = "gemini-embedding-001" # For ML.GENERATE_EMBEDDING()

# Limits to control cost during dev
SAMPLE_QUESTIONS_LIMIT = 10_000
EMBEDDINGS_LIMIT       = 1_000     # For quick demos
TSNE_SAMPLE_LIMIT      = 500       # t-SNE is O(n^2) (small for speed)

                               # --- Imports ---
import warnings
import re
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud
from google.cloud import bigquery
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ipywidgets as widgets
from google.api_core.exceptions import NotFound
from IPython.display import display, Markdown, HTML
from google.oauth2 import service_account
warnings.filterwarnings("ignore")
KEY_PATH = "/kaggle/input/service-key/ai-dev-helpdesk-00a61a917691.json"

# Create credentials
creds = service_account.Credentials.from_service_account_file(KEY_PATH)

# Initialize BigQuery client with credentials
client = bigquery.Client(credentials=creds, project=PROJECT_ID)

print("Authenticated as:", creds.service_account_email)
print("BigQuery client initialized:", client)



# Create dataset in case it doesn't exist (idempotent)
dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET}")
dataset_ref.location = LOCATION

try:
    client.get_dataset(dataset_ref)
except Exception:
    client.create_dataset(dataset_ref)
    print(f"Created dataset {PROJECT_ID}.{DATASET}")
else:
    print(f"Using dataset {PROJECT_ID}.{DATASET}")



# Create the sample table only if it doesn't exist
table_id = f"{PROJECT_ID}.{DATASET}.sample_stackoverflow_posts"

try:
    # We See if the treasure chest already exists
    client.get_table(table_id)
    print(f"Using existing table: {table_id}")
except NotFound:
    print(f"Table not found. Creating {table_id}...")
    create_sample_table_sql = f"""
    CREATE TABLE `{table_id}` AS
    SELECT
      id,
      title,
      body,
      tags,
      creation_date
    FROM
      `bigquery-public-data.stackoverflow.posts_questions`
    WHERE body IS NOT NULL
    LIMIT {SAMPLE_QUESTIONS_LIMIT};
    """
    client.query(create_sample_table_sql).result()
    print("Sample Stack Overflow questions table created.")


# We First Look at Tags & Body Lengths
import warnings

# Suppress BigQuery Storage API warning
warnings.filterwarnings("ignore", message=".*BigQuery Storage module not found.*")

query = f"""
SELECT tags, LENGTH(body) AS body_length
FROM `{PROJECT_ID}.{DATASET}.sample_stackoverflow_posts`
"""
df_explore = client.query(query).to_dataframe()
print(df_explore.shape)
print(df_explore.head)


plt.figure(figsize=(10,6))
sns.histplot(
    df_explore["body_length"],
    bins=50,
    kde=False,
    color="skyblue"
)

plt.title("Distribution of Question Body Lengths", fontsize=16)
plt.xlabel("Body Length (characters)", fontsize=12)
plt.ylabel("Number of Questions", fontsize=12)
plt.grid(axis='y', alpha=0.75)

plt.show()


df_explore['tags'].dropna().head(10).tolist()


# --- Function to parse pipe-separated tags ---
def parse_pipe_tags(tag_str):
    """Parse tags separated by | and return as a list."""
    if not isinstance(tag_str, str):
        return []
    return [t.strip() for t in tag_str.split('|') if t.strip()]

# --- Extract and flatten tags ---
all_tags = df_explore['tags'].dropna().tolist()
tags_flat = []
for t in all_tags:
    tags_flat.extend(parse_pipe_tags(t)) 

# --- Check if any tags exist ---
if len(tags_flat) == 0:
    display(Markdown("> **Note:** No tags parsed; check input format."))
else:
    # Count tag frequencies
    tag_freq = pd.Series(tags_flat).value_counts()

    # Generate WordCloud
    wordcloud = WordCloud(
        width=1000, 
        height=500, 
        background_color="white",
        colormap="viridis",  # nicer color palette
        max_words=200
    ).generate_from_frequencies(tag_freq.to_dict())

    # Display with Matplotlib
    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Stack Overflow Tags", fontsize=16)
    plt.tight_layout()
    plt.show()
    wordcloud.to_file("tags_wordcloud.png")


# Create summaries table in BigQuery using AI.GENERATE
create_summaries_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.question_summaries` AS
SELECT
  id,
  title,
  AI.GENERATE(
    ('Summarize the following Stack Overflow question in one short sentence: ', body),
    connection_id => '{CONNECTION_ID}',
    endpoint => '{GENERATION_ENDPOINT}'
  ).result AS summary
FROM `{PROJECT_ID}.{DATASET}.sample_stackoverflow_posts`
LIMIT 20  -- start small for quick demo; increase after testing
"""
client.query(create_summaries_sql).result()

print("âœ… Summaries table created in BigQuery.")

# Fetch the results into Python
df_summaries = client.query(f"""
    SELECT id, title, summary
    FROM `{PROJECT_ID}.{DATASET}.question_summaries`
    ORDER BY id
""").to_dataframe()

# Replace newlines for better display
df_summaries['summary'] = df_summaries['summary'].str.replace(r'\n', '<br>', regex=True)

# Style functions
def style_title(text):
    return f"<div style='padding:8px; border-radius:5px; background-color:#cce5ff; color:#000; border:1px solid #99ccff; font-weight:bold'>{text}</div>"

def style_summary(text):
    return f"<div style='padding:8px; border-radius:5px; background-color:#d4edda; color:#000; border:1px solid #a3d9a5; white-space: pre-wrap; overflow: hidden; text-overflow: ellipsis; max-height:120px;'>{text}</div>"

# Slice first 15 rows
df_slice = df_summaries[['title', 'summary']].head(15).copy()

# Apply styles
df_slice['title'] = df_slice['title'].apply(style_title)
df_slice['summary'] = df_slice['summary'].apply(style_summary)

# Render HTML table body only (no default header)
html_body = df_slice.to_html(escape=False, index=False, header=False)

# Wrap in scrollable container with custom header and portable CSS
html_table = f"""
<style>
.scrollable-table {{
    max-height:400px;
    overflow:auto;
    border:1px solid #ccc;
    border-radius:5px;
    font-family: sans-serif;
}}
.scrollable-table table {{
    border-collapse: collapse;
    width: 100%;
}}
.scrollable-table th {{
    background-color: #cce5ff;  
    color: #000;                 
    padding: 10px;
    border-bottom: 3px solid #99ccff;
    text-align: left;
}}
.scrollable-table td {{
    padding: 5px;
    vertical-align: top;
}}
</style>

<div class="scrollable-table">
    <table>
        <thead>
            <tr>
                <th>Developer Questions</th>
                <th>AI-Powered Response</th>
            </tr>
        </thead>
        <tbody>
            {html_body.replace('<table border="1" class="dataframe">', '').replace('</table>', '')}
        </tbody>
    </table>
</div>
"""

display(HTML(html_table))


# Create the remote embedding model in BigQuery
create_model_sql = f"""
CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET}.{EMBEDDING_MODEL}`
REMOTE WITH CONNECTION `{CONNECTION_ID}`
OPTIONS(
    ENDPOINT = 'text-embedding-004'  -- or 'gemini-text-embedding-001' if available
);
"""
client.query(create_model_sql).result()
print(f"âœ… Remote embedding model created: {PROJECT_ID}.{DATASET}.{EMBEDDING_MODEL}")


create_embeddings_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.question_embeddings` AS
SELECT
  CAST(id AS STRING) AS id,
  content AS summary,
  ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `{PROJECT_ID}.{DATASET}.{EMBEDDING_MODEL}`,
  (SELECT summary AS content, id FROM `{PROJECT_ID}.{DATASET}.question_summaries`),
  STRUCT(TRUE AS flatten_json_output, 'RETRIEVAL_DOCUMENT' AS task_type)
)
LIMIT {EMBEDDINGS_LIMIT};
"""


client.query(create_embeddings_sql).result()
print("âœ… Embeddings table created from summaries.")


df_embeddings = client.query(f"""
SELECT id, summary, embedding
FROM `{PROJECT_ID}.{DATASET}.question_embeddings`
LIMIT 10
""").to_dataframe()


df_slice = df_embeddings.head(10).copy()

# Wrap in scrollable container with custom CSS
html_table = f"""
<style>
.scrollable-table {{
    max-height:400px;
    overflow:auto;
    border:1px solid #ccc;
    border-radius:5px;
    font-family: sans-serif;
}}
.scrollable-table table {{
    border-collapse: collapse;
    width: 100%;
}}
.scrollable-table th {{
    background-color: #0056b3; 
    color: white;                
    padding: 10px;
    border-bottom: 3px solid #003d80;
    text-align: left;
}}
.scrollable-table td {{
    padding: 8px;
    vertical-align: top;
    border-bottom: 1px solid #ddd;
    max-width: 300px;
    word-wrap: break-word;
}}
</style>

<div class="scrollable-table">
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Summary</th>
                <th>Embedding</th>
            </tr>
        </thead>
        <tbody>
            {"".join(
                f"<tr><td>{row.id}</td><td>{row.summary}</td><td>{row.embedding}</td></tr>"
                for row in df_slice.itertuples()
            )}
        </tbody>
    </table>
</div>
"""

display(HTML(html_table))


def semantic_search_no_index(query_text: str, top_k: int = 5) -> pd.DataFrame:
    sql = f"""
    WITH query_table AS (
      SELECT '{query_text}' AS content
    ),
    query_embedding AS (
      SELECT ml_generate_embedding_result AS embedding
      FROM ML.GENERATE_EMBEDDING(
        MODEL `{PROJECT_ID}.{DATASET}.{EMBEDDING_MODEL}`,
        (SELECT content FROM query_table),
        STRUCT(TRUE AS flatten_json_output, 'RETRIEVAL_DOCUMENT' AS task_type)
      )
    )
    SELECT
      doc.id,
      doc.summary,
      (SELECT SUM(a*b)
         FROM UNNEST(doc.embedding) a WITH OFFSET
         JOIN UNNEST((SELECT embedding FROM query_embedding)) b WITH OFFSET USING(offset)
      ) /
      (SQRT((SELECT SUM(POW(x,2)) FROM UNNEST(doc.embedding) x)) *
       SQRT((SELECT SUM(POW(y,2)) FROM UNNEST((SELECT embedding FROM query_embedding)) y))) AS cosine_sim
    FROM `{PROJECT_ID}.{DATASET}.question_embeddings` AS doc
    LIMIT {EMBEDDINGS_LIMIT};
    """
    return client.query(sql).to_dataframe()

query_text = "How do I fix a 400 Bad Request error in Python requests?"
results_df = semantic_search_no_index(query_text, top_k=5)
results_df = results_df.sort_values(by="cosine_sim", ascending=False)

# --- Slice to top_k results ---
results_df = results_df.head(5).copy()

# --- Build HTML table ---
html_table = f"""
<style>
.semantic-table {{
    max-height:400px;
    overflow:auto;
    border:1px solid #ccc;
    border-radius:5px;
    font-family: sans-serif;
}}
.semantic-table table {{
    border-collapse: collapse;
    width: 100%;
}}
.semantic-table th {{
    background-color: #0056b3;
    color: white;
    padding: 10px;
    border-bottom: 3px solid #003d80;
    text-align: left;
}}
.semantic-table td {{
    padding: 8px;
    vertical-align: top;
    border-bottom: 1px solid #ddd;
    max-width: 300px;
    word-wrap: break-word;
}}
</style>

<div class="semantic-table">
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Summary</th>
                <th>Cosine Similarity</th>
            </tr>
        </thead>
        <tbody>
            {"".join(
                f"<tr><td>{row.id}</td><td>{row.summary}</td><td>{row.cosine_sim:.4f}</td></tr>"
                for row in results_df.itertuples()
            )}
        </tbody>
    </table>
</div>
"""

display(HTML(html_table))


# --- Fetch embeddings ---
emb_sql = f"""
SELECT embedding, summary
FROM `{PROJECT_ID}.{DATASET}.question_embeddings`
LIMIT {TSNE_SAMPLE_LIMIT}
"""
df_emb = client.query(emb_sql).to_dataframe()

if df_emb.empty:
    display(Markdown("> **No embeddings found.** Increase EMBEDDINGS_LIMIT and rerun embedding step."))
else:
    # Convert embeddings to numpy array
    emb_array = np.vstack(df_emb['embedding'].values)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto', perplexity=12)
    emb_2d = tsne.fit_transform(emb_array)

    # Add t-SNE coordinates to dataframe
    df_emb['x'] = emb_2d[:,0]
    df_emb['y'] = emb_2d[:,1]

    # --- Static scatter plot with matplotlib ---
    plt.figure(figsize=(12,8))
    plt.scatter(df_emb['x'], df_emb['y'], s=50, alpha=0.7, c='skyblue', edgecolors='k')

    # Optional: annotate a few points (first 10)
    for i, txt in enumerate(df_emb['summary'][:10]):
        plt.annotate(txt[:50]+"..." if len(txt)>50 else txt, (df_emb['x'].iloc[i], df_emb['y'].iloc[i]),
                     fontsize=9, alpha=0.8)

    plt.title('t-SNE Projection of Question Embeddings', fontsize=16)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig("tsne_question_embeddings.png", bbox_inches='tight')


preview_sql = f"""
SELECT id, title, SUBSTR(body, 1, 300) AS snippet, tags
FROM `{PROJECT_ID}.{DATASET}.sample_stackoverflow_posts`
LIMIT 5;
"""
df_preview = client.query(preview_sql).to_dataframe()
df_preview


# Preview rows
df_preview = client.query(preview_sql).to_dataframe()
df_preview.to_csv('preview_rows.csv', index=False)

# Fetch sample summaries
df_summaries = client.query(
    f"SELECT id, title, summary FROM `{PROJECT_ID}.{DATASET}.question_summaries` LIMIT 20"
).to_dataframe()

# Rename columns for a nicer CSV
df_summaries_renamed = df_summaries.rename(columns={
    'title': 'Developer Questions',
    'summary': 'AI Helpdesk Response'
})

# Save to CSV
df_summaries_renamed.to_csv('sample_summaries.csv', index=False)

print("CSV outputs saved: preview_rows.csv, sample_summaries.csv")

