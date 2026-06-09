from IPython.display import HTML

# Display Architecture pipeline

HTML(f'''
<div style="text-align: center; padding: 15px;">
    <a href="https://github.com/veyselserifoglu/bq-ai-patent-analyst/blob/main/doc/Patent%20Analysis%20Pipeline%20Architecture%20-%20PNG.png?raw=true" 
       target="_blank" 
       style="cursor: pointer; display: inline-block; text-decoration: none;">
        <div style="position: relative; display: inline-block;">
            <img src="https://github.com/veyselserifoglu/bq-ai-patent-analyst/blob/main/doc/Patent%20Analysis%20Pipeline%20Architecture%20-%20PNG.png?raw=true" 
                 width="300" 
                 height="200"
                 style="border: 2px solid #e0e0e0; border-radius: 8px; transition: all 0.3s ease; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
                 onmouseover="this.style.borderColor='#4285F4'; this.style.boxShadow='0 6px 12px rgba(66, 133, 244, 0.3)'"
                 onmouseout="this.style.borderColor='#e0e0e0'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.1)'">
            <div style="position: absolute; top: 8px; right: 8px; background: rgba(255,255,255,0.9); border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 14px;">
                â†—
            </div>
        </div>
    </a>
    <p style="margin-top: 12px; color: #5f6368; font-size: 13px; font-style: italic;">Click to explore the full architecture</p>
</div>
''')


# For visualization purposes
%pip install -q pyvis
%pip install -q plotly
%pip install -q ipywidgets


# BigQuery
import os
from google.cloud import bigquery
from kaggle_secrets import UserSecretsClient
import pandas as pd
from pyvis.network import Network
import plotly.express as px
from google.cloud import bigquery
from IPython.display import Image, display, HTML, IFrame
import ipywidgets as widgets
from ipywidgets import Layout
import warnings


# pd.set_option('display.max_colwidth', None)

# Suppress the specific UserWarning from the BigQuery client
# warnings.filterwarnings("ignore", message="BigQuery Storage module")


user_secrets = UserSecretsClient()
project_id = user_secrets.get_secret("GCP_PROJECT_ID")
gcp_key_json = user_secrets.get_secret("GCP_SA_KEY")
location = 'US'


# Write the key to a temporary file in the notebook's environment
key_file_path = 'gcp_key.json'
try:
    with open(key_file_path, 'w') as f:
        f.write(gcp_key_json)
    
    # Remove "> /dev/null 2>&1" to show the output.
    # Authenticate the gcloud tool using the key file
    !gcloud auth activate-service-account --key-file={key_file_path} > /dev/null 2>&1
    
    # Configure the gcloud tool to use your project
    !gcloud config set project {project_id} > /dev/null 2>&1
    
finally:
    # Securely delete the key file immediately after use
    if os.path.exists(key_file_path):
        os.remove(key_file_path)

# Enable the Vertex AI and BigQuery Connection APIs. Run only once Or Enable using the Cloud Interface.
# !gcloud services enable aiplatform.googleapis.com bigqueryconnection.googleapis.com > /dev/null 2>&1


# This command creates the connection resource. Remove "> /dev/null 2>&1" to show the output.
!bq mk --connection --location={location} --connection_type=CLOUD_RESOURCE llm-connection > /dev/null 2>&1


# This command shows the details of your connection. Remove "> /dev/null 2>&1" to show the output.
!bq show --connection --location={location} llm-connection > /dev/null 2>&1


# Initiate BigQuery client.
client = bigquery.Client(project=project_id, location=location)
client


# 1. Create the new dataset "patent_analysis"
patent_analysis = "patent_analysis"

create_dataset_query = f"""
CREATE SCHEMA IF NOT EXISTS `{project_id}.{patent_analysis}`
OPTIONS(location = '{location}');
"""
print(f"Creating dataset 'patent_analysis' in {location}...")
job = client.query(create_dataset_query)
try:
    job.result()
except Exception as e:
    print(f"â�Œ FAILED to create dataset. Error:\n\n{e}")


# 2. Create the AI model reference inside the new dataset
create_model_query = f"""
CREATE OR REPLACE MODEL `{project_id}.{patent_analysis}.gemini_vision_analyzer`
  REMOTE WITH CONNECTION `{location}.llm-connection`
  OPTIONS (endpoint = 'gemini-2.5-flash');
"""
print("\nCreating the AI model reference...")
job = client.query(create_model_query)
try:
    job.result()
except Exception as e:
    print(f"â�Œ FAILED to create the AI Model reference. Error:\n\n{e}")


# 3. Create the Object Table
# This query creates the "map" to the PDF files inside the local 'patent_analysis' dataset.
object_table_query = f"""
CREATE OR REPLACE EXTERNAL TABLE `{project_id}.{patent_analysis}.patent_documents_object_table`
WITH CONNECTION `{location}.llm-connection`
OPTIONS (
    object_metadata = 'SIMPLE',
    uris = ['gs://gcs-public-data--labeled-patents/*.pdf'] 
);
"""
print("Creating the object table...")
job = client.query(object_table_query)
try:
    job.result()
except Exception as e:
    print(f"â�Œ FAILED to create the object table. Error:\n\n{e}")


# 4. Create a remote connection for the embedding model.
sql_query = f"""
CREATE OR REPLACE MODEL `{project_id}.{patent_analysis}.embedding_model`
  REMOTE WITH CONNECTION `{location}.llm-connection`
  OPTIONS (endpoint = 'gemini-embedding-001');
"""

print("Creating the AI Embedding Model reference...")
job = client.query(sql_query)
try:
    job.result()
except Exception as e:
    print(f"â�Œ FAILED to create the AI Embedding Model reference. Error:\n\n{e}")


# 5. creates a helper function to perform L2 normalization on a vector.
create_classification_model = f"""
CREATE OR REPLACE FUNCTION `{project_id}.{patent_analysis}.L2_NORMALIZE`(vec ARRAY<FLOAT64>)
RETURNS ARRAY<FLOAT64> AS ((
  
  -- Calculate the L2 Norm (magnitude) of the vector.
  WITH vector_norm AS (
    SELECT SQRT(SUM(element * element)) AS norm
    FROM UNNEST(vec) AS element
  )
  
  -- Divide each element by the norm to create a unit vector.
  -- Handle the case where the norm is 0 to avoid division by zero errors.
  SELECT
    ARRAY_AGG(
      IF(norm = 0, 0, element / norm)
    )
  FROM
    UNNEST(vec) AS element, vector_norm
));
"""
print("Creating a Vector Normalization UDF...")
job = client.query(create_classification_model)
try:
    job.result()
except Exception as e:
    print(f"â�Œ FAILED to create the Vector Normalization reference. Error:\n\n{e}")


# 6. This creates a helper function to perform a weighted average of two vectors.
sql_query = f"""
CREATE OR REPLACE FUNCTION `{project_id}.{patent_analysis}.VECTOR_WEIGHTED_AVG`(
  vec1 ARRAY<FLOAT64>, weight1 FLOAT64,
  vec2 ARRAY<FLOAT64>, weight2 FLOAT64
)
RETURNS ARRAY<FLOAT64>
LANGUAGE js AS r'''
  if (!vec1 || !vec2 || vec1.length !== vec2.length) {{
    return null;
  }}
  let weighted_vec = [];
  for (let i = 0; i < vec1.length; i++) {{
    weighted_vec.push((vec1[i] * weight1) + (vec2[i] * weight2));
  }}
  return weighted_vec;
''';
"""

print("Creating a weighted average vector UDF...")
job = client.query(sql_query)
try:
    job.result()
except Exception as e:
    print(f"â�Œ FAILED to create the weighted average UDF reference. Error:\n\n{e}")


# 1. DataFrame Styler
def display_styled_df(df: pd.DataFrame, title: str):
    """
    Takes a DataFrame and returns a styled HTML table for better readability.
    """
    if df.empty:
        print("âš ï¸� DataFrame is empty.")
        return

    styler = df.style \
        .set_caption(f"<h3>{title}</h3>") \
        .set_properties(**{
            'text-align': 'left',
            'white-space': 'normal', # Crucial for wrapping long text
            'font-size': '14px',
            'vertical-align': 'top', # Aligns text to the top of the cell
            'border': '1px solid #444',
            'padding': '8px'
        }) \
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left'), ('font-size', '16px'), ('background-color', '#333')]},
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-size', '18px'), ('text-align', 'center')]}
        ])

    display(HTML(styler.to_html()))


# 1. Multimodal Analysis - only texts - ai_text_extraction table

prompt_text = """From this patent document, perform the following tasks:

1.  **Extract these fields**: title, inventor, abstract, 
    the **Filed**, the **Date of Patent**, the international classification code, and the applicant.
    
2.  **Translate**: If the original title and abstract are in German or French, translate them into English.

3.  **Identify Language**: Determine the original language of the document.

Return ONLY a valid JSON object with EXACTLY these ten keys: 
"title_en", "inventor", "abstract_en", "filed", "date_of_patent", "class_international", "applicant", and "original_language".

**Formatting Rule**: For any key that has multiple values (like "inventor" or "class_international" or "applicant"), 
combine them into a single string, separated by a comma and a space. For example: "Igor Karp, Lev Stesin".

The "original_language" value must be one of these three strings: 'EN', 'FR', or 'DE'.
If any other field is unavailable, use null as the value.
"""

# The main SQL query.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.ai_text_extraction` AS (
  WITH raw_json AS (
      SELECT
        uri,
        ml_generate_text_llm_result AS llm_result
      FROM
        ML.GENERATE_TEXT(
          MODEL `{project_id}.{patent_analysis}.gemini_vision_analyzer`,
          TABLE `{project_id}.{patent_analysis}.patent_documents_object_table`,
          STRUCT(
            '''{prompt_text}''' AS prompt,
            2048 AS max_output_tokens,
            0.2 AS temperature,
            TRUE AS flatten_json_output
          )
        )
    ),
    parsed_json AS (
      -- Step 2: Clean and parse the JSON output.
      SELECT
        uri,
        llm_result,
        SAFE.PARSE_JSON(
          REGEXP_REPLACE(llm_result, r'(?s)```json\\n(.*?)\\n```', r'\\1')
        ) AS json_data
      FROM
        raw_json
    )
  SELECT
    uri,
    llm_result,
    
    SAFE.JSON_VALUE(json_data, '$.original_language') AS original_language,
    SAFE.JSON_VALUE(json_data, '$.title_en') AS extracted_title_en,
    SAFE.JSON_VALUE(json_data, '$.inventor') AS extracted_inventor,
    SAFE.JSON_VALUE(json_data, '$.abstract_en') AS extracted_abstract_en,
    SAFE.JSON_VALUE(json_data, '$.filed') AS filed_date,
    SAFE.JSON_VALUE(json_data, '$.date_of_patent') AS official_patent_date,
    SAFE.JSON_VALUE(json_data, '$.class_international') AS class_international,
    SAFE.JSON_VALUE(json_data, '$.applicant') AS applican
    
  FROM
    parsed_json
);
"""

print("Attempting to create the ai text extraction table...")
job = client.query(sql_query)
try:
    job.result()
    print("âœ… Success: The `ai_text_extraction` table was created.")

    print("\nFetching a sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT 
        ate.uri, 
        ate.original_language,
        ate.extracted_title_en,
        ate.extracted_inventor, 
        ate.extracted_abstract_en,
        ate.filed_date,
        ate.class_international
    FROM `{project_id}.{patent_analysis}.ai_text_extraction` AS ate
    WHERE ate.extracted_title_en is not NULL
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, title="Sample of 5 Records from the `ai_text_extraction` Table")

except Exception as e:
    print(f"â�Œ FAILED: An error occurred. Error:\n\n{e}")


# 1. Multimodal Analysis - only extending ai_text_extraction table with the technical diagrams.

diagram_prompt_text = """
Describe this technical diagram from a patent document. 
What is its primary function and what key components are labeled?
"""

sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.ai_text_extraction` AS (

  WITH figures_with_object_ref AS (
      SELECT
        fig.*, obj.ref
      FROM
        `bigquery-public-data.labeled_patents.figures` AS fig
      JOIN
        `{project_id}.{patent_analysis}.patent_documents_object_table` AS obj
      ON
        fig.gcs_path = obj.uri
    ),
    
    generated_descriptions AS (
      SELECT
        gcs_path,
        ml_generate_text_llm_result AS diagram_description
      FROM
        ML.GENERATE_TEXT(
          MODEL `{project_id}.{patent_analysis}.gemini_vision_analyzer`,
          (
            SELECT
              gcs_path,
              [
                JSON_OBJECT('uri', ref.uri, 'bounding_poly', [
                  STRUCT(x_relative_min AS x, y_relative_min AS y),
                  STRUCT(x_relative_max AS x, y_relative_min AS y),
                  STRUCT(x_relative_max AS x, y_relative_max AS y),
                  STRUCT(x_relative_min AS x, y_relative_max AS y)
                ])
              ] AS contents,
              '''{diagram_prompt_text}''' AS prompt
            FROM
              figures_with_object_ref
          ),
          STRUCT(
            4096 AS max_output_tokens,
            0.2 AS temperature,
            TRUE AS flatten_json_output
          )
        )
    ),

    aggregated_descriptions AS (
      SELECT
        gcs_path,
        ARRAY_AGG(diagram_description IGNORE NULLS) AS diagram_descriptions
      FROM
        generated_descriptions
      GROUP BY
        gcs_path
    )

  SELECT
    T.*,
    S.diagram_descriptions
  FROM
    `{project_id}.{patent_analysis}.ai_text_extraction` AS T
  LEFT JOIN
    aggregated_descriptions AS S
  ON
    T.uri = S.gcs_path
);
"""

print("Attempting to extend the ai text extraction table with the diagram description...")
job = client.query(sql_query)
try:
    job.result()
    print("âœ… Success: The `ai_text_extraction` table was extended.")

    print("\nFetching a sample of 5 records from the table:")
    sql_select_sample_query = f"""
    SELECT 

        ate.uri, 
        ate.original_language,
        ate.extracted_title_en,
        ate.extracted_inventor,
        ate.filed_date,
        ate.diagram_descriptions
    
    FROM `{project_id}.{patent_analysis}.ai_text_extraction` AS ate
    WHERE ate.extracted_title_en is not NULL AND ARRAY_LENGTH(ate.diagram_descriptions) > 0
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, title="Sample of 5 Records from the `ai_text_extraction` Table, with diagrams descriptions")

except Exception as e:
    print(f"â�Œ FAILED: An error occurred. Error:\n\n{e}")


# 2. Knowledge Graph - patent_knowledge_graph table.

# Define the schema as a Python variable
schema = """
invention_domain STRING, problem_solved STRING, patent_type STRING, 
components ARRAY<STRUCT<component_name STRING, component_function STRING, connected_to ARRAY<STRING>>>
"""

# The prompt text remains the same
prompt_text = """
From the following patent text, perform these tasks:
1. Determine the high-level technical domain (e.g., 'Telecommunications', 'Medical Devices').
2. Provide a one-sentence summary of the core problem the invention solves.
3. Classify the patent as a 'Method', 'System', 'Apparatus', or a combination.
4. Extract all technical components into a nested list. 
For each component, provide its name, its primary function, and a list of other components it is connected to.

Here is the text:
"""

sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.patent_knowledge_graph` AS (
  SELECT
    t.uri,
    t.invention_domain,
    t.problem_solved,
    t.patent_type,
    t.components
  FROM
    AI.GENERATE_TABLE(
      MODEL `{project_id}.{patent_analysis}.gemini_vision_analyzer`,
      (
        SELECT
          uri,
          CONCAT(
            '''{prompt_text}''',
            '\\n\\n',
            IFNULL(extracted_title_en, ''),
            '\\n\\n',
            IFNULL(extracted_abstract_en, ''),
            '\\n\\nDiagrams:\\n',
            IFNULL(ARRAY_TO_STRING(diagram_descriptions, '\\n'), '')
          ) AS prompt
        FROM
          `{project_id}.{patent_analysis}.ai_text_extraction`
        WHERE
          extracted_abstract_en IS NOT NULL
      ),
      STRUCT(
        '''{schema}''' AS output_schema
      )
    ) AS t
);
"""

print("Attempting to create the patent knowledge graph...")
job = client.query(sql_query)
try:
    job.result()
    print("âœ… Success: The `patent_knowledge_graph` table was extended.")

    print("\nFetching a sample of 5 records from the table:")
    sql_select_sample_query = f"""
    SELECT 
    
        pkg.uri,
        pkg.invention_domain,
        pkg.problem_solved,
        pkg.patent_type,
        pkg.components
    
    FROM `{project_id}.{patent_analysis}.patent_knowledge_graph` AS pkg
    WHERE ARRAY_LENGTH(pkg.components) > 0 and pkg.invention_domain is not NULL
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, title="Sample of 5 Records from the `patent_knowledge_graph` Table")

except Exception as e:
    print(f"â�Œ FAILED: An error occurred. Error:\n\n{e}")


# 1. This query calculates the null percentage for key columns in the knowledge graph.
sql_completeness_check = f"""
SELECT
  COUNT(*) AS total_rows,
  ROUND(100 * COUNTIF(invention_domain IS NULL) / COUNT(*), 2) AS pct_null_domain,
  ROUND(100 * COUNTIF(problem_solved IS NULL) / COUNT(*), 2) AS pct_null_problem,
  ROUND(100 * COUNTIF(ARRAY_LENGTH(components) IS NULL OR ARRAY_LENGTH(components) = 0) / COUNT(*), 2) AS pct_empty_components
FROM
  `{project_id}.{patent_analysis}.patent_knowledge_graph`;
"""

print("--- Running Completeness Check ---")
try:
    df_completeness = client.query(sql_completeness_check).to_dataframe()
    display_styled_df(df_completeness, "Data Completeness and Null Rates (%)")
except Exception as e:
    print(f"â�Œ FAILED: The query failed. Error:\n\n{e}")


# 2. This query checks for duplicate URIs in the knowledge graph table.
sql_duplicate_check = f"""
SELECT
  uri,
  COUNT(*) AS num_occurrences
FROM
  `{project_id}.{patent_analysis}.patent_knowledge_graph`
GROUP BY
  uri
HAVING
  num_occurrences > 1;
"""

print("--- Running Uniqueness Check ---")
try:
    df_duplicates = client.query(sql_duplicate_check).to_dataframe()
    
    if df_duplicates.empty:
        print("âœ… Success: No duplicate patents found.")
    else:
        print("âš ï¸� Warning: Duplicate patents found! These URIs appear more than once:")
        display_styled_df(df_duplicates, "Duplicate Patent URIs")

except Exception as e:
    print(f"â�Œ FAILED: The query failed. Error:\n\n{e}")


# 3. This query validates the nested component schema for completeness.
sql_schema_check = f"""
SELECT
  COUNT(*) AS total_components,
  COUNTIF(c.component_name IS NULL) AS components_missing_name,
  COUNTIF(c.component_function IS NULL) AS components_missing_function
FROM
  `{project_id}.{patent_analysis}.patent_knowledge_graph` AS t,
  UNNEST(t.components) AS c;
"""

print("--- Running Schema Consistency Check ---")
try:
    df_schema = client.query(sql_schema_check).to_dataframe()
    display_styled_df(df_schema, "Component Schema Consistency")
except Exception as e:
    print(f"â�Œ FAILED: The query failed. Error:\n\n{e}")


# 4. This query finds patents with an anomalous number of components.
sql_outlier_check = f"""
WITH component_stats AS (
  SELECT
    uri,
    ARRAY_LENGTH(components) AS num_components,
    AVG(ARRAY_LENGTH(components)) OVER() AS avg_components,
    STDDEV(ARRAY_LENGTH(components)) OVER() AS stddev_components
  FROM
    `{project_id}.{patent_analysis}.patent_knowledge_graph`
)
SELECT
  uri,
  num_components
FROM
  component_stats
WHERE
  -- A standard statistical definition of an outlier
  num_components > avg_components + (3 * stddev_components);
"""

print("--- Running Outlier Detection ---")
try:
    df_outliers = client.query(sql_outlier_check).to_dataframe()
    
    if df_outliers.empty:
        print("âœ… Success: No significant outliers found in component counts.")
    else:
        print("âš ï¸� Warning: Potential outliers found. These patents have an unusually high number of components:")
        display_styled_df(df_outliers, "Patent Component Count Outliers")

except Exception as e:
    print(f"â�Œ FAILED: The query failed. Error:\n\n{e}")


# Distribution of component counts - Histogram

# --- Step 1: Fetch the component count for ALL patents ---
sql_all_counts = f"""
SELECT
  ARRAY_LENGTH(components) AS num_components
FROM
  `{project_id}.{patent_analysis}.patent_knowledge_graph`
WHERE
  ARRAY_LENGTH(components) > 0;
"""

print("--- Generating Distribution Plot with Outlier List ---")

try:
    df_all_counts = client.query(sql_all_counts).to_dataframe()
    
    # --- Step 2: Create the Histogram Figure ---
    fig = px.histogram(
        df_all_counts,
        x="num_components",
        title="<b>Distribution of Component Counts</b>",
        labels={"num_components": "Number of Components per Patent"}
    )

    # Add vertical lines for each outlier
    for index, row in df_outliers.iterrows():
        fig.add_vline(
            x=row['num_components'],
            line_width=2,
            line_dash="dash",
            line_color="red"
        )

    fig.update_layout(
        xaxis_title="<b>Number of Components</b>",
        yaxis_title="<b>Number of Patents</b>",
        font=dict(family="Arial, sans-serif", size=12),
        width=600 # Set a fixed width for the chart
    )
    
    # --- Step 3: Convert the chart and the list to HTML strings ---
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    outlier_list_html = "<h4>Potential Outliers:</h4><ul style='font-size: 12px; list-style-type: none; padding-left: 0;'>"
    for index, row in df_outliers.iterrows():
        short_name = row['uri'].split('/')[-1]
        outlier_list_html += f"<li style='margin-bottom: 5px;'>- {short_name} ({row['num_components']} components)</li>"
    outlier_list_html += "</ul>"

    # --- Step 4: Combine everything into a single HTML table for side-by-side display ---
    final_html = f"""
    <div style="display: flex; flex-direction: row; align-items: flex-start;">
        <div style="flex: 3;">{chart_html}</div>
        <div style="flex: 1; padding-left: 20px;">{outlier_list_html}</div>
    </div>
    """
    
    # Display the final combined HTML
    display(HTML(final_html))

except Exception as e:
    print(f"â�Œ FAILED: Could not generate the plot. Error:\n\n{e}")


# 5. Compare Companies' Patents.

# This query creates a summary table for each patent applicant.
sql_connection_density_query = f"""
WITH
  patent_connection_stats AS (
    SELECT
      T1.uri,
      T1.applican,
      T2.invention_domain,
      (
        SELECT SUM(ARRAY_LENGTH(c.connected_to))
        FROM UNNEST(T2.components) AS c
        WHERE c.connected_to IS NOT NULL
      ) AS total_connections
    FROM
      `{project_id}.{patent_analysis}.ai_text_extraction` AS T1
    JOIN
      `{project_id}.{patent_analysis}.patent_knowledge_graph` AS T2
    ON
      T1.uri = T2.uri
    WHERE
      T1.applican IS NOT NULL AND T2.invention_domain IS NOT NULL
  )

SELECT
  applican,
  COUNT(DISTINCT invention_domain) AS innovation_breadth,
  ROUND(AVG(total_connections), 2) AS average_connection_density,
  COUNT(uri) AS total_patents
FROM
  patent_connection_stats
WHERE
  total_connections > 0 -- Exclude patents with no connections to avoid skewing the average.
GROUP BY
  applican
HAVING
  COUNT(uri) > 1 -- Filter for applicants with more than one patent for a cleaner chart.
ORDER BY
  total_patents DESC;
"""

print("--- Calculating Enhanced Portfolio Metrics ---")
try:
    df_summary_enhanced = client.query(sql_connection_density_query).to_dataframe()
    print("âœ… Success: Enhanced metrics calculated.")
    display(df_summary_enhanced.head())
except Exception as e:
    print(f"â�Œ FAILED: The query failed. Error:\n\n{e}")

# Create the Interactive Bubble Chart using the new "connection density" metric.
fig = px.scatter(
    df_summary_enhanced,
    x="innovation_breadth",
    y="average_connection_density",
    size="total_patents",
    color="applican",
    hover_name="applican",
    log_x=True,
    size_max=60,
    title="<b>Strategic Patent Portfolio Analysis: Breadth vs. Connection Density</b>",
    labels={
        "innovation_breadth": "Innovation Breadth (Number of Domains)",
        "average_connection_density": "Average Connection Density (Connections per Patent)"
    }
)

# Customize the layout for a professional look
fig.update_layout(
    showlegend=False,
    xaxis_title="<b>Innovation Breadth â�¡ï¸�</b> (More Diverse)",
    yaxis_title="<b>Architectural Complexity â¬†ï¸�</b> (More Connections)"
)

display(HTML(fig.to_html()))


# This query creates a flat table of all components from all patents.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.patent_components_flat` AS (
  SELECT
    t.uri,
    t.invention_domain,
    c.component_name,
    c.component_function,
    c.connected_to
  FROM
    `{project_id}.{patent_analysis}.patent_knowledge_graph` AS t,
    UNNEST(t.components) AS c
  WHERE
    c.component_function IS NOT NULL
    AND c.component_name IS NOT NULL
);
"""

print("Attempting to create the flattened components table...")
job = client.query(sql_query)
try:
    job.result()
    print("âœ… Success: The `patent_components_flat` table was created.")

    print("\nFetching a sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT * FROM `{project_id}.{patent_analysis}.patent_components_flat` 
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, "Patent Components Flattened")

except Exception as e:
    print(f"â�Œ FAILED: An error occurred. Error:\n\n{e}")


# This query creates a single context vector for each patent, reading from ai_text_extraction table.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.patent_context_embeddings` AS (
  SELECT
    t.uri,
    t.ml_generate_embedding_result AS patent_context_vector
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `{project_id}.{patent_analysis}.embedding_model`,
      (
        SELECT
          uri,
          CONCAT(
            'Represent this technical patent for semantic search: \\n\\n', 
            'Patent Title: ', IFNULL(extracted_title_en, ''), '\\n\\n',
            'Applicant: ', IFNULL(applican, ''), '\\n\\n',
            'International Classification: ', IFNULL(class_international, ''), '\\n\\n',
            'Abstract: ', IFNULL(extracted_abstract_en, ''), '\\n\\n',
            'Diagram Descriptions: ', IFNULL(ARRAY_TO_STRING(diagram_descriptions, '\\n'), '')
          ) AS content
        FROM
          `{project_id}.{patent_analysis}.ai_text_extraction`
        WHERE
          extracted_title_en IS NOT NULL
      )
    ) AS t
);
"""

print("Attempting to create the patent context embeddings table...")
job = client.query(sql_query)
try:
    job.result() 
    print("âœ… Success: The `patent_context_embeddings` table was created.")

    print("\nFetching a sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT 
        uri, 
        ARRAY_LENGTH(patent_context_vector) as vector_dimensions 
    FROM `{project_id}.{patent_analysis}.patent_context_embeddings` 
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, "Patent Context Embedding Sample")

except Exception as e:
    print(f"â�Œ FAILED: An error occurred. Error:\n\n{e}")


# This query creates a single specific function vector for each individual component.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.component_function_embeddings` AS (
  SELECT
    t.uri,
    t.component_name,
    t.ml_generate_embedding_result AS component_function_vector
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `{project_id}.{patent_analysis}.embedding_model`,
      (
        SELECT
          uri,
          component_name,
          CONCAT(
            'Represent this technical patent for semantic search: \\n\\n',
            'A component named "', component_name, '" whose function is to ', component_function
          ) AS content
        FROM
          `{project_id}.{patent_analysis}.patent_components_flat`
      )
    ) AS t
);
"""

print("Attempting to create the component function embeddings table...")
job = client.query(sql_query)
try:
    job.result()
    print("âœ… Success: The `component_function_embeddings` table was created.")

    print("\nFetching a sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT 
        uri, 
        component_name,
        ARRAY_LENGTH(component_function_vector) as vector_dimensions 
    FROM `{project_id}.{patent_analysis}.component_function_embeddings` 
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, "Component Function Embedding Sample")

except Exception as e:
    print(f"â�Œ FAILED: An error occurred. Error:\n\n{e}")


# Normalization

def normalize_and_save_vectors(
    table_id: str,
    vector_column: str,
    client: bigquery.Client
):
    """
   Normalizes a vector column in a BigQuery table in-place by replacing
    the table with its normalized version.

    Args:
        table_id: The full ID of the table to update (e.g., "project.dataset.table").
        vector_column: The name of the column containing the vectors to normalize.
        client: An authenticated BigQuery client object.
    """


    # This SQL query selects all original columns and replaces the vector
    # column with its normalized version.
    sql_query = f"""
    CREATE OR REPLACE TABLE `{table_id}` AS (
      SELECT
        * EXCEPT({vector_column}),
        `{client.project}.{patent_analysis}.L2_NORMALIZE`({vector_column}) AS {vector_column}
      FROM
        `{table_id}`
    );
    """

    try:
        # Execute the query.
        job = client.query(sql_query)
        job.result()
    except Exception as e:
        print(f"â�Œ FAILED: An error occurred during normalization. Error:\n\n{e}")


# 1. Normalize the patent context embeddings.
print("--- Normalizing Patent Context Vectors ---")
normalize_and_save_vectors(
   table_id=f"{project_id}.{patent_analysis}.patent_context_embeddings",
   vector_column="patent_context_vector",
   client=client
)

# 2. Normalize the component function embeddings.
print("\n--- Normalizing Component Function Vectors ---")
normalize_and_save_vectors(
   table_id=f"{project_id}.{patent_analysis}.component_function_embeddings",
   vector_column="component_function_vector",
   client=client
)

print("\n--- Fetching a Diverse Sample of 5 Unique Patents ---")

# This query uses QUALIFY to get one component from 5 different patents.
sql_select_sample = f"""
SELECT
    uri,
    component_name,
    ARRAY_LENGTH(component_function_vector) as vector_dimensions
FROM
    `{project_id}.{patent_analysis}.component_function_embeddings`
QUALIFY
    ROW_NUMBER() OVER(PARTITION BY uri ORDER BY RAND()) = 1
LIMIT 5;
"""

try:
    df_sample = client.query(sql_select_sample).to_dataframe()
    display_styled_df(df_sample, "Normalized Embedding Sample")
except Exception as e:
    print(f"â�Œ FAILED to fetch a diverse sample. Error:\n\n{e}")


# This query rebuilds the search index using the UDF - weighted average function.
sql_query = f"""
CREATE OR REPLACE TABLE `{project_id}.{patent_analysis}.component_search_index` AS (
  SELECT
    flat.uri,
    flat.component_name,
    flat.component_function,
    -- Call our new UDF with the desired weights.
    `{project_id}.{patent_analysis}.VECTOR_WEIGHTED_AVG`(
      func.component_function_vector, 0.7, -- 70% weight to the function
      ctx.patent_context_vector, 0.3      -- 30% weight to the context
    ) AS combined_vector
  FROM
    `{project_id}.{patent_analysis}.patent_components_flat` AS flat
  JOIN
    `{project_id}.{patent_analysis}.patent_context_embeddings` AS ctx
  ON
    flat.uri = ctx.uri
  JOIN
    `{project_id}.{patent_analysis}.component_function_embeddings` AS func
  ON
    flat.uri = func.uri AND flat.component_name = func.component_name
);
"""

print("Attempting to create the final component search index table...")
job = client.query(sql_query)
try:
    job.result()
    print("âœ… Success: The `component_search_index` table was created.")

    print("\nFetching a diverse sample of 5 records from the new table:")
    sql_select_sample_query = f"""
    SELECT
        uri,
        component_name,
        ARRAY_LENGTH(combined_vector) as vector_dimensions
    FROM
        `{project_id}.{patent_analysis}.component_search_index`
    QUALIFY
        ROW_NUMBER() OVER(PARTITION BY uri ORDER BY RAND()) = 1
    LIMIT 5;
    """
    
    df_sample = client.query(sql_select_sample_query).to_dataframe()
    display_styled_df(df_sample, "Final Component Searching Sample")

except Exception as e:
    print(f"â�Œ FAILED: An error occurred. Error:\n\n{e}")

