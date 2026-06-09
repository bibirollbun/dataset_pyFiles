%pip install --upgrade bigframes google-cloud-automl google-cloud-translate google-ai-generativelanguage tensorflow 


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


PROJECT = "PROJECT"


import bigframes.pandas as bpd
bpd.options.bigquery.project = PROJECT
bpd.options.bigquery.ordering_mode = "partial" # Optional: partial ordering mode can accelerate executions and save costs
bpd.options.bigquery.use_cache = False

import bigframes.exceptions
import warnings
warnings.filterwarnings("ignore", category=bigframes.exceptions.AmbiguousWindowWarning)

import json
import gradio as gr
from PIL import Image
import io
import contextlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil


# # Setup embeddings and embedding model for future usage. Does not need to be run again

# from google.cloud import bigquery

# # Initialize BigQuery client
# client = bigquery.Client(project="PROJECT")

# # Create dataset if it doesn't exist
# client.query("""
# CREATE SCHEMA IF NOT EXISTS `PROJECT.embeddings`
# OPTIONS(location="US")
# """).result()

# # Create or replace the remote embedding model
# client.query("""
# CREATE OR REPLACE MODEL `PROJECT.embeddings.text_model`
# REMOTE WITH CONNECTION `us.bigquery-hackathon`
# OPTIONS(ENDPOINT = 'text-embedding-005')
# """).result()

# print("Embedding dataset and model ready.")


sys_prompt = """
You are an autonomous AI agent working with BigQuery data. You are not a query machine, you are an investigator. Your task is to uncover patterns, contrasts, and implications hidden in the dataset. Do not stop after a fixed number of steps unless the rules explicitly allow it. Keep probing until the picture feels rich, multi-dimensional, and trustworthy.

TOOLS:
- thought: Always the first tool in every step. Describe your reasoning, what you notice, what feels unanswered, and what to do next.
- SQL: Execute SQL queries against the dataset.
- python: Execute Python code. Never JSON escape. Never use python to predict data. Always use SQL FORECAST for predictions. When using python you can access `sql_result`, which holds the output from your last SQL query. It is a global variable - do not access it with `variables['sql_result']`, instead access it directly. Use save_var(name, value) to save variables for use in future code - variables you make are normally erased after the tool is used. Access saved data with variables[name]. The function and `variables` are already made - you don't need to make it yourself. Save only if you expect to reuse the data.
- exit: End the process with a final message.

GENERAL RULES:
- You must always use a tool. You cannot respond without using one.
- EVERY RESPONSE MUST CONTAIN A THOUGHT TOOL USAGE. ALWAYS.
- If you run SQL or python, that must be the last tool in your message.
- Never assume a table, column, or value exists. Always verify with schema.
- Never invent surrogate data, fabricate mappings, or hardcode values to fill gaps. If required information is absent, conclude that the path is impossible. Adapt with what is available or exit.
- Never access datasets you were not asked to. Never use prior knowledge of a dataset, even if you were trained on it. Use only domain knowledge unrelated to specific datasets plus query results.

EXPLORATION RULES:
- Always look for contrast. Compare groups, compare time periods, compare regions, compare metrics. A single view is never enough.
- If categories are messy, consolidate into meaningful groups before interpreting.
- Notice anomalies and outliers. Ask why they stand out and follow that thread with another query.
- You must explore at least four distinct perspectives. Time and category are mandatory. You must also explore at least two others such as geography, duration, anomalies, or seasonality.
- After finishing each perspective, ask yourself: what dimension have I not looked at yet? Then pursue it.
- Always design at least one query that could contradict or weaken your current conclusion. Only if the conclusion survives this self-challenge may you exit.
- Before you start digging, ask yourself: "How thorough should the search be: quick, moderate, or thorough?".

SQL FEATURES:
- Forecast numerical data:
    SELECT *
    FROM AI.FORECAST(
      (QUERY_STATEMENT),
      data_col => 'your_numeric_column',
      timestamp_col => 'your_date_column',
      horizon => N,
      confidence_level => N
    );

- Generate embeddings from text:
    SELECT *
    FROM ML.GENERATE_EMBEDDING(
      MODEL `bigquery-ai-hackathon-470403.embeddings.text_model`,
      (QUERY_STATEMENT),
      STRUCT(TRUE AS flatten_json_output)
    );

- Find nearest neighbors for a query embedding:
    SELECT *
    FROM VECTOR_SEARCH(
      TABLE base_table,
      'ml_generate_embedding_result',
      TABLE query_table,
      query_column_to_search => 'ml_generate_embedding_result',
      top_k => N,
      distance_type => 'COSINE'
    )
    WHERE distance < N;

SQL NOTES:
- If you generate embeddings and plan to reuse them (e.g. in clustering or search), immediately save them to a variable using save_var(). You are NEVER permitted to create datasets to save this data. Instead, ALWAYS use it in Python.
- Use embeddings for semantic tasks such as sentiment, semantic search, clustering, or nuanced queries. Keyword search is still allowed when it is more direct or efficient.
- These SQL features must never be run with Python. Always use the SQL tool for ML.GENERATE_EMBEDDING and VECTOR_SEARCH.
- The query passed into ML.GENERATE_EMBEDDING must alias the text field as `content`. You must filter out empty values and explicitly cap the result to 600 rows (hard limit).
- Always preserve any useful source fields (e.g., `title`, `body`, `id`, etc.) alongside `content` when creating embeddings. This ensures interpretability for downstream analysis like clustering or labeling.
- The ML.GENERATE_EMBEDDING output column is always named `ml_generate_embedding_result`. Never attempt to average or directly aggregate embedding vectors.
- For VECTOR_SEARCH, always use TABLE references (not inline SELECTs) for both base and query. Always use `ml_generate_embedding_result` as the column name. Distance is optional but recommended.
- When doing clustering or related tasks, you must examine real example values from each cluster before assigning labels. Never hardcode or assume them blindly.

VISUALS:
- Only make visualizations when the user asks for them.
- Always make visuals in python, not SQL.
- Always save images with plt.savefig().
- Never use markdown, base64, or other formats for images.
- NEVER hardcode values in Python. ALWAYS use SQL outputs instead using `sql_result` and `variables`.
- DO NOT MAKE IMAGES DURING DEEP DIVE. ONLY MAKE IMAGES IF YOU ARE TRYING TO SHOW THEM TO THE USER.

DATASET DISCOVERY:
1. Your very first step must ALWAYS be to list all tables in the dataset using project.dataset.INFORMATION_SCHEMA.TABLES. Do not use a WHERE clause. THIS IS REQURIED AND IS NOT NEGOTIABLE.
2. Immediately after, retrieve the schema (columns and data types) for those tables using project.dataset.INFORMATION_SCHEMA.COLUMNS. Do this only once at the start.
3. Use exact column names as given in the schema. Do not assume standard names like date exist unless confirmed.

USER REQUEST TYPES:
- Expect two types of queries: normal, and deep dive. Deep dive is ONLY for open-ended questions such as "How do I fix this?". Normal requests are everything else, from 
- If the request is a deep dive, your final message must connect at least three separate findings into one actionable recommendation. A plain summary of patterns is not enough. The recommendation must be detailed and genuinely insightful.
- If the request is a normal one, just answer it in full.

EXIT RULES:
- You cannot use exit with other tools in the same step.
- Exit if you are certain you cannot complete the request with available data.
- You may exit to ask for clarification if needed.
- Do not continue working after the request is fully satisfied.
- If you do not understand what the user wants at first, immediately exit and ask them.
"""


# The result of the model's last SQL query so it can access it in Python
sql_result = ""

# Store variables made by the model
variables = {}

# Fix cache issues that persist between sessions
bpd.close_session()

# Delete all files in "/kaggle/working" to allow the image detection to work
# If the AI overwrites an image from a previous conversation, it won't detect it since it's not a new image
folder = "/kaggle/working"
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

# Allow the model to save variables for future use
def save_var(name, val):
    global variables
    variables[name] = val

# Call the model to generate text
def call_model(prompt):
    query = f"""
        SELECT AI.GENERATE(
            '''{prompt}''',
            connection_id => "bigquery-ai-hackathon-470403.us.bigquery-hackathon",
            endpoint => "gemini-2.0-flash",
            output_schema => "tools ARRAY<STRUCT<name STRING, params STRING>>"
        ).full_response AS raw_output
    """
    
    df = bpd.read_gbq(query).to_pandas()
    outer = json.loads(df["raw_output"].iloc[0])
    return json.loads(outer["candidates"][0]["content"]["parts"][0]["text"])

# Runs the model's SQL query
def run_sql(sql: str):
    global sql_result
    
    df = bpd.read_gbq(sql).to_pandas()
    sql_result = df
    return df

# Start up the model's history with the system prompt
history = [{"role": "system", "content": sys_prompt}]

# Convert the history values into a single string for the model to see
def format_history():
    return "\n\n".join(f"[{h['role'].upper()}]\n{h['content']}" for h in history)

# Call the model and handle its output
# Model can call multiple tools in one output
def iter_message():
    # Add system message so it understands that *it* performed the previous actions
    # Without, it gets confused on what it's doing
    content = format_history() + "[SYSTEM] Now, perform the next action. All items in history have already been enacted. Do not repeat the last message."
    output = call_model(content)

    # Nudge the model so it doesn't get stuck
    if len(output["tools"]) == 0:
        return {"role": "system", "content": "You cannot use no tools! You must use a tool!"}
    
    for tool in output["tools"]:
        name = tool["name"]
        params = tool["params"]
    
        if name == "SQL":
            yield {"role": "assistant", "content": f"Ran SQL query: {params}"}

            try:
                response = run_sql(params)

                # Detect whether embeddings were generated
                if "ML.GENERATE_EMBEDDING" in params.upper():
                    yield {"role": "tool", "content": f"Generated embeddings with {len(response)} rows."}
                else:
                    yield {"role": "tool", "content": response.to_string()[:1000]}
            except Exception as e:
                yield {"role": "tool", "content": f"SQL query errored. Error[:500]: {str(e)[:500]}"}

        elif name == "python":
            # Nudge the model if it does something wrong
            if "base64" in params:
                yield {"role": "system", "content": "You cannot use base64 or markdown! You must use `plt.savefig()`!"}

            yield {"role": "assistant", "content": f"Python: ran code: {params}"}
            
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            before_files = set(os.listdir("."))

            try:
                # Run the code in an environment where we can see its text and image outputs
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    exec(params, {"sql_result": sql_result, "save_var": save_var, "variables": variables})
    
                text_output = buf_out.getvalue().strip()
                if text_output:
                    yield {"role": "tool", "content": f"Python: {text_output[:1500]}"}

                after_files = set(os.listdir("."))
                new_files = after_files - before_files
                for f in new_files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        yield {"role": "tool", "content": gr.Image(value=f)}
    
            except Exception as e:
                yield {"role": "tool", "content": f"Python: code errored. Fix the code and try again. Error[:500]: {str(e)[:500]}"}
        
        elif name == "thought":
            yield {"role": "assistant", "content": f"Thought: {params}"}

        elif name == "exit":
            val = True
            yield {"role": "assistant", "content": f"Exited deep dive. Message: {params}"}

# Prompt the model until it exits or reaches the message limit
def chat_loop(message, gradio_history):
    if not message.strip():
        return []
        
    history.append({"role": "user", "content": message})
    temp_hist = []
    
    i = 0
    done = False

    while done == False and i < 50:
        for output in iter_message():
            history.append(output)
            temp_hist.append(output)
            yield temp_hist

            if isinstance(output["content"], str) and output["content"].startswith("Exited deep dive."):
                done = True

        i += 1

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        type="messages",
        group_consecutive_messages=False,
        render_markdown=False, # To prevent Python code from rendering as Markdown
        height=650
    )

    gr.ChatInterface(
        fn=chat_loop,
        type="messages",
        chatbot=chatbot
    )

demo.launch(
    height=800,
    quiet=True
)


# Example prompts to the model, containing PROMPT.DATASET and the request
# Demonstrates various abilities, such as graph generation, embeddings, and more

# Open-ended prompt
"Using bigquery-public-data.chicago_crime.crime, investigate how crime types and frequencies vary across neighborhoods and seasons. Identify notable anomalies or shifts over time, and recommend strategies for resource allocation in law enforcement."

# Visualizations prompt
"Using bigquery-public-data.chicago_crime.crime, make a bar chart of the five most frequent crimes."

# Embeddings and clustering prompt
"Using `bigquery-public-data.stackoverflow.posts_questions`, cluster questions, then label each cluster and explain to me what it means."


# See a plain text version of the message history
print(format_history())

