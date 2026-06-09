import os
from kaggle_secrets import UserSecretsClient
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
import numpy as np
from numpy.linalg import norm
from google import genai
import vertexai
import pandas as pd
try:
    user_secrets = UserSecretsClient()
    os.environ['GOOGLE_API_KEY'] = user_secrets.get_secret("GOOGLE_API_KEY")
    print("✅ GOOGLE_API_KEY loaded successfully.")
except:
    print("🚨 Error: Could not find GOOGLE_API_KEY. Please add it to your Kaggle Secrets.")



client = genai.Client()
def get_embedding(text: str) -> np.ndarray:
    """Generates an embedding for a given text."""
    if not text:
        return np.zeros(768) # Assuming an embedding dimension of 768
    try:
        result = client.models.embed_content(model="gemini-embedding-001", contents=text,
                config=types.EmbedContentConfig(
                  task_type="retrieval_document",
                  ))
        return np.array(result.embeddings[0].values)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(768)


# We will store the abstract vectors here
abstract_vectors = np.array([])
df_safe_copy = pd.DataFrame() # To store the cleaned data

def build_semantic_index(df_input: pd.DataFrame):
    """
    (Run this once) Creates embeddings for all abstracts and stores them.
    """
    global abstract_vectors, df_safe_copy
    print("Building semantic index...")

    if df_input.empty:
        print("Database is empty, skipping index build.")
        return

    # 1. Clean and store a safe copy of the dataframe
    df_safe = df_input.copy()
    df_safe['abstract_text'] = df_safe['abstract_text'].fillna("").astype(str)
    # Reset index to ensure our vector index (0, 1, 2...) matches the df index
    df_safe = df_safe.reset_index(drop=True) 
    df_safe_copy = df_safe

    # 2. Create embeddings for all abstracts
    all_vectors = []
    for abstract in df_safe_copy['abstract_text']:
        all_vectors.append(get_embedding(abstract))

    # 3. Store in a single NumPy array for fast calculations
    abstract_vectors = np.array(all_vectors)
    print(f"✅ Index built. {len(abstract_vectors)} vectors created.")


# --- Load Dataset ---
DATA_PATH = "../input/pubmed-200k-rtc/PubMed_20k_RCT/train.csv"

try:
    # We only load a sample to keep it fast for this demo
    df_pubmed = pd.read_csv(DATA_PATH).sample(n=100, random_state=42)
    print(f"✅ Loaded {len(df_pubmed)} abstracts from PubMed dataset.")
    build_semantic_index(df_pubmed)
except FileNotFoundError:
    print(f"🚨 Error: Dataset not found at {DATA_PATH}. Please check the path.")
    df_pubmed = pd.DataFrame() # Create empty df to avoid errors


def search_local_database(query: str) -> dict:
    """
    Searches the local PubMed abstracts (vector index) for articles
    semantically similar to the query. Returns the top 3 matches.
    """
    global abstract_vectors, df_safe_copy

    if abstract_vectors.size == 0 or df_safe_copy.empty:
        return {"status":"error", "error_message":"The local database index has not been built."}

    # 1. Generate an embedding for the user's query
    try:
        query_vector = get_embedding(query)
    except Exception as e:
        return {"status":"error", "error_message": f"Could not generate query embedding: {e}"}

    # Calculate cosine similarity
    # It calculates the similarity between the query_vector and ALL abstract_vectors at once
    
    # Normalize vectors to unit length
    norm_query = query_vector / norm(query_vector)
    norm_abstracts = abstract_vectors / norm(abstract_vectors, axis=1, keepdims=True)
    
    # Compute dot product (which is cosine similarity for normalized vectors)
    similarities = np.dot(norm_abstracts, norm_query)

    # Get the indices of the top 3 most similar abstracts
    # We use np.argsort to get the indices, then flip for descending order
    top_3_indices = np.argsort(similarities)[::-1][:3]
    
    if not np.any(similarities[top_3_indices] > 0.5): # You can tune this 0.5 threshold
         return {"status":"error", "error_message":"No articles found in the local database with high confidence for that query."}

    # Build the output from the top 3 matches
    output = "Found these abstracts in the local database (semantic search):\n\n"
    for idx in top_3_indices:
        row = df_safe_copy.iloc[idx]
        similarity_score = similarities[idx]
        
        abstract_short = row['abstract_text'][:1200]
        output += (
            f"--- ABSTRACT (Match Score: {similarity_score:.4f}) ---\n"
            f"TARGET: {row['target']}\n"
            f"TEXT: {abstract_short}\n\n"
        )

    return {"status":"success", "output": output}


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


from google.adk.agents import LlmAgent
LocalSearchAgent = LlmAgent(
    name="LocalSearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite", 
        retry_options=retry_config
    ),
    instruction="""You MUST ALWAYS call the tool `search_local_database`.
Do NOT answer directly. Your only job is to retrieve matches from the local database.
Your goal is to find established research abstracts from the local database.
You will receive the user's original query.""",
    output_key="local_search_output",
    tools=[
        search_local_database
    ]
)


WebSearchAgent = LlmAgent(
    name="WebSearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a web search agent.
Your goal is to maximize the recall of relevant scientific evidence.
Find *newer* articles, news, or supplementary information. You will receive the user's original query.""",
    output_key="web_search_output",
    tools=[
        google_search
    ]
)


ParallelResearch = ParallelAgent(
    name="ParallelResearch",
    sub_agents=[LocalSearchAgent, WebSearchAgent]
)


SummarizerAgent = LlmAgent(
    name = "SummarizerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Your task is to parse and structure raw scientific text 
    and abstracts obtained by:
    1.  Local database: {local_search_output}
    2.  Web: {web_search_output}
    Combine and structure all the relevant findings.""",
    output_key="summary_output"
)


SynthesizerAgent = LlmAgent(
    name = "SynthesizerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a biomedical scientist responsible for producing a high-quality
medical evidence summary.
Interpret the extracted evidence from {summary_output} and Compare and synthesize findings.""",
)


rootAgent = SequentialAgent(
     name="MedicalLiteratureReviewPipeline",
    sub_agents=[ParallelResearch, SummarizerAgent, SynthesizerAgent],
)


APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

# InMemorySessionService stores conversations in RAM (temporary)
session_service = InMemorySessionService()
runner = Runner(agent=rootAgent, app_name=APP_NAME, session_service=session_service)
response = await runner.run_debug(
    "Summarize the clinical trial findings for sitagliptin in treating Type 2 Diabetes."
)

