!pip install -qU google-adk langchain-community langchain-google-genai faiss-cpu jq langchain langchain-text-splitters 
print("âœ… pip install successfully.")


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


!adk create helpful_code_assistant --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile helpful_code_assistant/agent.py

from typing import Annotated
from google.adk.agents import Agent, ParallelAgent, LlmAgent, SequentialAgent
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, ToolContext, FunctionTool
from google.genai import types
from google.adk.runners import Runner
from google.adk.models.google_llm import Gemini

from google.adk.code_executors import BuiltInCodeExecutor



print("âœ… ADK components imported successfully.")

APP_NAME = "code_troubleshooting_assistant"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

EMBEDDING_MODEL_NAME = "gemini-embedding-001"
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"

FOLDER_REPO_TO_BE_ANALYZED = "/kaggle/input/productioninsightagentsuite-dataset/capstone_ai_agents_intensive_fictional_library"
FAISS_CODE_INDEX_DIR = "/kaggle/working/faiss_code_index_code"


DEFAULT_METRIC_DATASET_FILE_PATH = "/kaggle/input/productioninsightagentsuite-dataset/metrics_sample/mid_metrics_sample.json"
FAISS_METRIC_INDEX_DIR = "/kaggle/working/faiss_code_index_metric"


print("âœ… Default variables defined.")



# Define helper functions that will be reused throughout the notebook

import os
import shutil

# Util to remove folder
def remove_folder(folder: str) -> bool:
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Default folder removed successfully.")
            return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False        
    return True

# Util to remove file
def remove_file(file_to_delete) -> bool:
    if os.path.exists(file_to_delete):
        try:
            os.remove(file_to_delete)
            print(f"File '{file_to_delete}' deleted successfully.")
            return True
        except OSError as e:
            print(f"Error deleting file '{file_to_delete}': {e}")
            return False
    else:
        print(f"File '{file_to_delete}' does not exist.")
        return False

# Util to print a list of file names inside a directory
def print_file_name(dir_path: str):
    for dirname, _, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            print(file_path)

# Util to verify if dataset exist
def dataset_exist() -> bool:
    if os.path.exists(DEFAULT_METRIC_DATASET_FILE_PATH):
        return True
    else:
        return False

print("âœ… Helper functions defined.")

import os

def clean_repo():
    # Clean git files to focus the code analysis only on the source files.
    remove_folder(FOLDER_REPO_TO_BE_ANALYZED + "/.git")
    
    print_file_name(FOLDER_REPO_TO_BE_ANALYZED)
    print("âœ… code filtered successfully.")

print("âœ… Code helper functions defined.")

from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

def load_code_chunks():
    
    # 1. --- Load Documents Recursively ---
    # DirectoryLoader is configured to:
    # - Look in the REPO_PATH.
    # - Use the **/*.py glob pattern to find ALL Python files, recursively.
    # - Use TextLoader to read the content of each file.
    loader = DirectoryLoader(
        FOLDER_REPO_TO_BE_ANALYZED,
        glob="**/*.py",           # Only load files ending in .py recursively
        loader_cls=TextLoader,    # Use the simple TextLoader to read content
        recursive=True,           # Search all subdirectories
        show_progress=True        # Show Directory Loader progress
    )    

    # Load the files into LangChain's Document format
    all_documents = loader.load()
    
    # --- 2. Intelligent Chunking (Substep 2) ---
    # Use the Python-aware splitter to maintain code structure
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Split ALL documents from the directory into structural chunks
    code_chunks = python_splitter.split_documents(all_documents)
    
    # --- 3. Verification ---
    print(f"Total files found in directory: {len(all_documents)}")
    print(f"Total structural code chunks created: {len(code_chunks)}")
    
    # Inspect the first chunk to verify content and metadata (source path)
    if code_chunks:
        print("\n--- Example Chunk ---")
        print(f"Source: {code_chunks[0].metadata['source']}")
        print(code_chunks[0].page_content)

    return code_chunks
print("âœ… Helper load_code_chunks functions defined.")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def save_code_faiss_index():
    # 1. Initialize the Embedding Model (Converts text to vectors)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # 2. Create the FAISS index (The Core Indexing step)
    # 'documents' comes from Substep 2's splitting process
    documents = load_code_chunks()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # 3. Save the index (So you don't have to re-run this step later)
    vectorstore.save_local(FAISS_CODE_INDEX_DIR)


print("âœ… Convert source code to vectors finished.")

def prepare_code_for_analysis():
    clean_repo()
    save_code_faiss_index()
print("âœ… Main function to fecth code created.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

print("âœ… retry_config created.")

code_analize_ingestor_agent = LlmAgent(
    name="code_analize_ingestor_agent",
    model=Gemini(
        model=DEFAULT_MODEL_NAME,
        retry_options=retry_config
    ),
    description="Agent responsible for the first step on a RAG pipeline for code, that includes fetching the source code from a given repository and preparing for the analysis phase",
    instruction="""Your job is responsible ONLY FOR CODE. 
        Please don't ask questions about URL and folder since you have already configured all the URLs and folders you need.
        You MUST strictly follow the step below and only use the available tool.
        Anytime someone ask to fecth code you MUST run prepare_code_for_analysis() tool
    """,
    tools=[prepare_code_for_analysis]
)

print("âœ… code_analize_ingestor_agent Agent defined.")


def clean_metric_folder() -> bool:
    return remove_folder(FAISS_METRIC_INDEX_DIR)
print("âœ… Metric helper functions defined.")

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
import jq
import json
from datetime import datetime


def load_metrics_chunks():
    jq_schema = "."
    # Load the file, treating each row as a separate document chunk
    loader = JSONLoader(
        file_path=DEFAULT_METRIC_DATASET_FILE_PATH,
        jq_schema=jq_schema,
        #json_lines=False,
        text_content=False,
    )
    dataset_documents = loader.load()
    print(dataset_documents)
    resource_query = '.[].[].resource.[].[] | {key, value }'
    resource_result = jq.all(resource_query, json.loads(dataset_documents[0].page_content))
    extracted_metadata=json.dumps(resource_result, indent=2)
    
    scope_spans_query = '.[].[].scope_spans'
    scope_spans_result = jq.all(scope_spans_query, json.loads(dataset_documents[0].page_content))
    extracted_page_content=json.dumps(scope_spans_result, indent=2)
    
    json_splitter = RecursiveJsonSplitter(max_chunk_size=50)
    split_chunks = json_splitter.split_json(scope_spans_result, convert_lists=True)
    metrics_documents = json_splitter.create_documents(split_chunks , convert_lists=True)
    
    
    current_timestamp = datetime.now()
    for metrics_document in metrics_documents:
        metrics_document.metadata['original_metadata'] = dataset_documents[0].metadata
        metrics_document.metadata['new_metadata'] = resource_result
        metrics_document.metadata['source'] = DEFAULT_METRIC_DATASET_FILE_PATH
    
        metrics_document.metadata['processed_date'] = current_timestamp.timestamp()
        metrics_document.metadata['processed_date_human'] = current_timestamp.strftime("%Y-%m-%d")

    return metrics_documents

print("âœ… Helper load_metrics_chunks functions defined.")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def save_metrics_faiss_index():
    # 1. Initialize the Embedding Model (Converts text to vectors)
    metrics_embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # 2. Create the FAISS index (The Core Indexing step)
    # 'documents' comes from Substep 2's splitting process
    metrics_documents = load_metrics_chunks()
    metrics_vectorstore = FAISS.from_documents(metrics_documents, metrics_embeddings)
    
    # 3. Save the index (So you don't have to re-run this step later)
    metrics_vectorstore.save_local(FAISS_METRIC_INDEX_DIR)


print("âœ… Convert metrics data to vectors finished.")

def prepare_metrics_for_analysis():
    dataset_exist()
    clean_metric_folder()
    save_metrics_faiss_index()
print("âœ… Main function to fecth metrics created.")

metrics_data_ingestor_agent = LlmAgent(
    name="metrics_data_ingestor_agent",
    model=Gemini(
        model=DEFAULT_MODEL_NAME,
        retry_options=retry_config
    ),
    description="Agent responsible for the first step on a RAG pipeline for Metrics data, including get data  from `metrics-sample` dataset and preparing for the analysis phase",
    instruction="""Your job is responsible ONLY FOR Metrics. 
        Please don't ask questions about URL and folder since you have already configured all the URLs and folders you need.
        You MUST strictly follow the step below and only use the available tool.        
        Anytime someone ask to fecth metrics you MUST run prepare_metrics_for_analysis() tool
    """,
    tools=[prepare_metrics_for_analysis]
)

print("âœ… metrics_data_assistant Agent defined.")

def faiss_code_exist() -> bool:
    if os.path.exists(FAISS_CODE_INDEX_DIR + "/index.pkl") :
        index_pkl_file_exist = True
        print("Code FAISS index.pkl ready to be consumed.")
    else:
        index_pkl_file_exist = False
        print(f"Code FAISS index.pkl - {FAISS_CODE_INDEX_DIR}/index.pkl -  not found.")
    if os.path.exists(FAISS_CODE_INDEX_DIR + "/index.faiss"):
        index_faiss_file_exist = True
        print("Code FAISS index.faiss ready to be consumed.")
    else:
        index_faiss_file_exist = False
        print(f"Code FAISS index.pkl - {FAISS_CODE_INDEX_DIR}/index.faiss -  not found.")
    
    if index_pkl_file_exist and index_faiss_file_exist:
        return True
    else:
        return False

def faiss_metrics_exist() -> bool:
    if os.path.exists(FAISS_METRIC_INDEX_DIR + "/index.pkl") :
        index_pkl_file_exist = True
        print("Metric FAISS index.pkl ready to be consumed.")
    else:
        index_pkl_file_exist = False
        print(f"Metric FAISS index.pkl - {FAISS_METRIC_INDEX_DIR}/index.pkl -  not found.")
    if os.path.exists(FAISS_METRIC_INDEX_DIR + "/index.faiss"):
        index_faiss_file_exist = True
        print("Metric FAISS index.faiss ready to be consumed.")
    else:
        index_faiss_file_exist = False
        print(f"Metric FAISS index.pkl - {FAISS_METRIC_INDEX_DIR}/index.faiss -  not found.")
    
    if index_pkl_file_exist and index_faiss_file_exist:
        return True
    else:
        return False

print("âœ… Faiss verificator helper defined.")

def is_all_faiis_created() -> bool:
    return faiss_code_exist() and faiss_metrics_exist()
print("âœ… Main function to valida FAIIS created.")

# The ingestion_validator runs *after* the parallel step to ensure both have finished their job.
ingestion_aggregator = LlmAgent(
    name="ingestion_aggregator",
    model=DEFAULT_MODEL_NAME,
    instruction="Your job is responsible ONLY FOR call is_all_faiis_created() tool and ensure that return True.",
    tools=[is_all_faiis_created]
)

print("âœ… aggregator_agent created.")

ingestion_agents = ParallelAgent(
    name="data_ingestion_agents",
    sub_agents=[metrics_data_ingestor_agent, code_analize_ingestor_agent],
)


ingestion_coordinator = SequentialAgent(
    name="ingestion_coordinator",
    sub_agents=[ingestion_agents, ingestion_aggregator],
)

print("âœ… Ingestion Coordinator Agent created.")

import os
from kaggle_secrets import UserSecretsClient

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def code_search_tool(query: str) -> str:
    """
    Use this tool to search the code repository for specific functions, classes,
    or implementation details related to the user's question.
    Input must be a single string query (the user's question).
    The output is the relevant source code snippets.
    """
    if code_retriever is None:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        try:
            vectorstore = FAISS.load_local(FAISS_CODE_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            code_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            print(f"FATAL ERROR: Could not load FAISS index. Run indexing steps first. Error: {e}")
            code_retriever = None
        if code_retriever is None:
            return "ERROR: The code repository index is not available."
        
    docs = code_retriever.invoke(query)
    formatted_context = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown File')
        content = doc.page_content
        formatted_context.append(f"--- FILE: {source} ---\n{content}\n")
    
    return "\n".join(formatted_context)
# -------------------------------------------------------------------------
print("âœ… Function code_search_tool successfully created.")

code_analize_agent = Agent(
    name="code_analize_assistant",
    model=Gemini(
        model=DEFAULT_MODEL_NAME,
        retry_options=retry_config
    ),
    description="A simple agent that can answer  code questions.",
    instruction="You are a helpful assistant for answering code questions ONLY. Use the code_search_tool() tool to search the knowledge base for relevant information before answering the user's questions.",
    tools=[code_search_tool],
    output_key="code_analize"
)

print("âœ… code_analize_agent Agent defined.")

import os
from kaggle_secrets import UserSecretsClient

# LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS



def get_metric_data(query: str) -> str:
    """
    Use this tool to search the metric data for specific usage, metric, trace, span
    of functions related to the user's question.
    Input must be a single string query (the user's question).
    The output is the relevant information about the metric found.
    """
    if metrics_code_retriever is None:
        metrics_embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        try:
            metrics_vectorstore = FAISS.load_local(FAISS_METRIC_INDEX_DIR, metrics_embeddings, allow_dangerous_deserialization=True)
            metrics_code_retriever = metrics_vectorstore.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            print(f"FATAL ERROR: Could not load FAISS index. Run indexing steps first. Error: {e}")
            metrics_code_retriever = None
        if metrics_code_retriever is None:
            return "ERROR: The metric repository index is not available."
        
    docs = metrics_code_retriever.invoke(query)
    formatted_context = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown File')
        content = doc.page_content
        formatted_context.append(f"--- FILE: {source} ---\n{content}\n")
    
    return "\n".join(formatted_context)
# -------------------------------------------------------------------------
print("âœ… Function get_metric_data successfully created.")

metrics_data_agent = Agent(
    name="metrics_data_assistant",
    model=Gemini(
        model=DEFAULT_MODEL_NAME,
        retry_options=retry_config
    ),
    description="A simple agent that can answer  metrics questions.",
    instruction="You are a helpful assistant for answering metrics questions ONLY. Use the get_metric_data() tool to search about metrics data from our telemetry system",
    tools=[get_metric_data],
    output_key="metrics_data"
)

print("âœ… code_analize_agent Agent defined.")

info_retriever = ParallelAgent(
    name="info_retriever_agent",
    sub_agents=[code_analize_agent, metrics_data_agent],
    
)

print("âœ… retreive_info Parallel Agent created.")


aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model=DEFAULT_MODEL_NAME,
        retry_options=retry_config
    ),
    instruction="""Combine information from the code_data and matching with the metrics_info information and produce a single executive summary:
    
    **code_data:**
    {code_analize}
    
    **metrics_info:**
    {metrics_data}

    The way to combine the metric information with the code is using the span_name
    
    Your summary should highlight the correlation between both, usage, and counters. The final summary should be around 200 words.""",
    output_key="executive_summary"
)

print("âœ… aggregator_agent created.")

root_agent = LlmAgent(
    name="agents_coordinator",
    model=Gemini(
        model=DEFAULT_MODEL_NAME,
        retry_options=retry_config
    ),
    instruction="""You are a code research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
        1. First, you MUST call the agent tool ingestion_coordinator() to prepare the files and index.
        2. Next, you MUST call the agent tool info_retriever() to fetch code data and metrics. This is trigger a parallel agent that will invoke two other agents:
        `code_analize_agent` and `metrics_data_agent` so you will have the RAG answer for both;
        3. After that you will have one output about code named code_analize and one output about metrics named metrics_data;
        4. Next, you MUST call the `aggregator_agent` tool to create a concise summary.
        5. Finally, present the final summary clearly to the user as your response.""",
    tools=[AgentTool (ingestion_coordinator), AgentTool(info_retriever), AgentTool(aggregator_agent)],     
)

print("âœ… Coordinator Agent created.")


url_prefix = get_adk_proxy_url()


!adk web --log_level DEBUG --url_prefix {url_prefix}

