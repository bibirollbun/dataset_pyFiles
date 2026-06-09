# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from google.adk.models.google_llm import Gemini
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder

APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


#%%writefile -a agent_logic.py
import ast
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools import google_search, AgentTool, ToolContext
from typing import Dict, Any, List, Optional, Callable
import datetime
import json
import textwrap
import uuid

from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

class EventsLogger:
    """
    Lightweight in-memory event logger and metrics collector.

    This class is intended for simple, structured logging and counting metrics inside an Agentic AI application 

    Attributes
    ----------
    logs : List[Dict[str, Any]]
        List of structured log entries. Each entry has:
        - "timestamp": UTC ISO-8601 string
        - "level": log severity (e.g. "INFO", "ERROR")
        - "message": human-readable description
        - "details": optional dict with extra context
    metrics : Dict[str, int]
        Dictionary of simple integer counters, keyed by metric name.
        Useful for counting events (e.g. "runs_started", "errors").
    """
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.metrics: Dict[str, int] = {}

    def log(self, level: str, message: str, details: Dict[str, Any] = None):
        """
        Add a structured log entry.

        Parameters
        ----------
        level : str
            Log level or severity (e.g. "INFO", "ERROR", "DEBUG").
        message : str
            Short, human-readable log message.
        details : Dict[str, Any], optional
            Additional structured context (payloads, IDs, timings, etc.).
            If not provided, an empty dict is stored.
        """
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "details": details or {},
        }
        self.logs.append(entry)

    def info(self, message: str, **kwargs):
        """
        Convenience helper for logging an informational message.
        Parameters
        ----------
        message : str
            Log message at INFO level."""
        self.log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs):
        """
        Convenience helper for logging an informational message.
        Parameters
        ----------
        message : str
            Log message at ERROR level."""
        self.log("ERROR", message, **kwargs)

    def incr(self, metric_name: str, amount: int = 1):
        """
        Increment a numeric metric counter.
        Parameters
        ----------
        metric_name : str
            Name of the metric to increment (e.g. "runs_started").
        amount : int, optional. Amount to add to the metric (default is 1).
        """
        self.metrics[metric_name] = self.metrics.get(metric_name, 0) + amount

    def dump(self):
        return {
            "logs": self.logs[-50:],          # last 50 log entries
            "metrics": self.metrics,
        }

logger = EventsLogger()
logger.info("Logger initialized")

class EventsLoggerPlugin(BasePlugin):
    """A custom plugin that logs every agent and LLM invocation."""

    def __init__(self, logger: 'EventsLogger') -> None:
        super().__init__(name="events_logger")
        # Store the shared EventsLogger instance
        self.logger = logger
        self.logger.info("EventsLoggerPlugin initialized.")

    # 1. Runs BEFORE any agent (root or sub-agent) is called.
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Log agent start and input data."""
        agent_id = agent.name
        
        # NOTE: Input data might be in the callback_context or the specific method's signature.
        # We'll log the agent ID and the type of call.
        self.logger.info(
            f"AGENT_START: {agent_id}",
            details={
                "call_type": "AgentInvocation",
                "parent_id": callback_context.agent_name, # Identifies the orchestrator
                # For input, you might need to inspect callback_context.inputs 
                # or a corresponding after_agent_callback for full I/O logging.
            }
        )
        self.logger.incr(f"{agent_id}_runs")

    # 2. Runs AFTER any agent (root or sub-agent) has finished its work.
    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Log agent completion and output."""
        agent_id = agent.name
        output_key = getattr(agent, "output_key", None)
        output_value = callback_context.state.get(output_key, "[No output found]") if output_key else "[No output key defined]"
        self.logger.info(
            f"AGENT_COMPLETE: {agent_id}",
            details={
                "call_type": "AgentCompletion",
                "status": "Success",
                "output_preview": str(output_value)[:100], # Log the output
                # The framework usually calculates duration, but you can calculate it here too.
            }
        )
        self.logger.incr(f"{agent_id}_success")
        
    # 3. Runs BEFORE a model is called (for every LLM prompt).
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Log LLM request content."""
        self.logger.info(
            "LLM_REQUEST: Model call initiated",
            details={
                "call_type": "LLM_Invocation",
                "source_agent": callback_context.agent_name,
                "prompt_preview": str(llm_request.contents[-1].parts[0].text)[:200],
            }
        )
        self.logger.incr("total_llm_calls")

    async def on_model_error_callback(
        self,*,callback_context: CallbackContext,error: Exception
    ) -> None:
        """Officially supported callback for LLM failures."""
        self.logger.error(
            f"MODEL_FAILURE: Agent {callback_context.agent_name} encountered an LLM error.",
            details={"error": str(error)}
        )


#import vertexai memory bank service for long-term memory persistence.
from google.adk.memory import VertexAiMemoryBankService



#%%writefile -a agent_logic.py
# Define the symptom extraction agent
symptom_agent = Agent(
    name="symptom_extractor",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Extracts key symptoms from customer medical chat input.",
    instruction=(
        """You receive a customer chat describing health concerns. 
        Identify and return ONLY a list of key symptoms mentioned. 
        Do NOT add extra commentary or unrelated information. "
        Reply with a python style list of symptom strings.
        The returned response must strictly be a LIST that can be used as python code"""
    ),
    output_key='symptoms_list'
)



#A fuzzy matcher that will be used to match a string to its closest resembling one from a list.
from fuzzywuzzy import process


#%%writefile -a agent_logic.py
def get_classifier_and_disease_encoder():
    '''
    Initializes the k nearest neighbor disease classifier to classify the disease based on a known set of symptoms.
    Returns:
        neigh: The k-nearest neighbor model which will be a globally accessible.
        ce: An ordinal encoder for encoding the disease into from a string to a number for allowing the model to make predictions.
        known_symptoms: A list of known and possible symptoms which act as a set of features for the classifier.
    '''
    disease_dataset = pd.read_csv('/kaggle/input/diseases-and-symptoms-dataset/Final_Augmented_dataset_Diseases_and_Symptoms.csv')
    known_symptoms = list(disease_dataset.columns[1:])
    X = disease_dataset.iloc[:,1:]
    y = disease_dataset.iloc[:,0]
    ce = OrdinalEncoder()
    y = ce.fit_transform(y.values.reshape(-1,1))
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    return ce, neigh, known_symptoms

ce, neigh,known_symptoms = get_classifier_and_disease_encoder()


import joblib
# Assuming 'knn_classifier' is your trained model
joblib.dump(neigh, 'knn_classifier.joblib')
joblib.dump(ce, 'ordinal_encoder.joblib')
# You can do this with the gcloud CLI or the Python SDK:
from google.cloud import storage

# You can do this with the gcloud CLI or the Python SDK:

storage_client = storage.Client(project="tidal-turbine-466223-s0")
bucket = storage_client.bucket('agentic-ai-buck01')
blob = bucket.blob("knn_classifier.joblib")
blob.upload_from_filename('knn_classifier.joblib')
blob = bucket.blob("ordinal_encoder.joblib")
blob.upload_from_filename('ordinal_encoder.joblib')


import os
from google.cloud import storage
from typing import List
def map_to_closest_symptom(symptom : str, known_symptom_list: List[str], threshold=70):
    """
    Maps a given symptom string to the closest known symptom using fuzzy matching.
    Returns None if similarity is below the threshold.
    Args: 
        symptom : a single symptom as a string
        known_symptom_list: a list of possible symptoms that are utilized as features by the ML model.
    Returns:
        The closest matching symptom from the known_symptom_list that resembles the symptom the most.
        Fuzzy matching is used here to match the strings.
    """
    match, score = process.extractOne(symptom, known_symptom_list)
    if score >= threshold:
        return match
    else:
        return None

    
def predict_disease(symptoms: List[str]):
    """
    Predicts a disease suffered by a patient based on the reported symptoms.
    Args:
        symptoms: a list of symptoms reported by the patient.
    Returns:
        The predicted disease based on those symptoms.
    """
    mapped_symptoms = []
    for sym in symptoms:
        mapped = map_to_closest_symptom(sym, known_symptoms)
        if mapped:
            mapped_symptoms.append(mapped)
    
    present_symptoms = np.zeros(len(known_symptoms))
    for symptom in mapped_symptoms:
        present_symptoms[known_symptoms.index(symptom)] = 1
    
    predicted_disease = ce.inverse_transform(neigh.predict(np.array(present_symptoms).reshape(1,-1)).reshape(-1,1))[0]#.to_array()
    return str(predicted_disease)

def precautions_recommender(disease: str):
    """
    Provides a set of recommendations to follow based on the predicted disease.
    Args:
        disease: The predicted disease.
    Returns:
        Precautions: A list of precautions to follow based on a specific disease, otherwise a message 
        suggesting that disease not present in database.
    """
    precautions_dataset = pd.read_csv('/kaggle/input/disease-and-symptoms-dataset/Disease precaution.csv')
    #Convert all disease to lowercase for convenient matching
    precautions_dataset['Disease'] = precautions_dataset['Disease'].str.lower()

    #Check if disease present in dataframe
    if disease.lower() in list(precautions_dataset['Disease'].unique()):
        #If yes, return the combined list of entries in all columns from 2nd till last which basically contain precautions.
        #We are ignoring 1st column because that itself corresponds to the disease.
        return precautions_dataset.loc[precautions_dataset['Disease']==disease.lower()].iloc[0,1:].dropna().astype(str).str.cat(sep=', ')
    else:
        return "Sorry, the mentioned disease is not present in the dataset."


#%%writefile -a agent_logic.py
disease_agent = Agent(
    name="disease_predictor",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Predicts disease from the list of suggested symptoms.",
    instruction=("You are a disease predictor agent that takes in a set of symptoms suffered by patient and predicts a potential disease."
        "First, you MUST call the symptom_agent to get the list of symptoms returned by it."
        "Based on the symptoms returned by the symptom_agent, use the predicted_disease function to predict a potential disease."
        "If the function cannot provide a valid response, try to predict based on your understanding."
        "Suggest and return the predicted disease without any additional commentary or extra information about it."
        "ONLY respond with the name of predicted disease."
    ),
    tools=[AgentTool(symptom_agent),predict_disease],
    output_key='predicted_disease'
)


google_search_agent = Agent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash"),
    description="Google Search agent",
    instruction="Use google search to fetch relevant info.",
    tools=[google_search],
)

precautions_agent = Agent(
    name="precaution_suggester",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Suggest a set of precautions to be taken based on predicted disease.",
    instruction=(
        "You will get the predicted disease suffered by a patient in {predicted_disease}"
        "Based on the {predicted_disease}, you will use your understanding, the response from 'precautions_recommender' tool (if any), and google_search_agent to provide a set of 3-5 medical precautions to be taken to prevent the disease from worsening."
        "Only provide a list of suggested precautions without adding any further commentary or extra information."
        "Provide the precautions as a concise list of 3-5 numbered items."
    ),
    tools=[AgentTool(disease_agent),AgentTool(google_search_agent),precautions_recommender],
    output_key='suggested_precautions'
)


diet_recommender = Agent(
    name="diet_recommender",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Suggest a set of precautions to be taken based on predicted disease.",
    instruction=(
        "You will get the predicted disease suffered by a patient as {predicted_disease} and the symptoms as {symptoms_list} from disease_agent and symptom_agent respectively."
        "Based on the {predicted_disease}, and google_search_agent, you will provide a diet recommendation and a simple meal plan for the concerned patient."
        "Also, explicitly mention a set of items the patient must absolutely avoid in their diet."
        "Your overall response should include always three things:"
        "- Suggested diet (e.g. low carb, high protein, high fiber intake, mediterranean, etc)."
        "- ALWAYS provide a simple meal plan that includes recommendations for 1) breakfast, 2) lunch, 3) dinner."
        "- Things to avoid (gluten, dairy, etc)."
    ),
    tools=[AgentTool(disease_agent),AgentTool(symptom_agent),AgentTool(google_search_agent)],
    output_key='dietary_suggestions'
)


exercise_recommender = Agent(
    name="exercise_recommender",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    description="Suggest a set of precautions to be taken based on predicted disease.",
    instruction=(
        "You will get the predicted disease suffered by a patient in {predicted_disease} from output of agents disease_agent and the symptoms as {symptoms_list} from output of symptom_agent."
        "Based on the {predicted_disease} and {symptoms_list}, you will provide a simple weekly and daily exercise plan for the patient."
        "Your overall response should include:"
        "- A daily or weekly workout plan with key exercises and yoga, along with their intensity levels."
        "- Lifestyle changes related to sleep, stress, and hygiene for disease management."
    ),
    tools=[AgentTool(disease_agent),AgentTool(symptom_agent)],
    output_key='exercise_recommender'
)


root_agent = Agent(
    name="healthcare_provider",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="A modular agent that predicts diseases and suggests strategies to manage them.",
    instruction=(
        "You are a healthcare management agent who predicts patient disease and provides concise lifestyle recommendations."
        "You will receive patient inputs describing his health issues and first call the disease_predictor and symptom_extractor agents to get the predicted disease and a list of symptom."
        "You will then pass the symptom list and predicted disease to the precautions_agent, diet_recommender, and exercise_recommender agents and ask each of them to generate recommendations."
        "You will then collect the suggestions provided by each of these agents and summarize them into a concise actionable plan."
        "Your final output should include the predicted disease and a concise summary of lifestyle recommendations."
        "You should dynamically update your response based on the new symptoms and description provided by the user."
    ),
    tools=[AgentTool(disease_agent),AgentTool(symptom_agent),AgentTool(precautions_agent),AgentTool(diet_recommender),AgentTool(exercise_recommender)],
    output_key='suggested_precautions'
)


import vertexai
from google.colab import auth

#authenticate the user.
auth.authenticate_user()
vertexai.init(
    project="tidal-turbine-466223-s0",
    location="us-central1",
    staging_bucket="agentic-ai-buck01",
)

from vertexai import agent_engines
app = agent_engines.AdkApp(
    agent=root_agent,
    enable_tracing=True,
)

#Create vertexAI client for different tasks such as deploying agent or persisting memories
client = vertexai.Client(
  project="tidal-turbine-466223-s0",
  location="us-central1"
)

#Creates an empty agent with no pre-defined configuration as a resource for storing memories
agent_engine = client.agent_engines.create()

print(agent_engine.api_resource.name)

agent_engine_id = agent_engine.api_resource.name.split("/")[-1]

#Initialize the VertexAiMemoryBankService and pass the 
memory_service = VertexAiMemoryBankService(
    project="tidal-turbine-466223-s0",
    location="us-central1",
    agent_engine_id=agent_engine_id
)



from google.adk.sessions import InMemorySessionService
from google.adk.memory import VertexAiMemoryBankService
from google.genai import types
vertex_memory_service = VertexAiMemoryBankService()

async def migrate_session_to_memory(in_memory_service, memory_service, app_name, user_id, session_id):
    # Retrieve the session from the in-memory session service
    session = await in_memory_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # Save session content/state into the memory service's long-term store
    await memory_service.add_session_to_memory(session)
    
# Initialize in-memory session service to store session data to memory 
session_service = InMemorySessionService()

#Create a session with name session_service
session = await session_service.create_session(
    app_name=APP_NAME, user_id=USER_ID, session_id='session_service'
)
runner = Runner(
    agent=root_agent,
    session_service=session_service,
    app_name=APP_NAME,
    plugins=[EventsLoggerPlugin(logger=logger)]
)



# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(event.content.parts[0].text)
    else:
        print("No queries!")


await run_session(
    runner,
    [
        "Hi, My name is Krishna. I have been feeling stomach cramps, fever, and nausea from a few days",
        "I am not having any cramps now, but now suffering from vomiting and nausea",  # This time, the agent should remember!
    ],
    "session_service",
)


await migrate_session_to_memory(session_service,memory_service,APP_NAME,USER_ID,session.id)


logger.logs




