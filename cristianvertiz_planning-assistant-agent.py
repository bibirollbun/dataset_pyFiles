import asyncio
import os
from datetime import datetime

import pymongo
from google.adk.agents import LlmAgent
from google.adk.memory import VertexAiMemoryBankService
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import AgentTool
from google.adk.tools import preload_memory
from google.genai import types
from openai import OpenAI
from pymongo import MongoClient
from pymongo.errors import OperationFailure
import vertexai
from vertexai import agent_engines

from kaggle_secrets import UserSecretsClient


try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    OPENAI_API_KEY = UserSecretsClient().get_secret("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' and 'OPENAI_API_KEY' to your Kaggle secrets. Details: {e}"
    )


try:
    MONGODB_URI = (
        f"mongodb+srv://{UserSecretsClient().get_secret('MONGODB_USERNAME')}:{UserSecretsClient().get_secret('MONGODB_PASSWORD')}@{UserSecretsClient().get_secret('MONGODB_HOST')}/?appName=planningAssistant"
    )
    
    MONGODB_DB_NAME = UserSecretsClient().get_secret("MONGODB_DB_NAME")
    MONGODB_COLLECTION_NAME = UserSecretsClient().get_secret("MONGODB_COLLECTION_NAME")
    MONGODB_CONNECTION_TIMEOUT_MS = UserSecretsClient().get_secret("MONGODB_CONNECTION_TIMEOUT_MS")
    MONGODB_SERVER_SELECTION_TIMEOUT_MS = UserSecretsClient().get_secret(
        "MONGODB_SERVER_SELECTION_TIMEOUT_MS"
    )
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'MONGODB_USERNAME', 'MONGODB_PASSWORD', 'MONGODB_HOST', 'MONGODB_DB_NAME', 'MONGODB_COLLECTION_NAME', 'MONGODB_CONNECTION_TIMEOUT_MS', 'MONGODB_SERVER_SELECTION_TIMEOUT_MS' to your Kaggle secrets. Details: {e}"
    )


try:
    SESSION_DB_URL = UserSecretsClient().get_secret("SESSION_DB_URL")
    os.environ["SESSION_DB_URL"] = SESSION_DB_URL

    # Set up Cloud Credentials in Kaggle
    user_secrets = UserSecretsClient()
    user_credential = user_secrets.get_gcloud_credential()
    user_secrets.set_tensorflow_credential(user_credential)

    
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'SESSION_DB_URL' and 'GOOGLE_APPLICATION_CREDENTIALS' to your Kaggle secrets. Details: {e}"
    )


APP_NAME = "agents"
USER_ID = "cvertiz"

async def run_session(
        runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")

print("âœ… Helper functions defined.")


def generate_embedding(text: str) -> dict:
    """
    Generates an embedding vector representation of the provided text using a
    pre-configured OpenAI embedding model. The function first checks if the input
    text is non-empty; if empty, it returns None. Otherwise, it sends the text to
    the OpenAI API and receives the corresponding embedding vector, which is then
    returned.

    Args:
        text (str): The input text for which the embedding is to be generated.
            Must not be an empty string.

    Returns:
        Dictionary with status and list of floats containing the embedding.
        Success: {"status": "success", "embedding": [0.000, ...]}
        Error: {"status": "error", "error_message": "Cannot create embedding: ..."}
    """
    if not text.strip():
        return {
            "status": "error",
            "error_message": "Cannot create embedding:: Input text cannot be empty.",
        }
    embedding = (
        openai_client.embeddings.create(input=[text], model="text-embedding-3-small")
        .data[0]
        .embedding
    )
    return {"status": "success", "embedding": embedding}


def find_similar_task_or_issue(query: str) -> dict:
    """Searches for tasks or issues in a Jira repository database with a description semantically similar to a given description, summary, or comments.

    Args:
        query: A string containing the user's query about Jira reported tasks or issues.
            This can be a task or issue description, or any other relevant information related to the
            task or issue in the Jira database.
    Example:
        query = "Is there any issue related to tests in the CppSuite?"
    Example:
        query = "Is there any task related to improving documentation?"
    Returns:
        Dictionary with status and list of dictionaries containing the key, description, fields.assignee, fields.priority, fields.status, fields.creator, fields.reporter, fields.issuetype, fields.project, and score for the task or issue.
        Success: {"status": "success", "documents": [{"key": str, "description": str, "fields.assignee": dict, "fields.priority": dict, "fields.status": dict, "fields.creator": dict, "fields.reporter": dict, "fields.issuetype": dict, "fields.project": dict, "score": float}, ...]}
        Error: {"status": "error", "error_message": "Cannot find similar task or issue: ..."}
    """

    # query by tag
    query_embedding_result = generate_embedding(query)

    if query_embedding_result["status"] == "error":
        return {
            "status": "error",
            "error_message": f'Cannot find similar task or issue: {query_embedding_result["error_message"]}',
        }

    query_embedding = query_embedding_result["embedding"]

    # Sample vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": 5,
            }
        },
        {
            "$project": {
                "_id": 0,
                "key": 1,
                "description": 1,
                "fields.assignee": 1,
                "fields.priority": 1,
                "fields.status": 1,
                "fields.creator": 1,
                "fields.reporter": 1,
                "fields.issuetype": 1,
                "fields.project": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        # Execute the aggregation pipeline
        documents = collection.aggregate(pipeline).to_list()
        return {"status": "success", "documents": documents}
    except pymongo.errors.OperationFailure as e:
        return {
            "status": "error",
            "error_message": f"Cannot find similar task or issue: {e}",
        } 


def get_risk_analysis(issue_keys: list[str]) -> dict:
    """
    Returns the analysis of Jira issue status transitions for the provided list of issue
    keys. The function computes the status transitions for each issue based on changelog
    data, including transition durations and overall ticket lifetime.

    The method uses an aggregation pipeline to query the Jira issues from the database,
    extracts their status transitions, and calculates the number of days spent in each
    status. In case an issue has no status transitions, a default self-loop transition
    is created to represent its lifetime in the same status.

    Parameters:
    issue_keys (list[str]): A list of Jira issue keys for which risk analysis is to be
                            performed.

    Returns:
        Dictionary with status and list of dictionaries containing issue key, ticket lifetime in days, source status, target status, transition days, from date, and to date.
        Success: {'status': 'success', 'transitions': [{'issue_key': str, 'ticket_lifetime_days': int, 'source_status': str, 'target_status': str, 'transition_days': int, 'from_date': str, 'to_date': str}, ...]}
        Error: {'status': 'error', 'error_message': 'Cannot get risk analysis: ...'}
    """

    # print(f"ğŸ”� [{get_jira_ticket_status_transitions}] State check: {len(tool_context.state.get('ticket_status_transitions', []))} transitions in context")
    # print(f"ğŸ”� [{get_jira_ticket_status_transitions}] Context ID: {id(tool_context.state)}")
    print(f"Processing {len(issue_keys)} tickets: {', '.join(issue_keys)}")

    pipeline = [
        {"$match": {"key": {"$in": issue_keys}}},
        {"$project": {"_id": 0, "key": 1, "changelog": 1, "fields.created": 1}},
    ]
    try:

        # Execute the aggregation pipeline
        documents = collection.aggregate(pipeline).to_list()
        # print(f"Obtained documents {documents} for the given key issues")

        # Get current date
        current_date = datetime.now()

        transitions = []

        for issue in documents:

            # Get creation date for ticket lifetime calculation
            creation_date = datetime.fromisoformat(
                issue["fields"]["created"].replace("Z", "+00:00")
            )
            ticket_lifetime_days = (
                current_date.replace(tzinfo=creation_date.tzinfo) - creation_date
            ).days + 1
            # print(f"   Ticket lifetime: {ticket_lifetime_days} days (created: {creation_date.strftime('%Y-%m-%d')})")

            # Parse transitions directly from changelog
            prev_status = "Backlog"
            prev_date = issue["fields"]["created"]

            # CRITICAL: Process changelog in chronological order (oldest first)
            for history in issue["changelog"]["histories"]:  # â†� Add reversed() here!
                for item in history["items"]:
                    if item["field"] == "status":
                        # Always record the transition
                        from_date = datetime.fromisoformat(
                            prev_date.replace("Z", "+00:00")
                        )
                        to_date = datetime.fromisoformat(
                            history["created"].replace("Z", "+00:00")
                        )
                        days = max(1, (to_date - from_date).days + 1)

                        transitions.append(
                            {
                                "issue_key": issue["key"],
                                "ticket_lifetime_days": ticket_lifetime_days,
                                "source_status": prev_status,
                                "target_status": item["toString"],
                                "transition_days": days,
                                "from_date": prev_date,
                                "to_date": history["created"],
                            }
                        )

                        prev_status = item["toString"]
                        prev_date = history["created"]

            # Handle case where ticket has no status changes (self-loop)
            if not transitions:
                # Create self-loop from creation to now with initial status
                initial_status = issue["fields"]["status"]["name"]
                days = max(1, ticket_lifetime_days)

                transitions.append(
                    {
                        "issue_key": issue["key"],
                        "ticket_lifetime_days": ticket_lifetime_days,
                        "source_status": initial_status,
                        "target_status": initial_status,
                        "transition_days": days,
                        "from_date": issue["fields"]["created"],
                        "to_date": current_date.isoformat(),
                    }
                )
    except pymongo.errors.OperationFailure as e:
        return {"status": "error", "error_message": f"Cannot get risk analysis: {e}"}
    return {"status": "success", "transitions": transitions}


def get_priority_analysis(keys: list[str]) -> dict:
    """
    Build the aggregation pipeline that extracts structured sprint data.

    This function constructs an aggregation pipeline that collects and transforms
    sprint-related data from a MongoDB collection, based on the specified keys.
    The resulting pipeline applies filtering, extraction, and transformation steps
    for structured sprint details. The function also handles potential query
    failures gracefully.

    Args:
        keys (list[str]): A list of string keys used to filter sprints.

    Returns:
        Dictionary with status and list of dictionaries containing issue_key, sprint_name, sprint_startDate, sprint_endDate, sprint_state, sprint_goal.
        Success: {'status': 'success', 'sprints': [{'issue_key': str, 'sprint_name': str, 'sprint_startDate': str, 'sprint_endDate': str, 'sprint_state': str, 'sprint_goal': str}, ...]}
        Error: {'status': 'error', 'error_message': 'Cannot get priority analysis: ...'}
    """

    print(f"fetching sprints for: {keys}")

    # Variable used within $map for each sprint string
    sprint_var = "$$sprint"

    def first_split_after(token: str):
        """
        Helper expression: Splits a string after the first occurrence of a specified token and retrieves the
        first part from the resulting split segments.

        Parameters:
        token: str
            The token used to split the string.

        Returns:
        dict
            A MongoDB aggregation expression that splits the input string based on the
            specified token, further splits the resulting substring by a comma, and
            retrieves the first part of the split result.
        """
        return {
            "$arrayElemAt": [
                {
                    "$split": [
                        {"$arrayElemAt": [{"$split": [sprint_var, token]}, 1]},
                        ",",
                    ]
                },
                0,
            ]
        }

    def date_fallback(date_field, fallback):
        """
        Helper expression: Generates a MongoDB query expression that conditionally replaces a null date
        value with a specified fallback value.

        This function creates an aggregation conditional expression using `$cond`.
        It checks if the given `date_field` is equal to "<null>", and if true, replaces
        it with the provided `fallback` value. Otherwise, it keeps the original
        `date_field` value.

        Args:
            date_field: The field containing the date value to be evaluated.
            fallback: The fallback value to use if `date_field` is equal to "<null>".

        Returns:
            dict: A MongoDB aggregation conditional expression structured to handle
            null date values with a fallback.
        """
        return {
            "$cond": {
                "if": {"$eq": [date_field, "<null>"]},
                "then": fallback,
                "else": date_field,
            }
        }

    # Build the initial match stage. Always require `fields.customfield_10557` to be non-null.
    match_stage = {
        "fields.customfield_10557": {"$ne": None},
        "key": {"$in": list(keys)},
    }

    pipeline = [
        {"$match": match_stage},
        {
            "$project": {
                "_id": 0,
                "key": 1,
                "sprints": {
                    "$map": {
                        "input": "$fields.customfield_10557",
                        "as": "sprint",
                        "in": {
                            "id": {"$toInt": first_split_after("id=")},
                            "rapidViewId": {
                                "$toInt": first_split_after("rapidViewId=")
                            },
                            "state": first_split_after("state="),
                            "name": first_split_after("name="),
                            "startDate": date_fallback(
                                first_split_after("startDate="), None
                            ),
                            "endDate": date_fallback(
                                first_split_after("endDate="), None
                            ),
                            "completeDate": date_fallback(
                                first_split_after("completeDate="), None
                            ),
                            "activatedDate": date_fallback(
                                first_split_after("activatedDate="), None
                            ),
                            "sequence": {"$toInt": first_split_after("sequence=")},
                            "goal": first_split_after("goal="),
                        },
                    }
                },
            }
        },
        # Flatten the sprint array so each sprint becomes its own document
        {"$unwind": "$sprints"},
        {
            "$project": {
                "key": 1,
                # Flatten fields from sprints into top-level fields
                "sprint_name": "$sprints.name",
                "sprint_startDate": "$sprints.startDate",
                "sprint_endDate": "$sprints.endDate",
                "sprint_state": "$sprints.state",
                "sprint_goal": "$sprints.goal",
            }
        },
    ]

    try:
        documents = collection.aggregate(pipeline).to_list()
    except pymongo.errors.OperationFailure as e:
        return {
            "status": "error",
            "error_message": f"Cannot get priority analysis: {e}",
        }

    print(f"sprints: {documents}")
    return {"status": "success", "sprints": documents}


# Connect to MongoDB cluster
mongo_client = MongoClient(
    MONGODB_URI,
    connectTimeoutMS=MONGODB_CONNECTION_TIMEOUT_MS,
    serverSelectionTimeoutMS=MONGODB_SERVER_SELECTION_TIMEOUT_MS,
)

db = mongo_client[MONGODB_DB_NAME]
collection = db[MONGODB_COLLECTION_NAME]


# Open API
openai_client = OpenAI()


# Retry configuration for Google GenAI
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,  # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


GOOGLE_CLOUD_PROJECT = "project-52f530b5-d60c-4b3c-bd0"
os.environ["GOOGLE_CLOUD_PROJECT"] = GOOGLE_CLOUD_PROJECT

GOOGLE_CLOUD_LOCATION="us-central1"

## Deployed agent id
DEPLOYED_AGENT_ID="1754130064628252672"

# Session
session_service = DatabaseSessionService(db_url=SESSION_DB_URL)

# Memory
memory_service = VertexAiMemoryBankService(
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
    agent_engine_id=DEPLOYED_AGENT_ID
)

# Helper for saving session into memory
async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )


analyze_risk_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="AnalyzeRiskAgent",
    instruction=""" 
You perform the analysis of Jira issues through **get_risk_analysis** tool, such as blocking issues, slow-moving issues (in-progress status too long).

What you can do:
1. You can check slow-moving issues (in-progress status too long) based on the information obtained by **get_risk_analysis** tool. You can use source status, target status, and transition days (number of days that the issue took to move from source status to target status).
2. You can check blocking issues based on the information obtained by **get_risk_analysis** tool. You can use source status, target status, and start date (from date). A blocking issue is an issue where source status is equals to target status, and start date (from date) is very old compared with the current date.

Your Workflow:
Always follow these steps precisely!
1. Send a LIST of all Jira issue keys the user wants to analyze to **get_risk_analysis** tool to get the status transitions for each Jira issue.
2. Use the following information to answer user's question:
    2.1 Use the Jira issue information retrieved in the parent agent **RootAgent** through **find_similar_task_or_issue** tool.
    2.2 Use the Jira issue status transition information retrieved in the step 1 that includes: source status, target status, transition days, start date (from date), and end date (to date).
    """,
    tools=[get_risk_analysis],
)


analyze_priority_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="AnalyzePriorityAgent",
    instruction=""" 
You perform priority analysis of Jira issues through **get_priority_analysis** tool, such as which to pull into next sprint and bring information related to the sprint.

What you can do:
1. You can check which to pull into next sprint based on:
    1.1 The retrieved Jira issue information in the parent agent **RootAgent** through **find_similar_task_or_issue** tool. You can use fields priority and status of Jira issue.
    1.2 The retrieved Jira issue sprints information from **get_priority_analysis** tool. You can use name, state, start date, end date, goal.
2. You can answer questions if they are related to issue sprint using information from **get_priority_analysis** tool.

Your Workflow:
Always follow these steps precisely!
1. Send a LIST of all Jira issue keys the user wants to analyze to **get_priority_analysis** tool to get sprint information for each Jira issue.
2. Use the following information to answer user's question:
    2.1 Use the Jira issue information retrieved in the parent agent **RootAgent** through **find_similar_task_or_issue** tool.
    2.2 Use the Jira issue sprints information retrieved in the step 1 that includes the following information for each Jira issue related to sprints: name, state, start date, end date, goal.  
    """,
    tools=[get_priority_analysis],
)


root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="RootAgent",
    instruction=""" 
You are the **Software Development Sprint Planning Agent**, a friendly and helpful virtual assistant for agile software development teams to help them for sprint planning. 
Start every conversation with a warm greeting, introduce yourself as the "Software Development Sprint Planning Agent", and ask how you can assist the user today. 
Your role is to help agile software development teams for sprint planning.
Yo have access to Jira repository through **find_similar_task_or_issue** tool.

What you can do:
1. Help users discover and explore reported tasks and issues in the Jira repository.
2. Help users on agile planning sessions guiding priority setting, estimation, workload balancing, risk identification and mitigation using a information in the Jira repository.

Your Workflow:
Always follow these steps precisely!

1. **Retrieve information** for tasks or issues in Jira repository with a description semantically similar to the userâ€™s request. Use **find_similar_task_or_issue** tool for this.
2. **Analyze the user's question** to determine their specific intent. 
3. **If the user's intent determined in the previous step is related to risk** such as blocking issues, slow-moving issues (in-progress too long), use **analyze_risk_agent** agent tool with a LIST of all ticket keys the user wants to analyze, obtained in the step one.
4. **If the user's intent determined in the previous step is related to priority or sprint** such as which to pull into next sprint or any other question related to the sprint, use **analyze_priority_agent** agent tool with a LIST of all ticket keys the user wants to analyze, obtained in the step one.
5. **Create a concrete response for the user** based on what was retrieved from the tool **find_similar_task_or_issue**, and agent tools **analyze_risk_agent** and **analyze_priority_agent**. 

Available tools:
1. **find_similar_task_or_issue**: Searches for tasks or issues in Jira repository with a description semantically similar to the userâ€™s request.

Available agent tools:
1. **analyze_risk_agent**: Analyze issues risk such as blocking issues, slow-moving issues (in-progress too long) with a LIST of all ticket keys the user wants to analyze.
2. **analyze_priority_agent**: Analyze issues priority such as which to pull into next sprint or any other question related to the sprint, with a LIST of all ticket keys the user wants to analyze.

Core guidelines:
- **Always search first**: If a user asks for a task or issue, call `find_similar_task_or_issue`.  
- **Answer user's questions using data retrieved**: Create a concrete response for the user based on what was retrieved from the tool **find_similar_task_or_issue** and agent tool **analyze_risk_agent** if it is risk related.
- **Handle missing tasks or issues**: If the requested task or issue is not in the repository, suggest similar tasks or issues returned by the search.  
- **Parallel tool use**: You may call multiple tools in parallel when appropriate (e.g., searching for several tasks or issues at once).  
- **Clarify only when necessary**: Ask for more details if the request is unclear and you cannot perform a search.  
- Keep your tone positive, approachable, and customer-focused throughout the interaction.  

Additional important instructions:
- **Multi-item requests**: If the user asks for several tasks or issues in one message, search for all items together.  
- **Fallback behavior**: If no results are found, apologize politely, and encourage the user to try a different task or issue.  
- **Stay focused**: Only handle tasks or issues discovery, and planning sessions guidance. Politely decline requests unrelated to tasks or issues.  
- **Answering task or issue questions**: If the question is about a task or issue (e.g., "Is there any task related to improve documentation and who worked on them?"), use the tools results and agent tools results to answer. If the information is not available, respond transparently that you donâ€™t have that detail.  

Remember: you are a professional yet friendly Software Development Sprint Planning assistant whose goal is to help users for tasks planning in a smooth, efficient, and enjoyable way.

    """,
    tools=[
        find_similar_task_or_issue,
        AgentTool(analyze_risk_agent),
        AgentTool(analyze_priority_agent),
        preload_memory
    ],
    after_agent_callback=auto_save_to_memory,  # Saves after each turn!
)


runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service
)



await run_session(runner, "hello, I am Cristian, â� Is there any task related to improve documentation and who worked on them? ", "conversation-01")


await run_session(runner, "Is there any task related to improve performance and which project does it belong to?")


await run_session(runner, "Show me the issues related to improve documentation that are slow-moving issues")


await run_session(runner, "Which issues related to vector search can be pull into the next sprint?")



await run_session(runner, "hello, what is my name?", "conversation-02")


# GOOGLE_CLOUD_PROJECT="project-52f530b5-d60c-4b3c-bd0"
# GOOGLE_CLOUD_LOCATION="us-central1"

# os.environ["GOOGLE_API_KEY"] = None
# os.environ.pop('GOOGLE_API_KEY', None)

# Initialize Vertex AI
vertexai.init()

# Get the most recently deployed agent
agents_list = list(agent_engines.list())
if agents_list:
    remote_agent = agents_list[0]  # Get the first (most recent) agent
    client = agent_engines
    print(f"âœ… Connected to deployed agent: {remote_agent.resource_name}")
else:
    print("â�Œ No agents found. Please deploy first.")


async for item in remote_agent.async_stream_query(
    message="What is my name?",
    user_id="cvertiz",
):
    print(item)


async for item in remote_agent.async_stream_query(
    message="Can you show me the issues related to improve documentation that are slow-moving issues?",
    user_id="user_42",
):
    print(item)




