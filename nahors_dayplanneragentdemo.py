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


import logging
import os

# Logging configuration
LOG_FILE = os.path.join(os.getcwd(), "logger.log")
LOG_LEVEL = logging.DEBUG
FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
STREAM_HANDLER_CONSOLE = True


def configure_logging():
    """Idempotently configure root logging to write to a file and console.

    This function can be imported safely multiple times. It will not remove
    existing handlers or truncate existing logs; instead it ensures a file
    handler writing to LOG_FILE exists so log records from all modules are
    captured.
    """
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    # Add a FileHandler if one writing to LOG_FILE isn't present yet
    file_handler_exists = False
    for h in list(root.handlers):
        try:
            if isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(LOG_FILE):
                file_handler_exists = True
                break
        except Exception:
            # Some handlers may not have baseFilename attribute
            continue

    if not file_handler_exists:
        fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(fh)

    # Ensure there's at least one console handler (StreamHandler) for interactive runs
    stream_exists = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not stream_exists:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(sh)

    # Redirect stdout/stderr to logging so print() output is captured in the logs.
    try:
        import sys

        class StreamToLogger:
            """Fake file-like object that redirects writes to a logger instance."""

            def __init__(self, logger, level=logging.INFO):
                self.logger = logger
                self.level = level

            def write(self, buf):
                for line in buf.rstrip().splitlines():
                    self.logger.log(self.level, line)

            def flush(self):
                pass

        stdout_logger = logging.getLogger('STDOUT')
        stderr_logger = logging.getLogger('STDERR')
        # Only redirect if not already redirected
        if not isinstance(sys.stdout, StreamToLogger) and not STREAM_HANDLER_CONSOLE:
            sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
        if not isinstance(sys.stderr, StreamToLogger) and not STREAM_HANDLER_CONSOLE:
            sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
    except Exception:
        # If redirection fails, don't crash the import
        root.exception('Failed to redirect stdout/stderr to logging')


# Configure on import; this ensures other modules that import the package
# get a consistent logging setup. Because configure_logging is idempotent,
# repeated imports won't duplicate handlers.
configure_logging()

print("Logging configured (file:", LOG_FILE, ")")



from google.genai import types
from google.adk.models.google_llm import Gemini
import os
from google.adk.sessions.database_session_service import DatabaseSessionService
from dotenv import load_dotenv
load_dotenv() # Load environment variables from a .env file if present
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
# --- Define Model Constants for easier use ---
AGENT_MODEL = os.getenv("MODEL_NAME","gemini-2.5-flash")
LLM_AS_JUDEGE_MODEL = os.getenv("LLM_AS_JUDGE_MODEL","gemini-2.5-flash")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)
base_model=Gemini(model=AGENT_MODEL, retry_options=retry_config)





import sqlite3
import os
import logging
from datetime import date, datetime
from dotenv import load_dotenv
load_dotenv("../.env")  # Load environment variables from a .env file if present
logger = logging.getLogger(__name__)

DEFAULT_DB = os.getenv('EVENTS_DB_PATH', os.path.join(os.getcwd(), 'events_db.sqlite'))

CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS future_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_date DATE,
    event_type TEXT,
    event_details TEXT
);
'''
CREATE_TABLE_RECURRING_SQL = '''
CREATE TABLE IF NOT EXISTS recurring_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 event_frequency TEXT NOT NULL,
 event_start_date DATE NOT NULL,
 event_type TEXT NOT NULL, 
 event_details TEXT,
 event_end_date DATE
);
'''
def init_events_db(db_path: str | None = None) -> str:
    """Create the SQLite database and  tables if not present.

    Returns the path to the database file.
    """
    db_path = db_path or DEFAULT_DB
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        cur.execute(CREATE_TABLE_RECURRING_SQL)
        conn.commit()
        logger.info('Initialized events DB at %s', db_path)
    finally:
        conn.close()
    return db_path

def add_recurring_event(event_frequency: str, event_start_date: date | str, event_type: str, event_details: str | None = None, event_end_date: date | str | None = None, db_path: str | None = None) -> int:
    """Insert a row into the recurring_events table.

    Returns the inserted row id.
    """
    db_path, conn = getConnection(db_path)
    try:
        cur = conn.cursor()
        # Accept datetime.date/datetime objects or strings. Store as ISO YYYY-MM-DD
        logger.debug(f"Adding recurring event: freq={event_frequency}, start_date={event_start_date}, type={event_type}, end_date={event_end_date} ,event_details={event_details}")    
        if isinstance(event_start_date, datetime):
            start_date_str = event_start_date.date().isoformat()
        elif isinstance(event_start_date, date):
            start_date_str = event_start_date.isoformat()
        else:
            # assume string-like
            start_date_str = str(event_start_date)

        if event_end_date is not None:
            if isinstance(event_end_date, datetime):
                end_date_str = event_end_date.date().isoformat()
            elif isinstance(event_end_date, date):
                end_date_str = event_end_date.isoformat()
            else:
                end_date_str = str(event_end_date)
        else:
            end_date_str = None

        cur.execute(
            "INSERT INTO recurring_events (event_frequency, event_start_date, event_type, event_details, event_end_date) VALUES (?, ?, ?, ?, ?)",
            (event_frequency, start_date_str, event_type, event_details, end_date_str),
        )
        conn.commit()
        rowid = cur.lastrowid
        logger.info('Inserted recurring event id %s into %s', rowid, db_path)
        return rowid
    except Exception as e:
        logger.error('Error inserting recurring event into %s: %s', db_path, e.with_traceback())
        raise e
    finally:
        conn.close()    
    
def fetch_recurring_events(start_date:str,end_date:str) -> list[dict]:
    """Fetch recurring events between start_date in YYYY-MM-DD format and end_date in YYYY-MM-DD format.

    Returns a list of dictionaries with keys: id, event_frequency, event_start_date, event_type, event_details, event_end_date.
    """
    db_path, conn = getConnection(None)
    try:
        cur = conn.cursor()
        # accept date object or string
        if start_date is None:
            start_date=datetime.now().date()
        if end_date is None:
            end_date=datetime.now().date()

        if isinstance(start_date, datetime):
            start_date_str = start_date.date().isoformat()
        elif isinstance(start_date, date):
            start_date_str = start_date.isoformat()
        else:
            start_date_str = str(start_date)

        if isinstance(end_date, datetime):
            end_date_str = end_date.date().isoformat()
        elif isinstance(end_date, date):
            end_date_str = end_date.isoformat()
        else:
            end_date_str = str(end_date)

        cur.execute("SELECT id, event_frequency, event_start_date, event_type, event_details, event_end_date FROM recurring_events WHERE event_start_date <= ? AND (event_end_date IS NULL OR event_end_date >= ?) ORDER BY event_start_date, id", (end_date_str, start_date_str))
        rows = cur.fetchall()

        result = []
        for r in rows:
            # convert stored date strings back to date objects when possible
            start_stored = r[2]
            end_stored = r[5]
            try:
                stored_start_date = datetime.strptime(start_stored, "%Y-%m-%d").date()
            except Exception:
                stored_start_date = start_stored
            if end_stored is None:
                stored_end_date = None
            else:
                try:
                    stored_end_date = datetime.strptime(end_stored, "%Y-%m-%d").date()
                except Exception:
                    stored_end_date = end_stored

            result.append({
                'id': r[0],
                'event_frequency': r[1],
                'event_start_date': stored_start_date,
                'event_type': r[3],
                'event_details': r[4],
                'event_end_date': stored_end_date,
            })
        logger.info('Fetched %d recurring events from %s between %s and %s', len(result), db_path, start_date, end_date)
        return result
    except Exception as e:
        logger.error('Error fetching recurring events from %s: %s', db_path, e.with_traceback())
        raise e 
    finally:
        conn.close()



def add_future_event(event_date: date | str | None, event_type: str, event_details: str | None = None, db_path: str | None = None) -> int:
    """Insert a row into the future_events table.

    Returns the inserted row id.
    """
    db_path, conn = getConnection(db_path)
    try:
        cur = conn.cursor()
        # Accept datetime.date/datetime objects or strings. Store as ISO YYYY-MM-DD
        if isinstance(event_date, datetime):
            event_date_str = event_date.date().isoformat()
        elif isinstance(event_date, date):
            event_date_str = event_date.isoformat()
        elif event_date is None:
            # maintain previous behavior: empty string when no date provided
            event_date_str = ''
        else:
            # assume string-like
            event_date_str = str(event_date)

        cur.execute(
            "INSERT INTO future_events (event_date, event_type, event_details) VALUES (?, ?, ?)",
            (event_date_str, event_type, event_details),
        )
        conn.commit()
        rowid = cur.lastrowid
        logger.info('Inserted event id %s into %s', rowid, db_path)
        return rowid
    except Exception as e:
        logger.error('Error inserting event into %s: %s', db_path, e.with_traceback)
        raise e
    finally:
        conn.close()

def getConnection(db_path):
    db_path = db_path or DEFAULT_DB
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    return db_path,conn


def fetch_date_events(date_param: str,db_path=None) -> list[dict]:
    """Fetch events for a specific date (datetime.date or YYYY-MM-DD string) or all events if date is None.

    Returns a list of dictionaries with keys: id, event_date (datetime.date or None), event_type, event_details.
    """
    db_path, conn = getConnection(db_path)
    try:
        cur = conn.cursor()
        if date is None:
            cur.execute("SELECT id, event_date, event_type, event_details FROM future_events ORDER BY event_date, id")
            rows = cur.fetchall()
        else:
            # accept date object or string
            if isinstance(date_param, datetime):
                date_str = date_param.date().isoformat()
            elif isinstance(date_param, date):
                date_str = date_param.isoformat()
            else:
                date_str = str(date_param)
            cur.execute("SELECT id, event_date, event_type, event_details FROM future_events WHERE event_date = ? ORDER BY id", (date_str,))
            rows = cur.fetchall()

        result = []
        for r in rows:
            # convert stored date string back to date object when possible
            stored = r[1]
            if stored is None or stored == '':
                stored_date = None
            else:
                try:
                    stored_date = datetime.strptime(stored, "%Y-%m-%d").date()
                except Exception:
                    # fallback: return raw string if parsing fails
                    stored_date = stored
            result.append({
                'id': r[0],
                'event_date': stored_date,
                'event_type': r[2],
                'event_details': r[3],
            })
        logger.info('Fetched %d events from %s for date=%s', len(result), db_path, date_param)
        return result
    finally:
        conn.close()



    path = init_events_db()
    print('DB initialized at', path)




import os
import asyncio
from google.adk.agents import Agent,BaseAgent,SequentialAgent
import google.adk as adk
import json
from datetime import date, datetime
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.tools import FunctionTool
from google.adk.events.event_actions import EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator
from google.genai import types
from dotenv import load_dotenv
load_dotenv("../.env")  # Load environment variables from a .env file if present
import logging
logger = logging.getLogger(__name__)
# Initialize events DB unless explicitly disabled
if os.getenv('INIT_EVENTS_DB', '1') not in ('0', 'false', 'False'):
  EVENTS_DB_PATH = os.getenv('EVENTS_DB_PATH', os.path.join(os.getcwd(), 'events_db.sqlite'))
  init_events_db(EVENTS_DB_PATH)
# --- Define Model Constants for easier use ---
AGENT_MODEL = os.getenv("MODEL_NAME")
prompt_template = """You are an event classification assistant. 
Your task is to read a short text describing an activity, occasion, or message, 
and classify it into one of the following categories:

- Birthday
- Anniversary
- Holiday (Christmas, New Year, Diwali, etc.)
- Shopping
- Meeting
- Festival
- Personal Greeting
- Work/Task
- Other

Rules:
1. Focus on the main intent of the text.
2. If the text mentions a birthday â†’ Birthday.
3. If the text mentions an anniversary â†’ Anniversary.
4. If the text mentions a holiday (Christmas, New Year, Diwali, etc.) â†’ Holiday.
5. If the text is about buying, shopping, or purchasing â†’ Shopping.
6. If the text is about a meeting, appointment, or scheduled gathering â†’ Meeting.
7. If the text is about a cultural/religious festival â†’ Festival.
8. If the text is a general greeting or well-wishing not tied to a birthday/holiday â†’ Personal Greeting.
9. If the text is about work, tasks, or professional duties â†’ Work/Task.
10. If none of the above apply â†’ Other.
11. If a date is mentioned in the text, extract it in YYYY-MM-DD format; otherwise, return null for the date.
12.Determine if the event is recurring or one-time based on the text. If recurring, specify the frequency (e.g., daily, weekly, monthly, yearly).
15.Present output in JSON format as follows:
Output format: 
{
  "event_text": "<original input text>",
  "category": "<one of the categories above>",
  "start_date": "<YYYY-MM-DD if present, else null>",
  "end_date": "<YYYY-MM-DD if present for recurring events, else null>",
  "is_recurring": "<True/False>",
  "event_date": "<YYYY-MM-DD if present, else null>",
  "recurrence_frequency": "<if recurring, specify frequency like daily, weekly, monthly, yearly; else null>"
}

"""
event_fetching_prompt = """
You are a calendar assistant that retrieves events for a specific date.
Given a date in YYYY-MM-DD format, fetch all events scheduled for that date from the database.
You have two types of events to consider: one-time events and recurring events.
When fetching events, follow these steps:
1. Query the one-time events table for events matching the given date.
2. Query the recurring events table for events that recur on the given date based on their frequency.
3. Combine the results from both queries.
4.Summarize the Output and present in text format as follows:
Dont show date if event_details is null or Details is empty.
- If events are found, list them under the date header.
- For one-time events, use the following format.
  Event Date: <event_date>
  - Event Type: <event_type>  
    - Details: <event_details>
- For recurring events, Analyse each event's frequency and determine if it occurs on the given date.
  - If it does, use the following format:
    Recurring Event (Frequency: <event_frequency>)
    - Event Date: <event_date>
    - Event Type: <event_type>
      - Details: <event_details>
5.If no events are found , return "No Events are present for this period".
"""
class StoreEvent(BaseAgent):
  async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
     json_details = ctx.session.state.get("event_classification_response")
     # Call your Python function with that value
     cleaned = clean_json_block(json_details)
     logger.info(f"Storing event with details: {cleaned}")
     result = add_event_wrapper(cleaned)
        
        # Yield the result back into the agent system
     yield Event(
    author=self.name,
    content=types.Content(role='user', parts=[types.Part(text=result)]),
    actions=EventActions()
  )


import json
import re

def clean_json_block(raw: str) -> str:
    # Remove triple backticks and optional language tag
    return re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()


     
def add_event_wrapper(details)-> str:
    """Wrapper for add_future_event to be used as a tool in the agent."""
    if not details or not details.strip():
        raise ValueError("Empty or missing event details")

    try:
      logger.debug(f"Adding event with details: {details}")
      dict=json.loads(details)
      event_date=dict.get("event_date",None)
      event_date=datetime.strptime(event_date,"%Y-%m-%d").date() if event_date else None
      event_type=dict.get("event_type")
      if not event_type:
          event_type=dict.get("category","Other")
      event_details=dict.get("event_details",None)
      if not event_details:
          event_details=dict.get("event_text")
      is_recurring=dict.get("is_recurring",False)
      if is_recurring==True:
          logger.debug("Event is recurring, adding to recurring_events table")
          # For recurring events, we might need additional fields like frequency and end_date
          recurrence_frequency=dict.get("recurrence_frequency",None)
          end_date=dict.get("end_date",None)
          end_date=datetime.strptime(end_date,"%Y-%m-%d").date() if end_date else None
          logger.debug(f"Before call to db :Recurrence frequency: {recurrence_frequency}, End date: {end_date}")
          row_id=add_recurring_event(recurrence_frequency,event_date,event_type,event_details,end_date)
          if row_id>0:
            return f"Recurring event added with id {row_id}"
      else:
        logger.debug("Event is one-time, adding to future_events table")
        row_id= add_future_event(event_date, event_type, event_details)
        if row_id>0:
            return f"One-time event added with id {row_id}"
    except Exception as e:
        logger.error(f"Error adding event:", exc_info=True)
    return "Failed to add event."
      
      

def get_event_classifier_agent():
    return Agent(
        model=base_model,
        name="calendar_agent",
        description="An agent that helps manage calendar events.IT takes the event and provide details about it.Is the event recurring or one time.Is the date fixed or variable.It classifies the event.as Anniversery,Festiwal,Work etc " \
        "",
        instruction=prompt_template,
        output_key="event_classification_response",

    ) 
def get_event_asisstant_agent():
    return Agent(
    model=base_model,
    name="calendar_assisstant",
    description="This agents provides the calendar details for the specific date",
    instruction=event_fetching_prompt,
    tools=[FunctionTool(fetch_date_events_wrapper),FunctionTool(fetch_recurring_events_wrapper)],
    output_key="event_fetching_response",

)
def fetch_date_events_wrapper(date_param:str) -> json:
    """Fetch events for a specific date (datetime.date or YYYY-MM-DD string) or all events if date is None.

    Returns json object in following format  {event_date :[{events_by_type:{event_type:[{event_date,event_details}]}}]
    """
    ret=fetch_date_events(date_param)
    if not ret or len(ret)==0:
        return {"event_date":date_param,"events_by_type":{}}
    else:
        events_by_type={}
        for r in ret:
            etype=r.get("event_type")
            edetails=r.get("event_details")
            edate=r.get("event_date")
            event_record={"event_date":edate,"event_details":edetails}
            if etype in events_by_type:
                events_by_type[etype].append(event_record)
            else:
                events_by_type[etype]=[event_record]
        return json.dumps({"event_date":date_param,"events_by_type":events_by_type},default=safe_json)
def fetch_recurring_events_wrapper(start_date:str,end_date:str) -> json:
    """Fetch recurring events between start_date in YYYY-MM-DD format and end_date in YYYY-MM-DD format.

    Returns json object in following format  {recurring_events :[{id,event_frequency,event_start_date,event_type,event_details,event_end_date}]}
    """
    ret=fetch_recurring_events(start_date,end_date)
    if not ret or len(ret)==0:
        return {"recurring_events":[]}
    else:
        return json.dumps({"recurring_events":ret},default=safe_json) 
    
def safe_json(obj):
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, datetime):
        # Return only the date part in ISO format
        return obj.date().isoformat()   # "YYYY-MM-DD"
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    if isinstance(obj, (Exception,str)):
        return str(obj)

    raise TypeError(f"Type {type(obj)} not serializable")
def get_event_aggregator_agent():
    return Agent(
        model=base_model,
        name="event_aggregator_agent",
        description="An agent that aggregates events published by StoreEvent agent.",  # Description of the agent's purpose
        instruction="""Summarize all events created by Agents sharing the same Parent as You .
        Provide a concise summary.
        """,
        output_key="event_aggregation_response",

    )

def get_event_storing_agent():
    return SequentialAgent(
   name="event_storing_agent",
   description="An agent that classifies and stores calendar events into the database.",
   sub_agents=[get_event_classifier_agent(),StoreEvent(name="StoreEvent"),get_event_aggregator_agent()],)
   


# @title Import necessary libraries
import os
import uuid
import asyncio
from google.adk.agents import Agent,SequentialAgent,ParallelAgent
from google.adk.tools import google_search

from google.adk.runners import Runner,InMemoryRunner
from dotenv import load_dotenv

load_dotenv("../.env")  # Load environment variables from a .env file if present
from google.genai import types # For creating message Content/Parts

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger(__name__)
logger.info("Libraries imported.")
# Keep external package verbosity lower
logging.getLogger("google.adk").setLevel(logging.INFO)








# @title Define Agent Interaction Function

from google.genai import types # For creating message Content/Parts

# Define helper functions that will be reused throughout the notebook

instruction= """You are a Routing Manager Agent .Your task is to help users manage their calendar events. 
If user_task_response in state is store_event or store_and_fetch,
You will Route request to event_storing_agent to classify and store events.
If user_task_response in state is fetch_event or store_and_fetch,
You will Route request to calendar_assisstant to fetch events for specific dates.
You will Route request to calendar_assisstant to fetch events for specific dates.
Here are the sub agents you can use:
event_storing_agent: An agent that classifies events into categories and determines if they are one-time or recurring.
calendar_assisstant: An agent that provides calendar details for a specific date.
Rules:
1.If the date or date range is in the past,reply that the event is in the past and cannot be fetched.
2.If storing events,ensure to capture the event details,type,date(if provided),and recurrence information.
3.If the User asks for events for a period like a week or month,use calendar_assisstant tool to fetch events for each date in that period and consolidate the results.
4.If No event are found reply with No events found for the given date.
Output Format: 
{
   [ "date": "<YYYY-MM-DD>",
  "events_by_type": {
    "<event_type>": [
      {
        "id": <event_id>,
        "event_date": "<YYYY-MM-DD>",
        "event_details": "<details>"
      },
      ]
      }
    ]
    """

date_determination_prompt="""
You are a helpful assistant that determines dates. Given a user query about calendar events, identify the specific date or date range mentioned.
Rules:
-PLease follow the below rules strictly:
-The date range should always be greated or equal to today's date.
-Classify if The event is a single date or a date range.
-Identify dates mentioned in various formats (e.g., "next Monday", "December 25th", "from Jan 1 to Jan 7").
-Consider relative dates based on the current date (e.g., "tomorrow", "next week").
-If it is a festival or holiday,event date should be that of the festival/holiday.
-You can use tool only to determine the date using following query ."What is the date for New Year 2025?"
-You cannot fetch events using the tool.

Output the result in JSON format as follows:
{
    
    "tool_code:"Yes",
    "date": "<YYYY-MM-DD if single date>",
    "start_date": "<YYYY-MM-DD if date range>",
    "end_date": "<YYYY-MM-DD if date range>"
"""
def get_date_determination_agent():
    """It returns date determination agent which can take text and return date or date range."""
    mod = Agent(
    model=base_model,
    name="date_determination_agent",
    description="An agent that identifies specific dates or date ranges from user queries about calendar events.",
    instruction=date_determination_prompt,
    tools=[google_search],
    output_key="date_determination_response",
)
    return mod
def get_User_Task_Agent():
    userAgent=Agent(
    model=base_model,
    name="user_task_agent",
    description="An agent that helpsto classify user task as storing event or fetching event or both ",
    output_key="user_task_response",
    instruction="""You are a helpful assistant that classifies user tasks related to calendar events.
Given a user query about calendar events, determine whether the user wants to store a new event, fetch existing events, or both.
Rules:
1. If the user query includes details about a new event (e.g., event description, date, recurrence), classify it as "store_event".
2. If the user query requests information about existing events (e.g., "What events do I have on...?", "List my events for..."), classify it as "fetch_event".
3. If the user query includes both storing and fetching requests, classify it as "store_and_fetch".
Output the result in Tools format as follows:
{   
    "tool_code:"Yes",
    "task_type": "<store_event | fetch_event | store_and_fetch>"
}
"""
)
    return userAgent
def get_Parallel_agent():
    return ParallelAgent(
    name="user_task_date_determination_agent",
    sub_agents=[get_User_Task_Agent(),get_date_determination_agent()],
)

manager_agent = Agent(
    model=base_model,
    name="Manager_Agent",
    description="A text chatbot helps user with events management.It aggregates and prepare date for final summary",  # Description of the agent's purpose
    instruction=instruction,
    output_key="root_agent_response",
    sub_agents=[get_event_storing_agent(),get_event_asisstant_agent()],
    

)
def get_Summary_agent():
    return Agent(
        model=base_model,
        name="Summary_Agent",
        description="An agent that summarizes the events fetched for the user",  # Description of the agent's purpose
        instruction=""" You are a helpful assistant that summarizes calendar events for the user.
        Your task is to  evalaute Evaluate root_agent_response,event_aggregation_response and event_fetching_response if present.
         Provide a concise summary of the events.
        """,
        output_key="root_agent_response",
        )
def get_flow_agent():
    return SequentialAgent(
        name="multi_tool_manager_agent",
        description="A text chatbot helps user with events management",  # Description of the agent's purpose
        sub_agents=[get_Parallel_agent(),manager_agent,get_Summary_agent()],
    )







  


agent=get_flow_agent()
eval_cases = [
    {"input": "I have a birthday party to attend next tuesday", "expected": "event added"},
    {"input": "what are my events  next week", "expected": "birthday"},
    {"input": "Add an alarm to take my medcine at 9 PM for next one week", "expected": "recurring event"},
]


async def run_eval_cases(eval_cases, agent):
    runner = InMemoryRunner(agent=agent, app_name="agents")
    results = []
    for case in eval_cases:
        output = await runner.run_debug(case["input"])

        texts = []
        for event in output:
            if event.content and event.content.parts:
                texts.extend([
                    part.text
                    for part in event.content.parts
                    if part.text and part.text.strip().lower() != "none"
                ])
        text = " \n".join(texts)

        results.append({
            "input": case["input"],
            "output": text,
            "expected": case["expected"],
            "pass": case["expected"].lower() in text.lower()
        })
    return results

async def run_evals():
    results = await run_eval_cases(eval_cases, agent)
    for r in results:
        print(f"Input: {r['input']}")
        print(f"Output: {r['output']}")
        print(f"Expected: {r['expected']}")
        print(f"Pass: {r['pass']}")
        print("-" * 40)
    return results
async def main():
    results = await run_eval_cases(eval_cases, agent)
    judged = []
    for case in results:
        verdict = await judge_case(case)
        judged.append({**case, "verdict": verdict})
    for r in judged:
        print(r)   

judge_agent= Agent(
        model=base_model,
        name="judge_agent",
        description="An agent that judges whether the agent output meets the expected behavior.",
        )
async def judge_case(case):

    judge_prompt = f"""
    You are an evaluator. Given the agent's output and the expected behavior,
    decide if the output satisfies the expectation.

    Input: {case['input']}
    Expected: {case['expected']}
    Output: {case['output']}

    Answer only with PASS or FAIL, and a oneâ€‘sentence justification.
    """


    runner = InMemoryRunner(agent=judge_agent, app_name="judge")
    output = await runner.run_debug(judge_prompt)

    # Collect text parts from the events
    texts = []
    for event in output:
        if event.content and event.content.parts:
            texts.extend([
                part.text
                for part in event.content.parts
                if part.text and part.text.strip().lower() != "none"
            ])
    return " ".join(texts)


await main()

