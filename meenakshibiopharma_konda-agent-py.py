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


!pip install fastapi
!pip install uvicorn
!pip install pydantic
!pip install aiohttp
!pip install nest_asyncio
!pip install google-genai
!pip install tqdm
!pip install pandas
!pip install pytesseract
!pip install Pillow
!pip install qrcode
!pip install python-multipart
!pip install requests
!pip install python-dotenv
!pip install adk





! pip install google adk


# agent/adk_emulator.py
import os, asyncio, base64, logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger("KONDA-ADK")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Try to import genai
GENAI_AVAILABLE = False
try:
    from google import genai
    GENAI_AVAILABLE = True
    logger.info("google-genai available")
except Exception:
    logger.info("google-genai not available, falling back to deterministic behavior")

@dataclass
class Part:
    mime_type: Optional[str] = None
    text: Optional[str] = None
    inline_data: Optional[str] = None
    @staticmethod
    def from_text(text: str): return Part(mime_type="text/plain", text=text)
    @staticmethod
    def from_inline_data(data: str, mime_type: str="image/png"): return Part(mime_type=mime_type, inline_data=data)

@dataclass
class Event:
    text: Optional[str]=None
    tool_call: Optional[Dict[str,Any]]=None
    tool_result: Optional[Dict[str,Any]]=None

class FunctionTool:
    def __init__(self, func: Callable, name: Optional[str]=None):
        self.func = func
        self.name = name or getattr(func, "__name__", "function_tool")
    async def call(self, *args, **kwargs):
        res = self.func(*args, **kwargs)
        if asyncio.iscoroutine(res):
            res = await res
        return res

class InMemorySessionService:
    def __init__(self):
        self._sessions: Dict[str, Dict[str,Any]] = {}
    async def get_session(self, app_name: str, user_id: str) -> Dict[str,Any]:
        key = f"{app_name}:{user_id}"
        return self._sessions.setdefault(key, {})
    async def set_session_state(self, app_name: str, user_id: str, state: Dict[str,Any]):
        key = f"{app_name}:{user_id}"
        self._sessions[key] = state

# Basic genai wrapper (safe)
def genai_generate(prompt: str, model: str="gemini-2.5-flash", max_output_tokens: int=512) -> str:
    if not GENAI_AVAILABLE:
        raise RuntimeError("genai not available")
    api_key = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = None
    try:
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
    except Exception:
        client = genai.Client()
    # Try client.generate_text or client.generate
    try:
        if hasattr(client, "generate_text"):
            resp = client.generate_text(model=model, input=prompt, max_output_tokens=max_output_tokens)
            # adapt response shapes
            if hasattr(resp, "text"): return resp.text
            if isinstance(resp, dict) and 'candidates' in resp: return resp['candidates'][0].get('content','')
            return str(resp)
    except Exception:
        pass
    try:
        if hasattr(client, "generate"):
            resp = client.generate(model=model, text=prompt)
            if hasattr(resp, "candidates") and len(resp.candidates)>0:
                c = resp.candidates[0]
                return getattr(c, 'content', str(c))
            return str(resp)
    except Exception:
        pass
    raise RuntimeError("genai call failed")

# Minimal LlmAgent class (deterministic or genai-enabled)
class LlmAgent:
    def __init__(self, name:str, instruction:str, tools:Optional[List[FunctionTool]]=None, input_keys:Optional[List[str]]=None, output_keys:Optional[List[str]]=None, model:str="gemini-2.5-flash"):
        self.name=name; self.instruction=instruction
        self.tools = {t.name: t for t in (tools or [])}
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.use_real_llm = False
        self.model = model
    async def run(self, session_state: Dict[str,Any], parts: List[Part], runner_events: List[Event]):
        logger.info(f"[{self.name}] run start")
        if self.use_real_llm and GENAI_AVAILABLE:
            # Basic prompt and call; for production parse structured plan JSON
            prompt = self.instruction + "\n\nContext:\n"
            for k in self.input_keys:
                prompt += f"{k}: {session_state.get(k)}\n"
            for p in parts:
                if p.text: prompt += "\nUser: " + p.text
            try:
                text_out = genai_generate(prompt, model=self.model)
                runner_events.append(Event(text=f"[{self.name}] Gemini: {text_out[:200]}"))
                # naive: if JSON in output and instructs a tool call, handle that (left to user customization)
            except Exception as e:
                runner_events.append(Event(text=f"[{self.name}] Gemini error: {e}"))
            return session_state
        # deterministic behavior implemented in orchestrator & specific agents
        runner_events.append(Event(text=f"[{self.name}] deterministic noop"))
        return session_state

class SequentialAgent:
    def __init__(self, name:str, sub_agents:List[LlmAgent], description:Optional[str]=None):
        self.name=name; self.sub_agents=sub_agents; self.description=description

class InMemoryRunner:
    def __init__(self, agent:SequentialAgent, app_name:str, session_service:InMemorySessionService, api_key:str=""):
        self.agent=agent; self.app_name=app_name; self.session_service=session_service; self.api_key=api_key
    async def run_async(self, user_id:str, new_message:List[Part]):
        state = await self.session_service.get_session(self.app_name, user_id)
        # discharge to sub agents
        for sub in self.agent.sub_agents:
            events = []
            st = await sub.run(state, new_message, events)
            state.update(st)
            for e in events:
                yield e
        await self.session_service.set_session_state(self.app_name, user_id, state)
        yield Event(text=f"[Runner] Completed keys: {list(state.keys())}")



# agent/tools.py
import asyncio, base64, json, uuid, os
from typing import Any, Dict, List
# Removed: from google.adk_emulator import FunctionTool

# Seva list
SEVA_LIST = [
    "Padmavati Kalyanotsavam",
    "Srinivasa Mangapuram Kalyanotsavam",
    "Srinivasa Divyanugraha Homam"
]

MOCK_TICKET_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDHBLDNn/qACwFgM3W/pX2gAAAABJRU5ErkJggg=="

def process_image_for_ocr(part) -> Dict[str,Any]:
    # Validate base64
    if not part or not getattr(part, "inline_data", None):
        return {"extracted_successfully": False}
    try:
        _ = base64.b64decode(part.inline_data)
        # Simulate successful OCR extraction
        extracted_data = {
            "ticket_id": "TTD-20250101-A9B3",
            "darshan_date": "2025-10-15",
            "pilgrim_name": "Srinivas Rao", # Mock extracted name
            "extracted_successfully": True
        }
        return extracted_data
    except Exception:
        return {"extracted_successfully": False}


import asyncio, base64, json, uuid, os
from typing import Any, Dict, List
# REMOVED: from agent.adk_emulator import FunctionTool, Part # This line caused the ModuleNotFoundError

# Assume FunctionTool and Part are available from previous cell execution (8mqJjEVfY3LU)
# Seva list
SEVA_LIST = [
    "Padmavati Kalyanotsavam",
    "Srinivasa Mangapuram Kalyanotsavam",
    "Srinivasa Divyanugraha Homam"
]

MOCK_TICKET_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDHBLDNn/qACwFgM3W/pX2gAAAABJRU5ErkJggg=="

def process_image_for_ocr(image_part: Part) -> Dict[str,Any]:
    """
    CUSTOM TOOL 1: Simulates processing an image for OCR to extract ticket details.

    The LLM will call this tool, providing the image data, and expects the extracted
    structured data in return. This demonstrates multimodal input handling.

    Args:
        image_part: The image content from the user's prompt.

    Returns:
        A dictionary containing the extracted ticket data, including pilgrim_name.
    """
    # In a real app, you would call a dedicated OCR service here.
    # We use a hardcoded result for this demonstration.
    if image_part and image_part.inline_data:
        # Simulate processing the ticket image
        extracted_data = {
            "ticket_id": "TTD-20250101-A9B3",
            "darshan_date": "2025-10-15",
            "pilgrim_name": "Srinivas Rao", # Extracted for use in booking
            "extracted_successfully": True
        }
        return extracted_data

    return {"extracted_successfully": False}

async def ticket_db_lookup(ticket_id: str) -> dict:
    """
    CUSTOM TOOL 2: Simulates a long-running database query (Long-running operation).

    This function is defined as 'async' to simulate an I/O bound operation
    (like a database call) that takes a noticeable amount of time.

    Args:
        ticket_id: The ID to query in the database.

    Returns:
        A dictionary containing the database verification status.
    """
    # Simulate a network/database delay (Long-running operation simulation)
    await asyncio.sleep(0.1) # Reduced sleep for faster emulator execution

    # Mock Database Logic: Check if the ID is valid and has been used
    if ticket_id.startswith("TTD-2025"): # Corresponds to the mock image
        # This ticket is valid and active
        db_response = {
            "status": "VALID_ACTIVE",
            "registration_user_id": "user1234",
            "is_used": False,
            "original_booking_location": "Hyderabad, India"
        }
    elif ticket_id.startswith("INVALID"): # Example for an invalid ticket
        db_response = {"status": "INVALID_ID", "is_used": True}
    else:
        db_response = {"status": "NOT_FOUND", "is_used": False}

    return db_response


async def book_single_seva(seva_name: str, ticket_id: str, pilgrim_name: str) -> dict:
    """Simulates the fast, asynchronous booking of a single seva."""
    # The "gigabyteseconds" constraint is simulated by a very fast/parallel execution model.
    await asyncio.sleep(0.01)

    # Mock success/failure logic
    # We simulate the Homam being sold out to show mixed results
    success = True if seva_name != "Srinivasa Divyanugraha Homam" else False

    if success:
        return {
            "seva_name": seva_name,
            "status": "BOOKED",
            "confirmation_code": f"CF-{seva_name.split()[0][:3]}-{ticket_id[-4:]}"
        }
    else:
        return {
            "seva_name": seva_name,
            "status": "FAILED (Sold Out)",
            "reason": "Quota exhausted for the requested date."
        }

async def book_sevas_tool(ticket_id: str, pilgrim_name: str) -> list[dict]:
    """
    CUSTOM TOOL 3: Books multiple sevas in parallel using asyncio.gather.
    This simulates the "in gigabyteseconds" fast execution requirement.

    Args:
        ticket_id: The verified ticket ID.
        pilgrim_name: The name of the pilgrim for booking.

    Returns:
        A list of dictionaries containing the booking status for each seva.
    """
    booking_tasks = [
        book_single_seva(seva_name, ticket_id, pilgrim_name)
        for seva_name in SEVA_LIST
    ]

    # Execute all booking tasks concurrently
    results = await asyncio.gather(*booking_tasks)

    return results

# Global FunctionTool instances
OCR_TOOL = FunctionTool(func=process_image_for_ocr, name="process_image_for_ocr")
DB_TOOL = FunctionTool(func=ticket_db_lookup, name="db_lookup_tool")
SEVA_TOOL = FunctionTool(func=book_sevas_tool, name="book_sevas_tool")


import os, asyncio, base64, logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger("KONDA-ADK")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Try to import genai
GENAI_AVAILABLE = False
try:
    from google import genai
    GENAI_AVAILABLE = True
    logger.info("google-genai available")
except Exception:
    logger.info("google-genai not available, falling back to deterministic behavior")

@dataclass
class Part:
    mime_type: Optional[str] = None
    text: Optional[str] = None
    inline_data: Optional[str] = None
    @staticmethod
    def from_text(text: str): return Part(mime_type="text/plain", text=text)
    @staticmethod
    def from_inline_data(data: str, mime_type: str="image/png"): return Part(mime_type=mime_type, inline_data=data)

@dataclass
class Event:
    text: Optional[str]=None
    tool_call: Optional[Dict[str,Any]]=None
    tool_result: Optional[Dict[str,Any]]=None

class EmulatorFunctionTool: # Renamed FunctionTool to EmulatorFunctionTool
    def __init__(self, func: Callable, name: Optional[str]=None):
        self.func = func
        self.name = name or getattr(func, "__name__", "function_tool")
    async def call(self, *args, **kwargs):
        res = self.func(*args, **kwargs)
        if asyncio.iscoroutine(res):
            res = await res
        return res

class InMemorySessionService:
    def __init__(self):
        self._sessions: Dict[str, Dict[str,Any]] = {}
    async def get_session(self, app_name: str, user_id: str) -> Dict[str,Any]:
        key = f"{app_name}:{user_id}"
        return self._sessions.setdefault(key, {})
    async def set_session_state(self, app_name: str, user_id: str, state: Dict[str,Any]):
        key = f"{app_name}:{user_id}"
        self._sessions[key] = state

# Basic genai wrapper (safe)
def genai_generate(prompt: str, model: str="gemini-2.5-flash", max_output_tokens: int=512) -> str:
    if not GENAI_AVAILABLE:
        raise RuntimeError("genai not available")
    api_key = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = None
    try:
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
    except Exception:
        client = genai.Client()
    # Try client.generate_text or client.generate
    try:
        if hasattr(client, "generate_text"):
            resp = client.generate_text(model=model, input=prompt, max_output_tokens=max_output_tokens)
            # adapt response shapes
            if hasattr(resp, "text"): return resp.text
            if isinstance(resp, dict) and 'candidates' in resp: return resp['candidates'][0].get('content','')
            return str(resp)
    except Exception:
        pass
    try:
        if hasattr(client, "generate"):
            resp = client.generate(model=model, text=prompt)
            if hasattr(resp, "candidates") and len(resp.candidates)>0:
                c = resp.candidates[0]
                return getattr(c, 'content', str(c))
            return str(resp)
    except Exception:
        pass
    raise RuntimeError("genai call failed")

# Minimal LlmAgent class (deterministic or genai-enabled)
class LlmAgent:
    def __init__(self, name:str, instruction:str, tools:Optional[List[EmulatorFunctionTool]]=None, input_keys:Optional[List[str]]=None, output_keys:Optional[List[str]]=None, model:str="gemini-2.5-flash"):
        self.name=name; self.instruction=instruction
        self.tools = {t.name: t for t in (tools or [])}
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.use_real_llm = False
        self.model = model
    async def run(self, session_state: Dict[str,Any], parts: List[Part], runner_events: List[Event]):
        logger.info(f"[{self.name}] run start")
        if self.use_real_llm and GENAI_AVAILABLE:
            # Basic prompt and call; for production parse structured plan JSON
            prompt = self.instruction + "\n\nContext:\n"
            for k in self.input_keys:
                prompt += f"{k}: {session_state.get(k)}\n"
            for p in parts:
                if p.text: prompt += "\nUser: " + p.text
            try:
                text_out = genai_generate(prompt, model=self.model)
                runner_events.append(Event(text=f"[{self.name}] Gemini: {text_out[:200]}"))
                # naive: if JSON in output and instructs a tool call, handle that (left to user customization)
            except Exception as e:
                runner_events.append(Event(text=f"[{self.name}] Gemini error: {e}"))
            return session_state
        # deterministic behavior implemented in orchestrator & specific agents
        runner_events.append(Event(text=f"[{self.name}] deterministic noop"))
        return session_state

class SequentialAgent:
    def __init__(self, name:str, sub_agents:List[LlmAgent], description:Optional[str]=None):
        self.name=name; self.sub_agents=sub_agents; self.description=description

class InMemoryRunner:
    def __init__(self, agent:SequentialAgent, app_name:str, session_service:InMemorySessionService, api_key:str=""):
        self.agent=agent; self.app_name=app_name; self.session_service=session_service; self.api_key=api_key
    async def run_async(self, user_id:str, new_message:List[Part]):
        state = await self.session_service.get_session(self.app_name, user_id)
        # discharge to sub agents
        for sub in self.agent.sub_agents:
            events = []
            st = await sub.run(state, new_message, events)
            state.update(st)
            for e in events:
                yield e
        await self.session_service.set_session_state(self.app_name, user_id, state)
        yield Event(text=f"[Runner] Completed keys: {list(state.keys())}")


import asyncio, base64, json, uuid, os
from typing import Any, Dict, List
from __main__ import EmulatorFunctionTool, Part # Explicitly import EmulatorFunctionTool and Part from the main notebook scope

# Seva list
SEVA_LIST = [
    "Padmavati Kalyanotsavam",
    "Srinivasa Mangapuram Kalyanotsavam",
    "Srinivasa Divyanugraha Homam"
]

MOCK_TICKET_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDHBLDNn/qACwFgM3W/pX2gAAAABJRU5ErkJggg=="

def process_image_for_ocr(image_part: Part) -> Dict[str,Any]:
    """
    CUSTOM TOOL 1: Simulates processing an image for OCR to extract ticket details.

    The LLM will call this tool, providing the image data, and expects the extracted
    structured data in return. This demonstrates multimodal input handling.

    Args:
        image_part: The image content from the user's prompt.

    Returns:
        A dictionary containing the extracted ticket data, including pilgrim_name.
    """
    # In a real app, you would call a dedicated OCR service here.
    # We use a hardcoded result for this demonstration.
    if image_part and image_part.inline_data:
        # Simulate processing the ticket image
        extracted_data = {
            "ticket_id": "TTD-20250101-A9B3",
            "darshan_date": "2025-10-15",
            "pilgrim_name": "Srinivas Rao", # Extracted for use in booking
            "extracted_successfully": True
        }
        return extracted_data

    return {"extracted_successfully": False}

async def ticket_db_lookup(ticket_id: str) -> dict:
    """
    CUSTOM TOOL 2: Simulates a long-running database query (Long-running operation).

    This function is defined as 'async' to simulate an I/O bound operation
    (like a database call) that takes a noticeable amount of time.

    Args:
        ticket_id: The ID to query in the database.

    Returns:
        A dictionary containing the database verification status.
    """
    # Simulate a network/database delay (Long-running operation simulation)
    await asyncio.sleep(0.1) # Reduced sleep for faster emulator execution

    # Mock Database Logic: Check if the ID is valid and has been used
    if ticket_id.startswith("TTD-2025"): # Corresponds to the mock image
        # This ticket is valid and active
        db_response = {
            "status": "VALID_ACTIVE",
            "registration_user_id": "user1234",
            "is_used": False,
            "original_booking_location": "Hyderabad, India"
        }
    elif ticket_id.startswith("INVALID"): # Example for an invalid ticket
        db_response = {"status": "INVALID_ID", "is_used": True}
    else:
        db_response = {"status": "NOT_FOUND", "is_used": False}

    return db_response


async def book_single_seva(seva_name: str, ticket_id: str, pilgrim_name: str) -> dict:
    """Simulates the fast, asynchronous booking of a single seva."""
    # The "gigabyteseconds" constraint is simulated by a very fast/parallel execution model.
    await asyncio.sleep(0.01)

    # Mock success/failure logic
    # We simulate the Homam being sold out to show mixed results
    success = True if seva_name != "Srinivasa Divyanugraha Homam" else False

    if success:
        return {
            "seva_name": seva_name,
            "status": "BOOKED",
            "confirmation_code": f"CF-{seva_name.split()[0][:3]}-{ticket_id[-4:]}"
        }
    else:
        return {
            "seva_name": seva_name,
            "status": "FAILED (Sold Out)",
            "reason": "Quota exhausted for the requested date."
        }

async def book_sevas_tool(ticket_id: str, pilgrim_name: str) -> list[dict]:
    """
    CUSTOM TOOL 3: Books multiple sevas in parallel using asyncio.gather.
    This simulates the "in gigabyteseconds" fast execution requirement.

    Args:
        ticket_id: The verified ticket ID.
        pilgrim_name: The name of the pilgrim for booking.

    Returns:
        A list of dictionaries containing the booking status for each seva.
    """
    booking_tasks = [
        book_single_seva(seva_name, ticket_id, pilgrim_name)
        for seva_name in SEVA_LIST
    ]

    # Execute all booking tasks concurrently
    results = await asyncio.gather(*booking_tasks)

    return results

# Global EmulatorFunctionTool instances
OCR_TOOL = EmulatorFunctionTool(func=process_image_for_ocr, name="process_image_for_ocr")
DB_TOOL = EmulatorFunctionTool(func=ticket_db_lookup, name="db_lookup_tool")
SEVA_TOOL = EmulatorFunctionTool(func=book_sevas_tool, name="book_sevas_tool")




# agent/orchestrator.py
import asyncio
# Removed: from google.adk.emulator import LlmAgent, SequentialAgent, InMemoryRunner, InMemorySessionService, Part, Event
# Removed: from google.adk.tools import OCR_TOOL, DB_TOOL, SEVA_TOOL

# Assume LlmAgent, SequentialAgent, InMemoryRunner, InMemorySessionService, Part, Event
# and OCR_TOOL, DB_TOOL, SEVA_TOOL are available from previous cell executions (8mqJjEVfY3LU and Zel7LtbzaEqu)
from __main__ import LlmAgent, SequentialAgent, InMemoryRunner, InMemorySessionService, Part, Event, OCR_TOOL, DB_TOOL, SEVA_TOOL

# Deterministic agent implementations (inherit LlmAgent but override run or use the deterministic path)
class OCRAgent(LlmAgent):
    async def run(self, session_state, parts, events):
        image_part = next((p for p in parts if p.inline_data and p.mime_type and p.mime_type.startswith("image")), None)
        if OCR_TOOL and image_part:
            events.append(Event(text=f"[OCRAgent] calling OCR"))
            res = await OCR_TOOL.call(image_part)
            events.append(Event(tool_call={"name":OCR_TOOL.name, "args":{}}))
            events.append(Event(tool_result={"result":res}))
            session_state.update({"ticket_id":res.get("ticket_id"), "pilgrim_name":res.get("pilgrim_name")})
        else:
            events.append(Event(text="[OCRAgent] No image found"))
        return session_state

class ValidationAgent(LlmAgent):
    async def run(self, session_state, parts, events):
        ticket_id = session_state.get("ticket_id")
        if ticket_id:
            events.append(Event(text="[ValidationAgent] calling DB lookup"))
            res = await DB_TOOL.call(ticket_id)
            events.append(Event(tool_call={"name":DB_TOOL.name,"args":{"ticket_id":ticket_id}}))
            events.append(Event(tool_result={"result":res}))
            session_state["db_status"] = res
        else:
            events.append(Event(text="[ValidationAgent] no ticket_id"))
        return session_state

class BookingAgent(LlmAgent):
    async def run(self, session_state, parts, events):
        db = session_state.get("db_status", {})
        if db.get("status") != "VALID_ACTIVE":
            events.append(Event(text="[BookingAgent] ticket invalid; aborting booking"))
            session_state["booking_status"] = {"error":"invalid_ticket"}
            return session_state
        ticket_id = session_state.get("ticket_id")
        pilgrim = session_state.get("pilgrim_name")
        if ticket_id and pilgrim:
            events.append(Event(text="[BookingAgent] calling book_sevas_tool"))
            res = await SEVA_TOOL.call(ticket_id, pilgrim)
            events.append(Event(tool_call={"name":SEVA_TOOL.name,"args":{"ticket_id":ticket_id,"pilgrim_name":pilgrim}}))
            events.append(Event(tool_result={"result":res}))
            session_state["booking_status"] = res
        else:
            events.append(Event(text="[BookingAgent] missing ticket_id or pilgrim_name"))
        return session_state

class AnomalyAgent(LlmAgent):
    async def run(self, session_state, parts, events):
        db = session_state.get("db_status",{})
        current_loc = session_state.get("current_user_location", "Unknown")
        orig = db.get("original_booking_location")
        flagged = False
        if db.get("status") != "VALID_ACTIVE":
            session_state["anomaly"] = {"verdict":"NOT_APPLICABLE"}
            events.append(Event(text="[AnomalyAgent] not applicable"))
            return session_state
        if orig and current_loc and orig.split(",")[-1].strip().lower() != current_loc.split(",")[-1].strip().lower():
            flagged = True
        session_state["anomaly"] = {"verdict":"POTENTIAL_RESALE_ANOMALY" if flagged else "OK"}
        events.append(Event(text=f"[AnomalyAgent] verdict: {session_state['anomaly']}"))
        return session_state

# Build sequential agent
def build_sequential_agent(use_genai=False):
    o = OCRAgent("OCRAgent","",tools=[OCR_TOOL]); v = ValidationAgent("ValidationAgent","",tools=[DB_TOOL])
    b = BookingAgent("BookingAgent","",tools=[SEVA_TOOL]); a = AnomalyAgent("AnomalyAgent","")
    seq = SequentialAgent("KONDA-Guardian",[o,v,b,a], description="OCR->Validate->Book->Anomaly")
    session = InMemorySessionService()
    runner = InMemoryRunner(agent=seq, app_name="KONDAApp", session_service=session, api_key="")
    # Optionally set use_real_llm on specific agents
    if use_genai:
        o.use_real_llm = True
    return runner, session



# server/app.py
from fastapi import FastAPI, UploadFile, File, Form
import asyncio
# Corrected imports to use globally available objects from __main__
from __main__ import build_sequential_agent, Part

app = FastAPI(title="KONDA Agent API")

runner, session_service = build_sequential_agent(use_genai=False)

@app.post("/book")
async def book_ticket(user_id: str = Form("guest"), location: str = Form("Unknown"), id_image: UploadFile = File(...)):
    # read image bytes and create Part
    data = await id_image.read()
    import base64
    b64 = base64.b64encode(data).decode("utf-8")
    prompt_part = Part.from_text(f"Please verify ticket. My current location is: {location}")
    img_part = Part.from_inline_data(b64, mime_type=id_image.content_type)
    outputs = []
    async for ev in runner.run_async(user_id=user_id, new_message=[prompt_part, img_part]):
        outputs.append(ev)
    # Return aggregated events and final session state
    final_state = await session_service.get_session("KONDAApp", user_id)
    return {"events":[e.text or {} for e in outputs], "final_state": final_state}


#uvicorn server.app:app --host 0.0.0.0 --port 8000



import os, asyncio, base64, logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger("KONDA-ADK")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Try to import genai
GENAI_AVAILABLE = False
try:
    from google import genai
    GENAI_AVAILABLE = True
    logger.info("google-genai available")
except Exception:
    logger.info("google-genai not available, falling back to deterministic behavior")

@dataclass
class Part:
    mime_type: Optional[str] = None
    text: Optional[str] = None
    inline_data: Optional[str] = None
    @staticmethod
    def from_text(text: str): return Part(mime_type="text/plain", text=text)
    @staticmethod
    def from_inline_data(data: str, mime_type: str="image/png"): return Part(mime_type=mime_type, inline_data=data)

@dataclass
class Event:
    text: Optional[str]=None
    tool_call: Optional[Dict[str,Any]]=None
    tool_result: Optional[Dict[str,Any]]=None

class EmulatorFunctionTool: # Renamed FunctionTool to EmulatorFunctionTool
    def __init__(self, func: Callable, name: Optional[str]=None):
        self.func = func
        self.name = name or getattr(func, "__name__", "function_tool")
    async def call(self, *args, **kwargs):
        res = self.func(*args, **kwargs)
        if asyncio.iscoroutine(res):
            res = await res
        return res

class InMemorySessionService:
    def __init__(self):
        self._sessions: Dict[str, Dict[str,Any]] = {}
    async def get_session(self, app_name: str, user_id: str) -> Dict[str,Any]:
        key = f"{app_name}:{user_id}"
        return self._sessions.setdefault(key, {})
    async def set_session_state(self, app_name: str, user_id: str, state: Dict[str,Any]):
        key = f"{app_name}:{user_id}"
        self._sessions[key] = state

# Basic genai wrapper (safe)
def genai_generate(prompt: str, model: str="gemini-2.5-flash", max_output_tokens: int=512) -> str:
    if not GENAI_AVAILABLE:
        raise RuntimeError("genai not available")
    api_key = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = None
    try:
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
    except Exception:
        client = genai.Client()
    # Try client.generate_text or client.generate
    try:
        if hasattr(client, "generate_text"):
            resp = client.generate_text(model=model, input=prompt, max_output_tokens=max_output_tokens)
            # adapt response shapes
            if hasattr(resp, "text"): return resp.text
            if isinstance(resp, dict) and 'candidates' in resp: return resp['candidates'][0].get('content','')
            return str(resp)
    except Exception:
        pass
    try:
        if hasattr(client, "generate"):
            resp = client.generate(model=model, text=prompt)
            if hasattr(resp, "candidates") and len(resp.candidates)>0:
                c = resp.candidates[0]
                return getattr(c, 'content', str(c))
            return str(resp)
    except Exception:
        pass
    raise RuntimeError("genai call failed")

# Minimal LlmAgent class (deterministic or genai-enabled)
class LlmAgent:
    def __init__(self, name:str, instruction:str, tools:Optional[List[EmulatorFunctionTool]]=None, input_keys:Optional[List[str]]=None, output_keys:Optional[List[str]]=None, model:str="gemini-2.5-flash"):
        self.name=name; self.instruction=instruction
        self.tools = {t.name: t for t in (tools or [])}
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.use_real_llm = False
        self.model = model
    async def run(self, session_state: Dict[str,Any], parts: List[Part], runner_events: List[Event]):
        logger.info(f"[{self.name}] run start")
        if self.use_real_llm and GENAI_AVAILABLE:
            # Basic prompt and call; for production parse structured plan JSON
            prompt = self.instruction + "\n\nContext:\n"
            for k in self.input_keys:
                prompt += f"{k}: {session_state.get(k)}\n"
            for p in parts:
                if p.text: prompt += "\nUser: " + p.text
            try:
                text_out = genai_generate(prompt, model=self.model)
                runner_events.append(Event(text=f"[{self.name}] Gemini: {text_out[:200]}"))
                # naive: if JSON in output and instructs a tool call, handle that (left to user customization)
            except Exception as e:
                runner_events.append(Event(text=f"[{self.name}] Gemini error: {e}"))
            return session_state
        # deterministic behavior implemented in orchestrator & specific agents
        runner_events.append(Event(text=f"[{self.name}] deterministic noop"))
        return session_state

class SequentialAgent:
    def __init__(self, name:str, sub_agents:List[LlmAgent], description:Optional[str]=None):
        self.name=name; self.sub_agents=sub_agents; self.description=description

class InMemoryRunner:
    def __init__(self, agent:SequentialAgent, app_name:str, session_service:InMemorySessionService, api_key:str=""):
        self.agent=agent; self.app_name=app_name; self.session_service=session_service; self.api_key=api_key
    async def run_async(self, user_id:str, new_message:List[Part]):
        state = await self.session_service.get_session(self.app_name, user_id)
        # discharge to sub agents
        for sub in self.agent.sub_agents:
            events = []
            st = await sub.run(state, new_message, events)
            state.update(st)
            for e in events:
                yield e
        await self.session_service.set_session_state(self.app_name, user_id, state)
        yield Event(text=f"[Runner] Completed keys: {list(state.keys())}")


# Demo-only function (no server)
def run_agent(user_input):
    # Call Gemini or your agent logic here
    return {"response": "Simulated agent output for: " + user_input}

run_agent("Verify TS1234")



from fastapi import FastAPI

app = FastAPI()

@app.post("/verify_ticket")
def verify_ticket(ticket_id: str):
    return {"verified": ticket_id.startswith("TS")}



