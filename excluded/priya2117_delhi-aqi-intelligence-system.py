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


from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


# Cell 1: Force upgrade to latest version
!pip uninstall -y langchain-google-genai
!pip install --upgrade --force-reinstall langchain-google-genai==2.0.5
!pip install --upgrade google-generativeai

# âš ï¸� IMPORTANT: Restart the kernel after this
# Go to: Runtime â†’ Restart Runtime


#Import Libraries
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Type
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Go to Kaggle -> Add-ons -> Secrets to add these
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

# Get API keys from Kaggle secrets
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
AQICN_API_KEY = user_secrets.get_secret("AQICN_API_KEY")  # Get from https://aqicn.org/data-platform/token/
NASA_FIRMS_KEY = user_secrets.get_secret("NASA_FIRMS_KEY")  # Get from https://firms.modaps.eosdis.nasa.gov/api/
OPENWEATHER_KEY = user_secrets.get_secret("OPENWEATHER_KEY")  # Get from https://openweathermap.org/api
SERPAPI_KEY = user_secrets.get_secret("SERPAPI_KEY")  # Get from https://serpapi.com/

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


import google.generativeai as genai

# Load your API key
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    GEMINI_KEY = secrets.get_secret("GOOGLE_API_KEY")
except:
    import os
    GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GEMINI_KEY)

# List all available models
print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  âœ… {model.name}")


import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["AQICN_API_KEY"] = secrets.get_secret("AQICN_API_KEY")
print("âœ… API key set!")



import os
import sys
import time
import json
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import traceback

# Core dependencies
import requests
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

# Visualization
import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
import plotly.express as px

# LLM Integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("âš ï¸� Google Generative AI not installed. Install: pip install google-generativeai")

# Async support
import aiohttp

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Delhi area coordinates (comprehensive list)
DELHI_STATIONS_COORDS  = {
    "Connaught Place": (28.6315, 77.2167),
    "Karol Bagh": (28.6510, 77.1900),
    "Mandir Marg": (28.6357, 77.2011),
    "Paharganj": (28.6431, 77.2160),
    "Darya Ganj": (28.6448, 77.2380),
    "Civil Lines": (28.6773, 77.2249),
    "Sadar Bazaar": (28.6624, 77.2000),
    "Lodhi Road": (28.5918, 77.2273),
    "Pusa": (28.6416, 77.1467),
    "Chanakyapuri": (28.5964, 77.1836),
    "AIIMS": (28.5661, 77.2100),
    "RK Puram": (28.5631, 77.1817),
    "Hauz Khas": (28.5494, 77.2001),
    "Green Park": (28.5595, 77.2066),
    "Malviya Nagar": (28.5273, 77.2147),
    "Saket": (28.5246, 77.2063),
    "Kalkaji": (28.5482, 77.2580),
    "Greater Kailash": (28.5410, 77.2434),
    "Okhla": (28.5308, 77.2714),
    "Dwarka": (28.5921, 77.0460),
    "Vasant Kunj": (28.5273, 77.1509),
    "Vasant Vihar": (28.5677, 77.1570),
    "Najafgarh": (28.6090, 76.9795),
    "Palam": (28.5758, 77.0944),
    "Punjabi Bagh": (28.6692, 77.1317),
    "Janakpuri": (28.6219, 77.0878),
    "Uttam Nagar": (28.6240, 77.0525),
    "Paschim Vihar": (28.6766, 77.0880),
    "Rajouri Garden": (28.6430, 77.1230),
    "Model Town": (28.7047, 77.1938),
    "Azadpur": (28.7090, 77.1805),
    "Rohini": (28.7419, 77.0672),
    "Bawana": (28.7964, 77.0389),
    "Narela": (28.8534, 77.0934),
    "Shahdara": (28.6699, 77.2878),
    "Seelampur": (28.6820, 77.2810),
    "Vivek Vihar": (28.6725, 77.3150),
    "Anand Vihar": (28.6467, 77.3167),
    "Mayur Vihar": (28.5965, 77.3100),
    "Laxmi Nagar": (28.6312, 77.2778),
    "Patparganj": (28.6291, 77.2923),
    "Sarita Vihar": (28.5286, 77.2952),
    "IGI Airport T3": (28.5562, 77.1000),
    "ITO": (28.6289, 77.2469),
}
# AQI Health Categories
AQI_CATEGORIES = {
    "good": {"range": (0, 50), "color": "#00E400", "emoji": "ğŸŸ¢"},
    "moderate": {"range": (51, 100), "color": "#FFFF00", "emoji": "ğŸŸ¡"},
    "unhealthy_sensitive": {"range": (101, 150), "color": "#FF7E00", "emoji": "ğŸŸ "},
    "unhealthy": {"range": (151, 200), "color": "#FF0000", "emoji": "ğŸ”´"},
    "very_unhealthy": {"range": (201, 300), "color": "#8F3F97", "emoji": "ğŸŸ£"},
    "hazardous": {"range": (301, 500), "color": "#7E0023", "emoji": "âš«"},
}



# =============================================================================
# OBSERVABILITY FRAMEWORK
# =============================================================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Span:
    """OpenTelemetry-compatible span for distributed tracing"""
    trace_id: str
    span_id: str
    name: str
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    
    def add_event(self, name: str, attributes: Dict = None):
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })
    
    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value
    
    def end(self, status: str = "OK"):
        self.end_time = datetime.now()
        self.status = status
    
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0

class DistributedTracer:
    """Production-grade distributed tracing system"""
    
    def __init__(self, service_name: str = "delhi-aqi-intelligence"):
        self.service_name = service_name
        self.spans: List[Span] = []
        self.current_trace_id: Optional[str] = None
        self.active_spans: Dict[str, Span] = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.service_name)
        logger.setLevel(logging.WARNING)

        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(trace_id)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_trace(self) -> str:
        self.current_trace_id = str(uuid.uuid4())[:8]
        return self.current_trace_id
    
    def start_span(self, name: str, parent_span_id: str = None) -> Span:
        span = Span(
            trace_id=self.current_trace_id or self.start_trace(),
            span_id=str(uuid.uuid4())[:8],
            name=name,
            parent_span_id=parent_span_id
        )
        self.spans.append(span)
        self.active_spans[span.span_id] = span
        self.log(LogLevel.DEBUG, f"Started span: {name}", {"span_id": span.span_id})
        return span
    
    def end_span(self, span: Span, status: str = "OK"):
        span.end(status)
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
    
    def log(self, level: LogLevel, message: str, extra: Dict = None):
        extra = extra or {}
        extra['trace_id'] = self.current_trace_id or 'N/A'
        
        log_func = getattr(self.logger, level.value.lower())
        log_func(message, extra={'trace_id': extra.get('trace_id', 'N/A')})
    
    def get_metrics(self) -> Dict[str, Any]:
        if not self.spans:
            return {}
        
        durations = [s.duration_ms() for s in self.spans if s.end_time]
        return {
            "total_spans": len(self.spans),
            "completed_spans": len([s for s in self.spans if s.end_time]),
            "failed_spans": len([s for s in self.spans if s.status != "OK"]),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            "spans_by_name": {name: len([s for s in self.spans if s.name == name]) 
                            for name in set(s.name for s in self.spans)}
        }

# Global tracer
tracer = DistributedTracer()

def traced(name: str = None):
    """Decorator for automatic span creation"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            span = tracer.start_span(span_name)
            span.set_attribute("function", func.__name__)
            
            try:
                result = func(*args, **kwargs)
                tracer.end_span(span, "OK")
                return result
            except Exception as e:
                span.add_event("error", {"error": str(e)})
                tracer.end_span(span, "ERROR")
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or func.__name__
            span = tracer.start_span(span_name)
            span.set_attribute("function", func.__name__)
            
            try:
                result = await func(*args, **kwargs)
                tracer.end_span(span, "OK")
                return result
            except Exception as e:
                span.add_event("error", {"error": str(e)})
                tracer.end_span(span, "ERROR")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator




# =============================================================================
# SESSIONS & MEMORY SYSTEM
# =============================================================================

@dataclass
class SessionState:
    """User session state management"""
    user_id: str
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    interaction_count: int = 0
    
    def set(self, key: str, value: Any):
        self.data[key] = value
        self.last_updated = datetime.now()
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def add_message(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.interaction_count += 1

class SessionService:
    """In-memory session service"""
    
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        tracer.log(LogLevel.INFO, "SessionService initialized")
    
    def create_session(self, user_id: str, profile: Dict = None) -> SessionState:
        session_id = str(uuid.uuid4())
        session = SessionState(
            user_id=user_id,
            session_id=session_id,
            data={"profile": profile or {}}
        )
        self.sessions[session_id] = session
        tracer.log(LogLevel.INFO, f"Session created: {session_id}", {"user_id": user_id})
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, key: str, value: Any) -> bool:
        session = self.sessions.get(session_id)
        if session:
            session.set(key, value)
            return True
        return False

class MemoryBank:
    """Long-term memory for historical patterns and predictions"""
    
    def __init__(self):
        self.memories: Dict[str, List[Dict]] = {}
        self.aqi_cache: Dict[str, Dict] = {}
        self.historical_patterns: Dict[str, Any] = {
            "seasonal": {},
            "weekly": {},
            "events": {}
        }
        tracer.log(LogLevel.INFO, "MemoryBank initialized")
    
    def add_memory(self, user_id: str, memory_type: str, content: Dict):
        if user_id not in self.memories:
            self.memories[user_id] = []
        
        self.memories[user_id].append({
            "id": str(uuid.uuid4())[:8],
            "type": memory_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def cache_aqi_data(self, location_key: str, data: Dict, ttl_minutes: int = 30):
        self.aqi_cache[location_key] = {
            "data": data,
            "cached_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=ttl_minutes)
        }
    
    def get_cached_aqi(self, location_key: str) -> Optional[Dict]:
        cached = self.aqi_cache.get(location_key)
        if cached and cached["expires_at"] > datetime.now():
            return cached["data"]
        return None
    
    def store_seasonal_pattern(self, month: int, stats: Dict):
        if month not in self.historical_patterns["seasonal"]:
            self.historical_patterns["seasonal"][month] = []
        
        self.historical_patterns["seasonal"][month].append({
            "year": datetime.now().year,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })




# =============================================================================
# LLM INTEGRATION
# =============================================================================

class GeminiClient:
    """Wrapper for Google Gemini API"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite"):
    #def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
    
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        genai.configure(api_key=api_key)
        env_model = os.getenv("GEMINI_MODEL")
        if env_model:
            env_model = env_model.strip().strip('"').strip("'")
        
        self.model_name = model or env_model or "gemini-2.0-flash-lite" 
        # Validate model name format (should be lowercase with hyphens)
        if " " in self.model_name or self.model_name[0].isupper():
          tracer.log(LogLevel.WARNING, f"Invalid model name '{self.model_name}', using gemini-2.0-flash-lite")
          self.model_name = "gemini-2.0-flash-lite"
    
        tracer.log(LogLevel.INFO, f"Using Gemini model: {repr(self.model_name)}")
    
        self.model = genai.GenerativeModel(self.model_name)
        tracer.log(
            LogLevel.INFO,
            f"Using Gemini model: {repr(self.model_name)}"
        )
 
       
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    
    @traced("llm_generate")
    def generate_content(self, prompt: str, temperature: float = None) -> str:
        """Generate content using Gemini"""
        try:
            config = self.generation_config.copy()
            if temperature is not None:
                config["temperature"] = temperature
            
            response = self.model.generate_content(
                prompt,
                generation_config=config
            )
            
            return response.text
        except Exception as e:
            tracer.log(LogLevel.ERROR, f"LLM generation error: {str(e)}")
            return ""
    
    @traced("llm_generate_json")
    def generate_json(self, prompt: str) -> Dict:
        """Generate JSON response"""
        json_prompt = f"""{prompt}

CRITICAL: Respond with ONLY valid JSON. No markdown, no backticks, no explanation.
Just the raw JSON object."""
        
        response = self.generate_content(json_prompt, temperature=0.3)
        
        # Clean response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            tracer.log(LogLevel.ERROR, f"Failed to parse JSON: {response[:100]}")
            return {}




# =============================================================================
# TOOLS & MCP INTEGRATION
# =============================================================================

class AQICNTool:
    """Tool for fetching AQI data from AQICN API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.waqi.info"
    
    @traced("tool_fetch_aqi_coordinates")
    def fetch_by_coordinates(self, lat: float, lon: float, area_name: str = "Unknown") -> Dict:
        """Fetch AQI by GPS coordinates"""
        try:
            url = f"{self.base_url}/feed/geo:{lat};{lon}/?token={self.api_key}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            
            if data.get("status") != "ok":
                return {"error": f"API error: {data.get('data')}"}
            
            aqi_data = data["data"]
            iaqi = aqi_data.get("iaqi", {})
            
            return {
                "name": area_name,
                "full_name": aqi_data.get("city", {}).get("name", area_name),
                "lat": lat,
                "lon": lon,
                "aqi": self._safe_float(aqi_data.get("aqi")),
                "pm25": self._safe_float(iaqi.get("pm25", {}).get("v")),
                "pm10": self._safe_float(iaqi.get("pm10", {}).get("v")),
                "o3": self._safe_float(iaqi.get("o3", {}).get("v")),
                "no2": self._safe_float(iaqi.get("no2", {}).get("v")),
                "timestamp": aqi_data.get("time", {}).get("s", "N/A"),
                "source": "AQICN",
                "uid": aqi_data.get("idx")
            }
        except Exception as e:
            tracer.log(LogLevel.ERROR, f"Error fetching AQI: {str(e)}")
            return {"error": str(e), "name": area_name}
    
    @traced("tool_fetch_aqi_bounds")
    def fetch_by_bounds(self, bounds: str = "28.40,76.80,28.95,77.40", limit: int = 20) -> List[Dict]:
        """Fetch all stations in bounding box"""
        stations = []
        url = f"{self.base_url}/map/bounds/?token={self.api_key}&latlng={bounds}"
        
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
            
            if data.get("status") != "ok":
                return []
            
            for st in data.get("data", [])[:limit]:
                uid = st.get("uid")
                detail_url = f"{self.base_url}/feed/@{uid}/?token={self.api_key}"
                detail_resp = requests.get(detail_url, timeout=5)
                detail_data = detail_resp.json()
                
                if detail_data.get("status") != "ok":
                    continue
                
                aqi_data = detail_data["data"]
                iaqi = aqi_data.get("iaqi", {})
                aqi_val = self._safe_float(aqi_data.get("aqi"))
                
                if aqi_val and aqi_val > 0:
                    stations.append({
                        "name": st.get("station", {}).get("name", f"Station {uid}"),
                        "lat": st.get("lat"),
                        "lon": st.get("lon"),
                        "aqi": int(aqi_val),
                        "pm25": self._safe_float(iaqi.get("pm25", {}).get("v")),
                        "pm10": self._safe_float(iaqi.get("pm10", {}).get("v")),
                        "timestamp": aqi_data.get("time", {}).get("s", "N/A"),
                        "source": "AQICN-bounds",
                        "uid": uid
                    })
                
                time.sleep(0.3)
        except Exception as e:
            tracer.log(LogLevel.ERROR, f"Bounds API error: {str(e)}")
        
        return stations
    
    def _safe_float(self, val) -> Optional[float]:
        if val is None or val == "N/A":
            return None
        try:
            return float(val)
        except:
            return None

class WeatherMCPServer:
    """MCP Server for weather data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    @traced("mcp_weather_current")
    async def get_current_weather(self, lat: float, lon: float) -> Dict:
        """Fetch current weather data"""
        try:
            url = f"{self.base_url}/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
            
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "wind_direction": data["wind"]["deg"],
                "description": data["weather"][0]["description"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            tracer.log(LogLevel.ERROR, f"Weather API error: {str(e)}")
            return {}

class NASAFIRMSTool:
    """Tool for NASA FIRMS fire/hotspot data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    
    @traced("tool_nasa_fires")
    def fetch_fire_hotspots(self, region_bounds: str, days: int = 1) -> List[Dict]:
        """
        Fetch fire hotspots (stubble burning detection)
        region_bounds: "north,west,south,east" e.g., "28,74,32,78" for Punjab/Haryana
        """
        try:
            url = f"{self.base_url}/{self.api_key}/VIIRS_SNPP_NRT/{days}/{region_bounds}"
            resp = requests.get(url, timeout=15)
            
            if resp.status_code != 200:
                return []
            
            fires = []
            lines = resp.text.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if not line:
                    continue
                
                fields = line.split(',')
                if len(fields) >= 9:
                    fires.append({
                        "lat": float(fields[0]),
                        "lon": float(fields[1]),
                        "brightness": float(fields[2]),
                        "confidence": fields[8],
                        "timestamp": fields[5]
                    })
            
            tracer.log(LogLevel.INFO, f"Found {len(fires)} fire hotspots")
            return fires
            
        except Exception as e:
            tracer.log(LogLevel.ERROR, f"NASA FIRMS error: {str(e)}")
            return []

class GeospatialTool:
    """Custom geospatial analysis tool"""
    
    @traced("tool_interpolate_aqi")
    def interpolate_aqi(self, known_points: List[Dict], target: Tuple[float, float]) -> float:
        """Interpolate AQI at target location using IDW"""
        if not known_points:
            return 0.0
        
        distances = []
        values = []
        
        for point in known_points:
            dist = distance.euclidean(target, (point['lat'], point['lon']))
            distances.append(dist)
            values.append(point['aqi'])
        
        # Inverse Distance Weighting
        weights = [1/d**2 if d > 0 else 1e10 for d in distances]
        interpolated = sum(w*v for w, v in zip(weights, values)) / sum(weights)
        
        return round(interpolated, 1)
    
    @traced("tool_find_hotspots")
    def find_pollution_hotspots(self, stations: List[Dict], threshold: int = 150) -> List[Dict]:
        """Identify pollution hotspots using clustering"""
        if len(stations) < 3:
            return []
        
        # Feature matrix [lat, lon, aqi]
        X = np.array([[s['lat'], s['lon'], s['aqi']/100] for s in stations])
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.03, min_samples=3).fit(X)
        
        hotspots = []
        for label in set(clustering.labels_):
            if label == -1:  # Noise
                continue
            
            cluster_points = [s for i, s in enumerate(stations) if clustering.labels_[i] == label]
            avg_aqi = sum(p['aqi'] for p in cluster_points) / len(cluster_points)
            
            if avg_aqi > threshold:
                hotspots.append({
                    "cluster_id": label,
                    "avg_aqi": round(avg_aqi, 1),
                    "station_count": len(cluster_points),
                    "stations": [p['name'] for p in cluster_points]
                })
        
        return sorted(hotspots, key=lambda x: x['avg_aqi'], reverse=True)



# =============================================================================
# AGENT FRAMEWORK
# =============================================================================

@dataclass
class AgentResult:
    """Result object for agent execution"""
    success: bool
    data: Any = None
    error: str = None
    timestamp: datetime = field(default_factory=datetime.now)

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.output_key: Optional[str] = None
    
    def run(self, context: Dict) -> AgentResult:
        raise NotImplementedError
    
    async def run_async(self, context: Dict) -> AgentResult:
        """Async execution (default: calls sync run)"""
        return self.run(context)


# =============================================================================
# DATA COLLECTION AGENTS
# =============================================================================

class DataFetcherAgent(BaseAgent):
    """Fetches AQI data from a single source"""
    
    def __init__(self, name: str, fetch_method: Callable, source_type: str):
        super().__init__(name, f"Fetches AQI data using {source_type}")
        self.fetch_method = fetch_method
        self.source_type = source_type
        self.output_key = f"{source_type}_data"
    
    @traced()
    def run(self, context: Dict) -> AgentResult:
        span = tracer.start_span(f"agent_{self.name}")
        span.set_attribute("source_type", self.source_type)
        
        try:
            tracer.log(LogLevel.INFO, f" {self.name} starting...")
            data = self.fetch_method()
            
            span.set_attribute("records_found", len(data) if isinstance(data, list) else 1)
            tracer.end_span(span, "OK")
            
            tracer.log(LogLevel.INFO, f" {self.name} completed: {len(data) if isinstance(data, list) else 1} records")
            return AgentResult(success=True, data=data)
            
        except Exception as e:
            tracer.end_span(span, "ERROR")
            tracer.log(LogLevel.ERROR, f" {self.name} failed: {str(e)}")
            return AgentResult(success=False, error=str(e))

class ParallelAgent:
    """Runs multiple agents in parallel (simulated)"""
    
    def __init__(self, name: str, sub_agents: List[BaseAgent]):
        self.name = name
        self.sub_agents = sub_agents
        self.output_key = "parallel_results"
    
    @traced("parallel_agent")
    def run(self, context: Dict) -> AgentResult:
        span = tracer.start_span(f"parallel_{self.name}")
        tracer.log(LogLevel.INFO, f" ParallelAgent '{self.name}' starting {len(self.sub_agents)} agents...")
        
        results = {}
        for agent in self.sub_agents:
            result = agent.run(context)
            if result.success:
                results[agent.output_key or agent.name] = result.data
        
        tracer.end_span(span, "OK")
        tracer.log(LogLevel.INFO, f" ParallelAgent completed with {len(results)} results")
        return AgentResult(success=True, data=results)

class SequentialAgent:
    """Runs agents in sequence, passing state between them"""
    
    def __init__(self, name: str, sub_agents: List[BaseAgent]):
        self.name = name
        self.sub_agents = sub_agents
        self.output_key = "sequential_results"
    
    @traced("sequential_agent")
    def run(self, context: Dict) -> AgentResult:
        span = tracer.start_span(f"sequential_{self.name}")
        tracer.log(LogLevel.INFO, f" SequentialAgent '{self.name}' starting...")
        
        state = context.copy()
        
        for i, agent in enumerate(self.sub_agents):
            tracer.log(LogLevel.DEBUG, f"  Step {i+1}/{len(self.sub_agents)}: {agent.name}")
            
            result = agent.run(state)
            
            if not result.success:
                tracer.end_span(span, "ERROR")
                return AgentResult(success=False, error=f"Step {agent.name} failed: {result.error}")
            
            # Merge results into state
            if hasattr(agent, 'output_key') and agent.output_key:
                if agent.output_key == "parallel_results" and isinstance(result.data, dict):
                    state.update(result.data)
                else:
                    state[agent.output_key] = result.data
        
        tracer.end_span(span, "OK")
        tracer.log(LogLevel.INFO, f" SequentialAgent completed all steps")
        return AgentResult(success=True, data=state)




# =============================================================================
# PROCESSING AGENTS
# =============================================================================

class DataDeduplicationAgent(BaseAgent):
    """Deduplicates station data from multiple sources"""
    
    def __init__(self):
        super().__init__("DataDeduplicator", "Removes duplicate stations")
        self.output_key = "deduplicated_stations"
    
    @traced("deduplicate")
    def run(self, context: Dict) -> AgentResult:
        tracer.log(LogLevel.INFO, " DataDeduplicator: Processing...")
        
        all_stations = []
        for key in ["bounds_data", "coords_data", "search_data"]:
            data = context.get(key, [])
            if isinstance(data, list):
                all_stations.extend(data)
        
        # Deduplicate
        unique_stations = []
        seen_keys = set()
        
        for st in all_stations:
            if not st or "error" in st:
                continue
            
            uid = st.get("uid")
            key = f"uid:{uid}" if uid else f"geo:{round(st.get('lat', 0), 3)}:{round(st.get('lon', 0), 3)}"
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_stations.append(st)
        
        tracer.log(LogLevel.INFO, f" Deduplicator: {len(unique_stations)} unique stations")
        return AgentResult(success=True, data=unique_stations)

class AnalysisAgent(BaseAgent):
    """Analyzes AQI data and generates insights"""
    
    def __init__(self):
        super().__init__("Analyzer", "Generates AQI insights")
        self.output_key = "analysis_results"
    
    @traced("analyze")
    def run(self, context: Dict) -> AgentResult:
        tracer.log(LogLevel.INFO, " Analyzer: Generating insights...")
        
        stations = context.get("deduplicated_stations", [])
        if not stations:
            return AgentResult(success=False, error="No station data")
        
        valid_stations = [s for s in stations if s.get("aqi") and s.get("aqi") > 0]
        if not valid_stations:
            return AgentResult(success=False, error="No valid AQI readings")
        
        aqis = [s["aqi"] for s in valid_stations]
        
        analysis = {
            "total_stations": len(valid_stations),
            "avg_aqi": round(sum(aqis) / len(aqis), 1),
            "max_aqi": max(aqis),
            "min_aqi": min(aqis),
            "median_aqi": sorted(aqis)[len(aqis) // 2],
            "most_polluted": sorted(valid_stations, key=lambda x: x["aqi"], reverse=True)[:5],
            "least_polluted": sorted(valid_stations, key=lambda x: x["aqi"])[:5],
            "category_distribution": self._get_distribution(valid_stations),
            "timestamp": datetime.now().isoformat()
        }
        
        tracer.log(LogLevel.INFO, f" Analyzer: Avg AQI = {analysis['avg_aqi']}")
        return AgentResult(success=True, data=analysis)
    
    def _get_distribution(self, stations: List[Dict]) -> Dict:
        dist = {"Good": 0, "Moderate": 0, "Unhealthy (Sensitive)": 0, 
                "Unhealthy": 0, "Very Unhealthy": 0, "Hazardous": 0}
        
        for s in stations:
            aqi = s.get("aqi", 0)
            if aqi <= 50: dist["Good"] += 1
            elif aqi <= 100: dist["Moderate"] += 1
            elif aqi <= 150: dist["Unhealthy (Sensitive)"] += 1
            elif aqi <= 200: dist["Unhealthy"] += 1
            elif aqi <= 300: dist["Very Unhealthy"] += 1
            else: dist["Hazardous"] += 1
        
        return dist




# =============================================================================
# LLM-POWERED INTELLIGENCE AGENTS
# =============================================================================

class NaturalLanguageQueryAgent(BaseAgent):
    """LLM-powered agent that understands natural language queries"""
    
    def __init__(self, llm_client: GeminiClient):
        super().__init__("NLQueryAgent", "Natural language query processor")
        self.llm = llm_client
        self.output_key = "nl_response"
    
    @traced("nl_query")
    def run(self, context: Dict) -> AgentResult:
        user_query = context.get("user_query", "")
        aqi_data = context.get("analysis_results", {})
        stations = context.get("deduplicated_stations", [])
        
        if not user_query:
            return AgentResult(success=False, error="No query provided")
        
        # Build context for LLM
        prompt = f"""You are an AI assistant specialized in Delhi air quality data.

Current AQI Summary:
- Average AQI: {aqi_data.get('avg_aqi', 'N/A')}
- Range: {aqi_data.get('min_aqi', 'N/A')} - {aqi_data.get('max_aqi', 'N/A')}
- Total Stations: {aqi_data.get('total_stations', 0)}

Most Polluted Areas:
{json.dumps([{"name": s["name"], "aqi": s["aqi"]} for s in aqi_data.get("most_polluted", [])[:3]], indent=2)}

Cleanest Areas:
{json.dumps([{"name": s["name"], "aqi": s["aqi"]} for s in aqi_data.get("least_polluted", [])[:3]], indent=2)}

User Query: {user_query}

Provide a helpful, conversational response that:
1. Directly answers the user's question
2. Includes specific AQI values when relevant
3. Provides health recommendations if appropriate
4. Suggests alternatives if needed

Keep response concise (2-3 paragraphs max)."""

        response = self.llm.generate_content(prompt)
        
        # Compute avg_aqi safely for fallback
        avg_aqi = aqi_data.get("avg_aqi")
        if avg_aqi is None:
            avg_aqi = 0.0
        else:
            try:
                avg_aqi = float(avg_aqi)
            except (TypeError, ValueError):
                avg_aqi = 0.0
        if not response:
           # Fallback response
           response = f"Current Delhi AQI is {avg_aqi:.1f}. "
        if avg_aqi > 300:
            response += "HAZARDOUS conditions. Stay indoors."
        elif avg_aqi > 200:
            response += "Very unhealthy. Avoid outdoor activities."
        elif avg_aqi > 150:
            response += "Unhealthy for everyone. Limit outdoor time."
        else:
            response += "Moderate air quality. Take normal precautions."

        
        return AgentResult(success=True, data=response)

class HealthRecommendationAgent(BaseAgent):
    """Generates personalized health recommendations"""
    
    def __init__(self, llm_client: GeminiClient):
        super().__init__("HealthAdvisor", "Personalized health recommendations")
        self.llm = llm_client
        self.output_key = "health_recommendations"
    
    @traced("health_recommendations")
    def run(self, context: Dict) -> AgentResult:
        aqi_data = context.get("analysis_results", {})
        user_profile = context.get("user_profile", {})
        
        avg_aqi = aqi_data.get("avg_aqi", 0)
        
        prompt = f"""Generate personalized health recommendations for Delhi air quality.
    

Current Air Quality:
- Average AQI: {avg_aqi}
- Category: {self._get_category(avg_aqi)}

User Profile:
- Age: {user_profile.get('age', 'Unknown')}
- Health Conditions: {user_profile.get('health_conditions', ['None'])}
- Activity Level: {user_profile.get('activity_level', 'Moderate')}

Provide recommendations in JSON format:
{{
    "overall_advice": "Brief overall recommendation",
    "outdoor_activities": "Guidance on outdoor activities",
    "protective_measures": "Masks, air purifiers, etc.",
    "vulnerable_group_warning": "Special warnings if applicable",
    "safe_areas": "Areas with better air quality for activities"
}}"""

        response = self.llm.generate_json(prompt)
        if not isinstance(response, dict) or not response:
          tracer.log(LogLevel.WARNING, "LLM unavailable, using rule-based fallback")
          # Rule-based fallback
          response = self._get_fallback_recommendations(avg_aqi, user_profile)

        
        return AgentResult(success=True, data=response)
    
    def _get_category(self, aqi: float) -> str:
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"
    def _get_fallback_recommendations(self, aqi: float, profile: Dict) -> Dict:
      """Rule-based recommendations when LLM unavailable"""
      if aqi <= 50:
         return {
            "overall_advice": "Air quality is good. Enjoy outdoor activities!",
            "outdoor_activities": "All outdoor activities safe for everyone.",
            "protective_measures": "No special measures needed.",
            "vulnerable_group_warning": "",
            "safe_areas": "All areas show good air quality."
         }
      elif aqi <= 100:
         return {
            "overall_advice": "Air quality is acceptable for most people.",
            "outdoor_activities": "Normal activities safe. Sensitive groups monitor symptoms.",
            "protective_measures": "Consider masks for sensitive individuals.",
            "vulnerable_group_warning": "People with asthma should limit prolonged exertion.",
            "safe_areas": "Look for areas with AQI below 100."
         }
      elif aqi <= 150:
         return {
            "overall_advice": "Unhealthy for sensitive groups.",
            "protective_measures": "N95 masks recommended. Use air purifiers indoors.",
            "vulnerable_group_warning": "People with respiratory conditions minimize exposure.",
            "safe_areas": "Seek areas with AQI below 100."
         }
      elif aqi <= 200:
        return {
            "overall_advice": "UNHEALTHY: Everyone may experience health effects.",
            "outdoor_activities": "AVOID prolonged outdoor activities.",
            "protective_measures": "N95 masks strongly recommended. Keep windows closed.",
            "vulnerable_group_warning": "Vulnerable groups should stay indoors.",
            "safe_areas": "Stay indoors when possible."
         }
      elif aqi <= 300:
         return {
            "overall_advice": "VERY UNHEALTHY: Health alert for everyone.",
            "outdoor_activities": "AVOID all outdoor activities.",
            "protective_measures": "N95/N99 masks ESSENTIAL. Air purifiers on high.",
            "vulnerable_group_warning": "EMERGENCY: Sensitive groups remain indoors.",
            "safe_areas": "STAY INDOORS. No safe outdoor areas."
         }
      else:  # 300+
         return {
            "overall_advice": "HAZARDOUS: Emergency conditions. AQI is " + str(int(aqi)),
            "outdoor_activities": "DO NOT go outside. Stay indoors with sealed windows.",
            "protective_measures": "N99/P100 masks only if emergency. Multiple air purifiers.",
            "vulnerable_group_warning": "EMERGENCY: Medical attention may be needed.",
            "safe_areas": "NO SAFE OUTDOOR AREAS. Remain indoors until AQI improves."
         }
   
class StubbleBurningAnalysisAgent(BaseAgent):
    """Analyzes NASA FIRMS data to correlate stubble burning with AQI"""
    
    def __init__(self, llm_client: GeminiClient, nasa_tool: NASAFIRMSTool):
        super().__init__("StubbleBurningAnalyzer", "Analyzes agricultural fire impact")
        self.llm = llm_client
        self.nasa_tool = nasa_tool
        self.output_key = "stubble_burning_analysis"
    
    @traced("stubble_burning_analysis")
    def run(self, context: Dict) -> AgentResult:
        # Fetch fire data from Punjab/Haryana region
        fires = self.nasa_tool.fetch_fire_hotspots(
            region_bounds="28,74,32,78",  # Punjab, Haryana
            days=2
        )
        
        aqi_data = context.get("analysis_results", {})
        
        if not fires:
            return AgentResult(success=True, data={
                "fire_count": 0,
                "impact": "No significant stubble burning detected",
                "confidence": "High"
            })
        
        # Analyze with LLM
        prompt = f"""Analyze the impact of agricultural fires on Delhi air quality.

Fire Hotspots Detected:
- Total fires: {len(fires)}
- High confidence fires: {len([f for f in fires if f['confidence'] == 'high'])}

Current Delhi AQI:
- Average: {aqi_data.get('avg_aqi')}
- Maximum: {aqi_data.get('max_aqi')}

Based on the fire count and AQI levels, provide analysis in JSON:
{{
    "fire_count": {len(fires)},
    "impact_assessment": "Low/Medium/High",
    "contribution_estimate": "Estimated % contribution to current AQI",
    "forecast": "Expected AQI trend in next 24-48 hours",
    "recommendations": "Actions for residents"
}}"""

        analysis = self.llm.generate_json(prompt)
        
        return AgentResult(success=True, data=analysis)


# =============================================================================
# VISUALIZATION AGENTS
# =============================================================================

class MapVisualizationAgent(BaseAgent):
    """Creates interactive Folium map"""
    
    def __init__(self, output_path: str = "delhi_aqi_map.html"):
        super().__init__("MapVisualizer", "Creates interactive map")
        self.output_path = output_path
        self.output_key = "map_path"
    
    @traced("create_map")
    def run(self, context: Dict) -> AgentResult:
        tracer.log(LogLevel.INFO, " Creating map...")
        
        stations = context.get("deduplicated_stations", [])
        analysis = context.get("analysis_results", {})
        
        if not stations:
            return AgentResult(success=False, error="No station data")
        
        # Create map
        m = folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles="OpenStreetMap")
        
        # Title
        avg_aqi = analysis.get("avg_aqi", "N/A")
        title_html = f"""
        <div style="position: fixed; top: 10px; left: 50px; width: 450px;
            background-color: white; border:2px solid grey; z-index:9999;
            font-size:14px; padding: 10px; border-radius: 5px;">
            <b> Delhi Air Quality Intelligence Map</b><br>
            <span style="font-size:12px">
                Average AQI: <b>{avg_aqi}</b> | Stations: {len(stations)}
            </span><br>
            <span style="font-size:10px; color:grey">
                Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Heatmap
        heat_data = [[s["lat"], s["lon"], s["aqi"]/50.0] for s in stations if s.get("aqi")]
        if heat_data:
            HeatMap(heat_data, min_opacity=0.3, max_opacity=0.8, radius=25, blur=25).add_to(m)
        
        # Markers
        for st in stations:
            if not st.get("aqi"):
                continue
            
            aqi = st["aqi"]
            color = self._get_color(aqi)
            
            popup_html = f"""
            <div style="font-family: Arial; width: 220px;">
                <h4 style="margin:0; color:{color}">{st['name']}</h4>
                <hr style="margin:5px 0;">
                <b>AQI:</b> <span style="font-size:18px; color:{color}"><b>{aqi}</b></span><br>
                <b>PM2.5:</b> {st.get('pm25', 'N/A')} Âµg/mÂ³<br>
                <b>PM10:</b> {st.get('pm10', 'N/A')} Âµg/mÂ³<br>
                <small>{st.get('timestamp', 'N/A')}</small>
            </div>
            """
            
            folium.CircleMarker(
                location=[st["lat"], st["lon"]],
                radius=15,
                popup=folium.Popup(popup_html, max_width=260),
                color="white",
                weight=2,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                tooltip=f"{st['name']}: AQI {aqi}"
            ).add_to(m)
        
        m.save(self.output_path)
        tracer.log(LogLevel.INFO, f" Map saved: {self.output_path}")

        try:
          from IPython.display import IFrame, display
          display(IFrame(src='delhi_aqi_map.html', width=800, height=600))
          tracer.log(LogLevel.INFO, " Map displayed inline")
        except:
         tracer.log(LogLevel.INFO, " Map saved (inline display not available)")
    
        return AgentResult(success=True, data={"map_file": "delhi_aqi_map.html"})
        
        
        #return AgentResult(success=True, data=self.output_path)
    
    def _get_color(self, aqi: int) -> str:
        if aqi <= 50: return "#00E400"
        elif aqi <= 100: return "#FFFF00"
        elif aqi <= 150: return "#FF7E00"
        elif aqi <= 200: return "#FF0000"
        elif aqi <= 300: return "#8F3F97"
        else: return "#7E0023"

class ReportGeneratorAgent(BaseAgent):
    """Generates comprehensive report"""
    
    def __init__(self):
        super().__init__("ReportGenerator", "Generates final report")
        self.output_key = "report"
    
    @traced("generate_report")
    def run(self, context: Dict) -> AgentResult:
        tracer.log(LogLevel.INFO, "ğŸ“‹ Generating report...")
        
        analysis = context.get("analysis_results", {})
        health_rec = context.get("health_recommendations", {})
        stubble_analysis = context.get("stubble_burning_analysis", {})
        map_path = context.get("map_path", "N/A")
        
        report = {
            "title": "Delhi Air Quality Intelligence Report",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_stations": analysis.get("total_stations", 0),
                "average_aqi": analysis.get("avg_aqi", "N/A"),
                "aqi_range": f"{analysis.get('min_aqi', 'N/A')} - {analysis.get('max_aqi', 'N/A')}",
                "overall_category": self._get_category(analysis.get("avg_aqi", 0))
            },
            "most_polluted_areas": [
                {"name": s["name"], "aqi": s["aqi"]} 
                for s in analysis.get("most_polluted", [])
            ],
            "cleanest_areas": [
                {"name": s["name"], "aqi": s["aqi"]} 
                for s in analysis.get("least_polluted", [])
            ],
            "category_distribution": analysis.get("category_distribution", {}),
            "health_recommendations": health_rec,
            "stubble_burning_impact": stubble_analysis,
            "map_file": map_path,
            "observability_metrics": tracer.get_metrics()
        }
        
        return AgentResult(success=True, data=report)
    
    def _get_category(self, avg_aqi) -> str:
        if not avg_aqi or avg_aqi == "N/A":
            return "Unknown"
        aqi = float(avg_aqi)
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"




# =============================================================================
# AGENT-TO-AGENT COMMUNICATION (A2A)
# =============================================================================

class A2AProtocol:
    """Agent-to-Agent communication protocol"""
    
    def __init__(self):
        self.message_queue: Dict[str, List[Dict]] = {}
        self.agent_registry: Dict[str, Dict] = {}
    
    def register_agent(self, agent_id: str, agent: BaseAgent, capabilities: List[str]):
        """Register agent with capabilities"""
        self.agent_registry[agent_id] = {
            "agent": agent,
            "capabilities": capabilities,
            "status": "available"
        }
        tracer.log(LogLevel.DEBUG, f"Registered agent: {agent_id}")
    
    def send_message(self, from_agent: str, to_agent: str, message: Dict):
        """Send message between agents"""
        if to_agent not in self.message_queue:
            self.message_queue[to_agent] = []
        
        self.message_queue[to_agent].append({
            "from": from_agent,
            "timestamp": datetime.now().isoformat(),
            "message": message
        })
    
    def get_messages(self, agent_id: str) -> List[Dict]:
        """Retrieve messages for an agent"""
        messages = self.message_queue.get(agent_id, [])
        self.message_queue[agent_id] = []  # Clear after reading
        return messages

class OrchestratorAgent(BaseAgent):
    """Master orchestrator using A2A protocol"""
    
    def __init__(self, llm_client: GeminiClient, a2a: A2AProtocol):
        super().__init__("Orchestrator", "Master coordinator")
        self.llm = llm_client
        self.a2a = a2a
    
    @traced("orchestrator")
    def run(self, context: Dict) -> AgentResult:
        """Coordinate multiple specialist agents via A2A"""
        user_request = context.get("user_query", "")
        
        # Plan execution
        plan_prompt = f"""Given this user request about Delhi air quality: "{user_request}"

Available capabilities:
- aqi_data: Fetch current AQI data
- health_advice: Generate health recommendations
- stubble_burning: Analyze agricultural fire impact
- visualization: Create maps and charts

Create an execution plan. Which capabilities are needed? Return JSON:
{{"required_capabilities": ["cap1", "cap2"], "reasoning": "why"}}"""

        plan = self.llm.generate_json(plan_prompt)
        
        # Execute via A2A
        results = {}
        for capability in plan.get("required_capabilities", []):
            # Find agent with this capability
            for agent_id, agent_info in self.a2a.agent_registry.items():
                if capability in agent_info["capabilities"]:
                    result = agent_info["agent"].run(context)
                    results[capability] = result.data
                    break
        
        return AgentResult(success=True, data={
            "plan": plan,
            "results": results
        })



# =============================================================================
# MAIN PIPELINE
# =============================================================================

class DelhiAQIIntelligenceSystem:
    """
    Complete Delhi AQI Intelligence System
    
    Demonstrates all Google ADK concepts:
    - Multi-agent architecture
    - LLM integration
    - Tool ecosystem (AQICN, NASA, Weather)
    - Sessions & Memory
    - Observability
    - A2A Protocol
    """
    
    def __init__(self, aqicn_key: str, gemini_key: str, 
                 nasa_key: str = None, weather_key: str = None):
        
        # Initialize services
        self.session_service = SessionService()
        self.memory_bank = MemoryBank()
        self.a2a_protocol = A2AProtocol()
        
        # Initialize LLM
        if GEMINI_AVAILABLE and gemini_key:
            self.llm = GeminiClient(gemini_key)
        else:
            self.llm = None
            tracer.log(LogLevel.WARNING, "LLM not available - some features disabled")
        
        # Initialize tools
        self.aqicn_tool = AQICNTool(aqicn_key)
        self.geospatial_tool = GeospatialTool()
        
        if nasa_key:
            self.nasa_tool = NASAFIRMSTool(nasa_key)
        else:
            self.nasa_tool = None
        
        if weather_key:
            self.weather_tool = WeatherMCPServer(weather_key)
        else:
            self.weather_tool = None
        
        # Build agent pipeline
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Construct the multi-agent pipeline"""
        
        # Data fetcher agents
        bounds_agent = DataFetcherAgent(
            "BoundsFetcher",
            lambda: self.aqicn_tool.fetch_by_bounds(),
            "bounds"
        )
        
        coords_agent = DataFetcherAgent(
            "CoordsFetcher",
            lambda: self._fetch_all_coords(),
            "coords"
        )
        
        # Parallel data collection
        parallel_fetcher = ParallelAgent(
            "DataCollectionGroup",
            [bounds_agent, coords_agent]
        )
        
        # Processing pipeline
        dedup_agent = DataDeduplicationAgent()
        analysis_agent = AnalysisAgent()
        
        # Intelligence agents (if LLM available)
        intelligence_agents = []
        if self.llm:
            health_agent = HealthRecommendationAgent(self.llm)
            intelligence_agents.append(health_agent)
            
            if self.nasa_tool:
                stubble_agent = StubbleBurningAnalysisAgent(self.llm, self.nasa_tool)
                intelligence_agents.append(stubble_agent)
        
        # Visualization
        map_agent = MapVisualizationAgent()
        report_agent = ReportGeneratorAgent()
        
        # Build sequential pipeline
        pipeline_agents = [
            parallel_fetcher,
            dedup_agent,
            analysis_agent,
        ] + intelligence_agents + [
            map_agent,
            report_agent
        ]
        
        self.pipeline = SequentialAgent("MainPipeline", pipeline_agents)
        
        # Register agents in A2A
        if self.llm:
            self.a2a_protocol.register_agent(
                "health_advisor",
                HealthRecommendationAgent(self.llm),
                ["health_advice", "recommendations"]
            )
    
    def _fetch_all_coords(self) -> List[Dict]:
        """Fetch AQI for all predefined coordinates"""
        results = []
        for name, (lat, lon) in DELHI_STATIONS_COORDS.items():
            # Check cache first
            cached = self.memory_bank.get_cached_aqi(f"{round(lat,3)}:{round(lon,3)}")
            if cached:
                results.append(cached)
            else:
                result = self.aqicn_tool.fetch_by_coordinates(lat, lon, name)
                if result and "error" not in result:
                    results.append(result)
                    self.memory_bank.cache_aqi_data(
                        f"{round(lat,3)}:{round(lon,3)}", 
                        result
                    )
                time.sleep(0.3)
        return results
    
    def run(self, user_id: str = "default_user", user_query: str = None,
            user_profile: Dict = None) -> Dict:
        """
        Execute the full intelligence pipeline
        
        Args:
            user_id: User identifier
            user_query: Natural language query (optional)
            user_profile: User health profile (optional)
            
        Returns:
            Comprehensive intelligence report
        """
        # Start trace
        trace_id = tracer.start_trace()
        main_span = tracer.start_span("pipeline_execution")
        
        print("\n" + "=" * 80)
        print("DELHI AQI INTELLIGENCE SYSTEM")
        print("=" * 80)
        print(f" Trace ID: {trace_id}")
        print(f" User: {user_id}")
        if user_query:
            print(f" Query: {user_query}")
        print("=" * 80)
        
        # Create session
        session = self.session_service.create_session(user_id, user_profile)
        
        # Build context
        context = {
            "session_id": session.session_id,
            "user_id": user_id,
            "user_query": user_query,
            "user_profile": user_profile or {},
            "config": {
                "delhi_bounds": "28.40,76.80,28.95,77.40"
            }
        }
        
        # Run pipeline
        print("\n Executing multi-agent pipeline...\n")
        result = self.pipeline.run(context)
        
        if result.success:
            report = result.data.get("report", {})
            
            # Process natural language query if provided
            if user_query and self.llm:
                nl_agent = NaturalLanguageQueryAgent(self.llm)
                nl_context = {**context, "analysis_results": result.data.get("analysis_results", {})}
                nl_result = nl_agent.run(nl_context)
                report["nl_response"] = nl_result.data
            
            # Store in memory
            self.memory_bank.add_memory(
                user_id,
                "aqi_query",
                {
                    "session_id": session.session_id,
                    "avg_aqi": report.get("summary", {}).get("average_aqi"),
                    "query": user_query
                }
            )
            
            # Update session
            session.set("last_report", report)
            if user_query:
                session.add_message("user", user_query)
                if "nl_response" in report:
                    session.add_message("assistant", report["nl_response"])
            
            # Print report
            self._print_report(report)
            
            tracer.end_span(main_span, "OK")
            return report
        else:
            tracer.end_span(main_span, "ERROR")
            print(f"\n Pipeline failed: {result.error}")
            return {"error": result.error}
    
    def _print_report(self, report: Dict):
        """Print formatted report"""
        print("\n" + "=" * 80)
        print("INTELLIGENCE REPORT")
        print("=" * 80)
        
        summary = report.get("summary", {})
        print(f"\n Air Quality Summary:")
        print(f"   â€¢ Total Stations: {summary.get('total_stations', 'N/A')}")
        print(f"   â€¢ Average AQI: {summary.get('average_aqi', 'N/A')}")
        print(f"   â€¢ Range: {summary.get('aqi_range', 'N/A')}")
        print(f"   â€¢ Overall: {summary.get('overall_category', 'N/A')}")
        
        print(f"\n Most Polluted:")
        for area in report.get("most_polluted_areas", [])[:3]:
            print(f"   â€¢ {area['name']}: AQI {area['aqi']}")
        
        print(f"\n Cleanest Areas:")
        for area in report.get("cleanest_areas", [])[:3]:
            print(f"   â€¢ {area['name']}: AQI {area['aqi']}")
        
        # Health recommendations
        health = report.get("health_recommendations", {})
        if health:
            print(f"\n Health Recommendations:")
            print(f"   â€¢ {health.get('overall_advice', 'N/A')}")
            print(f"   â€¢ Outdoor: {health.get('outdoor_activities', 'N/A')}")
            print(f"   â€¢ Protection: {health.get('protective_measures', 'N/A')}")
        
        # Stubble burning
        stubble = report.get("stubble_burning_impact", {})
        if stubble and stubble.get("fire_count", 0) > 0:
            print(f"\n Stubble Burning Impact:")
            print(f"   â€¢ Fires Detected: {stubble.get('fire_count', 0)}")
            print(f"   â€¢ Impact: {stubble.get('impact_assessment', 'N/A')}")
            print(f"   â€¢ Forecast: {stubble.get('forecast', 'N/A')}")
        
        # Natural language response
        if "nl_response" in report:
            print(f"\n AI Response:")
            print(f"   {report['nl_response']}")
        
        print(f"\n Map: {report.get('map_file', 'N/A')}")
        
        # Metrics
        metrics = report.get("observability_metrics", {})
        if metrics:
            print(f"\n Performance Metrics:")
            print(f"   â€¢ Total Spans: {metrics.get('total_spans', 0)}")
            print(f"   â€¢ Avg Duration: {metrics.get('avg_duration_ms', 0):.2f}ms")
            print(f"   â€¢ P95 Duration: {metrics.get('p95_duration_ms', 0):.2f}ms")
        
        print("\n" + "=" * 80)
    
    def query(self, query: str, user_id: str = "default_user") -> str:
        """
        Simplified query interface for conversational use
        
        Args:
            query: Natural language question
            user_id: User identifier
            
        Returns:
            Natural language response
        """
        result = self.run(user_id=user_id, user_query=query)
        return result.get("nl_response", "Unable to process query")
def display_map(map_file: str = "delhi_aqi_map.html"):
      """Display Folium map inline in Kaggle notebook"""
      try:
        from IPython.display import IFrame, display
        display(IFrame(src=map_file, width=900, height=600))
        print(f"Displaying interactive map: {map_file}")
      except Exception as e:
        print(f"Could not display map inline: {e}")
        print(f"Map saved to: {map_file}")
# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point"""
    
    # Load API keys
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        AQICN_KEY = secrets.get_secret("AQICN_API_KEY")
        GEMINI_KEY = secrets.get_secret("GOOGLE_API_KEY")
        NASA_KEY = secrets.get_secret("NASA_FIRMS_KEY") if "NASA_FIRMS_KEY" in dir(secrets) else None
        WEATHER_KEY = secrets.get_secret("OPENWEATHER_KEY") if "OPENWEATHER_KEY" in dir(secrets) else None
        print("Loaded API keys from Kaggle secrets")
    except:
        # Fallback to environment variables
        AQICN_KEY = os.getenv("AQICN_API_KEY")
        GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
        NASA_KEY = os.getenv("NASA_FIRMS_KEY")
        WEATHER_KEY = os.getenv("OPENWEATHER_KEY")
        print("Loaded API keys from environment")
    
    if not AQICN_KEY:
        print("AQICN_API_KEY not found!")
        print("Get a free key at: https://aqicn.org/data-platform/token/")
        return
    
    if not GEMINI_KEY:
        print("GOOGLE_API_KEY not found!")
        print("Get a key at: https://ai.google.dev/")
        print("Some features will be disabled.")
    
    # Initialize system
    system = DelhiAQIIntelligenceSystem(
        aqicn_key=AQICN_KEY,
        gemini_key=GEMINI_KEY,
        nasa_key=NASA_KEY,
        weather_key=WEATHER_KEY
    )
    
    # Example 1: Basic data collection
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic AQI Data Collection")
    print("="*80)
    
    report = system.run(user_id="demo_user")
    
    # Example 2: Natural language query
    if GEMINI_KEY:
        print("\n" + "="*80)
        print("EXAMPLE 2: Natural Language Query")
        print("="*80)
        
        response = system.query(
            "Is it safe to go jogging in Dwarka right now?",
            user_id="demo_user"
        )
        print(f"\nğŸ’¬ Response: {response}")
    
    # Example 3: Personalized recommendations
    if GEMINI_KEY:
        print("\n" + "="*80)
        print("EXAMPLE 3: Personalized Health Recommendations")
        print("="*80)
        
        user_profile = {
            "age": 35,
            "health_conditions": ["Asthma"],
            "activity_level": "High"
        }
        
        report = system.run(
            user_id="health_conscious_user",
            user_query="What precautions should I take today?",
            user_profile=user_profile
        )
    
    print("\n Demo completed!")
    
    print(f" Open {report.get('map_file', 'delhi_aqi_map.html')} to view the interactive map")
    display_map()
    
if __name__ == "__main__":
    main()




