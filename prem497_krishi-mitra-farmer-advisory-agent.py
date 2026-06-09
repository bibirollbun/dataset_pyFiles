


# Install required packages
!pip install -q google-adk google-genai requests


# ============================================
# IMPORTS AND CONFIGURATION
# ============================================
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('KrishiMitra')

print("Imports loaded successfully!")


# ============================================
# SESSION AND MEMORY MANAGEMENT
# Key Concept #3: Sessions & Memory
# ============================================

@dataclass
class FarmerProfile:
    """Stores farmer-specific information for personalized advice"""
    farmer_id: str
    name: str
    location: str
    crops: List[str] = field(default_factory=list)
    soil_type: str = "unknown"
    farm_size_acres: float = 0.0
    language: str = "en"

@dataclass 
class ConversationMessage:
    """Single message in conversation history"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: str = "system"

class InMemorySessionService:
    """
    In-memory session management for storing conversation history
    and farmer profiles. Implements the Sessions & Memory concept.
    """
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.farmer_profiles: Dict[str, FarmerProfile] = {}
        logger.info("InMemorySessionService initialized")
    
    def create_session(self, session_id: str, farmer_id: str = None) -> Dict:
        """Create a new session for a farmer"""
        self.sessions[session_id] = {
            'session_id': session_id,
            'farmer_id': farmer_id,
            'messages': [],
            'created_at': datetime.now(),
            'context': {}
        }
        logger.info(f"Session created: {session_id}")
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, message: ConversationMessage):
        """Add a message to session history"""
        if session_id in self.sessions:
            self.sessions[session_id]['messages'].append(message)
            logger.debug(f"Message added to session {session_id}")
    
    def get_history(self, session_id: str, limit: int = 10) -> List[ConversationMessage]:
        """Get recent conversation history"""
        if session_id in self.sessions:
            return self.sessions[session_id]['messages'][-limit:]
        return []
    
    def save_farmer_profile(self, profile: FarmerProfile):
        """Save farmer profile for long-term memory"""
        self.farmer_profiles[profile.farmer_id] = profile
        logger.info(f"Farmer profile saved: {profile.farmer_id}")
    
    def get_farmer_profile(self, farmer_id: str) -> Optional[FarmerProfile]:
        """Retrieve farmer profile"""
        return self.farmer_profiles.get(farmer_id)

# Initialize global session service
session_service = InMemorySessionService()
print("Session service initialized!")


# ============================================
# CUSTOM TOOLS
# Key Concept #2: Tools (Custom Tools)
# ============================================

# Simulated Agricultural Knowledge Base
CROP_DATABASE = {
    'rice': {
        'soil_type': ['clay', 'loamy'],
        'water_requirement': 'high',
        'season': 'kharif',
        'pests': ['stem borer', 'leaf folder', 'brown planthopper'],
        'fertilizer': 'NPK 20:10:10'
    },
    'wheat': {
        'soil_type': ['loamy', 'sandy loam'],
        'water_requirement': 'moderate',
        'season': 'rabi',
        'pests': ['aphids', 'rust', 'termites'],
        'fertilizer': 'NPK 12:32:16'
    },
    'cotton': {
        'soil_type': ['black', 'alluvial'],
        'water_requirement': 'moderate',
        'season': 'kharif',
        'pests': ['bollworm', 'whitefly', 'jassids'],
        'fertilizer': 'NPK 10:26:26'
    },
    'sugarcane': {
        'soil_type': ['loamy', 'clay loam'],
        'water_requirement': 'very high',
        'season': 'annual',
        'pests': ['top borer', 'pyrilla', 'white grub'],
        'fertilizer': 'NPK 150:60:60 kg/ha'
    }
}

PEST_SOLUTIONS = {
    'stem borer': 'Apply Carbofuran 3G @ 25kg/ha or use pheromone traps',
    'leaf folder': 'Spray Chlorpyriphos 20EC @ 2ml/L water',
    'brown planthopper': 'Avoid excess nitrogen, spray Imidacloprid',
    'aphids': 'Spray Dimethoate 30EC @ 1.5ml/L',
    'bollworm': 'Install pheromone traps, spray Spinosad 45SC',
    'whitefly': 'Yellow sticky traps, Neem oil spray 2%'
}

class WeatherTool:
    """
    Custom tool for fetching weather information.
    Simulates weather API responses for demonstration.
    """
    def __init__(self):
        self.name = "weather_tool"
        self.description = "Get weather forecast for farming decisions"
        logger.info("WeatherTool initialized")
    
    def get_weather(self, location: str) -> Dict:
        """Simulate weather data retrieval"""
        logger.info(f"WeatherTool: Fetching weather for {location}")
        # Simulated weather data
        weather_data = {
            'location': location,
            'temperature': 28,
            'humidity': 65,
            'rainfall_probability': 40,
            'forecast': 'Partly cloudy with chances of light rain',
            'advisory': 'Good conditions for field preparation. Delay spraying if rain expected.'
        }
        return weather_data

class CropAdvisoryTool:
    """
    Custom tool for crop-related queries.
    Uses the agricultural knowledge base.
    """
    def __init__(self):
        self.name = "crop_advisory_tool"
        self.description = "Get crop recommendations and cultivation advice"
        logger.info("CropAdvisoryTool initialized")
    
    def get_crop_info(self, crop_name: str) -> Dict:
        """Get information about a specific crop"""
        logger.info(f"CropAdvisoryTool: Getting info for {crop_name}")
        crop = crop_name.lower()
        if crop in CROP_DATABASE:
            return {'status': 'found', 'crop': crop, 'info': CROP_DATABASE[crop]}
        return {'status': 'not_found', 'message': f'Crop {crop_name} not in database'}
    
    def recommend_crops(self, soil_type: str, season: str) -> List[str]:
        """Recommend crops based on soil type and season"""
        logger.info(f"CropAdvisoryTool: Recommending for soil={soil_type}, season={season}")
        recommendations = []
        for crop, info in CROP_DATABASE.items():
            if soil_type.lower() in [s.lower() for s in info['soil_type']]:
                if info['season'].lower() == season.lower() or info['season'] == 'annual':
                    recommendations.append(crop)
        return recommendations

class PestManagementTool:
    """
    Custom tool for pest identification and management.
    """
    def __init__(self):
        self.name = "pest_management_tool"
        self.description = "Identify pests and get treatment recommendations"
        logger.info("PestManagementTool initialized")
    
    def get_pest_solution(self, pest_name: str) -> Dict:
        """Get solution for a specific pest"""
        logger.info(f"PestManagementTool: Finding solution for {pest_name}")
        pest = pest_name.lower()
        if pest in PEST_SOLUTIONS:
            return {'pest': pest_name, 'solution': PEST_SOLUTIONS[pest]}
        return {'pest': pest_name, 'solution': 'Consult local agricultural officer for specific treatment'}
    
    def identify_crop_pests(self, crop_name: str) -> List[str]:
        """Get common pests for a crop"""
        crop = crop_name.lower()
        if crop in CROP_DATABASE:
            return CROP_DATABASE[crop]['pests']
        return []

# Initialize tools
weather_tool = WeatherTool()
crop_tool = CropAdvisoryTool()
pest_tool = PestManagementTool()
print("Custom tools initialized!")


# ============================================
# MULTI-AGENT SYSTEM
# Key Concept #1: Multi-Agent System (Sequential Agents)
# Key Concept #4: Observability (Logging & Tracing)
# ============================================

class BaseAgent:
    """Base class for all agents with common functionality"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.trace_id = None
        logger.info(f"Agent initialized: {name}")
    
    def set_trace_id(self, trace_id: str):
        """Set trace ID for observability"""
        self.trace_id = trace_id
    
    def log_action(self, action: str, details: str = ""):
        """Log agent action with trace ID"""
        logger.info(f"[{self.trace_id}] {self.name}: {action} - {details}")

class RouterAgent(BaseAgent):
    """
    Routes user queries to appropriate specialist agents.
    Part of the sequential agent pipeline.
    """
    def __init__(self):
        super().__init__("RouterAgent", "Routes queries to specialist agents")
        self.route_keywords = {
            'weather': ['weather', 'rain', 'temperature', 'climate', 'monsoon'],
            'crop': ['crop', 'plant', 'grow', 'seed', 'harvest', 'fertilizer', 'soil'],
            'pest': ['pest', 'disease', 'insect', 'bug', 'worm', 'fungus', 'spray']
        }
    
    def route(self, query: str) -> str:
        """Determine which agent should handle the query"""
        query_lower = query.lower()
        self.log_action("Routing", f"Query: {query[:50]}...")
        
        for agent_type, keywords in self.route_keywords.items():
            if any(kw in query_lower for kw in keywords):
                self.log_action("Routed", f"To: {agent_type}_agent")
                return agent_type
        
        self.log_action("Routed", "To: general (default)")
        return 'general'

class WeatherAgent(BaseAgent):
    """Handles weather-related queries using WeatherTool"""
    def __init__(self, weather_tool: WeatherTool):
        super().__init__("WeatherAgent", "Provides weather information and farming advisories")
        self.tool = weather_tool
    
    def process(self, query: str, location: str = "India") -> str:
        """Process weather query and return advisory"""
        self.log_action("Processing", f"Weather query for {location}")
        weather_data = self.tool.get_weather(location)
        
        response = f"""Weather Advisory for {weather_data['location']}:
- Temperature: {weather_data['temperature']}°C
- Humidity: {weather_data['humidity']}%
- Rainfall Probability: {weather_data['rainfall_probability']}%
- Forecast: {weather_data['forecast']}
- Farming Advisory: {weather_data['advisory']}"""
        
        self.log_action("Completed", "Weather response generated")
        return response

class CropAgent(BaseAgent):
    """Handles crop-related queries using CropAdvisoryTool"""
    def __init__(self, crop_tool: CropAdvisoryTool):
        super().__init__("CropAgent", "Provides crop recommendations and cultivation advice")
        self.tool = crop_tool
    
    def process(self, query: str, **kwargs) -> str:
        """Process crop query and return recommendations"""
        self.log_action("Processing", f"Crop query: {query[:50]}...")
        
        # Check if asking about specific crop
        for crop in CROP_DATABASE.keys():
            if crop in query.lower():
                info = self.tool.get_crop_info(crop)
                if info['status'] == 'found':
                    crop_info = info['info']
                    response = f"""Information about {crop.title()}:
- Suitable Soil: {', '.join(crop_info['soil_type'])}
- Water Requirement: {crop_info['water_requirement']}
- Best Season: {crop_info['season']}
- Recommended Fertilizer: {crop_info['fertilizer']}
- Common Pests: {', '.join(crop_info['pests'])}"""
                    self.log_action("Completed", f"Crop info for {crop}")
                    return response
        
        # General crop recommendation
        soil = kwargs.get('soil_type', 'loamy')
        season = kwargs.get('season', 'kharif')
        recommendations = self.tool.recommend_crops(soil, season)
        
        if recommendations:
            response = f"Recommended crops for {soil} soil in {season} season: {', '.join(recommendations)}"
        else:
            response = "Please provide more details about your soil type and season for crop recommendations."
        
        self.log_action("Completed", "Crop recommendation generated")
        return response

class PestAgent(BaseAgent):
    """Handles pest-related queries using PestManagementTool"""
    def __init__(self, pest_tool: PestManagementTool):
        super().__init__("PestAgent", "Identifies pests and provides treatment solutions")
        self.tool = pest_tool
    
    def process(self, query: str, crop_name: str = None) -> str:
        """Process pest query and return solutions"""
        self.log_action("Processing", f"Pest query: {query[:50]}...")
        
        # Check for specific pest mentions
        for pest in PEST_SOLUTIONS.keys():
            if pest in query.lower():
                solution = self.tool.get_pest_solution(pest)
                response = f"""Pest: {solution['pest'].title()}
Solution: {solution['solution']}"""
                self.log_action("Completed", f"Solution for {pest}")
                return response
        
        # If crop mentioned, show common pests
        if crop_name:
            pests = self.tool.identify_crop_pests(crop_name)
            if pests:
                solutions = []
                for pest in pests:
                    sol = self.tool.get_pest_solution(pest)
                    solutions.append(f"- {pest.title()}: {sol['solution']}")
                response = f"Common pests in {crop_name.title()} and their solutions:\n" + "\n".join(solutions)
                self.log_action("Completed", f"Pest list for {crop_name}")
                return response
        
        response = "Please describe the pest symptoms or mention the crop name for specific pest solutions."
        self.log_action("Completed", "General pest guidance")
        return response

class ResponseAgent(BaseAgent):
    """Formats and delivers final response to user"""
    def __init__(self):
        super().__init__("ResponseAgent", "Formats final response for user")
    
    def format_response(self, agent_response: str, query_type: str) -> str:
        """Format the response with greeting and follow-up"""
        self.log_action("Formatting", f"Response type: {query_type}")
        
        greeting = "Namaste! Here's the information you requested:\n\n"
        follow_up = "\n\nIs there anything else you'd like to know about farming?"
        
        formatted = greeting + agent_response + follow_up
        self.log_action("Completed", "Response formatted")
        return formatted

print("Multi-Agent System classes defined!")


# ============================================
# KRISHI MITRA ORCHESTRATOR
# Main agent that coordinates the multi-agent system
# ============================================

import uuid

class KrishiMitraOrchestrator:
    """
    Main orchestrator for the Krishi Mitra multi-agent system.
    Coordinates sequential flow: Router -> Specialist Agent -> Response Agent
    """
    def __init__(self):
        # Initialize all agents
        self.router = RouterAgent()
        self.weather_agent = WeatherAgent(weather_tool)
        self.crop_agent = CropAgent(crop_tool)
        self.pest_agent = PestAgent(pest_tool)
        self.response_agent = ResponseAgent()
        
        # Map agent types to agent instances
        self.agents = {
            'weather': self.weather_agent,
            'crop': self.crop_agent,
            'pest': self.pest_agent
        }
        
        logger.info("KrishiMitra Orchestrator initialized with all agents")
    
    def process_query(self, query: str, session_id: str, farmer_id: str = None, **kwargs) -> str:
        """
        Process a user query through the multi-agent pipeline.
        
        Flow: Query -> Router -> Specialist Agent -> Response Agent -> User
        """
        # Generate trace ID for observability
        trace_id = str(uuid.uuid4())[:8]
        logger.info(f"[{trace_id}] Processing query: {query[:50]}...")
        
        # Set trace ID for all agents
        for agent in [self.router, self.response_agent] + list(self.agents.values()):
            agent.set_trace_id(trace_id)
        
        # Store user message in session
        user_message = ConversationMessage(
            role='user',
            content=query,
            agent_name='user'
        )
        session_service.add_message(session_id, user_message)
        
        # Step 1: Route the query
        query_type = self.router.route(query)
        
        # Step 2: Process with specialist agent
        if query_type in self.agents:
            specialist_agent = self.agents[query_type]
            if query_type == 'weather':
                location = kwargs.get('location', 'India')
                agent_response = specialist_agent.process(query, location)
            elif query_type == 'crop':
                agent_response = specialist_agent.process(query, **kwargs)
            elif query_type == 'pest':
                crop_name = kwargs.get('crop_name')
                agent_response = specialist_agent.process(query, crop_name)
        else:
            agent_response = "I can help you with weather forecasts, crop advice, and pest management. Please ask about any of these topics!"
        
        # Step 3: Format response
        final_response = self.response_agent.format_response(agent_response, query_type)
        
        # Store assistant message in session
        assistant_message = ConversationMessage(
            role='assistant',
            content=final_response,
            agent_name=query_type + '_agent'
        )
        session_service.add_message(session_id, assistant_message)
        
        logger.info(f"[{trace_id}] Query processing completed")
        return final_response

# Initialize the orchestrator
orchestrator = KrishiMitraOrchestrator()
print("Krishi Mitra Orchestrator ready!")


# ============================================
# DEMO: Testing the Krishi Mitra Agent
# ============================================

# Create a session for demo farmer
session_id = "demo_session_001"
farmer_id = "farmer_001"

# Create session and farmer profile
session_service.create_session(session_id, farmer_id)

# Create farmer profile
farmer = FarmerProfile(
    farmer_id=farmer_id,
    name="Rajan Kumar",
    location="Punjab, India",
    crops=["wheat", "rice"],
    soil_type="loamy",
    farm_size_acres=5.0,
    language="en"
)
session_service.save_farmer_profile(farmer)

print("="*60)
print("KRISHI MITRA - AI FARMER ADVISORY AGENT DEMO")
print("="*60)

# Test queries
test_queries = [
    "What is the weather forecast for today?",
    "Tell me about growing rice",
    "How to control stem borer pest in my crops?"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*60}")
    print(f"Query {i}: {query}")
    print("="*60)
    response = orchestrator.process_query(query, session_id, farmer_id)
    print(response)

# Show conversation history
print("\n" + "="*60)
print("CONVERSATION HISTORY")
print("="*60)
history = session_service.get_history(session_id)
for msg in history:
    print(f"[{msg.agent_name}] {msg.role}: {msg.content[:100]}...")

print("\n" + "="*60)
print("DEMO COMPLETED SUCCESSFULLY!")
print("Key Concepts Demonstrated:")
print("1. Multi-Agent System (Sequential Agents)")
print("2. Custom Tools (Weather, Crop, Pest)")
print("3. Sessions & Memory Management")
print("4. Observability (Logging & Tracing)")
print("="*60)

