# Install required packages
!pip install -q google-adk python-dotenv


# Import libraries
import os
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional, List
from google.adk.agents import LlmAgent
from google.adk.events import Event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Get API key from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    print("âœ… API key loaded from Kaggle Secrets")
except Exception as e:
    print(f"âš ï¸� Could not load from Kaggle Secrets: {e}")
    print("Please set GOOGLE_API_KEY in Kaggle Secrets or manually:")
    # Uncomment and add your key for testing:
    # os.environ['GOOGLE_API_KEY'] = 'your-api-key-here'


# In-memory message storage (replaces JMS queues)
class MessageStore:
    def __init__(self):
        self.questions = []
        self.responses = []
        self.summaries = []
    
    def add_question(self, message: Dict[str, Any]):
        self.questions.append(message)
        logger.info(f"Question stored: {message.get('question', '')[:50]}...")
    
    def add_response(self, message: Dict[str, Any]):
        self.responses.append(message)
        logger.info(f"Response stored (length: {len(message.get('response', ''))})")
    
    def add_summary(self, message: Dict[str, Any]):
        self.summaries.append(message)
        logger.info(f"Summary stored: {message.get('summary', '')[:50]}...")
    
    def get_stats(self) -> Dict[str, int]:
        return {
            'questions': len(self.questions),
            'responses': len(self.responses),
            'summaries': len(self.summaries)
        }
    
    def clear(self):
        self.questions.clear()
        self.responses.clear()
        self.summaries.clear()
        logger.info("ğŸ—‘ï¸� All messages cleared")

# Global message store
message_store = MessageStore()


class QuestionAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="question_agent",
            model="gemini-2.0-flash-exp",
            instruction="""Question analyzer for a car service system.
            
Analyze the user's question and provide a structured response with:
- question: The original question
- category: Type of question (service, pricing, scheduling, technical, general)
- priority: Priority level (low, medium, high, urgent)
- keywords: Key terms from the question

Return as JSON format."""
        )
    
    async def run_async(self, parent_context) -> AsyncGenerator[Event, None]:
        logger.info("=== QuestionAgent processing ===")
        
        question_content = getattr(parent_context, 'user_content', 'No question provided')
        if hasattr(question_content, 'parts'):
            question_text = ' '.join(part.text for part in question_content.parts if hasattr(part, 'text'))
        else:
            question_text = str(question_content)
        
        message = {
            "question": question_text,
            "timestamp": datetime.now().isoformat(),
            "session_id": getattr(parent_context, 'session_id', None)
        }
        
        message_store.add_question(message)
        return
        yield


class ResponseAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="response_agent",
            model="gemini-2.0-flash-exp",
            instruction="""You are a response formatter for a car service system.
            
Your task is to structure and format responses from the car service agent.
Take the agent's response and format it with:
- response: The formatted response text
- response_type: Type of response (answer, recommendation, instruction, information)
- confidence: Confidence level (low, medium, high)
- follow_up_needed: Whether follow-up is needed (true/false)

Return as JSON format."""
        )
    
    async def run_async(self, parent_context) -> AsyncGenerator[Event, None]:
        logger.info("=== ResponseAgent processing ===")
        
        response_text = getattr(parent_context, 'response_text', 'Response generated')
        
        message = {
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
            "session_id": getattr(parent_context, 'session_id', None)
        }
        
        message_store.add_response(message)
        return
        yield


class SummaryAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="summary_agent",
            model="gemini-2.0-flash-exp",
            instruction="""You are a conversation summarizer for a car service system.
            
Your task is to create concise summaries of interactions.
Generate a summary with:
- summary: Brief summary of the interaction
- key_points: List of main points discussed
- action_items: Any actions that need to be taken
- topics: Main topics covered

Return as JSON format."""
        )
    
    async def run_async(self, parent_context) -> AsyncGenerator[Event, None]:
        logger.info("=== SummaryAgent processing ===")
        
        summary_text = getattr(parent_context, 'summary_text', 'Conversation summary')
        question_content = getattr(parent_context, 'user_content', None)
        
        if question_content and hasattr(question_content, 'parts'):
            question_text = ' '.join(part.text for part in question_content.parts if hasattr(part, 'text'))
        else:
            question_text = str(question_content) if question_content else None
        
        message = {
            "summary": summary_text,
            "timestamp": datetime.now().isoformat(),
            "session_id": getattr(parent_context, 'session_id', None),
            "question": question_text
        }
        
        message_store.add_summary(message)
        return
        yield


SYSTEM_INSTRUCTION = """Car service event generator for a messaging system.

Each event should represent a realistic car service scenario and must include:
1. event_text: A detailed description of the car service event
2. main_characters: List of people involved (e.g., service manager, technician, customer)
3. car_info: Detailed information about the car (make, model, year, VIN, license plate, mileage)

Output Format:
Return a valid JSON object with the following structure:
{
  "events": [
    {
      "event_id": "unique_event_id",
      "event_text": "Detailed description of the service event",
      "main_characters": [
        {"name": "Character Name", "role": "Role in the event"}
      ],
      "car_info": {
        "make": "Car manufacturer",
        "model": "Car model",
        "year": 2020,
        "vin": "Vehicle Identification Number",
        "license_plate": "License plate number",
        "mileage": 50000
      },
      "timestamp": "ISO 8601 timestamp"
    }
  ]
}

Generate diverse car service events such as:
- Routine maintenance (oil change, tire rotation)
- Repair work (brake replacement, engine diagnostics)
- Inspection services
- Emergency repairs
- Warranty work"""


class JMSArtemisAgent(LlmAgent):
    def __init__(self):
        sub_agent_1 = QuestionAgent()
        sub_agent_2 = ResponseAgent()
        sub_agent_3 = SummaryAgent()
        
        super().__init__(
            name="jms_artemis_agent",
            model="gemini-2.0-flash-exp",
            instruction=SYSTEM_INSTRUCTION,
            sub_agents=[sub_agent_1, sub_agent_2, sub_agent_3]
        )
    
    async def run_async(self, parent_context) -> AsyncGenerator[Event, None]:
        logger.info("=== JMSArtemisAgent started ===")
        logger.info(f"Sub-agents: {[agent.name for agent in self.sub_agents]}")
        
        logger.info("Invoking QuestionAgent...")
        question_agent = self.sub_agents[0]
        async for event in question_agent.run_async(parent_context):
            pass
        
        logger.info("Running main agent...")
        response_text = ""
        async for event in super().run_async(parent_context):
            logger.info(f"Main agent event: {type(event)}")
            if hasattr(event, 'content'):
                response_text += str(event.content)
            elif hasattr(event, 'text'):
                response_text += str(event.text)
            yield event
        
        class SubAgentContext:
            def __init__(self, parent_ctx, response, summary):
                self.user_content = getattr(parent_ctx, 'user_content', None)
                self.session_id = getattr(parent_ctx, 'session_id', None)
                self.timestamp = getattr(parent_ctx, 'timestamp', None)
                self.response_text = response
                self.summary_text = summary
        
        question_text = str(getattr(parent_context, 'user_content', 'N/A'))
        summary_text = f"Q: {question_text[:50]}... | A: {response_text[:50]}..."
        sub_context = SubAgentContext(parent_context, response_text, summary_text)
        
        logger.info("Invoking ResponseAgent...")
        response_agent = self.sub_agents[1]
        async for event in response_agent.run_async(sub_context):
            pass
        
        logger.info("Invoking SummaryAgent...")
        summary_agent = self.sub_agents[2]
        async for event in summary_agent.run_async(sub_context):
            pass
        
        logger.info("=== JMSArtemisAgent completed ===")


# Create agent instance
agent = JMSArtemisAgent()
print("Agent created successfully")

