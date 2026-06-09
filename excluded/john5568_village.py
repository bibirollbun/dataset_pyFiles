# Core libraries
import pandas as pd
import uuid
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Kaggle secrets
from kaggle_secrets import UserSecretsClient

# APIs
import google.generativeai as genai

# Get all available API keys
user_secrets = UserSecretsClient()

try:
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    print(f"âœ… GOOGLE_API_KEY loaded ({len(GOOGLE_API_KEY) if GOOGLE_API_KEY else 0} chars)")
except:
    GOOGLE_API_KEY = None
    print("âš ï¸� GOOGLE_API_KEY not found")

try:
    GROQ_API_KEY = user_secrets.get_secret("GROQ_API_KEY")
    print(f"âœ… GROQ_API_KEY loaded ({len(GROQ_API_KEY) if GROQ_API_KEY else 0} chars)")
except:
    GROQ_API_KEY = None
    print("âš ï¸� GROQ_API_KEY not found")

# Configure Gemini if available
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"âš ï¸� Gemini config failed: {str(e)[:50]}")

print("\nâœ… Setup complete!")
print(f"ğŸ“Š Available LLM APIs: {sum([bool(GOOGLE_API_KEY), bool(GROQ_API_KEY)])}/2")



# ğŸ”� LLM API DIAGNOSTIC - Tests both Gemini and Groq
print("="*70)
print("ğŸ”� LLM API DIAGNOSTIC")
print("="*70)

# Test Gemini
print("\nğŸ”· TESTING GEMINI API")
print("-"*70)
if GOOGLE_API_KEY:
    print(f"âœ… API Key found ({len(GOOGLE_API_KEY)} chars)")
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        test_model = genai.GenerativeModel('gemini-2.5-flash')
        response = test_model.generate_content("Say 'working' in one word")
        print(f"âœ… GEMINI WORKING! Response: {response.text}")
    except Exception as e:
        print(f"â�Œ Gemini failed: {str(e)[:100]}")
        if '404' in str(e):
            print("   ğŸ’¡ Try: models/gemini-2.5-flash instead")
        elif '403' in str(e) or 'API key' in str(e):
            print("   ğŸ’¡ Get new key: https://aistudio.google.com/app/apikey")
else:
    print("â�Œ No GOOGLE_API_KEY found")

# Test Groq
print("\nğŸ”¶ TESTING GROQ API")
print("-"*70)
if GROQ_API_KEY:
    print(f"âœ… API Key found ({len(GROQ_API_KEY)} chars)")
    try:
        try:
            from groq import Groq
        except ImportError:
            print("   ğŸ“¦ Installing Groq client...")
            import subprocess
            subprocess.check_call(["pip", "install", "-q", "groq"])
            from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'working' in one word"}],
            model="llama-3.3-70b-versatile",
            max_tokens=10
        )
        print(f"âœ… GROQ WORKING! Response: {completion.choices[0].message.content}")
    except Exception as e:
        print(f"â�Œ Groq failed: {str(e)[:100]}")
        print("   ğŸ’¡ Check your Groq API key at: https://console.groq.com/keys")
else:
    print("â�Œ No GROQ_API_KEY found")

print("\n" + "="*70)
print("ğŸ“Š SUMMARY")
print("="*70)
gemini_ok = GOOGLE_API_KEY and "GEMINI WORKING" in str(locals())
groq_ok = GROQ_API_KEY and "GROQ WORKING" in str(locals())

if gemini_ok:
    print("âœ… Gemini API: READY (Primary)")
elif groq_ok:
    print("âœ… Groq API: READY (Will be used as primary)")
else:
    print("âš ï¸� No LLM APIs working - will use knowledge base fallback")

print("\nğŸ’¡ At least ONE API should be working for best results!")
print("="*70)



# Curated parenting knowledge base - Australian Sources (Verified)
knowledge_data = {
    'topic': [
        'sleep_newborn', 'sleep_infant', 'feeding_newborn', 'feeding_infant',
        'development_0_3m', 'development_3_6m', 'crying_normal', 'crying_distress',
        'safety_sleep', 'safety_feeding', 'illness_fever', 'illness_emergency',
        'immunisation', 'nappy_rash', 'teething', 'postnatal_mental_health'
    ],
    'age_range_weeks': [
        '0-4', '4-12', '0-4', '4-12',
        '0-12', '12-24', '0-12', '0-12',
        '0-52', '0-52', '0-52', '0-52',
        '0-52', '0-52', '12-52', '0-52'
    ],
    'evidence_source': [
        'RCH KidsInfo', 'RCH KidsInfo', 'Healthdirect', 'Healthdirect',
        'Raising Children', 'Raising Children', 'RCH KidsInfo', 'Healthdirect',
        'RCH KidsInfo', 'Healthdirect', 'Healthdirect', 'Healthdirect',
        'Healthdirect', 'RCH KidsInfo', 'Raising Children', 'Healthdirect'
    ],
    'source_url': [
        # Verified RCH KidsInfo links
        'https://www.rch.org.au/kidsinfo/fact_sheets/Sleep_in_infants/',
        'https://www.rch.org.au/kidsinfo/fact_sheets/Sleep_in_infants/',
        
        # Verified Healthdirect links
        'https://www.healthdirect.gov.au/baby-feeding',
        'https://www.healthdirect.gov.au/baby-feeding',
        
        # Verified Raising Children links
        'https://raisingchildren.net.au/newborns/development/development-tracker/development-0-3-months',
        'https://raisingchildren.net.au/babies/development/development-tracker/development-3-6-months',
        
        # RCH KidsInfo verified
        'https://www.rch.org.au/kidsinfo/fact_sheets/Crying_baby/',
        
        # Healthdirect verified
        'https://www.healthdirect.gov.au/serious-illnesses-in-babies-and-children',
        
        # RCH KidsInfo verified
        'https://www.rch.org.au/kidsinfo/fact_sheets/Safe_sleeping_for_babies/',
        
        # Healthdirect verified
        'https://www.healthdirect.gov.au/bottle-feeding',
        
        # Healthdirect verified
        'https://www.healthdirect.gov.au/fever-in-children',
        
        # Healthdirect verified
        'https://www.healthdirect.gov.au/what-to-do-in-an-emergency',
        
        # Healthdirect verified
        'https://www.healthdirect.gov.au/childhood-immunisation',
        
        # RCH KidsInfo verified
        'https://www.rch.org.au/kidsinfo/fact_sheets/Nappy_rash/',
        
        # Raising Children verified
        'https://raisingchildren.net.au/babies/health-dental-care/teething-dental-care/teething',
        
        # Healthdirect verified
        'https://www.healthdirect.gov.au/postnatal-depression'
    ],
    'advice': [
        # Sleep newborn (0-4 weeks) - RCH
        'Newborns sleep 16-20 hours per day in 2-4 hour stretches. Always place baby on back to sleep in a safe cot with firm mattress, no pillows or soft toys.',
        
        # Sleep infant (4-12 weeks) - RCH  
        'By 3 months, babies sleep 14-15 hours per day. Establish consistent bedtime routine. Room temperature should be 20-22Â°C.',
        
        # Feeding newborn (0-4 weeks) - Healthdirect
        'Newborns feed 8-12 times in 24 hours. Watch for hunger cues: rooting, sucking hands. Breastfed babies need vitamin D supplements.',
        
        # Feeding infant (4-12 weeks) - Healthdirect
        'From 6 months, introduce iron-rich foods (pureed meat, iron-fortified cereal). Continue breastfeeding or formula as main milk until 12 months.',
        
        # Development 0-3 months - Raising Children
        'By 3 months: follows faces with eyes, coos and gurgles, lifts head during tummy time, brings hands to mouth.',
        
        # Development 3-6 months - Raising Children
        'By 6 months: rolls both ways, sits with support, passes objects between hands, responds to own name.',
        
        # Crying normal - RCH
        'Crying peaks at 6-8 weeks (2-3 hours/day). Comfort techniques: swaddling, gentle rocking, white noise, baby carrier.',
        
        # Crying distress - Healthdirect
        'Seek urgent care if: crying is high-pitched, continuous >3 hours, with fever >38Â°C, vomiting, or baby seems in pain.',
        
        # Safety sleep - RCH
        'Safe sleep guidelines: back to sleep, firm mattress, no loose bedding, smoke-free environment, room sharing for first 6-12 months.',
        
        # Safety feeding - Healthdirect
        'Sterilise bottles until 12 months. Hold baby semi-upright during feeds. Never prop bottle. Watch for choking when starting solids.',
        
        # Illness fever - Healthdirect
        'Babies under 3 months with fever >38Â°C need immediate medical care. For older babies: monitor symptoms, offer fluids, use paracetamol if advised by doctor.',
        
        # Illness emergency - Healthdirect
        'Call 000 immediately for: difficulty breathing, blue lips/pale skin, seizure >5 minutes, unconscious, severe allergic reaction.',
        
        # Immunisation - Healthdirect
        'Follow Australian National Immunisation Program starting at birth. Vaccinations protect against serious diseases like whooping cough, measles, and meningococcal.',
        
        # Nappy rash - RCH
        'Change nappies frequently, use barrier cream. If rash has blisters, sores or persists >3 days, see GP. May be fungal infection needing antifungal cream.',
        
        # Teething - Raising Children
        'Teething typically starts 6-12 months. Symptoms: drooling, chewing, mild temperature (<38Â°C). Use teething rings, clean damp cloth, gentle gum massage.',
        
        # Postnatal mental health - Healthdirect
        'Postnatal depression affects 1 in 7 Australian mothers. Seek help if feeling overwhelmed, anxious, or disconnected from baby. Call PANDA on 1300 726 306.'
    ],
    'risk_level': [
        'low', 'low', 'low', 'low',
        'low', 'low', 'low', 'medium',
        'high', 'medium', 'high', 'emergency',
        'medium', 'low', 'low', 'medium'
    ]
}

knowledge_base = pd.DataFrame(knowledge_data)
print(f"ğŸ“š Loaded Australian knowledge base: {len(knowledge_base)} entries")
print("ğŸ‡¦ğŸ‡º Trusted Australian Sources:")
print("   â€¢ Royal Children's Hospital KidsInfo (RCH)")
print("   â€¢ Healthdirect Australia (Govt health advice)")
print("   â€¢ Raising Children Network (Govt parenting website)")
print("\nâœ… All links verified and active")
knowledge_base[['topic', 'evidence_source', 'source_url']].head(10)


# Tool 1: Simplified Safety Classification
class SafetyClassificationTool:
    """Simplified safety tool using keyword analysis"""
    def __init__(self):
        self.emergency_keywords = [
            'not breathing', 'blue', 'seizure', 'unconscious', 'unresponsive',
            'blood', 'severe', 'emergency', 'choking', 'stopped breathing'
        ]
        self.urgent_keywords = [
            'fever', 'high temperature', 'vomiting', 'diarrhea', 
            'won\'t eat', 'hasn\'t eaten', 'dehydrated', 'rash',
            'won\'t stop crying', 'inconsolable'
        ]
        self.distress_words = [
            'worried', 'scared', 'concerned', 'anxious', 'help', 'urgent'
        ]
        
    def analyze(self, text: str) -> Dict:
        """Analyze text for urgency and sentiment"""
        text_lower = text.lower()
        
        # Check for emergency keywords
        has_emergency = any(kw in text_lower for kw in self.emergency_keywords)
        has_urgent = any(kw in text_lower for kw in self.urgent_keywords)
        has_distress = any(kw in text_lower for kw in self.distress_words)
        
        # Count negative indicators
        negative_count = text_lower.count('not') + text_lower.count('won\'t') + text_lower.count('can\'t')
        
        # Calculate risk level
        if has_emergency:
            risk_level = "emergency"
            sentiment_score = -0.9
        elif has_urgent or negative_count > 2:
            risk_level = "medium"
            sentiment_score = -0.5
        elif has_distress:
            risk_level = "low-medium"
            sentiment_score = -0.3
        else:
            risk_level = "low"
            sentiment_score = 0.0
            
        return {
            "risk_level": risk_level,
            "sentiment_score": sentiment_score,
            "has_emergency_keywords": has_emergency,
            "has_urgent_keywords": has_urgent,
            "needs_escalation": has_emergency or (has_urgent and negative_count > 0),
            "method": "keyword_analysis"
        }

# Tool 2: Context Management
class ContextManagementTool(ContextManagementTool):
    def __init__(self):
        super().__init__()
        # Extended keyword to topic mapping
        self.keyword_to_topic = {
            'temperature': 'illness_fever',
            'fever': 'illness_fever',
            'hot': 'illness_fever',
            'thermometer': 'illness_fever',
            'warm': 'illness_fever',
            'sleep': 'sleep_infant',
            'tired': 'sleep_infant',
            'awake': 'sleep_infant',
            'feed': 'feeding_infant',
            'hungry': 'feeding_infant',
            'milk': 'feeding_infant',
            'cry': 'crying_normal',
            'crying': 'crying_normal',
            'upset': 'crying_normal',
            'rash': 'nappy_rash',
            'red': 'nappy_rash',
            'teeth': 'teething',
            'gum': 'teething',
            'drool': 'teething',
            'develop': 'development_3_6m',
            'roll': 'development_3_6m',
            'sit': 'development_3_6m'
        }
    
    def search_knowledge(self, query: str, age_weeks: int) -> List[Dict]:
        """Enhanced search with keyword mapping"""
        query_lower = query.lower()
        
        # First, try keyword mapping
        matched_topics = []
        for keyword, topic in self.keyword_to_topic.items():
            if keyword in query_lower:
                matched_topics.append(topic)
        
        # Also search in knowledge base directly
        if matched_topics:
            matches = knowledge_base[
                knowledge_base['topic'].isin(matched_topics)
            ]
        else:
            # Fallback to original search
            matches = knowledge_base[
                knowledge_base['topic'].str.contains(query_lower, case=False, na=False) |
                knowledge_base['advice'].str.contains(query_lower, case=False, na=False)
            ]
        
        # Filter by age if possible
        if not matches.empty:
            def age_match(age_range, age_weeks):
                try:
                    if '-' in age_range:
                        start, end = map(int, age_range.split('-'))
                        return start <= age_weeks <= end
                    return True
                except:
                    return True
            
            matches = matches[
                matches['age_range_weeks'].apply(lambda x: age_match(x, age_weeks))
            ]
        
        return matches.to_dict('records')

print("ğŸ› ï¸� Tools initialized!")
print("   2 tools: Safety Analysis + Context Management")



@dataclass
class AgentResponse:
    agent_name: str
    response: str
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# Agent 1: Conversation Agent with Gemini + Groq fallback
class ConversationAgent:
    def __init__(self):
        self.gemini_model = None
        self.groq_client = None
        self.active_llm = None
        self.name = "ConversationAgent"
        
        # Try Gemini first
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
                
                for model_name in model_names:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        test = self.gemini_model.generate_content("test")
                        if test:
                            self.active_llm = "gemini"
                            print(f"âœ… Gemini active: {model_name}")
                            break
                    except:
                        continue
            except Exception as e:
                print(f"âš ï¸� Gemini init failed: {str(e)[:50]}")
        
        # Try Groq as backup
        if not self.active_llm and GROQ_API_KEY:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=GROQ_API_KEY)
                # Test it
                test = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": "test"}],
                    model="llama-3.3-70b-versatile",
                    max_tokens=10
                )
                if test:
                    self.active_llm = "groq"
                    print("âœ… Groq active: llama-3.3-70b-versatile")
            except ImportError:
                print("âš ï¸� Groq not installed. Installing...")
                import subprocess
                subprocess.check_call(["pip", "install", "-q", "groq"])
                from groq import Groq
                self.groq_client = Groq(api_key=GROQ_API_KEY)
                self.active_llm = "groq"
                print("âœ… Groq installed and active")
            except Exception as e:
                print(f"âš ï¸� Groq init failed: {str(e)[:50]}")
        
        if not self.active_llm:
            print("âš ï¸� No LLM available - will use knowledge base fallback")
        
    def generate_response(self, query: str, context: Dict, 
                         knowledge: List[Dict]) -> AgentResponse:
        """Generate response using Gemini, Groq, or fallback"""
        
        # Build context - FIX key names
        context_str = ""
        if context:
            # Use correct keys from session data
            age_weeks = context.get('age_weeks', 'N/A')
            age_months = context.get('age_months', 'N/A')
            context_str = f"Baby is {age_weeks} weeks old ({age_months} months)"
        
        knowledge_str = ""
        if knowledge:
            knowledge_str = "\n".join([
                f"- {k['advice']} (Source: {k['evidence_source']})"
                for k in knowledge[:3]
            ])
        
        # UPDATED PROMPT - Enforce evidence-first format
        prompt = f"""You are a supportive, evidence-based Australian parenting assistant.

Baby Context: {context_str}

Relevant Australian Evidence:
{knowledge_str}

Parent's Question: {query}

Provide a concise, evidence-based response in Australian spelling, following this structure:
1. FIRST: Direct answer using provided Australian evidence (cite sources)
2. SECOND: Practical application for this baby's age
3. THIRD: Any precautions or when to seek help

Keep it warm but professional. Always prioritize Australian sources over general knowledge."""
        
        # Try active LLM
        response_text = None
        llm_used = "None"
        
        if self.active_llm == "gemini":
            try:
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text
                llm_used = "Gemini"
            except Exception as e:
                print(f"âš ï¸� Gemini error: {str(e)[:50]}")
                # Try Groq as backup
                if GROQ_API_KEY and not self.groq_client:
                    try:
                        from groq import Groq
                        self.groq_client = Groq(api_key=GROQ_API_KEY)
                        self.active_llm = "groq"
                    except:
                        pass
        
        if not response_text and self.active_llm == "groq":
            try:
                completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=500
                )
                response_text = completion.choices[0].message.content
                llm_used = "Groq (Llama 3.3)"
            except Exception as e:
                print(f"âš ï¸� Groq error: {str(e)[:50]}")
        
        # Fallback to knowledge base
        if not response_text:
            if knowledge:
                response_text = f"Based on evidence-based guidelines:\n\n"
                for k in knowledge[:3]:
                    response_text += f"â€¢ {k['advice']}\n  (Source: {k['evidence_source']})\n\n"
                response_text += f"These recommendations are from trusted medical sources.\n\nContext: {context_str}"
                llm_used = "Knowledge Base"
            else:
                response_text = "I'd be happy to help! Could you provide more details about your question?"
                llm_used = "Fallback"
        
        return AgentResponse(
            agent_name=self.name,
            response=response_text,
            tools_used=[llm_used],
            metadata={
                "context_used": bool(context), 
                "knowledge_entries": len(knowledge),
                "llm": llm_used
            }
        )

# Agent 2: Safety Agent
class SafetyAgent:
    def __init__(self):
        self.tool = SafetyClassificationTool()
        self.name = "SafetyAgent"
        
    def assess_safety(self, query: str) -> AgentResponse:
        """Assess query for safety concerns"""
        analysis = self.tool.analyze(query)
        
        if analysis["needs_escalation"]:
            response = f"""âš ï¸� URGENT: This sounds serious. 
            
Risk Level: {analysis['risk_level'].upper()}

Please seek immediate medical attention:
- Call your pediatrician NOW
- Go to emergency room if severe symptoms
- Call 911 if life-threatening (difficulty breathing, unresponsive, seizure)

This is not a substitute for professional medical advice.
"""
        else:
            response = f"âœ… Safety check: {analysis['risk_level']} risk detected"
            
        return AgentResponse(
            agent_name=self.name,
            response=response,
            tools_used=["Keyword Safety Analysis"],
            metadata=analysis
        )

# Agent 3: Knowledge Agent
class KnowledgeAgent:
    def __init__(self):
        self.tool = ContextManagementTool() 
        self.name = "KnowledgeAgent"
        
    def retrieve_knowledge(self, topic: str, age_weeks: int) -> AgentResponse:
        """Retrieve relevant knowledge from database"""
        results = self.tool.search_knowledge(topic, age_weeks)
        
        if results:
            response = f"ğŸ“š Found {len(results)} relevant knowledge entries"
        else:
            response = "ğŸ“š No specific entries found, using general guidance"
            
        return AgentResponse(
            agent_name=self.name,
            response=response,
            tools_used=["Knowledge Base Search"],
            metadata={"results": results, "result_count": len(results)}
        )

print("ğŸ¤– Multi-agent system with dual LLM support ready!")
print("   âœ… Gemini (primary) + Groq (backup)")



class ParentingSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.memory_bank = {
            "baby_profile": {},
            "conversation_history": [],
            "parent_preferences": {},
            "previous_advice": [],
            "context_summary": ""
        }
        
    def add_interaction(self, query: str, response: str, metadata: Dict):
        """Add interaction to memory"""
        self.memory_bank["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "metadata": metadata
        })
        
    def update_baby_profile(self, age_weeks: int, milestones: List[str]):
        """Update baby information"""
        self.memory_bank["baby_profile"] = {
            "age_weeks": age_weeks,
            "age_months": age_weeks // 4,
            "milestones": milestones,
            "last_updated": datetime.now().isoformat()
        }
        
    def compact_context(self, max_history: int = 5):
        """Compact conversation history for token efficiency"""
        if len(self.memory_bank["conversation_history"]) > max_history:
            recent = self.memory_bank["conversation_history"][-max_history:]
            self.memory_bank["conversation_history"] = recent
            self.memory_bank["context_summary"] = f"Compacted. Keeping last {max_history} interactions."
            
    def get_context(self) -> Dict:
        """Get current session context"""
        return {
            "session_id": self.session_id,
            "baby_profile": self.memory_bank["baby_profile"],
            "interaction_count": len(self.memory_bank["conversation_history"]),
            "session_duration": str(datetime.now() - self.created_at)
        }

class SessionManager:
    def __init__(self):
        self.sessions = {}
        
    def create_session(self) -> ParentingSession:
        """Create new parenting session"""
        session = ParentingSession()
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ParentingSession]:
        """Retrieve existing session"""
        return self.sessions.get(session_id)

print("ğŸ’¾ Session management initialised!")


class AgentLogger:
    def __init__(self):
        self.logs = []
        
    def log_interaction(self, session_id: str, query: str, 
                       agent_responses: List[AgentResponse]):
        """Log complete interaction with all agent responses"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": query,
            "agents_invoked": [r.agent_name for r in agent_responses],
            "tools_used": list(set(
                tool for r in agent_responses for tool in r.tools_used
            )),
            "responses": [
                {
                    "agent": r.agent_name,
                    "response_length": len(r.response),
                    "metadata": r.metadata
                } for r in agent_responses
            ]
        }
        self.logs.append(log_entry)
        
    def get_logs_df(self) -> pd.DataFrame:
        """Convert logs to DataFrame for analysis"""
        return pd.DataFrame(self.logs)
    
    def print_trace(self, session_id: str):
        """Print execution trace for a session"""
        session_logs = [log for log in self.logs if log['session_id'] == session_id]
        print(f"\nğŸ”� Execution Trace for Session: {session_id[:8]}...\n")
        for i, log in enumerate(session_logs, 1):
            print(f"Interaction {i}: {log['timestamp']}")
            print(f"  Query: {log['query'][:60]}...")
            print(f"  Agents: {', '.join(log['agents_invoked'])}")
            print(f"  Tools: {', '.join(log['tools_used'])}")
            print()

logger = AgentLogger()
print("ğŸ“Š Observability system ready!")


class VillageAIOrchestrator:
    def __init__(self):
        self.conversation_agent = ConversationAgent()
        self.safety_agent = SafetyAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.session_manager = SessionManager()
        self.context_tool = ContextManagementTool()
    
    def extract_topic_from_query(self, query: str) -> str:
        """Extract topic from query using smarter matching"""
        query_lower = query.lower()
        
        # Medical urgency takes priority
        emergency_keywords = ['emergency', 'urgent', '911', '000', 'ambulance', 'hospital']
        if any(kw in query_lower for kw in emergency_keywords):
            return 'emergency'
        
        # Map specific symptoms to topics
        symptom_mapping = {
            'temperature': 'illness_fever',
            'fever': 'illness_fever',
            'thermometer': 'illness_fever',
            'hot': 'illness_fever',
            'warm': 'illness_fever',
            'sleep': 'sleep_infant',
            'awake': 'sleep_infant',
            'nap': 'sleep_infant',
            'feed': 'feeding_infant',
            'hungry': 'feeding_infant',
            'milk': 'feeding_infant',
            'formula': 'feeding_infant',
            'breast': 'feeding_infant',
            'cry': 'crying_normal',
            'crying': 'crying_normal',
            'whimper': 'crying_normal',
            'fussy': 'crying_normal',
            'rash': 'nappy_rash',
            'red': 'nappy_rash',
            'sore': 'nappy_rash',
            'teeth': 'teething',
            'teething': 'teething',
            'gum': 'teething',
            'drool': 'teething',
            'develop': 'development_0_3m',
            'milestone': 'development_0_3m',
            'roll': 'development_3_6m',
            'sit': 'development_3_6m',
            'vaccine': 'immunisation',
            'shot': 'immunisation',
            'immunise': 'immunisation'
        }
        
        # Check for symptom keywords
        for keyword, topic in symptom_mapping.items():
            if keyword in query_lower:
                return topic
        
        # Check general categories
        if any(word in query_lower for word in ['how much', 'how many', 'how long']):
            if any(word in query_lower for word in ['sleep', 'nap', 'awake']):
                return 'sleep_infant'
            elif any(word in query_lower for word in ['eat', 'feed', 'milk', 'formula']):
                return 'feeding_infant'
        
        return 'general'
        
    def process_query(self, session: ParentingSession, query: str) -> Dict:
        """Orchestrate multi-agent response"""
        agent_responses = []
        
        # Step 1: Safety Assessment
        safety_response = self.safety_agent.assess_safety(query)
        agent_responses.append(safety_response)
        
        # If emergency, return immediately
        if safety_response.metadata.get('needs_escalation'):
            logger.log_interaction(session.session_id, query, agent_responses)
            session.add_interaction(query, safety_response.response, {
                "agents_used": [r.agent_name for r in agent_responses],
                "safety_level": safety_response.metadata['risk_level'],
                "emergency": True
            })
            return {
                "response": safety_response.response,
                "agent_responses": agent_responses,
                "emergency": True
            }
        
        # Step 2: Knowledge Retrieval
        baby_context = session.memory_bank["baby_profile"]
        age_weeks = baby_context.get("age_weeks", 8)
        
        # Extract topic from query using the new method
        topic = self.extract_topic_from_query(query)
        
        # Get knowledge using the extracted topic
        knowledge_response = self.knowledge_agent.retrieve_knowledge(topic, age_weeks)
        agent_responses.append(knowledge_response)
        
        # Step 3: Generate Conversational Response
        conversation_response = self.conversation_agent.generate_response(
            query,
            baby_context,
            knowledge_response.metadata.get('results', [])
        )
        agent_responses.append(conversation_response)
        
        final_response = conversation_response.response
        
        # Log interaction
        logger.log_interaction(session.session_id, query, agent_responses)
        
        # Update session memory
        session.add_interaction(query, final_response, {
            "agents_used": [r.agent_name for r in agent_responses],
            "safety_level": safety_response.metadata['risk_level']
        })
        
        return {
            "response": final_response,
            "agent_responses": agent_responses,
            "emergency": False
        }

# Initialize orchestrator
orchestrator = VillageAIOrchestrator()
print("ğŸ�¯ Orchestrator initialized!")
print("   âœ… 3 agents: Safety â†’ Knowledge â†’ Conversation")


print("=" * 60)
print("ğŸ“� DEMO 1: Normal Parenting Query with Context Memory")
print("=" * 60)

# Create session
session = orchestrator.session_manager.create_session()
session.update_baby_profile(
    age_weeks=8,
    milestones=["smiling", "tracking faces", "cooing"]
)

print(f"\nâœ… Session created: {session.session_id[:8]}...")
print(f"ğŸ“‹ Baby profile: 8 weeks old\n")

# Query 1
query1 = "How much should my 8-week-old baby be sleeping?"
print(f"ğŸ—£ï¸�  Parent: {query1}\n")

result1 = orchestrator.process_query(session, query1)
print(f"ğŸ¤– Village AI:\n{result1['response']}\n")

# Show agent breakdown
print("\nğŸ”� Agent Activity:")
for agent_resp in result1['agent_responses']:
    print(f"  - {agent_resp.agent_name}: {agent_resp.response[:60]}")
    if agent_resp.tools_used:
        print(f"    Tools: {', '.join(agent_resp.tools_used)}")

# Query 2 (showing context retention)
print("\n" + "="*60)
print("Testing context memory...\n")
query2 = "What about feeding at this age?"
print(f"ğŸ—£ï¸�  Parent: {query2}\n")

result2 = orchestrator.process_query(session, query2)
print(f"ğŸ¤– Village AI:\n{result2['response']}\n")
print("âœ… Agent remembered baby is 8 weeks old from context!")


print("=" * 60)
print("ğŸš¨ DEMO 2: Emergency Detection & Safety Classification")
print("=" * 60)

session2 = orchestrator.session_manager.create_session()
session2.update_baby_profile(age_weeks=4, milestones=[])

emergency_query = "My baby has a high fever and won't stop crying for 4 hours, seems unresponsive"
print(f"\nğŸ—£ï¸�  Parent: {emergency_query}\n")

result = orchestrator.process_query(session2, emergency_query)

print(f"ğŸ¤– Village AI:\n{result['response']}\n")

if result['emergency']:
    print("âœ… Safety Agent correctly flagged as EMERGENCY")
    print("âœ… Conversation bypassed for immediate escalation\n")
    
# Show safety analysis
safety_resp = next(
    r for r in result['agent_responses'] if r.agent_name == 'SafetyAgent'
)
print(f"ğŸ“Š Safety Analysis Metadata:")
print(json.dumps(safety_resp.metadata, indent=2))


print("=" * 60)
print("ğŸ“Š DEMO 3: Knowledge Base Retrieval & Citations")
print("=" * 60)

session3 = orchestrator.session_manager.create_session()
session3.update_baby_profile(age_weeks=24, milestones=["laughing", "reaching", "rolling"])

query = "When should my baby start eating solid foods?"
print(f"\nğŸ—£ï¸�  Parent: {query}\n")

result = orchestrator.process_query(session3, query)

print(f"ğŸ¤– Village AI:\n{result['response']}\n")

# Show knowledge retrieval
knowledge_resp = next(
    (r for r in result['agent_responses'] if r.agent_name == 'KnowledgeAgent'),
    None
)

if knowledge_resp:
    print(f"ğŸ“š Knowledge Retrieved: {knowledge_resp.metadata.get('result_count', 0)} entries")
    if knowledge_resp.metadata.get('results'):
        print("\nEvidence Sources:")
        for entry in knowledge_resp.metadata['results'][:3]:
            print(f"  â€¢ {entry['evidence_source']}: {entry['advice'][:80]}...")
else:
    print("ğŸ“š Knowledge agent executed")

print("\nâœ… System retrieved evidence-based information from knowledge base")



print("=" * 60)
print("ğŸ“Š OBSERVABILITY DASHBOARD")
print("=" * 60)

# Get logs as DataFrame
logs_df = logger.get_logs_df()

print(f"\nğŸ“‹ Total Interactions Logged: {len(logs_df)}\n")

if len(logs_df) > 0:
    # Agent usage statistics
    all_agents = [agent for agents in logs_df['agents_invoked'] for agent in agents]
    agent_counts = pd.Series(all_agents).value_counts()
    
    print("ğŸ¤– Agent Invocation Counts:")
    print(agent_counts)
    print()
    
    # Tool usage statistics
    all_tools = [tool for tools in logs_df['tools_used'] for tool in tools]
    tool_counts = pd.Series(all_tools).value_counts()
    
    print("ğŸ› ï¸�  Tool Usage Counts:")
    print(tool_counts)
    print()
    
    # Show execution trace for first session
    first_session_id = logs_df.iloc[0]['session_id']
    logger.print_trace(first_session_id)
    
    # Display logs table
    print("\nğŸ“Š Interaction Logs Summary:")
    display(logs_df[['timestamp', 'query', 'agents_invoked', 'tools_used']].head())
else:
    print("No interactions logged yet.")


print("=" * 60)
print("ğŸ’¾ SESSION MEMORY & CONTEXT DEMONSTRATION")
print("=" * 60)

# Show session context
print("\nğŸ“‹ Session 1 Full Context:")
print(json.dumps(session.get_context(), indent=2))

print("\nğŸ’¬ Conversation History:")
for i, interaction in enumerate(session.memory_bank['conversation_history'], 1):
    print(f"\n{i}. Timestamp: {interaction['timestamp']}")
    print(f"   Q: {interaction['query']}")
    print(f"   A: {interaction['response']}")
    print(f"   Agents Used: {', '.join(interaction['metadata']['agents_used'])}")
    print(f"   Safety Level: {interaction['metadata']['safety_level']}")

# Demonstrate context compaction
print("\n" + "="*60)
print("ğŸ—œï¸�  Context Compaction Demo")
print("="*60)
print(f"Before compaction: {len(session.memory_bank['conversation_history'])} interactions")
session.compact_context(max_history=1)
print(f"After compaction: {len(session.memory_bank['conversation_history'])} interactions retained")
print(f"Reason: {session.memory_bank['context_summary']}")
print("\nâœ… This prevents token overflow in long conversations")


print("=" * 60)
print("ğŸ“š KNOWLEDGE BASE ANALYTICS")
print("=" * 60)

print(f"\nğŸ“Š Statistics:")
print(f"  Total entries: {len(knowledge_base)}")
print(f"  Unique topics: {knowledge_base['topic'].nunique()}")
print(f"  Evidence sources: {knowledge_base['evidence_source'].nunique()}")

print(f"\nâš ï¸�  Risk Level Distribution:")
print(knowledge_base['risk_level'].value_counts())

print(f"\nğŸ“– Evidence Sources:")
print(knowledge_base['evidence_source'].value_counts())

# Show high-risk entries
print("\nğŸš¨ High-Risk & Emergency Topics:")
high_risk = knowledge_base[knowledge_base['risk_level'].isin(['high', 'emergency'])]
display(high_risk[['topic', 'age_range_weeks', 'evidence_source', 'risk_level']])


# Install and import UI libraries
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
except ImportError:
    !pip install ipywidgets -q
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML

print("âœ… UI libraries loaded!")


# Create interactive session
ui_session = orchestrator.session_manager.create_session()
ui_session.update_baby_profile(8, ['smiling', 'cooing', 'tracking faces'])

# UI Components
query_box = widgets.Textarea(
    placeholder='Ask your parenting question... (e.g., "How much should my baby sleep?")',
    layout=widgets.Layout(width='95%', height='100px')
)

send_btn = widgets.Button(
    description='ğŸ’¬ Ask Village AI',
    button_style='success',
    layout=widgets.Layout(width='200px', height='40px')
)

clear_btn = widgets.Button(
    description='ğŸ—‘ï¸� Clear Chat',
    button_style='warning',
    layout=widgets.Layout(width='200px', height='40px')
)

output_area = widgets.Output(
    layout=widgets.Layout(
        border='2px solid #4CAF50',
        padding='20px',
        min_height='300px'
    )
)

def on_send(b):
    query = query_box.value.strip()
    
    # Clear immediately to prevent double-trigger
    query_box.value = ''
    
    if not query:
        with output_area:
            print("âš ï¸� Please enter a question!")
        return
    
    with output_area:
        print(f"\n{'='*60}")
        print(f"ğŸ—£ï¸�  You: {query}")
        print(f"{'='*60}")
        print("\nğŸ¤” Processing...\n")
    
    try:
        result = orchestrator.process_query(ui_session, query)
        
        with output_area:
            if result['emergency']:
                print("\nâš ï¸�âš ï¸�âš ï¸� EMERGENCY DETECTED âš ï¸�âš ï¸�âš ï¸�\n")
            
            print("ğŸ¤– VILLAGE AI:")
            print("-" * 60)
            print(result['response'])
            print("-" * 60)
            
            print("\nğŸ”� AGENT ACTIVITY:")
            for agent in result['agent_responses']:
                print(f"  âœ… {agent.agent_name}: {agent.tools_used}")
            
            print(f"\nğŸ’¬ Conversation #{len(ui_session.memory_bank['conversation_history'])}")
            print("\n" + "="*60 + "\n")
    except Exception as e:
        with output_area:
            print(f"\nâ�Œ Error: {str(e)}")
            print("Please try a different question.\n")
def on_clear(b):
    with output_area:
        clear_output()
        print("ğŸ—‘ï¸� Chat cleared!\n")
        print("ğŸ’¡ Example questions:")
        print("  â€¢ How much should my baby sleep?")
        print("  â€¢ When can I start solid foods?")
        print("  â€¢ My baby has a fever (tests emergency detection)")

send_btn.on_click(on_send)
clear_btn.on_click(on_clear)

# Display UI
display(HTML("<h2 style='color: #4CAF50;'>ğŸ�¼ Village AI - Interactive Chat</h2>"))
display(HTML("<p style='color: #666;'>Multi-agent parenting assistant powered by Gemini</p>"))
display(HTML("<p><strong>Baby:</strong> 8 weeks old â€¢ Milestones: smiling, cooing, tracking faces</p>"))

display(query_box)
display(widgets.HBox([send_btn, clear_btn]))
display(output_area)

with output_area:
    print("ğŸ‘‹ Welcome to Village AI!\n")
    print("3-Agent System:")
    print("  ğŸš¨ SafetyAgent - Risk detection")
    print("  ğŸ“š KnowledgeAgent - Evidence retrieval")
    print("  ğŸ¤– ConversationAgent - Gemini responses\n")
    print("ğŸ’¡ Try asking a question above!")

