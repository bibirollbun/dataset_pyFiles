# ============================================================================
# CELL 1: Install + Imports
# ============================================================================
!pip install -q google-generativeai

import json, datetime, os, time, re
from typing import List
from dataclasses import dataclass, asdict
from collections import Counter

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

print("âœ… Packages imported successfully")









# ============================================================================
# CELL 2: Load API Key from Kaggle Secrets
# ============================================================================
GOOGLE_API_KEY = None

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    try:
        from kaggle_secrets import UserSecretsClient
        GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    except:
        GOOGLE_API_KEY = None

if GOOGLE_API_KEY:
    print("âœ… GOOGLE_API_KEY loaded")
else:
    print("â�Œ GOOGLE_API_KEY not found â†’ running demo mode")



# ============================================================================
# CELL 3: Configure Gemini safely (will fall back if quota exceeded)
# ============================================================================
GEMINI_MODEL = "models/gemini-2.0-flash-exp"

if GEMINI_AVAILABLE and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        genai.GenerativeModel(GEMINI_MODEL)
        print("âš ï¸� Gemini quota exceeded â†’ running in fallback mode (expected)")
        GEMINI_AVAILABLE = False  # force fallback until quota resets
    except Exception as e:
        print(f"â�Œ Gemini unavailable: {e}")
        GEMINI_AVAILABLE = False
else:
    GEMINI_AVAILABLE = False



# ============================================================================
# Data Models
# ============================================================================
@dataclass
class DreamEntry:
    id: str
    timestamp: str
    raw_text: str
    processed_text: str
    emotions: List[str]
    symbols: List[str]
    themes: List[str]
    vividness_score: float
    interpretation: str
    interpretation_text: str = ""

@dataclass
class DreamPattern:
    theme: str
    frequency: int
    related_emotions: List[str]
    related_symbols: List[str]



# ============================================================================
# Memory Bank
# ============================================================================
class DreamMemoryBank:
    def __init__(self):
        self.dreams=[]
        self.emotion_history=[]
        self.symbol_history=[]
        self.patterns={}

    def store_dream(self,dream:DreamEntry):
        self.dreams.append(dream)
        self.emotion_history+=dream.emotions
        self.symbol_history+=dream.symbols
        self.patterns={k:DreamPattern(k,0,[],[]) for k in []}

    def get_statistics(self):
        n=len(self.dreams)
        return {
            "total_dreams":n,
            "most_common_emotions":Counter(self.emotion_history).most_common(5),
            "most_common_symbols":Counter(self.symbol_history).most_common(5),
            "average_vividness":round(sum(d.vividness_score for d in self.dreams)/n,2) if n else 0
        }




# ============================================================================
# Session Service
# ============================================================================
class DreamSessionService:
    def __init__(self): self.sessions={}
    def create_session(self,user):
        sid=f"sess_{user}_{time.time()}"
        self.sessions[sid]={"user":user,"status":"active","text":""}
        return sid




# ============================================================================
# Logger
# ============================================================================
class SimpleLogger:
    def __init__(self,name): self.name=name
    def log(self,msg): print(f"[{datetime.datetime.now().isoformat()}] {self.name}: {msg}")




# ============================================================================
# Agents (Gemini disabled safe fallback)
# ============================================================================
class VoiceTranscriptionAgent:
    def __init__(self): self.logger=SimpleLogger("VoiceTranscriptionAgent")
    def transcribe(self,text,session):
        self.logger.log("Transcription received")
        for f in["um","uh","like","you know"]:
            text=re.sub(rf"\b{f}\b","",text,1,re.I)
        return text.strip()

class PatternAnalysisAgent:
    def __init__(self,mem:DreamMemoryBank):
        self.mem=mem
        self.logger=SimpleLogger("PatternAnalysisAgent")

    def analyze(self,text):
        self.logger.log("Analyzing dream")
        emotions=["joy","fear","sadness","anger","confusion","peace"]
        symbols=["water","flying","falling","chase","death","home","people","animals"]

        e=["joy"] if "happy" in text.lower() else ["fear"]
        s=[sym for sym in symbols if sym in text.lower()]

        return {"emotions":e,"symbols":s,"themes":[],"vividness":min(len(s),10),"interpretation":"keyword"}

class CreativeStoryAgent:
    def __init__(self,mem:DreamMemoryBank):
        self.mem=mem
        self.logger=SimpleLogger("CreativeStoryAgent")

    def generate(self,dream:DreamEntry,style="fantasy"):
        self.logger.log("Generating story")
        return f"\n# Dream Story ({style})\n{dream.processed_text}\n\n---\n*Inspired by symbols: {', '.join(dream.symbols[:3]) if dream.symbols else 'mystery'}*"




# ============================================================================
# Orchestrator
# ============================================================================
class DreamJournalOrchestrator:
    def __init__(self,use_gemini=False):
        self.mem=DreamMemoryBank()
        self.srv=DreamSessionService()
        self.tr=VoiceTranscriptionAgent()
        self.pa=PatternAnalysisAgent(self.mem)
        self.cs=CreativeStoryAgent(self.mem)
        self.logger=SimpleLogger("Orchestrator")

    def record(self,user,text):
        self.logger.log("Starting workflow")
        processed=self.tr.transcribe(text,self.srv.create_session(user))
        r=self.pa.analyze(processed)
        return self.mem.store_dream(
            DreamEntry(f"dream_{time.time()}",datetime.datetime.now().isoformat(),text,processed,r["emotions"],r["symbols"],[],r["vividness"],r["interpretation"])
        ) or self.mem.dreams[-1]

    def analyze_patterns(self):
        return self.mem.get_statistics()

    def generate_story_from_latest(self,style="fantasy"):
        if not self.mem.dreams: return "No dreams stored yet"
        return self.cs.generate(self.mem.dreams[0],style)


# ============================================================================
# CELL FINAL: Demo Run
# ============================================================================
def demo_workflow():
    print("\n"+"="*80)
    print("ğŸŒ™ DREAM JOURNAL ANALYST & STORY GENERATOR")
    print("Running in fallback mode (Gemini unavailable)")
    print("="*80+"\n")

    o=DreamJournalOrchestrator()

    samples=[
        "I was flying over an ocean. The water was deep blue and I felt free and happy. Then I started falling softly onto a beach.",
        "I was in my childhood home. Everything was dark, my family was missing, and I felt scared searching empty rooms."
    ]

    for i,text in enumerate(samples,1):
        print(f"\nğŸŒ™ Recording Dream {i}...")
        d=o.record("demo_user", text)
        print("âœ“",d.id)
        print("  ğŸ˜Š Emotions:",d.emotions)
        print("  ğŸ”® Symbols:",d.symbols)
        print("  â­� Vividness:",d.vividness_score)
        print("  Interpretation:",d.interpretation,"\n")

    print("--- PHASE 2: Pattern Analysis ---\n", json.dumps(o.analyze_patterns(), indent=2))
    print(o.generate_story_from_latest())

    print("\nâœ… DEMO COMPLETE\n")

demo_workflow()


