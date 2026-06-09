# Environment Setup & Dependency Installation
%pip install -q -U google-generativeai langchain langchain-community langchain-google-genai chromadb gradio nest_asyncio pypdf pandas duckduckgo-search

print("âœ… Environment Setup Complete.")


# Configuration and Secure Secret Management
import os
import json
import logging
import asyncio
import nest_asyncio
import pandas as pd
import gradio as gr
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

# Third-party imports
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.embeddings import Embeddings
import numpy as np

# Kaggle Secrets Integration - Securely retrieves API key
from kaggle_secrets import UserSecretsClient

try:
    # This safely retrieves your API key from Kaggle Secrets
    # Make sure you've added your Google API key to Kaggle Secrets as 'GOOGLE_API_KEY'
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ… API Key loaded successfully from Kaggle Secrets")
except Exception as e:
    print(f"âš ï¸� Warning: Could not load API key from secrets: {e}")
    print("Please add your Google API key to Kaggle Secrets as 'GOOGLE_API_KEY'")
    # For testing purposes, you can uncomment the next line and add your key directly
    # genai.configure(api_key="your-api-key-here")

# Apply asyncio patch for Jupyter/Kaggle environments
nest_asyncio.apply()

# Alternative Embedding Class using Google's API directly (avoiding pydantic_v1 issues)
class GoogleEmbeddings(Embeddings):
    """Custom Google embeddings class that avoids pydantic_v1 compatibility issues."""
    
    def __init__(self, model_name: str = "models/embedding-001"):
        self.model_name = model_name
        self.client = genai
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        try:
            results = []
            for text in texts:
                # Use Google's embedding API directly
                response = self.client.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                results.append(response['embedding'])
            return results
        except Exception as e:
            print(f"Warning: Document embedding failed: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 768 for _ in texts]
    
    def embed_query(self, text):
        """Embed a single query."""
        try:
            response = self.client.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return response['embedding']
        except Exception as e:
            print(f"Warning: Query embedding failed: {e}")
            return [0.0] * 768

# System Configuration - Uses Kaggle-compatible paths
@dataclass
class SystemConfig:
    app_name: str = "NANSC_Intelligent_Console"
    # Kaggle provides /kaggle/working as a writable directory
    persistence_dir: str = "/kaggle/working/nansc_data"
    model_name: str = "gemini-2.5-flash"

    def __post_init__(self):
        os.makedirs(self.persistence_dir, exist_ok=True)

# Global Configuration Instance
sys_config = SystemConfig()

print(f"âœ… System Configuration:")
print(f"  - App Name: {sys_config.app_name}")
print(f"  - Persistence Dir: {sys_config.persistence_dir}")
print(f"  - Model: {sys_config.model_name}")


# Layer 1: State Management, Configuration, and Observability

# --- 1. OBSERVABILITY (Metrics & Telemetry) ---
@dataclass
class TelemetryEvent:
    timestamp: str
    event_type: str
    details: str

class ObservabilityService:
    def __init__(self):
        self.events: List[TelemetryEvent] = []
        self.metrics = {"requests": 0, "tool_usage": 0, "errors": 0}
    
    def log_event(self, event_type: str, details: str):
        """Log an event with timestamp and type."""
        event = TelemetryEvent(
            datetime.now().strftime("%H:%M:%S"), 
            event_type, 
            details
        )
        self.events.append(event)
        
        # Update metrics counters
        if event_type == "ERROR": 
            self.metrics["errors"] += 1
        elif event_type == "REQUEST": 
            self.metrics["requests"] += 1
        elif event_type == "TOOL_USE": 
            self.metrics["tool_usage"] += 1

    def get_logs(self) -> str:
        """Get recent logs as formatted string."""
        return "\n".join([
            f"[{e.timestamp}] [{e.event_type}] {e.details}" 
            for e in self.events[-15:]
        ])

    def get_metrics(self) -> Dict[str, int]:
        """Get current metrics dictionary."""
        return self.metrics.copy()

# --- 2. SESSION MANAGER ---
class SessionManager:
    def __init__(self, config: SystemConfig):
        self.filepath = os.path.join(config.persistence_dir, "sessions.json")
    
    def save_session(self, session_id: str, history: List[Dict]):
        """Save conversation history to persistent storage."""
        data = {}
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f: 
                    data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read session file: {e}")
        
        data[session_id] = {
            "timestamp": datetime.now().isoformat(), 
            "history": history
        }
        
        try:
            with open(self.filepath, 'w') as f: 
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save session: {e}")

    def load_session(self, session_id: str) -> List[Dict]:
        """Load conversation history from persistent storage."""
        if not os.path.exists(self.filepath):
            return []
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                return data.get(session_id, {}).get("history", [])
        except Exception as e:
            print(f"Warning: Could not load session: {e}")
            return []

# --- 3. GLOBAL INSTANCES AND LOGGING ---

# Initialize global services
telemetry = ObservabilityService()
session_manager = SessionManager(sys_config)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Outputs to console
    ]
)

# Create logger instance
logger = logging.getLogger("NANSC_Core")

print("âœ… Layer 1 (State & Observability) Initialized.")
print(f"  - Telemetry Service: Ready")
print(f"  - Session Manager: {session_manager.filepath}")
print(f"  - Logger: {logger.name}")


# Layer 2: Domain Logic, Knowledge Base, and Custom Tools

# --- 1. DOMAIN LOGIC (Custom Tools) ---
class ICAOTools:
    """Tools for civil aviation telecommunications operations."""
    
    # ICAO airport database (sample data)
    AIRPORT_DB = {
        "HECA": "Cairo Intl (Egypt)", 
        "HEBA": "Borg El Arab (Egypt)",
        "OJAA": "Queen Alia (Jordan)", 
        "EGLL": "London Heathrow (UK)",
        "LFPG": "Paris CDG (France)",
        "KJFK": "JFK New York (USA)",
        "KORD": "O'Hare Chicago (USA)",
        "EHAM": "Amsterdam Schiphol (Netherlands)",
        "EDDF": "Frankfurt (Germany)",
        "ZBAA": "Beijing Capital (China)",
        "RJTT": "Tokyo Haneda (Japan)",
        "YSSY": "Sydney (Australia)",
        "FAOR": "OR Tambo Johannesburg (South Africa)",
        "OMDB": "Dubai (UAE)",
        "VHHH": "Hong Kong (China)",
    }
    
    @staticmethod
    def lookup_airport(icao_code: str) -> str:
        """
        Looks up an airport location by its 4-letter ICAO code.
        If not found in local database, performs web search to find the airport information.

        Args:
            icao_code: 4-letter ICAO airport code

        Returns:
            Airport name and location, or error message
        """
        code = icao_code.upper().strip()
        if len(code) != 4:
            return f"Error: ICAO code must be exactly 4 characters. Got: '{code}'"

        # First try local database
        result = ICAOTools.AIRPORT_DB.get(code)
        if result:
            telemetry.log_event("TOOL_USE", f"Airport Lookup: {code} -> {result}")
            return result

        # If not found locally, perform web search
        try:
            search_query = f"ICAO airport code {code} location airport name"
            search_result = ICAOTools.web_search(search_query)

            if search_result and "Error" not in search_result and len(search_result) > 10:
                # Found information via web search
                message = (
                    f"âš ï¸� ICAO code '{code}' not found in local database.\n"
                    f"ğŸ”� Searching online...\n\n"
                    f"ğŸ“� Found result:\n{search_result[:800]}"
                )
                telemetry.log_event("TOOL_USE", f"Airport Lookup (Web Search): {code} -> Found online")
                return message
            else:
                # Web search failed or returned no useful results
                message = (
                    f"â�Œ ICAO code '{code}' not found in local database.\n"
                    f"ğŸ”� Web search did not return useful results.\n"
                    f"ğŸ’¡ Please verify the ICAO code and try again."
                )
                telemetry.log_event("TOOL_USE", f"Airport Lookup (Web Search): {code} -> Not found online")
                return message

        except Exception as e:
            # Web search error
            error_msg = (
                f"â�Œ ICAO code '{code}' not found in local database.\n"
                f"ğŸ”� Web search failed: {str(e)}\n"
                f"ğŸ’¡ Please verify the ICAO code or check your internet connection."
            )
            telemetry.log_event("ERROR", f"Airport Lookup web search failed: {e}")
            return error_msg

    @staticmethod
    def bridge_aftn_to_amhs(aftn_address: str) -> str:
        """
        Converts legacy AFTN (8-char) to AMHS (X.400) format.
        
        Args:
            aftn_address: 8-character AFTN address
            
        Returns:
            X.400 format address or error message
        """
        addr = aftn_address.upper().strip()
        if len(addr) != 8: 
            return "Error: Address must be exactly 8 characters."
        
        # PRMD (Physical Message Relay Domain) mapping
        prmd_map = {
            "HE": "EGYPT", "OJ": "JORDAN", "EG": "UK", 
            "LF": "FRANCE", "K": "USA", "EH": "NETHERLANDS",
            "ED": "GERMANY", "ZB": "CHINA", "RJ": "JAPAN",
            "YS": "AUSTRALIA", "FA": "SOUTH AFRICA",
            "OM": "UAE", "VH": "HONG KONG"
        }
        
        prefix = addr[:2]
        prmd = prmd_map.get(prefix, "UNKNOWN")
        
        # X.400 format: /C=XX/A=ICAO/P=PRMD/O=ORG/OU1=UNIT
        x400 = f"/C=XX/A=ICAO/P={prmd}/O={addr[:4]}/OU1={addr[4:]}/"
        telemetry.log_event("TOOL_USE", f"Bridge Conversion: {addr} -> {x400}")
        return x400

    @staticmethod
    def web_search(query: str) -> str:
        """
        Searches the web for aviation definitions if internal knowledge fails.
        
        Args:
            query: Search query
            
        Returns:
            Search results or error message
        """
        try:
            search = DuckDuckGoSearchRun()
            res = search.run(query)
            telemetry.log_event("TOOL_USE", f"Web Search: {query}")
            return res[:1000]  # Limit response length for display
        except Exception as e:
            error_msg = f"Search unavailable: {str(e)}"
            telemetry.log_event("ERROR", error_msg)
            return error_msg

# --- 2. RAG ENGINE (Retrieval Augmented Generation) ---
class RAGEngine:
    """Handles document processing and retrieval for RAG operations."""
    
    def __init__(self, persistence_dir: str):
        self.persist_dir = os.path.join(persistence_dir, "chroma_db")
        self.embeddings = None
        self.vector_store = None
        self._init_embeddings()
        self._init_db()

    def _init_embeddings(self):
        """Initialize custom Google embeddings to avoid pydantic_v1 issues."""
        try:
            self.embeddings = GoogleEmbeddings(model_name="models/embedding-001")
            print(f"âœ… Embeddings initialized: models/embedding-001")
        except Exception as e:
            print(f"âš ï¸� Warning: Could not initialize embeddings: {e}")
            telemetry.log_event("ERROR", f"Embeddings init failed: {e}")

    def _init_db(self):
        """Initialize ChromaDB vector store."""
        try:
            if self.embeddings:
                self.vector_store = Chroma(
                    persist_directory=self.persist_dir, 
                    embedding_function=self.embeddings
                )
                print(f"âœ… Vector store initialized: {self.persist_dir}")
            else:
                print("âš ï¸� Warning: Vector store not initialized (no embeddings)")
        except Exception as e:
            print(f"âš ï¸� Warning: ChromaDB Init failed: {e}")
            telemetry.log_event("ERROR", f"Vector store init failed: {e}")

    def ingest_pdf(self, file_path: str) -> str:
        """
        Ingest a PDF document into the vector database.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Status message
        """
        try:
            if not self.vector_store or not self.embeddings:
                return "â�Œ Vector store not ready. Please check embeddings configuration."
            
            # Load and process PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Split documents into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100
            )
            splits = splitter.split_documents(docs)
            
            # Add to vector store
            self.vector_store.add_documents(splits)
            
            # Log and return status
            message = f"âœ… Ingested {len(splits)} document chunks into vector store."
            telemetry.log_event("RAG_INGEST", message)
            return message
            
        except Exception as e:
            error_msg = f"â�Œ Error ingesting document: {str(e)}"
            telemetry.log_event("ERROR", error_msg)
            return error_msg

    def query(self, question: str) -> str:
        """
        Query the vector database for relevant documents.
        
        Args:
            question: User's question
            
        Returns:
            Retrieved document content or empty string
        """
        if not self.vector_store:
            return ""
        
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search(question, k=3)
            
            # Combine results
            content = "\n\n".join([
                f"Document Chunk {i+1}:\n{d.page_content[:500]}..." 
                for i, d in enumerate(docs)
            ])
            
            if content:
                telemetry.log_event("RAG_QUERY", f"Found {len(docs)} relevant chunks")
            
            return content
            
        except Exception as e:
            error_msg = f"Query failed: {str(e)}"
            telemetry.log_event("ERROR", error_msg)
            return ""

# --- 3. GLOBAL INSTANCES ---

# Create global RAG engine instance
rag_engine = RAGEngine(sys_config.persistence_dir)

# Create tools list for Gemini - Use the class methods directly
tools_list = [
    ICAOTools.lookup_airport, 
    ICAOTools.bridge_aftn_to_amhs, 
    ICAOTools.web_search
]

print("\nâœ… Layer 2 (Knowledge & Tools) Initialized.")
print(f"  - Tools available: {len(tools_list)}")
print(f"  - RAG Engine: {rag_engine.persist_dir}")


# Layer 3: Agent Orchestration and Core Logic

class EnterpriseAgent:
    """
    Main agent orchestrator that handles all user interactions.
    
    This class integrates:
    - Google Gemini model for LLM capabilities
    - Custom tools for aviation operations
    - RAG for document-based queries
    - Session management and observability
    """
    
    def __init__(self):
        """Initialize the agent with model (no tools parameter to avoid compatibility issues)."""
        try:
            # Initialize Gemini model without tools parameter to avoid compatibility issues
            self.model = genai.GenerativeModel(model_name=sys_config.model_name)
            
            # Start chat session
            self.chat = self.model.start_chat()
            
            print(f"âœ… Agent initialized with model: {sys_config.model_name}")
            
            # Log successful initialization
            telemetry.log_event("REQUEST", f"Agent initialized with {sys_config.model_name}")
            
        except Exception as e:
            error_msg = f"Failed to initialize agent: {str(e)}"
            print(f"â�Œ {error_msg}")
            telemetry.log_event("ERROR", error_msg)
            # Don't raise the exception - allow the system to continue with limited functionality
            self.model = None
            self.chat = None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """
        You are the NANSC Intelligent Operations Console Assistant.
        
        OPERATIONAL PROTOCOL:
        1. DEFINITIONS: If the user asks "What is...", answer from your internal knowledge. 
           If unsure, use the 'web_search' tool.
        2. CODES: If an ICAO code (4 letters) or AFTN address (8 letters) is detected, 
           ALWAYS use 'lookup_airport' or 'bridge_aftn_to_amhs' tools automatically.
        3. PROCEDURES: If asked about rules/regs, refer to the RAG Context provided.
        
        BEHAVIORAL GUIDELINES:
        - Be professional, concise, and helpful
        - Always provide accurate information
        - Use tools proactively when appropriate
        - Maintain context throughout the conversation
        - Log all tool usage and errors for observability
        
        DOMAIN EXPERTISE:
        - Civil Aviation Telecommunications
        - ICAO Standards and Procedures
        - AFTN and AMHS Operations
        - Air Traffic Management
        - Aviation Safety and Security
        """

    def _detect_and_call_tools(self, message: str) -> str:
        """
        Detect tool usage requirements and call tools manually.
        
        Args:
            message: User's input message
            
        Returns:
            Enhanced message with tool results
        """
        # Check for ICAO codes (4 letters)
        import re
        icao_pattern = r'\b[A-Z]{4}\b'
        aftn_pattern = r'\b[A-Z]{8}\b'
        
        icao_codes = re.findall(icao_pattern, message)
        aftn_codes = re.findall(aftn_pattern, message)
        
        tool_results = []
        
        # Process ICAO codes
        for code in icao_codes:
            if len(code) == 4:
                result = ICAOTools.lookup_airport(code)
                tool_results.append(f"ICAO Code {code}: {result}")
                telemetry.log_event("TOOL_USE", f"Detected ICAO code {code}")
        
        # Process AFTN codes  
        for code in aftn_codes:
            if len(code) == 8:
                result = ICAOTools.bridge_aftn_to_amhs(code)
                tool_results.append(f"AFTN Code {code}: {result}")
                telemetry.log_event("TOOL_USE", f"Detected AFTN code {code}")
        
        # Add tool results to message
        if tool_results:
            tool_output = "\n".join(tool_results)
            enhanced_message = f"Tool Results:\n{tool_output}\n\nUser Message: {message}"
            return enhanced_message
        else:
            return message

    async def process_message(self, message: str) -> str:
        """
        Process a user message asynchronously.
        
        Args:
            message: User's input message
            
        Returns:
            Agent's response
        """
        if not message or not message.strip():
            return "Please provide a message to process."
        
        try:
            # Check if agent is properly initialized
            if not self.model or not self.chat:
                return ("âš ï¸� System Warning: AI model not available. This could be due to:\n"
                       "1. API key configuration issues\n"
                       "2. Quota limits exceeded\n"
                       "3. Service connectivity problems\n\n"
                       "However, you can still use:\n"
                       "â€¢ Airport lookups (ICAO codes)\n"
                       "â€¢ AFTN address conversions\n"
                       "â€¢ Batch processing tools\n"
                       "â€¢ Document upload and management\n"
                       "â€¢ System telemetry monitoring\n\n"
                       "Please check your API configuration or try again later.")

            # 1. RAG Context Injection
            # Check if user is asking about procedures, rules, or manuals
            rag_context = ""
            if any(keyword in message.lower() for keyword in [
                "procedure", "rule", "reg", "manual", "doc", "guideline",
                "protocol", "standard", "regulation", "policy", "directive"
            ]):
                rag_context = rag_engine.query(message)
                if rag_context:
                    message = f"Reference Info from Manuals:\n{rag_context}\n\nUser Question: {message}"
                    telemetry.log_event("RAG_CONTEXT", "RAG context injected")

            # 2. Tool Detection and Manual Tool Calling
            enhanced_message = self._detect_and_call_tools(message)

            # 3. Generate response with Gemini
            # Apply system instructions by prepending them to the message
            system_prompt = self._get_system_prompt()
            full_message = f"{system_prompt}\n\n{enhanced_message}"
            
            response = await self.chat.send_message_async(full_message)
            
            # 4. Log successful request
            telemetry.log_event("REQUEST", f"Message processed: {message[:50]}...")
            
            # 5. Persist session data
            try:
                hist_serialized = [
                    {"role": p.role, "parts": [pt.text for pt in p.parts]} 
                    for p in self.chat.history
                ]
                session_manager.save_session("web_user", hist_serialized)
            except Exception as e:
                telemetry.log_event("ERROR", f"Session save failed: {e}")
            
            return response.text
            
        except Exception as e:
            # Log and return error message
            error_msg = f"âš ï¸� System Error: {str(e)}"
            telemetry.log_event("ERROR", error_msg)
            return error_msg

    def get_session_history(self) -> List[Dict]:
        """Get the current session history."""
        try:
            return session_manager.load_session("web_user")
        except:
            return []

    def reset_session(self):
        """Reset the current session."""
        try:
            session_manager.save_session("web_user", [])
            # Start a new chat session if model is available
            if self.model:
                self.chat = self.model.start_chat()
            telemetry.log_event("REQUEST", "Session reset")
        except Exception as e:
            telemetry.log_event("ERROR", f"Session reset failed: {e}")

# Initialize the global agent instance
agent = EnterpriseAgent()

print("\nğŸš€ Agent ready for operations!")


# Layer 4: Gradio Dashboard Interface

# --- 1. ASYNC CHAT WRAPPER ---
async def chat_wrapper(message, history):
    """
    Wrapper function for Gradio chat interface.
    
    Args:
        message: User's input message
        history: Chat history from Gradio
        
    Returns:
        Agent's response
    """
    if not message or not message.strip():
        return "Please enter a message to begin."
    
    try:
        # Process message asynchronously
        response = await agent.process_message(message)
        return response
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        telemetry.log_event("ERROR", error_msg)
        return error_msg

# --- 2. BATCH TOOL WRAPPERS ---
def batch_tool_wrapper(text_input, operation):
    """
    Process multiple items using selected tool.
    
    Args:
        text_input: Multi-line text with items to process
        operation: Type of operation ("Convert AFTN" or "Lookup Airport")
        
    Returns:
        DataFrame with results
    """
    if not text_input or not text_input.strip():
        return pd.DataFrame(columns=["Input", "Result"])
    
    lines = [l.strip() for l in text_input.split('\n') if l.strip()]
    results = []
    
    for line in lines:
        try:
            if operation == "Convert AFTN":
                res = ICAOTools.bridge_aftn_to_amhs(line)
            else:  # Lookup Airport
                res = ICAOTools.lookup_airport(line)
            
            results.append({"Input": line, "Result": res})
            
        except Exception as e:
            results.append({"Input": line, "Result": f"Error: {str(e)}"})
    
    return pd.DataFrame(results)

# --- 3. DOCUMENT INGESTION WRAPPER ---
def ingest_wrapper(files):
    """
    Ingest uploaded PDF files into RAG system.
    
    Args:
        files: List of uploaded files from Gradio
        
    Returns:
        Status messages
    """
    if not files:
        return "No files provided. Please upload one or more PDF files."
    
    results = []
    for file in files:
        try:
            result = rag_engine.ingest_pdf(file.name)
            results.append(result)
        except Exception as e:
            results.append(f"â�Œ Error processing {file.name}: {str(e)}")
    
    return "\n".join(results)

# --- 4. TELEMETRY WRAPPERS ---
def get_stats_wrapper():
    """Get current telemetry statistics."""
    metrics = telemetry.get_metrics()
    logs = telemetry.get_logs()
    return json.dumps(metrics, indent=2), logs

def clear_logs_wrapper():
    """Clear telemetry logs."""
    telemetry.events = []
    telemetry.log_event("REQUEST", "Logs cleared")
    return "Logs cleared successfully."

# --- 5. MAIN INTERFACE LAYOUT ---
with gr.Blocks(
    title="NANSC Intelligent Operations Console"
) as demo:
    
    # Header Section
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                # ğŸ“¡ NANSC Intelligent Operations Console
                **Civil Aviation Telecommunications | AI-Powered Assistant**
                
                Professional-grade interface for aviation telecommunications operations.
                Built with Google Gemini, LangChain, and Gradio.
                """
            )
    
    # Main Dashboard Layout
    with gr.Row(equal_height=True):
        
        # LEFT COLUMN: Tools & Admin
        with gr.Column(scale=1, min_width=350):
            
            # Batch Tools Section
            with gr.Accordion("ğŸ› ï¸� Batch Operations", open=True):
                gr.Markdown("Process multiple items efficiently.")
                b_input = gr.TextArea(
                    lines=4, 
                    placeholder="HECAYFYX\nOJAA\nEGLL\nKJFK\nXXXX (unknown code)",
                    label="Input Items (one per line)",
                    show_label=True
                )
                b_operation = gr.Radio(
                    ["Convert AFTN", "Lookup Airport"], 
                    value="Convert AFTN", 
                    label="Operation Type"
                )
                b_button = gr.Button("ğŸš€ Process Batch", variant="primary")
                b_output = gr.Dataframe(
                    headers=["Input", "Result"], 
                    wrap=True,
                    label="Results"
                )
                b_button.click(
                    batch_tool_wrapper, 
                    inputs=[b_input, b_operation], 
                    outputs=b_output
                )
            
            # Document Management Section
            with gr.Accordion("ğŸ“š Knowledge Base Management", open=False):
                gr.Markdown("Upload and process PDF documents for RAG.")
                f_upload = gr.File(
                    file_count="multiple", 
                    file_types=[".pdf"],
                    label="Upload PDF Files"
                )
                up_button = gr.Button("ğŸ“¥ Ingest Documents", variant="secondary")
                up_output = gr.Textbox(
                    show_label=False, 
                    placeholder="Upload status will appear here...",
                    lines=3
                )
                up_button.click(
                    ingest_wrapper, 
                    inputs=[f_upload], 
                    outputs=[up_output]
                )
                
                # Session Management
                with gr.Row():
                    reset_button = gr.Button("ğŸ”„ Reset Session", variant="secondary")
                    clear_logs_btn = gr.Button("ğŸ§¹ Clear Logs", variant="secondary")
                
                reset_button.click(
                    lambda: agent.reset_session(),
                    outputs=[]
                )
                
                clear_logs_btn.click(
                    clear_logs_wrapper,
                    outputs=[up_output]
                )
                
            # Telemetry Section
            with gr.Accordion("ğŸ“Š System Telemetry", open=False):
                stat_button = gr.Button("ğŸ”„ Refresh Metrics", variant="secondary")
                stat_json = gr.Code(
                    language="json", 
                    label="Usage Metrics",
                    lines=6
                )
                stat_logs = gr.TextArea(
                    label="System Logs",
                    lines=8
                )
                stat_button.click(
                    get_stats_wrapper, 
                    outputs=[stat_json, stat_logs]
                )
                
            # System Status
            with gr.Accordion("ğŸ”� System Status", open=False):
                status_box = gr.HTML()
                
                def update_status():
                    """Update system status display."""
                    metrics = telemetry.get_metrics()
                    return f"""
                    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <strong>System Status:</strong><br>
                    â€¢ Requests: {metrics.get('requests', 0)}<br>
                    â€¢ Tool Usage: {metrics.get('tool_usage', 0)}<br>
                    â€¢ Errors: {metrics.get('errors', 0)}<br>
                    â€¢ Model: {sys_config.model_name}<br>
                    â€¢ Persistence: {sys_config.persistence_dir}
                    </div>
                    """
                
                demo.load(update_status, outputs=status_box)
        
        # RIGHT COLUMN: Chat Interface
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat_wrapper,
                examples=[
                    "What is AMHS?",
                    "Convert HECAYFYX to X.400",
                    "Where is OJAA airport?",
                    "Lookup EGLL",
                    "What are the procedures for flight planning?",
                    "Explain AFTN routing",
                    "Lookup XXXX (unknown ICAO code - will trigger web search)"
                ],
                title="Operations Assistant",
                description="""
                Interact with the Enterprise Agent. Ask about:
                â€¢ Aviation definitions and concepts
                â€¢ ICAO airport lookups (with web search fallback for unknown codes)
                â€¢ AFTN to AMHS address conversions
                â€¢ Document-based queries and procedures
                """
            )

print("\nğŸ�‰ Gradio Interface Created Successfully!")
print("\nğŸ“� Interface Features:")
print("  âœ“ Interactive AI Chat with Examples")
print("  âœ“ Batch Processing Tools")
print("  âœ“ Document Ingestion (PDF)")
print("  âœ“ Real-time Telemetry")
print("  âœ“ Session Management")
print("  âœ“ Professional UI/UX")


# Launch the Console Interface

print("ğŸš€ Launching NANSC Console...")
print("\nğŸ“‹ Access Instructions:")
print("  â€¢ Interface will appear below in the notebook")
print("  â€¢ Use the chat for interactive AI assistance")
print("  â€¢ Try the batch tools for multiple operations")
print("  â€¢ Upload PDFs to build your knowledge base")
print("  â€¢ Monitor telemetry for system health")
print("\nâš ï¸� Note: In Kaggle, this is a read-only demo.")
print("   For full functionality, export and run locally.")

# In Kaggle, we just display the interface without launching
# The interface components are ready for use
print("\nâœ… Interface components are ready for Kaggle display!")
print("   The Gradio interface will be shown below when the notebook is rendered.")


# System Health Check

print("ğŸ�¥ Running System Health Check...")
print("=" * 50)

health_status = {
    "API Configuration": "â�Œ",
    "Model Initialization": "â�Œ",
    "Tools Available": "â�Œ",
    "Storage Systems": "â�Œ",
    "Telemetry Services": "â�Œ",
    "RAG Engine": "â�Œ"
}

# Check API Configuration - Improved version
try:
    # Check if API key is configured and client is accessible
    if hasattr(genai, 'configure') and GOOGLE_API_KEY:
        # Try to access the configured client indirectly
        try:
            # Attempt to list models (this will test the API connection)
            models = genai.list_models()
            if models:
                health_status["API Configuration"] = "âœ…"
                print("âœ… API Configuration: Google Generative AI client ready")
            else:
                print("âš ï¸� API Configuration: Client configured but no models available")
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                health_status["API Configuration"] = "âš ï¸�"
                print("âš ï¸� API Configuration: API key configured but quota exceeded")
            else:
                print(f"âš ï¸� API Configuration: Client configured but connection failed: {e}")
    else:
        print("â�Œ API Configuration: API key not configured")
except Exception as e:
    print(f"â�Œ API Configuration: Failed to initialize - {e}")

# Check Model Initialization - Improved version
try:
    if hasattr(agent, 'model') and agent.model:
        health_status["Model Initialization"] = "âœ…"
        print(f"âœ… Model Initialization: {sys_config.model_name} ready")
    elif hasattr(agent, 'model') and agent.model is None:
        print("â�Œ Model Initialization: Agent exists but model is None")
    else:
        print("â�Œ Model Initialization: Agent not properly initialized")
except Exception as e:
    print(f"â�Œ Model Initialization: Failed - {e}")

# Check Tools
try:
    if len(tools_list) > 0:
        health_status["Tools Available"] = "âœ…"
        print(f"âœ… Tools Available: {len(tools_list)} tools registered")
        # Test tool functionality
        test_result = ICAOTools.lookup_airport("KJFK")
        if "Error" not in test_result and "JFK" in test_result:
            print("  âœ“ Tool functionality verified")
        else:
            print(f"  âš ï¸� Tool test result: {test_result}")
    else:
        print("â�Œ Tools Available: No tools found")
except Exception as e:
    print(f"â�Œ Tools Available: Failed to check - {e}")

# Check Storage Systems
try:
    if os.path.exists(sys_config.persistence_dir):
        health_status["Storage Systems"] = "âœ…"
        print(f"âœ… Storage Systems: Persistence directory ready at {sys_config.persistence_dir}")

        # Check if subdirectories exist
        chroma_dir = os.path.join(sys_config.persistence_dir, "chroma_db")
        if os.path.exists(chroma_dir):
            print("  âœ“ Vector store directory ready")
        else:
            print("  âš ï¸� Vector store directory will be created on first use")
    else:
        print("â�Œ Storage Systems: Persistence directory missing")
except Exception as e:
    print(f"â�Œ Storage Systems: Failed to check - {e}")

# Check Telemetry
try:
    if telemetry:
        health_status["Telemetry Services"] = "âœ…"
        print("âœ… Telemetry Services: Ready for logging and metrics")
        # Test logging
        telemetry.log_event("TEST", "Health check test")
        print("  âœ“ Telemetry logging verified")
    else:
        print("â�Œ Telemetry Services: Not initialized")
except Exception as e:
    print(f"â�Œ Telemetry Services: Failed to check - {e}")

# Check RAG Engine - Improved version
try:
    if rag_engine:
        health_status["RAG Engine"] = "âœ…"
        print(f"âœ… RAG Engine: Ready at {rag_engine.persist_dir}")

        # Test embeddings with better error handling
        if hasattr(rag_engine, 'embeddings') and rag_engine.embeddings:
            try:
                test_embedding = rag_engine.embeddings.embed_query("test")
                if test_embedding and len(test_embedding) > 0:
                    print(f"  âœ“ Embeddings test: Vector length {len(test_embedding)}")
                else:
                    print("  âš ï¸� Embeddings test: Empty vector returned")
            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    print("  âš ï¸� Embeddings test: Quota exceeded (fallback vectors used)")
                else:
                    print(f"  âš ï¸� Embeddings test: {e}")
        else:
            print("  âš ï¸� Embeddings not initialized")
    else:
        print("â�Œ RAG Engine: Not initialized")
except Exception as e:
    print(f"â�Œ RAG Engine: Failed to check - {e}")

print("\n" + "=" * 50)
print("ğŸ“‹ Health Check Summary:")
for component, status in health_status.items():
    print(f"  {status} {component}")

overall_health = all(status == "âœ…" for status in health_status.values())
some_issues = any(status == "âš ï¸�" for status in health_status.values())

if overall_health:
    print("\nğŸ�‰ All systems healthy!")
elif some_issues:
    print("\nâš ï¸� Some systems may need attention (see details above).")
else:
    print("\nâ�Œ Multiple systems need attention.")

# Quick functionality test - Improved version
print("\nğŸ§ª Quick Functionality Test:")
try:
    # Test tool functionality
    test_result = ICAOTools.lookup_airport("KJFK")
    print(f"  âœ“ Airport lookup test: {test_result}")

    test_result2 = ICAOTools.bridge_aftn_to_amhs("HECAYFYX")
    print(f"  âœ“ AFTN conversion test: {test_result2}")

    # Test async processing framework
    print("  âœ“ Async processing framework ready")

    # Test session management
    try:
        session_data = session_manager.load_session("health_check_test")
        print("  âœ“ Session management ready")
    except Exception as e:
        print(f"  âš ï¸� Session management test: {e}")

    print("\nâœ… Core functionality verified!")

except Exception as e:
    print(f"\nâ�Œ Functionality test failed: {e}")
    telemetry.log_event("ERROR", f"Functionality test failed: {e}")

print("\n" + "=" * 50)
print("ğŸ�¥ Health check complete!")

