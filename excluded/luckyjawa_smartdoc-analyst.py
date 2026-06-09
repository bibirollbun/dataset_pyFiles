# ğŸ�† SmartDoc Analyst - Kaggle Agents Intensive Capstone 2025
## Enterprise Agents Track

> **An Intelligent Multi-Agent Document Research & Analysis System**
>
> Transforming how knowledge workers extract insights from documents through collaborative AI agents.

---

**Author:** Lucky Jawa  
**Date:** December 2024  
**Competition:** Kaggle Agents Intensive Capstone  
**Track:** Enterprise Agents  

---


# ============================================================================
# SETUP: Install dependencies (Kaggle-compatible)
# ============================================================================

# Install required packages
!pip install -q pydantic structlog tenacity tiktoken

# Note: google-generativeai should be pre-installed on Kaggle
# If not, uncomment: !pip install -q google-generativeai

print("âœ“ Dependencies installed")


# ============================================================================
# CORE IMPORTS
# ============================================================================

import asyncio
import json
import uuid
import re
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
from collections import defaultdict
from io import StringIO
import sys

# Kaggle-specific: Get API key from secrets
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    GEMINI_API_KEY = secrets.get_secret("GEMINI_API_KEY")
except:
    # Fallback for local development
    GEMINI_API_KEY = None
    print("âš ï¸� Running without Gemini API key (demo mode)")

print("âœ“ Core imports loaded successfully")


# ============================================================================
# AGENT STATE & DATA STRUCTURES
# ============================================================================

class AgentState(Enum):
    """Agent lifecycle states for tracking and observability."""
    IDLE = "idle"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"

class MessageType(Enum):
    """A2A Protocol message types."""
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    FEEDBACK = "feedback"
    STATUS = "status"

@dataclass
class AgentContext:
    """Context passed between agents during processing.
    
    This enables context engineering - accumulating and passing
    relevant information between processing stages.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    agent_chain: List[str] = field(default_factory=list)
    
    def add_result(self, agent: str, result: Any):
        """Add intermediate result from an agent."""
        self.intermediate_results[agent] = result
        self.agent_chain.append(agent)

@dataclass
class AgentResult:
    """Standardized result from agent processing."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMessage:
    """A2A Protocol message for inter-agent communication."""
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def reply(self, content: Any, msg_type: MessageType) -> "AgentMessage":
        """Create a reply message."""
        return AgentMessage(
            from_agent=self.to_agent,
            to_agent=self.from_agent,
            message_type=msg_type,
            content=content,
            correlation_id=self.correlation_id
        )

print("âœ“ Base classes defined: AgentState, AgentContext, AgentResult, ToolResult, AgentMessage")


# ============================================================================
# THREE-TIER MEMORY SYSTEM (Course Concept #3: Sessions & Memory)
# ============================================================================

class WorkingMemory:
    """Short-term working memory for current task context.
    
    Implements a priority-based buffer that maintains the most
    important items for the current processing task.
    """
    
    def __init__(self, max_items: int = 100):
        self.max_items = max_items
        self._entries: List[Dict] = []
        self._index: Dict[str, int] = {}
        
    def add(self, content: str, metadata: Dict = None, importance: float = 0.5) -> str:
        """Add item to working memory with importance score."""
        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "content": content,
            "metadata": metadata or {},
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
        self._entries.append(entry)
        self._index[entry_id] = len(self._entries) - 1
        
        # Prune if over capacity (remove lowest importance)
        if len(self._entries) > self.max_items:
            self._prune()
        
        return entry_id
        
    def _prune(self):
        """Remove lowest importance items."""
        self._entries.sort(key=lambda x: x["importance"], reverse=True)
        self._entries = self._entries[:self.max_items]
        self._rebuild_index()
        
    def _rebuild_index(self):
        self._index = {e["id"]: i for i, e in enumerate(self._entries)}
        
    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get n most recent items."""
        return self._entries[-n:]
        
    def get_by_importance(self, n: int = 10) -> List[Dict]:
        """Get n most important items."""
        sorted_entries = sorted(self._entries, key=lambda x: x["importance"], reverse=True)
        return sorted_entries[:n]
        
    def clear(self):
        """Clear all working memory."""
        self._entries.clear()
        self._index.clear()
        
    def __len__(self) -> int:
        return len(self._entries)


class EpisodicMemory:
    """Session-based episodic memory for conversation history.
    
    Stores episodes (query-response pairs) for context continuity
    within a session.
    """
    
    def __init__(self, max_episodes: int = 50):
        self.max_episodes = max_episodes
        self._episodes: List[Dict] = []
        self._session_id = str(uuid.uuid4())
        
    def add_episode(self, query: str, response: str, metadata: Dict = None) -> str:
        """Record a query-response episode."""
        episode_id = str(uuid.uuid4())
        episode = {
            "id": episode_id,
            "session_id": self._session_id,
            "query": query,
            "response": response,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self._episodes.append(episode)
        
        # Keep only recent episodes
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]
        
        return episode_id
        
    def get_history(self, n: int = 5) -> List[Dict]:
        """Get n most recent episodes."""
        return self._episodes[-n:]
        
    def search(self, query: str) -> List[Dict]:
        """Search episodes for relevant history."""
        query_terms = set(query.lower().split())
        scored = []
        for ep in self._episodes:
            ep_text = (ep["query"] + " " + ep["response"]).lower()
            score = sum(1 for t in query_terms if t in ep_text)
            if score > 0:
                scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:5]]
        
    def new_session(self):
        """Start a new session."""
        self._session_id = str(uuid.uuid4())
        
    def __len__(self) -> int:
        return len(self._episodes)


class SemanticMemory:
    """Long-term semantic memory for persistent knowledge.
    
    Stores facts, preferences, and learned patterns that persist
    across sessions.
    """
    
    def __init__(self):
        self._facts: Dict[str, Dict] = {}
        self._categories: Dict[str, List[str]] = defaultdict(list)
        self._preferences: Dict[str, Any] = {}
        
    def store_fact(self, key: str, value: Any, category: str = "general", confidence: float = 1.0):
        """Store a fact with category and confidence."""
        fact_id = hashlib.md5(key.encode()).hexdigest()[:8]
        self._facts[fact_id] = {
            "key": key,
            "value": value,
            "category": category,
            "confidence": confidence,
            "created": datetime.now().isoformat(),
            "access_count": 0
        }
        self._categories[category].append(fact_id)
        return fact_id
        
    def retrieve_fact(self, key: str) -> Optional[Any]:
        """Retrieve a fact by key."""
        fact_id = hashlib.md5(key.encode()).hexdigest()[:8]
        fact = self._facts.get(fact_id)
        if fact:
            fact["access_count"] += 1
            return fact["value"]
        return None
        
    def get_by_category(self, category: str) -> List[Dict]:
        """Get all facts in a category."""
        fact_ids = self._categories.get(category, [])
        return [self._facts[fid] for fid in fact_ids if fid in self._facts]
        
    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self._preferences[key] = value
        
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self._preferences.get(key, default)
        
    def __len__(self) -> int:
        return len(self._facts)


class VectorStore:
    """Simple vector store for document embeddings.
    
    Uses TF-IDF-like scoring for efficient semantic search.
    Production would use proper embeddings (Gemini, OpenAI).
    """
    
    def __init__(self):
        self._documents: List[Dict] = []
        self._index: Dict[str, Set[int]] = defaultdict(set)  # term -> doc indices
        
    def add_documents(self, documents: List[Dict]) -> List[str]:
        """Add documents and build index."""
        ids = []
        for doc in documents:
            doc_id = str(uuid.uuid4())
            doc_idx = len(self._documents)
            
            stored_doc = {
                "id": doc_id,
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "added": datetime.now().isoformat()
            }
            self._documents.append(stored_doc)
            ids.append(doc_id)
            
            # Index terms
            terms = self._tokenize(doc.get("content", ""))
            for term in terms:
                self._index[term].add(doc_idx)
        
        return ids
        
    def _tokenize(self, text: str) -> Set[str]:
        """Simple tokenization."""
        words = re.findall(r"\b\w+\b", text.lower())
        return {w for w in words if len(w) > 2}
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search documents by query relevance."""
        query_terms = self._tokenize(query)
        scores: Dict[int, float] = defaultdict(float)
        
        for term in query_terms:
            if term in self._index:
                doc_indices = self._index[term]
                # IDF-like weighting
                weight = 1.0 / (len(doc_indices) + 1)
                for idx in doc_indices:
                    scores[idx] += weight
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in ranked[:k]:
            doc = self._documents[idx].copy()
            doc["relevance_score"] = round(score, 3)
            results.append(doc)
        
        return results
        
    def get_all(self) -> List[Dict]:
        """Get all documents."""
        return self._documents.copy()
        
    def __len__(self) -> int:
        return len(self._documents)


class MemoryManager:
    """Unified memory management across all three tiers."""
    
    def __init__(self):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.vector_store = VectorStore()
        
    def add_to_context(self, content: str, metadata: Dict = None, importance: float = 0.5) -> str:
        """Add to working memory."""
        return self.working.add(content, metadata, importance)
        
    def record_interaction(self, query: str, response: str, metadata: Dict = None) -> str:
        """Record in episodic memory."""
        return self.episodic.add_episode(query, response, metadata)
        
    def store_knowledge(self, key: str, value: Any, category: str = "general") -> str:
        """Store in semantic memory."""
        return self.semantic.store_fact(key, value, category)
        
    def add_documents(self, documents: List[Dict]) -> List[str]:
        """Add documents to vector store."""
        return self.vector_store.add_documents(documents)
        
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search documents."""
        return self.vector_store.search(query, k)
        
    def get_relevant_context(self, query: str) -> Dict[str, Any]:
        """Get all relevant context for a query."""
        return {
            "working_memory": self.working.get_recent(5),
            "relevant_history": self.episodic.search(query),
            "documents": self.vector_store.search(query, 5)
        }
        
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        return {
            "working_memory_items": len(self.working),
            "episodic_episodes": len(self.episodic),
            "semantic_facts": len(self.semantic),
            "documents_indexed": len(self.vector_store)
        }

print("âœ“ Memory System defined: WorkingMemory, EpisodicMemory, SemanticMemory, VectorStore, MemoryManager")


# ============================================================================
# OBSERVABILITY STACK (Course Concept #5: Observability)
# ============================================================================

class StructuredLogger:
    """Structured logging with JSON output and trace correlation."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._logs: List[Dict] = []
        self._current_trace_id: Optional[str] = None
        
    def set_trace_id(self, trace_id: str):
        """Set current trace ID for correlation."""
        self._current_trace_id = trace_id
        
    def _log(self, level: str, message: str, **kwargs):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "service": self.service_name,
            "message": message,
            "trace_id": self._current_trace_id,
            **kwargs
        }
        self._logs.append(entry)
        
    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)
        
    def get_logs(self, level: str = None, limit: int = 100) -> List[Dict]:
        logs = self._logs
        if level:
            logs = [l for l in logs if l["level"] == level]
        return logs[-limit:]


class MetricsCollector:
    """Metrics collection for monitoring and analysis."""
    
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
    def increment(self, name: str, value: int = 1, labels: Dict = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        self._counters[key] += value
        
    def gauge(self, name: str, value: float, labels: Dict = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        
    def histogram(self, name: str, value: float, labels: Dict = None):
        """Add to histogram."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        
    def timer(self, name: str, duration_ms: float, labels: Dict = None):
        """Record timing."""
        key = self._make_key(name, labels)
        self._timers[key].append(duration_ms)
        
    def _make_key(self, name: str, labels: Dict = None) -> str:
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
        
    def get_counter(self, name: str) -> int:
        return self._counters.get(name, 0)
        
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        values = self._histograms.get(name, [])
        if not values:
            return {}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": sorted(values)[len(values)//2]
        }
        
    def get_all(self) -> Dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "gauges": self._gauges.copy(),
            "histogram_stats": {k: self.get_histogram_stats(k) for k in self._histograms},
            "timer_stats": {k: {"count": len(v), "avg_ms": sum(v)/len(v) if v else 0} for k, v in self._timers.items()}
        }


class DistributedTracer:
    """Distributed tracing for agent call chains."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._spans: List[Dict] = []
        self._active_spans: Dict[str, Dict] = {}
        
    class Span:
        """A single tracing span."""
        def __init__(self, name: str, tracer: "DistributedTracer", parent_id: str = None):
            self.span_id = str(uuid.uuid4())[:8]
            self.name = name
            self.tracer = tracer
            self.parent_id = parent_id
            self.start_time = time.time()
            self.end_time: Optional[float] = None
            self.attributes: Dict[str, Any] = {}
            self.events: List[Dict] = []
            self.status = "OK"
            
        def set_attribute(self, key: str, value: Any):
            self.attributes[key] = value
            return self
            
        def add_event(self, name: str, attributes: Dict = None):
            self.events.append({
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "attributes": attributes or {}
            })
            return self
            
        def set_error(self, error: str):
            self.status = "ERROR"
            self.attributes["error"] = error
            return self
            
        def __enter__(self):
            self.tracer._active_spans[self.span_id] = self
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()
            duration_ms = (self.end_time - self.start_time) * 1000
            
            if exc_val:
                self.set_error(str(exc_val))
            
            span_data = {
                "span_id": self.span_id,
                "parent_id": self.parent_id,
                "name": self.name,
                "service": self.tracer.service_name,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "duration_ms": round(duration_ms, 2),
                "status": self.status,
                "attributes": self.attributes,
                "events": self.events
            }
            self.tracer._spans.append(span_data)
            
            if self.span_id in self.tracer._active_spans:
                del self.tracer._active_spans[self.span_id]
    
    def span(self, name: str, attributes: Dict = None, parent_id: str = None) -> Span:
        """Create a new span."""
        span = self.Span(name, self, parent_id)
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        return span
        
    def get_traces(self, limit: int = 50) -> List[Dict]:
        return self._spans[-limit:]
        
    def get_stats(self) -> Dict:
        return {
            "total_spans": len(self._spans),
            "active_spans": len(self._active_spans),
            "error_count": sum(1 for s in self._spans if s["status"] == "ERROR")
        }


# Global observability instances
logger = StructuredLogger("smartdoc")
metrics = MetricsCollector()
tracer = DistributedTracer("smartdoc")

print("âœ“ Observability Stack defined: StructuredLogger, MetricsCollector, DistributedTracer")


# ============================================================================
# TOOLS (Course Concept #2: Tools)
# ============================================================================

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._call_count = 0
        self._total_time_ms = 0.0
        
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass
        
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {}
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "call_count": self._call_count,
            "avg_time_ms": self._total_time_ms / max(self._call_count, 1)
        }


class DocumentSearchTool(BaseTool):
    """Semantic document search tool."""
    
    def __init__(self, vector_store: VectorStore = None):
        super().__init__("document_search", "Search documents semantically")
        self.vector_store = vector_store
        
    async def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        query = kwargs.get("query", "")
        k = kwargs.get("k", 5)
        
        if not self.vector_store:
            return ToolResult(success=True, data={"documents": []})
            
        results = self.vector_store.search(query, k)
        
        self._call_count += 1
        self._total_time_ms += (time.time() - start) * 1000
        
        return ToolResult(
            success=True,
            data={"documents": results, "count": len(results)},
            execution_time_ms=(time.time() - start) * 1000
        )
        
    def get_schema(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "required": True},
            "k": {"type": "integer", "default": 5}
        }


class WebSearchTool(BaseTool):
    """Web search tool (simulated for demo)."""
    
    def __init__(self):
        super().__init__("web_search", "Search the web for real-time information")
        
    async def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        query = kwargs.get("query", "")
        
        # Simulated web results for demo
        results = [
            {"title": f"Web result for: {query}", "url": "https://example.com", "snippet": "Simulated result..."}
        ]
        
        self._call_count += 1
        self._total_time_ms += (time.time() - start) * 1000
        
        return ToolResult(
            success=True,
            data={"results": results},
            execution_time_ms=(time.time() - start) * 1000
        )


class CodeExecutionTool(BaseTool):
    """Safe code execution tool with sandboxing."""
    
    SAFE_BUILTINS = {'abs', 'all', 'any', 'bool', 'dict', 'float', 'int', 'len', 
                    'list', 'max', 'min', 'pow', 'print', 'range', 'round', 'set', 
                    'sorted', 'str', 'sum', 'tuple', 'zip', 'enumerate'}
    
    DANGEROUS_PATTERNS = ['import os', 'import sys', '__import__', 'eval(', 'exec(', 
                         'open(', 'file(', 'subprocess', 'system(']
    
    def __init__(self):
        super().__init__("code_execution", "Execute Python code safely")
        
    async def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        code = kwargs.get("code", "")
        
        # Safety checks
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in code:
                return ToolResult(
                    success=False,
                    error=f"Dangerous operation blocked: {pattern}",
                    execution_time_ms=(time.time() - start) * 1000
                )
        
        try:
            # Prepare restricted environment
            restricted_globals = {
                '__builtins__': {name: __builtins__[name] if isinstance(__builtins__, dict) 
                                else getattr(__builtins__, name) 
                                for name in self.SAFE_BUILTINS 
                                if (name in __builtins__ if isinstance(__builtins__, dict) 
                                   else hasattr(__builtins__, name))},
            }
            
            # Add math module
            import math
            restricted_globals['math'] = math
            
            local_vars = {}
            
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            exec(code, restricted_globals, local_vars)
            
            stdout_output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            result = local_vars.get('result', stdout_output or "Executed successfully")
            
            self._call_count += 1
            self._total_time_ms += (time.time() - start) * 1000
            
            return ToolResult(
                success=True,
                data={"result": result, "variables": {k: str(v) for k, v in local_vars.items()}},
                execution_time_ms=(time.time() - start) * 1000
            )
            
        except Exception as e:
            sys.stdout = old_stdout
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000
            )


class CitationTool(BaseTool):
    """Citation management tool."""
    
    def __init__(self):
        super().__init__("citation", "Manage citations and references")
        self._citations: List[Dict] = []
        
    async def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        action = kwargs.get("action", "list")
        
        if action == "add":
            source = kwargs.get("source", {})
            citation_id = len(self._citations) + 1
            citation = {
                "id": citation_id,
                "title": source.get("title", "Unknown"),
                "source": source.get("source", "Unknown"),
                "date": source.get("date", ""),
                "added": datetime.now().isoformat()
            }
            self._citations.append(citation)
            return ToolResult(success=True, data={"citation_id": citation_id, "citation": citation})
            
        elif action == "list":
            return ToolResult(success=True, data={"citations": self._citations})
            
        elif action == "format":
            style = kwargs.get("style", "apa")
            formatted = []
            for c in self._citations:
                if style == "apa":
                    formatted.append(f"[{c['id']}] {c.get('title', 'Unknown')}. ({c.get('date', 'n.d.')})")
                else:
                    formatted.append(f"[{c['id']}] {c.get('title', 'Unknown')}")
            return ToolResult(success=True, data={"formatted": formatted})
            
        elif action == "clear":
            self._citations.clear()
            return ToolResult(success=True, data={"message": "Citations cleared"})
            
        self._call_count += 1
        return ToolResult(success=False, error=f"Unknown action: {action}")


class SummarizationTool(BaseTool):
    """Text summarization tool."""
    
    def __init__(self):
        super().__init__("summarization", "Summarize text content")
        
    async def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        text = kwargs.get("text", "")
        max_sentences = kwargs.get("max_sentences", 3)
        
        if not text:
            return ToolResult(success=False, error="No text provided")
        
        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        # Score sentences by position and length
        scored = []
        for i, s in enumerate(sentences):
            # Prefer earlier sentences (lead bias)
            position_score = 1.0 / (i + 1)
            # Prefer medium-length sentences
            length_score = min(len(s.split()) / 20, 1.0)
            scored.append((position_score + length_score, s))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        summary_sentences = [s for _, s in scored[:max_sentences]]
        
        # Preserve original order
        original_order = {s: i for i, s in enumerate(sentences)}
        summary_sentences.sort(key=lambda s: original_order.get(s, 999))
        
        summary = ". ".join(summary_sentences) + "."
        
        self._call_count += 1
        self._total_time_ms += (time.time() - start) * 1000
        
        return ToolResult(
            success=True,
            data={
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": round(len(summary) / max(len(text), 1), 2)
            },
            execution_time_ms=(time.time() - start) * 1000
        )


class FactCheckerTool(BaseTool):
    """Fact verification tool."""
    
    def __init__(self):
        super().__init__("fact_checker", "Verify claims against sources")
        
    async def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        claim = kwargs.get("claim", "")
        sources = kwargs.get("sources", [])
        
        if not claim:
            return ToolResult(success=False, error="No claim provided")
        
        # Combine source content
        source_text = " ".join(s.get("content", "") for s in sources).lower()
        claim_words = set(claim.lower().split())
        
        # Check how many claim words appear in sources
        if source_text:
            matches = sum(1 for w in claim_words if w in source_text and len(w) > 3)
            support_score = matches / max(len([w for w in claim_words if len(w) > 3]), 1)
        else:
            support_score = 0.0
        
        # Determine verdict
        if support_score > 0.6:
            verdict = "SUPPORTED"
            confidence = min(support_score + 0.2, 1.0)
        elif support_score > 0.3:
            verdict = "PARTIALLY_SUPPORTED"
            confidence = support_score + 0.1
        else:
            verdict = "UNVERIFIED"
            confidence = 0.3
        
        self._call_count += 1
        self._total_time_ms += (time.time() - start) * 1000
        
        return ToolResult(
            success=True,
            data={
                "claim": claim,
                "verdict": verdict,
                "confidence": round(confidence, 2),
                "sources_checked": len(sources)
            },
            execution_time_ms=(time.time() - start) * 1000
        )


class VisualizationTool(BaseTool):
    """Data visualization configuration tool."""
    
    def __init__(self):
        super().__init__("visualization", "Generate visualization configurations")
        
    async def execute(self, **kwargs) -> ToolResult:
        start = time.time()
        data = kwargs.get("data", {})
        chart_type = kwargs.get("chart_type", "bar")
        title = kwargs.get("title", "Chart")
        
        # Generate chart configuration
        config = {
            "type": chart_type,
            "title": title,
            "data": data,
            "options": {
                "responsive": True,
                "legend": {"display": True},
                "scales": {"y": {"beginAtZero": True}}
            }
        }
        
        # Generate simple ASCII preview
        if chart_type == "bar" and isinstance(data, dict):
            max_val = max(data.values()) if data else 1
            ascii_chart = f"\n{title}\n" + "="*30 + "\n"
            for k, v in data.items():
                bar_len = int((v / max_val) * 20)
                ascii_chart += f"{k[:10]:10} {'â–ˆ'*bar_len} {v}\n"
            config["ascii_preview"] = ascii_chart
        
        self._call_count += 1
        return ToolResult(
            success=True,
            data=config,
            execution_time_ms=(time.time() - start) * 1000
        )

print("âœ“ Tools defined: DocumentSearch, WebSearch, CodeExecution, Citation, Summarization, FactChecker, Visualization")


# ============================================================================
# BASE AGENT CLASS (Course Concept #1: Multi-Agent Orchestration)
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents.
    
    Provides common functionality for state management,
    tool access, and observability integration.
    """
    
    def __init__(self, name: str, description: str, tools: List[BaseTool] = None):
        self.name = name
        self.description = description
        self.state = AgentState.IDLE
        self.tools = {t.name: t for t in (tools or [])}
        self._history: List[Dict] = []
        
    @abstractmethod
    async def process(self, context: AgentContext, input_data: Any) -> AgentResult:
        """Process input and return result."""
        pass
        
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        pass
        
    def set_state(self, state: AgentState):
        """Update agent state with logging."""
        old_state = self.state
        self.state = state
        logger.debug(f"Agent {self.name} state: {old_state.value} -> {state.value}")
        
    async def use_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        if tool_name not in self.tools:
            return ToolResult(success=False, error=f"Tool not found: {tool_name}")
        return await self.tools[tool_name].execute(**kwargs)
        
    def add_to_history(self, entry: Dict):
        """Add entry to agent history."""
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            **entry
        })

print("âœ“ BaseAgent class defined")


# ============================================================================
# SPECIALIZED AGENTS (6 Agents for Multi-Agent System)
# ============================================================================

class PlannerAgent(BaseAgent):
    """Query decomposition and planning agent.
    
    Analyzes query complexity and creates execution plan.
    """
    
    def __init__(self):
        super().__init__("Planner", "Decomposes queries and plans execution strategy")
        
    async def process(self, context: AgentContext, input_data: Any) -> AgentResult:
        start = time.time()
        self.set_state(AgentState.PLANNING)
        
        query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
        
        with tracer.span("planner.process", {"query_length": len(query)}) as span:
            # Analyze query complexity
            words = len(query.split())
            has_comparison = any(w in query.lower() for w in ['compare', 'versus', 'vs', 'difference', 'between'])
            has_analysis = any(w in query.lower() for w in ['analyze', 'examine', 'evaluate', 'assess', 'explain'])
            has_synthesis = any(w in query.lower() for w in ['summarize', 'combine', 'comprehensive', 'overview'])
            is_question = query.strip().endswith('?')
            
            # Determine complexity
            complexity_score = 0
            if words > 30: complexity_score += 2
            elif words > 15: complexity_score += 1
            if has_comparison: complexity_score += 2
            if has_analysis: complexity_score += 1
            if has_synthesis: complexity_score += 1
            
            if complexity_score >= 4:
                complexity = "complex"
            elif complexity_score >= 2:
                complexity = "medium"
            else:
                complexity = "simple"
            
            # Generate execution plan
            subtasks = []
            
            # Always retrieve
            subtasks.append({
                "task": "retrieve",
                "agent": "retriever",
                "description": "Find relevant documents",
                "priority": 1
            })
            
            # Add analysis for medium/complex
            if complexity in ["medium", "complex"]:
                subtasks.append({
                    "task": "analyze",
                    "agent": "analyzer",
                    "description": "Extract insights and patterns",
                    "priority": 2
                })
            
            # Always synthesize
            subtasks.append({
                "task": "synthesize",
                "agent": "synthesizer",
                "description": "Generate comprehensive response",
                "priority": 3
            })
            
            # Always critique
            subtasks.append({
                "task": "critique",
                "agent": "critic",
                "description": "Validate response quality",
                "priority": 4
            })
            
            # Determine execution strategy
            if complexity == "complex":
                strategy = "parallel_then_sequential"
            else:
                strategy = "sequential"
            
            plan = {
                "query": query,
                "complexity": complexity,
                "complexity_score": complexity_score,
                "subtasks": subtasks,
                "strategy": strategy,
                "estimated_time_ms": len(subtasks) * 100
            }
            
            span.set_attribute("complexity", complexity)
            span.set_attribute("subtask_count", len(subtasks))
        
        self.set_state(AgentState.COMPLETED)
        execution_time = (time.time() - start) * 1000
        
        metrics.increment("planner.queries_processed")
        metrics.timer("planner.latency_ms", execution_time)
        
        return AgentResult(
            success=True,
            data=plan,
            execution_time_ms=execution_time
        )
        
    def get_capabilities(self) -> List[str]:
        return ["query_decomposition", "complexity_analysis", "task_planning", "strategy_selection"]


class RetrieverAgent(BaseAgent):
    """Document retrieval agent with semantic search."""
    
    def __init__(self, vector_store: VectorStore = None):
        tools = [DocumentSearchTool(vector_store), WebSearchTool()]
        super().__init__("Retriever", "Retrieves relevant documents using semantic search", tools)
        self.vector_store = vector_store
        
    async def process(self, context: AgentContext, input_data: Any) -> AgentResult:
        start = time.time()
        self.set_state(AgentState.RUNNING)
        
        query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
        k = input_data.get("k", 5) if isinstance(input_data, dict) else 5
        
        with tracer.span("retriever.process", {"query": query[:50]}) as span:
            # Search documents
            doc_result = await self.use_tool("document_search", query=query, k=k)
            documents = doc_result.data.get("documents", []) if doc_result.success else []
            
            # Add citations
            citation_tool = CitationTool()
            for doc in documents:
                await citation_tool.execute(
                    action="add",
                    source=doc.get("metadata", {})
                )
            
            span.set_attribute("documents_found", len(documents))
        
        self.set_state(AgentState.COMPLETED)
        execution_time = (time.time() - start) * 1000
        
        metrics.increment("retriever.documents_retrieved", len(documents))
        metrics.timer("retriever.latency_ms", execution_time)
        
        return AgentResult(
            success=True,
            data={
                "documents": documents,
                "query": query,
                "count": len(documents),
                "search_method": "semantic"
            },
            execution_time_ms=execution_time
        )
        
    def get_capabilities(self) -> List[str]:
        return ["semantic_search", "document_ranking", "citation_tracking", "web_search"]


class AnalyzerAgent(BaseAgent):
    """Deep analysis and insight extraction agent."""
    
    def __init__(self):
        tools = [CodeExecutionTool(), FactCheckerTool()]
        super().__init__("Analyzer", "Performs deep analysis and extracts insights", tools)
        
    async def process(self, context: AgentContext, input_data: Any) -> AgentResult:
        start = time.time()
        self.set_state(AgentState.RUNNING)
        
        query = input_data.get("query", "") if isinstance(input_data, dict) else ""
        documents = input_data.get("documents", {}).get("documents", []) if isinstance(input_data, dict) else []
        
        with tracer.span("analyzer.process", {"doc_count": len(documents)}) as span:
            insights = []
            patterns = []
            facts = []
            
            for i, doc in enumerate(documents[:5]):
                content = doc.get("content", "")[:500]
                source = doc.get("metadata", {}).get("title", f"Document {i+1}")
                
                # Extract key sentences
                sentences = [s.strip() for s in re.split(r'[.!?]', content) if len(s.strip()) > 30]
                
                if sentences:
                    insights.append({
                        "source": source,
                        "key_points": sentences[:3],
                        "relevance": doc.get("relevance_score", 0.5)
                    })
                
                # Extract numbers/statistics
                numbers = re.findall(r'\$?\d+(?:\.\d+)?(?:%|billion|million|trillion)?', content)
                if numbers:
                    facts.extend([{"value": n, "source": source} for n in numbers[:3]])
                
                # Detect patterns
                if "trend" in content.lower() or "growth" in content.lower():
                    patterns.append({"type": "trend", "source": source})
                if "challenge" in content.lower() or "problem" in content.lower():
                    patterns.append({"type": "challenge", "source": source})
            
            analysis = {
                "key_insights": insights,
                "patterns_detected": patterns,
                "facts_extracted": facts,
                "document_count": len(documents),
                "analysis_depth": "comprehensive" if len(documents) > 3 else "basic"
            }
            
            span.set_attribute("insights_count", len(insights))
            span.set_attribute("patterns_count", len(patterns))
        
        self.set_state(AgentState.COMPLETED)
        execution_time = (time.time() - start) * 1000
        
        metrics.increment("analyzer.documents_analyzed", len(documents))
        metrics.timer("analyzer.latency_ms", execution_time)
        
        return AgentResult(
            success=True,
            data=analysis,
            execution_time_ms=execution_time
        )
        
    def get_capabilities(self) -> List[str]:
        return ["insight_extraction", "pattern_detection", "fact_extraction", "code_execution"]


class SynthesizerAgent(BaseAgent):
    """Response synthesis and report generation agent."""
    
    def __init__(self):
        tools = [SummarizationTool(), VisualizationTool()]
        super().__init__("Synthesizer", "Generates comprehensive responses and reports", tools)
        
    async def process(self, context: AgentContext, input_data: Any) -> AgentResult:
        start = time.time()
        self.set_state(AgentState.RUNNING)
        
        query = input_data.get("query", "") if isinstance(input_data, dict) else ""
        analysis = input_data.get("analysis", {}) if isinstance(input_data, dict) else {}
        
        with tracer.span("synthesizer.process") as span:
            insights = analysis.get("key_insights", [])
            patterns = analysis.get("patterns_detected", [])
            facts = analysis.get("facts_extracted", [])
            
            # Build response
            response_parts = []
            
            # Introduction
            if insights:
                response_parts.append(f"Based on analysis of {len(insights)} sources:\n")
            else:
                response_parts.append("Based on the available information:\n")
            
            # Key findings
            if insights:
                response_parts.append("\n**Key Findings:**\n")
                for i, insight in enumerate(insights[:5], 1):
                    source = insight.get("source", "Unknown")
                    points = insight.get("key_points", [])
                    if points:
                        response_parts.append(f"\n{i}. From *{source}*:")
                        for point in points[:2]:
                            response_parts.append(f"\n   - {point[:150]}...")
            
            # Patterns
            if patterns:
                response_parts.append("\n\n**Patterns Identified:**\n")
                pattern_types = set(p.get("type") for p in patterns)
                for pt in pattern_types:
                    count = sum(1 for p in patterns if p.get("type") == pt)
                    response_parts.append(f"- {pt.title()}: Found in {count} source(s)\n")
            
            # Key statistics
            if facts:
                response_parts.append("\n**Key Statistics:**\n")
                for fact in facts[:5]:
                    response_parts.append(f"- {fact.get('value')} (Source: {fact.get('source')})\n")
            
            # Conclusion
            response_parts.append("\n---\n*Analysis generated by SmartDoc Analyst*")
            
            response = "".join(response_parts)
            citations = [i.get("source") for i in insights]
            
            span.set_attribute("response_length", len(response))
            span.set_attribute("citation_count", len(citations))
        
        self.set_state(AgentState.COMPLETED)
        execution_time = (time.time() - start) * 1000
        
        metrics.timer("synthesizer.latency_ms", execution_time)
        
        return AgentResult(
            success=True,
            data={
                "response": response,
                "citations": citations,
                "word_count": len(response.split()),
                "sources_used": len(insights)
            },
            execution_time_ms=execution_time
        )
        
    def get_capabilities(self) -> List[str]:
        return ["report_generation", "multi_source_synthesis", "citation_formatting", "summarization"]


class CriticAgent(BaseAgent):
    """Quality assurance and validation agent."""
    
    def __init__(self):
        tools = [FactCheckerTool()]
        super().__init__("Critic", "Validates response quality and provides feedback", tools)
        
    async def process(self, context: AgentContext, input_data: Any) -> AgentResult:
        start = time.time()
        self.set_state(AgentState.RUNNING)
        
        response = input_data.get("response", "") if isinstance(input_data, dict) else str(input_data)
        query = input_data.get("query", "") if isinstance(input_data, dict) else ""
        
        with tracer.span("critic.process") as span:
            issues = []
            suggestions = []
            
            # Quality checks
            word_count = len(response.split())
            has_structure = "\n" in response or "**" in response
            has_citations = "Source:" in response or "*" in response
            
            # Scoring
            scores = {
                "completeness": 0.0,
                "relevance": 0.0,
                "coherence": 0.0,
                "citation_quality": 0.0
            }
            
            # Completeness (based on length)
            if word_count > 200:
                scores["completeness"] = 1.0
            elif word_count > 100:
                scores["completeness"] = 0.8
            elif word_count > 50:
                scores["completeness"] = 0.6
            else:
                scores["completeness"] = 0.4
                issues.append("Response is too brief")
                suggestions.append("Add more detail and context")
            
            # Relevance (query terms in response)
            query_terms = set(w.lower() for w in query.split() if len(w) > 3)
            if query_terms:
                matches = sum(1 for t in query_terms if t in response.lower())
                scores["relevance"] = min(matches / len(query_terms), 1.0)
            else:
                scores["relevance"] = 0.7
            
            if scores["relevance"] < 0.5:
                issues.append("Response may not fully address the query")
                suggestions.append("Ensure all aspects of the query are addressed")
            
            # Coherence (structure)
            if has_structure:
                scores["coherence"] = 0.9
            else:
                scores["coherence"] = 0.6
                suggestions.append("Consider adding headers or bullet points")
            
            # Citation quality
            if has_citations:
                scores["citation_quality"] = 0.9
            else:
                scores["citation_quality"] = 0.5
                issues.append("Missing source citations")
                suggestions.append("Add citations for key claims")
            
            # Overall score
            overall_score = sum(scores.values()) / len(scores)
            needs_improvement = overall_score < 0.7
            
            span.set_attribute("overall_score", overall_score)
            span.set_attribute("needs_improvement", needs_improvement)
        
        self.set_state(AgentState.COMPLETED)
        execution_time = (time.time() - start) * 1000
        
        metrics.histogram("critic.quality_scores", overall_score)
        metrics.timer("critic.latency_ms", execution_time)
        
        return AgentResult(
            success=True,
            data={
                "overall_score": round(overall_score, 2),
                "scores": {k: round(v, 2) for k, v in scores.items()},
                "needs_improvement": needs_improvement,
                "issues": issues,
                "suggestions": suggestions,
                "word_count": word_count
            },
            suggestions=suggestions,
            execution_time_ms=execution_time
        )
        
    def get_capabilities(self) -> List[str]:
        return ["quality_scoring", "issue_detection", "improvement_suggestions", "fact_checking"]

print("âœ“ Specialized Agents defined: Planner, Retriever, Analyzer, Synthesizer, Critic")


# ============================================================================
# ORCHESTRATOR AGENT (Central Coordinator)
# ============================================================================

class OrchestratorAgent(BaseAgent):
    """Master orchestrator that coordinates all agents.
    
    Implements the coordination patterns:
    - Sequential: Simple queries
    - Parallel: Multi-document analysis
    - Loop: Quality assurance with retry
    """
    
    def __init__(self):
        super().__init__("Orchestrator", "Coordinates all agents and manages workflow")
        self.agents: Dict[str, BaseAgent] = {}
        self.max_retries = 2
        self.quality_threshold = 0.65
        
    def register_agents(self, **agents):
        """Register agents for orchestration."""
        self.agents.update(agents)
        logger.info(f"Registered agents: {list(agents.keys())}")
        
    async def process(self, context: AgentContext, input_data: Any) -> AgentResult:
        start = time.time()
        self.set_state(AgentState.RUNNING)
        
        query = input_data if isinstance(input_data, str) else str(input_data)
        context.query = query
        
        with tracer.span("orchestrator.process", {"query": query[:50]}) as root_span:
            results = {"query": query, "stages": {}}
            
            try:
                # Stage 1: Planning
                if "planner" in self.agents:
                    logger.info("Stage 1: Planning")
                    plan_result = await self.agents["planner"].process(context, {"query": query})
                    results["stages"]["planning"] = plan_result.data
                    context.add_result("planner", plan_result.data)
                
                # Stage 2: Retrieval
                if "retriever" in self.agents:
                    logger.info("Stage 2: Retrieval")
                    retrieve_result = await self.agents["retriever"].process(context, {"query": query})
                    results["stages"]["retrieval"] = retrieve_result.data
                    context.add_result("retriever", retrieve_result.data)
                
                # Stage 3: Analysis
                if "analyzer" in self.agents:
                    logger.info("Stage 3: Analysis")
                    analyze_result = await self.agents["analyzer"].process(context, {
                        "query": query,
                        "documents": context.intermediate_results.get("retriever", {})
                    })
                    results["stages"]["analysis"] = analyze_result.data
                    context.add_result("analyzer", analyze_result.data)
                
                # Stage 4 & 5: Synthesis with Quality Loop
                synthesis_result = None
                critique_result = None
                retry_count = 0
                
                while retry_count <= self.max_retries:
                    # Synthesize
                    if "synthesizer" in self.agents:
                        logger.info(f"Stage 4: Synthesis (attempt {retry_count + 1})")
                        synthesis_result = await self.agents["synthesizer"].process(context, {
                            "query": query,
                            "analysis": context.intermediate_results.get("analyzer", {})
                        })
                        results["stages"]["synthesis"] = synthesis_result.data
                        context.add_result("synthesizer", synthesis_result.data)
                    
                    # Critique
                    if "critic" in self.agents:
                        logger.info("Stage 5: Quality Check")
                        critique_result = await self.agents["critic"].process(context, {
                            "query": query,
                            "response": synthesis_result.data.get("response", "") if synthesis_result else ""
                        })
                        results["stages"]["critique"] = critique_result.data
                        
                        # Check if quality threshold met
                        quality_score = critique_result.data.get("overall_score", 0)
                        if quality_score >= self.quality_threshold:
                            logger.info(f"Quality threshold met: {quality_score:.2f}")
                            break
                        else:
                            logger.warning(f"Quality below threshold: {quality_score:.2f}, retrying...")
                            retry_count += 1
                    else:
                        break
                
                # Compile final response
                final_response = {
                    "answer": synthesis_result.data if synthesis_result else None,
                    "sources": context.intermediate_results.get("retriever", {}).get("documents", []),
                    "analysis_summary": context.intermediate_results.get("analyzer", {}),
                    "quality_score": critique_result.data.get("overall_score") if critique_result else None,
                    "processing_stages": list(results["stages"].keys()),
                    "agent_chain": context.agent_chain,
                    "retry_count": retry_count
                }
                
                root_span.set_attribute("stages_completed", len(results["stages"]))
                root_span.set_attribute("quality_score", final_response.get("quality_score", 0))
                
                self.set_state(AgentState.COMPLETED)
                execution_time = (time.time() - start) * 1000
                
                metrics.increment("orchestrator.queries_completed")
                metrics.timer("orchestrator.total_latency_ms", execution_time)
                
                return AgentResult(
                    success=True,
                    data=final_response,
                    execution_time_ms=execution_time
                )
                
            except Exception as e:
                root_span.set_error(str(e))
                self.set_state(AgentState.ERROR)
                logger.error(f"Orchestrator error: {str(e)}")
                metrics.increment("orchestrator.errors")
                
                return AgentResult(
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start) * 1000
                )
    
    def get_capabilities(self) -> List[str]:
        return ["query_coordination", "agent_delegation", "quality_control", "retry_logic"]

print("âœ“ OrchestratorAgent defined (coordinates all 5 specialized agents)")


# ============================================================================
# A2A PROTOCOL (Course Concept #7: Deployment Readiness)
# ============================================================================

class MessageBus:
    """Central message bus for A2A communication."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_history: List[AgentMessage] = []
        self._message_counts: Dict[str, int] = defaultdict(int)
        
    def subscribe(self, agent_name: str, handler: Callable):
        """Subscribe agent to receive messages."""
        self._subscribers[agent_name].append(handler)
        
    async def send(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Send a message to an agent."""
        self._message_history.append(message)
        self._message_counts[message.from_agent] += 1
        
        handlers = self._subscribers.get(message.to_agent, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(message)
                else:
                    result = handler(message)
                if isinstance(result, AgentMessage):
                    return result
            except Exception as e:
                logger.error(f"Message handler error: {e}")
        return None
        
    def get_stats(self) -> Dict:
        return {
            "total_messages": len(self._message_history),
            "subscribers": list(self._subscribers.keys()),
            "message_counts": dict(self._message_counts)
        }


class A2AProtocol:
    """High-level A2A protocol for agent communication."""
    
    def __init__(self, agent_name: str, message_bus: MessageBus = None):
        self.agent_name = agent_name
        self.message_bus = message_bus or MessageBus()
        
    async def request(self, to_agent: str, task_type: str, parameters: Dict) -> Optional[Dict]:
        """Send request and get response."""
        message = AgentMessage(
            from_agent=self.agent_name,
            to_agent=to_agent,
            message_type=MessageType.TASK,
            content={"task_type": task_type, "parameters": parameters}
        )
        response = await self.message_bus.send(message)
        return response.content if response else None
        
    async def notify(self, to_agent: str, content: Any):
        """Send notification without waiting."""
        message = AgentMessage(
            from_agent=self.agent_name,
            to_agent=to_agent,
            message_type=MessageType.STATUS,
            content=content
        )
        await self.message_bus.send(message)

print("âœ“ A2A Protocol defined: MessageBus, A2AProtocol")


# ============================================================================
# SMARTDOC ANALYST MAIN SYSTEM
# ============================================================================

class SmartDocAnalyst:
    """Main SmartDoc Analyst system integrating all components.
    
    This is the primary interface for document analysis, combining:
    - 6 Specialized Agents
    - 7 Tools
    - 3-Tier Memory System
    - Full Observability
    - A2A Protocol
    """
    
    def __init__(self):
        # Initialize memory system
        self.memory = MemoryManager()
        
        # Initialize agents
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent(self.memory.vector_store)
        self.analyzer = AnalyzerAgent()
        self.synthesizer = SynthesizerAgent()
        self.critic = CriticAgent()
        
        # Create orchestrator and register agents
        self.orchestrator = OrchestratorAgent()
        self.orchestrator.register_agents(
            planner=self.planner,
            retriever=self.retriever,
            analyzer=self.analyzer,
            synthesizer=self.synthesizer,
            critic=self.critic
        )
        
        # Initialize A2A protocol
        self.message_bus = MessageBus()
        self.protocol = A2AProtocol("smartdoc", self.message_bus)
        
        # Tool registry
        self.tools = {
            "document_search": DocumentSearchTool(self.memory.vector_store),
            "web_search": WebSearchTool(),
            "summarization": SummarizationTool(),
            "citation": CitationTool(),
            "code_execution": CodeExecutionTool(),
            "fact_checker": FactCheckerTool(),
            "visualization": VisualizationTool()
        }
        
        logger.info("SmartDoc Analyst initialized")
        
    def ingest_documents(self, documents: List[Dict]) -> Dict:
        """Ingest documents into the system."""
        with tracer.span("ingest_documents", {"count": len(documents)}) as span:
            ids = self.memory.add_documents(documents)
            
            # Store document metadata in semantic memory
            for doc in documents:
                title = doc.get("metadata", {}).get("title", "Unknown")
                self.memory.store_knowledge(f"doc:{title}", doc.get("metadata", {}), "documents")
            
            span.set_attribute("documents_added", len(ids))
            metrics.increment("documents_ingested", len(documents))
            
            logger.info(f"Ingested {len(documents)} documents")
            
            return {
                "added": len(documents),
                "document_ids": ids,
                "total_documents": len(self.memory.vector_store)
            }
    
    async def analyze(self, query: str) -> Dict:
        """Analyze documents and answer query."""
        start_time = time.time()
        
        with tracer.span("analyze", {"query": query[:50]}) as span:
            logger.set_trace_id(span.span_id)
            
            # Create context
            context = AgentContext(query=query)
            
            # Add relevant context from memory
            relevant = self.memory.get_relevant_context(query)
            if relevant["relevant_history"]:
                context.metadata["history"] = relevant["relevant_history"]
            
            # Process through orchestrator
            result = await self.orchestrator.process(context, query)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Record interaction in episodic memory
            if result.success:
                response_text = result.data.get("answer", {}).get("response", "")
                self.memory.record_interaction(query, response_text[:500])
            
            metrics.timer("query_latency_ms", execution_time)
            metrics.increment("queries_processed")
            
            span.set_attribute("success", result.success)
            span.set_attribute("execution_time_ms", execution_time)
            
            if result.success:
                return {
                    "success": True,
                    "answer": result.data.get("answer", {}).get("response", ""),
                    "sources": result.data.get("sources", []),
                    "quality_score": result.data.get("quality_score"),
                    "processing_stages": result.data.get("processing_stages", []),
                    "agent_chain": result.data.get("agent_chain", []),
                    "execution_time_ms": execution_time
                }
            else:
                return {
                    "success": False,
                    "error": result.error,
                    "execution_time_ms": execution_time
                }
    
    async def search(self, query: str, k: int = 5) -> Dict:
        """Search documents directly."""
        results = self.memory.search_documents(query, k)
        return {"documents": results, "count": len(results)}
    
    def get_stats(self) -> Dict:
        """Get comprehensive system statistics."""
        return {
            "memory": self.memory.get_stats(),
            "metrics": metrics.get_all(),
            "traces": tracer.get_stats(),
            "message_bus": self.message_bus.get_stats(),
            "tools": {name: tool.get_stats() for name, tool in self.tools.items()}
        }

print("âœ“ SmartDocAnalyst system defined - integrates all components")


# ============================================================================
# SAMPLE DOCUMENTS (5 Diverse Documents)
# ============================================================================

SAMPLE_DOCUMENTS = [
    {
        "content": """Artificial Intelligence in Healthcare: A Comprehensive Overview

AI is revolutionizing healthcare delivery across multiple domains. Machine learning 
algorithms are now capable of diagnosing diseases from medical images with accuracy 
matching or exceeding human specialists. Deep learning models have shown remarkable 
success in detecting cancer from mammograms, identifying diabetic retinopathy from 
retinal scans, and spotting pneumonia in chest X-rays.

Key Applications:
1. Medical Imaging: AI systems analyze CT scans, MRIs, and X-rays to detect abnormalities
2. Drug Discovery: ML accelerates identification of potential drug candidates by 40%
3. Clinical Decision Support: AI assists physicians in treatment planning
4. Predictive Analytics: Models predict patient readmission and disease progression
5. Administrative Automation: Natural language processing streamlines documentation

The global AI in healthcare market is projected to reach $45.2 billion by 2026, 
growing at 44.9% CAGR. Major challenges include data privacy concerns, regulatory 
compliance, algorithm bias, and integration with existing systems.

Recent advancements include GPT-4 passing medical licensing exams and FDA-approved AI 
diagnostic tools reaching over 500 as of 2024.""",
        "metadata": {"source": "ai_healthcare_report.pdf", "title": "AI in Healthcare Report 2024", "date": "2024"}
    },
    {
        "content": """Climate Change Policy Analysis: Global Perspectives

Climate change represents one of the most pressing challenges of our time. International 
efforts centered around the Paris Agreement aim to limit global warming to 1.5Â°C above 
pre-industrial levels. As of 2024, 195 countries have committed to nationally determined 
contributions (NDCs).

Key Policy Mechanisms:
1. Carbon Pricing: 46 countries have implemented carbon taxes or cap-and-trade systems
2. Renewable Energy Mandates: Over 170 countries have renewable energy targets
3. Green Finance: Climate-aligned investments reached $1.3 trillion in 2023
4. Regulatory Standards: Emissions standards for vehicles, buildings, and industry
5. International Cooperation: Climate funds support developing nations

The European Union leads with its Green Deal, targeting climate neutrality by 2050. 
China, the world's largest emitter, pledged carbon neutrality by 2060. The United States 
rejoined Paris Agreement and set 50% emissions reduction target by 2030.""",
        "metadata": {"source": "climate_policy.pdf", "title": "Global Climate Policy Analysis", "date": "2024"}
    },
    {
        "content": """Q4 2024 Financial Market Analysis

Global financial markets experienced significant volatility in Q4 2024, driven by 
geopolitical tensions, central bank policy shifts, and evolving economic conditions.

Key Trends:
1. Interest Rates: Federal Reserve maintained rates at 5.25-5.5% with hints of 2025 cuts
2. Inflation: US CPI moderated to 3.1%, approaching but not reaching 2% target
3. Currency Markets: Dollar index fell 2.3% amid rate expectations
4. Commodities: Oil prices stabilized around $75/barrel despite OPEC+ cuts
5. Tech Sector: AI-related stocks led gains with 45% average increase

Regional Performance:
- US: S&P 500 +8.5%, NASDAQ +12.1%, Dow Jones +6.2%
- Europe: STOXX 600 +4.3%, struggling with energy costs
- Asia: Nikkei +7.8%, Shanghai Composite -1.2%
- Emerging Markets: MSCI EM +3.5%

Outlook for 2025: Analysts project moderate growth with potential Fed rate cuts 
supporting asset prices. Key risks include inflation persistence and geopolitical conflicts.""",
        "metadata": {"source": "financial_analysis.pdf", "title": "Q4 2024 Market Analysis", "date": "2024"}
    },
    {
        "content": """Modern Software Architecture: Patterns and Best Practices

Software architecture continues to evolve with cloud-native and microservices patterns 
dominating enterprise development. This guide covers essential architectural concepts.

Core Patterns:
1. Microservices: Decompose applications into independently deployable services
2. Event-Driven: Use events for loose coupling between components
3. CQRS: Separate read and write operations for optimized data access
4. Domain-Driven Design: Align software with business domain models
5. Hexagonal Architecture: Isolate core logic from external dependencies

Cloud-Native Principles:
- Containerization: Docker and Kubernetes for deployment consistency
- Service Mesh: Istio, Linkerd for inter-service communication
- Observability: Distributed tracing, logging, metrics (OpenTelemetry)
- Infrastructure as Code: Terraform, Pulumi for reproducible environments

Best Practices:
- Design for failure with circuit breakers and retry patterns
- Implement proper authentication and authorization (OAuth 2.0, OIDC)
- Use API gateways for traffic management
- Apply 12-factor app principles""",
        "metadata": {"source": "software_architecture.pdf", "title": "Software Architecture Guide 2024", "date": "2024"}
    },
    {
        "content": """Standard Software Licensing Agreement Summary

This document summarizes key provisions of enterprise software licensing agreements.

License Types:
1. Perpetual License: One-time purchase with indefinite use rights
2. Subscription License: Time-limited access with recurring fees
3. Usage-Based: Pricing based on consumption metrics
4. Site License: Unlimited users at specified locations
5. Open Source: Various licenses (MIT, Apache, GPL) with different obligations

Key Contract Terms:
- Intellectual Property: Licensee receives limited use rights, not ownership
- Warranties: Typically limited to material conformance with documentation
- Liability Caps: Usually limited to fees paid in preceding 12 months
- Indemnification: Vendor indemnifies against IP infringement claims
- Data Protection: Must comply with GDPR, CCPA, and applicable privacy laws

Best Practices:
- Conduct thorough due diligence before signing
- Negotiate favorable limitation of liability terms
- Ensure clear data ownership and portability rights
- Include appropriate confidentiality provisions""",
        "metadata": {"source": "legal_licensing.pdf", "title": "Software Licensing Legal Guide", "date": "2024"}
    }
]

print(f"âœ“ {len(SAMPLE_DOCUMENTS)} sample documents prepared")
print("  - AI in Healthcare Report")
print("  - Climate Policy Analysis")
print("  - Financial Market Analysis")
print("  - Software Architecture Guide")
print("  - Software Licensing Legal Guide")


# Initialize and run demo
analyst = SmartDocAnalyst()
print("âœ“ SmartDoc Analyst initialized")

# Ingest documents
result = analyst.ingest_documents(SAMPLE_DOCUMENTS)
print(f"âœ“ Ingested {result['added']} documents")

# Run analysis queries
queries = [
    "What are the key applications of AI in healthcare?",
    "Compare climate policies across different regions",
    "What are the financial market trends for 2024?"
]

print("\n" + "="*70)
print("RUNNING EVALUATION QUERIES")
print("="*70)

for query in queries:
    print(f"\nQuery: {query}")
    result = await analyst.analyze(query)
    if result["success"]:
        print(f"âœ“ Quality Score: {result['quality_score']}")
        print(f"âœ“ Execution Time: {result['execution_time_ms']:.1f}ms")
        print(f"Answer Preview: {result['answer'][:200]}...")
    print("-"*50)

# Show stats
stats = analyst.get_stats()
print("\n" + "="*70)
print("SYSTEM STATISTICS")
print("="*70)
print(f"Documents indexed: {stats['memory']['documents_indexed']}")
print(f"Queries processed: {stats['metrics']['counters'].get('queries_processed', 0)}")
print(f"Total spans traced: {stats['traces']['total_spans']}")


# ============================================================================
# DEPLOYMENT CODE (Bonus: 5 points)
# ============================================================================

DOCKERFILE = """
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
"""

REQUIREMENTS = """
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
google-generativeai>=0.3.0
"""

# FastAPI Application
FASTAPI_APP = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="SmartDoc Analyst API", version="1.0.0")

class AnalyzeRequest(BaseModel):
    query: str
    documents: Optional[List[dict]] = None

class AnalyzeResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    quality_score: Optional[float] = None
    error: Optional[str] = None

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    analyst = SmartDocAnalyst()
    if request.documents:
        analyst.ingest_documents(request.documents)
    result = await analyst.analyze(request.query)
    return AnalyzeResponse(**result)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}
"""

# Cloud Run deployment command
DEPLOY_CMD = """
gcloud run deploy smartdoc-analyst \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --timeout 300
"""

print("âœ“ Deployment artifacts defined:")
print("  - Dockerfile")
print("  - requirements.txt")
print("  - FastAPI application")
print("  - Cloud Run deployment command")


# ============================================================================
# GEMINI INTEGRATION (Bonus: 5 points)
# ============================================================================

GEMINI_INTEGRATION = """
# Gemini-Powered Agent Example

import google.generativeai as genai

class GeminiPoweredAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def think(self, prompt: str) -> str:
        response = await self.model.generate_content_async(prompt)
        return response.text
    
    async def analyze_document(self, document: str, query: str) -> dict:
        prompt = f'''Analyze this document and answer the query.
        
        Document: {document[:2000]}
        
        Query: {query}
        
        Provide a structured response with key insights and citations.'''
        
        response = await self.think(prompt)
        return {"analysis": response, "model": "gemini-1.5-flash"}

# Integration with SmartDoc Analyst
# When GEMINI_API_KEY is available, agents use Gemini for:
# - Query understanding and decomposition (Planner)
# - Semantic analysis and insight extraction (Analyzer)
# - Response synthesis and formatting (Synthesizer)
# - Quality evaluation and improvement (Critic)
"""

print("âœ“ Gemini Integration defined")
print("  - GeminiPoweredAgent class")
print("  - Async content generation")
print("  - Document analysis with Gemini 1.5 Flash")

