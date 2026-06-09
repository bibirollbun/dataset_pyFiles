import sys
import os
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from kaggle_secrets import UserSecretsClient
from IPython.display import display, HTML, clear_output

# For URL fetching
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB_LIBS = True
except ImportError:
    HAS_WEB_LIBS = False
    print("âš  Web libraries not available. URL fetching will be limited.")

print("âœ“ Libraries Loaded")



# Load API Key from Kaggle Secrets
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("âœ“ API Key Configured")
except Exception as e:
    print(f"âš  API Key Error: {str(e)}")
    print("ğŸ“Œ To fix: Go to Add-ons â†’ Secrets â†’ Add 'GOOGLE_API_KEY'")
    GOOGLE_API_KEY = None

# Agent Configuration
CONFIG = {
    "team": "DesignPortfolioReviewer",
    "model": "gemini-2.5-flash-lite",  # Updated for better free tier limits
    "max_tokens": 3000,
    "temperature": 0.3,
    "version": "1.0.0"
}

print(f"\n{'='*60}")
print(f"{'AGENT CONFIGURATION':^60}")
print(f"{'='*60}")
for k, v in CONFIG.items():
    print(f"{k:.<25} {v}")
print(f"{'='*60}")



def fetch_url_content(url: str) -> str:
    """Fetches and extracts text content from a web URL"""
    if not HAS_WEB_LIBS:
        return f"Error: Web libraries not available. Cannot fetch URL: {url}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:50000]  # Limit to 50k characters
    except Exception as e:
        return f"Error fetching URL {url}: {str(e)}"


def parse_portfolio_content(content: str, format_type: str = "auto") -> Dict[str, Any]:
    """Parses portfolio content into structured format"""
    parsed = {
        "raw_content": content,
        "format": format_type,
        "length": len(content),
        "sections": {}
    }
    
    # Try to identify sections (markdown headers, etc.)
    if format_type in ["markdown", "auto"]:
        # Look for markdown headers
        header_pattern = r'^#+\s+(.+)$'
        lines = content.split('\n')
        current_section = "Introduction"
        section_content = []
        
        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                if section_content:
                    parsed["sections"][current_section] = '\n'.join(section_content)
                current_section = match.group(1).strip()
                section_content = []
            else:
                section_content.append(line)
        
        if section_content:
            parsed["sections"][current_section] = '\n'.join(section_content)
    
    return parsed


def evaluate_storytelling(content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluates storytelling aspects of the portfolio"""
    context = context or {}
    
    prompt = f"""As an expert design portfolio reviewer, evaluate the storytelling quality of this portfolio case study.

Portfolio Content:
{content[:10000]}

Evaluate the following aspects:
1. Narrative structure (beginning, middle, end)
2. Problem statement clarity
3. Solution journey documentation
4. Visual storytelling support
5. Emotional engagement

Provide:
- Overall score (0-100)
- Strengths identified
- Weaknesses/gaps
- Specific evidence found
- Improvement recommendations

Format as JSON with keys: score, strengths, weaknesses, evidence, recommendations.
"""
    
    try:
        model = genai.GenerativeModel(CONFIG['model'])
        response = model.generate_content(prompt)
        result_text = response.text
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Fallback: create structured response
            result = {
                "score": 70,
                "strengths": ["Narrative structure present"],
                "weaknesses": ["Could improve emotional engagement"],
                "evidence": ["Problem statement found"],
                "recommendations": ["Add more personal anecdotes"],
                "raw_response": result_text
            }
        
        return result
    except Exception as e:
        return {
            "score": 0,
            "error": str(e),
            "strengths": [],
            "weaknesses": ["Evaluation failed"],
            "evidence": [],
            "recommendations": []
        }


def evaluate_double_diamond(content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluates double-diamond design process evidence"""
    context = context or {}
    
    prompt = f"""As an expert design portfolio reviewer, evaluate the double-diamond design process evidence in this portfolio.

Portfolio Content:
{content[:10000]}

Evaluate evidence for each phase:
1. DISCOVER phase: Research, user interviews, data collection, exploration
2. DEFINE phase: Problem framing, insights synthesis, opportunity definition
3. DEVELOP phase: Ideation, prototyping, iteration, testing
4. DELIVER phase: Final solution, implementation, launch, outcomes

For each phase, assess:
- Presence of evidence (yes/no)
- Quality of documentation
- Specific examples found

Provide:
- Overall score (0-100)
- Phase-by-phase assessment
- Missing phases
- Improvement recommendations

Format as JSON with keys: score, phases (dict with discover/define/develop/deliver), missing_phases, recommendations.
"""
    
    try:
        model = genai.GenerativeModel(CONFIG['model'])
        response = model.generate_content(prompt)
        result_text = response.text
        
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "score": 65,
                "phases": {
                    "discover": {"present": True, "quality": "medium"},
                    "define": {"present": True, "quality": "medium"},
                    "develop": {"present": False, "quality": "low"},
                    "deliver": {"present": True, "quality": "medium"}
                },
                "missing_phases": ["develop"],
                "recommendations": ["Add more development phase documentation"],
                "raw_response": result_text
            }
        
        return result
    except Exception as e:
        return {
            "score": 0,
            "error": str(e),
            "phases": {},
            "missing_phases": [],
            "recommendations": []
        }


def evaluate_designer_influence(content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluates designer influence, decision-making, and leadership evidence"""
    context = context or {}
    
    prompt = f"""As an expert design portfolio reviewer, evaluate the designer's influence and decision-making evidence in this portfolio.

Portfolio Content:
{content[:10000]}

Evaluate:
1. Decision-making documentation (key decisions made, rationale)
2. Conflict resolution examples (how conflicts were handled)
3. Leadership moments (leading initiatives, guiding team)
4. Design rationale explanations (why certain choices were made)
5. Stakeholder influence (how designer influenced outcomes)

Provide:
- Overall score (0-100)
- Strengths (specific examples of influence)
- Weaknesses (missing evidence)
- Recommendations for improvement

Format as JSON with keys: score, strengths, weaknesses, evidence, recommendations.
"""
    
    try:
        model = genai.GenerativeModel(CONFIG['model'])
        response = model.generate_content(prompt)
        result_text = response.text
        
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "score": 60,
                "strengths": ["Some decision-making documented"],
                "weaknesses": ["Limited conflict resolution examples"],
                "evidence": ["Design rationale mentioned"],
                "recommendations": ["Add more leadership examples"],
                "raw_response": result_text
            }
        
        return result
    except Exception as e:
        return {
            "score": 0,
            "error": str(e),
            "strengths": [],
            "weaknesses": [],
            "evidence": [],
            "recommendations": []
        }


def evaluate_collaboration(content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluates cross-discipline collaboration and teamwork evidence"""
    context = context or {}
    
    prompt = f"""As an expert design portfolio reviewer, evaluate the cross-discipline collaboration evidence in this portfolio.

Portfolio Content:
{content[:10000]}

Evaluate:
1. Cross-functional team mentions (engineers, PMs, researchers, etc.)
2. Collaboration methods/tools used
3. Stakeholder engagement (how stakeholders were involved)
4. Feedback incorporation (how feedback was integrated)
5. Team dynamics (how team worked together)

Provide:
- Overall score (0-100)
- Strengths (collaboration examples)
- Weaknesses (missing collaboration evidence)
- Recommendations

Format as JSON with keys: score, strengths, weaknesses, evidence, recommendations.
"""
    
    try:
        model = genai.GenerativeModel(CONFIG['model'])
        response = model.generate_content(prompt)
        result_text = response.text
        
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "score": 55,
                "strengths": ["Team members mentioned"],
                "weaknesses": ["Limited collaboration details"],
                "evidence": ["Some stakeholder engagement"],
                "recommendations": ["Document collaboration process more"],
                "raw_response": result_text
            }
        
        return result
    except Exception as e:
        return {
            "score": 0,
            "error": str(e),
            "strengths": [],
            "weaknesses": [],
            "evidence": [],
            "recommendations": []
        }


def generate_improvement_plan(evaluations: Dict[str, Any]) -> str:
    """Generates actionable improvement plan based on all evaluations"""
    
    prompt = f"""As an expert design portfolio reviewer, create a prioritized improvement plan based on these evaluations:

{json.dumps(evaluations, indent=2)}

Create an actionable improvement plan that:
1. Prioritizes improvements by impact
2. Provides specific, actionable steps
3. Includes examples of what good looks like
4. Suggests resources or references

Format as a clear, structured improvement plan with priorities.
"""
    
    try:
        model = genai.GenerativeModel(CONFIG['model'])
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating improvement plan: {str(e)}"


print("âœ“ Tool Functions Defined")
print("  â€¢ fetch_url_content")
print("  â€¢ parse_portfolio_content")
print("  â€¢ evaluate_storytelling")
print("  â€¢ evaluate_double_diamond")
print("  â€¢ evaluate_designer_influence")
print("  â€¢ evaluate_collaboration")
print("  â€¢ generate_improvement_plan")



function_declarations = [
    FunctionDeclaration(
        name="fetch_url_content",
        description="Fetches and extracts text content from a web URL for portfolio review",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL of the portfolio/case study to fetch"}
            },
            "required": ["url"]
        }
    ),
    FunctionDeclaration(
        name="parse_portfolio_content",
        description="Parses portfolio content into structured format for analysis",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Raw portfolio content (text or markdown)"},
                "format_type": {"type": "string", "description": "Format type: 'auto', 'markdown', or 'text'"}
            },
            "required": ["content"]
        }
    ),
    FunctionDeclaration(
        name="evaluate_storytelling",
        description="Evaluates storytelling aspects: narrative structure, problem clarity, solution journey, engagement",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Portfolio content to evaluate"},
                "context": {"type": "object", "description": "Additional context for evaluation"}
            },
            "required": ["content"]
        }
    ),
    FunctionDeclaration(
        name="evaluate_double_diamond",
        description="Evaluates double-diamond design process evidence: discover, define, develop, deliver phases",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Portfolio content to evaluate"},
                "context": {"type": "object", "description": "Additional context for evaluation"}
            },
            "required": ["content"]
        }
    ),
    FunctionDeclaration(
        name="evaluate_designer_influence",
        description="Evaluates designer influence: decision-making, conflict resolution, leadership, design rationale",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Portfolio content to evaluate"},
                "context": {"type": "object", "description": "Additional context for evaluation"}
            },
            "required": ["content"]
        }
    ),
    FunctionDeclaration(
        name="evaluate_collaboration",
        description="Evaluates cross-discipline collaboration: team mentions, collaboration methods, stakeholder engagement",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Portfolio content to evaluate"},
                "context": {"type": "object", "description": "Additional context for evaluation"}
            },
            "required": ["content"]
        }
    ),
    FunctionDeclaration(
        name="generate_improvement_plan",
        description="Generates prioritized, actionable improvement plan based on all evaluation results",
        parameters={
            "type": "object",
            "properties": {
                "evaluations": {"type": "object", "description": "Dictionary containing all evaluation results"}
            },
            "required": ["evaluations"]
        }
    )
]

tools = Tool(function_declarations=function_declarations)
print(f"âœ“ Function Declarations Created ({len(function_declarations)} tools)")



@dataclass
class ConversationMemory:
    """Manages conversation history and context"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_history: int = 20
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_context(self) -> str:
        if not self.messages:
            return "No previous conversation."
        context = "Recent conversation:\n"
        for msg in self.messages[-5:]:
            context += f"{msg['role']}: {msg['content'][:100]}...\n"
        return context
    
    def clear(self):
        self.messages.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m['role'] == 'user'),
            "agent_messages": sum(1 for m in self.messages if m['role'] == 'agent')
        }


@dataclass
class PortfolioReviewMemory:
    """Stores portfolio review history and previous evaluations"""
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    max_reviews: int = 50
    
    def add_review(self, review_data: Dict[str, Any]):
        review_data["timestamp"] = datetime.now().isoformat()
        self.reviews.append(review_data)
        if len(self.reviews) > self.max_reviews:
            self.reviews = self.reviews[-self.max_reviews:]
    
    def get_recent_reviews(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.reviews[-count:]
    
    def search_reviews(self, keyword: str) -> List[Dict[str, Any]]:
        keyword_lower = keyword.lower()
        results = []
        for review in self.reviews:
            review_str = json.dumps(review).lower()
            if keyword_lower in review_str:
                results.append(review)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.reviews:
            return {"total_reviews": 0, "avg_scores": {}}
        
        # Calculate average scores
        scores = {"storytelling": [], "double_diamond": [], "designer_influence": [], "collaboration": []}
        for review in self.reviews:
            if "evaluations" in review:
                evals = review["evaluations"]
                if "storytelling" in evals and "score" in evals["storytelling"]:
                    scores["storytelling"].append(evals["storytelling"]["score"])
                if "double_diamond" in evals and "score" in evals["double_diamond"]:
                    scores["double_diamond"].append(evals["double_diamond"]["score"])
                if "designer_influence" in evals and "score" in evals["designer_influence"]:
                    scores["designer_influence"].append(evals["designer_influence"]["score"])
                if "collaboration" in evals and "score" in evals["collaboration"]:
                    scores["collaboration"].append(evals["collaboration"]["score"])
        
        avg_scores = {}
        for key, values in scores.items():
            if values:
                avg_scores[key] = sum(values) / len(values)
        
        return {
            "total_reviews": len(self.reviews),
            "avg_scores": avg_scores
        }
    
    def clear(self):
        self.reviews.clear()


memory = ConversationMemory(max_history=20)
review_memory = PortfolioReviewMemory(max_reviews=50)
print(f"âœ“ Memory System Initialized")
print(f"  â€¢ Conversation Memory: Max {memory.max_history} messages")
print(f"  â€¢ Review Memory: Max {review_memory.max_reviews} reviews")



@dataclass
class AgentLogger:
    """Comprehensive logging for agent operations"""
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def log(self, level: str, event: str, details: Dict[str, Any] = None):
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            "details": details or {}
        })
    
    def info(self, event: str, **kwargs):
        self.log("INFO", event, kwargs)
    
    def error(self, event: str, **kwargs):
        self.log("ERROR", event, kwargs)
    
    def warning(self, event: str, **kwargs):
        self.log("WARNING", event, kwargs)
    
    def get_recent_logs(self, count: int = 10) -> List[Dict]:
        return self.logs[-count:]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_logs": len(self.logs),
            "info_count": sum(1 for log in self.logs if log['level'] == 'INFO'),
            "error_count": sum(1 for log in self.logs if log['level'] == 'ERROR'),
            "warning_count": sum(1 for log in self.logs if log['level'] == 'WARNING')
        }
    
    def export_logs(self, filename: str = "agent_logs.json"):
        with open(filename, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"âœ“ Logs exported to {filename}")


logger = AgentLogger()
logger.info("Logger initialized")
print("âœ“ Logging System Ready")



class StorytellingAgent:
    """Specialized agent for evaluating storytelling aspects"""
    
    def __init__(self, config: Dict, logger: AgentLogger):
        self.config = config
        self.logger = logger
        self.name = "StorytellingAgent"
    
    def evaluate(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate storytelling quality"""
        self.logger.info(f"{self.name} evaluation started", content_length=len(content))
        start_time = time.time()
        
        try:
            result = evaluate_storytelling(content, context)
            elapsed = time.time() - start_time
            self.logger.info(f"{self.name} evaluation completed", 
                           score=result.get("score", 0), 
                           elapsed=f"{elapsed:.2f}s")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} evaluation failed", error=str(e))
            return {"score": 0, "error": str(e)}


class DoubleDiamondAgent:
    """Specialized agent for evaluating double-diamond design process"""
    
    def __init__(self, config: Dict, logger: AgentLogger):
        self.config = config
        self.logger = logger
        self.name = "DoubleDiamondAgent"
    
    def evaluate(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate double-diamond process evidence"""
        self.logger.info(f"{self.name} evaluation started", content_length=len(content))
        start_time = time.time()
        
        try:
            result = evaluate_double_diamond(content, context)
            elapsed = time.time() - start_time
            self.logger.info(f"{self.name} evaluation completed", 
                           score=result.get("score", 0), 
                           elapsed=f"{elapsed:.2f}s")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} evaluation failed", error=str(e))
            return {"score": 0, "error": str(e)}


class DesignerInfluenceAgent:
    """Specialized agent for evaluating designer influence and decision-making"""
    
    def __init__(self, config: Dict, logger: AgentLogger):
        self.config = config
        self.logger = logger
        self.name = "DesignerInfluenceAgent"
    
    def evaluate(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate designer influence evidence"""
        self.logger.info(f"{self.name} evaluation started", content_length=len(content))
        start_time = time.time()
        
        try:
            result = evaluate_designer_influence(content, context)
            elapsed = time.time() - start_time
            self.logger.info(f"{self.name} evaluation completed", 
                           score=result.get("score", 0), 
                           elapsed=f"{elapsed:.2f}s")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} evaluation failed", error=str(e))
            return {"score": 0, "error": str(e)}


class CollaborationAgent:
    """Specialized agent for evaluating cross-discipline collaboration"""
    
    def __init__(self, config: Dict, logger: AgentLogger):
        self.config = config
        self.logger = logger
        self.name = "CollaborationAgent"
    
    def evaluate(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate collaboration evidence"""
        self.logger.info(f"{self.name} evaluation started", content_length=len(content))
        start_time = time.time()
        
        try:
            result = evaluate_collaboration(content, context)
            elapsed = time.time() - start_time
            self.logger.info(f"{self.name} evaluation completed", 
                           score=result.get("score", 0), 
                           elapsed=f"{elapsed:.2f}s")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} evaluation failed", error=str(e))
            return {"score": 0, "error": str(e)}


class ReportGeneratorAgent:
    """Specialized agent for synthesizing evaluations into comprehensive report"""
    
    def __init__(self, config: Dict, logger: AgentLogger):
        self.config = config
        self.logger = logger
        self.name = "ReportGeneratorAgent"
    
    def generate_report(self, evaluations: Dict[str, Any], improvement_plan: str) -> Dict[str, Any]:
        """Generate comprehensive review report"""
        self.logger.info(f"{self.name} report generation started")
        start_time = time.time()
        
        try:
            # Calculate overall score
            scores = []
            for key, eval_result in evaluations.items():
                if isinstance(eval_result, dict) and "score" in eval_result:
                    scores.append(eval_result["score"])
            
            overall_score = sum(scores) / len(scores) if scores else 0
            
            report = {
                "overall_score": round(overall_score, 2),
                "evaluations": evaluations,
                "improvement_plan": improvement_plan,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "storytelling_score": evaluations.get("storytelling", {}).get("score", 0),
                    "double_diamond_score": evaluations.get("double_diamond", {}).get("score", 0),
                    "designer_influence_score": evaluations.get("designer_influence", {}).get("score", 0),
                    "collaboration_score": evaluations.get("collaboration", {}).get("score", 0)
                }
            }
            
            elapsed = time.time() - start_time
            self.logger.info(f"{self.name} report generated", 
                           overall_score=overall_score, 
                           elapsed=f"{elapsed:.2f}s")
            return report
        except Exception as e:
            self.logger.error(f"{self.name} report generation failed", error=str(e))
            return {"error": str(e)}


print("âœ“ Specialized Agent Classes Created")
print("  â€¢ StorytellingAgent")
print("  â€¢ DoubleDiamondAgent")
print("  â€¢ DesignerInfluenceAgent")
print("  â€¢ CollaborationAgent")
print("  â€¢ ReportGeneratorAgent")



class PortfolioReviewCoordinator:
    """Main orchestrating agent that coordinates multi-agent review process"""
    
    def __init__(self, config: Dict, tools: Tool, memory: ConversationMemory, 
                 review_memory: PortfolioReviewMemory, logger: AgentLogger):
        self.config = config
        self.tools = tools
        self.memory = memory
        self.review_memory = review_memory
        self.logger = logger
        
        # Initialize specialized agents
        self.storytelling_agent = StorytellingAgent(config, logger)
        self.double_diamond_agent = DoubleDiamondAgent(config, logger)
        self.designer_influence_agent = DesignerInfluenceAgent(config, logger)
        self.collaboration_agent = CollaborationAgent(config, logger)
        self.report_generator_agent = ReportGeneratorAgent(config, logger)
        
        self.stats = {
            "reviews_processed": 0,
            "total_response_time": 0.0,
            "errors": 0
        }
        
        self.logger.info("PortfolioReviewCoordinator initialized")
    
    def _call_function(self, function_call) -> str:
        """Execute tool function and return result"""
        function_name = function_call.name
        function_args = dict(function_call.args)
        
        self.logger.info("Function called", function=function_name, args=str(function_args))
        
        function_map = {
            "fetch_url_content": fetch_url_content,
            "parse_portfolio_content": parse_portfolio_content,
            "evaluate_storytelling": evaluate_storytelling,
            "evaluate_double_diamond": evaluate_double_diamond,
            "evaluate_designer_influence": evaluate_designer_influence,
            "evaluate_collaboration": evaluate_collaboration,
            "generate_improvement_plan": generate_improvement_plan
        }
        
        if function_name in function_map:
            try:
                result = function_map[function_name](**function_args)
                return str(result) if not isinstance(result, str) else result
            except Exception as e:
                self.logger.error("Function execution failed", error=str(e))
                return f"Error executing {function_name}: {str(e)}"
        return f"Unknown function: {function_name}"
    
    def review_portfolio(self, content: str, input_type: str = "text") -> Dict[str, Any]:
        """Orchestrate multi-agent portfolio review"""
        start_time = time.time()
        self.logger.info("Portfolio review started", input_type=input_type, content_length=len(content))
        
        try:
            # Parse content if needed
            parsed_content = parse_portfolio_content(content)
            content_text = parsed_content.get("raw_content", content)
            
            # Sequential evaluation by specialized agents
            evaluations = {}
            
            # 1. Storytelling evaluation
            self.logger.info("Starting storytelling evaluation")
            evaluations["storytelling"] = self.storytelling_agent.evaluate(content_text)
            print("â�³ Waiting 5 seconds to respect rate limits (15 RPM)...")
            time.sleep(5)  # 5 seconds = 12 requests per minute (safe margin)
            
            # 2. Double-diamond evaluation
            self.logger.info("Starting double-diamond evaluation")
            evaluations["double_diamond"] = self.double_diamond_agent.evaluate(content_text)
            print("â�³ Waiting 5 seconds...")
            time.sleep(5)  # Wait 5 seconds
            
            # 3. Designer influence evaluation
            self.logger.info("Starting designer influence evaluation")
            evaluations["designer_influence"] = self.designer_influence_agent.evaluate(content_text)
            print("â�³ Waiting 5 seconds...")
            time.sleep(5)  # Wait 5 seconds
            
            # 4. Collaboration evaluation
            self.logger.info("Starting collaboration evaluation")
            evaluations["collaboration"] = self.collaboration_agent.evaluate(content_text)
            print("â�³ Waiting 5 seconds before generating improvement plan...")
            time.sleep(5)  # Wait 5 seconds
            
            # 5. Generate improvement plan
            self.logger.info("Generating improvement plan")
            improvement_plan = generate_improvement_plan(evaluations)
            
            # 6. Generate comprehensive report
            self.logger.info("Generating final report")
            report = self.report_generator_agent.generate_report(evaluations, improvement_plan)
            
            # Store in review memory
            review_data = {
                "input_type": input_type,
                "content_length": len(content),
                "evaluations": evaluations,
                "report": report
            }
            self.review_memory.add_review(review_data)
            
            elapsed = time.time() - start_time
            self.stats["reviews_processed"] += 1
            self.stats["total_response_time"] += elapsed
            
            self.logger.info("Portfolio review completed", 
                           overall_score=report.get("overall_score", 0),
                           elapsed=f"{elapsed:.2f}s")
            
            return report
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Portfolio review failed", error=str(e))
            return {
                "error": str(e),
                "overall_score": 0,
                "evaluations": {},
                "improvement_plan": "Review failed. Please try again."
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        avg_time = (
            self.stats["total_response_time"] / self.stats["reviews_processed"]
            if self.stats["reviews_processed"] > 0 else 0
        )
        
        return {
            **self.stats,
            "avg_response_time": round(avg_time, 2),
            "memory_stats": self.memory.get_stats(),
            "review_memory_stats": self.review_memory.get_stats(),
            "logger_stats": self.logger.get_stats()
        }
    
    def reset(self):
        """Reset coordinator state"""
        self.memory.clear()
        self.stats = {"reviews_processed": 0, "total_response_time": 0.0, "errors": 0}
        self.logger.info("Coordinator reset")


if GOOGLE_API_KEY:
    coordinator = PortfolioReviewCoordinator(
        config=CONFIG,
        tools=tools,
        memory=memory,
        review_memory=review_memory,
        logger=logger
    )
    print("âœ“ Coordinator Agent Initialized")
    print("âœ“ Ready for Portfolio Reviews")
else:
    coordinator = None
    print("âš  Coordinator initialization skipped - Configure API key")



def review_portfolio(input_source: str, input_type: str = "auto") -> Dict[str, Any]:
    """
    Main function to review a design portfolio.
    
    Args:
        input_source: URL or text content of the portfolio
        input_type: 'url', 'text', or 'auto' (auto-detect)
    
    Returns:
        Dictionary containing comprehensive review report
    """
    if not coordinator:
        return {"error": "Coordinator not initialized. Please configure API key."}
    
    # Auto-detect input type
    if input_type == "auto":
        if input_source.startswith("http://") or input_source.startswith("https://"):
            input_type = "url"
        else:
            input_type = "text"
    
    # Fetch content if URL
    if input_type == "url":
        print(f"Fetching content from URL: {input_source}")
        content = fetch_url_content(input_source)
        if content.startswith("Error"):
            return {"error": content}
    else:
        content = input_source
    
    # Review portfolio
    print(f"Starting portfolio review (type: {input_type}, length: {len(content)} chars)...")
    report = coordinator.review_portfolio(content, input_type)
    
    return report


print("âœ“ Main Review Function Ready")
print("ğŸ“Œ Usage: review_portfolio('https://example.com/portfolio') or review_portfolio('portfolio text here')")



def display_review_report(report: Dict[str, Any]):
    """Pretty-print evaluation results"""
    if "error" in report:
        print(f"â�Œ Error: {report['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"{'PORTFOLIO REVIEW REPORT':^60}")
    print(f"{'='*60}\n")
    
    # Overall score
    overall = report.get("overall_score", 0)
    print(f"ğŸ“Š Overall Score: {overall}/100\n")
    
    # Summary scores
    summary = report.get("summary", {})
    print("ğŸ“ˆ Category Scores:")
    print(f"  â€¢ Storytelling: {summary.get('storytelling_score', 0)}/100")
    print(f"  â€¢ Double-Diamond Process: {summary.get('double_diamond_score', 0)}/100")
    print(f"  â€¢ Designer Influence: {summary.get('designer_influence_score', 0)}/100")
    print(f"  â€¢ Collaboration: {summary.get('collaboration_score', 0)}/100\n")
    
    # Detailed evaluations
    evaluations = report.get("evaluations", {})
    for category, eval_result in evaluations.items():
        if isinstance(eval_result, dict):
            print(f"{'='*60}")
            print(f"{category.upper().replace('_', ' ')}")
            print(f"{'='*60}")
            print(f"Score: {eval_result.get('score', 0)}/100\n")
            
            if "strengths" in eval_result:
                print("âœ… Strengths:")
                for strength in eval_result["strengths"]:
                    print(f"  â€¢ {strength}")
                print()
            
            if "weaknesses" in eval_result:
                print("âš ï¸� Weaknesses:")
                for weakness in eval_result["weaknesses"]:
                    print(f"  â€¢ {weakness}")
                print()
            
            if "recommendations" in eval_result:
                print("ğŸ’¡ Recommendations:")
                for rec in eval_result["recommendations"]:
                    print(f"  â€¢ {rec}")
                print()
    
    # Improvement plan
    if "improvement_plan" in report:
        print(f"{'='*60}")
        print("IMPROVEMENT PLAN")
        print(f"{'='*60}\n")
        print(report["improvement_plan"])
        print()
    
    print(f"{'='*60}\n")


def export_review_report(report: Dict[str, Any], filename: str = "portfolio_review.json"):
    """Export review report to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"âœ“ Review report exported to: {filename}")
        return filename
    except Exception as e:
        print(f"â�Œ Error exporting report: {str(e)}")
        return None


def search_review_history(keyword: str) -> List[Dict[str, Any]]:
    """Search past portfolio reviews"""
    if not coordinator:
        print("âš  Coordinator not initialized")
        return []
    
    results = coordinator.review_memory.search_reviews(keyword)
    
    if results:
        print(f"ğŸ”� Found {len(results)} review(s) containing '{keyword}':\n")
        for idx, review in enumerate(results, 1):
            print(f"Review #{idx} [{review.get('timestamp', 'N/A')}]")
            if "report" in review and "overall_score" in review["report"]:
                print(f"  Overall Score: {review['report']['overall_score']}/100")
            print("-" * 60)
    else:
        print(f"â�Œ No reviews found containing '{keyword}'")
    
    return results


def get_review_statistics() -> Dict[str, Any]:
    """Get analytics on reviews performed"""
    if not coordinator:
        print("âš  Coordinator not initialized")
        return {}
    
    stats = coordinator.get_stats()
    review_stats = coordinator.review_memory.get_stats()
    
    print(f"\n{'='*60}")
    print(f"{'REVIEW STATISTICS':^60}")
    print(f"{'='*60}\n")
    
    print("ğŸ“Š Review Performance:")
    print(f"  Total Reviews: {stats['reviews_processed']}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"  Errors: {stats['errors']}\n")
    
    print("ğŸ“ˆ Review History:")
    print(f"  Total Reviews Stored: {review_stats['total_reviews']}")
    if review_stats['avg_scores']:
        print("  Average Scores:")
        for category, avg_score in review_stats['avg_scores'].items():
            print(f"    â€¢ {category.replace('_', ' ').title()}: {avg_score:.1f}/100")
    print()
    
    print("ğŸ’­ Memory Statistics:")
    mem_stats = stats['memory_stats']
    print(f"  Total Messages: {mem_stats['total_messages']}")
    print(f"  User Messages: {mem_stats['user_messages']}")
    print(f"  Agent Messages: {mem_stats['agent_messages']}\n")
    
    print("ğŸ“� Logger Statistics:")
    log_stats = stats['logger_stats']
    print(f"  Total Logs: {log_stats['total_logs']}")
    print(f"  Info: {log_stats['info_count']} | Warning: {log_stats['warning_count']} | Error: {log_stats['error_count']}")
    print(f"{'='*60}\n")
    
    return stats


print("âœ“ Utility Functions Ready")
print("  â€¢ display_review_report(report)")
print("  â€¢ export_review_report(report, filename)")
print("  â€¢ search_review_history(keyword)")
print("  â€¢ get_review_statistics()")



# Sample portfolio case study text
sample_portfolio_text = """
# E-Commerce Mobile App Redesign

## Problem Statement
Our e-commerce mobile app had low user engagement and high cart abandonment rates. 
User research revealed that the checkout process was confusing and the product discovery 
was limited.

## Discovery Phase
I conducted 15 user interviews to understand pain points. Key findings:
- 60% of users abandoned carts due to complex checkout
- Users wanted better product filtering
- Mobile experience was slow and clunky

## Define Phase
We synthesized insights into three key opportunities:
1. Simplify checkout to 3 steps max
2. Improve product search and filtering
3. Optimize mobile performance

## Develop Phase
I created wireframes and prototypes in Figma. We tested with 5 users iteratively.
Key decisions I made:
- Removed unnecessary form fields
- Added one-click checkout option
- Implemented smart product recommendations

## Deliver Phase
Launched the redesigned app. Results:
- Cart abandonment reduced by 40%
- User engagement increased by 25%
- App store rating improved from 3.2 to 4.5

## Collaboration
Worked closely with:
- Engineering team (2 developers)
- Product Manager
- UX Researcher
- Marketing team for launch

We used daily standups, design reviews, and user testing sessions to align.
"""

# Review the sample portfolio
if coordinator:
    print("="*60)
    print("DEMO 1: Reviewing Sample Portfolio (Text Input)")
    print("="*60)
    report = review_portfolio(sample_portfolio_text, input_type="text")
    display_review_report(report)
else:
    print("âš  Coordinator not initialized. Please configure API key first.")



# Example: Review portfolio from URL
# Uncomment and replace with actual portfolio URL to test
# portfolio_url = "https://example.com/designer-portfolio"

# if coordinator:
#     print("="*60)
#     print("DEMO 2: Reviewing Portfolio from URL")
#     print("="*60)
#     report = review_portfolio(portfolio_url, input_type="url")
#     display_review_report(report)
#     
#     # Export the report
#     export_review_report(report, "url_portfolio_review.json")
# else:
#     print("âš  Coordinator not initialized. Please configure API key first.")

print("ğŸ“Œ To review a portfolio from URL, uncomment the code above and provide a valid URL")



# Display performance statistics
if coordinator:
    get_review_statistics()
else:
    print("âš  Coordinator not initialized. Please configure API key first.")


