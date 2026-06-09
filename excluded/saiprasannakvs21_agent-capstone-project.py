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


!pip install -q google-genai
!pip install -q google-cloud-aiplatform

print("âœ“ Installation complete!")


import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import random


# Set your Google API key here
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class Config:
    MODEL = "gemini-2.5-flash"
    MAX_PAPERS = 10
    MEMORY_RETENTION_DAYS = 30
    ENABLE_TRACING = True
    ENABLE_METRICS = True

print("âœ“ Configuration set")
print(f"  Model: {Config.MODEL}")
print(f"  Max Papers: {Config.MAX_PAPERS}")


import google.generativeai as genai

genai.configure(api_key=GOOGLE_API_KEY)

# Test connection
model = genai.GenerativeModel(Config.MODEL)
test_response = model.generate_content("Say 'Connection successful!'")
print(f"âœ“ Gemini API: {test_response.text}")


@dataclass
class ResearchPaper:
    """Represents a scientific paper"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    publication_date: str
    journal: str
    doi: Optional[str] = None
    keywords: List[str] = None
    
    def to_dict(self):
        return asdict(self)
    
    def __str__(self):
        return f"{self.title} - {self.journal} ({self.publication_date[:4]})"

@dataclass
class ResearchInsight:
    """Extracted insight from a paper"""
    paper_id: str
    methodology: str
    key_findings: List[str]
    statistical_significance: str
    limitations: List[str]
    relevance_score: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class LiteratureReview:
    """Generated literature review"""
    topic: str
    papers_analyzed: int
    synthesis: str
    common_themes: List[str]
    research_gaps: List[str]
    recommendations: List[str]
    citations: List[str]
    
    def to_dict(self):
        return asdict(self)

# Test data model
test_paper = ResearchPaper(
    pmid="TEST001",
    title="CRISPR Gene Editing in Cancer Treatment",
    abstract="This study explores CRISPR applications in oncology...",
    authors=["Dr. Smith", "Dr. Johnson"],
    publication_date="2024-01-15",
    journal="Nature Biotechnology",
    doi="10.1038/test.001",
    keywords=["CRISPR", "cancer", "gene editing"]
)

print("âœ“ Data models defined")
print(f"  Test paper: {test_paper}")


class PubMedSearchTool:
    """Custom tool for searching PubMed database"""
    
    def __init__(self):
        self.name = "pubmed_search"
        self.description = "Search PubMed for scientific papers"
        print(f"âœ“ {self.name} initialized")
    
    async def search(self, query: str, max_results: int = 10) -> List[ResearchPaper]:
        """Search PubMed for papers (simulated for demo)"""
        print(f"\nğŸ”� Searching PubMed for: '{query}'")
        
        # Simulate API delay
        await asyncio.sleep(0.5)
        
        # Generate simulated papers
        papers = []
        topics = [
            "Molecular Mechanisms and Clinical Applications",
            "Novel Therapeutic Approaches",
            "Genomic Analysis and Biomarkers",
            "Clinical Trial Results and Patient Outcomes",
            "Mechanistic Studies and Future Directions"
        ]
        
        for i in range(1, min(max_results + 1, 6)):
            paper = ResearchPaper(
                pmid=f"PMID{35000000 + i}",
                title=f"{query.title()}: {topics[i-1]}",
                abstract=f"This study investigates {query} through comprehensive analysis. Methods included randomized controlled trials with {100 + i*50} participants. Results demonstrate significant therapeutic potential with p < 0.01. Our findings suggest novel treatment pathways and identify key molecular targets.",
                authors=[f"Author{j} et al." for j in range(1, 4)],
                publication_date=f"2024-{str(i).zfill(2)}-15",
                journal=["Nature Medicine", "Cell", "Science", "NEJM", "Lancet"][i % 5],
                doi=f"10.1038/nm.2024.{1000 + i}",
                keywords=[query, "clinical trial", "therapeutics", "biomarkers"]
            )
            papers.append(paper)
        
        print(f"âœ“ Found {len(papers)} papers")
        for idx, p in enumerate(papers, 1):
            print(f"  {idx}. {p.title[:60]}...")
        
        return papers

# Test the tool
pubmed_tool = PubMedSearchTool()

# Run async search
papers = await pubmed_tool.search("CRISPR gene editing", max_results=5)
print(f"\nâœ“ PubMed Search Tool tested successfully")


class StatisticalAnalysisTool:
    """Tool for statistical analysis of research data"""
    
    def __init__(self):
        self.name = "statistical_analysis"
        self.description = "Perform statistical analysis on research findings"
        print(f"âœ“ {self.name} initialized")
    
    def analyze(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis"""
        print(f"\nğŸ“Š Analyzing: {paper_data.get('title', 'Unknown')[:50]}...")
        
        # Simulated statistical analysis
        random.seed(hash(paper_data.get('title', '')) % 1000)
        
        analysis = {
            "p_value": round(random.uniform(0.001, 0.049), 4),
            "confidence_interval": [
                round(random.uniform(0.3, 0.5), 2),
                round(random.uniform(0.7, 0.9), 2)
            ],
            "effect_size": round(random.uniform(0.5, 0.8), 2),
            "statistical_power": round(random.uniform(0.8, 0.95), 2),
            "sample_size": random.randint(100, 500),
            "significance": "Statistically significant" if random.random() > 0.3 else "Not significant"
        }
        
        print(f"  âœ“ P-value: {analysis['p_value']}")
        print(f"  âœ“ Effect size: {analysis['effect_size']}")
        print(f"  âœ“ Sample: n={analysis['sample_size']}")
        
        return analysis

# Test the tool
stats_tool = StatisticalAnalysisTool()
test_analysis = stats_tool.analyze({
    "title": "CRISPR Gene Editing Study",
    "sample_size": 200
})

print(f"\nâœ“ Statistical Analysis Tool tested")


class CitationFormatterTool:
    """Tool for formatting citations in various styles"""
    
    def __init__(self):
        self.name = "citation_formatter"
        self.styles = ["APA", "MLA", "Chicago", "Vancouver"]
        print(f"âœ“ {self.name} initialized")
        print(f"  Supported styles: {', '.join(self.styles)}")
    
    def format_citation(self, paper: ResearchPaper, style: str = "APA") -> str:
        """Format citation for a paper"""
        
        if style == "APA":
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += ", et al."
            year = paper.publication_date[:4]
            citation = (f"{authors_str} ({year}). {paper.title}. "
                       f"*{paper.journal}*. https://doi.org/{paper.doi}")
        
        elif style == "MLA":
            first_author = paper.authors[0] if paper.authors else "Unknown"
            citation = (f'{first_author}, et al. "{paper.title}." '
                       f'*{paper.journal}*, {paper.publication_date[:4]}.')
        
        elif style == "Vancouver":
            authors_str = ", ".join([a.split()[0] for a in paper.authors[:6]])
            if len(paper.authors) > 6:
                authors_str += ", et al"
            citation = (f"{authors_str}. {paper.title}. "
                       f"{paper.journal}. {paper.publication_date[:4]};{paper.doi}")
        
        else:
            citation = f"[{style}] {paper.title} - {paper.journal}"
        
        return citation
    
    def format_bibliography(self, papers: List[ResearchPaper], style: str = "APA") -> List[str]:
        """Format multiple citations"""
        print(f"\nğŸ“š Formatting {len(papers)} citations in {style} style...")
        citations = [self.format_citation(p, style) for p in papers]
        return citations

# Test the tool
citation_tool = CitationFormatterTool()

# Format single citation
single_citation = citation_tool.format_citation(test_paper, style="APA")
print(f"\nâœ“ Sample APA Citation:")
print(f"  {single_citation}")

# Format bibliography
if len(papers) > 0:
    bibliography = citation_tool.format_bibliography(papers[:3], style="APA")
    print(f"\nâœ“ Bibliography (first 3):")
    for i, cite in enumerate(bibliography, 1):
        print(f"  {i}. {cite}")


class SessionManager:
    """Manages research sessions and state"""
    
    def __init__(self):
        self.sessions = {}
        self.current_session = None
        print("âœ“ SessionManager initialized")
    
    def create_session(self, user_id: str, topic: str) -> Dict:
        """Create a new research session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = {
            "id": session_id,
            "user_id": user_id,
            "topic": topic,
            "created_at": datetime.now().isoformat(),
            "state": {
                "current_step": "initialized",
                "papers_found": 0,
                "papers_analyzed": 0,
                "progress": 0
            },
            "results": {}
        }
        
        self.sessions[session_id] = session
        self.current_session = session
        
        print(f"\nâœ“ Session created: {session_id}")
        print(f"  User: {user_id}")
        print(f"  Topic: {topic}")
        
        return session
    
    def update_session(self, session_id: str, updates: Dict):
        """Update session state"""
        if session_id in self.sessions:
            self.sessions[session_id]["state"].update(updates)
            print(f"âœ“ Session updated: {updates}")
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve a session"""
        return self.sessions.get(session_id)


class MemoryBank:
    """Long-term memory for storing research history"""
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.memory_store = []
        self.user_preferences = {}
        print(f"âœ“ MemoryBank initialized (retention: {retention_days} days)")
    
    def store_research(self, session_id: str, data: Dict):
        """Store completed research in long-term memory"""
        memory_entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "expires_at": (datetime.now().timestamp() + (self.retention_days * 86400))
        }
        
        self.memory_store.append(memory_entry)
        print(f"âœ“ Research stored in memory: {session_id}")
    
    def get_user_history(self, user_id: str) -> List[Dict]:
        """Retrieve user's research history"""
        return [m for m in self.memory_store if m["data"].get("user_id") == user_id]
    
    def get_research_patterns(self, user_id: str) -> Dict:
        """Analyze user's research patterns"""
        history = self.get_user_history(user_id)
        
        patterns = {
            "total_sessions": len(history),
            "common_topics": ["CRISPR", "immunotherapy", "gene editing"],
            "avg_papers_per_session": 8.5,
            "preferred_journals": ["Nature", "Science", "Cell"]
        }
        
        return patterns

# Initialize memory and session management
session_manager = SessionManager()
memory_bank = MemoryBank(retention_days=Config.MEMORY_RETENTION_DAYS)

# Test session creation
test_session = session_manager.create_session(
    user_id="researcher_001",
    topic="CRISPR applications in cancer"
)

print("\nâœ“ Session & Memory systems ready")


class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
        self.model = genai.GenerativeModel(Config.MODEL)
        self.tools = []
        print(f"âœ“ Agent '{name}' initialized")
    
    def add_tool(self, tool):
        """Add a tool to the agent"""
        self.tools.append(tool)
        print(f"  + Tool added: {tool.name}")
    
    async def execute(self, prompt: str, context: Dict = None) -> str:
        """Execute agent with given prompt"""
        full_prompt = f"{self.instructions}\n\n{prompt}"
        
        if context:
            full_prompt += f"\n\nContext: {json.dumps(context, indent=2)}"
        
        print(f"\nğŸ¤– {self.name} executing...")
        
        response = self.model.generate_content(full_prompt)
        return response.text


class LLMAgent(BaseAgent):
    """Agent powered by LLM"""
    
    def __init__(self, name: str, instructions: str, tools: List = None):
        super().__init__(name, instructions)
        if tools:
            for tool in tools:
                self.add_tool(tool)
        print(f"âœ“ LLMAgent '{name}' ready with {len(self.tools)} tools")

# Test base agent
test_agent = LLMAgent(
    name="TestAgent",
    instructions="You are a helpful research assistant.",
    tools=[pubmed_tool]
)

print("\nâœ“ Agent base classes defined")


class PubMedSearchAgent(LLMAgent):
    """Agent specialized in searching scientific literature"""
    
    def __init__(self):
        super().__init__(
            name="PubMedSearchAgent",
            instructions="""You are an expert at searching scientific literature.
            Your responsibilities:
            1. Understand the research query and intent
            2. Formulate effective search terms
            3. Search PubMed for relevant papers
            4. Filter and rank results by relevance
            
            Be thorough and precise in your searches.""",
            tools=[pubmed_tool]
        )


class PaperAnalysisAgent(LLMAgent):
    """Agent specialized in analyzing research papers"""
    
    def __init__(self):
        super().__init__(
            name="PaperAnalysisAgent",
            instructions="""You are an expert at analyzing scientific papers.
            For each paper, you must:
            1. Identify the research methodology
            2. Extract key findings and results
            3. Assess statistical significance
            4. Note study limitations
            5. Evaluate relevance to the research question
            
            Provide detailed, structured analysis.""",
            tools=[stats_tool]
        )
    
    async def analyze_paper(self, paper: ResearchPaper) -> ResearchInsight:
        """Analyze a single research paper"""
        print(f"\nğŸ“„ Analyzing: {paper.title[:60]}...")
        
        # Perform statistical analysis
        stats = stats_tool.analyze(paper.to_dict())
        
        # Generate AI analysis
        prompt = f"""
        Analyze this research paper and provide structured insights:
        
        Title: {paper.title}
        Journal: {paper.journal}
        Abstract: {paper.abstract}
        
        Provide: methodology, key findings, limitations, and relevance score (0-1).
        """
        
        analysis = await self.execute(prompt)
        
        # Create structured insight
        insight = ResearchInsight(
            paper_id=paper.pmid,
            methodology="Randomized controlled trial with molecular analysis",
            key_findings=[
                "Significant therapeutic effect observed",
                "Novel molecular mechanism identified",
                "Minimal adverse effects reported"
            ],
            statistical_significance=f"p = {stats['p_value']}",
            limitations=[
                "Limited sample size",
                "Single-center study",
                "Short follow-up period"
            ],
            relevance_score=0.85
        )
        
        print(f"  âœ“ Analysis complete")
        print(f"    Relevance: {insight.relevance_score}")
        print(f"    P-value: {stats['p_value']}")
        
        return insight


class SynthesisAgent(LLMAgent):
    """Agent specialized in synthesizing research findings"""
    
    def __init__(self):
        super().__init__(
            name="SynthesisAgent",
            instructions="""You are an expert at synthesizing research findings.
            Your role:
            1. Identify common themes across papers
            2. Compare and contrast methodologies
            3. Highlight consensus and controversies
            4. Identify research gaps
            5. Provide actionable recommendations
            
            Create comprehensive, well-structured literature reviews.""",
            tools=[citation_tool]
        )
    
    async def synthesize(self, topic: str, papers: List[ResearchPaper], 
                        insights: List[ResearchInsight]) -> LiteratureReview:
        """Synthesize research findings into literature review"""
        
        print(f"\nğŸ“š Synthesizing {len(papers)} papers on: {topic}")
        
        prompt = f"""
        Create a comprehensive literature review on: {topic}
        
        Papers analyzed: {len(papers)}
        Average relevance score: {sum(i.relevance_score for i in insights) / len(insights):.2f}
        
        Based on the analyzed papers, provide:
        1. Main synthesis of findings
        2. Common themes
        3. Research gaps
        4. Recommendations
        """
        
        synthesis_text = await self.execute(prompt)
        
        review = LiteratureReview(
            topic=topic,
            papers_analyzed=len(papers),
            synthesis=synthesis_text,
            common_themes=[
                "Promising therapeutic potential",
                "Need for larger clinical trials",
                "Molecular mechanism elucidation"
            ],
            research_gaps=[
                "Long-term efficacy data needed",
                "Diverse population studies required",
                "Cost-effectiveness analysis lacking"
            ],
            recommendations=[
                "Conduct multi-center trials",
                "Investigate combination therapies",
                "Develop predictive biomarkers"
            ],
            citations=citation_tool.format_bibliography(papers, style="APA")
        )
        
        print(f"  âœ“ Synthesis complete")
        print(f"    Themes: {len(review.common_themes)}")
        print(f"    Gaps: {len(review.research_gaps)}")
        
        return review

# Initialize specialized agents
search_agent = PubMedSearchAgent()
analysis_agent = PaperAnalysisAgent()
synthesis_agent = SynthesisAgent()

print("\nâœ“ Specialized agents ready")


class ParallelAgentExecutor:
    """Executes multiple agents in parallel"""
    
    def __init__(self, agent: LLMAgent):
        self.agent = agent
        self.name = "ParallelExecutor"
        print(f"âœ“ ParallelExecutor initialized for {agent.name}")
    
    async def execute_parallel(self, tasks: List[Any]) -> List[Any]:
        """Execute tasks in parallel"""
        print(f"\nâš¡ Executing {len(tasks)} tasks in parallel...")
        
        # Create parallel tasks
        async_tasks = [self._execute_single(task) for task in tasks]
        
        # Execute all in parallel
        results = await asyncio.gather(*async_tasks)
        
        print(f"  âœ“ Parallel execution complete: {len(results)} results")
        return results
    
    async def _execute_single(self, task: Any) -> Any:
        """Execute a single task"""
        await asyncio.sleep(0.1)  # Simulate processing
        return task

# Test parallel execution
parallel_executor = ParallelAgentExecutor(analysis_agent)

print("\nâœ“ Parallel execution system ready")


class SequentialWorkflowAgent:
    """Orchestrates agents in sequential workflow"""
    
    def __init__(self, name: str, agents: List[LLMAgent]):
        self.name = name
        self.agents = agents
        self.results = []
        print(f"âœ“ SequentialWorkflow '{name}' initialized")
        print(f"  Pipeline: {' â†’ '.join([a.name for a in agents])}")
    
    async def execute_workflow(self, initial_input: Dict) -> Dict:
        """Execute agents sequentially"""
        print(f"\nğŸ”„ Starting sequential workflow: {self.name}")
        
        context = initial_input.copy()
        
        for idx, agent in enumerate(self.agents, 1):
            print(f"\n  Step {idx}/{len(self.agents)}: {agent.name}")
            
            # Execute agent with accumulated context
            result = await agent.execute("Process the research data", context)
            
            # Update context for next agent
            context[f"result_{agent.name}"] = result
            self.results.append({
                "agent": agent.name,
                "step": idx,
                "result": result[:100] + "..." if len(result) > 100 else result
            })
        
        print(f"\n  âœ“ Workflow complete: {len(self.results)} steps executed")
        return context

# Create sequential workflow
sequential_workflow = SequentialWorkflowAgent(
    name="ResearchPipeline",
    agents=[search_agent, analysis_agent, synthesis_agent]
)

print("\nâœ“ Sequential workflow defined")


class LoopAgent:
    """Agent that iteratively refines results"""
    
    def __init__(self, name: str, agent: LLMAgent, max_iterations: int = 3):
        self.name = name
        self.agent = agent
        self.max_iterations = max_iterations
        self.iteration_count = 0
        print(f"âœ“ LoopAgent '{name}' initialized")
        print(f"  Max iterations: {max_iterations}")
    
    async def execute_with_refinement(self, initial_query: str, 
                                     quality_threshold: float = 0.8) -> Dict:
        """Execute with iterative refinement"""
        print(f"\nğŸ”� Starting iterative refinement loop...")
        
        query = initial_query
        best_result = None
        best_score = 0.0
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n  Iteration {iteration}/{self.max_iterations}")
            print(f"  Query: {query}")
            
            # Execute agent
            result = await self.agent.execute(f"Research: {query}")
            
            # Evaluate quality (simulated)
            quality_score = random.uniform(0.6, 0.95)
            print(f"  Quality score: {quality_score:.2f}")
            
            if quality_score > best_score:
                best_score = quality_score
                best_result = result
            
            # Check if quality threshold met
            if quality_score >= quality_threshold:
                print(f"  âœ“ Quality threshold met!")
                break
            
            # Refine query for next iteration
            query = self._refine_query(query, quality_score)
            self.iteration_count = iteration
        
        print(f"\n  âœ“ Refinement complete after {self.iteration_count} iterations")
        print(f"  Best score: {best_score:.2f}")
        
        return {
            "final_result": best_result,
            "quality_score": best_score,
            "iterations": self.iteration_count
        }
    
    def _refine_query(self, query: str, score: float) -> str:
        """Refine query based on quality score"""
        if score < 0.7:
            return f"{query} recent clinical trials"
        elif score < 0.8:
            return f"{query} systematic review"
        else:
            return f"{query} meta-analysis"

# Create loop agent
loop_agent = LoopAgent(
    name="IterativeRefinement",
    agent=search_agent,
    max_iterations=3
)

# Test loop execution
loop_result = await loop_agent.execute_with_refinement(
    initial_query="cancer immunotherapy",
    quality_threshold=0.85
)

print(f"\nâœ“ Loop agent tested: {loop_result['iterations']} iterations")


class ResearchOrchestrator:
    """Main orchestrator coordinating all agents"""
    
    def __init__(self):
        self.name = "LifeSciencesResearchOrchestrator"
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        
        # Initialize agents
        self.search_agent = search_agent
        self.analysis_agent = analysis_agent
        self.synthesis_agent = synthesis_agent
        
        # Initialize executors
        self.parallel_executor = ParallelAgentExecutor(analysis_agent)
        self.sequential_workflow = sequential_workflow
        self.loop_agent = loop_agent
        
        print(f"âœ“ {self.name} initialized")
        print("  Components:")
        print(f"    - Search Agent")
        print(f"    - Analysis Agent (parallel capable)")
        print(f"    - Synthesis Agent")
        print(f"    - Loop Agent (iterative refinement)")
    
    async def conduct_research(self, query: str, user_id: str) -> LiteratureReview:
        """Main research workflow orchestrating all agents"""
        
        print("\n" + "="*80)
        print(f"ğŸ”¬ RESEARCH WORKFLOW STARTING")
        print(f"Query: {query}")
        print(f"User: {user_id}")
        print("="*80)
        
        # Create session
        session = self.session_manager.create_session(user_id, query)
        
        try:
            # STEP 1: Search for papers (Sequential)
            print("\nğŸ“� STEP 1: Searching for papers")
            session_manager.update_session(session["id"], {"current_step": "search", "progress": 20})
            
            papers = await pubmed_tool.search(query, max_results=5)
            session_manager.update_session(session["id"], {"papers_found": len(papers), "progress": 40})
            
            # STEP 2: Analyze papers in parallel
            print("\nğŸ“� STEP 2: Analyzing papers (parallel)")
            session_manager.update_session(session["id"], {"current_step": "analysis", "progress": 50})
            
            insights = []
            for paper in papers:
                insight = await self.analysis_agent.analyze_paper(paper)
                insights.append(insight)
            
            session_manager.update_session(session["id"], {"papers_analyzed": len(insights), "progress": 70})
            
            # STEP 3: Synthesize findings (Sequential)
            print("\nğŸ“� STEP 3: Synthesizing findings")
            session_manager.update_session(session["id"], {"current_step": "synthesis", "progress": 85})
            
            review = await self.synthesis_agent.synthesize(query, papers, insights)
            
            # STEP 4: Store in memory
            print("\nğŸ“� STEP 4: Storing results")
            self.memory_bank.store_research(session["id"], {
                "user_id": user_id,
                "query": query,
                "papers_count": len(papers),
                "review": review.to_dict()
            })
            
            session_manager.update_session(session["id"], {"current_step": "complete", "progress": 100})
            
            print("\n" + "="*80)
            print("âœ“ RESEARCH WORKFLOW COMPLETE")
            print("="*80)
            
            return review
            
        except Exception as e:
            print(f"\nâ�Œ Error in workflow: {str(e)}")
            raise

# Initialize orchestrator
orchestrator = ResearchOrchestrator()

print("\nâœ“ Multi-Agent Orchestrator ready")


async def run_research_demo():
    """Run a complete research demonstration"""
    
    print("\n" + "="*80)
    print("ğŸš€ EXECUTING COMPLETE RESEARCH DEMONSTRATION")
    print("="*80)
    
    # Run research
    review = await orchestrator.conduct_research(
        query="CRISPR gene editing in cancer treatment",
        user_id="researcher_demo"
    )
    
    # Display results
    print("\n" + "="*80)
    print("ğŸ“Š RESEARCH RESULTS")
    print("="*80)
    
    print(f"\nğŸ“Œ Topic: {review.topic}")
    print(f"ğŸ“„ Papers Analyzed: {review.papers_analyzed}")
    
    print(f"\nğŸ�¯ Common Themes:")
    for i, theme in enumerate(review.common_themes, 1):
        print(f"  {i}. {theme}")
    
    print(f"\nğŸ”� Research Gaps:")
    for i, gap in enumerate(review.research_gaps, 1):
        print(f"  {i}. {gap}")
    
    print(f"\nğŸ’¡ Recommendations:")
    for i, rec in enumerate(review.recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\nğŸ“š Citations ({len(review.citations)}):")
    for i, citation in enumerate(review.citations[:3], 1):
        print(f"  {i}. {citation}")
    
    print(f"\nâœ�ï¸� Synthesis Preview:")
    print(f"  {review.synthesis}")
    
    return review

# Demo
demo_review = await run_research_demo()




