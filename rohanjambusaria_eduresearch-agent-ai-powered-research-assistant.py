# ============================================================================
# OFFLINE EDURESEARCH AGENT 
# Handles any arxiv dataset format with automatic column detection & cleaning
# ============================================================================

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import logging
import textwrap
import warnings
warnings.filterwarnings('ignore')

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EduResearchOffline")

print("="*80)
print("ğŸ�“ OFFLINE EDURESEARCH AGENT - INITIALIZING")
print("="*80)

# ---------- Step 1: Find & Load Dataset ----------
print("\nğŸ“� Scanning /kaggle/input for datasets...\n")

csv_files = []
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith('.csv'):
            full_path = os.path.join(dirname, filename)
            csv_files.append(full_path)
            print(f"Found: {full_path}")

if not csv_files:
    raise FileNotFoundError("No CSV files found in /kaggle/input. Please attach a dataset.")

# Use the first CSV (or modify index if multiple exist)
DATA_PATH = csv_files[0]
print(f"\nâœ… Using dataset: {DATA_PATH}\n")

# Load raw data
raw_df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(raw_df)} rows")
print(f"Original columns: {list(raw_df.columns)}\n")

# ---------- Step 2: Smart Column Detection & Cleaning ----------
print("ğŸ”� Detecting title and abstract columns...\n")

# Possible column name variations
TITLE_VARIATIONS = [
    'title', 'Title', 'TITLE', 'paper_title', 'titles', 
    'paper', 'Paper', 'name', 'Name', 'article_title'
]
ABSTRACT_VARIATIONS = [
    'abstract', 'Abstract', 'ABSTRACT', 'paper_abstract', 
    'summary', 'Summary', 'description', 'Description', 'text'
]

title_col = None
abstract_col = None

# Find title column
for col in raw_df.columns:
    if col in TITLE_VARIATIONS:
        title_col = col
        break
    # Partial match
    if any(var.lower() in col.lower() for var in ['title', 'paper', 'name']):
        title_col = col
        break

# Find abstract column
for col in raw_df.columns:
    if col in ABSTRACT_VARIATIONS:
        abstract_col = col
        break
    # Partial match
    if any(var.lower() in col.lower() for var in ['abstract', 'summary', 'description']):
        abstract_col = col
        break

print(f"Detected title column: {title_col}")
print(f"Detected abstract column: {abstract_col}\n")

if not title_col or not abstract_col:
    print("âš ï¸� Could not auto-detect columns. Available columns:")
    for i, col in enumerate(raw_df.columns):
        print(f"  [{i}] {col}")
    print("\nğŸ”§ Manual override: Set TITLE_COL_INDEX and ABSTRACT_COL_INDEX")
    
    # MANUAL OVERRIDE (uncomment and set if needed)
    # TITLE_COL_INDEX = 0  # index from list above
    # ABSTRACT_COL_INDEX = 1
    # title_col = raw_df.columns[TITLE_COL_INDEX]
    # abstract_col = raw_df.columns[ABSTRACT_COL_INDEX]
    
    if not title_col or not abstract_col:
        raise ValueError("Could not detect title/abstract columns. Use manual override above.")

# ---------- Step 3: Create Clean Dataset ----------
print("ğŸ§¹ Cleaning and normalizing data...\n")

papers_df = pd.DataFrame()
papers_df['paper_title'] = raw_df[title_col].astype(str).str.strip()
papers_df['paper_abstract'] = raw_df[abstract_col].astype(str).str.strip()

# Add date if available
date_cols = [c for c in raw_df.columns if 'date' in c.lower() or 'year' in c.lower()]
if date_cols:
    papers_df['paper_date'] = raw_df[date_cols[0]].astype(str)
else:
    papers_df['paper_date'] = 'NA'

# Add categories if available
cat_cols = [c for c in raw_df.columns if 'categor' in c.lower() or 'subject' in c.lower()]
if cat_cols:
    papers_df['paper_category'] = raw_df[cat_cols[0]].astype(str)
else:
    papers_df['paper_category'] = 'General'

# Remove rows with missing critical data
papers_df = papers_df[
    (papers_df['paper_title'] != 'nan') & 
    (papers_df['paper_title'] != '') &
    (papers_df['paper_abstract'] != 'nan') & 
    (papers_df['paper_abstract'] != '') &
    (papers_df['paper_abstract'].str.len() > 50)  # meaningful abstracts only
].reset_index(drop=True)

print(f"âœ… Clean dataset ready: {len(papers_df)} valid papers")
print(f"Columns: {list(papers_df.columns)}\n")

# Preview
display(papers_df.head())
print("\n" + "="*80)



@dataclass
class Paper:
    title: str
    abstract: str
    category: str = "General"
    date: str = "NA"
    paper_id: str = ""

@dataclass
class Summary:
    paper_id: str
    key_findings: List[str]
    methodology: str
    limitations: List[str]
    research_gap: str
    relevance_score: float

@dataclass
class Citation:
    paper_id: str
    apa: str

@dataclass
class ResearchSession:
    session_id: str
    query: str
    papers: List[Paper]
    summaries: List[Summary]
    citations: List[Citation]
    literature_review: str
    research_plan: Dict
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

print("âœ… Data models defined")



class OfflineResearchTools:
    """Offline tools with robust error handling."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        logger.info(f"Tools initialized with {len(self.df)} papers")

    def search(self, query: str, max_results: int = 10) -> Tuple[pd.DataFrame, List[float]]:
        """Enhanced keyword search with scoring."""
        logger.info(f"Searching for: '{query}'")
        
        q_tokens = [w.lower() for w in query.split() if len(w) > 2]
        
        # Create searchable text
        text = (
            self.df['paper_title'].fillna("") + " " +
            self.df['paper_abstract'].fillna("") + " " +
            self.df['paper_category'].fillna("")
        ).str.lower()
        
        # Scoring function
        def score_fn(t: str) -> float:
            score = 0
            for tok in q_tokens:
                # Title match = higher weight
                if tok in self.df.loc[text[text == t].index[0], 'paper_title'].lower():
                    score += 3
                # Abstract match
                elif tok in t:
                    score += 1
            return score
        
        scores = text.apply(lambda t: sum(
            3 if tok in self.df.loc[i, 'paper_title'].lower() else (1 if tok in t else 0)
            for tok in q_tokens
        ) for i, t in enumerate(text))
        
        # Get top results
        top_idx = scores.nlargest(max_results).index
        result_df = self.df.loc[top_idx].copy()
        result_df['relevance_score'] = scores.loc[top_idx].values
        
        logger.info(f"Found {len(result_df)} papers")
        return result_df, scores.loc[top_idx].tolist()

    def format_citation(self, row: pd.Series, idx: int) -> str:
        """Generate APA-style citation."""
        title = row.get('paper_title', 'Untitled')[:200]
        year = str(row.get('paper_date', 'n.d.'))[:4]
        
        return f"[{idx}] {title}. ({year}). arXiv preprint."

print("âœ… Research tools defined")



class OfflineSearchAgent:
    def __init__(self, tools: OfflineResearchTools):
        self.tools = tools

    def search(self, query: str, num_papers: int = 8) -> Tuple[List[Paper], pd.DataFrame]:
        df_res, scores = self.tools.search(query, max_results=num_papers)
        
        papers = []
        for idx, (_, row) in enumerate(df_res.iterrows(), 1):
            papers.append(Paper(
                paper_id=f"paper_{idx}",
                title=row['paper_title'],
                abstract=row['paper_abstract'],
                category=row.get('paper_category', 'General'),
                date=str(row.get('paper_date', 'NA'))[:10]
            ))
        
        return papers, df_res


class OfflineAnalysisAgent:
    """Extract insights from abstracts using rule-based NLP."""

    def analyze(self, paper: Paper) -> Summary:
        text = paper.abstract.strip()
        
        # Sentence splitting
        sentences = [s.strip() + "." for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 20]
        
        # Extract key findings (first 3 meaningful sentences)
        key_findings = sentences[:3] if sentences else ["Abstract unavailable"]
        
        # Simple methodology detection
        method_keywords = ['method', 'approach', 'algorithm', 'model', 'framework', 'technique']
        methodology = next(
            (s for s in sentences if any(kw in s.lower() for kw in method_keywords)),
            "Methodology not explicitly stated in abstract."
        )
        
        # Research gap heuristic
        gap_keywords = ['future', 'limitation', 'challenge', 'however', 'need', 'require']
        research_gap = next(
            (s for s in sentences if any(kw in s.lower() for kw in gap_keywords)),
            "Further research needed to explore practical applications."
        )
        
        # Relevance score (0-10 based on abstract length and completeness)
        relevance = min(len(text) / 300, 1.0) * 10
        
        return Summary(
            paper_id=paper.paper_id,
            key_findings=key_findings,
            methodology=methodology,
            limitations=["Limitations not detailed in abstract"],
            research_gap=research_gap,
            relevance_score=round(relevance, 1)
        )


class OfflineSynthesisAgent:
    """Generate literature review from summaries."""

    def synthesize(self, query: str, papers: List[Paper], summaries: List[Summary]) -> str:
        lines = []
        
        # Introduction
        lines.append(f"LITERATURE REVIEW: {query.upper()}\n")
        lines.append(f"This review synthesizes {len(papers)} relevant papers from the offline dataset.\n")
        
        # Group by category if available
        categories = {}
        for p, s in zip(papers, summaries):
            cat = p.category if p.category != 'nan' else 'General'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((p, s))
        
        # Thematic summary
        lines.append("KEY THEMES:\n")
        for i, (p, s) in enumerate(zip(papers[:5], summaries[:5]), 1):
            lines.append(f"{i}. {p.title}")
            lines.append(f"   Year: {p.date} | Relevance: {s.relevance_score}/10")
            if s.key_findings:
                lines.append(f"   Finding: {s.key_findings[0][:150]}...")
            lines.append("")
        
        # Research gaps
        lines.append("\nIDENTIFIED RESEARCH GAPS:")
        for i, s in enumerate(summaries[:3], 1):
            lines.append(f"{i}. {s.research_gap[:200]}")
        
        return "\n".join(lines)


class OfflineCitationAgent:
    def __init__(self, tools: OfflineResearchTools):
        self.tools = tools

    def generate(self, papers: List[Paper], df_res: pd.DataFrame) -> List[Citation]:
        citations = []
        for idx, (p, (_, row)) in enumerate(zip(papers, df_res.iterrows()), 1):
            apa = self.tools.format_citation(row, idx)
            citations.append(Citation(paper_id=p.paper_id, apa=apa))
        return citations


class OfflinePlanningAgent:
    """Generate research project plan."""

    def create_plan(self, query: str, papers: List[Paper], summaries: List[Summary]) -> Dict:
        # Extract domains from categories
        domains = list(set(p.category for p in papers if p.category != 'nan'))[:3]
        
        return {
            "project_title": f"Research Project: {query}",
            "objectives": [
                f"Conduct comprehensive literature review on '{query}'",
                f"Analyze {len(papers)} core papers across domains: {', '.join(domains)}",
                "Identify methodological approaches and compare effectiveness",
                "Synthesize findings and propose novel research directions"
            ],
            "phases": [
                {"phase": "Literature Collection", "duration": "1 week", "tasks": [
                    "Complete offline paper search",
                    "Organize papers by relevance and category"
                ]},
                {"phase": "Analysis & Note-taking", "duration": "2 weeks", "tasks": [
                    "Read and annotate selected papers",
                    "Extract key methodologies and results",
                    "Create comparison matrix"
                ]},
                {"phase": "Synthesis & Writing", "duration": "2 weeks", "tasks": [
                    "Draft literature review sections",
                    "Identify research gaps",
                    "Propose future work"
                ]},
                {"phase": "Review & Presentation", "duration": "1 week", "tasks": [
                    "Revise and polish document",
                    "Prepare presentation slides",
                    "Practice defense"
                ]}
            ],
            "timeline": "6 weeks total",
            "deliverables": [
                "10-15 page literature review",
                "Annotated bibliography of core papers",
                "Research proposal for future work",
                "Presentation slides"
            ],
            "resources_needed": [
                "Access to full-text papers (via university library)",
                "Reference management software (Zotero/Mendeley)",
                "Collaboration tools for team projects"
            ]
        }

print("âœ… All agents defined")



class OfflineOrchestrator:
    """Main controller coordinating all agents."""

    def __init__(self, tools: OfflineResearchTools):
        self.tools = tools
        self.search_agent = OfflineSearchAgent(tools)
        self.analysis_agent = OfflineAnalysisAgent()
        self.synthesis_agent = OfflineSynthesisAgent()
        self.citation_agent = OfflineCitationAgent(tools)
        self.planning_agent = OfflinePlanningAgent()
        self.sessions: Dict[str, ResearchSession] = {}

    def process_query(self, query: str, num_papers: int = 8) -> ResearchSession:
        session_id = f"session_{int(datetime.now().timestamp())}"
        
        print("\n" + "="*80)
        print(f"ğŸ�“ PROCESSING QUERY: {query}")
        print("="*80)

        try:
            # Step 1: Search
            print("\n[1/5] ğŸ”� Searching offline dataset...")
            papers, df_res = self.search_agent.search(query, num_papers=num_papers)
            print(f"       âœ… Found {len(papers)} papers")

            # Step 2: Analyze
            print("\n[2/5] ğŸ“Š Analyzing abstracts...")
            summaries = []
            for i, p in enumerate(papers, 1):
                print(f"       Analyzing {i}/{len(papers)}: {p.title[:50]}...")
                summaries.append(self.analysis_agent.analyze(p))
            print(f"       âœ… Analysis complete")

            # Step 3: Synthesize
            print("\n[3/5] ğŸ“� Generating literature review...")
            literature_review = self.synthesis_agent.synthesize(query, papers, summaries)
            print(f"       âœ… Review generated ({len(literature_review)} chars)")

            # Step 4: Citations
            print("\n[4/5] ğŸ“š Formatting citations...")
            citations = self.citation_agent.generate(papers, df_res)
            print(f"       âœ… {len(citations)} citations formatted")

            # Step 5: Plan
            print("\n[5/5] ğŸ“‹ Creating research plan...")
            plan = self.planning_agent.create_plan(query, papers, summaries)
            print(f"       âœ… Plan created")

            session = ResearchSession(
                session_id=session_id,
                query=query,
                papers=papers,
                summaries=summaries,
                citations=citations,
                literature_review=literature_review,
                research_plan=plan
            )
            
            self.sessions[session_id] = session
            
            print("\n" + "="*80)
            print("âœ… QUERY PROCESSING COMPLETE")
            print("="*80 + "\n")
            
            return session
            
        except Exception as e:
            print(f"\nâ�Œ ERROR: {e}")
            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise

print("âœ… Orchestrator defined")



class OfflineUI:
    def __init__(self, orchestrator: OfflineOrchestrator):
        self.orchestrator = orchestrator

    def display(self, session: ResearchSession):
        print("\n" + "â–ˆ"*80)
        print("ğŸ�“ EDURESEARCH AGENT - RESULTS SUMMARY")
        print("â–ˆ"*80)
        
        print(f"\nğŸ“Œ Query: {session.query}")
        print(f"â�±ï¸�  Time: {session.created_at}")
        print(f"ğŸ“š Papers Analyzed: {len(session.papers)}")
        
        # Papers
        print("\n" + "â”€"*80)
        print("ğŸ“„ TOP PAPERS")
        print("â”€"*80)
        for i, (p, s) in enumerate(zip(session.papers[:6], session.summaries[:6]), 1):
            print(f"\n{i}. {p.title}")
            print(f"   ğŸ“… {p.date} | ğŸ�·ï¸�  {p.category} | â­� Relevance: {s.relevance_score}/10")
            if s.key_findings:
                print(f"   ğŸ’¡ {s.key_findings[0][:120]}...")
        
        # Literature Review
        print("\n" + "â”€"*80)
        print("ğŸ“� LITERATURE REVIEW")
        print("â”€"*80)
        print(session.literature_review)
        
        # Citations
        print("\n" + "â”€"*80)
        print("ğŸ“š CITATIONS (APA Style)")
        print("â”€"*80)
        for c in session.citations[:8]:
            print(f"   {c.apa}")
        
        # Plan
        print("\n" + "â”€"*80)
        print("ğŸ“‹ RESEARCH PROJECT PLAN")
        print("â”€"*80)
        plan = session.research_plan
        print(f"\nğŸ�¯ {plan['project_title']}")
        print(f"\nâ�±ï¸�  Timeline: {plan['timeline']}")
        
        print("\nğŸ“Œ Objectives:")
        for obj in plan['objectives']:
            print(f"   â€¢ {obj}")
        
        print("\nğŸ“… Phases:")
        for phase in plan['phases']:
            print(f"   {phase['phase']} ({phase['duration']})")
        
        print("\nğŸ“¦ Deliverables:")
        for d in plan['deliverables']:
            print(f"   â€¢ {d}")
        
        print("\n" + "â–ˆ"*80 + "\n")

print("âœ… UI defined")



# Initialize the complete offline system
tools = OfflineResearchTools(papers_df)
orchestrator = OfflineOrchestrator(tools)
ui = OfflineUI(orchestrator)

print("\n" + "ğŸš€"*40)
print("OFFLINE EDURESEARCH AGENT - READY")
print("ğŸš€"*40)
print(f"\nâœ… Dataset: {len(papers_df)} papers loaded")
print("âœ… All agents initialized")
print("âœ… System ready for queries\n")



# Demo 1: Computer Vision
query1 = "Decision Forests vs. Deep Networks"
session1 = orchestrator.process_query(query1, num_papers=6)
ui.display(session1)




# Demo 2: NLP
query2 = "transformer models for sentiment analysis"
session2 = orchestrator.process_query(query2, num_papers=6)
ui.display(session2)



class AgentEvaluator:
    @staticmethod
    def evaluate_session(session: ResearchSession) -> Dict[str, float]:
        metrics = {
            "num_papers": len(session.papers),
            "avg_relevance": np.mean([s.relevance_score for s in session.summaries]),
            "review_length": len(session.literature_review),
            "num_citations": len(session.citations),
            "plan_phases": len(session.research_plan.get('phases', [])),
            "plan_deliverables": len(session.research_plan.get('deliverables', []))
        }
        
        # Overall quality score (0-1)
        quality = (
            (metrics["avg_relevance"] / 10) * 0.4 +
            min(metrics["review_length"] / 1000, 1.0) * 0.3 +
            (metrics["plan_phases"] / 5) * 0.3
        )
        metrics["overall_quality"] = round(quality, 3)
        
        return metrics

# Evaluate all sessions
evaluator = AgentEvaluator()

print("\n" + "="*80)
print("ğŸ“ˆ PERFORMANCE EVALUATION")
print("="*80)

for sid, sess in orchestrator.sessions.items():
    metrics = evaluator.evaluate_session(sess)
    print(f"\nQuery: {sess.query}")
    print(f"  Overall Quality: {metrics['overall_quality']:.3f}/1.000")
    print(f"  Papers: {metrics['num_papers']}")
    print(f"  Avg Relevance: {metrics['avg_relevance']:.1f}/10")
    print(f"  Review Length: {metrics['review_length']} chars")
    print(f"  Plan Phases: {metrics['plan_phases']}")

print("\n" + "="*80)
print(f"ğŸ“Š Total Sessions: {len(orchestrator.sessions)}")
print("="*80 + "\n")


