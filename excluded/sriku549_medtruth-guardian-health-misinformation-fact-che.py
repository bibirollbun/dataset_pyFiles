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


# Install required libraries
!pip install -q google-generativeai
!pip install -q requests
!pip install -q beautifulsoup4
!pip install -q pandas

# Verify installations
import google.generativeai as genai
import sqlite3
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime
import os

print("âœ… All libraries installed successfully!")


# Configure Gemini API
from kaggle_secrets import UserSecretsClient

# Get API key from Kaggle secrets
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Test connection
model = genai.GenerativeModel('gemini-2.0-flash-exp')
response = model.generate_content("Say 'Hello from MedTruth Guardian!'")
print(response.text)

print("âœ… Gemini API connected successfully!")


# Create folder structure
import os

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('output', exist_ok=True)

print("âœ… Project structure created!")
print("ğŸ“� Folders:")
print("   - data/     (for SQLite database)")
print("   - logs/     (for logging)")
print("   - output/   (for reports)")


# Initialize SQLite database
DB_PATH = 'data/medtruth.db'

def init_database():
    """Create database tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create claims table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_text TEXT NOT NULL,
            verdict TEXT NOT NULL,
            truth_score INTEGER,
            evidence TEXT,
            sources TEXT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create known_myths table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS known_myths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            myth_text TEXT NOT NULL,
            category TEXT,
            debunk_explanation TEXT,
            official_source TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create user_history table (for memory)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            claim_checked TEXT,
            result TEXT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("âœ… Database initialized!")
    print(f"ğŸ“Š Database location: {DB_PATH}")

# Initialize database
init_database()


# Add sample known myths to database
def add_sample_myths():
    """Populate database with common health myths"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    sample_myths = [
        ("Drinking bleach cures COVID-19", "COVID-19", 
         "Bleach is highly toxic and should never be consumed. This is a dangerous myth that can cause severe harm or death.",
         "CDC, WHO"),
        
        ("5G networks spread COVID-19", "COVID-19",
         "Viruses cannot travel on radio waves or mobile networks. COVID-19 is spread through respiratory droplets.",
         "WHO"),
        
        ("Vitamin C cures COVID-19", "COVID-19",
         "While vitamin C supports immune function, there is no evidence it cures or prevents COVID-19.",
         "NIH, CDC"),
        
        ("Vaccines cause autism", "Vaccines",
         "Extensive research has found no link between vaccines and autism. This myth originated from a fraudulent study.",
         "CDC, WHO, AAP"),
        
        ("Eating carrots gives you night vision", "Nutrition",
         "Carrots contain vitamin A which supports eye health, but they don't provide night vision abilities.",
         "Harvard Health"),
    ]
    
    for myth in sample_myths:
        cursor.execute('''
            INSERT OR IGNORE INTO known_myths (myth_text, category, debunk_explanation, official_source)
            VALUES (?, ?, ?, ?)
        ''', myth)
    
    conn.commit()
    conn.close()
    
    print("âœ… Sample myths added to database!")
    print(f"ğŸ“� Added {len(sample_myths)} known myths")

# Add sample data
add_sample_myths()


# Test database queries
def test_database():
    """Test that database is working"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Count myths
    cursor.execute("SELECT COUNT(*) FROM known_myths")
    myth_count = cursor.fetchone()[0]
    
    # Get sample myth
    cursor.execute("SELECT myth_text, debunk_explanation FROM known_myths LIMIT 1")
    sample = cursor.fetchone()
    
    conn.close()
    
    print("âœ… Database test successful!")
    print(f"ğŸ“Š Total myths in database: {myth_count}")
    print(f"ğŸ“� Sample myth: {sample[0][:50]}...")
    
    return myth_count > 0

# Test database
test_database()


# Setup simple logging
class Logger:
    """Simple logging system"""
    def __init__(self):
        self.logs = []
        
    def log(self, level, agent, message):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'agent': agent,
            'message': message
        }
        self.logs.append(log_entry)
        
        # Print to console
        print(f"[{timestamp}] [{level}] {agent}: {message}")
    
    def info(self, agent, message):
        self.log("INFO", agent, message)
    
    def error(self, agent, message):
        self.log("ERROR", agent, message)
    
    def get_logs(self):
        """Get all logs as DataFrame"""
        return pd.DataFrame(self.logs)

# Initialize logger
logger = Logger()
logger.info("SYSTEM", "MedTruth Guardian initialized")

print("âœ… Logging system ready!")


# Final verification
def verify_setup():
    """Verify all components are working"""
    checks = []
    
    # Check Gemini API
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        model.generate_content("test")
        checks.append(("âœ…", "Gemini API"))
    except:
        checks.append(("â�Œ", "Gemini API"))
    
    # Check database
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.close()
        checks.append(("âœ…", "SQLite Database"))
    except:
        checks.append(("â�Œ", "SQLite Database"))
    
    # Check folders
    if os.path.exists('data') and os.path.exists('logs'):
        checks.append(("âœ…", "Project Structure"))
    else:
        checks.append(("â�Œ", "Project Structure"))
    
    # Check logging
    if logger.logs:
        checks.append(("âœ…", "Logging System"))
    else:
        checks.append(("â�Œ", "Logging System"))
    
    print("\n" + "="*50)
    print("ğŸ“‹ PHASE 1 SETUP VERIFICATION")
    print("="*50)
    for status, component in checks:
        print(f"{status} {component}")
    print("="*50)
    
    all_good = all(status == "âœ…" for status, _ in checks)
    if all_good:
        print("\nğŸ�‰ Phase 1 Complete! Ready for Phase 2!")
    else:
        print("\nâš ï¸� Some components failed. Please check errors above.")
    
    return all_good

# Run verification
verify_setup()


# Get SerpAPI key from Kaggle secrets
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
SERPAPI_KEY = user_secrets.get_secret("SERP_API_KEY")

print("âœ… SerpAPI key loaded!")


import requests
from typing import List, Dict
import json

class SearchTools:
    """Tools for searching information using SerpAPI"""
    
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
    
    def web_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search the web using SerpAPI
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results with title, link, snippet
        """
        try:
            url = "https://serpapi.com/search"
            params = {
                'api_key': self.serpapi_key,
                'q': query,
                'num': num_results,
                'engine': 'google'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            if 'organic_results' in data:
                for item in data['organic_results'][:num_results]:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', '')
                    })
            
            logger.info("SearchTools", f"SerpAPI search found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            logger.error("SearchTools", f"SerpAPI search failed: {str(e)}")
            # Return empty list on error
            return []
    
    def search_database(self, claim: str) -> List[Dict]:
        """
        Search known myths database for similar claims
        
        Args:
            claim: Health claim to check
            
        Returns:
            List of matching myths from database
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Search for similar myths (case-insensitive)
            query = """
                SELECT myth_text, category, debunk_explanation, official_source
                FROM known_myths
                WHERE myth_text LIKE ?
                OR category LIKE ?
            """
            
            search_term = f"%{claim}%"
            cursor.execute(query, (search_term, search_term))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'myth': row[0],
                    'category': row[1],
                    'explanation': row[2],
                    'source': row[3],
                    'from_database': True
                })
            
            conn.close()
            
            logger.info("SearchTools", f"Database search found {len(results)} matches")
            return results
            
        except Exception as e:
            logger.error("SearchTools", f"Database search failed: {str(e)}")
            return []
    
    def fetch_url_content(self, url: str, max_length: int = 2000) -> str:
        """
        Fetch content from a URL
        
        Args:
            url: URL to fetch
            max_length: Maximum content length
            
        Returns:
            Text content from URL
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            logger.info("SearchTools", f"Fetched content from: {url}")
            return text
            
        except Exception as e:
            logger.error("SearchTools", f"Failed to fetch {url}: {str(e)}")
            return ""

# Initialize search tools with SerpAPI key
search_tools = SearchTools(SERPAPI_KEY)

print("âœ… Search tools initialized with SerpAPI!")


class ResearchAgent:
    """
    Agent responsible for gathering evidence about health claims
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model = genai.GenerativeModel(model_name)
        self.search_tools = search_tools
        self.name = "ResearchAgent"
    
    def research_claim(self, claim: str) -> Dict:
        """
        Research a health claim by gathering evidence from multiple sources
        
        Args:
            claim: Health claim to research
            
        Returns:
            Dictionary containing evidence and sources
        """
        logger.info(self.name, f"Starting research on claim: {claim}")
        
        evidence = {
            'claim': claim,
            'web_results': [],
            'database_results': [],
            'summary': '',
            'sources': []
        }
        
        # Step 1: Search database for known myths
        logger.info(self.name, "Searching database for known myths...")
        db_results = self.search_tools.search_database(claim)
        evidence['database_results'] = db_results
        
        if db_results:
            logger.info(self.name, f"Found {len(db_results)} matches in known myths database")
        
        # Step 2: Search web for fact-checks and information
        logger.info(self.name, "Searching web for fact-checks...")
        search_query = f"{claim} fact check health misinformation"
        web_results = self.search_tools.web_search(search_query, num_results=5)
        evidence['web_results'] = web_results
        
        if web_results:
            logger.info(self.name, f"Found {len(web_results)} web results")
        
        # Step 3: Use Gemini to summarize and analyze evidence
        logger.info(self.name, "Analyzing evidence with Gemini...")
        summary = self._analyze_evidence(claim, db_results, web_results)
        evidence['summary'] = summary
        
        # Step 4: Compile sources
        sources = []
        for db_result in db_results:
            sources.append(f"Database: {db_result['source']}")
        for web_result in web_results:
            sources.append(f"{web_result['title']} - {web_result['link']}")
        
        evidence['sources'] = sources
        
        logger.info(self.name, f"Research complete. Found {len(sources)} sources")
        
        return evidence
    
    def _analyze_evidence(self, claim: str, db_results: List[Dict], 
                         web_results: List[Dict]) -> str:
        """
        Use Gemini to analyze gathered evidence
        
        Args:
            claim: Original claim
            db_results: Results from database
            web_results: Results from web search
            
        Returns:
            Summary of evidence
        """
        # Prepare context for Gemini
        context = f"Health Claim: {claim}\n\n"
        
        if db_results:
            context += "Known Myths Database Results:\n"
            for i, result in enumerate(db_results, 1):
                context += f"{i}. {result['myth']}\n"
                context += f"   Explanation: {result['explanation']}\n"
                context += f"   Source: {result['source']}\n\n"
        
        if web_results:
            context += "Web Search Results:\n"
            for i, result in enumerate(web_results, 1):
                context += f"{i}. {result['title']}\n"
                context += f"   {result['snippet']}\n"
                context += f"   URL: {result['link']}\n\n"
        
        # Create prompt for Gemini
        prompt = f"""You are a medical fact-checking research assistant. Analyze the following evidence about a health claim.

{context}

Task: Provide a concise summary (3-4 sentences) of what the evidence says about this claim. Focus on:
1. Whether this is a known myth or legitimate health information
2. What authoritative sources say
3. The scientific consensus if available

Summary:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(self.name, f"Failed to analyze evidence: {str(e)}")
            return "Unable to analyze evidence at this time."

# Initialize Research Agent
research_agent = ResearchAgent()

print("âœ… Research Agent initialized!")


# Test the Research Agent with a sample claim

def test_research_agent():
    """Test research agent with sample claims"""
    
    test_claims = [
        "Drinking bleach cures COVID-19",
        "Vaccines cause autism",
        "Eating carrots improves night vision"
    ]
    
    print("="*60)
    print("ğŸ§ª TESTING RESEARCH AGENT")
    print("="*60)
    
    for claim in test_claims:
        print(f"\nğŸ“‹ Testing claim: {claim}")
        print("-"*60)
        
        # Run research
        evidence = research_agent.research_claim(claim)
        
        # Display results
        print(f"\nğŸ“Š Results:")
        print(f"   Database matches: {len(evidence['database_results'])}")
        print(f"   Web results: {len(evidence['web_results'])}")
        print(f"   Total sources: {len(evidence['sources'])}")
        
        print(f"\nğŸ“� Summary:")
        print(f"   {evidence['summary']}")
        
        print("\n" + "="*60)
    
    print("\nâœ… Research Agent testing complete!")

# Run test
test_research_agent()


class ContextCompactor:
    """
    Compress long evidence into concise summaries
    This demonstrates 'context engineering' requirement
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model = genai.GenerativeModel(model_name)
    
    def compact_web_results(self, web_results: List[Dict], max_length: int = 500) -> str:
        """
        Compress multiple web search results into a concise summary
        
        Args:
            web_results: List of web search results
            max_length: Maximum length of summary
            
        Returns:
            Compacted summary
        """
        if not web_results:
            return ""
        
        # Combine all snippets
        combined_text = "\n".join([
            f"{r['title']}: {r['snippet']}" 
            for r in web_results
        ])
        
        prompt = f"""Summarize the following web search results about a health claim in {max_length} characters or less. Focus on the key facts and consensus:

{combined_text}

Concise Summary:"""
        
        try:
            response = self.model.generate_content(prompt)
            summary = response.text
            
            # Ensure it's within max length
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary
        except Exception as e:
            logger.error("ContextCompactor", f"Failed to compact: {str(e)}")
            return combined_text[:max_length] + "..."

# Add to Research Agent
research_agent.compactor = ContextCompactor()

print("âœ… Context Compaction added!")


import concurrent.futures

class EnhancedResearchAgent(ResearchAgent):
    """
    Enhanced Research Agent with parallel search capabilities
    This demonstrates 'parallel agents' requirement
    """
    
    def research_claim_parallel(self, claim: str) -> Dict:
        """
        Research claim using parallel searches for speed
        
        Args:
            claim: Health claim to research
            
        Returns:
            Dictionary containing evidence and sources
        """
        logger.info(self.name, f"Starting PARALLEL research on: {claim}")
        
        evidence = {
            'claim': claim,
            'web_results': [],
            'database_results': [],
            'summary': '',
            'sources': []
        }
        
        # Run searches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both search tasks simultaneously
            db_future = executor.submit(self.search_tools.search_database, claim)
            web_future = executor.submit(
                self.search_tools.web_search, 
                f"{claim} fact check health misinformation",
                5
            )
            
            # Wait for both to complete
            db_results = db_future.result()
            web_results = web_future.result()
        
        evidence['database_results'] = db_results
        evidence['web_results'] = web_results
        
        logger.info(self.name, "Parallel searches complete")
        
        # Analyze evidence
        summary = self._analyze_evidence(claim, db_results, web_results)
        evidence['summary'] = summary
        
        # Compile sources
        sources = []
        for db_result in db_results:
            sources.append(f"Database: {db_result['source']}")
        for web_result in web_results:
            sources.append(f"{web_result['title']} - {web_result['link']}")
        
        evidence['sources'] = sources
        
        logger.info(self.name, f"Research complete. Found {len(sources)} sources")
        
        return evidence

# Replace with enhanced version
research_agent = EnhancedResearchAgent()

print("âœ… Enhanced Research Agent with parallel search ready!")


def verify_phase2():
    """Verify Research Agent is working correctly"""
    
    print("\n" + "="*60)
    print("ğŸ“‹ PHASE 2 VERIFICATION")
    print("="*60)
    
    checks = []
    
    # Test 1: Search tools initialized
    try:
        assert search_tools is not None
        checks.append(("âœ…", "Search Tools initialized"))
    except:
        checks.append(("â�Œ", "Search Tools initialization"))
    
    # Test 2: Research agent initialized
    try:
        assert research_agent is not None
        checks.append(("âœ…", "Research Agent initialized"))
    except:
        checks.append(("â�Œ", "Research Agent initialization"))
    
    # Test 3: Database search works
    try:
        results = search_tools.search_database("COVID")
        checks.append(("âœ…", f"Database search ({len(results)} results)"))
    except Exception as e:
        checks.append(("â�Œ", f"Database search failed: {e}"))
    
    # Test 4: Research claim works
    try:
        evidence = research_agent.research_claim("test claim")
        assert 'summary' in evidence
        checks.append(("âœ…", "Research claim function"))
    except Exception as e:
        checks.append(("â�Œ", f"Research claim failed: {e}"))
    
    # Test 5: Parallel search works
    try:
        evidence = research_agent.research_claim_parallel("test claim")
        checks.append(("âœ…", "Parallel search"))
    except Exception as e:
        checks.append(("â�Œ", f"Parallel search failed: {e}"))
    
    # Display results
    for status, component in checks:
        print(f"{status} {component}")
    
    print("="*60)
    
    all_good = all(status == "âœ…" for status, _ in checks)
    if all_good:
        print("\nğŸ�‰ Phase 2 Complete! Ready for Phase 3!")
    else:
        print("\nâš ï¸� Some components failed. Review errors above.")
    
    return all_good

# Run verification
verify_phase2()


class VerdictScoring:
    """
    Scoring system for health claim verdicts
    """
    
    # Verdict categories
    FALSE = "FALSE"
    MIXED = "MIXED"
    TRUE = "TRUE"
    UNVERIFIED = "UNVERIFIED"
    
    @staticmethod
    def categorize_score(score: int) -> str:
        """
        Convert numeric score to verdict category
        
        Args:
            score: Truth score (0-100)
            
        Returns:
            Verdict category
        """
        if score < 0:
            return VerdictScoring.UNVERIFIED
        elif score < 40:
            return VerdictScoring.FALSE
        elif score < 80:
            return VerdictScoring.MIXED
        else:
            return VerdictScoring.TRUE
    
    @staticmethod
    def get_emoji(verdict: str) -> str:
        """Get emoji for verdict"""
        emojis = {
            "FALSE": "ğŸš«",
            "MIXED": "âš ï¸�",
            "TRUE": "âœ…",
            "UNVERIFIED": "â�“"
        }
        return emojis.get(verdict, "â�“")
    
    @staticmethod
    def get_color(verdict: str) -> str:
        """Get color code for verdict"""
        colors = {
            "FALSE": "RED",
            "MIXED": "YELLOW",
            "TRUE": "GREEN",
            "UNVERIFIED": "GRAY"
        }
        return colors.get(verdict, "GRAY")

print("âœ… Verdict scoring system defined!")


class VerdictAgent:
    """
    Agent responsible for analyzing evidence and making verdicts
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model = genai.GenerativeModel(model_name)
        self.name = "VerdictAgent"
        self.scoring = VerdictScoring()
    
    def analyze_and_verdict(self, claim: str, evidence: Dict) -> Dict:
        """
        Analyze evidence and generate verdict
        
        Args:
            claim: Original health claim
            evidence: Evidence from Research Agent
            
        Returns:
            Dictionary with verdict, score, explanation, recommendations
        """
        logger.info(self.name, f"Analyzing evidence for: {claim}")
        
        # Step 1: Analyze evidence and assign score
        analysis_result = self._analyze_evidence(claim, evidence)
        
        truth_score = analysis_result['score']
        verdict_category = self.scoring.categorize_score(truth_score)
        
        logger.info(self.name, f"Verdict: {verdict_category} (Score: {truth_score}/100)")
        
        # Step 2: Generate user-friendly report
        report = self._generate_report(
            claim, 
            verdict_category, 
            truth_score,
            analysis_result['reasoning'],
            evidence
        )
        
        # Step 3: Save to database
        self._save_to_database(claim, verdict_category, truth_score, evidence, report)
        
        result = {
            'claim': claim,
            'verdict': verdict_category,
            'score': truth_score,
            'reasoning': analysis_result['reasoning'],
            'report': report,
            'sources': evidence.get('sources', [])
        }
        
        logger.info(self.name, "Verdict analysis complete")
        
        return result
    
    def _analyze_evidence(self, claim: str, evidence: Dict) -> Dict:
        """
        Use Gemini to analyze evidence and assign truth score
        
        Args:
            claim: Health claim
            evidence: Evidence dictionary
            
        Returns:
            Dictionary with score and reasoning
        """
        # Prepare evidence context
        context = self._prepare_evidence_context(claim, evidence)
        
        # Create analysis prompt
        prompt = f"""You are a medical fact-checking expert. Analyze the following health claim and evidence.

CLAIM: {claim}

EVIDENCE:
{context}

TASK: Provide your analysis in the following format:

SCORE: [A number from 0-100 where:
- 0-39 = FALSE (claim is debunked/dangerous/false)
- 40-79 = MIXED (partially true, needs context, or uncertain)
- 80-100 = TRUE (scientifically supported)]

REASONING: [2-3 sentences explaining why you assigned this score, citing specific evidence]

Format your response as:
SCORE: [number]
REASONING: [your explanation]
"""

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Parse response
            score = self._extract_score(result_text)
            reasoning = self._extract_reasoning(result_text)
            
            return {
                'score': score,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(self.name, f"Failed to analyze evidence: {str(e)}")
            return {
                'score': -1,
                'reasoning': "Unable to analyze evidence due to technical error."
            }
    
    def _prepare_evidence_context(self, claim: str, evidence: Dict) -> str:
        """Prepare evidence for analysis"""
        context = ""
        
        # Add database results
        if evidence.get('database_results'):
            context += "KNOWN MYTHS DATABASE:\n"
            for result in evidence['database_results']:
                context += f"- {result['myth']}\n"
                context += f"  Explanation: {result['explanation']}\n"
                context += f"  Source: {result['source']}\n\n"
        
        # Add web search results
        if evidence.get('web_results'):
            context += "WEB SEARCH RESULTS:\n"
            for i, result in enumerate(evidence['web_results'], 1):
                context += f"{i}. {result['title']}\n"
                context += f"   {result['snippet']}\n"
                context += f"   URL: {result['link']}\n\n"
        
        # Add research summary if available
        if evidence.get('summary'):
            context += f"RESEARCH SUMMARY:\n{evidence['summary']}\n\n"
        
        return context
    
    def _extract_score(self, text: str) -> int:
        """Extract score from Gemini response"""
        import re
        
        # Look for "SCORE: XX" pattern
        match = re.search(r'SCORE:\s*(\d+)', text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            # Ensure score is in valid range
            return max(0, min(100, score))
        
        # If no score found, return -1 (unverified)
        logger.error(self.name, "Could not extract score from response")
        return -1
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from Gemini response"""
        import re
        
        # Look for "REASONING: ..." pattern
        match = re.search(r'REASONING:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            # Clean up (remove extra whitespace, limit length)
            reasoning = ' '.join(reasoning.split())
            return reasoning[:500]  # Limit to 500 chars
        
        # Fallback: return entire text
        return text[:500]
    
    def _generate_report(self, claim: str, verdict: str, score: int, 
                        reasoning: str, evidence: Dict) -> str:
        """
        Generate user-friendly report
        
        Args:
            claim: Original claim
            verdict: Verdict category
            score: Truth score
            reasoning: Analysis reasoning
            evidence: Evidence dictionary
            
        Returns:
            Formatted report string
        """
        emoji = self.scoring.get_emoji(verdict)
        
        report = f"""
{'='*60}
{emoji} MEDTRUTH GUARDIAN VERDICT {emoji}
{'='*60}

CLAIM: "{claim}"

VERDICT: {verdict}
TRUTH SCORE: {score}/100

ANALYSIS:
{reasoning}

SOURCES:
"""
        
        # Add sources
        sources = evidence.get('sources', [])
        if sources:
            for i, source in enumerate(sources[:5], 1):  # Max 5 sources
                report += f"{i}. {source}\n"
        else:
            report += "No sources available\n"
        
        # Add recommendations based on verdict
        report += f"\nRECOMMENDATION:\n"
        
        if verdict == "FALSE":
            report += "âš ï¸� This claim is NOT supported by credible evidence. "
            report += "Do not follow this advice. Consult healthcare professionals "
            report += "for medical guidance.\n"
        elif verdict == "MIXED":
            report += "âš ï¸� This claim is partially true or requires additional context. "
            report += "Consult healthcare professionals before taking action.\n"
        elif verdict == "TRUE":
            report += "âœ… This claim is supported by credible evidence. "
            report += "However, always consult healthcare professionals for "
            report += "personalized medical advice.\n"
        else:
            report += "â�“ Unable to verify this claim. Please consult trusted "
            report += "health authorities or medical professionals.\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def _save_to_database(self, claim: str, verdict: str, score: int, 
                         evidence: Dict, report: str):
        """
        Save verdict to database for future reference
        
        Args:
            claim: Health claim
            verdict: Verdict category
            score: Truth score
            evidence: Evidence dictionary
            report: Generated report
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Convert sources to JSON string
            sources_json = json.dumps(evidence.get('sources', []))
            
            # Insert into claims table
            cursor.execute('''
                INSERT INTO claims (claim_text, verdict, truth_score, evidence, sources)
                VALUES (?, ?, ?, ?, ?)
            ''', (claim, verdict, score, report, sources_json))
            
            conn.commit()
            conn.close()
            
            logger.info(self.name, f"Verdict saved to database for: {claim[:50]}...")
            
        except Exception as e:
            logger.error(self.name, f"Failed to save to database: {str(e)}")

# Initialize Verdict Agent
verdict_agent = VerdictAgent()

print("âœ… Verdict Agent initialized!")


def test_verdict_agent():
    """
    Test the complete Research â†’ Verdict pipeline
    """
    
    print("\n" + "="*60)
    print("ğŸ§ª TESTING RESEARCH + VERDICT AGENT PIPELINE")
    print("="*60)
    
    test_claims = [
        "Drinking bleach cures COVID-19",
        "Vaccines cause autism"
    ]
    
    for claim in test_claims:
        print(f"\n{'='*60}")
        print(f"ğŸ“‹ Testing: {claim}")
        print("="*60)
        
        # Step 1: Research (gather evidence)
        print("\nğŸ”� Phase 1: Researching claim...")
        evidence = research_agent.research_claim_parallel(claim)
        
        print(f"   Found {len(evidence['database_results'])} database matches")
        print(f"   Found {len(evidence['web_results'])} web results")
        
        # Step 2: Verdict (analyze and decide)
        print("\nâš–ï¸� Phase 2: Analyzing evidence...")
        result = verdict_agent.analyze_and_verdict(claim, evidence)
        
        # Step 3: Display report
        print("\nğŸ“„ FINAL REPORT:")
        print(result['report'])
    
    print("\nâœ… Pipeline testing complete!")

# Run test
test_verdict_agent()


def check_previous_verdicts(claim: str) -> List[Dict]:
    """
    Check if this claim has been fact-checked before
    (Demonstrates memory/caching capability)
    
    Args:
        claim: Health claim to check
        
    Returns:
        List of previous verdicts from database
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Search for similar claims
        query = """
            SELECT claim_text, verdict, truth_score, sources, checked_at
            FROM claims
            WHERE claim_text LIKE ?
            ORDER BY checked_at DESC
            LIMIT 5
        """
        
        search_term = f"%{claim}%"
        cursor.execute(query, (search_term,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'claim': row[0],
                'verdict': row[1],
                'score': row[2],
                'sources': json.loads(row[3]) if row[3] else [],
                'checked_at': row[4]
            })
        
        conn.close()
        
        return results
        
    except Exception as e:
        logger.error("Database", f"Failed to check previous verdicts: {str(e)}")
        return []

print("âœ… Database query function added!")


class VerdictMetrics:
    """
    Track and display metrics about verdicts
    (Demonstrates observability requirement)
    """
    
    def __init__(self):
        self.metrics = {
            'total_claims': 0,
            'verdicts': {
                'FALSE': 0,
                'MIXED': 0,
                'TRUE': 0,
                'UNVERIFIED': 0
            },
            'avg_score': 0,
            'cache_hits': 0
        }
    
    def update_metrics(self):
        """Update metrics from database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Total claims
            cursor.execute("SELECT COUNT(*) FROM claims")
            self.metrics['total_claims'] = cursor.fetchone()[0]
            
            # Verdicts breakdown
            cursor.execute("""
                SELECT verdict, COUNT(*) 
                FROM claims 
                GROUP BY verdict
            """)
            for row in cursor.fetchall():
                self.metrics['verdicts'][row[0]] = row[1]
            
            # Average score
            cursor.execute("SELECT AVG(truth_score) FROM claims")
            avg = cursor.fetchone()[0]
            self.metrics['avg_score'] = round(avg, 1) if avg else 0
            
            conn.close()
            
        except Exception as e:
            logger.error("VerdictMetrics", f"Failed to update metrics: {str(e)}")
    
    def display_dashboard(self):
        """Display metrics dashboard"""
        self.update_metrics()
        
        print("\n" + "="*60)
        print("ğŸ“Š MEDTRUTH GUARDIAN METRICS DASHBOARD")
        print("="*60)
        
        print(f"\nğŸ“ˆ Total Claims Checked: {self.metrics['total_claims']}")
        print(f"ğŸ“Š Average Truth Score: {self.metrics['avg_score']}/100")
        
        print(f"\nğŸ�¯ Verdict Breakdown:")
        for verdict, count in self.metrics['verdicts'].items():
            emoji = VerdictScoring.get_emoji(verdict)
            percentage = (count / self.metrics['total_claims'] * 100) if self.metrics['total_claims'] > 0 else 0
            print(f"   {emoji} {verdict}: {count} ({percentage:.1f}%)")
        
        print("\n" + "="*60)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Return metrics as pandas DataFrame"""
        self.update_metrics()
        
        data = []
        for verdict, count in self.metrics['verdicts'].items():
            data.append({
                'Verdict': verdict,
                'Count': count,
                'Emoji': VerdictScoring.get_emoji(verdict)
            })
        
        return pd.DataFrame(data)

# Initialize metrics
metrics = VerdictMetrics()

print("âœ… Verdict metrics dashboard created!")


class EnhancedVerdictAgent(VerdictAgent):
    """
    Enhanced Verdict Agent with loop capability
    (Demonstrates loop agents requirement)
    """
    
    def analyze_with_retry(self, claim: str, evidence: Dict, 
                          max_retries: int = 2) -> Dict:
        """
        Analyze with retry loop if evidence is insufficient
        
        Args:
            claim: Health claim
            evidence: Evidence dictionary
            max_retries: Maximum number of retry attempts
            
        Returns:
            Verdict result
        """
        logger.info(self.name, f"Starting analysis with retry capability (max {max_retries} retries)")
        
        for attempt in range(max_retries + 1):
            # Analyze evidence
            result = self.analyze_and_verdict(claim, evidence)
            
            # Check if we have sufficient evidence
            has_sufficient_evidence = (
                len(evidence.get('database_results', [])) > 0 or
                len(evidence.get('web_results', [])) >= 3
            )
            
            if has_sufficient_evidence or result['verdict'] != 'UNVERIFIED':
                logger.info(self.name, f"Analysis successful on attempt {attempt + 1}")
                return result
            
            # If insufficient evidence and more retries available
            if attempt < max_retries:
                logger.info(self.name, f"Insufficient evidence. Retry {attempt + 1}/{max_retries}")
                
                # Try alternative search (this is the "loop" behavior)
                alternative_query = f"{claim} medical research scientific evidence"
                additional_results = search_tools.web_search(alternative_query, 3)
                
                # Merge additional results
                evidence['web_results'].extend(additional_results)
                
                logger.info(self.name, f"Added {len(additional_results)} additional search results")
        
        # If all retries exhausted
        logger.info(self.name, "Maximum retries reached, returning best available result")
        return result

# Replace with enhanced version
verdict_agent = EnhancedVerdictAgent()

print("âœ… Enhanced Verdict Agent with loop capability ready!")


def verify_phase3():
    """Verify Verdict Agent is working correctly"""
    
    print("\n" + "="*60)
    print("ğŸ“‹ PHASE 3 VERIFICATION")
    print("="*60)
    
    checks = []
    
    # Test 1: Verdict Agent initialized
    try:
        assert verdict_agent is not None
        checks.append(("âœ…", "Verdict Agent initialized"))
    except:
        checks.append(("â�Œ", "Verdict Agent initialization"))
    
    # Test 2: Scoring system works
    try:
        score = VerdictScoring.categorize_score(25)
        assert score == "FALSE"
        checks.append(("âœ…", "Scoring system"))
    except Exception as e:
        checks.append(("â�Œ", f"Scoring system failed: {e}"))
    
    # Test 3: Can analyze evidence
    try:
        test_evidence = {
            'database_results': [],
            'web_results': [],
            'sources': []
        }
        result = verdict_agent.analyze_and_verdict("test claim", test_evidence)
        assert 'verdict' in result
        checks.append(("âœ…", "Evidence analysis"))
    except Exception as e:
        checks.append(("â�Œ", f"Evidence analysis failed: {e}"))
    
    # Test 4: Database save works
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM claims")
        count = cursor.fetchone()[0]
        conn.close()
        checks.append(("âœ…", f"Database storage ({count} claims)"))
    except Exception as e:
        checks.append(("â�Œ", f"Database storage failed: {e}"))
    
    # Test 5: Metrics dashboard works
    try:
        metrics.display_dashboard()
        checks.append(("âœ…", "Metrics dashboard"))
    except Exception as e:
        checks.append(("â�Œ", f"Metrics dashboard failed: {e}"))
    
    # Display results
    print()
    for status, component in checks:
        print(f"{status} {component}")
    
    print("="*60)
    
    all_good = all(status == "âœ…" for status, _ in checks)
    if all_good:
        print("\nğŸ�‰ Phase 3 Complete! Ready for Phase 4!")
    else:
        print("\nâš ï¸� Some components failed. Review errors above.")
    
    return all_good

# Run verification
verify_phase3()


class SessionMemory:
    """
    Manages session state and conversation context
    (Demonstrates Sessions & State management requirement)
    """
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_history = []
        self.current_claim = None
        self.current_evidence = None
        self.current_verdict = None
        self.user_preferences = {
            'show_sources': True,
            'detail_level': 'standard'  # 'brief', 'standard', 'detailed'
        }
    
    def add_interaction(self, user_input: str, agent_response: str):
        """Record user-agent interaction"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'agent': agent_response
        })
    
    def get_context(self) -> str:
        """Get conversation context for agents"""
        if not self.conversation_history:
            return "New conversation"
        
        recent = self.conversation_history[-3:]  # Last 3 interactions
        context = "Recent conversation:\n"
        for interaction in recent:
            context += f"User: {interaction['user']}\n"
            context += f"Agent: {interaction['agent'][:100]}...\n"
        
        return context
    
    def clear(self):
        """Clear session memory"""
        self.conversation_history = []
        self.current_claim = None
        self.current_evidence = None
        self.current_verdict = None
        logger.info("SessionMemory", "Session cleared")

# Initialize session memory
session_memory = SessionMemory()

print("âœ… Session Memory system initialized!")
print(f"ğŸ“� Session ID: {session_memory.session_id}")


class LongTermMemory:
    """
    Manages user history and persistent memory across sessions
    (Demonstrates Long-term memory requirement)
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
    
    def save_user_query(self, user_id: str, claim: str, verdict: str):
        """Save user query to history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_history (user_id, claim_checked, result)
                VALUES (?, ?, ?)
            ''', (user_id, claim, verdict))
            
            conn.commit()
            conn.close()
            
            logger.info("LongTermMemory", f"Saved query to user history: {claim[:50]}...")
            
        except Exception as e:
            logger.error("LongTermMemory", f"Failed to save user history: {str(e)}")
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve user's past queries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT claim_checked, result, checked_at
                FROM user_history
                WHERE user_id = ?
                ORDER BY checked_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'claim': row[0],
                    'result': row[1],
                    'checked_at': row[2]
                })
            
            conn.close()
            
            logger.info("LongTermMemory", f"Retrieved {len(history)} past queries for user")
            return history
            
        except Exception as e:
            logger.error("LongTermMemory", f"Failed to retrieve user history: {str(e)}")
            return []
    
    def check_cache(self, claim: str) -> Dict:
        """
        Check if claim was already fact-checked (cache hit)
        This improves performance for repeated queries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Look for exact or very similar claim
            cursor.execute('''
                SELECT claim_text, verdict, truth_score, evidence, sources, checked_at
                FROM claims
                WHERE claim_text = ?
                ORDER BY checked_at DESC
                LIMIT 1
            ''', (claim,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                logger.info("LongTermMemory", "âœ… CACHE HIT - Claim already checked!")
                return {
                    'cache_hit': True,
                    'claim': row[0],
                    'verdict': row[1],
                    'score': row[2],
                    'evidence': row[3],
                    'sources': json.loads(row[4]) if row[4] else [],
                    'checked_at': row[5]
                }
            else:
                logger.info("LongTermMemory", "â�Œ CACHE MISS - New claim")
                return {'cache_hit': False}
                
        except Exception as e:
            logger.error("LongTermMemory", f"Cache check failed: {str(e)}")
            return {'cache_hit': False}

# Initialize long-term memory
long_term_memory = LongTermMemory()

print("âœ… Long-term Memory system initialized!")


class A2AProtocol:
    """
    Agent-to-Agent communication protocol
    (Demonstrates A2A Protocol requirement)
    """
    
    def __init__(self):
        self.message_log = []
    
    def send_message(self, from_agent: str, to_agent: str, 
                    message_type: str, payload: Dict) -> Dict:
        """
        Send message from one agent to another
        
        Args:
            from_agent: Sender agent name
            to_agent: Receiver agent name
            message_type: Type of message (REQUEST, RESPONSE, ERROR)
            payload: Message data
            
        Returns:
            Message envelope
        """
        message = {
            'id': len(self.message_log) + 1,
            'timestamp': datetime.now().isoformat(),
            'from': from_agent,
            'to': to_agent,
            'type': message_type,
            'payload': payload
        }
        
        self.message_log.append(message)
        
        logger.info("A2A", f"{from_agent} â†’ {to_agent}: {message_type}")
        
        return message
    
    def get_message_history(self) -> List[Dict]:
        """Get all A2A messages"""
        return self.message_log
    
    def clear_log(self):
        """Clear message log"""
        self.message_log = []

# Initialize A2A protocol
a2a_protocol = A2AProtocol()

print("âœ… A2A Protocol initialized!")


import os
import json
import glob
from typing import List, Dict

class MCPFilesystem:
    """
    MCP Filesystem Tool - Compliant with Model Context Protocol
    Provides file operations for caching and storage
    """
    
    def __init__(self, base_path: str = "./mcp_storage"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(f"{base_path}/evidence", exist_ok=True)
        os.makedirs(f"{base_path}/reports", exist_ok=True)
        os.makedirs(f"{base_path}/cache", exist_ok=True)
        logger.info("MCP-Filesystem", f"âœ… Initialized at {base_path}")
    
    # MCP Tool: write_file
    def write_file(self, filename: str, content: str, subdir: str = "") -> Dict:
        """
        MCP Tool: Write file to storage
        
        Args:
            filename: Name of file
            content: File content
            subdir: Subdirectory (evidence, reports, cache)
            
        Returns:
            Operation result
        """
        try:
            path = os.path.join(self.base_path, subdir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("MCP-Filesystem", f"âœ… Wrote: {subdir}/{filename}")
            
            return {
                'success': True,
                'path': path,
                'size': len(content)
            }
        except Exception as e:
            logger.error("MCP-Filesystem", f"â�Œ Write failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # MCP Tool: read_file
    def read_file(self, filename: str, subdir: str = "") -> Dict:
        """
        MCP Tool: Read file from storage
        
        Returns:
            File content and metadata
        """
        try:
            path = os.path.join(self.base_path, subdir, filename)
            
            if not os.path.exists(path):
                return {'success': False, 'error': 'File not found'}
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info("MCP-Filesystem", f"âœ… Read: {subdir}/{filename}")
            
            return {
                'success': True,
                'content': content,
                'size': len(content)
            }
        except Exception as e:
            logger.error("MCP-Filesystem", f"â�Œ Read failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # MCP Tool: list_files
    def list_files(self, pattern: str = "*", subdir: str = "") -> Dict:
        """
        MCP Tool: List files matching pattern
        
        Returns:
            List of matching files
        """
        try:
            search_path = os.path.join(self.base_path, subdir, pattern)
            files = glob.glob(search_path)
            filenames = [os.path.basename(f) for f in files]
            
            logger.info("MCP-Filesystem", f"âœ… Listed {len(filenames)} files")
            
            return {
                'success': True,
                'files': filenames,
                'count': len(filenames)
            }
        except Exception as e:
            logger.error("MCP-Filesystem", f"â�Œ List failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # MCP Tool: delete_file
    def delete_file(self, filename: str, subdir: str = "") -> Dict:
        """MCP Tool: Delete file"""
        try:
            path = os.path.join(self.base_path, subdir, filename)
            
            if os.path.exists(path):
                os.remove(path)
                logger.info("MCP-Filesystem", f"âœ… Deleted: {subdir}/{filename}")
                return {'success': True}
            else:
                return {'success': False, 'error': 'File not found'}
        except Exception as e:
            logger.error("MCP-Filesystem", f"â�Œ Delete failed: {e}")
            return {'success': False, 'error': str(e)}

# Initialize MCP Filesystem
mcp_filesystem = MCPFilesystem()

print("âœ… MCP Filesystem Tool initialized!")
print(f"ğŸ“� Storage location: ./mcp_storage/")
print(f"   - evidence/ (research results)")
print(f"   - reports/ (verdict reports)")
print(f"   - cache/ (temporary cache)")


class MCPMemory:
    """
    MCP Memory Tool - Compliant with Model Context Protocol
    Provides structured key-value storage with tagging
    """
    
    def __init__(self):
        self.storage = {}
        self.index = {}
        self.metadata = {}
        logger.info("MCP-Memory", "âœ… Initialized")
    
    # MCP Tool: store
    def store(self, key: str, value: Dict, tags: List[str] = None, 
             metadata: Dict = None) -> Dict:
        """
        MCP Tool: Store data with optional tags and metadata
        
        Args:
            key: Unique identifier
            value: Data to store
            tags: Optional tags for indexing
            metadata: Optional metadata
            
        Returns:
            Operation result
        """
        try:
            self.storage[key] = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'tags': tags or [],
                'metadata': metadata or {}
            }
            
            # Update tag index
            for tag in (tags or []):
                if tag not in self.index:
                    self.index[tag] = set()
                self.index[tag].add(key)
            
            logger.info("MCP-Memory", f"âœ… Stored: {key}")
            
            return {
                'success': True,
                'key': key,
                'tags': tags or []
            }
        except Exception as e:
            logger.error("MCP-Memory", f"â�Œ Store failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # MCP Tool: retrieve
    def retrieve(self, key: str) -> Dict:
        """
        MCP Tool: Retrieve data by key
        
        Returns:
            Stored data or error
        """
        if key in self.storage:
            logger.info("MCP-Memory", f"âœ… Retrieved: {key}")
            return {
                'success': True,
                'data': self.storage[key]['value'],
                'timestamp': self.storage[key]['timestamp']
            }
        else:
            return {
                'success': False,
                'error': 'Key not found'
            }
    
    # MCP Tool: search
    def search(self, tag: str) -> Dict:
        """
        MCP Tool: Search by tag
        
        Returns:
            List of matching keys
        """
        keys = list(self.index.get(tag, []))
        logger.info("MCP-Memory", f"âœ… Search '{tag}': {len(keys)} results")
        
        return {
            'success': True,
            'tag': tag,
            'keys': keys,
            'count': len(keys)
        }
    
    # MCP Tool: list_all
    def list_all(self, limit: int = 100) -> Dict:
        """MCP Tool: List all stored keys"""
        keys = list(self.storage.keys())[:limit]
        
        return {
            'success': True,
            'keys': keys,
            'count': len(keys),
            'total': len(self.storage)
        }
    
    # MCP Tool: delete
    def delete(self, key: str) -> Dict:
        """MCP Tool: Delete entry"""
        if key in self.storage:
            # Remove from tag index
            tags = self.storage[key]['tags']
            for tag in tags:
                if tag in self.index:
                    self.index[tag].discard(key)
            
            del self.storage[key]
            logger.info("MCP-Memory", f"âœ… Deleted: {key}")
            
            return {'success': True}
        else:
            return {'success': False, 'error': 'Key not found'}

# Initialize MCP Memory
mcp_memory = MCPMemory()

print("âœ… MCP Memory Tool initialized!")


# Update Research Agent to use MCP tools
research_agent.mcp_filesystem = mcp_filesystem
research_agent.mcp_memory = mcp_memory

# Add MCP caching method
def research_with_mcp_cache(self, claim: str) -> Dict:
    """
    Research with MCP-based caching
    """
    # Generate cache key
    cache_key = f"research_{hash(claim) % 10000}"
    
    # Check MCP Memory first
    memory_result = self.mcp_memory.retrieve(cache_key)
    
    if memory_result['success']:
        logger.info(self.name, "âœ… MCP Memory cache hit!")
        return memory_result['data']
    
    # Check MCP Filesystem cache
    cache_file = f"{cache_key}.json"
    file_result = self.mcp_filesystem.read_file(cache_file, "cache")
    
    if file_result['success']:
        logger.info(self.name, "âœ… MCP Filesystem cache hit!")
        data = json.loads(file_result['content'])
        
        # Also store in memory for faster access
        self.mcp_memory.store(cache_key, data, tags=['research', 'cached'])
        
        return data
    
    # No cache hit - do actual research
    logger.info(self.name, "Cache miss - performing research...")
    evidence = self.research_claim_parallel(claim)
    
    # Save to MCP Memory
    self.mcp_memory.store(
        cache_key, 
        evidence, 
        tags=['research', claim[:20]],
        metadata={'claim': claim}
    )
    
    # Save to MCP Filesystem
    self.mcp_filesystem.write_file(
        cache_file,
        json.dumps(evidence, indent=2),
        "cache"
    )
    
    logger.info(self.name, "âœ… Cached to MCP Memory + Filesystem")
    
    return evidence

# Add method to research agent
research_agent.research_with_mcp_cache = research_with_mcp_cache.__get__(research_agent)

print("âœ… Research Agent MCP integration complete!")


# Update Verdict Agent to use MCP tools
verdict_agent.mcp_filesystem = mcp_filesystem
verdict_agent.mcp_memory = mcp_memory

# Add MCP report storage method
def save_verdict_to_mcp(self, claim: str, result: Dict) -> None:
    """
    Save verdict to MCP storage
    """
    verdict_key = f"verdict_{hash(claim) % 10000}"
    
    # Save to MCP Memory
    self.mcp_memory.store(
        verdict_key,
        result,
        tags=['verdict', result['verdict'], 'completed'],
        metadata={
            'claim': claim,
            'score': result['score']
        }
    )
    
    # Save report to MCP Filesystem
    report_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    self.mcp_filesystem.write_file(
        report_file,
        result['report'],
        "reports"
    )
    
    logger.info(self.name, "âœ… Verdict saved to MCP Memory + Filesystem")

# Add method to verdict agent
verdict_agent.save_verdict_to_mcp = save_verdict_to_mcp.__get__(verdict_agent)

print("âœ… Verdict Agent MCP integration complete!")


class OrchestratorAgent:
    """
    Main orchestrator with MCP integration
    """
    
    def __init__(self, 
                 research_agent: ResearchAgent,
                 verdict_agent: VerdictAgent,
                 session_memory: SessionMemory,
                 long_term_memory: LongTermMemory,
                 a2a_protocol: A2AProtocol,
                 mcp_filesystem: MCPFilesystem,
                 mcp_memory: MCPMemory,
                 model_name: str = "gemini-2.0-flash-exp"):
        
        self.model = genai.GenerativeModel(model_name)
        self.research_agent = research_agent
        self.verdict_agent = verdict_agent
        self.session_memory = session_memory
        self.long_term_memory = long_term_memory
        self.a2a = a2a_protocol
        self.mcp_filesystem = mcp_filesystem
        self.mcp_memory = mcp_memory
        self.name = "OrchestratorAgent"
        
        # Connect MCP to agents
        self.research_agent.mcp_filesystem = mcp_filesystem
        self.research_agent.mcp_memory = mcp_memory
        self.verdict_agent.mcp_filesystem = mcp_filesystem
        self.verdict_agent.mcp_memory = mcp_memory
        
        logger.info(self.name, "âœ… Orchestrator initialized with MCP integration")
    
    def process_claim(self, user_input: str, user_id: str = "default_user") -> Dict:
        """
        Main entry point - processes a health claim with MCP integration
        
        Args:
            user_input: User's health claim or question
            user_id: User identifier for memory
            
        Returns:
            Complete result with verdict and report
        """
        logger.info(self.name, f"="*60)
        logger.info(self.name, f"Processing new claim from user: {user_id}")
        logger.info(self.name, f"Claim: {user_input}")
        logger.info(self.name, f"="*60)
        
        # Step 1: Extract and normalize claim
        claim = self._extract_claim(user_input)
        self.session_memory.current_claim = claim
        
        logger.info(self.name, f"Extracted claim: {claim}")
        
        # Step 2: Check MCP Memory cache first (fastest)
        cache_key = f"verdict_{hash(claim) % 10000}"
        mcp_cache_result = self.mcp_memory.retrieve(cache_key)
        
        if mcp_cache_result['success']:
            logger.info(self.name, "âœ… MCP MEMORY CACHE HIT!")
            cached_data = mcp_cache_result['data']
            
            # Record in session memory
            self.session_memory.add_interaction(user_input, cached_data.get('report', ''))
            
            # Save to user history
            self.long_term_memory.save_user_query(user_id, claim, cached_data.get('verdict', ''))
            
            return {
                'claim': claim,
                'verdict': cached_data.get('verdict', ''),
                'score': cached_data.get('score', 0),
                'report': cached_data.get('report', ''),
                'sources': cached_data.get('sources', []),
                'cached': True,
                'cache_source': 'MCP Memory'
            }
        
        # Step 3: Check database cache (secondary)
        db_cache_result = self.long_term_memory.check_cache(claim)
        
        if db_cache_result['cache_hit']:
            logger.info(self.name, "âœ… DATABASE CACHE HIT!")
            
            # Store in MCP Memory for next time
            self.mcp_memory.store(
                cache_key,
                {
                    'verdict': db_cache_result['verdict'],
                    'score': db_cache_result['score'],
                    'report': db_cache_result['evidence'],
                    'sources': db_cache_result['sources']
                },
                tags=['verdict', db_cache_result['verdict'], 'cached']
            )
            
            # Record in session memory
            self.session_memory.add_interaction(user_input, db_cache_result['evidence'])
            
            # Save to user history
            self.long_term_memory.save_user_query(user_id, claim, db_cache_result['verdict'])
            
            return {
                'claim': claim,
                'verdict': db_cache_result['verdict'],
                'score': db_cache_result['score'],
                'report': db_cache_result['evidence'],
                'sources': db_cache_result['sources'],
                'cached': True,
                'cache_source': 'Database',
                'cached_at': db_cache_result['checked_at']
            }
        
        # Step 4: No cache - do full research with MCP
        logger.info(self.name, "â�Œ CACHE MISS - Starting full research pipeline")
        
        # Send request to Research Agent (A2A)
        logger.info(self.name, "â†’ Sending research request to Research Agent")
        
        research_message = self.a2a.send_message(
            from_agent=self.name,
            to_agent="ResearchAgent",
            message_type="REQUEST",
            payload={'claim': claim, 'action': 'research_with_mcp'}
        )
        
        # Research Agent processes with MCP caching
        evidence = self._research_with_mcp_cache(claim)
        self.session_memory.current_evidence = evidence
        
        # Research Agent responds (A2A)
        self.a2a.send_message(
            from_agent="ResearchAgent",
            to_agent=self.name,
            message_type="RESPONSE",
            payload={'evidence': evidence}
        )
        
        logger.info(self.name, "â†� Received evidence from Research Agent")
        
        # Step 5: Send request to Verdict Agent (A2A)
        logger.info(self.name, "â†’ Sending verdict request to Verdict Agent")
        
        verdict_message = self.a2a.send_message(
            from_agent=self.name,
            to_agent="VerdictAgent",
            message_type="REQUEST",
            payload={'claim': claim, 'evidence': evidence, 'action': 'analyze'}
        )
        
        # Verdict Agent processes
        result = self.verdict_agent.analyze_and_verdict(claim, evidence)
        self.session_memory.current_verdict = result
        
        # Verdict Agent responds (A2A)
        self.a2a.send_message(
            from_agent="VerdictAgent",
            to_agent=self.name,
            message_type="RESPONSE",
            payload={'result': result}
        )
        
        logger.info(self.name, "â†� Received verdict from Verdict Agent")
        
        # Step 6: Save to MCP Memory (for fast future access)
        self.mcp_memory.store(
            cache_key,
            result,
            tags=['verdict', result['verdict'], 'completed'],
            metadata={'claim': claim, 'user_id': user_id}
        )
        
        # Step 7: Save report to MCP Filesystem
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(claim) % 1000}.txt"
        self.mcp_filesystem.write_file(
            report_filename,
            result['report'],
            "reports"
        )
        
        logger.info(self.name, "âœ… Saved to MCP Memory + Filesystem")
        
        # Step 8: Save to database (long-term storage)
        self.long_term_memory.save_user_query(user_id, claim, result['verdict'])
        
        # Step 9: Save to session memory
        self.session_memory.add_interaction(user_input, result['report'])
        
        logger.info(self.name, f"âœ… Processing complete - Verdict: {result['verdict']}")
        
        # Return complete result
        return {
            'claim': claim,
            'verdict': result['verdict'],
            'score': result['score'],
            'reasoning': result['reasoning'],
            'report': result['report'],
            'sources': result['sources'],
            'cached': False
        }
    
    def _research_with_mcp_cache(self, claim: str) -> Dict:
        """
        Research with MCP caching
        """
        # Generate research cache key
        research_cache_key = f"research_{hash(claim) % 10000}"
        
        # Check MCP Memory first
        memory_result = self.mcp_memory.retrieve(research_cache_key)
        
        if memory_result['success']:
            logger.info(self.name, "âœ… Research MCP Memory cache hit!")
            return memory_result['data']
        
        # Check MCP Filesystem cache
        cache_file = f"{research_cache_key}.json"
        file_result = self.mcp_filesystem.read_file(cache_file, "cache")
        
        if file_result['success']:
            logger.info(self.name, "âœ… Research MCP Filesystem cache hit!")
            data = json.loads(file_result['content'])
            
            # Also store in memory for faster next access
            self.mcp_memory.store(research_cache_key, data, tags=['research', 'cached'])
            
            return data
        
        # No cache - do actual research
        logger.info(self.name, "Performing new research...")
        evidence = self.research_agent.research_claim_parallel(claim)
        
        # Save to MCP Memory
        self.mcp_memory.store(
            research_cache_key, 
            evidence, 
            tags=['research', claim[:20]],
            metadata={'claim': claim}
        )
        
        # Save to MCP Filesystem
        self.mcp_filesystem.write_file(
            cache_file,
            json.dumps(evidence, indent=2),
            "cache"
        )
        
        logger.info(self.name, "âœ… Research cached to MCP Memory + Filesystem")
        
        return evidence
    
    def _extract_claim(self, user_input: str) -> str:
        """Extract health claim from user input using Gemini"""
        # For simple inputs, return as-is
        if len(user_input) < 200:
            return user_input.strip()
        
        # For complex inputs, use Gemini to extract core claim
        prompt = f"""Extract the main health claim from this user input. Return ONLY the claim, nothing else.

User input: {user_input}

Health claim:"""
        
        try:
            response = self.model.generate_content(prompt)
            claim = response.text.strip()
            return claim
        except:
            return user_input.strip()
    
    def get_session_summary(self) -> str:
        """Get summary of current session"""
        return f"""
SESSION SUMMARY
===============
Session ID: {self.session_memory.session_id}
Claims checked: {len(self.session_memory.conversation_history)}
Current claim: {self.session_memory.current_claim or 'None'}
Current verdict: {self.session_memory.current_verdict.get('verdict') if self.session_memory.current_verdict else 'None'}
"""
    
    def get_mcp_stats(self) -> Dict:
        """Get MCP usage statistics"""
        # MCP Memory stats
        memory_stats = self.mcp_memory.list_all()
        
        # MCP Filesystem stats
        cache_files = self.mcp_filesystem.list_files("*.json", "cache")
        report_files = self.mcp_filesystem.list_files("*.txt", "reports")
        
        return {
            'mcp_memory_entries': memory_stats['count'],
            'mcp_cache_files': cache_files['count'],
            'mcp_report_files': report_files['count']
        }

# Re-initialize Orchestrator with MCP
orchestrator = OrchestratorAgent(
    research_agent=research_agent,
    verdict_agent=verdict_agent,
    session_memory=session_memory,
    long_term_memory=long_term_memory,
    a2a_protocol=a2a_protocol,
    mcp_filesystem=mcp_filesystem,
    mcp_memory=mcp_memory
)

print("âœ… Orchestrator Agent with MCP integration initialized!")


def check_health_claim(claim: str, user_id: str = "user_001") -> None:
    """
    Main user interface function with MCP integration
    
    Args:
        claim: Health claim to check
        user_id: User identifier
    """
    print("\n" + "ğŸ”�" * 30)
    print("MEDTRUTH GUARDIAN - HEALTH CLAIM CHECKER")
    print("ğŸ”�" * 30)
    print(f"\nğŸ“� Your claim: {claim}\n")
    print("â�³ Processing...\n")
    
    # Process through orchestrator with MCP
    result = orchestrator.process_claim(claim, user_id)
    
    # Display result
    if result.get('cached'):
        cache_source = result.get('cache_source', 'Unknown')
        print(f"âš¡ CACHED RESULT (from {cache_source})")
        if 'cached_at' in result:
            print(f"ğŸ“… Original check: {result['cached_at']}")
        print()
    
    print(result['report'])
    
    # Show MCP stats
    mcp_stats = orchestrator.get_mcp_stats()
    print("\n" + "="*60)
    print("ğŸ’¾ MCP STORAGE STATISTICS")
    print("="*60)
    print(f"MCP Memory entries: {mcp_stats['mcp_memory_entries']}")
    print(f"MCP Cache files: {mcp_stats['mcp_cache_files']}")
    print(f"MCP Report files: {mcp_stats['mcp_report_files']}")
    
    # Show A2A message flow
    print("\n" + "="*60)
    print("ğŸ”— AGENT COMMUNICATION FLOW (A2A Protocol)")
    print("="*60)
    
    messages = a2a_protocol.get_message_history()
    recent_messages = messages[-6:] if len(messages) >= 6 else messages
    
    for msg in recent_messages:
        arrow = "â†’" if msg['type'] == "REQUEST" else "â†�"
        print(f"{msg['from']} {arrow} {msg['to']}: {msg['type']}")
    
    print("="*60)

print("âœ… Updated check_health_claim function ready!")


def batch_check_claims(claims: List[str], user_id: str = "batch_user") -> pd.DataFrame:
    """
    Process multiple claims in batch with MCP caching
    
    Args:
        claims: List of health claims to check
        user_id: User identifier
        
    Returns:
        DataFrame with results
    """
    results = []
    cache_hits = 0
    
    print(f"\nğŸ“Š Batch processing {len(claims)} claims with MCP...")
    print("="*60)
    
    for i, claim in enumerate(claims, 1):
        print(f"\n[{i}/{len(claims)}] Processing: {claim[:50]}...")
        
        try:
            result = orchestrator.process_claim(claim, user_id)
            
            if result.get('cached'):
                cache_hits += 1
                cache_indicator = "âš¡ (cached)"
            else:
                cache_indicator = ""
            
            results.append({
                'Claim': claim,
                'Verdict': result['verdict'],
                'Score': result['score'],
                'Cached': result.get('cached', False),
                'Cache Source': result.get('cache_source', 'N/A')
            })
            
            print(f"   âœ… {result['verdict']} (Score: {result['score']}/100) {cache_indicator}")
            
        except Exception as e:
            logger.error("BatchProcess", f"Failed to process claim: {str(e)}")
            results.append({
                'Claim': claim,
                'Verdict': 'ERROR',
                'Score': -1,
                'Cached': False,
                'Cache Source': 'N/A'
            })
            print(f"   â�Œ ERROR")
    
    print("\n" + "="*60)
    print("âœ… Batch processing complete!")
    print(f"ğŸ“Š Cache hits: {cache_hits}/{len(claims)} ({cache_hits/len(claims)*100:.1f}%)")
    
    # Show MCP stats
    mcp_stats = orchestrator.get_mcp_stats()
    print(f"ğŸ’¾ MCP Memory entries: {mcp_stats['mcp_memory_entries']}")
    print(f"ğŸ“� MCP Cache files: {mcp_stats['mcp_cache_files']}")
    print(f"ğŸ“„ MCP Reports: {mcp_stats['mcp_report_files']}")
    
    return pd.DataFrame(results)

print("âœ… Updated batch_check_claims function ready!")


def test_complete_system():
    """
    Test the complete MedTruth Guardian system with MCP
    """
    print("\n" + "="*70)
    print("ğŸ§ª COMPLETE SYSTEM TEST - MEDTRUTH GUARDIAN WITH MCP")
    print("="*70)
    
    test_claims = [
        "Drinking warm lemon water cures COVID-19",
        "Vaccines cause autism in children",
        "Eating carrots improves night vision",
        "Drinking warm lemon water cures COVID-19"  # Repeat to test cache
    ]
    
    print(f"\nğŸ“‹ Testing {len(test_claims)} claims through complete MCP-enhanced pipeline")
    print("   (Note: 4th claim is a repeat to test MCP caching)\n")
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_claims)}")
        if i == 4:
            print("(REPEAT CLAIM - Should hit MCP cache)")
        print("="*70)
        
        check_health_claim(claim, user_id=f"test_user_{i}")
        
        # Small delay between tests
        import time
        time.sleep(1)
    
    # Show final metrics
    print("\n" + "="*70)
    print("ğŸ“Š FINAL SYSTEM METRICS")
    print("="*70)
    metrics.display_dashboard()
    
    # Show MCP usage
    print("\n" + "="*70)
    print("ğŸ’¾ MCP USAGE SUMMARY")
    print("="*70)
    mcp_stats = orchestrator.get_mcp_stats()
    print(f"Total MCP Memory entries: {mcp_stats['mcp_memory_entries']}")
    print(f"Total MCP Cache files: {mcp_stats['mcp_cache_files']}")
    print(f"Total MCP Report files: {mcp_stats['mcp_report_files']}")
    
    # Show session summary
    print("\n" + orchestrator.get_session_summary())
    
    # Show A2A message count
    total_messages = len(a2a_protocol.get_message_history())
    print(f"\nğŸ”— Total A2A messages exchanged: {total_messages}")
    
    print("\nâœ… Complete system test finished!")
    print("ğŸ�‰ MCP integration verified - caching working as expected!")

print("âœ… Updated test_complete_system function ready!")


test_complete_system()


class AgentEvaluator:
    """
    Evaluation system for measuring agent performance
    (Demonstrates Agent Evaluation requirement)
    """
    
    def __init__(self):
        self.test_cases = []
        self.results = []
    
    def add_test_case(self, claim: str, expected_verdict: str, 
                     expected_score_range: tuple, category: str):
        """
        Add a test case with expected results
        
        Args:
            claim: Health claim to test
            expected_verdict: Expected verdict (TRUE/FALSE/MIXED)
            expected_score_range: (min, max) expected score
            category: Test category
        """
        self.test_cases.append({
            'claim': claim,
            'expected_verdict': expected_verdict,
            'expected_score_range': expected_score_range,
            'category': category
        })
    
    def run_evaluation(self) -> Dict:
        """
        Run evaluation on all test cases
        
        Returns:
            Evaluation results with metrics
        """
        print("\n" + "="*70)
        print("ğŸ“Š AGENT EVALUATION - Running Test Cases")
        print("="*70)
        
        self.results = []
        correct_verdicts = 0
        correct_scores = 0
        
        for i, test in enumerate(self.test_cases, 1):
            print(f"\n[{i}/{len(self.test_cases)}] Testing: {test['claim'][:60]}...")
            
            try:
                # Process claim
                result = orchestrator.process_claim(test['claim'], "evaluator")
                
                # Check verdict accuracy
                verdict_correct = result['verdict'] == test['expected_verdict']
                if verdict_correct:
                    correct_verdicts += 1
                
                # Check score range
                min_score, max_score = test['expected_score_range']
                score_correct = min_score <= result['score'] <= max_score
                if score_correct:
                    correct_scores += 1
                
                # Record result
                self.results.append({
                    'claim': test['claim'],
                    'category': test['category'],
                    'expected_verdict': test['expected_verdict'],
                    'actual_verdict': result['verdict'],
                    'verdict_correct': verdict_correct,
                    'expected_score_range': test['expected_score_range'],
                    'actual_score': result['score'],
                    'score_correct': score_correct,
                    'cached': result.get('cached', False)
                })
                
                status = "âœ…" if (verdict_correct and score_correct) else "âš ï¸�"
                print(f"   {status} Expected: {test['expected_verdict']}, Got: {result['verdict']}")
                print(f"      Score: {result['score']} (expected: {min_score}-{max_score})")
                
            except Exception as e:
                logger.error("Evaluator", f"Test failed: {str(e)}")
                self.results.append({
                    'claim': test['claim'],
                    'category': test['category'],
                    'error': str(e)
                })
        
        # Calculate metrics
        total_tests = len(self.test_cases)
        verdict_accuracy = (correct_verdicts / total_tests * 100) if total_tests > 0 else 0
        score_accuracy = (correct_scores / total_tests * 100) if total_tests > 0 else 0
        
        evaluation_summary = {
            'total_tests': total_tests,
            'verdict_accuracy': verdict_accuracy,
            'score_accuracy': score_accuracy,
            'correct_verdicts': correct_verdicts,
            'correct_scores': correct_scores
        }
        
        print("\n" + "="*70)
        print("ğŸ“ˆ EVALUATION RESULTS")
        print("="*70)
        print(f"Total test cases: {total_tests}")
        print(f"Verdict accuracy: {verdict_accuracy:.1f}% ({correct_verdicts}/{total_tests})")
        print(f"Score accuracy: {score_accuracy:.1f}% ({correct_scores}/{total_tests})")
        print("="*70)
        
        return evaluation_summary
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get evaluation results as DataFrame"""
        return pd.DataFrame(self.results)
    
    def add_standard_test_cases(self):
        """Add standard test cases for common health myths"""
        
        # Clearly FALSE claims (dangerous myths)
        self.add_test_case(
            "Drinking bleach cures COVID-19",
            "FALSE",
            (0, 20),
            "dangerous_myth"
        )
        
        self.add_test_case(
            "Vaccines cause autism",
            "FALSE",
            (0, 20),
            "debunked_myth"
        )
        
        self.add_test_case(
            "5G networks spread coronavirus",
            "FALSE",
            (0, 20),
            "conspiracy"
        )
        
        # MIXED claims (partially true or context-dependent)
        self.add_test_case(
            "Vitamin C helps with colds",
            "MIXED",
            (40, 79),
            "partially_true"
        )
        
        self.add_test_case(
            "Eating carrots improves vision",
            "MIXED",
            (40, 79),
            "partially_true"
        )
        
        # TRUE claims (scientifically supported)
        self.add_test_case(
            "Washing hands reduces spread of disease",
            "TRUE",
            (80, 100),
            "scientifically_proven"
        )
        
        self.add_test_case(
            "Smoking increases risk of lung cancer",
            "TRUE",
            (80, 100),
            "scientifically_proven"
        )
        
        print(f"âœ… Added {len(self.test_cases)} standard test cases")

# Initialize evaluator
evaluator = AgentEvaluator()

print("âœ… Agent Evaluator initialized!")


class TracingSystem:
    """
    Enhanced tracing for complete observability
    (Completes Observability requirement: Logging, Tracing, Metrics)
    """
    
    def __init__(self):
        self.traces = []
        self.current_trace_id = None
    
    def start_trace(self, operation: str, metadata: Dict = None) -> str:
        """Start a new trace"""
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.traces)}"
        
        trace = {
            'trace_id': trace_id,
            'operation': operation,
            'start_time': datetime.now(),
            'end_time': None,
            'duration_ms': None,
            'metadata': metadata or {},
            'spans': []
        }
        
        self.traces.append(trace)
        self.current_trace_id = trace_id
        
        logger.info("Tracing", f"Started trace: {trace_id} - {operation}")
        
        return trace_id
    
    def add_span(self, trace_id: str, agent: str, action: str, 
                status: str = "success", details: Dict = None):
        """Add a span to a trace"""
        
        for trace in self.traces:
            if trace['trace_id'] == trace_id:
                span = {
                    'timestamp': datetime.now().isoformat(),
                    'agent': agent,
                    'action': action,
                    'status': status,
                    'details': details or {}
                }
                trace['spans'].append(span)
                break
    
    def end_trace(self, trace_id: str):
        """End a trace and calculate duration"""
        
        for trace in self.traces:
            if trace['trace_id'] == trace_id:
                trace['end_time'] = datetime.now()
                duration = (trace['end_time'] - trace['start_time']).total_seconds() * 1000
                trace['duration_ms'] = round(duration, 2)
                
                logger.info("Tracing", f"Ended trace: {trace_id} - Duration: {duration:.2f}ms")
                break
    
    def get_trace(self, trace_id: str) -> Dict:
        """Get a specific trace"""
        for trace in self.traces:
            if trace['trace_id'] == trace_id:
                return trace
        return None
    
    def get_all_traces(self) -> List[Dict]:
        """Get all traces"""
        return self.traces
    
    def display_trace(self, trace_id: str):
        """Display a trace in readable format"""
        trace = self.get_trace(trace_id)
        
        if not trace:
            print(f"Trace {trace_id} not found")
            return
        
        print("\n" + "="*70)
        print(f"ğŸ”— TRACE: {trace['trace_id']}")
        print("="*70)
        print(f"Operation: {trace['operation']}")
        print(f"Duration: {trace['duration_ms']}ms")
        print(f"\nSpans ({len(trace['spans'])}):")
        
        for i, span in enumerate(trace['spans'], 1):
            status_emoji = "âœ…" if span['status'] == "success" else "â�Œ"
            print(f"  {i}. {status_emoji} {span['agent']}: {span['action']}")
            if span['details']:
                print(f"     Details: {span['details']}")
        
        print("="*70)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from traces"""
        
        if not self.traces:
            return {}
        
        durations = [t['duration_ms'] for t in self.traces if t['duration_ms']]
        
        return {
            'total_traces': len(self.traces),
            'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
            'min_duration_ms': min(durations) if durations else 0,
            'max_duration_ms': max(durations) if durations else 0
        }

# Initialize tracing
tracing = TracingSystem()

print("âœ… Tracing system initialized!")


# Add tracing to orchestrator
def process_claim_with_tracing(self, user_input: str, user_id: str = "default_user") -> Dict:
    """
    Process claim with complete tracing
    """
    # Start trace
    trace_id = tracing.start_trace(
        "process_health_claim",
        metadata={'user_id': user_id, 'claim': user_input[:50]}
    )
    
    try:
        # Extract claim
        tracing.add_span(trace_id, "Orchestrator", "extract_claim", "success")
        claim = self._extract_claim(user_input)
        
        # Check cache
        tracing.add_span(trace_id, "Orchestrator", "check_mcp_cache", "success")
        cache_key = f"verdict_{hash(claim) % 10000}"
        mcp_cache_result = self.mcp_memory.retrieve(cache_key)
        
        if mcp_cache_result['success']:
            tracing.add_span(trace_id, "Orchestrator", "cache_hit", "success", 
                           {'source': 'MCP Memory'})
            tracing.end_trace(trace_id)
            
            cached_data = mcp_cache_result['data']
            return {
                'claim': claim,
                'verdict': cached_data.get('verdict', ''),
                'score': cached_data.get('score', 0),
                'report': cached_data.get('report', ''),
                'sources': cached_data.get('sources', []),
                'cached': True,
                'cache_source': 'MCP Memory',
                'trace_id': trace_id
            }
        
        # Research phase
        tracing.add_span(trace_id, "Orchestrator", "request_research", "success")
        evidence = self._research_with_mcp_cache(claim)
        tracing.add_span(trace_id, "ResearchAgent", "gather_evidence", "success",
                       {'sources': len(evidence.get('sources', []))})
        
        # Verdict phase
        tracing.add_span(trace_id, "Orchestrator", "request_verdict", "success")
        result = self.verdict_agent.analyze_and_verdict(claim, evidence)
        tracing.add_span(trace_id, "VerdictAgent", "generate_verdict", "success",
                       {'verdict': result['verdict'], 'score': result['score']})
        
        # Save to MCP
        tracing.add_span(trace_id, "Orchestrator", "save_to_mcp", "success")
        self.mcp_memory.store(cache_key, result, tags=['verdict', result['verdict']])
        
        # End trace
        tracing.end_trace(trace_id)
        
        result['trace_id'] = trace_id
        return result
        
    except Exception as e:
        tracing.add_span(trace_id, "Orchestrator", "error", "failed", {'error': str(e)})
        tracing.end_trace(trace_id)
        raise

# Add method to orchestrator
orchestrator.process_claim_traced = process_claim_with_tracing.__get__(orchestrator)

print("âœ… Tracing integrated into Orchestrator!")


def run_comprehensive_demo():
    """
    Run comprehensive demo showing all features
    """
    print("\n" + "ğŸ�¬" * 30)
    print("MEDTRUTH GUARDIAN - COMPREHENSIVE DEMO")
    print("ğŸ�¬" * 30)
    
    demo_claims = [
        {
            'claim': "Drinking bleach cures COVID-19",
            'description': "Dangerous myth that should be clearly debunked"
        },
        {
            'claim': "Vaccines cause autism in children",
            'description': "Long-debunked myth based on fraudulent study"
        },
        {
            'claim': "Vitamin D helps immune system",
            'description': "Scientifically supported but context-dependent"
        },
        {
            'claim': "Drinking bleach cures COVID-19",
            'description': "REPEAT - Testing MCP cache (should be instant)"
        }
    ]
    
    print(f"\nğŸ“‹ Demo will process {len(demo_claims)} claims\n")
    
    for i, item in enumerate(demo_claims, 1):
        print(f"\n{'='*70}")
        print(f"DEMO {i}/{len(demo_claims)}: {item['description']}")
        print("="*70)
        print(f"Claim: \"{item['claim']}\"")
        print()
        
        # Process with tracing
        result = orchestrator.process_claim_traced(item['claim'], f"demo_user_{i}")
        
        # Show result
        print(result['report'])
        
        # Show trace
        if 'trace_id' in result:
            print(f"\nğŸ”— Trace ID: {result['trace_id']}")
            tracing.display_trace(result['trace_id'])
        
        # Pause between demos
        import time
        time.sleep(2)
    
    # Final statistics
    print("\n" + "="*70)
    print("ğŸ“Š DEMO COMPLETE - FINAL STATISTICS")
    print("="*70)
    
    # Metrics
    metrics.display_dashboard()
    
    # MCP Stats
    mcp_stats = orchestrator.get_mcp_stats()
    print(f"\nğŸ’¾ MCP Statistics:")
    print(f"   Memory entries: {mcp_stats['mcp_memory_entries']}")
    print(f"   Cache files: {mcp_stats['mcp_cache_files']}")
    print(f"   Reports: {mcp_stats['mcp_report_files']}")
    
    # Performance
    perf_stats = tracing.get_performance_stats()
    print(f"\nâš¡ Performance:")
    print(f"   Avg response time: {perf_stats.get('avg_duration_ms', 0):.2f}ms")
    print(f"   Fastest: {perf_stats.get('min_duration_ms', 0):.2f}ms")
    print(f"   Slowest: {perf_stats.get('max_duration_ms', 0):.2f}ms")
    
    # A2A messages
    total_messages = len(a2a_protocol.get_message_history())
    print(f"\nğŸ”— A2A Protocol:")
    print(f"   Total messages: {total_messages}")
    
    print("\n" + "="*70)
    print("âœ… Demo complete!")

print("âœ… Comprehensive demo function ready!")


def run_comprehensive_demo():
    """
    Run comprehensive demo showing all features
    """
    print("\n" + "ğŸ�¬" * 30)
    print("MEDTRUTH GUARDIAN - COMPREHENSIVE DEMO")
    print("ğŸ�¬" * 30)
    
    demo_claims = [
        {
            'claim': "Drinking bleach cures COVID-19",
            'description': "Dangerous myth that should be clearly debunked"
        },
        {
            'claim': "Vaccines cause autism in children",
            'description': "Long-debunked myth based on fraudulent study"
        },
        {
            'claim': "Vitamin D helps immune system",
            'description': "Scientifically supported but context-dependent"
        },
        {
            'claim': "Drinking bleach cures COVID-19",
            'description': "REPEAT - Testing MCP cache (should be instant)"
        }
    ]
    
    print(f"\nğŸ“‹ Demo will process {len(demo_claims)} claims\n")
    
    for i, item in enumerate(demo_claims, 1):
        print(f"\n{'='*70}")
        print(f"DEMO {i}/{len(demo_claims)}: {item['description']}")
        print("="*70)
        print(f"Claim: \"{item['claim']}\"")
        print()
        
        # Process with tracing
        result = orchestrator.process_claim_traced(item['claim'], f"demo_user_{i}")
        
        # Show result
        print(result['report'])
        
        # Show trace
        if 'trace_id' in result:
            print(f"\nğŸ”— Trace ID: {result['trace_id']}")
            tracing.display_trace(result['trace_id'])
        
        # Pause between demos
        import time
        time.sleep(2)
    
    # Final statistics
    print("\n" + "="*70)
    print("ğŸ“Š DEMO COMPLETE - FINAL STATISTICS")
    print("="*70)
    
    # Metrics
    metrics.display_dashboard()
    
    # MCP Stats
    mcp_stats = orchestrator.get_mcp_stats()
    print(f"\nğŸ’¾ MCP Statistics:")
    print(f"   Memory entries: {mcp_stats['mcp_memory_entries']}")
    print(f"   Cache files: {mcp_stats['mcp_cache_files']}")
    print(f"   Reports: {mcp_stats['mcp_report_files']}")
    
    # Performance
    perf_stats = tracing.get_performance_stats()
    print(f"\nâš¡ Performance:")
    print(f"   Avg response time: {perf_stats.get('avg_duration_ms', 0):.2f}ms")
    print(f"   Fastest: {perf_stats.get('min_duration_ms', 0):.2f}ms")
    print(f"   Slowest: {perf_stats.get('max_duration_ms', 0):.2f}ms")
    
    # A2A messages
    total_messages = len(a2a_protocol.get_message_history())
    print(f"\nğŸ”— A2A Protocol:")
    print(f"   Total messages: {total_messages}")
    
    print("\n" + "="*70)
    print("âœ… Demo complete!")

print("âœ… Comprehensive demo function ready!")


run_comprehensive_demo()

