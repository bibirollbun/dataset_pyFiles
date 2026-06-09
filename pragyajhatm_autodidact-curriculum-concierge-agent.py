import warnings
warnings.filterwarnings("ignore")
import os

# --- INSTALLATIONS ---
!pip install -q -U ddgs langchain langchain-community langchain-core langchain-google-genai

# --- IMPORTS ---
import os
import json
import time
from kaggle_secrets import UserSecretsClient 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from IPython.display import display, Markdown

# --- DIRECT IMPORT FOR SEARCH ---
try:
    from ddgs import DDGS
except ImportError:
    print("âš ï¸� DDGS not found. Restart Session and run again.")

# --- API KEY SETUP ---
api_key = os.getenv("GOOGLE_API_KEY") 
try:
    user_secrets = UserSecretsClient()
    secret_key = user_secrets.get_secret("GOOGLE_API_KEY")
    if secret_key:
        api_key = secret_key
        os.environ["GOOGLE_API_KEY"] = api_key
except:
    pass

# --- CUSTOM SEARCH TOOL ---
def custom_search_func(query):
    try:
        results = []
        with DDGS() as ddgs:
            time.sleep(1)
            ddg_results = ddgs.text(query, max_results=5)
            if ddg_results:
                for r in ddg_results:
                    results.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}")
        
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search Error: {e}"

search_tool = Tool(
    name="duckduckgo_search",
    description="Search for tutorials.",
    func=custom_search_func
)

# --- INITIALIZATION ---
try:
    llm_logic = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.2, 
        google_api_key=api_key
    )
    
    llm_creative = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.7, 
        google_api_key=api_key
    )
    
    print("âœ… System Initialized: Agents Ready.")
except Exception as e:
    print(f"â�Œ Initialization Error: {e}")


class ArchitectAgent:
    def __init__(self, llm):
        self.llm = llm
        
    # Here we prompt the architect agent to create the curriculum
    def create_syllabus(self, topic, timeframe):
        print(f"ğŸ�—ï¸� Architect: Designing {timeframe} path for '{topic}'...")
        
        prompt = f"""
        You are the best Curriculum Designer in the world.
        Create a strictly structured learning path for: {topic} over {timeframe}.
        
        OUTPUT FORMAT:
        Return ONLY a raw JSON list of strings.
        Each string must be a specific, searchable sub-topic.
        Create up to 10 topics as can be completed within the timeframe.
        
        Example: ["Python variables vs lists", "Python functions tutorial", "Python error handling", "Python file I/O"]
        """

        # The result must be formatted as JSON, else the program should take the fallback route
        try:
            response = self.llm.invoke(prompt).content
            clean_json = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"âš ï¸� Architect Error: {e}")
            return [f"{topic} basics", f"{topic} core concepts", f"{topic} advanced features", f"{topic} best practices"]

print("âœ… Architect Agent Online.")


class ScoutAgent:
    def __init__(self, llm, tool):
        self.llm = llm
        self.tool = tool
        
    # Here we prompt the scout agent to research relevant topics
    def fetch_resources(self, topic_list):
        print(f"ğŸ•µï¸� Scout: Researching {len(topic_list)} modules...")
        research_data = {}

        for topic in topic_list:
            print(f"   > Searching for: {topic}")
            try:
                raw_results = self.tool.run(f"best free tutorial for {topic}")
                
                summary_prompt = f"""
                Analyze these search results for '{topic}':
                {raw_results}
                
                Extract the 3 to 7 best resources. For each, keep the Title and URL.
                Discard any results that look like ads or generic spam.
                """
                
                summary = self.llm.invoke(summary_prompt).content
                research_data[topic] = summary
                
            except Exception as e:
                print(f"   âš ï¸� Search failed for {topic}: {e}")
                research_data[topic] = "No reliable data found."
                
        return research_data

print("âœ… Scout Agent Online.")


class ReviewerAgent:
    def __init__(self, llm):
        self.llm = llm
        
    # Here we prompt the reviewer agent to review the result and format in markdown
    def compile_final_guide(self, topic, research_data):
        print("âš–ï¸� Reviewer: Auditing resources and compiling final guide...")
        
        data_str = json.dumps(research_data, indent=2)
        
        prompt = f"""
        You are a strict Quality Assurance Auditor.
        I have collected research notes for a course on "{topic}".
        
        YOUR TASK:
        1. Review the notes below.
        2. Filter out any broken or suspicious-looking links.
        3. Compile a professional Markdown Syllabus.
        
        FORMATTING RULES:
        - Group content by "Module".
        - Add a "Final Project Idea" section at the end.
        - Use emojis to make it engaging (e.g., ğŸ“š, ğŸ”—, ğŸ’¡).

        RAW NOTES:
        {data_str}
        """
        
        return self.llm.invoke(prompt).content

print("âœ… Reviewer Agent Online.")


# --- USER INPUT ---
TOPIC = input("Enter your topic: ")
TIMEFRAME = input("Enter the duration of study: ")

# --- INSTANTIATE AGENTS ---
# We use 'llm_logic' (low temp) for planning and 'llm_creative' (high temp) for writing
architect = ArchitectAgent(llm_logic)
scout = ScoutAgent(llm_logic, search_tool)
reviewer = ReviewerAgent(llm_creative)

# --- EXECUTE THE CHAIN ---
print(f"\nğŸš€ STARTING AUTODIDACT AGENT FOR: {TOPIC}\n" + "="*50)

# Step 1: Architect plans the course
syllabus = architect.create_syllabus(TOPIC, TIMEFRAME)
print(f"\nğŸ“‹ Syllabus Generated: {syllabus}\n")

# Step 2: Scout goes to the web (This takes 10-20 seconds)
raw_resources = scout.fetch_resources(syllabus)

# Step 3: Reviewer writes the final markdown
final_output = reviewer.compile_final_guide(TOPIC, raw_resources)

# --- DISPLAY RESULT ---
print("\n" + "="*50 + "\nDONE. RENDERED OUTPUT BELOW:\n")
display(Markdown(final_output))


# --- RUN THIS CELL TO DISPLAY ARCHITECTURE DIAGRAM ---
from graphviz import Digraph

dot = Digraph(comment='AutoDidact Architecture')
dot.attr(rankdir='LR', size='15')

# Define Nodes
dot.node('U', 'User Input\n(Topic + Time)', shape='ellipse', style='filled', fillcolor='lightblue')
dot.node('A', 'Agent 1: Architect\n(Gemini Flash)', shape='box', style='filled', fillcolor='lightgrey')
dot.node('S', 'Agent 2: Scout\n(DuckDuckGo Tool)', shape='box', style='filled', fillcolor='lightyellow')
dot.node('R', 'Agent 3: Reviewer\n(Gemini Flash)', shape='box', style='filled', fillcolor='lightgreen')
dot.node('O', 'Final Output\n(Markdown Curriculum)', shape='note', style='filled', fillcolor='white')

# Define Edges
dot.edge('U', 'A', label=' "Teach me X" ')
dot.edge('A', 'S', label=' JSON Syllabus ')
dot.edge('S', 'R', label=' Raw Content ')
dot.edge('R', 'O', label=' Curated Guide ')

# Render
dot

