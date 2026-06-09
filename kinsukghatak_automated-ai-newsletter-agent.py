# pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

try:
    youtube_api_key = UserSecretsClient().get_secret("youtube_api_key")
    os.environ["youtube_api_key"] = youtube_api_key
    print("âœ… youtube_api_key  key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'youtube_api_key' to your Kaggle secrets. Details: {e}"
    )


# Import required libraries

from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent

from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

import asyncio
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import MCPToolset
from google.genai import types
from mcp.client.stdio import StdioServerParameters
from datetime import datetime, timedelta
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
import xml.etree.ElementTree as ET
import requests



print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=15,  # Delay multiplier
    initial_delay=3,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Research Agent: Its job is to use the google_search tool and present findings.
research_agent_google = Agent(
    name="ResearchAgent_google",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are a specialized research agent for discovering top AI/ML news articles from 
    the period mentioned by the user.


    Your INDEPENDENT task:
    1. Use the google_search tool to find the most relevant and widely discussed AI/ML news from 
    the specific timeline share by the user
    2. Execute 4-5 targeted searches to get comprehensive coverage:
       - "artificial intelligence news latest"
       - "machine learning breakthrough latest"
       - "AI company announcement recent"
       - "generative AI LLM news"
       - "AI research development 2024 , 2025"
    
    3. From ALL search results, identify the 4-5 MOST SIGNIFICANT news stories
    
    4. For EACH story, you MUST extract and include:
       - Headline/Title of the article
       - Source/Publication name (e.g., TechCrunch, Reuters, OpenAI Blog)
       - Brief summary (2-3 sentences) explaining what happened and why it matters
       - **THE COMPLETE URL** - This is CRITICAL! Always include the full clickable link
    
    Selection criteria for top stories:
    - Breaking news or major announcements
    - Significant product launches or updates
    - Important research breakthroughs
    - Industry-shaping developments
    - High credibility sources (avoid random blogs)
    
    Output format - Use EXACTLY this structure:
    
    â€¢ **[Article Headline]** - [Source Name]
      Summary: [2-3 sentences explaining the news and its significance]
      ğŸ”— URL: [FULL URL HERE - must be included!]
]
    
    [Continue for all 4-5 stories...]
    
    CRITICAL REQUIREMENTS:
    âœ“ ALWAYS include the complete URL for every article
    âœ“ Make sure URLs are clickable and complete (start with http:// or https://)
    âœ“ If a search result doesn't provide a clear URL, skip that result and find another
    âœ“ Verify each story is from the provided timeline or very recent
    âœ“ Prioritize authoritative sources (major news outlets, company blogs, research institutions)
    âœ“ Each summary should explain WHAT happened, WHO is involved, and WHY it matters
    
    Example of CORRECT formatting:
    
    â€¢ **OpenAI Launches GPT-5 with Breakthrough Reasoning Capabilities** - OpenAI Blog
      Summary: OpenAI announced GPT-5, featuring enhanced reasoning abilities that surpass previous models in complex problem-solving tasks. The model shows significant improvements in mathematical reasoning, coding, and multi-step logical deduction. This release marks a major advancement in AI capabilities for enterprise and research applications.
      ğŸ”— URL: https://openai.com/blog/gpt-5-announcement

    
    QUALITY CHECKLIST before submitting:
    â–¡ Did I include 4-5 news stories?
    â–¡ Does EVERY story have a complete URL?
    â–¡ Are all stories from this week or very recent?
    â–¡ Do summaries explain the significance, not just repeat headlines?
    â–¡ Are sources credible and authoritative?
    â–¡ Is the formatting consistent across all entries?

    """,
    tools=[google_search],
    output_key="research_findings_google",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… ResearchAgent_google created.")


"""
MCP-Based GitHub Trending Agent for AI Newsletter
Using mcp-github-trending MCP Server with Google ADK

This agent discovers trending GitHub repositories and developers in the AI space.
"""

print("âœ… Starting MCP-based GitHub Trending Agent setup...")


# Configure connection to GitHub Trending MCP server
# Using uvx (recommended) - automatically downloads and runs the package
github_trending_params = StdioServerParameters(
    command="uvx",
    args=["mcp-github-trending"],
    timeout=60 
)



github_tools = MCPToolset(
    connection_params=github_trending_params
)

print(f"âœ… Connected to GitHub Trending MCP Server")

# Create the research agent with MCP tools
research_agent_github = Agent(
    name="ResearchAgent_GitHub_MCP",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent for discovering trending GitHub repositories 
    in AI and machine learning.
    
    Your task:
    1. Use the get_trending_repos tool to find the most popular AI/ML , GenAI repositories trending 
    this week or for the timeline shared by the user.
    2. Focus on repositories that are:
       - Related to AI, machine learning, deep learning, or LLMs or GenAI
       - Actively gaining stars and attention
       - From the past week (use "weekly" timeframe) or from the timeline shared  by user
    
    Tool Parameters to use:
    - language: Try "python", "jupyter-notebook", or leave empty for all languages
    - since: Use "weekly" to get repositories trending this week
    - spoken_language: Use "en" for English descriptions
    
    Multiple searches strategy:
    1. First search: General AI repositories (no language filter)
    2. Second search: Python AI projects specifically
    3. Third search: If needed, search for specific AI subfields (LLMs, computer vision, etc.)
    
    After gathering results:
    1. Analyze and identify the 3-5 most significant AI/ML repositories
    2. Create a concise summary with:
       - Repository name and full path (owner/repo)
       - Clear description of what it does
       - Star count and stars gained in the current period
       - Programming language
       - Direct GitHub URL
       - Why it's noteworthy for AI enthusiasts
    
    Format as a well-structured bulleted list with engaging descriptions.
    
    Example output format:
    â€¢ **repository-name** (owner/repository-name)
      Description of the project and its significance
      â­� 10,000 stars (+500 this week) | Language: Python
      ğŸ”— https://github.com/owner/repository-name
    """,
    tools=[github_tools],
    output_key="research_findings_github_mcp",
)
    
print(f"âœ… GitHub MCP Based tool created")





def get_youtube_trending_videos(topic: str = "artificial intelligence", max_results: int = 5) -> str:
    """
    Fetches trending YouTube videos on AI topics from the past week.
    
    Args:
        topic: The topic to search for (default: "artificial intelligence")
        max_results: Number of videos to return (default: 5)
    
    Returns:
        String with formatted video information
    """
    try:
        #Youtube API key is set 
        
        API_KEY = youtube_api_key 

        # Calculate date for the past week
        week_ago = (datetime.now() - timedelta(days=7)).isoformat() + "Z"
        
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": topic,
            "type": "video",
            "order": "viewCount",
            "publishedAfter": week_ago,
            "maxResults": max_results,
            "key": API_KEY,
            "relevanceLanguage": "en"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("items"):
            return "No trending videos found for this week."
        
        result = []
        for video in data["items"]:
            video_id = video["id"]["videoId"]
            snippet = video["snippet"]
            result.append(
                f"â€¢ {snippet['title']}\n"
                f"  ğŸ“º Channel: {snippet['channelTitle']}\n"
                f"  ğŸ“… Published: {snippet['publishedAt'][:10]}\n"
                f"  ğŸ”— https://www.youtube.com/watch?v={video_id}\n"
            )
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Error fetching YouTube videos: {str(e)}"


youtube_trending_tool = FunctionTool(get_youtube_trending_videos)

research_agent_youtube = Agent(
    name="ResearchAgent_YouTube",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent for finding trending AI videos and podcasts on YouTube.
    Use the get_youtube_trending_videos tool to discover the most popular AI-related videos from this week.

    CRITICAL INSTRUCTION: 
    - You MUST use the get_youtube_trending_videos function tool to search YouTube.
    - Do NOT use google_search.
    - Do NOT reference any previous research from other agents.
    - Do NOT use information from session state.
    - Your task is INDEPENDENT and SEPARATE from other agents.


    Your task:
    1. Call the tool to get trending videos
    2. Filter your results on detailed long videos like editorials / tutorials/ talks / webinars or podcasts and not shorts
    3. Analyze the results and identify the most valuable content
    3. Create a concise summary highlighting 3-5 top videos with:
       - Video title and creator
       - Brief description of content
       - Publication date
       - Direct link to the video (clickable URL)
       - Why it's relevant to AI enthusiasts
       - Key insights and summaries from that Video
    
    Format your output as a bulleted list with clear sections.""",
    tools=[youtube_trending_tool],
    output_key="research_findings_youtube",
)

print(f"âœ… YouTube Custom Function Based tool created")




def search_arxiv_papers(query: str = "artificial intelligence", max_results: int = 10) -> str:
    """
    Search arXiv papers directly using the official arXiv API.
    
    Args:
        query: Search query for paper titles and abstracts
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        String with formatted paper information
    """
    try:
        # Official arXiv API endpoint
        base_url = "http://export.arxiv.org/api/query"
        
        # Build search query - search in all fields and filter by Computer Science
        search_query = f"all:{query} AND cat:cs.*"
        
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",  # Sort by most recent
            "sortOrder": "descending"
        }
        
        # Make request to arXiv API
        print(f"   Searching arXiv for: '{query}'...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Namespaces for arXiv API
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        # Extract entries
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            return f"No papers found for query: {query}"
        
        print(f"   Found {len(entries)} papers")
        
        result = []
        for idx, entry in enumerate(entries, 1):
            # Extract paper details
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            
            # Authors
            authors = entry.findall('atom:author', ns)
            author_names = [a.find('atom:name', ns).text for a in authors[:3]]
            if len(authors) > 3:
                author_str = f"{', '.join(author_names)} et al."
            else:
                author_str = ', '.join(author_names)
            
            # Abstract
            abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            # Truncate abstract for readability
            if len(abstract) > 350:
                abstract = abstract[:347] + "..."
            
            # Published date
            published = entry.find('atom:published', ns).text[:10]  # YYYY-MM-DD
            
            # ArXiv ID and links
            arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Categories
            categories = entry.findall('atom:category', ns)
            cat_list = [c.get('term') for c in categories[:3]]
            
            result.append(
                f"â€¢ **{title}**\n"
                f"  ğŸ‘¥ Authors: {author_str}\n"
                f"  ğŸ“� Abstract: {abstract}\n"
                f"  ğŸ�·ï¸� Categories: {', '.join(cat_list)}\n"
                f"  ğŸ“… Published: {published}\n"
                f"  ğŸ”— arXiv: {arxiv_url}\n"
                f"  ğŸ“„ PDF: {pdf_url}\n"
            )
        
        return "\n".join(result)
        
    except requests.exceptions.Timeout:
        return "â�Œ ArXiv API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"â�Œ Error connecting to ArXiv API: {str(e)}"
    except ET.ParseError as e:
        return f"â�Œ Error parsing ArXiv response: {str(e)}"
    except Exception as e:
        return f"â�Œ Unexpected error: {str(e)}"


# Create the FunctionTool
arxiv_search_tool = FunctionTool(search_arxiv_papers)

print("âœ… ArXiv search function created")


# ============================================================================
# ARXIV RESEARCH AGENT
# ============================================================================

research_agent_arxiv = Agent(
    name="ResearchAgent_ArXiv",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent for discovering groundbreaking AI/ML research papers from arXiv.

CRITICAL INSTRUCTION:
- You MUST use the search_arxiv_papers function tool to search arXiv.
- Do NOT use google_search or any other tool.
- Do NOT reference any previous research from other agents.
- Do NOT use information from session state.
- Your task is INDEPENDENT and SEPARATE from other agents.


Your task:
1. Use the search_arxiv_papers tool to find recent AI/ML papers
2. Execute 4-5 focused searches with different queries:
   - "artificial intelligence machine learning"
   - "large language model transformer"
   - "computer vision diffusion generative"
   - "reinforcement learning deep learning"
   - "neural architecture optimization"

3. Each search returns up to 10 papers with:
   - Title and authors
   - Abstract summary
   - Categories (e.g., cs.AI, cs.LG)
   - Published date
   - arXiv and PDF URLs

4. From ALL search results (40-50 papers total), select the TOP 5 most significant papers

Selection criteria:
- Published within the date span the user has provided or within the last 2 weeks  (check dates!)
- Novel techniques or architectures
- Significant contributions to the field
- Practical applications or impact
- From reputable authors/institutions (if identifiable)
- Clear, well-written abstracts

5. For each of the TOP 5 papers, provide detailed analysis:

â€¢ **[Paper Title]**
  Authors: [Author list]
  Summary: [2-3 sentences in YOUR OWN WORDS explaining what the paper does and its contribution. Don't just copy the abstract - interpret it!]
  Key innovation: [What's novel or groundbreaking about this work?]
  Practical impact: [Why does this matter? How could it be applied?]
  Categories: [arXiv categories]
  ğŸ“„ arXiv ID: [ID] | ğŸ”— [arXiv URL]
  ğŸ“¥ PDF: [PDF URL]
  ğŸ“… Published: [Date]

CRITICAL REQUIREMENTS:
âœ“ Make 4-5 searches to get comprehensive coverage (40-50 papers total)
âœ“ ALWAYS check publication dates - prioritize papers from the last 1-2 weeks or as epr user input
âœ“ Select ONLY the 5 most impactful papers across ALL searches
âœ“ Write summaries in YOUR OWN WORDS - don't just copy abstracts
âœ“ Explain WHY each paper matters, not just WHAT it does
âœ“ Include ALL links (arXiv URL and PDF URL) for each paper
âœ“ Focus on papers with clear, significant contributions

Example of CORRECT output:

â€¢ **Scaling Laws for Neural Language Models Beyond 100 Trillion Parameters**
  Authors: Smith, J., Johnson, A., et al.
  Summary: This paper investigates how language model performance scales with model size beyond current limits, discovering new power-law relationships that remain stable even at 100T+ parameters. The authors provide empirical evidence and theoretical frameworks for predicting optimal model sizes given compute budgets. This work has immediate implications for planning next-generation LLM development and resource allocation.
  Key innovation: First empirical study of scaling behavior at extreme model sizes with validated predictive formulas
  Practical impact: Enables companies to make data-driven decisions about model architecture and compute investment for future LLMs
  Categories: cs.LG, cs.CL, cs.AI
  ğŸ“„ arXiv ID: 2024.12345 | ğŸ”— https://arxiv.org/abs/2024.12345
  ğŸ“¥ PDF: https://arxiv.org/pdf/2024.12345.pdf
  ğŸ“… Published: 2024-11-25

Quality checklist:
â–¡ Did I search 4-5 different queries?
â–¡ Did I review 40-50 papers total?
â–¡ Are all 5 selected papers from the last 1-2 weeks?
â–¡ Did I write original summaries (not copied abstracts)?
â–¡ Did I explain the practical impact of each paper?
â–¡ Are ALL URLs included for each paper?
â–¡ Is the output well-formatted and consistent?

Remember: Focus on papers that push boundaries and have real-world applications, not just incremental improvements.""",
    tools=[arxiv_search_tool],
    output_key="research_findings_arxiv",
)

print("âœ… ArXiv Research Agent created")





# This agent coordinates all 4 research subagents to run in sequence one after the other


# Create the sequential research coordinator
research_coordinator = SequentialAgent(
    name="ResearchCoordinator",
    sub_agents=[
        research_agent_google,
        research_agent_github,
        research_agent_youtube,
        research_agent_arxiv
    ]
)

print("âœ… Research Coordinator created")





# This agent takes all research findings and creates a polished newsletter

newsletter_formatter = Agent(
    name="NewsletterFormatter",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a professional newsletter writer and editor. Your job is to create a polished, engaging AI newsletter from research findings.
    
    You will receive research findings from 4 sources:
    1. research_findings_google - Top AI news articles
    2. research_findings_github_mcp - Trending GitHub repositories
    3. research_findings_youtube - Top YouTube videos
    4. research_findings_arxiv - Research papers from arXiv
    
    Your task:
    1. Review ALL the research findings
    2. Organize them into a coherent, engaging newsletter structure
    3. Write a compelling introduction
    4. Present each section with clear headers
    5. Add smooth transitions between sections
    6. Include a closing summary with key takeaways
    7. Maintain a professional yet engaging tone
    
    Newsletter Structure:
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    ğŸ“° AI WEEKLY NEWSLETTER
    [Date Range]
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    
    ğŸ‘‹ INTRODUCTION
    [Write a 2-3 paragraph introduction that:
     - Highlights the week's major themes
     - Teases the most exciting content
     - Sets an engaging tone]
    
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ğŸ”¥ TOP NEWS & ARTICLES
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    [Present the top news articles from research_findings_google
     - Keep the best 3-5 articles
     - Add context and why each matters
     - Include links]
    
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ğŸ’» TRENDING OPEN SOURCE PROJECTS
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    [Present trending GitHub repos from research_findings_github_mcp
     - Highlight 3-5 most interesting repos
     - Explain what each does and why it's trending
     - Include star counts and links]
    
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ğŸ�¥ MUST-WATCH VIDEOS
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    [Present top YouTube videos from research_findings_youtube
     - Select 3-5 most valuable videos
     - Explain what viewers will learn
     - Include channel names and links]
    
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ğŸ“š RESEARCH HIGHLIGHTS
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    [Present research papers from research_findings_arxiv
     - Feature 3-5 most significant papers
     - Explain contributions in accessible language
     - Highlight practical implications
     - Include arXiv links]
    
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ğŸ’¡ KEY TAKEAWAYS
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    [Write a closing section with:
     - 3-5 key themes or trends from the week
     - What readers should pay attention to
     - Brief forward-looking statement]
    
    â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    
    Writing Guidelines:
    - Use clear, engaging language (avoid jargon when possible)
    - Make it scannable with headers and bullets
    - Balance technical depth with accessibility
    - Be enthusiastic but professional
    - Ensure all links are included
    - Keep total length to ~1500-2000 words
    - Use emojis sparingly for visual interest
    
    Quality checklist:
    âœ“ Introduction hooks the reader
    âœ“ Each section has clear value
    âœ“ Transitions flow naturally
    âœ“ All important links included
    âœ“ Tone is consistent throughout
    âœ“ Key takeaways provide real insights
    âœ“ No grammatical errors
    âœ“ Newsletter feels cohesive and professional""",
    tools=[],
    output_key="final_newsletter",
)

print("âœ… Newsletter Formatter Agent created")




# This is the main orchestrator that manages the entire workflow

root_orchestrator = SequentialAgent(
    name="RootOrchestrator",
    sub_agents=[
        research_coordinator,        # Step 2: Run all research agents in parallel
        newsletter_formatter,        # Step 3: Format the newsletter
    ]
)

print("âœ… Root Orchestrator Agent created (Sequential workflow)")





runner = InMemoryRunner(agent=root_orchestrator)
response = await runner.run_debug(
    "Develop the AI newsletter as per the designed sequence with the sub agents for the period November 19 to November 26 ,2025. For the GitHub agent since it doesn't take into account the specific date sequence search the repos for last one week"
)


