# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


%pip install feedparser


import os
import feedparser
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool


try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext

print("âœ… ADK components imported successfully.")


# --------------------------------------------------------------------------------
# TOOL DEFINITION (The "Agent" Capability)
# --------------------------------------------------------------------------------

def fetch_aws_announcements():
    """
    Fetches the latest announcements from the AWS What's New RSS feed.
    Returns a list of the 15 most recent announcements with titles and summaries.
    """
    rss_url = "https://aws.amazon.com/about-aws/whats-new/recent/feed/"
    print(f"\n[System] Agent is fetching live data from: {rss_url}...")
    
    feed = feedparser.parse(rss_url)
    
    if feed.bozo:
        return "Error: Could not parse the AWS RSS feed. It might be down or invalid."

    news_items = []
    # Get top 15 entries to keep context window manageable but informative
    for entry in feed.entries[:15]:
        item = {
            "title": entry.title,
            "date": entry.published,
            "summary": entry.description, # AWS feed usually puts the body here
            "link": entry.link
        }
        news_items.append(item)

    return news_items
print("fetch announcements defined.")
print(f"ğŸ’³ Test: {fetch_aws_announcements()}")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
print(retry_config)


# AWS bews feed agent with custom function tools
aws_news_feed_agent = LlmAgent(
    name="aws_news_feed_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
        You are an expert AWS Cloud Assistant. 
        Your goal is to answer user questions about the latest AWS features and announcements. 
        You have access to a tool 'fetch_aws_announcements' which retrieves real-time data from the AWS RSS feed. 
        ALWAYS use this tool if the user asks about 'latest news', 'recent updates', or specific new features. 
        When answering, provide the specific date of the announcement if available. 
        Also give the complete webpage so that I can check for more details.
        If the user asks a question not related to recent news, you can answer from your general knowledge.
        """,
    tools=[fetch_aws_announcements]
)

print("âœ… AWS fetch news feed agent created with custom function tools")
print("ğŸ”§ Available tools:")
print("  â€¢ fetch_aws_announcements - Fetches the latest aws news announcements")



# Test the aws news feed agent
aws_news_feed_runner = InMemoryRunner(agent=aws_news_feed_agent)
_ = await aws_news_feed_runner.run_debug(
    "Summarize the latest 2 announcements from AWS rss news feed"
)


# Test the aws news feed agent
aws_news_feed_runner = InMemoryRunner(agent=aws_news_feed_agent)
_ = await aws_news_feed_runner.run_debug(
    "Summarize the latest lambda and s3 announcements"
)

