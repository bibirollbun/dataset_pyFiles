# Cell 1:

!pip install google-adk python-dotenv > /dev/null

import os
import asyncio
import gc
import aiohttp
from kaggle_secrets import UserSecretsClient

# Get API key from Kaggle secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = api_key

print("âœ… API key loaded from Kaggle secrets")


# Cell 2:

from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, FunctionTool
from google.genai import types
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.runners import InMemoryRunner

print("âœ… ADK components imported")


# Cell 3a:

retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=2,
    http_status_codes=[429, 500, 503],
)

print("âœ… Configuration set")


# Cell 3b:
# ğŸ�¯ CHANGE THESE VALUES TO SIMULATE USER INPUT

USER_TOPIC = "Explain the benefits of candles in winter."  # â†� CHANGE THIS
BLOG_LENGTH = 200  # â†� CHANGE THIS (200-2000 words)

print("ğŸ�¯ USER INPUT SIMULATION")
print("=" * 50)
print(f"ğŸ“� Topic: {USER_TOPIC}")
print(f"ğŸ“� Length: {BLOG_LENGTH} words")
print("=" * 50)
print("ğŸ’¡ To change topic/length: Edit the values above and re-run this cell")


# Cell 4:

def create_agents(blog_length):
    
    initial_research_agent = Agent(
        name="InitialResearchAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config,
        ),
        instruction="""Based on the user's prompt, gather novel, surprising, relevant and high-quality information 
        from your knowledge base and the web to assist in building the context to help write an inspiring
        yet accurate blog post. Focus on novelty, key facts, lateral connections into other areas
        and sources. Keep it concise yet diversity rich.""",
        output_key="research_info",
        tools=[google_search],
    )

    initial_writer_agent = Agent(
        name="InitialWriterAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config,
        ),
        instruction=f"""Based on the user's prompt and the {{research_info}}, write the 
        first draft of a blog with max {blog_length + 50} words.
        Output only the blog text and title, with no introduction or explanation.""",
        output_key="current_blog",
    )

    critic_agent = Agent(
        name="CriticAgent",
        model=Gemini(
            model="gemini-2.5-flash",
            retry_options=retry_config,
        ),
        instruction="""You are a constructive but nerdish blog critic. 
        FACT CHECK the blog {current_blog} comparing to knowledge base and web-sources 
        and point out wrong or inaccurate facts. Then critique the blog: 
        - Evaluate the blog's logic structure, it's unique or novel information, and pacing.
        - Evaluate if the blog is well-written and complete. If you agree, you MUST respond with the exact phrase: "APPROVED"
        - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
        output_key="critique",
        tools=[google_search],
    )

    def exit_loop():
        return {
            "status": "approved",
            "message": "Story approved. Exiting refinement loop.",
        }
# This agent allows to define specific text styles, corporate branding or degree of humor
    
    refiner_agent = Agent(
        name="RefinerAgent",
        model=Gemini(
            model="gemini-2.5-flash",
            retry_options=retry_config,
        ),
        instruction=f"""You are a blog refiner. You have a blog draft and critique.

        Blog Draft: {{current_blog}}
        Critique: {{critique}}

        Your task is to analyze the critique.
        - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function.
        - OTHERWISE, rewrite the blog draft to fully incorporate the feedback from the critique.

        Requirements:
        - Unique examples, tension and surprising angles
        - Practical actionable content
        - Dissent or nuance
        - Game theory or tradeoffs where appropriate
        - A hint of British humour              
        - No repetition

        Max length is {blog_length} words.
        """,
        output_key="current_blog",
        tools=[FunctionTool(exit_loop)],
    )

    story_refinement_loop = LoopAgent(
        name="StoryRefinementLoop",
        sub_agents=[critic_agent, refiner_agent],
        max_iterations=2,
    )

    quality_agent = Agent(
        name="QualityAgent",
        model=Gemini(
            model="gemini-2.5-flash",
            retry_options=retry_config,
        ),
        instruction="""You determine the quality of a blog. You have a current blog {current_blog}.

        Judge based on:
        - Novel content compared to typical blogs
        - Factual accuracy
        - Writing style
        - Practical actionable insights
        - Depth and nuance
        - Overall value

        Provide a rating from 1 to 10. Provide the number only.

        Blog Draft: {current_blog}
        """,
        output_key="quality_rating",
        tools=[google_search],
    )

    root_agent = SequentialAgent(
        name="BlogPipeline",
        sub_agents=[
            initial_research_agent,
            initial_writer_agent,
            story_refinement_loop,
            quality_agent,
        ],
    )
    
    return root_agent

print("âœ… Agent creation function defined")


# Cell 5:

async def generate_blog(user_prompt, blog_length):
    print(f"ğŸ�¬ Generating {blog_length}-word blog for: '{user_prompt}'")
    print("=" * 60)

    # Create fresh agents with the specified blog length
    root_agent = create_agents(blog_length)
    
    runner = None
    try:
        runner = InMemoryRunner(agent=root_agent)

        response = await asyncio.wait_for(
            runner.run_debug(user_prompt),
            timeout=300,
        )

        print("\n" + "=" * 60)
        print("ğŸ“� FINAL BLOG")
        print("=" * 60)

        # Extract the final blog from the response - improved logic
        final_blog = None
        blog_versions = []
        
        for event in response:
            # Look for blog content in state changes
            if hasattr(event, "actions") and event.actions and event.actions.state_delta:
                state = event.actions.state_delta
                if "current_blog" in state and state["current_blog"]:
                    content = state["current_blog"].strip()
                    # Only capture substantial blog content, not approval messages
                    if (len(content) > 100 and 
                        not content.startswith("APPROVED") and 
                        "approved" not in content.lower() and
                        "exiting" not in content.lower() and
                        "exit" not in content.lower()):
                        blog_versions.append(("state", content))
            
            # Also look for blog content in agent outputs
            if hasattr(event, "agent_name") and hasattr(event, "content"):
                agent_name = event.agent_name
                if (agent_name == "InitialWriterAgent" or 
                    agent_name == "RefinerAgent" or 
                    agent_name == "BlogPipeline"):
                    if hasattr(event, "content") and event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                text = part.text.strip()
                                # Capture substantial blog content
                                if (len(text) > 100 and 
                                    not text.startswith("APPROVED") and 
                                    "approved" not in text.lower() and
                                    "exiting" not in text.lower() and
                                    "exit" not in text.lower()):
                                    blog_versions.append((f"agent_{agent_name}", text))

        # Get the final blog - prefer the last one found
        if blog_versions:
            # Take the last blog version (most recent)
            final_blog = blog_versions[-1][1]
            print(f"âœ… Found {len(blog_versions)} blog versions, using the most recent")
        else:
            # Fallback: try to find any blog content in the entire response
            for event in response:
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            text = part.text.strip()
                            if len(text) > 200:  # Likely a blog post
                                final_blog = text
                                break
                    if final_blog:
                        break

        if final_blog:
            # Save to file for Kaggle output
            with open('/kaggle/working/generated_blog.txt', 'w') as f:
                f.write(final_blog)
            print("ğŸ’¾ Blog saved to 'generated_blog.txt'")
            
            # Display the final blog content
            print("\n" + "=" * 60)
            print("FINAL BLOG:")
            print("=" * 60)
            print(final_blog)
        else:
            print("â�Œ Could not extract final blog from response")
            print("Available blog versions found:")
            for source, content in blog_versions:
                print(f"\n--- From {source} ---")
                print(content[:500] + "..." if len(content) > 500 else content)

        print("\n" + "=" * 60)
        print("âœ… Blog generation complete")

    except asyncio.TimeoutError:
        print("â�° TIMEOUT: Operation took too long (5+ minutes)")
    except Exception as e:
        print(f"â�Œ ERROR: {e}")
    finally:
        # Enhanced cleanup to prevent unclosed session warnings
        cleanup_attempts = 0
        max_cleanup_attempts = 3
        
        while cleanup_attempts < max_cleanup_attempts:
            try:
                # Close runner first
                if runner is not None and hasattr(runner, "close"):
                    await runner.close()
                    print("ğŸ”’ Runner closed successfully")
                
                # Force garbage collection multiple times
                for _ in range(3):
                    await asyncio.sleep(0.1)
                    gc.collect()
                
                # Close any remaining aiohttp sessions and connectors
                sessions_closed = 0
                connectors_closed = 0
                
                # Get all objects and close sessions/connectors
                objects_to_check = list(gc.get_objects())
                for obj in objects_to_check:
                    try:
                        # Close ClientSessions
                        if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
                            await obj.close()
                            sessions_closed += 1
                        # Close TCPConnectors
                        elif isinstance(obj, aiohttp.TCPConnector) and not obj.closed:
                            await obj.close()
                            connectors_closed += 1
                    except (AttributeError, RuntimeError, Exception):
                        pass  # Ignore any errors during cleanup
                
                if sessions_closed > 0 or connectors_closed > 0:
                    print(f"ğŸ”’ Cleaned up {sessions_closed} sessions and {connectors_closed} connectors (attempt {cleanup_attempts + 1})")
                
                # If we didn't find anything to clean, break early
                if sessions_closed == 0 and connectors_closed == 0:
                    break
                    
            except Exception as e:
                print(f"âš ï¸� Warning during cleanup attempt {cleanup_attempts + 1}: {e}")
            
            cleanup_attempts += 1
        
        # Final garbage collection
        gc.collect()
        print("ğŸ”’ Cleanup completed")


print("ğŸš€ Starting Corporate Blogger with your chosen topic...")
await generate_blog(USER_TOPIC, BLOG_LENGTH)

