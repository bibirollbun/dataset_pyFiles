import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    IMAGE_API_KEY = UserSecretsClient().get_secret("IMAGE_API_KEY")
    os.environ["IMAGE_API_KEY"] = IMAGE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import uuid
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import google_search, AgentTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("âœ… ADK components imported successfully.")


#config retry options:
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# MCP integration with Everything Server
# mcp_image_server = McpToolset(
#     connection_params=StdioConnectionParams(
#         server_params=StdioServerParameters(
#             command="npx",  # Run MCP server via npx
#             args=[
#                 "-y",  # Argument for npx to auto-confirm install
#                 "@nanana-ai/mcp-server-nano-banana",
#             ],
#             tool_filter=['text_to_image'],
#             env={
#                         "NANANA_API_TOKEN": IMAGE_API_KEY,
#                     }
#         ),
#         timeout=30,
#     )
# )
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "@pollinations/model-context-protocol"
            ]
        ),
        timeout=30,
    )
)

print("âœ… MCP Tool created")


# Create image agent with MCP integration
image_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction="Use the MCP Tool to generate images for user queries",
    tools=[mcp_image_server],
)


from google.adk.runners import InMemoryRunner

runner = InMemoryRunner(agent=image_agent)


response = await runner.run_debug("Image of a Eiffel tower", verbose=True)


from IPython.display import display, Image as IPImage
import re

for event in response:
    if event.content and event.content.parts:
        for part in event.content.parts:
            if hasattr(part, "function_response") and part.function_response:
                for item in part.function_response.response.get("content", []):
                    if "imageUrl" in item.get("text"):
                        display(IPImage(url=re.search("(?P<url>https?://[^\s]+)", item.get("text")).group("url")))


LARGE_IMAGE_THRESHOLD = 1


def generating_images(
    num_images: int, query: str,tool_context: ToolContext
) -> dict:
    """Generating Images. Requires approval if more than 1 images are to be generated (LARGE_IMAGE_THRESHOLD).

    Args:
        num_images: Number of images to generate

    Returns:
        Dictionary with 
    """

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 1: Single Image (â‰¤1) auto-approve
    if num_images <= LARGE_IMAGE_THRESHOLD:
        # Create image agent with MCP integration
        #image_agent = LlmAgent(
        #    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        #    name="image_agent",
        #    instruction="Use the MCP Tool to generate images for user queries",
        #    tools=[mcp_image_server],
        #)
        #runner = InMemoryRunner(agent=image_agent)
        #response = runner.run_debug(query)
        #for event in response:
        #    if event.content and event.content.parts:
        #        for part in event.content.parts:
        #            if hasattr(part, "function_response") and part.function_response:
        #                for item in part.function_response.response.get("content", []):
        #                    if "Image URL" in item.get("text"):
        #                        display(IPImage(url=re.search("(?P<url>https?://[^\s]+)", item.get("text")).group("url")))
        return {
            "status": "approved",
            "num_images": num_images,
            "query": query,
            "message": f"Auto-Approved with number of images: {num_images} for query: {query}."
        }

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 2: This is the first time this tool is called. Large num of images need human approval - PAUSE here.
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"âš ï¸� Large Num Of IMAGES: {num_images} TO generate. Do you want to approve?",
            payload={"num_images": num_images, "query": query},
        )
        return {  # This is sent to the Agent
            "status": "pending",
            "query": query,
            "message": f"Large Num Of IMAGES: {num_images} to generate requires approval",
        }

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 3: The tool is called AGAIN and is now resuming. Handle approval response - RESUME here.
    if tool_context.tool_confirmation.confirmed:
        #image_agent = LlmAgent(
        #    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        #    name="image_agent",
        #    instruction="Use the MCP Tool to generate images for user queries",
        #    tools=[mcp_image_server],
        #)
        #runner = InMemoryRunner(agent=image_agent)
        #response = runner.run_debug(query)
        #for event in response:
        #    if event.content and event.content.parts:
        #        for part in event.content.parts:
        #            if hasattr(part, "function_response") and part.function_response:
        #                for item in part.function_response.response.get("content", []):
        #                    if "Image URL" in item.get("text"):
        #                        display(IPImage(url=re.search("(?P<url>https?://[^\s]+)", item.get("text")).group("url")))
        return {
            "status": "approved",
            "num_images": num_images,
            "query": query,
            "message": f"Request approved with number of images: {num_images} for query: {query}."
        }
    else:
        return {
            "status": "rejected",
            "message": f"User didnt approve for generating huge num of images: {num_images} for given query: {query}",
        }


print("âœ… Long-running functions created!")


# def tool_generate_image(response: str) -> None:
#     """Tool to generate the images using the tool and query given.

#     Args:
#         response: Response given by agent.

#     Returns:
#         None
#     """
#     #runner = InMemoryRunner(agent=tool)
#     #response = runner.run_debug(query)
#     for event in response:
#         if event.content and event.content.parts:
#             for part in event.content.parts:
#                 if hasattr(part, "function_response") and part.function_response:
#                     for item in part.function_response.response.get("content", []):
#                         if "Image URL" in item.get("text"):
#                             display(IPImage(url=re.search("(?P<url>https?://[^\s]+)", item.get("text")).group("url")))


# Create image generation agent with pausable tool
image_generator_agent = LlmAgent(
    name="image_generation_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are an Image Generation assistant.
  
  When users request to generate image:
   1. Use the 'generating_images' tool with the number of images to be displyed and user's query.
   2. If the query status is 'pending', inform the user that approval is required.
   3. After receiving the final result, provide a clear summary including: 
       - Request status (approved or rejected)
       - Query request.
   4. Once Request status is 'approved', use the 'image_agent' tool along with specific Query request and count to search for the images the user wants.
   5. Give Summary where each image is displayed with title and image URL information.
   6. Keep responses concise but informative.

   If any tool returns status "error", explain the issue to the user clearly.
  """,
    tools=[FunctionTool(func=generating_images), AgentTool(image_agent)]
    #tools=[FunctionTool(func=generating_images), tool_generate_image],
)

print("âœ… Image Generator Agent created!")


# Wrap the agent in a resumable app - THIS IS THE KEY FOR LONG-RUNNING OPERATIONS!
generator_app = App(
    name="image_coordinator",
    root_agent=image_generator_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

print("âœ… Resumable app created!")


session_service = InMemorySessionService()

# Create runner with the resumable app
generator_runner = Runner(
    app=generator_app,  # Pass the app instead of the agent
    session_service=session_service,
)

print("âœ… Runner created!")


def check_for_approval(events):
    """Check if events contain an approval request.

    Returns:
        dict with approval details or None
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None


def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}\n")
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    for item in part.function_response.response.get("content", []):
                        if "imageUrl" in item.get("text"):
                            display(IPImage(url=re.search("(?P<url>https?://[^\s]+)", item.get("text")).group("url")))


def create_approval_response(approval_info, approved):
    """Create approval response message."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )


print("âœ… Helper functions defined")


# Session and Runner
# async def setup_session_and_runner():
#     session_service = InMemorySessionService()
#     session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
#     runner = Runner(agent=image_agent, session_service=session_service)
#     return session, runner


# Agent Interaction
# async def call_agent_async(query):
    # content = types.Content(role='user', parts=[types.Part(text=query)])
    # session, runner = await setup_session_and_runner()
    # events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    # async for event in events:
    #     if event.content and event.content.parts:
    #         for part in event.content.parts:
    #             if hasattr(part, "function_response") and part.function_response:
    #                 for item in part.function_response.response.get("content", []):
    #                     if "Image URL" in item.get("text"):
    #                         display(IPImage(url=re.search("(?P<url>https?://[^\s]+)", item.get("text")).group("url")))


async def run_image_workflow(query: str, auto_approve: bool = True):
    """Runs a Image generation workflow with approval handling.

    Args:
        query: User's request for Image generation.
        auto_approve: Whether to auto-approve large number of Images (simulates human decision)
    """

    print(f"\n{'='*60}")
    print(f"User > {query}\n")

    # Generate unique session ID
    session_id = f"order_{uuid.uuid4().hex[:8]}"

    # Create session
    await session_service.create_session(
        app_name="image_coordinator", user_id="test_user", session_id=session_id
    )

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 1: Send initial request to the Agent. If num_images > 1, the Agent returns the special `adk_request_confirmation` event
    async for event in generator_runner.run_async(
        user_id="test_user", session_id=session_id, new_message=query_content
    ):
        events.append(event)

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 2: Loop through all the events generated and check if `adk_request_confirmation` is present.
    approval_info = check_for_approval(events)

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    # STEP 3: If the event is present, it's a large order - HANDLE APPROVAL WORKFLOW
    if approval_info:
        print(f"â�¸ï¸�  Pausing for approval...")
        print(f"ğŸ¤” Human Decision: {'APPROVE âœ…' if auto_approve else 'REJECT â�Œ'}\n")

        # PATH A: Resume the agent by calling run_async() again with the approval decision
        async for event in generator_runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=create_approval_response(
                approval_info, auto_approve
            ),  # Send human decision here
            invocation_id=approval_info[
                "invocation_id"
            ],  # Critical: same invocation_id tells ADK to RESUME
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}\n")
            #for event in response:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_response") and part.function_response:
                        for item in part.function_response.response.get("content", []):
                            if "imageUrl" in item.get("text"):
                                display(IPImage(url=re.search("(?P<url>https?://[^\s]+)", item.get("text")).group("url")))

    

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    else:
        # PATH B: If the `adk_request_confirmation` is not present - no approval needed - image printed successfully.
        print_agent_response(events)

    print(f"{'='*60}\n")


print("âœ… Workflow function ready")





# Demo 1: It's a small order. Agent receives auto-approved status from tool
await run_image_workflow("Generate 1 image of Rose")


# Demo 2: Workflow simulates human decision: APPROVE âœ…
await run_image_workflow("Generate 2 images of Sunflower", auto_approve=True)


# Demo 3: Workflow simulates human decision: REJECT â�Œ
await run_image_workflow("Generate 5 images of food.", auto_approve=False)




