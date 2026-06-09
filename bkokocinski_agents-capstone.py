import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
) 

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# This agent runs ONCE at the beginning to create the first draft.
initial_writer_agent = Agent(
    name="InitialWriterAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Based on the content of the provided Wikipedia page (or a topic summarized from it),
    write the first draft of a YouTube video script (around 100-150 words).
    The script should be engaging, informative, and formatted for a presenter.""",
    output_key="current_script",  # Stores the first draft in the state.
)

print("âœ… initial_writer_agent created.")



# This agent's only job is to provide feedback or the approval signal. It has no tools.
critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction=""""You are a constructive YouTube Script critic. Review the script provided below.
    Script: {current_script}
    Evaluate the script's Hook/Engagement, Information Clarity, and Flow.
    If the script is strong and ready for production, you MUST respond with the exact phrase: "APPROVED"
    Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
    output_key="critique",  # Stores the feedback in the state.
)

print("âœ… critic_agent created.")


# This is the function that the RefinerAgent will call to exit the loop.
def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the script is finished and no more changes are needed."""
    return {"status": "approved", "message": "Script approved. Exiting refinement loop."}


print("âœ… exit_loop function created.")


# This agent refines the story based on critique OR calls the exit_loop function.
refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a YouTube Script refiner. You have a script draft and a critique.
    Script Draft: {current_script} Critique: {critique}
    Your task is to analyze the critique.
    IF the critique is EXACTLY "APPROVED", you MUST call the 'exit_loop' function and nothing else.
    OTHERWISE, rewrite the Script Draft to fully incorporate the feedback from the critique.
    The rewritten script should maintain the original length constraints""",
    output_key="current_script",  # It overwrites the story with the new, refined version.
    tools=[
        FunctionTool(exit_loop)
    ],  # The tool is now correctly initialized with the function reference.
)

print("âœ… refiner_agent created.")


# The LoopAgent contains the agents that will run repeatedly: Critic -> Refiner.
story_refinement_loop = LoopAgent(
    name="StoryRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2,  # Prevents infinite loops
)

# The root agent is a SequentialAgent that defines the overall workflow: Initial Write -> Refinement Loop.
root_agent = SequentialAgent(
    name="StoryPipeline",
    sub_agents=[initial_writer_agent, story_refinement_loop],
)

print("âœ… Loop and Sequential Agents created.")



runner = InMemoryRunner(
    agent=root_agent,
    plugins=[
        LoggingPlugin()
    ],
)
response = await runner.run_debug(
    "Write a script based on this wipedia article: Magnolia is a large genus of about 210 to 340[a] flowering plant species in the subfamily Magnolioideae of the family Magnoliaceae. The natural range of Magnolia species is disjunct, with a main center in east, south and southeast Asia and a secondary center in eastern North America, Central America, the West Indies, and some species in South America. Magnolias are evergreen or deciduous trees or shrubs known for their large, fragrant, bowl- or star-shaped flowers with numerous spirally arranged reproductive parts, producing cone-like fruits in autumn that open to reveal seeds. The genus Magnolia was first named in 1703 by Charles Plumier, honoring Pierre Magnol, with early taxonomy refined by Linnaeus in the 18th century based on American and later Asian species. Modern molecular phylogenetic studies have revealed complex relationships leading to taxonomic debates about merging related genera like Michelia with Magnolia. Magnolia species are valued horticulturally for their early and showy flowering, used culinarily in various edible forms, employed in traditional medicine for their bioactive compounds like magnolol and honokiol, and harvested for timber, with hybridization enhancing desirable traits. Magnolia is an ancient genus that dates back to the Cretaceous. Fossilized specimens of M. acuminata have been found dating to 20 million years ago (mya), and fossils of plants identifiably belonging to the Magnoliaceae date to 95 mya.[4] They are theorized to have evolved to encourage pollination by beetles as they existed prior to the evolution of bees.[5] Another aspect of Magnolia considered to represent an ancestral state is that the flower bud is enclosed in a bract rather than in sepals; the perianth parts are undifferentiated and called tepals rather than distinct sepals and petals. Magnolia shares the tepal characteristic with several other flowering plants near the base of the flowering plant lineage, such as Amborella and Nymphaea (as well as with many more recently derived plants, such as Lilium). Magnolias are culturally significant symbols, serving as official flowers and trees in various regions like Shanghai, Mississippi, Louisiana, North Korea, and Seoul, and are closely associated with the Southern United States. In the arts, magnolias symbolize both beauty and resilience, as seen in the play and film Steel Magnolias, while also evoking the contrasting brutality of lynching in the song Strange Fruit and Southern stereotypes in political commentary."
    )




