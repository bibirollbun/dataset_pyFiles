ğŸ�‹ The Used Car "Lemon Detector" Agent
A Hierarchical Agentic Workflow for Safer Car Buying

1. ğŸš¨ Problem Statement
"Pristine condition." "Drives like a dream." "First to see will buy."

We have all seen used car ads like this. They are often filled with subjective marketing fluff designed to distract from the mechanical reality of the vehicle. Buying a used car is one of the most stressful financial decisions people make because of a fundamental Information Asymmetry: the dealer knows the car's secrets, but the buyer doesn't.

Crucial data existsâ€”in government MOT history databases, owner forums, and reliability indicesâ€”but it is fragmented. A buyer has to manually cross-reference a dealer's glowing description against a dry, technical list of past mechanical failures. It is tedious, technical, and all too easy to miss a "red flag" that could cost thousands in repairs later.

The Goal: Build an AI Agent that instantly cuts through the marketing noise to assess the mechanical reality of a vehicle before the buyer even steps foot in the dealership.

2. ğŸ¤– The Solution: Why Agents?
Standard automation cannot solve this problem effectively. A simple script might be able to scrape data, but it cannot understand context. It doesn't know that a "suspension clunk" mentioned in a forum is related to the "advisory on near-side pin" in an MOT report.

Agents are the perfect solution because this task requires three distinct cognitive steps:

Contextual Understanding: Separating facts (trim level, warranty) from fluff ("wow look at this!").

Dynamic Research: Actively "Googling" common faults for specific models (e.g., Nissan Juke vs. Honda Jazz) in real-time.

Synthesis & Judgment: Weighing the dealer's subjective promise against objective government data to render a verdict (Green / Amber / Red). This requires maintaining state and context across a multi-turn conversation.

3. ğŸ�—ï¸� Architecture & Design
This project utilizes a Hierarchical Agentic Workflow built with the Google Agent Development Kit (ADK) and powered by Gemini 2.5 Flash-Lite.

The Team Structure
Instead of one giant prompt, I created a team of specialized tools coordinated by a central intelligence:

The Brain (Main Agent): A "Consultant" agent that manages the user session. It possesses Gatekeeper Logic, refusing to give a verdict unless it has sufficient data (Description + MOT History).

The Scribe (Data Processor Tool): A specialized sub-agent wrapped as a tool. It takes raw, messy text pastes and structures them into clean JSON-like facts, calculating Pass/Fail ratios.

The Detective (Researcher Tool): A tool equipped with Google Search. It looks for model-specific common faults (e.g., "Nissan Juke DCT gearbox issues") to detect known "ticking time bombs."

The Memory (Session Management): A robust state-management layer (InMemoryRunner) that enables comparative analysis across different vehicles in the same session.

System Diagram
Code snippet

graph TD
    User([User]) -->|Chat Input| Main[Main Agent: The Consultant]
    
    subgraph "Session State (InMemoryRunner)"
        Memory[(Conversation History)]
    end
    
    Main <-->|Read/Write| Memory
    
    subgraph "Tool Belt"
        Processor[Data Processor Tool]
        Researcher[Researcher Tool]
    end
    
    Main -->|Delegate: Clean Text| Processor
    Main -->|Delegate: Check Faults| Researcher
    
    Researcher -->|Query| Google[Google Search]
    Processor -->|Extract| Facts[Structured Specs & MOT Stats]
    
    Facts --> Main
    Google --> Main
    
    Main -->|Synthesized Verdict| User
4. ğŸ�“ Key Concepts Applied
This submission demonstrates the following advanced agent concepts:

âœ… Multi-Agent System: A hierarchical structure where a Main Agent delegates tasks to specialized sub-agents (Processor and Researcher).

âœ… Sessions & Memory: Implemented InMemorySessionService with namespace introspection to solve persistence issues in Notebook environments. The agent remembers previous cars to answer questions like "How does this compare to the first car?"

âœ… Tools: Integration of built-in tools (Google Search) and custom-built tools (wrapping an agent as an AgentTool).

âœ… Async Streaming: Implemented async/await patterns to stream the agent's thought process in real-time within the Notebook.

5. ğŸ› ï¸� Setup & Usage Instructions
Prerequisites
To run this notebook, you need a Google Gemini API Key saved in Kaggle Secrets.

Add API Key: Go to Add-ons -> Secrets -> Add a new secret. Label it GOOGLE_API_KEY and paste your key.

Install Dependencies: Run the first cell to install google-adk and google-genai.

Run All: Execute all cells in order. The final cell will start the interactive chat loop.

How to use the tool
Once the interactive loop starts:

Paste the Dealer's Description: Copy the text from an Autotrader or eBay ad.

Paste the MOT History: Copy the table data from the UK Government MOT check website.

Get the Verdict: The agent will analyze the data, check for known faults, and give you a Green, Amber, or Red light rating.

6. ğŸš€ Future Roadmap
If I had more time, I would expand this agent to become a fully autonomous "Hunter":

Direct API Integration: Connect directly to the UK Government MOT API to remove the need for copy-pasting.

Computer Vision: Add a tool to analyze car photos for rust, panel gaps, or mismatched paint.

Negotiation Mode: Allow the agent to draft an email to the dealer using the identified "Red Flags" as leverage for a price reduction.


# Install dependencies
pip install google-adk
pip install phi
pip install phidata sqlalchemy


# Configure Gemini API Key
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# Import ADK components
import os
from typing import Dict
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


# Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Onboarding guide
def print_onboarding_guide():
    guide = """
    ğŸš— **Used Car Analysis Tool - Ready**
    
    To get a Green/Red light verdict, please provide the following:
    
    1. **Car Spec:** (e.g., 2015 Ford Fiesta 1.0 Ecoboost)
    2. **Price:** (e.g., Â£5,000)
    3. **Ad Description:** (Copy and paste the dealer's text here)
    4. **MOT History:** (Copy and paste the table from the gov.uk website)
    
    ğŸ‘‰ *Paste the information below and press Enter!*
    """
    print(guide)

# Run this once at startup
print_onboarding_guide()


# Resaercher Agent: Its job is to use the google_search tool and present findings.
research_agent = Agent(
    name="Research_Assistant",
    model=Gemini(
        id="gemini-2.5-flash-lite", 
        retry_options=retry_config
    ),
    tools=[google_search],
    instruction="""
    You are a factual research assistant.
    1. When asked about a Dealer: Search for "Dealer Name + Reviews" on Trustpilot, Google Maps, and forums. Summarize the sentiment.
    2. When asked about Costs: Search for "Car Model + Year + MPG Real World" and "Car Model + Tax Band UK".
    3. When asked about Faults: Search for "Car Model + Common Problems".
    Always return concise summaries with sources.
    """
)

# Wrap it as a Tool
# This allows Main Agent to say: "I need to call the Research_Assistant tool."
research_tool = AgentTool(agent=research_agent) 

print("âœ… Research Agent created and wrapped as a Tool.")


# Processer Agent: Its job is to process the MOT history and dealer description to transform them into structured, clean data.
processor_agent = Agent(
    name="Data_Processor",
    model=Gemini(
        id="gemini-2.5-flash-lite", 
        retry_options=retry_config
    ),
    # Note: No external tools needed (like Google Search). 
    # Its only job is logic and text processing.
    instruction="""
    You are an expert Data Analyst for used cars. 
    Your job is to take raw, messy text (Dealer descriptions and MOT history) and extract structured facts.
    
    Output a clean summary including:
    1. **Vehicle Specs**: Make, Model, Year, Engine Size, Transmission.
    2. **MOT Health Check**: 
       - Calculate the approximate Pass/Fail ratio.
       - Identify RECURRING faults (e.g., "Suspension bushes mentioned in 2020, 2021, and 2022").
       - Flag 'Red Flags' specifically: Corrosion/Rust, Oil Leaks, Mileage discrepancies (clocking).
    3. **Dealer Claims**: Extract key value-adds like "Full Service History (FSH)", "Timing belt changed", or "Warranty included". Ignore subjective fluff like "Drives like a dream".
    
    If data is missing, explicitly state "Not mentioned in text".
    """
)

# Wrap it as a tool for the Main Agent
processing_tool = AgentTool(agent=processor_agent)

print("âœ… Data Processing Agent created and wrapped.")


# Main Agent
main_agent = Agent(
    name="Car_Buying_Consultant",
    model=Gemini(id="gemini-2.5-flash-lite"),
    tools=[research_tool, processing_tool],     
    instruction="""
    You are an expert Used Car Consultant. Your goal is to protect the buyer from lemons.

    ### MEMORY & COMPARISON INSTRUCTIONS:
    You have access to the conversation history. 
    1. **Track Candidates:** Keep a mental list of cars the user has already shown you in this session.
    2. **Compare:** When analyzing a new car, explicitly compare it to the previous ones if they were in the same category. 
       - Example: "This BMW has higher mileage than the Audi you showed me earlier, but a better service history."
    3. **Context Management:** Since the chat history is shortened, if you see a car that is a strong "Green Light," summarize its details briefly in your final verdict so you don't lose track of it in future turns.

    ## PHASE 0: INPUT VALIDATION (CRITICAL)
    Before calling ANY tools, check if the user has provided the necessary data.
    To perform a proper analysis, you need at least two of the following:
    1. **The Car Details** (Make, Model, Year, Engine).
    2. **The Dealer Description** (The text from the ad).
    3. **The MOT History** (The list of passes/fails).
    
    **IF DATA IS MISSING:**
    Do not hallucinate. Do not guess. 
    Politely ask the user to provide the missing pieces. Tell them: "To give you a safety rating, I need the MOT history and the ad description. Please paste them here."
    
    ### YOUR WORKFLOW:
    1. **Analyze Data**: When given raw text/links, ALWAYS use the `Data_Processor` tool first to clean data, and extract facts and red flags.
    2. **Verify Facts**: Use the `Research_Assistant` tool to check the specific car model's common faults and the dealer's reputation.
    3. **Synthesize**: Compare the 'Data Processor' findings (what the car is) with the 'Research' findings (what the car SHOULD be).
    
    ### FINAL OUTPUT FORMAT:
    You must output your final answer strictly in this structure:

    ## 1. Executive Summary: The Verdict
    (One concise paragraph. Is this a gem or a grenade?)

    ## 2. Analysis: Description vs. MOT History
    (Does the dealer's story match the government data? Highlight inconsistencies.)

    ## 3. Red Flags
    * **Mechanical:** (e.g., Corrosion, Oil leaks)
    * **Administrative:** (e.g., Gaps in history, short ownership)
    * **Model Specific:** (e.g., "This engine is known for timing chain failures")

    ## 4. Overall Condition Assessment
    (Combine the physical condition with the reputation of the model.)

    ## 5. Action Plan & Questions for the Dealer
    * (Question 1)
    * (Question 2)
    * (Next step for the buyer)

    ## 6. Final Verdict: [GREEN / AMBER / RED]
    (If not Green, explicitly state what evidence is needed to upgrade it.)

    # ALWAYS end your response with a hidden section:
    [[Summary of Top Contenders]]:
    1. 2012 BMW - Red Flag (Corrosion)
    2. 2015 Ford - Green Light (Price Â£5k)
    
    Carry this summary forward in your mind for future comparisons.
    """
)

print("âœ… Main Agent created with Reporting Standards.")




# To handle the intricacies of the InMemoryRunner namespace in this notebook environment, 
# I implemented an introspection method that auto-detects the runner's app configuration to ensure session persistence is never lost

import sys
import uuid
import asyncio
from google.genai import types
from google.adk.runners import InMemoryRunner

# --- DEFINING THE WRAPPER ---
async def start_aligned_session():
    # 1. SETUP
    runner = InMemoryRunner(agent=main_agent)
    
    # --- STEP 1: INTROSPECTION (FIND THE HIDDEN APP NAME) ---
    # We stop guessing. We look inside the runner to see what name IT wants.
    target_app_name = "default" # Fallback
    
    # Check common internal attributes where the name might be hidden
    if hasattr(runner, 'app_name'):
        target_app_name = runner.app_name
    elif hasattr(runner, '_app_name'):
        target_app_name = runner._app_name
        
    print(f"ğŸ•µï¸� DETECTED RUNNER APP NAME: '{target_app_name}'")
    
    # 2. CREATE SESSION (Using the detected name)
    session_id = f"session-{str(uuid.uuid4())[:8]}"
    user_id = "local_user"

    print(f"â�³ Creating Session: {session_id} in App: '{target_app_name}'...")

    try:
        await runner.session_service.create_session(
            app_name=target_app_name,
            user_id=user_id,
            session_id=session_id
        )
        print("âœ… Session Memory Active!")
        
    except Exception as e:
        print(f"â�Œ Setup Failure: {e}")
        return

    # 3. CHAT LOOP
    print(f"\nğŸš— Car Consultant Ready!")
    print("--------------------------------------------------")
    print("ğŸ“‹ INSTRUCTIONS:")
    print("1. Paste the Car Ad/Description text and press Enter.")
    print("2. Wait for the Consultant to ask for the MOT history.")
    print("3. Paste the MOT history text and press Enter.")
    print("--------------------------------------------------\n")

    while True:
        try:
            user_input = input("You: ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        if user_input.lower() in ["exit", "quit"]:
            print("Bye!")
            break
        
        if not user_input.strip():
            continue
            
        print("\nğŸ¤– Consultant is thinking... (This may take 10-20s)\n")
        
        try:
            message_content = types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )

            # 4. RUN AGENT
            # We assume the runner uses its own internal app_name, which matches what we just used.
            response_stream = runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message_content
            )

            async for event in response_stream:
                try:
                    if hasattr(event, 'text') and event.text:
                        print(event.text, end="", flush=True)
                    elif hasattr(event, 'part') and hasattr(event.part, 'text'):
                            print(event.part.text, end="", flush=True)
                    elif hasattr(event, 'content') and event.content.parts:
                        print(event.content.parts[0].text, end="", flush=True)
                except:
                    pass
            
            print("\n")
            
        except Exception as e:
            print(f"â�Œ Error during chat: {e}")
            
            # --- FINAL FALLBACK DIAGNOSTIC ---
            print("\nğŸ”� DEBUGGING DUMP:")
            try:
                # If it failed, print ALL keys in the storage so we see where it went
                if hasattr(runner.session_service, '_sessions'):
                    store = runner.session_service._sessions
                    print(f"   Available App Names in Store: {list(store.keys())}")
                    for app in store:
                         print(f"   -> Sessions in '{app}': {list(store[app].keys())}")
            except:
                pass
            
            import traceback
            traceback.print_exc()
            break 

    print("\n" + "-"*50 + "\n")

# --- EXECUTE ---
await start_aligned_session()


# Example 1: A Nissan Juke I found on Autotrader
ğŸ•µï¸� DETECTED RUNNER APP NAME: 'InMemoryRunner'
â�³ Creating Session: session-35ddebb4 in App: 'InMemoryRunner'...
âœ… Session Memory Active!

ğŸš— Car Consultant Ready!
--------------------------------------------------
ğŸ“‹ INSTRUCTIONS:
1. Paste the Car Ad/Description text and press Enter.
2. Wait for the Consultant to ask for the MOT history.
3. Paste the MOT history text and press Enter.
--------------------------------------------------

You:  Autotrader carsSkip to contentSkip to footer Saved Sign in CarsVansBikesMotorhomesCaravansTrucksFarmPlantElectric bikes Main site menuVehicle typesCurrently in the cars channel Used cars New cars Sell your car Value your car Car reviews Car leasing Electric cars Buy a car online  Back to top   Key information   Pricing   Overview   Description   Key features   Running costs   Vehicle history   Meet the seller   Before you buy   Expert reviews   Back to results    Front Right  Rear Left  Side Right  Front  Rear  Interior Front  Infotainment System   Gallery From Sytner Select Warrington  Warrington 3 miles away 4.4  More seller information Nissan Juke 1.0 DIG-T Tekna DCT Auto Euro 6 (s/s) 5dr Â£14,999  Â£334 below market average   Includes admin fee   Great price Get it on finance Â£360.06 per month*  Use our finance calculator to get a more personalised monthly cost.  *admin fee not included in monthly price  Use the finance calculator Overview Mileage  13,002 miles  Registration  2020 (70 reg)  Fuel type  Petrol  Body type  SUV  Engine  1.0L  Gearbox  Automatic  Doors  5  Seats  5  Emission class  Euro 6  Body colour  Black   View all spec and features Description One owner from new | 360 Camera | Rear View Camera | Satellite Navigation | BOSE Sound System | 19" Alloy Wheels | Power Folding Door Mirrors | DAB Digital Radio | Adaptive Cruise Control | Heated Seats | Climate Control | Privacy Glass | Sytner Select Warranty | Sytner Select Vehicle | 100's more available, visit sytner.co.uk/sytner-select/ to view our full range.   Read full description Key features  Infotainment System  Driver Instruments  Infotainment System  Infotainment System  Boot  Infotainment System  Infotainment System  Running costs  Better fuel economy  Lower emissions COâ‚‚ emissions  143g/km  Insurance group  12E  Tax per year  Â£195  Get an insurance quote From our trusted partner MoneySuperMarket  View all running costs This vehicleâ€™s history Owners  Contact seller  Keys  Contact seller  Service history  Contact seller  Basic history check 5 checks passed   Not recorded as stolen  Not recorded as scrapped  Not imported from another country  Not exported out of the UK  Never been written off  View all checks and history Meet the seller Seller Logo Sytner Select Warrington  Warrington  â€¢  3 miles  4.4 Visit seller website (01925) 903142 Message  Get directions  Show location Visit seller profile Our most recent awards  Retailer Awards 2019  WINNER  View more seller information Before you buy Work out some of the most important costs for this car before you go ahead  Get a part exchange quote Get a free, no-commitment Autotrader guide price for your old car Use the finance calculator See how much this car might cost per month with finance provided by the seller Expert reviews for the Nissan Juke 4  This rating comes from our Autotrader vehicle experts, and is based on running costs, reliability, safety, comfort, features and power.  Read our experts review More vehicles from this seller Carousel slide 1 SEAT Arona 1.5 TSI EVO FR Sport Euro 6 (s/s) 5dr  Â£11,099  Lower price 5 seats 5 doors SUV Petrol Manual 1.5 litres 20  Carousel slide 2 Vauxhall Insignia 1.5i Turbo GPF SRi VX Line Nav Grand Sport Euro 6 (s/s) 5dr  Â£12,749  Lower price 5 seats 5 doors Hatchback Petrol Manual 1.5 litres 20  Carousel slide 3 Vauxhall Insignia 1.5i Turbo GPF SRi Nav Grand Sport Euro 6 (s/s) 5dr  Â£7,849  Good price 5 seats 5 doors Hatchback Petrol Manual 1.5 litres 20  Carousel slide 4 Land Rover Range Rover Velar 2.0 P250 R-Dynamic SE Auto 4WD Euro 6 (s/s) 5dr  Â£22,949  Good price 5 seats 5 doors SUV Petrol Automatic 2.0 litres 20  View more vehicles from Sytner Select Warrington Buying a car safely Learn how to stay safe and protect your money with our handy guide  Read our guide on buying safely Monthly finance price example Â£360.06  per month (HP)  What is HP / CS finance? Get a part exchange quote Representative example:  Monthly payments Â£360.06, Term 48 months, Contract length 48 months, Car price Â£14,900, Cash deposit Â£1,500, Annual mileage 10000, Total amount of credit Â£13,400, Total amount payable Â£18,782.88, Representative APR 13.9%, Total charges payable Â£3,882.88, Fixed rate of interest pa 13.05%, Option to purchase fee Â£10.  Enquire now Work out your monthly payment Finance available with Car Shops Limited  Car Shops Limited trading as Sytner Select Warrington is authorised and regulated by the Financial Conduct Authority (firm reference number is 447727). Car Shops Limited is a credit broker and not a lender.   Who can get car finance?  How do sellers offer finance?  Will the seller earn a commission?  Will Autotrader earn commission? Contact seller (01925) 903142 Message More ways to connect Sytner Select Warrington  Warrington  â€¢  3 miles  Visit seller website  Back to topof the page  Security advice Contact us About Autotrader Careers Investor information Privacy policies and terms Terms & conditions External wellbeing support Manage cookies Products & services Buying advice Quick search Autotrader for dealers Help us improve our website  Send feedback Copyright Â© Auto Trader Limited 2025. Auto Trader Limited (trading as Autotrader) is authorised and regulated by the Financial Conduct Authority. Our FCA firm reference number is 735711. Our FCA authorisation includes credit broking and insurance introductions. We are not a lender. Read more about our role and about fees and commissions Registered office and headquarters 4th Floor 1 Tony Wilson Place Manchester M15 4FN United Kingdom Registered number: 03909628  Back Description One owner from new | 360 Camera | Rear View Camera | Satellite Navigation | BOSE Sound System | 19" Alloy Wheels | Power Folding Door Mirrors | DAB Digital Radio | Adaptive Cruise Control | Heated Seats | Climate Control | Privacy Glass | Sytner Select Warranty | Sytner Select Vehicle | 100's more available, visit sytner.co.uk/sytner-select/ to view our full range.  Vehicle registered: 04/11/2020 Buying with Sytner Select Warrington At Sytner Select Warrington we take pride in displaying over 300 individually chosen vehicles. Our experienced Team will be able to guide you through a relaxed sales experience, allowing you to view fully mechanically and cosmetically prepared vehicles, all supplied at competitive prices alongside a Sytner Select Warranty that can cover your chosen vehicle for up to three years.   As part of the Sytner Group we have access to some of the worlds leading brands, so whether your shopping for a Luxury 4x4 or an Economical Hatchback we feel sure we can offer you the very best of choice.   We look forward to seeing you soon

ğŸ¤– Consultant is thinking... (This may take 10-20s)

WARNING:google_genai.types:Warning: there are non-text parts in the response: ['thought_signature', 'function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
To give you a safety rating, I need the MOT history. Please paste it here. In the meantime, I will process the car details and dealer description you've provided.

None
WARNING:google_genai.types:Warning: there are non-text parts in the response: ['thought_signature'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
Thank you for providing the car advertisement. I have processed the details:

**Vehicle:** Nissan Juke 1.0 DIG-T Tekna DCT Auto Euro 6 (s/s) 5dr
**Year:** 2020 (Registered 04/11/2020)
**Mileage:** 13,002 miles
**Engine:** 1.0L Petrol
**Transmission:** Automatic
**Ownership:** One owner from new (dealer claim)
**Key Features:** 360 Camera, Rear View Camera, Satellite Navigation, BOSE Sound System, 19" Alloy Wheels, Adaptive Cruise Control, Heated Seats, Climate Control, Privacy Glass.
**History Check:** Passed basic history checks (not stolen, scrapped, imported, exported, or written off).
**Warranty:** Sytner Select Warranty included.

To give you a comprehensive safety rating, I still need two crucial pieces of information:

1.  **The full MOT history:** Please paste the complete MOT history for this vehicle. This will show any past advisories or failures.
2.  **The service history:** The ad states "Contact seller" for service history. Please get these details from the dealer.

Without the MOT history, I cannot assess the vehicle's long-term mechanical health and identify potential recurring issues. Without the service history, I cannot verify if the car has been maintained properly according to the manufacturer's schedule.

Please provide these details so I can complete the analysis.

You:  Check the MOT history of a vehicle Help us make this service better. Give us your feedback.  Back MJ70 HLX NISSAN JUKE TEKNA DIG-T S-A Check another vehicle  Colour Black Fuel type Petrol Date registered 4 November 2020 MOT valid until 18 September 2026 Get an MOT reminder by email or text.  Download test certificates  You can get information corrected on your MOT history (such as mileage or vehicle details) if it's wrong. Show all sections  MOT history Check mileage recorded at test, expiry date, and test outcome Hide Date tested 19 September 2025 PASS Mileage 14,545 miles Test location View test location MOT test number 8870 1501 1333 Expiry date 18 September 2026 Date tested 11 September 2025 FAIL Mileage 14,544 miles Test location View test location MOT test number 3060 5864 9285 Do not drive until repaired (dangerous defects): Nearside Front Tyre has a tear, caused by separation or partial failure of its structure cut in tyre to chords (5.2.3 (d) (ii)) Offside Front Tyre has a tear, caused by separation or partial failure of its structure (5.2.3 (d) (ii)) Repair immediately (major defects): Nearside Front Windscreen wiper does not clear the windscreen effectively (3.4 (b) (ii)) Offside Front Windscreen wiper does not clear the windscreen effectively (3.4 (b) (ii)) Nearside Rear Tyre has a cut in excess of the requirements deep enough to reach the ply or cords (5.2.3 (d) (i)) Monitor and repair if necessary (advisories): Offside Rear Tyre slightly damaged/cracking or perishing (5.2.3 (d) (ii)) What are defects and advisories? Date tested 21 October 2024 PASS Mileage 12,461 miles Test location View test location MOT test number 3088 8599 0626 Expiry date 3 November 2025 Date tested 25 October 2023 PASS Mileage 10,254 miles Test location View test location MOT test number 8370 0218 0601 Expiry date 3 November 2024 The MOT test changed on 20 May 2018 Defects are now categorised according to their severity - dangerous, major, and minor. Find out more.   Check for vehicle recalls See if NISSAN JUKE TEKNA DIG-T S-A MJ70HLX has outstanding recalls Show Cookies Terms and conditions Privacy notice Accessibility statement MOT history API Service status Built by the Driver & Vehicle Standards Agency   All content is available under the Open Government Licence v3.0, except where otherwise stated Â© Crown copyright

ğŸ¤– Consultant is thinking... (This may take 10-20s)

WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call', 'thought_signature'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
NoneNone
WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call', 'thought_signature'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
NoneNone
WARNING:google_genai.types:Warning: there are non-text parts in the response: ['thought_signature'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
## 1. Executive Summary: The Verdict

This 2020 Nissan Juke presents as an attractive, low-mileage SUV from a reputable dealer, however, a recent MOT failure due to dangerous and major defects raises significant concerns about its immediate condition and prior maintenance. While the dealer claims the car is "fully mechanically and cosmetically prepared" and offers a warranty, the MOT history directly contradicts the "mechanically prepared" aspect just before its current sale.

## 2. Analysis: Description vs. MOT History

The dealer's description highlights a "one owner from new" vehicle with desirable features and states that it is "fully mechanically and cosmetically prepared," backed by a "Sytner Select Warranty." However, the MOT history reveals a concerning fail on September 11, 2025, just before a pass on September 19, 2025. This fail included **dangerous defects** (tears in both front tyres) and **major defects** (ineffective front windscreen wipers and a cut in a rear tyre). While these issues were subsequently rectified for the pass, the fact that a vehicle described as "fully mechanically prepared" by a major dealer group recently had such critical safety issues is a red flag. The current mileage on the ad (13,002 miles) is lower than the last MOT pass (14,545 miles), indicating the mileage on the ad is likely outdated, or the car has been driven further since the ad was created. The service history is still listed as "Contact seller," which needs to be verified to ensure proper maintenance.

## 3. Red Flags

*   **Mechanical:**
    *   **Recent MOT Failure:** The car failed an MOT on September 11, 2025, with multiple "dangerous" and "major" defects related to tyres and wipers. While fixed, it indicates a period where the car was in an unsafe condition, raising questions about immediate pre-sale preparation and the "mechanically prepared" claim.
    *   **Tyre Condition:** Multiple severe tyre defects (tears, cuts, perishing) were noted in the September 2025 MOT failure. Even though they passed later, it suggests either neglect or significant wear, and the condition of the currently fitted tyres needs careful inspection.
    *   **Model Specific (Potential):** The 1.0 DIG-T engine, while relatively new, can be susceptible to carbon buildup. Some owners report a perceived lack of power. There are also reports of unreliability for the 7-speed Dual-Clutch Transmission (DCT) paired with this engine. Older 1.2 DIG-T engines had timing chain issues, so while not widespread for the 1.0, it's something to be aware of.
*   **Administrative:**
    *   **Missing Service History:** The ad states "Contact seller" for service history. This is a critical omission that needs to be clarified immediately. Without a full service history, it's impossible to verify proper maintenance, which is crucial for modern turbocharged engines like the DIG-T.
    *   **Mileage Discrepancy:** The mileage in the ad (13,002 miles) is lower than the mileage recorded at the last MOT (14,545 miles). This needs clarification from the dealer.

## 4. Overall Condition Assessment

The car appears cosmetically well-appointed with good features and low mileage for its age. The dealer (Sytner Select Warrington) generally has an excellent reputation for customer service. However, the recent MOT failure is a significant concern. While the dangerous defects were rectified to secure a pass, the fact they existed just days before the car's current advertisement suggests a reactive rather than proactive approach to mechanical preparation, contradicting the dealer's claim. The lack of immediately available service history makes it difficult to ascertain the long-term mechanical care of the vehicle. The known potential issues with the 1.0 DIG-T engine (carbon buildup, perceived power limitations) and the DCT gearbox add another layer of caution.

## 5. Action Plan & Questions for the Dealer

*   **Question 1:** Can you provide the full service history of the vehicle, including proof of all scheduled maintenance?
*   **Question 2:** Can you explain why the vehicle failed its MOT with dangerous and major defects on September 11, 2025, just before being advertised as "fully mechanically prepared"? Please provide documentation of the repairs carried out.
*   **Question 3:** What is the current mileage of the vehicle, and why does it differ from the mileage stated in the advertisement?
*   **Question 4:** Can I see the vehicle and inspect the condition of the tyres that were replaced/repaired, and can I have an independent inspection done?
*   **Next step for the buyer:** Do not proceed without satisfactory answers and documentation for all these points. An independent pre-purchase inspection by a trusted mechanic is highly recommended, focusing on the engine (for signs of carbon buildup or turbo issues) and the DCT transmission.

## 6. Final Verdict: AMBER

This Juke is an **AMBER LIGHT**. While it boasts low mileage and a good specification from a reputable dealer, the recent MOT failure with dangerous defects and the currently missing service history are major red flags. These issues need to be thoroughly investigated and verified as resolved with proper documentation before considering a purchase. Without clear evidence of the MOT repairs and a full service history, this car remains a gamble.

[[Summary of Top Contenders]]:
1. 2020 Nissan Juke 1.0 DIG-T Tekna DCT Auto - Amber Light (Recent MOT fail with dangerous defects, missing service history, mileage discrepancy, potential engine/gearbox issues).

