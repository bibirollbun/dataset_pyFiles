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


from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.runners import InMemoryRunner 
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


# Sales Agent: Scans tender portals, identifies RFPs <= 3 months, and summerizes requirements.
sales_agent = Agent(
    name="SalesAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction=""" 
    STRICT RULES (must follow):
1. ONLY return RFPs whose due date is within the next 90 days.
2. DO NOT return expired or past RFPs.
3. DO NOT repeat the same RFP more than once.
4. If Google Search returns old tenders â€” IGNORE them.

Your tasks:
1. Use the Google Search tool to find RFP base on the prompt given
   
2. Extract only ACTIVE tenders with:
   - Future due dates
   - Date â‰¤ 90 days from today
3. For each valid RFP, extract:
   - RFP_Title
   - Authority
   - Due_Date
   - Product_Requirements
   - Testing_Requirements (if any)
4. Mark `"Is_Qualified": true` only when:
   - Due date â‰¤ 90 days
   - It belongs to office supplies category

If no active RFPs exist â†’ Output:
{ "message": "No active RFPs available within next 90 days." }

FINAL OUTPUT MUST BE:
A JSON array with UNIQUE entries, no duplicates, and only valid tenders.
    """,
    tools=[google_search],
    output_key="sales_agent_output",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… sales_agent created.")


technical_agent = Agent(
    name="TechnicalAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
    You are the Technical Agent responsible for product-spec matching.

    INPUTS YOU WILL RECEIVE:
    - RFP summary (product requirements, specs)
    - OEM product catalog (list of SKUs with specs)

    YOUR TASKS:
    1. Extract the list of products from "Scope of Supply" in the RFP.
    2. For each product, match it with the **top 3 OEM SKUs** using a Spec-Match % formula.
       - All specs have equal weight.
       - Spec Match % = (number of matched spec attributes / total attributes) * 100
    3. Create a comparison table:
       - RFP Spec
       - OEM Product 1 Spec
       - OEM Product 2 Spec
       - OEM Product 3 Spec
       - Match %
    4. Choose the **Top 1 SKU** for each RFP item.
    5. Output ONLY valid JSON in this format:

    {
      "RFP_Products": [...],
      "Top_Matches": [
        {
          "RFP_Item": "...",
          "Top3_SKUs": [
            {"SKU": "...", "Match_Percentage": 95},
            {"SKU": "...", "Match_Percentage": 89},
            {"SKU": "...", "Match_Percentage": 82}
          ],
          "Comparison_Table": [...],
          "Selected_SKU": "SKU123"
        }
      ]
    }

    STRICT RULES:
    - No hallucination: only use products from catalog.
    - Only valid JSON.
    - No additional text outside JSON.
    """,
    output_key="technical_agent_output"
)

print("âœ… Technical Agent created.")



pricing_agent = Agent(
    name="PricingAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=os.environ["GOOGLE_API_KEY"],
        retry_options=retry_config
    ),
    instruction="""
    You are the Pricing Agent responsible for generating RFP cost estimates.

    INPUTS YOU WILL RECEIVE:
    - Recommended SKUs & comparison table from Technical Agent
    - List of required tests / acceptance criteria from Sales Agent
    - Synthetic Pricing Tables:
        * Product_Prices = {"SKU": price_per_unit}
        * Test_Prices = {"Type Test": 5000, "Routine Test": 1500, ...}
        * Logistics_Factors = {"base_rate": 5%, "distance_multiplier": ...}

    YOUR TASKS:
    1. For each selected SKU:
       - Retrieve unit price from synthetic product price table.
       - Multiply quantity if provided (else assume 1).
    2. Add cost of required tests.
    3. Add logistics cost (base_rate * material cost).
    4. Calculate:
       - Material Price
       - Test Price
       - Logistics Price
       - Total Price

    FINAL OUTPUT MUST BE STRICT JSON in this format:

    {
      "Pricing": [
        {
          "SKU": "SKU123",
          "Unit_Price": 250,
          "Quantity": 100,
          "Material_Cost": 25000,
          "Testing_Cost": 5000,
          "Logistics_Cost": 1250,
          "Total_Cost": 31250
        }
      ],
      "Grand_Total": 31250
    }

    RULES:
    - No hallucinated SKUs or tests.
    - Only JSON output.
    """,
    output_key="pricing_agent_output"
)

print("âœ… Pricing Agent created.")



parallel_agent = ParallelAgent(
    name="Parallel_tech_price",
    sub_agents=[technical_agent, pricing_agent],
)
print("âœ… Parallel Agent created.")

final_agent = SequentialAgent(
    name="RFP_Sequential_Orchestrator",
    sub_agents=[sales_agent,parallel_agent],
)

print("âœ… Sequential Agent created.")


session_service = InMemorySessionService()
runner = InMemoryRunner(agent=final_agent)
print("âœ… Runner created.")

response = await runner.run_debug(
    "Find active RFPs and tenders for office supplies, stationery, printers, IT peripherals, or office furniture in India, issued by government departments, PSUs, or private enterprises, with submission deadlines within the next 3 months.")
print("ğŸš€ Running agent...")

