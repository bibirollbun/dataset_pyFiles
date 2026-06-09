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


import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
import concurrent.futures

# --- CONFIGURATION ---
MODEL_NAME = "gemini-2.0-flash"
MAX_RETRIES = 5

INITIAL_PAIRS = [
    'GBP/USD (Forex)',
    'XAU/USD (Gold/Metal)',
    'USD/JPY (Forex)',
]

# --- CLIENT INITIALIZATION ---
try:
    API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not API_KEY:
        raise KeyError("GOOGLE_API_KEY not found in environment variables.")
        
    client = genai.Client(api_key=API_KEY)
    print("Gemini Client initialized successfully.")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize Gemini Client. Details: {e}")
    print("Ensure your API key is valid and was loaded in the preceding block.")
    client = None

AnalysisSchema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "pair": types.Schema(type=types.Type.STRING, description="The currency or commodity pair analyzed."),
        "decision": types.Schema(type=types.Type.STRING, enum=["Buy", "Sell", "Hold"], description="The final trading recommendation."),
        "winningRate": types.Schema(type=types.Type.NUMBER, description="The calculated confidence level (0-100) as a percentage."),
        "rationale": types.Schema(type=types.Type.STRING, description="A brief explanation for the decision, mentioning key technical indicators.")
    },
    required=["pair", "decision", "winningRate", "rationale"]
)

# --- CORE AGENT LOGIC (Gemini API Wrapper in Python) ---

def fetch_analysis(pair: str) -> Dict[str, Any]:
    """
    Simulates a Specialized Analysis Agent making a decision via the Gemini API.
    """
    failure_result = {
        "pair": pair,
        "decision": 'N/A',
        "winningRate": 0,
        "rationale": 'API Client failed to initialize. Check API Key.',
        "sources": []
    }
    
    if not client:
        return failure_result

    # IMPORTANT: Modified system prompt to prioritize Yahoo Finance and other reliable sources.
    # FIX: Corrected indentation for system_prompt and properly defined generation_config
    system_prompt = "You are a specialized Forex and Metals Technical Analyst. Based on the market data from your tools, which should primarily target reliable financial sources like Yahoo Finance, Bloomberg, or reputable trading platforms, provide a structured analysis. Focus on key indicators (like RSI, MACD, Moving Averages). Your ENTIRE output must be a single JSON object, wrapped in a markdown code block (```json{...}```). This JSON object MUST include the following keys: 'pair', 'decision' (Buy, Sell, or Hold), 'winningRate' (0-100%), and a detailed 'rationale' explaining your decision based on technical indicators and market conditions. DO NOT include any conversational text outside the markdown block."
    user_query = f"Provide technical analysis for {pair} today, suggesting a 'Buy', 'Sell', or 'Hold' decision. The 'winningRate' must be a numerical percentage (0-100) representing confidence. The analysis should be based on current, real-time data."

    # FIX: Removed response_mime_type and response_schema as they conflict with the 'google_search' tool.
    generation_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[{"google_search": {}}]
    )

    for attempt in range(MAX_RETRIES):
        try:
            print(f"[{pair}] Attempt {attempt + 1}: Fetching real-time market data...")
            
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_query,
                config=generation_config
            )

            if response.text:
                json_string = response.text.strip()
                
                match = re.search(r"```json\s*(.*?)\s*```", json_string, re.DOTALL)
                
                if match:
                    json_string = match.group(1).strip()
                
                try:
                    data = json.loads(json_string)
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse JSON response. Raw text received: {response.text}")
                
                sources = []
                if response.candidates and response.candidates[0].grounding_metadata:
                    metadata = response.candidates[0].grounding_metadata
                    
                    if hasattr(metadata, 'attributions'):
                        attributions = metadata.attributions
                    elif hasattr(metadata, 'grounding_attributions'):
                         attributions = metadata.grounding_attributions
                    else:
                        attributions = []

                    sources = [
                        {
                            "uri": attr.web.uri,
                            "title": attr.web.title,
                        } for attr in attributions if attr.web
                    ]
                
                raw_rationale = data.get("rationale")
                
                if not raw_rationale:
                    raw_rationale = (
                        data.get("summary") or 
                        data.get("overallAnalysis") or 
                        data.get("technicalAnalysis")
                    )
                
                if raw_rationale:
                    if isinstance(raw_rationale, dict):
                        final_rationale = raw_rationale.get("summary", str(raw_rationale))
                    else:
                        final_rationale = str(raw_rationale)
                else:
                    final_rationale = "Model failed to provide a detailed rationale."

                normalized_data = {
                    "pair": pair,
                    "decision": data.get("decision", "N/A"),
                    "winningRate": int(data.get("winningRate", 0)), 
                    "rationale": final_rationale,
                    "sources": sources
                }
                
                print(f"[{pair}] Analysis Complete: {normalized_data['decision']} ({normalized_data['winningRate']}%)")
                return normalized_data
            
            raise ValueError("No valid content returned from API.")

        except APIError as e:
            print(f"[{pair}] API Error on attempt {attempt + 1}: {e}")
        except Exception as e:
            print(f"[{pair}] Processing Error on attempt {attempt + 1}: {e}")
        
        if attempt < MAX_RETRIES - 1:
            delay = 2 ** attempt
            print(f"[{pair}] Retrying in {delay} seconds...")
            time.sleep(delay)

    return {
        "pair": pair,
        "decision": 'N/A',
        "winningRate": 0,
        "rationale": 'Analysis failed after multiple retries. Check console for details.',
        "sources": []
    }


# --- ORCHESTRATOR EXECUTION ---

def orchestrate_analysis(pairs: List[str]) -> List[Dict[str, Any]]:
    """
    Coordinates all specialized analysis agents using concurrent execution.
    """
    print("\n--- Starting Multi-Agent Trading Orchestrator (Concurrent) ---")
    if not client:
        print("Orchestration aborted due to API client error.")
        return []

    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_pair = {executor.submit(fetch_analysis, pair): pair for pair in pairs}
        
        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'[{pair}] Generated an exception: {exc}')
                results.append({
                    "pair": pair,
                    "decision": 'N/A',
                    "winningRate": 0,
                    "rationale": f'Execution failed: {exc}',
                    "sources": []
                })

    print("\n--- Orchestration Complete ---")
    return results

final_results = orchestrate_analysis(INITIAL_PAIRS)

print("\n--- Final Consolidated Trading Recommendations ---")
for result in final_results:
    print(f"\nAsset: {result.get('pair', 'Unknown Asset')}")
    print(f"Recommendation: {result.get('decision', 'N/A')}")
    print(f"Confidence: {result.get('winningRate', 0)}%")
    print(f"Rationale: {result.get('rationale', 'Rationale missing from analysis.')}")
    if result.get('sources'):
        print("--- Grounding Sources ---")
        for source in result['sources']:
            print(f"  - {source['title']} ({source['uri']})")
    
try:
    import pandas as pd
    
    required_keys = ['pair', 'decision', 'winningRate', 'rationale']
    clean_results = []
    for res in final_results:
        clean_results.append({k: res.get(k, 'N/A') for k in required_keys})

    df = pd.DataFrame(clean_results)
    
    # print("\n--- Results DataFrame ---")
    # print(df.to_markdown(index=False))
except ImportError:
    print("\nInstall pandas ('!pip install pandas') to display results in a DataFrame.")
except Exception as e:
     print(f"Failed to create DataFrame: {e}")

