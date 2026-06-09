!pip install --upgrade openai  > /dev/null 2>&1
!pip install pypdf==3.17.3  > /dev/null 2>&1
!pip install faiss-cpu==1.7.4  > /dev/null 2>&1
!pip install tiktoken==0.5.2  > /dev/null 2>&1
!pip install numpy==1.24.3 > /dev/null 2>&1


import os
import json
import time
import random
from datetime import datetime
from kaggle_secrets import UserSecretsClient
from openai import OpenAI


#'=============================================================================
# JSON of questions
#'=============================================================================

questions_data = {
    "A. State Of the Art (SOTA) for archaeological site detection": [
        {"number": 1, "question": "What are Amazonian Dark Earths, and can they be used to detect Amazonian archaeological sites from satellite images?"},
        {"number": 2, "question": "What is the current state of the art for detecting archaeological sites in the Amazon Forest using satellite or aerial images?"},
        {"number": 3, "question": "What machine learning models have successfully identified archaeological features beneath forest canopy?"},
        {"number": 4, "question": "How effective is LIDAR technology for detecting earthworks and geoglyphs in the Amazon, and what are its limitations?"},
        {"number": 5, "question": "What remote sensing techniques are most effective in distinguishing natural vs. anthropogenic landscape features?"},
        {"number": 6, "question": "How are radar (e.g., Sentinel-1 SAR) and hyperspectral data being combined with LIDAR for sub-canopy detection?"},
        {"number": 7, "question": "What benchmark datasets and evaluation metrics are commonly used to compare canopy-penetrating algorithms?"}
    ],

    "B. Datasets": [
        {"number": 8, "question": "What open-access LIDAR datasets cover portions of the Brazilian Amazon?"},
        {"number": 9, "question": "Where can we access free high-resolution satellite imagery of the Amazon region?"},
        {"number": 10, "question": "What public multispectral satellite data exists for the target region?"},
        {"number": 11, "question": "Are there any open archaeological databases containing verified site locations in the Amazon?"},
        {"number": 12, "question": "What digital elevation models (DEMs) are available covering the Amazon basin?"},
        {"number": 13, "question": "What historical map collections have been digitized that include Amazon exploration routes?"},
        {"number": 14, "question": "Which public platforms curate Indigenous territorial boundaries and ethno-linguistic data layers?"},
        {"number": 15, "question": "What national or state-level open-data portals provide ancillary layers (hydrology, soils, roads, deforestation) that could help in site prediction?"}
    ],

    "C. Known archaeological sites": [
        {"number": 16, "question": "What are the locations of known archaeological sites in the Amazon Rainforest (give coordinates so I can locate them on a map)"},
        {"number": 17, "question": "What are the distinctive features of Kuhikugu and other confirmed ancient settlements in the Xingu River region?"},
        {"number": 18, "question": "What common geographical features are shared among verified archaeological sites in the Amazon?"},
        {"number": 19, "question": "What dating techniques have been used to establish chronologies for known Amazon sites?"},
        {"number": 20, "question": "How large were the largest confirmed pre-Columbian settlements in the Amazon?"},
        {"number": 21, "question": "Which ceramic typologies are most commonly recovered from major Amazonian sites, and how do they aid cultural attribution?"}
    ],

    "D. Suspected archaeological sites": [
        {"number": 22, "question": "What are the locations of suspected archaeological sites in the Brazilian Amazon Rainforest from the literature that have not yet been confirmed"},
        {"number": 23, "question": "What locations match descriptions from historical accounts but haven't been verified on the ground?"},
        {"number": 24, "question": "What unusual vegetation patterns or geometric anomalies have been reported but not investigated?"},
        {"number": 25, "question": "What geographic areas have high potential for archaeological sites but remain unsurveyed?"},
        {"number": 26, "question": "Where have Indigenous communities reported ancestral sites that archaeologists haven't formally documented?"},
        {"number": 27, "question": "Which clusters of non-natural soil mounds (terras pretas) are visible on soil maps but untested archaeologically?"}
    ],

    "E. Historical text from expedition": [
        {"number": 28, "question": "What primary accounts from Francisco de Orellana's 1541-1542 expedition mention large settlements?"},
        {"number": 29, "question": "What details did Percy Fawcett record about \"Z\" in his journals and correspondence?"},
        {"number": 30, "question": "What geographical landmarks in historical accounts can be correlated with modern maps?"},
        {"number": 31, "question": "Which colonial-era maps show settlements that don't appear in modern records?"},
        {"number": 32, "question": "What Spanish, Portuguese, and British expedition records from 1500-1800 describe large Amazon populations?"},
        {"number": 33, "question": "How do accounts from different European explorers contradict or corroborate each other?"},
        {"number": 34, "question": "What references to earthworks, causeways, or canals appear in Jesuit or missionary writings?"},
        {"number": 35, "question": "What case studies show text-mining pipelines leading directly to new archaeological discoveries in densely forested regions?"},
        {"number": 36, "question": "Which OCR workflows give the best accuracy on 16th–19th-century Portuguese and Spanish expedition journals?"}
    ],

    "F. Indigenous knowledge and oral histories": [
        {"number": 37, "question": "What Indigenous oral histories describe ancient cities or large settlements?"},
        {"number": 38, "question": "How do current Indigenous land use patterns reflect historical settlement patterns?"},
        {"number": 39, "question": "What traditional ecological knowledge might indicate former human habitation?"},
        {"number": 40, "question": "Which Indigenous groups maintain knowledge of ancient earthworks or structures?"},
        {"number": 41, "question": "What place names in Indigenous languages correlate with archaeological features?"},
        {"number": 42, "question": "How have recent collaborative archaeology projects integrated Indigenous mapping methods (e.g., oral cartography)?"},
        {"number": 43, "question": "In which oral traditions are specific landscape modifications—such as raised fields, causeways, or fish-weirs—explicitly described or ritually commemorated?"},
        {"number": 44, "question": "How do seasonal calendars, myth cycles, and ceremonial itineraries embed spatial information about former habitation zones?"},
        {"number": 45, "question": "Which Indigenous languages retain directional or locational morphemes (up-river, old village, sacred clearing) that may encode archaeological clues?"},
        {"number": 46, "question": "How have community-led participatory mapping or story-mapping projects revealed undocumented archaeological features?"},
        {"number": 47, "question": "How can text-mined colonial sources be cross-validated against living oral traditions to resolve conflicting site locations?"}
    ],

    "G. Environmental and geological context": [
        {"number": 48, "question": "What soil types are most associated with long-term human settlements in the Amazon?"},
        {"number": 49, "question": "How does river migration in the Amazon basin affect the preservation of archaeological sites?"},
        {"number": 50, "question": "What vegetation patterns might indicate historical human modification?"},
        {"number": 51, "question": "Which areas have terrain suitable for large settlements (flood-free, accessible by river)?"},
        {"number": 52, "question": "How have climate changes over the past 2000 years affected settlement patterns?"},
        {"number": 53, "question": "What reproducibility frameworks (e.g., Jupyter, Docker, Kedro) are ideal for transparent analysis pipelines?"}
    ],

    "H. Technical approach": [
        {"number": 54, "question": "What image processing techniques best enhance subtle terrain differences in satellite imagery?"},
        {"number": 55, "question": "How can we filter false positives when identifying potential geometric patterns?"},
        {"number": 56, "question": "What machine learning architectures work best for identifying anthropogenic features under forest canopy?"},
        {"number": 57, "question": "How can we correlate multiple data sources (LIDAR, multispectral, historical) in a unified analysis?"},
        {"number": 58, "question": "What verification methods can be used to confirm potential site discoveries remotely?"}
    ],

    "I. Geographical consideration": [
        {"number": 59, "question": "What locations in the Amazon forest are the most likely to be linked to archaeological sites (for ex water proximity, proximity to known settlement…)?"},
        {"number": 60, "question": "How does modern deforestation correlate with exposure of buried archaeological features?"},
        {"number": 61, "question": "Which geomorphological provinces (e.g., interfluvial plateaus vs. river terraces) show the highest density of known sites?"},
        {"number": 62, "question": "How strong is the spatial autocorrelation between confirmed sites and large navigable waterways (main stems vs. secondary tributaries)?"},
        {"number": 63, "question": "What distance-decay threshold from documented Terra Preta patches correlates with a significant uptick in undiscovered features?"},
        {"number": 64, "question": "Can kernel-density or hotspot analyses of known sites reveal settlement corridors along paleo-channels now hidden beneath forest?"},
        {"number": 65, "question": "Which combinations of slope (°), elevation (m), and flood-return interval best discriminate occupied vs. unused terraces?"},
        {"number": 66, "question": "How do soil-fertility maps (e.g., phosphorus, organic carbon) overlap with clusters of anthropogenic dark earths?"},
        {"number": 67, "question": "What role do confluences, oxbow-lake margins, or seasonally flooded várzea islands play in site distribution?"},
        {"number": 68, "question": "How does least-cost path analysis between paired earthwork complexes suggest regional road or causeway networks?"},
        {"number": 69, "question": "Which eco-regional boundaries (e.g., terra firme to várzea transition) mark sharp drops in site density?"},
        {"number": 70, "question": "How do modern roadcuts, logging gaps, and fire scars reveal subsurface geoglyphs when viewed in time-series imagery?"},
        {"number": 71, "question": "What machine-learning feature-importance rankings emerge when predicting site presence using variables such as water distance, Terra Preta proximity, and canopy-height variance?"},
        {"number": 72, "question": "How can river-migration reconstructions (based on sediment-core or remote-sensing chronologies) refine estimates of former shoreline settlement belts?"},
        {"number": 73, "question": "What is the spatial relationship between sacred-grove distributions (identified through ethnobotanical surveys) and archaeological mound fields?"},
        {"number": 74, "question": "How do modern Indigenous reserve boundaries correspond to historical occupation cores inferred from ceramic-style catchments?"},
        {"number": 75, "question": "Which hydro-geomorphic settings (levees, scroll bars, ancient levee breaches) display the highest proportion of untested magnetic anomalies?"},
        {"number": 76, "question": "How can multi-criteria suitability models (AHP, MaxEnt, or Bayesian networks) integrate hydrology, soils, and cultural distance to rank unexplored polygons for survey priority?"}
    ]
}

#'=============================================================================
# JSON of models
#'=============================================================================

# Web-enabled models using :online suffix
models_data = [
    "openai/gpt-4.1-mini:online",
    "anthropic/claude-3-haiku:online",
    "meta-llama/llama-4-scout:online"
]


#'=============================================================================
# System instruction with your four formatting requirements ----
#'=============================================================================

SYSTEM_INSTRUCTION = """\
You are asked to produce a comprehensive report in **Markdown**.  
Please ensure you:
- **Add a meaningful title** to your report.
- **Restate the question** verbatim at the very start.
- Add a reference section
- Structure headings and sections using proper Markdown syntax.
"""


OPEN_ROUTER_KEY = None

#'=============================================================================
# Load API keys from secrets and set environment variables ----
#'=============================================================================
def get_openrouter_key():
    """Get the OpenRouter API key from Kaggle secrets only once"""
    global OPEN_ROUTER_KEY
    
    # If key already retrieved, return it
    if OPEN_ROUTER_KEY:
        return OPEN_ROUTER_KEY
    
    # Otherwise, get it from Kaggle secrets
    try:
        user_secrets = UserSecretsClient()
        OPEN_ROUTER_KEY = user_secrets.get_secret("open_router_key_250521")
        return OPEN_ROUTER_KEY
    except Exception as e:
        print(f"⚠️ Error retrieving OpenRouter API key: {str(e)}")
        print("Please make sure you have set up a valid 'openrouter_key' in Kaggle Secrets.")
        raise

#'=============================================================================
# OpenAI client method for OpenRouter API ----
#'=============================================================================
def create_chat_completion(
    model: str,
    messages: list[dict],
    max_retries: int = 3,
    backoff_factor: float = 2.0
):
    """Create chat completion using OpenAI client with OpenRouter"""
    # Get API key once
    api_key = get_openrouter_key()
    
    # Initialize OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    # Add retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            # Add a small delay before each request to avoid rate limiting
            if attempt > 0:
                backoff_time = backoff_factor ** attempt + random.uniform(0, 1)
                print(f"Retry attempt {attempt+1}/{max_retries} for {model}, waiting {backoff_time:.2f}s")
                time.sleep(backoff_time)
            
            # Updated parameters according to documentation
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_headers={
                    "HTTP-Referer": "https://kaggle.com",
                    "X-Title": "Kaggle Notebook"
                },
                extra_body={}
            )
            
            # Validate response contains expected structure
            if not hasattr(response, 'choices') or len(response.choices) == 0:
                raise ValueError(f"Invalid response format from API: missing choices")
            return response
            
        except Exception as e:
            print(f"Error on attempt {attempt+1} with {model}: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full stack trace
            
            # Continue to next retry unless it's the last attempt
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed for {model}")
                return None
    
    return None

#'=============================================================================
# Direct API implementation as a fallback option ----
#'=============================================================================
def create_chat_completion_direct(
    model: str,
    messages: list[dict],
    max_retries: int = 3
):
    """Direct API call using requests - fallback method"""
    import requests
    
    # Get API key once
    api_key = get_openrouter_key()
    
    # Only use OpenRouter as requested
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kaggle.com",
        "X-Title": "Kaggle Notebook"  # Added according to documentation
    }
    
    payload = {
        "model": model,
        "messages": messages
    }
    
    # Implement retry logic
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                backoff_time = 2 ** attempt + random.random()
                print(f"Direct API retry attempt {attempt+1}/{max_retries}, waiting {backoff_time:.2f}s")
                time.sleep(backoff_time)
                
            response = requests.post(
                base_url,
                headers=headers,
                json=payload,
                timeout=90  # 90-second timeout
            )
            
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
            
        except Exception as e:
            print(f"Direct API error on attempt {attempt+1}: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full stack trace
            
            if attempt == max_retries - 1:
                return None
    
    return None

#'=============================================================================
# Extract web citations if present (optional) ----
#'=============================================================================
def extract_web_citations(response):
    """Extract web search citations from the response if available"""
    citations = []
    
    try:
        # For OpenAI client response
        if hasattr(response, 'choices'):
            message = response.choices[0].message
            if hasattr(message, 'annotations'):
                for annotation in message.annotations:
                    if annotation.type == 'url_citation':
                        citations.append({
                            'url': annotation.url_citation.url,
                            'title': annotation.url_citation.title,
                            'content': getattr(annotation.url_citation, 'content', None)
                        })
        # For direct API response
        elif isinstance(response, dict) and 'choices' in response:
            message = response['choices'][0]['message']
            if 'annotations' in message:
                for annotation in message['annotations']:
                    if annotation['type'] == 'url_citation':
                        citations.append({
                            'url': annotation['url_citation']['url'],
                            'title': annotation['url_citation']['title'],
                            'content': annotation['url_citation'].get('content', None)
                        })
    except Exception as e:
        print(f"  Note: Could not extract web citations: {str(e)}")
    
    return citations

#'=============================================================================
# Main execution loop  ----
#'=============================================================================
def process_questions():
    
    # Track results for reporting
    results_log = []
    
    # Pre-fetch API key once to handle any errors upfront
    try:
        get_openrouter_key()
        print("✓ Successfully retrieved OpenRouter API key")
        print("✓ Web search enabled for all models using :online suffix")
    except Exception as e:
        print(f"✗ Failed to retrieve OpenRouter API key: {str(e)}")
        print("Cannot proceed without API key. Please check your Kaggle secrets.")
        return
    
    for section, questions in questions_data.items():
        print(f"\nProcessing section: {section}")
        
        for q in questions:
            number = q["number"]
            text = q["question"]
            print(f"\nQuestion {number}: {text[:50]}..." if len(text) > 50 else f"\nQuestion {number}: {text}")
            
            # Build message stack
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": text},
            ]
            
            for model in models_data:
                print(f"  Using model: {model} (web search enabled)")
                
                try:
                    # Try with OpenAI client first
                    response = create_chat_completion(
                        model=model, 
                        messages=messages
                    )
                    
                    # If the first method failed, try direct API call as fallback
                    if response is None:
                        print(f"  Trying fallback direct API approach for {model}...")
                        response_json = create_chat_completion_direct(model=model, messages=messages)
                        
                        if response_json is None:
                            raise ValueError("Both API approaches failed")
                            
                        content = response_json["choices"][0]["message"]["content"]
                        
                        # Extract citations from direct API response
                        citations = extract_web_citations(response_json)
                    else:
                        # Extract content from the successful OpenAI client response
                        content = response.choices[0].message.content
                        
                        # Extract citations from OpenAI client response
                        citations = extract_web_citations(response)
                    
                    # Build a timestamped, filesystem-safe filename
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    safe_model = model.replace("/", "_").replace(":", "_")
                    filename = f"question_{number}_{safe_model}_{timestamp}.md"
                    
                    # Write out the Markdown-formatted answer
                    with open(filename, "w", encoding="utf-8") as out_f:
                        out_f.write(content)
                        
                        # Optionally append citations at the end
                        if citations:
                            out_f.write("\n\n---\n## Web Sources Used\n\n")
                            for i, citation in enumerate(citations, 1):
                                out_f.write(f"{i}. [{citation['title']}]({citation['url']})\n")
                    
                    print(f"  ✓ Successfully saved response to {filename}")
                    if citations:
                        print(f"  ✓ Found {len(citations)} web citations")
                    
                    results_log.append({
                        "status": "success", 
                        "question": number, 
                        "model": model, 
                        "filename": filename,
                        "timestamp": timestamp,
                        "web_citations_count": len(citations)
                    })
                    
                except Exception as e:
                    print(f"  ✗ Error with model {model}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())  # Print full stack trace
                    
                    results_log.append({
                        "status": "error", 
                        "question": number, 
                        "model": model, 
                        "error": str(e),
                        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S")
                    })
                
                # Add a delay between requests to prevent rate limiting
                time.sleep(3)
    
    # Save completion results log
    log_filename = f"completion_results_log_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        json.dump(results_log, log_file, indent=2)
    
    # Print summary statistics
    success_count = sum(1 for r in results_log if r["status"] == "success")
    error_count = sum(1 for r in results_log if r["status"] == "error")
    
    print(f"\nProcessing complete!")
    print(f"Successful completions: {success_count}")
    print(f"Failed completions: {error_count}")
    if success_count + error_count > 0:
        print(f"Success rate: {success_count/(success_count+error_count)*100:.1f}%")
    print(f"Results log saved to: {log_filename}")
    
    # Cost estimation
    total_requests = success_count
    plugin_cost = total_requests * 0.02  # $0.02 per request with default 5 results
    print(f"\nEstimated web search cost: ${plugin_cost:.2f} (assuming default 5 results per search)")

# Run the processing with all the questions and models
if __name__ == "__main__":
    process_questions()

