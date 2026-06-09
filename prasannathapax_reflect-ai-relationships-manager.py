import os
import google.generativeai as genai
from google.api_core import exceptions
import json
import time
from datetime import datetime
import re
import textwrap
import urllib.parse
import threading
import queue
import math
import concurrent.futures
import logging
import sys
from tqdm.notebook import tqdm

try:
    from kaggle_secrets import UserSecretsClient
    from IPython.display import display, FileLink, IFrame, HTML
except ImportError:
    print("Not in Kaggle/Colab env. FileLink/UserSecretsClient may not work.")
    class MockUserSecretsClient:
        def get_secret(self, name):
            if name == "GOOGLE_API_KEY":
                return os.environ.get("GOOGLE_API_KEY")
            return None
    UserSecretsClient = MockUserSecretsClient
    FileLink = lambda x: print(f"FileLink: {x} (local mock)")
    IFrame = lambda src, width, height: print(f"IFrame: {src} (width={width}, height={height})")
    HTML = lambda x: print(f"HTML: {x} (local mock)")
display(IFrame(src="https://prasannathapa.in/reflect_ai_output.html", width='100%', height=1500))



FILE_LINK = "/kaggle/input/groupchat/chat.txt"
ENABLE_INTERACTIVE_TAGGING = False #For kaggle no input execution

NUM_ANALYZER_WORKERS = 5 #Number of chucks to process in parallel
NUM_PERSON_MERGE_WORKERS = 5 #Merge these many people in parallel
ANALYSIS_QUEUE_SIZE = 20 #the max limit of chunks in memory
MAX_API_RETRIES = 15 # Max number of API call attempts (default is 3)

MODEL_NAME = 'gemini-2.5-flash-lite'
CHAR_CHUNK_SIZE = 20000
OUTPUT_HTML_FILE_NAME = "report.html"


# --- Logger Setup ---
log_filename = "pipeline.log"
#open in 'w' (write) mode to clear the file
with open(log_filename, "w") as f:
    pass # Opening in 'w' mode and closing truncates the file
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(threadName)-12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(log_filename), # Log to a file
        # logging.StreamHandler(sys.stdout)   # Log to the console (Kaggle output)
    ]
)

# Silence noisy Google API logs to keep our output clean
logging.getLogger("google.api_core").setLevel(logging.WARNING)
logging.info("Logger configured. Pipeline settings loaded.")


try:
    API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    if not API_KEY:
        raise Exception("GOOGLE_API_KEY secret not found or is empty.")
    
    os.environ["GOOGLE_API_KEY"] = API_KEY
    genai.configure(api_key=API_KEY)
    
    # This line acts as a live validation check for the API key
    model_check = genai.GenerativeModel(MODEL_NAME)
    
    logging.info(f"Gemini API key setup complete. Using model: {MODEL_NAME}\n")

except Exception as e:
    logging.critical(f"Authentication Error: {e}")
    logging.critical("    Please make sure 'GOOGLE_API_KEY' is added to Kaggle Secrets")
    logging.critical("    or set as an environment variable for local execution.")
    
    # This variable acts as a global flag to halt the pipeline if auth fails
    model_check = None


ANALYZER_SYSTEM_PROMPT = textwrap.dedent("""
    You are a meticulous data analyst AI. Your task is to analyze a chunk
    of a chat log and extract detailed information.
    - Identify ALL persons mentioned (speakers only not outsiders).
    - For each speaker, infer their hobbies, good traits, and bad traits.
    - Quantify traits and relationship bonds from 0 to 100. If no evidence, return an empty array [].
    - Extract a timeline of moods (happiness, anger, sadness, fear, surprise, disgust) for each person.
    - Extract places they visited or plan to visit.
    - Extract common important dates (birthdays, anniversaries, events).
    - Extract topics that led to positive discussion (topics_to_discuss).
    - Extract topics that led to conflict or anger (topics_to_avoid).
    - ***For topics, focus on generic, recurring themes or interests (e.g., "Planning trips", "Jokes about work").***
    - ***Avoid one-time past events like "completed project X" unless it's a major, defining milestone.***
    - ***Extract only the top 3-5 most important topics for 'topics_to_discuss' and 'topics_to_avoid' in this chunk.***
    - Use "YYYY-MM-DD" format for all dates. If year is missing, infer it (assume current year).
    - You MUST return a JSON object that strictly conforms to the provided schema.
""")

ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "persons": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "hobbies_interests": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "good_traits": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "trait": {"type": "STRING"},
                                "percentage": {"type": "NUMBER"}
                            },
                            "required": ["trait", "percentage"]
                        }
                    },
                    "bad_traits": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "trait": {"type": "STRING"},
                                "percentage": {"type": "NUMBER"}
                            },
                            "required": ["trait", "percentage"]
                        }
                    },
                    "relationships": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "with_person": {"type": "STRING"},
                                "bond_score": {"type": "NUMBER"}
                            },
                            "required": ["with_person", "bond_score"]
                        }
                    },
                    "places_visited": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "date": {"type": "STRING"},
                                "place": {"type": "STRING"}
                            },
                            "required": ["date", "place"]
                        }
                    },
                    "mood_timeline": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "date": {"type": "STRING"},
                                "happiness": {"type": "NUMBER"},
                                "anger": {"type": "NUMBER"},
                                "sadness": {"type": "NUMBER"},
                                "fear": {"type": "NUMBER"},
                                "surprise": {"type": "NUMBER"},
                                "disgust": {"type": "NUMBER"}
                            },
                            "required": ["date", "happiness", "anger", "sadness", "fear", "surprise", "disgust"]
                        }
                    }
                },
                "required": ["name", "hobbies_interests", "good_traits", "bad_traits", "relationships", "places_visited", "mood_timeline"]
            }
        },
        "common_important_dates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "date": {"type": "STRING"},
                    "event": {"type": "STRING"}
                },
                "required": ["date", "event"]
            }
        },
        "topics_to_discuss": {
            "type": "ARRAY",
            "description": "Topics that seem to generate positive engagement.",
            "items": {"type": "STRING"}
        },
        "topics_to_avoid": {
            "type": "ARRAY",
            "description": "Topics that seem to cause anger or sadness.",
            "items": {"type": "STRING"}
        }
    },
    "required": ["persons", "common_important_dates", "topics_to_discuss", "topics_to_avoid"]
}

ANALYZER_GEN_CONFIG = genai.types.GenerationConfig(
    response_mime_type="application/json",
    response_schema=ANALYSIS_SCHEMA
) if model_check else None


MERGER_SYSTEM_PROMPT = textwrap.dedent("""
    You are a data synthesis AI. You will be given two JSON analysis
    reports about the same person. Your job is to merge them into a
    single, updated report.

    - Concatenate simple timelines (moods, places).
    - Merge lists of hobbies, topics_to_discuss, and topics_to_avoid, and de-duplicate them.
    - For 'good_traits', 'bad_traits', and 'relationships', you must
      synthesize the data. If a trait or person appears in both,
      calculate a new, *weighted average* of the score.
    - Return ONLY the single, merged JSON object for this person.
""")

MERGER_GEN_CONFIG = genai.types.GenerationConfig(
    response_mime_type="application/json"
) if model_check else None


INSIGHTS_SYSTEM_PROMPT = textwrap.dedent("""
    You are a high-level strategic analyst. Based on the final JSON summary
    of a chat, provide actionable gift suggestions for each person.
    Return only JSON.
""")

INSIGHTS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "gift_suggestions": {
            "type": "ARRAY",
            "description": "Gift ideas for *each* person, based on their profile.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "for_person": {"type": "STRING"},
                    "gift_idea": {"type": "STRING"},
                    "reason": {"type": "STRING"}
                }
            }
        }
    },
    "required": ["gift_suggestions"]
}

INSIGHTS_GEN_CONFIG = genai.types.GenerationConfig(
    response_mime_type="application/json",
    response_schema=INSIGHTS_SCHEMA
) if model_check else None


class ChatAnalyzerAgent:
    def __init__(self, model_name, max_retries=3):
        if not model_check:
            self.model = None
            logging.error("❌ ChatAnalyzerAgent not initialized. API key is missing.")
            return
        try:
            self.model = genai.GenerativeModel(
                model_name,
                system_instruction=ANALYZER_SYSTEM_PROMPT,
                generation_config=ANALYZER_GEN_CONFIG
            )
            self.max_retries = max_retries
            logging.info("✅ ChatAnalyzerAgent initialized.")
        except Exception as e:
            logging.error(f"Error initializing ANALYZER_MODEL: {e}")
            self.model = None

    def analyze(self, chat_chunk_text, thread_name="Analyzer"):
        if self.model is None:
            logging.warning("ANALYZER_MODEL is not initialized. Skipping.")
            return None

        # Loop up to max_retries, attempting the API call
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(chat_chunk_text)
                return json.loads(response.text)

            except (exceptions.ResourceExhausted, exceptions.DeadlineExceeded) as e:
                # Handle rate limits and timeouts with exponential backoff
                wait_time = 5 * (attempt + 1)
                logging.warning(f"Rate limit or timeout, retrying in {wait_time}s... (Attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)

            except (json.JSONDecodeError, ValueError) as e:
                # Handle cases where the AI returns malformed JSON
                logging.warning(f"ERROR: AI returned invalid JSON. Retrying... (Attempt {attempt + 1}/{self.max_retries})")
                try:
                    logging.debug(f"Raw response part: {response.text[:200]}")
                except Exception:
                    pass
                time.sleep(5 * (attempt + 1))

            except Exception as e:
                # Handle other unexpected API errors
                logging.warning(f"An unexpected error occurred: {e} (Attempt {attempt + 1}/{self.max_retries})")
                try:
                    logging.debug(f"Debug info: {response.prompt_feedback}")
                except Exception:
                    pass
                time.sleep(5 * (attempt + 1))
                
        logging.error(f"Chunk analysis failed after {self.max_retries} attempts.")
        return None

class ChatMergerAgent:
    def __init__(self, model_name, max_retries=3):
        if not model_check:
            self.model = None
            logging.error("❌ ChatMergerAgent not initialized. API key is missing.")
            return

        try:
            self.model = genai.GenerativeModel(
                model_name,
                system_instruction=MERGER_SYSTEM_PROMPT,
                generation_config=MERGER_GEN_CONFIG
            )
            # This agent uses an internal thread pool to merge multiple people in parallel
            self.person_merge_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=NUM_PERSON_MERGE_WORKERS, 
                thread_name_prefix="PersonMerger"
            )
            self.max_retries = max_retries
            logging.info("✅ ChatMergerAgent initialized (with parallel person-merging).")
        except Exception as e:
            logging.error(f"Error initializing MERGER_MODEL: {e}")
            self.model = None

    def _merge_single_person(self, name, old_p, new_p, thread_name="PersonMerger"):
        # This function runs in the internal thread pool
        logging.info(f"merging data for {name}")
        
        merged_person_json = None

        if self.model:
            prompt = f"Merge the following two analysis objects for '{name}'.\n\nOLD DATA:\n{json.dumps(old_p, indent=2)}\n\nNEW DATA:\n{json.dumps(new_p, indent=2)}\n\nReturn ONLY the merged JSON object for '{name}'."
            
            for attempt in range(self.max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    # Extract JSON from the response, handling potential markdown fences
                    match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    
                    if match:
                        merged_person_json = json.loads(match.group(0))
                        break # Success
                    else:
                        logging.warning(f"AI Merge failed for {name}, (no JSON found in response). Retrying... (Attempt {attempt + 1}/{self.max_retries})")

                except (exceptions.ResourceExhausted, exceptions.DeadlineExceeded) as e:
                    wait_time = 5 * (attempt + 1)
                    logging.warning(f"Rate limit for {name}, retrying in {wait_time}s... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"ERROR: AI returned invalid JSON for {name}. Retrying... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(5 * (attempt + 1))
                
                except Exception as e:
                    logging.warning(f"Merge AI call failed for {name}: {e}. Retrying... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(5 * (attempt + 1))
            
            if not merged_person_json:
                logging.error(f"AI Merge for {name} FAILED after {self.max_retries} attempts. Falling back to manual list merge.")
            
        # Fallback logic: if AI merge fails or model is off, manually combine the data
        if not merged_person_json:
            merged_person_json = new_p 
            merged_person_json["name"] = name 

        # Manually merge (or re-merge) simple lists to ensure correctness
        old_hobbies = set(old_p.get("hobbies_interests", []))
        new_hobbies = set(new_p.get("hobbies_interests", []))
        merged_person_json["hobbies_interests"] = list(old_hobbies | new_hobbies)

        old_places = old_p.get("places_visited", [])
        new_places = new_p.get("places_visited", [])
        valid_old_places = [p for p in old_places if isinstance(p, dict) and 'date' in p and 'place' in p]
        valid_new_places = [p for p in new_places if isinstance(p, dict) and 'date' in p and 'place' in p]
        merged_person_json["places_visited"] = valid_old_places + valid_new_places

        old_moods = old_p.get("mood_timeline", [])
        new_moods = new_p.get("mood_timeline", [])
        valid_old_moods = [m for m in old_moods if isinstance(m, dict) and 'date' in m]
        valid_new_moods = [m for m in new_moods if isinstance(m, dict) and 'date' in m]
        merged_person_json["mood_timeline"] = valid_old_moods + valid_new_moods

        # Validate nested objects to ensure they are well-formed
        good_traits = merged_person_json.get("good_traits", [])
        merged_person_json["good_traits"] = [t for t in good_traits if isinstance(t, dict) and 'trait' in t and 'percentage' in t]

        bad_traits = merged_person_json.get("bad_traits", [])
        merged_person_json["bad_traits"] = [t for t in bad_traits if isinstance(t, dict) and 'trait' in t and 'percentage' in t]

        relationships = merged_person_json.get("relationships", [])
        merged_person_json["relationships"] = [r for r in relationships if isinstance(r, dict) and 'with_person' in r and 'bond_score' in r]
        
        return merged_person_json


    def merge(self, old_json, new_json, thread_name="Merger"):
        logging.info("Merging analysis...")
        if not old_json:
            return new_json
        if not new_json:
            return old_json

        # Failsafe if the model failed to initialize
        if self.model is None:
            logging.error("MERGER_MODEL is not initialized. Falling back to simple append.")
            old_json["persons"].extend(new_json.get("persons", []))
            old_json["common_important_dates"].extend(new_json.get("common_important_dates", []))
            old_json["topics_to_discuss"].extend(new_json.get("topics_to_discuss", []))
            old_json["topics_to_avoid"].extend(new_json.get("topics_to_avoid", []))
            return old_json

        # Initialize the new cumulative JSON object
        merged_json = {
            "persons": [],
            "common_important_dates": old_json.get("common_important_dates", []) + \
                                      new_json.get("common_important_dates", []),
            "topics_to_discuss": old_json.get("topics_to_discuss", []) + \
                                   new_json.get("topics_to_discuss", []),
            "topics_to_avoid": old_json.get("topics_to_avoid", []) + \
                                 new_json.get("topics_to_avoid", [])
        }

        # Create dictionaries for fast lookups of people
        old_persons = {p["name"]: p for p in old_json.get("persons", [])}
        new_persons = {p["name"]: p for p in new_json.get("persons", [])}
        all_names = set(old_persons.keys()) | set(new_persons.keys())

        merged_persons_list = []
        futures = []
        
        # Submit a merge task for each person to the internal thread pool
        for name in all_names:
            if not name: continue
            
            old_p = old_persons.get(name)
            new_p = new_persons.get(name)

            if old_p and not new_p:
                merged_persons_list.append(old_p)
                continue
            if not old_p and new_p:
                merged_persons_list.append(new_p)
                continue
            
            # Submit the AI-powered merge to the thread pool
            future = self.person_merge_executor.submit(
                self._merge_single_person, name, old_p, new_p, thread_name
            )
            futures.append(future)

        # Collect results as they are completed
        for future in concurrent.futures.as_completed(futures):
            try:
                merged_person = future.result()
                if merged_person:
                    merged_persons_list.append(merged_person)
            except Exception as e:
                logging.error(f"A person-merge task failed: {e}")

        merged_json["persons"] = merged_persons_list

        # De-duplicate common lists like dates and topics
        seen_dates = set()
        unique_dates = []
        for d in merged_json["common_important_dates"]:
            if isinstance(d, dict) and d.get("date") and d.get("event"):
                key = (d.get("date"), d.get("event"))
                if key not in seen_dates:
                    seen_dates.add(key)
                    unique_dates.append(d)
        merged_json["common_important_dates"] = unique_dates

        merged_json["topics_to_discuss"] = sorted(list(set(merged_json.get("topics_to_discuss", []))))
        merged_json["topics_to_avoid"] = sorted(list(set(merged_json.get("topics_to_avoid", []))))

        return merged_json


class ChatInsightsAgent:
    def __init__(self, model_name, max_retries=3):
        if not model_check:
            self.model = None
            logging.error("❌ ChatInsightsAgent not initialized. API key is missing.")
            return

        try:
            self.model = genai.GenerativeModel(
                model_name,
                system_instruction=INSIGHTS_SYSTEM_PROMPT,
                generation_config=INSIGHTS_GEN_CONFIG
            )
            self.max_retries = max_retries
            logging.info("✅ ChatInsightsAgent initialized.")
        except Exception as e:
            logging.error(f"Error initializing INSIGHTS_MODEL: {e}")
            self.model = None

    def get_insights(self, final_merged_json):
        logging.info("--- [Insights] Calling 'Final Insights' agent (for gifts only)... ---")
        if self.model is None:
            logging.warning("INSIGHTS_MODEL is not initialized. Skipping.")
            return {"gift_suggestions": []}

        try:
            # Create a compacted summary to save tokens
            summary_for_insights = {
                "persons": [
                    {
                        "name": p["name"],
                        "hobbies": p.get("hobbies_interests", []),
                        "traits": p.get("good_traits", [])
                    }
                    for p in final_merged_json.get("persons", [])
                ]
            }
            
            # Retry logic for the final API call
            for attempt in range(self.max_retries):
                try:
                    response = self.model.generate_content(json.dumps(summary_for_insights))
                    return json.loads(response.text)
                
                except (exceptions.ResourceExhausted, exceptions.DeadlineExceeded) as e:
                    wait_time = 5 * (attempt + 1)
                    logging.warning(f"Rate limit, retrying in {wait_time}s... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                
                except Exception as e:
                    logging.error(f"Final insights failed: {e}")
                    return {"gift_suggestions": []}

            logging.error(f"Final insights failed after {self.max_retries} attempts.")
            return {"gift_suggestions": []}

        except Exception as e:
            logging.error(f"Final insights failed: {e}")
            return {"gift_suggestions": []}


import re
import logging

def get_speakers_from_chunk(chat_chunk_text):
    try:
        # Regex to find names matching the pattern: [timestamp] Speaker Name: (Whatsapp export)
        speaker_pattern = re.compile(r'^\s*\[.*?\]\s*([^:]+):', re.MULTILINE)
        all_speakers = speaker_pattern.findall(chat_chunk_text)
        # Return a clean, unique set of lowercase speaker names
        return set(name.strip().lower() for name in all_speakers)
    except Exception as e:
        logging.error(f"Error parsing speakers: {e}", exc_info=True)
        return set()

def filter_non_interactive(analysis_json, chat_chunk_text):
    logging.debug("Running non-interactive (automatic) filter...")
    if not analysis_json or not analysis_json.get("persons"):
        return analysis_json
    speaker_names_lower = get_speakers_from_chunk(chat_chunk_text)
    if not speaker_names_lower:
        logging.warning("No speakers found in chunk, cannot filter.")
        return analysis_json
    
    # This function is a placeholder; the main filtering happens
    # in the _filter_final_json method of the pipeline.
    return analysis_json

def map_and_filter_chunk_members(analysis_json, persistent_member_map):
    if not analysis_json or not analysis_json.get("persons"):
        return analysis_json, persistent_member_map
    
    # Find all names mentioned by the AI in this chunk
    current_chunk_names = set(p.get("name") for p in analysis_json.get("persons", []))
    relationship_names = set()
    for p in analysis_json.get("persons", []):
        for bond in p.get("relationships", []):
            if bond.get("with_person"):
                relationship_names.add(bond.get("with_person"))
    
    all_found_names = current_chunk_names.union(relationship_names)
    
    # Identify names that haven't been seen and mapped before
    newly_found_names = all_found_names - set(persistent_member_map.keys())
    
    if newly_found_names:
        # This section is interactive and uses print() for user prompts
        print("\n--- 🕵️ Member Identification (Tagging) ---")
        print("Please map the AI-found names to a 'main name' or 'tag'.")
        print("If a person is not a member (an outsider), just press ENTER.")
        
        for ai_name in sorted(list(newly_found_names)):
            if not ai_name: continue
            try:
                prompt = f"    AI found: '{ai_name}' -> Main Name [ENTER for outsider]: "
                user_input = input(prompt).strip()
                if user_input:
                    # Map the AI name (e.g., "Bob") to the user's tag (e.g., "Robert")
                    persistent_member_map[ai_name] = user_input
                    print(f"        ✅ Mapped '{ai_name}' => '{user_input}'.")
                else:
                    # Mark this name as an outsider to be filtered out
                    persistent_member_map[ai_name] = None
                    print(f"        ❌ Marked '{ai_name}' as an outsider.")
            except Exception as e:
                logging.error(f"An error occurred during input: {e}", exc_info=True)
                break
    
    # Filter the JSON based on the mapping
    filtered_persons = []
    for person in analysis_json.get("persons", []):
        original_name = person.get("name")
        main_name_tag = persistent_member_map.get(original_name)
        
        # Only process people who are not marked as outsiders (None)
        if main_name_tag is not None:
            person["name"] = main_name_tag # Standardize the name
            
            # Also standardize names within the relationships
            if "relationships" in person:
                updated_relationships = []
                for bond in person.get("relationships", []):
                    original_with_person = bond.get("with_person")
                    mapped_with_person_tag = persistent_member_map.get(original_with_person)
                    
                    # Only keep relationships with other known members
                    if mapped_with_person_tag is not None:
                        bond["with_person"] = mapped_with_person_tag
                        updated_relationships.append(bond)
                person["relationships"] = updated_relationships
            filtered_persons.append(person)
            
    analysis_json["persons"] = filtered_persons
    return analysis_json, persistent_member_map


class HTMLReportGenerator:
    def __init__(self, final_json_data, final_insights_data):
        logging.info("--- Segment 4: Generating HTML Report ---")
        self.final_json = final_json_data
        self.persons = final_json_data.get("persons", [])
        self.common_dates = final_json_data.get("common_important_dates", [])
        self.topics_to_discuss = final_json_data.get("topics_to_discuss", [])
        self.topics_to_avoid = final_json_data.get("topics_to_avoid", [])
        self.final_insights = final_insights_data

    def create_ics_file_content(self):
        # Generates a standard iCalendar (.ics) file string
        ics_content = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//GeminiAI//ChatAnalyzer//EN"]
        for event in self.common_dates:
            try:
                dt = datetime.strptime(event["date"], "%Y-%m-%d")
                date_str = dt.strftime("%Y%m%d")
            except (ValueError, TypeError):
                continue
            ics_content.append("BEGIN:VEVENT")
            ics_content.append(f"DTSTART;VALUE=DATE:{date_str}")
            ics_content.append(f"SUMMARY:{event['event']}")
            ics_content.append(f"UID:{event['date']}-{event['event']}@gemini.ai")
            ics_content.append("END:VEVENT")
        ics_content.append("END:VCALENDAR")
        return "\r\n".join(ics_content)

    def generate_html(self):
        # This is the main template for the self-contained HTML file.
        # It includes all CSS and the Chart.js tab-switching logic.
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Chat Analysis Report</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
                
                :root {{
                    --bg-color: #202124;
                    --card-color: #2d2e30;
                    --border-color: #3c4043;
                    --text-primary: #e8eaed;
                    --text-secondary: #bdc1c6;
                    --accent-color: #82aaff;
                    --accent-color-dark: #1a1a1a;
                    --color-green: #57e389;
                    --color-red: #ff8b82;
                    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    --shadow: 0 4px 10px rgba(0,0,0,0.3);
                }}

                body {{
                    font-family: var(--font-family);
                    background-color: var(--bg-color);
                    color: var(--text-primary);
                    margin: 0; 
                    padding: 24px;
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 1400px;
                    margin: auto;
                    background: var(--card-color);
                    border-radius: 12px;
                    border: 1px solid var(--border-color);
                    box-shadow: var(--shadow);
                    overflow: hidden;
                }}
                header {{
                    background-color: var(--bg-color);
                    color: var(--text-primary);
                    padding: 24px 40px;
                    text-align: left;
                    border-bottom: 1px solid var(--border-color);
                }}
                header h1 {{ 
                    margin: 0; 
                    font-size: 2em; 
                    font-weight: 600;
                }}
                .tab-buttons {{
                    display: flex;
                    background: var(--card-color);
                    padding: 0 20px;
                    flex-wrap: wrap;
                    border-bottom: 1px solid var(--border-color);
                }}
                .tab-btn {{
                    padding: 16px 24px;
                    border: none;
                    background: none;
                    cursor: pointer;
                    font-size: 1.05em;
                    font-weight: 600;
                    color: var(--text-secondary);
                    position: relative;
                    transition: color 0.3s;
                    border-bottom: 4px solid transparent;
                    margin-bottom: -1px;
                }}
                .tab-btn:hover {{ color: var(--text-primary); }}
                .tab-btn.active {{
                    color: var(--accent-color);
                    border-bottom: 4px solid var(--accent-color);
                }}
                
                .tab-content {{ 
                    display: none; 
                    padding: 40px;
                }}
                .tab-content.active {{ display: block; }}

                .grid-container {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 25px;
                }}

                @media (max-width: 900px) {{
                    .grid-container {{
                        grid-template-columns: 1fr;
                    }}
                }}
                
                .chart-card.full-width {{
                    grid-column: 1 / -1;
                }}
                
                .chart-card {{
                    background: #3c4043;
                    border: 1px solid var(--border-color);
                    border-radius: 12px;
                    padding: 24px;
                    box-shadow: var(--shadow);
                }}
                .chart-card h3 {{
                    margin-top: 0;
                    margin-bottom: 20px;
                    font-weight: 600;
                    font-size: 1.25em;
                    color: var(--accent-color);
                }}
                
                .chart-card h3.title-green {{ color: var(--color-green); }}
                .chart-card h3.title-red {{ color: var(--color-red); }}

                .chart-wrapper {{
                    position: relative;
                    height: 350px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                .chart-wrapper canvas {{
                    max-width: 100%;
                    max-height: 100%;
                }}
                
                .list-card ul {{ 
                    padding-left: 20px; 
                    margin-top: 10px; 
                    margin-bottom: 0;
                }}
                .list-card li {{ 
                    margin-bottom: 12px; 
                    line-height: 1.5; 
                    color: var(--text-secondary);
                }}
                .list-card p {{
                    color: var(--text-secondary);
                    margin-top: 0;
                }}
                .list-card strong {{ 
                    color: var(--text-primary); 
                    font-weight: 600;
                }}
                
                .list-title {{
                    display: block;
                    font-weight: 600;
                    color: var(--text-primary);
                    margin-bottom: 8px; 
                    margin-top: 15px;
                }}
                .list-title:first-of-type {{
                    margin-top: 0;
                }}

                .download-btn {{
                    display: inline-block;
                    background: var(--accent-color);
                    color: var(--accent-color-dark);
                    padding: 10px 18px;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    margin-top: 20px;
                    transition: background 0.3s, transform 0.2s;
                }}
                .download-btn:hover {{ 
                    background: #a9c3ff;
                    transform: translateY(-2px);
                }}

                .scrollable-list-container {{
                    max-height: 316px;
                    overflow-y: auto;
                    padding-right: 10px;
                }}
                .scrollable-list-container.small {{
                    max-height: 350px;
                }}
                .scrollable-list-container ul {{ padding-left: 15px; margin: 0; }}
                .scrollable-list-container::-webkit-scrollbar {{ width: 8px; }}
                .scrollable-list-container::-webkit-scrollbar-track {{ background: #3c4043; border-radius: 4px; }}
                .scrollable-list-container::-webkit-scrollbar-thumb {{ background: #777; border-radius: 4px; }}
                .scrollable-list-container::-webkit-scrollbar-thumb:hover {{ background: #999; }}
            </style>
        </head>
        <body>
            <div class="container">
                <header><h1>AI Chat Analysis Report</h1></header>
                
                <div class="tab-buttons">
                    <button class="tab-btn active" onclick="openTab(event, 'Common')">Common Insights</button>
                    {self.generate_tab_buttons()}
                </div>
                
                <div id="Common" class="tab-content active">
                    {self.generate_common_tab()}
                </div>
                
                {self.generate_person_tabs()}
                
            </div>
            
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {{ tabcontent[i].style.display = "none"; }}
                    tablinks = document.getElementsByClassName("tab-btn");
                    for (i = 0; i < tablinks.length; i++) {{ tablinks[i].className = tablinks[i].className.replace(" active", ""); }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
                
                document.addEventListener("DOMContentLoaded", () => {{
                    try {{
                        {self.generate_chart_js()}
                    }} catch (e) {{ console.error("Error rendering charts:", e); }}
                }});
            </script>
        </body>
        </html>
        """
        return textwrap.dedent(html)

    def generate_tab_buttons(self):
        buttons_html = ""
        for person in self.persons:
            name = person.get("name", "Unknown")
            safe_name = re.sub(r'\W+', '', name) 
            buttons_html += f'<button class="tab-btn" onclick="openTab(event, \'{safe_name}\')">{name}</button>\n'
        return buttons_html

    def generate_common_tab(self):
        # Dynamically build the HTML for each card
        # If data is empty, the card (including title) will not be rendered
        
        dates_card_html = ""
        dos_card_html = ""
        donts_card_html = ""

        if self.common_dates:
            ics_data = self.create_ics_file_content()
            ics_href = f"data:text/calendar;charset=utf-8,{urllib.parse.quote(ics_data)}"
            dates_html = "".join([f"<li><strong>{item['date']}:</strong> {item['event']}</li>" for item in self.common_dates])
            dates_card_html = f"""
            <div class="list-card chart-card">
                <h3>Important Dates</h3>
                <div class="scrollable-list-container">
                    <ul>{dates_html}</ul>
                </div>
                <a href="{ics_href}" target="_blank" download="chat_calendar.ics" class="download-btn">Download .ics Calendar</a>
            </div>
            """

        if self.topics_to_discuss:
            dos_html = "".join([f"<li>{item}</li>" for item in self.topics_to_discuss])
            dos_card_html = f"""
            <div class="list-card chart-card">
                <h3 class="title-green">Conversation: Do's</h3>
                <p>Generic topics that generate positive engagement:</p>
                <div class="scrollable-list-container small">
                    <ul>{dos_html}</ul>
                </div>
            </div>
            """
            
        if self.topics_to_avoid:
            donts_html = "".join([f"<li>{item}</li>" for item in self.topics_to_avoid])
            donts_card_html = f"""
            <div class="list-card chart-card">
                <h3 class="title-red">Conversation: Don'ts</h3>
                <p>Generic topics that may cause conflict or sadness:</p>
                <div class="scrollable-list-container small">
                    <ul>{donts_html}</ul>
                </div>
            </div>
            """

        return f"""
        <div class="grid-container">
            {dates_card_html}
            {dos_card_html}
            {donts_card_html}
        </div>
        """

    def generate_person_tabs(self):
        tabs_html = ""
        for person in self.persons:
            name = person.get("name", "Unknown")
            safe_name = re.sub(r'\W+', '', name)
            
            # Conditionally build each card's HTML
            mood_card_html = ""
            trait_card_html = ""
            bond_card_html = ""
            interests_card_html = ""
            places_card_html = ""

            mood_data = person.get("mood_timeline", [])
            if mood_data:
                mood_card_html = f"""
                <div class="chart-card full-width">
                    <h3>Mood Timeline</h3>
                    <canvas id="moodChart-{safe_name}"></canvas>
                </div>
                """

            good_traits = person.get("good_traits", [])
            bad_traits = person.get("bad_traits", [])
            if good_traits or bad_traits:
                trait_card_html = f"""
                <div class="chart-card">
                    <h3>Personality Traits (Good vs. Bad)</h3>
                    <div class="chart-wrapper"> 
                        <canvas id="traitChart-{safe_name}"></canvas>
                    </div>
                </div>
                """
            
            # Only render the relationship chart if there are other people
            if len(self.persons) >= 2:
                bond_card_html = f"""
                <div class="chart-card">
                    <h3>Relationship Bonds (Radar)</h3>
                    <div class="chart-wrapper">
                        <canvas id="bondChart-{safe_name}"></canvas>
                    </div>
                </div>
                """

            hobbies_data = person.get("hobbies_interests", [])
            gifts = [g for g in self.final_insights.get("gift_suggestions", []) if g.get("for_person") == name]
            
            if hobbies_data or gifts:
                hobbies_html = "".join([f"<li>{item}</li>" for item in hobbies_data])
                gifts_html = "".join([f"<li><strong>{g['gift_idea']}:</strong> {g['reason']}</li>" for g in gifts])

                interests_card_html = f"""
                <div class="chart-card">
                    <h3>Interests & Gift Ideas</h3>
                    <strong class="list-title">Hobbies & Interests:</strong>
                    <div class="scrollable-list-container small">
                        <ul>{hobbies_html if hobbies_html else "<li>No specific interests found.</li>"}</ul>
                    </div>
                    <strong class="list-title">Gift Suggestions:</strong>
                    <div class="scrollable-list-container small">
                        <ul>{gifts_html if gifts_html else "<li>No specific gift ideas found.</li>"}</ul>
                    </div>
                </div>
                """

            places_data = person.get("places_visited", [])
            if places_data:
                places_html = "".join([f"<li><strong>{item['date']}:</strong> {item['place']}</li>" for item in places_data])
                places_card_html = f"""
                <div class="list-card chart-card">
                    <h3>Places Timeline</h3>
                    <div class="scrollable-list-container">
                        <ul>{places_html}</ul>
                    </div>
                </div>
                """

            # Assemble the final HTML for this person's tab
            tabs_html += f"""
            <div id="{safe_name}" class="tab-content">
                <div class="grid-container">
                    {mood_card_html}
                    {trait_card_html}
                    {bond_card_html}
                    {interests_card_html}
                    {places_card_html}
                </div>
            </div>
            """
        return tabs_html

    def generate_chart_js(self):
        # This method generates the JavaScript <script> content
        js_chunks = []
        all_names = [p["name"] for p in self.persons]
        
        # Define chart colors
        chart_text_color = '#e8eaed'
        chart_grid_color = 'rgba(232, 234, 237, 0.15)'
        chart_grid_color_zero = 'rgba(232, 234, 237, 0.4)'
        
        accent_color = '#82aaff'
        accent_color_faded = 'rgba(130, 170, 255, 0.2)'
        color_green = '#57e389'
        color_green_faded = 'rgba(87, 227, 137, 0.2)'
        color_red = '#ff8b82'
        color_red_faded = 'rgba(255, 139, 130, 0.2)'
        color_sadness = '#89cff0'
        color_sadness_faded = 'rgba(137, 207, 240, 0.2)'
        color_surprise = '#f9d162'
        color_surprise_faded = 'rgba(249, 209, 98, 0.2)'
        color_disgust = '#7d5c41'
        color_disgust_faded = 'rgba(125, 92, 65, 0.2)'
        color_fear = '#c39eff'
        color_fear_faded = 'rgba(195, 158, 255, 0.2)' 

        for person in self.persons:
            name = person.get("name", "Unknown")
            safe_name = re.sub(r'\W+', '', name)
            
            # --- Mood Chart JS ---
            mood_data = person.get("mood_timeline", [])
            if mood_data: # Only generate JS if data exists
                try:
                    # Sort moods by date to ensure the line chart is chronological
                    mood_data.sort(key=lambda x: datetime.strptime(x.get('date', '1970-01-01'), '%Y-%m-%d'))
                except ValueError:
                    logging.warning(f"Could not sort mood data for {name}, dates may be out of order.")

                mood_labels = []
                for m in mood_data:
                    try:
                        dt = datetime.strptime(m.get('date', ''), '%Y-%m-%d')
                        mood_labels.append(dt.strftime('%b %d, %y'))
                    except ValueError:
                        mood_labels.append(m.get('date', 'Unknown'))

                # Extract data for each emotion
                happiness_data = [m.get("happiness", 50) for m in mood_data]
                anger_data = [m.get("anger", 50) for m in mood_data]
                sadness_data = [m.get("sadness", 50) for m in mood_data]
                fear_data = [m.get("fear", 50) for m in mood_data]
                surprise_data = [m.get("surprise", 50) for m in mood_data]
                disgust_data = [m.get("disgust", 50) for m in mood_data]
                
                js_chunks.append(f"""
                var ctxMood = document.getElementById('moodChart-{safe_name}');
                if (ctxMood) {{
                    new Chart(ctxMood, {{
                        type: 'line',
                        data: {{
                            labels: {json.dumps(mood_labels)},
                            datasets: [
                                {{ label: 'Happiness', data: {json.dumps(happiness_data)}, borderColor: {json.dumps(color_green)}, backgroundColor: {json.dumps(color_green_faded)}, fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 0 }},
                                {{ label: 'Anger', data: {json.dumps(anger_data)}, borderColor: {json.dumps(color_red)}, backgroundColor: {json.dumps(color_red_faded)}, fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 0 }},
                                {{ label: 'Sadness', data: {json.dumps(sadness_data)}, borderColor: {json.dumps(color_sadness)}, backgroundColor: {json.dumps(color_sadness_faded)}, fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 0 }},
                                {{ label: 'Fear', data: {json.dumps(fear_data)}, borderColor: {json.dumps(color_fear)}, backgroundColor: {json.dumps(color_fear_faded)}, fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 0 }},
                                {{ label: 'Surprise', data: {json.dumps(surprise_data)}, borderColor: {json.dumps(color_surprise)}, backgroundColor: {json.dumps(color_surprise_faded)}, fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 0 }},
                                {{ label: 'Disgust', data: {json.dumps(disgust_data)}, borderColor: {json.dumps(color_disgust)}, backgroundColor: {json.dumps(color_disgust_faded)}, fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 0 }}
                            ]
                        }},
                        options: {{ 
                            responsive: true,
                            plugins: {{ legend: {{ labels: {{ color: {json.dumps(chart_text_color)} }} }}, tooltip: {{ intersect: false, mode: 'index' }} }},
                            scales: {{ 
                                y: {{ stacked: true, beginAtZero: true, grid: {{ color: {json.dumps(chart_grid_color)} }}, ticks: {{ color: {json.dumps(chart_text_color)} }} }},
                                x: {{ grid: {{ color: {json.dumps(chart_grid_color)} }}, ticks: {{ color: {json.dumps(chart_text_color)} }} }}
                            }}
                        }}
                    }});
                }}
                """)
            
            # --- Trait Chart JS ---
            good_traits = person.get("good_traits", [])
            bad_traits = person.get("bad_traits", [])
            if good_traits or bad_traits: 
                trait_labels = [t['trait'] for t in bad_traits] + [t['trait'] for t in good_traits]
                # Use negative values for bad traits to create a diverging bar chart
                trait_data = [-t['percentage'] for t in bad_traits] + [t['percentage'] for t in good_traits]
                
                js_chunks.append(f"""
                var ctxTrait = document.getElementById('traitChart-{safe_name}');
                if (ctxTrait) {{
                    new Chart(ctxTrait, {{
                        type: 'bar',
                        data: {{
                            labels: {json.dumps(trait_labels)},
                            datasets: [{{
                                label: 'Trait Score',
                                data: {json.dumps(trait_data)},
                                backgroundColor: (ctx) => ctx.raw < 0 ? 'rgba(255, 139, 130, 0.7)' : 'rgba(87, 227, 137, 0.7)',
                                borderColor: (ctx) => ctx.raw < 0 ? {json.dumps(color_red)} : {json.dumps(color_green)},
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true, maintainAspectRatio: true, indexAxis: 'y',
                            scales: {{
                                x: {{
                                    ticks: {{ callback: (value) => Math.abs(value) + '%', color: {json.dumps(chart_text_color)} }},
                                    grid: {{ color: (ctx) => ctx.tick.value === 0 ? {json.dumps(chart_grid_color_zero)} : {json.dumps(chart_grid_color)} }}
                                }},
                                y: {{ grid: {{ display: false }}, ticks: {{ color: {json.dumps(chart_text_color)} }} }}
                            }},
                            plugins: {{
                                legend: {{ display: false }},
                                tooltip: {{ callbacks: {{ label: (ctx) => ctx.label + ': ' + Math.abs(ctx.raw) + '%' }} }}
                            }}
                        }}
                    }});
                }}
                """)
            
            # --- Bond Chart JS ---
            if len(all_names) >= 2: 
                bond_data = person.get("relationships", [])
                bond_labels = []
                bond_scores = []
                # Map bond scores for all people in the chat
                for other_name in all_names:
                    bond_labels.append(other_name)
                    if other_name == name:
                        bond_scores.append(100) # Bond with self is 100
                    else:
                        bond_score = 0 # Default bond
                        for bond in bond_data:
                            if bond.get("with_person") == other_name:
                                bond_score = bond.get("bond_score", 0)
                                break
                        bond_scores.append(bond_score)
                
                js_chunks.append(f"""
                var ctxBond = document.getElementById('bondChart-{safe_name}');
                if (ctxBond) {{
                    if ({len(bond_labels)} >= 2) {{
                        new Chart(ctxBond, {{
                            type: 'radar',
                            data: {{
                                labels: {json.dumps(bond_labels)},
                                datasets: [{{
                                    label: 'Bond Score',
                                    data: {json.dumps(bond_scores)},
                                    fill: true,
                                    backgroundColor: {json.dumps(accent_color_faded)},
                                    borderColor: {json.dumps(accent_color)},
                                    pointBackgroundColor: {json.dumps(accent_color)},
                                    pointBorderColor: '#fff',
                                    pointHoverBackgroundColor: '#fff',
                                    pointHoverBorderColor: {json.dumps(accent_color)}
                                }}]
                            }},
                            options: {{ 
                                responsive: true, maintainAspectRatio: true,
                                scales: {{ 
                                    r: {{ 
                                        beginAtZero: true, max: 100,
                                        ticks: {{ backdropColor: 'transparent', stepSize: 20, color: {json.dumps(chart_text_color)} }},
                                        angleLines: {{ color: {json.dumps(chart_grid_color)} }},
                                        grid: {{ color: {json.dumps(chart_grid_color)} }},
                                        pointLabels: {{ color: {json.dumps(chart_text_color)} }}
                                    }} 
                                }},
                                elements: {{ line: {{ tension: 0 }} }},
                                plugins: {{ legend: {{ display: false }} }}
                            }}
                        }});
                    }} else {{
                        var ctx = ctxBond.getContext('2d');
                        ctx.font = '16px Inter';
                        ctx.fillStyle = '#999';
                        ctx.textAlign = 'center';
                        ctx.fillText('Need at least 2 people for radar chart', ctxBond.width/2, ctxBond.height/2);
                    }}
                }}
                """)
            
        return "\n".join(js_chunks)


    def save_report(self, filename=OUTPUT_HTML_FILE_NAME):
        html_content = self.generate_html()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"HTML Report saved: '{filename}'")
        return filename


class ChatAnalysisPipeline:
    def __init__(self, file_path, num_analyzers, queue_size):
        self.file_path = file_path
        self.num_analyzer_workers = num_analyzers
        
        # Bounded queue to hold results from Analyzers, provides backpressure
        self.analysis_queue = queue.Queue(maxsize=queue_size)
        
        # This dictionary holds the final, cumulative JSON
        self.final_merged_json = {}
        
        # A lock to prevent multiple threads from writing to final_merged_json simultaneously
        self.merger_lock = threading.Lock()
        
        # Queue to hold raw text chunks for the Analyzer threads
        self.task_queue = queue.Queue()
                
        # (Optional) Holds mappings for user name aliases
        self.member_tag_map = {}
        self.filtering_lock = threading.Lock() 
        
        # Initialize all three agents, passing the global retry config
        self.analyzer_agent = ChatAnalyzerAgent(MODEL_NAME, max_retries=MAX_API_RETRIES)
        self.merger_agent = ChatMergerAgent(MODEL_NAME, max_retries=MAX_API_RETRIES)
        self.insights_agent = ChatInsightsAgent(MODEL_NAME, max_retries=MAX_API_RETRIES)
        
        # Placeholder for the TQDM progress bar instance
        self.progress_bar = None

    def _filter_and_map(self, analysis_json, chunk):
        # This function is used to map AI-found names to a main "tag"
        global member_tag_map
        if ENABLE_INTERACTIVE_TAGGING:
            with self.filtering_lock:
                logging.info(f"Acquiring lock for interactive tagging...")
                filtered_json, self.member_tag_map = map_and_filter_chunk_members(
                    analysis_json, 
                    self.member_tag_map
                )
                logging.info(f"Releasing lock.")
            return filtered_json
        else:
            # If not interactive, just run the automatic filter
            return filter_non_interactive(analysis_json, chunk)

    def _analyzer_worker(self):
        # This function runs in a loop for each Analyzer thread
        thread_name = threading.current_thread().name
        while True:
            chunk = None 
            try:
                # Get a raw text chunk from the task queue
                chunk = self.task_queue.get()
                if chunk is None:
                    break # Stop the thread if a None (poison pill) is received

                logging.info(f"Analyzing chunk of {len(chunk)} chars...")
                new_analysis_json = self.analyzer_agent.analyze(chunk, thread_name)
                
                if not isinstance(new_analysis_json, dict):
                    logging.warning(f"AI returned invalid type (expected dict, got {type(new_analysis_json)}). Skipping chunk.")
                    continue

                # Basic validation to skip empty/useless AI results
                has_persons = bool(new_analysis_json.get("persons"))
                has_topics = bool(new_analysis_json.get("topics_to_discuss") or new_analysis_json.get("topics_to_avoid"))
                
                if not has_persons and not has_topics:
                    logging.info(f"No people or topics found by AI. Skipping chunk.")
                    continue
                
                # (Optional) Run the name-tagging filter
                new_analysis_json = self._filter_and_map(new_analysis_json, chunk)
                
                logging.debug(f"Submitting result to analysis_queue (bounded)...")
                # Put the JSON result onto the next queue for the Merger
                self.analysis_queue.put(new_analysis_json)
                
            except Exception as e:
                logging.error(f"Error in analyzer worker: {e}", exc_info=True)
            finally:
                if chunk is not None:
                    # Signal to the queue that this task is complete
                    self.task_queue.task_done()
                    
        logging.info(f"Analyzer worker shutting down.")


    def _merger_worker(self):
        # This function runs in a single thread to merge all results
        thread_name = threading.current_thread().name
        logging.info(f"Single Merger worker started... waiting for analysis results.")
        
        while True:
            new_json = None
            try:
                # Get an analyzed JSON chunk from the analysis queue
                new_json = self.analysis_queue.get()
                
                if new_json is None:
                    logging.info(f"Got shutdown signal.")
                    self.analysis_queue.task_done()
                    break # Stop the thread
                
                # Use a lock to ensure only this thread can modify the final JSON
                with self.merger_lock:
                    logging.debug(f"Locking to merge... (Queue size: {self.analysis_queue.qsize()})")
                    
                    self.final_merged_json = self.merger_agent.merge(
                        self.final_merged_json, 
                        new_json, 
                        thread_name
                    )
                    
                    logging.debug(f"Merge complete. Releasing lock.")
                
            except Exception as e:
                logging.error(f"Error in merger worker: {e}", exc_info=True)
            finally:
                if new_json is not None:
                    self.analysis_queue.task_done()
                    # Update the progress bar by 1 after a chunk is successfully merged
                    if self.progress_bar:
                        self.progress_bar.update(1)
                    
        logging.info(f"Merger worker shutting down.")

    def _filter_final_json(self, final_json):
        # Cleans the final JSON by removing people with no data
        logging.info("--- [Pipeline] Running final AI-heuristic filter on merged JSON... ---")
        if not final_json or not final_json.get("persons"):
            return final_json
            
        filtered_persons = []
        speaker_names = set()

        # First pass: find all valid speakers
        for person in final_json.get("persons", []):
            original_name = person.get("name")
            if not original_name:
                continue
            
            # Check if the person has any meaningful data
            has_mood = bool(person.get("mood_timeline"))
            has_traits = bool(person.get("good_traits") or person.get("bad_traits"))
            has_hobbies = bool(person.get("hobbies_interests"))
            has_places = bool(person.get("places_visited"))
            has_relations = bool(person.get("relationships"))

            if has_mood or has_traits or has_hobbies or has_places or has_relations:
                filtered_persons.append(person)
                speaker_names.add(original_name.lower())
            
        original_count = len(final_json.get("persons", []))
        filtered_count = len(filtered_persons)
        
        if filtered_count < original_count:
            logging.info(f"Removed {original_count - filtered_count} outsider(s) (no details found).")
        
        # Second pass: filter relationships to only include other valid speakers
        for person in filtered_persons:
             if "relationships" in person:
                  updated_relationships = []
                  for bond in person.get("relationships", []):
                       original_with_person = bond.get("with_person")
                       
                       if original_with_person and original_with_person.lower() in speaker_names:
                            updated_relationships.append(bond)
                  person["relationships"] = updated_relationships
                  
        final_json["persons"] = filtered_persons
        return final_json

    def _streaming_chunk_loader(self):
        # Runs in one thread, reading the file in chunks and feeding the task_queue
        logging.info("--- [Pipeline] Starting streaming chunk loader...")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(CHAR_CHUNK_SIZE)
                    if not chunk:
                        break # End of file
                    self.task_queue.put(chunk)
            
            logging.info("--- [Pipeline] Chunk loader finished.")
        except Exception as e:
            logging.error(f"Failed to read chat file: {e}", exc_info=True)
            
    def run(self):
        logging.info("--- Segment 5: Starting Concurrent Pipeline (Streaming) ---")
        if not model_check:
            logging.critical("Main process halted. Please configure your GOOGLE_API_KEY.")
            return

        # Calculate total chunks to initialize the progress bar
        try:
            file_size_in_bytes = os.path.getsize(self.file_path)
            total_chunks = math.ceil(file_size_in_bytes / CHAR_CHUNK_SIZE)
            logging.info(f"File Size: {file_size_in_bytes} bytes. Estimated Chunks: {total_chunks}")
            # Create the TQDM progress bar instance
            self.progress_bar = tqdm(total=total_chunks, desc="Processing Chunks", unit="chunk")
        except Exception as e:
            logging.error(f"Could not create progress bar: {e}")
            total_chunks = 0
        
        # Start all the worker threads
        loader_thread = threading.Thread(target=self._streaming_chunk_loader, name="ChunkLoader")
        loader_thread.daemon = True
        loader_thread.start()
        
        analyzer_threads = []
        for i in range(self.num_analyzer_workers):
            t = threading.Thread(target=self._analyzer_worker, name=f"Analyzer-{i+1}")
            t.daemon = True 
            t.start()
            analyzer_threads.append(t)

        merger_thread = threading.Thread(target=self._merger_worker, name="Merger-1")
        merger_thread.daemon = True
        merger_thread.start()

        # --- Graceful Shutdown Sequence ---
        
        # 1. Wait for the file to be fully read
        loader_thread.join()
        logging.info("--- [Pipeline] File reading is complete. ---")

        # 2. Wait for all chunks to be picked up by analyzers
        logging.info("--- [Pipeline] Waiting for all chunks to be analyzed... ---")
        self.task_queue.join()
        logging.info("--- [Pipeline] All chunks have been processed by analyzers. ---")
        
        # 3. Send "poison pill" (None) to stop analyzer threads
        for _ in analyzer_threads:
            self.task_queue.put(None)
        for t in analyzer_threads:
            t.join()
        logging.info("--- [Pipeline] All Analyzer threads have stopped. ---")

        # 4. Send "poison pill" (None) to stop the merger thread
        logging.info("--- [Pipeline] Signaling Merger to shut down... ---")
        self.analysis_queue.put(None)
        merger_thread.join()
        logging.info("--- [Pipeline] Merger thread has stopped. ---")
        
        # 5. Close the progress bar
        if self.progress_bar:
            self.progress_bar.close()
            logging.info("Progress bar complete.")
        
        logging.info("[Pipeline] All chunks processed and merged.")

        # --- Final Report Generation ---
        
        final_merged_json = self.final_merged_json
        
        if not ENABLE_INTERACTIVE_TAGGING:
            try:
                final_merged_json = self._filter_final_json(final_merged_json)
            except Exception as e:
                logging.error(f"Failed during final filtering step: {e}", exc_info=True)

        # Final check if any data was successfully processed
        if not final_merged_json:
             logging.error("No data was analyzed successfully (JSON is null). Exiting.")
             return
                 
        has_persons = bool(final_merged_json.get("persons"))
        has_topics = bool(final_merged_json.get("topics_to_discuss") or final_json.get("topics_to_avoid"))
        has_dates = bool(final_merged_json.get("common_important_dates"))

        if not has_persons and not has_topics and not has_dates:
            logging.error("No data was analyzed successfully (no persons, topics, or dates found). Exiting.")
            return

        # Call the final agent to get gift suggestions
        raw_insights = self.insights_agent.get_insights(final_merged_json)
        
        final_insights = {} 
        if isinstance(raw_insights, dict):
            final_insights = raw_insights
        else:
            logging.warning(f"Insights agent returned invalid type (got {type(raw_insights)}), using empty default.")
        final_insights.setdefault("gift_suggestions", [])

        # Generate and save the final HTML file
        report_generator = HTMLReportGenerator(final_merged_json, final_insights)
        report_filename = report_generator.save_report()

        logging.info("--- Process Complete ---")
        
        # Use print for final user-facing output
        try:
            display(HTML(f"To open in new tab click {FileLink(report_filename)._repr_html_()}"))
            display(IFrame(report_filename, width='100%', height=2300)) 
        except NameError:
            print(f"Please find '{report_filename}' in your file directory.")


def main():
    # Main entry point for the script
    if not model_check:
        logging.critical("❌ Main process halted. API key is not configured.")
        return
    
    try:
        # 1. Initialize the pipeline with all the configured settings
        pipeline = ChatAnalysisPipeline(
            file_path=FILE_LINK,
            num_analyzers=NUM_ANALYZER_WORKERS,
            queue_size=ANALYSIS_QUEUE_SIZE
        )
        
        # 2. Start the entire concurrent analysis process
        pipeline.run()
        
    except FileNotFoundError:
        logging.critical(f"❌ CRITICAL ERROR: File not found at {FILE_LINK}")
    except Exception as e:
        logging.critical(f"❌ An unexpected error occurred in main: {e}", exc_info=True)
       
if __name__ == "__main__":
    main()


# --- Display Logs After Execution ---
# This code will run after main() is complete (or has failed)
# It prints the full log file for review and then clears it.
WIDTH = 85

# --- PIPELINE LOGS ---
print("\n" + "="*WIDTH)
print("PIPELINE LOGS".center(WIDTH))
print("="*WIDTH + "\n")

try:
    # Open and read the log file to print its contents
    with open(log_filename, "r") as f:
        print(f.read())
except FileNotFoundError:
    print(f"--- No log file ({log_filename}) was found. ---".center(WIDTH))
except Exception as e:
    print(f"--- Error reading log file: {e} ---".center(WIDTH))

# --- BLOG POST ---
print("\n" + "="*WIDTH)
print("Read the blog post!".center(WIDTH))
print("="*WIDTH + "\n")
display(IFrame(src="https://blog.prasannathapa.in/reflect-personal-ai-relationship-manager/", width='100%', height=800))

# --- AUTHOR SECTION ---
print("\n" + "="*WIDTH)
print("Author: Prasanna Thapa 🤝 Lets connect".center(WIDTH))
print("="*WIDTH + "\n")
display(IFrame(src="https://prasannathapa.in", width='100%', height=800))


