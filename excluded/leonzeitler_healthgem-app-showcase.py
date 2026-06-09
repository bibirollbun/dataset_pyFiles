%%capture
# Install required dependencies
!pip install pyvis==0.3.2 langchain-ollama==0.3.6
!curl -fsSL https://ollama.com/install.sh | sh


%%capture
import time
import subprocess
from subprocess import DEVNULL

# Start ollama server and download model
subprocess.Popen('ollama serve', stdout=DEVNULL, stderr=DEVNULL, shell=True)
time.sleep(3)  # Let the ollama server finish start up
!ollama pull hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q8_K_XL


# Localization
class LocalizationManager:
    """Manages translations, localization, and language-specific content for the health tracking app."""
    
    def __init__(self):
        self.current_language = 'English'
        self._init_medical_specialties()
        self._init_ui_texts()
        self._init_examples()
    
    def _init_medical_specialties(self):
        """Initialize medical specialties translations."""
        self.medical_specialties = {
            'English': [
                'primary care', 'cardiology', 'gastroenterology', 'pulmonology', 
                'neurology', 'endocrinology', 'nephrology', 'rheumatology', 
                'immunology', 'infectious diseases', 'dermatology', 'oncology', 
                'psychiatry', 'orthopedics', 'otolaryngology', 'ophthalmology', 
                'urology', 'gynecology', 'dentistry'
            ],
            'German': [
                'allgemeinmedizin', 'kardiologie', 'gastroenterologie', 'pneumologie',
                'neurologie', 'endokrinologie', 'nephrologie', 'rheumatologie',
                'immunologie', 'infektiologie', 'dermatologie', 'onkologie',
                'psychiatrie', 'orthopÃ¤die', 'hals-nasen-ohrenheilkunde', 'augenheilkunde',
                'urologie', 'gynÃ¤kologie', 'zahnmedizin'
            ]
        }
        
        # Create translation mappings for all language pairs
        self.specialty_mappings = {}
        languages = list(self.medical_specialties.keys())
        
        for source_lang in languages:
            for target_lang in languages:
                if source_lang != target_lang:
                    mapping_key = f"{source_lang}_to_{target_lang}"
                    source_specialties = self.medical_specialties[source_lang]
                    target_specialties = self.medical_specialties[target_lang]
                    
                    # Create mapping dictionary
                    self.specialty_mappings[mapping_key] = dict(zip(source_specialties, target_specialties))
    
    def _init_ui_texts(self):
        """Initialize UI text translations."""
        self.ui_texts = {
            'app_title': {'English': '# HealthGem', 'German': '# HealthGem'},
            'welcome_message': {
                'English': 'Welcome! Please describe your medical condition, symptom, or action.',
                'German': 'Willkommen! Bitte beschreiben Sie Ihren medizinischen Zustand, ein Symptom oder eine MaÃŸnahme.'
            },
            'processing': {
                'English': 'Processing your input',
                'German': 'Ihre Eingabe wird verarbeitet'
            },
            'done_message': {
                'English': 'âœ… Done! I\'ve added the new events to your health graph.',
                'German': 'âœ… Fertig! Ich habe die neuen Ereignisse zu Ihrem Gesundheitsgraph hinzugefÃ¼gt.'
            },
            'error_message': {
                'English': 'âš ï¸� Could not extract a valid event. Please try rephrasing your input.',
                'German': 'âš ï¸� Konnte kein gÃ¼ltiges Ereignis extrahieren. Bitte formulieren Sie Ihre Eingabe um.'
            },
            'link_found': {
                'English': 'I\'ve processed your input. I found a possible connection to a past event about',
                'German': 'Ich habe Ihre Eingabe verarbeitet. Ich fand eine mÃ¶gliche Verbindung zu einem vergangenen Ereignis Ã¼ber'
            },
            'link_question': {
                'English': 'Should I link the new event to this one?',
                'German': 'Soll ich das neue Ereignis mit diesem verknÃ¼pfen?'
            },
            'linked_events': {
                'English': 'ğŸ”— I\'ve linked the events.',
                'German': 'ğŸ”— Ich habe die Ereignisse verknÃ¼pft.'
            },
            'separate_event': {
                'English': 'â›“ï¸�â€�ğŸ’¥ I\'ve added it as a separate event.',
                'German': 'â›“ï¸�â€�ğŸ’¥ Ich habe es als separates Ereignis hinzugefÃ¼gt.'
            },
            'no_events_selected': {
                'English': 'âš ï¸� No events were selected for deletion.',
                'German': 'âš ï¸� Keine Ereignisse fÃ¼r die LÃ¶schung ausgewÃ¤hlt.'
            },
            'events_deleted': {
                'English': 'âœ… Successfully deleted {count} specific event(s).',
                'German': 'âœ… {count} spezifische(s) Ereignis(se) erfolgreich gelÃ¶scht.'
            },
            'graph_legend': {
                'English': '''
                **Graph Legend**
                <br><span style="display:inline-block; width:12px; height:12px; background-color:#616161; border-radius:3px; margin-right: 5px;"></span> **Category**: A medical speciality or anatomical part.
                <br><span style="display:inline-block; width:12px; height:12px; background:linear-gradient(to right, #ffffb3, #ef5350); border-radius:50%; margin-right: 5px; border: 2px solid #ef5350;"></span> **Symptom**: Seriousness from low (yellow) to high (red).
                <br><span style="display:inline-block; width:12px; height:12px; background-color:#A5D6A7; border-radius:50%; margin-right: 5px; border: 2px solid #4CAF50;"></span> **Action Taken**: Medication, therapy, or self-care.
                <br><span style="display:inline-block; width:12px; height:12px; background-color:#90CAF9; border-radius:50%; margin-right: 5px; border: 2px solid #2196F3;"></span> **Visit**: An appointment with a medical professional.
                ''',
                'German': '''
                **Graph-Legende**
                <br><span style="display:inline-block; width:12px; height:12px; background-color:#616161; border-radius:3px; margin-right: 5px;"></span> **Kategorie**: Eine medizinische Fachrichtung oder anatomischer Teil.
                <br><span style="display:inline-block; width:12px; height:12px; background:linear-gradient(to right, #ffffb3, #ef5350); border-radius:50%; margin-right: 5px; border: 2px solid #ef5350;"></span> **Symptom**: Schweregrad von niedrig (gelb) bis hoch (rot).
                <br><span style="display:inline-block; width:12px; height:12px; background-color:#A5D6A7; border-radius:50%; margin-right: 5px; border: 2px solid #4CAF50;"></span> **MaÃŸnahme**: Medikation, Therapie oder Selbstpflege.
                <br><span style="display:inline-block; width:12px; height:12px; background-color:#90CAF9; border-radius:50%; margin-right: 5px; border: 2px solid #2196F3;"></span> **Besuch**: Ein Termin bei einem Arzt.
                '''
            },
            'filter_event_type': {'English': 'Filter by Event Type', 'German': 'Nach Ereignistyp filtern'},
            'filter_anatomical_part': {'English': 'Filter by Anatomical Part', 'German': 'Nach anatomischem Teil filtern'},
            'language_label': {'English': 'Language (graph translation takes some time)', 'German': 'Sprache (GraphÃ¼bersetzung dauert etwas)'},
            'enter_event': {'English': 'Enter a new health event', 'German': 'Geben Sie ein neues Gesundheitsereignis ein'},
            'placeholder_text': {
                'English': 'e.g., \'Woke up yesterday with a blinding headache\'',
                'German': 'z.B., \'Bin gestern mit starken Kopfschmerzen aufgewacht\''
            },
            'add_event_button': {'English': 'Add Event to Graph (Shift+Enter)', 'German': 'Ereignis zum Graph hinzufÃ¼gen (Shift+Enter)'},
            'chat_history': {'English': 'Chat History', 'German': 'Chat-Verlauf'},
            'no_separate': {'English': 'No, it\'s separate', 'German': 'Nein, es ist separat'},
            'yes_link': {'English': 'Yes, link them', 'German': 'Ja, verknÃ¼pfen'},
            'try_example': {'English': 'Try an Example', 'German': 'Beispiel ausprobieren'},
            'event_history_deletion': {'English': 'Event History & Deletion', 'German': 'Ereignisverlauf & LÃ¶schung'},
            'select_events_delete': {'English': 'Select event entries to delete', 'German': 'EreigniseintrÃ¤ge zum LÃ¶schen auswÃ¤hlen'},
            'delete_selected': {'English': 'Delete Selected Entries', 'German': 'AusgewÃ¤hlte EintrÃ¤ge lÃ¶schen'},
            'recenter_button': {'English': 'Recenter', 'German': 'Zentrieren'},
            'reset_button': {'English': 'Reset', 'German': 'ZurÃ¼cksetzen'},
            'zoom_in_button': {'English': 'Zoom In', 'German': 'Hineinzoomen'},
            'zoom_out_button': {'English': 'Zoom Out', 'German': 'Herauszoomen'},
            'processing_graph': {'English': 'Processing and updating graph...', 'German': 'Graph wird verarbeitet und aktualisiert...'},
            'all_filter': {'English': 'All', 'German': 'Alle'},
            'event_types': {
                'symptom': {'English': 'Symptom', 'German': 'Symptom'},
                'action_taken': {'English': 'Action Taken', 'German': 'MaÃŸnahme'},
                'visit': {'English': 'Visit', 'German': 'Besuch'}
            },
            'tooltip_fields': {
                'event_type': {'English': 'Event Type', 'German': 'Ereignistyp'},
                'medical_speciality': {'English': 'Medical Speciality', 'German': 'Medizinische Fachrichtung'},
                'anatomical_part': {'English': 'Anatomical Part', 'German': 'Anatomischer Teil'},
                'specific_body_region': {'English': 'Body Region', 'German': 'KÃ¶rperregion'},
                'user_commentary': {'English': 'User Commentary', 'German': 'Benutzerkommentar'},
                'descriptors': {'English': 'Descriptors', 'German': 'Beschreibungen'},
                'symptom_term': {'English': 'Symptom Term', 'German': 'Symptombegriff'},
                'symptom_quality': {'English': 'Symptom Quality', 'German': 'SymptomqualitÃ¤t'},
                'perceived_trigger': {'English': 'Perceived Trigger', 'German': 'Wahrgenommener AuslÃ¶ser'},
                'seriousness_score': {'English': 'Seriousness Score', 'German': 'Schweregrad-Bewertung'},
                'action_name': {'English': 'Action Name', 'German': 'MaÃŸnahme'},
                'description': {'English': 'Description', 'German': 'Beschreibung'},
                'outcome': {'English': 'Outcome', 'German': 'Ergebnis'},
                'reason': {'English': 'Reason', 'German': 'Grund'},
                'diagnosis': {'English': 'Diagnosis', 'German': 'Diagnose'},
                'recommendation': {'English': 'Recommendation', 'German': 'Empfehlung'}
            }
        }
    
    def _init_examples(self):
        """Initialize example prompts for both languages."""
        self.example_prompts = {
            'English': [
                'Woke up yesterday with a blinding headache.',
                'Finally saw the neurologist today about the weird headaches. She diagnosed it as a classic migraine and told me to take the prescribed medication when one starts.',
                'Just remembered that weird, itchy rash I had on my arm about half a year ago. It lasted a bit and then just vanished.',
                'I took an antihistamine last week and it made me incredibly drowsy the whole day.',
                'That awful heartburn is back again tonight. Seems to happen every time I have tomato sauce.',
                'My left shoulder has been a bit stiff and achy all morning, probably from how I slept on it.',
            ],
            'German': [
                'Ich bin gestern mit rasenden Kopfschmerzen aufgewacht.',
                'Heute war ich endlich bei der Neurologin wegen der seltsamen Kopfschmerzen. Sie diagnostizierte eine klassische MigrÃ¤ne und sagte mir, ich solle die verschriebenen Medikamente nehmen, wenn eine MigrÃ¤ne beginnt.',
                'Mir ist gerade dieser komische, juckende Ausschlag eingefallen, den ich vor einem halben Jahr am Arm hatte. Er war eine Zeit lang da und verschwand dann einfach.',
                'Ich habe letzte Woche ein Antihistaminikum genommen und es hat mich den ganzen Tag unglaublich schlÃ¤frig gemacht.',
                'Dieses schreckliche Sodbrennen ist heute Abend wieder da. Das passiert anscheinend immer, wenn ich TomatensoÃŸe esse.',
                'Meine linke Schulter ist schon den ganzen Morgen etwas steif und schmerzt, wahrscheinlich, weil ich falsch darauf gelegen habe.',
            ]
        }
    
    def set_language(self, language):
        """Set the current language for the app."""
        if language in ['English', 'German']:
            self.current_language = language
    
    def get_text(self, key, **kwargs):
        """Get translated text for the current language."""
        if key in self.ui_texts:
            text = self.ui_texts[key].get(self.current_language, self.ui_texts[key]['English'])
            return text.format(**kwargs) if kwargs else text
        return key
    
    def get_tooltip_text(self, field_key):
        """Get translated text for tooltip field labels."""
        if field_key in self.ui_texts['tooltip_fields']:
            return self.ui_texts['tooltip_fields'][field_key].get(
                self.current_language, 
                self.ui_texts['tooltip_fields'][field_key]['English']
            )
        return field_key.replace('_', ' ').title()
    
    def get_medical_specialties(self):
        """Get medical specialties for the current language."""
        return self.medical_specialties[self.current_language]
    
    def get_specialty_mapping(self, source_language, target_language):
        """Get specialty mapping between two languages."""
        if source_language == target_language:
            return {}
        
        mapping_key = f'{source_language}_to_{target_language}'
        return self.specialty_mappings.get(mapping_key, {})
    
    def format_specialty_mapping_for_prompt(self, source_language, target_language):
        """Format specialty mapping for use in translation prompts."""
        mapping = self.get_specialty_mapping(source_language, target_language)
        if not mapping:
            return 'No specialty mappings available.'
        
        formatted_lines = []
        for source, target in mapping.items():
            formatted_lines.append(f'- {source} â†’ {target}')
        
        return "\n".join(formatted_lines)
    
    def get_example_prompts(self):
        """Get example prompts for the current language."""
        return self.example_prompts[self.current_language]
    
    def get_event_type_translation(self, event_type):
        """Get translation for event type."""
        if event_type in self.ui_texts['event_types']:
            return self.ui_texts['event_types'][event_type][self.current_language]
        return event_type

localization = LocalizationManager()



# Prompts

from datetime import datetime

class PromptManager:
    """Manages LLM prompts for the health tracking app."""
    
    def __init__(self, localization_manager):
        self.localization = localization_manager
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize LLM prompts."""
        self.prompts = {
            'time_extraction_prompt': '''
                Today is {current_datetime}. User describes an event: {user_input}.
                Based on this event and the current date, what is the most likely date it occurred? Only output the estimated date in ISO format (YYYY-MM-DD HH:MM:SS).
            ''',
            'extraction_prompt': '''
                You are an expert medical data extractor. Your task is to analyze the user's input and extract every single medical event mentioned.

                ### Instructions
                1.  **Identify all events:** Carefully read the user's input and identify every distinct event. An event can be a 'symptom', 'action_taken', or 'visit'. It can be a single event or a series of related events. If you are not confident only create one event thats best fitting.
                2.  **Create one object per event:** For each event you identify, create one JSON object. 
                3.  **Generate a JSON array:** Your final output must be a single JSON array containing all the event objects you created.
                4.  **Strictly JSON output:** Your response must ONLY be the valid JSON array, with no additional text, explanations, or markdown formatting.
                5.  **Follow the schema:** Adhere strictly to the JSON schema, field types, and options provided below. If a specific piece of information is not available in the user's text, use 'null' for that field.
                
                ### Event Definitions
                -   **symptom:** A physical or mental feature indicating a medical condition.
                -   **action_taken:** An action the user has *already performed* or a medication they have *already taken*. This must be a completed act, such as taking a pill, performing a specific exercise, or applying a cream. If a doctor told the user to do something, it is NOT an 'action_taken' unless the user confirms they have already done it.
                -   **visit:** An appointment with a medical professional, such as a doctor, nurse, or therapist, for diagnosis, consultation, recommendations or prescriptions.
                -   **Distinguish Actions vs. Recommendations:** This is a critical rule. An action the user *has already taken* is an 'action_taken'. A suggestion, prescription, or order from a medical professional is a recommendation and belongs inside a 'visit' event. Do NOT create an 'action_taken' event for a recommendation.

                ### Rules for Adherence
                -   **Strictly Adhere to Options:** For any field with a list of Options[], you **MUST** use one of the exact strings from the provided list.
                -   **No Synonyms:** Do not use synonyms, abbreviations, or variations.
                -   **No Merging:** You MUST NOT merge multiple distinct events into one object.

                ### JSON Schema
                {{
                    "event_type": Options['symptom', 'action_taken', 'visit'], 
                    "medical_speciality": Options{medical_specialties}, // desc: MUST be estimated, choose the best fitting
                    "anatomical_part": str, // desc: the specific anatomical part of the body that is affected or targeted by an action taken.
                    "descriptors": {{
                        // desc: The descriptors of this object will change based on the "event_type"
                    }}
                }}

                // descriptors for "event_type" == "symptom"
                "descriptors": {{
                    "symptom_term": str, // desc: The layman's term for the medical symptom
                    "symptom_quality": str, // desc: The nature of the symptom
                    "perceived_trigger": str, // desc: What the user thinks caused the event.
                    "seriousness_score": float // desc: You MUST estimate this score on a scale of 0.0 (no issue) to 1.0 (emergency)
                }}

                // descriptors for "event_type" == "action_taken"
                "descriptors": {{
                    "action_name": str, // desc: Name of intervention, treatment, action, medication or self-care that was actively taken by the user. 
                    "description": str, // desc: More detailed description
                    "outcome": str // desc: Outcome of the action taken
                }}

                // descriptors for "event_type" == "visit"
                "descriptors": {{
                    "reason": str, // desc: Reason for the visit with the medical professional
                    "diagnosis": str,
                    "recommendation": str // desc: Prescribed therapy, medication or recommended actions by the medical professional
                }}

                User Input:

            ''',
            'linking_prompt': '''
                You are an expert medical data analyst. Your task is to determine if a new health event is a continuation of, or directly related to, a previous event from the user's history.

                - Analyze the new event and compare it against the historical events provided.
                - The historical events all occurred in the same anatomical location.
                - If you are confident the new event is a direct follow-up or related to a specific past event, respond ONLY with the "relation_id" of that historical event.
                - If there is no clear, direct connection, respond ONLY with the word "null".

                Do not provide any explanations or extra text. Your entire response must be either the "relation_id" or "null".

            ''',
            'translation_prompt': '''
                You are an expert medical translator. Your task is to translate medical health event data from one language to another while preserving all the semantic meaning and medical accuracy.

                ### Instructions
                1. **Translate all text fields**: Translate all string values in the provided JSON data to the target language.
                2. **Preserve structure**: Keep the exact same JSON structure, field names, and data types.
                3. **Medical accuracy**: Ensure medical terms are translated accurately using proper medical terminology.
                4. **Preserve null values**: Keep any null values as null.
                5. **Preserve numbers**: Keep all numeric values (timestamps, scores) unchanged.
                6. **Preserve IDs**: Keep all IDs and relation_ids unchanged.
                7. **Output format**: Return ONLY the translated JSON, no additional text or formatting.

                ### Translation Rules
                - Translate from: {source_language} to: {target_language}
                - Use appropriate medical terminology in the target language
                - Maintain the same level of detail and specificity
                - For anatomical parts, use standard medical terminology in the target language
                - For medical specialties, use ONLY the provided specialties from the reference list below

                ### Medical Specialties Translation Reference
                {specialties_mapping}

                ### Important Notes
                - When translating medical_speciality fields, you MUST use the exact terms from the specialties reference above
                - Do not create new medical specialty terms - only use the provided mappings
                - If a specialty is not in the reference, keep it unchanged

                JSON data to translate:

            '''
        }
    
    def get_prompt(self, prompt_name, **kwargs):
        """Get prompt with medical specialties for current language and format with additional parameters."""
        if prompt_name in self.prompts:
            prompt = self.prompts[prompt_name]
            if '{medical_specialties}' in prompt:
                specialties = self.localization.get_medical_specialties()
                prompt = prompt.replace('{medical_specialties}', str(specialties))
            if '{current_datetime}' in prompt:
                current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                prompt = prompt.replace('{current_datetime}', str(current_datetime))
            
            # Handle additional kwargs for time extraction
            for key, value in kwargs.items():
                placeholder = '{' + key + '}'
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, str(value))
            
            return prompt
        return ''

prompts = PromptManager(localization)


# Model Setup and Core AI Logic

import json
import uuid
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model='hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q8_K_XL',
    temperature=0
    
    # temperature=1.0,
    # top_k=64,
    # top_p=0.95,
    # min_p=0.0
)

# Prompt generation functions
def get_extraction_prompt():
    """Get the extraction prompt in the current language."""
    return prompts.get_prompt('extraction_prompt')

def get_time_extraction_prompt():
    """Get the time extraction prompt."""
    return prompts.get_prompt('time_extraction_prompt')

def get_linking_prompt():
    """Get the linking prompt in the current language."""
    return prompts.get_prompt('linking_prompt')

def get_translation_prompt():
    """Get the translation prompt."""
    return prompts.get_prompt('translation_prompt')

def create_extraction_chain():
    """Create extraction chain with current language prompt."""
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', get_extraction_prompt()),
        ('user', '{user_input}')
    ])
    return prompt_template | model | StrOutputParser()

def create_time_extraction_chain():
    """Create time extraction chain."""
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', get_time_extraction_prompt()),
        ('user', '{user_input}')
    ])
    return prompt_template | model | StrOutputParser()

def create_linking_chain():
    """Create linking chain with current language prompt."""
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', get_linking_prompt()),
        ('user', 'New Event:\n{new_event}\n\nHistorical Events:\n{historical_events}')
    ])
    return prompt_template | model | StrOutputParser()

def create_translation_chain():
    """Create translation chain for translating graph entries."""
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', get_translation_prompt()),
        ('user', '{json_data}')
    ])
    return prompt_template | model | StrOutputParser()


# Based on Google's business email assistant example (https://github.com/google-gemini/gemma-cookbook/tree/main/Demos/business-email-assistant/email-processing-webapp)
def format_response(text):
    """Format the model's structured response to a json string and finally to a python dict."""
    # Clean up the response text
    text = text.replace('```json', '').replace('```', '').strip().lower()
    return json.loads(text)

def print_data(data):
    """Pretty print python dict for debugging purposes."""
    print(json.dumps(data, indent=2))

def extract_timestamps_from_input(user_input):
    """Extract timestamp from user input using a separate LLM call."""
    try:
        time_extraction_chain = create_time_extraction_chain()
        response = time_extraction_chain.invoke({'user_input': user_input})
        
        timestamp_str = response.strip().replace('```', '').replace('`', '')
        
        try:
            dt = datetime.fromisoformat(timestamp_str)
            return dt.timestamp()
        except (ValueError, TypeError):
            print(f'Failed to parse timestamp: {timestamp_str}')
            return datetime.now().timestamp()
            
    except Exception as e:
        print(f'Error extracting timestamp: {e}')
        return datetime.now().timestamp()

def generate_event(user_input):
    """Generate desired json response based on the given template structure."""
    try:
        extracted_timestamp = extract_timestamps_from_input(user_input)
        
        extraction_chain = create_extraction_chain()
        response = extraction_chain.invoke({'user_input': user_input})
        events = format_response(response)

        relation_id = str(uuid.uuid4())
        
        for event in events:
            event['timestamp'] = extracted_timestamp
            event['user_input'] = user_input
            event['relation_id'] = relation_id
        
        return events
    except Exception as e:
        print(f'Error generating event: {e}')
        print(f'Response: {response}')
        raise


# Graph Modifications

def find_proposed_link(new_events, full_history):
    """Use the LLM to find ONE proposed link and return the proposal."""
    for new_event in new_events:
        anatomical_part = new_event.get('anatomical_part')
        if not anatomical_part:
            continue

        current_relation_id = new_event.get('relation_id')
        relevant_history = [
            e for e in full_history
            if (e.get('anatomical_part') == anatomical_part and 
                e.get('relation_id') != current_relation_id)
        ]

        if relevant_history:
            try:
                linking_chain = create_linking_chain()
                response = linking_chain.invoke({
                    'new_event': json.dumps(new_event, indent=2),
                    'historical_events': json.dumps(relevant_history, indent=2)
                })
                linked_id = response.strip().lower().replace('`', '')
                
                past_event_candidate = next(
                    (e for e in relevant_history if e['relation_id'] == linked_id), 
                    None
                )
                
                if past_event_candidate:
                    desc = past_event_candidate.get('descriptors', {})
                    past_event_term = (
                        desc.get('symptom_term') or 
                        desc.get('action_name') or 
                        desc.get('reason') or 
                        'past event'
                    )
                    past_user_input = past_event_candidate.get('user_input', 'No original text found.')
                    
                    return {
                        'new_event_relation_id': new_event['relation_id'],
                        'past_event_relation_id': linked_id,
                        'past_event_term': past_event_term,
                        'past_event_user_input': past_user_input
                    }
            except Exception:
                pass  # Silently handle errors during link proposal
    return None

def translate_graph_entries(graph_data, target_language, current_language):
    """Translate all graph entries between languages."""
    if not graph_data or target_language == current_language:
        return graph_data
    
    try:
        import copy
        translated_graph = copy.deepcopy(graph_data)
        
        # Get the specialty mapping for the translation prompt
        specialty_mapping = localization.format_specialty_mapping_for_prompt(current_language, target_language)
        
        # Prepare the translation prompt with all necessary parameters
        translation_prompt = get_translation_prompt().format(
            source_language=current_language,
            target_language=target_language,
            specialties_mapping=specialty_mapping
        )
        
        # Translate in batches to avoid overwhelming the model
        batch_size = 5
        for i in range(0, len(translated_graph), batch_size):
            batch = translated_graph[i:i + batch_size]
            
            # Prepare JSON data for translation
            json_data = json.dumps(batch, indent=2)
            
            # Create a custom prompt template for this translation
            custom_translation_prompt = ChatPromptTemplate.from_messages([
                ('system', translation_prompt),
                ('user', '{json_data}')
            ])
            
            # Create chain with custom prompt
            custom_chain = custom_translation_prompt | model | StrOutputParser()
            
            # Get translation
            response = custom_chain.invoke({'json_data': json_data})
            translated_batch = format_response(response)
            
            # Update the translated graph with the batch results
            for j, translated_event in enumerate(translated_batch):
                if i + j < len(translated_graph):
                    # Preserve original non-translatable fields
                    original_event = translated_graph[i + j]
                    translated_event['timestamp'] = original_event['timestamp']
                    translated_event['relation_id'] = original_event['relation_id']
                    if 'linked_to_relation_id' in original_event:
                        translated_event['linked_to_relation_id'] = original_event['linked_to_relation_id']
                    
                    translated_graph[i + j] = translated_event
        
        return translated_graph
        
    except Exception as e:
        # Print error for debugging but still return original data
        print(f'Translation error: {e}')
        return graph_data
    
def filter_graph_data(graph_data, event_type, anatomical_part, time_range):
    """Filter the graph data based on selected criteria, including a time range."""
    if not graph_data: 
        print('Debug: No graph data provided')
        return []

    start_ts, end_ts = None, None
    if time_range:
        if isinstance(time_range, dict):
            # Handle Plotly-style time range
            if not time_range.get('xaxis.autorange'):
                start_dt_str = time_range.get('xaxis.range[0]')
                end_dt_str = time_range.get('xaxis.range[1]')
                try:
                    if start_dt_str: start_ts = datetime.fromisoformat(start_dt_str.split('.')[0]).timestamp()
                    if end_dt_str: end_ts = datetime.fromisoformat(end_dt_str.split('.')[0]).timestamp()
                except (ValueError, TypeError):
                    start_ts, end_ts = None, None
        elif isinstance(time_range, tuple) and len(time_range) == 2:
            # Handle timeline-style time range (start_iso, end_iso)
            try:
                start_ts = datetime.fromisoformat(time_range[0].replace('Z', '+00:00')).timestamp()
                end_ts = datetime.fromisoformat(time_range[1].replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError, AttributeError):
                start_ts, end_ts = None, None

    all_text = localization.get_text('all_filter')
    filtered_events = []
    
    for event in graph_data:
        matches_type = (event_type == all_text or event.get('event_type') == event_type)
        matches_part = (anatomical_part == all_text or event.get('anatomical_part') == anatomical_part)
        
        matches_time = True
        if start_ts and end_ts and (event_ts := event.get('timestamp')):
            matches_time = start_ts <= event_ts <= end_ts

        if matches_type and matches_part and matches_time:
            filtered_events.append(event)
            
    return filtered_events


# General Helper

def get_heatmap_color(score):
    """Calculate a color from yellow to red based on a 0.0-1.0 score."""
    score = max(0.0, min(1.0, float(score)))
    r = 255
    g = int(255 * (1 - score**1.5))
    b = int(64 * (1 - score**1.5))
    return f'#{r:02x}{g:02x}{b:02x}'

def hex_to_rgba(hex_color, opacity):
    """Convert a hex color string to an rgba string with specified opacity."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {opacity})'


# Graph Generation

import base64
from pyvis.network import Network


def format_event_details_as_text(event, translator):
    """Format an event into a human-readable plain text string for tooltips."""
    details = []
    
    def add_detail(key, value):
        if value is not None:
            label = translator.get_tooltip_text(key)
            display_value = str(value).capitalize() if isinstance(value, str) else value
            details.append(f'{label}: {display_value}')

    event_type_key = event.get('event_type', 'event_type')
    event_type_value = translator.get_event_type_translation(event_type_key)
    details.append(f'{translator.get_tooltip_text("event_type")}: {event_type_value.capitalize()}')
    details.append('â”€' * 23)

    add_detail('medical_speciality', event.get('medical_speciality'))
    add_detail('anatomical_part', event.get('anatomical_part'))
    
    descriptors = event.get('descriptors', {})
    if descriptors:
        details.append('â”€' * 23)
        for key, value in descriptors.items():
            add_detail(key, value)
            
    if user_input := event.get('user_input'):
        details.append('â”€' * 23)
        details.append(f'"{user_input}"')

    return '\n'.join(details)

def add_main_event_node(net, event, event_id, time_context, localization):
    """Add a main node for each event."""
    descriptors = event.get('descriptors', {})
    event_type = event.get('event_type')
    color_map = {
        'symptom': lambda: (get_heatmap_color(descriptors.get('seriousness_score', 0.0)), descriptors.get('symptom_term')),
        'action_taken': lambda: ('#4CAF50', descriptors.get('action_name')),
        'visit': lambda: ('#2196F3', descriptors.get('reason'))
    }
    hex_color, raw_label = color_map.get(event_type, lambda: ('#9E9E9E', 'Unknown Event'))()
    label = raw_label.capitalize() if isinstance(raw_label, str) else raw_label
    tooltip_text = format_event_details_as_text(event, localization)
    net.add_node(
        event_id, label=label, shape='dot', size=18,
        color={'border': hex_color, 'background': hex_to_rgba(hex_color, 0.3), 'highlight': {'border': hex_color, 'background': hex_to_rgba(hex_color, 0.5)}, 'hover': {'border': hex_color, 'background': hex_to_rgba(hex_color, 0.5)}},
        borderWidth=3, font={'size': 16, 'face': 'sans-serif'}, title=tooltip_text,
        shadow={'enabled': True, 'size': 5, 'x': 2, 'y': 2, 'color': 'rgba(0,0,0,0.2)'}
    )

def add_and_connect_tag_nodes(net, event, event_id, created_tag_nodes):
    """Add nodes for medical speciality/parts and connect them to the event."""
    speciality = event.get('medical_speciality')
    anatomical_part = event.get('anatomical_part')
    if not speciality and not anatomical_part: return
    if speciality and speciality not in created_tag_nodes:
        net.add_node(
            speciality, label=speciality.capitalize(), shape='box', color='#616161',
            font={'color': 'white', 'size': 22, 'face': 'sans-serif'}, borderWidth=0,
            shapeProperties={'borderRadius': 4}, shadow={'enabled': True, 'size': 5, 'x': 2, 'y': 2, 'color': 'rgba(0,0,0,0.2)'}
        )
        created_tag_nodes.add(speciality)
    if anatomical_part and anatomical_part not in created_tag_nodes:
        net.add_node(
            anatomical_part, label=anatomical_part.capitalize(), shape='box',
            color={'background': '#FFFFFF', 'border': '#BDBDBD'}, borderWidth=1,
            font={'color': '#212121', 'size': 18, 'face': 'sans-serif'}, shapeProperties={'borderRadius': 4},
            shadow={'enabled': True, 'size': 5, 'x': 2, 'y': 2, 'color': 'rgba(0,0,0,0.1)'}
        )
        created_tag_nodes.add(anatomical_part)
    if speciality and anatomical_part:
        net.add_edge(speciality, anatomical_part, color='#BDBDBD')
        net.add_edge(anatomical_part, event_id, color='#BDBDBD')
    elif anatomical_part:
        net.add_edge(anatomical_part, event_id, color='#BDBDBD')
    elif speciality:
        net.add_edge(speciality, event_id, color='#BDBDBD')

def add_graph_nodes_and_edges(net, graph_data, localization):
    """Populate the pyvis network with nodes and edges."""
    if not graph_data:
        return
        
    created_tag_nodes = set()
    created_nodes = set()
    existing_relation_ids = set()
    
    # Collect all relation IDs that exist in this dataset
    for event in graph_data:
        relation_id = event.get('relation_id')
        if relation_id:
            existing_relation_ids.add(relation_id)
    
    # Group events by relation_id for processing
    relation_to_events = {}
    for event in graph_data:
        relation_id = event.get('relation_id')
        if relation_id:
            if relation_id not in relation_to_events:
                relation_to_events[relation_id] = []
            relation_to_events[relation_id].append(event)
    
    # Create nodes and internal edges for each relation_id group
    for relation_id, events in relation_to_events.items():
        # Sort events by timestamp for consistent ordering
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        for i, event in enumerate(sorted_events):
            # Create unique node ID for each event
            node_id = f'rel_{relation_id}_{i}'
            
            # Add the main event node
            add_main_event_node(net, event, node_id, None, localization)
            add_and_connect_tag_nodes(net, event, node_id, created_tag_nodes)
            created_nodes.add(node_id)
            
            # Connect to previous event in same relation (same user input)
            if i > 0:
                prev_node_id = f'rel_{relation_id}_{i-1}'
                net.add_edge(prev_node_id, node_id, color='#87CEEB', dashes=True, title='Linked from same input')
    
    # Add edges for actively linked events (between different relations)
    for relation_id, events in relation_to_events.items():
        for i, event in enumerate(events):
            linked_relation_id = event.get('linked_to_relation_id')
            
            if (linked_relation_id and 
                linked_relation_id in existing_relation_ids and
                linked_relation_id != relation_id):
                
                # Current event node
                current_node_id = f'rel_{relation_id}_{i}'
                
                # Target the first event of the linked relation
                target_node_id = f'rel_{linked_relation_id}_0'
                
                # Only create edge if both nodes exist
                if (current_node_id in created_nodes and 
                    target_node_id in created_nodes):
                    net.add_edge(target_node_id, current_node_id, color='#87CEEB', dashes=[5, 5], title='Actively linked', width=2)

def set_network_options(net):
    """Set predefined physics and edge options for the graph network."""
    net.options.edges.smooth.enabled = True
    net.options.edges.smooth.type = 'dynamic'
    net.options.interaction.hover = True
    net.options.interaction.hover_connected_edges = False
    net.options.interaction.tooltip_delay = 200
    net.options.physics.min_velocity = 0.75
    net.options.physics.solver = 'forceAtlas2Based'
    net.options.physics.forceAtlas2Based = {'gravitationalConstant': -100, 'centralGravity': 0.01, 'springLength': 200, 'springConstant': 0.05, 'damping': 0.8, 'avoidOverlap': 0.5}

def generate_graph_iframe(graph_data, localization):
    """Create the pyvis graph from data and return a self-contained iframe using Base64 Data URI."""
    net = Network(notebook=True, height='600px', width='100%', cdn_resources='in_line')
    if graph_data:
        add_graph_nodes_and_edges(net, graph_data, localization)
        set_network_options(net)
    
    html_content = net.generate_html()
    
    # Custom CSS for tooltip styling to match timeline tooltips
    tooltip_css = '''
    <style>
        .vis-tooltip {
            background: white !important;
            border: 1px solid #ccc !important;
            border-radius: 4px !important;
            padding: 8px !important;
            font-size: 12px !important;
            font-family: sans-serif !important;
            color: #333 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            max-width: 300px !important;
            white-space: pre-line !important;
            z-index: 1000 !important;
            line-height: 1.4 !important;
        }
        .custom-graph-buttons { 
            position: absolute; 
            top: 20px; 
            left: 10px; 
            z-index: 1000; 
            display: flex; 
            flex-direction: column; 
            gap: 5px; 
        }
        .custom-graph-buttons button { 
            background-color: #f0f0f0; 
            border: 1px solid #ccc; 
            border-radius: 4px; 
            padding: 5px 10px; 
            cursor: pointer; 
            font-family: sans-serif; 
            font-size: 12px; 
        }
        .custom-graph-buttons button:hover { 
            background-color: #e0e0e0; 
        }
    </style>
    '''
    
    button_injection = f'''
    <div class="custom-graph-buttons">
        <button id="recenterBtn">{localization.get_text('recenter_button')}</button>
    </div>
    <script type="text/javascript">
        function setupCustomButtons() {{
            if (typeof network !== 'undefined') {{
                document.getElementById('recenterBtn').addEventListener('click', () => {{
                    network.fit();
                }});
            }} else {{
                setTimeout(setupCustomButtons, 100);
            }}
        }}
        setupCustomButtons();
    </script>
    '''
    
    final_html = html_content.replace('</head>', tooltip_css + '</head>').replace('</body>', button_injection + '</body>')
    # Encode as base64 for iframe with proper UTF-8 encoding
    html_content_bytes = final_html.encode('utf-8')
    base64_encoded_html = base64.b64encode(html_content_bytes).decode('ascii')
    iframe_src = f'data:text/html;charset=utf-8;base64,{base64_encoded_html}'
    return f'<iframe src="{iframe_src}" width="100%" height="620px" style="border:none; border-radius: 0px;"></iframe>'


# Timeline 

def generate_timeline_iframe(graph_data, localization, reset_key=None):
    """Create a custom interactive timeline with zoom/pan functionality."""
    if not graph_data:
        return '<div style="height: 200px; display: flex; align-items: center; justify-content: center; background: #f9f9f9; border: 0px solid #e0e0e0; border-radius: 8px; color: #888;">No events to display on timeline.</div>'
    
    # Add a reset key to force timeline to start fresh when needed
    import time
    if reset_key is None:
        reset_key = int(time.time() * 1000)  # Use current timestamp as default
    
    # Prepare timeline data with logical Y-positioning by event type
    timeline_events = []
    sorted_events = sorted(graph_data, key=lambda x: x['timestamp'])
    
    # Define fixed Y-positions for each event type (consistent positioning)
    y_positions = {
        'visit': 30,        # Top section (blue)
        'symptom': 60,      # Middle section (red/yellow gradient)
        'action_taken': 90  # Bottom section (green)
    }
    
    for event in sorted_events:
        event_type = event.get('event_type', 'symptom')
        
        # Use fixed Y position based on event type
        y_pos = y_positions.get(event_type, 60)

        descriptors = event.get('descriptors', {})
        
        color_map = {
            'symptom': get_heatmap_color(descriptors.get('seriousness_score', 0.0)), 
            'action_taken': '#4CAF50', 
            'visit': '#2196F3'
        }
        label_map = {
            'symptom': descriptors.get('symptom_term', 'Symptom'), 
            'action_taken': descriptors.get('action_name', 'Action'), 
            'visit': descriptors.get('reason', 'Visit')
        }
        
        # Create simplified tooltip with main label, medical speciality and anatomical part
        tooltip_parts = []
        
        # Add the main event label
        event_label = str(label_map.get(event_type, 'Event')).capitalize()
        tooltip_parts.append(f"Event: {event_label}")
        
        # Add medical speciality and anatomical part
        if event.get('medical_speciality'):
            tooltip_parts.append(f"Medical Speciality: {event['medical_speciality'].capitalize()}")
        if event.get('anatomical_part'):
            tooltip_parts.append(f"Anatomical Part: {event['anatomical_part'].capitalize()}")
        
        simple_tooltip = '\\n'.join(tooltip_parts) if tooltip_parts else 'No additional details'
        
        timeline_events.append({
            'timestamp': event['timestamp'],
            'date': datetime.fromtimestamp(event['timestamp']).date().isoformat(),
            'event_type': event_type,
            'x': 0,  # Will be calculated in JavaScript
            'y': y_pos,
            'label': str(label_map.get(event_type, 'Event')).capitalize(),
            'color': color_map.get(event_type, '#9E9E9E'),
            'tooltip': simple_tooltip.replace("'", "\\'").replace('"', '\\"'),
            'relation_id': event.get('relation_id', ''),
            'linked_to_relation_id': event.get('linked_to_relation_id', ''),
            'original_index': len(timeline_events)  # Track original position for connections
        })
    
    # Create custom timeline HTML
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; padding: 0; font-family: sans-serif; }}
            .timeline-container {{ 
                width: 100%; 
                height: 150px; 
                background: white; 
                border: 0px solid #e0e0e0; 
                border-radius: 0px; 
                position: relative; 
                overflow: hidden;
                cursor: grab;
            }}
            .timeline-container:active {{ cursor: grabbing; }}
            .timeline-svg {{ width: 100%; height: 100%; }}
            .event-circle {{ 
                cursor: pointer; 
                stroke: white; 
                stroke-width: 2; 
                opacity: 0.8;
                transition: all 0.2s ease;
            }}
            .event-circle:hover {{ 
                opacity: 1; 
                stroke-width: 3;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
            }}
            .event-circle.selected {{ 
                stroke-width: 4; 
                opacity: 1;
                filter: drop-shadow(0 3px 6px rgba(0,0,0,0.3));
            }}
            .event-circle.connected {{ 
                stroke-width: 3; 
                opacity: 0.9;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15));
            }}
            .connection-line {{ 
                stroke-width: 2; 
                fill: none; 
                opacity: 0.6; 
                transition: all 0.2s ease;
            }}
            .connection-line.highlighted {{ 
                stroke-width: 3; 
                opacity: 0.8;
                filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));
            }}
            .same-input-line {{ stroke: #87CEEB; stroke-dasharray: 3,3; }}
            .active-link-line {{ stroke: #87CEEB; stroke-dasharray: 5,5; }}
            .axis-line {{ stroke: #e0e0e0; stroke-width: 1; }}
            .axis-text {{ font-size: 12px; fill: #666; text-anchor: middle; }}
            .axis-label {{ font-size: 11px; fill: #666; text-anchor: start; font-weight: bold; }}
            .timeline-controls {{
                position: absolute;
                top: 10px;
                left: 10px;
                display: flex;
                gap: 5px;
            }}
            .timeline-button {{
                background: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 10px;
                cursor: pointer;
            }}
            .timeline-button:hover {{ background: #e0e0e0; }}
            .tooltip {{
                position: absolute;
                background: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
                z-index: 1000;
                display: none;
                max-width: 300px;
                white-space: pre-line;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="timeline-container" id="timelineContainer_{reset_key}">
            <svg class="timeline-svg" id="timelineSvg_{reset_key}">
                <!-- Timeline content will be drawn here -->
            </svg>
            <div class="timeline-controls">
                <button class="timeline-button" onclick="resetZoom_{reset_key}()">{localization.get_text('reset_button')}</button>
                <button class="timeline-button" onclick="zoomIn_{reset_key}()">{localization.get_text('zoom_in_button')}</button>
                <button class="timeline-button" onclick="zoomOut_{reset_key}()">{localization.get_text('zoom_out_button')}</button>
                <!-- <button class="timeline-button" onclick="showAllEvents_{reset_key}()">Show All</button> -->
            </div>
            <div class="tooltip" id="tooltip_{reset_key}"></div>
        </div>
        
        <script>
            (function() {{
                // Encapsulate everything in a unique scope using the reset key
                const TIMELINE_ID = '{reset_key}';
                const events = {timeline_events};
            let svgWidth = 800;
            let svgHeight = 140;
            let margin = {{ left: 80, right: 40, top: 20, bottom: 30 }};
            let plotWidth = svgWidth - margin.left - margin.right;
            let plotHeight = svgHeight - margin.top - margin.bottom;
            
            // Convert timestamps to dates and find min/max
            const timestamps = events.map(e => e.timestamp * 1000); // Convert to milliseconds
            const minDate = new Date(Math.min(...timestamps));
            const maxDate = new Date(Math.max(...timestamps));
            const totalRange = maxDate - minDate || 86400000; // At least 1 day
            
            // Zoom and pan state
            let zoomLevel = 1;
            let panOffset = 0;
            let isDragging = false;
            let lastMouseX = 0;
            
            // Minimum zoom range for detailed viewing (2 weeks in milliseconds)
            const minZoomRange = 14 * 24 * 60 * 60 * 1000; // 2 weeks
            const maxZoomLevel = Math.max(10, totalRange / minZoomRange); // Dynamic max zoom based on data range
            
            // Track current selection state
            let currentSelectedRelationId = null;
            let currentLinkedToRelationId = null;
            
            function getScale() {{
                return plotWidth / (totalRange / zoomLevel);
            }}
            
            function getXPosition(timestamp) {{
                const date = new Date(timestamp * 1000);
                const relativePos = ((date - minDate) / totalRange);
                return margin.left + relativePos * plotWidth * zoomLevel + panOffset;
            }}
            
            function drawTimeline() {{
                const svg = document.getElementById('timelineSvg_' + TIMELINE_ID);
                svg.innerHTML = '';
                
                // Update SVG dimensions
                const container = document.getElementById('timelineContainer_' + TIMELINE_ID);
                svgWidth = container.clientWidth;
                plotWidth = svgWidth - margin.left - margin.right;
                svg.setAttribute('width', svgWidth);
                svg.setAttribute('height', svgHeight);
                
                // Reset selection state when redrawing (since DOM elements are recreated)
                currentSelectedRelationId = null;
                currentLinkedToRelationId = null;
                
                // Calculate X positions for all events first
                events.forEach(event => {{
                    event.x = getXPosition(event.timestamp);
                }});
                
                // Draw connections first (so they appear behind the circles)
                drawConnections(svg);
                
                // Draw Y-axis type labels and reference lines
                svg.innerHTML += `<line class="axis-line" x1="85" y1="50" x2="${{svgWidth - margin.right}}" y2="50" stroke="#e0e0e0" stroke-width="1" stroke-dasharray="2,2" opacity="0.5"></line>`;
                svg.innerHTML += `<text class="axis-label" x="5" y="54" style="font-size: 11px; fill: #2196F3; font-weight: bold;">{localization.get_event_type_translation('visit')}</text>`;
                
                svg.innerHTML += `<line class="axis-line" x1="85" y1="80" x2="${{svgWidth - margin.right}}" y2="80" stroke="#e0e0e0" stroke-width="1" stroke-dasharray="2,2" opacity="0.5"></line>`;
                svg.innerHTML += `<text class="axis-label" x="5" y="84" style="font-size: 11px; fill: #ef5350; font-weight: bold;">{localization.get_event_type_translation('symptom')}</text>`;
                
                svg.innerHTML += `<line class="axis-line" x1="85" y1="110" x2="${{svgWidth - margin.right}}" y2="110" stroke="#e0e0e0" stroke-width="1" stroke-dasharray="2,2" opacity="0.5"></line>`;
                svg.innerHTML += `<text class="axis-label" x="5" y="114" style="font-size: 11px; fill: #4CAF50; font-weight: bold;">{localization.get_event_type_translation('action_taken')}</text>`;
                
                // Draw events
                events.forEach((event, index) => {{
                    const x = event.x;
                    const y = margin.top + event.y;
                    
                    // Only draw if visible - use different margins based on zoom level
                    const leftMargin = zoomLevel === 1 ? margin.left - 5 : margin.left + 10;
                    if (x >= leftMargin && x <= svgWidth - margin.right + 20) {{
                        svg.innerHTML += `
                            <circle class="event-circle" 
                                cx="${{x}}" cy="${{y}}" r="7" 
                                fill="${{event.color}}"
                                data-relation-id="${{event.relation_id}}"
                                data-linked-to="${{event.linked_to_relation_id || ''}}"
                                data-event-index="${{index}}"
                                onmouseover="showTooltip_${{TIMELINE_ID}}(event, '${{event.tooltip.replace(/'/g, "\\'")}}')"
                                onmouseout="hideTooltip_${{TIMELINE_ID}}()"
                                onclick="selectRelatedEvents_${{TIMELINE_ID}}('${{event.relation_id}}', '${{event.linked_to_relation_id}}')">
                            </circle>
                        `;
                    }}
                }});
                
                // Draw dynamic date labels and vertical grid lines
                drawDateLabels(svg);
            }}
            
            function drawDateLabels(svg) {{
                // Calculate visible time range based on the SAME scaling used for events
                const scaleStartTime = minDate.getTime();
                const scaleEndTime = maxDate.getTime();
                
                // Calculate pixels per millisecond for the current zoom level
                const pixelsPerMs = (plotWidth * zoomLevel) / totalRange;
                
                // Calculate the currently visible time range using the exact same logic as event positioning
                const visibleStartRatio = Math.max(0, -panOffset / (plotWidth * zoomLevel));
                const visibleEndRatio = Math.min(1, (svgWidth - margin.left - margin.right - panOffset) / (plotWidth * zoomLevel));
                
                const visibleStartTime = scaleStartTime + visibleStartRatio * totalRange;
                const visibleEndTime = scaleStartTime + visibleEndRatio * totalRange;
                
                // Define time intervals with minimum pixel spacing
                const intervals = [
                    {{ ms: 24 * 60 * 60 * 1000, format: 'day', minWidth: 60 }},
                    {{ ms: 7 * 24 * 60 * 60 * 1000, format: 'week', minWidth: 80 }},
                    {{ ms: 30 * 24 * 60 * 60 * 1000, format: 'month', minWidth: 100 }},
                    {{ ms: 90 * 24 * 60 * 60 * 1000, format: 'quarter', minWidth: 120 }},
                    {{ ms: 365 * 24 * 60 * 60 * 1000, format: 'year', minWidth: 150 }}
                ];
                
                // Find the best interval for current zoom level
                let selectedInterval = intervals[intervals.length - 1]; // Default to largest
                for (let interval of intervals) {{
                    const pixelSpacing = interval.ms * pixelsPerMs;
                    if (pixelSpacing >= interval.minWidth) {{
                        selectedInterval = interval;
                        break;
                    }}
                }}
                
                // Calculate the interval step and generate label positions
                const intervalMs = selectedInterval.ms;
                
                // Find the first label position before or at the visible start
                const firstLabelTime = Math.floor(visibleStartTime / intervalMs) * intervalMs;
                
                // Generate labels at regular intervals
                for (let time = firstLabelTime; time <= visibleEndTime + intervalMs; time += intervalMs) {{
                    // Calculate X position using the EXACT SAME formula as events
                    const labelX = margin.left + ((time - minDate.getTime()) / totalRange) * plotWidth * zoomLevel + panOffset;
                    
                    // Only draw if label is visible and doesn't overlap with y-axis labels
                    if (labelX >= margin.left + 10 && labelX <= svgWidth - margin.right + 50) {{
                        // Draw vertical grid line
                        svg.innerHTML += `<line class="axis-line" x1="${{labelX}}" y1="35" x2="${{labelX}}" y2="${{svgHeight - 25}}" stroke="#f0f0f0" stroke-width="1" opacity="0.7"></line>`;
                        
                        // Format date label based on interval type
                        const date = new Date(time);
                        let dateStr;
                        
                        if (selectedInterval.format === 'day') {{
                            dateStr = date.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }});
                        }} else if (selectedInterval.format === 'week') {{
                            dateStr = date.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }});
                        }} else if (selectedInterval.format === 'month') {{
                            dateStr = date.toLocaleDateString('en-US', {{ month: 'short', year: '2-digit' }});
                        }} else if (selectedInterval.format === 'quarter') {{
                            const quarter = Math.floor(date.getMonth() / 3) + 1;
                            dateStr = `Q${{quarter}} '${{date.getFullYear().toString().slice(-2)}}`;
                        }} else {{
                            dateStr = date.getFullYear().toString();
                        }}
                        
                        svg.innerHTML += `<text class="axis-text" x="${{labelX}}" y="${{svgHeight - 8}}">${{dateStr}}</text>`;
                    }}
                }}
            }}
            
            function drawConnections(svg) {{
                // Create mapping from relation_id to events for quick lookup
                const relationToEvents = {{}};
                events.forEach(event => {{
                    const relationId = event.relation_id;
                    if (!relationToEvents[relationId]) {{
                        relationToEvents[relationId] = [];
                    }}
                    relationToEvents[relationId].push(event);
                }});
                
                // Draw same user input connections (for events with same relation_id)
                Object.values(relationToEvents).forEach(relatedEvents => {{
                    if (relatedEvents.length > 1) {{
                        // Sort by timestamp to connect in chronological order
                        relatedEvents.sort((a, b) => a.timestamp - b.timestamp);
                        
                        for (let i = 0; i < relatedEvents.length - 1; i++) {{
                            const event1 = relatedEvents[i];
                            const event2 = relatedEvents[i + 1];
                            
                            // Only draw if both events are visible - use different margins based on zoom level
                            const leftMargin = zoomLevel === 1 ? margin.left - 5 : margin.left + 10;
                            if (event1.x >= leftMargin && event1.x <= svgWidth - margin.right + 20 &&
                                event2.x >= leftMargin && event2.x <= svgWidth - margin.right + 20) {{
                                
                                const x1 = event1.x;
                                const y1 = margin.top + event1.y;
                                const x2 = event2.x;
                                const y2 = margin.top + event2.y;
                                
                                // Create curved connection with offset to avoid overlapping other events
                                const path = createOffsetPath(x1, y1, x2, y2, events, margin.top);
                                
                                svg.innerHTML += `<path d="${{path}}" class="connection-line same-input-line" 
                                    data-from-relation="${{event1.relation_id}}" 
                                    data-to-relation="${{event2.relation_id}}"
                                    title="Same user input connection"></path>`;
                            }}
                        }}
                    }}
                }});
                
                // Draw active link connections (linked_to_relation_id)
                events.forEach(event => {{
                    if (event.linked_to_relation_id) {{
                        // Find the target event(s) with the linked relation_id
                        const targetEvents = relationToEvents[event.linked_to_relation_id];
                        if (targetEvents && targetEvents.length > 0) {{
                            // Connect to the first (earliest) event with that relation_id
                            const targetEvent = targetEvents[0];
                            
                            // Only draw if both events are visible
                            const leftMargin = zoomLevel === 1 ? margin.left - 5 : margin.left + 10;
                            if (event.x >= leftMargin && event.x <= svgWidth - margin.right + 20 &&
                                targetEvent.x >= leftMargin && targetEvent.x <= svgWidth - margin.right + 20) {{
                                
                                const x1 = targetEvent.x;
                                const y1 = margin.top + targetEvent.y;
                                const x2 = event.x;
                                const y2 = margin.top + event.y;
                                
                                // Create curved connection with offset
                                const path = createOffsetPath(x1, y1, x2, y2, events, margin.top);
                                
                                svg.innerHTML += `<path d="${{path}}" class="connection-line active-link-line" 
                                    data-from-relation="${{targetEvent.relation_id}}" 
                                    data-to-relation="${{event.relation_id}}"
                                    title="Active link connection"></path>`;
                            }}
                        }}
                    }}
                }});
            }}
            
            function createOffsetPath(x1, y1, x2, y2, allEvents, topMargin) {{
                // Calculate if there are events between these two points that might be overlapped
                const minX = Math.min(x1, x2);
                const maxX = Math.max(x1, x2);
                const minY = Math.min(y1, y2);
                const maxY = Math.max(y1, y2);
                
                // Check for events in the path area
                let hasIntermediateEvents = false;
                allEvents.forEach(event => {{
                    const eventX = event.x;
                    const eventY = topMargin + event.y;
                    if (eventX > minX && eventX < maxX && 
                        eventY >= minY - 15 && eventY <= maxY + 15) {{
                        hasIntermediateEvents = true;
                    }}
                }});
                
                // Create a curved path with offset if there are intermediate events
                const offset = hasIntermediateEvents ? 25 : 10;
                const midX = (x1 + x2) / 2;
                const midY = Math.min(y1, y2) - offset;
                
                // Create quadratic bezier curve
                return `M ${{x1}} ${{y1}} Q ${{midX}} ${{midY}} ${{x2}} ${{y2}}`;
            }}
            
            function showTooltip(event, text) {{
                const tooltip = document.getElementById('tooltip_' + TIMELINE_ID);
                tooltip.style.display = 'block';
                tooltip.innerHTML = text.replace(/\\\\n/g, '<br>');
                
                // Get mouse position relative to the timeline container
                const rect = document.getElementById('timelineContainer_' + TIMELINE_ID).getBoundingClientRect();
                const mouseX = event.clientX - rect.left;
                const mouseY = event.clientY - rect.top;
                
                // Position tooltip near mouse but keep it in bounds
                let tooltipX = mouseX + 10;
                let tooltipY = mouseY - 10;
                
                // Keep tooltip within container bounds
                if (tooltipX + 300 > rect.width) tooltipX = mouseX - 310;
                if (tooltipY < 0) tooltipY = mouseY + 20;
                
                tooltip.style.left = tooltipX + 'px';
                tooltip.style.top = tooltipY + 'px';
            }}
            
            function hideTooltip() {{
                document.getElementById('tooltip_' + TIMELINE_ID).style.display = 'none';
            }}
            
            function selectRelatedEvents(relationId, linkedToRelationId) {{
                console.log('Timeline event clicked, relation_id:', relationId, 'linked_to_relation_id:', linkedToRelationId);
                
                // Check if this event is part of the currently selected group (for deselection)
                let isPartOfCurrentSelection = false;
                
                if (currentSelectedRelationId !== null) {{
                    // Find all currently connected relation IDs
                    const currentConnectedIds = findAllConnectedRelationIds(currentSelectedRelationId, currentLinkedToRelationId);
                    
                    // Check if the clicked event is part of the current selection
                    isPartOfCurrentSelection = currentConnectedIds.has(relationId);
                }}
                
                if (isPartOfCurrentSelection) {{
                    // Deselect - reset to show all events but maintain zoom
                    console.log('Deselecting connected event group, maintaining zoom level');
                    currentSelectedRelationId = null;
                    currentLinkedToRelationId = null;
                    clearSelections();
                    
                    // Send message to show all events in current time range (maintain zoom)
                    const currentRange = getCurrentVisibleRange();
                    try {{
                        window.parent.postMessage({{
                            type: 'timeline-filter',
                            action: 'filter',
                            startDate: currentRange.start,
                            endDate: currentRange.end,
                            timestamp: Date.now()
                        }}, '*');
                        console.log('Deselection message sent, maintaining zoom level');
                    }} catch (e) {{
                        console.error('Failed to post deselection message to parent:', e);
                    }}
                }} else {{
                    // New selection - show connected events
                    console.log('Selecting new event, showing connected');
                    currentSelectedRelationId = relationId;
                    currentLinkedToRelationId = linkedToRelationId;
                    
                    // Visual highlighting - clear previous selections
                    clearSelections();
                    
                    // Highlight clicked event and connected events
                    highlightConnectedEvents(relationId, linkedToRelationId);
                    
                    // Use postMessage to communicate with parent window
                    try {{
                        window.parent.postMessage({{
                            type: 'timeline-filter',
                            action: 'show-connected',
                            relationId: relationId,
                            linkedToRelationId: linkedToRelationId || '',
                            timestamp: Date.now()
                        }}, '*');
                        console.log('PostMessage sent successfully');
                    }} catch (e) {{
                        console.error('Failed to post message to parent:', e);
                    }}
                }}
            }}
            
            function findAllConnectedRelationIds(selectedRelationId, linkedToRelationId) {{
                // Find all connected relation IDs for the current selection
                const connectedRelationIds = new Set([selectedRelationId]);
                if (linkedToRelationId) {{
                    connectedRelationIds.add(linkedToRelationId);
                }}
                
                // Find events that link to our target relation_id
                events.forEach(event => {{
                    if (event.linked_to_relation_id === selectedRelationId) {{
                        connectedRelationIds.add(event.relation_id);
                    }} else if (event.relation_id === selectedRelationId && event.linked_to_relation_id) {{
                        connectedRelationIds.add(event.linked_to_relation_id);
                    }}
                }});
                
                // Recursively find all connected events (in case of chains)
                let changed = true;
                while (changed) {{
                    changed = false;
                    events.forEach(event => {{
                        const eventRelationId = event.relation_id;
                        const eventLinkedTo = event.linked_to_relation_id;
                        
                        if (connectedRelationIds.has(eventRelationId) && eventLinkedTo && !connectedRelationIds.has(eventLinkedTo)) {{
                            connectedRelationIds.add(eventLinkedTo);
                            changed = true;
                        }} else if (connectedRelationIds.has(eventLinkedTo) && !connectedRelationIds.has(eventRelationId)) {{
                            connectedRelationIds.add(eventRelationId);
                            changed = true;
                        }}
                    }});
                }}
                
                return connectedRelationIds;
            }}
            
            function clearSelections() {{
                console.log('Clearing all timeline selections');
                
                // Remove all selection classes from circles
                const circles = document.querySelectorAll('.event-circle');
                circles.forEach(circle => {{
                    circle.classList.remove('selected', 'connected');
                }});
                
                // Remove all highlighting from connection lines
                const lines = document.querySelectorAll('.connection-line');
                lines.forEach(line => {{
                    line.classList.remove('highlighted');
                }});
                
                console.log('Timeline selections cleared');
            }}
            
            function highlightConnectedEvents(selectedRelationId, linkedToRelationId) {{
                // Use the same logic as selection to find all connected relation IDs
                const connectedRelationIds = findAllConnectedRelationIds(selectedRelationId, linkedToRelationId);
                
                // Highlight circles by finding them via data-relation-id attribute (not by index)
                const circles = document.querySelectorAll('.event-circle');
                circles.forEach(circle => {{
                    const relationId = circle.getAttribute('data-relation-id');
                    if (relationId && connectedRelationIds.has(relationId)) {{
                        if (relationId === selectedRelationId) {{
                            circle.classList.add('selected');
                        }} else {{
                            circle.classList.add('connected');
                        }}
                    }}
                }});
                
                // Highlight connection lines
                const lines = document.querySelectorAll('.connection-line');
                lines.forEach(line => {{
                    const fromRelation = line.getAttribute('data-from-relation');
                    const toRelation = line.getAttribute('data-to-relation');
                    
                    // Highlight line if both endpoints are connected
                    if (fromRelation && toRelation && 
                        connectedRelationIds.has(fromRelation) && 
                        connectedRelationIds.has(toRelation)) {{
                        line.classList.add('highlighted');
                    }}
                }});
            }}
            
            function getCurrentVisibleRange() {{
                // Calculate currently visible time range based on zoom and pan
                const visibleStartX = margin.left - panOffset;
                const visibleEndX = svgWidth - margin.right - panOffset;
                
                const startRatio = Math.max(0, (visibleStartX - margin.left) / (plotWidth * zoomLevel));
                const endRatio = Math.min(1, (visibleEndX - margin.left) / (plotWidth * zoomLevel));
                
                const visibleStartTime = minDate.getTime() + startRatio * totalRange;
                const visibleEndTime = minDate.getTime() + endRatio * totalRange;
                
                return {{
                    start: new Date(visibleStartTime).toISOString(),
                    end: new Date(visibleEndTime).toISOString()
                }};
            }}
            
            function updateVisibleRange() {{
                // Update the main graph based on currently visible timeline range
                const range = getCurrentVisibleRange();
                console.log('Posting visible range update to parent:', range);
                
                try {{
                    window.parent.postMessage({{
                        type: 'timeline-filter',
                        action: 'filter',
                        startDate: range.start,
                        endDate: range.end,
                        timestamp: Date.now()
                    }}, '*');
                }} catch (e) {{
                    console.error('Failed to post message to parent:', e);
                }}
            }}
            
            function resetZoom() {{
                zoomLevel = 1;
                panOffset = 0;
                currentSelectedRelationId = null;
                currentLinkedToRelationId = null;
                clearSelections(); // Clear visual highlighting
                drawTimeline();
                // Reset graph filter to show all events
                console.log('Posting reset message to parent');
                
                try {{
                    window.parent.postMessage({{
                        type: 'timeline-filter',
                        action: 'reset',
                        timestamp: Date.now()
                    }}, '*');
                }} catch (e) {{
                    console.error('Failed to post message to parent:', e);
                }}
            }}
            
            function showAllEvents() {{
                // Reset to show all events in the main graph
                console.log('Posting show all events message to parent');
                currentSelectedRelationId = null;
                currentLinkedToRelationId = null;
                clearSelections(); // Clear visual highlighting
                
                try {{
                    window.parent.postMessage({{
                        type: 'timeline-filter',
                        action: 'reset',
                        timestamp: Date.now()
                    }}, '*');
                }} catch (e) {{
                    console.error('Failed to post message to parent:', e);
                }}
            }}
            
            function zoomIn() {{
                zoomLevel = Math.min(zoomLevel * 1.5, maxZoomLevel);
                drawTimeline();
                // Update graph filter after zoom
                setTimeout(updateVisibleRange, 100);
            }}
            
            function zoomOut() {{
                zoomLevel = Math.max(zoomLevel / 1.5, 0.5);
                // Apply proper panning constraints when zooming out
                const maxPanRight = 0;
                const maxPanLeft = -(plotWidth * (zoomLevel - 1));
                panOffset = Math.max(Math.min(panOffset, maxPanRight), maxPanLeft);
                drawTimeline();
                // Update graph filter after zoom
                setTimeout(updateVisibleRange, 100);
            }}
            
            // Mouse event handlers
            document.getElementById('timelineContainer_' + TIMELINE_ID).addEventListener('mousedown', (e) => {{
                // Don't start dragging if clicking on an event circle
                if (e.target.classList.contains('event-circle')) {{
                    return;
                }}
                
                isDragging = true;
                lastMouseX = e.clientX;
                e.preventDefault(); // Prevent text selection
            }});
            
            // Handle click on empty areas (separate from mousedown to avoid conflicts with dragging)
            document.getElementById('timelineContainer_' + TIMELINE_ID).addEventListener('click', (e) => {{
                // Only handle click if we're not dragging and not clicking on an event circle
                if (!isDragging && !e.target.classList.contains('event-circle') && currentSelectedRelationId !== null) {{
                    // Reset selection and notify parent, but maintain current zoom
                    console.log('Clicking on empty timeline area, resetting selection but maintaining zoom');
                    currentSelectedRelationId = null;
                    currentLinkedToRelationId = null;
                    clearSelections();
                    
                    // Send message to show all events in current time range (maintain zoom)
                    const currentRange = getCurrentVisibleRange();
                    try {{
                        window.parent.postMessage({{
                            type: 'timeline-filter',
                            action: 'filter',
                            startDate: currentRange.start,
                            endDate: currentRange.end,
                            timestamp: Date.now()
                        }}, '*');
                        console.log('Reset message sent from empty area click, maintaining zoom');
                    }} catch (e) {{
                        console.error('Failed to post reset message from empty area click:', e);
                    }}
                }}
            }});
            
            document.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    const deltaX = e.clientX - lastMouseX;
                    panOffset += deltaX;
                    
                    // Apply proper panning constraints
                    // When zoomed in (zoomLevel > 1), we can pan in both directions
                    // panOffset should be positive to pan right, negative to pan left
                    const maxPanRight = 0; // Can't pan beyond the right edge
                    const maxPanLeft = -(plotWidth * (zoomLevel - 1)); // Can pan left when zoomed
                    panOffset = Math.max(Math.min(panOffset, maxPanRight), maxPanLeft);
                    
                    lastMouseX = e.clientX;
                    drawTimeline();
                }}
            }});
            
            document.addEventListener('mouseup', (e) => {{
                if (isDragging) {{
                    isDragging = false;
                    // Update graph filter after panning is complete
                    setTimeout(updateVisibleRange, 100);
                }}
            }});
            
            // Wheel zoom
            document.getElementById('timelineContainer_' + TIMELINE_ID).addEventListener('wheel', (e) => {{
                e.preventDefault();
                const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
                const oldZoomLevel = zoomLevel;
                zoomLevel = Math.max(0.5, Math.min(maxZoomLevel, zoomLevel * zoomFactor));
                
                // Adjust pan offset to zoom towards mouse position
                const rect = document.getElementById('timelineContainer_' + TIMELINE_ID).getBoundingClientRect();
                const mouseX = e.clientX - rect.left - margin.left; // Adjust for margin
                const zoomRatio = zoomLevel / oldZoomLevel;
                
                // Calculate zoom adjustment relative to the plot area
                const zoomAdjustment = mouseX * (zoomRatio - 1);
                panOffset = panOffset * zoomRatio - zoomAdjustment;
                
                // Apply proper panning constraints (allow panning in both directions when zoomed)
                const maxPanRight = 0;
                const maxPanLeft = -(plotWidth * (zoomLevel - 1));
                panOffset = Math.max(Math.min(panOffset, maxPanRight), maxPanLeft);
                
                drawTimeline();
                // Update graph filter after zoom
                setTimeout(updateVisibleRange, 100);
            }});
            
            // Resize handler
            window.addEventListener('resize', () => {{
                setTimeout(drawTimeline, 100);
            }});
            
            // Global function assignments with unique names
            window['resetZoom_' + TIMELINE_ID] = resetZoom;
            window['zoomIn_' + TIMELINE_ID] = zoomIn;
            window['zoomOut_' + TIMELINE_ID] = zoomOut;
            window['showAllEvents_' + TIMELINE_ID] = showAllEvents;
            window['selectRelatedEvents_' + TIMELINE_ID] = selectRelatedEvents;
            window['showTooltip_' + TIMELINE_ID] = showTooltip;
            window['hideTooltip_' + TIMELINE_ID] = hideTooltip;
            
            // Initial draw
            drawTimeline();
            }})(); // End of scoped function
        </script>
    </body>
    </html>
    '''
    
    # Encode as base64 for iframe with proper UTF-8 encoding
    html_content_bytes = html_content.encode('utf-8')
    base64_encoded_html = base64.b64encode(html_content_bytes).decode('ascii')
    iframe_src = f'data:text/html;charset=utf-8;base64,{base64_encoded_html}'
    return f'<iframe src="{iframe_src}" width="100%" height="180px" style="border:none; border-radius: 0px;"></iframe>'


def handle_timeline_filter(start_date, end_date, filter_type, filter_part):
    """Handle timeline filtering updates from the custom timeline component."""
    if start_date and end_date:
        time_range = (start_date, end_date)
        filtered_data = filter_graph_data(graph, filter_type, filter_part, time_range)
        return generate_graph_iframe(filtered_data, localization), time_range
    else:
        # Reset to show all events
        filtered_data = filter_graph_data(graph, filter_type, filter_part, None)
        return generate_graph_iframe(filtered_data, localization), None

def handle_timeline_relayout(relayout_data, filter_type, filter_part):
    """Handle timeline relayout events (zoom/pan) and update graph accordingly."""
    time_range = None
    if relayout_data and isinstance(relayout_data, dict):
        # Check for zoom/pan events that include time range
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            time_range = {
                'xaxis.range[0]': relayout_data['xaxis.range[0]'],
                'xaxis.range[1]': relayout_data['xaxis.range[1]'],
                'xaxis.autorange': False
            }
        elif 'xaxis.autorange' in relayout_data and relayout_data['xaxis.autorange']:
            time_range = {'xaxis.autorange': True}
    
    filtered_data = filter_graph_data(graph, filter_type, filter_part, time_range)
    return generate_graph_iframe(filtered_data, localization), time_range

def filter_connected_events(graph_data, relation_id, linked_to_relation_id=None):
    """Filter graph data to show only events connected to the specified relation_id."""
    if not graph_data or not relation_id:
        return graph_data
    
    connected_events = []
    
    # Find all relation IDs that are connected to the target relation_id
    connected_relation_ids = {relation_id}
    
    # Add the linked_to_relation_id if provided
    if linked_to_relation_id:
        connected_relation_ids.add(linked_to_relation_id)
    
    # Find all events that link to our target relation_id
    for event in graph_data:
        if event.get('linked_to_relation_id') == relation_id:
            connected_relation_ids.add(event.get('relation_id'))
        elif event.get('relation_id') == relation_id and event.get('linked_to_relation_id'):
            connected_relation_ids.add(event.get('linked_to_relation_id'))
    
    # Recursively find all connected events (in case of chains)
    changed = True
    while changed:
        changed = False
        for event in graph_data:
            event_relation_id = event.get('relation_id')
            event_linked_to = event.get('linked_to_relation_id')
            
            if event_relation_id in connected_relation_ids and event_linked_to and event_linked_to not in connected_relation_ids:
                connected_relation_ids.add(event_linked_to)
                changed = True
            elif event_linked_to in connected_relation_ids and event_relation_id not in connected_relation_ids:
                connected_relation_ids.add(event_relation_id)
                changed = True
    
    # Filter events to only include those with connected relation IDs
    for event in graph_data:
        if event.get('relation_id') in connected_relation_ids:
            connected_events.append(event)
    
    return connected_events


# Timeline Javascript Integration

timeline_js = '''
<script>
let timelineUpdateTimeout;
let currentTimelineRange = null;

// Function to find the hidden input more reliably
function findTimelineInput() {
    // Wait a bit for DOM to be ready
    let input = null;
    
    // Try different approaches to find the input
    input = document.querySelector('#timeline-filter-input input') ||
            document.querySelector('#timeline-filter-input textarea') ||
            document.querySelector('[data-testid="textbox"] input') ||
            document.querySelector('[data-testid="textbox"] textarea');
    
    // Try finding by label text
    if (!input) {
        const labels = document.querySelectorAll('label');
        for (let label of labels) {
            if (label.textContent.includes('Timeline Filter')) {
                const container = label.closest('.block');
                if (container) {
                    input = container.querySelector('input') || container.querySelector('textarea');
                    break;
                }
            }
        }
    }
    
    // Last resort: find any visible input/textarea that looks like our filter input
    if (!input) {
        const allInputs = document.querySelectorAll('input, textarea');
        for (let inp of allInputs) {
            // Look for the input with our specific element ID or nearby
            const parent = inp.closest('[id*="timeline-filter"]');
            if (parent) {
                input = inp;
                break;
            }
        }
    }
    
    console.log('Timeline input search result:', input);
    console.log('All available inputs:', document.querySelectorAll('input, textarea'));
    return input;
}

// Listen for postMessage events from timeline iframe
window.addEventListener('message', function(event) {
    console.log('Received postMessage:', event.data);
    
    if (event.data && event.data.type === 'timeline-filter') {
        const hiddenInput = findTimelineInput();
        console.log('Found timeline input for postMessage:', hiddenInput);
        
        if (hiddenInput) {
            let filterData;
            
            if (event.data.action === 'show-connected') {
                // Handle connected events action
                filterData = JSON.stringify({
                    action: 'show-connected',
                    relationId: event.data.relationId,
                    linkedToRelationId: event.data.linkedToRelationId,
                    timestamp: Date.now()
                });
            } else {
                // Handle other actions (filter, reset)
                filterData = JSON.stringify({
                    action: event.data.action,
                    startDate: event.data.startDate,
                    endDate: event.data.endDate,
                    timestamp: Date.now()
                });
            }
            
            console.log('Setting filter data from postMessage:', filterData);
            
            // Set the value and trigger events
            hiddenInput.value = filterData;
            hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
            hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
            hiddenInput.dispatchEvent(new Event('blur', { bubbles: true }));
            
            console.log('PostMessage events dispatched, input value now:', hiddenInput.value);
        } else {
            console.error('Could not find timeline filter input element for postMessage');
        }
    }
}, false);

// Function to update graph based on timeline filter
window.updateTimelineFilter = function(startDate, endDate) {
    console.log('updateTimelineFilter called:', startDate, endDate);
    currentTimelineRange = [startDate, endDate];
    
    // Clear any existing timeout
    clearTimeout(timelineUpdateTimeout);
    
    // Debounce updates to avoid too many calls during panning/zooming
    timelineUpdateTimeout = setTimeout(() => {
        const hiddenInput = findTimelineInput();
        console.log('Found timeline input:', hiddenInput);
        
        if (hiddenInput) {
            const filterData = JSON.stringify({
                action: 'filter',
                startDate: startDate,
                endDate: endDate,
                timestamp: Date.now()
            });
            console.log('Setting filter data:', filterData);
            
            // Set the value and trigger events
            hiddenInput.value = filterData;
            
            // Try multiple event types to trigger Gradio
            hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
            hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
            hiddenInput.dispatchEvent(new Event('blur', { bubbles: true }));
            
            // Also try triggering on the container
            const container = hiddenInput.closest('.block');
            if (container) {
                container.dispatchEvent(new Event('change', { bubbles: true }));
            }
            
            console.log('Events dispatched, input value now:', hiddenInput.value);
        } else {
            console.error('Could not find timeline filter input element');
            // Debug: show all available inputs
            console.log('Available inputs:', document.querySelectorAll('input, textarea'));
        }
    }, 200);
};

// Function to reset timeline filter
window.resetTimelineFilter = function() {
    console.log('resetTimelineFilter called');
    currentTimelineRange = null;
    
    const hiddenInput = findTimelineInput();
    if (hiddenInput) {
        const filterData = JSON.stringify({
            action: 'reset',
            timestamp: Date.now()
        });
        console.log('Setting reset data:', filterData);
        
        hiddenInput.value = filterData;
        hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
        hiddenInput.dispatchEvent(new Event('change', { bubbles: true }));
        hiddenInput.dispatchEvent(new Event('blur', { bubbles: true }));
        
        console.log('Reset events dispatched');
    } else {
        console.error('Could not find timeline filter input element for reset');
    }
};

</script>
'''


# Gradio Management
import gradio as gr

app_css = '''
.graph-container {
    padding: 0 !important;
    background: none !important;
    border: none !important;
    border-radius: 0 !important;
}
#input-row { align-items: stretch; }
.left-col { display: flex; flex-direction: column; }
#submit-button { margin-top: auto; }
footer { display: none !important; }
.loading-dots {
    display: inline-block;
    position: relative;
    width: 40px;
    height: 10px;
    margin-left: 8px;
}
.loading-dots div {
    position: absolute;
    top: 0;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #999;
    animation: loading-dots 1.2s linear infinite;
}
.loading-dots div:nth-child(1) { left: 4px; animation-delay: 0s; }
.loading-dots div:nth-child(2) { left: 14px; animation-delay: 0.4s; }
.loading-dots div:nth-child(3) { left: 24px; animation-delay: 0.8s; }
@keyframes loading-dots {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}
'''


def handle_timeline_filter_input(filter_input, filter_type, filter_part):
    """Handle timeline filter updates from the JavaScript bridge."""
    
    if not filter_input:
        return gr.update(), None
    
    try:
        filter_data = json.loads(filter_input)
        action = filter_data.get('action')
        
        if action == 'filter':
            start_date = filter_data.get('startDate')
            end_date = filter_data.get('endDate')
            if start_date and end_date:
                time_range = (start_date, end_date)
                filtered_data = filter_graph_data(graph, filter_type, filter_part, time_range)
                return generate_graph_iframe(filtered_data, localization), time_range
        elif action == 'show-connected':
            # Show only connected events based on relation_id, but still respect dropdown filters
            relation_id = filter_data.get('relationId')
            linked_to_relation_id = filter_data.get('linkedToRelationId')
            if relation_id:
                # Filter to show only connected events
                connected_events = filter_connected_events(graph, relation_id, linked_to_relation_id)
                # Apply dropdown filters to the connected events
                filtered_connected_events = filter_graph_data(connected_events, filter_type, filter_part, None)
                return generate_graph_iframe(filtered_connected_events, localization), None
        elif action == 'reset':
            # Reset to show all events
            filtered_data = filter_graph_data(graph, filter_type, filter_part, None)
            return generate_graph_iframe(filtered_data, localization), None
        elif action == 'test':
            # Return a test response to verify the connection works
            return gr.update(), None
            
    except (json.JSONDecodeError, KeyError) as e:
        print(f"â�Œ Error parsing filter input: {e}")  # Debug
        print(f"   Raw input was: {filter_input}")  # Debug
    
    # Return current state if parsing fails
    return gr.update(), None

def update_graph_on_filter(event_type, anatomical_part, time_range):
    """Called when a filter dropdown changes."""
    all_text = localization.get_text('all_filter')
    part_choices = [all_text] + sorted(list(set(e.get('anatomical_part') for e in graph if e.get('anatomical_part'))))
    
    # Apply filters to get the data
    filtered_data = filter_graph_data(graph, event_type, anatomical_part, time_range)
    
    # Check if we have any events after filtering
    if not filtered_data and time_range:
        # If no events match the current filters with time range, 
        # try without time range to avoid empty results
        filtered_data = filter_graph_data(graph, event_type, anatomical_part, None)
        # If we still have no events, then the anatomical part filter is the issue
        if not filtered_data:
            # Reset anatomical part to 'All' to prevent empty results
            anatomical_part = all_text
            filtered_data = filter_graph_data(graph, event_type, all_text, time_range)
    
    return generate_graph_iframe(filtered_data, localization), gr.update(choices=part_choices), generate_timeline_iframe(filtered_data, localization), time_range


def handle_filter_change_with_timeline_reset(event_type, anatomical_part, current_timeline_range):
    """Handle filter changes and reset timeline range to avoid filter conflicts."""
    all_text = localization.get_text('all_filter')
    
    # Reset timeline range to avoid conflicts between timeline filtering and dropdown filtering
    # This ensures that when users change dropdown filters, they see all relevant events
    # without being constrained by a previous timeline zoom/selection
    timeline_range = None
    
    return update_graph_on_filter(event_type, anatomical_part, timeline_range)

def process_new_submission(user_input_text, chat_history, filter_type, filter_part, timeline_range):
    """Handle the user's initial text submission."""
    # Reset timeline range immediately upon submission (before any processing)
    timeline_range = None
    
    # Check if input is empty or just whitespace
    if not user_input_text or not user_input_text.strip():
        # Don't process empty input, just return current state without changes
        filtered_data = filter_graph_data(graph, filter_type, filter_part, timeline_range)
        yield chat_history, gr.update(interactive=True), gr.update(visible=False), None, None, None, generate_graph_iframe(filtered_data, localization), gr.update(), gr.update(), generate_timeline_iframe(graph, localization), gr.update(value=timeline_range)
        return
    
    chat_history.append({'role': 'user', 'content': user_input_text})
    chat_history.append({'role': 'assistant', 'content': localization.get_text('processing') + '<div class="loading-dots"><div></div><div></div><div></div></div>'})
    # During loading, show the current graph state (don't apply timeline filtering which might hide events)
    current_filtered_data = filter_graph_data(graph, filter_type, filter_part, None)
    # Reset timeline by regenerating it completely with a new timestamp (this will reset zoom state)
    import time
    timeline_reset_key = int(time.time() * 1000)  # Unique timestamp for fresh timeline
    yield chat_history, gr.update(interactive=False), gr.update(visible=False), None, None, None, generate_graph_iframe(current_filtered_data, localization), gr.update(), gr.update(), generate_timeline_iframe(graph, localization, timeline_reset_key), gr.update(value=timeline_range)
    
    new_events = generate_event(user_input_text)
    if not new_events:
        chat_history[-1]['content'] = localization.get_text('error_message')
        final_filtered_graph = filter_graph_data(graph, filter_type, filter_part, timeline_range)
        yield chat_history, gr.update(interactive=True), gr.update(visible=False), None, None, None, generate_graph_iframe(final_filtered_graph, localization), gr.update(), gr.update(), generate_timeline_iframe(graph, localization), gr.update(value=timeline_range)
        return
    
    proposed_link = find_proposed_link(new_events, graph)
    if proposed_link:
        past_term = proposed_link['past_event_term']
        past_input = proposed_link['past_event_user_input']
        chat_history[-1]['content'] = f"{localization.get_text('link_found')} **'{past_term}'**."
        chat_history.append({'role': 'assistant', 'content': f'> \\"{past_input}\\"'})
        confirmation_question = localization.get_text('link_question')
        # Use filtered data for consistency, even during confirmation
        final_filtered_graph = filter_graph_data(graph, filter_type, filter_part, timeline_range)
        yield (chat_history, gr.update(interactive=False), gr.update(visible=True), gr.update(value=confirmation_question), new_events, proposed_link, generate_graph_iframe(final_filtered_graph, localization), create_event_checklist_update(graph), gr.update(), generate_timeline_iframe(graph, localization), gr.update(value=timeline_range))
    else:
        graph.extend(new_events)
        graph.sort(key=lambda x: x['timestamp'])
        chat_history[-1]['content'] = localization.get_text('done_message')
        final_filtered_graph = filter_graph_data(graph, filter_type, filter_part, None)
        all_text = localization.get_text('all_filter')
        part_choices = [all_text] + sorted(list(set(e.get('anatomical_part') for e in graph if e.get('anatomical_part'))))
        yield (chat_history, gr.update(interactive=True), gr.update(visible=False), gr.update(), [], None, generate_graph_iframe(final_filtered_graph, localization), create_event_checklist_update(graph), gr.update(choices=part_choices), generate_timeline_iframe(graph, localization), gr.update(value=timeline_range))

def handle_confirmation(confirmation_choice, current_staged_events, link_proposal, chat_history, filter_type, filter_part):
    """Handle the user's choice on a proposed link."""
    if confirmation_choice == 'Yes':
        for event in current_staged_events:
            if event['relation_id'] == link_proposal['new_event_relation_id']:
                event['linked_to_relation_id'] = link_proposal['past_event_relation_id']
                break
        chat_history.append({'role': 'assistant', 'content': localization.get_text('linked_events')})
    else:
        chat_history.append({'role': 'assistant', 'content': localization.get_text('separate_event')})
    
    graph.extend(current_staged_events)
    graph.sort(key=lambda x: x['timestamp'])
    
    # Apply current filters to ensure consistency between graph and timeline
    filtered_data = filter_graph_data(graph, filter_type, filter_part, None)
    all_text = localization.get_text('all_filter')
    part_choices = [all_text] + sorted(list(set(e.get('anatomical_part') for e in graph if e.get('anatomical_part'))))
    return (chat_history, gr.update(interactive=True), gr.update(visible=False), [], None, generate_graph_iframe(filtered_data, localization), create_event_checklist_update(graph), gr.update(choices=part_choices), generate_timeline_iframe(graph, localization), gr.update(value=None))

def create_event_checklist_update(graph_data):
    """Create updated choices for the event checklist."""
    choices = []
    indexed_graph = list(enumerate(graph_data))
    sorted_indexed_graph = sorted(indexed_graph, key=lambda x: x[1]['timestamp'], reverse=True)
    
    for original_index, event in sorted_indexed_graph:
        descriptors = event.get('descriptors', {})
        event_type_translated = localization.get_event_type_translation(event.get('event_type', 'unknown'))
        label_map = {'symptom': descriptors.get('symptom_term', 'N/A'), 'action_taken': descriptors.get('action_name', 'N/A'), 'visit': descriptors.get('reason', 'N/A')}
        label = label_map.get(event.get('event_type'), 'N/A')
        event_time = datetime.fromtimestamp(event['timestamp']).strftime('%Y-%m-%d')
        display_text = f'[{event_time}] {event_type_translated.capitalize()}: {label} ({event.get("anatomical_part", "N/A")})'
        choices.append((display_text, original_index))
        
    return gr.update(choices=choices, value=[])

def delete_selected_events_by_index(selected_indices, chat_history, filter_type, filter_part):
    """Delete selected events by their indices."""
    global graph
    if not selected_indices:
        chat_history.append({'role': 'assistant', 'content': localization.get_text('no_events_selected')})
        all_text = localization.get_text('all_filter')
        part_choices = [all_text] + sorted(list(set(e.get('anatomical_part') for e in graph if e.get('anatomical_part'))))
        filtered_data = filter_graph_data(graph, filter_type, filter_part, None)
        return chat_history, generate_graph_iframe(filtered_data, localization), gr.update(), gr.update(choices=part_choices), generate_timeline_iframe(graph, localization), gr.update(value=None)
    
    indices_to_delete = set(selected_indices)
    graph = [event for i, event in enumerate(graph) if i not in indices_to_delete]

    chat_history.append({'role': 'assistant', 'content': localization.get_text('events_deleted', count=len(selected_indices))})
    all_text = localization.get_text('all_filter')
    part_choices = [all_text] + sorted(list(set(e.get('anatomical_part') for e in graph if e.get('anatomical_part'))))
    filtered_data = filter_graph_data(graph, filter_type, filter_part, None)
    return (chat_history, generate_graph_iframe(filtered_data, localization), create_event_checklist_update(graph), gr.update(choices=part_choices), generate_timeline_iframe(graph, localization), gr.update(value=None))

def handle_language_change(language):
    """Handle language change and update UI components."""
    global graph
    
    current_language = localization.current_language
    localization.set_language(language)
    
    if language != current_language and graph:
        try:
            graph = translate_graph_entries(graph, language, current_language)
        except Exception:
            pass
    
    all_text = localization.get_text('all_filter')
    event_type_choices = [all_text, 'symptom', 'action_taken', 'visit']
    part_choices = [all_text] + sorted(list(set(e.get('anatomical_part') for e in graph if e.get('anatomical_part'))))
    new_chatbot = [{'role': 'assistant', 'content': localization.get_text('welcome_message')}]
    updated_checklist = create_event_checklist_update(graph)
    updated_examples_html = create_examples_html(localization.get_example_prompts(), localization.get_text('try_example'))
    
    return (
        new_chatbot,
        generate_graph_iframe(graph, localization),
        generate_timeline_iframe(graph, localization),
        gr.update(choices=updated_checklist['choices'], label=localization.get_text('select_events_delete')),
        gr.update(label=localization.get_text('filter_event_type'), choices=event_type_choices, value=all_text),
        gr.update(label=localization.get_text('filter_anatomical_part'), choices=part_choices, value=all_text),
        gr.update(value=localization.get_text('app_title')),
        gr.update(value=localization.get_text('graph_legend')),
        gr.update(label=localization.get_text('enter_event'), placeholder=localization.get_text('placeholder_text')),
        gr.update(value=localization.get_text('add_event_button')),
        gr.update(value=localization.get_text('no_separate')),
        gr.update(value=localization.get_text('yes_link')),
        gr.update(),  # Remove label update for Column component
        gr.update(value=localization.get_text('delete_selected')),
        gr.update(value=localization.get_text('link_question')),
        gr.update(label=localization.get_text('chat_history')),
        gr.update(value=updated_examples_html),
        gr.update(value=None) # Reset timeline range state
    )

def clear_textbox():
    """Clear the text input box."""
    return gr.update(value='')


# Examples html
def create_examples_html(examples, label):
    """Create HTML for clickable examples."""
    html = f'<div style="margin-bottom: 10px;"><strong>{label}</strong></div>'
    html += '<div style="display: flex; flex-direction: column; gap: 8px;">'
    
    for example in examples:
        # Escape quotes and backticks for JavaScript
        escaped_example = example.replace('`', '\\`').replace("'", "\\'").replace('"', '\\"')
        html += f'''
        <div style="border: 1px solid #ddd; border-radius: 6px; padding: 12px; cursor: pointer; 
                    background: #f9f9f9; transition: background-color 0.2s;"
             onclick="document.querySelector('#main-textbox textarea').value = `{escaped_example}`;
                     document.querySelector('#main-textbox textarea').dispatchEvent(new Event('input', {{bubbles: true}}));"
             onmouseover="this.style.backgroundColor='#f0f0f0'"
             onmouseout="this.style.backgroundColor='#f9f9f9'">
            <div style="font-size: 14px; color: #555; line-height: 1.4;">{example}</div>
        </div>
        '''
    html += '</div>'
    return html


# Session-persistent variables
app = None

# DISCLAIMER: These events were generated with a slightly different extraction prompt and without time extraction at that time.
graph = [
  {
    "event_type": "symptom",
    "medical_speciality": "dentistry",
    "anatomical_part": "gums",
    "descriptors": {
      "symptom_term": "sensitive gums",
      "symptom_quality": "puffy",
      "perceived_trigger": "stress from work",
      "seriousness_score": 0.5
    },
    "timestamp": 1749177600.0,
    "user_input": "My gums have been really sensitive and puffy for a while now, especially when I brush. I'm blaming it on stress from work.",
    "relation_id": "233e7f1e-d10a-4081-ab91-ddf7919ec592"
  },
  {
    "event_type": "symptom",
    "medical_speciality": "dermatology",
    "anatomical_part": "shin",
    "descriptors": {
      "symptom_term": "scratch",
      "symptom_quality": "deep",
      "perceived_trigger": "thorny branch",
      "seriousness_score": 0.6
    },
    "timestamp": 1752885600.0,
    "user_input": "Great weekend hike today. Got a pretty deep scratch from a thorny branch on my shin though. Cleaned it up.",
    "relation_id": "0a648103-0eae-48fe-b2df-c4ab38993646"
  },
  {
    "event_type": "action_taken",
    "medical_speciality": "primary care",
    "anatomical_part": "shin",
    "descriptors": {
      "action_name": "cleaning",
      "description": "cleaned the scratch",
      "outcome": "clean"
    },
    "timestamp": 1752885600.0,
    "user_input": "Great weekend hike today. Got a pretty deep scratch from a thorny branch on my shin though. Cleaned it up.",
    "relation_id": "0a648103-0eae-48fe-b2df-c4ab38993646"
  },
  {
    "event_type": "symptom",
    "medical_speciality": "dermatology",
    "anatomical_part": "shin",
    "descriptors": {
      "symptom_term": "scratch",
      "symptom_quality": "red",
      "perceived_trigger": "scratch",
      "seriousness_score": 0.3
    },
    "timestamp": 1753491600.0,
    "user_input": "That scratch on my shin from last week is taking forever to heal. It still looks red and isn't fully closed.",
    "relation_id": "e74ac727-532f-4987-9b4b-cbbac23a1913",
    "linked_to_relation_id": "0a648103-0eae-48fe-b2df-c4ab38993646"
  },
  {
    "event_type": "visit",
    "medical_speciality": "dentistry",
    "anatomical_part": "gums",
    "descriptors": {
      "reason": "gums",
      "diagnosis": None,
      "recommendation": "be more disciplined with flossing"
    },
    "timestamp": 1751843400.0,
    "user_input": "Was at the dentist again for my stupid gums. He just tells me to be more disciplined with flossing.",
    "relation_id": "a1130635-2aca-41f9-a615-c30b2821acba",
    "linked_to_relation_id": "233e7f1e-d10a-4081-ab91-ddf7919ec592"
  },
  {
    "event_type": "action_taken",
    "medical_speciality": "dentistry",
    "anatomical_part": "gums",
    "descriptors": {
      "action_name": "flossing",
      "description": "be more disciplined with flossing",
      "outcome": None
    },
    "timestamp": 1751843400.0,
    "user_input": "Was at the dentist again for my stupid gums. He just tells me to be more disciplined with flossing.",
    "relation_id": "a1130635-2aca-41f9-a615-c30b2821acba"
  },
  {
    "event_type": "symptom",
    "medical_speciality": "dentistry",
    "anatomical_part": "gums",
    "descriptors": {
      "symptom_term": "bleeding gums",
      "symptom_quality": "inflamed",
      "perceived_trigger": "flossing",
      "seriousness_score": 0.6
    },
    "timestamp": 1753764000.0,
    "user_input": "Knew it. I've been flossing like crazy for weeks and my gums are still bleeding and inflamed. This is so frustrating.",
    "relation_id": "34ee49be-923b-4be0-b7ab-ffe80797b73e",
    "linked_to_relation_id": "a1130635-2aca-41f9-a615-c30b2821acba"
  },
  {
    "event_type": "symptom",
    "medical_speciality": "dentistry",
    "anatomical_part": "gums",
    "descriptors": {
      "symptom_term": "inflammation",
      "symptom_quality": "inflamed",
      "perceived_trigger": "flossing",
      "seriousness_score": 0.6
    },
    "timestamp": 1753764000.0,
    "user_input": "Knew it. I've been flossing like crazy for weeks and my gums are still bleeding and inflamed. This is so frustrating.",
    "relation_id": "34ee49be-923b-4be0-b7ab-ffe80797b73e"
  },
  {
    "event_type": "symptom",
    "medical_speciality": "primary care",
    "anatomical_part": None,
    "descriptors": {
      "symptom_term": "fatigue",
      "symptom_quality": "massive wall",
      "perceived_trigger": "canteen food",
      "seriousness_score": 0.6
    },
    "timestamp": 1753932600.0,
    "user_input": "Hit a massive wall of fatigue after lunch today. The canteen food always makes me so sleepy and groggy.",
    "relation_id": "e2436270-2919-4c73-83e6-b87317e843da"
  },
  {
    "event_type": "symptom",
    "medical_speciality": "primary care",
    "anatomical_part": None,
    "descriptors": {
      "symptom_term": "sleepiness",
      "symptom_quality": "groggy",
      "perceived_trigger": "canteen food",
      "seriousness_score": 0.4
    },
    "timestamp": 1753932600.0,
    "user_input": "Hit a massive wall of fatigue after lunch today. The canteen food always makes me so sleepy and groggy.",
    "relation_id": "e2436270-2919-4c73-83e6-b87317e843da"
  },
  {
    "event_type": "symptom",
    "medical_speciality": "endocrinology",
    "anatomical_part": "body",
    "descriptors": {
      "symptom_term": "constant thirst",
      "symptom_quality": "annoying",
      "perceived_trigger": "unknown",
      "seriousness_score": 0.6
    },
    "timestamp": 1753856400.0,
    "user_input": "The constant thirst is getting really annoying. I feel like I'm drinking water all day long but am never satisfied.",
    "relation_id": "d92167a1-2426-4551-86a5-934fbd53c93a"
  },
  {
    "event_type": "symptom",
    "medical_speciality": "gastroenterology",
    "anatomical_part": "bladder",
    "descriptors": {
      "symptom_term": "frequent urination",
      "symptom_quality": "disruptive",
      "perceived_trigger": "unknown",
      "seriousness_score": 0.6
    },
    "timestamp": 1754008800.0,
    "user_input": "Woke up twice last night to use the bathroom. This is becoming a regular thing now, it's really messing with my sleep.",
    "relation_id": "66ed8863-257f-47a6-bd62-8dcf4e562cc9"
  },
  {
    "event_type": "symptom",
    "medical_speciality": "primary care",
    "anatomical_part": "sleep",
    "descriptors": {
      "symptom_term": "sleep disruption",
      "symptom_quality": "interrupted",
      "perceived_trigger": "frequent urination",
      "seriousness_score": 0.4
    },
    "timestamp": 1754008800.0,
    "user_input": "Woke up twice last night to use the bathroom. This is becoming a regular thing now, it's really messing with my sleep.",
    "relation_id": "66ed8863-257f-47a6-bd62-8dcf4e562cc9"
  }
]


# Gradio Interface

# Sort initial graph data by timestamp
if graph:
    graph.sort(key=lambda x: x['timestamp'])

initial_checklist_update = create_event_checklist_update(graph)

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=app_css, head=timeline_js) as app_instance:
    # State Management
    staged_events = gr.State([])
    potential_link = gr.State(None)
    timeline_range = gr.State(None)

    # Main Layout
    with gr.Column(elem_classes=['graph-container']):
        html_output = gr.HTML(value=generate_graph_iframe(graph, localization))
    
    with gr.Column(elem_classes=['timeline-container']):
        timeline_viewer = gr.HTML(value=generate_timeline_iframe(graph, localization))
        
    # Hidden input for timeline communication
    timeline_filter_input = gr.Textbox(
        elem_id="timeline-filter-input",
        visible=False,  # Make it visible for debugging
        value="",
        label="Timeline Filter (Debug)",
        container=False
    )

    # Filter Controls
    with gr.Row():
        filter_type = gr.Dropdown(
            label=localization.get_text('filter_event_type'),
            choices=[localization.get_text('all_filter'), 'symptom', 'action_taken', 'visit'],
            value=localization.get_text('all_filter')
        )
        filter_part = gr.Dropdown(
            label=localization.get_text('filter_anatomical_part'),
            choices=[localization.get_text('all_filter')] + sorted(list(set(e.get('anatomical_part') for e in graph if e.get('anatomical_part')))),
            value=localization.get_text('all_filter')
        )
        language_selection = gr.Dropdown(
            label=localization.get_text('language_label'),
            choices=['English', 'German'],
            value='English'
        )
            
    # Input and Chat Section
    with gr.Row(elem_id='input-row'):
        # Left Column (Input Controls)
        with gr.Column(scale=2, elem_classes=['left-col']):
            with gr.Row():
                title_markdown = gr.Markdown(localization.get_text('app_title'))
                
            legend_markdown = gr.Markdown(value=localization.get_text('graph_legend'))
            text_input = gr.Textbox(
                label=localization.get_text('enter_event'),
                placeholder=localization.get_text('placeholder_text'),
                lines=7,
                elem_id='main-textbox'
            )
            submit_btn = gr.Button(localization.get_text('add_event_button'), variant='primary', elem_id='submit-button')
            
            examples_html = gr.HTML(
                value=create_examples_html(localization.get_example_prompts(), localization.get_text('try_example'))
            )

        # Right Column (Chat & Event List)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                [{'role': 'assistant', 'content': localization.get_text('welcome_message')}],
                label=localization.get_text('chat_history'),
                type='messages',
                height=450
            )

            # Confirmation Row
            with gr.Row(visible=False) as confirmation_row:
                confirmation_markdown = gr.Markdown(localization.get_text('link_question'))
                no_btn = gr.Button(localization.get_text('no_separate'))
                yes_btn = gr.Button(localization.get_text('yes_link'), variant='primary')
    
            #with gr.Accordion(localization.get_text('event_history_deletion'), open=True) as event_accordion:
            with gr.Column() as event_accordion:
                history_checklist = gr.CheckboxGroup(
                    label=localization.get_text('select_events_delete'),
                    choices=initial_checklist_update['choices']
                )
                delete_btn = gr.Button(localization.get_text('delete_selected'), variant='stop')

    # Define Component Lists for Handlers
    outputs_submission = [chatbot, text_input, confirmation_row, confirmation_markdown, staged_events, potential_link, html_output, history_checklist, filter_part, timeline_viewer, timeline_range]
    outputs_confirmation = [chatbot, text_input, confirmation_row, staged_events, potential_link, html_output, history_checklist, filter_part, timeline_viewer, timeline_range]
    outputs_deletion = [chatbot, html_output, history_checklist, filter_part, timeline_viewer, timeline_range]
    outputs_lang_change = [
        chatbot, html_output, timeline_viewer, history_checklist, filter_type, filter_part, 
        title_markdown, legend_markdown, text_input, submit_btn, no_btn, yes_btn, 
        event_accordion, delete_btn, confirmation_markdown, chatbot, examples_html, timeline_range
    ]

    # Event Handlers
    language_selection.change(
        fn=handle_language_change,
        inputs=[language_selection],
        outputs=outputs_lang_change
    )

    submit_btn.click(
        fn=process_new_submission,
        inputs=[text_input, chatbot, filter_type, filter_part, timeline_range],
        outputs=outputs_submission
    ).then(fn=clear_textbox, inputs=None, outputs=text_input)

    text_input.submit(
        fn=process_new_submission,
        inputs=[text_input, chatbot, filter_type, filter_part, timeline_range],
        outputs=outputs_submission
    ).then(fn=clear_textbox, inputs=None, outputs=text_input)
        
    yes_btn.click(
        fn=handle_confirmation,
        inputs=[gr.State('Yes'), staged_events, potential_link, chatbot, filter_type, filter_part],
        outputs=outputs_confirmation
    )
    
    no_btn.click(
        fn=handle_confirmation,
        inputs=[gr.State('No'), staged_events, potential_link, chatbot, filter_type, filter_part],
        outputs=outputs_confirmation
    )
    
    delete_btn.click(
        fn=delete_selected_events_by_index,
        inputs=[history_checklist, chatbot, filter_type, filter_part],
        outputs=outputs_deletion
    )

    filter_type.change(
        fn=handle_filter_change_with_timeline_reset,
        inputs=[filter_type, filter_part, timeline_range],
        outputs=[html_output, filter_part, timeline_viewer, timeline_range]
    )
    
    filter_part.change(
        fn=handle_filter_change_with_timeline_reset,
        inputs=[filter_type, filter_part, timeline_range],
        outputs=[html_output, filter_part, timeline_viewer, timeline_range]
    )
    
    # Timeline filter handler
    timeline_filter_input.change(
        fn=handle_timeline_filter_input,
        inputs=[timeline_filter_input, filter_type, filter_part],
        outputs=[html_output, timeline_range]
    )

# Launch the app
app = app_instance
app.launch(share=True)


# Errors will be displayed in this cell
# In case the Gradio Interface cell runs endlessly, the most likely reason is that gradio's share link server is currently down: https://status.gradio.app

