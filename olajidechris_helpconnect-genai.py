import os

try: 
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
except:
    from kaggle_secrets import UserSecretsClient
    GEMINI_API_KEY =  UserSecretsClient().get_secret("GEMINI_API_KEY")

if GEMINI_API_KEY: print('GEMINI_API_KEY loaded successfully.')


class DatabaseTools:
    def __init__(self):
        self.resources = {}
        self.needs = {}
        self.next_resource_id = 1
        self.next_need_id = 1

    def _get_db(self, category):
        if category == 'resources':
            return self.resources
        elif category == 'needs':
            return self.needs
        else:
            raise ValueError("Invalid category. Must be 'resources' or 'needs'.")

    def _get_next_id(self, category):
        if category == 'resources':
            current_id = self.next_resource_id
            self.next_resource_id += 1
            return current_id
        elif category == 'needs':
            current_id = self.next_need_id
            self.next_need_id += 1
            return current_id

    def add_entry(self, category, entry_data):
        db = self._get_db(category)
        entry_id = self._get_next_id(category)
        db[entry_id] = entry_data
        print(f"Added {category} entry with ID: {entry_id}")
        return entry_id

    def get_entry(self, category, entry_id):
        db = self._get_db(category)
        return db.get(entry_id)

    def update_entry(self, category, entry_id, new_data):
        db = self._get_db(category)
        if entry_id in db:
            db[entry_id].update(new_data)
            print(f"Updated {category} entry with ID: {entry_id}")
            return True
        print(f"Error: {category} entry with ID {entry_id} not found.")
        return False

    def list_entries(self, category):
        db = self._get_db(category)
        return db.items()

    def delete_entry(self, category, entry_id):
        db = self._get_db(category)
        if entry_id in db:
            del db[entry_id]
            print(f"Deleted {category} entry with ID: {entry_id}")
            return True
        print(f"Error: {category} entry with ID {entry_id} not found.")
        return False

# Initialize the database tools
db_tools = DatabaseTools()

print("DatabaseTools class and instance 'db_tools' created successfully.")


import requests
import json
import os

# Placeholder for Google Search API - In a real scenario, this would use a proper library or direct API calls
# For simplicity, we'll simulate a search result or use a basic web search if possible.
class GoogleSearchTool:
    def __init__(self, api_key=None):
        self.api_key = api_key
        # In a real scenario, you'd configure a search client here (e.g., Google Custom Search API, SerpAPI)
        print("Google Search Tool initialized. (Note: Actual search functionality may require a dedicated API setup).")

    def search(self, query, num_results=3):
        # This is a placeholder. In a real application, you would use a library like google-api-python-client
        # or integrate with a service like SerpAPI or directly use Google Custom Search API.
        # For demonstration purposes, we'll return a mock result or a basic web search simulation.
        if self.api_key: # Assuming GEMINI_API_KEY could be used for some Google services, but not directly for 'Google Search' as a general web search.
            print(f"Performing a simulated search for: {query}")
            # Example mock response structure
            mock_results = [
                {"title": f"Result 1 for {query}", "link": f"https://example.com/result1_{query.replace(' ', '-')}", "snippet": "This is a snippet for the first search result."},
                {"title": f"Result 2 for {query}", "link": f"https://example.com/result2_{query.replace(' ', '-')}", "snippet": "Another snippet with more details."}
            ]
            return mock_results
        else:
            print("No API key provided for Google Search. Returning generic mock results.")
            mock_results = [
                {"title": f"Generic Result 1 for {query}", "link": f"https://generic.com/result1", "snippet": "This is a generic snippet for the first search result."}
            ]
            return mock_results


class CodeExecutionTool:
    def execute_code(self, code_string, globals_dict=None, locals_dict=None):
        try:
            # Define a dictionary to capture stdout/stderr
            output_capture = io.StringIO()
            sys.stdout = output_capture
            sys.stderr = output_capture

            # Create a safe environment for execution
            if globals_dict is None:
                globals_dict = {}
            if locals_dict is None:
                locals_dict = {}

            # Prevent access to built-in functions like open, import, etc.
            safe_globals = {"__builtins__": {
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'range': range, 'type': type, 'list': list, 'dict': dict,
                'set': set, 'tuple': tuple, 'sum': sum, 'min': min, 'max': max,
                'abs': abs, 'round': round, 'dir': dir, 'getattr': getattr,
                'hasattr': hasattr, 'setattr': setattr, 'delattr': delattr,
                'isinstance': isinstance, 'issubclass': issubclass, 'callable': callable
            }}
            # Merge user-provided globals/locals while ensuring safety
            safe_globals.update(globals_dict)

            # Execute the code
            exec(code_string, safe_globals, locals_dict)
            return {"status": "success", "output": output_capture.getvalue()}
        except Exception as e:
            return {"status": "error", "output": output_capture.getvalue() + f"Error: {e}"}
        finally:
            # Restore stdout/stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

# Initialize built-in tools
import io
import sys
google_search_tool = GoogleSearchTool(api_key=GEMINI_API_KEY) # Using GEMINI_API_KEY as a generic indicator of API availability
code_execution_tool = CodeExecutionTool()

print("GoogleSearchTool and CodeExecutionTool classes instantiated successfully.")



class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, tool_instance):
        self.tools[name] = tool_instance
        print(f"Tool '{name}' registered.")

    def get_tool(self, name):
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found.")
        return self.tools[name]

# Initialize the ToolManager
tool_manager = ToolManager()

# Register the custom and built-in tools
tool_manager.register_tool('database_tools', db_tools)
tool_manager.register_tool('google_search', google_search_tool)
tool_manager.register_tool('code_execution', code_execution_tool)

print("ToolManager initialized and tools registered successfully.")


class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, session_id, initial_data=None):
        if initial_data is None:
            initial_data = {}
        if session_id not in self.sessions:
            self.sessions[session_id] = initial_data
            print(f"Session '{session_id}' created successfully.")
            return True
        print(f"Session '{session_id}' already exists.")
        return False

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def update_session(self, session_id, new_data):
        if session_id in self.sessions:
            self.sessions[session_id].update(new_data)
            print(f"Session '{session_id}' updated successfully.")
            return True
        print(f"Session '{session_id}' not found. Cannot update.")
        return False

print("SessionManager class defined.")


class MemoryBank:
    def __init__(self):
        self.memory = {}

    def add_to_memory(self, identifier, entry_data):
        if identifier not in self.memory:
            self.memory[identifier] = []
        self.memory[identifier].append(entry_data)
        print(f"Entry added to memory for '{identifier}'.")

    def retrieve_memory(self, identifier):
        return self.memory.get(identifier, [])

print("MemoryBank class defined.")


session_manager = SessionManager()
memory_bank = MemoryBank()

print("SessionManager and MemoryBank instances created successfully.")


import google.generativeai as genai

print("GenerativeAI library imported.")


class ResourceAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')
        self.google_search = self.tool_manager.get_tool('google_search')

        # Define the system instruction for the LLM
        self.system_instruction = (
            "You are the Resource Agent, an AI designed for crisis response. Your primary responsibilities include "
            "identifying, vetting, and managing available resources (e.g., supplies, volunteers, equipment). "
            "You have access to a `database_tools` for managing resources and `google_search` for verification. "
            "When identifying new resources, use the `database_tools.add_entry` function. "
            "When vetting resources, use `google_search` to find external information and `database_tools.update_entry` to update their status. "
            "When managing availability, use `database_tools.update_entry`. "
            "Always record significant actions and outcomes in the `memory_bank` associated with the current session identifier. "
            "Provide clear, concise responses and justifications for your actions."
        )

        print("ResourceAgent class initialized with LLM, tool_manager, session_manager, and memory_bank.")


class ResourceAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')
        self.google_search = self.tool_manager.get_tool('google_search')

        # Define the system instruction for the LLM
        self.system_instruction = (
            "You are the Resource Agent, an AI designed for crisis response. Your primary responsibilities include "
            "identifying, vetting, and managing available resources (e.g., supplies, volunteers, equipment). "
            "You have access to a `database_tools` for managing resources and `google_search` for verification. "
            "When identifying new resources, use the `database_tools.add_entry` function. "
            "When vetting resources, use `google_search` to find external information and `database_tools.update_entry` to update their status. "
            "When managing availability, use `database_tools.update_entry`. "
            "Always record significant actions and outcomes in the `memory_bank` associated with the current session identifier. "
            "Provide clear, concise responses and justifications for your actions."
        )

        print("ResourceAgent class initialized with LLM, tool_manager, session_manager, and memory_bank.")

    def identify_resource(self, resource_description, session_id):
        # Get current session or create a new one if it doesn't exist
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        # Use LLM to process the resource description and extract structured data
        prompt = f"Given the following resource description, extract its type, quantity, location, and any special notes. Output in JSON format: {resource_description}"

        # For demonstration, we'll simulate LLM response and tool use
        # In a real scenario, self.model.generate_content would be used
        # response = self.model.generate_content(prompt)
        # extracted_data = json.loads(response.text) # Assuming LLM outputs valid JSON

        # Mock LLM extraction for now
        if "water" in resource_description.lower():
            extracted_data = {"type": "Water Bottles", "quantity": "1000", "location": "Warehouse A", "notes": "Sealed, 500ml bottles"}
        elif "volunteer" in resource_description.lower():
            extracted_data = {"type": "Volunteers", "quantity": "10", "location": "Community Center", "notes": "Available for 8 hours/day"}
        else:
            extracted_data = {"type": "Unknown", "quantity": "N/A", "location": "N/A", "notes": resource_description}

        # Add the extracted resource to the database
        resource_id = self.db_tools.add_entry('resources', extracted_data)

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "identified_resource", "resource_id": resource_id, "data": extracted_data}
        )
        print(f"Resource identified and added with ID: {resource_id}")
        return resource_id

print("ResourceAgent class now includes the identify_resource method.")


class ResourceAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')
        self.google_search = self.tool_manager.get_tool('google_search')

        # Define the system instruction for the LLM
        self.system_instruction = (
            "You are the Resource Agent, an AI designed for crisis response. Your primary responsibilities include "
            "identifying, vetting, and managing available resources (e.g., supplies, volunteers, equipment). "
            "You have access to a `database_tools` for managing resources and `google_search` for verification. "
            "When identifying new resources, use the `database_tools.add_entry` function. "
            "When vetting resources, use `google_search` to find external information and `database_tools.update_entry` to update their status. "
            "When managing availability, use `database_tools.update_entry`. "
            "Always record significant actions and outcomes in the `memory_bank` associated with the current session identifier. "
            "Provide clear, concise responses and justifications for your actions."
        )

    def identify_resource(self, resource_description, session_id):
        # Get current session or create a new one if it doesn't exist
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        # Use LLM to process the resource description and extract structured data
        prompt = f"Given the following resource description, extract its type, quantity, location, and any special notes. Output in JSON format: {resource_description}"

        # Mock LLM extraction for now
        if "water" in resource_description.lower():
            extracted_data = {"type": "Water Bottles", "quantity": "1000", "location": "Warehouse A", "notes": "Sealed, 500ml bottles"}
        elif "volunteer" in resource_description.lower():
            extracted_data = {"type": "Volunteers", "quantity": "10", "location": "Community Center", "notes": "Available for 8 hours/day"}
        else:
            extracted_data = {"type": "Unknown", "quantity": "N/A", "location": "N/A", "notes": resource_description}

        # Add the extracted resource to the database
        resource_id = self.db_tools.add_entry('resources', extracted_data)

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "identified_resource", "resource_id": resource_id, "data": extracted_data}
        )
        print(f"Resource identified and added with ID: {resource_id}")
        return resource_id

    def vet_resource(self, resource_id, session_id):
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        resource = self.db_tools.get_entry('resources', resource_id)
        if not resource:
            print(f"Error: Resource with ID {resource_id} not found for vetting.")
            return False

        # Simulate LLM assessment (or use actual LLM call if integrated)
        # For now, we'll assume a basic check for 'verified' status based on content
        is_verified = False
        vetting_notes = ""

        if "verified" in resource.get("notes", "").lower() or "trusted" in resource.get("type", "").lower():
            is_verified = True
            vetting_notes = "Resource already marked as verified or from trusted source."
        else:
            # Use google_search to simulate external verification
            search_query = f"verify {resource.get('type', '')} {resource.get('location', '')} crisis relief organization"
            search_results = self.google_search.search(search_query)

            # Simple logic to determine verification from mock search results
            if any("legitimate" in r["snippet"].lower() for r in search_results):
                is_verified = True
                vetting_notes = "External search results indicate legitimacy."
            else:
                vetting_notes = "External search did not provide conclusive verification. Further manual review needed."

        # Update resource status in database
        update_successful = self.db_tools.update_entry('resources', resource_id, {'vetted': is_verified, 'vetting_notes': vetting_notes})

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "vetted_resource", "resource_id": resource_id, "status": is_verified, "notes": vetting_notes}
        )

        if update_successful:
            print(f"Resource ID {resource_id} vetted. Status: {is_verified}. Notes: {vetting_notes}")
            return True
        return False

print("ResourceAgent class now includes the vet_resource method.")


class ResourceAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')
        self.google_search = self.tool_manager.get_tool('google_search')

        # Define the system instruction for the LLM
        self.system_instruction = (
            "You are the Resource Agent, an AI designed for crisis response. Your primary responsibilities include "
            "identifying, vetting, and managing available resources (e.g., supplies, volunteers, equipment). "
            "You have access to a `database_tools` for managing resources and `google_search` for verification. "
            "When identifying new resources, use the `database_tools.add_entry` function. "
            "When vetting resources, use `google_search` to find external information and `database_tools.update_entry` to update their status. "
            "When managing availability, use `database_tools.update_entry`. "
            "Always record significant actions and outcomes in the `memory_bank` associated with the current session identifier. "
            "Provide clear, concise responses and justifications for your actions."
        )

    def identify_resource(self, resource_description, session_id):
        # Get current session or create a new one if it doesn't exist
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        # Use LLM to process the resource description and extract structured data
        prompt = f"Given the following resource description, extract its type, quantity, location, and any special notes. Output in JSON format: {resource_description}"

        # Mock LLM extraction for now
        if "water" in resource_description.lower():
            extracted_data = {"type": "Water Bottles", "quantity": "1000", "location": "Warehouse A", "notes": "Sealed, 500ml bottles"}
        elif "volunteer" in resource_description.lower():
            extracted_data = {"type": "Volunteers", "quantity": "10", "location": "Community Center", "notes": "Available for 8 hours/day"}
        else:
            extracted_data = {"type": "Unknown", "quantity": "N/A", "location": "N/A", "notes": resource_description}

        # Add the extracted resource to the database
        resource_id = self.db_tools.add_entry('resources', extracted_data)

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "identified_resource", "resource_id": resource_id, "data": extracted_data}
        )
        print(f"Resource identified and added with ID: {resource_id}")
        return resource_id

    def vet_resource(self, resource_id, session_id):
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        resource = self.db_tools.get_entry('resources', resource_id)
        if not resource:
            print(f"Error: Resource with ID {resource_id} not found for vetting.")
            return False

        # Simulate LLM assessment (or use actual LLM call if integrated)
        # For now, we'll assume a basic check for 'verified' status based on content
        is_verified = False
        vetting_notes = ""

        if "verified" in resource.get("notes", "").lower() or "trusted" in resource.get("type", "").lower():
            is_verified = True
            vetting_notes = "Resource already marked as verified or from trusted source."
        else:
            # Use google_search to simulate external verification
            search_query = f"verify {resource.get('type', '')} {resource.get('location', '')} crisis relief organization"
            search_results = self.google_search.search(search_query)

            # Simple logic to determine verification from mock search results
            if any("legitimate" in r["snippet"].lower() for r in search_results):
                is_verified = True
                vetting_notes = "External search results indicate legitimacy."
            else:
                vetting_notes = "External search did not provide conclusive verification. Further manual review needed."

        # Update resource status in database
        update_successful = self.db_tools.update_entry('resources', resource_id, {'vetted': is_verified, 'vetting_notes': vetting_notes})

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "vetted_resource", "resource_id": resource_id, "status": is_verified, "notes": vetting_notes}
        )

        if update_successful:
            print(f"Resource ID {resource_id} vetted. Status: {is_verified}. Notes: {vetting_notes}")
            return True
        return False

    def manage_availability(self, resource_id, new_status, session_id):
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        resource = self.db_tools.get_entry('resources', resource_id)
        if not resource:
            print(f"Error: Resource with ID {resource_id} not found for availability management.")
            return False

        # Update resource status in database
        update_successful = self.db_tools.update_entry('resources', resource_id, {'availability': new_status})

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "manage_availability", "resource_id": resource_id, "new_status": new_status}
        )

        if update_successful:
            print(f"Resource ID {resource_id} availability updated to: {new_status}.")
            return True
        return False

print("ResourceAgent class now includes the manage_availability method.")


model = genai.GenerativeModel('gemini-pro', safety_settings={'HARASSMENT':'block_none', 'HATE_SPEECH':'block_none', 'SEXUALLY_EXPLICIT':'block_none', 'DANGEROUS':'block_none'}) # Initialize with GEMINI_API_KEY if needed, or configure separately.
genai.configure(api_key=GEMINI_API_KEY)

print("GenerativeModel instantiated successfully.")


resource_agent = ResourceAgent(model, tool_manager, session_manager, memory_bank)

print("ResourceAgent instantiated successfully.")


class NeedsAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')
        self.google_search = self.tool_manager.get_tool('google_search')

        # Define the system instruction for the LLM
        self.system_instruction = (
            "You are the Needs Agent, an AI designed for crisis response. Your primary responsibilities include "
            "identifying, verifying, and prioritizing immediate needs in crisis scenarios. "
            "You have access to `database_tools` for managing needs and `google_search` for verification. "
            "When identifying new needs, use the `database_tools.add_entry` function. "
            "When verifying needs, use `google_search` to find external information and `database_tools.update_entry` to update their status. "
            "When prioritizing needs, use `database_tools.update_entry` to set priority levels. "
            "Always record significant actions and outcomes in the `memory_bank` associated with the current session identifier. "
            "Provide clear, concise responses and justifications for your actions."
        )

        print("NeedsAgent class initialized with LLM, tool_manager, session_manager, and memory_bank.")

print("NeedsAgent class definition started.")


class NeedsAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')
        self.google_search = self.tool_manager.get_tool('google_search')

        # Define the system instruction for the LLM
        self.system_instruction = (
            "You are the Needs Agent, an AI designed for crisis response. Your primary responsibilities include "
            "identifying, verifying, and prioritizing immediate needs in crisis scenarios. "
            "You have access to `database_tools` for managing needs and `google_search` for verification. "
            "When identifying new needs, use the `database_tools.add_entry` function. "
            "When verifying needs, use `google_search` to find external information and `database_tools.update_entry` to update their status. "
            "When prioritizing needs, use `database_tools.update_entry` to set priority levels. "
            "Always record significant actions and outcomes in the `memory_bank` associated with the current session identifier. "
            "Provide clear, concise responses and justifications for your actions."
        )

    def identify_need(self, need_description, session_id):
        # Get current session or create a new one if it doesn't exist
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        # Use LLM to process the need description and extract structured data
        prompt = f"Given the following need description, extract its type, quantity, location, urgency, and any special notes. Output in JSON format: {need_description}"

        # For demonstration, we'll simulate LLM response and tool use
        # In a real scenario, self.model.generate_content would be used
        # response = self.model.generate_content(prompt)
        # extracted_data = json.loads(response.text) # Assuming LLM outputs valid JSON

        # Mock LLM extraction for now
        if "medical supplies" in need_description.lower():
            extracted_data = {"type": "Medical Supplies", "quantity": "Critical", "location": "Field Hospital C", "urgency": "High", "notes": "Bandages, antiseptic, pain relievers"}
        elif "shelter" in need_description.lower():
            extracted_data = {"type": "Temporary Shelter", "quantity": "For 50 families", "location": "Community Park A", "urgency": "High", "notes": "Tents, blankets, sleeping bags"}
        elif "food" in need_description.lower():
            extracted_data = {"type": "Food Rations", "quantity": "For 200 people", "location": "Distribution Point B", "urgency": "Medium", "notes": "Non-perishable items"}
        else:
            extracted_data = {"type": "Unknown Need", "quantity": "N/A", "location": "N/A", "urgency": "Low", "notes": need_description}

        # Add the extracted need to the database
        need_id = self.db_tools.add_entry('needs', extracted_data)

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "identified_need", "need_id": need_id, "data": extracted_data}
        )
        print(f"Need identified and added with ID: {need_id}")
        return need_id

print("NeedsAgent class now includes the identify_need method.")


class NeedsAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')
        self.google_search = self.tool_manager.get_tool('google_search')

        # Define the system instruction for the LLM
        self.system_instruction = (
            "You are the Needs Agent, an AI designed for crisis response. Your primary responsibilities include "
            "identifying, verifying, and prioritizing immediate needs in crisis scenarios. "
            "You have access to `database_tools` for managing needs and `google_search` for verification. "
            "When identifying new needs, use the `database_tools.add_entry` function. "
            "When verifying needs, use `google_search` to find external information and `database_tools.update_entry` to update their status. "
            "When prioritizing needs, use `database_tools.update_entry` to set priority levels. "
            "Always record significant actions and outcomes in the `memory_bank` associated with the current session identifier. "
            "Provide clear, concise responses and justifications for your actions."
        )

    def identify_need(self, need_description, session_id):
        # Get current session or create a new one if it doesn't exist
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        # Use LLM to process the need description and extract structured data
        prompt = f"Given the following need description, extract its type, quantity, location, urgency, and any special notes. Output in JSON format: {need_description}"

        # Mock LLM extraction for now
        if "medical supplies" in need_description.lower():
            extracted_data = {"type": "Medical Supplies", "quantity": "Critical", "location": "Field Hospital C", "urgency": "High", "notes": "Bandages, antiseptic, pain relievers"}
        elif "shelter" in need_description.lower():
            extracted_data = {"type": "Temporary Shelter", "quantity": "For 50 families", "location": "Community Park A", "urgency": "High", "notes": "Tents, blankets, sleeping bags"}
        elif "food" in need_description.lower():
            extracted_data = {"type": "Food Rations", "quantity": "For 200 people", "location": "Distribution Point B", "urgency": "Medium", "notes": "Non-perishable items"}
        else:
            extracted_data = {"type": "Unknown Need", "quantity": "N/A", "location": "N/A", "urgency": "Low", "notes": need_description}

        # Add the extracted need to the database
        need_id = self.db_tools.add_entry('needs', extracted_data)

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "identified_need", "need_id": need_id, "data": extracted_data}
        )
        print(f"Need identified and added with ID: {need_id}")
        return need_id

    def verify_need(self, need_id, session_id):
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        need = self.db_tools.get_entry('needs', need_id)
        if not need:
            print(f"Error: Need with ID {need_id} not found for verification.")
            return False

        is_verified = False
        verification_notes = ""

        if "verified" in need.get("notes", "").lower():
            is_verified = True
            verification_notes = "Need already marked as verified."
        else:
            # Use google_search to simulate external verification
            search_query = f"verify need for {need.get('type', '')} in {need.get('location', '')}"
            search_results = self.google_search.search(search_query)

            if any("confirmed" in r["snippet"].lower() or "official report" in r["snippet"].lower() for r in search_results):
                is_verified = True
                verification_notes = "External search results confirm legitimacy."
            else:
                verification_notes = "External search did not provide conclusive verification. Further assessment needed."

        # Update need status in database
        update_successful = self.db_tools.update_entry('needs', need_id, {'verified': is_verified, 'verification_notes': verification_notes})

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "verified_need", "need_id": need_id, "status": is_verified, "notes": verification_notes}
        )

        if update_successful:
            print(f"Need ID {need_id} verified. Status: {is_verified}. Notes: {verification_notes}")
            return True
        return False

print("NeedsAgent class now includes the verify_need method.")


class NeedsAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')
        self.google_search = self.tool_manager.get_tool('google_search')

        # Define the system instruction for the LLM
        self.system_instruction = (
            "You are the Needs Agent, an AI designed for crisis response. Your primary responsibilities include "
            "identifying, verifying, and prioritizing immediate needs in crisis scenarios. "
            "You have access to `database_tools` for managing needs and `google_search` for verification. "
            "When identifying new needs, use the `database_tools.add_entry` function. "
            "When verifying needs, use `google_search` to find external information and `database_tools.update_entry` to update their status. "
            "When prioritizing needs, use `database_tools.update_entry` to set priority levels. "
            "Always record significant actions and outcomes in the `memory_bank` associated with the current session identifier. "
            "Provide clear, concise responses and justifications for your actions."
        )

    def identify_need(self, need_description, session_id):
        # Get current session or create a new one if it doesn't exist
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        # Use LLM to process the need description and extract structured data
        prompt = f"Given the following need description, extract its type, quantity, location, urgency, and any special notes. Output in JSON format: {need_description}"

        # Mock LLM extraction for now
        if "medical supplies" in need_description.lower():
            extracted_data = {"type": "Medical Supplies", "quantity": "Critical", "location": "Field Hospital C", "urgency": "High", "notes": "Bandages, antiseptic, pain relievers"}
        elif "shelter" in need_description.lower():
            extracted_data = {"type": "Temporary Shelter", "quantity": "For 50 families", "location": "Community Park A", "urgency": "High", "notes": "Tents, blankets, sleeping bags"}
        elif "food" in need_description.lower():
            extracted_data = {"type": "Food Rations", "quantity": "For 200 people", "location": "Distribution Point B", "urgency": "Medium", "notes": "Non-perishable items"}
        else:
            extracted_data = {"type": "Unknown Need", "quantity": "N/A", "location": "N/A", "urgency": "Low", "notes": need_description}

        # Add the extracted need to the database
        need_id = self.db_tools.add_entry('needs', extracted_data)

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "identified_need", "need_id": need_id, "data": extracted_data}
        )
        print(f"Need identified and added with ID: {need_id}")
        return need_id

    def verify_need(self, need_id, session_id):
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        need = self.db_tools.get_entry('needs', need_id)
        if not need:
            print(f"Error: Need with ID {need_id} not found for verification.")
            return False

        is_verified = False
        verification_notes = ""

        if "verified" in need.get("notes", "").lower():
            is_verified = True
            verification_notes = "Need already marked as verified."
        else:
            # Use google_search to simulate external verification
            search_query = f"verify need for {need.get('type', '')} in {need.get('location', '')}"
            search_results = self.google_search.search(search_query)

            if any("confirmed" in r["snippet"].lower() or "official report" in r["snippet"].lower() for r in search_results):
                is_verified = True
                verification_notes = "External search results confirm legitimacy."
            else:
                verification_notes = "External search did not provide conclusive verification. Further assessment needed."

        # Update need status in database
        update_successful = self.db_tools.update_entry('needs', need_id, {'verified': is_verified, 'verification_notes': verification_notes})

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "verified_need", "need_id": need_id, "status": is_verified, "notes": verification_notes}
        )

        if update_successful:
            print(f"Need ID {need_id} verified. Status: {is_verified}. Notes: {verification_notes}")
            return True
        return False

    def prioritize_need(self, need_id, priority_level, session_id):
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        need = self.db_tools.get_entry('needs', need_id)
        if not need:
            print(f"Error: Need with ID {need_id} not found for prioritization.")
            return False

        # Update need priority in database
        update_successful = self.db_tools.update_entry('needs', need_id, {'priority': priority_level})

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "prioritized_need", "need_id": need_id, "new_priority": priority_level}
        )

        if update_successful:
            print(f"Need ID {need_id} priority updated to: {priority_level}.")
            return True
        return False

print("NeedsAgent class now includes the prioritize_need method.")


needs_agent = NeedsAgent(model, tool_manager, session_manager, memory_bank)

print("NeedsAgent instantiated successfully.")


class MatchingAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')

        print("MatchingAgent class initialized with LLM, tool_manager, session_manager, and memory_bank.")

print("MatchingAgent class definition started.")


class MatchingAgent:
    def __init__(self, model, tool_manager, session_manager, memory_bank):
        self.model = model
        self.tool_manager = tool_manager
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.db_tools = self.tool_manager.get_tool('database_tools')

        # Define the system instruction for the LLM
        self.system_instruction = (
            "You are the Matching Agent, an AI designed for crisis response. Your primary responsibility is to "
            "intelligently match available resources with verified needs, considering various constraints and real-time data. "
            "You have access to `database_tools` for managing resources and needs. "
            "When performing matches, you will retrieve resources and needs, apply matching logic, and update their statuses "
            "in the database using `database_tools.update_entry`. "
            "Always record significant actions and outcomes (especially matched pairs) in the `memory_bank` "
            "associated with the current session identifier. "
            "Provide clear, concise responses and justifications for your matching decisions."
        )

        print("MatchingAgent class initialized with LLM, tool_manager, session_manager, memory_bank, and system instruction.")

    def match_resources_to_needs(self, session_id):
        session = self.session_manager.get_session(session_id)
        if not session:
            self.session_manager.create_session(session_id, {'history': []})
            session = self.session_manager.get_session(session_id)

        available_resources = {res_id: data for res_id, data in self.db_tools.list_entries('resources') if data.get('vetted', False) and data.get('availability', 'available') == 'available'}
        prioritized_needs = {need_id: data for need_id, data in self.db_tools.list_entries('needs') if data.get('verified', False) and data.get('priority') in ['High', 'Medium', 'Critical'] and data.get('status', 'open') == 'open'}

        matches = []
        matched_resource_ids = set()
        matched_need_ids = set()

        # Simple LLM-powered matching simulation (could be expanded with actual LLM calls for semantic matching)
        # For this simulation, we'll try to match 'Medical Supplies' to 'medical supplies' needs, etc.
        for need_id, need_data in prioritized_needs.items():
            if need_id in matched_need_ids: # Skip if already matched
                continue

            best_match_resource_id = None
            for resource_id, resource_data in available_resources.items():
                if resource_id in matched_resource_ids: # Skip if already used
                    continue

                # Basic matching logic based on type and quantity/urgency
                if (need_data['type'] == resource_data['type'] or
                    (need_data['type'].lower().startswith('medical') and resource_data['type'].lower().startswith('medical')) or
                    (need_data['type'].lower().startswith('food') and resource_data['type'].lower().startswith('food')) or
                    (need_data['type'].lower().startswith('shelter') and resource_data['type'].lower().startswith('temporary shelter'))):

                    # Further refine with quantity/urgency considerations (simplified)
                    if need_data['urgency'] == 'High' or need_data['urgency'] == 'Critical':
                        best_match_resource_id = resource_id
                        break # Prioritize urgent needs with any matching resource
                    elif need_data['urgency'] == 'Medium' and resource_data.get('quantity', '1') != 'N/A':
                        best_match_resource_id = resource_id
                        break

            if best_match_resource_id:
                matches.append({
                    'need_id': need_id,
                    'resource_id': best_match_resource_id,
                    'need_type': need_data['type'],
                    'resource_type': available_resources[best_match_resource_id]['type']
                })
                matched_need_ids.add(need_id)
                matched_resource_ids.add(best_match_resource_id)

        if not matches:
            print("No suitable matches found for the current needs and resources.")
            return []

        # Update database for matched resources and needs
        for match in matches:
            self.db_tools.update_entry('resources', match['resource_id'], {'availability': 'allocated', 'status': 'matched_to_need', 'matched_need_id': match['need_id']})
            self.db_tools.update_entry('needs', match['need_id'], {'status': 'fulfilled', 'matched_resource_id': match['resource_id']})

        # Record action in memory bank
        self.memory_bank.add_to_memory(
            session_id,
            {"action": "matched_resources_to_needs", "matches": matches}
        )

        print(f"Successfully matched {len(matches)} resource(s) to need(s).")
        for match in matches:
            print(f"  Need ID {match['need_id']} ({match['need_type']}) matched with Resource ID {match['resource_id']} ({match['resource_type']}).")

        return matches

print("MatchingAgent class now includes system_instruction and match_resources_to_needs method.")


matching_agent = MatchingAgent(model, tool_manager, session_manager, memory_bank)

print("MatchingAgent instantiated successfully.")


class CrisisResponseSystem:
    def __init__(self, resource_agent, needs_agent, matching_agent, session_manager, memory_bank, tool_manager):
        self.resource_agent = resource_agent
        self.needs_agent = needs_agent
        self.matching_agent = matching_agent
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.tool_manager = tool_manager
        self.db_tools = self.tool_manager.get_tool('database_tools')
        print("CrisisResponseSystem initialized.")

print("CrisisResponseSystem class definition started.")


class CrisisResponseSystem:
    def __init__(self, resource_agent, needs_agent, matching_agent, session_manager, memory_bank, tool_manager):
        self.resource_agent = resource_agent
        self.needs_agent = needs_agent
        self.matching_agent = matching_agent
        self.session_manager = session_manager
        self.memory_bank = memory_bank
        self.tool_manager = tool_manager
        self.db_tools = self.tool_manager.get_tool('database_tools')
        print("CrisisResponseSystem initialized.")

    def run_crisis_workflow(self, scenario_name="default_crisis_scenario"):
        print(f"\n--- Starting Crisis Workflow: {scenario_name} ---")
        session_id = scenario_name

        # a. Create a new session
        self.session_manager.create_session(session_id, {'workflow_stage': 'initialization'})
        print(f"Session '{session_id}' created for the workflow.")

        # b. Use the resource_agent to identify and vet resources
        print("\n--- Resource Agent: Identifying and Vetting Resources ---")
        resource_id_1 = self.resource_agent.identify_resource("1000 water bottles at Warehouse A, donated by Red Cross", session_id)
        self.resource_agent.vet_resource(resource_id_1, session_id)
        self.resource_agent.manage_availability(resource_id_1, 'available', session_id)

        resource_id_2 = self.resource_agent.identify_resource("5 medical doctors and 3 nurses available for 8 hours at Community Center", session_id)
        self.resource_agent.vet_resource(resource_id_2, session_id)
        self.resource_agent.manage_availability(resource_id_2, 'available', session_id)

        print("Current Resources in DB:")
        for res_id, res_data in self.db_tools.list_entries('resources'):
            print(f"  ID: {res_id}, Data: {res_data}")

        # c. Use the needs_agent to identify, verify, and prioritize needs
        print("\n--- Needs Agent: Identifying, Verifying, and Prioritizing Needs ---")
        need_id_1 = self.needs_agent.identify_need("Urgent need for medical supplies (bandages, antiseptic, pain relievers) at Field Hospital C", session_id)
        self.needs_agent.verify_need(need_id_1, session_id)
        self.needs_agent.prioritize_need(need_id_1, 'High', session_id)

        need_id_2 = self.needs_agent.identify_need("Need for temporary shelter for 50 families at Community Park A", session_id)
        self.needs_agent.verify_need(need_id_2, session_id)
        self.needs_agent.prioritize_need(need_id_2, 'Critical', session_id)

        print("Current Needs in DB:")
        for n_id, n_data in self.db_tools.list_entries('needs'):
            print(f"  ID: {n_id}, Data: {n_data}")

        # d. Use the matching_agent to attempt to match resources with needs
        print("\n--- Matching Agent: Matching Resources to Needs ---")
        self.session_manager.update_session(session_id, {'workflow_stage': 'matching'})
        matched_pairs = self.matching_agent.match_resources_to_needs(session_id)

        # e. Print the current state and memory entries
        print("\n--- Workflow Complete: Final State ---")
        print("\nFinal Resources in DB:")
        for res_id, res_data in self.db_tools.list_entries('resources'):
            print(f"  ID: {res_id}, Data: {res_data}")

        print("\nFinal Needs in DB:")
        for n_id, n_data in self.db_tools.list_entries('needs'):
            print(f"  ID: {n_id}, Data: {n_data}")

        print(f"\nMemory entries for session '{session_id}':")
        for entry in self.memory_bank.retrieve_memory(session_id):
            print(f"  - {entry}")

        print(f"--- Crisis Workflow: {scenario_name} Completed ---")

print("CrisisResponseSystem class now includes the run_crisis_workflow method.")


crisis_system = CrisisResponseSystem(
    resource_agent,
    needs_agent,
    matching_agent,
    session_manager,
    memory_bank,
    tool_manager
)

print("CrisisResponseSystem instantiated successfully.")


crisis_system.run_crisis_workflow("hurricane_response_2023")

