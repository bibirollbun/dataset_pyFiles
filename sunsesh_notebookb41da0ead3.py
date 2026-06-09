# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import logging
import warnings

# Suppress specific Google GenAI warnings
#logging.getLogger("google_genai.types").setLevel(logging.ERROR)
#logging.getLogger("google.genai.types").setLevel(logging.ERROR)

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")
# Suppress general warnings to keep the output clean
#warnings.filterwarnings("ignore")


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search, AgentTool, ToolContext, load_memory, preload_memory
from google.adk.code_executors import BuiltInCodeExecutor

print("âœ… ADK components imported successfully.")


def show_python_code_and_result(response):
    for i in range(len(response)):
        # Check if the response contains a valid function call result from the code executor
        if (
            (response[i].content.parts)
            and (response[i].content.parts[0])
            and (response[i].content.parts[0].function_response)
            and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])


print("âœ… Helper functions defined.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)
print("âœ… Retry confguration complete successfully.")



from datetime import date, datetime
def verify_eligibility(inv_number: int, inv_date:str, product_id:int, amount:int, ) -> dict:
    """Looks up the invoice details (invoice number as integer, date in yyyy-mm-dd format, 
    amount as integer, product id as integer) in the arrray of (simulated) invoices (in real life, it will be a 
    database) stored as dict in the same module. If it exists and the invoice is not more than 90 days old, 
    the tool returns true and if not, false

    Args:
        inv_number, inv_date, product_id, amount

    Returns:
        Dictionary with status and any error message
        Success: {"status": "ok", "message": "ok"}
        Error: {"status": "notok","message":"Invalid Invoice" or "Eligible period has been exceeded"}
    """
    # This simulates looking up the invoice details.
    invoice_database = [
        [34789,'2025-01-01',6574,300],
        [46789,'2025-10-01',6434,200],
        [58432,'2025-09-10',6438,250]
    ]        
    for sale in invoice_database:
        if sale[0]== inv_number and sale[1] == inv_date and sale[2] == product_id and sale[3] == amount:
            if (datetime.now().date() - datetime.strptime(inv_date,'%Y-%m-%d').date()).days <= 90:
                return {"status":"ok", "message":"ok"}   
            else:
                return {"status":"notok", "message":"Eligible Period has been exceeded"}
    return {"status":"notok","message":"Invalid invoice"}

print("âœ… Invoice and Eligibility Verification function created")
print(f"ğŸ’³ Test: {verify_eligibility(46789,'2025-10-01',6434,200)}")


from google import genai
def check_mattress_condition(file_path, mime_type, inv_number:int):
    """
    Courtesy: Google Search
    
    Reads a local file and store its content to the Gemini API for analysis.

    Args:
        file_path (str): The path to the local file.
        mime_type (str): The MIME type of the file (e.g., 'text/plain', 'application/pdf', 'image/jpeg').
        prompt_text (str): The prompt to send along with the file content.
    """
    
    try:
        with open("/kaggle/input/images/" + file_path, "rb") as f:
            file_content = f.read()

        # Prepare the content for the Gemini API
        file_part = types.Part.from_bytes(data=file_content, mime_type=mime_type)

        # Create a Gemini client (assuming you have authenticated)
        client = genai.Client() # Or configure with your API key if not using default credentials

        # Send the content to the Gemini model and ask it to analyze
        prompt_text="""Review the file and produce an output as below:
            if the image is not proper, say, 'Invalid Image for invoice %s'
            if the image is proper and the mattress as the image is free of any damages or if there is no soiled portion,
            say 'The mattress seems Ok. (Invoice Reference:%s) We will get back to you after one more manual inspection, 
            if required for further steps'
            Else, say 'Ineligible for return as the mattress (%s) is damaged or spoiled'
        """ % (inv_number, inv_number, inv_number)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=[prompt_text, file_part]
        )
        print(response.text)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
check_mattress_condition("damageno.jpeg", "image/jpeg",324)


# validator agent with custom function tools
validator_agent = LlmAgent(
    name="validator_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a smart agent to validate the request for the refund when a customer wants to return the  
    mattress subject to certain conditions handled by the below actions:

    1. Use 'verify_eligibility()' to validate the invoice details and the period within which the return can take place
    2. Use 'check_mattress_condition()' to ensure that the mattress is free of any soiling or damages.

    You may return the result produced by the second tool. 
    
    If the first tool returns status "notok", explain the issue to the user clearly.
    """,
    tools=[load_memory, verify_eligibility, check_mattress_condition],
)

print("âœ… validator_agent created")
print("ğŸ”§ Available tools:")
print("  â€¢ verify_eligibility- Looks up the correctness of the invoice and that the complaint is made within 90 days of purchase")
print("  â€¢ check_mattress_condition - Verify the condition of the mattress based on the image uploaded")

