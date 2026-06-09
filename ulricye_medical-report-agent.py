%pip install -q google-adk pymupdf4llm


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


from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.adk.tools import google_search, AgentTool, ToolContext
from google.genai import types
import pymupdf4llm
import pathlib

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")


input_dir = '../input'


# md_text = pymupdf4llm.to_markdown(f'{input_dir}/sample-medical-reports/sample cardiology report.pdf')

# # now work with the markdown text, e.g. store as a UTF8-encoded file
# clean_text = md_text.encode('utf-8', errors='ignore').decode('utf-8')

# pathlib.Path("output.md").write_bytes(clean_text.encode('utf-8'))


import fitz  # PyMuPDF
import pathlib

def pdf_to_clean_markdown(file_path: str) -> str:
    """Extract text from pdf.

    Args:
        pdf_path: The path to the pdf file.

    Returns:
        Text from the pdf.
    """
    doc = fitz.open(file_path)
    text = ""
    
    for page in doc:
        text += page.get_text("text", sort=True)
    
    doc.close()
    
    return text

def pdf_to_md(file_name: str) -> dict:
    """Convert pdf to md file.

    Args:
        file_name: The name of the file to convert.

    Returns:
        Dictionary with status and file path.
        Success: {"status": "success", "path": './file.md'}
        Error: {"status": "error", "error_message": "Unsupported file type"}
    """
    print(f'filename: {file_name}')
    file_path = f'../input/sample-medical-reports/{file_name}'
    filename_with_ext = os.path.basename(file_path)
    _, ext = os.path.splitext(filename_with_ext)
    if ext.lower() != '.pdf':
        return {
            "status": "error",
            "error_message": f"Unsupported file type: {ext}",
        }

    md_text = pdf_to_clean_markdown(file_path)
    pathlib.Path(f'{file_name}.md').write_bytes(md_text.encode('utf-8'))

    return {"status": "success", "path": f'{file_name}.md'}

print("âœ… PDF to MD function created")
print(f"ðŸ’± Test: {pdf_to_md('sample cardiology report.pdf')}")


# Pay attention to the docstring, type hints, and return value.
def get_fee_for_payment_method(method: str) -> dict:
    """Looks up the transaction fee percentage for a given payment method.

    This tool simulates looking up a company's internal fee structure based on
    the name of the payment method provided by the user.

    Args:
        method: The name of the payment method. It should be descriptive,
                e.g., "platinum credit card" or "bank transfer".

    Returns:
        Dictionary with status and fee information.
        Success: {"status": "success", "fee_percentage": 0.02}
        Error: {"status": "error", "error_message": "Payment method not found"}
    """
    # This simulates looking up a company's internal fee structure.
    fee_database = {
        "platinum credit card": 0.02,  # 2%
        "gold debit card": 0.035,  # 3.5%
        "bank transfer": 0.01,  # 1%
    }

    fee = fee_database.get(method.lower())
    if fee is not None:
        return {"status": "success", "fee_percentage": fee}
    else:
        return {
            "status": "error",
            "error_message": f"Payment method '{method}' not found",
        }


print("âœ… Fee lookup function created")
print(f"ðŸ’³ Test: {get_fee_for_payment_method('platinum credit card')}")


def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Looks up and returns the exchange rate between two currencies.

    Args:
        base_currency: The ISO 4217 currency code of the currency you
                       are converting from (e.g., "USD").
        target_currency: The ISO 4217 currency code of the currency you
                         are converting to (e.g., "EUR").

    Returns:
        Dictionary with status and rate information.
        Success: {"status": "success", "rate": 0.93}
        Error: {"status": "error", "error_message": "Unsupported currency pair"}
    """

    # Static data simulating a live exchange rate API
    # In production, this would call something like: requests.get("api.exchangerates.com")
    rate_database = {
        "usd": {
            "eur": 0.93,  # Euro
            "jpy": 157.50,  # Japanese Yen
            "inr": 83.58,  # Indian Rupee
        }
    }

    # Input validation and processing
    base = base_currency.lower()
    target = target_currency.lower()

    # Return structured result with status
    rate = rate_database.get(base, {}).get(target)
    if rate is not None:
        return {"status": "success", "rate": rate}
    else:
        return {
            "status": "error",
            "error_message": f"Unsupported currency pair: {base_currency}/{target_currency}",
        }


print("âœ… Exchange rate function created")
print(f"ðŸ’± Test: {get_exchange_rate('USD', 'EUR')}")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


research_agent = Agent(
    name="ResearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
    tools=[google_search],
    output_key="research_findings",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… research_agent created.")


summarizer_agent = Agent(
    name="SummarizerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Read the provided research findings and create a concise summary as a bulleted list with 3-5 key points. 
    Include citations for each point when available.""",
    output_key="final_summary",
)

print("âœ… summarizer_agent created.")


medical_agent = LlmAgent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer questions about medical reports.",
    instruction="""You are a smart medical report assistant.

    For medical report inquiry:
    1. Ask the user for the medical report file name if not given
    2. Use `pdf_to_md()` to convert pdf to markdown file. This returns a file path.
    3. Read the content from the markdown file path returned by `pdf_to_md`.
    4. Analyze the markdown content to answer the question
    5. Output the answer in simple text that non-medical people can understand.
    6. Otherwise, follow this sequential workflow:
       a. First use `ResearchAgent` tool to find relevant information based on the question
       b. Then use `SummarizerAgent` tool to summarize the research findings
    7. Present the final summary to the user in simple words.
    
    If any tool returns status "error", explain the issue to the user clearly.
    
    Important: Always use the tools in the correct sequence when research is needed.
    """,
    tools=[pdf_to_md, AgentTool(research_agent), AgentTool(summarizer_agent)],
)

print("âœ… Root Agent defined.")


medical_runner = InMemoryRunner(agent=medical_agent)
candidates = await medical_runner.run_debug(
    "What is EEG?"
)




