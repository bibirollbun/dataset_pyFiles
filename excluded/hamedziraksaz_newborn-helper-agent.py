from kaggle_secrets import UserSecretsClient
import os

# Load GOOGLE_API_KEY from Kaggle user secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = api_key

print("GOOGLE_API_KEY is set:", "GOOGLE_API_KEY" in os.environ)


from typing import List, Dict

def make_step(title: str, description: str, german_label: str) -> Dict[str, str]:
    """
    Build a single checklist step item.

    Parameters:
        title: Short title of the step.
        description: Brief English explanation of the task.
        german_label: Key German label(s) found on official forms.

    Returns:
        A dictionary representing one step in the checklist.
    """
    return {
        "title": title,
        "description": description,
        "german_label": german_label,
    }


def newborn_admin_timeline(
    child_age_days: int,
    has_health_insurance: bool,
    has_residence_permit: bool,
) -> List[Dict[str, str]]:
    """
    Produce a general checklist of newborn administrative tasks for families in Germany.
    This function provides general guidance only and is not legal or financial advice.

    Parameters:
        child_age_days: Age of the newborn in days (approximate timeline support).
        has_health_insurance: Whether parents already have German health insurance.
        has_residence_permit: Whether parents hold a residence permit.

    Returns:
        A list of dictionaries, each describing one recommended administrative step.
    """
    steps: List[Dict[str, str]] = []

    # 1) Birth registration
    steps.append(
        make_step(
            title="Birth registration at Standesamt",
            description=(
                "Use the hospital birth notification (Geburtsanzeige) to register the "
                "birth at the Standesamt. Bring passports, residence permits, and "
                "marriage certificate if applicable."
            ),
            german_label="Geburtsanzeige / Standesamt",
        )
    )

    # 2) Health insurance
    if has_health_insurance:
        steps.append(
            make_step(
                title="Register baby with health insurance",
                description=(
                    "Notify your Krankenkasse and complete the newborn registration "
                    "form. Typical labels: 'Familienversicherung', 'Mitgliedsnummer', "
                    "'Geburtsdatum des Kindes'."
                ),
                german_label="Familienversicherung / Neugeborenes",
            )
        )

    # 3) Address registration
    steps.append(
        make_step(
            title="Update registration of address (if required)",
            description=(
                "Depending on local rules, notify the BÃ¼rgeramt/MeldebehÃ¶rde that the "
                "child is living in your household. Look for labels like 'Anmeldung' "
                "or 'Ã„nderung der Meldedaten'."
            ),
            german_label="Anmeldung / Melderegister",
        )
    )

    # 4) Family benefits
    steps.append(
        make_step(
            title="Prepare documents for family benefits",
            description=(
                "Gather required documents for child benefits (e.g. birth certificate, "
                "tax ID when available, bank account details). Follow official forms "
                "and instructions."
            ),
            german_label="Leistungen fÃ¼r Familien (z.B. Kindergeld)",
        )
    )

    # 5) Residence reminder (if no permit)
    if not has_residence_permit:
        steps.append(
            make_step(
                title="Clarify residence status for the child",
                description=(
                    "If the parents do not yet hold a residence permit, consult the "
                    "AuslÃ¤nderbehÃ¶rde or an accredited advisory service. This system "
                    "cannot provide legal advice."
                ),
                german_label="AuslÃ¤nderbehÃ¶rde / Aufenthalt des Kindes",
            )
        )

    return steps



from google.adk.agents import LlmAgent

DEFAULT_MODEL = "gemini-2.0-flash"

newborn_agent = LlmAgent(
    name="newborn_admin_helper",
    model=DEFAULT_MODEL,
    description="Helps new parents with a clear newborn admin checklist in Germany.",
    instruction="""
You are a calm, multilingual assistant for families living in Germany whose baby was
recently born and who have limited German proficiency.

Your goals:
- Explain typical administrative steps after birth in a clear checklist.
- Use the `newborn_admin_timeline` tool to build a structured list of steps.
- Show important German form labels so users can match them to the fields.
- Never give legal, medical, or financial advice.
- Always encourage users to verify details on official German authority websites.
- Keep the tone supportive, respectful, and simple.

When you answer:
1. Decide reasonable values for:
   - child_age_days (approximate)
   - has_health_insurance (True/False)
   - has_residence_permit (True/False)
2. Call the tool `newborn_admin_timeline` with those values.
3. Turn the result into a human-friendly checklist with:
   - numbered steps,
   - short explanation in the user's language (here: English in the demo),
   - and the key German label(s) for each step.

Important safety:
- Do NOT promise eligibility.
- Do NOT invent deadlines; say they may exist and must be checked officially.
""",
    tools=[newborn_admin_timeline],
)


import asyncio
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from IPython.display import Markdown, display


APP_NAME = "newborn_helper_app"
USER_ID = "demo_user"
SESSION_ID = "session_1"


async def main():
    # 1) Create a session
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # 2) Create runner
    runner = Runner(
        agent=newborn_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # 3) Sample user query
    query = (
        "Our baby was recently born in Germany. We have limited German skills and "
        "need a simple checklist of the essential administrative steps."
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    # 4) Run the agent and capture the final response text
    final_text = None
    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    # 5) Display as a styled â€œUI cardâ€�
    if final_text:
        card_md = f"""
<style>
.newborn-card {{
  border-radius: 12px;
  border: 1px solid #e0e0e0;
  padding: 18px 22px;
  margin-top: 8px;
  background: #fcfcff;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.newborn-card h2, .newborn-card h3 {{
  margin-top: 0.4rem;
}}
.newborn-card p {{
  margin-bottom: 0.4rem;
}}
.newborn-card ol {{
  padding-left: 1.2rem;
}}
.newborn-card em {{
  color: #555;
}}
</style>

<div class="newborn-card">
{final_text}
</div>
"""
        display(Markdown(card_md))
    else:
        print("No final response received. Check logs above.")


await main()


