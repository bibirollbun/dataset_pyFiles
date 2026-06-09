# Run this cell first
!pip install --quiet google-generativeai
# optional pretty printing
!pip install --quiet rich


from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# Load API key from Kaggle Secrets
api_key = UserSecretsClient().get_secret("GEMINI_API_KEY")
print("Loaded Key?:", api_key is not None)

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Add it in Notebook → Secrets.")

# Configure API
genai.configure(api_key=api_key)

# Use a supported model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Test generation
response = model.generate_content("Hello Gemini, this is a test message!")
print(response.text)


def call_gemini(prompt: str, model_name: str = "gemini-1.5-flash", max_output_tokens: int = 512):
    """
    Simple wrapper to call Gemini and return text.
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt, max_output_tokens=max_output_tokens)
    # If the SDK returns an object, pick .text ; adapt if different
    return getattr(response, "text", str(response))


def summarize_email(email_text: str) -> str:
    prompt = f"""
You are an email analysis assistant. Given the email below:
1) Provide 5 concise bullet points summarizing the main ideas.
2) Provide a 2-line short summary.
3) Extract action items in a numbered list with deadlines if any.

Email:
\"\"\"{email_text}\"\"\"
"""
    return call_gemini(prompt)


def detect_urgency(email_text: str) -> str:
    prompt = f"""
Classify the urgency of this email as one of: LOW, MEDIUM, HIGH.
Then provide a one-sentence rationale.

Email:
\"\"\"{email_text}\"\"\"
"""
    return call_gemini(prompt)


def extract_action_items(email_text: str) -> str:
    prompt = f"""
Extract action items from the email below. For each item, return:
- task (short)
- who (if mentioned)
- deadline (if mentioned, in ISO or plain text)
Format as a numbered list.

Email:
\"\"\"{email_text}\"\"\"
"""
    return call_gemini(prompt)


def generate_replies(email_text: str) -> str:
    prompt = f"""
Generate three reply drafts to the email below. Label them:
1) Formal corporate
2) Friendly professional
3) Very short (1-2 lines)

Each draft should:
- Acknowledge the email
- Address action items briefly
- Propose next steps if required

Email:
\"\"\"{email_text}\"\"\"
"""
    return call_gemini(prompt)


def final_polished_reply(email_text: str) -> str:
    prompt = f"""
Create one final polished reply to the email below.
Rules:
- Keep it polite and professional.
- Explicitly list the next steps and deadlines you will follow.
- Keep it reasonably brief (4-8 sentences).
Email:
\"\"\"{email_text}\"\"\"
"""
    return call_gemini(prompt)


def email_agent(email_text: str) -> dict:
    outputs = {}
    outputs["summary"] = summarize_email(email_text)
    outputs["urgency"] = detect_urgency(email_text)
    outputs["action_items"] = extract_action_items(email_text)
    outputs["reply_options"] = generate_replies(email_text)
    outputs["final_reply"] = final_polished_reply(email_text)
    return outputs


def call_gemini(prompt, model_name="models/gemini-2.5-flash"):
    """
    Sends a prompt to Gemini and returns the text response.
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    
    # Return text field if exists
    return getattr(response, "text", str(response))


# Paste any email here to test interactively
test_email = "Hello, this is a test email."
result = email_agent(test_email)
print("\n--- Final reply ---\n")
print(result["final_reply"])


import json

def save_result(email_text: str, result: dict, filename: str = "email_agent_output.json"):
    payload = {
        "email": email_text,
        "result": result
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved to {filename}")

# Now this will work
save_result(test_email, result)

