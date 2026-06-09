from kaggle_secrets import UserSecretsClient
import os

# Load secret from Kaggle
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("OPENAI_API_KEY")

# Put it into an environment variable (optional but useful)
os.environ["OPENAI_API_KEY"] = api_key

print(os.environ.get("OPENAI_API_KEY"))



# -------------------------------------------------------------
# Automated Email Reply Assistant (Simulated) - Full Working Code
# -------------------------------------------------------------

# This version works without OpenAI API.
# It generates simple polite replies based on basic rules.

def generate_reply(email_text):
    """
    Simulates an email reply.
    Rules:
    - If email contains "meeting", respond with scheduling reply
    - If email contains "question", respond politely
    - Otherwise, send a generic polite acknowledgement
    """
    email_lower = email_text.lower()
    
    if "meeting" in email_lower:
        reply = """Hello,

Thank you for reaching out. I am available to schedule a meeting.
Please let me know a suitable time.

Best regards,"""
    elif "question" in email_lower:
        reply = """Hello,

Thank you for your email. I will be happy to answer your question.
Please provide more details if needed.

Best regards,"""
    else:
        reply = """Hello,

Thank you for your email. I have received it and will get back to you shortly.

Best regards,"""
    
    return reply

# -------------------------------------------------------------
# Example Emails
# -------------------------------------------------------------

emails = [
    "Hi, can we schedule a meeting this Friday afternoon to discuss the project status?",
    "I have a question about the last report you sent.",
    "Just wanted to say thank you for your help!"
]

# -------------------------------------------------------------
# Generate Replies
# -------------------------------------------------------------
for i, email in enumerate(emails, start=1):
    print(f"Email {i}: {email}\n")
    reply = generate_reply(email)
    print("Generated Reply:\n", reply)
    print("-" * 50)


