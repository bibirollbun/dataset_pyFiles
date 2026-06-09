import json
import datetime

class Coordinator:
    def __init__(self):
        self.conversation_log = []

    def ask(self, user_message):
        # Simple intent classification (placeholder)
        if "cancel" in user_message.lower():
            intent = "cancel_subscription"
        elif "invoice" in user_message.lower():
            intent = "billing_issue"
        elif "refund" in user_message.lower():
            intent = "refund_request"
        elif "help" in user_message.lower():
            intent = "general_support"
        else:
            intent = "unknown"

        response = {
            "timestamp": str(datetime.datetime.now()),
            "user_query": user_message,
            "detected_intent": intent,
            "agent_response": f"Processed message with intent: {intent}",
        }

        # Store each interaction for capstone evaluation
        self.conversation_log.append(response)
        return response


# ------------------------------------------------------------
# Enhanced Test Script for Capstone Project
# ------------------------------------------------------------
support_agent = Coordinator()

test_messages = [
    "I want to cancel my subscription.",
    "My invoice amount is wrong.",
    "I need a refund please.",
    "Hello, I need help.",
    "Is anyone there?",
    "Please update my address.",
]

print("==== CUSTOMER SUPPORT AGENT DEMO ====\n")

for user_msg in test_messages:
    print(f"[USER]: {user_msg}")
    output = support_agent.ask(user_msg)
    print("[AGENT OUTPUT]:")
    print(json.dumps(output, indent=4))
    print("-" * 70)

# ------------------------------------------------------------
# Export Conversation Log (extra feature for capstone)
# ------------------------------------------------------------
print("\n==== COMPLETE CONVERSATION LOG ====")
print(json.dumps(support_agent.conversation_log, indent=4))


