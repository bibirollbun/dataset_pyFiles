def law_ai_agent(situation_type, details):
    """
    Simulates a basic AI agent that provides relevant law/rights 
    based on a simplified, categorized situation.

    Args:
        situation_type (str): A keyword describing the legal area (e.g., 'consumer_goods', 'tenant').
        details (str): A brief description of what happened.

    Returns:
        tuple: (Relevant Law/Right Title, Plain Language Explanation)
    """

    # --- 1. Consumer Goods / Purchases ---
    if situation_type.lower() == "consumer_goods":
        if "broken" in details.lower() or "not working" in details.lower():
            return (
                "Right to Goods of Satisfactory Quality",
                "**The Law:** Goods must be of a quality that a reasonable person would consider satisfactory, taking into account the price and description. If they break or don't work soon after purchase, you generally have a right to a **refund, repair, or replacement** from the seller. You should notify the seller as soon as possible."
            )
        elif "misrepresented" in details.lower() or "not as described" in details.lower():
            return (
                "Right to Goods Matching Description",
                "**The Law:** The product you receive must match any description given to you by the seller (e.g., on the box, website, or told verbally). If it doesn't, you usually have a right to a **refund**."
            )
        else:
            return (
                "General Consumer Protection",
                "**The Law:** You have a right to fair dealing. Always keep your **receipt** or proof of purchase, as it is key evidence for any claim."
            )

    # --- 2. Other Simple Legal Areas (You would expand here) ---
    elif situation_type.lower() == "rental_deposit":
        return (
            "Tenant's Right to Deposit Protection",
            "**The Law (General Principle):** Your landlord must secure your security deposit in a government-approved scheme and give you the scheme's details within a certain timeframe. This ensures you can get it back easily if you meet the terms of your lease."
        )

    # --- Default/No Match ---
    else:
        return (
            "Information Not Found / General Advice",
            "**The Law (Action Step):** This situation is complex or not in our database yet. Always document everything (photos, emails, dates) and consider consulting a **legal professional** or a **specialized free legal aid service** in your area."
        )

# --- Sample Problem and Solution ---
print("--- LAW AI AGENT SIMULATION ---")
print("\n[SCENARIO 1: Faulty Product]")

# The user input
problem_area = "consumer_goods"
problem_details = "I bought a brand new blender yesterday, but it started making a grinding noise and stopped working this morning. I lost the receipt."

print(f"\n**User Situation:**")
print(f"Area: **{problem_area}**")
print(f"Details: {problem_details}")

# Run the AI Agent
law_title, law_explanation = law_ai_agent(problem_area, problem_details)

print("\n--- ðŸ’¡ LAW AI AGENT SOLUTION ---")
print(f"## {law_title}")
print(law_explanation)
print("\n---")


import pandas as pd

# 1. LAW AI AGENT KNOWLEDGE BASE (Simplified Dictionary)
# In a real project, this data would come from a database (SQL or CSV).
LEGAL_KNOWLEDGE = {
    "consumer_faulty_product": {
        "title": "Right to Goods of Satisfactory Quality",
        "keywords": ["broken", "not working", "faulty", "failed"],
        "explanation": "**The Law:** Goods must be fit for their purpose and of satisfactory quality. If a product fails within a short period (e.g., 30 days), you generally have the right to a **full refund** or a **replacement/repair** from the seller. Keep proof of purchase (receipt)!",
        "action_step": "Contact the retailer immediately and demand a remedy based on the fault."
    },
    "tenant_deposit_return": {
        "title": "Tenant's Right to Security Deposit Return",
        "keywords": ["deposit", "returned", "not receiving", "landlord kept"],
        "explanation": "**The Law:** Landlords must return the security deposit shortly after the lease ends, typically within 14-30 days, unless there are justified deductions for damage beyond normal wear and tear or unpaid rent. Deductions must be itemized and explained.",
        "action_step": "Send a formal written demand letter to the landlord requesting the deposit back, specifying the deadline."
    },
    "digital_data_sharing": {
        "title": "Right to Data Privacy and Consent",
        "keywords": ["data", "shared", "consent", "private"],
        "explanation": "**The Law:** Companies generally need your explicit and informed consent before collecting, processing, or sharing your personal data (e.g., email, location, habits) with third parties. You usually have the right to know what data they hold.",
        "action_step": "Review the company's privacy policy and send a formal email asking them to disclose what data they have shared and to exercise your 'Right to Be Forgotten,' if applicable."
    }
}

def law_ai_agent_solver(situation_description):
    """
    Analyzes a user's situation and finds the closest matching law from the knowledge base.
    """
    situation_lower = situation_description.lower()
    
    # Simple keyword search logic
    for key, law_data in LEGAL_KNOWLEDGE.items():
        for keyword in law_data["keywords"]:
            if keyword in situation_lower:
                return law_data

    # If no match is found
    return {
        "title": "Information Not Found / General Advice",
        "explanation": "**The Law (Action Step):** Your problem doesn't match a simple category. Please consult a **legal aid service** or provide more details. Always document everything (photos, emails, dates).",
        "action_step": "Seek professional consultation."
    }


# --- SAMPLE PROBLEM FLOWS ---

def run_sample_case(case_number, area, problem):
    """Helper function to run and print a case neatly."""
    print(f"\n--- ðŸ“‚ SAMPLE CASE {case_number}: {area} ---")
    print(f"**User Situation:** {problem}")
    
    # Run the AI Agent
    result = law_ai_agent_solver(problem)
    
    # Display the Solution
    print("\nðŸ’¡ **LAW AI AGENT SOLUTION**")
    print(f"## {result['title']}")
    print(f"**Law Explained:** {result['explanation']}")
    print(f"**Action Plan:** {result['action_step']}")
    print("-" * 50)


# --- Running the Cases ---
print("--- LAW AI AGENT SIMULATION START ---")

# 1. Consumer Rights Case
run_sample_case(
    1,
    "Consumer Rights (Product Fault)",
    "I bought a TV five days ago, but the screen is now completely **broken** and **not working**. They say it's my fault."
)

# 2. Landlord-Tenant Case
run_sample_case(
    2,
    "Landlord-Tenant Law (Deposit)",
    "My lease ended a month ago, and the **landlord kept** my full **deposit** without giving me any reason or invoice for repairs."
)

# 3. Digital Privacy Case
run_sample_case(
    3,
    "Digital Privacy (Data Sharing)",
    "I'm worried about my personal **data** like email and location being **shared** by a social media company without my clear **consent**."
)

# 4. No Match Case (General Advice)
run_sample_case(
    4,
    "No Match / Complex Issue",
    "My neighbor keeps throwing trash on my driveway, and I don't know what to do."
)

