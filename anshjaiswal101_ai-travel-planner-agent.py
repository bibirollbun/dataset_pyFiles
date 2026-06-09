import os
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("travel")
genai.configure(api_key=secret_value_0)


MODEL_NAME = "gemini-2.5-flash"  # or the model used in your course

def build_prompt(destination, days, budget, traveller_type, interests, style):
    system_instructions = """
You are an expert travel planner.

Given:
- Destination
- Trip dates OR number of days
- Budget
- Traveller type
- Interests

You must return a detailed, day-wise itinerary.

Rules:
- Start with a short overview paragraph.
- Then create a section: "Summary of Trip".
- Then create: "Day-wise Itinerary".
  For each day include: Morning, Afternoon, Evening.
- Mention approximate budget level per day: (Low / Medium / High).
- End with a "Tips & Recommendations" section.
- Respond in clear bullet points and headings.
"""
    user_message = f"""
User Details:
Destination: {destination}
Number of Days: {days}
Budget: {budget}
Traveller Type: {traveller_type}
Interests: {interests}
Travel Style: {style}

Now create a personalized itinerary.
"""
    return system_instructions + "\n\n" + user_message

def generate_itinerary(destination, days, budget, traveller_type, interests, style):
    prompt = build_prompt(destination, days, budget, traveller_type, interests, style)
    
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    print("=== AI Travel Planner ===")
    destination = input("Destination: ")
    days = input("Number of days: ")
    budget = input("Budget (Low/Medium/High or amount): ")
    traveller_type = input("Traveller type (solo/couple/family/friends): ")
    interests = input("Interests (comma-separated): ")
    style = input("Travel style (relaxed/packed/adventure/etc.): ")

    itinerary = generate_itinerary(destination, days, budget, traveller_type, interests, style)
    print("\n=== Generated Itinerary ===\n")
    print(itinerary)


