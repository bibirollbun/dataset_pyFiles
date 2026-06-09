import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from IPython.display import display

print("âœ… All libraries imported successfully.")


user_profile = {
    'name': 'Alice',
    'age': 30,
    'budget': 500,  # Adjusted for a more interesting plan
    'trip_days': 3,
    'interests': ['culture', 'anime', 'food'],
    'destination': 'Tokyo'
}
attractions_data = [
    {'name': 'Tokyo Tower', 'category': 'culture', 'cost': 20, 'duration_hr': 2},
    {'name': 'Akihabara Tour', 'category': 'anime', 'cost': 40, 'duration_hr': 3},
    {'name': 'Tsukiji Sushi Market', 'category': 'food', 'cost': 25, 'duration_hr': 2},
    {'name': 'Skytree Visit', 'category': 'culture', 'cost': 30, 'duration_hr': 2.5},
    {'name': 'Ramen Museum', 'category': 'food', 'cost': 15, 'duration_hr': 1.5},
    {'name': 'Shibuya Anime Shops', 'category': 'anime', 'cost': 0, 'duration_hr': 2},
    {'name': 'Imperial Palace', 'category': 'culture', 'cost': 0, 'duration_hr': 3},
]
df_attractions = pd.DataFrame(attractions_data)

print("\n--- User Profile and Attraction Data Loaded ---")
display(df_attractions.head())


# This dictionary simulates a database where the agent stores user info
agent_memory = {}

def save_user_profile(user_id, profile):
    """Saves a user's profile to the agent's memory."""
    agent_memory[user_id] = profile
    print(f"\nğŸ§  Profile for user '{user_id}' saved to agent memory.")

def get_user_profile(user_id):
    """Retrieves a user's profile from memory."""
    return agent_memory.get(user_id, "User not found.")

# Let's save and retrieve our sample user
save_user_profile("u001", user_profile)
retrieved_profile = get_user_profile("u001")
print("Retrieved Profile:", retrieved_profile['name'])


# This agent finds hotels that fit the user's budget.
hotel_data = [
    {'name': 'Tokyo Budget Inn', 'stars': 3, 'cost_per_night': 60, 'category': 'budget'},
    {'name': 'Shinjuku Comfort', 'stars': 4, 'cost_per_night': 120, 'category': 'standard'},
    {'name': 'Five Star Imperial', 'stars': 5, 'cost_per_night': 220, 'category': 'luxury'}
]
hotel_df = pd.DataFrame(hotel_data)

def hotel_agent(budget_per_night):
    """Recommends hotels based on the nightly budget."""
    return hotel_df[hotel_df['cost_per_night'] <= budget_per_night]

# Calculate daily budget and get hotel recommendations
daily_budget = user_profile['budget'] / user_profile['trip_days']
recommended_hotels = hotel_agent(daily_budget)
print("\n--- ğŸ�¨ Hotel Agent Recommendations ---")
print(f"Based on a daily budget of ${daily_budget:.2f}, we recommend:")
display(recommended_hotels)


# This agent creates a plan of activities based on interests, time, and budget.
def activity_agent(df, profile):
    """Generates a plan of activities based on user profile."""
    user_interests = profile['interests']
    trip_days = profile['trip_days']
    time_limit = trip_days * 6  # Max 6 hours of activities per day
    total_time = 0
    budget_remaining = profile['budget']

    activities = []
    # Prioritize interests
    for interest in user_interests:
        matches = df[df['category'] == interest].sample(frac=1)  # Shuffle to get variety
        for _, activity in matches.iterrows():
            if (total_time + activity['duration_hr'] <= time_limit) and (budget_remaining - activity['cost'] >= 0):
                activities.append(activity)
                total_time += activity['duration_hr']
                budget_remaining -= activity['cost']
    
    plan = pd.DataFrame(activities)
    return plan, profile['budget'] - budget_remaining

# Generate the final activity plan
final_plan, used_amount = activity_agent(df_attractions, user_profile)
print("\n--- ğŸ�¡ Activity Agent Itinerary ---")
display(final_plan)
print(f"ğŸ’¸ Total spent on activities: ${used_amount}")


# This agent turns the list of activities into a timed schedule.
def build_schedule(plan_df):
    """Builds a daily schedule from a DataFrame of activities."""
    if plan_df.empty:
        return pd.DataFrame()
        
    start_time = datetime.today().replace(hour=9, minute=0)
    schedule = []

    for i, row in plan_df.iterrows():
        duration = timedelta(hours=row['duration_hr'])
        end_time = start_time + duration
        schedule.append({
            'Activity': row['name'], 
            'Start': start_time.strftime('%H:%M'), 
            'End': end_time.strftime('%H:%M'),
            'Type': row['category']
        })
        start_time = end_time + timedelta(minutes=30)  # Add 30 min rest time
    
    return pd.DataFrame(schedule)

# Create and display the visual schedule
schedule = build_schedule(final_plan)
print("\n--- ğŸ“… Visual Schedule ---")
display(schedule)


# This simulates a friendly, text-based response from the agent.
def agent_response(user_name, plan_df):
    """Generates a friendly summary of the plan."""
    reply = f"ğŸ‘‹ Hi {user_name}, here's your personalized itinerary for {user_profile.get('destination', 'your trip')}:\n\n"
    if plan_df.empty:
        return reply + "No activities could be planned with the current constraints."
        
    for i, row in plan_df.iterrows():
        reply += f"ğŸ”¹ {row['name']} â€” ({row['category']}, ${row['cost']}) for {row['duration_hr']} hours\n"
    return reply

# Generate and print the agent's reply
print("\n--- ğŸ’¬ Agent's Final Response ---")
# **FIXED**: Using `final_plan` which was generated in Step 4
print(agent_response(user_profile['name'], final_plan))


# This creates the final output file for submission.
if not final_plan.empty:
    # **FIXED**: Using `final_plan` to create the submission file
    plan_file = final_plan[['name', 'category', 'cost', 'duration_hr']]
    plan_file.columns = ['Activity', 'Category', 'Cost($)', 'Duration(Hours)']

    # Save to submission.csv
    plan_file.to_csv('submission.csv', index=False)

    print("\n--- ğŸ“„ Final Plan for Submission ---")
    print("âœ… Saved your itinerary to submission.csv")
    display(plan_file)
else:
    print("\n--- ğŸ“„ Final Plan for Submission ---")
    print("âš ï¸� No plan generated, so submission file was not created.")

