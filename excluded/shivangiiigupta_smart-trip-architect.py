import random
import json
from datetime import datetime, timedelta

# -----------------------------
# MEMORY AGENT
# -----------------------------
class MemoryAgent:
    def __init__(self):
        self.session_memory = {}
        self.long_term_memory = {}

    def store_session(self, user_id, info):
        self.session_memory[user_id] = info

    def store_long_term(self, user_id, info):
        if user_id not in self.long_term_memory:
            self.long_term_memory[user_id] = []
        self.long_term_memory[user_id].append(info)

    def retrieve_session(self, user_id):
        return self.session_memory.get(user_id, {})

    def retrieve_long_term(self, user_id):
        return self.long_term_memory.get(user_id, [])

# -----------------------------
# INPUT AGENT
# -----------------------------
class InputAgent:
    def collect_input(self, query):
        data = {"destination": None, "days": None, "budget": None, "interests": []}
        text = query.lower()
        destinations = ["goa", "manali", "jaipur", "delhi", "mumbai", "agra", "bangalore"]
        for d in destinations:
            if d in text:
                data["destination"] = d.capitalize()
        for word in text.split():
            if word.isdigit():
                num = int(word)
                if 1 <= num <= 30:
                    data["days"] = num
        if "budget" in text or "rs" in text or "â‚¹" in text:
            nums = [int(s) for s in text.split() if s.isdigit()]
            if nums:
                data["budget"] = nums[0]
        possible_interests = ["food", "adventure", "history", "nature", "shopping", "photography"]
        found_interests = [i for i in possible_interests if i in text]
        data["interests"] = found_interests if found_interests else ["general"]
        return data

# -----------------------------
# TOOL AGENT â€” Offline realistic cost
# -----------------------------
class ToolAgent:
    activity_prices = {
        "Goa": {
            "Visit scenic viewpoint": 200,
            "Adventure activity": 1500,
            "Local restaurant": 500,
            "Museum / historic site": 300,
            "Shopping": 1000
        },
        "Manali": {
            "Visit scenic viewpoint": 250,
            "Adventure activity": 1200,
            "Local restaurant": 400,
            "Museum / historic site": 200,
            "Shopping": 800
        }
    }

    def get_weather(self, destination):
        options = ["Sunny", "Cloudy", "Rainy", "Cool", "Humid"]
        return random.choice(options)

    def estimate_food_cost(self, days):
        return days * 500

    def estimate_travel_cost(self, days):
        return days * 300

    def estimate_activity_cost(self, destination, day_activities):
        city_prices = self.activity_prices.get(destination, {})
        total = 0
        for act in day_activities:
            total += city_prices.get(act, 400)
        return total

# -----------------------------
# LLM Agent â€” makes itinerary attractive
# -----------------------------
class LLMAgent:
    def describe_day(self, day_info):
        desc = f"ðŸŒž Day {day_info['day']} ({day_info['date']}):\n"
        for act in day_info['activities']:
            # Simple template to make it feel more human-like
            if "scenic" in act:
                desc += f"   - Enjoy a beautiful scenic viewpoint and take amazing photos.\n"
            elif "Adventure" in act:
                desc += f"   - Get your adrenaline pumping with an exciting adventure activity!\n"
            elif "Local restaurant" in act:
                desc += f"   - Savor delicious local cuisine at a popular restaurant.\n"
            elif "Museum" in act:
                desc += f"   - Explore the history and culture at a fascinating museum.\n"
            elif "Shopping" in act:
                desc += f"   - Shop for souvenirs and local crafts at famous markets.\n"
            else:
                desc += f"   - {act}\n"
        desc += f"   Estimated travel: {day_info['estimated_travel_km']} km (~{day_info['estimated_travel_time_hr']} hr)\n"
        desc += f"   Estimated cost for the day: â‚¹{day_info['day_activity_cost']}\n"
        return desc

# -----------------------------
# PLANNER AGENT
# -----------------------------
class PlannerAgent:
    def __init__(self, tools: ToolAgent, llm: LLMAgent):
        self.tools = tools
        self.llm = llm

    def generate_itinerary(self, info, memory_agent: MemoryAgent, user_id="default_user"):
        destination = info["destination"]
        days = info["days"] or 3
        budget = info["budget"] or 10000
        interests = info["interests"]

        itinerary = []
        for day in range(1, days + 1):
            activities = []
            if "nature" in interests:
                activities.append("Visit scenic viewpoint")
            if "food" in interests:
                activities.append("Local restaurant")
            if "adventure" in interests:
                activities.append("Adventure activity")
            if "history" in interests:
                activities.append("Museum / historic site")
            if "shopping" in interests:
                activities.append("Shopping")
            if not activities:
                activities.append("Explore city attractions")

            travel_km = random.randint(5, 30)
            travel_time_hr = round(travel_km / 30, 1)
            day_cost = self.tools.estimate_activity_cost(destination, activities)

            day_info = {
                "day": day,
                "date": str(datetime.now().date() + timedelta(days=day-1)),
                "activities": activities[:2],
                "estimated_travel_km": travel_km,
                "estimated_travel_time_hr": travel_time_hr,
                "day_activity_cost": day_cost
            }
            day_info['llm_description'] = self.llm.describe_day(day_info)
            itinerary.append(day_info)

        total_activity_cost = sum(day["day_activity_cost"] for day in itinerary)
        total_food = self.tools.estimate_food_cost(days)
        total_travel = self.tools.estimate_travel_cost(days)
        total_estimate = total_activity_cost + total_food + total_travel

        memory_agent.store_session(user_id, info)
        memory_agent.store_long_term(user_id, {"destination": destination, "days": days, "budget": budget, "interests": interests})

        plan = {
            "destination": destination,
            "days": days,
            "interests": interests,
            "budget_input": budget,
            "estimated_total_cost": total_estimate,
            "cost_breakdown": {
                "food": total_food,
                "local_travel": total_travel,
                "activities": total_activity_cost
            },
            "weather": self.tools.get_weather(destination),
            "itinerary": itinerary
        }
        return plan

# -----------------------------
# MAIN EXECUTION
# -----------------------------
def run_travel_agent(user_query, user_id="user_1"):
    memory = MemoryAgent()
    input_agent = InputAgent()
    tools = ToolAgent()
    llm = LLMAgent()
    planner = PlannerAgent(tools, llm)
    info = input_agent.collect_input(user_query)
    plan = planner.generate_itinerary(info, memory, user_id)
    return plan

# -----------------------------
# EXAMPLE RUN
# -----------------------------
query = "I want to go to Goa for 5 days with a budget of 20000 INR. I like food, adventure, and nature."
result = run_travel_agent(query, user_id="user_1")

print("==== Travel Planner Agent Output ====")
print(f"Destination: {result['destination']}")
print(f"Days: {result['days']}")
print(f"Interests: {', '.join(result['interests'])}")
print(f"Estimated Total Cost: â‚¹{result['estimated_total_cost']}")
print(f"Weather Forecast: {result['weather']}")
print("Cost Breakdown:", result['cost_breakdown'])
print("\nDay-wise Itinerary (LLM-enhanced descriptions):")
for day in result['itinerary']:
    print(day['llm_description'])

# save for Kaggle notebook display
with open("travel_plan_llm.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nItinerary saved to travel_plan_llm.json")


