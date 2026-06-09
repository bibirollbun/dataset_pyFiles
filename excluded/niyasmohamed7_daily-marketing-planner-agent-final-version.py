import pandas as pd
import random



store_name = "My Small Shop"

products = [
    {"name": "Red T-Shirt", "category": "Clothing", "price": 499},
    {"name": "Blue Jeans", "category": "Clothing", "price": 999},
    {"name": "Sports Shoes", "category": "Footwear", "price": 1499},
    {"name": "Smart Watch", "category": "Electronics", "price": 1999},
]



def promotion_planner_agent(store_name, products, total_days, minutes_per_day):
    channels = [
        "WhatsApp Status",
        "Instagram Reel",
        "Facebook Post",
        "Local Poster",
        "SMS Broadcast",
    ]

    goals = [
        "Increase awareness",
        "Clear old stock",
        "Promote new arrival",
        "Increase online orders",
        "Drive foot traffic to shop",
    ]

    plan = []

    for day in range(1, total_days + 1):
        product = products[(day - 1) % len(products)]  # rotate products
        channel = random.choice(channels)
        goal = random.choice(goals)

        content_idea = (
            f"Create a {channel.lower()} highlighting {product['name']} "
            f"({product['category']}) with a small discount or offer. "
            f"Focus on: {goal.lower()}."
        )

        plan.append({
            "Day": day,
            "Store": store_name,
            "Focus Product": product["name"],
            "Channel": channel,
            "Goal": goal,
            "Time (minutes)": minutes_per_day,
            "Content Idea": content_idea,
            "Approx Price": product["price"],
        })

    return plan



total_days = 7
minutes_per_day = 45

promotion_plan = promotion_planner_agent(
    store_name=store_name,
    products=products,
    total_days=total_days,
    minutes_per_day=minutes_per_day,
)

df_plan = pd.DataFrame(promotion_plan)
df_plan


