from IPython.display import Image
Image("/kaggle/input/image-food/food_waste.jpg")


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
import joblib


def generate_synthetic_cafeteria_data(n=2000, seed=42):
    # Set random seeds so results are reproducible
    np.random.seed(seed)
    random.seed(seed)

    # Define the types of food items served in the cafeteria
    food_types = ["Fruit", "Vegetables", "Meat", "Dairy", "Bread", "Soup", "Dessert", "Sandwich"]
    rows = []

    for _ in range(n):
        # Pick a random school size (number of students present that day)
        attendance = int(np.random.choice(range(120, 520)))

        # Simulate the daily temperature (average around 15°C, but can vary)
        temp = int(np.random.normal(loc=15, scale=10))
        temp = max(-5, min(40, temp))  # keep it within realistic bounds

        # Randomly choose which food item is being served
        food = random.choice(food_types)

        # Each food has a baseline popularity (fraction of students who usually take it)
        popularity = {
            "Fruit": 0.30,
            "Vegetables": 0.45,
            "Meat": 0.6,
            "Dairy": 0.25,
            "Bread": 0.5,
            "Soup": 0.35,
            "Dessert": 0.4,
            "Sandwich": 0.55
        }
        base_ratio = popularity[food]

        # Adjust demand depending on the weather:
        # - Hot days → more fruit/vegetables/desserts
        # - Cold days → more soup, meat, bread
        if food in ["Soup", "Meat", "Bread"]:
            weather_factor = 1.0 + max(0, (12 - temp)) * 0.03   # colder → more demand
            weather_factor -= max(0, (temp - 25)) * 0.01        # very hot → less demand
        elif food in ["Fruit", "Vegetables", "Dessert"]:
            weather_factor = 1.0 + max(0, (temp - 22)) * 0.04   # hotter → more demand
            weather_factor -= max(0, (12 - temp)) * 0.02        # very cold → less demand
        else:  # Dairy & Sandwich are less sensitive to weather
            weather_factor = 1.0 + (12 - temp) * 0.005

        # Add weekday effect:
        # - Slightly lower demand on Mondays
        # - Higher demand on Thursdays and Fridays
        weekday = random.randint(0, 4)  # 0=Mon ... 4=Fri
        weekday_factor = 1.0 + (0.05 if weekday in [3,4] else -0.02 if weekday == 0 else 0.0)

        # Calculate expected demand and add some random noise
        expected = attendance * base_ratio * weather_factor * weekday_factor
        consumed = int(max(0, expected + np.random.normal(0, attendance * 0.03)))

        # Kitchen usually prepares a bit more than expected to avoid shortages
        produced = int(min(attendance + int(attendance * 0.2),
                           consumed + np.random.randint(0, int(attendance * 0.15))))

        # Make sure values are logical (can’t consume more than attendance, waste ≥ 0)
        consumed = min(consumed, attendance)
        waste = max(0, produced - consumed)

        # Save the record
        rows.append({
            "attendance": attendance,
            "temperature": temp,
            "weekday": weekday,
            "Menu_Item": food,
            "meals_consumed": consumed,
            "meals_produced": produced,
            "waste": waste
        })

    # Build dataframe and calculate waste percentage
    df = pd.DataFrame(rows)
    df["waste_percent"] = df["waste"] / df["meals_produced"] * 100
    return df

# Generate dataset with 3000 rows
df = generate_synthetic_cafeteria_data(n=3000)
df.head()


# Quick check: dataset shape and summary
print("Dataset shape:", df.shape)
display(df.describe(include="all").T)

# Plot consumption vs temperature for Fruit and Soup
plt.figure(figsize=(12,5))
sns.scatterplot(
    data=df[df['Menu_Item'].isin(['Fruit','Soup'])],
    x='temperature', y='meals_consumed',
    hue='Menu_Item', alpha=0.4
)
plt.title("Meals consumed vs temperature (Fruit vs Soup)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Meals Consumed")
plt.show()


features = ["attendance", "temperature", "weekday", "Menu_Item"]
target = "meals_consumed"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

cat_features = ["Menu_Item"]

# Create and train CatBoost
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=7,
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

# train using Pool to pass categorical features by name
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

model.fit(train_pool, eval_set=test_pool, use_best_model=True, early_stopping_rounds=50)


preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.2f}, R2: {r2:.3f}")

# Plot true vs predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.3)
plt.plot([0, max(y_test.max(), preds.max())], [0, max(y_test.max(), preds.max())], 'r--')
plt.xlabel("Actual consumed")
plt.ylabel("Predicted consumed")
plt.title("Actual vs Predicted (meals_consumed)")
plt.show()
joblib.dump(model, "catboost_demand_model.joblib")


def predict_consumption(model, attendance, temperature, weekday, menu_item):
    row = pd.DataFrame([{
        "attendance": attendance,
        "temperature": temperature,
        "weekday": weekday,
        "Menu_Item": menu_item
    }])
    return float(model.predict(row)[0])

# Example
example_pred = predict_consumption(model, attendance=200, temperature=10, weekday=2, menu_item="Fruit")
print("Predicted consumed:", round(example_pred,2))


import os
from google import genai
from kaggle_secrets import UserSecretsClient

class SmartMealGuardianAgent:
    def __init__(self, model):
        # Keep a reference to the trained CatBoost model
        self.model = model

    def predict_consumption(self, attendance, temperature, weekday, menu_item):
        # Use the trained model to predict how many portions will actually be eaten
        pred = predict_consumption(self.model, attendance, temperature, weekday, menu_item)
        # Put a logical cap: predictions can’t go below 0 or above ~110% of attendance
        pred = max(0, min(pred, attendance * 1.1))
        return round(pred, 2)

    def estimate_waste(self, produced, predicted_consumed):
        # Compare how many meals were produced vs. how many are expected to be eaten
        produced = int(produced)
        waste = max(0, produced - int(round(predicted_consumed)))
        # Calculate waste percentage relative to what was produced
        waste_pct = (waste / produced) * 100 if produced > 0 else 0
        return int(waste), round(waste_pct, 2)

    def generate_recommendation(self, attendance, temperature, weekday, menu_item, produced=None):
        """
        Predict demand, estimate waste, and then ask Gemini to turn this into
        clear, actionable advice for the cafeteria staff.
        """
        # Step 1: Predict how many portions will be consumed
        pred_consumed = self.predict_consumption(attendance, temperature, weekday, menu_item)

        # Step 2: Decide how much to produce (add a buffer depending on food type)
        buffer_map = {"Soup": 0.2, "Fruit": 0.05, "Meat": 0.1, "Dessert": 0.15}
        if produced is None:
            produced = int(pred_consumed * (1 + buffer_map.get(menu_item, 0.1)))

        # Step 3: Estimate waste based on predicted consumption vs. production
        waste, waste_pct = self.estimate_waste(produced, pred_consumed)

        # Step 4: Build a short summary of the situation
        summary = (
            f"Predicted consumption for {menu_item} with attendance={attendance}, "
            f"temperature={temperature}°C, weekday={weekday}: {pred_consumed:.0f} portions. "
            f"Planned production: {produced} -> estimated waste {waste} portions ({waste_pct}%)."
        )

        # Step 5: Create a prompt for Gemini to turn numbers into human-friendly recommendations
        prompt = f"""
You are SmartMeal Guardian assistant. Given the following cafeteria summary, propose concise, prioritized operational actions for the kitchen staff to reduce waste and improve satisfaction.
Summary:
{summary}

Provide:
1) Top 3 immediate actions (e.g., reduce batch size, change portioning).
2) Medium term actions (data-driven changes).
3) One communication to students to reduce waste (short).
Limit to 120-180 words.
"""

        # Step 6: Call Gemini with the prompt
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("MY_GEMINI_KEY")   
        if not api_key:
            raise RuntimeError("MY_GEMINI_KEY not set in environment.")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = response.text

        # Step 7: Return everything together (numbers + Gemini recommendation)
        return {
            "summary": summary,
            "predicted_consumed": pred_consumed,
            "produced": produced,
            "waste": waste,
            "waste_percent": waste_pct,
            "recommendation": text
        }

# Usage example
agent = SmartMealGuardianAgent(model)

# Predict demand and get Gemini’s advice for Fruit on a mild day
res = agent.generate_recommendation(attendance=300, temperature=20, weekday=2, menu_item="Fruit")
print(res["summary"])
print(res["recommendation"])


# Define test scenarios (edge cases)
buffer_map = {"Soup": 0.2, "Fruit": 0.05, "Meat": 0.1, "Dessert": 0.15}

cases = [
    {"attendance": 200, "temperature": 1,  "weekday": 2, "menu_item": "Fruit"},
    {"attendance": 200, "temperature": 1,  "weekday": 2, "menu_item": "Soup"},
    {"attendance": 200, "temperature": 30, "weekday": 2, "menu_item": "Fruit"},
    {"attendance": 200, "temperature": 30, "weekday": 2, "menu_item": "Soup"},
]

# Run predictions for each case
print("### Edge Case Tests ###")
for c in cases:
    pred = agent.predict_consumption(c["attendance"], c["temperature"], c["weekday"], c["menu_item"])
    buffer = buffer_map.get(c["menu_item"], 0.1)
    produced = int(pred * (1 + buffer))
    waste, waste_pct = agent.estimate_waste(produced, pred)
    print(f"{c} => pred: {pred:.2f}, produced: {produced}, waste: {waste} ({waste_pct:.2f}%)")


# Example cycle
agent = SmartMealGuardianAgent(model)

# Case: Fruit on a hot day
res = agent.generate_recommendation(attendance=300, temperature=30, weekday=2, menu_item="Fruit")
print(res["summary"])
print(res["recommendation"])

# Case: Soup on a cold day
res = agent.generate_recommendation(attendance=300, temperature=5, weekday=3, menu_item="Soup")
print(res["summary"])
print(res["recommendation"])


plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="temperature", y="waste_percent", hue="Menu_Item", alpha=0.4)
plt.title("Waste % vs Temperature by Food Type")
plt.xlabel("Temperature (°C)")
plt.ylabel("Waste Percentage (%)")
plt.show()


plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="attendance", y="meals_consumed", hue="Menu_Item", alpha=0.4)
plt.title("Meals Consumed vs Attendance by Food Type")
plt.xlabel("Attendance")
plt.ylabel("Meals Consumed")
plt.show()

daily_waste = df.groupby("weekday")["waste"].mean()
plt.figure(figsize=(8,5))
daily_waste.plot(kind="line", marker="o", color="tomato")
plt.title("Average Waste by Weekday")
plt.xlabel("Weekday (0=Mon ... 4=Fri)")
plt.ylabel("Average Waste Portions")
plt.show()

