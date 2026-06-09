# ==========================================
# AI-Based Smart Traffic Control System
# Kaggle Capstone - Baseline Implementation
# ==========================================

# -----------------------
# 1. Imports
# -----------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error


# -----------------------
# 2. Synthetic Data Generation
# -----------------------

def generate_synthetic_traffic_data(n_samples=5000, random_state=42):
    """
    Generate synthetic traffic data for a 4-way intersection.
    Features:
        - north_count, east_count, south_count, west_count
        - time_of_day (0-23 hour)
        - is_weekend (0 or 1)
    Targets:
        - next_green_direction (categorical: 'N', 'E', 'S', 'W')
        - green_duration_sec (float)
    """
    rng = np.random.default_rng(random_state)

    # Random vehicle counts per direction (0 to 50)
    north = rng.integers(0, 51, size=n_samples)
    east = rng.integers(0, 51, size=n_samples)
    south = rng.integers(0, 51, size=n_samples)
    west = rng.integers(0, 51, size=n_samples)

    # Time of day: 0-23
    time_of_day = rng.integers(0, 24, size=n_samples)

    # Weekend flag
    is_weekend = rng.integers(0, 2, size=n_samples)

    # Stack for convenience
    counts = np.vstack([north, east, south, west]).T

    # Decide "true" next green direction: mostly the max queue, but add noise
    directions = np.array(["N", "E", "S", "W"])
    max_idx = counts.argmax(axis=1)

    # Add a bit of randomness: sometimes prioritize slightly less crowded direction
    noise = rng.random(n_samples)
    noisy_idx = max_idx.copy()
    switch_mask = noise < 0.1  # 10% of the time we pick a non-max direction
    random_alt = rng.integers(0, 4, size=n_samples)
    noisy_idx[switch_mask] = random_alt[switch_mask]

    next_green_direction = directions[noisy_idx]

    # Decide green duration (in seconds) based on total vehicles
    total_cars = counts.sum(axis=1)
    # Base 10s + 0.8s per vehicle, clipped between 10 and 90s
    green_duration = 10 + 0.8 * total_cars
    green_duration = np.clip(green_duration, 10, 90)

    # Small random noise
    green_duration = green_duration + rng.normal(0, 5, size=n_samples)
    green_duration = np.clip(green_duration, 10, 90)

    df = pd.DataFrame({
        "north_count": north,
        "east_count": east,
        "south_count": south,
        "west_count": west,
        "time_of_day": time_of_day,
        "is_weekend": is_weekend,
        "next_green_direction": next_green_direction,
        "green_duration_sec": green_duration
    })

    return df


# Generate dataset
df = generate_synthetic_traffic_data(n_samples=8000)
print("Sample of synthetic data:")
print(df.head())


# -----------------------
# 3. Train / Test Split
# -----------------------

X = df[["north_count", "east_count", "south_count", "west_count",
        "time_of_day", "is_weekend"]]

y_direction = df["next_green_direction"]
y_duration = df["green_duration_sec"]

X_train, X_test, y_dir_train, y_dir_test, y_dur_train, y_dur_test = train_test_split(
    X, y_direction, y_duration, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])


# -----------------------
# 4. Train Models
# -----------------------

# 4.1 Classifier for next_green_direction
clf_direction = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

clf_direction.fit(X_train, y_dir_train)

# 4.2 Regressor for green_duration_sec
reg_duration = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

reg_duration.fit(X_train, y_dur_train)


# -----------------------
# 5. Evaluate Models
# -----------------------

# 5.1 Direction classifier performance
y_dir_pred = clf_direction.predict(X_test)
print("\n=== Classification Report: Next Green Direction ===")
print(classification_report(y_dir_test, y_dir_pred))

# 5.2 Duration regressor performance
y_dur_pred = reg_duration.predict(X_test)
mae = mean_absolute_error(y_dur_test, y_dur_pred)
print("\n=== Regression Performance: Green Duration (sec) ===")
print(f"Mean Absolute Error: {mae:.2f} seconds")


# -----------------------
# 6. Smart Traffic Controller Class
# -----------------------

class SmartTrafficController:
    def __init__(self, direction_model, duration_model):
        """
        direction_model: classifier predicting 'N','E','S','W'
        duration_model: regressor predicting green duration in seconds
        """
        self.direction_model = direction_model
        self.duration_model = duration_model

    def decide_signal(self, north_count, east_count, south_count, west_count,
                      time_of_day, is_weekend):
        """
        Returns:
            - next_dir: 'N','E','S','W'
            - green_duration: float (seconds)
        """
        x = pd.DataFrame([{
            "north_count": north_count,
            "east_count": east_count,
            "south_count": south_count,
            "west_count": west_count,
            "time_of_day": time_of_day,
            "is_weekend": is_weekend
        }])

        next_dir = self.direction_model.predict(x)[0]
        green_duration = float(self.duration_model.predict(x)[0])

        # Safety clipping
        green_duration = max(10.0, min(90.0, green_duration))

        return next_dir, green_duration


# Instantiate controller
controller = SmartTrafficController(clf_direction, reg_duration)


# -----------------------
# 7. Demo: Simulating Real-Time Decisions
# -----------------------

def simulate_controller_step(controller, north, east, south, west, hour, weekend):
    next_dir, duration = controller.decide_signal(
        north_count=north,
        east_count=east,
        south_count=south,
        west_count=west,
        time_of_day=hour,
        is_weekend=weekend
    )
    print(f"Traffic State: N={north}, E={east}, S={south}, W={west}, hour={hour}, weekend={weekend}")
    print(f"Controller Decision: Give GREEN to {next_dir} for ~{duration:.1f} seconds\n")


# Example scenarios
print("\n=== Demo: Controller Decisions ===")
simulate_controller_step(controller, north=40, east=10, south=5, west=2, hour=9, weekend=0)
simulate_controller_step(controller, north=10, east=30, south=25, west=5, hour=18, weekend=0)
simulate_controller_step(controller, north=5, east=5, south=5, west=5, hour=2, weekend=1)
simulate_controller_step(controller, north=0, east=45, south=40, west=10, hour=20, weekend=0)


# -----------------------
# 8. Notes for Report / Presentation (Not Code, Just Guidance)
# -----------------------
# - This is a simplified AI-based controller using supervised learning.
# - Real system would use live sensors / CCTV to estimate vehicle counts.
# - You could extend this by:
#       * adding average waiting time as feature or label
#       * including separate models for each phase pattern
#       * using RL (reinforcement learning) for more advanced control.
# - In explanation, clearly show:
#       * data preprocessing pipeline
#       * model training + evaluation
#       * how real-time decision is made from input features.



# ============================================================
# AI-Based Smart Traffic Control System - Kaggle Capstone
# ============================================================

# -----------------------
# 1. Imports
# -----------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error

import matplotlib.pyplot as plt
import joblib

# For nicer prints
pd.set_option("display.max_columns", None)


# -----------------------
# 2. Synthetic Data Generation
# -----------------------

def generate_synthetic_traffic_data(n_samples=8000, random_state=42):
    """
    Generate synthetic data for a 4-way intersection.
    
    Features:
        - north_count, east_count, south_count, west_count : vehicle counts per lane
        - time_of_day : hour (0-23)
        - is_weekend : 0 (weekday) or 1 (weekend)
    
    Targets:
        - next_green_direction : 'N','E','S','W'
        - green_duration_sec : duration for green in seconds
    """
    rng = np.random.default_rng(random_state)

    north = rng.integers(0, 51, size=n_samples)
    east  = rng.integers(0, 51, size=n_samples)
    south = rng.integers(0, 51, size=n_samples)
    west  = rng.integers(0, 51, size=n_samples)

    time_of_day = rng.integers(0, 24, size=n_samples)
    is_weekend  = rng.integers(0, 2, size=n_samples)

    counts = np.vstack([north, east, south, west]).T
    directions = np.array(["N", "E", "S", "W"])

    # Basic rule: choose lane with maximum queue
    max_idx = counts.argmax(axis=1)

    # Add 10% randomness so it's not trivial
    noise = rng.random(n_samples)
    noisy_idx = max_idx.copy()
    switch_mask = noise < 0.10
    random_alt = rng.integers(0, 4, size=n_samples)
    noisy_idx[switch_mask] = random_alt[switch_mask]

    next_green_direction = directions[noisy_idx]

    # Green duration depends on total vehicles
    total_cars = counts.sum(axis=1)
    # Base 10s + 0.8s per vehicle, clipped [10, 90]
    green_duration = 10 + 0.8 * total_cars
    green_duration = np.clip(green_duration, 10, 90)

    # Add Gaussian noise
    green_duration = green_duration + rng.normal(0, 5, size=n_samples)
    green_duration = np.clip(green_duration, 10, 90)

    df = pd.DataFrame({
        "north_count": north,
        "east_count": east,
        "south_count": south,
        "west_count": west,
        "time_of_day": time_of_day,
        "is_weekend": is_weekend,
        "next_green_direction": next_green_direction,
        "green_duration_sec": green_duration
    })

    return df


# Generate dataset
df = generate_synthetic_traffic_data(n_samples=8000)
print("Sample data:")
print(df.head())


# -----------------------
# 3. Basic EDA
# -----------------------

print("\nDataset shape:", df.shape)
print("\nDescribe numeric columns:")
print(df.describe())

# Add total vehicles column for analysis
df["total_vehicles"] = df["north_count"] + df["east_count"] + df["south_count"] + df["west_count"]

# Histogram of counts
plt.figure()
df[["north_count", "east_count", "south_count", "west_count"]].hist(bins=20)
plt.suptitle("Vehicle Count Distribution per Direction")
plt.tight_layout()
plt.show()

# Green duration distribution
plt.figure()
df["green_duration_sec"].hist(bins=20)
plt.title("Green Duration (sec) Distribution")
plt.xlabel("Duration (sec)")
plt.ylabel("Frequency")
plt.show()

# Total vehicles vs duration
plt.figure()
plt.scatter(df["total_vehicles"], df["green_duration_sec"], alpha=0.3)
plt.title("Total Vehicles vs Green Duration")
plt.xlabel("Total Vehicles")
plt.ylabel("Green Duration (sec)")
plt.grid(True)
plt.show()

# Average load per hour
avg_by_hour = df.groupby("time_of_day")["total_vehicles"].mean()
plt.figure()
avg_by_hour.plot(kind="bar")
plt.title("Average Total Vehicles by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Average Vehicles")
plt.xticks(rotation=0)
plt.show()


# -----------------------
# 4. Train/Test Split
# -----------------------

feature_cols = ["north_count", "east_count", "south_count", "west_count",
                "time_of_day", "is_weekend"]

X = df[feature_cols]
y_dir = df["next_green_direction"]
y_dur = df["green_duration_sec"]

X_train, X_test, y_dir_train, y_dir_test, y_dur_train, y_dur_test = train_test_split(
    X, y_dir, y_dur, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])


# -----------------------
# 5. Train Models
# -----------------------

# Classifier for direction
clf_direction = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
clf_direction.fit(X_train, y_dir_train)

# Regressor for duration
reg_duration = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
reg_duration.fit(X_train, y_dur_train)


# -----------------------
# 6. Evaluation
# -----------------------

# Direction classification
y_dir_pred = clf_direction.predict(X_test)
print("\n=== Classification Report: Next Green Direction ===")
print(classification_report(y_dir_test, y_dir_pred))

cm = confusion_matrix(y_dir_test, y_dir_pred, labels=["N", "E", "S", "W"])
cm_df = pd.DataFrame(cm,
                     index=["N_true", "E_true", "S_true", "W_true"],
                     columns=["N_pred", "E_pred", "S_pred", "W_pred"])
print("\nConfusion Matrix:")
print(cm_df)

# Duration regression
y_dur_pred = reg_duration.predict(X_test)
mae = mean_absolute_error(y_dur_test, y_dur_pred)
print("\n=== Regression: Green Duration (sec) ===")
print(f"Mean Absolute Error: {mae:.2f} sec")

# Feature importance plots
fi_clf = pd.Series(clf_direction.feature_importances_, index=feature_cols).sort_values(ascending=False)
fi_reg = pd.Series(reg_duration.feature_importances_, index=feature_cols).sort_values(ascending=False)

plt.figure()
fi_clf.plot(kind="bar")
plt.title("Feature Importance (Direction Classifier)")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
fi_reg.plot(kind="bar")
plt.title("Feature Importance (Duration Regressor)")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -----------------------
# 7. Smart Traffic Controller Class
# -----------------------

class SmartTrafficController:
    """
    Wraps the trained models into a simple interface.
    Given traffic state, it decides:
        - which direction to give green signal
        - for how many seconds
    """
    def __init__(self, direction_model, duration_model,
                 min_green=10.0, max_green=90.0):
        self.direction_model = direction_model
        self.duration_model = duration_model
        self.min_green = min_green
        self.max_green = max_green

    def decide_signal(self, north_count, east_count, south_count, west_count,
                      time_of_day, is_weekend):
        x = pd.DataFrame([{
            "north_count": north_count,
            "east_count": east_count,
            "south_count": south_count,
            "west_count": west_count,
            "time_of_day": time_of_day,
            "is_weekend": is_weekend
        }])

        direction = self.direction_model.predict(x)[0]
        duration  = float(self.duration_model.predict(x)[0])

        duration = max(self.min_green, min(self.max_green, duration))
        return direction, duration


controller = SmartTrafficController(clf_direction, reg_duration)


# -----------------------
# 8. Simple Demo Scenarios
# -----------------------

def simulate_controller_step(controller, north, east, south, west, hour, weekend):
    direction, duration = controller.decide_signal(
        north_count=north,
        east_count=east,
        south_count=south,
        west_count=west,
        time_of_day=hour,
        is_weekend=weekend
    )
    print(f"State: N={north}, E={east}, S={south}, W={west}, hour={hour}, weekend={weekend}")
    print(f"Decision: GREEN to {direction} for ~{duration:.1f} seconds\n")


print("\n=== Demo Decisions ===")
simulate_controller_step(controller, 40, 10, 5, 2, hour=9,  weekend=0)
simulate_controller_step(controller, 10, 30, 25, 5, hour=18, weekend=0)
simulate_controller_step(controller, 5, 5, 5, 5,  hour=2,  weekend=1)
simulate_controller_step(controller, 0, 45, 40, 10, hour=20, weekend=0)


# -----------------------
# 9. Multi-Step Intersection Simulation
# -----------------------

def simulate_intersection(controller, steps=15, time_of_day=18, is_weekend=0,
                          random_state=42, initial_state=None):
    """
    Very simplified intersection simulation:

    - At each step:
        * Use controller to pick green direction + duration
        * Reduce queue in that direction based on capacity
        * Add random new arrivals to all directions

    This is NOT real traffic physics, just a demo to show how AI decisions evolve.
    """
    rng = np.random.default_rng(random_state)

    if initial_state is None:
        state = {
            "north": rng.integers(0, 30),
            "east": rng.integers(0, 30),
            "south": rng.integers(0, 30),
            "west": rng.integers(0, 30),
        }
    else:
        state = initial_state.copy()

    history = []

    for step in range(steps):
        n, e, s, w = state["north"], state["east"], state["south"], state["west"]

        direction, duration = controller.decide_signal(
            north_count=n,
            east_count=e,
            south_count=s,
            west_count=w,
            time_of_day=time_of_day,
            is_weekend=is_weekend
        )

        # Assume 1 car passes every 2 seconds on the green lane
        capacity = int(duration / 2)

        if direction == "N":
            passed = min(capacity, state["north"])
            state["north"] -= passed
        elif direction == "E":
            passed = min(capacity, state["east"])
            state["east"] -= passed
        elif direction == "S":
            passed = min(capacity, state["south"])
            state["south"] -= passed
        elif direction == "W":
            passed = min(capacity, state["west"])
            state["west"] -= passed
        else:
            passed = 0

        # Random arrivals on every lane (0–10)
        state["north"] += rng.integers(0, 11)
        state["east"]  += rng.integers(0, 11)
        state["south"] += rng.integers(0, 11)
        state["west"]  += rng.integers(0, 11)

        total = state["north"] + state["east"] + state["south"] + state["west"]

        history.append({
            "step": step,
            "north": state["north"],
            "east": state["east"],
            "south": state["south"],
            "west": state["west"],
            "green_to": direction,
            "green_duration": duration,
            "total_vehicles": total,
            "cars_passed_this_step": passed
        })

    return pd.DataFrame(history)


sim_df = simulate_intersection(controller, steps=15, time_of_day=18, is_weekend=0)
print("\nSimulation (head):")
print(sim_df.head())

plt.figure()
plt.plot(sim_df["step"], sim_df["total_vehicles"], marker="o")
plt.title("Total Vehicles Over Simulation Steps")
plt.xlabel("Step")
plt.ylabel("Total Vehicles")
plt.grid(True)
plt.show()


# -----------------------
# 10. Save & Load Models
# -----------------------

joblib.dump(clf_direction, "direction_model.pkl")
joblib.dump(reg_duration, "duration_model.pkl")
print("\nModels saved: direction_model.pkl, duration_model.pkl")

# Example: load and use again
loaded_clf = joblib.load("direction_model.pkl")
loaded_reg = joblib.load("duration_model.pkl")
loaded_controller = SmartTrafficController(loaded_clf, loaded_reg)

test_dir, test_dur = loaded_controller.decide_signal(
    north_count=15, east_count=25, south_count=5, west_count=0,
    time_of_day=8, is_weekend=0
)
print("\nLoaded Controller Test:")
print(f"Next green: {test_dir}, Duration: {test_dur:.1f} sec")



# ============================================================
# AI-Based Smart Traffic Control System - Submission Version
# ============================================================

# 1. Imports
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error

import joblib


# 2. Synthetic Data Generation
def generate_synthetic_traffic_data(n_samples=8000, random_state=42):
    """
    Generate synthetic traffic data for a 4-way intersection.

    Features:
        - north_count, east_count, south_count, west_count
        - time_of_day (0-23)
        - is_weekend (0/1)

    Targets:
        - next_green_direction: 'N','E','S','W'
        - green_duration_sec: float
    """
    rng = np.random.default_rng(random_state)

    north = rng.integers(0, 51, size=n_samples)
    east  = rng.integers(0, 51, size=n_samples)
    south = rng.integers(0, 51, size=n_samples)
    west  = rng.integers(0, 51, size=n_samples)

    time_of_day = rng.integers(0, 24, size=n_samples)
    is_weekend  = rng.integers(0, 2, size=n_samples)

    counts = np.vstack([north, east, south, west]).T
    directions = np.array(["N", "E", "S", "W"])

    # Choose lane with max queue + 10% randomness
    max_idx = counts.argmax(axis=1)
    noise = rng.random(n_samples)
    noisy_idx = max_idx.copy()
    switch_mask = noise < 0.10
    random_alt = rng.integers(0, 4, size=n_samples)
    noisy_idx[switch_mask] = random_alt[switch_mask]
    next_green_direction = directions[noisy_idx]

    # Green duration based on total vehicles
    total_cars = counts.sum(axis=1)
    green_duration = 10 + 0.8 * total_cars
    green_duration = np.clip(green_duration, 10, 90)

    # Add noise
    green_duration = green_duration + rng.normal(0, 5, size=n_samples)
    green_duration = np.clip(green_duration, 10, 90)

    df = pd.DataFrame({
        "north_count": north,
        "east_count": east,
        "south_count": south,
        "west_count": west,
        "time_of_day": time_of_day,
        "is_weekend": is_weekend,
        "next_green_direction": next_green_direction,
        "green_duration_sec": green_duration
    })
    return df


# 3. Smart Traffic Controller Class
class SmartTrafficController:
    """
    Wraps the trained models:
        - direction_model: predicts which lane gets green ('N','E','S','W')
        - duration_model: predicts green time (seconds)
    """
    def __init__(self, direction_model, duration_model,
                 min_green=10.0, max_green=90.0):
        self.direction_model = direction_model
        self.duration_model = duration_model
        self.min_green = min_green
        self.max_green = max_green

    def decide_signal(self, north_count, east_count, south_count, west_count,
                      time_of_day, is_weekend):
        x = pd.DataFrame([{
            "north_count": north_count,
            "east_count": east_count,
            "south_count": south_count,
            "west_count": west_count,
            "time_of_day": time_of_day,
            "is_weekend": is_weekend
        }])

        direction = self.direction_model.predict(x)[0]
        duration  = float(self.duration_model.predict(x)[0])

        duration = max(self.min_green, min(self.max_green, duration))
        return direction, duration


# 4. Main Training + Evaluation Pipeline
def train_and_evaluate(random_state=42):
    # Generate data
    df = generate_synthetic_traffic_data(n_samples=8000, random_state=random_state)

    feature_cols = ["north_count", "east_count", "south_count", "west_count",
                    "time_of_day", "is_weekend"]
    X = df[feature_cols]
    y_dir = df["next_green_direction"]
    y_dur = df["green_duration_sec"]

    X_train, X_test, y_dir_train, y_dir_test, y_dur_train, y_dur_test = train_test_split(
        X, y_dir, y_dur, test_size=0.2, random_state=random_state
    )

    # Classifier (next direction)
    clf_direction = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=random_state
    )
    clf_direction.fit(X_train, y_dir_train)

    # Regressor (green duration)
    reg_duration = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=random_state
    )
    reg_duration.fit(X_train, y_dur_train)

    # Evaluation
    y_dir_pred = clf_direction.predict(X_test)
    print("=== Classification Report: Next Green Direction ===")
    print(classification_report(y_dir_test, y_dir_pred))

    cm = confusion_matrix(y_dir_test, y_dir_pred, labels=["N", "E", "S", "W"])
    cm_df = pd.DataFrame(cm,
                         index=["N_true", "E_true", "S_true", "W_true"],
                         columns=["N_pred", "E_pred", "S_pred", "W_pred"])
    print("\nConfusion Matrix:")
    print(cm_df)

    y_dur_pred = reg_duration.predict(X_test)
    mae = mean_absolute_error(y_dur_test, y_dur_pred)
    print("\n=== Regression: Green Duration (sec) ===")
    print(f"Mean Absolute Error: {mae:.2f} seconds")

    # Build controller object
    controller = SmartTrafficController(clf_direction, reg_duration)

    return df, clf_direction, reg_duration, controller


# 5. Simple Demo with the Trained Controller
def demo_controller(controller):
    scenarios = [
        {"north": 40, "east": 10, "south": 5,  "west": 2,  "hour": 9,  "weekend": 0},
        {"north": 10, "east": 30, "south": 25, "west": 5,  "hour": 18, "weekend": 0},
        {"north": 5,  "east": 5,  "south": 5,  "west": 5,  "hour": 2,  "weekend": 1},
        {"north": 0,  "east": 45, "south": 40, "west": 10, "hour": 20, "weekend": 0},
    ]

    print("\n=== Demo: Controller Decisions on Sample Scenarios ===")
    for i, sc in enumerate(scenarios, start=1):
        direction, duration = controller.decide_signal(
            north_count=sc["north"],
            east_count=sc["east"],
            south_count=sc["south"],
            west_count=sc["west"],
            time_of_day=sc["hour"],
            is_weekend=sc["weekend"]
        )
        print(f"Scenario {i}: N={sc['north']}, E={sc['east']}, S={sc['south']}, W={sc['west']}, "
              f"hour={sc['hour']}, weekend={sc['weekend']}")
        print(f"  -> GREEN to {direction} for ~{duration:.1f} seconds\n")


# 6. (Optional) Simple Multi-Step Simulation
def simulate_intersection(controller, steps=10, time_of_day=18, is_weekend=0,
                          random_state=42):
    """
    Simple intersection simulation:
        - Each step, controller selects lane + green time
        - Some cars pass, new cars arrive at random
    """
    rng = np.random.default_rng(random_state)

    state = {
        "north": rng.integers(0, 30),
        "east":  rng.integers(0, 30),
        "south": rng.integers(0, 30),
        "west":  rng.integers(0, 30),
    }

    history = []

    for step in range(steps):
        n, e, s, w = state["north"], state["east"], state["south"], state["west"]

        direction, duration = controller.decide_signal(
            north_count=n,
            east_count=e,
            south_count=s,
            west_count=w,
            time_of_day=time_of_day,
            is_weekend=is_weekend
        )

        capacity = int(duration / 2)  # 1 car per 2 seconds
        if direction == "N":
            passed = min(capacity, state["north"])
            state["north"] -= passed
        elif direction == "E":
            passed = min(capacity, state["east"])
            state["east"] -= passed
        elif direction == "S":
            passed = min(capacity, state["south"])
            state["south"] -= passed
        elif direction == "W":
            passed = min(capacity, state["west"])
            state["west"] -= passed
        else:
            passed = 0

        # New arrivals
        state["north"] += rng.integers(0, 11)
        state["east"]  += rng.integers(0, 11)
        state["south"] += rng.integers(0, 11)
        state["west"]  += rng.integers(0, 11)

        total = state["north"] + state["east"] + state["south"] + state["west"]

        history.append({
            "step": step,
            "north": state["north"],
            "east": state["east"],
            "south": state["south"],
            "west": state["west"],
            "green_to": direction,
            "green_duration": duration,
            "total_vehicles": total,
            "cars_passed": passed
        })

    return pd.DataFrame(history)


# 7. Save Models for "Deployment" Style
def save_models(direction_model, duration_model,
                dir_path="direction_model.pkl", dur_path="duration_model.pkl"):
    joblib.dump(direction_model, dir_path)
    joblib.dump(duration_model, dur_path)
    print(f"Models saved to: {dir_path}, {dur_path}")


# 8. Main Execution Block (run everything)
if __name__ == "__main__":
    # Train + evaluate
    df, clf_direction, reg_duration, controller = train_and_evaluate()

    # Demo on a few scenarios
    demo_controller(controller)

    # Run a simple simulation and show first few rows
    sim_df = simulate_intersection(controller, steps=10, time_of_day=18, is_weekend=0)
    print("Simulation result (head):")
    print(sim_df.head())

    # Save models
    save_models(clf_direction, reg_duration)


