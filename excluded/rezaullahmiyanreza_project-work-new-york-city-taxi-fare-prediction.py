import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


 #load Data
# Load first 1 million rows to keep memory usage low
df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows=1_000_000,
                 parse_dates=['pickup_datetime'])


# ðŸ“… Extract day of the week (0=Monday, 6=Sunday)
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['day_name'] = df['pickup_datetime'].dt.day_name()


# ðŸ‘¤ Step 4: Hypothesis 1 - Fares vary by day of week
day_avg_fare = df.groupby('day_name')['fare_amount'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])


# ðŸ“Š Ploting for Hypothesis 1
plt.figure(figsize=(8,5))
sns.barplot(x=day_avg_fare.index, y=day_avg_fare.values, palette="viridis")
plt.title("Hypothesis 1: Average Fare by Day of the Week")
plt.ylabel("Average Fare ($)")
plt.xlabel("Day of Week")
plt.xticks(rotation=55)
plt.grid(True)
plt.show()


# ðŸ‘¥ Step 5: Hypothesis 2 - More passengers = higher fare?
passenger_avg_fare = df.groupby('passenger_count')['fare_amount'].mean()


# ðŸ“Š Plot for Hypothesis 2
plt.figure(figsize=(8, 6))
sns.barplot(x=passenger_avg_fare.index, y=passenger_avg_fare.values, palette="coolwarm")
plt.title("Hypothesis 2: Average Fare by Number of Passengers")
plt.xlabel("Passenger Count")
plt.ylabel("Average Fare ($)")
plt.grid(True)
plt.show()


# âœ… Hypothesis 3: Average Fare by Distance

from numpy import sqrt

# Approximate Euclidean distance (not actual road distance)
df['distance'] = ((df['pickup_longitude'] - df['dropoff_longitude'])**2 + 
                  (df['pickup_latitude'] - df['dropoff_latitude'])**2) ** 0.5

# Remove unrealistic distance outliers
df = df[df['distance'] < 0.5]

# Bin the distances
df['distance_bin'] = pd.cut(df['distance'], 
                            bins=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 
                            labels=["<0.01", "0.01â€“0.02", "0.02â€“0.05", "0.05â€“0.1", 
                                    "0.1â€“0.2", "0.2â€“0.3", "0.3â€“0.4", "0.4â€“0.5"])

# Group by distance bin
distance_fare = df.groupby('distance_bin', observed=True)['fare_amount'].mean()
df = df.dropna(subset=['distance_bin'])


# âœ… Plot
plt.figure(figsize=(10,6))
sns.barplot(x=distance_fare.index, y=distance_fare.values, palette="crest")
plt.title("Hypothesis 3: Average Fare by Distance Range")
plt.xlabel("Distance Bin (Euclidean Degrees)")
plt.ylabel("Average Fare ($)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# âœ… Optional: Show values
print("Average Fare by Distance Range:\n", distance_fare)


#Print mean fares
print("Average Fare by Day of the Week:\n", day_avg_fare)
print("\nAverage Fare by Passenger Count:\n", passenger_avg_fare)

