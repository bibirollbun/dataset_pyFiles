# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

session = {}


    base_path = "/kaggle/input/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/"
    
    daily_activity = pd.read_csv(base_path + "dailyActivity_merged.csv")
    sleep = pd.read_csv(base_path + "sleepDay_merged.csv", on_bad_lines='skip')
    heartrate = pd.read_csv(base_path + "heartrate_seconds_merged.csv")
    weight = pd.read_csv(base_path + "weightLogInfo_merged.csv")
    
    session = {
        'daily_activity_raw': daily_activity,
        'sleep_raw': sleep,
        'heartrate_raw': heartrate,
        'weight_raw': weight
    }


def calculate_sleep_efficiency(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=['TotalMinutesAsleep', 'TotalTimeInBed'])
    df['SleepEfficiency'] = df['TotalMinutesAsleep'] / df['TotalTimeInBed']
    return df

def aggregate_activity(df):
    df = df.drop_duplicates()
    agg = df.groupby('Id').agg({
        'TotalSteps': 'sum',
        'Calories': 'sum',
        'TotalDistance': 'sum'
    }).reset_index()
    return agg

def process_health(hr_df, weight_df):
    hr_df = hr_df.drop_duplicates()
    weight_df = weight_df.drop_duplicates()
    hr_avg = hr_df.groupby('Id')['Value'].mean().reset_index()
    weight_avg = weight_df.groupby('Id')['WeightKg'].mean().reset_index()
    health_df = hr_avg.merge(weight_avg, on='Id', how='outer')
    return health_df



def cleaning_agent():
    print("[Cleaning Agent] Started")
    
    # Clean daily activity
    df = session['daily_activity_raw']
    df = df.drop_duplicates()
    df = df.dropna(subset=['TotalSteps', 'Calories', 'TotalDistance'])
    session['daily_activity_raw'] = df
    
    # Clean sleep
    df = session['sleep_raw']
    df = df.drop_duplicates()
    df = df.dropna(subset=['TotalMinutesAsleep', 'TotalTimeInBed'])
    session['sleep_raw'] = df
    
    # Clean heart rate
    df = session['heartrate_raw']
    df = df.drop_duplicates()
    session['heartrate_raw'] = df
    
    # Clean weight
    df = session['weight_raw']
    df = df.drop_duplicates()
    session['weight_raw'] = df
    
    print("[Cleaning Agent] Completed")



# Sleep Agent (parallel-style processing)
def sleep_agent():
    print("[Sleep Agent] Started")
    session["sleep_df"] = calculate_sleep_efficiency(session["sleep_raw"])
    print(f"[Sleep Agent] Completed - rows: {len(session['sleep_df'])}")

# Activity Agent (sequential-style processing)
def activity_agent():
    print("[Activity Agent] Started")
    time.sleep(1)  # simulate compute delay
    session["activity_df"] = aggregate_activity(session["daily_activity_raw"])
    print(f"[Activity Agent] Completed - rows: {len(session['activity_df'])}")

# Health Agent (chunk / loop processing)
def health_agent(chunk_size=100000):
    print("[Health Agent] Started")

    hr_df = session["heartrate_raw"]
    weight_df = session["weight_raw"]

    chunks = [hr_df[i:i+chunk_size] for i in range(0, len(hr_df), chunk_size)]
    results = []

    for i, chunk in enumerate(chunks):
        print(f"[Health Agent] Processing chunk {i+1}/{len(chunks)}")
        results.append(process_health(chunk, weight_df))

    session["health_df"] = pd.concat(results, ignore_index=True)
    print(f"[Health Agent] Completed - rows: {len(session['health_df'])}")

sleep_agent()
activity_agent()
health_agent()


# --- Sleep Agent ---
def sleep_agent():
    print("[Sleep Agent] Started")
    df = session["sleep_raw"].drop_duplicates().dropna(subset=['TotalMinutesAsleep', 'TotalTimeInBed'])
    df['SleepEfficiency'] = df['TotalMinutesAsleep'] / df['TotalTimeInBed']
    session["sleep_df"] = df
    print(f"[Sleep Agent] Completed - rows: {len(df)}")

# --- Activity Agent ---
def activity_agent():
    print("[Activity Agent] Started")
    df = session["daily_activity_raw"].drop_duplicates()
    session["activity_df"] = df.groupby('Id', as_index=False)[['TotalSteps','Calories','TotalDistance']].sum()
    print(f"[Activity Agent] Completed - rows: {len(session['activity_df'])}")

# --- Health Agent ---
def health_agent():
    print("[Health Agent] Started")
    hr_df = session["heartrate_raw"].drop_duplicates()
    weight_df = session["weight_raw"].drop_duplicates()
    hr_avg = hr_df.groupby('Id')['Value'].mean().reset_index()
    weight_avg = weight_df.groupby('Id')['WeightKg'].mean().reset_index()
    session["health_df"] = hr_avg.merge(weight_avg, on='Id', how='outer')
    print(f"[Health Agent] Completed - rows: {len(session['health_df'])}")

# -------------------------------
# Call agents
# -------------------------------
sleep_agent()
activity_agent()
health_agent()




cleaning_agent()


from concurrent.futures import ThreadPoolExecutor

# Run sleep and activity in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(sleep_agent)
    executor.submit(activity_agent)

# Run health agent sequentially
health_agent()

print("All agents executed.")



summary_df = sleep_df.merge(activity_df, on='Id', how='outer').merge(health_df, on='Id', how='outer')


# Start from your merged summary_df
clean_summary_df = summary_df.copy()

# Handle missing values
# Fill missing numeric values with median (robust to outliers)
for column in numeric_cols:
    median_value = clean_summary_df[column].median()
    clean_summary_df[column] = clean_summary_df[column].fillna(median_value)

# Check the cleaned dataframe
print(clean_summary_df.head())


# Observability: check shapes and sample rows after merging
print({k: v.shape for k,v in session.items() if isinstance(v, pd.DataFrame)})



summary_df = summary_df.drop_duplicates(subset=['Id'])


# Example Insights

# 1. Average sleep efficiency
avg_sleep_efficiency = summary_df['SleepEfficiency'].mean()
print(f"Average Sleep Efficiency: {avg_sleep_efficiency:.2f}")

# 2. Top 5 users by total steps
top_steps = summary_df[['Id', 'TotalSteps']].sort_values(by='TotalSteps', ascending=False).head()
print("\nTop 5 Users by Total Steps:")
print(top_steps)

# 3. Correlation between activity and calories burned
corr = summary_df[['TotalSteps', 'Calories']].corr().iloc[0,1]
print(f"\nCorrelation between Total Steps and Calories Burned: {corr:.2f}")

# 4. Average heart rate and weight
avg_hr = summary_df['Value'].mean()
avg_weight = summary_df['WeightKg'].mean()
print(f"\nAverage Heart Rate: {avg_hr:.2f} bpm")
print(f"Average Weight: {avg_weight:.2f} kg")


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of Sleep Efficiency
plt.figure(figsize=(8,5))
sns.histplot(summary_df['SleepEfficiency'], bins=20, kde=True)
plt.title('Distribution of Sleep Efficiency')
plt.xlabel('Sleep Efficiency')
plt.ylabel('Count')
plt.show()

# Total Steps vs Calories
plt.figure(figsize=(8,5))
sns.scatterplot(data=summary_df, x='TotalSteps', y='Calories')
plt.title('Total Steps vs Calories Burned')
plt.xlabel('Total Steps')
plt.ylabel('Calories')
plt.show()

# Heart Rate vs Weight
plt.figure(figsize=(8,5))
sns.scatterplot(data=summary_df, x='WeightKg', y='Value')
plt.title('Heart Rate vs Weight')
plt.xlabel('Weight (kg)')
plt.ylabel('Average Heart Rate (bpm)')
plt.show()



# Save summary_df to CSV for report
summary_df.to_csv("fitbit_summary_report.csv", index=False)
print("Summary report saved as 'fitbit_summary_report.csv'")

# Remove duplicate users before ranking
summary_df_unique = summary_df.drop_duplicates(subset=["Id"])

# Highlight top users
# Sort by TotalSteps descending
top_users = summary_df.sort_values(by='TotalSteps', ascending=False).head(10)

# Replace NaN in SleepEfficiency for display
top_users_display = top_users.copy()
top_users_display['SleepEfficiency'] = top_users_display['SleepEfficiency'].fillna(
    top_users_display['SleepEfficiency'].median()
)

# Print top 10 active users
print("\nTop 10 Active Users:")
print(top_users_display[['Id','TotalSteps','Calories','SleepEfficiency']])


# Recreate session dictionary
base_path = "/kaggle/input/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/"

daily_activity = pd.read_csv(base_path + "dailyActivity_merged.csv")
sleep = pd.read_csv(base_path + "sleepDay_merged.csv", on_bad_lines='skip')
heartrate = pd.read_csv(base_path + "heartrate_seconds_merged.csv")
weight = pd.read_csv(base_path + "weightLogInfo_merged.csv")

session = {
    'daily_activity_raw': daily_activity,
    'sleep_raw': sleep,
    'heartrate_raw': heartrate,
    'weight_raw': weight
}

# Check all DataFrames in session
for name, df in session.items():
    if isinstance(df, pd.DataFrame):
        print(f"{name}: {df.shape}")


