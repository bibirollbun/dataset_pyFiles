import numpy as np
import pandas as pd
import json
import os
import glob
import dask.dataframe as dd


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Read and display a few lines to understand the structure
file_path  = "/kaggle/input/otto-recommender-system/train.jsonl"
# with open(file_path, "r", encoding="utf-8") as f:
#     for _ in range(5):  # Read first 5 lines
#         print(json.loads(f.readline()))


# Function to process the file in chunks
def process_jsonl_in_chunks(file_path, chunk_size=100000):
    """Reads large JSONL file efficiently in chunks and converts it to a DataFrame."""
    data = []
    batch = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)  # Load each JSON object

            # Extract session ID and expand the events
            session_id = record["session"]
            for event in record["events"]:
                event["session"] = session_id  # Add session ID to each event
                data.append(event)

            # Process in chunks
            if len(data) >= chunk_size:
                df = pd.DataFrame(data)
                df.to_parquet(f"output_chunk_{batch}.parquet", index=False)  # Save to Parquet
                print(f"Processed {batch * chunk_size} rows...")
                data = []  # Reset list
                batch += 1

    # Process the remaining data
    if data:
        df = pd.DataFrame(data)
        df.to_parquet(f"output_chunk_{batch}.parquet", index=False)
        print(f"Final batch processed: {batch * chunk_size} rows.")

    print("Processing complete. Data saved as Parquet files.")



# Run processing function
#process_jsonl_in_chunks(file_path)


# files = glob.glob("output_chunk_*.parquet")
# df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
# df.to_parquet("final_output.parquet", index=False)


# Load Parquet file
file_path = "/kaggle/input/otto-parquet-full-training-file/final_output.parquet"
df = dd.read_parquet(file_path)


# Path to the test.jsonl file
file_path = "/kaggle/input/otto-recommender-system/test.jsonl"

# Step 1: Read the JSONL file directly
with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Step 2: Expand the nested 'events' column
flattened_data = []
for record in data:
    session_id = record["session"]
    for event in record["events"]:
        event["session"] = session_id  # Add session ID to each event
        flattened_data.append(event)

# Step 3: Convert to DataFrame
test_df = pd.DataFrame(flattened_data)




test_df.shape


test_df.head()


df.head(5)


# Convert Unix timestamp to datetime (milliseconds assumed)
df["datetime"] = dd.to_datetime(df["ts"], unit="ms")


# Convert Unix timestamp to datetime (milliseconds assumed)
test_df["datetime"] = pd.to_datetime(test_df["ts"], unit="ms")


nb_unique_sessions = df['session'].nunique().compute()
print(f"The number of unique sessions: {nb_unique_sessions}")
print(f"The number of total events: {df.shape[0].compute()}")
print(f"Earliest Timestamp in the test set: {df['datetime'].min().compute()}")
print(f"Latest Timestamp in the test set: {df['datetime'].max().compute()}")
print(f"The number of unique article ids (product codes): {df['aid'].nunique().compute()}")

# Check for missing values (NaNs) in each column
missing_values = df.isnull().sum().compute()
print('Check for missing values')
print('-------------------------------------')
missing_values


# Count occurrences of each event type
event_counts = df["type"].value_counts().compute().reset_index()
event_counts.columns = ["Event Type", "Count"]

# Calculate percentage
event_counts["Percentage %"] = (event_counts["Count"] / event_counts["Count"].sum()) * 100


event_counts.style.set_table_styles(
    [
        {"selector": "th", "props": [("background-color", "#016FD0"), ("color", "white"), ("font-weight", "bold")]}
    ]
).hide(axis="index").background_gradient(subset=["Percentage %"], cmap="Blues")  # Blue-Green gradient


nb_unique_sessions_test = test_df['session'].nunique()
print(f"The number of unique sessions: {nb_unique_sessions_test}")
print(f"The number of total events: {test_df.shape[0]}")
print(f"Earliest Timestamp in the training set: {test_df['datetime'].min()}")
print(f"Latest Timestamp in the training set: {test_df['datetime'].max()}")
print(f"The number of unique article ids (product codes): {test_df['aid'].nunique()}")


# 1. Overlap in Session IDs
train_sessions = set(df["session"].unique().compute())
test_sessions = set(test_df["session"].unique())

session_overlap = train_sessions.intersection(test_sessions)
num_session_overlap = len(session_overlap)

# 2. Overlap in Article IDs (AIDs)
train_aids = set(df["aid"].unique().compute())
test_aids = set(test_df["aid"].unique())

aid_overlap = train_aids.intersection(test_aids)
num_aid_overlap = len(aid_overlap)

# 3. Display Results
print(f"Total overlapping session IDs: {num_session_overlap}")
print(f"Total overlapping article IDs (AIDs): {num_aid_overlap}")

# Percentage overlap for context
session_overlap_percentage = (num_session_overlap / len(test_sessions)) * 100
aid_overlap_percentage = (num_aid_overlap / len(test_aids)) * 100

print(f"Session ID Overlap Percentage: {session_overlap_percentage:.2f}%")
print(f"AID Overlap Percentage: {aid_overlap_percentage:.2f}%")



# Step 1: Group by session and count the number of events per session
session_lengths = df.groupby("session").size().compute().reset_index(name="num_events")

# Step 2: Display the distribution of session lengths
# Basic descriptive statistics
session_length_stats = session_lengths["num_events"].describe()

session_length_stats


session_length_stats.iloc[1]


session_length_stats.iloc[7]


# Define event types to keep
event_types_to_keep = ["carts", "orders"]

# Step 1: Identify sessions that contain "cart" or "order"
session_filter = df[df["type"].isin(event_types_to_keep)]["session"].unique().compute()

# Step 2: Filter the original DataFrame to keep only these sessions
filtered_df = df[df["session"].isin(session_filter)]

# Define event type mappings
filtered_df["is_click"] = (filtered_df["type"] == "clicks").astype("int8")
filtered_df["is_cart"] = (filtered_df["type"] == "carts").astype("int8")
filtered_df["is_order"] = (filtered_df["type"] == "orders").astype("int8")


filtered_df.head()


num_rows = filtered_df.shape[0].compute()  # Compute row count
num_columns = len(filtered_df.columns)  # Column count

print(f"Shape of DataFrame: ({num_rows}, {num_columns})")
print(f"The number of unique sessions that have at least one cart or order: {filtered_df['session'].nunique().compute()}")


# Step 1: Group by session and sum up the click, cart, and order counts

session_event_counts = (
    filtered_df.groupby("session")
    .agg(
        num_events=("is_click", "size"),         # Count total events per session
        total_clicks=("is_click", "sum"),         # Sum of clicks
        total_carts=("is_cart", "sum"),           # Sum of carts
        total_orders=("is_order", "sum")          # Sum of orders
    )
    .compute()
    .reset_index()
)

# Rename columns for clarity
session_event_counts.columns = ["session_id", "num_events", "num_clicks", "num_carts", "num_orders"]


session_event_counts.head()


session_event_counts["num_events"].describe()


total_sessions = session_event_counts.shape[0]
clicks_carts = session_event_counts[(session_event_counts['num_carts'] > 0) & (session_event_counts['num_orders'] == 0) ].shape[0]
clicks_carts_orders = session_event_counts[(session_event_counts['num_carts'] > 0) & (session_event_counts['num_orders'] > 0) ].shape[0]
clicks_ordrers = session_event_counts[(session_event_counts['num_orders'] > 0) & (session_event_counts['num_carts'] == 0) ].shape[0]

print(f"Number of unique sessions that contain at least one extra event except for clicks: {total_sessions} - ~{round((total_sessions / nb_unique_sessions) * 100)}% of the total number of unique sessions in the dataset.")

print("-------------------------------------------------------------------------------------------------------------------------------------")
print("The percentages below are calculated based on the number of unique sessions that contain at least one extra event except for clicks!")
print(f"Number of unique sessions that contain clicks and carts events: {clicks_carts} - ~{round((clicks_carts / total_sessions) * 100)}%")
print(f"Number of unique sessions that contain clicks, carts and orders events: {clicks_carts_orders} - ~{round((clicks_carts_orders / total_sessions) * 100)}%")
print(f"Number of unique sessions that contain clicks and orders events: {clicks_ordrers} - ~{round((clicks_ordrers / total_sessions) * 100)}%")


only_clicks = df[~df['session'].isin(session_filter)]['session'].nunique().compute()
print(f"Number of unique sessions that contain ONLY clicks events: {only_clicks} ~{round((only_clicks / nb_unique_sessions) * 100)}% of the total number of unique sessions in the dataset.")


only_clicks_df = df[~df['session'].isin(session_filter)]
only_clicks_stats = (
    only_clicks_df.groupby("session")
    .agg(
        num_events=("session", "size"),         # Count total events per session
    )
    .compute()
    .reset_index()
)


only_clicks_stats["num_events"].describe()


# Extract time-based features
df["hour"] = df["datetime"].dt.hour

# Extract the weekday name
df["weekday"] = df["datetime"].dt.strftime("%A")  # Full weekday name (e.g., "Monday")

# Extract day of the week as a number (0 = Monday, 6 = Sunday)
df["weekday_number"] = df["datetime"].dt.weekday

# Extract the month name
df["month"] = df["datetime"].dt.month # Full month name (e.g., "August")


# Compute distribution of timestamps (e.g., peak activity hours)
hourly_activity = df["hour"].value_counts().compute().reset_index()
hourly_activity.columns = ["Hour", "Event Count"]
hourly_activity["Percentage %"] = (hourly_activity["Event Count"] / hourly_activity["Event Count"].sum()) * 100
hourly_activity = hourly_activity.sort_values("Hour", ascending=True)


hourly_activity.style.set_table_styles(
    [
        {"selector": "th", "props": [("background-color", "#016FD0"), ("color", "white"), ("font-weight", "bold")]}
    ]
).hide(axis="index").background_gradient(subset=["Percentage %"], cmap="Blues")  # Blue-Green gradient


# Compute distribution of timestamps (e.g., peak activity hours)
daily_activity = df["weekday_number"].value_counts().compute().reset_index()
daily_activity.columns = ["Day", "Event Count"]
daily_activity["Percentage %"] = (daily_activity["Event Count"] / daily_activity["Event Count"].sum()) * 100
daily_activity = daily_activity.sort_values("Day", ascending=True)

# Mapping of day numbers to weekday names
day_mapping = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

# Assuming df is your DataFrame
daily_activity["Weekday"] = daily_activity["Day"].map(day_mapping)


daily_activity.style.set_table_styles(
    [
        {"selector": "th", "props": [("background-color", "#016FD0"), ("color", "white"), ("font-weight", "bold")]}
    ]
).hide(axis="index").background_gradient(subset=["Percentage %"], cmap="Blues")  # Blue-Green gradient


# Compute session start and end timestamps
session_times = df.groupby("session")["datetime"].agg(["min", "max"]).compute()
session_times["session_duration"] = (session_times["max"] - session_times["min"]).dt.total_seconds()
# Add session duration in minutes
session_times["session_duration_minutes"] = session_times["session_duration"] / 60
session_times["session_duration_hours"] = session_times["session_duration"] / 3600


session_times.head()


# Compute descriptive statistics
session_duration_stats = session_times["session_duration_hours"].describe().reset_index()
session_duration_stats.columns = ["statistic", "value"]
session_duration_stats


df["is_click"] = (df["type"] == "clicks").astype("int8")
df["is_cart"] = (df["type"] == "carts").astype("int8")
df["is_order"] = (df["type"] == "orders").astype("int8")


# Step 1: Count clicks, carts, and orders per product (aid)
product_event_counts = df.groupby("aid")[["is_click", "is_cart", "is_order"]].sum().compute().reset_index()

# Rename columns for clarity
product_event_counts.columns = ["aid", "num_clicks", "num_carts", "num_orders"]


# Compute total number of clicks, carts, and orders
total_clicks = product_event_counts["num_clicks"].sum()
total_carts = product_event_counts["num_carts"].sum()
total_orders = product_event_counts["num_orders"].sum()


# Step 2: Calculate Conversion Rate (orders / clicks)
product_event_counts["conversion_rate"] = product_event_counts["num_orders"] / product_event_counts["num_clicks"]
product_event_counts["conversion_rate"] = product_event_counts["conversion_rate"].fillna(0)  # Handle division by zero

# Add percentage columns
product_event_counts["click_percentage"] = (product_event_counts["num_clicks"] / total_clicks) * 100
product_event_counts["cart_percentage"] = (product_event_counts["num_carts"] / total_carts) * 100
product_event_counts["order_percentage"] = (product_event_counts["num_orders"] / total_orders) * 100


product_event_counts = product_event_counts.sort_values("conversion_rate", ascending=False)

product_event_counts.head(20).style.set_table_styles(
    [
        {"selector": "th", "props": [("background-color", "#016FD0"), ("color", "white"), ("font-weight", "bold")]}
    ]
).hide(axis="index").background_gradient(subset=["conversion_rate"], cmap="Blues")  # Blue-Green gradient


# Step 1: Sort products by the number of orders in descending order
product_event_counts = product_event_counts.sort_values("num_orders", ascending=False)
product_event_counts["cumulative_orders"] = product_event_counts["num_orders"].cumsum() / product_event_counts["num_orders"].sum()
product_event_counts


(product_event_counts[
product_event_counts['cumulative_orders'] <= 0.8
].shape[0] /  product_event_counts.shape[0]) * 100






