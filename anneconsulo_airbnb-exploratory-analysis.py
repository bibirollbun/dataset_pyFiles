# Core
import pandas as pd
import numpy as np

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Interactive + missingness)
!pip -q install plotly missingno duckdb
import plotly.express as px
import missingno as msno

# Impuration
from sklearn.impute import SimpleImputer, KNNImputer

# SQL (DuckDB)
import duckdb

# Display options
pd.options.display.float_format = "{:,.2f}".format


train_users = pd.read_csv("/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip")
sessions = pd.read_csv("/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip")

age_gender = pd.read_csv("/kaggle/input/airbnb-recruiting-new-user-bookings/age_gender_bkts.csv.zip")
countries = pd.read_csv("/kaggle/input/airbnb-recruiting-new-user-bookings/countries.csv.zip")

train_users.shape, train_users.head(3)


train_users.info()
train_users.describe(include="all").T
train_users.isna().mean().sort_values(ascending=False).head(10)


# 4.1 Dates

for col in["date_account_created", "timestamp_first_active", "date_first_booking"]:
    if col in train_users.columns:
        train_users[col] = pd.to_datetime(train_users[col], errors="coerce")

# 4.2 Age sanity check
train_users["age"] = pd.to_numeric(train_users["age"], errors="coerce")
train_users.loc[(train_users["age"] < 15) | (train_users["age"] > 90), "age"] = np.nan

# 4.3 Booking flag
train_users["booked"] = (train_users["country_destination"].fillna("NDF") != "NDF").astype(int)


msno.bar(train_users[["age","gender","country_destination","booked"]], figsize=(8,4))

# Missing flag
train_users["age_missing"] = train_users["age"].isna().astype(int)

# Booking rate by missing vs present
booking_rate = (train_users.groupby("age_missing")["booked"].mean()
                   .rename({0:"Age present", 1:"Age missing"}))
booking_rate


train_users["age"].plot(kind="hist", bins=30, figsize=(7,4))
plt.title("Age Distribution (before imputation)")
plt.xlabel("Age"); plt.ylabel("Users"); plt.show()


g = (train_users.groupby("gender")["booked"].mean()
        .sort_values(ascending=False))
g.plot(kind="bar", figsize=(6,4))
plt.title("Booking Rate by Gender"); plt.ylabel("Booking Rate");plt.show()


bins = [0, 17, 25, 35, 45, 55, 65, 100]
labels = ["0-17", "18-25", "26-35", "36-45", "46-55", "56-65", "65+"]

train_users["age_group"] = pd.cut(train_users["age"], bins=bins, labels=labels, right=True)

booking_rate_age = (
    train_users.groupby("age_group")["booked"]
    .mean()
    .reset_index(name="booking_rate")
)

print(booking_rate_age)


train_users["booked"] = (train_users["country_destination"] != "NDF").astype(int)

booking_rate_age = (train_users.groupby("age_group")["booked"]
                    .mean()
                    .reset_index(name="booking_rate"))

plt.figure(figsize=(8,4))
sns.barplot(x="age_group", y="booking_rate", data=booking_rate_age, color="skyblue")
plt.title("Booking Rate by Age Group")
plt.ylabel("Booking Rate")
plt.xlabel("Age Group")
plt.xticks(rotation=45)
plt.show()


dest_share = (train_users["country_destination"]
             .value_counts(normalize=True)
             .rename_axis("country_destination")
             .reset_index(name="share"))
dest_share.head(10)


train_users["country_destination"].value_counts().head(10).plot(kind="bar", figsize=(8,4))
plt.title("Top Destination Countries (by users)"); plt.ylabel("Users"); plt.show()


# Group by Age and Destination
age_dest = (
    train_users.groupby(["age_group", "country_destination"])
    .size()
    .reset_index(name="count")
)
age_dest["percentage"] = age_dest.groupby("age_group")["count"].transform(lambda x: x / x.sum())

# Keep only top destinations for better visualization
top_destinations = train_users["country_destination"].value_counts().head(5).index
age_dest = age_dest[age_dest["country_destination"].isin(top_destinations)]

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=age_dest,
    x="age_group",
    y="percentage",
    hue="country_destination"
)
plt.title("Destination Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Proportion")
plt.legend(title="Destination")
plt.show()


duckdb.query("""
    SELECT country_destination, COUNT(*) AS users
    FROM train_users
    WHERE booked = 1
    GROUP BY 1
    ORDER BY users DESC
    LIMIT 10
""").to_df()


df_imp = train_users.copy()
for g, med in df_imp.groupby("gender")["age"].median().items():
    df_imp.loc[(df_imp["gender"]==g) & (df_imp["age"].isna()), "age"] = med
df_imp["age"].fillna(df_imp["age"].median(), inplace=True)


# Define age bands
bins = [0, 18, 25, 35, 50, 65, 100]
labels = ["0-17", "18-24", "25-34", "35-49", "50-64", "65+"]
train_users["age_band"] = pd.cut(train_users["age"], bins=bins, labels=labels)
df_imp["age_band"] = pd.cut(df_imp["age"], bins=bins, labels=labels)

# Booking rates before imputation
rate_before = (
    train_users.groupby("age_band")["booked"]
    .mean()
    .reset_index(name="booking_rate_before")
)

# Booking rates after imputation
rate_after = (
    df_imp.groupby("age_band")["booked"]
    .mean()
    .reset_index(name="booking_rate_after")
)

# Merge compration
comparison = pd.merge(rate_before, rate_after, on="age_band")
print(comparison)


results = pd.DataFrame({
    "age_band": ["0-17", "18-24", "25-34", "35-49", "50-64", "65+"],
    "booking_rate_before": [0.49, 0.51, 0.58, 0.53, 0.51, 0.51],
    "booking_rate_after":  [0.49, 0.51, 0.37, 0.53, 0.51, 0.51]
})

# Plot
x = np.arange(len(results["age_band"]))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width/2, results["booking_rate_before"], width, label="Before Imputation")
ax.bar(x + width/2, results["booking_rate_after"], width, label="After Imputation")

# Labels and styling
ax.set_title("Booking Rates by Age Band (Before vs After Imputation)")
ax.set_xlabel("Age Band")
ax.set_ylabel("Booking Rate")
ax.set_xticks(x)
ax.set_xticklabels(results["age_band"])
ax.legend()

plt.show()


# Gender x Age Band
cross = (df_imp.groupby(["gender", "age_band"], as_index=False)
        .agg(users=("booked", "size"),
            booking_rate=("booked", "mean")))


display(cross.sort_values("booking_rate", ascending=False))


plt.figure(figsize=(8,5))
sns.barplot(data=cross, x="age_band", y="booking_rate", hue="gender")
plt.title("Booking Rate by Gender x Age Band")
plt.ylabel("Booking Rate")
plt.show()


sessions.groupby("action")["secs_elapsed"].sum().sort_values(ascending=False).head(10)


avg_time = (
    sessions.groupby("action")["secs_elapsed"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

avg_time


# Merge session with bookings

sessions_merged = sessions.merge(
    train_users[["id", "booked"]], left_on="user_id", right_on="id", how="left"
)

# Compare top actions by booking outcome
booked_vs_not = (
    sessions_merged.groupby(["booked","action"])["secs_elapsed"]
    .sum()
    .reset_index()
    .sort_values(["booked", "secs_elapsed"], ascending=[True, False])
)

# Top 10 actions for booked users
booked_vs_not[booked_vs_not["booked"]==1].head(10)


# Build Conversion Funnel

funnel = pd.DataFrame({
    "stage": ["Signups", "Confirmed Email","Searched", "Booked"],
    "users": [
        train_users.shape[0],
        sessions[sessions["action"]== "confirm_email"]["user_id"].nunique(),
        sessions[sessions["action"] == "search_results"]["user_id"].nunique(),
        train_users["booked"].sum()
    ]
})

funnel.to_csv("export_funnel.csv", index=False)
cross.to_csv("export_cross.csv", index=False)
booked_vs_not.to_csv("export_booked_vs_not.csv", index=False)
avg_time.to_csv("export_avg_time.csv", index=False)

print("CSV exports completed: funnel, cross, booked_vs_not, avg_time")


# Count bookings by country and age band
booking_counts = train_users.groupby(["country_destination", "age_band"]).size().unstack(fill_value=0)

plt.figure(figsize=(12,6))
sns.heatmap(booking_counts, cmap="Blues", annot=True, fmt="d", cbar_kws={'label': 'Number of Bookings'})
plt.title("Booking Counts by Country & Age Band")
plt.ylabel("Country")
plt.xlabel("Age Band")
plt.show()


# Count sessions per user
sessions_per_user = sessions.groupby("user_id").size().reset_index(name="session_count")

# Merge with bookings
sessions_merged = sessions_per_user.merge(
    train_users[["id", "booked"]], 
    left_on="user_id", 
    right_on="id", 
    how="left"
)

plt.figure(figsize=(10,6))
sns.histplot(data=sessions_merged, x="session_count", hue="booked", bins=50, kde=False)
plt.xlim(0,100) 
plt.title("Distribution of Session Counts per User")
plt.xlabel("Number of Sessions (Actions)")
plt.ylabel("Number of Users")
plt.show()


# Ensure datetime format
train_users["date_account_created"] = pd.to_datetime(train_users["date_account_created"])
train_users["date_first_booking"] = pd.to_datetime(train_users["date_first_booking"], errors="coerce")

# Calculate time to book
train_users["days_to_book"] = (train_users["date_first_booking"] - train_users["date_account_created"]).dt.days

# Filter only booked users
time_to_book = train_users[train_users["booked"] == 1]["days_to_book"].dropna()

plt.figure(figsize=(8,5))
plt.hist(time_to_book, bins=50, color="steelblue", edgecolor="black")
plt.title("Distribution of Time to First Booking")
plt.xlabel("Days from Signup to Booking")
plt.ylabel("Number of Users")
plt.show()

