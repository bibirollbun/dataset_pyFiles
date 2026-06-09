import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
competitions_path = "/kaggle/input/meta-kaggle/Competitions.csv"
kernels_path = "/kaggle/input/meta-kaggle/Kernels.csv"
users_path = "/kaggle/input/meta-kaggle/Users.csv"

# Load datasets
competitions = pd.read_csv(competitions_path)
kernels = pd.read_csv(kernels_path)
users = pd.read_csv(users_path, usecols=[
    'Id', 'UserName', 'DisplayName', 'PerformanceTier', 'Country'
])

# Preview column names
print("Competition Columns:", competitions.columns.tolist())
print("Kernel Columns:", kernels.columns.tolist())
print("User Columns:", users.columns.tolist())

# Convert date columns safely
if 'DeadlineUnixTime' in competitions.columns:
    competitions['Deadline'] = pd.to_datetime(competitions['DeadlineUnixTime'], unit='ms', errors='coerce')
elif 'Deadline' in competitions.columns:
    competitions['Deadline'] = pd.to_datetime(competitions['Deadline'], errors='coerce')
else:
    print("No 'Deadline' column found.")

if 'CreationDate' in kernels.columns:
    kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'], errors='coerce')
else:
    print("No 'CreationDate' column in kernels.")

# Top 10 most participated competitions
if 'TotalTeams' in competitions.columns:
    top_comps = competitions.sort_values('TotalTeams', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_comps, x='Title', y='TotalTeams')
    plt.title("Top 10 Competitions by Team Participation")
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.show()

# Kernel creation trends
if 'CreationDate' in kernels.columns:
    kernels['YearMonth'] = kernels['CreationDate'].dt.to_period('M')
    monthly_kernel_counts = kernels['YearMonth'].value_counts().sort_index()

    plt.figure(figsize=(14, 6))
    monthly_kernel_counts.plot()
    plt.title("Kernels Created Per Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Kernels")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Users by country
top_countries = users['Country'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 Countries by User Count")
plt.xlabel("Number of Users")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# PerformanceTier distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=users, x='PerformanceTier', order=sorted(users['PerformanceTier'].dropna().unique()))
plt.title("User Performance Tiers")
plt.xlabel("Tier")
plt.ylabel("Count")
plt.tight_layout()
plt.show()





