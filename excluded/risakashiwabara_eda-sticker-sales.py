import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge,Lasso


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=["date"])
original_train_df = train_df.copy()
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=["date"])


product_counts = train_df["product"].value_counts()
plt.figure(figsize=(4, 5))
product_counts.plot(kind="bar", color="skyblue")
plt.title("Product Count Distribution", fontsize=14)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# Group by 'country' and calculate total num_sold
country_sales = train_df.groupby("country")["num_sold"].sum()

# Plot
plt.figure(figsize=(4, 5))
country_sales.plot(kind="bar", color=["skyblue", "orange"])
plt.title("Total Sales by Country", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=0)
plt.show()


# Group by 'product' and calculate total num_sold
product_sales = train_df.groupby("product")["num_sold"].sum()

# Plot
plt.figure(figsize=(4, 4))
product_sales.plot(kind="pie", autopct='%1.1f%%', startangle=90, cmap="Set3")
plt.title("Percentage of Sales by Product", fontsize=16)
plt.ylabel("")  # Remove the y-axis label for cleaner visualization
plt.show()


# Convert 'date' to datetime and group by 'date' to calculate total num_sold
train_df["date"] = pd.to_datetime(train_df["date"])
time_sales = train_df.groupby("date")["num_sold"].sum()

# Plot
plt.figure(figsize=(5, 4))
time_sales.plot(kind="line", marker="o", color="green")
plt.title("Sales Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.grid(axis="both", linestyle="--", alpha=0.7)
plt.show()


# Pivot table for stacking by 'country' and 'product'
stacked_data = train_df.pivot_table(
    index="country", columns="product", values="num_sold", aggfunc="sum"
)

# Plot
stacked_data.plot(kind="bar", stacked=True, figsize=(5, 6), cmap="Set2")
plt.title("Sales by Country and Product", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.legend(title="Product", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# Create a pivot table for the heatmap
heatmap_data = train_df.pivot_table(
    index="country", columns="product", values="num_sold", aggfunc="sum"
)

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Number Sold'})

plt.title("Heatmap of Sales by Country and Product", fontsize=16)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.tight_layout()
plt.show()

