import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ğ“Ñ€Ğ°Ñ„Ğ¸ĞºÑƒÑƒĞ´Ñ‹Ğ³ notebook Ğ´Ñ�Ñ�Ñ€ ÑˆÑƒÑƒĞ´ Ñ…Ğ°Ñ€ÑƒÑƒĞ»Ğ°Ñ…
%matplotlib inline


# Kaggle Ğ¾Ñ€Ñ‡Ğ½Ñ‹ Ğ·Ğ°Ğ¼Ğ°Ğ°Ñ� Ğ´Ğ°Ñ‚Ğ° ÑƒĞ½ÑˆĞ¸Ñ…
path = "/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv"
df = pd.read_csv(path)


# Ğ­Ñ…Ğ½Ğ¸Ğ¹ 5 Ğ¼Ó©Ñ€
print(df.head())

# Ğ¥Ñ�Ğ¼Ğ¶Ñ�Ñ�, Ñ‚Ó©Ñ€Ó©Ğ», Ñ…Ğ¾Ğ¾Ñ�Ğ¾Ğ½ ÑƒÑ‚Ğ³Ğ° Ğ³Ñ�Ñ… Ğ¼Ñ�Ñ‚
print(df.info())

# Ğ¢Ğ¾Ğ¾Ğ½ Ñ…ÑƒĞ²ÑŒÑ�Ğ°Ğ³Ñ‡Ğ´Ñ‹Ğ½ Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸Ğº
print(df.describe())


print("Ğ¥Ğ¾Ğ¾Ñ�Ğ¾Ğ½ ÑƒÑ‚Ğ³Ğ°:\n", df.isnull().sum())
print("\nĞ”Ğ°Ğ²Ñ…Ğ°Ñ€Ğ´Ñ�Ğ°Ğ½ Ğ¼Ó©Ñ€Ğ¸Ğ¹Ğ½ Ñ‚Ğ¾Ğ¾:", df.duplicated().sum())


df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Ğ¢Ğ¾Ğ¾Ğ½ Ñ…ÑƒĞ²ÑŒÑ�Ğ°Ğ³Ñ‡Ğ´Ñ‹Ğ½ Ñ‚Ğ°Ñ€Ñ…Ğ°Ğ»Ñ‚", fontsize=14)
plt.show()


corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    print(f"ğŸŸ¢ {col} Ğ±Ğ°Ğ³Ğ°Ğ½Ñ‹Ğ½ Ğ´Ğ°Ğ²Ñ‚Ğ°Ğ¼Ğ¶:")
    print(df[col].value_counts().head())
    print("-" * 50)


plt.style.use("default")
sns.set_theme(style="whitegrid")

path = "/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv"
df = pd.read_csv(path)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.describe())


df["month"] = pd.to_datetime(df["month"], format="%Y-%b")

# Ğ¡Ğ°Ñ€ Ğ±Ò¯Ñ€Ğ¸Ğ¹Ğ½ Ğ´ÑƒĞ½Ğ´Ğ°Ğ¶ Ò¯Ğ½Ñ�
monthly_price = df.groupby("month")["price_new_house_transactions_nearby_sectors"].mean()

plt.figure(figsize=(12,6))
monthly_price.plot()
plt.title("ğŸ“ˆ Ğ¡Ğ°Ñ€ Ğ±Ò¯Ñ€Ğ¸Ğ¹Ğ½ Ğ´ÑƒĞ½Ğ´Ğ°Ğ¶ ÑˆĞ¸Ğ½Ñ� Ğ±Ğ°Ğ¹Ñ€Ğ½Ñ‹ Ò¯Ğ½Ñ� (CNY/mÂ²)")
plt.xlabel("Ğ¡Ğ°Ñ€")
plt.ylabel("Ò®Ğ½Ñ� (CNY)")
plt.show()


avg_by_sector = df.groupby("sector")[["price_new_house_transactions_nearby_sectors", 
                                      "area_new_house_transactions_nearby_sectors",
                                      "num_new_house_transactions_nearby_sectors"]].mean().sort_values(
                                      "price_new_house_transactions_nearby_sectors", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=avg_by_sector["price_new_house_transactions_nearby_sectors"][:10],
            y=avg_by_sector.index[:10])
plt.title("ğŸ’° Ğ¥Ğ°Ğ¼Ğ³Ğ¸Ğ¹Ğ½ Ó©Ğ½Ğ´Ó©Ñ€ Ğ´ÑƒĞ½Ğ´Ğ°Ğ¶ Ò¯Ğ½Ğ¸Ğ¹Ğ½ 10 Ñ�ĞµĞºÑ‚Ğ¾Ñ€")
plt.xlabel("Ğ”ÑƒĞ½Ğ´Ğ°Ğ¶ Ò¯Ğ½Ñ� (CNY/mÂ²)")
plt.ylabel("Ğ¡ĞµĞºÑ‚Ğ¾Ñ€")
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(x="area_new_house_transactions_nearby_sectors",
                y="price_new_house_transactions_nearby_sectors",
                hue="sector",
                data=df, alpha=0.7)
plt.title("ğŸ�—ï¸� Ğ¢Ğ°Ğ»Ğ±Ğ°Ğ¹ Ğ±Ğ° Ò¯Ğ½Ñ� Ñ…Ğ¾Ğ¾Ñ€Ğ¾Ğ½Ğ´Ñ‹Ğ½ Ñ…Ğ°Ğ¼Ğ°Ğ°Ñ€Ğ°Ğ»")
plt.xlabel("Ğ¢Ğ°Ğ»Ğ±Ğ°Ğ¹ (mÂ²)")
plt.ylabel("Ò®Ğ½Ñ� (CNY/mÂ²)")
plt.legend([],[], frameon=False)
plt.show()

# =====================


plt.figure(figsize=(8,6))
sns.regplot(x="period_new_house_sell_through_nearby_sectors",
            y="price_new_house_transactions_nearby_sectors",
            data=df)
plt.title("ğŸ•’ Ğ‘Ğ¾Ñ€Ğ»ÑƒÑƒĞ»Ğ°Ğ»Ñ‚Ñ‹Ğ½ Ñ…ÑƒĞ³Ğ°Ñ†Ğ°Ğ° Ğ±Ğ° Ò¯Ğ½Ñ� Ñ…Ğ¾Ğ¾Ñ€Ğ¾Ğ½Ğ´Ñ‹Ğ½ Ñ…Ğ°Ğ¼Ğ°Ğ°Ñ€Ğ°Ğ»")
plt.xlabel("Ğ‘Ğ¾Ñ€Ğ»ÑƒÑƒĞ»Ğ°Ğ»Ñ‚Ñ‹Ğ½ Ñ…ÑƒĞ³Ğ°Ñ†Ğ°Ğ° (period)")
plt.ylabel("Ò®Ğ½Ñ� (CNY/mÂ²)")
plt.show()


plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.title("ğŸ”— Ğ¥ÑƒĞ²ÑŒÑ�Ğ°Ğ³Ñ‡Ğ´Ñ‹Ğ½ Ñ…Ğ°Ğ¼Ğ°Ğ°Ñ€Ğ»Ñ‹Ğ½ Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†")
plt.show()

