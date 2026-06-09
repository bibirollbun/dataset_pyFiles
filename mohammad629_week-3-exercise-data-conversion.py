# -----------------------------
# ğŸ“¦ 1. Ú©ØªØ§Ø¨Ø®Ø§Ù†Ù‡â€ŒÙ‡Ø§ Ùˆ Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ø¯Ø§Ø¯Ù‡
# -----------------------------

import pandas as pd

# Ù�Ø§ÛŒÙ„ ØªÙ…ÛŒØ²Ø´Ø¯Ù‡ Ø§Ø² ØªÙ…Ø±ÛŒÙ† Ù‡Ù�ØªÙ‡ Ø¯ÙˆÙ… Ø±Ø§ Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ú©Ù†
df = pd.read_csv("/kaggle/input/DontGetKicked/training.csv")

# Ù†Ù…Ø§ÛŒØ´ Ú†Ù†Ø¯ Ø³Ø·Ø± Ø§ÙˆÙ„ Ø¯Ø§Ø¯Ù‡
print("âœ… Ø´Ú©Ù„ Ø§ÙˆÙ„ÛŒÙ‡ Ø¯Ø§Ø¯Ù‡:", df.shape)
df.head()



# -----------------------------
# ğŸ”¹ 2. Ù†Ø±Ù…Ø§Ù„â€ŒØ³Ø§Ø²ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ (Ø¨Ø¯ÙˆÙ† Ø®Ø·Ø§)
# -----------------------------

from sklearn.preprocessing import MinMaxScaler

# Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ù‡Ø¯Ù� Ø§ÙˆÙ„ÛŒÙ‡
num_cols = ['VehOdo', 'VehicleAge', 'MMRAcquisitionRetailCleanPrice']

# Ø¨Ø±Ø±Ø³ÛŒ Ø§ÛŒÙ†Ú©Ù‡ Ú©Ø¯ÙˆÙ…â€ŒÛŒÚ© Ø§Ø² Ø§ÛŒÙ† Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ Ø¯Ø± Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… ÙˆØ¬ÙˆØ¯ Ø¯Ø§Ø±Ù†Ø¯
available_cols = [col for col in num_cols if col in df.columns]

# Ø§Ú¯Ø± Ù‡ÛŒÚ†â€ŒÚ©Ø¯Ø§Ù… Ø§Ø² Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ ÙˆØ¬ÙˆØ¯ Ù†Ø¯Ø§Ø´ØªÙ†Ø¯ØŒ Ù‡Ø´Ø¯Ø§Ø± Ø¨Ø¯Ù‡
if not available_cols:
    print("âš ï¸� Ù‡ÛŒÚ†â€ŒÚ©Ø¯Ø§Ù… Ø§Ø² Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ø§Ù†ØªØ®Ø§Ø¨ÛŒ Ø¯Ø± Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… ÙˆØ¬ÙˆØ¯ Ù†Ø¯Ø§Ø±Ù†Ø¯.")
else:
    # Ø³Ø§Ø®Øª Ø´ÛŒØ¡ Ù†Ø±Ù…Ø§Ù„â€ŒØ³Ø§Ø²
    scaler = MinMaxScaler()

    # Ø§Ø¹Ù…Ø§Ù„ Ù†Ø±Ù…Ø§Ù„â€ŒØ³Ø§Ø²ÛŒ Ù�Ù‚Ø· Ø±ÙˆÛŒ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ù…ÙˆØ¬ÙˆØ¯
    df[available_cols] = scaler.fit_transform(df[available_cols])

    # Ù†Ù…Ø§ÛŒØ´ Ù†ØªØ§ÛŒØ¬
    print("ğŸ“Š Ø¨Ø¹Ø¯ Ø§Ø² Ù†Ø±Ù…Ø§Ù„â€ŒØ³Ø§Ø²ÛŒ:")
    display(df[available_cols].head())



# -----------------------------
# ğŸ”¹ 3. Ø³Ø§Ø®Øª ÙˆÛŒÚ˜Ú¯ÛŒ Ø¬Ø¯ÛŒØ¯ (Feature Construction)
# -----------------------------

# Ø¨Ø±Ø±Ø³ÛŒ ÙˆØ¬ÙˆØ¯ Ø³ØªÙˆÙ† VehYear
if 'VehYear' in df.columns:
    # Ø³Ø§Ø®Øª ÙˆÛŒÚ˜Ú¯ÛŒ Ø³Ù† Ø®ÙˆØ¯Ø±Ùˆ
    df['CarAge'] = 2025 - df['VehYear']
else:
    print("âš ï¸� Ø³ØªÙˆÙ† VehYear Ø¯Ø± Ø¯Ø§Ø¯Ù‡ ÙˆØ¬ÙˆØ¯ Ù†Ø¯Ø§Ø±Ø¯ØŒ ÙˆÛŒÚ˜Ú¯ÛŒ CarAge Ø³Ø§Ø®ØªÙ‡ Ù†Ø´Ø¯.")

# Ø¨Ø±Ø±Ø³ÛŒ ÙˆØ¬ÙˆØ¯ Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ù‚ÛŒÙ…Øª Ø¨Ø±Ø§ÛŒ Ø§Ø®ØªÙ„Ø§Ù� Ù‚ÛŒÙ…Øª
price_cols = ['MMRAcquisitionAuctionAveragePrice', 'MMRCurrentAuctionAveragePrice']
if all(col in df.columns for col in price_cols):
    df['Price_Diff'] = df['MMRAcquisitionAuctionAveragePrice'] - df['MMRCurrentAuctionAveragePrice']
else:
    print("âš ï¸� ÛŒÚ©ÛŒ Ø§Ø² Ø³ØªÙˆÙ†â€ŒÙ‡Ø§ÛŒ Ù‚ÛŒÙ…Øª Ø¯Ø± Ø¯Ø§Ø¯Ù‡ ÙˆØ¬ÙˆØ¯ Ù†Ø¯Ø§Ø±Ø¯ØŒ ÙˆÛŒÚ˜Ú¯ÛŒ Price_Diff Ø³Ø§Ø®ØªÙ‡ Ù†Ø´Ø¯.")

# Ù†Ù…Ø§ÛŒØ´ Ú†Ù†Ø¯ Ø³Ø·Ø± Ø§Ø² ÙˆÛŒÚ˜Ú¯ÛŒâ€ŒÙ‡Ø§ÛŒ Ø¬Ø¯ÛŒØ¯ (Ø¯Ø± ØµÙˆØ±Øª ÙˆØ¬ÙˆØ¯)
available_cols = [col for col in ['VehYear', 'CarAge', 'Price_Diff'] if col in df.columns]
print("âœ¨ ÙˆÛŒÚ˜Ú¯ÛŒâ€ŒÙ‡Ø§ÛŒ Ø¬Ø¯ÛŒØ¯ Ø³Ø§Ø®ØªÙ‡ Ø´Ø¯Ù†Ø¯:")
df[available_cols].head()



# -----------------------------
# ğŸ”¹ 4. Ú¯Ø³Ø³ØªÙ‡â€ŒØ³Ø§Ø²ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§
# -----------------------------

# ØªÙ‚Ø³ÛŒÙ… Ù…ØªØºÛŒØ± VehicleAge Ø¨Ù‡ Ø³Ù‡ Ø¨Ø§Ø²Ù‡
df['VehicleAge_Bin'] = pd.cut(df['VehicleAge'], bins=3, labels=['Low', 'Medium', 'High'])

# ØªÙ‚Ø³ÛŒÙ… Ù…ØªØºÛŒØ± VehOdo Ø¨Ù‡ 4 Ø¯Ø³ØªÙ‡ Ù…Ø³Ø§ÙˆÛŒ
df['VehOdo_Bin'] = pd.cut(df['VehOdo'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Ù†Ù…Ø§ÛŒØ´ Ù†Ù…ÙˆÙ†Ù‡â€ŒØ§ÛŒ Ø§Ø² Ú¯Ø³Ø³ØªÙ‡â€ŒØ³Ø§Ø²ÛŒ
print("ğŸ“¦ Ø¯Ø§Ø¯Ù‡ Ú¯Ø³Ø³ØªÙ‡â€ŒØ³Ø§Ø²ÛŒ Ø´Ø¯Ù‡:")
df[['VehicleAge', 'VehicleAge_Bin', 'VehOdo', 'VehOdo_Bin']].head()



# -----------------------------
# ğŸ”¹ 5. Ù‡Ù…ÙˆØ§Ø±Ø³Ø§Ø²ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ (Smoothing)
# -----------------------------

# Ø§Ø³ØªÙ�Ø§Ø¯Ù‡ Ø§Ø² Ù…ÛŒØ§Ù†Ú¯ÛŒÙ† Ù…ØªØ­Ø±Ú© Ø±ÙˆÛŒ Ù‚ÛŒÙ…Øª Ø®Ø±Ø¯Ù‡â€ŒÙ�Ø±ÙˆØ´ÛŒ Ù�Ø¹Ù„ÛŒ
df['Smoothed_Price'] = df['MMRCurrentRetailAveragePrice'].rolling(window=5).mean()

# Ù†Ù…Ø§ÛŒØ´ Ú†Ù†Ø¯ Ø³Ø·Ø± Ø§ÙˆÙ„ Ø¨Ø±Ø§ÛŒ Ù…Ù‚Ø§ÛŒØ³Ù‡ Ù‚Ø¨Ù„ Ùˆ Ø¨Ø¹Ø¯ Ø§Ø² Ù‡Ù…ÙˆØ§Ø±Ø³Ø§Ø²ÛŒ
print("ğŸª¶ Ø¯Ø§Ø¯Ù‡ Ù‡Ù…ÙˆØ§Ø±Ø³Ø§Ø²ÛŒâ€ŒØ´Ø¯Ù‡ (Ù†Ù…ÙˆÙ†Ù‡):")
df[['MMRCurrentRetailAveragePrice', 'Smoothed_Price']].head(10)



# -----------------------------
# ğŸ’¾ 6. Ø°Ø®ÛŒØ±Ù‡ Ø¯Ø§Ø¯Ù‡â€ŒÛŒ Ù†Ù‡Ø§ÛŒÛŒ
# -----------------------------

df.to_csv("/kaggle/working/DontGetKicked_transformed.csv", index=False)

print("âœ… Ù�Ø§ÛŒÙ„ Ù†Ù‡Ø§ÛŒÛŒ Ø°Ø®ÛŒØ±Ù‡ Ø´Ø¯: DontGetKicked_transformed.csv")





