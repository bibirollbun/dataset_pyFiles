import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display plots inline
%matplotlib inline
file_path = '/kaggle/input/meta-kaggle/Competitions.csv'
competitions_df = pd.read_csv(file_path)
competitions_df.columns = competitions_df.columns.str.strip()



competitions_df['EnabledDate'] = pd.to_datetime(competitions_df['EnabledDate'], errors='coerce')
competitions_df['Year'] = competitions_df['EnabledDate'].dt.year



plt.figure(figsize=(12,6))
competitions_df['Year'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Number of Competitions per Year')
plt.xlabel('Year')
plt.ylabel('Number of Competitions')
plt.grid(True)
plt.tight_layout()
plt.show()



competitions_df['TotalTeams'] = pd.to_numeric(competitions_df['TotalTeams'], errors='coerce')
top_teams = competitions_df.sort_values('TotalTeams', ascending=False)[['Title', 'TotalTeams']].head(10)

plt.figure(figsize=(12,6))
sns.barplot(data=top_teams, y='Title', x='TotalTeams', palette='magma')
plt.title('Top 10 Competitions by Number of Teams')
plt.xlabel('Number of Teams')
plt.ylabel('')
plt.tight_layout()
plt.show()



competitions_df.to_csv('/kaggle/working/competitions_cleaned.csv', index=False)



# Drop NaNs for rewards
rewarded = competitions_df.dropna(subset=['RewardQuantity'])

plt.figure(figsize=(12,6))
sns.histplot(rewarded['RewardQuantity'], bins=30, kde=True, color='green')
plt.title('Distribution of Competition Rewards')
plt.xlabel('Reward Amount (USD)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



output_path = "/kaggle/working/competitions_cleaned.csv"
competitions_df.to_csv(output_path, index=False)
print("File saved at:", output_path)


