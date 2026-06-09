import pandas as pd
import matplotlib.pyplot as plt


# Read the data
data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
data


data.describe()


data.describe(include='object')


# Remove outliners
data = data[data['Number_of_Ads'] < 10]
data = data[data['Episode_Length_minutes'] < 300]
data = data[data['Episode_Length_minutes'] != 0]
data = data[data['Host_Popularity_percentage'] <= 100]
data = data[data['Guest_Popularity_percentage'] <= 100]
data = data[data['Episode_Length_minutes'] >= data['Listening_Time_minutes']]


numerical_data = data.select_dtypes(include=['number']).drop(['id'], axis=1)

# Set up subplots
num_features = numerical_data.shape[1]
cols = 3  # Number of columns of subplots
rows = (num_features + cols - 1) // cols  # Compute required rows

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
axes = axes.flatten()  # Flatten in case it's a 2D array

for i, col in enumerate(numerical_data.columns):
    axes[i].boxplot(numerical_data[col].dropna())
    axes[i].set_title(f'Box Plot of {col}')
    axes[i].set_ylabel(col)

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


data = data[data['Number_of_Ads'] < 10]


numerical_data = data.select_dtypes(include=['number']).drop(['id'], axis=1)

# Set up subplots
num_features = numerical_data.shape[1]
cols = 3  # Number of columns of subplots
rows = (num_features + cols - 1) // cols  # Compute required rows

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
axes = axes.flatten()  # Flatten in case it's a 2D array

for i, col in enumerate(numerical_data.columns):
    axes[i].boxplot(numerical_data[col].dropna())
    axes[i].set_title(f'Box Plot of {col}')
    axes[i].set_ylabel(col)

# Hide any empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


data[data['Episode_Length_minutes'] > 300]


data[data['Podcast_Name'] == 'Home & Living']


data['Episode'] = data['Episode_Title'].str.split(' ').apply(lambda x: x[1]).astype(int)


data[(data['Podcast_Name'] == 'Home & Living') & (data['Episode'] < 16)]


data[(data['Podcast_Name'] == 'Home & Living') & (data['Episode'] == 3)]


print("With popular host")
data[data['Host_Popularity_percentage'] > 80]['Episode_Sentiment'].value_counts() / len(data[data['Host_Popularity_percentage'] > 80])


print("Without popular host")
data['Episode_Sentiment'].value_counts() / len(data)


# Let's try to visualize this:
with_popular = data[data['Host_Popularity_percentage'] > 80]['Episode_Sentiment'].value_counts(normalize=True)
overall = data['Episode_Sentiment'].value_counts(normalize=True)

# Combine into a single DataFrame for easy plotting
df_compare = pd.DataFrame({
    'With Popular Host': with_popular,
    'Overall': overall
}).fillna(0)

# Plot
df_compare.plot(kind='bar', figsize=(8, 5))
plt.title("Episode Sentiment Distribution")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.legend(title='Host Type')
plt.tight_layout()
plt.show()


# Let's try to visualize this:
with_popular = data[data['Host_Popularity_percentage'] > 80]['Listening_Time_minutes'].mean()
overall = data['Listening_Time_minutes'].mean()

# Combine into a single DataFrame for easy plotting
df_compare = pd.DataFrame({
    'Listening Time (minutes)': [with_popular, overall]
}, index=['With Popular Host', 'Overall'])

# Plot
df_compare.plot(kind='bar', figsize=(8, 5), legend=False)
plt.title("Average Listening Time by Host Popularity")
plt.ylabel("Listening Time (minutes)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("Difference: ", with_popular - overall, "minutes")


def categorize_host_popularity(score):
    if score <= 20:
        return 'Niche or Newcomer'
    elif score <= 40:
        return 'Emerging'
    elif score <= 60:
        return 'Established'
    elif score <= 80:
        return 'Popular'
    else:
        return 'Celebrity / Influential'

# Apply the function to create a new column
data['Host_Popularity_Category'] = data['Host_Popularity_percentage'].apply(categorize_host_popularity)



# Group and sort by average listening time
df_compare = data.groupby('Host_Popularity_Category')['Listening_Time_minutes'].mean()
df_compare = df_compare.sort_values()  # <-- sort by values instead of category order

# Plot the sorted results
df_compare.plot(kind='bar', figsize=(8, 5), color='coral')
plt.title("Average Listening Time by Host Popularity Category")
plt.ylabel("Listening Time (minutes)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()


# Reshape to wide format
df_compare = data.groupby('Host_Popularity_Category')['Episode_Sentiment'].value_counts(normalize=True).unstack()

# Optional: sort categories by 'Positive' or total sentiment strength
df_compare = df_compare.sort_values(by='Positive', ascending=False)

# Plot grouped bar chart
df_compare.plot(kind='bar', figsize=(10, 6))
plt.title("Episode Sentiment Distribution by Host Popularity")
plt.ylabel("Proportion")
plt.xlabel("Host Popularity Category")
plt.xticks(rotation=30, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()



# Group and sort by average listening time
df_compare = data.groupby('Host_Popularity_Category')['Episode_Length_minutes'].mean()
df_compare = df_compare.sort_values()  # <-- sort by values instead of category order

# Plot the sorted results
df_compare.plot(kind='bar', figsize=(8, 5), color='coral')
plt.title("Average Listening Time by Episode Length Minutes")
plt.ylabel("Listening Time (minutes)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()


def simplify_popularity(percentage):
    if percentage >= 81:
        return 'Famous'
    elif percentage >= 61:
        return 'Well-Known'
    elif percentage >= 41:
        return 'Known'
    elif percentage >= 21:
        return 'Less Known'
    else:
        return 'New Face'

# Apply to DataFrame
data['Guest_Popularity_Category'] = data['Guest_Popularity_percentage'].apply(simplify_popularity)


# Group and sort by average listening time
df_compare = data.groupby('Guest_Popularity_Category')['Listening_Time_minutes'].mean()
df_compare = df_compare.sort_values()  # <-- sort by values instead of category order

# Plot the sorted results
df_compare.plot(kind='bar', figsize=(8, 5), color='coral')
plt.title("Average Listening Time by Guest Popularity Category")
plt.ylabel("Listening Time (minutes)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()


# Reshape to wide format
df_compare = data.groupby('Guest_Popularity_Category')['Episode_Sentiment'].value_counts(normalize=True).unstack()

# Optional: sort categories by 'Positive' or total sentiment strength
df_compare = df_compare.sort_values(by='Positive', ascending=False)

# Plot grouped bar chart
df_compare.plot(kind='bar', figsize=(10, 6))
plt.title("Episode Sentiment Distribution by Guest Popularity")
plt.ylabel("Proportion")
plt.xlabel("Guest Popularity Category")
plt.xticks(rotation=30, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()


# Group and sort by average listening time
df_compare = data.groupby('Genre')['Listening_Time_minutes'].mean()
df_compare = df_compare.sort_values()  # <-- sort by values instead of category order

# Plot the sorted results
df_compare.plot(kind='bar', figsize=(8, 5), color='coral')
plt.title("Average Listening Time by Genre")
plt.ylabel("Listening Time (minutes)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

