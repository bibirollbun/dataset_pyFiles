import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# Set display options for wide tables
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

# Set seaborn style for nicer plots
sns.set(style="whitegrid", palette="pastel")

# Load dataset
train_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Drop unused columns
train_df.drop(columns=['Unnamed: 0', 'id'], inplace=True, errors='ignore')

# Print Analysis Questions
questions = [
    "1. What is the distribution of the target variable `satisfaction`?",
    "2. How is customer satisfaction distributed across different `Gender` categories?",
    "3. How does `Type of Travel` relate to satisfaction levels?",
    "4. What is the relationship between `Class` and customer satisfaction?",
    "5. What is the distribution of `Flight Distance` among customers?",
    "6. How does the `Cleanliness` rating differ between satisfied and dissatisfied customers?"
]

print(" Key Questions for Analysis:\n")
for q in questions:
    print(q)

# Basic dataset info
print("\nDataset Info:")
print(train_df.info())

# Descriptive statistics table with styling for better readability
desc_stats = train_df.describe(include='all').transpose()
cols_to_show = ['count', 'unique', 'top', 'freq', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
desc_stats_subset = desc_stats[cols_to_show]

print("\nDescriptive Statistics:")
display(desc_stats_subset.style.background_gradient(cmap='Blues'))

# Check and display missing values (only if exist)
missing = train_df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    print("\nMissing Values:")
    display(missing.to_frame(name='Missing Count').style.background_gradient(cmap='Reds'))
else:
    print("\nNo missing values detected.")

# Function to display count tables nicely
def show_counts(col, color):
    print(f"\n{col} Distribution Table:")
    counts = train_df[col].value_counts().to_frame(name='Count')
    display(counts.style.bar(subset=['Count'], color=color))

# Show count tables for categorical columns
show_counts('satisfaction', 'skyblue')
show_counts('Gender', 'lightgreen')
show_counts('Type of Travel', 'orange')
show_counts('Class', 'violet')

# Plot: Target Class Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='satisfaction', data=train_df, palette='Set2')
plt.title('Target Class Distribution')
plt.xlabel('Satisfaction')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Plot: Gender vs Satisfaction
plt.figure(figsize=(8,5))
sns.countplot(x='Gender', hue='satisfaction', data=train_df, palette='Set1')
plt.title('Gender vs Satisfaction')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Plot: Type of Travel vs Satisfaction
plt.figure(figsize=(10,6))
sns.countplot(x='Type of Travel', hue='satisfaction', data=train_df, palette='Set3')
plt.title('Type of Travel vs Satisfaction')
plt.xlabel('Type of Travel')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot: Class vs Satisfaction
plt.figure(figsize=(8,5))
sns.countplot(x='Class', hue='satisfaction', data=train_df, palette='Paired')
plt.title('Travel Class vs Satisfaction')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Plot: Flight Distance distribution with KDE and transparency
plt.figure(figsize=(10,6))
sns.histplot(train_df['Flight Distance'], bins=30, kde=True, color='mediumslateblue', alpha=0.6)
plt.title('Flight Distance Distribution')
plt.xlabel('Flight Distance')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot: Cleanliness rating by Satisfaction with boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x='satisfaction', y='Cleanliness', data=train_df, palette='coolwarm')
plt.title('Cleanliness Rating by Satisfaction')
plt.xlabel('Satisfaction')
plt.ylabel('Cleanliness')
plt.tight_layout()
plt.show()

# Plot: Correlation heatmap with annotation for numeric features
plt.figure(figsize=(14,10))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='vlag', center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


