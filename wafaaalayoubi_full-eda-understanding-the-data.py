# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from wordcloud import WordCloud, STOPWORDS


# Set plot styles for a professional look
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# Load the datasets
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


# Print the size of the datasets
print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")
print("-" * 30)

# Display the first 5 rows of the training data
print("First 5 rows of the training data:")
display(train_df.head())
print("-" * 30)

# Get a concise summary of the dataframe
print("Training data info:")
train_df.info()
print("-" * 30)

# Check for any missing values
print("Missing values in training data:")
print(train_df.isnull().sum())


# Create a new dataframe containing only the rows with a misconception label
labeled_df = train_df.dropna(subset=['Misconception']).copy()

print(f"Shape of the filtered dataframe: {labeled_df.shape}")
print("-" * 30)


# Calculate the number of unique misconceptions
num_unique_misconceptions = labeled_df['Misconception'].nunique()
print(f"Number of unique misconceptions: {num_unique_misconceptions}")
print("-" * 30)


# Get the top 15 most common misconceptions
top_15_misconceptions = labeled_df['Misconception'].value_counts().nlargest(15)

print("Top 15 most common misconceptions:")
print(top_15_misconceptions)
print("-" * 30)


# --- Visualization ---

# Set up the plot
plt.figure(figsize=(12, 8))
sns.barplot(y=top_15_misconceptions.index, x=top_15_misconceptions.values, palette='viridis')

# Add labels and title
plt.title('Top 15 Most Frequent Student Misconceptions', fontsize=18)
plt.xlabel('Number of Occurrences', fontsize=12)
plt.ylabel('Misconception', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()


# We will work with the labeled_df since these are the texts we'll actually train on.

# --- Feature Engineering: Text Lengths ---

# 1. Character Count
labeled_df['text_char_count'] = labeled_df['StudentExplanation'].str.len()

# 2. Word Count (by splitting on spaces)
labeled_df['text_word_count'] = labeled_df['StudentExplanation'].str.split().str.len()


# --- Statistical Summary ---
print("Statistical summary of text lengths:")
display(labeled_df[['text_char_count', 'text_word_count']].describe())


# --- Visualization of Distributions ---

plt.figure(figsize=(16, 6))

# Plot for Character Count
plt.subplot(1, 2, 1)
sns.histplot(labeled_df['text_char_count'], bins=50, color='skyblue', kde=True)
plt.title('Distribution of Character Count in Explanations')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')

# Plot for Word Count
plt.subplot(1, 2, 2)
sns.histplot(labeled_df['text_word_count'], bins=50, color='salmon', kde=True)
plt.title('Distribution of Word Count in Explanations')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

plt.suptitle('Analysis of Student Explanation Lengths', fontsize=20, y=1.02)
plt.tight_layout()
plt.show()


# Let's look at some of the longest explanations to see what they're like
print("\nTop 5 Longest Explanations (by word count):")
pd.set_option('display.max_colwidth', None) # Show full text
display(labeled_df.nlargest(5, 'text_word_count')[['StudentExplanation', 'text_word_count', 'Misconception']])


# --- Word Cloud for All Explanations ---

# Combine all student explanations into a single string of text
all_text = ' '.join(labeled_df['StudentExplanation'].astype(str))

# Define stopwords (common words to ignore, like 'the', 'a', 'is')
# We can add custom words if we see noise, e.g., 'because'
stopwords = set(STOPWORDS)
stopwords.update(["because", "it", "is", "the", "to", "and", "i", "of", "that", "in"])

# Create the word cloud object
wordcloud_all = WordCloud(
    stopwords=stopwords,
    background_color="white",
    width=800,
    height=400,
    colormap='cividis'
).generate(all_text)

# Display the generated image
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis("off")
plt.title('Most Common Words Across All Explanations', fontsize=20)
plt.show()


# --- Word Cloud for a Specific Misconception ---

# Let's analyze the second most common misconception: 'Additive'
additive_text = ' '.join(labeled_df[labeled_df['Misconception'] == 'Additive']['StudentExplanation'].astype(str))

wordcloud_additive = WordCloud(
    stopwords=stopwords,
    background_color="white",
    width=800,
    height=400,
    colormap='magma'
).generate(additive_text)

# Display the generated image
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud_additive, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words for the 'Additive' Misconception", fontsize=20)
plt.show()




