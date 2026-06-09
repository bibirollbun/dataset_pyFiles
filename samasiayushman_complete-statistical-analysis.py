!pip install -q textstat
!pip install -q gensim pyLDAvis  # For more sophisticated topic modeling
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the data is already loaded in a DataFrame called 'data'
# If not, you can load it with:
data = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')


print("Basic Descriptive Statistics:")
print(data.describe(include='all'))


print("\nCategory Counts:")
print(data['Category'].value_counts())


print("\nMisconception Counts:")
print(data['Misconception'].value_counts(dropna=False))


data['Explanation_Length'] = data['StudentExplanation'].str.len()
print("\nExplanation Length Statistics:")
print(data['Explanation_Length'].describe())

# Visualization
plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
data['Category'].value_counts().plot(kind='bar', color=['green', 'blue', 'gray'])
plt.title('Distribution of Answer Categories')
plt.ylabel('Count')


plt.subplot(1, 2, 2)
plt.hist(data['Explanation_Length'].dropna(), bins=10, color='purple')
plt.title('Distribution of Explanation Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


from scipy.stats import chi2_contingency


# First, create a contingency table (though Misconception is mostly NaN here)
contingency_table = pd.crosstab(data['Category'], data['Misconception'])

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test for independence between Category and Misconception:")
print(f"Chi2 Statistic: {chi2:.4f}, p-value: {p:.4f}")


from scipy.stats import f_oneway

# Group explanation lengths by category
groups = [group['Explanation_Length'].dropna() for name, group in data.groupby('Category')]

# Perform one-way ANOVA
f_stat, p_val = f_oneway(*groups)
print(f"\nANOVA for Explanation Length by Category:")
print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")


from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['StudentExplanation'].dropna().fillna(''))
word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print("\nMost common words in explanations:")
print(word_counts.sum().sort_values(ascending=False).head(10))


from textstat import flesch_reading_ease

# Calculate readability scores
data['Readability'] = data['StudentExplanation'].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) else None
)

# Compare readability between correct/incorrect answers
print("Readability by Category:")
print(data.groupby('Category')['Readability'].describe())

# Visualization
plt.figure(figsize=(8, 5))
data.boxplot(column='Readability', by='Category', grid=False)
plt.title('Readability Scores by Answer Category')
plt.suptitle('')
plt.ylabel('Flesch Reading Ease Score')
plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


tfidf = TfidfVectorizer(max_features=50, stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['StudentExplanation'].fillna(''))

# Convert sparse matrix to dense array for easier manipulation
dense_tfidf = tfidf_matrix.toarray()

# Get most important words by category
for category in data['Category'].unique():
    mask = (data['Category'] == category).values
    if sum(mask) > 0:  # Only calculate if there are samples
        avg_tfidf = dense_tfidf[mask].mean(axis=0)
        sorted_tfidf = avg_tfidf.argsort()[-5:][::-1]  # Top 5 words
        print(f"\nTop words for {category}:")
        print([tfidf.get_feature_names_out()[i] for i in sorted_tfidf])
    else:
        print(f"\nNo samples for category {category}")

# Topic Modeling (if you had more data)
if len(data) > 10:  # Lowered threshold for demonstration
    n_components = min(3, len(data)-1)  # Ensure we don't request more components than samples
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(tfidf_matrix)
    print("\nDiscovered Topics:")
    for idx, topic in enumerate(lda.components_):
        print(f"Topic {idx}:")
        print([tfidf.get_feature_names_out()[i] for i in topic.argsort()[-5:]])



incorrect = data[data['Category'].str.contains('Incorrect')]

print("\nCommon incorrect patterns:")
print(incorrect['MC_Answer'].value_counts().head(5))

# Extract numerical patterns from incorrect answers
import re

def extract_fractions(text):
    return re.findall(r'\d+/\d+', str(text))

data['Fractions'] = data['MC_Answer'].apply(extract_fractions)

print("\nMost common incorrect fractions:")
all_fracs = [frac for sublist in data['Fractions'] for frac in sublist]
pd.Series(all_fracs).value_counts().head(5)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Feature engineering
data['Explanation_Length'] = data['StudentExplanation'].str.len()
data['Contains_Because'] = data['StudentExplanation'].str.contains('because', case=False).fillna(False)
data['Num_Fractions'] = data['Fractions'].str.len()


X = data[['Explanation_Length', 'Contains_Because', 'Num_Fractions']].fillna(0)
y = data['Category'].str.contains('Correct')  # Binary target

# Train/test split
if len(data) > 20:  # Only run if sufficient data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("\nModel Performance:")
    print(classification_report(y_test, model.predict(X_test)))

