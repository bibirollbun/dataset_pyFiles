# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')


# This cell is used to store helpful functions

################################################
# Function to load dataset
################################################
def load_dataset(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    print('='*100)
    print('Loading Dataset...')
    print('='*100)  
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        display(df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

################################################
# Function to show dattaset basic information
################################################
def show_dataset_basic_info(df):
    print("="*100)
    print("DATASET BASIC INFORMATION")
    print("="*100)

    print(f"Total number of articles: {len(df)}")
    print(f"Number of features: {df.shape[1]}")

    # Column information
    print("\nColumn Names and Data Types:")
    print(df.dtypes)

    # Check for missing values
    print("\nMissing Values:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])

################################################
# Function to show dattaset overall information
################################################
def display_overall_statistics(df, text_col):   
    print("="*100)
    print("DATASET OVERALL INFORMATION")
    print("="*100)

    df = df.copy()

    # Basic features
    df['text_length'] = df[text_col].apply(len)
    df['word_count'] = df[text_col].str.split().apply(len)
    df['sentence_count'] = df[text_col].apply(lambda x: len(str(x).split('.')))
    df['avg_word_length'] = df[text_col].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) 
        if len(str(x).split()) > 0 else 0
    )
    
    # Print numbers
    print(f"Average Text length: {df['text_length'].mean():.2f} characters")
    print(f"Average word count: {df['word_count'].mean():.2f} words")
    print(f"Average sentence count: {df['sentence_count'].mean():.2f} sentences")
    print(f"Average word length: {df['avg_word_length'].mean():.2f} characters")
    
    total_words = df['word_count'].sum()
    print(f"Total number of words in dataset: {total_words}")

    # Unique words
    all_words = " ".join(df[text_col]).split()
    unique_words = set(all_words)
    print(f"Total number of unique words in dataset: {len(unique_words)}")

    # Words by category
    print("\nTotal Word Count by Category:")
    print(df.groupby('Category')['word_count'].sum())

    # Most common words
    all_text = " ".join(df[text_col]).lower()
    words = re.findall(r"\b\w+\b", all_text)
    word_freq = Counter(words)

    print("\nMost Common Words:")
    for word, freq in word_freq.most_common(20):
        print(f"{word}: {freq}")
    
################################################
# Function to clean the dataset
################################################
def clean_text(text):
    """
    Clean the input text by removing special characters, numbers, and extra spaces.

    Parameters:
    text (str): Input text to be cleaned.

    Returns:
    str: Cleaned text.
    """

    # Get stopwords (common English words)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))

    # Add common news-related stopwords
    additional_stopwords = ['said', 'say', 'says', 'also', 
                            'one', 'two', 'new', 'year', 'years', 'time', 'mr', 'bn']
    stop_words.update(additional_stopwords)

    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remov number
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

################################################
# Function to extract top words for each topic
################################################
def display_nmf_topics(tfidf_vectorizer, H, n_top_words=10):
    print("="*100)
    print("TOP WORDS IN EACH TOPIC")
    print("="*100)

    """
    Display the top words in each NMF topic.

    Parameters
    ----------
    tfidf_vectorizer : TfidfVectorizer
        The fitted vectorizer used to extract feature names.
    H : ndarray of shape (n_topics, n_features)
        The topic-word matrix from NMF (components_).
        Each row corresponds to a topic and each column corresponds to a word weight.
    n_top_words : int, optional
        Number of top words to display per topic.

    Notes
    -----
    - For each topic (row of H), the function finds the indices of the highest-weighted terms.
    - The corresponding words and weights are printed in descending order.
    """
    # Extract vocabulary words from the vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()

    for topic_idx, topic in enumerate(H):
        # Sort the word weights and pick the indices of the top n words (highest weights)
        top_features_indices = topic.argsort()[:-n_top_words - 1:-1]

        # Convert indices to actual words
        top_features = [feature_names[i] for i in top_features_indices]

        # Extract the corresponding weight values
        top_weights = topic[top_features_indices]

        # Print each word with its NMF weight
        print(f"Topic {topic_idx + 1}:")
        for i in range(n_top_words):
            print(f"   {top_features[i]} ({top_weights[i]:.4f})")
        print()

    
################################################
# Function to run experiments with different pipelines
################################################
def run_experiments_with_pipelines(
 X_train, y_train, X_val, y_val,
    reducers_dict,
    classifiers_dict,
    component_sizes,
    topic_display_func=None,
    n_top_words=10,
    max_iter=2000
):
    results = []
    best_acc = -1
    best_pipeline = None
    best_info = None

    """"
    Run experiments with different dimensionality reducers and classifiers.
    Parameters:
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - reducers_dict: Dictionary of dimensionality reducers (name: class).
    - classifiers_dict: Dictionary of classifiers (name: class).
    - component_sizes: List of component sizes to try for reducers.
    - topic_display_func: Function to display topics (for NMF).
    - n_top_words: Number of top words to display per topic.
    - max_iter: Maximum iterations for NMF.
    Returns:
    - results_df: DataFrame with results.
    - best_pipeline: Best performing pipeline.
    - best_info: Info about the best model.
    """

    print("Training classifiers on dataset variants...\n")

    for reducer_name, reducer_class in reducers_dict.items():
        # If reducer is None (TF-IDF baseline), only run once (no components loop)
        if reducer_class is None:
            component_sizes_to_use = [None]
        else:
            component_sizes_to_use = component_sizes

        for n_comp in component_sizes_to_use:
            # handle n_comp == None for TF-IDF baseline
            if n_comp is None:
                dataset_label = reducer_name
                reducer = None
            else:
                dataset_label = f"{reducer_name} ({n_comp})"
                # initialize reducer as before
                if reducer_name == 'NMF':
                    reducer = reducer_class(n_components=n_comp, max_iter=max_iter, random_state=42)
                elif reducer_name == 'SVD':
                    reducer = reducer_class(n_components=n_comp, random_state=42, n_iter=10)
                else:
                    reducer = reducer_class(n_components=n_comp)

            # Build pipeline without classifier to get data shape after tfidf+reducer
            steps = [('tfidf', TfidfVectorizer())]
            if reducer is not None:
                steps.append(('dim_red', reducer))
            base_pipe = Pipeline(steps)
            # Fit on training data to get transformed shape
            X_train_transformed = base_pipe.fit_transform(X_train)
            print("="*50)
            print(f"Dataset: {dataset_label} Shape: {X_train_transformed.shape}")
            print("="*50)

            for clf_name, clf in classifiers_dict.items():
                # Skip Naive Bayes on SVD due to negative values
                if reducer_name == 'SVD' and clf_name == 'Naive Bayes':
                    print(f"Skipping Naive Bayes on {dataset_label} (negative values present).")
                    continue

                # Now add classifier and fit full pipeline
                full_steps = steps + [('clf', clf)]
                pipe = Pipeline(full_steps)

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_val)

                acc = accuracy_score(y_val, y_pred)
                macro_f1 = f1_score(y_val, y_pred, average='macro')

                print(f"{clf_name} | Acc: {acc:.4f} | F1 (macro): {macro_f1:.4f}")

                results.append({
                    'Dataset': dataset_label,
                    'Classifier': clf_name,
                    'Accuracy': acc,
                    'Macro F1-Score': macro_f1,
                    'Model': pipe
                })

                if topic_display_func is not None and reducer_name == 'NMF':
                    print("\nTop words for each NMF topic:")
                    nmf_model = pipe.named_steps['dim_red']
                    tfidf_vectorizer = pipe.named_steps['tfidf']
                    topic_word_matrix = nmf_model.components_
                    topic_display_func(tfidf_vectorizer, topic_word_matrix, n_top_words)

                if acc > best_acc:
                    best_acc = acc
                    best_pipeline = pipe
                    best_info = {
                        'Reducer': reducer_name,
                        'Classifier': clf_name,
                        'Components': n_comp,
                        'Accuracy': acc,
                        'Macro F1-Score': macro_f1
                    }

    results_df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("BEST PIPELINE SUMMARY")
    print("="*60)
    print(f"Reducer: {best_info['Reducer']}")
    print(f"Classifier: {best_info['Classifier']}")
    print(f"Components: {best_info['Components']}")
    print(f"Accuracy: {best_info['Accuracy']:.4f}")
    print(f"Macro F1-Score: {best_info['Macro F1-Score']:.4f}")
    print("Pipeline steps:")
    for step_name, step_obj in best_pipeline.named_steps.items():
        print(f" - {step_name}: {step_obj}")
    print("="*60)

    return results_df, best_pipeline, best_info

################################################
# Function to train best model on full data and predict on test set
################################################

def train_and_predict_best_model(best_pipeline, X_full_train, y_full_train, X_test, test_ids):
    """
    Fits the best pipeline on full training data, predicts on test data,
    and returns a DataFrame with predictions and IDs.

    Parameters:
    - best_pipeline: sklearn Pipeline (unfitted or fitted on partial data)
    - X_full_train: iterable or array-like of full training texts (train + val combined)
    - y_full_train: array-like labels for full training data
    - X_test: iterable or array-like of test texts (unlabeled)
    - test_ids: array-like of test sample identifiers (e.g., ArticleId)

    Returns:
    - test_results_df: pd.DataFrame with columns ['ArticleId', 'PredictedCategory']
    - test_probs (optional): np.array with probabilities or None
    """

    # 1. Fit best pipeline on full training data
    best_pipeline.fit(X_full_train, y_full_train)

    # 2. Predict categories on test set
    test_preds = best_pipeline.predict(X_test)

    # 3. Predict probabilities if available (e.g., SVM with probability=True)
    if hasattr(best_pipeline.named_steps['clf'], 'predict_proba'):
        test_probs = best_pipeline.predict_proba(X_test)
    else:
        test_probs = None

    # 4. Create results DataFrame with IDs and predicted categories
    test_results_df = pd.DataFrame({
        'ArticleId': test_ids,
        'PredictedCategory': test_preds
    })

    return test_results_df, test_probs






# Load trainig dataset
train_df = load_dataset('/kaggle/input/learn-ai-bbc/BBC News Train.csv')


# Basic information about the dataset
show_dataset_basic_info(train_df)


# Analyze the distribution of categories
print("="*50)
print("CATEGORY DISTRIBUTION ANALYSIS")
print("="*50)

# Check unique categories
categories = train_df['Category'].unique()
print(f"Unique categories: {categories}")
print(f"Number of categories: {len(categories)}")

# Calculate distribution
category_dist = train_df['Category'].value_counts()
print("\nCategory Distribution:")
print(category_dist)

# Calculate percentages
category_percent = (train_df['Category'].value_counts(normalize=True) * 100).round(2)
print("\nCategory Distribution (Percentage):")
print(category_percent)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar plot
axes[0].bar(category_dist.index, category_dist.values, color='skyblue', edgecolor='black')
axes[0].set_title('Category Distribution - Count', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Category', fontsize=12)
axes[0].set_ylabel('Number of Articles', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Add count labels on bars
for i, v in enumerate(category_dist.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Pie chart
colors = plt.cm.Set3(np.linspace(0, 1, len(category_dist)))
axes[1].pie(category_dist.values, labels=category_dist.index, autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Category Distribution - Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Statistical summary
print("\nStatistical Summary of Distribution:")
print(f"Mean articles per Category: {category_dist.mean():.2f}")
print(f"Standard deviation: {category_dist.std():.2f}")
print(f"Min articles: {category_dist.min()} ({category_dist.idxmin()})")
print(f"Max articles: {category_dist.max()} ({category_dist.idxmax()})")


# Look at the text in each category
print("="*50)
print("TEXT EXPLORATION BY CATEGORY")
for category in categories:
    print(f"\nCategory: {category}")
    sample_texts = train_df[train_df['Category'] == category]['Text'].sample(2, random_state=42).values
    for i, text in enumerate(sample_texts, 1):
        print(f" Sample {i}: {text[:200]}...")  # Print first 200 characters


# Print overall statistics befor processing
display_overall_statistics(train_df, 'Text')


# Create cleaned Text column
train_df['cleaned_text'] = train_df['Text'].apply(clean_text)

# Display overall statistics
display_overall_statistics(train_df, 'cleaned_text')


# Create a CountVectorizer instance
count_vectorizer = CountVectorizer()

# Fit and transform the cleaned text data
count_matrix = count_vectorizer.fit_transform(train_df['cleaned_text'])

# Convert to DataFrame for easier handling
count_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out(),
                         index=train_df['Category']  )
print("\nCount Vectorizer DataFrame Shape:", count_df.shape)

# Result the word counts per category
category_word_counts = count_df.groupby('Category').sum()
category_word_counts = category_word_counts.reset_index()

# Define a function to plot top N words per category
def plot_top_n_words_per_category(category_word_counts, n=10):
    categories = category_word_counts['Category'].values
    num_categories = len(categories)
    
    fig, axes = plt.subplots(num_categories, 1, figsize=(10, num_categories * 4))
    
    for i, category in enumerate(categories):
        word_counts = category_word_counts[category_word_counts['Category'] == category].drop('Category', axis=1).T
        word_counts.columns = ['count']
        top_words = word_counts.sort_values(by='count', ascending=False).head(n)
        
        axes[i].barh(top_words.index, top_words['count'], color='teal', edgecolor='black')
        axes[i].set_title(f'Top {n} Words in Category: {category}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Word Count')
        axes[i].invert_yaxis()  # Highest counts on top
        
        # Add count labels on bars
        for j, v in enumerate(top_words['count']):
            axes[i].text(v + 1, j, str(v), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Plot top 10 words per category
plot_top_n_words_per_category(category_word_counts, n=10)




# Map category in training set to integer labels
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['Category'])
train_df


# Define a function to tokenize the cleaned text
train_df['tokenized_text'] = train_df['cleaned_text'].apply(lambda x: x.split())
train_df


# Split the dataset into training and validation sets
X = train_df['cleaned_text']
y = train_df['label_encoded']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# Run experiments with different reducers and classifiers

reducers = {
    'TF-IDF (Original)': None,
    'SVD': TruncatedSVD,
    'NMF': NMF
}

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
    'Naive Bayes': MultinomialNB()
}

component_sizes = [5, 10, 20]

results_df, best_pipe, best_info = run_experiments_with_pipelines(
    X_train, y_train, X_val, y_val,
    reducers_dict=reducers,
    classifiers_dict=classifiers,
    component_sizes=component_sizes,
    max_iter=2000
)

# To display results nicely (in Jupyter):
results_df.style.format({
    'Accuracy': '{:.6f}',
    'Macro F1-Score': '{:.6f}'
})



# Process the test dataset and make predictions using the best model
# Load test dataset
test_df = load_dataset('/kaggle/input/learn-ai-bbc/BBC News Test.csv')
# Clean the test dataset
test_df['cleaned_text'] = test_df['Text'].apply(clean_text)
# Make predictions on the test dataset
test_results_df, test_probs = train_and_predict_best_model(
    best_pipe,
    X_full_train=pd.concat([X_train, X_val]),
    y_full_train=pd.concat([y_train, y_val]),
    X_test=test_df['cleaned_text'],
    test_ids=test_df['ArticleId']
)

# Map predicted integer labels back to original category names
test_results_df['PredictedCategory'] = label_encoder.inverse_transform(test_results_df['PredictedCategory'])
# Display test predictions
test_results_df.head()



# Read the semple submission file
submission = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Sample Solution.csv')

# Prepare submission DataFrame
# Replace 'Category' column with our predictions
submission['Category'] = test_results_df['PredictedCategory']

# Save submission file
submission.to_csv('submission.csv', index=False)

