import numpy as np

# Get the unique values from the sparse y_train matrix
# Convert to dense array and then find unique values
unique_values_y_train = np.unique(y_train.toarray())

print("Unique values in y_train:")
print(unique_values_y_train)

# Optionally, print the data type
print("\nData type of y_train values:")
print(y_train.dtype)


from sklearn.model_selection import train_test_split
import numpy as np # Import numpy for potential array handling

# Assuming you have a variable 'y' containing your labels
# y should have a shape (n_samples,) where n_samples is the number of rows in final_features_with_type

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    final_features_with_type,  # Use the combined feature matrix
    y,  # Your labels goes here
    test_size=0.2,  # Example: 20% for testing
    random_state=42  # Example: for reproducibility
)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Calculate the length of each question
multi_choice_df.loc[:, 'question_length'] = multi_choice_df['question'].apply(len)

# Display statistics about the question lengths
print("Statistics about question lengths:")
display(multi_choice_df['question_length'].describe())


import jsonlines
import pandas as pd

# Load the JSONL file
data = []
with jsonlines.open('/content/curebench_testset_phase2.jsonl') as reader:
    for obj in reader:
        data.append(obj)

# Convert to a pandas DataFrame for easier inspection
df = pd.DataFrame(data)

# Display the first few rows and the columns
display(df.head())
display(df.info())


import pandas as pd

# Perform one-hot encoding on the 'question_type' column
question_type_one_hot = pd.get_dummies(model_input_df_combined['question_type'], prefix='question_type')

# Display the first few rows of the one-hot encoded features
print("One-hot encoded 'question_type' features (first 5 rows):")
display(question_type_one_hot.head())

# Display the columns of the one-hot encoded features
print("\nColumns (question types) in the one-hot encoded features:")
print(question_type_one_hot.columns)


# Display the shape of the final sparse matrix
print("Shape of the final combined sparse matrix:", final_sparse_matrix.shape)


from scipy.sparse import hstack, coo_matrix
import pandas as pd

# Create a list to hold the expanded one-hot encoded rows
expanded_question_type_rows = []

# Iterate through each question in model_input_df_combined
for index, row in model_input_df_combined.iterrows():
    question_type_row = question_type_one_hot.loc[[index]] # Get the one-hot encoded row for this question
    num_options = len(row['options']) # Get the number of options for this question

    # Repeat the question type row for each option
    repeated_rows = pd.concat([question_type_row] * num_options, ignore_index=True)
    expanded_question_type_rows.append(repeated_rows)

# Concatenate the list of repeated rows into a single DataFrame
expanded_question_type_one_hot = pd.concat(expanded_question_type_rows, ignore_index=True)


if final_sparse_matrix.shape[0] == expanded_question_type_one_hot.shape[0]:
    # Convert the expanded one-hot encoded DataFrame to a sparse SciPy matrix for stacking
    # Use coo_matrix directly on the DataFrame values
    expanded_question_type_one_hot_sparse = coo_matrix(expanded_question_type_one_hot.astype(float).values)

    # Combine the final_sparse_matrix with the expanded one-hot encoded question type features
    final_features_with_type = hstack([final_sparse_matrix, expanded_question_type_one_hot_sparse])

    print("Shape of the final feature matrix including question type:", final_features_with_type.shape)
else:
    print("Error: The number of rows in final_sparse_matrix and expanded_question_type_one_hot do not match after expansion.")


from sklearn.model_selection import train_test_split
import numpy as np

# Example: If your data includes both features and target
# Assuming the last column is your target variable
X = final_sparse_matrix[:, :-1]  # All columns except last
y = final_sparse_matrix[:, -1]   # Last column as target

# Or if you're working with a DataFrame:
# X = your_dataframe.drop('target_column', axis=1)
# y = your_dataframe['target_column']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


from sklearn.model_selection import train_test_split

# Assuming you have a variable 'y' containing your labels
# y should have a shape (n_samples,) where n_samples is the number of rows in final_sparse_matrix

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    final_sparse_matrix,
    y,  # Your labels goes here
    test_size=0.2,  # Example: 20% for testing
    random_state=42  # Example: for reproducibility
)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


from scipy.sparse import vstack

# Convert the list of sparse matrices into a single sparse matrix
final_sparse_matrix = vstack(combined_tfidf_vectors)

# Display the shape of the final sparse matrix
print("Shape of the final combined sparse matrix:", final_sparse_matrix.shape)

# You can now use final_sparse_matrix as the feature matrix for your model


# Calculate the number of tokens for each question
model_input_df_combined['num_tokens_question'] = model_input_df_combined['tokenized_question'].apply(len)

# Calculate the average number of tokens per question
average_tokens_per_question = model_input_df_combined['num_tokens_question'].mean()

print(f"Average number of tokens per question: {average_tokens_per_question:.2f}")


from scipy.sparse import hstack

# Concatenate the question and option TF-IDF vectors for each pair
# We'll store the results in a list of sparse matrices
combined_tfidf_vectors = []

for question_vector, option_vector in tfidf_pairs:
    # Concatenate the sparse vectors horizontally
    combined_vector = hstack([question_vector, option_vector])
    combined_tfidf_vectors.append(combined_vector)

# combined_tfidf_vectors is now a list of sparse matrices, where each matrix
# represents the concatenated TF-IDF features for a question-option pair.
# You might need to convert this list into a single sparse matrix or a dense
# NumPy array depending on your model's input requirements.

print(f"Generated {len(combined_tfidf_vectors)} combined TF-IDF vectors.")
# Example: Display the shape of the first combined vector
if len(combined_tfidf_vectors) > 0:
    print(f"Shape of first combined TF-IDF vector: {combined_tfidf_vectors[0].shape}")

# Note: To use this with scikit-learn models, you might need to convert the list
# of sparse matrices into a single sparse matrix using scipy.sparse.vstack
# or convert to a dense array if memory allows:
# from scipy.sparse import vstack
# final_sparse_matrix = vstack(combined_tfidf_vectors)
# final_dense_array = final_sparse_matrix.todense()


import numpy as np

# Get the TF-IDF matrix for questions and the vectorizer
tfidf_matrix_questions = tfidf_vectorizer.fit_transform(model_input_df_combined['lemmatized_question_str'])
question_tfidf_vectorizer = tfidf_vectorizer # Keep the vectorizer to transform questions if needed later

# Get the TF-IDF matrix for options and the vectorizer
# Fit on all options globally to have a consistent vocabulary
all_lemmatized_options_str = []
for options_list in model_input_df_combined['lemmatized_options']:
    for option_tokens in options_list:
        all_lemmatized_options_str.append(' '.join(option_tokens))

tfidf_vectorizer_options = TfidfVectorizer(max_features=5000)
tfidf_matrix_options = tfidf_vectorizer_options.fit_transform(all_lemmatized_options_str)
option_tfidf_vectorizer = tfidf_vectorizer_options # Keep the vectorizer

# Create a list to store the (question_tfidf_vector, option_tfidf_vector) pairs
tfidf_pairs = []
labels = [] # Assuming you have labels for which option is correct (you would add this based on your ground truth data)

option_index = 0 # To keep track of the index in the flattened options matrix

for index, row in model_input_df_combined.iterrows():
    question_vector = tfidf_matrix_questions[index]
    options_list = row['lemmatized_options'] # Use lemmatized options to match the vectorizer

    for option_tokens in options_list:
        # Get the TF-IDF vector for the current option from the pre-computed matrix
        option_vector = tfidf_matrix_options[option_index]

        # Append the pair of sparse vectors (or convert to dense if needed for your model)
        tfidf_pairs.append((question_vector, option_vector))

        # If you had labels, you would append them here
        # labels.append(your_label_for_this_option)

        option_index += 1

# tfidf_pairs is now a list of tuples, where each tuple contains the sparse TF-IDF vector
# for a question and a sparse TF-IDF vector for one of its options.
# You can further process this list, e.g., convert to a dense NumPy array if required by your model.

print(f"Generated {len(tfidf_pairs)} question-option TF-IDF pairs.")
# Example: Display the shape of the first pair's vectors (sparse matrix shapes)
if len(tfidf_pairs) > 0:
    print(f"Shape of first question TF-IDF vector: {tfidf_pairs[0][0].shape}")
    print(f"Shape of first option TF-IDF vector: {tfidf_pairs[0][1].shape}")

# Note: You would typically concatenate or combine these vectors based on your model's input layer.
# For example: combined_vector = np.hstack([question_vector.todense(), option_vector.todense()])


# Display the shape of the tfidf_matrix_options
print("Shape of tfidf_matrix_options:", tfidf_matrix_options.shape)


# Flatten the list of lists of lemmatized options into a single list of strings
# Each string is a lemmatized option
all_lemmatized_options_str = []
for options_list in model_input_df_combined['lemmatized_options']:
    for option_tokens in options_list:
        all_lemmatized_options_str.append(' '.join(option_tokens))

# Initialize TfidfVectorizer for options (can use the same vocabulary as questions or a new one)
# For simplicity, let's use a new one here, but combining is also an option
tfidf_vectorizer_options = TfidfVectorizer(max_features=5000) # Adjust parameters as needed

# Fit and transform all lemmatized options
tfidf_matrix_options = tfidf_vectorizer_options.fit_transform(all_lemmatized_options_str)

# Display the shape of the resulting TF-IDF matrix for options
print("Shape of TF-IDF matrix for all options:", tfidf_matrix_options.shape)

# Note: This matrix represents all individual options. You would need to structure this
# appropriately for your model input, e.g., by associating options back to their questions.


from sklearn.feature_extraction.text import TfidfVectorizer

# Convert the list of tokens back to strings for TF-IDF
model_input_df_combined['lemmatized_question_str'] = model_input_df_combined['lemmatized_question'].apply(lambda tokens: ' '.join(tokens))

# Initialize TfidfVectorizer
# You can adjust parameters like max_features, min_df, max_df, ngram_range
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Example: consider top 5000 terms

# Fit and transform the lemmatized questions
tfidf_matrix_questions = tfidf_vectorizer.fit_transform(model_input_df_combined['lemmatized_question_str'])

# Display the shape of the resulting TF-IDF matrix and the first few feature names
print("Shape of TF-IDF matrix for questions:", tfidf_matrix_questions.shape)
print("\nFirst 50 feature names (terms):")
print(tfidf_vectorizer.get_feature_names_out()[:50])


# Apply lemmatization to the tokenized options
model_input_df_combined['lemmatized_options'] = model_input_df_combined['tokenized_options'].apply(lambda options_list: [spacy_lemmatize(option_tokens) for option_tokens in options_list])

# Display the first few rows with the new column
print("DataFrame with lemmatized options (first 5 rows):")
display(model_input_df_combined[['tokenized_options', 'lemmatized_options']].head())


# Function to lemmatize a list of tokens using spaCy
def spacy_lemmatize(tokens):
    # Recreate a spaCy Doc from the tokens to access lemma_
    doc = nlp(" ".join(tokens)) # Join tokens back to string for nlp processing
    return [token.lemma_ for token in doc]

# Apply lemmatization to the tokenized questions
model_input_df_combined['lemmatized_question'] = model_input_df_combined['tokenized_question'].apply(spacy_lemmatize)

# Display the first few rows with the new column
print("DataFrame with lemmatized questions (first 5 rows):")
display(model_input_df_combined[['tokenized_question', 'lemmatized_question']].head())


# Get the list of stop words from spaCy
stop_words = spacy.lang.en.STOP_WORDS

# Function to remove stop words from a list of tokens
def remove_stop_words(tokens):
    return [token for token in tokens if token.lower() not in stop_words]

# Apply stop word removal to the tokenized questions
model_input_df_combined['question_no_stopwords'] = model_input_df_combined['tokenized_question'].apply(remove_stop_words)

# Display the first few rows with the new column
print("DataFrame with tokenized questions after removing stop words (first 5 rows):")
display(model_input_df_combined[['tokenized_question', 'question_no_stopwords']].head())


import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to tokenize text using spaCy
def spacy_tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

# Apply tokenization to the 'question' column
model_input_df_combined['tokenized_question'] = model_input_df_combined['question'].apply(spacy_tokenize)

# Apply tokenization to each option in the 'options' list
model_input_df_combined['tokenized_options'] = model_input_df_combined['options'].apply(lambda options_list: [spacy_tokenize(option) for option in options_list])

# Display the first few rows with the new tokenized columns
print("DataFrame with spaCy tokenized questions and options (first 5 rows):")
display(model_input_df_combined.head())


# Install spaCy
%pip install spacy

# Download a spaCy model
!python -m spacy download en_core_web_sm


# Check for any missing values before tokenization
print("Missing values in 'question' column:", model_input_df_combined['question'].isnull().sum())
print("Missing values in 'options' column:", model_input_df_combined['options'].isnull().sum())

# Handle missing values if any
model_input_df_combined = model_input_df_combined.dropna(subset=['question', 'options'])

# Optional: Add token counts for analysis
model_input_df_combined['question_token_count'] = model_input_df_combined['tokenized_question'].apply(len)
model_input_df_combined['options_token_count'] = model_input_df_combined['tokenized_options'].apply(
    lambda options: [len(option) for option in options]
)

# Display token count statistics
print("\nQuestion token count statistics:")
print(model_input_df_combined['question_token_count'].describe())

print("\nSample of tokenized data:")
for i in range(min(3, len(model_input_df_combined))):
    print(f"\nRow {i}:")
    print(f"Original question: {model_input_df_combined.iloc[i]['question']}")
    print(f"Tokenized question: {model_input_df_combined.iloc[i]['tokenized_question']}")
    print(f"Original options: {model_input_df_combined.iloc[i]['options']}")
    print(f"Tokenized options: {model_input_df_combined.iloc[i]['tokenized_options']}")


# Check the data types
print("Data types:")
print(model_input_df_combined[['question', 'options']].dtypes)

# Check if 'options' is actually a list
print("\nSample of options column:")
print(model_input_df_combined['options'].iloc[0])
print("Type:", type(model_input_df_combined['options'].iloc[0]))

# If options is stored as string instead of list, you might need to evaluate it
import ast
if isinstance(model_input_df_combined['options'].iloc[0], str):
    model_input_df_combined['options'] = model_input_df_combined['options'].apply(ast.literal_eval)


# Enhanced tokenization function
def enhanced_tokenize_text(text):
    """
    Enhanced tokenization with lowercase conversion
    """
    if not isinstance(text, str):
        return []
    return nltk.word_tokenize(text.lower())

# Or using spaCy for more advanced tokenization
# import spacy
# nlp = spacy.load("en_core_web_sm")
#
# def spacy_tokenize(text):
#     doc = nlp(text)
#     return [token.text for token in doc]


# 1. Verify the tokenization worked correctly
print("\nVerifying tokenization:")
print("Original question:", model_input_df_combined['question'].iloc[0])
print("Tokenized question:", model_input_df_combined['tokenized_question'].iloc[0])
print("Original options:", model_input_df_combined['options'].iloc[0])
print("Tokenized options:", model_input_df_combined['tokenized_options'].iloc[0])

# 2. Check token counts for analysis
model_input_df_combined['question_token_count'] = model_input_df_combined['tokenized_question'].apply(len)
model_input_df_combined['options_token_count'] = model_input_df_combined['tokenized_options'].apply(
    lambda options: sum(len(option) for option in options)
)

print(f"\nAverage tokens per question: {model_input_df_combined['question_token_count'].mean():.2f}")
print(f"Average tokens per option set: {model_input_df_combined['options_token_count'].mean():.2f}")

# 3. Check for any empty tokenizations (shouldn't happen with your data)
empty_questions = model_input_df_combined['tokenized_question'].apply(len) == 0
empty_options = model_input_df_combined['tokenized_options'].apply(lambda x: any(len(opt) == 0 for opt in x))

print(f"Empty tokenized questions: {empty_questions.sum()}")
print(f"Options with empty tokenization: {empty_options.sum()}")


# Lowercasing (if you want case-insensitive processing)
model_input_df_combined['tokenized_question_lower'] = model_input_df_combined['tokenized_question'].apply(
    lambda tokens: [token.lower() for token in tokens]
)

# Remove punctuation (if needed)
import string
def remove_punctuation(tokens):
    return [token for token in tokens if token not in string.punctuation]

model_input_df_combined['tokenized_question_clean'] = model_input_df_combined['tokenized_question'].apply(remove_punctuation)

# For model input, you might want to convert tokens to indices using a vocabulary
from sklearn.feature_extraction.text import CountVectorizer

# Flatten tokenized questions back to strings for vectorization
model_input_df_combined['question_text'] = model_input_df_combined['tokenized_question'].apply(
    lambda tokens: ' '.join(tokens)
)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(model_input_df_combined['question_text'])
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


# DataFrame with tokenized questions and options (first 5 rows):
# Total rows: [number_of_rows]


# Check if the DataFrame exists
if 'model_input_df_combined' not in locals():
    print("DataFrame not found. Please make sure it's defined.")
    # You might need to load your data first
    # model_input_df_combined = pd.read_csv('your_file.csv')


# Check available columns
print("Available columns:", model_input_df_combined.columns.tolist())

# Check if required columns exist
required_columns = ['question', 'options']
missing_columns = [col for col in required_columns if col not in model_input_df_combined.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")


import nltk
try:
    nltk.download('punkt')
except Exception as e:
    print(f"NLTK download error: {e}")
    # Alternative: use split() as simple tokenizer
    def tokenize_text(text):
        return text.split()


import nltk
import pandas as pd

try:
    nltk.download('punkt', quiet=True)
    print("NLTK punkt downloaded successfully")
except:
    print("Using fallback tokenizer")

def tokenize_text(text):
    """Tokenize text with error handling"""
    if not isinstance(text, str):
        return []
    try:
        return nltk.word_tokenize(text)
    except:
        # Fallback to simple whitespace tokenization
        return text.split()

# Verify DataFrame exists and has required columns
if 'model_input_df_combined' in locals() and hasattr(model_input_df_combined, 'columns'):
    required_columns = ['question', 'options']
    if all(col in model_input_df_combined.columns for col in required_columns):

        # Apply tokenization
        model_input_df_combined['tokenized_question'] = model_input_df_combined['question'].apply(tokenize_text)

        model_input_df_combined['tokenized_options'] = model_input_df_combined['options'].apply(
            lambda options_list: [tokenize_text(option) for option in options_list]
            if isinstance(options_list, list) else []
        )

        # Display results
        print("DataFrame with tokenized questions and options (first 5 rows):")
        print(f"Total rows: {len(model_input_df_combined)}")
        print(f"Columns: {model_input_df_combined.columns.tolist()}")
        display(model_input_df_combined.head())

    else:
        print("Required columns not found in DataFrame")
else:
    print("DataFrame 'model_input_df_combined' not found")


# Check data types and sample data
print("DataFrame info:")
print(f"Shape: {model_input_df_combined.shape}")
print(f"Columns: {model_input_df_combined.columns.tolist()}")
print("\nData types:")
print(model_input_df_combined.dtypes)
print("\nFirst few rows of original data:")
display(model_input_df_combined[['question', 'options']].head(2))

# Check if options column contains actual lists
print("\nSample of options column:")
sample_options = model_input_df_combined['options'].iloc[0]
print(f"Type: {type(sample_options)}")
print(f"Content: {sample_options}")


# Check if the tokenized columns exist
if 'tokenized_question' in model_input_df_combined.columns and 'tokenized_options' in model_input_df_combined.columns:
    print("âœ… Tokenization completed successfully!")
    print(f"Shape of DataFrame: {model_input_df_combined.shape}")

    # Show a sample of the tokenized data
    print("\nğŸ“Š Sample of tokenized data:")
    for i in range(min(2, len(model_input_df_combined))):
        print(f"\nRow {i}:")
        print(f"Question: {model_input_df_combined['question'].iloc[i][:100]}...")
        print(f"Tokenized: {model_input_df_combined['tokenized_question'].iloc[i][:10]}...")  # First 10 tokens
        print(f"Options: {model_input_df_combined['options'].iloc[i]}")
        print(f"Tokenized options: {[[tokens[:5] for tokens in model_input_df_combined['tokenized_options'].iloc[i]]]}")  # First 5 tokens per option
else:
    print("â�Œ Tokenized columns not found. Running tokenization...")

    # Your tokenization code here
    import nltk
    nltk.download('punkt')

    def tokenize_text(text):
        return nltk.word_tokenize(text)

    model_input_df_combined['tokenized_question'] = model_input_df_combined['question'].apply(tokenize_text)
    model_input_df_combined['tokenized_options'] = model_input_df_combined['options'].apply(
        lambda options_list: [tokenize_text(option) for option in options_list]
    )


# 1. Analyze token statistics
print("\nğŸ”� Token Analysis:")
model_input_df_combined['question_token_count'] = model_input_df_combined['tokenized_question'].apply(len)
model_input_df_combined['total_options_tokens'] = model_input_df_combined['tokenized_options'].apply(
    lambda options: sum(len(opt) for opt in options)
)

print(f"Average tokens per question: {model_input_df_combined['question_token_count'].mean():.2f}")
print(f"Average total tokens per options: {model_input_df_combined['total_options_tokens'].mean():.2f}")
print(f"Max tokens in a question: {model_input_df_combined['question_token_count'].max()}")
print(f"Min tokens in a question: {model_input_df_combined['question_token_count'].min()}")

# 2. Check for any potential issues
empty_tokens = model_input_df_combined['question_token_count'] == 0
if empty_tokens.any():
    print(f"âš ï¸�  Found {empty_tokens.sum()} questions with no tokens")

# 3. Preview the tokenized data
print("\nğŸ“‹ First 3 rows preview:")
display(model_input_df_combined[['question', 'tokenized_question', 'options', 'tokenized_options']].head(3))


# Option A: Convert to vocabulary indices for neural networks
from sklearn.feature_extraction.text import CountVectorizer
import itertools

# Create a vocabulary from all tokens
all_question_tokens = list(itertools.chain(*model_input_df_combined['tokenized_question']))
all_option_tokens = list(itertools.chain(*itertools.chain(*model_input_df_combined['tokenized_options'])))
all_tokens = all_question_tokens + all_option_tokens

print(f"Total unique tokens: {len(set(all_tokens))}")

# Option B: For transformer models, you might want to use their tokenizers
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Option C: Create word embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert tokens back to text for vectorization
model_input_df_combined['question_text'] = model_input_df_combined['tokenized_question'].apply(' '.join)
vectorizer = TfidfVectorizer(max_features=5000)
X_questions = vectorizer.fit_transform(model_input_df_combined['question_text'])
print(f"TF-IDF matrix shape: {X_questions.shape}")


from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

print("ğŸš¨ Using Anomaly Detection (recommended for your data distribution)")

# Treat the single non-zero value as an anomaly
model = IsolationForest(
    contamination=0.0001,  # Very low contamination rate
    random_state=42,
    n_estimators=100
)

# Fit the model (unsupervised)
model.fit(X_train)

# Predict anomalies (-1 for anomalies, 1 for normal)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert to binary (0 for normal, 1 for anomaly)
y_train_pred_binary = (y_train_pred == -1).astype(int)
y_test_pred_binary = (y_test_pred == -1).astype(int)

print("Anomaly Detection Results:")
print(f"Training - Predicted anomalies: {np.sum(y_train_pred_binary)}")
print(f"Testing - Predicted anomalies: {np.sum(y_test_pred_binary)}")


from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

print("ğŸš¨ Using Anomaly Detection (recommended for your data distribution)")

# Treat the single non-zero value as an anomaly
model = IsolationForest(
    contamination=0.0001,  # Very low contamination rate
    random_state=42,
    n_estimators=100
)

# Fit the model (unsupervised) using X_train, which should include question types
model.fit(X_train)

# Predict anomalies (-1 for anomalies, 1 for normal)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert to binary (0 for normal, 1 for anomaly)
y_train_pred_binary = (y_train_pred == -1).astype(int)
y_test_pred_binary = (y_test_pred == -1).astype(int)

print("Anomaly Detection Results:")
print(f"Training - Predicted anomalies: {np.sum(y_train_pred_binary)}")
print(f"Testing - Predicted anomalies: {np.sum(y_test_pred_binary)}")


from sklearn.utils import resample
import numpy as np

# Assuming you have your training data X_train and y_train
# X_train is your training features, y_train is your training labels

# Convert y_train to a dense array and flatten before comparison and indexing
y_train_dense = y_train.toarray().ravel() if hasattr(y_train, 'toarray') else y_train.ravel()

# Separate majority and minority classes
# Use integer comparison after potential rounding if needed, but be aware of label meaning
# Based on unique values [0. 0.17380664], 0 is majority. Let's assume anything non-zero is minority for this code.
majority_indices = np.where(y_train_dense == 0)[0]
minority_indices = np.where(y_train_dense != 0)[0] # Treat non-zero as minority

# Get the corresponding data from X_train
majority_class_X = X_train[majority_indices]
minority_class_X = X_train[minority_indices]

majority_labels = y_train_dense[majority_indices]
minority_labels = y_train_dense[minority_indices]


# Upsample the minority class
if minority_class_X.shape[0] > 0: # Check number of rows for sparse matrix
    minority_upsampled_X = resample(
        minority_class_X,
        replace=True,     # sample with replacement
        n_samples=majority_class_X.shape[0],  # match majority class (use shape[0] for sparse matrix)
        random_state=42
    )
    # Create labels for upsampled minority (assign the minority value or a representative label)
    # Assuming minority value is 0.1738... or similar, let's assign a single label like 1 for upsampled.
    minority_upsampled_y = np.ones(minority_upsampled_X.shape[0]) * np.unique(minority_labels)[0] if len(np.unique(minority_labels)) > 0 else np.ones(minority_upsampled_X.shape[0]) # Use the actual minority value or 1


    # Combine majority class with upsampled minority class
    # Convert sparse to dense if necessary for hstack/vstack depending on model input
    X_balanced = np.vstack((majority_class_X.toarray(), minority_upsampled_X.toarray()))
    y_balanced = np.hstack((majority_labels, minority_upsampled_y))

    # Now fit your model (assuming model is defined)
    # model.fit(X_balanced, y_balanced)
    print("Data balanced. X_balanced and y_balanced created.")
    print("Shape of X_balanced:", X_balanced.shape)
    print("Shape of y_balanced:", y_balanced.shape)

else:
    print("No minority class found to upsample.")
    # If no minority, balanced data is just the original training data
    X_balanced = X_train
    y_balanced = y_train_dense
    print("No minority class found. Balanced data is the original training data.")
    print("Shape of X_balanced:", X_balanced.shape)
    print("Shape of y_balanced:", y_balanced.shape)


import numpy as np
from scipy.sparse import issparse

# Convert any sparse matrices to dense arrays and ensure consistent dimensions
X_dense = X.toarray() if issparse(X) else np.array(X)
y_dense = y.toarray().ravel() if issparse(y) else np.array(y).ravel()

# Check and fix dimension mismatch
print(f"X shape: {X_dense.shape}, y shape: {y_dense.shape}")

# If dimensions don't match, fix the issue
if len(X_dense) != len(y_dense):
    print("Fixing dimension mismatch...")
    # Take the minimum length to ensure consistency
    min_length = min(len(X_dense), len(y_dense))
    X_dense = X_dense[:min_length]
    y_dense = y_dense[:min_length]
    print(f"Fixed shapes - X: {X_dense.shape}, y: {y_dense.shape}")

# Train model with class weighting to handle imbalance
model.fit(X_dense, y_dense)
print("Model trained successfully!")


import numpy as np
from scipy.sparse import issparse

# Handle the dimension mismatch
try:
    # Convert to proper arrays
    X_array = X.toarray() if issparse(X) else np.array(X)
    y_array = y.toarray().ravel() if issparse(y) else np.array(y).ravel()

    # Ensure consistent sample count
    if len(X_array) != len(y_array):
        # If X has fewer samples, it might be transposed
        if X_array.shape[0] == y_array.shape[0]:
            # Shapes already match
            pass
        elif X_array.shape[1] == y_array.shape[0]:
            # X might be transposed - transpose it
            X_array = X_array.T
        else:
            # Use the minimum common length
            min_len = min(len(X_array), len(y_array))
            X_array = X_array[:min_len]
            y_array = y_array[:min_len]

    print(f"Final shapes - X: {X_array.shape}, y: {y_array.shape}")

    # Train the model
    model.fit(X_array, y_array)
    print("âœ… Model training completed successfully!")

except Exception as e:
    print(f"Error: {e}")
    # Last resort: debug the shapes
    print(f"Debug - X type: {type(X)}, shape: {getattr(X, 'shape', 'No shape')}")
    print(f"Debug - y type: {type(y)}, shape: {getattr(y, 'shape', 'No shape')}")


import numpy as np
from scipy.sparse import issparse

# Debug the original data
print("Original data shapes:")
print(f"X type: {type(X)}, shape: {X.shape if hasattr(X, 'shape') else 'No shape'}")
print(f"y type: {type(y)}, shape: {y.shape if hasattr(y, 'shape') else 'No shape'}")

# Force consistent shapes
min_samples = min(X.shape[0] if hasattr(X, 'shape') else len(X),
                  y.shape[0] if hasattr(y, 'shape') else len(y))

# Convert and trim to consistent size
if issparse(X):
    X_final = X[:min_samples].toarray()
else:
    X_final = np.array(X)[:min_samples]

if issparse(y):
    y_final = y[:min_samples].toarray().ravel()
else:
    y_final = np.array(y)[:min_samples].ravel()

print(f"Final training shapes - X: {X_final.shape}, y: {y_final.shape}")

# Train the model
model.fit(X_final, y_final)
print("âœ… Model trained successfully!")





from sklearn.model_selection import train_test_split
import numpy as np # Import numpy for potential array handling

# Assuming you have a variable 'y' containing your CORRECT discrete labels
# y should have a shape (n_samples,) where n_samples is the number of rows in final_features_with_type

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    final_features_with_type,  # Use the combined feature matrix
    y,  # Your CORRECT discrete labels goes here
    test_size=0.2,  # Example: 20% for testing
    random_state=42  # Example: for reproducibility
)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

print("ğŸš¨ Using Anomaly Detection (recommended for your data distribution)")

# Treat the single non-zero value as an anomaly
model = IsolationForest(
    contamination=0.0001,  # Very low contamination rate
    random_state=42,
    n_estimators=100
)

# Fit the model (unsupervised) using X_train, which should include question types
model.fit(X_train)

# Predict anomalies (-1 for anomalies, 1 for normal)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert to binary (0 for normal, 1 for anomaly)
y_train_pred_binary = (y_train_pred == -1).astype(int)
y_test_pred_binary = (y_test_pred == -1).astype(int)

print("Anomaly Detection Results:")
print(f"Training - Predicted anomalies: {np.sum(y_train_pred_binary)}")
print(f"Testing - Predicted anomalies: {np.sum(y_test_pred_binary)}")


from sklearn.linear_model import LogisticRegression
import numpy as np

print("ğŸ”§ Manual Binary Classification Approach")

# Create binary labels: 0 for zeros, 1 for the non-zero value
# Ensure y_train_dense is a dense array
y_train_dense = y_train.toarray().ravel() if hasattr(y_train, 'toarray') else y_train.ravel()
y_test_dense = y_test.toarray().ravel() if hasattr(y_test, 'toarray') else y_test.ravel()

y_train_binary = (y_train_dense != 0).astype(int)
y_test_binary = (y_test_dense != 0).astype(int)

print(f"Class distribution: {dict(zip(*np.unique(y_train_binary, return_counts=True)))}")

# Since we have only 1 positive sample, we need special handling
if np.sum(y_train_binary) == 1:
    print("âš ï¸�  Only one positive sample detected - using special training strategy")

    # Find the positive sample index
    positive_idx = np.where(y_train_binary == 1)[0][0]

    # Ensure X_train is treated as an array for slicing/repeating and convert to dense
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    if not isinstance(X_train_dense, np.ndarray):
         X_train_dense = np.array(X_train_dense) # Ensure it's a NumPy array

    # Get the positive sample
    X_positive = X_train_dense[positive_idx:positive_idx+1]  # Single row

    # Create balanced dataset with equal samples per class
    n_positive_samples = 100  # Number of positive samples we want
    n_negative_samples = 100  # Equal number of negative samples

    # Get subset of negative samples
    negative_indices = np.where(y_train_binary == 0)[0][:n_negative_samples]
    X_negative_subset = X_train_dense[negative_indices]
    y_negative_subset = y_train_binary[negative_indices]

    # Create copies of the positive sample
    X_positive_copies = np.repeat(X_positive, n_positive_samples, axis=0)
    y_positive_copies = np.ones(n_positive_samples)

    # Create balanced dataset
    X_balanced = np.vstack([X_negative_subset, X_positive_copies])
    y_balanced = np.hstack([y_negative_subset, y_positive_copies])

    print(f"Balanced dataset shape: {X_balanced.shape}")
    print(f"Balanced class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")

    # Train on balanced data
    model = LogisticRegression(
        random_state=42,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced'
    )

    model.fit(X_balanced, y_balanced)

else:
    # Normal training if we have more positive samples
    X_train_array = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    if not isinstance(X_train_array, np.ndarray):
         X_train_array = np.array(X_train_array)

    model = LogisticRegression(
        random_state=42,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train_array, y_train_binary)

# Evaluate
X_test_array = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
if not isinstance(X_test_array, np.ndarray):
     X_test_array = np.array(X_test_array)

y_pred = model.predict(X_test_array)
print(f"Test predictions - Class 1: {np.sum(y_pred)}, Class 0: {np.sum(y_pred == 0)}")


from sklearn.linear_model import LogisticRegression
import numpy as np

# Define the Logistic Regression model
# You might need to adjust parameters like C, solver, max_iter based on your data and performance
model = LogisticRegression(random_state=42, solver='liblinear') # Using liblinear for potential large sparse data

# Train the model using the training data
print("Training the Logistic Regression model...")

# Ensure y_train is a dense 1D array with discrete labels
if hasattr(y_train, 'toarray'): # Check if it's a sparse matrix
    y_train_dense = y_train.toarray().ravel()
else:
    y_train_dense = y_train.ravel() # Assume it's already dense, just flatten

# Check unique values and type to give a more specific error message if labels are still continuous
unique_labels = np.unique(y_train_dense)
if np.issubdtype(y_train_dense.dtype, np.floating) and len(unique_labels) > 2 or (len(unique_labels) == 2 and not all(np.isclose(unique_labels, [0, 1]))):
     print("Error: y_train still contains continuous or unexpected non-binary labels. Please ensure it has discrete labels (e.g., integers).")
elif np.issubdtype(y_train_dense.dtype, np.floating) and len(unique_labels) <= 2:
     # It might be binary but as floats (e.g., 0.0, 1.0). Convert to int.
     print("Warning: y_train appears to be binary but as floats. Attempting to convert to integers.")
     y_train_discrete = y_train_dense.astype(int)
     model.fit(X_train, y_train_discrete)
else:
    # Assume it's already discrete (integers or objects/strings)
    model.fit(X_train, y_train_dense)

print("Model training complete.")


# Display the shape of X_train and X_test
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)



from sklearn.linear_model import LogisticRegression
import numpy as np

print("ğŸ”§ Simple Binary Classification with Class Weights")

# Create binary labels
y_train_dense = y_train.toarray().ravel() if hasattr(y_train, 'toarray') else y_train.ravel()
y_test_dense = y_test.toarray().ravel() if hasattr(y_test, 'toarray') else y_test.ravel()

y_train_binary = (y_train_dense != 0).astype(int)
y_test_binary = (y_test_dense != 0).astype(int)

print(f"Class distribution: {dict(zip(*np.unique(y_train_binary, return_counts=True)))}")

# Convert X to dense arrays
X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

if not isinstance(X_train_dense, np.ndarray):
    X_train_dense = np.array(X_train_dense)
if not isinstance(X_test_dense, np.ndarray):
    X_test_dense = np.array(X_test_dense)

# For extreme imbalance with only 1 positive sample, use strong class weighting
model = LogisticRegression(
    random_state=42,
    solver='liblinear',
    max_iter=1000,
    class_weight={0: 1, 1: 1000},  # Heavy weight for the single positive class
    C=0.01,  # Strong regularization
    penalty='l2'
)

# Train on original data (no manual balancing needed)
model.fit(X_train_dense, y_train_binary)

# Evaluate
y_pred = model.predict(X_test_dense)
print(f"Test predictions - Class 1: {np.sum(y_pred)}, Class 0: {np.sum(y_pred == 0)}")


import numpy as np
from scipy.sparse import issparse

# Convert any sparse matrices to dense arrays
X_dense = X.toarray() if issparse(X) else np.array(X)
y_dense = y.toarray().ravel() if issparse(y) else np.array(y).ravel()

# Train model with class weighting to handle imbalance
model.fit(X_dense, y_dense)


from sklearn.linear_model import LogisticRegression
import numpy as np

print("ğŸ”§ Manual Binary Classification Approach")

# Create binary labels: 0 for zeros, 1 for the non-zero value
# Ensure y_train_dense is a dense array
y_train_dense = y_train.toarray().ravel() if hasattr(y_train, 'toarray') else y_train.ravel()
y_test_dense = y_test.toarray().ravel() if hasattr(y_test, 'toarray') else y_test.ravel()


y_train_binary = (y_train_dense != 0).astype(int)
y_test_binary = (y_test_dense != 0).astype(int)

print(f"Class distribution: {dict(zip(*np.unique(y_train_binary, return_counts=True)))}")

# Since we have only 1 positive sample, we need special handling
if np.sum(y_train_binary) == 1:
    print("âš ï¸�  Only one positive sample detected - using special training strategy")

    # Find the positive sample index
    positive_idx = np.where(y_train_binary == 1)[0][0]

    # Create reasonable number of copies (much smaller than negative samples)
    n_copies = 100

    # Get the positive sample (ensure it's a proper 2D array with correct feature dimension)
    X_train_array = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    if not isinstance(X_train_array, np.ndarray):
         X_train_array = np.array(X_train_array)

    X_positive = X_train_array[positive_idx:positive_idx+1] # This slice should be (1, n_features)


    # Create copies of the positive sample (dense, should be (n_copies, n_features))
    X_positive_copies = np.repeat(X_positive, n_copies, axis=0)

    # Use a SUBSET of negative samples (not all of them)
    negative_indices = np.where(y_train_binary == 0)[0]
    n_negatives = min(1000, len(negative_indices))  # Use only 1000 negative samples max

    # Get subset of negative samples (ensure it's a proper 2D array with correct feature dimension)
    X_negative_subset = X_train_array[negative_indices[:n_negatives]]
    y_negative_subset = y_train_binary[negative_indices[:n_negatives]]

    # Create balanced dataset
    # Ensure both parts are dense NumPy arrays with consistent feature dimensions before stacking
    if X_negative_subset.shape[1] == X_positive_copies.shape[1]:
        X_balanced = np.vstack([X_negative_subset, X_positive_copies])
        y_balanced = np.hstack([y_negative_subset, np.ones(n_copies)])

        print(f"Balanced dataset shape: {X_balanced.shape}")
        print(f"Balanced class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")

        # Train on balanced data
        model = LogisticRegression(
            random_state=42,
            solver='liblinear',
            max_iter=1000,
            class_weight='balanced'
        )

        model.fit(X_balanced, y_balanced)
        print("Model training complete on balanced data.")

    else:
        print(f"Error: Feature dimensions mismatch before stacking. Negative subset features: {X_negative_subset.shape[1]}, Positive copies features: {X_positive_copies.shape[1]}")


else:
    # Normal training if we have more positive samples
    # Ensure X_train is treated as an array for fitting
    X_train_array = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    if not isinstance(X_train_array, np.ndarray):
         X_train_array = np.array(X_train_array)

    model = LogisticRegression(
        random_state=42,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train_array, y_train_binary)
    print("Model training complete on original training data.")


# Evaluate
# Ensure X_test is treated as an array for prediction
X_test_array = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
if not isinstance(X_test_array, np.ndarray):
     X_test_array = np.array(X_test_array)

# Ensure y_test_binary is 1D
y_test_binary_flat = y_test_binary.ravel() if hasattr(y_test_binary, 'ravel') else y_test_binary

y_pred = model.predict(X_test_array)
print(f"Test predictions - Class 1: {np.sum(y_pred)}, Class 0: {np.sum(y_pred == 0)}")


from sklearn.linear_model import LogisticRegression
import numpy as np

print("ğŸ”§ Simplified Binary Classification")

# Create binary labels
y_train_binary = (y_train_dense != 0).astype(int)
y_test_binary = (y_test_dense != 0).astype(int)

print(f"Class distribution: {dict(zip(*np.unique(y_train_binary, return_counts=True)))}")

# For extreme imbalance, use class weights directly - no manual balancing needed
model = LogisticRegression(
    random_state=42,
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced'  # This automatically handles the imbalance
)

# Convert to dense arrays if they are sparse
if hasattr(X_train, 'toarray'):
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
else:
    X_train_dense = X_train
    X_test_dense = X_test

# Train on original data
model.fit(X_train_dense, y_train_binary)

# Evaluate
y_pred = model.predict(X_test_dense)
print(f"Test predictions - Class 1: {np.sum(y_pred)}, Class 0: {np.sum(y_pred == 0)}")


from sklearn.linear_model import LogisticRegression

# Define the Logistic Regression model
# You might need to adjust parameters like C, solver, max_iter based on your data and performance
model = LogisticRegression(random_state=42, solver='liblinear') # Using liblinear for potential large sparse data

# Train the model using the training data
print("Training the Logistic Regression model...")

# Ensure y_train is a dense 1D array with discrete labels
if hasattr(y_train, 'toarray'): # Check if it's a sparse matrix
    y_train_dense = y_train.toarray().ravel()
else:
    y_train_dense = y_train.ravel() # Assume it's already dense, just flatten

# Check unique values and type to give a more specific error message if labels are still continuous
unique_labels = np.unique(y_train_dense)
if np.issubdtype(y_train_dense.dtype, np.floating) and len(unique_labels) > 2 or (len(unique_labels) == 2 and not all(np.isclose(unique_labels, [0, 1]))):
     print("Error: y_train still contains continuous or unexpected non-binary labels. Please ensure it has discrete labels (e.g., integers).")
elif np.issubdtype(y_train_dense.dtype, np.floating) and len(unique_labels) <= 2:
     # It might be binary but as floats (e.g., 0.0, 1.0). Convert to int.
     print("Warning: y_train appears to be binary but as floats. Attempting to convert to integers.")
     y_train_discrete = y_train_dense.astype(int)
     model.fit(X_train, y_train_discrete)
else:
    # Assume it's already discrete (integers or objects/strings)
    model.fit(X_train, y_train_dense)

print("Model training complete.")


from sklearn.svm import OneClassSVM

print("ğŸ�¯ Using One-Class SVM")

# Train only on the majority class (zeros)
zero_indices = np.where(y_train_dense == 0)[0]
X_zeros = X_train[zero_indices]

model = OneClassSVM(
    nu=0.01,  # Expected outlier fraction
    kernel='rbf',
    gamma='scale'
)

model.fit(X_zeros)

# Predict outliers
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert to binary (1 for inliers, -1 for outliers â†’ 0 for normal, 1 for anomaly)
y_train_binary = (y_train_pred == -1).astype(int)
y_test_binary = (y_test_pred == -1).astype(int)

print(f"Training anomalies detected: {np.sum(y_train_binary)}")
print(f"Testing anomalies detected: {np.sum(y_test_binary)}")


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# First, let's properly analyze and fix the target variable
print("ğŸ”� Analyzing target variable...")
print(f"y_train_dense shape: {y_train_dense.shape}")
print(f"Unique values: {np.unique(y_train_dense)}")
print(f"Value counts: {dict(zip(*np.unique(y_train_dense, return_counts=True)))}")

# Convert continuous labels to discrete classes
print("\nğŸ”„ Converting to discrete classes...")

# Strategy 1: Handle the single anomalous value (0.1738)
# Since we have 7970 zeros and 1 non-zero, let's treat this as binary classification
y_train_discrete = (y_train_dense != 0).astype(int)  # Convert to binary: 0 vs non-zero
y_test_discrete = (y_test_dense != 0).astype(int)

print(f"After conversion - Unique classes: {np.unique(y_train_discrete)}")
print(f"Class distribution: {dict(zip(*np.unique(y_train_discrete, return_counts=True)))}")

# Check if we have enough samples for cross-validation
if np.sum(y_train_discrete == 1) < 5:  # If less than 5 positive samples
    print("âš ï¸�  Very few positive samples. Using stratified CV might not work well.")

    # Strategy 2: Use different CV strategy or handle the imbalance
    from sklearn.model_selection import KFold

    # Use regular KFold instead of stratified
    cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)  # Reduced splits
    print("Using KFold CV with 3 splits due to class imbalance")
else:
    from sklearn.model_selection import StratifiedKFold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Using StratifiedKFold CV with 5 splits")

# Update the model to handle class imbalance
model_imbalanced = LogisticRegression(
    random_state=42,
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced',  # Important for imbalanced data
    C=1.0
)

# First, let's make sure the model can train on the full data
print("\nğŸ§ª Testing model on full training data...")
try:
    model_imbalanced.fit(X_train, y_train_discrete)
    print("âœ… Model trained successfully on full data")

    # Now try cross-validation
    print("\nğŸ“Š Performing cross-validation...")
    cv_scores = cross_val_score(
        model_imbalanced,
        X_train,
        y_train_discrete,
        cv=cv_strategy,
        scoring='accuracy'
    )
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

except Exception as e:
    print(f"â�Œ Error during training: {e}")

    # Alternative: Manual cross-validation for extreme imbalance
    print("\nğŸ”„ Trying manual cross-validation...")
    manual_cv_scores = []

    for train_idx, val_idx in cv_strategy.split(X_train, y_train_discrete):
        try:
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train_discrete[train_idx], y_train_discrete[val_idx]

            # Check if both classes are present
            if len(np.unique(y_train_fold)) > 1:
                model_fold = LogisticRegression(
                    random_state=42,
                    solver='liblinear',
                    max_iter=1000,
                    class_weight='balanced'
                )
                model_fold.fit(X_train_fold, y_train_fold)
                score = model_fold.score(X_val_fold, y_val_fold)
                manual_cv_scores.append(score)
            else:
                print(f"Skipping fold with only one class")

        except Exception as fold_error:
            print(f"Fold failed: {fold_error}")

    if manual_cv_scores:
        print(f"Manual CV scores: {manual_cv_scores}")
        print(f"Mean manual CV accuracy: {np.mean(manual_cv_scores):.4f}")


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define the Logistic Regression model
model = LogisticRegression(
    random_state=42,
    solver='liblinear',
    max_iter=1000,
    C=1.0
)

# Convert continuous labels to discrete classes
print("Converting continuous labels to discrete classes...")

# Option A: If you have a small number of unique values that should be treated as classes
unique_values = np.unique(y_train_dense)
print(f"Unique values in y_train: {unique_values}")
print(f"Number of unique values: {len(unique_values)}")

if len(unique_values) <= 100:  # If reasonable number of unique values
    # Use LabelEncoder to convert to integers
    label_encoder = LabelEncoder()
    y_train_discrete = label_encoder.fit_transform(y_train_dense)
    y_test_discrete = label_encoder.transform(y_test_dense)

    print(f"Converted to {len(np.unique(y_train_discrete))} classes")
    print(f"Class distribution after encoding: {dict(zip(*np.unique(y_train_discrete, return_counts=True)))}")

else:
    # Option B: If too many unique values, bin them into categories
    print("Too many unique values for classification. Binning into categories...")

    # Bin into 5 categories (adjust based on your needs)
    y_train_discrete = pd.cut(y_train_dense, bins=5, labels=False)
    y_test_discrete = pd.cut(y_test_dense, bins=5, labels=False)

    print(f"Binned into {len(np.unique(y_train_discrete))} categories")
    print(f"Class distribution after binning: {dict(zip(*np.unique(y_train_discrete, return_counts=True)))}")

# Now train the model with discrete labels
print("\nTraining the Logistic Regression model...")
model.fit(X_train, y_train_discrete)
print("Model training complete.")

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report

train_accuracy = accuracy_score(y_train_discrete, y_train_pred)
test_accuracy = accuracy_score(y_test_discrete, y_test_pred)

print(f"\nğŸ“Š Model Performance:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nğŸ“‹ Classification Report:")
print(classification_report(y_test_discrete, y_test_pred))


import joblib

# Save the trained model
joblib.dump(model, 'logistic_regression_model.pkl')
print("Model saved as 'logistic_regression_model.pkl'")

# Load the model later
# loaded_model = joblib.load('logistic_regression_model.pkl')


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of question token counts
plt.figure(figsize=(10, 6))
sns.histplot(model_input_df_combined['question_token_count'], bins=30, kde=True)
plt.title('Distribution of Question Token Counts')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.show()

# Flatten the list of lists in 'options_token_count'
all_options_token_counts = [count for sublist in model_input_df_combined['options_token_count'] for count in sublist]

# Visualize the distribution of option token counts
plt.figure(figsize=(10, 6))
sns.histplot(all_options_token_counts, bins=30, kde=True)
plt.title('Distribution of Option Token Counts')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.show()

# Optionally, display statistics for option token counts
print("\nStatistics about option token counts:")
import numpy as np
print(pd.Series(all_options_token_counts).describe())


import nltk
nltk.download('punkt')


# Use a shell command to list files in the default content directory
!ls /content/


# Display the first few rows and the columns of the loaded test data (df)
print("Contents of the loaded test data (df):")
display(df.head())
display(df.info())


import jsonlines
import pandas as pd

# Load the CURE-Bench training data file
# Assuming 'curebench_training_data.jsonl' is in the default Colab directory (/content/)
training_data = []
try:
    with jsonlines.open('curebench_training_data.jsonl') as reader:
        for obj in reader:
            training_data.append(obj)

    # Convert to a pandas DataFrame
    training_df = pd.DataFrame(training_data)

    print("Training data loaded successfully from curebench_training_data.jsonl.")
    display(training_df.head())
    display(training_df.info())

except FileNotFoundError:
    print("Error: 'curebench_training_data.jsonl' not found. Please make sure the file is uploaded to your Colab environment (e.g., in /content/) or provide the correct path.")
except Exception as e:
    print(f"An error occurred while loading the training data: {e}")


# Install NLTK
%pip install nltk


# Extract options for open_ended questions in the same list format
open_ended_options_list = open_ended_df['options'].apply(lambda x: list(x.values()))

# Create a temporary DataFrame for open_ended questions with structured data
open_ended_model_input_df = pd.DataFrame({
    'id': open_ended_df['id'],
    'question': open_ended_df['question'],
    'options': open_ended_options_list
})

# Add a question_type column to both dataframes before concatenating
model_input_df['question_type'] = 'structured_options' # For multi_choice and open_ended_multi_choice
open_ended_model_input_df['question_type'] = 'open_ended'

# Combine the structured dataframes
model_input_df_combined = pd.concat([model_input_df, open_ended_model_input_df], ignore_index=True)

# Display the first few rows of the combined structured data
print("Combined structured data for model input (first 5 rows):")
display(model_input_df_combined.head())

# Display information about the combined structured data
display(model_input_df_combined.info())

# Display the value counts for the new 'question_type' column to confirm combination
print("\nValue counts for question types in combined dataframe:")
display(model_input_df_combined['question_type'].value_counts())


# Create a new DataFrame with structured data for model input
# Combine multi-choice and open-ended multi-choice questions as they both have structured options
model_input_df = pd.DataFrame({
    'id': pd.concat([multi_choice_df['id'], open_ended_multi_choice_df['id']]),
    'question': pd.concat([multi_choice_df['question'], open_ended_multi_choice_df['question']]),
    'options': pd.concat([multi_choice_options, open_ended_multi_choice_options])
})

# Display the first few rows of the structured data
print("Structured data for model input (first 5 rows):")
display(model_input_df.head())

# Display information about the structured data
display(model_input_df.info())


# Extract answer choices for open_ended_multi_choice questions
open_ended_multi_choice_options = open_ended_multi_choice_df['options'].apply(lambda x: list(x.values()))

# Display the first few extracted options
print("Examples of extracted answer choices for open_ended_multi_choice questions:")
for i in range(min(5, len(open_ended_multi_choice_options))):
    print(f"Row {open_ended_multi_choice_options.index[i]}: {open_ended_multi_choice_options.iloc[i]}")


# Extract answer choices for multi-choice questions
multi_choice_options = multi_choice_df['options'].apply(lambda x: list(x.values()))

# Display the first few extracted options
print("Examples of extracted answer choices for multi-choice questions:")
for i in range(min(5, len(multi_choice_options))):
    print(f"Row {multi_choice_options.index[i]}: {multi_choice_options.iloc[i]}")


# Display examples of the 'options' column for each question type

print("Examples of 'options' for 'multi_choice' questions:")
for i in range(min(3, len(multi_choice_df))):
    print(f"Row {multi_choice_df.index[i]}: {multi_choice_df['options'].iloc[i]}")

print("\nExamples of 'options' for 'open_ended' questions:")
for i in range(min(3, len(open_ended_df))):
    print(f"Row {open_ended_df.index[i]}: {open_ended_df['options'].iloc[i]}")

print("\nExamples of 'options' for 'open_ended_multi_choice' questions:")
for i in range(min(3, len(open_ended_multi_choice_df))):
    print(f"Row {open_ended_multi_choice_df.index[i]}: {open_ended_multi_choice_df['options'].iloc[i]}")


from scipy import stats

# Extract question lengths for each group
multi_choice_lengths = multi_choice_df['question_length']
open_ended_lengths = open_ended_df['question_length']

# Perform independent samples t-test
ttest_result = stats.ttest_ind(multi_choice_lengths, open_ended_lengths)

# Display the results
print(f"Independent Samples t-test results:")
print(f"  Test Statistic: {ttest_result.statistic:.4f}")
print(f"  P-value: {ttest_result.pvalue:.4f}")

# Interpret the results
alpha = 0.05
if ttest_result.pvalue < alpha:
    print("\nInterpretation: The difference in mean question lengths between multi-choice and open-ended questions is statistically significant (p < 0.05).")
else:
    print("\nInterpretation: The difference in mean question lengths between multi-choice and open-ended questions is not statistically significant (p >= 0.05).")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Combine the question lengths from both dataframes for plotting
combined_lengths = pd.concat([
    multi_choice_df[['question_length']].assign(type='multi_choice'),
    open_ended_df[['question_length']].assign(type='open_ended')
])

plt.figure(figsize=(10, 6))
sns.boxplot(data=combined_lengths, x='type', y='question_length')
plt.title('Comparison of Question Lengths by Type')
plt.xlabel('Question Type')
plt.ylabel('Question Length')
plt.show()


# Calculate the length of each question for open_ended_df
open_ended_df.loc[:, 'question_length'] = open_ended_df['question'].apply(len)

# Display statistics about the open_ended question lengths
print("Statistics about open_ended question lengths:")
display(open_ended_df['question_length'].describe())


open_ended_multi_choice_df = df[df['question_type'] == 'open_ended_multi_choice']
display(open_ended_multi_choice_df.head())
display(open_ended_multi_choice_df.info())


open_ended_df = df[df['question_type'] == 'open_ended']
display(open_ended_df.head())
display(open_ended_df.info())


# Display some examples of the 'question' column
print("Examples of 'question' column:")
for i in range(5):
    print(f"Row {i}: {multi_choice_df['question'].iloc[i]}")

print("\nExamples of 'options' column:")
# Display some examples of the 'options' column
for i in range(5):
    print(f"Row {i}: {multi_choice_df['options'].iloc[i]}")


multi_choice_df = df[df['question_type'] == 'multi_choice']
display(multi_choice_df.head())
display(multi_choice_df.info())


# Display the shape of X_train and y_train
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)


%pip install jsonlines


# Example usage in your competition notebook:

"""
# COMPETITION SUBMISSION NOTEBOOK TEMPLATE

# 1. Import the pipeline
from submission import create_prediction_pipeline, train_model, predict, save_model

# 2. Prepare your data (assuming you have final_sparse_matrix and labels)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    final_sparse_matrix,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Create and train model
model = create_prediction_pipeline()
model = train_model(model, X_train, y_train)

# 4. Save the trained model
save_model(model, 'competition_model.pkl')

# 5. Make predictions on test set
predictions, anomaly_scores = predict(model, X_test)

# 6. Format predictions for submission
submission_df = pd.DataFrame({
    'id': test_ids,  # Your test sample IDs
    'prediction': predictions
})

# 7. Save submission file
submission_df.to_csv('submission.csv', index=False)

print("ğŸ�¯ Submission file created: submission.csv")
print(f"ğŸ“Š Predictions - Normal: {(predictions == 0).sum()}, Anomaly: {(predictions == 1).sum()}")
"""


# submission.py
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import joblib
import sys

def create_prediction_pipeline():
    """
    Create and return the prediction pipeline for the competition
    """

    # Initialize the One-Class SVM model with optimized parameters
    model = OneClassSVM(
        nu=0.01,  # Expected outlier fraction - tuned for your data distribution
        kernel='rbf',
        gamma='scale',
        # Removed random_state as it's not a valid parameter for OneClassSVM
    )

    return model

def train_model(model, X_train, y_train):
    """
    Train the model on the training data

    Parameters:
    - model: The sklearn model instance
    - X_train: Training features (sparse matrix or array)
    - y_train: Training labels (used to identify majority class)
    """

    print("ğŸ”� Identifying majority class for One-Class SVM...")

    # Convert to numpy array if sparse
    if hasattr(X_train, 'toarray'):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = np.array(X_train)

    # Convert y_train to numpy and identify majority class (zeros)
    if hasattr(y_train, 'toarray'):
        y_train_dense = y_train.toarray().ravel()
    else:
        y_train_dense = np.array(y_train).ravel()

    # Find indices of majority class (zeros)
    zero_indices = np.where(y_train_dense == 0)[0]
    X_zeros = X_train_dense[zero_indices]

    print(f"ğŸ“Š Training on {len(X_zeros)} majority class samples")
    print(f"ğŸ“ˆ Total training samples: {len(X_train_dense)}")
    print(f"ğŸ�¯ Expected anomaly rate (nu): {model.nu}")

    # Train the One-Class SVM on majority class only
    model.fit(X_zeros)

    print("âœ… Model training completed")

    return model

def predict(model, X_test):
    """
    Make predictions on test data

    Parameters:
    - model: Trained One-Class SVM model
    - X_test: Test features (sparse matrix or array)

    Returns:
    - predictions: Binary predictions (0 for normal, 1 for anomaly)
    - anomaly_scores: Raw anomaly scores (more negative = more anomalous)
    """

    # Convert to numpy array if sparse
    if hasattr(X_test, 'toarray'):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = np.array(X_test)

    # Make predictions (-1 for outliers, 1 for inliers)
    predictions = model.predict(X_test_dense)

    # Convert to binary (0 for normal, 1 for anomaly)
    binary_predictions = (predictions == -1).astype(int)

    # Get anomaly scores
    anomaly_scores = model.decision_function(X_test_dense)

    return binary_predictions, anomaly_scores

def save_model(model, filepath='one_class_svm_model.pkl'):
    """
    Save the trained model to disk
    """
    joblib.dump(model, filepath)
    print(f"ğŸ’¾ Model saved to {filepath}")

def load_model(filepath='one_class_svm_model.pkl'):
    """
    Load a trained model from disk
    """
    model = joblib.load(filepath)
    print(f"ğŸ“¥ Model loaded from {filepath}")
    return model

def evaluate_predictions(y_true, y_pred, anomaly_scores=None):
    """
    Evaluate model predictions (for validation)
    """
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    print("\nğŸ“Š PREDICTION EVALUATION:")
    print("=" * 50)

    # Basic statistics
    unique, counts = np.unique(y_pred, return_counts=True)
    prediction_dist = dict(zip(unique, counts))
    print(f"Prediction distribution: {prediction_dist}")

    # If we have true labels, compute metrics
    if y_true is not None:
        print(f"\nTrue distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")

        # Classification report
        print("\nğŸ“‹ Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

        # Confusion matrix
        print("ğŸ�¯ Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        # AUC-ROC if we have anomaly scores
        if anomaly_scores is not None and len(np.unique(y_true)) > 1:
            try:
                auc_score = roc_auc_score(y_true, -anomaly_scores)  # More negative = more anomalous
                print(f"\nğŸ“ˆ AUC-ROC Score: {auc_score:.4f}")
            except:
                print("\nğŸ“ˆ AUC-ROC: Could not compute (need both classes)")

# Example usage pattern for the competition
def main():
    """
    Main execution function - template for competition submission
    """

    # Example: How to use the pipeline
    print("ğŸš€ COMPETITION PREDICTION PIPELINE")
    print("=" * 50)

    # 1. Create model
    model = create_prediction_pipeline()
    print("âœ… Model initialized")

    # 2. Train model (you would replace this with your actual training data)
    # model = train_model(model, X_train, y_train)

    # 3. Save model
    # save_model(model, 'submission_model.pkl')

    # 4. Load model for inference
    # loaded_model = load_model('submission_model.pkl')

    # 5. Make predictions
    # predictions, scores = predict(loaded_model, X_test)

    print("\nğŸ“� SUBMISSION INSTRUCTIONS:")
    print("1. Train the model using train_model() with your data")
    print("2. Save the model using save_model()")
    print("3. In your submission notebook, load the model and make predictions")
    print("4. Use predict() function to get final binary predictions")
    print("5. Submit the predictions in the required format")

if __name__ == "__main__":
    main()


# submission.py
"""
Competition Submission Pipeline - One-Class SVM for Anomaly Detection
Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class CompetitionPipeline:
    """
    A complete pipeline for competition submission using One-Class SVM
    for highly imbalanced anomaly detection problems.
    """

    def __init__(self, nu=0.01, kernel='rbf', gamma='scale', random_state=42):
        """
        Initialize the One-Class SVM model with competition parameters.

        Parameters:
        - nu: Expected outlier fraction (0.01 = 1% anomalies expected)
        - kernel: SVM kernel type
        - gamma: Kernel coefficient
        - random_state: Random seed for reproducibility
        """
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            random_state=random_state
        )
        self.is_trained = False

    def prepare_data(self, X, y=None):
        """
        Convert input data to proper format for training/prediction.

        Parameters:
        - X: Feature matrix (sparse or dense)
        - y: Optional labels for training

        Returns:
        - X_processed: Processed feature matrix
        - y_processed: Processed labels (if provided)
        """
        # Convert sparse matrix to dense array if needed
        if hasattr(X, 'toarray'):
            X_processed = X.toarray()
        else:
            X_processed = np.array(X)

        # Process labels if provided
        if y is not None:
            if hasattr(y, 'toarray'):
                y_processed = y.toarray().ravel()
            else:
                y_processed = np.array(y).ravel()
            return X_processed, y_processed

        return X_processed

    def train(self, X_train, y_train):
        """
        Train the One-Class SVM model on majority class data.

        Parameters:
        - X_train: Training features
        - y_train: Training labels (used to identify majority class)
        """
        print("ğŸš€ Starting model training...")

        # Prepare data
        X_processed, y_processed = self.prepare_data(X_train, y_train)

        # Analyze class distribution
        unique_classes, class_counts = np.unique(y_processed, return_counts=True)
        print(f"ğŸ“Š Training data class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"   Class {cls}: {count} samples ({count/len(y_processed)*100:.2f}%)")

        # Identify majority class (assumed to be class 0)
        majority_class = 0
        majority_indices = np.where(y_processed == majority_class)[0]
        X_majority = X_processed[majority_indices]

        print(f"ğŸ�¯ Training One-Class SVM on {len(X_majority)} majority class samples")
        print(f"ğŸ“ˆ Model parameters: nu={self.model.nu}, kernel={self.model.kernel}")

        # Train the model
        self.model.fit(X_majority)
        self.is_trained = True

        print("âœ… Model training completed successfully!")
        return self

    def predict(self, X_test, return_scores=False):
        """
        Make predictions on test data.

        Parameters:
        - X_test: Test features
        - return_scores: Whether to return anomaly scores

        Returns:
        - predictions: Binary predictions (0=normal, 1=anomaly)
        - anomaly_scores: Optional anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        # Prepare test data
        X_processed = self.prepare_data(X_test)

        # Make predictions (-1 for outliers, 1 for inliers)
        raw_predictions = self.model.predict(X_processed)

        # Convert to binary (0 for normal, 1 for anomaly)
        binary_predictions = (raw_predictions == -1).astype(int)

        # Get anomaly scores (more negative = more anomalous)
        anomaly_scores = self.model.decision_function(X_processed)

        print(f"ğŸ“Š Prediction summary:")
        print(f"   Normal samples (0): {(binary_predictions == 0).sum()}")
        print(f"   Anomaly samples (1): {(binary_predictions == 1).sum()}")

        if return_scores:
            return binary_predictions, anomaly_scores
        else:
            return binary_predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data with true labels.

        Parameters:
        - X_test: Test features
        - y_test: True test labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")

        # Make predictions
        predictions, scores = self.predict(X_test, return_scores=True)

        # Prepare true labels
        _, y_true = self.prepare_data(X_test, y_test)

        print("\n" + "="*60)
        print("ğŸ�¯ MODEL EVALUATION RESULTS")
        print("="*60)

        # Basic statistics
        print(f"\nğŸ“ˆ PREDICTION DISTRIBUTION:")
        unique_pred, counts_pred = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_pred, counts_pred):
            print(f"   Predicted class {pred}: {count} samples")

        print(f"\nğŸ“Š TRUE DISTRIBUTION:")
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        for true, count in zip(unique_true, counts_true):
            print(f"   True class {true}: {count} samples")

        # Classification metrics
        print(f"\nğŸ“‹ CLASSIFICATION REPORT:")
        print(classification_report(y_true, predictions, target_names=['Normal', 'Anomaly']))

        print(f"ğŸ�¯ CONFUSION MATRIX:")
        print(confusion_matrix(y_true, predictions))

        # AUC-ROC score
        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, -scores)  # More negative = more anomalous
                print(f"\nâ­� AUC-ROC SCORE: {auc:.4f}")
            except Exception as e:
                print(f"\nâš ï¸�  Could not compute AUC-ROC: {e}")

        return predictions, scores

    def save_model(self, filepath='competition_model.pkl'):
        """
        Save the trained model to disk.

        Parameters:
        - filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")

        joblib.dump(self.model, filepath)
        print(f"ğŸ’¾ Model saved successfully to: {filepath}")

    def load_model(self, filepath='competition_model.pkl'):
        """
        Load a trained model from disk.

        Parameters:
        - filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"ğŸ“¥ Model loaded successfully from: {filepath}")
        return self

# Utility functions for competition submission
def create_submission_file(predictions, sample_ids=None, output_path='submission.csv'):
    """
    Create a competition submission file.

    Parameters:
    - predictions: Model predictions (0/1)
    - sample_ids: Optional sample IDs
    - output_path: Path for submission file
    """
    if sample_ids is None:
        sample_ids = range(len(predictions))

    submission_df = pd.DataFrame({
        'id': sample_ids,
        'prediction': predictions
    })

    submission_df.to_csv(output_path, index=False)
    print(f"ğŸ“„ Submission file created: {output_path}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Normal (0): {(predictions == 0).sum()}")
    print(f"   Anomaly (1): {(predictions == 1).sum()}")

    return submission_df

def main():
    """
    Example usage of the competition pipeline.
    """
    print("ğŸ�¯ COMPETITION SUBMISSION PIPELINE")
    print("="*50)
    print("This pipeline uses One-Class SVM for anomaly detection")
    print("Perfect for highly imbalanced datasets")
    print("="*50)

    # Example usage pattern
    print("\nğŸ“� USAGE EXAMPLE:")
    print("""
    # Initialize pipeline
    pipeline = CompetitionPipeline(nu=0.01)

    # Train model
    pipeline.train(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Create submission file
    create_submission_file(predictions, test_ids, 'my_submission.csv')

    # Save model for future use
    pipeline.save_model('trained_model.pkl')
    """)

if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of 'question_type'
plt.figure(figsize=(8, 5))
sns.countplot(data=df, y='question_type', order=df['question_type'].value_counts().index)
plt.title('Distribution of Question Types')
plt.xlabel('Count')
plt.ylabel('Question Type')
plt.show()

# Display the counts as well
print("\nCounts of each question type:")
display(df['question_type'].value_counts())


# submission.py
"""
Competition Submission Pipeline - One-Class SVM for Anomaly Detection
Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class CompetitionPipeline:
    """
    A complete pipeline for competition submission using One-Class SVM
    for highly imbalanced anomaly detection problems.
    """

    def __init__(self, nu=0.01, kernel='rbf', gamma='scale', random_state=42):
        """
        Initialize the One-Class SVM model with competition parameters.

        Parameters:
        - nu: Expected outlier fraction (0.01 = 1% anomalies expected)
        - kernel: SVM kernel type
        - gamma: Kernel coefficient
        - random_state: Random seed for reproducibility
        """
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            random_state=random_state
        )
        self.is_trained = False

    def prepare_data(self, X, y=None):
        """
        Convert input data to proper format for training/prediction.

        Parameters:
        - X: Feature matrix (sparse or dense)
        - y: Optional labels for training

        Returns:
        - X_processed: Processed feature matrix
        - y_processed: Processed labels (if provided)
        """
        # Convert sparse matrix to dense array if needed
        if hasattr(X, 'toarray'):
            X_processed = X.toarray()
        else:
            X_processed = np.array(X)

        # Process labels if provided
        if y is not None:
            if hasattr(y, 'toarray'):
                y_processed = y.toarray().ravel()
            else:
                y_processed = np.array(y).ravel()
            return X_processed, y_processed

        return X_processed

    def train(self, X_train, y_train):
        """
        Train the One-Class SVM model on majority class data.

        Parameters:
        - X_train: Training features
        - y_train: Training labels (used to identify majority class)
        """
        print("ğŸš€ Starting model training...")

        # Prepare data
        X_processed, y_processed = self.prepare_data(X_train, y_train)

        # Analyze class distribution
        unique_classes, class_counts = np.unique(y_processed, return_counts=True)
        print(f"ğŸ“Š Training data class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"   Class {cls}: {count} samples ({count/len(y_processed)*100:.2f}%)")

        # Identify majority class (assumed to be class 0)
        majority_class = 0
        majority_indices = np.where(y_processed == majority_class)[0]
        X_majority = X_processed[majority_indices]

        print(f"ğŸ�¯ Training One-Class SVM on {len(X_majority)} majority class samples")
        print(f"ğŸ“ˆ Model parameters: nu={self.model.nu}, kernel={self.model.kernel}")

        # Train the model
        self.model.fit(X_majority)
        self.is_trained = True

        print("âœ… Model training completed successfully!")
        return self

    def predict(self, X_test, return_scores=False):
        """
        Make predictions on test data.

        Parameters:
        - X_test: Test features
        - return_scores: Whether to return anomaly scores

        Returns:
        - predictions: Binary predictions (0=normal, 1=anomaly)
        - anomaly_scores: Optional anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        # Prepare test data
        X_processed = self.prepare_data(X_test)

        # Make predictions (-1 for outliers, 1 for inliers)
        raw_predictions = self.model.predict(X_processed)

        # Convert to binary (0 for normal, 1 for anomaly)
        binary_predictions = (raw_predictions == -1).astype(int)

        # Get anomaly scores (more negative = more anomalous)
        anomaly_scores = self.model.decision_function(X_processed)

        print(f"ğŸ“Š Prediction summary:")
        print(f"   Normal samples (0): {(binary_predictions == 0).sum()}")
        print(f"   Anomaly samples (1): {(binary_predictions == 1).sum()}")

        if return_scores:
            return binary_predictions, anomaly_scores
        else:
            return binary_predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data with true labels.

        Parameters:
        - X_test: Test features
        - y_test: True test labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")

        # Make predictions
        predictions, scores = self.predict(X_test, return_scores=True)

        # Prepare true labels
        _, y_true = self.prepare_data(X_test, y_test)

        print("\n" + "="*60)
        print("ğŸ�¯ MODEL EVALUATION RESULTS")
        print("="*60)

        # Basic statistics
        print(f"\nğŸ“ˆ PREDICTION DISTRIBUTION:")
        unique_pred, counts_pred = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_pred, counts_pred):
            print(f"   Predicted class {pred}: {count} samples")

        print(f"\nğŸ“Š TRUE DISTRIBUTION:")
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        for true, count in zip(unique_true, counts_true):
            print(f"   True class {true}: {count} samples")

        # Classification metrics
        print(f"\nğŸ“‹ CLASSIFICATION REPORT:")
        print(classification_report(y_true, predictions, target_names=['Normal', 'Anomaly']))

        print(f"ğŸ�¯ CONFUSION MATRIX:")
        print(confusion_matrix(y_true, predictions))

        # AUC-ROC score
        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, -scores)  # More negative = more anomalous
                print(f"\nâ­� AUC-ROC SCORE: {auc:.4f}")
            except Exception as e:
                print(f"\nâš ï¸�  Could not compute AUC-ROC: {e}")

        return predictions, scores

    def save_model(self, filepath='competition_model.pkl'):
        """
        Save the trained model to disk.

        Parameters:
        - filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")

        joblib.dump(self.model, filepath)
        print(f"ğŸ’¾ Model saved successfully to: {filepath}")

    def load_model(self, filepath='competition_model.pkl'):
        """
        Load a trained model from disk.

        Parameters:
        - filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"ğŸ“¥ Model loaded successfully from: {filepath}")
        return self

# Utility functions for competition submission
def create_submission_file(predictions, sample_ids=None, output_path='submission.csv'):
    """
    Create a competition submission file.

    Parameters:
    - predictions: Model predictions (0/1)
    - sample_ids: Optional sample IDs
    - output_path: Path for submission file
    """
    if sample_ids is None:
        sample_ids = range(len(predictions))

    submission_df = pd.DataFrame({
        'id': sample_ids,
        'prediction': predictions
    })

    submission_df.to_csv(output_path, index=False)
    print(f"ğŸ“„ Submission file created: {output_path}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Normal (0): {(predictions == 0).sum()}")
    print(f"   Anomaly (1): {(predictions == 1).sum()}")

    return submission_df

def main():
    """
    Example usage of the competition pipeline.
    """
    print("ğŸ�¯ COMPETITION SUBMISSION PIPELINE")
    print("="*50)
    print("This pipeline uses One-Class SVM for anomaly detection")
    print("Perfect for highly imbalanced datasets")
    print("="*50)

    # Example usage pattern
    print("\nğŸ“� USAGE EXAMPLE:")
    print("""
    # Initialize pipeline
    pipeline = CompetitionPipeline(nu=0.01)

    # Train model
    pipeline.train(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Create submission file
    create_submission_file(predictions, test_ids, 'my_submission.csv')

    # Save model for future use
    pipeline.save_model('trained_model.pkl')
    """)

if __name__ == "__main__":
    main()


# Example usage notebook cell:
"""
# COMPETITION SUBMISSION - QUICK START

# 1. Import the pipeline
from submission import CompetitionPipeline, create_submission_file

# 2. Initialize and train
pipeline = CompetitionPipeline(nu=0.01)
pipeline.train(X_train, y_train)

# 3. Make predictions
predictions = pipeline.predict(X_test)

# 4. Create submission (assuming you have test_ids)
create_submission_file(predictions, test_ids, 'final_submission.csv')

# 5. Optional: Save model for future
pipeline.save_model('final_model.pkl')
"""


# Simply copy the code above into a new file called 'submission.py'
# Or use this to create the file:
with open('submission.py', 'w') as f:
    f.write("# Paste the entire submission.py code here")


# submission.py
"""
Competition Submission Pipeline - One-Class SVM for Anomaly Detection
Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class CompetitionPipeline:
    """
    A complete pipeline for competition submission using One-Class SVM
    for highly imbalanced anomaly detection problems.
    """

    def __init__(self, nu=0.01, kernel='rbf', gamma='scale', random_state=42):
        """
        Initialize the One-Class SVM model with competition parameters.

        Parameters:
        - nu: Expected outlier fraction (0.01 = 1% anomalies expected)
        - kernel: SVM kernel type
        - gamma: Kernel coefficient
        - random_state: Random seed for reproducibility
        """
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            random_state=random_state
        )
        self.is_trained = False

    def prepare_data(self, X, y=None):
        """
        Convert input data to proper format for training/prediction.

        Parameters:
        - X: Feature matrix (sparse or dense)
        - y: Optional labels for training

        Returns:
        - X_processed: Processed feature matrix
        - y_processed: Processed labels (if provided)
        """
        # Convert sparse matrix to dense array if needed
        if hasattr(X, 'toarray'):
            X_processed = X.toarray()
        else:
            X_processed = np.array(X)

        # Process labels if provided
        if y is not None:
            if hasattr(y, 'toarray'):
                y_processed = y.toarray().ravel()
            else:
                y_processed = np.array(y).ravel()
            return X_processed, y_processed

        return X_processed

    def train(self, X_train, y_train):
        """
        Train the One-Class SVM model on majority class data.

        Parameters:
        - X_train: Training features
        - y_train: Training labels (used to identify majority class)
        """
        print("ğŸš€ Starting model training...")

        # Prepare data
        X_processed, y_processed = self.prepare_data(X_train, y_train)

        # Analyze class distribution
        unique_classes, class_counts = np.unique(y_processed, return_counts=True)
        print(f"ğŸ“Š Training data class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"   Class {cls}: {count} samples ({count/len(y_processed)*100:.2f}%)")

        # Identify majority class (assumed to be class 0)
        majority_class = 0
        majority_indices = np.where(y_processed == majority_class)[0]
        X_majority = X_processed[majority_indices]

        print(f"ğŸ�¯ Training One-Class SVM on {len(X_majority)} majority class samples")
        print(f"ğŸ“ˆ Model parameters: nu={self.model.nu}, kernel={self.model.kernel}")

        # Train the model
        self.model.fit(X_majority)
        self.is_trained = True

        print("âœ… Model training completed successfully!")
        return self

    def predict(self, X_test, return_scores=False):
        """
        Make predictions on test data.

        Parameters:
        - X_test: Test features
        - return_scores: Whether to return anomaly scores

        Returns:
        - predictions: Binary predictions (0=normal, 1=anomaly)
        - anomaly_scores: Optional anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        # Prepare test data
        X_processed = self.prepare_data(X_test)

        # Make predictions (-1 for outliers, 1 for inliers)
        raw_predictions = self.model.predict(X_processed)

        # Convert to binary (0 for normal, 1 for anomaly)
        binary_predictions = (raw_predictions == -1).astype(int)

        # Get anomaly scores (more negative = more anomalous)
        anomaly_scores = self.model.decision_function(X_processed)

        print(f"ğŸ“Š Prediction summary:")
        print(f"   Normal samples (0): {(binary_predictions == 0).sum()}")
        print(f"   Anomaly samples (1): {(binary_predictions == 1).sum()}")

        if return_scores:
            return binary_predictions, anomaly_scores
        else:
            return binary_predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data with true labels.

        Parameters:
        - X_test: Test features
        - y_test: True test labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")

        # Make predictions
        predictions, scores = self.predict(X_test, return_scores=True)

        # Prepare true labels
        _, y_true = self.prepare_data(X_test, y_test)

        print("\n" + "="*60)
        print("ğŸ�¯ MODEL EVALUATION RESULTS")
        print("="*60)

        # Basic statistics
        print(f"\nğŸ“ˆ PREDICTION DISTRIBUTION:")
        unique_pred, counts_pred = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_pred, counts_pred):
            print(f"   Predicted class {pred}: {count} samples")

        print(f"\nğŸ“Š TRUE DISTRIBUTION:")
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        for true, count in zip(unique_true, counts_true):
            print(f"   True class {true}: {count} samples")

        # Classification metrics
        print(f"\nğŸ“‹ CLASSIFICATION REPORT:")
        print(classification_report(y_true, predictions, target_names=['Normal', 'Anomaly']))

        print(f"ğŸ�¯ CONFUSION MATRIX:")
        print(confusion_matrix(y_true, predictions))

        # AUC-ROC score
        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, -scores)  # More negative = more anomalous
                print(f"\nâ­� AUC-ROC SCORE: {auc:.4f}")
            except Exception as e:
                print(f"\nâš ï¸�  Could not compute AUC-ROC: {e}")

        return predictions, scores

    def save_model(self, filepath='competition_model.pkl'):
        """
        Save the trained model to disk.

        Parameters:
        - filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")

        joblib.dump(self.model, filepath)
        print(f"ğŸ’¾ Model saved successfully to: {filepath}")

    def load_model(self, filepath='competition_model.pkl'):
        """
        Load a trained model from disk.

        Parameters:
        - filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"ğŸ“¥ Model loaded successfully from: {filepath}")
        return self

def create_submission_file(predictions, sample_ids=None, output_path='submission.csv'):
    """
    Create a competition submission file.

    Parameters:
    - predictions: Model predictions (0/1)
    - sample_ids: Optional sample IDs
    - output_path: Path for submission file
    """
    if sample_ids is None:
        sample_ids = range(len(predictions))

    submission_df = pd.DataFrame({
        'id': sample_ids,
        'prediction': predictions
    })

    submission_df.to_csv(output_path, index=False)
    print(f"ğŸ“„ Submission file created: {output_path}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Normal (0): {(predictions == 0).sum()}")
    print(f"   Anomaly (1): {(predictions == 1).sum()}")

    return submission_df

# Example usage function
def run_example_pipeline():
    """
    Demonstrate the complete competition pipeline.
    """
    print("ğŸ�¯ COMPETITION SUBMISSION PIPELINE DEMO")
    print("="*50)

    # This is just a demo - replace with your actual data
    print("ğŸ“� Replace this with your actual data loading code:")
    print("""
    # Load your data
    from sklearn.model_selection import train_test_split

    # Assuming you have:
    # final_sparse_matrix - your feature matrix
    # y - your labels
    # test_ids - your test sample IDs

    X_train, X_test, y_train, y_test = train_test_split(
        final_sparse_matrix,
        y,
        test_size=0.2,
        random_state=42
    )
    """)

    print("\nğŸš€ PIPELINE USAGE:")
    print("""
    # 1. Initialize pipeline
    pipeline = CompetitionPipeline(nu=0.01)

    # 2. Train model
    pipeline.train(X_train, y_train)

    # 3. Make predictions
    predictions = pipeline.predict(X_test)

    # 4. Create submission file
    submission_df = create_submission_file(predictions, test_ids, 'final_submission.csv')

    # 5. Optional: Save model
    pipeline.save_model('final_model.pkl')

    # 6. Optional: Evaluate performance
    pipeline.evaluate(X_test, y_test)
    """)

if __name__ == "__main__":
    run_example_pipeline()


# Save the submission pipeline to a file and download it
with open('submission.py', 'w') as f:
    f.write('''"""
Competition Submission Pipeline - One-Class SVM for Anomaly Detection
Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class CompetitionPipeline:
    """
    A complete pipeline for competition submission using One-Class SVM
    for highly imbalanced anomaly detection problems.
    """

    def __init__(self, nu=0.01, kernel='rbf', gamma='scale', random_state=42):
        """
        Initialize the One-Class SVM model with competition parameters.

        Parameters:
        - nu: Expected outlier fraction (0.01 = 1% anomalies expected)
        - kernel: SVM kernel type
        - gamma: Kernel coefficient
        - random_state: Random seed for reproducibility
        """
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            random_state=random_state
        )
        self.is_trained = False

    def prepare_data(self, X, y=None):
        """
        Convert input data to proper format for training/prediction.

        Parameters:
        - X: Feature matrix (sparse or dense)
        - y: Optional labels for training

        Returns:
        - X_processed: Processed feature matrix
        - y_processed: Processed labels (if provided)
        """
        # Convert sparse matrix to dense array if needed
        if hasattr(X, 'toarray'):
            X_processed = X.toarray()
        else:
            X_processed = np.array(X)

        # Process labels if provided
        if y is not None:
            if hasattr(y, 'toarray'):
                y_processed = y.toarray().ravel()
            else:
                y_processed = np.array(y).ravel()
            return X_processed, y_processed

        return X_processed

    def train(self, X_train, y_train):
        """
        Train the One-Class SVM model on majority class data.

        Parameters:
        - X_train: Training features
        - y_train: Training labels (used to identify majority class)
        """
        print("ğŸš€ Starting model training...")

        # Prepare data
        X_processed, y_processed = self.prepare_data(X_train, y_train)

        # Analyze class distribution
        unique_classes, class_counts = np.unique(y_processed, return_counts=True)
        print(f"ğŸ“Š Training data class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"   Class {cls}: {count} samples ({count/len(y_processed)*100:.2f}%)")

        # Identify majority class (assumed to be class 0)
        majority_class = 0
        majority_indices = np.where(y_processed == majority_class)[0]
        X_majority = X_processed[majority_indices]

        print(f"ğŸ�¯ Training One-Class SVM on {len(X_majority)} majority class samples")
        print(f"ğŸ“ˆ Model parameters: nu={self.model.nu}, kernel={self.model.kernel}")

        # Train the model
        self.model.fit(X_majority)
        self.is_trained = True

        print("âœ… Model training completed successfully!")
        return self

    def predict(self, X_test, return_scores=False):
        """
        Make predictions on test data.

        Parameters:
        - X_test: Test features
        - return_scores: Whether to return anomaly scores

        Returns:
        - predictions: Binary predictions (0=normal, 1=anomaly)
        - anomaly_scores: Optional anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")

        # Prepare test data
        X_processed = self.prepare_data(X_test)

        # Make predictions (-1 for outliers, 1 for inliers)
        raw_predictions = self.model.predict(X_processed)

        # Convert to binary (0 for normal, 1 for anomaly)
        binary_predictions = (raw_predictions == -1).astype(int)

        # Get anomaly scores (more negative = more anomalous)
        anomaly_scores = self.model.decision_function(X_processed)

        print(f"ğŸ“Š Prediction summary:")
        print(f"   Normal samples (0): {(binary_predictions == 0).sum()}")
        print(f"   Anomaly samples (1): {(binary_predictions == 1).sum()}")

        if return_scores:
            return binary_predictions, anomaly_scores
        else:
            return binary_predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data with true labels.

        Parameters:
        - X_test: Test features
        - y_test: True test labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")

        # Make predictions
        predictions, scores = self.predict(X_test, return_scores=True)

        # Prepare true labels
        _, y_true = self.prepare_data(X_test, y_test)

        print("\\n" + "="*60)
        print("ğŸ�¯ MODEL EVALUATION RESULTS")
        print("="*60)

        # Basic statistics
        print(f"\\nğŸ“ˆ PREDICTION DISTRIBUTION:")
        unique_pred, counts_pred = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_pred, counts_pred):
            print(f"   Predicted class {pred}: {count} samples")

        print(f"\\nğŸ“Š TRUE DISTRIBUTION:")
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        for true, count in zip(unique_true, counts_true):
            print(f"   True class {true}: {count} samples")

        # Classification metrics
        print(f"\\nğŸ“‹ CLASSIFICATION REPORT:")
        print(classification_report(y_true, predictions, target_names=['Normal', 'Anomaly']))

        print(f"ğŸ�¯ CONFUSION MATRIX:")
        print(confusion_matrix(y_true, predictions))

        # AUC-ROC score
        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, -scores)  # More negative = more anomalous
                print(f"\\nâ­� AUC-ROC SCORE: {auc:.4f}")
            except Exception as e:
                print(f"\\nâš ï¸�  Could not compute AUC-ROC: {e}")

        return predictions, scores

    def save_model(self, filepath='competition_model.pkl'):
        """
        Save the trained model to disk.

        Parameters:
        - filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")

        joblib.dump(self.model, filepath)
        print(f"ğŸ’¾ Model saved successfully to: {filepath}")

    def load_model(self, filepath='competition_model.pkl'):
        """
        Load a trained model from disk.

        Parameters:
        - filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"ğŸ“¥ Model loaded successfully from: {filepath}")
        return self

def create_submission_file(predictions, sample_ids=None, output_path='submission.csv'):
    """
    Create a competition submission file.

    Parameters:
    - predictions: Model predictions (0/1)
    - sample_ids: Optional sample IDs
    - output_path: Path for submission file
    """
    if sample_ids is None:
        sample_ids = range(len(predictions))

    submission_df = pd.DataFrame({
        'id': sample_ids,
        'prediction': predictions
    })

    submission_df.to_csv(output_path, index=False)
    print(f"ğŸ“„ Submission file created: {output_path}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Normal (0): {(predictions == 0).sum()}")
    print(f"   Anomaly (1): {(predictions == 1).sum()}")

    return submission_df

def run_example_pipeline():
    """
    Demonstrate the complete competition pipeline.
    """
    print("ğŸ�¯ COMPETITION SUBMISSION PIPELINE")
    print("="*50)
    print("Perfect for highly imbalanced datasets (like 7970 zeros vs 1 anomaly)")
    print("="*50)

    print("\\nğŸš€ QUICK START:")
    print("""
    # Initialize and train
    pipeline = CompetitionPipeline(nu=0.01)
    pipeline.train(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Create submission
    create_submission_file(predictions, test_ids, 'submission.csv')
    """)

if __name__ == "__main__":
    run_example_pipeline()
''')

print("âœ… submission.py file created successfully!")

# Download the file
from google.colab import files
files.download('submission.py')

