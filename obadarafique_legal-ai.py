# This project uses NLTK for text processing.
# We need to download specific packages for tokenization, stop words, and lemmatization.
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_unverified_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("NLTK packages downloaded successfully.")
import re
import random
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

print("Libraries imported successfully.")
# A small, representative sample of the Indian Constitution for our chatbot's knowledge base.
constitution_text = """
Article 14: Equality before law.
The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.
_ARTICLE_SEPARATOR_
Article 19: Protection of certain rights regarding freedom of speech, etc.
(1) All citizens shall have the right
(a) to freedom of speech and expression;
(b) to assemble peaceably and without arms;
(c) to form associations or unions;
(d) to move freely throughout the territory of India;
(e) to reside and settle in any part of the territory of India; and
(f) to practise any profession, or to carry on any occupation, trade or business.
_ARTICLE_SEPARATOR_
Article 21: Protection of life and personal liberty.
No person shall be deprived of his life or personal liberty except according to procedure established by law.
_ARTICLE_SEPARATOR_
Article 32: Remedies for enforcement of rights conferred by this Part.
(1) The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed.
(2) The Supreme Court shall have power to issue directions or orders or writs, including writs in the nature of habeas corpus, mandamus, prohibition, quo warranto and certiorari, whichever may be appropriate, for the enforcement of any of the rights conferred by this Part.
"""

# Let's parse this text into a more usable dictionary format.
constitution_data = {}
articles = constitution_text.strip().split('_ARTICLE_SEPARATOR_')

for article in articles:
    match = re.search(r'Article (\d+):', article)
    if match:
        article_num = match.group(1)
        constitution_data[article_num] = article.strip()

print("Constitution data loaded and parsed successfully.")
print(f"Loaded {len(constitution_data)} articles.")
# Example: Accessing Article 21
print("\n--- Example: Article 21 ---")
print(constitution_data.get('21', 'Article not found.'))
# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and preprocesses a given text."""
    # 1. Lowercasing and remove punctuation
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # 2. Tokenization
    tokens = word_tokenize(text)
    
    # 3. Stop word removal and Lemmatization
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(processed_tokens)

# Example of preprocessing
original_text = "Can you please explain the details of Article 14 to me?"
processed_text = preprocess_text(original_text)
print(f"Original: {original_text}")
print(f"Processed: {processed_text}")
# Define our intents and create training data
intents = {
    "greet": {
        "patterns": ["hello", "hi", "hey", "good morning", "greetings"],
        "responses": ["Hello! How can I help you with the Indian Constitution today?", "Hi there! What would you like to know?"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "take care", "exit"],
        "responses": ["Goodbye! Feel free to ask more questions anytime.", "Take care!"]
    },
    "get_article": {
        "patterns": [
            "what is article 14", "explain article 21", "tell me about article 32",
            "can you provide details on article 19", "article 14", "article 21 information"
        ],
        "responses": ["Here is the information on Article {article_num}:\n{content}"]
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "appreciate it"],
        "responses": ["You're welcome!", "Glad I could help!"]
    }
}

# Prepare data for Scikit-learn
X_train = []
y_train = []

for intent, data in intents.items():
    for pattern in data['patterns']:
        X_train.append(pattern)
        y_train.append(intent)

print(f"Training data created. Number of samples: {len(X_train)}")
print("\n--- Sample Data ---")
for i in range(5):
    print(f"Pattern: '{X_train[i]}' -> Intent: '{y_train[i]}'")
    # Create a pipeline for text classification
intent_classifier_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text)),
    ('clf', SVC(kernel='linear', probability=True)) # Use linear kernel for text data
])

# Train the classifier
intent_classifier_pipeline.fit(X_train, y_train)

print("Intent classification model trained successfully.")

# Test the model
test_query = "can you tell me about article 14"
predicted_intent = intent_classifier_pipeline.predict([test_query])[0]
print(f"\n--- Test ---")
print(f"Query: '{test_query}'")
print(f"Predicted Intent: '{predicted_intent}'")
def get_response(user_query):
    """
    Generates a response to a user query by predicting intent and retrieving information.
    """
    # Predict the intent
    predicted_intent = intent_classifier_pipeline.predict([user_query])[0]
    
    # --- Intent-based Response Logic ---
    
    if predicted_intent == "get_article":
        # Extract article number using regular expressions
        match = re.search(r'\b(\d+)\b', user_query)
        if match:
            article_num = match.group(1)
            content = constitution_data.get(article_num)
            if content:
                response = random.choice(intents['get_article']['responses']).format(article_num=article_num, content=content)
            else:
                response = f"I'm sorry, I don't have information on Article {article_num} in my current database."
        else:
            response = "I see you're asking about an article, but I couldn't identify the number. Please specify, for example: 'what is article 14?'"
            
    elif predicted_intent in intents:
        # For simple intents like greet, goodbye, thanks
        response = random.choice(intents[predicted_intent]['responses'])
        
    else:
        # Fallback response
        response = "I'm not sure how to answer that. You can ask me to explain a specific article of the Indian Constitution (e.g., 'tell me about article 21')."
        
    return response

# Test the response generation
print("--- Response Generation Test ---")
print(f"User: hello")
print(f"Vidhi: {get_response('hello')}")
print("\n")
print(f"User: explain article 21 please")
print(f"Vidhi: {get_response('explain article 21 please')}")
def chat():
    """
    Initiates a command-line chat session with the bot.
    """
    print("---------------------------------------------------------")
    print("Welcome to Vidhi - Your Legal AI Assistant!")
    print("Ask me questions about the Indian Constitution.")
    print("Type 'exit' or 'bye' to end the chat.")
    print("---------------------------------------------------------")
    
    while True:
        user_input = input("You: ")
        
        # Check for goodbye intent to exit loop
        processed_input = preprocess_text(user_input)
        if intent_classifier_pipeline.predict([processed_input])[0] == 'goodbye':
            print("Vidhi:", random.choice(intents['goodbye']['responses']))
            break
            
        response = get_response(user_input)
        print("Vidhi:", response)

# To start the chat, uncomment and run the following line in your notebook:
# chat()

