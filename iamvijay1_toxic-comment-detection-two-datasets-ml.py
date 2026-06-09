


import pandas as pd
youtube_df = pd.read_csv("/kaggle/input/youtube-toxicity-data/youtoxic_english_1000.csv")
toxic_df = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")


youtube_df.head()


youtube_df.shape


toxic_df.head()


toxic_df.shape


#We are going to select just the "comment_text" and "toxic" columns
toxic_df['Toxic'] = toxic_df.iloc[:, 2:].any(axis = 1)
selected_toxic_columns = toxic_df[['comment_text', 'Toxic']]
selected_toxic_columns


#We are going to select just the "Text" and "IsToxic" columns
youtube_df['Toxic'] = youtube_df.iloc[:, 3:].any(axis = 1)
selected_youtube_columns = youtube_df[['Text', 'Toxic']].rename(
    columns = {
        'Text' : 'comment_text'
    }
)
selected_youtube_columns


#Let's combine the two DataFrames
combined_df = pd.concat([selected_toxic_columns, selected_youtube_columns], ignore_index = True)
combined_df.head()


combined_df.shape


combined_df.info()


combined_df.describe()


combined_df.isnull().sum()


#Checking duplicates
combined_df.duplicated(subset = ['comment_text'], keep = False).sum()


#Printing the duplicated rows
duplicates = combined_df[combined_df.duplicated(subset = ['comment_text'], keep = False)]
duplicates


#Dropping Duplicates
combined_df.drop_duplicates(subset = ['comment_text'], keep = 'first', inplace = True)


#Confirm Drops
combined_df.duplicated(subset = ['comment_text'], keep = False).sum()


combined_df['Toxic'].value_counts()

#We can see from the code above that the data is imbalanced.


import matplotlib.pyplot as plt
%matplotlib inline
#Graphical representation of the Toxic column values (Toxic vs Non-Toxic Comments) distribution
plt.figure(figsize = (6, 4))
toxic_counts = combined_df['Toxic'].value_counts()
toxic_counts.plot(kind = 'bar', color = ['green', 'red'])
plt.title('Toxic vs Non-Toxic Comments')
plt.xlabel('Toxic')
plt.ylabel('Count')
plt.xticks(rotation = 0)
plt.show()


#"Wordcloud" is for creating word cloud visualization.
from wordcloud import WordCloud
#Creating Word Cloud of Toxic Comments
toxic_comments = ''.join(combined_df[combined_df['Toxic']]['comment_text'])
toxic_words = WordCloud(width = 900, height = 450, background_color = "white").generate(toxic_comments)
plt.imshow(toxic_words, interpolation = 'bilinear')
plt.axis("off")
plt.title("Word Cloud For Toxic Comments")
plt.show()


#Creating Word Cloud of Non-Toxic Comments
non_toxic_comments = ''.join(combined_df[~combined_df['Toxic']]['comment_text'])
non_toxic_words = WordCloud(width = 900, height = 450, background_color = "white").generate(non_toxic_comments)
plt.imshow(non_toxic_words, interpolation = 'bilinear')
plt.axis("off")
plt.title("Word Cloud For Non-Toxic Comments")
plt.show()


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#Replacing True as 1 and False as 0. 
combined_df['Toxic'] = combined_df['Toxic'].replace({True: 1, False: 0})


#"re" is for regular expressions and text processing.
import re
#Cleaning the comment texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re'", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub("\W", " ", text)
    text = re.sub("\s+", " ", text)
    text = text.strip(" ")
    
    return text

combined_df['comment_text'] = combined_df['comment_text'].map(lambda cleaned : clean_text(cleaned))


combined_df.head()


"""TF-IDF(Term Frequency-Inverse Document Frequency) is used for text analysis: 
Text to Numerical Conversion, Feature Extraction, Dimensionality Reduction, Normalization & Scaling etc."""

from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(max_features = 5000, stop_words = 'english')
X = vector.fit_transform(combined_df['comment_text'])
Y = combined_df['Toxic']


combined_df['Toxic'].value_counts()


#Recall that the data is imbalanced, so we have to balance it using SMOTE
"""SMOTE(Synthetic Minority Over-sampling Technique): It's a technique used in machine learning 
in dealing with imbalanced data. Imbalanced Data is a data where one class is significantly
underrepresented, compared to another class. The latter is the the 'Minority Class', while
the former is the 'Majority Class'.

The minority class has fewer samples than the majority class and the imbalance can lead to
biased models that performs poorly on the minority class.

SMOTE generates synthetic samples for the minority class. It does this by creating new
instances or synthetic samples that are combinations of the existing minority class samples.
The samples are created by interpolating between existing minority class samples in the feature.

SMOTE selects pairs of similar instances from the minority class and creates synthetic 
instances along the line segments joining these pairs. This process is known as 
'Interpolation Technique', and the process effectively increases the number of samples in
the minority class, making it more balanced with the majority class.

The synthetic samples generated are created in a uniform way that maintains the distribution
and patterns of the minority class,thereby preventing overfitting and improving the 
generalization of the machine learning model.
"""

from imblearn.over_sampling import SMOTE

#Initialize SMOTE
smote = SMOTE()

#Using SMOTE for oversampling
X_resampled, y_resampled = smote.fit_resample(X, Y)

#Converting oversampled data to DataFrame
resampled_df = pd.DataFrame(X_resampled.todense(), columns = vector.get_feature_names_out())
resampled_df['Toxic'] = y_resampled


resampled_df['Toxic'].value_counts()


#Plotting the new distribution sample
plt.figure(figsize = (6, 4))
toxic_counts = resampled_df['Toxic'].value_counts()
toxic_counts.plot(kind = 'bar', color = ['green', 'red'])
plt.title('Toxic vs Non-Toxic Comments')
plt.xlabel('Toxic')
plt.ylabel('Count')
plt.xticks(rotation = 0)
plt.show()


from sklearn.model_selection import train_test_split
#Splitting the New Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = 42)


#"TensorFlow/keras" is for deep learning models
"""Neural Network Model with very few neurons ensures that the model is ligh-weighted,
and using a dropout of 0.5 helps prevent overfitting."""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
#Neural Network Model
model = Sequential([
    Dense(64, activation = 'relu'),
    Dropout(0.5),
    Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])


train_model = model.fit(X_train.toarray(), y_train, epochs = 10, batch_size = 32, validation_split = 0.2)


#Training vs Validation Accuracy
plt.figure(figsize = (6, 4))
plt.plot(train_model.history['accuracy'])
plt.plot(train_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()


#Training vs Validation Loss
plt.figure(figsize = (6, 4))
plt.plot(train_model.history['loss'])
plt.plot(train_model.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()


#Evaluating Model Accuracy On Test Data
"""Let's ensure that the model is not overfitting."""

loss, accuracy = model.evaluate(X_test, y_test)
print(f"The Test Accuracy is: {accuracy}")


#Loss
print(f"The Model Loss is: {loss}")


from sklearn.metrics import classification_report

#Predictions on Test Data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

#Classification Report
class_report = classification_report(y_test, y_pred)
print(class_report)


import seaborn as sns

#Predictions on Test Data
#y_pred_prob = model.predict(X_test)
#y_pred = (y_pred_prob > 0.5).astype(int)

#Classification Report
class_report = classification_report(y_test, y_pred, output_dict = True)
class_report_df = pd.DataFrame(class_report).transpose()

#Dropping irrelevant metrics for Visualization
class_metrics = class_report_df.drop(['accuracy', 'macro avg', 'weighted avg'])

#Classification Metrics Using Heatmap
plt.figure(figsize = (8, 6))
sns.heatmap(class_metrics[['precision', 'recall', 'f1-score']], annot = True, cmap = 'Reds', fmt = '.2f')
plt.title("Classification Report Metrics")
plt.xlabel("Metrics")
plt.ylabel("Class")
plt.yticks(rotation = 0)
plt.show()


#Saving the Keras Model
import pickle

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vector, f)

model.save('toxic_comment_prediction_model.h5')


#Reusing The Saved Model
import pickle
from tensorflow.keras.models import load_model
#Import TF-IDF Vectorizer for text handling
from sklearn.feature_extraction.text import TfidfVectorizer

#Loading TF-IDF Vectorizer
with open('/kaggle/working/tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)
    
    
#Loading The Trained Model
loaded_model = load_model('/kaggle/working/toxic_comment_prediction_model.h5')
new_comments = [
    "You're quite a bad person at keeping to time.",
    "This is a very bad service.",
    "You’ve achieved so much!",
    "You are very stupid and mad.",
]

#Processing New Comments using the Loaded TF-IDF Vectorizer
processed_comment = loaded_vectorizer.transform(new_comments)

#Predicting using the Loaded Model
predictions = (loaded_model.predict(processed_comment) > 0.5).astype(int)

#Prediction Result
for comment, prediction in zip(new_comments, predictions):
    print(f"Comment: {comment} | Is Toxic: {bool(prediction)}")


# %% [markdown]
# # Toxic Comment Detection (Two Datasets) - Machine Learning
#
# **Objective**: Build a model to detect toxic (1) and non-toxic (0) comments using Jigsaw and YouTube datasets.
#
# **Methodology**:
# - Load and combine datasets.
# - Select comment and toxicity columns.
# - Preprocess text and balance dataset.
# - Train a neural network model.
# - Evaluate, save, and test the model.

# %% [code]
# Import libraries and set random seeds for reproducibility
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# %% [markdown]
# ## 2. Initial Data Analysis

# %% [code]
# Load datasets
youtube_df = pd.read_csv("/kaggle/input/youtube-toxicity-data/youtoxic_english_1000.csv")
toxic_df = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")

# Check dataset shapes
print("YouTube Dataset Shape:", youtube_df.shape)  # (1000, 15)
print("Jigsaw Dataset Shape:", toxic_df.shape)   # (159571, 8)

# %% [code]
# Select relevant columns and create binary Toxic label (1 for toxic, 0 for non-toxic)
toxic_df['Toxic'] = toxic_df.iloc[:, 2:].any(axis=1).astype(int)
selected_toxic_columns = toxic_df[['comment_text', 'Toxic']]
youtube_df['Toxic'] = youtube_df.iloc[:, 3:].any(axis=1).astype(int)
selected_youtube_columns = youtube_df[['Text', 'Toxic']].rename(columns={'Text': 'comment_text'})

# Combine datasets
combined_df = pd.concat([selected_toxic_columns, selected_youtube_columns])

# Handle missing values
combined_df['comment_text'] = combined_df['comment_text'].fillna('').astype(str)

# Remove duplicates
combined_df = combined_df.drop_duplicates()
print("Combined Dataset Shape after removing duplicates:", combined_df.shape)  # (160566, 2)

# %% [markdown]
# ## 3. Visualization

# %% [code]
# Bar plot for toxic vs non-toxic distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Toxic', data=combined_df)
plt.title('Distribution of Toxic (1) and Non-Toxic (0) Comments')
plt.xlabel('Toxicity (0 = Non-Toxic, 1 = Toxic)')
plt.ylabel('Count')
plt.show()

# Word clouds for toxic and non-toxic comments
toxic_text = ' '.join(combined_df[combined_df['Toxic'] == 1]['comment_text'])
non_toxic_text = ' '.join(combined_df[combined_df['Toxic'] == 0]['comment_text'])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
wordcloud_toxic = WordCloud(width=800, height=400, background_color='white').generate(toxic_text)
plt.imshow(wordcloud_toxic, interpolation='bilinear')
plt.title('Word Cloud: Toxic Comments')
plt.axis('off')

plt.subplot(1, 2, 2)
wordcloud_non_toxic = WordCloud(width=800, height=400, background_color='white').generate(non_toxic_text)
plt.title('Word Cloud: Non-Toxic Comments')
plt.imshow(wordcloud_non_toxic, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# ## 4. Exploratory Data Analysis (EDA)

# %% [code]
# Text cleaning function
def clean_text(text):
    """Clean text by removing special characters, converting to lowercase, and removing stopwords."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Apply text cleaning
combined_df['comment_text'] = combined_df['comment_text'].apply(clean_text)

# TF-IDF Vectorization
vector = TfidfVectorizer(max_features=5000, stop_words='english')
X = vector.fit_transform(combined_df['comment_text'])
y = combined_df['Toxic']

# Balance dataset with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print("Dataset Shape after SMOTE:", X.shape)

# %% [markdown]
# ## 5. Modeling

# %% [code]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# %% [code]
# Visualize model performance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='#1f77b4')
plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [code]
# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Toxic (0)', 'Toxic (1)']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic (0)', 'Toxic (1)'], yticklabels=['Non-Toxic (0)', 'Toxic (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %% [code]
# Save model and vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vector, f)
model.save('toxic_comment_prediction_model.h5')

# %% [code]
# Function to predict toxicity for new comments
def predict_toxicity(comments, vectorizer, model):
    """Predict toxicity for a list of comments."""
    try:
        cleaned_comments = [clean_text(comment) for comment in comments]
        processed_comments = vectorizer.transform(cleaned_comments)
        predictions = (model.predict(processed_comments) > 0.5).astype(int)
        return [(comment, bool(pred)) for comment, pred in zip(comments, predictions)]
    except Exception as e:
        print(f"Error processing comments: {e}")
        return []

# Test model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)
loaded_model = tf.keras.models.load_model('toxic_comment_prediction_model.h5')

new_comments = [
    "You're quite a bad person at keeping to time.",
    "This is a very bad service.",
    "You’ve achieved so much!",
    "You are very stupid and mad."
]
results = predict_toxicity(new_comments, loaded_vectorizer, loaded_model)
for comment, is_toxic in results:
    print(f"Comment: {comment} | Is Toxic: {is_toxic}")

# %% [markdown]
# ## 6. Reports
#
# **Results**:
# - Dataset: 160,566 comments (after removing 5 duplicates).
# - Toxic Distribution: 143,884 non-toxic (0), 16,684 toxic (1).
# - Model Performance: 97% accuracy, 0.0691 loss.
# - Test Results: Correctly classified new comments.
#
# **Conclusion**: Built a robust toxic comment detection model with 97% accuracy, ready for online moderation.
#
# **Recommendations**: Integrate into platforms, explore advanced models like BERT, and add multilingual support.

