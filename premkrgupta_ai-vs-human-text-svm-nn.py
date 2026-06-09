import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , ConfusionMatrixDisplay 


filePath ="/kaggle/input/llm-detect-ai-generated-text/train_essays.csv" 
textDetection_df =pd.read_csv(filePath)
textDetection_df.head()


print(f"Column Names:\n{textDetection_df.columns}")
print('-' * 60)
print(f"Shape of DataFrame (rows, columns): {textDetection_df.shape}") # Rows and Columns
print('-' * 60)
print("Data Types and Non-null Counts:")
print(textDetection_df.info()) # Data Types Check
print('-' * 60)
print(f'no of nulls .is: {textDetection_df.isnull().sum().sum()}') # Missing Values
print(f'no of duplicates: {textDetection_df.duplicated().sum()}') # Duplicates
print('-' * 60)
print("Basic Statistical Summary for Numerical Columns:")
print(textDetection_df.describe()) # Basic Stats
print('-' * 60)
print("Missing Values Per Column:")
print(textDetection_df.isna().sum()) # Missing Values

categorical_cols = textDetection_df.select_dtypes(include='object')

##class balance would refer to whether the dataset contains equal or unequal numbers of samples for each class. for Example the value counts of the target variable in a classification problem. If the dataset is imbalanced, it may affect the performance of the model.
# Value Counts  shows how many times each unique value appears in a column categorical column mostly for classification problems only or a single column only at each time

print(f"Value Counts for Target Column 'generated':")
print(textDetection_df['generated'].value_counts())


# Visualize the class balance
sns.countplot(x='generated', data=textDetection_df, color='purple')
plt.title('Class Balance')  
plt.show()


textDetection_df['generated'].value_counts()


# Word Clouds for Each Class
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Step 1: Separate the text data into two groups
student_text = ' '.join(textDetection_df[textDetection_df['generated'] == 0]['text'])

ai_text = ' '.join(textDetection_df[textDetection_df['generated'] == 1]['text'])


# Step 2: Create the WordCloud objects
wordcloud_student = WordCloud(width=800, height=400, background_color='white').generate(student_text)
wordcloud_ai = WordCloud(width=800, height=400, background_color='black').generate(ai_text)


# Step 3: Display the generated images using matplotlib 
plt.figure(figsize=(20, 10))

# Display student word cloud in the first subplot
plt.imshow(wordcloud_student, interpolation='bilinear')
plt.title('Most Common Words in Student-Written Essays', fontsize=16)
plt.axis('off') # Hide the axes

plt.show()


plt.figure(figsize=(20, 10))
# Display AI word cloud in the second subplot
plt.imshow(wordcloud_ai, interpolation='bilinear')
plt.title('Most Common Words in AI-Generated Essays', fontsize=16)
plt.axis('off') # Hide the axes
plt.show()


# Step 1: Create a new column for word count 
# This calculates the number of words in each essay
textDetection_df['word_count'] = textDetection_df['text'].str.split().str.len()


# Step 2: Visualize the distribution using a histogram
plt.figure(figsize=(12, 7))

# Use seaborn's histplot to compare the distributions
# The 'hue' parameter automatically creates separate histograms for each class
sns.histplot(data=textDetection_df, x='word_count', hue='generated', kde=True, element='step')

plt.title('Distribution of Essay Word Count (Student vs. AI)', fontsize=16)
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Number of Essays', fontsize=12)
plt.legend(title='Generated', labels=['AI (1)', 'Student (0)'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Drop rows where 'text' is null
textDetection_df = textDetection_df.dropna(subset=['text'])

# Drop empty texts like "    "
textDetection_df['text'] = textDetection_df['text'].astype(str)
textDetection_df = textDetection_df[textDetection_df['text'].str.strip().astype(bool)]

# Drop duplicates based on text
textDetection_df = textDetection_df.drop_duplicates(subset=['text'])


print(f'no of nulls is: {textDetection_df.isnull().sum().sum()}') # Missing Values
print(f'no of duplicates: {textDetection_df.duplicated().sum()}') # Duplicates


print(textDetection_df['text'].head(10))


from sklearn.utils import resample

# Separate majority and minority classes
df_majority = textDetection_df[textDetection_df['generated'] == 0]
df_minority = textDetection_df[textDetection_df['generated'] == 1]

# Upsample the minority class to match the majority class
df_minority_oversampled = resample(df_minority, 
                                 replace=True,     # Sample with replacement to make copies
                                 n_samples=len(df_majority), # Match the number of majority samples
                                 random_state=42) 

# Combine the majority class with the upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_oversampled])

# Display the new class balance
print("New Balanced Value Counts:")
print(df_balanced['generated'].value_counts())

sns.countplot(x='generated', data=df_balanced)
plt.title('Balanced Classes (After Oversampling)')
plt.show()


#clean the text data to prevent using irrelevant words or nonmeaningful ones
import re

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove numbers and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

x = df_balanced['text']
y = df_balanced['generated']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Apply cleaning to all text entries
# Convert x_train and x_test to DataFrames to add a new column
x_train = x_train.to_frame()
x_test = x_test.to_frame()

# Apply cleaning to all text entries
x_train['clean_text'] = x_train['text'].apply(clean_text)
x_test['clean_text'] = x_test['text'].apply(clean_text)



# Converting text to features using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', min_df=5, max_df=0.9) #only keep words that appear in 5+ docs & ignore words that appear in 90%+ of docs
features_from_text_train = tfidf_vectorizer.fit_transform(x_train['clean_text']).toarray()
features_from_text_test = tfidf_vectorizer.transform(x_test['clean_text']).toarray() #features_from_text is now a feature matrix
#features_from_text.shape:  (number of essays, number of words), each row is one essay and each column is one word

print("Non-zero values:", np.count_nonzero(features_from_text_train))
print("Shape:", features_from_text_train.shape)
# After fitting the vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names[:20])  # print first 20 words


features_from_text_train


scaler = StandardScaler()
x_traScal = scaler.fit_transform(features_from_text_train)
x_tesScal = scaler.transform(features_from_text_test)
#########################################################################################
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x_tra_pca = pca.fit_transform(x_traScal)
x_tes_pca = pca.transform(x_tesScal)


#converting features_from_text matrix to a DataFrame to see each word and its corresponding TF-IDF value score of importance
import pandas as pd

features_from_textdf = pd.DataFrame(features_from_text_train, columns=feature_names)
print(features_from_textdf.head())


# Showing correlation between the features' words
top10words = features_from_textdf.mean().sort_values(ascending=False).head(10).index.tolist()
correlation_matrix1 = features_from_textdf[top10words].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix1, cmap='coolwarm', annot=True)
plt.title("Correlation Between TF-IDF Features")
plt.show()



import plotly.express as px

top_20_words = features_from_textdf.mean().sort_values(ascending=False).head(20)

fig = px.bar(
    x=top_20_words.values,
    y=top_20_words.index,
    labels={'x': 'Avg TF-IDF Score', 'y': 'Word'},
    title='Top 20 Most Important Words (by Average TF-IDF Score)',
    color=top_20_words.values,
    color_continuous_scale='Viridis'
)

fig.show()


svm = { 'linear svm' : SVC(kernel='linear') ,
       'svc (RBF kernel)': SVC(kernel='rbf') }

print('# SVM MODELS # \n')
for name , model in svm.items():
    model.fit(x_tra_pca,y_train)
    y_pred = model.predict(x_tes_pca)
    print('Accuracy of model ' , name , accuracy_score(y_test,y_pred) , '\n')
    print(classification_report(y_test,y_pred))
    print('########################################################\n')



poly_svm = SVC(kernel='poly', degree=10,C=0.11, coef0=0.9)
poly_svm.fit(x_tra_pca, y_train)

#svm polynomial model
y_pred_poly = poly_svm.predict(x_tes_pca)
print('Accuracy of model ' , 'Polynomial SVM' , accuracy_score(y_test,y_pred_poly) , '\n')
print(classification_report(y_test, y_pred_poly))
print('########################################################\n')


lin_model = SVC(kernel='linear')
lin_model.fit(x_tra_pca,y_train)
y_pred_lin = lin_model.predict(x_tes_pca)

conf_lin = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_lin))   
conf_lin.plot()
plt.title('Confusion Matrix for Linear SVM')
plt.show()


conf_poly = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_poly))
conf_poly.plot()
plt.title('Confusion Matrix for Polynomial SVM')
plt.show()


RBF_model = SVC(kernel='rbf')
RBF_model.fit(x_tra_pca,y_train)
y_pred_RBF = RBF_model.predict(x_tes_pca)

conf_RBF = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_RBF))
conf_RBF.plot()
plt.title('Confusion Matrix for RBF SVM')
plt.show()


from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# Step 1: Define hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (64, 32), (64, 32, 16)],
    'activation': ['relu', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

# Step 2: Define base model
base_mlp = MLPClassifier(max_iter=200, early_stopping=True, random_state=42)

# Step 3: Grid search
grid_search = GridSearchCV(
    base_mlp, param_grid, cv=5,
    n_jobs=-1, verbose=1, scoring='accuracy'
)
grid_search.fit(x_tra_pca, y_train)

# Step 4: Get the best model
best_model = grid_search.best_estimator_
print("\n Best Hyperparameters:")
print(grid_search.best_params_)


# Step 5: Evaluate on test set
y_predNN = best_model.predict(x_tes_pca)
acc = accuracy_score(y_test, y_predNN)
print(f"\nTest Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_predNN))

# Step 6: Cross-validation on training set
cv_scores = cross_val_score(best_model, x_tra_pca, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores (5-fold): {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# Step 7: Plot loss curve
plt.plot(best_model.loss_curve_)
plt.title("Neural Network Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Step 8: Save the best model
joblib.dump(best_model, "best_nn_model.pkl")
print(" Best model saved as 'best_nn_model.pkl'")


conf_nn = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_predNN))   
conf_nn.plot()
plt.title('Confusion Matrix for NN')
plt.show()


print("Model Comparison Summary:")
summary_data = {
    "Model": ["SVM-Linear", "SVM-RBF", "SVM-Poly", "Neural Net (Best)"],
    "Test Accuracy": [accuracy_score(y_test, y_pred_lin),
                      accuracy_score(y_test, y_pred_RBF),
                      accuracy_score(y_test, y_pred_poly),
                      accuracy_score(y_test, y_predNN)],
}
summary_df = pd.DataFrame(summary_data)
print(summary_df)
best_model_name = summary_df.loc[summary_df['Test Accuracy'].idxmax(), 'Model']
print(f'Best Model is: {best_model_name}')


import joblib

# Save the TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Save the StandardScaler
joblib.dump(scaler, 'scaler.pkl')

# Save the PCA model
joblib.dump(pca, 'pca.pkl')

print("All necessary components (vectorizer, scaler, pca, and model) are now saved!")


import pandas as pd
import joblib

test_df = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
submission_df = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')

x_test_cleaned = test_df['text'].apply(clean_text)

x_test_features = tfidf_vectorizer.transform(x_test_cleaned).toarray()

x_test_scaled = scaler.transform(x_test_features)

x_test_pca = pca.transform(x_test_scaled)

results = best_model.predict_proba(x_test_pca)[:, 1]
submission_df['generated'] = results


submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(submission_df.head())

