import zipfile
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from xgboost import XGBClassifier
from nltk.corpus import stopwords


def load_json_from_zip(zip_path, json_file_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(json_file_name) as f:
            return json.load(f)

train = load_json_from_zip('/kaggle/input/whats-cooking/train.json.zip', "train.json")


#converting list to DataFrame
df = pd.DataFrame(train)


df.head()


df.shape


df.info()


#cuisine distribution
unique_cuisines = df['cuisine'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=unique_cuisines.values, y=unique_cuisines.index, palette="viridis")
plt.title('Cuisine Distribution', fontsize=16)
plt.xlabel('Number of Recipes', fontsize=12)
plt.ylabel('Cuisine Types', fontsize=12)
plt.show()


#top 20 ingredients across all cuisines
all_ingredients = df['ingredients'].explode()
ingredient_counts = Counter(all_ingredients)
top_ingredients = ingredient_counts.most_common(20)
plt.figure(figsize=(10, 6))
sns.barplot(x=[count for _, count in top_ingredients], y=[ingredient for ingredient, _ in top_ingredients], palette="viridis")
plt.title("Top 20 Ingredients Across All Cuisines", fontsize=16)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Ingredients", fontsize=12)
plt.show()


#top ingredients for each cuisine
cuisine_words = {}
for cuisine in df['cuisine'].unique():
    ingredients = df[df['cuisine'] == cuisine]['ingredients'].explode()
    cuisine_words[cuisine] = Counter(ingredients).most_common(10)
for cuisine, words in cuisine_words.items():
    plt.figure(figsize=(8, 4))
    sns.barplot(x=[count for _, count in words], y=[word for word, _ in words], palette="magma")
    plt.title(f"Top Ingredients for {cuisine.title()} Cuisine", fontsize=14)
    plt.xlabel("Frequency", fontsize=10)
    plt.ylabel("Ingredients", fontsize=10)
    plt.show()


#cleaning the text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\(|\)', '', text) 
    text = text.lower() 
    return text

df['ingredients'] = df['ingredients'].apply(lambda x: [clean_text(word) for word in x])


#removing stopwords
stop_words = set(stopwords.words('english'))
df['ingredients'] = df['ingredients'].apply(lambda x: [word for word in x if word not in stop_words])


#combining ingredients into a single string for vectorization
df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(x))


#encoding the target
label_encoder = LabelEncoder()
df['cuisine'] = label_encoder.fit_transform(df['cuisine'])

#TF-IDF vectorizing and splitting the data
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['ingredients_str'])
y = df['cuisine']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#ML modelling with XGBoost and classification report
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)

print(classification_report(y_test, xgb_pred))


#confusion matrix, precision, recall, and f1-score plot
report = classification_report(y_test, xgb_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(12, 6))
metrics = ['precision', 'recall', 'f1-score']
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i + 1)
    sns.barplot(x=report_df.index[:-3], y=report_df[metric][:-3], palette="viridis")
    plt.xticks(rotation=90)
    plt.title(f'{metric.capitalize()} for Each Class')
    plt.xlabel('Classes')
    plt.ylabel(metric.capitalize())

plt.tight_layout()
plt.show()

conf_matrix = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()

print("Summary for Weighted Avg Metrics:")
print(f"Precision: {report['weighted avg']['precision']:.2f}")
print(f"Recall: {report['weighted avg']['recall']:.2f}")
print(f"F1-Score: {report['weighted avg']['f1-score']:.2f}")



test = load_json_from_zip('/kaggle/input/whats-cooking/test.json.zip', "test.json")
test_df = pd.DataFrame(test)

#cleaning the text
test_df['ingredients'] = test_df['ingredients'].apply(lambda x: [clean_text(word) for word in x])
test_df['ingredients'] = test_df['ingredients'].apply(lambda x: [word for word in x if word not in stop_words])

#combining ingredients into a single string
test_df['ingredients_str'] = test_df['ingredients'].apply(lambda x: ' '.join(x))

#transforming using the same vectorizer used in training
x_test_transformed = vectorizer.transform(test_df['ingredients_str'])

#making predictions using trained model
test_df['cuisine'] = label_encoder.inverse_transform(xgb_model.predict(x_test_transformed))

#preparing the submission file
submission = test_df[['id', 'cuisine']]
submission.to_csv("submission.csv", index=False)

