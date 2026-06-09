# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# Load from Kaggle or local path
df = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")  # or use the file you have already

# Show class distribution
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['num_labels'] = df[label_cols].sum(axis=1)
print(df[label_cols].sum())



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
df[label_cols].sum().sort_values().plot(kind='barh', color='salmon')
plt.title("Class Distribution")
plt.show()



df['num_labels'] = df[label_cols].sum(axis=1)
print(df['num_labels'].value_counts())



import matplotlib.pyplot as plt

# Compute class distribution
label_counts = df['num_labels'].value_counts().sort_index()

# Plot
plt.figure(figsize=(8,4))
label_counts.plot(kind='barh', color='salmon')
plt.title("Number of Labels per Comment")
plt.xlabel("Number of Comments")
plt.ylabel("Number of Labels")
plt.show()



df_toxic = df[df[label_cols].sum(axis=1) > 0]   # all toxic rows
df_nontoxic = df[df[label_cols].sum(axis=1) == 0].sample(n=5000, random_state=42)  # balance it a bit
df_sample = pd.concat([df_toxic, df_nontoxic])



df_sample


import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words('english')]
    return " ".join(words)

df_sample['clean_text'] = df_sample['comment_text'].apply(preprocess)



df_sample.to_csv("/kaggle/working/processed.csv")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_sample['clean_text'])
y = df_sample[label_cols].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

logreg = OneVsRestClassifier(LogisticRegression(class_weight="balanced"))
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred, target_names=label_cols))



from sklearn.metrics import accuracy_score
report = classification_report(y_test, y_pred, target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], output_dict=True, zero_division=0)
model_evaluation = {'Logistic Regression' : {'F1_Score':report['weighted avg']['f1-score'],
                                            'Accuracy': accuracy_score(y_test,y_pred),
                                            'Recall':report['weighted avg']['recall'],
                                            'Precision':report['weighted avg']['precision']}}


model_evaluation


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier


rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf, target_names=label_cols))


report = classification_report(y_test, y_pred_rf, target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], output_dict=True, zero_division=0)
random_forest = {'Random Forest' : {'F1_Score':report['weighted avg']['f1-score'],
                                            'Accuracy': accuracy_score(y_test,y_pred_rf),
                                            'Recall':report['weighted avg']['recall'],
                                            'Precision':report['weighted avg']['precision']}}


model_evaluation['Random Forest'] = random_forest['Random Forest']
model_evaluation


xgb = MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1))
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost Performance:")
print(classification_report(y_test, y_pred_xgb, target_names=label_cols))



report = classification_report(y_test, y_pred_xgb, target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], output_dict=True, zero_division=0)
xg_boost = {'XG Boost' : {'F1_Score':report['weighted avg']['f1-score'],
                                            'Accuracy': accuracy_score(y_test,y_pred_xgb),
                                            'Recall':report['weighted avg']['recall'],
                                            'Precision':report['weighted avg']['precision']}}


model_evaluation['XG Boost'] = xg_boost['XG Boost']
model_evaluation


import matplotlib.pyplot as plt
import numpy as np


# Extract metric names and model names
metrics = ['F1_Score', 'Accuracy', 'Recall', 'Precision']
models = list(model_evaluation.keys())

# Convert to a 2D list for plotting
metric_values = [[model_evaluation[model][metric] for model in models] for metric in metrics]

# Plot settings
x = np.arange(len(models))  # label locations
width = 0.2  # width of the bars

plt.figure(figsize=(10, 6))

# Plot each metric as a group of bars
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, metric_values[i], width=width, label=metric)

# Add labels and formatting
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Comparison: F1 Score, Accuracy, Recall, Precision')
plt.xticks(x + width * 1.5, models)
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show plot
plt.show()



from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer(max_features=5000)
X_counts = vectorizer.fit_transform(df_sample['clean_text'])
words = np.array(vectorizer.get_feature_names_out())

def top_words_for_class(class_index):
    mask = (df_sample[label_cols[class_index]] == 1).values  # Convert to NumPy array
    toxic_words = X_counts[mask].sum(axis=0)
    sorted_indices = np.argsort(np.asarray(toxic_words).flatten())[::-1]
    return words[sorted_indices[:10]]

for i, label in enumerate(label_cols):
    print(f"Top words in '{label}': {top_words_for_class(i)}")



from sklearn.linear_model import LogisticRegressionCV
import seaborn as sns
import matplotlib.pyplot as plt

lengths = df['comment_text'].apply(lambda x: len(x.split()))
probs = logreg.predict_proba(X_test)  # shape: [samples, labels]
errors = np.abs(probs[:, 0] - y_test[:, 0])  # example: 'toxic' label

sns.scatterplot(x=lengths[:len(errors)], y=errors)
plt.xlabel("Comment Length")
plt.ylabel("Prediction Error")
plt.title("Error vs. Length â†’ Evidence of Heteroscedasticity")
plt.show()


from sklearn.model_selection import train_test_split

# Step 1: Split the raw data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df_sample['comment_text'], df_sample[label_cols], test_size=0.2, random_state=42
)

# Step 2: Vectorize test set
X_test = tfidf.transform(X_test_text)

# Step 3: Predict
probs = logreg.predict_proba(X_test)
errors = np.abs(probs[:, 0] - y_test.iloc[:, 0])  # 'toxic' class

# Step 4: Compute lengths for matching test comments
lengths = X_test_text.apply(lambda x: len(x.split()))

mask = lengths < 200
# Step 5: Plot â€” now perfectly aligned
sns.scatterplot(x=lengths[mask], y=errors[mask])
plt.xlabel("Comment Length")
plt.ylabel("Prediction Error")
plt.title("Error vs. Length â†’ Evidence of Heteroscedasticity")
plt.show()



!pip install transformers sentencepiece datasets


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# T5-based paraphraser
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
paraphrase_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

# Pipeline for easy generation
paraphraser = pipeline("text2text-generation", model=paraphrase_model, tokenizer=paraphrase_tokenizer)


# Focus on rare classes
rare_df = df_sample[df_sample[["threat", "severe_toxic", "identity_hate"]].sum(axis=1) > 0]
samples_to_augment = rare_df.sample(n=500, random_state=42)

def truncate_text(text, max_tokens=450):
    tokens = text.split()
    return ' '.join(tokens[:max_tokens])

samples_to_augment["comment_text"] = samples_to_augment["comment_text"].apply(truncate_text)



augmented_texts = []

for text in samples_to_augment["comment_text"]:
    input_text = f"paraphrase: {text} </s>"
    # Generate 3 paraphrases per sample to boost minority class count
    result = paraphraser(
    input_text,
    max_length=256,
    num_return_sequences=3,     # ğŸ”� more paraphrases
    num_beams=10,
    do_sample=True,
    top_k=120,
    top_p=0.95
    )
    paraphrased = result[0]['generated_text']
    augmented_texts.append(paraphrased)



augmented_texts


augmented_df = samples_to_augment.copy()
augmented_df["comment_text"] = augmented_texts

# Combine original + augmented
balanced_df = pd.concat([df_sample, augmented_df], ignore_index=True)

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save if needed
balanced_df.to_csv("balanced_textual_toxic.csv", index=False)


balanced_df.shape


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
balanced_df[label_cols].sum().sort_values().plot(kind='barh', color='salmon')
plt.title("Class Distribution")
plt.show()


balanced_df['num_labels'] = balanced_df[label_cols].sum(axis=1)
print(balanced_df[label_cols].sum())




