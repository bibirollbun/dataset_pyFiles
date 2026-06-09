import numpy as np
import pandas as pd
import os


df_train = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
df_train.head()


train_data_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"

data = []

for _ , row in df_train.iterrows():
    folder_id = str(row[0])
    real_text_id = str(row[1])

    if len(folder_id) ==1:
        folder_name = f'article_000{folder_id}'

    else :
        folder_name = f'article_00{folder_id}'

    folder_path = os.path.join(train_data_path , folder_name)
    for file in ['1' , '2']:
        filename = f'file_{file}.txt'
        file_path = os.path.join(folder_path , filename)

        try:
            with open(file_path , 'r') as f:
                text = f.read()
        except FileNotFoundError:
            print('No file found ', file_path)

        label = 1 if file ==  real_text_id  else 0 

        data.append({'text' : text , 'label' : label})

df_train = pd.DataFrame(data)


## class frequency visualisation ##
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize  = (6 , 4))
sns.countplot(x = 'label' , data = df_train)
plt.title('Real vs Fake ')
plt.xlabel("Fake- 0 or Real - 1")
plt.ylabel("Frequency")
plt.show()


df_train.info()


## data cleaning 
import re
import emoji

def clean_text(text):
    # Convert emojis to descriptive text
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace(":", " ").replace("_", " ")
    # Remove URLs
    text = re.sub(r'http\\S+|www\\.[^ ]+', '', text)
    return text

def preprocess_text(df):
    cleaned_text = df['text'].apply(clean_text)
    return cleaned_text

# Example usage:
df_train['text'] = preprocess_text(df_train)  # Applies to all rows


## Data Cleaning ##
import nltk
from nltk.stem import WordNetLemmatizer

def word_cleaning(df):
    corpus = []
    for i in range(0 , len(df)):
        sentence = re.sub(r'\W' , ' ' , str(df[i]))
        sentence = sentence.lower()
        sentence = re.sub(r'\s+', ' ', sentence)
        corpus.append(sentence)
    return corpus


def word_lemmatization(df):
    lemmatizer = WordNetLemmatizer()
    corpus = []
    for i in df :
        words = nltk.word_tokenize(str(i))
        words = [lemmatizer.lemmatize(word) for word in words ]
        corpus.append(' '.join(words))
    return corpus

df_train['text'] = word_cleaning(df_train['text'])
df_train['text'] = word_lemmatization(df_train['text'])        
        


## Word2Vec 
from gensim.models import Word2Vec

df_train['tokens'] = df_train['text'].astype(str).apply(nltk.word_tokenize)
w2v_model  = Word2Vec(sentences = df_train['tokens']  , vector_size = 100 , window = 5 , min_count = 1 , sg = 0 )



## converting words to vector ##
def word_vector(model , token):
    vecs = [ model.wv[word] for word in token if word in model.wv]
    return np.mean(vecs , axis = 0 )  if vecs else np.zeros(model.vector_size)

X = np.array([word_vector(w2v_model ,token ) for token in df_train['tokens'] ])
y = df_train['label'].values


## Train test split ##
from sklearn.model_selection import train_test_split


trainx , testx , trainy , testy  = train_test_split(X , y , random_state = 0 , test_size = 0.25)
print(trainx.shape)
print(testx.shape)


## Hyperparameter estimation ##
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import optuna 


def objective(trial):
    var_smoothing = trial.suggest_float('var_smoothing', 1e-9, 1e-5, log=True)
    model = GaussianNB(var_smoothing=var_smoothing)
    return cross_val_score(model , trainx , trainy , cv = 5 , scoring = "accuracy").mean()


study = optuna.create_study(direction = 'maximize')
study.optimize(objective , n_trials = 50)

print("Best params:", study.best_trial.params)
print("Best cross-val-score:", study.best_trial.value)



##model training ##
model = GaussianNB(**study.best_trial.params)

model.fit(trainx , trainy)

pred_val = model.predict(testx)


## Metrics calculation on val data 
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix , roc_curve, roc_auc_score

print(f'Accuracy Score {accuracy_score(testy , pred_val)}')
print()
print()
print(f'Classification Report{classification_report(testy , pred_val)}')
print()
print(f'Confusion Matrix {confusion_matrix(testy , pred_val)}')
print()

auc_score = roc_auc_score(testy  , model.predict_proba(testx)[:, 1])
print(f'auc score {auc_score}')


fpr, tpr, thresholds = roc_curve(testy ,model.predict_proba(testx)[:, 1])


plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()



### Data preparation 
test_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'

# Initialize list to hold test data
test_data = []

# Loop through each article folder
for folder in sorted(os.listdir(test_path)):
    folder_path = os.path.join(test_path, folder)
    if os.path.isdir(folder_path):
        for file_id in ["1", "2"]:
            file_name = f"file_{file_id}.txt"
            file_path = os.path.join(folder_path, file_name)

            try:
                with open(file_path, 'r') as f:
                    text = f.read()
                test_data.append({
                    "folder": folder,
                    "file_id": file_id,
                    "text": text
                })
            except FileNotFoundError:
                print(f"File not found: {file_path}")

# Create DataFrame
test_df = pd.DataFrame(test_data)

# Save for reference
test_df.to_csv("test_individual_texts.csv", index=False)

print("✅ Done! Test data read like train format.")
print(test_df.head())


test_df['text'] = word_cleaning(test_df['text'])
test_df['text'] = word_lemmatization(test_df['text'])      


test_df['tokens'] = test_df['text'].astype(str).apply(nltk.word_tokenize)
X_test = np.array([word_vector( w2v_model ,  token ) for token in test_df['tokens']])


## Predicting on test data
preds  = model.predict(X_test)
test_df['preds'] =  preds



import pandas as pd

# Example: test_df['folder'] = 'article_1501', 'article_1502', etc.
# Extract numeric ID from folder name
test_df['id'] = test_df['folder'].str.extract(r'(\d+)').astype(int)

# Decide which text is real based on predicted labels:
# For each pair (1 and 2), choose the one predicted as 'Real' (1), or default to 1
submission_rows = []
for i in range(0, len(test_df), 2):
    id_val = test_df.iloc[i]['id']
    pred1 = test_df.iloc[i]['preds']
    pred2 = test_df.iloc[i + 1]['preds']

    # Which one is predicted as real?
    if pred1 == 1 and pred2 != 1:
        real_text = 1
    elif pred2 == 1 and pred1 != 1:
        real_text = 2
    else:
        # If both are Real or both are Fake, pick the first
        real_text = 1

    submission_rows.append({'id': id_val, 'real_text_id': real_text})

# Create DataFrame and save
submission_df = pd.DataFrame(submission_rows)
submission_df = submission_df.sort_values('id')


submission_df.to_csv('submission.csv' , index = False)




