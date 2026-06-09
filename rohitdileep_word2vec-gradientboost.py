import pandas as pd
import numpy as np
import os


df_train  = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
df_train.head()


### Data preparation 
train_data_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"

data = []
for _ , row in df_train.iterrows():
    folder_id  = str(row[0])
    real_text_id =  str(row[1])

    if len(folder_id) ==  1:
        folder_name =  f"article_000{folder_id}"
    else :
        folder_name = f"article_00{folder_id}"
    folder_name =  os.path.join(train_data_path , folder_name )
    for file_id in ["1" , "2"] :
        file_name = f"file_{file_id}.txt"
        file_path = os.path.join(folder_name , file_name )
    
        try :
            with open(file_path , 'r') as f:
                text  = f.read()
    
        except FileNotFoundError:
            print('No file found' , file_path)

        label = 1 if file_id == real_text_id  else 0 
        data.append({'text' : text , 'label' : label})


df_train = pd.DataFrame(data)
df_train.head()


df_train.drop_duplicates(subset = 'text' , inplace = True)


import matplotlib.pyplot as plt 
import seaborn as sns

plt.figure(figsize=  (6 ,  4) )
sns.countplot(x = 'label' , data = df_train)
plt.title('real vs Fake')
plt.xlabel("Fake-0 or Real-1")
plt.ylabel('Frequency')
plt.show()


## data cleaning ##
import nltk , emoji , re 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    text = emoji.demojize(text , delimiters = (" " , " "))
    text = text.replace(":" , " ").replace("_" , " ")
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text =  re.sub(r'\W' , " " , text)
    tokens = word_tokenize(text)
    lemmatize = [lemmatizer.lemmatize(token) for token in tokens ] 
    return lemmatize


df_train['tokens'] = df_train['text'].astype('str').apply(clean_and_lemmatize)


##Word to vector conversion ###
from gensim.models import Word2Vec

w2v_model =  Word2Vec(sentences = df_train['tokens'] , vector_size = 100 , window = 5 , sg = 0)

def word2vector(model , tokens):
    vecs = [model.wv[word] for word in tokens if word in model.wv ]
    return np.mean(vecs , axis = 0 ) if vecs else np.zeros(model.vector_size)


X = np.array([word2vector(w2v_model , sentences) for sentences in df_train['tokens']])
y = df_train['label'].values


## train-test split 

from sklearn.model_selection import train_test_split

trainx , testx , trainy , testy  = train_test_split(X , y , test_size = 0.20 , random_state = 0 )
print(trainx.shape)
print(testx.shape)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import optuna


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

    model = GradientBoostingClassifier(**params, random_state=0)
    model.fit(trainx , trainy)
    pred = model.predict(testx)

    # Use cross-validation for robust evaluation
    score = accuracy_score(testy , pred)
    return score

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)



print("Best params:", study.best_params)
print("Best score:", study.best_value)


##model training ##
model = GradientBoostingClassifier(**study.best_trial.params)

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


## Data Preprocessing for submission
test_df['tokens'] = test_df['text'].astype('str').apply(clean_and_lemmatize)
X_test = np.array([word2vector( w2v_model ,  token ) for token in test_df['tokens']])


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




