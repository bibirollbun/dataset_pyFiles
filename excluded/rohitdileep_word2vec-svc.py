import numpy as np 
import pandas as pd
import re , nltk


df_train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df_train.head()


##Checking whether class is imbalace or no. ## 
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Count values
value_counts = df_train['rule_violation'].value_counts()

# Plot
plt.figure(figsize=(6,4))
sns.barplot(x=value_counts.index, y=value_counts.values, palette="Set2")

# Labels and title
plt.xlabel("Rule Violation")
plt.ylabel("Count")
plt.title("Distribution of Rule Violation Labels")
plt.xticks([0, 1], ['Not Violation (0)', 'Violation (1)'])
plt.show()


## Test data ###
df_test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
df_test.head()


df_train.info()


df_test.info()


df_train['body'] = df_train['body']
df_test['body'] = df_test['body'] 


## Data Cleaning 
import nltk , emoji , re 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer =  WordNetLemmatizer()

def clean_and_lemmatize(text: str):
    # 1. Convert emojis to text (keep them as tokens)
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # 2. Replace URLs with a placeholder
    text = re.sub(r'http\S+|www\.\S+', ' <URL> ', text)
    
    # 3. Replace mentions and hashtags (optional, if present in dataset)
    text = re.sub(r'@\w+', ' <USER> ', text)
    text = re.sub(r'#\w+', ' <HASHTAG> ', text)
    
    # 4. Normalize case
    text = text.lower()
    
    # 5. Keep only words and placeholders (remove other punctuation)
    text = re.sub(r'[^a-zA-Z0-9<> ]', ' ', text)
    
    # 6. Tokenize
    tokens = word_tokenize(text)
    
    # 7. Lemmatize each token
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.strip()]
    
    return lemmatized

    
df_train['body'] = df_train['body'].astype('str').apply(clean_and_lemmatize)  



from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec


### Word2vec embeddings ###

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split

# Tokenize the text column
# final_subset['tokens'] = final_subset['body'].astype(str).apply(word_tokenize)

# Train Word2Vec model (CBOW: sg=0)
w2v_model = Word2Vec(sentences=df_train['body'], vector_size=100, window=5, min_count=1, sg=0)

# Function to vectorize each sentence by averaging its word vectors
def vectorize_word(tokens, model):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

# Create feature matrix X by averaging word vectors for each tokenized sentence
X = np.array([vectorize_word(tokens, w2v_model) for tokens in df_train['body']])

# Target variable
y = df_train['rule_violation'].values

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


print(f'Train data shape {X_train.shape}')
print(f'Val data shape {X_val.shape}')


## Data Normalization ##
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

trainx_scale = sc.fit_transform(X_train)
valx_scale = sc.transform(X_val)





# import optuna
# from sklearn.svm import SVC
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold
# import numpy as np

# # Assuming you already have:
# # trainx_scale, y_train  (full training data, scaled)

# def objective(trial):
#     # Hyperparameters to tune
#     C = trial.suggest_float('C', 1, 500, log=True)
#     kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
#     gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    
#     # Create model
#     model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
    
#     # Stratified K-Fold for balanced class splits
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
#     auc_scores = []
    
#     for train_idx, val_idx in skf.split(trainx_scale, y_train):
#         X_train_fold, X_val_fold = trainx_scale[train_idx], trainx_scale[val_idx]
#         y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
#         model.fit(X_train_fold, y_train_fold)
        
#         y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
#         auc = roc_auc_score(y_val_fold, y_pred_proba)
#         auc_scores.append(auc)
    
#     # Return mean AUC across folds
#     return np.mean(auc_scores)


# # Optuna Study
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)

# print("Best parameters:", study.best_params)
# print("Best mean AUC:", study.best_value)



### model training ##
from sklearn.svm import SVC
parameters = {'C': 497.2800756864728, 'kernel': 'rbf', 'gamma': 'scale'}
model = SVC(** parameters ,  probability=True)
model.fit(trainx_scale , y_train)


## inference on val data ## 

pred_val = model.predict(valx_scale)


## Metrics calculation on val data 
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix , roc_curve, roc_auc_score

print(f'Accuracy Score {accuracy_score(y_val , pred_val)}')
print()
print()
print(f'Classification Report{classification_report(y_val , pred_val)}')
print()
print(f'Confusion Matrix {confusion_matrix(y_val , pred_val)}')
print()

auc_score =  roc_auc_score(y_val, model.decision_function(valx_scale))
print(f'auc score {auc_score}')


## Plotting ROC curve ##
fpr, tpr, thresholds = roc_curve(y_val ,model.decision_function(valx_scale))


plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()


## Test data preprocessing for inference ##
df_test['body'] = df_test['body'].astype('str').apply(clean_and_lemmatize)  
X_test = np.array([vectorize_word(tokens, w2v_model) for tokens in df_test['body']])


testx_scale = sc.transform(X_test)


### Submission 
submission  = pd.DataFrame({'row_id' : df_test.row_id , 'rule_violation' : model.predict_proba(X_test)[:, 1]} )


submission.head()


##Saving .csv file

submission.to_csv("submission.csv" , index = False)




