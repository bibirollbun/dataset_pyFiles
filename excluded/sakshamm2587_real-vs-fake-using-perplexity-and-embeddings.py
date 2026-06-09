import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna


def read_texts_from_dir(dir_path):
  """
  Reads the texts from a given directory and saves them in the pd.DataFrame with columns ['id', 'file_1', 'file_2'].

  Params:
    dir_path (str): path to the directory with data
  """
  # Count number of directories in the provided path
  dir_count = sum(os.path.isdir(os.path.join(root, d)) for root, dirs, _ in os.walk(dir_path) for d in dirs)
  data=[0 for _ in range(dir_count)]
  print(f"Number of directories: {dir_count}")

  # For each directory, read both file_1.txt and file_2.txt and save results to the list
  i=0
  for folder_name in sorted(os.listdir(dir_path)):
    folder_path = os.path.join(dir_path, folder_name)
    if os.path.isdir(folder_path):
      try:
        with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
          text1 = f1.read().strip()
        with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
          text2 = f2.read().strip()
        index = int(folder_name[-4:])
        data[i]=(index, text1, text2)
        i+=1
      except Exception as e:
        print(f"Error reading directory {folder_name}: {e}")

  # Change list with results into pandas DataFrame
  df = pd.DataFrame(data, columns=['id', 'file_1', 'file_2']).set_index('id')
  return df

train_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
train_df=read_texts_from_dir(train_path)
test_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
test_df=read_texts_from_dir(test_path)
y_train_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
y_train=pd.read_csv(y_train_path).drop(columns=['id']).squeeze()


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2-xl" # For faster swithc to gpt2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
def score(text, tokenizer=tokenizer, model=model):
    # Handle empty or whitespace-only input
    if not text or text.strip() == "":
        return 1e9   
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids
    if input_ids.numel() == 0:  # If the text is empty, we assign it highest perplexity
        return 1e9
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss
    # Convert loss â†’ perplexit
    perplexity = torch.exp(neg_log_likelihood)
    return perplexity.item()



tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

def get_cls_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]  # wrap single string in list
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Pass through RoBERTa
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract [CLS] token embeddings
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embeddings


human=[]
ai=[]
for idx,row in train_df.iterrows():
    f1=row["file_1"]
    f2=row["file_2"]
    label=y_train[idx]
    if label==1:
        human.append(f1)
        ai.append(f2)
    else:
        human.append(f2)
        ai.append(f1)


import numpy as np
from tqdm import tqdm

human_embeddings = []
ai_embeddings = []

for i in tqdm(range(len(human))):
    human_embeddings.append(get_cls_embeddings(human[i])[0])  
    ai_embeddings.append(get_cls_embeddings(ai[i])[0])

human_embeddings = np.array(human_embeddings)
ai_embeddings = np.array(ai_embeddings)

# Labels
human_labels = np.ones(len(human))  # 1 for human
ai_labels = np.zeros(len(ai))       # 0 for AI

# Final dataset
X = np.vstack([human_embeddings, ai_embeddings])
y = np.hstack([human_labels, ai_labels])
print("X shape:", X.shape, "y shape:", y.shape)


num_trial=10


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
import numpy as np

def objective_xgb(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "tree_method": "hist",# or 'gpu_hist' if GPU is available
        "device" : "cuda",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean()
study = optuna.create_study(direction="maximize")
study.optimize(objective_xgb, n_trials=num_trial, show_progress_bar=True)
print("Best params:", study.best_params)
best_model = xgb.XGBClassifier(**study.best_params)
best_model.fit(X, y)


from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
import numpy as np

def objective_svc(trial):
    params = {
        "C": trial.suggest_float("C", 0.001, 100, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "probability": True,  # Needed to use predict_proba later
        "random_state": 42
    }
    
    model = SVC(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean()
    

study_svc = optuna.create_study(direction="maximize")
study_svc.optimize(objective_svc, n_trials=num_trial, show_progress_bar=True)
print("Best params (SVC):", study_svc.best_params)
best_svc = SVC(**study_svc.best_params, probability=True, random_state=42)
best_svc.fit(X, y)


def objective_lr(trial):
    params = {
        "C": trial.suggest_float("C", 1e-3, 100, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),  # l1 works only with 'liblinear' solver
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "max_iter": 5000,
        "class_weight": "balanced",
        "random_state": 42
    }
    model = LogisticRegression(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean()

study_lr = optuna.create_study(direction="maximize")
study_lr.optimize(objective_lr, n_trials=num_trial, show_progress_bar=True)
print("Best LR params:", study_lr.best_params)

best_lr = LogisticRegression(**study_lr.best_params, max_iter=5000, class_weight="balanced", random_state=42)
best_lr.fit(X, y)


def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }
    model = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean()


study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=num_trial, show_progress_bar=True)
print("Best RF params:", study_rf.best_params)

best_rf = RandomForestClassifier(**study_rf.best_params, class_weight="balanced", random_state=42, n_jobs=-1)
best_rf.fit(X, y)


def predict_text_proba(text, model1=best_model, model2=best_svc, model3=best_rf, model4=best_lr, threshold=0.5):
    emb = get_cls_embeddings(text)[0].reshape(1, -1)
    prob1 = model1.predict_proba(emb)[0, 1]
    prob2 = model2.predict_proba(emb)[0, 1]
    prob3 = model3.predict_proba(emb)[0, 1]
    prob4 = model4.predict_proba(emb)[0, 1]
    return (prob1+prob2+prob3+prob4)/4.0


predicted_labels=[]
for idx, row in tqdm(test_df.iterrows()):
    file_1=row['file_1']
    file_2=row['file_2']
    score1=score(file_1)
    score2=score(file_2)
    if min(score1,score2)<0.33*max(score1,score2): # The threshold is that abs(score1-score2)/max(score1,score2) should be more than 0.33
        if score1<score2:
            predicted_labels.append(1)
        else:
            predicted_labels.append(2)
    else:
        score1=predict_text_proba(file_1) # If there is no significant difference between perplexities, we use RoBERTa embeddings
        score2=predict_text_proba(file_2)
        if score1<score2:
            predicted_labels.append(2)
        else:
            predicted_labels.append(1)


print(len(predicted_labels))
submission = pd.DataFrame({
    "id": range(len(predicted_labels)), 
    "real_text_id": predicted_labels
})
submission.to_csv("submission.csv", index=False)

