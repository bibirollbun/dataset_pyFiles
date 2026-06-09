import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns




!pip install transformers datasets scikit-learn pandas seaborn matplotlib



!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip


df=pd.read_csv('/kaggle/working/train.csv')


df.head()


label_cols=['toxic' , 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']
df[label_cols].sum().sort_values().plot(kind='barh')
plt.title('Label Distribution')
plt.show()



df['comment_length'] = df['comment_text'].apply(lambda x : len(x.split()))
df['comment_length'].plot(kind='hist' , bins=50)


import re

def clean_text(text):
    text=re.sub(r"http\S+" , "" , text) #removing URLs
    text=re.sub(r"\n" , " " , text)
    return text


df['clean_text']=df['comment_text'].apply(clean_text)


from sklearn.model_selection import train_test_split

df['clean_text']=df['comment_text'].fillna("")

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['labels']= df[label_cols].values.tolist()


train_texts, val_texts, train_labels , val_labels = train_test_split(
    df['clean_text'].tolist() , df['labels'].tolist() , test_size=0.1
)



from transformers import BertTokenizer

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings=tokenizer(train_texts , truncation =True , padding=True , max_length=128)
val_encodings=tokenizer(val_texts , truncation=True , padding=True , max_length=128)



import torch

class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToxicDataset(train_encodings, train_labels)
val_dataset = ToxicDataset(val_encodings, val_labels)



from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6,
    problem_type="multi_label_classification"
)


from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}



from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
import os
os.environ["WANDB_DISABLED"] = "true"

class PrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(logs)

training_args = TrainingArguments(
    output_dir="./results",                # still required
    run_name="toxic-bert-run",
    per_device_train_batch_size=4,         # smaller to avoid OOM
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    do_eval=True,
    logging_dir="./logs",
    logging_steps=100,
    
    save_strategy="no",                 # ✅ disables checkpoint saving
    load_best_model_at_end=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[PrintCallback()]
)

trainer.train()



def predict(text):
    # Make sure model is in eval mode
    model.eval()

    # Tokenize and move inputs to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    # Also move model to correct device
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()
    
    return dict(zip(label_cols, probs[0]))

text = "You are the worst human being."
prediction = predict(text)

for label, score in prediction.items():
    print(f"{label}: {score:.2f}")


!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip


# Load test data
test_df = pd.read_csv("/kaggle/working/test.csv")
test_texts = test_df["comment_text"].fillna("").tolist()

# Tokenize test data
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
test_dataset = ToxicDataset(test_encodings, labels=[[0]*6]*len(test_df))  # Dummy labels



# Get predictions
preds_raw = trainer.predict(test_dataset)
logits = preds_raw.predictions

# Convert logits to probabilities using sigmoid
probs = torch.sigmoid(torch.tensor(logits)).numpy()

# Apply threshold
threshold = 0.5
final_preds = (probs >= threshold).astype(int)



# Load sample submission file to get proper format
sample_submission = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

# Fill in predictions
sample_submission[label_cols] = final_preds

# Save to CSV
sample_submission.to_csv("submission.csv", index=False)














