import pandas as pd
import numpy as np
import os


df_train  = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')
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


df_train.info()


df_train.drop_duplicates(subset='text', inplace=True)


## class frequency visualisation ##
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize  = (6 , 4))
sns.countplot(x = 'label' , data = df_train)
plt.title('Real vs Fake ')
plt.xlabel("Fake- 0 or Real - 1")
plt.ylabel("Frequency")
plt.show()


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




from sklearn.model_selection import train_test_split


trainx , testx , trainy , testy  = train_test_split(df_train['text'] , df_train['label'] , random_state = 0 , test_size = 0.25)
print(trainx.shape)
print(testx.shape)



## Loading Tokenizer ##
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")



## Converting dataset to tensorflow format #
import tensorflow as tf
train_encodings =  tokenizer(list(trainx) , padding = True , truncation = True , return_tensors="tf")
test_encodings =  tokenizer(list(testx) , padding = True , truncation = True , return_tensors="tf")

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    trainy
)).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    testy
)).batch(8)


## Loading model 
from transformers import TFDistilBertForSequenceClassification

model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)



## Early stopping and callbacks ##
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True
)



### model training ##
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    callbacks=[early_stop, checkpoint]
)



import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

y_true = []
y_probs = []

for batch in test_dataset:
    inputs, labels = batch
    outputs = model(inputs, training=False)

    # outputs.logits shape: (batch_size, 1)
    logits = outputs.logits  # raw logits

    probs = tf.squeeze(tf.nn.sigmoid(logits), axis=-1)  # convert logits to probability
    y_probs.extend(probs.numpy())
    y_true.extend(labels.numpy())

y_probs = np.array(y_probs)
y_true = np.array(y_true)

# Convert probabilities to class predictions
y_pred = (y_probs >= 0.5).astype(int)


from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, digits=4))



from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
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
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                test_data.append({
                    "folder": folder,
                    "file_id": file_id,
                    "text": text
                })
            except FileNotFoundError:
                print(f" File not found: {file_path}")

# Create DataFrame
test_df = pd.DataFrame(test_data)

# Save for reference
test_df.to_csv("test_individual_texts.csv", index=False)

print("✅ Done! Test data read like train format.")
print(test_df.head())





## for inference ## 

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load the same model architecture
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

# Load trained weights
model.load_weights("best_model.h5")


test_df['text'] = preprocess_text(test_df) 
inputs = tokenizer(list(test_df['text']) , padding=True, truncation=True, return_tensors="tf")





def run_batched_inference(texts, batch_size=16):
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="tf")

        # Predict logits
        outputs = model(inputs, training=False)
        logits = outputs.logits  # shape: (batch_size, 1)

        # Convert logits to probabilities
        probs = tf.squeeze(tf.nn.sigmoid(logits), axis=-1)  # shape: (batch_size,)
        preds = (probs >= 0.5).numpy().astype(int)  # binary thresholding

        all_preds.extend(preds)
        all_probs.extend(probs.numpy())

    return np.array(all_preds), np.array(all_probs)

# Run on test data
preds, probs = run_batched_inference(test_df['text'].tolist(), batch_size=32)



test_df['preds'] =  preds
test_df['predicted_label_text'] = test_df['preds'].map({1: 'Real', 0: 'Fake'})


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


submission_df.head()


submission_df.to_csv("submission.csv", index=False)





