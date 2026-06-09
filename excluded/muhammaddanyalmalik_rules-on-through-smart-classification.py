import kagglehub
kagglehub.login()


# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

jigsaw_agile_community_rules_path = kagglehub.competition_download('jigsaw-agile-community-rules')
keras_distil_bert_keras_distil_bert_base_en_uncased_3_path = kagglehub.model_download('keras/distil_bert/Keras/distil_bert_base_en_uncased/3')

print('Data source import complete.')


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import torch
import os
from datasets import Dataset, Features, Value

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["WANDB_DISABLED"] = "true"

# Load dataset (edit these paths if needed)
TRAIN_PATH = os.path.join(jigsaw_agile_community_rules_path, "/kaggle/input/jigsaw-agile-community-rules/train.csv")
TEST_PATH  = os.path.join(jigsaw_agile_community_rules_path, "/kaggle/input/jigsaw-agile-community-rules/test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Show columns for debugging:
print("Train columns:", train_df.columns)
print("Test columns:", test_df.columns)
display(train_df.head())

# Select only columns needed
train_df = train_df[['body', 'rule_violation']].copy()
test_df  = test_df[['row_id', 'body']].copy()
train_df['rule_violation'] = train_df['rule_violation'].astype('float32')


features = Features({
    'body': Value('string'),
    'rule_violation': Value('float32')
})

train_ds = Dataset.from_pandas(train_df, features=features)
train_ds = train_ds.rename_column("rule_violation", "labels")

test_features = Features({
    'body': Value('string')
})
test_ds = Dataset.from_pandas(test_df[['body']], features=test_features)

# Use DistilBERT from offline folder
MODEL_PATH = keras_distil_bert_keras_distil_bert_base_en_uncased_3_path
from transformers import DistilBertTokenizerFast # Use DistilBertTokenizerFast

vocab_file_path = os.path.join(MODEL_PATH, "assets", "tokenizer", "vocabulary.txt")
print("Vocabulary file path:", vocab_file_path)
tokenizer = DistilBertTokenizerFast(vocab_file=vocab_file_path)

def tokenize_function(example):
    return tokenizer(example['body'], truncation=True, padding='max_length', max_length=128)

train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

train_ds = train_ds.remove_columns(['body'])
test_ds = test_ds.remove_columns(['body'])
# train_ds.set_format("torch") # Remove PyTorch format setting
# test_ds.set_format("torch") # Remove PyTorch format setting


print("Train dataset sample:", train_ds[0])


from transformers import TFDistilBertForSequenceClassification, DataCollatorWithPadding, TrainingArguments
import tensorflow as tf
from transformers import DistilBertTokenizerFast, DistilBertConfig
import os
import numpy as np
from datasets import Dataset, Features, Value

# Use DistilBERT from offline folder
MODEL_PATH = keras_distil_bert_keras_distil_bert_base_en_uncased_3_path
config_path = os.path.join(MODEL_PATH, "config.json")
vocab_file_path = os.path.join(MODEL_PATH, "assets", "tokenizer", "vocabulary.txt")

config = DistilBertConfig.from_json_file(config_path)
config.num_labels = 1 # Set num_labels in the config

tokenizer = DistilBertTokenizerFast(vocab_file=vocab_file_path)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")


features = Features({
    'body': Value('string'),
    'rule_violation': Value('float32')
})

train_ds = Dataset.from_pandas(train_df, features=features)
train_ds = train_ds.rename_column("rule_violation", "labels")

test_features = Features({
    'body': Value('string')
})
test_ds = Dataset.from_pandas(test_df[['body']], features=test_features)


def tokenize_function(example):
    return tokenizer(example['body'], truncation=True, padding='max_length', max_length=128)

train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

train_ds = train_ds.remove_columns(['body'])
test_ds = test_ds.remove_columns(['body'])


train_input_ids = np.array(train_ds['input_ids'])
train_attention_mask = np.array(train_ds['attention_mask'])
train_labels = tf.cast(np.array(train_ds['labels']), tf.float32) # Cast labels to float32 after converting to numpy

train_features = {'input_ids': train_input_ids, 'attention_mask': train_attention_mask}
train_tf_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(16) # Use batch size directly

test_input_ids = np.array(test_ds['input_ids'])
test_attention_mask = np.array(test_ds['attention_mask'])

test_features = {'input_ids': test_input_ids, 'attention_mask': test_attention_mask}
# Handle test set without labels for prediction later
if 'labels' in test_ds.features:
    test_labels = tf.cast(np.array(test_ds['labels']), tf.float32)
    test_tf_ds = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(16)
else:
    test_tf_ds = tf.data.Dataset.from_tensor_slices(test_features).batch(16) # Use batch size directly


# Load model configuration and then load weights from the Keras file
model = TFDistilBertForSequenceClassification(config)

# Load weights from the Keras checkpoint
weights_path = os.path.join(MODEL_PATH, "model.weights.h5")
model.load_weights(weights_path)


# Explicitly compile the model
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True) # Use from_logits=True as the model outputs logits
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

model.fit(train_tf_ds, epochs=30, verbose=1)


# Use the trained model for prediction
train_logits = model.predict(train_tf_ds).logits.squeeze() 

# Since the model outputs logits, apply sigmoid to get probabilities
train_probs = tf.sigmoid(train_logits).numpy()

# The rest of the code for evaluation remains the same
true_labels = train_df["rule_violation"].values

print(classification_report(true_labels, train_probs > 0.5, digits=4))
cm = confusion_matrix(true_labels, train_probs > 0.5)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.show()


# Use the trained model for prediction on the test set
# Ensure test_tf_ds is created correctly without labels if they don't exist in the original test_ds
test_logits = model.predict(test_tf_ds).logits.squeeze() # Use model.predict and access logits

# Apply sigmoid to get probabilities
test_probs = tf.sigmoid(test_logits).numpy()

submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "rule_violation": test_probs
})
submission.to_csv("submission.csv", index=False)
submission.head()




