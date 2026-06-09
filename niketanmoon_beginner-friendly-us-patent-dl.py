## I have added this to submit to competition package manager section
## https://www.kaggle.com/discussions/product-feedback/532336
# Paste the below command in package manager: This is useful when running notebook without internet
# !pip install -q datasets


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


path = Path("/kaggle/input/us-patent-phrase-to-phrase-matching")


# viewing the folder
!ls {path}


# Loading training data
train_df = pd.read_csv(path/'train.csv')


train_df.head()


train_df.describe(include='object')


train_df.describe()


train_df.info()


train_df["input"] = "CONTEXT: " + train_df.context + "; TARGET: " + train_df.target + "; ANCHOR: " + train_df.anchor


train_df.head()


## change small to large if you want to train on large model for better accuracy
model_name = "microsoft/deberta-v3-small"


from datasets import Dataset, DatasetDict
train_ds = Dataset.from_pandas(train_df)


train_ds


from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer_path = "/kaggle/input/autotokenizer/tokenizer"
model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# model_tokenizer.save_pretrained("/kaggle/working/tokenizer/")


## Example of how the tokenizer tokenizes a sentence
model_tokenizer.tokenize("Hi, I'm Niketan! I am revisiting NLP again.")


def tokenise_dataset(row):
    return model_tokenizer(row["input"])


## Tokenizing all the rows
tokenized_train_ds = train_ds.map(tokenise_dataset, batched=True)


first_row = tokenized_train_ds[0]
first_row


first_row['input'], first_row['input_ids']


## How is the input id assigned
model_tokenizer.tokenize(first_row["input"])


print(model_tokenizer.vocab['▁of'])
print(model_tokenizer.convert_ids_to_tokens(265))


tokenized_train_ds = tokenized_train_ds.rename_columns({"score": "labels"})


# using train_test_split
full_train_dataset_dict = tokenized_train_ds.train_test_split(0.20, seed=42)
train_dataset = full_train_dataset_dict["train"]
val_dataset = full_train_dataset_dict["test"]


def get_preprocessed_dataset(df, tokenization_fn):
    df["input"] = "CONTEXT: " + df.context + "; TARGET: " + df.target + "; ANCHOR: " + df.anchor
    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenization_fn, batched=True)
    return ds
    


test_df = pd.read_csv(path/"test.csv")
test_dataset = get_preprocessed_dataset(test_df, tokenise_dataset)


test_dataset


def get_corr_dataset(test_pred):
    x, y = test_pred # unpacking data
    # pearson correlation: r is the [0][1] or [1][0] value in matrix
    return {
        "pearson": np.corrcoef(x,y)[0][1]
    }


from transformers import TrainingArguments, Trainer


batch_size = 128
epochs = 4
lr = 8e-5


args = TrainingArguments(
    'outputs', 
    learning_rate=lr, 
    warmup_ratio=0.1, 
    lr_scheduler_type='cosine', 
    fp16=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none"
)


# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
# model.save_pretrained("deberta-v3-small-model")
model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/deberta-v3-small-model/pytorch/default/1/deberta-v3-small", num_labels=1)
trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=model_tokenizer, compute_metrics=get_corr_dataset)


# training the model
trainer.train();


test_preds = trainer.predict(test_dataset).predictions.astype(float)
test_preds


## Clipping data between 0 and 1
## clip will change values < 0 to 0 and values > 1 to 1
test_preds = np.clip(test_preds, 0, 1)
test_preds


import datasets
submission = datasets.Dataset.from_dict({
    'id': test_dataset['id'],
    'score': test_preds
})
submission.to_csv('submission.csv', index=False)


## change small to large if you want to train on large model for better accuracy
# model_name = "microsoft/deberta-v3-small"

# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# model_tokenizer = AutoTokenizer.from_pretrained(model_name)
# model_tokenizer.save_pretrained("/kaggle/working/tokenizer/")



# Load the uploaded file
# copy your own path from input
# tokenizer_path = "/kaggle/input/autotokenizer/tokenizer"
# model_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Do the same thing for model


