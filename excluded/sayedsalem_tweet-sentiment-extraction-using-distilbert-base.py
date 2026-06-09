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


!nvidia-smi



import matplotlib.pyplot as plt
import seaborn as sns

import re

from sklearn.model_selection import train_test_split

from transformers import (AutoTokenizer, 
                          AutoModelForQuestionAnswering, 
                          TrainingArguments, 
                          Trainer, 
                          TrainerCallback,
                          pipeline)

from datasets import Dataset, ClassLabel

from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
train_df


test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test_df


sub_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
sub_df


train_df.info()


train_df.isna().sum()


# Check nan values
train_df.dropna(inplace=True)


# Random check if there is any useless words need to be removed
print(train_df['text'].sample(10).to_list())


# # Preprocess our text data
# def remove_urls(row):
#   # in we will remove the url that is in text and not in selected_text
#   text = row.text
#   selected_text = row.selected_text
#   if not re.search(r'https?://\S+|www\.\S+', selected_text):
#     return re.sub(r'https?://\S+|www\.\S+', '', text)
#   return text

# train_df['cleaned_text'] = train_df.apply(remove_urls, axis=1)


# Let's remove extra spaces
def remove_extra_spaces(text):
  return re.sub(r'\s+', ' ', text)

train_df['text'] = train_df['text'].apply(remove_extra_spaces)
train_df['selected_text'] = train_df['selected_text'].apply(remove_extra_spaces)
test_df['text'] = test_df['text'].apply(remove_extra_spaces)


sentiment_counts = train_df['sentiment'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Distribution of Tweet Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()


train_df['text_len'] = train_df['text'].apply(lambda x: len(str(x)))
train_df['selected_text_len'] = train_df['selected_text'].apply(lambda x: len(str(x)))

test_df['text_len'] = test_df['text'].apply(lambda x: len(str(x)))

print("Train DataFrame Length Analysis:")
display(train_df[['text_len', 'selected_text_len']].describe())

print("\nTest DataFrame Length Analysis:")
display(test_df[['text_len']].describe())


# You can also visualize the distributions if needed
plt.figure(figsize=(12, 6))
sns.histplot(train_df['text_len'], bins=50, color='skyblue', label='Train Text Length', kde=True)
sns.histplot(train_df['selected_text_len'], bins=50, color='salmon', label='Train Selected Text Length', kde=True)
sns.histplot(test_df['text_len'], bins=50, color='lightgreen', label='Test Text Length', kde=True)
plt.title('Distribution of Text and Selected Text Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# model_name = "roberta-base"
model_name = "distilbert-base-uncased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_batched(examples):
    questions = [q.strip() for q in examples["sentiment"]]
    contexts = [c.strip() for c in examples["text"]]
    answers = examples["selected_text"]
    
    inputs = tokenizer(
        questions,
        contexts,
        max_length=150,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        context = contexts[i]
        
        start_char = context.find(answer)
        end_char = start_char + len(answer)
        
        sequence_ids = inputs.sequence_ids(i)

        ctx_start = sequence_ids.index(1)
        ctx_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        token_start_index = 0
        token_end_index = 0
        
        for j in range(ctx_start, ctx_end + 1):
            if offset[j][0] <= start_char < offset[j][1]:
                token_start_index = j
            if offset[j][0] < end_char <= offset[j][1]:
                token_end_index = j
        
        start_positions.append(token_start_index)
        end_positions.append(token_end_index)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


train_df.columns


hf_train_df = Dataset.from_pandas(train_df)

cols_to_remove = [col for col in train_df.columns if col != 'sentiment']
tokenized_train_data = hf_train_df.map(preprocess_batched, batched=True, remove_columns=cols_to_remove)
tokenized_train_data


model = AutoModelForQuestionAnswering.from_pretrained(model_name)


class_label_feature = ClassLabel(names=train_df['sentiment'].unique().tolist())

tokenized_data_with_sentiment = tokenized_train_data.cast_column(
    "sentiment", class_label_feature
)

train_val_split = tokenized_data_with_sentiment.train_test_split(
    test_size=0.1,
    seed=42,
    stratify_by_column="sentiment" # Use the column name as a string
)

train_val_split


train_dataset = train_val_split['train']
validation_dataset = train_val_split['test']

train_dataset = train_dataset.remove_columns(['sentiment'])
validation_dataset = validation_dataset.remove_columns(['sentiment'])

train_dataset, validation_dataset


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    # max_steps=5,
    report_to="none",         
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)


trainer.train()


def preprocess_test_data(examples):
    questions = [q.lower().strip() for q in examples["sentiment"]]
    contexts = [c.lower().strip() for c in examples["text"]]
    
    inputs = tokenizer(
        questions,
        contexts,
        max_length=150,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
    )
    
    # inputs["example_id"] = [i for i in range(len(inputs["input_ids"]))] # to save the order of the samples
    return inputs


test_df.columns


hf_test_df = Dataset.from_pandas(test_df)

test_dataset = hf_test_df.map(
    preprocess_test_data,
    batched=True,
    # remove_columns=test_df.columns.to_list()
)

test_dataset


raw_predictions = trainer.predict(test_dataset)
raw_predictions


start_logits, end_logits = raw_predictions.predictions
start_logits, end_logits


final_predictions = []

for i, example in enumerate(test_dataset):
    
    start_logit = start_logits[i]
    end_logit = end_logits[i]
    
    offsets = example["offset_mapping"]
    original_text = example["text"] 
    
    start_index = np.argmax(start_logit)
    end_index = np.argmax(end_logit)
    
    if start_index < len(offsets) and end_index < len(offsets) and offsets[start_index] and offsets[end_index]:
        start_char, _ = offsets[start_index]
        _, end_char = offsets[end_index]
        
        predicted_answer = original_text[start_char:end_char]
    else:
        predicted_answer = original_text

    if start_index > end_index:
        predicted_answer = original_text
        
    final_predictions.append({
        'textID': example['textID'],
        'text': original_text,
        'sentiment': example['sentiment'],
        'selected_text': predicted_answer
    })


test_predictions_df = pd.DataFrame(final_predictions)
test_predictions_df.head()


submission_df = test_predictions_df[['textID','selected_text']]
submission_df.head()


submission_df.to_csv('submission.csv', index=False)




