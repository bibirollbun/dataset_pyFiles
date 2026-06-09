import pandas as pd

train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

sample = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

print(train.info)
print('='*70)
print(test.info)
print('='*70)
print(sample.info)


from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

def create_input_text(df):
    return "Comment: " + df['body'].fillna('') + \
           " | Rule: " + df['rule'].fillna('') + \
           " | Positive Example 1: " + df['positive_example_1'].fillna('') + \
           " | Positive Example 2: " + df['positive_example_2'].fillna('') + \
           " | Negative Example 1: " + df['negative_example_1'].fillna('') + \
           " | Negative Example 2: " + df['negative_example_2'].fillna('')

train_df['text'] = create_input_text(train_df)
test_df['text'] = create_input_text(test_df)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].tolist(),
    train_df['rule_violation'].tolist(),
    test_size=0.2,
    random_state=42
)

model_name = '/kaggle/input/distilbertbaseuncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=512)


class ViolationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = ViolationDataset(train_encodings, train_labels)
val_dataset = ViolationDataset(val_encodings, val_labels)
test_dataset = ViolationDataset(test_encodings)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    do_eval=True,
    eval_steps=800,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

predictions = trainer.predict(test_dataset)

probs = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=-1)
violation_probs = probs[:, 1].tolist()

sample_submission['rule_violation'] = violation_probs
sample_submission.to_csv('submission.csv', index=False)

print("submission.csv created successfully!")

