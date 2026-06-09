import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict
from sentence_transformers import training_args, SentenceTransformer, SentenceTransformerTrainingArguments, util
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import CrossEntropyLoss, BinaryCrossEntropyLoss

from tqdm.notebook import tqdm
tqdm.pandas()

import torch
import os
os.environ["WANDB_DISABLED"] = "true"

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
sample_submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")
print('Complete dataset files loaded')


train_df.head(2).to_dict(orient="records")


training_pairs = []

for i, row in train_df.iterrows():
    anchor = row["rule"]

    # Positive pairs
    training_pairs.append((anchor, row["positive_example_1"], 1))
    training_pairs.append((anchor, row["positive_example_2"], 1))

    # Negative pairs
    training_pairs.append((anchor, row["negative_example_1"], 0))
    training_pairs.append((anchor, row["negative_example_2"], 0))

df_pairs = pd.DataFrame(training_pairs, columns=["anchor", "other_text", "label"])


print(f"Created {len(df_pairs)} training pairs")
print(f"Label distribution: {df_pairs['label'].value_counts()}")


train_data = {
    "sentence1": [],
    "sentence2": [],
    "label": []
}

train_data = {
    "sentence1": df_pairs.anchor.tolist(),
    "sentence2": df_pairs.other_text.tolist(),
    "label": df_pairs.label.tolist()
}

ds = Dataset.from_dict(train_data)
train_valtest = ds.train_test_split(test_size=0.2, seed=42, shuffle=True)
val_test = train_valtest['test'].train_test_split(test_size=0.4, seed=42, shuffle=False)

dataset_dict = DatasetDict({
    'train': train_valtest['train'],
    'validation': val_test['train'], 
    'test': val_test['test']
})


model = CrossEncoder(model_name_or_path="/kaggle/input/cross-encoderms-marco-minilm-l-6-v2/transformers/v1/1/ms-marco-MiniLM-L-6-v2")
loss = BinaryCrossEntropyLoss(model)


args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir="models/ce-test",
    # Optional training parameters:
    num_train_epochs=2,
    per_device_train_batch_size=16, # 8
    per_device_eval_batch_size=16, # 8
    learning_rate=3e-5, #2e-5 
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True, 
    bf16=False,  
    batch_sampler=training_args.BatchSamplers.BATCH_SAMPLER, 
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Or "accuracy" if you log it
    greater_is_better=False,
    logging_steps=1,
)


trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    loss=loss,
)
trainer.train()


model.predict([(train_df.iloc[0].rule, train_df.iloc[0].body)])[0]


def predict_rule_violation(model, text, rule, threshold=0.5):
    """Predict if text violates the given rule"""
    text_formatted = text  
    rule_context = f"Rule: {rule}"
    score = model.predict([(rule_context, text_formatted)])[0]
    return score


test_df['prediction'] = test_df.apply(
    lambda x: predict_rule_violation(
        model, 
        x['rule'], 
        x['body']
    ), 
    axis=1
)


# sumission
submission = test_df[["row_id", "prediction"]].rename(columns={"prediction": "rule_violation"})
submission.to_csv("submission.csv", index=False)


submission




