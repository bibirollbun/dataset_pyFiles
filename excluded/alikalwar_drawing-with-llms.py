import pandas as pd
import numpy as np
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split


# Set seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# Load data
train_df = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')
test_df = pd.read_csv('/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv')


# Create input sequences
train_df['input_text'] = "generate drawing: " + train_df['description'].str.lower()
test_df['input_text'] = "generate drawing: " + test_df['description'].str.lower()


# Split data
train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=SEED)


# Model initialization with modern tokenizer settings
MODEL_NAME = 't5-small'

# Load tokenizer with modern settings
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


# Updated Dataset Class for dynamic tokenization
class DrawingDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_length=256):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_text': self.data.iloc[idx]['input_text'],
            'target_text': self.data.iloc[idx]['drawing_commands']
        }


# Create datasets
train_dataset = DrawingDataset(train_data)
val_dataset = DrawingDataset(val_data)


# Data Collator for seq2seq tasks
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding='longest',
    max_length=256,
    return_tensors='pt'
)


# Modern Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=7,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-4,
    weight_decay=0.02,
    warmup_steps=300,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=100,
    fp16=True,
    load_best_model_at_end=True,
    report_to='none',
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    generation_max_length=256,
    generation_num_beams=5
)


# Updated Trainer without deprecated tokenizer argument
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer  # Still required for decoding but handled differently
)


# Start training
trainer.train()


# Optimized Generation Function
def generate_commands(text):
    inputs = tokenizer(
        f"generate drawing: {text.lower()}",
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=256,
        num_beams=7,
        temperature=0.85,
        repetition_penalty=3.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Generate predictions
test_df['generated_commands'] = test_df['description'].apply(generate_commands)


# Create submission
submission = test_df[['id', 'generated_commands']].rename(
    columns={'generated_commands': 'drawing_commands'}
)
submission.to_csv('submission.csv', index=False)

print("Final submission ready for download!")


# Set up package structure
!mkdir -p /kaggle/working/drawing_model
!touch /kaggle/working/drawing_model/__init__.py
!touch /kaggle/working/drawing_model/model.py


%%writefile /kaggle/working/drawing_model/model.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class DrawingModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
        
    def predict(self, text):
        input_text = f"generate drawing: {text.lower()}"
        inputs = self.tokenizer(
            input_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=256,
            num_beams=7,
            temperature=0.85,
            repetition_penalty=3.0,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# METADATA FILE
%%writefile /kaggle/working/metadata.json
{
    "id": "yourusername/drawing-with-llms-submission",
    "title": "T5-Based Drawing Command Generator",
    "code_file": "submission.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": true,
    "enable_gpu": true,
    "enable_internet": true,
    "dataset_sources": ["drawing-with-llms"],
    "competition_sources": ["drawing-with-llms"]
}


# %% [code] -- MAIN SUBMISSION SCRIPT --
%%writefile /kaggle/working/submission.py
import pandas as pd
from drawing_model.model import DrawingModel

def main():
    # Initialize model
    model = DrawingModel()
    
    # Load test data
    test_df = pd.read_csv("/kaggle/input/drawing-with-llms/test.csv")
    
    # Generate predictions
    test_df["drawing_commands"] = test_df["description"].apply(model.predict)
    
    # Create submission
    test_df[["id", "drawing_commands"]].to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()


# %% [code] -- VALIDATION CELL --
# Test the submission script
!python /kaggle/working/submission.py

# %% [code] -- FINAL PACKAGING CELL --
# Create submission zip
!cd /kaggle/working && zip -r submission.zip .

