%%writefile infer.py

import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_length", type=int)
    
    args = parser.parse_args()
    MODEL_NAME = args.model_path
    MAX_LENGTH = args.max_length

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def preprocess_function(examples):
        return tokenizer(examples['problem'], max_length=MAX_LENGTH, padding=False, truncation=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    training_args = TrainingArguments(
        ".", 
        per_device_eval_batch_size=4,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args, 
        tokenizer=tokenizer,
    )
    
    prompts = {
        "Algebra": 0,
        "Combinatorics": 1,
        "Geometry": 2,
        "Number Theory": 3
    }
    
    ka_map = {
        "Algebra": 0,
        "Combinatorics": 5,
        "Geometry": 1,
        "Number Theory": 4
    }
    
    id2label = {v: ka_map[k] for k, v in prompts.items()}
    print(id2label)
    
    test = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv") \
           .rename(columns={"Question": "problem"})
    print(test)  # Changed display() to print() since display is typically a Jupyter notebook function
    
    ds_comp = Dataset.from_pandas(test)
    ds_comp_enc = ds_comp.map(preprocess_function, batched=True)
    
    predictions = trainer.predict(ds_comp_enc).predictions
    print(predictions[0])
    
    labels = [id2label[int(pred.argmax())] for pred in predictions]
    
    test["label"] = labels
    test[["id", "label"]].to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()


INFERENCE_MODEL_PATH = "/kaggle/input/bogoconic1-topic-prediction-exp2-modernbert-base"
INFERENCE_MAX_LENGTH = 1024


!accelerate launch --num_processes 2 infer.py \
      --model_path $INFERENCE_MODEL_PATH \
      --max_length $INFERENCE_MAX_LENGTH

