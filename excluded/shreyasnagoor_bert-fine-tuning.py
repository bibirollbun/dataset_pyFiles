pip install transformers datasets torch tqdm accelerate



import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForQuestionAnswering, TrainingArguments, Trainer



# Adjust the paths to match your dataset structure in Kaggle
data_files = {
    "train": "/kaggle/input/chaii-hindi-and-tamil-question-answering/train.csv",
    "test": "/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv"
}

dataset = load_dataset("csv", data_files=data_files)



# Load the multilingual tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answer_texts = examples["answer_text"]
    answer_starts = examples["answer_start"]

    inputs = tokenizer(
        questions, 
        contexts, 
        truncation=True, 
        padding="max_length", 
        max_length=384, 
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(inputs["offset_mapping"]):
        answer_start = answer_starts[i]
        answer_text = answer_texts[i]
        answer_end = answer_start + len(answer_text)

        start_index = None
        end_index = None

        for j, (start, end) in enumerate(offsets):
            if start_index is None and start <= answer_start < end:
                start_index = j
            if end_index is None and start < answer_end <= end:
                end_index = j

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = 0

        start_positions.append(start_index)
        end_positions.append(end_index)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs.pop("offset_mapping")
    return inputs
# Filter out examples where answer_text is None
dataset = dataset.filter(lambda example: example["answer_text"] is not None)

# Apply the preprocessing function to the dataset
encoded_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names  # Remove all original columns
)




# Load the pre-trained BERT model for QA
model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./chaii-qa-model",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=200,
    save_total_limit=2,
    remove_unused_columns=False,  # This line keeps all columns needed for the model
    report_to="none"  # (Optional) disables extra logging in Kaggle notebooks
)


# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"]
)

# Start training
trainer.train()



data_files = {
    "train": "/kaggle/input/chaii-hindi-and-tamil-question-answering/train.csv",
    "test": "/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv"
}

dataset = load_dataset("csv", data_files=data_files)



print(dataset["test"].column_names)


# Define a preprocessing function for test data
def preprocess_test_data(examples):
    # Tokenize question and context
    tokenized_inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384
    )
    return tokenized_inputs

# Assuming your test dataset is stored in encoded_dataset["test"]
encoded_dataset["test"] = dataset["test"].map(preprocess_test_data, batched=True)

# Optionally, view the new columns in your test dataset:
print(encoded_dataset["test"].column_names)



def answer_question(question, context):
    # Tokenize the question and context with truncation, padding to max_length=512
    inputs = tokenizer(
        question, 
        context, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding="max_length"
    )
    # Move the tensors to the model's device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the start and end logits, using dim=-1 to ensure proper reduction
    answer_start = torch.argmax(outputs.start_logits, dim=-1)
    answer_end = torch.argmax(outputs.end_logits, dim=-1) + 1
    
    # Convert predicted tokens to string
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer

def add_prediction(example):
    # Get the prediction for a single example using the answer_question function
    example["prediction"] = answer_question(example["question"], example["context"])
    return example

# Apply the prediction function to the test dataset
predicted_dataset = encoded_dataset["test"].map(add_prediction)

# Print out the predictions
for pred in predicted_dataset:
    print("Question:", pred["question"])
    print("Prediction:", pred["prediction"])
    print("------")



model.save_pretrained("chaii-qa-model")
tokenizer.save_pretrained("chaii-qa-model")



def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))





