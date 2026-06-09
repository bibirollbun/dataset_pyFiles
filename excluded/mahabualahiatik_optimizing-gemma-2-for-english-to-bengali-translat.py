!pip install sacrebleu


# Install pip3-autoremove if not already installed
!pip install pip3-autoremove

# Uninstall old Torch and related packages
!pip-autoremove torch torchvision torchaudio -y

# Install Torch, TorchVision, and TorchAudio with CUDA 12.1
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121 --quiet

# Install additional required libraries
!pip install unsloth datasets trl --quiet



import pandas as pd

from pprint import pprint

import csv

# Function to convert CSV data into the desired JSON format

def convert_csv_to_json_format(csv_file):

    list_ds = []  # Initialize an empty list to store the formatted data
    # Open and read the CSV file from the specified path

    with open(csv_file, newline='', encoding='utf-8') as file:

        csv_reader = csv.DictReader(file)

        # Loop through each row in the CSV

        for row in csv_reader:

            english_sentence = row['english_caption']

            bangla_sentence = row['bengali_caption']

            # Append the formatted dictionaries to the list

            list_ds.append({

                "instruction": "Translate this to English",

                "input": bangla_sentence,

                "output": english_sentence

            })

            list_ds.append({

                "instruction": "Translate this to Bangla",

                "input": english_sentence,

                "output":  bangla_sentence

            })



    return list_ds  # Return the populated list



# Define the path to your CSV file

csv_file = '/kaggle/input/english-to-bengali-for-machine-translation/english to bengali.csv'  



# Call the function to convert the CSV into the desired format

list_ds = convert_csv_to_json_format(csv_file)

# Now print the result

pprint(list_ds[:2])  #print 2 lines



from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None  # Auto-detect based on hardware
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-2-9b-it", 
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)



import datasets

# Updated prompt name to reflect the translation task
translation_prompt = """Below is a task instruction paired with input text. Your job is to provide an accurate translation.

### Task:
{}

### Input Text:
{}

### Translation:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Ensure EOS token is defined

# Optimized function name and implementation
def format_translation_prompts(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    # Using list comprehension for readability and efficiency
    texts = [
        translation_prompt.format(instruction, input_text, output) + EOS_TOKEN
        for instruction, input_text, output in zip(instructions, inputs, outputs)
    ]
    return {"text": texts}

# Convert your DataFrame to a Hugging Face Dataset
df = pd.DataFrame(list_ds)
dataset = datasets.Dataset.from_pandas(df)

# Apply the formatting function to add the 'text' field
dataset = dataset.map(format_translation_prompts, batched=True)

# Print the final dataset to verify
print(dataset)



from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=25,
        max_steps=500,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer_stats = trainer.train()




FastLanguageModel.for_inference(model) # Unsloth has 2x faster inference!
inputs = tokenizer(
[
    translation_prompt.format(
        "Translate to English", # instruction
        "হাসনাত বলেন, ‘আমরা উদ্বেগের সঙ্গে লক্ষ্য করছি- সরকার এখনো ঘোষণাপত্রের ব্যাপারে দৃশ্যমান কোনো উদ্যোগ নেয়নি।", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


from sacrebleu import corpus_bleu

# Prepare references and hypotheses
def evaluate_bleu(test_dataset, model, tokenizer, translation_prompt, device="cuda"):
    references, hypotheses = [], []

    for example in test_dataset:
        # Format the input text using the alpaca_prompt
        input_text = translation_prompt.format(
            example['instruction'], example['input'], ""
        )
        reference = example['output']  # The ground truth response

        try:
            # Tokenize the input, ensuring consistent formatting and length
            inputs = tokenizer(
                [input_text],  # Wrap input in a list for batch processing
                return_tensors="pt",
                truncation=True,
                max_length=512,  # Set a maximum input length
                padding="max_length"  # Ensure consistent input size
            ).to(device)

            # Generate the hypothesis using the model
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,  # Limit the output length
                use_cache=True  # Speed up generation
            )

            # Decode the generated output
            hypothesis = tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0]  # Decode the first (and only) hypothesis

            # Append the reference and hypothesis to their respective lists
            references.append([reference])  # BLEU expects a list of references for each hypothesis
            hypotheses.append(hypothesis)

        except Exception as e:
            # Handle errors gracefully to avoid crashing
            print(f"Error generating text for input: {input_text}\n{e}")
            continue

    # Compute BLEU score
    try:
        bleu = corpus_bleu(hypotheses, references)
        print(f"BLEU Score: {bleu.score:.2f}")
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        bleu = None

    return bleu


# Ensure you have a small test dataset to validate
test_dataset = dataset.select(range(100))  # Use first 100 samples for testing

# Call the BLEU evaluation function
bleu_score = evaluate_bleu(
    test_dataset=test_dataset,
    model=model,
    tokenizer=tokenizer,
    translation_prompt=translation_prompt,  # Ensure this matches your prompt format
    device="cuda"
)



model_name = "Gemma2_BanglaEnglish"  

# Save the fine-tuned model and tokenizer locally
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)

print(f"Fine-tuned model and tokenizer saved as '{model_name}'")


