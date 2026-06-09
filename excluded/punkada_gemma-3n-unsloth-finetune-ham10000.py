# Cell 1: Perform a single, atomic installation to prevent dependency conflicts.

# By listing all packages and their pinned versions in one command, we force pip
# to find a compatible set without overriding our choices.
!pip install "unsloth[kaggle-new]" "transformers==4.54.1" "timm==1.0.19"

print("\nâœ… Installation complete.")


# Cell 2: Verify Installed Library Versions
import transformers
import timm

print("--- Checking Installed Library Versions ---")
print(f"âœ… transformers: {transformers.__version__}")
print(f"âœ… timm:         {timm.__version__}")
print("-----------------------------------------")
print("Expected: transformers==4.54.1 and timm==1.0.19")


# Cell 3 (WITH PRE-EMPTIVE COMPILER DISABLE)
import unsloth
import torch

# --- INCREASE THE COMPILER CACHE LIMIT ---
print("Increasing PyTorch JIT compiler cache limit...")
torch._dynamo.config.cache_size_limit = 512
print("Cache limit increased.")

from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it",
    dtype = None,
    max_seq_length = 2048,
    load_in_4bit = True,
    # We can leave this as False, as the global disable is stronger.
    use_gradient_checkpointing = False,
)


# Cell 4
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 32,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 32,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,              # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    modules_to_save = [               # Important for adapting to new tasks
        "lm_head",
        "embed_tokens",
    ],
    # Forcefully disable re-entrant checkpointing at the model level
    gradient_checkpointing_kwargs = {"use_reentrant" : False},
)


# Cell 5

import pandas as pd
from datasets import Dataset, Image
import os

# --- Define Paths ---
BASE_PATH = "/kaggle/input/skin-cancer-mnist-ham10000/"
METADATA_PATH = os.path.join(BASE_PATH, "HAM10000_metadata.csv")
IMAGES_PART1_PATH = os.path.join(BASE_PATH, "HAM10000_images_part_1")
IMAGES_PART2_PATH = os.path.join(BASE_PATH, "HAM10000_images_part_2")

# --- Load Metadata ---
print("Loading metadata from CSV...")
df = pd.read_csv(METADATA_PATH)

# --- Create a mapping from image_id to its full file path ---
image_paths_part1 = {os.path.splitext(f)[0]: os.path.join(IMAGES_PART1_PATH, f) for f in os.listdir(IMAGES_PART1_PATH)}
image_paths_part2 = {os.path.splitext(f)[0]: os.path.join(IMAGES_PART2_PATH, f) for f in os.listdir(IMAGES_PART2_PATH)}
full_image_paths = {**image_paths_part1, **image_paths_part2}

# --- Add file paths and full diagnosis names to the dataframe ---
print("Mapping image IDs to file paths...")
df['image_path'] = df['image_id'].map(full_image_paths)

# --- Create a dictionary for full diagnosis names for better captions ---
dx_full_names = {
    'akiec': "Actinic Keratoses and Intraepithelial Carcinoma / Bowen's disease",
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}
df['dx_full'] = df['dx'].map(dx_full_names)

# --- Clean up data ---
# Drop rows where the image file might be missing
df.dropna(subset=['image_path'], inplace=True)

# --- FIX THE 'age' COLUMN TYPE MISMATCH ---
# 1. Fill missing age values with the string 'unknown'.
# 2. Explicitly convert the ENTIRE column to the string data type.
# This creates a consistent type that PyArrow can handle without errors.
print("Cleaning and standardizing data types...")
df['age'] = df['age'].fillna('unknown').astype(str)


# --- Convert pandas DataFrame to Hugging Face Dataset ---
print("Converting to Hugging Face Dataset object...")
# This will now work without the ArrowInvalid error
dataset = Dataset.from_pandas(df)

# --- Load the actual image data into the 'image' column ---
# This is the final step to create the dataset in the format the trainer expects
print("Loading image data into the dataset...")
dataset = dataset.cast_column("image_path", Image(decode=True)).rename_column("image_path", "image")

print("\n ðŸ¤™ðŸ˜ŽðŸ¤™  HAM10000 dataset loaded and prepared successfully.")
print(f"Total samples loaded: {len(dataset)}")


# Cell 6: Inspect the dataset
dataset


# Cell 6.5 

from datasets import ClassLabel

# ---
PROTOTYPING_PERCENTAGE = 0.20 
# ---

# The train_test_split function requires the stratification column to be of the 'ClassLabel' type. 
print("Casting 'dx' column to ClassLabel for stratification...")
dataset = dataset.cast_column("dx", ClassLabel(names=df['dx'].unique().tolist()))


# --- DATASET SPLITTING ---
print("Performing a stratified 80/20 train-test split...")
full_dataset_split = dataset.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column='dx' 
)

# Isolate the full 80% training set and the fixed 20% test set.
full_train_dataset = full_dataset_split['train']
test_dataset = full_dataset_split['test']

# Check PROTOTYPING_PERCENTAGE to decide on the final training set size.
if PROTOTYPING_PERCENTAGE < 1.0:
    # This is a PROTOTYPE run.
    num_prototype_samples = int(len(full_train_dataset) * PROTOTYPING_PERCENTAGE)
    train_dataset = full_train_dataset.select(range(num_prototype_samples))
    print(" ðŸ¤™ðŸ˜ŽðŸ¤™  PROTOTYPING MODE ENABLED")
    print(f"   Using {PROTOTYPING_PERCENTAGE * 100}% of the training data ({num_prototype_samples} samples).")
else:
    # This is a FULL run.
    train_dataset = full_train_dataset
    print(" ðŸ¤™ðŸ˜ŽðŸ¤™  FULL RUN MODE ENABLED")
    print("   Using the entire training dataset.")

print("\n--- Final Dataset Sizes for this Run ---")
print(f"Training examples: {len(train_dataset)}")
print(f"Test examples:     {len(test_dataset)} (This is a fixed 20% of the total for consistent evaluation)")

# You can optionally inspect the features of the new dataset to confirm the change
print("\n--- Dataset Features After Casting ---")
print(test_dataset.features)


# Cell 7: View a sample image
test_dataset[20]["image"]


# Cell 8: View the corresponding caption
# View the corresponding metadata
print(f"Diagnosis Code: {test_dataset[20]['dx']}")
print(f"Full Diagnosis: {test_dataset[20]['dx_full']}")
print(f"Age: {test_dataset[20]['age']}")
print(f"Sex: {test_dataset[20]['sex']}")
print(f"Localization: {test_dataset[20]['localization']}")


# Cell 9

# New instruction for the dermatologist domain
instruction = "You are an expert dermatologist. Describe the lesion in this image based on the provided context."

def generate_caption_from_metadata(sample):
    """Creates a descriptive sentence from the HAM10000 metadata."""
    
    # Start with the core diagnosis
    caption = f"A dermatoscopic image of a {sample['dx_full']} lesion."
    
    # Add localization
    if sample.get('localization') and sample['localization'] != 'unknown':
        caption += f" It is located on the {sample['localization']}."
        
    # Add patient demographics
    age = sample.get('age', 'unknown')
    sex = sample.get('sex', 'unknown')
    
    # The 'if' statement correctly handles the 'unknown' case.
    # Inside the 'if', we now convert the string to a float first, then to an int.
    if age != 'unknown' and sex != 'unknown':
        caption += f" The patient is a {int(float(age))}-year-old {sex}."
    elif age != 'unknown':
        caption += f" The patient is {int(float(age))} years old."
        
    # Add confirmation type
    if sample.get('dx_type'):
        dx_type_map = {
            'histo': 'confirmed by histopathology',
            'follow_up': 'diagnosed via follow-up examination',
            'consensus': 'diagnosed by expert consensus',
            'confocal': 'confirmed by in-vivo confocal microscopy'
        }
        confirmation_text = dx_type_map.get(sample['dx_type'], '')
        if confirmation_text:
            caption += f" The diagnosis was {confirmation_text}."
            
    return caption

def convert_to_conversation(sample):
    """
    Converts a sample from the HAM10000 dataset into the required
    conversational format for the model.
    """
    generated_caption = generate_caption_from_metadata(sample)
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": generated_caption}]},
    ]
    return {"messages": conversation}

pass


# Cell 10: Apply the conversion to the entire dataset
converted_train_dataset = [convert_to_conversation(sample) for sample in train_dataset]
print(f"Converted {len(converted_train_dataset)} samples for training.")


# Cell 11

# Generate and print the caption for a sample to verify
print("--- Generated Caption ---")
print(generate_caption_from_metadata(train_dataset[20]))

# Display the image for context
print("\n--- Image ---")
display(train_dataset[20]["image"])


# Cell 12
from unsloth import get_chat_template

processor = get_chat_template(
    processor,
    "gemma-3n"
)


# Cell 13 (Improved with eos_token_id)
import torch
from transformers import TextStreamer

# The compiler fix from Cell 3 is still active and working.

FastVisionModel.for_inference(model)  # Enable for inference

image = test_dataset[20]["image"]
instruction = """You are a highly specialized labeling AI. Your ONLY function is to identify the primary diagnosis in the image and output it in a single, specific sentence.

You MUST follow this format EXACTLY:
"A dermatoscopic image of a [Full Diagnosis Name] lesion."

Do NOT include any other text, headers, markdown, or explanations. Generate ONLY the single sentence for the following image:"""

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

if image.mode == 'L':
    image = image.convert('RGB')

inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(processor, skip_prompt=True)
print("\nPre-training generation:")

# --- IMPROVEMENT APPLIED HERE ---
# We tell the generator to stop as soon as it produces the End of Sentence (EOS) token.
# This is much more efficient than waiting for max_new_tokens.
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64, # Reduced max tokens as a safeguard
                   eos_token_id = processor.eos_token_id)


# Cell 14
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_train_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor, resize=512),
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,

        gradient_checkpointing = False,
        gradient_checkpointing_kwargs = {"use_reentrant" : False},

        max_grad_norm = 0.3,
        warmup_steps = 5,
        # max_steps = 100, # Increase for a full run, e.g., num_train_epochs = 1
        num_train_epochs = 2,
        learning_rate = 2e-4,
        logging_steps = 1,
        save_strategy="steps",
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",

        # Arguments for vision fine-tuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    )
)


# Cell 15: Show current memory stats before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Cell 16: Start training
trainer_stats = trainer.train()


# Cell 17: Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# Cell 18 (Improved with eos_token_id)
FastVisionModel.for_inference(model)

image = test_dataset[20]["image"]
instruction = """You are a highly specialized labeling AI. Your ONLY function is to identify the primary diagnosis in the image and output it in a single, specific sentence.

You MUST follow this format EXACTLY:
"A dermatoscopic image of a [Full Diagnosis Name] lesion."

Do NOT include any other text, headers, markdown, or explanations. Generate ONLY the single sentence for the following image:"""

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

if image.mode == 'L':
    image = image.convert('RGB')

inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(processor, skip_prompt=True)
print("Post-training generation:")

# --- SAME IMPROVEMENT APPLIED HERE ---
# Add eos_token_id to stop generation efficiently.
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 254,
                   eos_token_id = processor.eos_token_id)


# Cell 19 
# --- 1. Local Saving (Good practice to do this first) ---
print("Saving adapters and processor locally...")
model.save_pretrained("lora_model")
processor.save_pretrained("lora_model")
print("Local save complete.")

# --- Log into the Hugging Face Hub to push the model
# --- uncomment to enable 
'''
# --- 2. Authenticate with Hugging Face ---
from huggingface_hub import login

# Replace "hf_..." with your HF token
HF_TOKEN = "hf_" # PASTE YOUR WRITE TOKEN HERE
login(token=HF_TOKEN)
print("Successfully logged into Hugging Face.")

# --- 3. Define Your Model Repository ID ---
# Use your HF username and a creative name for your model.
HF_REPO_ID = "YOUR_USER/YOUR-MODEL-REPO" 
print(f"Preparing to upload to: {HF_REPO_ID}")

# --- 4. Push the Adapters and Processor to the Hub ---
# This will create the repository if it doesn't exist.
print("Uploading model adapters...")
model.push_to_hub(HF_REPO_ID)
print("Model upload complete.")

print("Uploading processor...")
processor.push_to_hub(HF_REPO_ID)
print("Processor upload complete.")

print(f"\n ðŸ¤™ðŸ˜ŽðŸ¤™  Successfully uploaded your model to: https://huggingface.co/{HF_REPO_ID}")
'''


#cell 20
# Select ONLY 1 to save! (Both not needed!)

# Save locally to 16bit
if False: model.save_pretrained_merged("unsloth_finetune", processor,)

# To export and save to your Hugging Face account
if False: model.push_to_hub_merged("YOUR-USER/YOUR-MODEL-NAME", processor, token = "PUT_HERE")

