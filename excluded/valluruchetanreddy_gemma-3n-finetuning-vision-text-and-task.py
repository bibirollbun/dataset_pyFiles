# Install required packages
!pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
!pip install --no-deps trl peft accelerate bitsandbytes

# Import libraries
from unsloth import FastVisionModel
import torch
from transformers import TextStreamer
from unsloth import is_bfloat16_supported
from datasets import load_dataset

# Load Gemma 3n E4B model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name= "unsloth/Gemma-3n-E2B-Instruct"
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

# Configure LoRA for vision fine-tuning
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,    # Enable vision layer fine-tuning
    finetune_language_layers=True,  # Enable language layer fine-tuning
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,                          # LoRA rank
    lora_alpha=16,                 # LoRA alpha
    lora_dropout=0,
    bias="none",
    random_state=3407,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"],
)

# Prepare your vision dataset (example format)
def format_dataset(examples):
    texts = []
    for i in range(len(examples["image"])):
        text = f"""<|user|>
<image>
{examples["question"][i]}
<|assistant|>
{examples["answer"][i]}<|end_of_text|>"""
        texts.append(text)
    return {"text": texts}

# Load and format your dataset
dataset = load_dataset("your_vision_dataset")  # Replace with your dataset
dataset = dataset.map(format_dataset, batched=True)

# Training arguments
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=30,
    ),
)

# Start training
trainer_stats = trainer.train()

# Save the fine-tuned model
model.save_pretrained("gemma_3n_finetuned")
tokenizer.save_pretrained("gemma_3n_finetuned")



# Convert fine-tuned model to TensorFlow format
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
model_name = "./gemma_3n_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Convert to TensorFlow SavedModel format
try:
    # First convert to TensorFlow model
    tf_model = TFAutoModelForCausalLM.from_pretrained(
        model_name, 
        from_tf=False,
        return_dict=True
    )
    
    # Save as TensorFlow SavedModel
    tf_model.save_pretrained("./gemma_3n_tf_model", saved_model=True)
    
except Exception as e:
    print(f"Direct conversion failed: {e}")
    print("Using alternative conversion method...")
    
    # Alternative: Use mediapipe converter for Gemma models
    from mediapipe.tasks.python.genai import converter
    
    def gemma_convert_config(backend="cpu"):
        input_ckpt = './gemma_3n_finetuned/'
        vocab_model_file = './gemma_3n_finetuned/tokenizer.model'
        output_dir = './intermediate/'
        output_tflite_file = f'./gemma_3n_finetuned_{backend}.tflite'
        
        return converter.ConversionConfig(
            input_ckpt=input_ckpt,
            ckpt_format='safetensors',
            model_type='GEMMA_2B',  # Adjust based on your model size
            backend=backend,
            output_dir=output_dir,
            combine_file_only=False,
            vocab_model_file=vocab_model_file,
            output_tflite_file=output_tflite_file
        )
    
    # Convert to TFLite using MediaPipe converter
    config = gemma_convert_config("cpu")
    converter.convert_checkpoint(config)



# Convert TFLite to MediaPipe .task format
import os
import shutil
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer

def create_task_file(tflite_path, task_output_path, model_name="gemma_3n_finetuned"):
    """Create MediaPipe .task file from TFLite model"""
    try:
        # For Gemma 3n models, the .task format is actually a zip containing multiple files
        # This is a simplified version - actual implementation may vary
        
        # Read the TFLite model
        with open(tflite_path, 'rb') as f:
            tflite_model = f.read()
        
        # Create metadata for the model
        model_meta = metadata_writer.ModelMetadataWriter(
            name=model_name,
            description=f"Fine-tuned {model_name} for vision tasks",
            version="1.0.0"
        )
        
        # Write the task file (simplified - actual format is more complex)
        with open(task_output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Task file created: {task_output_path}")
        return True
        
    except Exception as e:
        print(f"Task file creation failed: {e}")
        print("Note: Gemma 3n .task files have a complex zip structure.")
        print("Consider using pre-converted models from Hugging Face:")
        print("- google/gemma-3n-E2B-it-litert-preview")
        print("- google/gemma-3n-E4B-it-litert-preview")
        return False

# Create the .task file
create_task_file("./gemma_3n_finetuned.tflite", "./gemma_3n_finetuned.task")


