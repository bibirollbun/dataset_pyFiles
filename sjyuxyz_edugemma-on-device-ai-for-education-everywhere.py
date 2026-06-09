# Uncomment and run if you need to install dependencies
# !pip install unsloth
# !pip install torch
# !pip install trl
# !pip install datasets
# !pip install wandb


from unsloth import FastModel
import torch
import os

# Set dynamo cache size limit
torch._dynamo.config.cache_size_limit = 32

MODEL_NAME = "unsloth/gemma-3n-E4B-it"
TRAIN_DATA_PATH = "data_processed/sft_dataset_10k.json"
EVAL_DATA_PATH = None
OUTPUT_DIR = "models/sft_lora-r-16_gemma-3n-E4B-it"
USE_WANDB = True


def load_model_and_tokenizer(model_name: str = "unsloth/gemma-3n-E4B-it"):
    """Load the base model and tokenizer with Unsloth optimizations."""
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        max_seq_length=32768,
        load_in_8bit=False,
        load_in_4bit=False,
        full_finetuning=False,
    )

    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    return model, tokenizer


def load_lora_model(base_model):
    """Apply LoRA configuration to the base model."""
    model = FastModel.get_peft_model(
        base_model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    return model


def formatting_prompts_func(examples, tokenizer):
    """Format conversations for training."""
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ).removeprefix('<bos>') 
        for convo in convos
    ]
    return {"text": texts}


def prepare_data(data_path: str, tokenizer):
    """Load and prepare dataset for training."""
    from datasets import load_dataset
    from unsloth.chat_templates import standardize_data_formats
    
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Dataset loaded: {dataset}")
    
    # Standardize to ShareGPT format
    dataset = standardize_data_formats(dataset)
    
    # Apply formatting
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
        num_proc=os.cpu_count() // 2
    )
    return dataset


def train(model, tokenizer, train_dataset, eval_dataset=None, 
          use_wandb=True, output_dir="./models", **kwargs):
    """Train the model with SFTTrainer."""
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    # Initialize wandb if requested
    if use_wandb:
        import wandb
        wandb.init(
            project="gemma3n-unsloth-edutech", 
            name=f"sft_lora-r-16_gemma-3n-E4B-it"
        )
        report_to = "wandb"
    else:
        report_to = "none"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,  # Effective batch size = 16
            warmup_ratio=0.03,
            num_train_epochs=2,
            # max_steps=60,  # Uncomment to limit training steps
            learning_rate=1e-5,
            logging_steps=10,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            report_to=report_to,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=250,
            save_total_limit=5,
        ),
    )

    # Train only on model responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    
    # Start training
    trainer_stats = trainer.train()

    # Save final model
    model.save_pretrained_merged(
        output_dir, 
        tokenizer, 
        save_method="merged_16bit"
    )

    print(f"✅ Final model saved to {output_dir}")
    return trainer, trainer_stats


# Load base model and tokenizer
print("Loading model and tokenizer...")
model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
print("✅ Model and tokenizer loaded successfully")


# Apply LoRA to the model
print("Applying LoRA configuration...")
model = load_lora_model(model)
print("✅ LoRA configuration applied")

# Print model info
model.print_trainable_parameters()


# Load and prepare training dataset
print("Preparing training dataset...")
train_dataset = prepare_data(TRAIN_DATA_PATH, tokenizer)
print(f"✅ Training dataset prepared with {len(train_dataset)} samples")

# Optionally load evaluation dataset
eval_dataset = None
if EVAL_DATA_PATH:
    print("Preparing evaluation dataset...")
    eval_dataset = prepare_data(EVAL_DATA_PATH, tokenizer)
    print(f"✅ Evaluation dataset prepared with {len(eval_dataset)} samples")


# Preview a sample from the dataset
if len(train_dataset) > 0:
    sample_idx = 0
    print(f"Sample {sample_idx} from training data:")
    print("-" * 50)
    print(train_dataset[sample_idx]["text"][:500])  # Show first 500 chars
    print("-" * 50)


# Start training
print("Starting training...")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Using Weights & Biases: {USE_WANDB}")

trainer, trainer_stats = train(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    use_wandb=USE_WANDB,
    output_dir=OUTPUT_DIR
)

print("\n✅ Training completed!")

