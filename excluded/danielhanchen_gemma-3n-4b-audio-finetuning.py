%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth


%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, processor = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it", # Or "unsloth/gemma-3n-E2B-it"
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


from transformers import TextStreamer
# Helper function for inference
def do_gemma_3n_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **processor.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        do_sample = False,
        streamer = TextStreamer(processor, skip_prompt = True),
    )


from datasets import load_dataset,Audio,concatenate_datasets

dataset = load_dataset("unsloth/Emilia-DE-B000000", split="train")

# Select a single audio sample to reserve for testing.
# This index is chosen from the full dataset before we create the smaller training split.
test_audio = dataset[7546]

dataset = dataset.select(range(3000))

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


from IPython.display import Audio, display
print(test_audio['text'])
Audio(test_audio['audio']['array'],rate=test_audio['audio']['sampling_rate'])


messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an assistant that transcribes speech accurately.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": test_audio['audio']['array']},
            {"type": "text", "text": "Please transcribe this audio."}
        ]
    }
]

do_gemma_3n_inference(messages, max_new_tokens = 256)


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 8,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                 # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,              # We support rank stabilized LoRA
    loftq_config = None,             # And LoftQ
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",

        # Audio layers
        "post", "linear_start", "linear_end",
        "embedding_projection",
    ],
    modules_to_save=[
        "lm_head",
        "embed_tokens",
        "embed_audio",
    ],
)


def format_intersection_data(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    for idx in range(len(samples["audio"])):
        audio = samples["audio"][idx]["array"]
        label = str(samples["text"][idx])

        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant that transcribes speech accurately.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Please transcribe this audio."}
                ]
            },
            {
                "role": "assistant",
                "content":[{"type": "text", "text": label}]
            }
        ]
        formatted_samples["messages"].append(message)
    return formatted_samples


dataset = dataset.map(format_intersection_data, batched=True, batch_size=4, num_proc=4)


def collate_fn(examples):
    texts = []
    audios = []
    
    for example in examples:
        # Apply chat template to get text
        text = processor.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        ).strip()
        texts.append(text)
    
        # Extract audios
        audios.append(example["audio"]["array"])
    
    # Tokenize the texts and process the images
    batch = processor(
        text=texts, audio=audios, return_tensors="pt", padding=True
    )
    
    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    
    # Use Gemma3n specific token masking
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if hasattr(processor.tokenizer, 'image_token_id'):
        labels[labels == processor.tokenizer.image_token_id] = -100
    if hasattr(processor.tokenizer, 'audio_token_id'):
        labels[labels == processor.tokenizer.audio_token_id] = -100
    if hasattr(processor.tokenizer, 'boi_token_id'):
        labels[labels == processor.tokenizer.boi_token_id] = -100
    if hasattr(processor.tokenizer, 'eoi_token_id'):
        labels[labels == processor.tokenizer.eoi_token_id] = -100
    
    
    batch["labels"] = labels
    return batch


from trl import SFTTrainer, SFTConfig


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=processor.tokenizer,
    data_collator=collate_fn,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        warmup_ratio = 0.1,
        #max_steps = 60,
        num_train_epochs = 1,          # Set this instead of max_steps for full training runs
        learning_rate = 5e-5,
        logging_steps = 10,
        save_strategy="steps",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",            # For Weights and Biases

        # You MUST put the below items for audio finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 2,
        max_length = 2048,
    )
)


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an assistant that transcribes speech accurately.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": test_audio['audio']['array']},
            {"type": "text", "text": "Please transcribe this audio."}
        ]
    }
]

do_gemma_3n_inference(messages, max_new_tokens = 256)


model.save_pretrained("gemma-3n")  # Local saving
processor.save_pretrained("gemma-3n")
# model.push_to_hub("HF_ACCOUNT/gemma-3n", token = "...") # Online saving
# processor.push_to_hub("HF_ACCOUNT/gemma-3n", token = "...") # Online saving


if False:
    from unsloth import FastModel
    model, processor = FastModel.from_pretrained(
        model_name = "gemma-3n", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-3N?",}]
}]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 128, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(processor, skip_prompt = True),
)


if True: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-3n", processor)


if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-3N-finetune", processor,
        token = "hf_..."
    )


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma-3N-finetune",
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )


if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "gemma-3N-finetune",
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = "HF_ACCOUNT/gemma-3N-finetune-gguf",
        token = "hf_...",
    )

