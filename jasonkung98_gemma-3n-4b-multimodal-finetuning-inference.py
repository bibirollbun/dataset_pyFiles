%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
    !pip install --no-deps unsloth


%%capture
# Install latest transformers for Gemma 3N
!pip install transformers==4.53.2 # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


from unsloth import FastVisionModel # FastLanguageModel for LLMs
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

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3n-E4B-it",
    max_seq_length = 2048,
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)


from transformers import TextStreamer
import gc
# Helper function for inference
def do_gemma_3n_inference(model, messages, max_new_tokens = 1024):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(processor, skip_prompt = True),
    )
    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()


walkway_link = "/kaggle/input/sample-image/0.jpg"

messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : walkway_link },
        { "type": "text",  "text" : "Can you describe the basic visual information of this image? Provide a detailed overview of the surroundings and provide guidance based on your analysis." }
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_3n_inference(model, messages, max_new_tokens = 1024)


messages = [{
    "role": "user",
    "content": [{ "type" : "text",
                  "text" : "What is the most convienient way to go to Kuala Lumpur Sentral from The Exchange TRX?" }]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 1024)


from IPython.display import Audio, display
Audio("/kaggle/input/tts-sample/ElevenLabs_Text_to_Speech_audio.mp3")


audio_file = "/kaggle/input/tts-sample/ElevenLabs_Text_to_Speech_audio.mp3"

messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : audio_file },
        { "type": "text",  "text" : "What is this audio about?" }
    ]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : audio_file },
        { "type": "image", "image" : walkway_link },
        { "type": "text",  "text" : "What is this audio and image about? "\
                                    "How are they related?" }
    ]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)


from datasets import load_dataset
dataset = load_dataset("jasonkung98/gemma3n_wad_dataset_v2", split = "test[:1]")


dataset[0]


dataset[0]["image"]


dataset[0]["scene_summary"]


dataset[0]["reminder"]


instruction = "Can you describe the basic visual information of this image: <image_soft_token>. Provide a detailed overview of the surroundings and provide guidance based on your analysis."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["reminder"]} ]
        },
    ]
    return { "messages" : conversation }
pass


converted_dataset = [convert_to_conversation(sample) for sample in dataset]


converted_dataset[0]


from unsloth import get_chat_template

processor = get_chat_template(
    processor,
    "gemma-3"
)


FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[0]["image"]
instruction = "Can you describe the basic visual information of this image: <image_soft_token>. Provide a detailed overview of the surroundings and provide guidance based on your analysis."

messages = [
    { "role": "user",
        "content" : [
        {"type" : "text",  "text"  : instruction},
        {"type" : "image", "image" : dataset[0]["image"]} ]
    },
    { "role" : "assistant",
        "content" : [
        {"type" : "text",  "text"  : dataset[0]["reminder"]} ]
    },
]

input_text = processor.apply_chat_template(messages, add_generation_prompt = True)
inputs = processor(
    image,
    input_text,
    add_special_tokens = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(processor, skip_prompt = True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024,
                   use_cache = True, temperature = 1.5, min_p = 0.1)


from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = processor,
    data_collator = UnslothVisionDataCollator(model, processor), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)


# @title Show current memory stats
import gc
gc.collect()
torch.cuda.empty_cache()
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


# @title Show final memory and time stats
gc.collect()
torch.cuda.empty_cache()
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


FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[0]["image"]
instruction = "Can you describe the basic visual information of this image: <image_soft_token>. Provide a detailed overview of the surroundings and provide guidance based on your analysis."

messages = [
    { "role": "user",
        "content" : [
        {"type" : "text",  "text"  : instruction},
        {"type" : "image", "image" : dataset[0]["image"]} ]
    },
    { "role" : "assistant",
        "content" : [
        {"type" : "text",  "text"  : dataset[0]["reminder"]} ]
    },
]

input_text = processor.apply_chat_template(messages, add_generation_prompt = True)
inputs = processor(
    image,
    input_text,
    add_special_tokens = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(processor, skip_prompt = True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024,
                   use_cache = True, temperature = 1.5, min_p = 0.1)


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HF_TOKEN")


# model.save_pretrained("gemma-3n")  # Local saving
# tokenizer.save_pretrained("gemma-3n")
model.push_to_hub("jasonkung98/gemma-3n", token = secret_value_0) # Online saving
processor.push_to_hub("jasonkung98/gemma-3n", token = secret_value_0) # Online saving


if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
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


if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-3n-finetune", tokenizer)


if True: # Change to True to upload finetune
    model.push_to_hub_merged(
        "jasonkung98/gemma-3n-finetune", processor,
        token = secret_value_0
    )


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma-3n-finetune",
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )


if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "gemma-3n-finetune",
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = "jasonkung98/gemma-3n-finetune-Q8_0-GGUF",
        token = secret_value_0,
    )

