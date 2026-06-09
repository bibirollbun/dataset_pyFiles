%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N
!pip install unsloth
!pip install accelerate
!pip install trl


from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it", # Or "unsloth/gemma-3n-E2B-it"
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_audio_layers      = True,
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


import json
import random

# Load your data
with open("/kaggle/input/english-cree-translations/filtered_english_cree.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare conversations
conversations = []

for english, cree in data.items():
    conversations.append({
        "conversations": [
            {"role": "user", "content": f"Translate this to Cree: {english}"},
            {"role": "assistant", "content": cree}
        ]
    })

# Shuffle for randomness
random.shuffle(conversations)

# Save to JSONL file
output_path = "cree_translation_unsloth_format.jsonl"
with open(output_path, "w", encoding="utf-8") as out_file:
    for convo in conversations:
        out_file.write(json.dumps(convo, ensure_ascii=False) + "\n")

print(f"âœ… Saved {len(conversations)} conversation entries to '{output_path}'")



from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


from unsloth.chat_templates import standardize_data_formats
from datasets import load_dataset

dataset = load_dataset("json", data_files="cree_translation_unsloth_format.jsonl", split="train")
dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
    return {
        "text": [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in examples["conversations"]
        ]
    }

dataset = dataset.map(formatting_prompts_func, batched=True)



from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        max_steps = 100,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
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


# added in response to too many recompilations due to varied input shapes
torch._dynamo.config.cache_size_limit = 32
trainer_stats = trainer.train()


import json
import random

with open("/kaggle/input/indigenous-stories/indigenous_stories_expanded.json", "r", encoding="utf-8") as f:
    stories = json.load(f)

# Randomly select one story to preview
story = random.choice(stories)


# from unsloth.chat_templates import get_chat_template
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "gemma-3",
# )

from transformers import TextStreamer
import gc

# Helper function for inference
def do_gemma_3n_inference(model, messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
        tokenize = True,
        return_dict = True,
    ).to("cuda")
    
    _ = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()




messages = [{
    "role" : "user",
    "content": [
        { "type": "text",  "text" : story}, 
        { "type": "text",  "text": "Create a simplified bedtime version of this story \
        for a child"}
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


messages = [{
    "role" : "user",
    "content": [
        { "type": "text",  "text" : story},
        { "type": "text",  "text": "What is the spiritual significance of this story and lessons that can be learnt?"}
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


# audio_file = "/kaggle/input/childrens-song/Wake Up.mp3"

# messages = [{
#     "role" : "user",
#     "content": [
#         { "type": "audio", "audio" : audio_file },
#         { "type": "text",  "text" : "What is this audio about?" }
#     ]
# }]
# inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt = True, # Must add for generation
#         tokenize = True,
#         return_dict = True,
#         return_tensors = "pt",
#     ).to("cuda")
# _ = model.generate(
#     **inputs,
#     max_new_tokens = 256,
#     temperature = 1.0, top_p = 0.95, top_k = 64,
#     streamer = TextStreamer(tokenizer, skip_prompt = True),
# )
# # Cleanup to reduce VRAM usage
# del inputs
# torch.cuda.empty_cache()
# gc.collect()


# !pip install gTTS pydub

# from gtts import gTTS
# from pydub import AudioSegment
# from pydub.playback import play
# import IPython.display as ipd

# # 1. Generate TTS from lyrics
# tts = gTTS(text=output_lyrics, lang='en')
# tts.save("output_audio.mp3")

# # 2. Play back the result in Kaggle
# ipd.Audio("output_audio.mp3")



model.save_pretrained("gemma-3n")  # Local saving
tokenizer.save_pretrained("gemma-3n")

