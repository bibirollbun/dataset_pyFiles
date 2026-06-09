%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N
!pip install unsloth
!pip install accelerate
!pip install trl


# Install TTS itself but DON'T pull deps (avoids breaking Kaggle's stack).
!pip -q install --no-deps TTS==0.22.0
# Add only the tiny runtime bits we actually need (these match Kaggle well).
!pip -q install --no-deps soundfile==0.12.1 unidecode==1.3.8 librosa==0.10.2.post1 scipy==1.11.4 numpy==1.26.4



# Load model
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

# Load your data (english-cree translation pairs)
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



# model training

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




# simple cree translation test
phrase = "Hello my name is Jeff"

messages = [{
    "role" : "user",
    "content": [
        { "type": "text",  "text" : phrase},
        { "type": "text",  "text": "Translate this phrase from english to cree"}
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


!pip -q install --no-deps TTS==0.22.0 soundfile==0.12.1 unidecode==1.3.8 librosa==0.10.2.post1 scipy==1.11.4 numpy==1.26.4
!pip install gtts


# Quick test
from gtts import gTTS
tts = gTTS("hello", lang='en')
tts.save("/kaggle/working/quick_test.mp3")


# --- Google TTS with Cree Phonetic Mapping ---

# Step 1: Install gTTS (Google Text-to-Speech)
import subprocess
import sys
print("ğŸ“¦ Installing Google TTS...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "gtts"])
print("âœ“ gTTS installed successfully!")

# Step 2: Import required libraries
from gtts import gTTS
from IPython.display import Audio, display
import os

# Step 3: Create phonetic mapping for Cree characters
print("\nğŸ”¤ Setting up Cree phonetic mapping...")

def cree_to_phonetic(text):
    """Convert Cree text with diacritics to phonetic approximation"""
    
    # Cree vowel mappings (approximate pronunciation)
    phonetic_map = {
        # Long vowels
        'Ä�': 'ah',      # long 'a' sound
        'Ä“': 'ay',      # long 'e' sound  
        'Ä«': 'ee',      # long 'i' sound
        'Å�': 'oh',      # long 'o' sound
        'Å«': 'oo',      # long 'u' sound
        
        # Circumflex variants
        'Ã¢': 'ah',
        'Ãª': 'ay', 
        'Ã®': 'ee',
        'Ã´': 'oh',
        'Ã»': 'oo',
        
        # Other common Cree sounds
        'Ã½': 'y',
        
        # Common Cree syllables for better pronunciation
        'kÃ®': 'kee',    # common prefix
        'nÃ®': 'nee',    # common in words
        'sÃ®': 'see',    # common sound
        'tÃ®': 'tee',    # common sound
        'wÃ®': 'wee',    # common sound
    }
    
    result = text.lower()  # Convert to lowercase for consistency
    
    # Apply syllable mappings first (longer patterns)
    for cree_sound, phonetic in phonetic_map.items():
        if len(cree_sound) > 1:  # Multi-character patterns first
            result = result.replace(cree_sound, phonetic)
    
    # Then apply single character mappings
    for cree_sound, phonetic in phonetic_map.items():
        if len(cree_sound) == 1:  # Single characters
            result = result.replace(cree_sound, phonetic)
    
    return result

# Step 4: Test with your Cree text
cree_text = "Nihkwa nitahtowin Jeff awiso" # previously translated phrase in cree (my name is Jeff)
phonetic_text = cree_to_phonetic(cree_text)

print(f"Original Cree: {cree_text}")
print(f"Phonetic:      {phonetic_text}")

# Step 5: Generate speech with Google TTS
print(f"\nğŸ�µ Generating speech...")

try:
    # Create TTS object
    tts = gTTS(
        text=phonetic_text, 
        lang='en',          # Use English voice
        slow=False          # Normal speed
    )
    
    # Save to file
    output_file = "/kaggle/working/cree_gtts.wav"
    tts.save(output_file)
    print(f"âœ“ Audio saved to: {output_file}")
    
    # Play the audio
    display(Audio(output_file))
    
    print("ğŸ�‰ SUCCESS! Google TTS generated Cree audio")
    
except Exception as e:
    print(f"â�Œ Error: {e}")
    print("ğŸ’¡ Make sure you have internet connection for Google TTS")

# Step 6: Optional - Create a reusable function
print(f"\nğŸ“� You can now use this function for any Cree text:")

def speak_cree(text, filename=None):
    """Convert Cree text to speech and save/play it"""
    if filename is None:
        filename = "/kaggle/working/cree_speech.wav"
    
    phonetic = cree_to_phonetic(text)
    print(f"Converting: '{text}' â†’ '{phonetic}'")
    
    try:
        tts = gTTS(text=phonetic, lang='en', slow=False)
        tts.save(filename)
        display(Audio(filename))
        return filename
    except Exception as e:
        print(f"Error: {e}")
        return None

# Test use:
print(f"\nğŸ”„ Testing with another Cree phrase...")
test_phrase = "mÄ�sÄ�skÄ�sÄ“n kÄ«hci"  # (example phrase)
speak_cree(test_phrase)

print(f"\nâœ… Setup complete! You can now:")
print(f"   speak_cree('your cree text here')")
print(f"   to convert any Cree text to speech")


model.save_pretrained("gemma-3n")  # Local saving
tokenizer.save_pretrained("gemma-3n")

