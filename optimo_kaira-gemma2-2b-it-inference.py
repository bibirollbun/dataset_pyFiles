# let's use the parameter efficient finetuning library for LORA finetuning
!pip --q install peft


import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM




MODEL_NAME = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2"

FINETUNED_PATH = "/kaggle/input/kaira-gemma2-2b-it/transformers/default/1"
# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, FINETUNED_PATH)

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
# Ensure the model is in evaluation mode
model.eval()


# Function to generate slang translation
def generate_slang_translation(paragraph, max_length=1024):
    prefix = "Réécris le texte suivant en argot:\n"
    input_text = prefix + paragraph
    # inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)

    input_text = USER_CHAT_TEMPLATE.format(prompt=input_text)

    input_ids = tokenizer(input_text, return_tensors="pt", truncation=None).to("cuda")

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            max_length=max_length
        )

    # Decode the generated text
    output_text = tokenizer.decode(output_ids[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return output_text


your_text_to_translate = """
Bonjour! Pour utiliser ce modèle c'est vraiment extrèmement simple, pas besoin de s'arracher les cheveux: il vous suffit de remplacer ce petit texte par celui que vous voulez traduire! C'est quand même plutôt simple non ?!"""

slang_translation = generate_slang_translation(your_text_to_translate)

print(f"Texte original:\n{your_text_to_translate}\n")

print(f"Traduction en argot:\n{slang_translation}")


