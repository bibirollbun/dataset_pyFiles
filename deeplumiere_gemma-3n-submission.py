%%capture
!pip install unsloth
# !pip install transformers>=4.54.1


%%capture
# !pip install --no-deps git+https://github.com/huggingface/transformers.git 
!pip install transformers -U
!pip install --no-deps --upgrade timm


import pkg_resources
transformers_version = pkg_resources.get_distribution("transformers").version
print(f"Current Transformers version: {transformers_version}")

def version_to_tuple(version_str):
    return tuple(map(int, (version_str.split('.'))))

if version_to_tuple(transformers_version) < version_to_tuple("0.34.1"):
    print("Transformers version is less than 0.34.1. Reinstalling...")
    !pip install --no-deps git+https://github.com/huggingface/transformers.git
else:
    print("Transformers version is 0.34.1 or higher. No reinstallation needed.")


from unsloth import FastModel
import torch
from IPython.display import Image, display
import gc
import json

torch._dynamo.config.cache_size_limit = 256


fourbit_models = [
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
]
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",
    dtype = None,
    max_seq_length = 4096,
    load_in_4bit = True,
    full_finetuning = False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = True, 
    finetune_language_layers   = True,  
    finetune_attention_modules = True,  
    finetune_mlp_modules       = True,  

    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


def do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens = 4096):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda:0")

    generated_token_ids = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
    )

    
    generated_text = tokenizer.decode(generated_token_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    del inputs
    del generated_token_ids 
    torch.cuda.empty_cache()
    gc.collect()

    return generated_text


report_image = "https://raw.githubusercontent.com/DeepLumiere/MedOCR/master/beta/img.png"

# Display the image directly from the URL
display(Image(url=report_image))


report_image = "https://raw.githubusercontent.com/DeepLumiere/MedOCR/master/beta/img.png"

list_tests_messages = [{
    "role": "user",
    "content": [
        { "type": "image", "image": report_image },
        { "type": "text", "text": "From this medical report image, please list all the distinct medical test names you can identify. Present them as a comma-separated list." }
    ]
}]

print("--- Extracting Test Features from the Report Image ---")
print("Please wait, the model is analyzing the image...")

raw_test_list_output = do_gemma_3n_inference(model, tokenizer, list_tests_messages, max_new_tokens=128)

test_names = []
if ',' in raw_test_list_output:
    test_names = [name.strip() for name in raw_test_list_output.split(',') if name.strip()]
else:
    for line in raw_test_list_output.split('\n'):
        clean_line = line.strip()
        if clean_line and (clean_line.startswith(('* ', '- ', '• ')) or (clean_line[0].isdigit() and '. ' in clean_line[:3])):
            clean_line = clean_line.split('. ', 1)[-1].strip() 
        if clean_line:
            test_names.append(clean_line)

print("\n--- Identified Test Features (Copy/Type one into the next cell) ---")
items_per_row = 3
for i, test_name in enumerate(test_names):
    print(f"{test_name:<30}", end="") 
    if (i + 1) % items_per_row == 0:
        print() 
print()
print("---------------------------------------------------------------")


finder = input("Enter the test name you want to review: ")

messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : report_image },
        { "type": "text",  "text" : finder + " . Find value, review whether it's in normal range, if not give potential reasons." }
    ]
}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda:0")
outputs = model.generate(
    **inputs,
    max_new_tokens = 2048,
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)

labreport = do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens = 128)


print(labreport)


model.save_pretrained("gemma-3n") 
tokenizer.save_pretrained("gemma-3n")


