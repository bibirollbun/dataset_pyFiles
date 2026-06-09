!pip install autoawq --upgrade


#reference: https://pypi.org/project/autoawq/
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
quant_path = '/kaggle/working/'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
print('Loading Model')
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print('Quantizing Model')
# Quantize
model.quantize(tokenizer, quant_config=quant_config)

print('Saving Model')
# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')


from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
from awq.utils.utils import get_best_device

device = get_best_device()

quant_path = "/kaggle/working/"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Convert prompt to tokens
prompt_template = """\
<|system|>
</s>
<|user|>
{prompt}</s>
<|assistant|>"""

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

tokens = tokenizer(
    prompt_template.format(prompt=prompt), 
    return_tensors='pt'
).input_ids.to(device)

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer
    #max_seq_len=128
)

generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(generated_text)

