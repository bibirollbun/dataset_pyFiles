!pip install transformers accelerate


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "/kaggle/input/qwen-3/transformers/4b/1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map=None
)
model.to("cuda")

model.eval()


input_text = "こんにちは、あなたは誰ですか？"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
    **inputs,
    max_new_tokens=100,       # 最大トークン数を増やす
    do_sample=True,           # サンプリングありにして多様な生成
    temperature=0.7,          # 生成の多様性を調整
    top_p=0.9,                # nucleus sampling の確率質量の閾値
    eos_token_id=tokenizer.eos_token_id  # 終了トークンで止める
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

