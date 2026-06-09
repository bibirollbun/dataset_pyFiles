!pip install cairosvg


%%capture
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth
!pip install vllm


from transformers import AutoProcessor, AutoModel
import cairosvg

from PIL import Image
import io

smodel = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to("cuda:1")
sprocessor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")


from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)


from unsloth import is_bfloat16_supported
import torch
max_seq_length = 2048  # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Generate SVG code to visually represent the following text description, while respecting the given constraints.
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>"A red circle with a blue square inside"</description>
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 384">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>

Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted and no ellipses.

<description>"{}"</description>

Respond in the following format:
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 384">
...
</svg>
"""

XML_COT_FORMAT = """\
<answer>
{answer}
</answer>
"""


# uncomment middle messages for 1-shot prompting
def make_dataset(split = "train") -> Dataset:
    data = load_dataset('csv', data_files='/kaggle/input/drawing-with-llms/train.csv')[split]
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['description']}
        ],
    }) # type: ignore
    return data # type: ignore

dataset = make_dataset()

# Reward functions 
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r'^<answer>\n<svg viewBox="0 0 384 384" width="384" height="384">\n.*?\n</svg>\n</answer>\n$'
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    print('-'*20, f"\nResponse:\n{responses[0]}")
    return [0.5 if match else 0.0 for match in matches]



def svg_to_png(svg_code: str):
        # Ensure SVG has proper size attributes
        if 'viewBox' not in svg_code:
            svg_code = svg_code.replace(
                '<svg', f'<svg viewBox="0 0 384 384"'
            )

        # Convert SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
        return Image.open(io.BytesIO(png_data)).convert('RGB').resize((384,384))

def svg_score(target_class, svg):
    target_class = "SVG illustration of " +  target_class # add
    # Convert SVG to PNG
    try:
        image = svg_to_png(svg)
        display(image)
        # Preprocess image and text
        inputs = sprocessor(
            text=[target_class], images=image, padding="max_length", return_tensors="pt"
        ).to('cuda:1')
    
        # Get features and normalize
        with torch.no_grad():
            outputs = smodel(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
    
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    
            # Calculate similarity scores
            similarities = (image_features @ text_features.T).squeeze()
        score = similarities.item()
    except:
        score = 0
    return score

def svg_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    prompts = [prompt for prompt in kwargs['description']]
    scores =  [svg_score(target_class_, svg_) for target_class_, svg_ in zip(prompts, contents)]
    return scores


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = 2048,
    max_completion_length = 2048,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 5,
    save_steps = 5,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        svg_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()




