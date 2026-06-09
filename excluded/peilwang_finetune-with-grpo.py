%%capture
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth
!pip install vllm


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


from dataclasses import dataclass, field

from defusedxml import ElementTree as etree


@dataclass(frozen=True)
class SVGConstraints:
    """Defines constraints for validating SVG documents.

    Attributes
    ----------
    max_svg_size : int, default=10000
        Maximum allowed size of an SVG file in bytes.
    allowed_elements : dict[str, set[str]]
        Mapping of the allowed elements to the allowed attributes of each element.
    """

    max_svg_size: int = 10000
    allowed_elements: dict[str, set[str]] = field(
        default_factory=lambda: {
            'common': {
                'id',
                'clip-path',
                'clip-rule',
                'color',
                'color-interpolation',
                'color-interpolation-filters',
                'color-rendering',
                'display',
                'fill',
                'fill-opacity',
                'fill-rule',
                'filter',
                'flood-color',
                'flood-opacity',
                'lighting-color',
                'marker-end',
                'marker-mid',
                'marker-start',
                'mask',
                'opacity',
                'paint-order',
                'stop-color',
                'stop-opacity',
                'stroke',
                'stroke-dasharray',
                'stroke-dashoffset',
                'stroke-linecap',
                'stroke-linejoin',
                'stroke-miterlimit',
                'stroke-opacity',
                'stroke-width',
                'transform',
            },
            'svg': {
                'width',
                'height',
                'viewBox',
                'preserveAspectRatio',
            },
            'g': {'viewBox'},
            'defs': set(),
            'symbol': {'viewBox', 'x', 'y', 'width', 'height'},
            'use': {'x', 'y', 'width', 'height', 'href'},
            'marker': {
                'viewBox',
                'preserveAspectRatio',
                'refX',
                'refY',
                'markerUnits',
                'markerWidth',
                'markerHeight',
                'orient',
            },
            'pattern': {
                'viewBox',
                'preserveAspectRatio',
                'x',
                'y',
                'width',
                'height',
                'patternUnits',
                'patternContentUnits',
                'patternTransform',
                'href',
            },
            'linearGradient': {
                'x1',
                'x2',
                'y1',
                'y2',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'radialGradient': {
                'cx',
                'cy',
                'r',
                'fx',
                'fy',
                'fr',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'stop': {'offset'},
            'filter': {
                'x',
                'y',
                'width',
                'height',
                'filterUnits',
                'primitiveUnits',
            },
            'feBlend': {'result', 'in', 'in2', 'mode'},
            'feColorMatrix': {'result', 'in', 'type', 'values'},
            'feComposite': {
                'result',
                'style',
                'in',
                'in2',
                'operator',
                'k1',
                'k2',
                'k3',
                'k4',
            },
            'feFlood': {'result'},
            'feGaussianBlur': {
                'result',
                'in',
                'stdDeviation',
                'edgeMode',
            },
            'feMerge': {
                'result',
                'x',
                'y',
                'width',
                'height',
                'result',
            },
            'feMergeNode': {'result', 'in'},
            'feOffset': {'result', 'in', 'dx', 'dy'},
            'feTurbulence': {
                'result',
                'baseFrequency',
                'numOctaves',
                'seed',
                'stitchTiles',
                'type',
            },
            'path': {'d'},
            'rect': {'x', 'y', 'width', 'height', 'rx', 'ry'},
            'circle': {'cx', 'cy', 'r'},
            'ellipse': {'cx', 'cy', 'rx', 'ry'},
            'line': {'x1', 'y1', 'x2', 'y2'},
            'polyline': {'points'},
            'polygon': {'points'},
        }
    )

    def validate_svg(self, svg_code: str) -> None:
        """Validates an SVG string against a set of predefined constraints.

        Parameters
        ----------
        svg_code : str
            The SVG string to validate.

        Raises
        ------
        ValueError
            If the SVG violates any of the defined constraints.
        """
        # Check file size
        if len(svg_code.encode('utf-8')) > self.max_svg_size:
            raise ValueError('SVG exceeds allowed size')

        # Parse XML
        tree = etree.fromstring(
            svg_code.encode('utf-8'),
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )

        elements = set(self.allowed_elements.keys())

        # Check elements and attributes
        for element in tree.iter():
            # Check for disallowed elements
            tag_name = element.tag.split('}')[-1]
            if tag_name not in elements:
                raise ValueError(f'Disallowed element: {tag_name}')

            # Check attributes
            for attr, attr_value in element.attrib.items():
                # Check for disallowed attributes
                attr_name = attr.split('}')[-1]
                if (
                    attr_name not in self.allowed_elements[tag_name]
                    and attr_name not in self.allowed_elements['common']
                ):
                    raise ValueError(f'Disallowed attribute: {attr_name}')

                # Check for embedded data
                if 'data:' in attr_value.lower():
                    raise ValueError('Embedded data not allowed')
                if ';base64' in attr_value:
                    raise ValueError('Base64 encoded content not allowed')

                # Check that href attributes are internal references
                if attr_name == 'href':
                    if not attr_value.startswith('#'):
                        raise ValueError(
                            f'Invalid href attribute in <{tag_name}>. Only internal references (starting with "#") are allowed. Found: "{attr_value}"'
                        )


svg_validator = SVGConstraints()


import re

def extract_svg(text):
    match = re.search(r'<generated_svg>(.*?)</generated_svg>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(data):
    
    instruction = 'Generate svg code for an image that looks like:'
    inputs       = data["prompt"]
    outputs      = data["completion"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input.split('Generate svg code for an image that looks like:')[1], output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
    
from datasets import load_dataset
dataset = load_dataset("thesantatitan/deepseek-svg-dataset", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


def is_valid_data(data):
    svg_code = extract_svg(data['text'])
    try:
        svg_validator.validate_svg(svg_code)
        return True
    except:
        return False


filtered_dataset = dataset.filter(is_valid_data)


filtered_dataset


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 2e-6,
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
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 2048,
    max_completion_length = 2048,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 20,
    save_steps = 20,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)


# Reward functions 
def format_reward_func(completions, **kwargs) -> list[float]:
    try:
        svg_validator.validate_svg(svg_code)
        pattern = r'^<generated_svg>\s*<svg.*?</svg>\s*</generated_svg>$'
        # responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in completions]
        return [0.5 if match else 0.3 for match in matches]
    except:
        return 0

def count_xml(text) -> float:
    count = 0.0
    if text.count("\n<svg>\n") == 1:
        count += 0.125
    if text.count("\n</svg>\n") == 1:
        count += 0.125
    if text.count("\n<generated_svg>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</generated_svg>\n")[-1])*0.001
    if text.count("\n</generated_svg>") == 1:
        count += 0.125
        count -= (len(text.split("\n</generated_svg>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<generated_svg>\n.*?\n</generated_svg>\n$"
    # responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<generated_svg>.*?</generated_svg>"
    # responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        format_reward_func,
        strict_format_reward_func,
        soft_format_reward_func
    ],
    args = training_args,
    train_dataset = filtered_dataset,
)
trainer.train()


model.save_lora("/kaggle/working/grpo_saved_lora")


FastLanguageModel.for_inference(model) # Unsloth has 2x faster inference!
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Generate svg code for an image that looks like:", # instruction
        "a goose winning a gold medal", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask,
                         max_new_tokens = 512, use_cache = True)
tokenizer.batch_decode(outputs)













