import time
from datetime import datetime, timedelta
stopped_at = datetime.now() + timedelta(hours=11, minutes=30)
start_at = time.perf_counter()
print("Start time:", str(stopped_at))


!pip install -q -U gemma-template
!pip install -q evaluate rouge_score sacrebleu nltk


%%capture
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y


!pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install -q 'unsloth==2025.1.1'
!pip uninstall -q transformers -y
!pip install -q 'transformers==4.47.1'


import os
import sys
import json
import random
from pathlib import Path

model_name = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2"
project_id = "gemma-template-gemma-2b-it-v2-competition-v2"

seed = 3407

if 'google.colab' in sys.modules:
    # Running on Colab
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    os.environ['WANDB_API_KEY'] = userdata.get('WANDB_TOKEN')
elif os.path.exists('/kaggle/working'):
    # Running on Kaggle
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ['HF_TOKEN'] = user_secrets.get_secret("HF_TOKEN")
    os.environ['WANDB_API_KEY'] = user_secrets.get_secret("WANDB_TOKEN")
else:
    # Not running on Colab or Kaggle
    raise EnvironmentError('This notebook is designed to run on Google Colab or Kaggle.')


try:
    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 3072 # Choose any! We auto support RoPE Scaling internally!
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    
    model.config.use_cache = False
    model.print_trainable_parameters()
except (Exception, RuntimeError):
    # Test template, tokenizer required.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


tokenizer.padding_side = "right"


from gemma_template import Template, gemma_template, vietnamese_gemma_template
from gemma_template.__version__ import __version__
print(__version__)


from gemma_template import Template, FieldPosition, GEMMA_TEMPLATE, INPUT_TEMPLATE, OUTPUT_TEMPLATE, INSTRUCTION_TEMPLATE, PROMPT_TEMPLATE


"""<start_of_turn>user
{{ input }}<end_of_turn>
<start_of_turn>model
{{ output }}<end_of_turn>

"""


"""{{ system_prompt }}
{% if instruction %}\n{{ instruction }}\n{% endif %}
{% if prompt_structure %}{{ prompt_structure }}\n{% else %}{{ prompt }}\n{% endif %}
# Text:
{{ input }}
{% if topic_value %}\nTopics: {{ topic_value }}\n{% endif %}{% if keyword_value %}Keywords: {{ keyword_value }}\n{% endif %}
"""


"""{% if structure_fields %}{% for field in structure_fields %}## **{{ field.label.custom or field.label.default }}:**\n{% if field.key == 'title' %}### {% endif%}{{ field.value }}\n\n{% endfor %}{% else %}{{ output }}{% endif %}"""


"""# Role:
You are a highly skilled professional content writer, linguistic analyst, and multilingual expert specializing in structured writing and advanced text processing.

# Task:
Your primary objectives are:
1. Simplification: Rewrite the input text or document to ensure it is accessible and easy to understand for a general audience while preserving the original meaning and essential details.
2. Lexical and Grammatical Analysis: Analyze and refine vocabulary and grammar using unigrams (single words), bigrams (two words), and trigrams (three words) to enhance readability and depth.
3. Structure and Organization: Ensure your response adheres strictly to the prescribed structure format.
4. Language Consistency: Respond in the same language as the input text unless explicitly directed otherwise.

# Additional Guidelines:
1. Provide a rewritten, enhanced version of the input text, ensuring professionalism, clarity, and improved structure.
2. Focus on multilingual proficiency, using complex vocabulary, grammar to improve your responses.
3. Preserve the context and cultural nuances of the original text when rewriting.

# Text Analysis:
Example 1: Unigrams (single words){% for word in unigrams %}\n{{ word }} => {{ language }}{% endfor %}
Text Analysis 3: These are common {{ language }} words, indicating the text is in {{ language }}.

Example 2: Bigrams (two words){% for word in bigrams %}\n{{ word }} => {{ language }}{% endfor %}
Text Analysis 2: Frequent bigrams in {{ language }} confirm the language context.

Example 3: Trigrams (three words){% for word in trigrams %}\n{{ word }} => {{ language }}{% endfor %}
Text Analysis 3: Trigrams further validate the linguistic analysis and the necessity to respond in {{ language }}.

# Conclusion of Text Analysis:
The linguistic analysis confirms the text is predominantly in {{ language }}. Consequently, the response should be structured and written in {{ language }} to align with the original text and context.
"""


"""{% if prompt %}\n\n# Input Text:\n{{ prompt }}\n\n{% endif %}{% if structure_fields %}# Response Structure Format
You must follow the response structure:

{% for field in structure_fields %}{{ field.label }}\n{% endfor %}
By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism, structure and alignment with user expectations.\n
{% endif %}"""


"""Gemma open models are built _____ the same _____ and technology as Gemini models. Gemma 2 comes in 2B, 9B _____ 27B and Gemma 1 comes in 2B and 7B sizes."""


"""Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes."""


import json
from datasets import Dataset, DatasetDict, load_dataset

def load_from_json_file(path: str = "/kaggle/input/gemma-template/train-gemma-template.json") -> Dataset:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        return Dataset.from_list(data)


def load_from_huggingface_hub(path: str = "twodev/gemma-template", split: str = "train") -> Dataset:
    dataset = load_dataset("twodev/gemma-template")
    return dataset[split]

try:
    dataset = load_from_json_file("/kaggle/input/gemma-template/train-gemma-template.json")
except FileNotFoundError:
    dataset = load_from_huggingface_hub("twodev/gemma-template")

try:
    dataset = dataset.map(lambda example: {"categories": list(set(example["categories"][:5])) if isinstance(example["categories"], list) else []})  # maximum 5 categories to avoid duplication
    print("Categories:", dataset[1]['categories'])
    print("Categories:", dataset[15]['categories'])
except:
    pass

try:
    dataset = dataset.map(lambda example: {"tags": list(set(example["tags"][:5])) if isinstance(example["tags"], list) else []})  # maximum 5 tags to avoid duplication
    print("Tags:", dataset[1]['tags'])
    print("Tags:", dataset[15]['tags'])
except:
    pass


def convert_to_conversations_dataset(data: Dataset, mapping_field: dict[str, list[str]]) -> Dataset:
    """
    Converts a dataset into a conversational format suitable for fine-tuning language models.

    Notes:
        - The `gemma_template._gen_bullet_list_style` method is used to format `openai` responses as a `number`, `dash` and `asterisk` bullet list when the field value is a list.
    """
    
    template = """<start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"""
    outputs = []
    for item in data:
        messages = item["messages"]
        for field in mapping_field:
            if field in item["origin_data"] and mapping_field[field]:
                value = item["origin_data"][field]
                messages.append(
                    {
                        "role": "user",
                        "content": random.choice(mapping_field[field]),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            gemma_template._generate_bullet_style(value, "asterisk")
                            if isinstance(value, list)
                            else value
                        ),
                    }
                )

        outputs_ = [
            template.format(
                input="\n\n".join([messages[0]['content'], messages[1]['content']]),
                output=messages[2]['content']
            )
        ]

        if len(messages) > 3:
            prompts = []
            for idx in range(3, len(messages), 2):
                try:
                    prompt = template.format(
                        input=messages[idx]["content"],
                        output=messages[idx + 1]["content"],
                    )
                    prompts.append(prompt.strip())
                except IndexError:
                    pass

            random.shuffle(prompts)
            outputs_.extend(prompts)

        outputs_.append("")
        sep = tokenizer.eos_token + "\n"
        outputs.append({"text": sep.join(outputs_) + sep})

    if outputs:
        return Dataset.from_list(outputs)


def process_fn(
    instance: Template,
    data: Dataset, 
    excluded_fields: list[str] = (), 
    output_format = 'text',
    max_hidden_ratio = 0.15, 
    max_hidden_words = 0.05, 
    min_chars_length = 2, 
    max_chars_length = 8,
    max_concurrency: int = 4,
    n_words = 5,
    is_remove_data = True,
    **kwargs
) -> Dataset:
    """
    Processes a dataset for fine-tuning language models, supporting formats like `text`, `alpaca`, and `gpt`.

    Args:
        instance (Template): A template instance for dataset processing.
        data (Dataset): The input dataset to be processed.
        excluded_fields (list[str]): Fields to exclude when generating conversational datasets.
        output_format (str): Format of the processed dataset. Options are 'text', 'alpaca', and 'gpt'.
        max_hidden_ratio (Union[float]):
            Percentage of documents that need to be word masked. Min: 0, Max: 1. Default: 0.
        max_hidden_words (Optional[str]):
            Replace words in the document with '____'. The `max_hidden` parameter must be greater than 0.
            Use `int`: exact number of words to be masked, `float`: percentage of number of words to be masked.
        min_chars_length (int):
            Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
        max_chars_length (int):
            Maximum character of a word, used to create unigrams, bigrams and trigrams. Default is 0.
        max_concurrency (int):
            Maximum number of concurrent threads for processing data. Default is 4.
        n_words (int): Number of words frequently used to create unigrams, bigrams and trigrams.
        is_remove_data (bool): Whether to remove specific fields from the dataset. Defaults to True.
        **kwargs: Additional configuration parameters.

    Returns:
        Dataset or DatasetDict: The processed dataset in the specified format.

    Notes:
        - The `output_format` parameter determines the dataset's structure:
            - `'text'`: Standard format for unsloth fine-tuning.
            - `'alpaca'` or `'openai'`: Formats for frameworks like LLaMA-Factory.
        - Using `output_format='openai'` and `is_remove_data=False` with `excluded_fields` generates conversational datasets.

    """
    ds = instance.load_dataset(
        data, 
        output_format=output_format, 
        excluded_fields=excluded_fields,
        max_hidden_ratio=max_hidden_ratio, 
        max_hidden_words=max_hidden_words, 
        min_chars_length=min_chars_length, 
        max_chars_length=max_chars_length,
        max_concurrency=max_concurrency,
        n_words=n_words,
        is_close_async_loop=False,  # Avoid `RuntimeError` by Notebook
        is_remove_data=is_remove_data,
    )
    if output_format == 'openai' and not is_remove_data:
        mapping_field = {
            field: getattr(instance, field, None)
            for field in excluded_fields
            if getattr(instance, field, None)
        }
        ds = convert_to_conversations_dataset(ds, mapping_field=mapping_field)
    else:
        ds = ds.map(lambda x: {"text": [text + tokenizer.eos_token for text in x["text"]]}, batched = True)  # Append eos token.
    return ds


def print_verify(data, is_masked: bool = True, task_name: str = "TASK"):
    print(task_name + "*" * 45)
    for item in data:
        if item.get("is_masked") == is_masked:
            if is_masked:
                print("HIDDEN TEXT: YES" + "*" * 45)
            else:
                print("HIDDEN TEXT: NO" + "*" * 45)
                
            print("\n")
            print(item['text'])
            print("=" * 60)
            print("*" * 30, " DATA ATTRS ", "*" * 30)
            print("Masked Text:", item['is_masked'])
            print("Language Code:", item['analysis']['language_code'])
            print("Language:", item['analysis']['language'])
            print("Categories:", item['analysis']['topic_value'])
            print("Keywords:", item['analysis']['keyword_value'])
            print("Unigrams:", item['analysis']['unigrams'])
            print("Bigrams:", item['analysis']['bigrams'])
            print("Trigrams:", item['analysis']['trigrams'])
            print("VALID TASK: YES")
            print("*" * 30, " TASK DONE ", "*" * 30)
            print("=" * 60)
            print("\n")
            
            return

    print("VALID TASK: NO")
    print("*" * 30, " TASK DONE ", "*" * 30)
    print("=" * 60)
    print("\n")


total_rows = len(dataset)
percent_idx = int(total_rows * 0.1)

dataset_mapping = {
    "Combined response including title and document": {
        "data": dataset.select(range(0, percent_idx)),
        "excluded_fields": ["description", "main_points", "categories", "tags"]
    },
    "Combined response including title, document and description": {
        "data": dataset.select(range(percent_idx, percent_idx * 2)),
        "excluded_fields": ["main_points", "categories", "tags"]
    },
    "Combined response including title, document and main points": {
        "data": dataset.select(range(percent_idx * 2, percent_idx * 3)),
        "excluded_fields": ["description", "categories", "tags"]
        
    },
    "Combined response including title, document and categories and tags": {
        "data": dataset.select(range(percent_idx * 3, percent_idx * 4)),
        "excluded_fields": ["description", "main_points"]
        
    },
}

print("TOTAL ROWS:", total_rows)
print("\n")
print("*" * 30, " DATASET INFO ", "*" * 30)
print(dataset_mapping)


language_ratio_size = 0.5  # Ratio between English and local language.


input_datasets = []
for task, item in dataset_mapping.items():
    print("Prepare dataset for task:", task)
    split_dataset = item['data'].train_test_split(test_size=language_ratio_size)

    # prepare dataset use instruction and structure English language.
    english_dataset = process_fn(
        gemma_template, 
        split_dataset["train"], 
        excluded_fields=item['excluded_fields']
    )
    input_datasets.append(english_dataset)
                                                       
    # prepare dataset use instruction and structure Vietnamese language.
    vietnamese_dataset = process_fn(
        vietnamese_gemma_template, 
        split_dataset["test"], 
        excluded_fields=item['excluded_fields']
    )
    
    input_datasets.append(vietnamese_dataset)
    
# Test with hidden mask test
print_verify(input_datasets[0] if input_datasets else [], is_masked=True, task_name="ENGLISH VERSION: {}".format(task.upper()))
# Test without hidden mask
print_verify(input_datasets[1] if len(input_datasets) > 1 else [], is_masked=False, task_name="VIETNAMESE VERSION: {}".format(task.upper()))


print(input_datasets)


# empty instruction template
gemma_template.instruction_template = []  
vietnamese_gemma_template.instruction_template = []

no_instruction_dataset = dataset.select(range(percent_idx*4, percent_idx*5))
print(no_instruction_dataset)

split_dataset = no_instruction_dataset.train_test_split(test_size=language_ratio_size)

# prepare dataset use instruction and structure English language.
english_no_instruction_dataset = process_fn(gemma_template, split_dataset["train"])
input_datasets.append(english_no_instruction_dataset)
                                                   
# prepare dataset use instruction and structure Vietnamese language.
vietnamese_no_instruction_dataset = process_fn(vietnamese_gemma_template,  split_dataset["test"])
input_datasets.append(vietnamese_no_instruction_dataset)

# print verify
print_verify(english_no_instruction_dataset, is_masked=False, task_name="ENGLISH NO INSTRUCTION VERSION")
print_verify(vietnamese_no_instruction_dataset, is_masked=False, task_name="VIETNAMESE NO INSTRUCTION VERSION")


from gemma_template.constants import INSTRUCTION_TEMPLATE, VIETNAMESE_INSTRUCTION_TEMPLATE

# Reset the template as instructed.
gemma_template.instruction_template = [INSTRUCTION_TEMPLATE]
vietnamese_gemma_template.instruction_template = [INSTRUCTION_TEMPLATE]

conversations_dataset = dataset.select(range(percent_idx*5, percent_idx*6))
split_dataset = conversations_dataset.train_test_split(test_size=language_ratio_size)
excluded_fields = ["title", "description", "main_points", "categories", "tags"]

# prepare dataset use instruction and structure English language.
english_conversations_dataset = process_fn(
    gemma_template, 
    split_dataset["train"], 
    excluded_fields=excluded_fields,
    output_format="openai",
    is_remove_data=False,
)
input_datasets.append(english_conversations_dataset)

                                                   
# prepare dataset use instruction and structure Vietnamese language.
vietnamese_conversations_dataset = process_fn(
    vietnamese_gemma_template, 
    split_dataset["test"], 
    excluded_fields=excluded_fields,
    output_format="openai",
    is_remove_data=False,
)
input_datasets.append(vietnamese_conversations_dataset)

# print verify
print(english_conversations_dataset['text'][0])


from gemma_template.constants import INSTRUCTION_TEMPLATE, VIETNAMESE_INSTRUCTION_TEMPLATE

# Reset the template as instructed.
gemma_template.instruction_template = [INSTRUCTION_TEMPLATE]
vietnamese_gemma_template.instruction_template = [INSTRUCTION_TEMPLATE]

instruction_dataset = dataset.select(range(percent_idx*6, len(dataset)))
print(instruction_dataset)

split_dataset = instruction_dataset.train_test_split(test_size=language_ratio_size)

# prepare dataset use instruction and structure English language.
english_instruction_dataset = process_fn(gemma_template, split_dataset["train"])
input_datasets.append(english_instruction_dataset)
                                                   
# prepare dataset use instruction and structure Vietnamese language.
vietnamese_instruction_dataset = process_fn(vietnamese_gemma_template,  split_dataset["test"])
input_datasets.append(vietnamese_instruction_dataset)

# print verify
print_verify(english_instruction_dataset, is_masked=True, task_name="ENGLISH INSTRUCTION VERSION")
print_verify(vietnamese_instruction_dataset, is_masked=False, task_name="VIETNAMESE INSTRUCTION VERSION")


from datasets import concatenate_datasets

wramup_dataset, train_dataset = [], []
for input_dataset in input_datasets:
    split_dataset = input_dataset.train_test_split(test_size=0.05)
    wramup_dataset.append(split_dataset['test'])
    train_dataset.append(split_dataset['train'])
    
wramup_dataset = concatenate_datasets(wramup_dataset).shuffle(seed=seed)
train_dataset = concatenate_datasets(train_dataset).shuffle(seed=seed)
train_dataset = concatenate_datasets([wramup_dataset, train_dataset])

# verify dataset
print(train_dataset)


elapsed = stopped_at - datetime.now()
print("Elapsed prepared dataset: %s. Took: %.2f seconds" % (str(elapsed), elapsed.total_seconds()))


import os
import json
import torch
from pathlib import Path
from transformers import TrainerCallback, Trainer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

README = """
# Training Arguments
Project Id: {project_id}
Training Steps: {step}

### Hyperparameters:
{hyperparameters}
"""

class HubCallback(TrainerCallback):
    def __init__(self, trainer: Trainer, project_id, stopped_at, save_every_n_minutes=60):
        super().__init__()
        self.trainer = trainer
        self.project_id = project_id
        self.stopped_at = stopped_at
        self.save_every_n_minutes = save_every_n_minutes
        self.save_after = datetime.now() + timedelta(minutes=save_every_n_minutes)

    def on_step_begin(self, train_args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if datetime.now().timestamp() > self.stopped_at.timestamp():
            self.save_checkpoint(train_args)
            control.should_training_stop = True
            control.should_save = True
        else:
            if datetime.now().timestamp() > self.save_after.timestamp():
                self.save_after = datetime.now() + timedelta(minutes=self.save_every_n_minutes)
                self.save_checkpoint(train_args)

    def save_checkpoint(self, train_args: TrainingArguments, output_dir: str = ".checkpoint/"):
        try:
            print("*" * 30, " SAVE CHECKPOINT ", "*" * 30)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.trainer.save_model(output_dir, _internal_call=True)
            self.trainer.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            torch.save(self.trainer.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            try:
                with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as fp:
                    readme_doc = README.format(
                        project_id=self.project_id,
                        step=self.trainer.state.global_step,
                        hyperparameters=json.dumps(train_args.to_dict(), ensure_ascii=False, indent=4)
                    )
                    fp.write(readme_doc)
            except:
                pass

            print("*" * 30, " PROCESSED PUSH TO HUB ", "*" * 30)
            self.push_to_hub(train_args, output_dir)
            print("*" * 30, " COMPLETED PUSH TO HUB ", "*" * 30)

        except Exception as e:
            print("FAILED TO SAVE CHECKPOINT:", e)

    def push_to_hub(self, train_args: TrainingArguments, output_dir: str = ".checkpoint/"):
        self.trainer.tokenizer.save_pretrained(output_dir)
        self.trainer.model.save_pretrained(output_dir)
        self.trainer.tokenizer.push_to_hub(
            self.project_id,
            private=train_args.hub_private_repo,
            token=train_args.hub_token
        )
        self.trainer.model.push_to_hub(
            self.project_id,
            commit_message="Training steps: {}".format(self.trainer.state.global_step),
            private=train_args.hub_private_repo,
            token=train_args.hub_token
        )


import os
import wandb

run = wandb.init(
    project=project_id,
)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

train_args = TrainingArguments(
    # ---Output settings--
    # Output directory where model predictions and checkpoints will be stored
    output_dir = f".results/{project_id}",
    logging_dir = f".results/{project_id}/logs",
    overwrite_output_dir = True,
    # No eval running
    do_eval = False,
    # Save strategy
    save_strategy = "steps",
    # Save steps
    save_steps = 300,
    # Save total limit
    save_total_limit = 1,
    # Batch size per GPU core for training
    per_device_train_batch_size = 2,
    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 4,
    # Train epochs
    num_train_epochs = 1,
    # Learning rate
    learning_rate = 2e-4,
    # Enable float16 precision
    fp16 = not torch.cuda.is_bf16_supported(),
    # Enable bfloat16 precision. False then fp16 is True
    bf16 = torch.cuda.is_bf16_supported(),
    # Logging: Log every X update step
    logging_steps = 3,
    # Optimizer to use
    optim = "adamw_8bit",
    # Weight decay
    weight_decay = 0.1,
    lr_scheduler_type = "linear",
    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.05,
    hub_private_repo = True,
    remove_unused_columns = True,
    seed = seed,
)


from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=train_args,
)

# Add HubCallback to save model every hour and stop after 11 hours 30 minutes
trainer.add_callback(HubCallback(trainer, project_id, stopped_at))

# Start training
trainer = trainer.train()


try:
    model.save_pretrained(project_id)
    tokenizer.save_pretrained(project_id)
except:
    pass
    
try:
    with open(os.path.join(project_id, "README.md"), "w", encoding="utf-8") as fp:
        readme_doc = README.format(
            project_id=project_id,
            step=trainer.state.global_step,
            hyperparameters=json.dumps(train_args.to_dict(), ensure_ascii=False, indent=4)
        )
        fp.write(readme_doc)
except:
    pass


try:
    model.push_to_hub(project_id)
    tokenizer.push_to_hub(project_id)
except:
    pass


elapsed = stopped_at - datetime.now()
print("Elapsed:", str(elapsed))


import torch

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    project_id,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

device = "cuda" if torch.cuda.is_available() else "cpu"

stopped_at = stopped_at + timedelta(minutes=15)

def is_expired(expired_at):
    if datetime.now().timestamp() > expired_at.timestamp():
        return True
    return False


try:
    test_dataset = load_from_json_file("/kaggle/input/gemma-template/test-gemma-template.json")
except FileNotFoundError:
    test_dataset = load_from_huggingface_hub("twodev/gemma-template", split='test')

total_rows = len(test_dataset)
percent_idx = int(total_rows * 0.1)

dataset_mapping = {
    "Combined response including title and document": {
        "data": test_dataset.select(range(0, percent_idx)),
        "excluded_fields": ["description", "main_points", "categories", "tags"]
    },
    "Combined response including title, document and description": {
        "data": test_dataset.select(range(percent_idx, percent_idx * 2)),
        "excluded_fields": ["main_points", "categories", "tags"]
    },
    "Combined response including title, document and main points": {
        "data": test_dataset.select(range(percent_idx * 2, percent_idx * 3)),
        "excluded_fields": ["description", "categories", "tags"]
        
    },
    "Combined response including title, document and categories and tags": {
        "data": test_dataset.select(range(percent_idx * 3, percent_idx * 4)),
        "excluded_fields": ["description", "main_points"]
        
    },
    "Prompt Structure Format": {
        "data": test_dataset.select(range(percent_idx * 4, len(test_dataset))),
        "excluded_fields": []
        
    },
}

print(dataset_mapping)


GEMMA_PROMPT_TEMPLATE = """<start_of_turn>user
{input}<end_of_turn>
<start_of_turn>model

"""

PROMPT_TEMPLATE = """{% if prompt %}\n\n{{ prompt }}\n\n{% endif %}{% if structure_fields %}# Response Structure Format
You must follow the response structure:

{% for field in structure_fields %}{{ field.label }}\n{% endfor %}
{% endif %}"""

VIETNAMESE_PROMPT_TEMPLATE = """{% if prompt %}\n\n{{ prompt }}\n\n{% endif %}{% if structure_fields %}# Định Dạng Cấu Trúc Phản Hồi
Bạn phải tuân theo cấu trúc phản hồi:

{% for field in structure_fields %}{{ field.label }}\n{% endfor %}
{% endif %}"""

gemma_template.instruction_template = []  
gemma_template.prompt_template = [PROMPT_TEMPLATE]
vietnamese_gemma_template.instruction_template = []
vietnamese_gemma_template.prompt_template = [VIETNAMESE_PROMPT_TEMPLATE]

eval_dataset = []
input_datasets = []
for task, item in dataset_mapping.items():
    if is_expired(stopped_at):
        break
        
    print("Prepare dataset for task:", task)
    split_dataset = item['data'].train_test_split(test_size=language_ratio_size)

    # prepare dataset use instruction and structure English language.
    english_dataset = gemma_template.load_dataset(
        split_dataset["train"], 
        excluded_fields=item['excluded_fields'],
        output_format='alpaca',
    )
    english_dataset = english_dataset.map(lambda x: {"task": ["English" for _ in x["input"]], }, batched=True)
    input_datasets.append(english_dataset)
                                                       
    # prepare dataset use instruction and structure Vietnamese language.
    vietnamese_dataset = vietnamese_gemma_template.load_dataset(
        split_dataset["test"], 
        excluded_fields=item['excluded_fields'],
        output_format='alpaca',
    )
    vietnamese_dataset = vietnamese_dataset.map(lambda x: {"task": ["Vietnamese" for _ in x["input"]], }, batched=True)
    input_datasets.append(vietnamese_dataset)

from datasets import concatenate_datasets

if input_datasets:
    eval_dataset = concatenate_datasets(input_datasets).shuffle(seed=42)
    eval_dataset = eval_dataset.map(lambda x: {"prompt": [GEMMA_PROMPT_TEMPLATE.format(input=input_str) for input_str in x["input"]], }, batched=True)
    print(eval_dataset)
    print(eval_dataset[0]['prompt'])


import json
import evaluate

def clean_response(response: str):
    response = response.split("<start_of_turn>model")[-1].split("<end_of_turn>")
    return response[0].strip()


google_bleu = evaluate.load("google_bleu")
rouge = evaluate.load('rouge')
eval_responses = []

for idx, item in enumerate(eval_dataset):

    # Remove `is_expired` for fully eval, this code is avoid Kaggle limit.
    if is_expired(stopped_at):
        break

    task = str(item['task']).upper()
    input_str = item['prompt']
    output_str = item['output'].strip()
    predictions, references = [output_str], []
    input_ids = tokenizer(input_str, return_tensors="pt").to(device)
    outputs = model.generate(**input_ids, max_new_tokens=1024)

    model_references = []
    rouge_score, google_bleu_score = {}, {}

    try:
        for output in outputs:
            model_response = tokenizer.decode(output)
            model_references.append(model_response)
            references.append(clean_response(model_response))
        
        if not (predictions and references):
            continue

        try:
            rouge_score = rouge.compute(predictions=predictions, references=references)
            rouge_score = {k: float(v) for k, v in rouge_score.items()}
        except:
            pass

        try:
            google_bleu_score = google_bleu.compute(predictions=predictions, references=references)
        except:
            pass
            
    except:
        pass
    
    try:
        item.update({"rouge": rouge_score, "google_bleu": google_bleu_score, "model_references": model_references})
        eval_responses.append(json.loads(json.dumps(item, default=str)))
    except:
        pass


def is_valid_score(r: dict, field: str):
    if isinstance(r, dict):
        if r.get(field):
            return True
        
def write_json(obj, path: str = "dump.json", *, ensure_ascii=False, indent=4):
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(obj, json_file, ensure_ascii=ensure_ascii, indent=4, default = str)


try:
    write_json(eval_responses, project_id + "/example_eval.json")
except:
    pass


total_rows = len(eval_responses)
try:
    rouge_mapping = {
        "rouge1": sum([r['rouge']['rouge1'] for r in eval_responses if is_valid_score(r, "rouge")]) / total_rows,
        "rouge2": sum([r['rouge']['rouge2'] for r in eval_responses if is_valid_score(r, "rouge")]) / total_rows,
        "rougeL": sum([r['rouge']['rougeL'] for r in eval_responses if is_valid_score(r, "rouge")]) / total_rows,
        "rougeLSum": sum([r['rouge']['rougeLsum'] for r in eval_responses if is_valid_score(r, "rouge")]) / total_rows,
    }
    print("AVG ROUGE SCORE:", str(rouge_mapping))
except:
    pass

try:
    google_bleu_mapping = {"google_bleu": sum([r['google_bleu']['google_bleu'] for r in eval_responses if is_valid_score(r, "google_bleu")]) / total_rows}
    print("AVG GOOGLE BLEU:", str(google_bleu_mapping))
    print("*" * 90)
    print("\n")
except:
    pass

try:
    for response in eval_responses[:5]:
        print("=" * 90)
        print("*" * 30, " ORIGIN OUTPUT ", "*" * 30)
        print(response['output'])
        print("*" * 30, " MODEL OUTPUT ", "*" * 30)
        print(clean_response(response['model_references'][0]))
        print("*" * 30, " SCORE ", "*" * 30)
        print("ROUGE SCORE:", response['rouge'])
        print("GOOGLE BLEU SCORE:", response['google_bleu'])
        print("=" * 90)
except:
    pass


!rm -rf .checkpoint wandb .results


# import pandas as pd
# import numpy as np
# import torch
# import json
# import glob
# import logging
# import os
# import argparse
# import time

# from string import Template
# from pathlib import Path
# from tqdm import tqdm
# from typing import Any

# import warnings
# warnings.simplefilter("ignore")

# def predict(args):
#     device = args.device
#     folder = args.folder

#     path = args.project_id.split('/')[-1].strip()
#     filename = f"./logs/{args.project_id}.log"
    
#     ## create directory
#     directory_path = './logs'
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)
        
#     # Configure logging
#     logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
#     logging.info(f'Model name: {args.project_id}')

#     # Create empty lists to store data
#     ids = []
#     questions = []
#     choices_A = []
#     choices_B = []
#     choices_C = []
#     choices_D = []
#     choices_E = []

#     # Read JSONL files
#     data_path = Path(folder)
#     jsonl_files = list(data_path.glob(args.dataset))

#     for file in jsonl_files:
#         with open(file, "r", encoding="utf-8") as f:
#             lines = f.readlines()
#             for line in lines:
#                 data = json.loads(line)
#                 ids.append(data["id"])
#                 questions.append(data["question"])
#                 choices = data["choices"]
#                 try:
#                     choices_A.append(choices[0])
#                 except:
#                     choices_A.append('')
#                 try:
#                     choices_B.append(choices[1])
#                 except:
#                     choices_B.append('')
#                 try:
#                     choices_C.append(choices[2])
#                 except:
#                     choices_C.append('')
#                 try:
#                     choices_D.append(choices[3])
#                 except:
#                     choices_D.append('')
#                 try:
#                     choices_E.append(choices[4])
#                 except:
#                     choices_E.append('')

#     # Create a DataFrame
#     df = pd.DataFrame({
#         "id": ids,
#         "prompt": questions,
#         "A": choices_A,
#         "B": choices_B,
#         "C": choices_C,
#         "D": choices_D,
#         "E": choices_E
#     })
#     logging.info(df.head())

#     preamble = \
#         'Chỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau: '

#     template = Template(args.template)

#     def format_input(df, idx):
#         prompt = df.loc[idx, 'prompt']
#         a = df.loc[idx, 'A']
#         b = df.loc[idx, 'B']
#         c = df.loc[idx, 'C']
#         d = df.loc[idx, 'D']
#         e = df.loc[idx, 'E']

#         input_text = template.substitute(
#             preamble=preamble, prompt=prompt, a=a, b=b, c=c, d=d, e=e)

#         return input_text

#     inputs = args.tokenizer(format_input(df, 0), return_tensors="pt").to(device)
#     outputs = args.model.generate(**inputs, max_new_tokens=1)
#     answer = args.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     logging.info('Contruct a toy eg')
#     logging.info("Generated answer: %s", answer)

#     answers = []

#     start = time.time()
#     for idx in tqdm(df.index):
#         inputs = args.tokenizer(format_input(df, idx), return_tensors="pt").to(device)
#         outputs = args.model.generate(**inputs, max_new_tokens=args.max_new_tokens)
#         answer_decoded = args.tokenizer.batch_decode(outputs, skip_special_tokens=True)

#         last_element = answer_decoded[-1]
#         answer = last_element.split()[-1]
#         if "án:" in answer:
#             answer = "-"

#         answers.append(answer)

#     end = time.time()
#     duration = end - start
#     print('Time taken for running inference: ', duration)

#     df['answer'] = answers
#     logging.info(df.head())

#     return df

# from dataclasses import dataclass

# @dataclass
# class EvalArgs:
#     model: Any
#     tokenizer: Any
#     project_id: str = "gemma-2b"
#     folder: str = "./vmlu_v1.5"  # please visit to https://vmlu.ai/ for download eval dataset data.
#     dataset: str = "test.jsonl"
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     template: str = "$preamble\n\n$prompt\n\n $a\n $b\n $c\n $d\n $e\nĐáp án:"
#     max_new_tokens: int = 1

# eval_args = EvalArgs(model=model, tokenizer=tokenizer, project_id=project_id)
# df = predict(eval_args)
# df[['id','answer']].to_csv("./gemma-2b-vmlu-benchmark.csv", index = False)


print("Took: %.2f seconds." % (time.perf_counter() - start_at))
print("Task End:", str(datetime.now() - stopped_at))

