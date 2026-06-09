!pip install -U kagglehub
!pip install -q 'transformers==4.47.1'
!pip install  accelerate datasets peft trl bitsandbytes --quiet


%%capture
!pip install unsloth
!pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git


from unsloth import FastLanguageModel
import torch
modelName = "/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2/"
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=modelName,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


instruction = {
    'determination':"""Transform the given input text in Darija from its indefinite forms to their corresponding definite forms in Darija.
    Maintain the structure and order of the words as in the input.""",
    'conj_past':"""Conjugate the given verb in Darija into its past tense for all pronouns (nta,nti,howa,hia,7na,ntoma,homa).""",
    'conj_present':"""Conjugate the given verb in Darija into its present tense for all pronouns (ana, nta, nti, howa, hiya, 7na, ntouma, homa).""",
    'imperative':'Generate the imperative conjugations of the given verb in Darija for specified pronouns (nta, nti, ntouma).',
    'pluralization':"""Pluralize the given nouns in Darija.
    Maintain the structure and order of the words as in the input.""",
    'nominalization':"""Perform nominalization on the given verbs in Darija. Convert the verbs into their corresponding noun forms.
    Maintain the structure and order of the words as in the input.""",
    'name_darija_to_arab':"""Convert names from Darija to Arabic.
    Maintain the structure and order of the words as in the input.""",
    'darija_to_arab':"""Convert names from Darija to Arabic.
    Maintain the structure and order of the words as in the input.""",
    'darija_arabic_to_arabic':"""Translate words from Darija Arabic to Arabic.
    Maintain the structure and order of the words as in the input.""",
    'darija_arabic_to_darija':"""Translate words from Darija Arabic to Darija.
    The input consists of words in Darija Arabic, separated by /n/n for each word.
    Provide multiple possible ways a word can be written in Darija, if applicable.
    Maintain the structure and order of the words as in the input.""" ,
    'alpaca':"""Perform question answering and provide the output in Arabic""",
    'textgen':"""Complete the text by generating a continuation for the given input."""
}


import kagglehub
# modified darija-eng-arabic-linguistic_dataset
path = kagglehub.dataset_download("aminemontasir/moroccan-arabic-darija-task-dataset")


from datasets import load_dataset
ds = load_dataset('FreedomIntelligence/alpaca-gpt4-arabic')
ds


def preprocess(batch):
  batch['instruction'] = instruction['alpaca']
  batch['input'] = batch['conversations'][0]['value']
  batch['output'] = batch['conversations'][1]['value']
  return batch
ds1 = ds['train'].map(preprocess,remove_columns=['conversations','id'])
ds1 = ds1.shuffle(47).select(range(5000))
ds1


ds = load_dataset("csv", data_files="/kaggle/input/moroccan-arabic-darija-task-dataset/darija_tasks.csv")
ds


def preprocess(batch):
  batch['instruction'] = instruction[batch['types']]
  batch['input'] = batch['inputs']
  batch['output'] = batch['targets']
  return batch
ds2 = ds['train'].map(preprocess,remove_columns=['inputs', 'targets', 'types'])
ds2


ds = load_dataset('AbderrahmanSkiredj1/moroccan_darija_wikipedia_dataset')
ds


def transform_text(batch):
    inputs = []
    outputs = []
    inst = []
    for text in batch['text']:
        words = text.split()
        inputs.append(" ".join(words[:10]))  
        outputs.append(" ".join(words[10:])) 
        inst.append(instruction['textgen'])
    return {'instruction':inst,'input': inputs, 'output': outputs}
ds3 = ds['train'].map(transform_text, batched=True,remove_columns=['text'])
ds3


print(ds1,ds2,ds3)


from datasets import concatenate_datasets
dataset = concatenate_datasets([ds1,ds2,ds3])
dataset = dataset.shuffle(seed=42)
dataset


gemma_prompt = """<start_of_turn>user Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}

### Input:
{}

<end_of_turn>
<start_of_turn>model 
### Response:
{}<end_of_turn>"""
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = gemma_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
dataset = dataset.map(formatting_prompts_func, batched = True,)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30, # for good result more than 4000
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


trainer_stats = trainer.train()


# Local saving
model.save_pretrained("saved_model")  
tokenizer.save_pretrained("saved_model")


import kagglehub

# Download latest version
path = kagglehub.model_download("aminemontasir/gemma_2-9b_darija/transformers/default")

print("Path to model files:", path)



!pip install transformers peft


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2/")
model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2/",
    device_map="auto",
    torch_dtype=torch.bfloat16
)


from peft import PeftModel

lora_weights_path = "/kaggle/input/gemma_2-9b_darija/transformers/default/1"  # Replace with the path to your LoRA weights
model = PeftModel.from_pretrained(model, lora_weights_path)



gemma_prompt = """<start_of_turn>user Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}

### Input:
{}

<end_of_turn>
<start_of_turn>model 
### Response:
"""
input_text = gemma_prompt.format(instruction['alpaca'],"شرح ليا كيفاش نتجاوز سرعة الضو")



input_ids = tokenizer(input_text, return_tensors="pt").to('cuda')
outputs = model.generate(**input_ids,max_new_tokens=364, use_cache=True)
print(tokenizer.decode(outputs[0]))

