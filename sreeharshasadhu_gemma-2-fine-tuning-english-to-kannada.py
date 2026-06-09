!nvidia-smi


%%capture

!pip install pip3-autoremove
!pip install datasets transformers numpy pandas polars rich tqdm sacrebleu seaborn wandb tqdm
!pip-autoremove torch torchvision torchaudio -y
!pip install "torch==2.4.0" "xformers==0.0.27.post2" triton torchvision torchaudio
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"


!pip install sacrebleu
!pip install transformers==4.47.1


from kaggle_secrets import UserSecretsClient

# load the huggingface token
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("huggingface_token")
# Authenticate with Hugging Face
from huggingface_hub import login
login(hf_token)


import os
import datasets
import pandas as pd
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, pipeline
from tqdm.autonotebook import tqdm
from unsloth import FastLanguageModel
import warnings

# load presets
os.environ["WANDB_DISABLED"] = "true"
pd.set_option("float_format", "{:f}".format)
pd.set_option("display.max_colwidth", None)
tqdm.pandas()
warnings.filterwarnings("ignore")


from datasets import load_dataset

ds = load_dataset("Cognitive-Lab/Kannada-Instruct-dataset")
ds = ds["train"]
df = ds.to_pandas()
instructions = df[["original_instruction", "translated_instruction"]]
outputs = df[["original_output", "translated_output"]]

# Rename columns to match the expected format

instructions.columns = ["eng", "kan"]
outputs.columns = ["eng", "kan"]

# Concatenate the instructions and outputs

df = pd.concat([instructions, outputs], axis=0)

# Reset the index to avoid duplicate indices

df.reset_index(drop=True, inplace=True)

df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)
dataset = pd.DataFrame({"translation": df.apply(lambda row: {"eng": row["eng"], "kan": row["kan"]}, axis=1)})
print(dataset.shape)

from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset, test_size=0.1, random_state=42)
test, val = train_test_split(test, test_size=0.1, random_state=42)

train.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

print(f"Train: {train.shape}")
print(f"Validation: {val.shape}")
print(f"Test: {test.shape}")


# setting global parameters for LLM
MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True
MODEL_NAME = "unsloth/gemma-2-2b-it-bnb-4bit"

# load the LLM & tokeknizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    token=hf_token,
)


# example
eng_sentence = train.translation[2].get("eng")
kannada_sentence = train.translation[2].get("kan")


def prompt_translation_func(eng_sentence, kannada_sentence):
    """Apply the prompt template and return the prompt with eos_token"""
    prompt_template = [
        {
            "role": "user",
            "content": f"Translate the sentence into kannada and only return the translation.\nInput: {eng_sentence}",
        },
        {
            "role": "assistant",
            "content": f"Output: {kannada_sentence}",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_template,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt.strip("\n") + tokenizer.eos_token


print(prompt_translation_func(eng_sentence, kannada_sentence))


train["prompt"] = train.translation.progress_apply(
    lambda data: prompt_translation_func(data.get("eng"), data.get("kan"))
)
train["prompt_length"] = train.prompt.progress_apply(
    lambda text: len(tokenizer.encode(text))
)

val["prompt"] = val.translation.progress_apply(
    lambda data: prompt_translation_func(data.get("eng"), data.get("kan"))
)
val["prompt_length"] = val.prompt.progress_apply(
    lambda text: len(tokenizer.encode(text))
)

test["prompt"] = test.translation.progress_apply(
    lambda data: prompt_translation_func(data.get("eng"), data.get("kan"))
)
test["prompt_length"] = test.prompt.progress_apply(
    lambda text: len(tokenizer.encode(text))
)


print(f"Train: {train.prompt[10]}")
print(f"Validation: {val.prompt[10]}")
print(f"Test: {test.prompt[10]}")


print(train.prompt_length.describe())
print(train[train.prompt_length <= 2048].shape[0] / train.shape[0])


train_v1 = train[train.prompt_length <= 2048].sample(10_000, random_state=42)
val_v1 = val[val.prompt_length <= 2048]
test_v1 = test[test.prompt_length <= 2048]

print(train_v1.shape)
print(val_v1.shape)
print(test_v1.shape)


train_dataset = datasets.Dataset.from_pandas(train_v1)
eval_dataset = datasets.Dataset.from_pandas(val_v1)
test_dataset = datasets.Dataset.from_pandas(test_v1)


# lora parameters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # "embed_tokens",
        "lm_head",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# training parameters
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="prompt",
    eval_dataset=eval_dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        warmup_steps=5,
        # max_steps = 300,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=500,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="output",
        report_to="none",
    ),
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(
    torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(
    f"Peak reserved memory for training % of max memory = {lora_percentage} %."
)


model.push_to_hub_merged(
    "abhi334/gemma-eng-to-kannada",
    tokenizer,
    save_method="merged_16bit",
    token=hf_token
)


torch.cuda.empty_cache()

pipe = pipeline(
    "text-generation",
    model="abhi334/gemma-eng-to-kannada",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)



%%time

# example
eng_sentence = test.translation[2121].get("eng")
hi_sentence = test.translation[2121].get("kan")


def generate_response(text):
    messages = [
        {
            "role": "user",
            "content": f"Translate the sentence into kannada and only return the translation.\nInput: {text}",
        },
    ]

    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

    return assistant_response


print(eng_sentence)
print(hi_sentence)
print(
    generate_response(eng_sentence)
)  # instead of eng_sentence we can pass our own sentence


test_sample = test.sample(100, random_state=42)
print(test_sample.shape)
test_sample["english"] = test_sample.translation.apply(lambda x: x.get("eng"))
test_sample["kannada"] = test_sample.translation.apply(lambda x: x.get("kan"))
test_sample.reset_index(drop=True, inplace=True)

test_sample["predictions"] = test_sample.english.progress_apply(
    lambda text: generate_response(text)
)
test_sample.reset_index(drop=True, inplace=True)

test_sample.head()


import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

nltk.download("punkt")


def calculate_bleu_score(reference, candidate):
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)

    return sentence_bleu([reference_tokens], candidate_tokens)


bleu_score = calculate_bleu_score(
    test_sample.kannada[10], test_sample.predictions[10]
)
print(f"BLEU score: {bleu_score}")


test_sample["BLEU_Score"] = test_sample[
    ["kannada", "predictions"]
].progress_apply(
    lambda inputs: calculate_bleu_score(inputs[0], inputs[1]), axis=1
)
print(f"Average BLEU Score: {test_sample['BLEU_Score'].mean().round(3)}")

