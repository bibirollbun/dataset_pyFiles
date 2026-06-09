!pip install xformers transformers accelerate datasets peft trl bitsandbytes unsloth  --q
!pip -q uninstall transformers -y
!pip -q install transformers==4.47.1



import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True



from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)




model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
        "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
)




alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


from datasets import load_dataset
dataset = load_dataset("NLPT/Telugu_Dilaog_Dataset", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)




trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer_stats = trainer.train()




model.save_pretrained("/content/drive/MyDrive/gemma/unsloth_finetuned/adapter/telugu_lora_model_gemma-2-27b-it-bnb-4bit")
tokenizer.save_pretrained("/content/drive/MyDrive/gemma/unsloth_finetuned/adapter/telugu_lora_model_tokenizer_gemma-2-27b-it-bnb-4bit")



inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Answer accurately",
            "ఈ క్రింది సంఖ్యను శాస్త్రీయ నోటేషన్ లో మార్చండి: 0.8970",
            "",
        )
    ],
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.batch_decode(outputs))



if True:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/kaggle/input/gemma2-telugu-instruct-9b-qlora-4/transformers/v1/1",
        max_seq_length=1024,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""



# General Knowledge Questions
general_knowledge_questions = [
    "తెలుగు భాష ఎంత పురాతనమైందో చెప్పండి?",  # Explain how ancient the Telugu language is.
    "భారత రాజ్యాంగాన్ని రచించిన ప్రధాన వ్యక్తి ఎవరు?",  # Who is the main author of the Indian Constitution?
    "సూర్యుడి చుట్టూ భూమి తిరగడానికి ఎంత సమయం పడుతుంది?",  # How long does it take for the Earth to revolve around the Sun?
]

for question in general_knowledge_questions:
    print(f"Q: {question}")
    inputs = tokenizer([alpaca_prompt.format("Answer accurately", question, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    print(f"A: {tokenizer.batch_decode(outputs)[0]}")



# Cultural and Literary Context Questions
cultural_questions = [
    "వేమన పద్యాలలో ముఖ్యమైన సందేశం ఏమిటి?",  # What is the main message in Vemana’s poetry?
    "తెలుగులో 2 ప్రముఖ కావ్యాలను వివరించండి.",  # Describe prominent epics in Telugu.
    "తెలుగు నవలల్లో, ప్రముఖ రచయితలు ఎవరు?",  # Who are the prominent authors in Telugu novels?
]

for question in cultural_questions:
    print(f"Q: {question}")
    inputs = tokenizer([alpaca_prompt.format("Answer accurately", question, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    print(f"A: {tokenizer.batch_decode(outputs)[0]}")



# Educational Assistance Questions
education_questions = [
    "పెద్ద సంఖ్యలను శాస్త్రీయ నోటేషన్‌లో ఎలా రాయాలో వివరించండి: 0.567",  # Explain scientific notation.
    "తెలుగు వాక్యనిర్మాణంపై క్లుప్తంగా వివరణ ఇవ్వండి.",  # Briefly explain Telugu sentence structure.
]

for question in education_questions:
    print(f"Q: {question}")
    inputs = tokenizer([alpaca_prompt.format("Answer accurately", question, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    print(f"A: {tokenizer.batch_decode(outputs)[0]}")



# Domain-Specific Knowledge Questions
domain_questions = [
    "గురుత్వాకర్షణ శక్తి ఎలా పనిచేస్తుంది?",  # How does gravity work?
    "కృత్రిమ మేధస్సు గురించి తెలుగులో ఒక క్లుప్త వివరణ ఇవ్వండి.",  # Explain AI in Telugu.
]

for question in domain_questions:
    print(f"Q: {question}")
    inputs = tokenizer([alpaca_prompt.format("Explain in detail", question, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    print(f"A: {tokenizer.batch_decode(outputs)[0]}")


