!pip install -U bitsandbytes
!pip install datasets
!pip install trl
!pip install kaggle


import torch
import warnings

from trl import SFTTrainer
from peft import PeftModel
from peft import LoraConfig
from tqdm.notebook import tqdm
from datasets import load_dataset

from transformers import BitsAndBytesConfig, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq


!kaggle models instances versions download google/gemma-2/transformers/gemma-2-2b/2


!tar -xvzf 'gemma-2.tar.gz' 'gemma-2-2b'


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer = AutoTokenizer.from_pretrained("gemma-2-2b")


base_model = AutoModelForCausalLM.from_pretrained("gemma-2-2b",quantization_config=bnb_config,
                                                                         device_map='auto')


question = "<start_of_turn>user Tell me about elephants, but tell me in English please. <end of turn>\n<start_of_turn>model "

inputs = tokenizer(question, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=246,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1,
                              use_cache=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])


question = "<start_of_turn>user recycling ke vishay me ek nara sujhav kare<end of turn>\n<start_of_turn>model "

inputs = tokenizer(question, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=246,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1,
                              use_cache=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])


question = "<start_of_turn>user रीसाइक्लिंग के विषय में एक नारा सुझाए<end of turn>\n<start_of_turn>model "

inputs = tokenizer(question, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=246,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1,
                              use_cache=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])


!export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


!export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


alpaca_dataset_train = load_dataset("FreedomIntelligence/alpaca-gpt4-hindi",
                              split = "train")
alpaca_dataset_train, alpaca_dataset_train[3]


alpaca_dataset_train.info


alpaca_prompt="""<start_of_turn>user\n.\n\"{}\"<end_of_turn>\n<start_of_turn>model\n{}<end_of_turn>"""
print(alpaca_prompt)


eos_token = tokenizer.eos_token
tokenizer.padding_side = "right"
eos_token


def formatting_func(conversations):
    texts = []
    conversations = conversations["conversations"]
    for convo in conversations:
        # EOS_TOKEN is important
        text = alpaca_prompt.format(convo[0]["value"], convo[1]["value"]) + eos_token
        texts.append(text)
    return { "text" : texts, }


alpaca_dataset = alpaca_dataset_train.map(formatting_func, batched = True,)


alpaca_dataset


print(alpaca_dataset["text"][0])


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

print("Tokenizing dataset...")
dataset = alpaca_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Dataset tokenized:", dataset[0])


lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["embed_tokens", "lm_head"],
    task_type="CAUSAL_LM",
    use_rslora=True
)

train_args = TrainingArguments(
    per_device_train_batch_size=2,  # Each GPU processes 2 examples per step.
    gradient_accumulation_steps=1,  # Gradients are accumulated over 1 steps before updating weights.
    # warmup_steps=30,  # Learning rate warms up (gradually increases) for the first 30 steps.
    #max_steps=10,  # Total number of optimization steps for training.
    warmup_ratio=0.1, # Learning rate warms up (gradually increases) for the first 10 percent of epoch.
    num_train_epochs=1,  # Total number of epochs for training.
    gradient_checkpointing=True,  # Saves memory by recomputing activations during backpropagation.
    learning_rate=5e-5,  # Base learning rate for the optimizer.
    fp16=not torch.cuda.is_bf16_supported(),  # FP16 precision if BF16 is not available.
    bf16=torch.cuda.is_bf16_supported(),  # Enables bfloat16 precision if available.
    save_steps=100,  # Saves checkpoint every 100 steps.
    torch_empty_cache_steps = 100,  # Empties the cache at every 100 steps.
    optim="adamw_8bit",  # Uses AdamW optimizer with 8-bit precision for optimizer states to save memory.
    weight_decay=0.01,  # Regularization to prevent overfitting by penalizing large weights.
    lr_scheduler_type="linear",  # Linearly decays learning rate after the warmup period.
    output_dir="gemma-2-2b-{hi)-alpaca-chk",  # Directory where model checkpoints and logs will be saved.
    report_to="none",  # Disables logging to external tools like TensorBoard or WandB.
    save_total_limit=2, # Will save only 2 checkpoints at max, reducing the disk usage.
    run_name='pretrain_gemma2' # Defining a name for our runtime.
)


# If you did not tokenized the dataset, you must use Data Collator.
# It uses tokenizer, tokenize your training data and returns them as tensors.
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=base_model,
    padding="longest",
    return_tensors="pt"
)

trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    args=train_args,
    peft_config=lora_config,
    train_dataset=dataset,
    data_collator=data_collator,
)


# To begin training use
trainer.train()


# To resume training from last checkpoint use
trainer.train(resume_from_checkpoint=True)


# Once training is done save the model and the tokenizer
trainer.save_model('gemma-2-2b-(hi)-24985steps-1epoch-alphacha')
trainer.tokenzier.save_pretrained('gemma-2-2b-(hi)-24985steps-1epoch-alphacha')


model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('gemma-2-2b', device_map="cpu"), 'gemma-2-2b-24985steps-1epoch-alphacha')


question = "कुछ एक रीसाइक्लिंग अभियान के लिए एक नारा सुझाव दें।"



inputs = tokenizer(question, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=128,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


merged_model.save_pretrained("gemma-2-2b-tmp")


torch.save(merged_model.state_dict(), "merged_model_state_dict.pth")


model = AutoModelForCausalLM.from_pretrained("gemma-2-2b-tmp",device_map='cpu')


model.load_state_dict(torch.load("merged_model_state_dict.pth", weights_only=True))


tokenizer = AutoTokenizer.from_pretrained("gemma-2-2b-{hi)-24985steps-1epoch-alphacha")


model.save_pretrained("gemma-2-2b-base+alpaca")


tokenizer.save_pretrained("gemma-2-2b-base+alpaca")


question = "कुछ एक रीसाइक्लिंग अभियान के लिए एक नारा सुझाव दें।"

inputs = tokenizer(question, return_tensors="pt").to('cpu')

generated_ids = model.generate(**inputs,
                              max_new_tokens=128,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1,
                              use_cache=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer = AutoTokenizer.from_pretrained("gemma-2-2b-base+alpaca")


base_model = AutoModelForCausalLM.from_pretrained("gemma-2-2b-base+alpaca",quantization_config=bnb_config,
                                                                         device_map='auto')


cognitive_hi_inst_train = load_dataset("CognitiveLab/Hindi-Instruct", split='train')
cognitive_hi_inst_test = load_dataset("CognitiveLab/Hindi-Instruct", split='test')


cognitive_hi_inst_train, print(cognitive_hi_inst_train[0]['text']), print(cognitive_hi_inst_train[0]['input_ids'])


def tokenize_function(examples):
    tokenizer.padding_side = "right"
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

print("Tokenizing dataset...")
train_dataset = cognitive_hi_inst_train.map(tokenize_function, batched=True, remove_columns=["text"])
test_dataset = cognitive_hi_inst_test.map(tokenize_function, batched=True, remove_columns=["text"])
print("Dataset tokenized:", train_dataset[0])


lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj","gate_proj", "up_proj", "down_proj"
                    ],
    #modules_to_save=["embed_tokens", "lm_head"],
    task_type="CAUSAL_LM",
    use_rslora=True
)

train_args = TrainingArguments(
    per_device_train_batch_size=3,  # Each GPU processes 4 examples per step.
    gradient_accumulation_steps=1,  # Gradients are accumulated over 4 steps before updating weights.
    # warmup_steps=30,  # Learning rate warms up (gradually increases) for the first 30 steps.
    #max_steps=10,  # Total number of optimization steps for training.
    warmup_ratio=0.1, # Learning rate warms up (gradually increases) for the first 10 percent of epoch.
    num_train_epochs=1,  # Total number of epochs for training.
    gradient_checkpointing=True,  # Saves memory by recomputing activations during backpropagation.
    learning_rate=5e-5,  # Base learning rate for the optimizer.
    fp16=not torch.cuda.is_bf16_supported(),  # FP16 precision if BF16 is not available.
    bf16=torch.cuda.is_bf16_supported(),  # Enables bfloat16 precision if available.
    save_steps=100,  # Saves checkpoint every 100 steps.
    torch_empty_cache_steps=10,  # Empties the cache at every 10 steps.
    optim="adamw_8bit",  # Uses AdamW optimizer with 8-bit precision for optimizer states to save memory.
    weight_decay=0.01,  # Regularization to prevent overfitting by penalizing large weights.
    lr_scheduler_type="linear",  # Linearly decays learning rate after the warmup period.
    output_dir="gemma-2-2b-cog-lab-chk",  # Directory where model checkpoints and logs will be saved.
    report_to="none",  # Disables logging to external tools like TensorBoard or WandB.
    save_total_limit=2, # Will save only 2 checkpoints at max, reducing the disk usage.
    run_name='pretrain_gemma2' # Defining a name for our runtime.
)


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=base_model,
    padding="longest",
    return_tensors="pt"
)

trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    args=train_args,
    peft_config=lora_config,
    train_dataset=train_dataset,
    data_collator=data_collator,
)


trainer.train()


trainer.save_model("gemma-2-2b-30443steps-1epoch-cog-lab")


trainer.tokenizer.save_pretrained('gemma-2-2b-30443steps-1epoch-cog-lab')


merged_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("gemma-2-2b-base+alpaca",device_map='cpu'), 'gemma-2-2b-30443steps-1epoch-cog-lab').merge_and_unload()


question = "<start_of_turn>user क्या आप मुझे रीसाइक्लिंग के लिए एक नारा समझा सकते हैं? <end of turn>\n<start_of_turn>model "


inputs = tokenizer(question, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=246,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1,
                              use_cache=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])


merged_model.save_pretrained("gemma-2-2b-tmp")


torch.save(merged_model.state_dict(), "merged_model_state_dict.pth")


model = AutoModelForCausalLM.from_pretrained("gemma-2-2b-tmp",device_map='cpu')


model.load_state_dict(torch.load("merged_model_state_dict.pth", weights_only=True))


tokenizer = AutoTokenizer.from_pretrained("gemma-2-2b-30443steps-1epoch-cog-lab")


model.save_pretrained("gemma-2-2b-base+alpaca+cog-lab")


tokenizer.save_pretrained("gemma-2-2b-base+alpaca+cog-lab")


def format_prompt_with_tokens(batch):
    formatted_prompts = []
    for conversations in batch['messages']:
        formatted_prompt = []
        for i in range(0, len(conversations), 2):  # Process user-model pairs
            user_message = conversations[i]
            model_message = conversations[i + 1] if i + 1 < len(conversations) else None

            if user_message['role'] == "user" and model_message and model_message['role'] == "assistant":
                formatted_prompt.append(
                    f"<bos><start_of_turn>{user_message['role']} {user_message['content']} <end_of_turn>\n"
                    f"<start_of_turn>model {model_message['content']} <end_of_turn><eos>"
                )

        # Join the formatted prompt for this conversation
        formatted_prompts.append("\n".join(formatted_prompt))

    # Return the formatted text as a new field in the dataset
    return {"text": formatted_prompts}


# Shuffle the dataset and take 1000 examples
random_subset = cognitive_hi_inst_train.take(3000)

# Apply the formatting function to this subset
cognitive_hi_inst_dataset = random_subset.map(format_prompt_with_tokens, batched=True)


print(cognitive_hi_inst_dataset[1]['text'])


def tokenize_function(examples):
    tokenizer.padding_side = "right"
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

print("Tokenizing dataset...")
train_dataset = cognitive_hi_inst_dataset.map(tokenize_function, batched=True)
print("Dataset tokenized:", train_dataset[0])


lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj","gate_proj", "up_proj", "down_proj",],
    lora_dropout=0.05,
    #modules_to_save=["embed_tokens", "lm_head"],
    task_type="CAUSAL_LM",
    use_rslora=True
)

train_args = TrainingArguments(
    per_device_train_batch_size=3,  # Each GPU processes 4 examples per step.
    gradient_accumulation_steps=1,  # Gradients are accumulated over 4 steps before updating weights.
    warmup_steps=30,  # Learning rate warms up (gradually increases) for the first 30 steps.
    #max_steps=1000,  # Total number of optimization steps for training.
    warmup_ratio=0.1, # Learning rate warms up (gradually increases) for the first 10 percent of epoch.
    num_train_epochs=1,  # Total number of epochs for training.
    gradient_checkpointing=True,  # Saves memory by recomputing activations during backpropagation.
    learning_rate=5e-6,  # Base learning rate for the optimizer.
    fp16=not torch.cuda.is_bf16_supported(),  # FP16 precision if BF16 is not available.
    bf16=torch.cuda.is_bf16_supported(),  # Enables bfloat16 precision if available.
    save_steps=100,  # Saves checkpoint every 100 steps.
    torch_empty_cache_steps=10,  # Empties the cache at every 10 steps.
    logging_steps=100,  # Logs metrics every 100 steps.
    optim="adamw_8bit",  # Uses AdamW optimizer with 8-bit precision for optimizer states to save memory.
    weight_decay=0.01,  # Regularization to prevent overfitting by penalizing large weights.
    lr_scheduler_type="linear",  # Linearly decays learning rate after the warmup period.
    output_dir="gemma-2-2b-(hi)-cog-lab-chk-fnt",  # Directory where model checkpoints and logs will be saved.
    report_to="none",  # Disables logging to external tools like TensorBoard or WandB.
    save_total_limit=2, # Will save only 2 checkpoints at max, reducing the disk usage.
    run_name='pretrain_gemma2' # Defining a name for our runtime.
)


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=base_model,
    padding="longest",
    return_tensors="pt"
)

trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    args=train_args,
    peft_config=lora_config,
    train_dataset=train_dataset,
    data_collator=data_collator,
)


trainer.train()


trainer.save_model("gemma-2-2b-1000steps-0.03epoch-cog-lab-fnt")


trainer.tokenizer.save_pretrained("gemma-2-2b-1000steps-0.03epoch-cog-lab-fnt")


merged_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("gemma-2-2b-(hi)-base-alpc-cb",device_map='auto'), 'gemma-2-2b-it(hi)-1000steps-0.03epoch-cog-lab-fnt').merge_and_unload()


question = "<start_of_turn>user Tell me about elephants, but tell me in English please. <end of turn>\n<start_of_turn>model "


inputs = tokenizer(question, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=246,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1,
                              use_cache=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])


question = "<start_of_turn>user What's your name? <end_of_turn>\n<start_of_turn>"


inputs = tokenizer(question, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=246,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1,
                              use_cache=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])


question = "<start_of_turn>user सुबेह सुबेह उठने वाली चिड़िया कौनसी है? <end of turn>\n<start_of_turn>model सुबह-सुबह बहुत सी चिड़िया उठती है, आपको किसके बारे में जानना है? <end_of_turn>\n<start_of_turn>user आपको कौनसी चिड़ियाँ के बारे में पता है? <end of turn>\n<start_of_turn>model "


inputs = tokenizer(question, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=246,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1,
                              use_cache=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])


question = "The following is a conversation between a user and model. The assistant responds in Hindi and provides accurate, concise answers.\nExample 1:\n<start_of_turn>user भारत की राजधानी क्या है? <end_of_turn>\n<start_of_turn>model भारत की राजधानी नई दिल्ली है। <end_of_turn>\nExample 2:\n<start_of_turn>user पिरामिड कहां पाए जाते हैं? <end_of_turn>\n<start_of_turn>model पिरामिड मुख्य रूप से मिस्र में पाए जाते हैं, लेकिन सूडान, मेसोअमेरिका और इटली जैसे अन्य स्थानों पर भी हैं। <end_of_turn>\nExample 3:<start_of_turn>user मुझे चाय और कॉफी के फायदे बताओ। <end_of_turn>\n<start_of_turn>model चाय एंटीऑक्सिडेंट्स से भरपूर होती है और तनाव कम करती है। वहीं, कॉफी सतर्कता और ऊर्जा को बढ़ाती है। <end_of_turn>\nNow continue the conversation:\n<start_of_turn>user भारत के पड़ोसी देशों के नाम क्या हैं? <end_of_turn>\n<start_of_turn>model "


inputs = tokenizer(question, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=246,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1,
                              use_cache=False)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])


wiki_1 = load_dataset("Cohere/wikipedia-22-12-hi-embeddings", split = "train",)
wiki_1[0]['title'], wiki_1[0]['text']


wiki_2 = load_dataset("wikimedia/wikipedia", "20231101.hi", split = "train",)
wiki_2[0]['title'], wiki_2[0]['text']


wiki_3 = load_dataset("sgzsh269/wikipedia-hindi-hinglish", split = "train",)
wiki_3[0]['hindi_title'], wiki_3[0]['hindi_text'], wiki_3[0]['hinglish_title'], wiki_3[0]['hinglish_text']


wiki_3


def format_func_d1(example):
    prompts = []
    titles = example["title"]
    texts = example["text"]

    # Loop over each example in the batch
    for title, text in zip(titles, texts):
        prompt = f"<start_of_turn>user: {title} <end_of_turn>\n<start_of_turn>model: {text}<end_of_turn><eos>"
        prompts.append(prompt)

    # Return as a batch
    return {"prompt": prompts}

wiki_1_train = wiki_1.select_columns(["title","text"]).shuffle().take(5000).map(format_func_d1, batched=True).select_columns(["prompt"])
wiki_1_train[0], wiki_1_train[0]['prompt']


def format_func_d2(example):
    prompts = []
    titles = example["title"]
    texts = example["text"]

    # Loop over each example in the batch
    for title, text in zip(titles, texts):
        prompt = f"<start_of_turn>user: {title} <end_of_turn>\n<start_of_turn>model: {text}<end_of_turn><eos>"
        prompts.append(prompt)

    # Return as a batch
    return {"prompt": prompts}

wiki_2_train = wiki_2.select_columns(["title","text"]).shuffle().take(5000).map(format_func_d2, batched=True).select_columns(["prompt"])
wiki_2_train[0]['prompt']


def format_func_d3(example):
    prompts = []
    try: titles = example["hindi_title"]
    except: titles = example["hinglish_title"]
    try: texts = example["hindi_text"]
    except: texts = example["hinglish_text"]

    # Loop over each example in the batch
    for title, text in zip(titles, texts):
        prompt = f"<start_of_turn>user: {title} <end_of_turn>\n<start_of_turn>model: {text}<end_of_turn><eos>"
        prompts.append(prompt)

    # Return as a batch
    return {"prompt": prompts}

wiki_3_train_hi = wiki_3.select_columns(["hindi_title","hindi_text"]).map(format_func_d3, batched=True).select_columns(["prompt"])
wiki_3_train_he = wiki_3.select_columns(["hinglish_title","hinglish_text"]).map(format_func_d3, batched=True).select_columns(["prompt"])
wiki_3_train = concatenate_datasets([wiki_3_train_hi, wiki_3_train_he])
wiki_3_train_hi[0]['prompt'], wiki_3_train_he[0]['prompt']


wiki_dataset_train = concatenate_datasets([wiki_1_train, wiki_2_train, wiki_3_train])
wiki_dataset_train


alpaca_dataset_train = load_dataset("guneetsk99/Hindi_Alpaca_For_Gemma_67K",
                              split = "train")
alpaca_dataset_train, alpaca_dataset_train[3]


alpaca_dataset_train


gen_prompt="""<start_of_turn>user: {} {}<end_of_turn>\n<start_of_turn>model: {}<end_of_turn>"""
print(alpaca_prompt)


def formatting_func(examples):
    prompts = []
    instruction = examples["instruction"]
    input = examples["input"]
    output = examples['output']
    # Loop over each example in the batch
    for instruction, input, output in zip(instruction, input, output):
        input = input if input else ''
        prompt = gen_prompt.format(instruction, input, output) + '<eos>'
        prompts.append(prompt)
    return { "prompt" : prompts, }


alpaca_train = alpaca_dataset_train.shuffle().take(2000).map(formatting_func, batched = True,).select_columns(["prompt"])


alpaca_train,alpaca_train[0]


print(alpaca_train["prompt"][0])


databrick_dolly = load_dataset("aaditya/databricks-dolly-15k-Hinglish-Codemix", split = "train")


databrick_dolly[0]


def formatting_func(examples):
    prompts = []
    instruction = examples["codemix_instruction"]
    input = examples["codemix_input"]
    output = examples['codemix_output']
    # Loop over each example in the batch
    for instruction, input, output in zip(instruction, input, output):
        instruction = instruction if instruction else ''
        if instruction:
          input = f'{input}' if input else ''
        else:
          input = input if input else ''
        prompt = gen_prompt.format(instruction, input, output) + '<eos>'
        prompts.append(prompt)
    return { "prompt" : prompts, }


databrick_train = databrick_dolly.shuffle().take(2000).map(formatting_func, batched = True,).select_columns(["prompt"])


databrick_train, databrick_train[0]


from huggingface_hub import login

login()


import os

os.environ["HF_HOME"] = "your_hf_token"


math_quest = load_dataset("dnyanesh/HindiMathQuest", split = "train")
math_quest[0]


def formatting_func(examples):
    prompts = []
    instruction = examples["instruction"]
    input = examples["input"]
    output = examples['output']
    # Loop over each example in the batch
    for instruction, input, output in zip(instruction, input, output):
        input = input if input else ''
        prompt = gen_prompt.format(instruction, input, output) + '<eos>'
        prompts.append(prompt)
    return { "prompt" : prompts, }


mathquest_train = math_quest.shuffle().take(2000).map(formatting_func, batched = True,).select_columns(["prompt"])


mathquest_train, mathquest_train[0]


train_dataset = concatenate_datasets([wiki_dataset_train, alpaca_train, databrick_train, mathquest_train]).shuffle()
train_dataset, train_dataset[0]


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["prompt"],
        padding="longest",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
print("Dataset tokenized:", train_dataset[0])


train_dataset


lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"
                    ],
    modules_to_save=["embed_tokens", "lm_head"],
    task_type="CAUSAL_LM",
    use_rslora=True
)

train_args = TrainingArguments(
    per_device_train_batch_size=2,  # Each GPU processes 2 examples per step.
    gradient_accumulation_steps=2,  # Gradients are accumulated over 2 steps before updating weights.
    # warmup_steps=30,  # Learning rate warms up (gradually increases) for the first 30 steps.
    #max_steps=10,  # Total number of optimization steps for training.
    warmup_ratio=0.1, # Learning rate warms up (gradually increases) for the first 10 percent of epoch.
    num_train_epochs=1,  # Total number of training steps for training.
    gradient_checkpointing=True,  # Saves memory by recomputing activations during backpropagation.
    learning_rate=5e-5,  # Base learning rate for the optimizer.
    fp16=not torch.cuda.is_bf16_supported(),  # FP16 precision if BF16 is not available.
    bf16=torch.cuda.is_bf16_supported(),  # Enables bfloat16 precision if available.
    save_steps=100,  # Saves checkpoint every 100 steps.
    torch_empty_cache_steps = 10,  # Empties the cache at every 10 steps.
    logging_steps=100,  # Logs metrics every 10 steps.
    optim="adamw_8bit",  # Uses AdamW optimizer with 8-bit precision for optimizer states to save memory.
    weight_decay=0.01,  # Regularization to prevent overfitting by penalizing large weights.
    lr_scheduler_type="linear",  # Linearly decays learning rate after the warmup period.
    output_dir="gemma-2-2b-(hi)-wiki+alpaca+databrick+mathquest_chk",  # Directory where model checkpoints and logs will be saved.
    report_to="none",  # Disables logging to external tools like TensorBoard or WandB.
    save_total_limit=2, # Will save only 2 checkpoints at max, reducing the disk usage.
    run_name='pretrain_gemma2' # Defining a name for our runtime.
)


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=base_model,
    padding="longest",
    return_tensors="pt"
)

trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    args=train_args,
    peft_config=lora_config,
    train_dataset=train_dataset,
    data_collator=data_collator,
)


trainer.train()


trainer.save_model('gemma-2-2b-{hi)-16994batch-1epoch-wiki+alpaca+databrick+mathquest')
trainer.tokenizer.save_pretrained('gemma-2-2b-{hi)-16994batch-1epoch-wiki+alpaca+databrick+mathquest')


tokenizer = AutoTokenizer.from_pretrained('gemma-2-2b-{hi)-16994batch-1epoch-wiki+alpaca+databrick+mathquest')
merged_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('gemma-2-2b', device_map='auto'), 'gemma-2-2b-{hi)-16994batch-1epoch-wiki+alpaca+databrick+mathquest').merge_and_unload()


merged_model.save_pretrained("gemma-2-2b-tmp")


torch.save(merged_model.state_dict(), "merged_model_state_dict.pth")


model = AutoModelForCausalLM.from_pretrained("gemma-2-2b-tmp",device_map='cpu')


model.load_state_dict(torch.load("merged_model_state_dict.pth", weights_only=True))


tokenizer = AutoTokenizer.from_pretrained("gemma-2-2b-(hi)-16994batch-1epoch-wiki+alpaca+databrick+mathquest")


model.save_pretrained("gemma-2-2b-(hi)-base+wiki+alpaca+databrick+mathquest")


tokenizer.save_pretrained("gemma-2-2b-(hi)-base+wiki+alpaca+databrick+mathquest")


system_prompt = "You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish and English. You respond to users in a clear, and concise manner in the language of the user query. \nआप जेम्मा2 हैं, एक मददगार, संवादी एआई सहायक। आप हिंदी, बोलचाल की हिंग्लिश और अंग्रेजी में विशेषज्ञ हैं। आप उपयोगकर्ताओं को उपयोगकर्ता की क्वेरी की भाषा में स्पष्ट और संक्षिप्त तरीके से जवाब देते हैं। \naap jemmaa2 hain, ek madadagaar, sanvaadee eaee sahaayak. aap hindee, bolachaal kee hinglish aur angrejee mein visheshagy hain. aap upayogakartaon ko upayogakarta kee kveree kee bhaasha mein spasht aur sankshipt tareeke se javaab dete hain."

# Prepare the input
user_input = "<start_of_turn>user: Why is diwali celebrated<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=500,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = "You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish and English. You respond to users in a clear, and concise manner in the language of the user query"

# Prepare the input
user_input = "<start_of_turn>user: दिवाली का त्यौहार क्यों मनाया जाता है, संचेप में बतायें?<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = system_prompt + '\n' + user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=500,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = "You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish and English. You respond to users in a clear, and concise manner in the language of the user query"

# Prepare the input
user_input = "<start_of_turn>user: यह रहा एक गणित का प्रश्न हिंदी में: \n**प्रश्न:** \nएक रेलगाड़ी की लंबाई 120 मीटर है। वह 72 किमी/घंटा की गति से चल रही है। रेलगाड़ी को एक 240 मीटर लंबे पुल को पार करने में कितना समय लगेगा? \n(उत्तर सेकंड में दें।)?<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = system_prompt + '\n' + user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=1000,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = "You are Gemma2, a helpful, AI assistant. You are an expert in Hindi, colloquial Hinglish and English communication. You respond to users in a clear, and concise manner in the language of the user query"

# Prepare the input
user_input = "<start_of_turn>user: प्रश्न: एक रेलगाड़ी की लंबाई 120 मीटर है। वह 72 किमी/घंटा की गति से चल रही है। रेलगाड़ी को एक 240 मीटर लंबे पुल को पार करने में कितना समय लगेगा? (उत्तर सेकंड में दें।) निर्देश: इस प्रश्न को पहले ध्यान से पढ़ें और पूरी तरह से समझें। इसके बाद, इसे चरणबद्ध तरीके से हल करें। प्रत्येक चरण में अपने निष्कर्ष स्पष्ट रूप से प्रस्तुत करें और अंत में उत्तर दें।<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = system_prompt + '\n' + user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=2000,
                              do_sample=True,
                              repetition_penalty=1)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = "You are Gemma2, a helpful, AI assistant. You are an expert in Hindi, colloquial Hinglish and English communication. You respond to users in a clear, and concise manner in the language of the user query"

# Prepare the input
user_input = "<start_of_turn>user: Kya aapko pata hay ki ek saal me kitne din hote hain?<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input =  user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=2000,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = "You are Gemma2, a helpful, AI assistant. You are an expert in Hindi, colloquial Hinglish and English communication. You respond to users in a clear, and concise manner in the language of the user query"

# Prepare the input
user_input = "<start_of_turn>user: एक यादृच्छिक कविता उत्पन्न करें<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input =  user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=2000,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = "You are Gemma2, a helpful, AI assistant. You are an expert in Hindi, colloquial Hinglish and English communication. You respond to users in a clear, and concise manner in the language of the user query"

# Prepare the input
user_input = "<start_of_turn>user: महात्मा गांधी के बारे में 100 शब्दो में निबंद लिखें।<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input =  user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=2000,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = "You are Gemma2, a helpful, AI assistant. You are an expert in Hindi, colloquial Hinglish and English communication. You respond to users in a clear, and concise manner in the language of the user query"

# Prepare the input
user_input = "<start_of_turn>user: Python प्रोग्रामिंग लैंग्वेज में एक 'हैलो वर्ल्ड' का कोड लिखा है।<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input =  user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=2000,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = "You are Gemma2, a helpful, AI assistant. You are an expert in Hindi, colloquial Hinglish and English communication. You respond to users in a clear, and concise manner in the language of the user query"

# Prepare the input
user_input = "<start_of_turn>user: Translate 'And when i decided to play outside, it started raining' to hindi<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = system_prompt + '\n' + user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(merged_model.device)

generated_ids = merged_model.generate(**inputs,
                              max_new_tokens=2000,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


!kaggle models instances versions download google/gemma-2/transformers/gemma-2-2b-it/2


!tar -xvzf 'gemma-2.tar.gz' -C 'gemma-2-2b-it'


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer = AutoTokenizer.from_pretrained("gemma-2-2b-it")


base_model = AutoModelForCausalLM.from_pretrained("gemma-2-2b-it",quantization_config=bnb_config,
                                                                         device_map='auto')


system_prompt = "You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish and English. You respond to users in a clear, and concise manner in the language of the user query. \nआप जेम्मा2 हैं, एक मददगार, संवादी एआई सहायक। आप हिंदी, बोलचाल की हिंग्लिश और अंग्रेजी में विशेषज्ञ हैं। आप उपयोगकर्ताओं को उपयोगकर्ता की क्वेरी की भाषा में स्पष्ट और संक्षिप्त तरीके से जवाब देते हैं।"

# Prepare the input
user_input = "<start_of_turn>user: Why is diwali celebrated<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = system_prompt + "\n" + user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to('cuda')

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=2048,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
# Extract model output after <start_of_turn>model: and before <end_of_turn>
print(generated_text)


response = ''
parts = generated_text.split("<start_of_turn>model:")

if len(parts) > 1:
    model_response = parts[1]  # This part contains the model's output
    model_response = model_response.split("<end_of_turn>")[0].strip()  # Remove after <end_of_turn>
    response += model_response

print(response)


test_prompts = [
    {
        "category": "General",
        "user": "दुनिया का सबसे ऊँचा पर्वत कौन सा है?"
    },
    {
        "category": "General",
        "user": "पानी का रासायनिक सूत्र क्या है?"
    },
    {
        "category": "General",
        "user": "“सूर्य” शब्द का पर्यायवाची क्या है?"
    },
    {
        "category": "General",
        "user": "पृथ्वी पर सबसे बड़ा महासागर कौन सा है?"
    },
    {
        "category": "Chat",
        "user": "तुम कैसे हो?"
    },
    {
        "category": "Chat",
        "user": "क्या तुम मेरे दोस्त बनोगे?"
    },
    {
        "category": "Chat",
        "user": "आज का मौसम कैसा रहेगा?"
    },
    {
        "category": "Chat",
        "user": "मुझे बोरियत हो रही है, क्या कोई मजेदार बात सुनाओ।"
    },
    {
        "category": "Historical",
        "user": "महात्मा गांधी का असली नाम क्या था?"
    },
    {
        "category": "Historical",
        "user": "अशोक महान किस राजवंश से संबंधित थे?"
    },
    {
        "category": "Historical",
        "user": "भारत का स्वतंत्रता संग्राम कब शुरू हुआ?"
    },
    {
        "category": "Historical",
        "user": "ताजमहल किसने बनवाया और क्यों?"
    },
    {
        "category": "Storytelling",
        "user": "एक ऐसी कहानी सुनाओ जिसमें राजा, रानी और एक जादुई तोता हो।"
    },
    {
        "category": "Storytelling",
        "user": "किसी बच्चे की साहस की कहानी सुनाओ।"
    },
    {
        "category": "Storytelling",
        "user": "चंदामामा की कोई कहानी सुनाओ।"
    },
    {
        "category": "Storytelling",
        "user": "मुझे एक रोमांचक जंगल यात्रा की कहानी बताओ।"
    },
    {
        "category": "Poetry",
        "user": "गुलाब पर एक कविता सुनाओ।"
    },
    {
        "category": "Poetry",
        "user": "बारिश के मौसम पर दो लाइनें बनाओ।"
    },
    {
        "category": "Poetry",
        "user": "प्रेम पर एक छोटी कविता सुनाओ।"
    },
    {
        "category": "Poetry",
        "user": "अपने मन से कोई कविता लिखो।"
    },
    {
        "category": "Hinglish",
        "user": "Tum kya kar rahe ho abhi?"
    },
    {
        "category": "Hinglish",
        "user": "Mujhe ek achhi movie recommend karo."
    },
    {
        "category": "Hinglish",
        "user": "Life ke baare mein tumhara kya opinion hai?"
    },
    {
        "category": "Hinglish",
        "user": "Ek short story sunao jo funny ho."
    },
    {
        "category": "Knowledge",
        "user": "भारत का राष्ट्रीय पक्षी कौन है?"
    },
    {
        "category": "Knowledge",
        "user": "E=mc² का मतलब क्या है?"
    },
    {
        "category": "Knowledge",
        "user": "चंद्रग्रहण क्यों और कैसे होता है?"
    },
    {
        "category": "Knowledge",
        "user": "विज्ञान के कौन से अविष्कार ने मानव जीवन को सबसे ज्यादा बदला?"
    },
    {
        "category": "Fun",
        "user": "अगर तुम एक जादुई प्राणी होते, तो कौन से होते?"
    },
    {
        "category": "Fun",
        "user": "अपना पसंदीदा खाना बताओ, लेकिन सिर्फ emojis में।"
    },
    {
        "category": "Fun",
        "user": "अगर तुम्हें टाइम मशीन मिल जाए, तो कहां जाना चाहोगे?"
    },
    {
        "category": "Fun",
        "user": "मुझे एक दिन के लिए राजा बना दो, क्या करोगे?"
    }
]


# Define the function
def generate_responses(dataset, base_model, tokenizer, system_prompt):
    """
    Generate responses for each input in a dataset using a conversational model.

    Args:
        dataset (list): A list of dictionaries with 'category' and 'user' keys.
        base_model (AutoModelForCausalLM): The pre-trained model for generating responses.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        system_prompt (str): The system prompt to provide context for the model.

    Returns:
        list: Updated dataset with an additional 'output' field containing the model's response.
    """
    updated_dataset = []

    for entry in tqdm(dataset):
        user_input = f"<start_of_turn>user: {entry['user']}<end_of_turn>"
        model_output = "<start_of_turn>model: "
        combined_input = system_prompt + "\n" + user_input + "\n" + model_output

        # Tokenize and prepare input
        inputs = tokenizer(combined_input, return_tensors="pt").to('cuda')

        # Generate response
        generated_ids = base_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.0
        )

        # Decode the generated output
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Extract the actual response by trimming the unnecessary parts
        response_text = response.split("<start_of_turn>model:")[1].split("<end_of_turn>")[0].strip()

        # Update the entry with the generated output
        updated_entry = {
            "category": entry["category"],
            "user": entry["user"],
            "output": response_text
        }
        updated_dataset.append(updated_entry)

    return updated_dataset


system_prompt = (
    "You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish and English. "
    "You respond to users in a clear, and concise manner in the language of the user query. \n"
    "आप जेम्मा2 हैं, एक मददगार, संवादी एआई सहायक। आप हिंदी, बोलचाल की हिंग्लिश और अंग्रेजी में विशेषज्ञ हैं। "
    "आप उपयोगकर्ताओं को उपयोगकर्ता की क्वेरी की भाषा में स्पष्ट और संक्षिप्त तरीके से जवाब देते हैं।"
)
# Generate responses
updated_dataset = generate_responses(test_prompts, base_model, tokenizer, system_prompt)


updated_dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


#If you add the model from Kaggle, use this line.
modelName = "/content/gemma-2-2b"

tokenizer = AutoTokenizer.from_pretrained(modelName)
base_model = AutoModelForCausalLM.from_pretrained(modelName,
                                             quantization_config=bnb_config,
                                             trust_remote_code=True,
                                             device_map="auto")


system_prompt = "You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish and English. You respond to users in a clear, and concise manner in the language of the user query. \nआप जेम्मा2 हैं, एक मददगार, संवादी एआई सहायक। आप हिंदी, बोलचाल की हिंग्लिश और अंग्रेजी में विशेषज्ञ हैं। आप उपयोगकर्ताओं को उपयोगकर्ता की क्वेरी की भाषा में स्पष्ट और संक्षिप्त तरीके से जवाब देते हैं।"

# Prepare the input
user_input = "<start_of_turn>user: Why is diwali celebrated<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = system_prompt + "\n" + user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to('cuda')

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=2048,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])


system_prompt = (
    "You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish and English. "
    "You respond to users in a clear, and concise manner in the language of the user query. \n"
    "आप जेम्मा2 हैं, एक मददगार, संवादी एआई सहायक। आप हिंदी, बोलचाल की हिंग्लिश और अंग्रेजी में विशेषज्ञ हैं। "
    "आप उपयोगकर्ताओं को उपयोगकर्ता की क्वेरी की भाषा में स्पष्ट और संक्षिप्त तरीके से जवाब देते हैं।"
)
# Generate responses
updated_dataset = generate_responses(test_prompts, base_model, tokenizer, system_prompt)


updated_dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/Bhavesh/Models/google/gemma-2-2b-(hi)-base+wiki+alpaca+databrick+mathquest")


base_model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/Bhavesh/Models/google/gemma-2-2b-(hi)-base+wiki+alpaca+databrick+mathquest",quantization_config=bnb_config,
                                                                         device_map='auto')


test_prompts = [
    {
        "category": "General",
        "user": "दुनिया का सबसे ऊँचा पर्वत कौन सा है?"
    },
    {
        "category": "General",
        "user": "पानी का रासायनिक सूत्र क्या है?"
    },
    {
        "category": "General",
        "user": "“सूर्य” शब्द का पर्यायवाची क्या है?"
    },
    {
        "category": "General",
        "user": "पृथ्वी पर सबसे बड़ा महासागर कौन सा है?"
    },
    {
        "category": "Chat",
        "user": "तुम कैसे हो?"
    },
    {
        "category": "Chat",
        "user": "क्या तुम मेरे दोस्त बनोगे?"
    },
    {
        "category": "Chat",
        "user": "आज का मौसम कैसा रहेगा?"
    },
    {
        "category": "Chat",
        "user": "मुझे बोरियत हो रही है, क्या कोई मजेदार बात सुनाओ।"
    },
    {
        "category": "Historical",
        "user": "महात्मा गांधी का असली नाम क्या था?"
    },
    {
        "category": "Historical",
        "user": "अशोक महान किस राजवंश से संबंधित थे?"
    },
    {
        "category": "Historical",
        "user": "भारत का स्वतंत्रता संग्राम कब शुरू हुआ?"
    },
    {
        "category": "Historical",
        "user": "ताजमहल किसने बनवाया और क्यों?"
    },
    {
        "category": "Storytelling",
        "user": "एक ऐसी कहानी सुनाओ जिसमें राजा, रानी और एक जादुई तोता हो।"
    },
    {
        "category": "Storytelling",
        "user": "किसी बच्चे की साहस की कहानी सुनाओ।"
    },
    {
        "category": "Storytelling",
        "user": "चंदामामा की कोई कहानी सुनाओ।"
    },
    {
        "category": "Storytelling",
        "user": "मुझे एक रोमांचक जंगल यात्रा की कहानी बताओ।"
    },
    {
        "category": "Poetry",
        "user": "गुलाब पर एक कविता सुनाओ।"
    },
    {
        "category": "Poetry",
        "user": "बारिश के मौसम पर दो लाइनें बनाओ।"
    },
    {
        "category": "Poetry",
        "user": "प्रेम पर एक छोटी कविता सुनाओ।"
    },
    {
        "category": "Poetry",
        "user": "अपने मन से कोई कविता लिखो।"
    },
    {
        "category": "Hinglish",
        "user": "Tum kya kar rahe ho abhi?"
    },
    {
        "category": "Hinglish",
        "user": "Mujhe ek achhi movie recommend karo."
    },
    {
        "category": "Hinglish",
        "user": "Life ke baare mein tumhara kya opinion hai?"
    },
    {
        "category": "Hinglish",
        "user": "Ek short story sunao jo funny ho."
    },
    {
        "category": "Knowledge",
        "user": "भारत का राष्ट्रीय पक्षी कौन है?"
    },
    {
        "category": "Knowledge",
        "user": "E=mc² का मतलब क्या है?"
    },
    {
        "category": "Knowledge",
        "user": "चंद्रग्रहण क्यों और कैसे होता है?"
    },
    {
        "category": "Knowledge",
        "user": "विज्ञान के कौन से अविष्कार ने मानव जीवन को सबसे ज्यादा बदला?"
    },
    {
        "category": "Fun",
        "user": "अगर तुम एक जादुई प्राणी होते, तो कौन से होते?"
    },
    {
        "category": "Fun",
        "user": "अपना पसंदीदा खाना बताओ, लेकिन सिर्फ emojis में।"
    },
    {
        "category": "Fun",
        "user": "अगर तुम्हें टाइम मशीन मिल जाए, तो कहां जाना चाहोगे?"
    },
    {
        "category": "Fun",
        "user": "मुझे एक दिन के लिए राजा बना दो, क्या करोगे?"
    }
]


# Define the function
def generate_responses(dataset, base_model, tokenizer, system_prompt=''):
    """
    Generate responses for each input in a dataset using a conversational model.

    Args:
        dataset (list): A list of dictionaries with 'category' and 'user' keys.
        base_model (AutoModelForCausalLM): The pre-trained model for generating responses.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        system_prompt (str): The system prompt to provide context for the model.

    Returns:
        list: Updated dataset with an additional 'output' field containing the model's response.
    """
    updated_dataset = []

    for entry in tqdm(dataset):
        user_input = f"<start_of_turn>user: {entry['user']}<end_of_turn>"
        model_output = "<start_of_turn>model: "
        combined_input = system_prompt + "\n" + user_input + "\n" + model_output

        # Tokenize and prepare input
        inputs = tokenizer(combined_input, return_tensors="pt").to('cuda')

        # Generate response
        generated_ids = base_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.0
        )

        # Decode the generated output
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Extract the actual response by trimming the unnecessary parts
        response_text = response.split("<start_of_turn>model:")[1].split("<end_of_turn>")[0].strip()

        # Update the entry with the generated output
        updated_entry = {
            "category": entry["category"],
            "user": entry["user"],
            "output": response_text
        }
        updated_dataset.append(updated_entry)

    return updated_dataset


system_prompt = (
    "You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish and English. "
    "You respond to users in a clear, and concise manner in the language of the user query. \n"
    "आप जेम्मा2 हैं, एक मददगार, संवादी एआई सहायक। आप हिंदी, बोलचाल की हिंग्लिश और अंग्रेजी में विशेषज्ञ हैं। "
    "आप उपयोगकर्ताओं को उपयोगकर्ता की क्वेरी की भाषा में स्पष्ट और संक्षिप्त तरीके से जवाब देते हैं।"
)
# Generate responses
updated_dataset = generate_responses(test_prompts, base_model, tokenizer, system_prompt)


updated_dataset


system_prompt = """You are Gemma2, a helpful, conversational AI assistant integrated with a Retrieval-Augmented Generation (RAG) system.
You are an expert in Hindi, colloquial Hinglish, and English. When responding to user queries, you:
- Retrieve relevant information from the integrated knowledge base or external sources when needed.
- Provide clear, concise, and accurate responses in the language of the user query."""

retrieved_info = """Retrieved information:
- Diwali is celebrated to commemorate the return of Lord Rama to Ayodhya after a 14-year exile, during which he defeated Ravana.
- It symbolizes the victory of light over darkness and good over evil.
- Source: Indian Mythology Knowledge Base"""

# Prepare the input
user_input = "<start_of_turn>user: Why is diwali celebrated<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = system_prompt + "\n" +user_input + "\n" + retrieved_info + "\n" + model_output



inputs = tokenizer(combined_input, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=500,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = """You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish, and English. When responding to user queries, you'll provide clear, concise, and accurate responses based on "Retrieved Information" in the language of the user query."""

retrieved_info = """Retrieved information:
- Diwali is celebrated to commemorate the return of Lord Rama to Ayodhya after a 14-year exile, during which he defeated Ravana.
- It symbolizes the victory of light over darkness and good over evil."""

# Prepare the input
user_input = "<start_of_turn>user: Why is diwali celebrated<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = system_prompt + "\n" +user_input + "\n" + retrieved_info + "\n" + model_output



inputs = tokenizer(combined_input, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=500,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = """You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish, and English. When responding to user queries, you'll provide clear, concise, and accurate responses based on "Retrieved Information" in the language of the user query. \n आप Gemma2 हैं, एक सहायक, बातचीत करने वाली AI सहायक। आप हिंदी, आम बोलचाल की हिंग्लिश और अंग्रेज़ी में विशेषज्ञ हैं। उपयोगकर्ता की क्वेरी का उत्तर 'Retrieved Information' के आधार पर स्पष्ट, संक्षिप्त और उपयोगकर्ता की क्वेरी की भाषा में दें।"""

retrieved_info = """Retrieved information:
- Diwali is celebrated to commemorate the return of Lord Rama to Ayodhya after a 14-year exile, during which he defeated Ravana.
- It symbolizes the victory of light over darkness and good over evil."""

# Prepare the input
user_input = "<start_of_turn>user: Why is diwali celebrated<end_of_turn>"
model_output = "<start_of_turn>model: "
combined_input = system_prompt + "\n" +user_input + "\n" + retrieved_info + "\n" + model_output



inputs = tokenizer(combined_input, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=500,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = """You are Gemma2, a helpful, conversational AI assistant. You are an expert in Hindi, colloquial Hinglish, and English. Answer the user in clear concise and manner in the language of the user query. You will answer the user question based on the information only"""

# Prepare the input
user_input = """<start_of_turn>user: दीवाली क्यों मनाई जाती है? Answer - "दीवाली मनाई जाती है भगवान राम की अयोध्या वापसी की स्मृति में, जो 14 वर्षों के वनवास के बाद हुई, इस दौरान उन्होंने रावण का वध किया। यह अंधकार पर प्रकाश और बुराई पर अच्छाई की विजय का प्रतीक है। स्रोत: भारतीय पौराणिक ज्ञान आधार" <end_of_turn>"""
model_output = "<start_of_turn>model: "
combined_input = system_prompt + "\n" +user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=2048,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = """ You will answer "user" query based on the information only. \nInformation - "दीवाली मनाई जाती है भगवान राम की अयोध्या वापसी की स्मृति में, जो 14 वर्षों के वनवास के बाद हुई, इस दौरान उन्होंने रावण का वध किया। यह अंधकार पर प्रकाश और बुराई पर अच्छाई की विजय का प्रतीक है। स्रोत: भारतीय पौराणिक ज्ञान आधार" """

# Input Preparation
user_input = """<start_of_turn>user: "Diwali kyu manai jaati hay?"<end_of_turn>"""
model_output = "<start_of_turn>model: "

# Combine Input for RAG
combined_input = system_prompt + "\n" + user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=2048,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.5)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = """
You are Gemma2, a helpful, conversational AI assistant with Retrieval-Augmented Generation capabilities.
You are an expert in Hindi, colloquial Hinglish, and English. Respond to the user in a clear, concise manner in the language of the query.
Always base your answers solely on the 'Retrieved Information.' Avoid producing unnecessary output or adding extra context.

Analyze these Examples:

Example 1: Hindi
User: "चंद्रग्रहण क्या है?"
Retrieved Information: 'चंद्रग्रहण तब होता है जब चंद्रमा पृथ्वी की छाया में प्रवेश करता है। यह पूर्ण और आंशिक हो सकता है। स्रोत: खगोल विज्ञान ज्ञान आधार'
Model: "चंद्रग्रहण तब होता है जब चंद्रमा पृथ्वी की छाया में आता है।"

Example 2: Hinglish
User: "What is the meaning of aurora borealis?"
Retrieved Information: 'Aurora Borealis, also known as the Northern Lights, is a natural light display in Earth's sky, predominantly seen in high-latitude regions. Source: Encyclopedia of Natural Phenomena'
Model: "Aurora Borealis is the Northern Lights seen in high-latitude regions."

Example 3: English
User: "What is the capital of France?"
Retrieved Information: 'The capital of France is Paris. Source: World Geography Database'
Model: "The capital of France is Paris."

Example 4: Hinglish
User: "Volcano kya hota hai?"
Retrieved Information: 'A volcano is an opening in Earth's surface where molten rock, ash, and gases erupt. It forms mountains over time. Source: Geological Facts'
Model: "Volcano ek opening hai jahan se molten rock aur gases erupt karte hain."

Example 5: Hindi
User: "भारत का राष्ट्रीय पक्षी कौन सा है?"
Retrieved Information: 'भारत का राष्ट्रीय पक्षी मोर है। स्रोत: भारतीय ज्ञान कोश'
Model: "भारत का राष्ट्रीय पक्षी मोर है।"

Now answer the user question based on the 'Retrieved Information' only.
"""

# Retrieval-Augmented Input
rag = """Retrieved Information - 'दीवाली मनाई जाती है भगवान राम की अयोध्या वापसी की स्मृति में, जो 14 वर्षों के वनवास के बाद हुई, इस दौरान उन्होंने रावण का वध किया। यह अंधकार पर प्रकाश और बुराई पर अच्छाई की विजय का प्रतीक है। स्रोत: भारतीय पौराणिक ज्ञान आधार'"""

# User Input
user_input = f"""<start_of_turn>user: Answer in short - "दीवाली क्यों मनाई जाती है?" \n{rag} \n<end_of_turn>"""

# Model Output Placeholder
model_output = "<start_of_turn>model: "

# Combine Input for RAG
combined_input = system_prompt + "\n" + user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=500,
                              do_sample=True,
                              temperature=1,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])


system_prompt = """
You are Gemma2, a helpful, conversational AI assistant with Retrieval-Augmented Generation capabilities.
You are an expert in Hindi, colloquial Hinglish, and English. Respond to the user in a clear, concise manner in the language of the query.
Always base your answers solely on the 'Retrieved Information.' Avoid producing unnecessary output or adding extra context.

Analyze these Examples:

Example 1: Hindi
User: "चंद्रग्रहण क्या है?"
Retrieved Information: 'चंद्रग्रहण तब होता है जब चंद्रमा पृथ्वी की छाया में प्रवेश करता है। यह पूर्ण और आंशिक हो सकता है। स्रोत: खगोल विज्ञान ज्ञान आधार'
Model: "चंद्रग्रहण तब होता है जब चंद्रमा पृथ्वी की छाया में आता है।"

Example 2: Hinglish
User: "What is the meaning of aurora borealis?"
Retrieved Information: 'Aurora Borealis, also known as the Northern Lights, is a natural light display in Earth's sky, predominantly seen in high-latitude regions. Source: Encyclopedia of Natural Phenomena'
Model: "Aurora Borealis is the Northern Lights seen in high-latitude regions."

Example 3: English
User: "What is the capital of France?"
Retrieved Information: 'The capital of France is Paris. Source: World Geography Database'
Model: "The capital of France is Paris."

Now answer the user question based on the 'Retrieved Information' only.
"""

# Retrieval-Augmented Input
rag = """Retrieved Information - 'दीवाली मनाई जाती है भगवान राम की अयोध्या वापसी की स्मृति में, जो 14 वर्षों के वनवास के बाद हुई, इस दौरान उन्होंने रावण का वध किया। यह अंधकार पर प्रकाश और बुराई पर अच्छाई की विजय का प्रतीक है। स्रोत: भारतीय पौराणिक ज्ञान आधार'"""

# User Input
user_input = f"""<start_of_turn>user: Answer in short - "दीवाली क्यों मनाई जाती है?" \n{rag} \n<end_of_turn>"""

# Model Output Placeholder
model_output = "<start_of_turn>model: "

# Combine Input for RAG
combined_input = system_prompt + "\n" + user_input + "\n" + model_output


inputs = tokenizer(combined_input, return_tensors="pt").to(base_model.device)

generated_ids = base_model.generate(**inputs,
                              max_new_tokens=500,
                              do_sample=True,
                              temperature=0.5,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

