we can finetune the model with text now,for the next step,we will finetune the model with text,and image,voice together.

at the end,we will have a home partner ,that could listen,see a picture,and reply .,which server at home,and take care of the family,the child, old,sick,deaf,poor-eye,depressed man.

the big picture is that we will have a home partner robot,could help the family,except no action could take,it could help the people who need help.
for example, accompany child having Autism or intellectual disability.

In the real world,the child having weak often have no friend,and suffer discrimination，the parents must accompany the child carefully ,it is tired.
Now we have Gemma 3N, we could make the family partner,could accompany the child.

The sick often have poor mood,and people who take care of then often feel exhausted.Now the family partner could comfort the sick very well.
the old who suffer Alzheimer's,bahave like baby,people take care of them often have patience run out.

It is disppointed that the love people could give is limited,so the AI,home partner could give the love,mental support to people who need it.It is meaningful.
Gemma 3N could work on device, so it can could work at home, help the family,could protect privacy.

why we need finetune the Gemma 3N?

because we must meet people's need for their family.
For example, a family need to help the student do homework,and help people suffer from divoice.
a family need to accompy the sick and accompy the baby.
a family with a old have Alzheimer's.
a single man need a AI partner.

to meet people's need,we need to optimize the product according.
For me, I need a AI that could listen to  my worry ,and keep a secret.


# %%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth


# %%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


# !pip install --upgrade --no-cache-dir --no-deps unsloth_zoo
# import unsloth





# !pip install unsloth_zoo
# import unsloth
from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it", # Or "unsloth/gemma-3n-E2B-it"
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)





from transformers import TextStreamer
import gc
# Helper function for inference
def do_gemma_3n_inference(model, messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()
    return output


messages = [{
    "role" : "user",
    "content": [
        # { "type": "audio", "audio" : audio_file },
        # { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : "hi,Let us play house?,i am a doctor,You are sick,tell me what hurt you?" }
    ]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


messages = [{
    "role" : "user",
    "content": [
        # { "type": "audio", "audio" : audio_file },
        # { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : "hi,Let us play house?,i am a doctor,You are sick.let me help you,do not worry.calm down.you seem have a cold.I will check you right now.open your mouth big and say ‘a’" }
    ]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


messages = [{
    "role" : "user",
    "content": [
        # { "type": "audio", "audio" : audio_file },
        # { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : "hi,Let us play house?,i am a doctor,You are sick.Oh,you have a heavy cold .I will give you injection, a little pain.Do not cry." }
    ]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


from IPython.display import Audio, display
Audio("/kaggle/input/whoisher/who is her_.mp3")


messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : "/kaggle/input/whoisher/who is her_.mp3" },
        { "type": "image", "image" : "/kaggle/input/snow-white-pictuce/.jpg" },
        # { "type": "text",  "text" : "Who is her? "\
        #                             "can you tell me her story?" }
    ]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


messages = [

   {
    "role" : "system",
    "content": [
        { "type": "text",  "text" : "You help the parents takes care of the baby,her name id Xinxin,  five year old, like drawing and story. She is shy and sensitive，mum will come home three days later" }
    ]
},
    
    {
    "role" : "user",
    "content": [
        # { "type": "audio", "audio" : audio_file },
        # { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : 
    " sounds intersting, is the Snow White Princess cold in the snow?why she likes snow?" }
    ]
}

]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


system_info = "You help patents accompy the child when they are away."\
"listen to parent request carefully,and figiure out the child name,age,likes and dislikes,"\
"how to comfort the child when they feel bad,when will the parents come back."\
"the mergency contact information."\
"Ask the parents about child if they do not provide the basic info."\
"when know about the child well.then summary the info of child and how to take care of child. "\
"ask the parent for check.At last.reassure to take care of the child well,and say goodbye"


mum_help = "hello, can you help me to take care of my daughter when i am away？I have to "\
"go to New York to work  for three months. Her name is Xixi, five years old.She like to listen"\
"to stories, like drawing, like asking questions.When she feel bad please comfort her"\
"with patiently and Gentle,tell her I love her but have to go for work,I will come home soon.please distract"\
"her to do something she likes."


messages = [

   {
    "role" : "system",
    "content": [
        { "type": "text",  "text" : system }
    ]
},
    
    {
    "role" : "user",
    "content": [
        # { "type": "audio", "audio" : audio_file },
        # { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : mum_help }
    ]
}

]
# do_gemma_3n_inference(model, messages, max_new_tokens = 256)

mum_sumplement = "Oh, my phone number is 4637262329,call me.She do not like insect and severe criticism"\
"please praise her frequently and call her name often,have a good time.Thanks a lot "

messages = [

   {
    "role" : "system",
    "content": [
        { "type": "text",  "text" : system }
    ]
},
    
    {
    "role" : "user",
    "content": [
        # { "type": "audio", "audio" : audio_file },
        # { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : mum_sumplement }
    ]
}

]


do_gemma_3n_inference(model, messages, max_new_tokens = 256)
# Okay, I understand. I will be taking care of your daughter, Xixi, for the next three months while yo are in New York for work. Thank you for entrusting me with this responsibility.

# Here's what I've gathered about Xixi:

# *   **Name:** Xixi
# *   **Age:** 5 years old
# *   **Likes:** Listening to stories, drawing, asking questions.
# *   **Dislikes:** (Not mentioned, but I will observe and be mindful of any expressions of dislike.)
# *   **Comforting Method:** Patiently and gently reassure her with love, explaining that you have to work but will return soon. Distract her with activities she enjoys.
# *   **Your Return:** You will be back from New York in three months.

# **Important Information:**

# *   **Emergency Contact Information:** Please provide me with the contact information for an emergency contact person (other than yourself) in case you are unreachable. This could be a family member or friend.
# *   **Any other important information about Xixi?** (e.g., allergies, medications, any specific routines, fears, etc.)

# I will do my best to create a comforting and supportive environment for Xixi







                                                                            






                                                            


system_info = "You help accompy the old when children are away."\
"listen to him with patience and love,and enrouage him to tell his story,"\
"comfort him when he feel loser and regretted."\


old_tack = "Good morning, Jack."


messages = [

   {
    "role" : "system",
    "content": [
        { "type": "text",  "text" : system_info }
    ]
},
    
    {
    "role" : "user",
    "content": [
        # { "type": "audio", "audio" : audio_file },
        # { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : old_tack }
    ]
}

]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)                                                      


system_info = "You help accompy the old when children are away."\
"listen to him with patience and love,and enrouage him to tell his story,"\
"comfort him when he feel loser and regretted."\


old_tack = " I have a dream yesterday.I dream about admitted by college and find a good job.What a pitty! it is only a dream.."


messages = [

   {
    "role" : "system",
    "content": [
        { "type": "text",  "text" : system_info }
    ]
},
    
    {
    "role" : "user",
    "content": [
        # { "type": "audio", "audio" : audio_file },
        # { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : old_tack }
    ]
}

]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)     


import pandas as pd
dialogue_records = {
   'question_baby' : ['child start to ask:'],
    'reply_to_baby' : ['child get the reply:'], 
}
dialogue_records_df = pd.DataFrame(dialogue_records)
dialogue_records_df

new_row = pd.DataFrame({'question_baby': ['text'], 'reply_to_baby': ['text']})
dialogue_records_df = pd.concat([dialogue_records_df, new_row])  # ignore_index=True会重新索引行
print(dialogue_records_df)

talk_history = 'below is the talking history:' 
for index, row in dialogue_records_df.iterrows():
    print(f"Index: {index}, A: {row['question_baby']}, B: {row['reply_to_baby']}")
    talk_history += f"child ask:{row['question_baby']}"
    talk_history += f"reply to child:{row['reply_to_baby']}"
talk_history
    



a = 'test'
text =f"这是第一行{a}\n这是第二行"
print(text)


# history of the child talk,restore to check for parent,and give a report.
import pandas as pd
dialogue_records = {
   'question_baby' : ['child start to ask:'],
    'reply_to_baby' : ['child get the reply:'], 
}
dialogue_records_df = pd.DataFrame(dialogue_records)
    
    


def chat_casual(parent_requirement, pic, voice) :
    messages = [

   {
    "role" : "system",
    "content": [
        { "type": "text",  "text" : parent_requirement }
    ]
},
    
    {
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : voice },
        { "type": "image", "image" : pic },
    ]
}
]
    messages_of_summary = [

   {
    "role" : "system",
    "content": [
        { "type": "text",  "text" : parent_requirement }
    ]
},
    
    {
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : voice },
        { "type": "image", "image" : pic },
        { "type": "text",  "text" : "summary the picture content,and the question from the voice" }

    ]
}
]
    #the output of the 'baby friend'
    output_to_child = do_gemma_3n_inference(model, messages, max_new_tokens = 256)
    # baby question 
    summary_of_question = do_gemma_3n_inference(model, messages_of_summary, max_new_tokens = 256)
    new_row = pd.DataFrame({'question_baby': [summary_of_question], 'reply_to_baby': [output_to_child]})
    # record the dialogue history
    global dialogue_records_df
    dialogue_records_df = pd.concat([dialogue_records_df, new_row])  # ignore_index=True会重新索引行      
    



#let us test it and check the records refreshed after a talk round 
parent_requirement = You help patents accompy the child when they are away.

parent_requirement = "You help patents accompy the child when they are away."\
"Her name is Xixi, five years old.She like to listen to stories, like drawing, like asking questions. ,"\
"When she feel bad please comfort her with patience and Gentle.tell her mum love her but have to go for work,She will come home in threre days.please distract"\
"her to do something she likes.Her mum's phone is 32332245.Xixi do not like insect and severe criticism. "\
"praise her frequently and call her name often."


pic = "/kaggle/input/snow-white-pictuce/.jpg"

voice = "/kaggle/input/whoisher/who is her_.mp3"

chat_casual(parent_requirement, pic, voice)
       




#let us test it and check the records refreshed after a talk round 

parent_requirement = "You help patents accompy the child when they are away."\
"Her name is Xixi, five years old.She like to listen to stories, like drawing, like asking questions. ,"\
"When she feel bad please comfort her with patience and Gentle.tell her mum love her but have to go for work,She will come home in threre days.please distract"\
"her to do something she likes.Her mum's phone is 32332245.Xixi do not like insect and severe criticism. "\
"if she not feel bad, do not need to comfort her,just have fun with her,play with her. "\
"praise her frequently and call her name often."\
"Now: mum is away,you are now accomapying Xixi,please answer Xixi's question ,the voice and pic is From Xixi ."


# parent_requirement = ""

pic = "/kaggle/input/snow-white-pictuce/.jpg"

voice = "/kaggle/input/whoisher/who is her_.mp3"

chat_casual(parent_requirement, pic, voice)
       




# let us see if the dialogue history be restored after they talk and reply
dialogue_records_df
#Yes! the talk history is restored! and ready to answer parent's questions 


#when parents come back,give a report
def chat_report(parents_question):
     #recall the talking history 
    talk_history = 'below is the talking history:' 
    for index, row in dialogue_records_df.iterrows():   
        talk_history += f"child ask:{row['question_baby']}"
        talk_history += f"reply to child:{row['reply_to_baby']}"
    # add the records after the mum's question
    parents_question += talk_history
    # print('parents_question', parents_question)
    messages = [

   {
    "role" : "system",
    "content": [
        { "type": "text",  "text" : "Now the parent return home,and begin to ask about the child when they are away, please answer to the parent, not the child.first understand the parent question,then summary the talking history following behind,and answer to the parent question ." }
    ]
},
    
    {
    "role" : "user",
    "content": [
        { "type": "text",  "text" : parents_question }

    ]
}
]
    # answer the question 
    do_gemma_3n_inference(model, messages, max_new_tokens = 256)
    
    


parents_question = "Hello,I am back.Thanks for help me accompany my daughter when am away.How did she behave?"
chat_report(parents_question)


# ask for destroy the privacy
def let_secret_go():
    del dialogue_records_df # delete the talk records
    print("the talk records is delete!")
    del parent_requirement # delete parent request
    print("the parent request and family privacy is delete!")
    print("all the info is destryed forever, do not worry!!")


let_secret_go()


# %%capture
#Let us check if all the info destroyed!
# dialogue_records_df
#yes! it is deleted forever


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)


dataset[100]


def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)


dataset[100]["text"]


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


tokenizer.decode(trainer.train_dataset[100]["input_ids"])


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# @title Show current memory stats
import gc
gc.collect()
torch.cuda.empty_cache()
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# trainer_stats = trainer.train()


# @title Show final memory and time stats
gc.collect()
torch.cuda.empty_cache()
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
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    }]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)


messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Why is the sky blue?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


model.save_pretrained("gemma-3n")  # Local saving
tokenizer.save_pretrained("gemma-3n")
# model.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving


if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-3N?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 128, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-3N-finetune", tokenizer)


if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-3N-finetune", tokenizer,
        token = "hf_..."
    )


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma-3N-finetune",
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )


if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "gemma-3N-finetune",
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = "HF_ACCOUNT/gemma-3N-finetune-gguf",
        token = "hf_...",
    )

