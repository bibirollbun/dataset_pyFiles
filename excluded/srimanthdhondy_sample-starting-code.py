import os


#Switching to the required directory for saving the output
os.chdir("../working")


import os

# Set the environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Verify that the environment variable is set
print("PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


!pip install peft trl


!pip install --upgrade bitsandbytes


!pip install --upgrade transformers


import transformers
#Checking the version of transformers package
print(transformers.__version__)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import TrainingArguments,Trainer,BitsAndBytesConfig,DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
import pandas as pd
from peft import LoraConfig, AutoPeftModelForCausalLM, TaskType, PeftModel
from trl import SFTConfig, SFTTrainer


device = "auto"
model_path = "ibm-granite/granite-3.1-2b-base"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 # if not set will throw a warning about slow speeds when training
)

model = AutoModelForCausalLM.from_pretrained(
  model_path,
  #quantization_config=bnb_config,
  device_map="auto"
)
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#tokenizer.padding_side = "right"


'''dataset = load_dataset('csv', data_files='/kaggle/input/pde-heat-equation-dataset/merged_dataset.csv')

# Access the 'train' split and perform train-validation split
dataset_split = dataset['train']  # Select the single split from the dataset
train_test_split = dataset_split.train_test_split(test_size=0.2)  # 80% train, 20% validation

# Convert back to a DatasetDict
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

# Access train and validation datasets
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]'''


#train_dataset


'''validation_dataset'''


'''def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['Question'])):
        text = f"### Question: {example['Question'][i]}\n ### Answer: {example['Answer'][i]}"
        output_texts.append(text)
    return output_texts

#response_template = " ### Answer:"
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)'''


#metric = cosine_similarity
#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    return metric(X=predictions, Y=labels)


'''# Apply qLoRA
qlora_config = LoraConfig(
    r=16,  # The rank of the Low-Rank Adaptation
    lora_alpha=32,  # Scaling factor for the adapted layers
    target_modules=["q_proj", "v_proj"],  # Layer names to apply LoRA to
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Initialize the SFTTrainer
training_args = TrainingArguments(
    output_dir="/kaggle/working/",
    learning_rate=5e-4,
    #per_device_train_batch_size=1,
    #per_device_eval_batch_size=1,
    num_train_epochs=5,
    logging_steps=100,
    fp16=True,
    save_total_limit=1,
    report_to="none",
    run_name="Train1",
    #eval_strategy='steps',
    #eval_steps=100,
    save_strategy='steps',
    auto_find_batch_size = True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    peft_config = qlora_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    #compute_metrics=compute_metrics
)'''


#trainer.train()


#trainer.save_model("/kaggle/working/modified_granite_model")


#model.save_pretrained("modified_granite_model")
#tokenizer.save_pretrained("modified_granite_model")


def load_model():
    device = "auto"
    model_path = "/kaggle/input/ibm_fine_tuned_heat_equation_v3/pytorch/default/1/modified_granite_model_lora_adapter_v3"
    model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
    )
    #merged_model = PeftModel.from_pretrained(model, lora_model_path)
    #merged_model.merge_and_unload()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    return model, tokenizer


def get_model_response(model, tokenizer, prompt, max_tokens=300):
    input_text = f"You are helpful assistant.You are designed to solve partial differential eqautions, especially Heat Equation.You are only supposed to generate the solution to the given heat equation and nothing more than that.I once remind you that, you are only supposed to generate the solution to the given problem and nothing more.Question:\n{prompt}\n\nAnswer:"
    #input_text = f"<|system|>\nYou are helpful assistant.You are designed to solve partial differential eqautions, especially Heat Equation.You are only supposed to generate the solution to the given heat equation and nothing more than that.I once remind you that, you are only supposed to generate the solution to the given problem and nothing more.\n<|user|>\n{prompt}\n<|assistant|>\n"
    input_tokens = tokenizer(input_text, return_tensors="pt").to(model.device)
    output = model.generate(
        **input_tokens,
        max_new_tokens=max_tokens,
        temperature=1.1,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        num_beams=2,
    )
    
    # Decode and clean up the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response[len(input_text):].strip()
    return response


def model_inference():
    model, tokenizer = load_model()
    
    cases = {
        "Case1Q1": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        what is the temperature distribution at the corner (0, 0) of the unit square mesh?""",
        "Case1Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        how does the temperature change with respect to the position along the x-axis at y = 0.5?""",
        "Case1Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        if we increase the coeeficient of pi in the force function what will happen?""",
        "Case2Q1": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        explain why the temperature is zero at both x=0 and x=1, and what this means physically.""",
        "Case2Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        at what coordinates does the maximum temperature occur, and what determines this location?""",
        "Case2Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        how does the temperature profile change along the vertical line x=0.5 compared to x=0.25?""",
        "Case3Q1": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        what is the temperature at the corner (0, 0) of the unit square mesh?""",
        "Case3Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        what physical significance does the boundary condition u(0,y)=0 have in the context of heat diffusion on the unit square mesh?""",
        "Case3Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        what does the boundary condition u(1,y)=y(1−y) represent physically in this heat diffusion problem?""",
        "Case4Q1": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        what can you infer about the decay rate of temperature!""",
        "Case4Q2": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        Comment on the physical interpretation of why the spatial pattern remains unchanged while only the amplitude decreases with time@""",
        "Case4Q3": """Analyze this steady-state heat equation solution T(x,y) = x^2+y^2 in a unit square domain and tell 
        What is the effect of alpha on the deacy rate of heat dissipation#"""
    }
    
    results = []
    
    for case, prompt in cases.items():
        print(f"\nTesting {case}...")
        try:
            response = get_model_response(model, tokenizer, prompt)
            results.append(response)
            print(f"\nResponse for {case}:")
            print(response)
        
        except Exception as e:
            print(f"Error in {case}: {str(e)}")
            results.append(f"Error: {str(e)}")
    
    return results


!rm /kaggle/working/submission.csv


results = model_inference()


df = pd.DataFrame({"Id" : list(range(1,13)), "Answer": results})
df.to_csv('submission.csv',index = False)

