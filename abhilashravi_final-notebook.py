import os


#Switching to the required directory for saving the output
os.chdir("../working")


!pip install transformers==4.46.0


import transformers
#Checking the version of transformers package
print(transformers.__version__)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


import pandas as pd


device = "auto"
model_path = "ibm-granite/granite-3.1-2b-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)


import os
from vtk import vtkUnstructuredGridReader

def process_vtk_files(directory):
    data_pairs = []
    for file in os.listdir(directory):
        if file.endswith(".vtk"):
            file_path = os.path.join(directory, file)
            reader = vtkUnstructuredGridReader()
            reader.SetFileName(file_path)
            reader.Update()
            data = reader.GetOutput()
            # Extract specific fields (e.g., temperature distribution, grid points)
            # Convert to input-output text format
            input_text = f"Grid Points: {data.GetNumberOfPoints()}"
            output_text = "Heat Equation Solution: ..."
            data_pairs.append((input_text, output_text))
    return data_pairs

# Use the directory containing your .vtk files
data = process_vtk_files("/kaggle/working")
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load the pretrained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("modified_granite_model")
tokenizer = AutoTokenizer.from_pretrained("modified_granite_model")

# Load and tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)

# Prepare data
data_pairs = [{"input": inp, "output": out} for inp, out in data]
dataset = Dataset.from_list(data_pairs)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")



# model.save_pretrained("modified_granite_model")
# tokenizer.save_pretrained("modified_granite_model")


def load_model():
    device = "auto"
    model_path = "./finetuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    model.eval()
    return model, tokenizer


def get_model_response(model, tokenizer, prompt, max_tokens=300):
    input_text = f"Question: {prompt}\n\nAnswer:"
    input_tokens = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    output = model.generate(
        **input_tokens,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
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


results = model_inference()


df = pd.DataFrame({"Id" : list(range(1,13)), "Answer": results})
df.to_csv('submission.csv',index = False)




