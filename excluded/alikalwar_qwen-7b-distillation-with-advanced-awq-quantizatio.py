from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to('cuda')


from datasets import load_dataset

# Load MMLU dataset
mmlu = load_dataset("hendrycks_test", "abstract_algebra", split="test")

# Function to evaluate model accuracy
def evaluate_model(dataset, num_samples=100):
    correct = 0
    for i, data in enumerate(dataset):
        if i >= num_samples:
            break
        input_text = data['question'] + " " + " ".join(data['choices'])
        inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
        outputs = model.generate(inputs.input_ids, max_length=50)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if data['answer'] in answer:
            correct += 1
    accuracy = correct / num_samples
    return accuracy

# Evaluate on a subset of MMLU
accuracy = evaluate_model(mmlu)
print(f"MMLU Accuracy: {accuracy * 100:.2f}%")



math_problem = "Integrate the function f(x) = x^2 from 0 to 1."
inputs = tokenizer(math_problem, return_tensors="pt").to('cuda')
outputs = model.generate(inputs.input_ids, max_length=100)
solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Problem: {math_problem}\nSolution: {solution}")


code_prompt = "Write a Python function to calculate the factorial of a number."
inputs = tokenizer(code_prompt, return_tensors="pt").to('cuda')
outputs = model.generate(inputs.input_ids, max_length=100)
code_snippet = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {code_prompt}\nGenerated Code:\n{code_snippet}")


import ipywidgets as widgets
from IPython.display import display

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs.input_ids, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create text box and button
text_box = widgets.Textarea(
    value='',
    placeholder='Type your prompt here',
    description='Prompt:',
    disabled=False
)
button = widgets.Button(description="Generate")

# Display output
output = widgets.Output()

def on_button_click(b):
    with output:
        output.clear_output()
        prompt = text_box.value
        response = generate_response(prompt)
        print(f"Response:\n{response}")

button.on_click(on_button_click)

display(text_box, button, output)

