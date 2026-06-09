import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ["TORCH_COMPILE_SKIP"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# torch._dynamo.config.suppress_errors = True
# torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
# torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul
# torch.backends.cudnn.allow_tf32 = True

import json
import re
import ast
import torch._inductor.config as config
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_dataset, Dataset
from kaggle_secrets import UserSecretsClient
import kagglehub
from kagglehub import KaggleDatasetAdapter
from transformers.models.gemma3 import Gemma3ForConditionalGeneration, Gemma3Processor
from huggingface_hub import login, snapshot_download
import gc

# config.triton.cudagraphs = False
# config.max_autotune_gemm = False
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(f"CUDA devices visible: {torch.cuda.device_count()}")
# os.environ["WANDB_DISABLED"] = "true"
# user_secrets = UserSecretsClient()
# hf_token = user_secrets.get_secret("HF_TOKEN")
# kaggle_key = user_secrets.get_secret("KAGGLE_KEY")
# kaggle_username = user_secrets.get_secret("KAGGLE_USERNAME")

# !pip install -U "huggingface_hub[cli]"

# !pip install kagglehub[hf-datasets]
# !hf auth login --token $hf_token --add-to-git-credential

# os.environ['HF_TOKEN'] = hf_token
# os.environ['KAGGLE_KEY'] = kaggle_key
# os.environ['KAGGLE_USERNAME'] = kaggle_username

# !kaggle competitions download -c arc-prize-2025
# !unzip arc-prize-2025.zip

# model_name = "google/gemma-3-1b-it"

model_path = "/kaggle/input/gemma-3/transformers/gemma-3-270m-it/1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             # token=hf_token,
                                             trust_remote_code=False,
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="eager",
                                             device_map="balanced",
                                             low_cpu_mem_usage=True,
                                            )

tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          # token=hf_token,
                                          trust_remote_code=False,
                                          torch_dtype=torch.bfloat16,
                                          # use_fast=True,
                                         )

model = torch.nn.DataParallel(model) # This wraps the model
model.to(device)
model.eval()
# model = torch.compile(model)


def preprocess_function(examples):
    # Combine prompt and solution for training
    texts = [f"Prompt: {p}\nlabels: {s}" for p, s in zip(examples['prompt'], examples['labels'])]
    tokenized_inputs = tokenizer(texts, truncation=True, padding='longest')#, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

    return tokenized_inputs

def first_list(response):
    refined_pattern = r"\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\]"

    # Use regex to find the potential list-of-lists string first
    list_string_match = re.findall(refined_pattern, response)
    if list_string_match:  # Check if the list is not empty
      return list_string_match[0]
    else:
      return response[0:100]


def string_to_list(string):
    # print("this is string", list(string))
    main_list = []
    sub_list = []
    bracket_counter = 0
    for i in list(string):
        if i.isnumeric():
            sub_list.append(int(i))
            bracket_counter = 0
        elif i == "]" and bracket_counter == 1:
            # print("this is main_list", main_list)
            return main_list
        elif i == "]":
            main_list.append(sub_list)
            sub_list = []
            bracket_counter+=1
            continue
        else: 
            continue


def generate_prediction(output):
    submission_list = []
    puzzle = list(output.items())

    prompt_id = puzzle[0][0]
    prompt = puzzle[0][1]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode(): # Disable gradient calculation for inference
      try:
        attempt_1 = model.module.generate(
            **inputs,
            max_new_tokens=5000,
            do_sample=True,
            # temperature=0.7,
            # top_k=40,
            # top_p=0.5,
            # repetition_penalty=1,
            use_cache=True,
        ).to(device)

      except Exception as e:
        submission_list.append([prompt_id, [{"attempt_1":[[0]], "attempt_2":[[0]]}]])
        print(f"An error occurred: {e}")
        return submission_list


    # print("Generation complete for attempt_1.")

    response_text_1 = tokenizer.decode(attempt_1[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    first_list_1 = first_list(response_text_1)
    list_attempt_1 = string_to_list(first_list_1)

    with torch.inference_mode(): # Disable gradient calculation for inference
      try:
        attempt_2 = model.module.generate(
            **inputs,
            max_new_tokens=5000,
            do_sample=True,
            temperature=0.1,
            top_k=9,
            top_p=0.8,
            repetition_penalty=1,
            use_cache=True,
        ).to(device)

      except Exception as e:
        submission_list.append([prompt_id, [{"attempt_1":[[0]], "attempt_2":[[0]]}]])
        print(f"An error occurred: {e}")
        return submission_list

    response_text_2 = tokenizer.decode(attempt_2[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    first_list_2 = first_list(response_text_2)
    list_attempt_2 = string_to_list(first_list_2)
    
    # Save the last processed index after each puzzle
    if list_attempt_1 == None:
        first_attempt_1 = [[0]]
        submission_list.append([prompt_id, [{"attempt_1":list_attempt_1, "attempt_2":list_attempt_2}]])
    elif list_attempt_2 == None:
        list_attempt_2 = [[0]]
        submission_list.append([prompt_id, [{"attempt_1":list_attempt_1, "attempt_2":list_attempt_2}]])
    else:
        submission_list.append([prompt_id, [{"attempt_1":list_attempt_1, "attempt_2":list_attempt_2}]])
    del response_text_1, first_list_1, response_text_2, first_list_2, list_attempt_1, list_attempt_2

    # Clear memory
    del prompt, prompt_id, inputs, attempt_1, attempt_2
    gc.collect()
    torch.cuda.empty_cache()
    
    return submission_list


def main():
        counter = 0
        prompt_type = 1
        
        in_testing_challenges_file_path = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
        # eval_file_path = '/kaggle/input/arc-prize-prompts-v2/evaluation_prompts_v2.csv'
        shot_template = "Input {index}:\n{train_input}\n\nOutput {index}:\n{train_output}\n\n---\n\n"
        
        submission_dump = {}
        task_counter = 0
        with open(in_testing_challenges_file_path, 'r', encoding='utf-8') as f:
            challenge_parser = json.load(f)
            # print("this is the first item in the list", list(parser)[0].items())
            test_challenge_dict_items = challenge_parser.items()

            for record in test_challenge_dict_items:
                
                task_counter +=1
                
                task_id = record[0]
                task_train_data = record[1]["train"]
                task_test_data = record[1]["test"]

                if prompt_type not in [1,2,3,4,5]:
                    prompt_type = 1
                match prompt_type:
                    case 1:
                        prompt = f"this task requires an expert pattern recognition system helping to solve ARC Prize 2025 puzzles. These puzzles test abstract reasoning through grid transformations.\n\n## Task Description\nGiven input-output grid pairs, identify the transformation pattern and apply it to a new input.\n\n## Grid Representation\n- Each grid is a 2D array where numbers represent distinct cell labels/colors\n- Common labels: 0 (black/empty), 5 (gray), 6 (magenta/background), 7 (orange), 9 (maroon/border)\n- Focus on spatial relationships, symmetries, and structural changes between input and output\n\n**STRICT INSTRUCTION:** **RESPOND ONLY with the resultant list for the Test Output.** **DO NOT include any explanation, preamble, reasoning, code block formatting, markdown headers, or any text other than the nested list structure itself.**\n\n## Training Examples\n\n"

                        # add the few shot learning case examples
                        for index, pair in enumerate(task_train_data):
                            prompt += shot_template.format(index=index+1, train_input=pair['input'], train_output=pair['output'])
                        prompt += f"## Analysis Strategy\n\n1. Identify structural elements (borders, dividers, regions)\n2. Compare corresponding regions between input and output\n3. Look for transformations: reflection, rotation, value swapping, or merging\n4. Check for consistency across all examples\n\n## Test Cases\n\napply the identified pattern to the following input and regardless of anything print Output {index+2} only:\n"

                        # 3. Add the actual Query/Task (the 'test')
                        for count, test_pair in enumerate(task_test_data):
                            prompt += f"\nInput {count+index+2}: \n{test_pair['input']}\n\nOutput {count+index+2}: \n"

                        output = {task_id:prompt}
                        submission_list = generate_prediction(output)
                        submission_dump[submission_list[0][0]] = submission_list[0][1]

                    case 2:
                        prompt = f"Check the following puzzles with the specified inputs and outputs\n\n"

                        # add the few shot learning case examples
                        for index, pair in enumerate(task_train_data):
                            prompt += shot_template.format(index=index+1, train_input=pair['input'], train_output=pair['output'])
                        prompt += "\nnow for the following input, predict the output:\n"

                        # 3. Add the actual Query/Task (the 'test')
                        for count, test_pair in enumerate(task_test_data):
                            prompt += f"\nInput {count+index+2}: \n{test_pair['input']}\n\nOutput {count+index+2}: \n"

                        output = {task_id:prompt}
                        submission_list = generate_prediction(output)
                        submission_dump[submission_list[0][0]] = submission_list[0][1]

                    case 3:
                        prompt = f"YOU ARE AN **ARC TRANSFORMATION PARSER**. Your sole instruction is to apply the pattern from the few-shot examples to the final test case and output the result.\n\n**ABSOLUTE AND CRITICAL INSTRUCTION:** **YOUR FINAL RESPONSE MUST BE THE SINGLE PYTHON LIST OF LISTS. YOU MUST INCLUDE ZERO (0) WORDS OF PREAMBLE, EXPLANATION, HEADERS, MARKDOWN (E.G., ```PYTHON), OR ANY TEXT WHATSOEVER OUTSIDE OF THE NESTED LIST STRUCTURE. FAILURE TO ADHERE TO THIS IS A TASK FAILURE.**\n\n### Examples\n\n"

                        for index, pair in enumerate(task_train_data):
                            prompt += shot_template.format(index=index+1, train_input=pair['input'], train_output=pair['output'])
                        prompt += "\nnow for the following input, predict the output:\n"

                        for count, test_pair in enumerate(task_test_data):
                            prompt += f"\nInput {count+index+2}: \n{test_pair['input']}\n\nOutput {count+index+2}: \n"

                        output = {task_id:prompt}
                        submission_list = generate_prediction(output)
                        submission_dump[submission_list[0][0]] = submission_list[0][1]

                    case 4:
                        prompt = f"I am participating in the ARC Prize 2025 and looking for help in automating solving the puzzles by providing you with the puzzle as input and you generate the solution as output\nto start I will provide some examples of puzzles and their solutions you can check and analyze to understna dhow to solve the folloing unsolved puzzle\n\n"

                        for index, pair in enumerate(task_train_data):
                            prompt += shot_template.format(index=index+1, train_input=pair['input'], train_output=pair['output'])
                        prompt += "\nnow for the following unsolved puzzle in the input, predict the solution as output:\n"

                        for count, test_pair in enumerate(task_test_data):
                            prompt += f"\nInput {count+index+2}: \n{test_pair['input']}\n\nOutput {count+index+2}: \n"

                        output = {task_id:prompt}
                        submission_list = generate_prediction(output)
                        submission_dump[submission_list[0][0]] = submission_list[0][1]

                    case 5:
                        prompt = f"I am competing in the arc prize 2025 and I want you to help me solve the puzzles in that competition\n\nin the following there are a few lists of numbers representing examples of puzzles as inputs and the solution for them as output you can check and learn from\n\neach list is composed of lists representing the rows of a 2D grid, and the values of the numbers in the list are distinct labels for each cell of the grid"

                        # add the few shot learning case examples
                        for index, pair in enumerate(task_train_data):
                            prompt += shot_template.format(index=index+1, train_input=pair['input'], train_output=pair['output'])
                        prompt += "\nnow similar to the examples, please gemma check the following Input, and generate the Output\n"

                        # 3. Add the actual Query/Task (the 'test')
                        for count, test_pair in enumerate(task_test_data):
                            prompt += f"\nInput {count+index+2}: \n{test_pair['input']}\n\nOutput {count+index+2}: \n"

                        output = {task_id:prompt}
                        submission_list = generate_prediction(output)
                        submission_dump[submission_list[0][0]] = submission_list[0][1]
                
                del output, submission_list, prompt
                gc.collect()
                torch.cuda.empty_cache()
                counter+=1
                print(counter)
                # print(submission_dump)
                prompt_type += 1

        output_submission_file = "submission.json" # You can name your submission file
        
        with open(output_submission_file, "w") as f:
            json.dump(submission_dump, f) # Using indent for readability
            # print("this is the length of the submission data", len(submission_dump.items()))


        # eval_dataframe = pd.read_csv(eval_file_path)

        # eval_ids = list(eval_dataframe["input_id"])
        # eval_prompts = list(eval_dataframe["prompt"])
        # eval_labels = list(eval_dataframe["labels"])


        # for i in range(len(eval_prompts)):
        #     response = gemma_lm.generate(eval_prompts[i], max_length=10000, strip_prompt=True)
        #     output = last_list(response)
        #     # print("this is response\n\n", output, "\n\n")
        #     if output == eval_labels[i]:
        #         correct+=1
                
        # accuracy = correct/len(labels)

        # print("this is accuracy", accuracy)


if __name__ == "__main__":
    main()


import torch
import threading
from your_model import YourModel # Your pre-trained model class

def run_inference(rank, input_data, results):
    """Function to run on a specific GPU."""
    device = torch.device(f"cuda:{rank}")
    
    # 1. Load Model and move to device
    model = YourModel() 
    model.load_state_dict(torch.load("your_weights.pth"))
    model.to(device)
    model.eval()

    # 2. Run Inference
    with torch.no_grad():
        # Move inputs to the assigned device
        input_tensors = input_data.to(device) 
        output = model(input_tensors)
        results[rank] = output.cpu() # Collect results back on CPU

# --- Execution ---
# Load your full dataset/tasks (e.g., 256 samples)
full_input_data = torch.randn(256, 100) 

# Split the data
data_gpu0 = full_input_data[:128]
data_gpu1 = full_input_data[128:]

# Dictionary to hold the results from each thread/GPU
results = {}

# Create and start threads
thread0 = threading.Thread(target=run_inference, args=(0, data_gpu0, results))
thread1 = threading.Thread(target=run_inference, args=(1, data_gpu1, results))

thread0.start()
thread1.start()

# Wait for both threads to complete
thread0.join()
thread1.join()

# Combine the results
final_output = torch.cat([results[0], results[1]], dim=0)
print("Inference completed and results combined.")

