from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os

import warnings
warnings.filterwarnings('ignore')



# Define paths to the datasets
training_solutions_path = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json' if os.path.exists('/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json') else 'arc-agi_training_solutions.json'  
evaluation_solutions_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json' if os.path.exists('/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json') else 'arc-agi_evaluation_solutions.json' 
evaluation_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json' if os.path.exists('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json') else 'arc-agi_evaluation_challenges.json'
sample_submission_path = '/kaggle/input/arc-prize-2025/sample_submission.json' if os.path.exists('/kaggle/input/arc-prize-2025/sample_submission.json') else 'sample_submission.json' 
training_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json' if os.path.exists('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json') else 'arc-agi_training_challenges.json'
test_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json' if os.path.exists('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json') else 'arc-agi_test_challenges.json'



# Load the training data
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

training_solutions = load_json(training_solutions_path)
evaluation_solutions = load_json(evaluation_solutions_path)
evaluation_challenges = load_json(evaluation_challenges_path)
sample_submission = load_json(sample_submission_path)
training_challenges = load_json(training_challenges_path)
test_challenges = load_json(test_challenges_path)


# Model loading and setup
try:
    import kagglehub
    model_name = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")
    print(f"Model downloaded from Kaggle Hub: {model_name}")
except ImportError:
    print("Kaggle Hub not available.  Make sure you have kaggle installed and are in a kaggle environment with internet access.")
    model_name = "Qwen/Qwen-7B"  # or any other appropriate Qwen model you have access to
except Exception as e:
     print(f"Error downloading from Kaggle Hub: {e}.  Using a local or alternative model.")
     model_name = "Qwen/Qwen-7B" # Replace this if needed to a suitable model
     print(f"Trying model {model_name}")




# Check for CUDA availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) #trust_remote_code needed for some models
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    model.eval() # Set model to evaluation mode

    if (device == "cuda"):
        model = model.to(device) # Move the model to the GPU
except Exception as e:
    print(f"Error loading model: {e}")
    raise  # Re-raise the exception so the program stops if the model cannot be loaded.



class QwenChatbot:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.history = []

    def generate_response(self, user_input, enable_thinking=True, max_tokens=512):  # Added max_tokens
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)  # Move input to the same device as the model

        with torch.no_grad():  # Disable gradient calculation for inference
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens  # Use max_tokens here
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        #parsing thinking content
        try:
            # rindex finding 151668 ()
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return thinking_content, response



# --- Demonstration and ARC Task Integration ---

# Initialize chatbot
chatbot = QwenChatbot(model, tokenizer)

# Example Usage with the provided test cases.
if __name__ == "__main__":

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    thinking_content, response_1 = chatbot.generate_response(user_input_1)
    print(f"Thinking: {thinking_content}")
    print(f"Bot: {response_1}")
    print("----------------------")

    # Second input with /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    thinking_content, response_2 = chatbot.generate_response(user_input_2, enable_thinking=False) # explicitly disable thinking for the demo
    print(f"Thinking: {thinking_content}") # should be empty
    print(f"Bot: {response_2}")
    print("----------------------")

    # Third input with /think
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    thinking_content, response_3 = chatbot.generate_response(user_input_3, enable_thinking=True)
    print(f"Thinking: {thinking_content}")
    print(f"Bot: {response_3}")

    print("----------------------")


    


# --- Example of integrating with the ARC dataset ---
# Let's try to answer one of the training challenges.
sample_challenge_id = list(training_challenges.keys())[0]  # Get the first challenge ID
challenge = training_challenges[sample_challenge_id]

# Construct a prompt that describes the task and provides examples
prompt = f"Solve the following abstract reasoning challenge.  Here's the challenge:\n\n"
for i, task in enumerate(challenge['train']): # Use examples from the training set to build the prompt
     prompt += f"Input {i+1}:\n"
     prompt += str(task['input']) + "\n"
     prompt += f"Output {i+1}:\n"
     prompt += str(task['output']) + "\n\n"

# Add the test input.  Tell the model to predict this output
prompt += "Now, predict the output for the following test input:\n"
prompt += str(challenge['test'][0]['input']) + "\n"
prompt += "Output:\n"  # Ask the model to generate the output

# Generate the response
print("----------------------")
print("Solving ARC Challenge:")
print(f"Challenge ID: {sample_challenge_id}")
thinking_content, arc_response = chatbot.generate_response(prompt)
print(f"Thinking: {thinking_content}")
print(f"Model's Prediction:\n{arc_response}")
print("----------------------")




