import openai

from kaggle_secrets import UserSecretsClient
api_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

def ask_llm(prompt, model="gpt-4.1-nano"):
    """Sends a prompt to the specified OpenAI model and returns the response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Adjust temperature for more or less creative responses
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"


# --- Example Question ---
question = "I have 5 apples and I buy 2 more. I then give 3 apples to my friend. How many apples do I have left?"

# --- 1. Standard Chain-of-Thought (CoT) Prompt ---
cot_prompt = f"""
Q: {question}
A: Let's think step by step.
"""

print("--- Standard CoT Prompt ---")
cot_response = ask_llm(cot_prompt)
print(f"Prompt:\n{cot_prompt}")
print(f"\nResponse:\n{cot_response}\n")
print("-" * 30)


# --- 2. Re-Reading (RE2) + CoT Prompt ---
re2_prompt = f"""
Q: {question}
Read the question again: {question}
A: Let's think step by step.
"""

print("--- RE2 + CoT Prompt ---")
re2_response = ask_llm(re2_prompt)
print(f"Prompt:\n{re2_prompt}")
print(f"\nResponse:\n{re2_response}\n")


# --- Samples from the Grade School Math 8K (GSM8K) Dataset ---
gsm8k_samples = [
    {
        "question": "Natalia sold 48 liters of milk in the morning and 32 liters of milk in the evening. She sold the milk for $4 per liter. How much money did she earn in total?",
        "ground_truth_answer": "Natalia sold 48 + 32 = 80 liters of milk in total. She earned 80 * 4 = $320. The final answer is 320."
    },
    {
        "question": "A clown had 45 balloons. He gave 15 to a group of children and then bought 10 more. How many balloons does the clown have now?",
        "ground_truth_answer": "The clown started with 45 balloons. He gave away 15, so he had 45 - 15 = 30 balloons. He then bought 10 more, so he now has 30 + 10 = 40 balloons. The final answer is 40."
    },
    {
        "question": "James is planting a garden. He has 3 rows of carrots and in each row, he plants 12 carrots. He also has 2 rows of tomatoes and in each row, he plants 8 tomatoes. How many vegetables did he plant in total?",
        "ground_truth_answer": "James planted 3 * 12 = 36 carrots. He planted 2 * 8 = 16 tomatoes. In total, he planted 36 + 16 = 52 vegetables. The final answer is 52."
    }
]

# --- Loop through each sample and test both prompting methods ---
for i, sample in enumerate(gsm8k_samples):
    question = sample["question"]
    ground_truth = sample["ground_truth_answer"]

    print(f"================== SAMPLE {i+1} ==================")
    print(f"QUESTION: {question}")
    print(f"GROUND TRUTH: {ground_truth}\n")

    # --- 1. Standard Chain-of-Thought (CoT) Prompt ---
    cot_prompt = f"Q: {question}\nA: Let's think step by step."
    print("--- Testing Standard CoT Prompt ---")
    cot_response = ask_llm(cot_prompt)
    print(f"Response:\n{cot_response}\n")

    # --- 2. Re-Reading (RE2) + CoT Prompt ---
    re2_prompt = f"Q: {question}\nRead the question again: {question}\nA: Let's think step by step."
    print("--- Testing RE2 + CoT Prompt ---")
    re2_response = ask_llm(re2_prompt)
    print(f"Response:\n{re2_response}\n")
    print("===========================================\n")


import openai
import pandas as pd
import random

# Securely fetch your OpenAI API key from Kaggle Secrets
from kaggle_secrets import UserSecretsClient
api_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

def ask_llm(prompt, model="gpt-4.1-mini"):
    """Sends a prompt to a powerful OpenAI model and returns the response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            # For Olympiad-level math, a low temperature is crucial for logical consistency
            temperature=0.0, 
            max_tokens=4096  # Increased tokens for very complex, multi-step solutions
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# --- Load the Challenging AIMO Dataset from Kaggle ---
try:
    # Path to the dataset after adding it to a Kaggle notebook
    dataset_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
    df = pd.read_csv(dataset_path)
    
    # Take a random sample of challenging problems to test
    num_samples = 1
    if len(df) >= num_samples:
        # Use a random seed for reproducibility if needed, or let it be random each time
        samples_df = df.sample(n=num_samples, random_state=random.randint(1, 1000))
    else:
        samples_df = df
        
except FileNotFoundError:
    print("Dataset not found. Please ensure you have added the 'ai-mathematical-olympiad-prize' dataset to your Kaggle notebook.")
    # Create a dummy DataFrame to prevent the rest of the code from crashing
    samples_df = pd.DataFrame(columns=['problem', 'answer'])


# --- Loop through each sample and test both prompting methods ---
for i, row in samples_df.iterrows():
    question = row['problem']
    # In this dataset, 'answer' is the final numerical result
    ground_truth_answer = row['answer']

    print(f"================== AIMO SAMPLE {i+1} ==================")
    print(f"QUESTION: {question}")
    print(f"GROUND TRUTH ANSWER (for verification): {ground_truth_answer}\n")

    # --- 1. Standard Chain-of-Thought (CoT) Prompt ---
    cot_prompt = f"Solve the following advanced mathematical problem. Think step by step and show all your reasoning before giving the final answer.\n\nProblem: {question}\n\nLet's think step by step:"
    print("--- Testing Standard CoT Prompt ---")
    cot_response = ask_llm(cot_prompt)
    print(f"Response:\n{cot_response}\n")

    # --- 2. Re-Reading (RE2) + CoT Prompt ---
    re2_prompt = f"Solve the following advanced mathematical problem. Think step by step and show all your reasoning before giving the final answer.\n\nProblem: {question}\nRead the problem again: {question}\n\nLet's think step by step:"
    print("--- Testing RE2 + CoT Prompt ---")
    re2_response = ask_llm(re2_prompt)
    print(f"Response:\n{re2_response}\n")
    print("======================================================\n")

