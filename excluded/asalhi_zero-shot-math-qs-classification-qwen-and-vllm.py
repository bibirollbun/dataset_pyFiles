!pip install "vllm[torch,flash-attn]"


from vllm import LLM, SamplingParams
import pandas as pd
import warnings
import re
import os

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm_model_pth = '/kaggle/input/qwen2.5/transformers/7b-instruct/1'

# Load model
llm = LLM(model=llm_model_pth,
          trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
          tensor_parallel_size=2,      # The number of GPUs to use for distributed execution with tensor parallelism
          dtype="half",
          gpu_memory_utilization=0.95,)

# Load test data
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.0,
                                 max_tokens=512)

# Build system+user messages for each test question
def build_chat_messages(question):
    return [
        {
            "role": "system",
            "content": (
                "Given a math problem written in natural language, your goal is to predict its correct topic "
                "from eight predefined categories. You are a helpful and harmless assistant. You are Qwen developed by Alibaba. "
                "You should think step-by-step. ٌeturn the final answer (topic id) within \\boxed{}.\n Do NOT SOLVE, only predict the topic"
                "Only choose a number from 0 to 7, each representing a topic label:\n"
                "0 - Algebra\n"
                "1 - Geometry and Trigonometry\n"
                "2 - Calculus and Analysis\n"
                "3 - Probability and Statistics\n"
                "4 - Number Theory\n"
                "5 - Combinatorics and Discrete Math\n"
                "6 - Linear Algebra\n"
                "7 - Abstract Algebra and Topology\n"
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

# Extract label from \boxed{n}
def extract_boxed_label(text):
    match = re.search(r"\\boxed\{(\d)\}", text)
    if match:
        label = int(match.group(1))
        return label if 0 <= label <= 7 else 0 #most common label is zero so we default to that if needed
    return 0

# Inference loop
results = []
for qid, question in zip(test_df["id"], test_df["Question"]):
    messages = build_chat_messages(question)
    response = llm.chat(messages, sampling_params)[0].outputs[0].text.strip()
    print("Res: ", response)
    label = extract_boxed_label(response)
    print("final: ",label)
    results.append((qid, label))

# Save predictions
output_df = pd.DataFrame(results, columns=["id", "label"])
output_df.to_csv("submission.csv", index=False)


