import pandas as pd
import re
from openai import OpenAI
from tqdm.notebook import tqdm


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
OPENAI_API_KEY = user_secrets.get_secret("api_key")


client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.avalai.ir/v1"
)

MODEL_NAME = "cf.llama-3.3-70b-instruct-fp8-fast"


test_df = pd.read_csv("/kaggle/input/prompt-engineering-math/test_with_translation.csv")
test_df.head()


test_df['translation'][10]


SYSTEM_PROMPT = """You are a world-class Mathematical Expert with perfect logic. 
Your goal is to solve math problems with absolute precision.

Rules:
1. Logical Reasoning: Solve the problem step-by-step.
2. Double Check: Before concluding, re-calculate the final step.
3. Formatting: Your final answer must be ONLY a number (integer or decimal).
4. No Trailing Zeros: If the answer is a whole number like 12.0, you MUST write it as 12.
5. Delimiter: You must end your response with: "Final Answer: [number]"

Example 0:
Problem: If x + 5 = 17, what is x?
Step-by-step: 
- Subtract 5 from both sides.
- 17 - 5 = 12.
- The result is an integer.
Final Answer: 12


Example 1:
Problem: "Calculate 2^10 - 512"
Step 1: Calculate 2^10 = 1024
Step 2: Subtract: 1024 - 512 = 512
Step 3: Verify: 512 + 512 = 1024 ✓
Final Answer: 512

Example 2:
Problem: "A rectangle has perimeter 36 and length is twice the width. Find the area."
Step 1: Let width = w, then length = 2w
Step 2: Perimeter formula: 2(w + 2w) = 36
Step 3: Simplify: 6w = 36, so w = 6
Step 4: Length = 2(6) = 12
Step 5: Area = 6 × 12 = 72
Step 6: Verify perimeter: 2(6 + 12) = 36 ✓
Final Answer: 72

Example 3:
Problem: "Find 15% of 80"
Step 1: Convert to decimal: 15% = 0.15
Step 2: Multiply: 0.15 × 80 = 12
Step 3: Verify: 12/80 = 0.15 = 15% ✓
Final Answer: 12

Example 4:
Problem: "How many integers from 1 to 100 are divisible by both 3 and 5?"
Step 1: Numbers divisible by both 3 and 5 are divisible by LCM(3,5) = 15
Step 2: Count multiples of 15 from 1 to 100: 15, 30, 45, 60, 75, 90
Step 3: Count them: 6 numbers
Step 4: Verify: 100 ÷ 15 = 6.67, so floor(6.67) = 6 ✓
Final Answer: 6

"""


def generate_user_prompt(problem_text):
    return f"Problem: {problem_text}\n\nSolve it step-by-step and provide the final answer in the required format."


def clean_final_answer(raw_text):
    try:
        if "Final Answer:" in raw_text:
            raw_val = raw_text.split("Final Answer:")[-1].strip()
        else:
            raw_val = re.findall(r"[-+]?\d*\.?\d+", raw_text)[-1]
        clean_val = re.sub(r"[^\d\.-]", "", raw_val)
        
        num = float(clean_val)
        
        if num == int(num):
            return str(int(num))
        else:
            return str(num)
    except:
        return "0"

results = []


for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    problem_id = row['problem_id']
    problem_text = row['translation']
    
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": generate_user_prompt(problem_text)}
            ],
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=1000
        )
        
        raw_output = completion.choices[0].message.content
        final_ans = clean_final_answer(raw_output)
        
        results.append({
            "problem_id": problem_id,
            "answer": final_ans
        })
        
    except Exception as e:
        print(f"Error on ID {problem_id}: {e}")
        results.append({"problem_id": problem_id, "answer": "0"})
        time.sleep(1)

submission = pd.DataFrame(results)
submission['answer'] = submission['answer'].astype(str).str.strip()

submission.to_csv("submission.csv", index=False)
print(submission.head(10))




