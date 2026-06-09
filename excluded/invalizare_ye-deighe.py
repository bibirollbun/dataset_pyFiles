
!pip install -q google-generativeai tqdm



import os
import pandas as pd
import google.generativeai as genai
import re
import time
from collections import Counter




test_df = pd.read_csv('/kaggle/input/prompt-engineering-math/test_with_translation.csv')
print(f"Loaded {len(test_df)} problems")




#  MetisAI API
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
metis_api_key = user_secrets.get_secret("METIS_API")


from google.api_core.client_options import ClientOptions



import google.generativeai as genai
from google.api_core.client_options import ClientOptions

genai.configure(api_key=metis_api_key, transport='rest',
                client_options=ClientOptions(api_endpoint="https://api.tapsage.com"))

model = genai.GenerativeModel("gemini-2.5-pro")
# response = model.generate_content("Explain Gemini to me like I am a kid")
# print(response.text, end="")

# ++++++++++++++++++++++++++++++++++++++++++++++
# genai.configure(api_key=metis_api_key, transport='rest',
#                 client_options=ClientOptions(api_endpoint="https://api.tapsage.com"))


# # Create model with custom base_url
# model = genai.GenerativeModel(
#     "gemini-2.5-pro",
#     generation_config={
#         "temperature": 0.2,
#         "max_output_tokens": 2048,
#     }
# )
 
 
SYSTEM_INSTRUCTION = """You are an elite mathematics expert specializing in solving mathematic problems with perfect accuracy.

CRITICAL RULES:
1. Read carefully - problems may have TeX notation ($, \\frac, \\sqrt, \\cdot, etc.)
2. Think step-by-step with explicit reasoning
3. Double-check ALL arithmetic calculations
4. Verify answer makes logical sense
5. Final answer must be ONE NUMBER ONLY (integer or decimal with dot separator)

SOLVING STRATEGY:
1. Identify problem type (algebra, geometry, probability, calculus, etc.)
2. Extract given information
3. Determine what to find
4. Apply appropriate formulas/methods
5. Calculate carefully (show work)
6. Verify answer reasonableness
7. Format: \\boxed{final_number}

COMMON PITFALLS TO AVOID:
- Decimal comma vs dot (use dot: 0.5 not 0,5)
- Order of operations errors
- Sign errors in algebra
- Unit conversions
- Rounding too early
- Misreading problem constraints"""



 
def build_prompt(translation):
    """Build comprehensive prompt with few-shot examples"""
    return f"""{SYSTEM_INSTRUCTION}

EXAMPLE 1:
Problem: Find the value of the expression $4.8 \\cdot 2.5$
Solution:
Type: Arithmetic multiplication
Given: 4.8 and 2.5
Find: Product
Step 1: Multiply 4.8 × 2.5
Step 2: 4.8 × 2.5 = 12.0
Verification: 12 ÷ 2.5 = 4.8 ✓
\\boxed{{12}}

EXAMPLE 2:
Problem: Solve equation: $x^2 - 5x + 6 = 0$
Solution:
Type: Quadratic equation
Given: $x^2 - 5x + 6 = 0$
Find: Value of x
Step 1: Factor $(x-2)(x-3) = 0$
Step 2: Solutions are x = 2 or x = 3
Step 3: If asking for one root, typically give the smaller positive: 2
Verification: $2^2 - 5(2) + 6 = 4 - 10 + 6 = 0$ ✓
\\boxed{{2}}

EXAMPLE 3:
Problem: A right triangle has legs 3 and 4. Find hypotenuse.
Solution:
Type: Geometry - Pythagorean theorem
Given: legs a=3, b=4
Find: hypotenuse c
Step 1: Apply $c^2 = a^2 + b^2$
Step 2: $c^2 = 3^2 + 4^2 = 9 + 16 = 25$
Step 3: $c = \\sqrt{{25}} = 5$
Verification: $3^2 + 4^2 = 9 + 16 = 25 = 5^2$ ✓
\\boxed{{5}}

EXAMPLE 4:
Problem: What is the probability of rolling a 6 on a fair die?
Solution:
Type: Probability
Given: Fair 6-sided die
Find: P(rolling 6)
Step 1: Total outcomes = 6
Step 2: Favorable outcomes = 1
Step 3: P = 1/6 ≈ 0.166666...
Step 4: Express as decimal: 0.166667 or exact fraction
Verification: 0 ≤ 0.166667 ≤ 1 ✓
\\boxed{{0.166667}}

NOW SOLVE THIS PROBLEM:
{translation}

Follow the exact format above:
- State problem type
- List given information
- Show complete step-by-step solution
- Verify your answer
- Provide final answer in \\boxed{{number}}

Remember: Answer must be a single number (integer or finite decimal with dot separator)."""




def extract_answer(text):
    """Extract numerical answer with multiple fallback methods"""
    text = str(text)
    
    # Method 1: Find \\boxed{number}
    patterns = [
        r'\\boxed\{([+-]?\d+\.?\d*)\}',
        r'\\boxed\{([+-]?\d*\.\d+)\}',
        r'boxed\{([+-]?\d+\.?\d*)\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(',', '.')
    
    # Method 2: Look for "Final answer:" or similar
    final_patterns = [
        r'[Ff]inal answer[:\s]+([+-]?\d+\.?\d*)',
        r'[Aa]nswer[:\s]+([+-]?\d+\.?\d*)',
        r'[Rr]esult[:\s]+([+-]?\d+\.?\d*)',
    ]
    
    for pattern in final_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(',', '.')
    
    # Method 3: Find last number in text
    numbers = re.findall(r'[+-]?\d+\.?\d*', text)
    if numbers:
        # Filter out years, IDs, etc (numbers > 10000 likely not answers)
        valid_numbers = [n for n in numbers if n and float(n) < 1000000]
        if valid_numbers:
            return valid_numbers[-1].replace(',', '.')
    
    return "0"




def get_answer_with_consensus(translation, n_samples=3):
    """Generate multiple answers and use majority voting"""
    answers = []
    prompt = build_prompt(translation)
    
    for attempt in range(n_samples):
        try:
            response = model.generate_content(prompt)
            answer = extract_answer(response.text)
            answers.append(answer)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            continue
    
    if not answers:
        return "0"
    
    # Return most common answer
    answer_counts = Counter(answers)
    most_common = answer_counts.most_common(1)[0]
    
    # If no clear majority and we have different answers, return first valid one
    if len(answer_counts) > 1 and most_common[1] == 1:
        return answers[0]
    
    return most_common[0]

# Generate answers for all problems
print("\nGenerating answers...")
answers = []

for idx, row in test_df.iterrows():
    translation = row['translation']
    
    # Use consensus with 3 attempts for better accuracy
    answer = get_answer_with_consensus(translation, n_samples=3)
    answers.append(answer)
    
    print(f"Problem {idx+1}/100 (ID: {row['problem_id']}): {answer}")
    
    # Rate limiting to avoid API throttling
    time.sleep(0.3)

# Add answers to dataframe
test_df['answer'] = answers



 
submission_df = test_df[['problem_id', 'answer']].copy()

# Clean answers: ensure proper format
submission_df['answer'] = submission_df['answer'].astype(str).str.strip()
submission_df['answer'] = submission_df['answer'].str.replace(',', '.')

# Handle edge cases
problematic = submission_df[submission_df['answer'].isin(['', '.', 'nan', 'None'])]
if len(problematic) > 0:
    print(f"\nWarning: Found {len(problematic)} problematic answers, replacing with 0")
    submission_df.loc[problematic.index, 'answer'] = '0'

# Save submission
submission_df.to_csv('submission.csv', index=False)

print(f"\n✓ Submission file created with {len(submission_df)} rows!")
print("\nFirst 10 submissions:")
print(submission_df.head(10))
print("\nLast 5 submissions:")
print(submission_df.tail()) 




