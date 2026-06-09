import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# トークナイザーとモデルのロード
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

# 入力テキストのトークン化
inputs = tokenizer("1+1=", return_tensors="pt")

# モデルによる推論とlogitsの取得
with torch.no_grad():
    outputs = model(**inputs, output_logits=True)
    logits = outputs.logits

# logitsのshapeを表示
print(logits.shape)

# 生成されたトークン列の長さを確認
print(inputs.input_ids.shape)


# 最後のトークンのlogitsを取得
last_token_logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)

# 最も確率の高いトークンIDを予測
predicted_token_id = torch.argmax(last_token_logits, dim=-1)

# トークンIDをテキストに変換
predicted_text = tokenizer.decode(predicted_token_id)

# 結果の表示
print(f"Predicted token ID: {predicted_token_id.item()}")
print(f"Predicted text: {predicted_text}")


import torch.nn.functional as F

# 最後のトークンのlogitsを取得
last_token_logits = outputs.logits[0, -1, :]  # [vocab_size]

# logitsから確率を計算
last_token_probs = F.softmax(last_token_logits, dim=-1)

# 確率の高い上位5つのトークンIDを取得
top_5_probs, top_5_ids = torch.topk(last_token_probs, 5)

# 対応するlogitsを取得
top_5_logits = last_token_logits[top_5_ids]

# トークンIDをテキストに変換
top_5_tokens = tokenizer.batch_decode(top_5_ids)

# 結果の表示
print("Top 5 predicted tokens:")
for i in range(5):
    print(f"Token: {top_5_tokens[i]}, Logit: {top_5_logits[i].item():.4f}, Probability: {top_5_probs[i].item():.4f}")



def get_top_probabilities_with_temperature(logits, temperature, top_k=5):
    """logitsに温度パラメータを適用し、確率上位k個のトークンとその確率を返す"""

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)
    return top_probs, top_ids

temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]  #温度パラメータのリスト

for temp in temperatures:
    top_probs, top_ids = get_top_probabilities_with_temperature(last_token_logits, temp)
    top_tokens = tokenizer.batch_decode(top_ids)

    print(f"\nTemperature: {temp}")
    for i in range(len(top_tokens)):
        print(f"Token: {top_tokens[i]}, Probability: {top_probs[i].item():.4f}")


# embedding層の確認

input_ids = tokenizer.encode("1+1=", return_tensors="pt")

embeddings = model.get_input_embeddings()
input_embeddings = embeddings(input_ids)
print("\n入力埋め込みの形状:", input_embeddings.shape)


data = {'QuestionId': 101,
 'ConstructId': 579,
 'ConstructName': 'Express one quantity as a percentage of another mentally',
 'SubjectId': 233,
 'SubjectName': 'Percentages of an Amount',
 'CorrectAnswer': 'B',
 'QuestionText': 'What is \\( 8 \\) out of \\( 40 \\) as a percentage?',
 'AnswerAText': '\\( 8.4 \\% \\)',
 'AnswerBText': '\\( 20 \\% \\)',
 'AnswerCText': '\\( 16 \\% \\)',
 'AnswerDText': '\\( 24 \\% \\)',
 'MisconceptionAId': 1786,
 'MisconceptionBId': -1,
 'MisconceptionCId': 658,
 'MisconceptionDId': -1,
 'MisconceptionAName': 'Converts a fraction to a percentage by writing the numerator followed by the denominator',
 'MisconceptionBName': None,
 'MisconceptionCName': 'Thinks they double the numerator to turn a fraction into a percentage',
 'MisconceptionDName': None}

# 演習データの準備
question_text = data['QuestionText']

# retrievalで本来得られる候補だが、簡単のために事前に定義しておく
misconceptions_candidates = [
    data['MisconceptionAName'],
    data['MisconceptionCName'],
    "Does not know that angles in a triangle sum to 180 degrees",
    "Uses dividing fractions method for multiplying fractions"
]

print("--- Step 1: 疑似 Retrieval 結果 (Re-ranking 対象候補) ---")
print(f"質問: {question_text}")
print("候補 Misconception:")
for i, mc in enumerate(misconceptions_candidates):
    print(f"  {i+1}. {mc}")
print("-" * 30)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys # For exiting script on critical errors

# --- Configuration ---
# Specify the model name (e.g., 'Qwen/Qwen1.5-1.8B-Chat')
# Ensure this model is available on Hugging Face Hub.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" # Example using Qwen1.5

# --- Helper Functions ---

def load_model_and_tokenizer(model_name: str) -> tuple:
    """Loads the tokenizer and model from Hugging Face Hub."""
    print(f"Attempting to load model: {model_name}")
    try:
        # Determine device (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",  # Use appropriate dtype (e.g., bfloat16)
            device_map="auto",  # Automatically map model to available device(s)
            trust_remote_code=True
        )
        print("Model and tokenizer loaded successfully.")
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}", file=sys.stderr)
        print("Please ensure the model name is correct, you have internet access, "
              "and required dependencies are installed.", file=sys.stderr)
        sys.exit(1) # Exit if model loading fails

def get_token_ids(tokenizer, words: list[str]) -> dict[str, int | None]:
    """Gets the token IDs for the given words."""
    token_ids = {}
    print("\nGetting token IDs...")
    for word in words:
        try:
            # Encode the word, ensuring not to add special tokens automatically
            # Taking the first token assuming the word itself is a single token
            token_id = tokenizer.encode(word, add_special_tokens=False)[0]
            token_ids[word] = token_id
            print(f"  Token ID for '{word}': {token_id}")
        except IndexError:
            token_ids[word] = None
            print(f"  Warning: Could not get a valid token ID for '{word}'.")
        except Exception as e:
            token_ids[word] = None
            print(f"  Error getting token ID for '{word}': {e}")
    # Critical check: Ensure we have IDs for core decision tokens ('Yes', 'No')
    if token_ids.get("Yes") is None or token_ids.get("No") is None:
         print("Error: Could not obtain essential token IDs for 'Yes' or 'No'. Exiting.", file=sys.stderr)
         sys.exit(1)
    return token_ids

# --- Main Logic Function ---

def get_yes_probability(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    question: str,
    wrong_answer: str,
    misconception: str,
    yes_token_id: int
) -> float:
    """
    Calculates the probability of the model predicting 'Yes' given the context.

    Args:
        tokenizer: The loaded tokenizer.
        model: The loaded language model.
        device: The device to run inference on ('cuda' or 'cpu').
        question: The question text.
        wrong_answer: The incorrect answer text.
        misconception: The potential misconception text.
        yes_token_id: The token ID for 'Yes'.

    Returns:
        The probability (0.0 to 1.0) of the 'Yes' token, or -1.0 on error.
    """
    # Construct the prompt in English
    prompt_text = f"""Question: {question}
Incorrect Answer: {wrong_answer}
Misconception: "{misconception}"
Is this misconception the likely cause of the incorrect answer? Respond Yes or No.
Response:""" # The model should predict 'Yes' or 'No' next

    # Prepare input for the model using the chat template if appropriate
    # (Qwen chat models often require specific formatting)
    messages = [
        # {"role": "system", "content": "You are an AI assistant evaluating reasoning errors."}, # Optional system prompt
        {"role": "user", "content": prompt_text}
    ]
    try:
        # Apply chat template for proper formatting
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Ensures the prompt ends correctly for generation
        )
        inputs = tokenizer([input_text], return_tensors="pt").to(device)
    except Exception as e:
        print(f"Error during tokenization or chat template application: {e}", file=sys.stderr)
        return -1.0

    # Perform inference
    with torch.no_grad(): # Disable gradient calculations for efficiency
        try:
            outputs = model(**inputs)
            logits = outputs.logits
        except Exception as e:
            print(f"Error during model inference: {e}", file=sys.stderr)
            return -1.0

    # Calculate probabilities for the next token
    # Get logits for the last token in the sequence
    last_token_logits = logits[0, -1, :]
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(last_token_logits, dim=-1)

    # Get the probability of the 'Yes' token
    try:
        yes_probability = probabilities[yes_token_id].item()
        return yes_probability
    except IndexError:
        print(f"Error: Token ID {yes_token_id} ('Yes') is out of bounds "
              f"for vocabulary size ({probabilities.shape[-1]}).", file=sys.stderr)
        return -1.0
    except Exception as e:
        print(f"Error retrieving 'Yes' probability: {e}", file=sys.stderr)
        return -1.0

# --- Main Execution Block ---

if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

    # Get necessary token IDs
    # Add other relevant words if needed (e.g., ' はい', ' いいえ' if checking Japanese too)
    target_words = ["Yes", "No"]
    token_ids = get_token_ids(tokenizer, target_words)
    yes_token_id = token_ids["Yes"]
    no_token_id = token_ids["No"] # Keep No token ID for potential comparison

    # --- Example Usage ---
    # Sample data (replace with your actual data source)
    data_sample = {
        'QuestionText': 'What is the capital of Japan?',
        'AnswerAText': 'Osaka',
        'MisconceptionAName': 'Confusing the second largest city with the capital.'
    }

    print("\n--- Calculating Probability ---")
    # Calculate the probability
    yes_prob = get_yes_probability(
        tokenizer,
        model,
        device,
        data['QuestionText'],
        data['AnswerAText'],
        data['MisconceptionAName'],
        yes_token_id
    )

    # --- Display Results ---
    print("\n--- Results ---")
    print(f"Question:       {data['QuestionText']}")
    print(f"Incorrect Answer: {data['AnswerAText']}")
    print(f"Misconception:  {data['MisconceptionAName']}")

    if yes_prob >= 0:
        print(f"\nProbability that the misconception caused the error ('Yes' probability): {yes_prob:.4f}")
        # Optionally, calculate and show 'No' probability for comparison
        # Note: Need probabilities from the get_yes_probability function or re-calculate
        # For simplicity, we only show 'Yes' here based on the function's return.
    else:
        print("\nFailed to calculate the probability due to an error.")



# Load model and tokenizer
tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

# Get necessary token IDs (crucially check "Yes" and "No")
target_words = ["Yes", "No"]
token_ids = get_token_ids(tokenizer, target_words)
yes_token_id = token_ids["Yes"]
# no_token_id = token_ids["No"] # Available if needed for comparison

# --- Evaluate Each Misconception Candidate ---
print("\n--- Evaluating Misconception Candidates ---")
results = {} # Store results {misconception: probability}

for i, mc_candidate in enumerate(misconceptions_candidates):
    print(f"\nProcessing Candidate {i+1}/{len(misconceptions_candidates)}: \"{mc_candidate}\"")

    # Calculate the probability for the current candidate
    yes_prob = get_yes_probability(
        tokenizer,
        model,
        device,
        data['QuestionText'],
        data['AnswerAText'],
        mc_candidate,           # Current misconception candidate
        yes_token_id
    )

    # Store and print the result for this candidate
    results[mc_candidate] = yes_prob
    if yes_prob >= 0:
        print(f"  -> 'Yes' Probability: {yes_prob:.4f}")
    else:
        print(f"  -> Failed to calculate probability.")

# --- Final Summary ---
print("\n--- Final Results Summary ---")
print(f"Question: {question_text}")
print(f"Incorrect Answer (Example): {data['AnswerAText']}")
print("\n'Yes' Probability for each Misconception Candidate:")
# Sort results by probability (descending) for potential re-ranking
sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

for misconception, probability in sorted_results:
    if probability >= 0:
        print(f"  {probability:.4f} : \"{misconception}\"")
    else:
        print(f"  [Error] : \"{misconception}\"")






import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from typing import List, Dict, Tuple, Optional # Added type hinting

# Prepare exercise data
question_text = data['QuestionText']
incorrect_answer_text = data['AnswerAText']

print("--- Input Data ---")
print(f"Question: {question_text}")
print(f"Incorrect Answer (Example): {incorrect_answer_text}")
print("\nMisconception Candidates:")
for i, mc in enumerate(misconceptions_candidates):
    print(f"  {i+1}. {mc}")
print("-" * 30)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# --- Helper Functions (Mostly reusable, minor adjustments needed) ---

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """Loads the tokenizer and model from Hugging Face Hub."""
    print(f"\nAttempting to load model: {model_name}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        print("Model and tokenizer loaded successfully.")
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model or tokenizer '{model_name}': {e}", file=sys.stderr)
        # Fallback mechanism if the primary model fails
        fallback_model = "Qwen/Qwen1.5-1.8B-Chat"
        if model_name != fallback_model:
             print(f"\nAttempting to load fallback model: {fallback_model}...")
             try:
                 # Retry loading with the fallback model name
                 return load_model_and_tokenizer(fallback_model)
             except Exception as fallback_e:
                 print(f"Error loading fallback model '{fallback_model}': {fallback_e}", file=sys.stderr)
                 print("Exiting due to model loading failure.", file=sys.stderr)
                 sys.exit(1)
        else:
             # If fallback itself failed or was the original model
             print("Exiting due to model loading failure.", file=sys.stderr)
             sys.exit(1)


def get_token_ids(tokenizer: AutoTokenizer, words: List[str]) -> Dict[str, Optional[int]]:
    """Gets the token IDs for the given words, trying with/without leading space."""
    token_ids: Dict[str, Optional[int]] = {}
    print("\nGetting token IDs...")
    for word in words:
        token_id = None
        try:
            # Try encoding directly first
            encoded = tokenizer.encode(word, add_special_tokens=False)
            if encoded:
                token_id = encoded[0]
            else:
                # If direct encoding fails or is empty, try with a leading space
                encoded_with_space = tokenizer.encode(" " + word, add_special_tokens=False)
                if encoded_with_space:
                    token_id = encoded_with_space[0]
                    print(f"  Note: Used ' {word}' for encoding.")

            if token_id is not None:
                 token_ids[word] = token_id
                 print(f"  Token ID for '{word}': {token_id}")
            else:
                 # If both attempts failed
                 token_ids[word] = None
                 print(f"  Warning: Could not get a valid token ID for '{word}' (tried both direct and with space).")

        except Exception as e:
            token_ids[word] = None
            print(f"  Error getting token ID for '{word}': {e}")

    # Critical check: Ensure we have IDs for all target choices
    missing_ids = [word for word in words if token_ids.get(word) is None]
    if missing_ids:
         print(f"Error: Could not obtain essential token IDs for: {', '.join(missing_ids)}. Exiting.", file=sys.stderr)
         sys.exit(1)

    return token_ids

# --- New Main Logic Function ---

def get_choice_probabilities(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    question: str,
    wrong_answer: str,
    misconceptions: List[str],
    choice_token_ids: Dict[str, int]
) -> Optional[Dict[str, float]]:
    """
    Calculates the probability of the model predicting each choice number.

    Args:
        tokenizer: The loaded tokenizer.
        model: The loaded language model.
        device: The device to run inference on ('cuda' or 'cpu').
        question: The question text.
        wrong_answer: The incorrect answer text.
        misconceptions: A list of misconception strings.
        choice_token_ids: A dictionary mapping choice strings (e.g., "1") to their token IDs.

    Returns:
        A dictionary mapping choice strings to their probabilities (0.0 to 1.0),
        or None if an error occurred.
    """
    # Construct the prompt with numbered choices
    prompt_choices = "\n".join([f"{i+1}. \"{mc}\"" for i, mc in enumerate(misconceptions)])

    prompt_text = f"""Context:
Question: {question}
Incorrect Answer: {wrong_answer}

Possible Misconceptions causing the incorrect answer:
{prompt_choices}

Instruction: Identify the number corresponding to the most likely misconception that led to the incorrect answer. Respond with only the number (1, 2, 3, or 4).
Most likely cause number:""" # Model predicts the number next

    messages = [{"role": "user", "content": prompt_text}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([input_text], return_tensors="pt").to(device)
    except Exception as e:
        print(f"Error during tokenization/template: {e}", file=sys.stderr)
        return None

    # Perform inference
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            logits = outputs.logits
        except Exception as e:
            print(f"Error during model inference: {e}", file=sys.stderr)
            return None

    # Calculate probabilities for the next token
    last_token_logits = logits[0, -1, :]
    probabilities = torch.softmax(last_token_logits, dim=-1)

    # Extract probabilities for the target choice tokens
    choice_probabilities: Dict[str, float] = {}
    vocab_size = probabilities.shape[-1]
    all_found = True
    for choice_str, token_id in choice_token_ids.items():
        if token_id is None: # Should have been caught by get_token_ids, but double-check
             print(f"Error: Token ID for choice '{choice_str}' is missing.", file=sys.stderr)
             all_found = False
             continue
        try:
            if 0 <= token_id < vocab_size:
                 choice_probabilities[choice_str] = probabilities[token_id].item()
            else:
                 print(f"Error: Token ID {token_id} for choice '{choice_str}' is out of bounds "
                       f"(Vocabulary size: {vocab_size}). Setting probability to 0.", file=sys.stderr)
                 choice_probabilities[choice_str] = 0.0
                 all_found = False # Treat as error if out of bounds
        except Exception as e:
            print(f"Error retrieving probability for choice '{choice_str}' (ID: {token_id}): {e}", file=sys.stderr)
            choice_probabilities[choice_str] = -1.0 # Indicate error for this choice
            all_found = False

    # Return probabilities only if all were successfully retrieved (or handled)
    return choice_probabilities if all_found else None


# Load model and tokenizer
# The load function now includes fallback logic
tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

# Define the target choices (numbers as strings)
target_choices = [str(i) for i in range(1, len(misconceptions_candidates) + 1)] # ["1", "2", "3", "4"]

# Get token IDs for the choices
choice_token_ids = get_token_ids(tokenizer, target_choices)

# --- Calculate Choice Probabilities ---
print("\n--- Calculating Choice Probabilities ---")

choice_probs = get_choice_probabilities(
    tokenizer,
    model,
    device,
    question_text,
    incorrect_answer_text,
    misconceptions_candidates,
    choice_token_ids # Pass the dictionary of choice string -> token ID
)

# --- Display Results ---
print("\n--- Results: Probabilities for Each Choice Number ---")
if choice_probs is not None:
    # Sort by probability descending for clarity
    sorted_probs = sorted(choice_probs.items(), key=lambda item: item[1], reverse=True)

    total_prob = sum(p for p in choice_probs.values() if p >= 0) # Sum of valid probabilities
    print(f"Choice Number | Probability | Misconception")
    print("-" * 60)
    for choice_num_str, probability in sorted_probs:
         choice_index = int(choice_num_str) - 1 # Convert "1" -> 0, "2" -> 1, etc.
         misconception_text = misconceptions_candidates[choice_index]
         if probability >= 0:
             print(f"      {choice_num_str}       |  {probability:.4f}     | \"{misconception_text}\"")
         else:
             print(f"      {choice_num_str}       |  [Error]    | \"{misconception_text}\"")
    print("-" * 60)
    print(f"Sum of probabilities for choices {list(choice_probs.keys())}: {total_prob:.4f}")
    # Note: This sum might not be 1.0, as it only includes the specific target tokens.
else:
    print("\nFailed to calculate choice probabilities due to an error during processing.")






import pandas as pd
import numpy as np

df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')

# データ数を取得
data_num = len(df)

# train と test の分割比率を指定
train_rate = 0.8
train_num = int(data_num * train_rate)

# index を分割
np.random.seed(42)
train_index = np.random.choice(data_num, train_num, replace=False)
valid_index = list(set(range(data_num)) - set(train_index))

# train と test に分割
train_df = df.iloc[train_index]
valid_df = df.iloc[valid_index]

print("train_df の形状:", train_df.shape)
print("valid_df の形状:", valid_df.shape)


def preprocess(df):    
    result = []
    for i, row in df.iterrows():
        for option in ['A', 'B', 'C', 'D']:
            if pd.isnull(row[f'Misconception{option}Id']):
                continue
            result.append(
                {
                    'ConstructId': row['ConstructId'],
                    'ConstructName': row['ConstructName'],
                    'SubjectId': row['SubjectId'],
                    'SubjectName': row['SubjectName'],
                    'CorrectAnswer': row['CorrectAnswer'],
                    'IsCorrect': row['CorrectAnswer']==option,
                    'Option': option,
                    'AnswerText': row[f'Answer{option}Text'],
                    'MisconceptionId': int(row[f'Misconception{option}Id']),
                }
            )
    df = pd.DataFrame(result)

    misconception_mapping_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')
    df = df.merge(
        misconception_mapping_df,
        on="MisconceptionId",
        how="left"
    )

    return df

train_df = preprocess(train_df)
valid_df = preprocess(valid_df)

print("train_df の形状:", train_df.shape)
display(train_df.head())


print("valid_df の形状:", valid_df.shape)
display(train_df.head())


training_data = [    
    # --- 分数 ---
    {
        'QuestionText': 'Calculate \\( \\frac{1}{2} + \\frac{1}{3} \\)',
        'AnswerText': '\\( \\frac{2}{5} \\)', # 分子同士、分母同士を足す間違い
        'MisconceptionName': 'Adds fractions by adding numerators and denominators separately',
        'IsCorrect': True
    },
    {
        'QuestionText': 'Calculate \\( \\frac{1}{2} + \\frac{1}{3} \\)',
        'AnswerText': '\\( \\frac{1}{5} \\)', # 何か別の計算間違い
        'MisconceptionName': 'Adds fractions by adding numerators and denominators separately',
        'IsCorrect': False # 上記の特定のMisconceptionとは異なる間違い
    },
    # --- 小数 ---
    {
        'QuestionText': 'What is \\( 0.5 \\times 0.2 \\)?',
        'AnswerText': '\\( 1.0 \\)', # 小数点の位置間違い
        'MisconceptionName': 'Misplaces the decimal point in multiplication of decimals',
        'IsCorrect': True
    },
    {
        'QuestionText': 'What is \\( 0.5 \\times 0.2 \\)?',
        'AnswerText': '\\( 0.7 \\)', # 掛け算を足し算と間違える
        'MisconceptionName': 'Confuses multiplication operation with addition',
        'IsCorrect': True
     },
    # --- 簡単な方程式 ---
    {
        'QuestionText': 'Solve for x: \\( 2x + 3 = 11 \\)',
        'AnswerText': '\\( x = 7 \\)', # 移項時に符号を変え忘れる (11+3)/2
        'MisconceptionName': 'Forgets to change the sign of a term when moving it across the equals sign',
        'IsCorrect': True
    },
    {
        'QuestionText': 'Solve for x: \\( 2x + 3 = 11 \\)',
        'AnswerText': '\\( x = 5.5 \\)', # 最初に2で割ってしまう 3を引くのを忘れる
        'MisconceptionName': 'Performs operations in the wrong order when solving equations',
        'IsCorrect': True # 操作順序の間違い
    },
    # --- 割合 ---
    {
        'QuestionText': 'What is 20% off a 500 yen item?',
        'AnswerText': '\\( 100 \\text{ yen} \\)', # 割引額を答えてしまう
        'MisconceptionName': 'Calculates the discount amount instead of the final price after discount',
        'IsCorrect': True
    },
    {
        'QuestionText': 'What is 20% off a 500 yen item?',
        'AnswerText': '\\( 480 \\text{ yen} \\)', # 20%を20円と勘違いして引く
        'MisconceptionName': 'Subtracts the percentage value directly as a fixed amount',
        'IsCorrect': True
    },
]

prompt_template = """Determine if the provided misconception is relevant to the student's answer for the given question. Respond with only 'Yes' or 'No'.

Question: {QuestionText}
Student's Answer: {AnswerText}
Misconception Candidate: {MisconceptionName}

Is the misconception relevant? (Yes/No)
Answer:{Answer}"""


# 必要なライブラリのインストール
!pip install trl -q


import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling, # TRLがない場合の代替として使うこともあるが、今回はtrlを使う
)
# DataCollatorForCompletionOnlyLM は trl ライブラリに含まれます
# pip install trl transformers datasets accelerate bitsandbytes torch
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    print("TRL library not found. Please install it: pip install trl")
    # trl がない場合は、代替手段を考えるか、インストールを促して終了する
    exit()

# --- 1. データ準備 ---
training_data_list = [
    # --- 分数 ---
    {
        'QuestionText': 'Calculate \\( \\frac{1}{2} + \\frac{1}{3} \\)',
        'AnswerText': '\\( \\frac{2}{5} \\)', # 分子同士、分母同士を足す間違い
        'MisconceptionName': 'Adds fractions by adding numerators and denominators separately',
        'IsCorrect': True
    },
    {
        'QuestionText': 'Calculate \\( \\frac{1}{2} + \\frac{1}{3} \\)',
        'AnswerText': '\\( \\frac{1}{5} \\)', # 何か別の計算間違い
        'MisconceptionName': 'Adds fractions by adding numerators and denominators separately',
        'IsCorrect': False # 上記の特定のMisconceptionとは異なる間違い
    },
    # --- 小数 ---
    {
        'QuestionText': 'What is \\( 0.5 \\times 0.2 \\)?',
        'AnswerText': '\\( 1.0 \\)', # 小数点の位置間違い
        'MisconceptionName': 'Misplaces the decimal point in multiplication of decimals',
        'IsCorrect': True
    },
    {
        'QuestionText': 'What is \\( 0.5 \\times 0.2 \\)?',
        'AnswerText': '\\( 0.7 \\)', # 掛け算を足し算と間違える
        'MisconceptionName': 'Confuses multiplication operation with addition',
        'IsCorrect': True
     },
    # --- 簡単な方程式 ---
    {
        'QuestionText': 'Solve for x: \\( 2x + 3 = 11 \\)',
        'AnswerText': '\\( x = 7 \\)', # 移項時に符号を変え忘れる (11+3)/2
        'MisconceptionName': 'Forgets to change the sign of a term when moving it across the equals sign',
        'IsCorrect': True
    },
    {
        'QuestionText': 'Solve for x: \\( 2x + 3 = 11 \\)',
        'AnswerText': '\\( x = 5.5 \\)', # 最初に2で割ってしまう 3を引くのを忘れる
        'MisconceptionName': 'Performs operations in the wrong order when solving equations',
        'IsCorrect': True # 操作順序の間違い
    },
    # --- 割合 ---
    {
        'QuestionText': 'What is 20% off a 500 yen item?',
        'AnswerText': '\\( 100 \\text{ yen} \\)', # 割引額を答えてしまう
        'MisconceptionName': 'Calculates the discount amount instead of the final price after discount',
        'IsCorrect': True
    },
    {
        'QuestionText': 'What is 20% off a 500 yen item?',
        'AnswerText': '\\( 480 \\text{ yen} \\)', # 20%を20円と勘違いして引く
        'MisconceptionName': 'Subtracts the percentage value directly as a fixed amount',
        'IsCorrect': True
    },
]

prompt_template = """Determine if the provided misconception is relevant to the student's answer for the given question. Respond with only 'Yes' or 'No'.

Question: {QuestionText}
Student's Answer: {AnswerText}
Misconception Candidate: {MisconceptionName}

Is the misconception relevant? (Yes/No)
Answer:{Answer}"""

# プロンプト形式にデータを整形
formatted_data = []
for item in training_data_list:
    answer = "Yes" if item['IsCorrect'] else "No"
    formatted_text = prompt_template.format(
        QuestionText=item['QuestionText'],
        AnswerText=item['AnswerText'],
        MisconceptionName=item['MisconceptionName'],
        Answer=answer
    )
    formatted_data.append({"text": formatted_text})

# Hugging Face Datasetを作成
dataset = Dataset.from_list(formatted_data)

# --- 2. モデルとトークナイザーのロード ---
model_name = "Qwen/Qwen2.5-0.5B-Instruct" # Qwen2.5はまだリリースされていない可能性 or Qwen1.5を意図している可能性を考慮
output_dir = "./qwen-finetuned-answer" # ファインチューニング後のモデル保存先

# 利用可能なデバイスを選択 (GPUがあればGPU、なければCPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Qwenモデルは通常pad_tokenが設定されていないため、追加する
# EOSトークンを使用するのが一般的
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    # 量子化などが必要な場合はここに追加 (例: load_in_8bit=True)
).to(device) # モデルをデバイスに移動

# --- 3. データセットのトークン化 ---
def tokenize_function(examples):
    # `tokenizer`はテキストをトークンIDのリストに変換します
    # truncation=True で長すぎるシーケンスを切り捨てます
    # padding=False はここでは不要（DataCollatorが行うため）
    return tokenizer(examples["text"], truncation=True, max_length=512) # 必要に応じてmax_length調整

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# --- 4. DataCollatorの設定 ---
# "Answer:" 以降のトークンのみを学習対象とする DataCollator
# response_templateはリスト形式でIDを指定する必要がある
response_template_ids = tokenizer.encode("Answer:", add_special_tokens=False)
print(f"Response template 'Answer:' encoded as: {response_template_ids}")

# TRLのDataCollatorForCompletionOnlyLMを使用
# response_template には Answer: の token id を指定
# instruction_template は Answer: より前の部分ですが、今回は response_template だけ指定すれば十分です
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
    mlm=False # マスク言語モデリングではなく、因果言語モデリングを行う
)

# --- 5. トレーニング設定 ---
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,           # エポック数を1に設定
    per_device_train_batch_size=2, # メモリに応じて調整
    gradient_accumulation_steps=4, # メモリに応じて調整 (batch_size * accumulation_steps が実効バッチサイズ)
    learning_rate=2e-5,           # 学習率
    logging_dir='./logs',         # ログの保存先
    logging_steps=10,             # ログ出力の頻度
    save_strategy="epoch",        # エポックごとにモデルを保存
    fp16=torch.cuda.is_available(), # GPUが利用可能なら半精度浮動小数点数を使用
    # bf16=True, # Ampere以降のGPUで利用可能ならbf16の方が安定しやすい
    report_to="none" # wandbなどのレポートツールを使わない場合
)

# --- 6. Trainerの設定とトレーニング実行 ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=collator, # ここで設定したDataCollatorを使用
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning finished.")

# --- 7. モデルの保存 ---
print(f"Saving fine-tuned model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir) # トークナイザーも一緒に保存
print("Model saved successfully.")

# --- (オプション) ファインチューニング後のモデルで推論テスト ---
print("\nTesting the fine-tuned model:")
test_prompt_text = """Determine if the provided misconception is relevant to the student's answer for the given question. Respond with only 'Yes' or 'No'.

Question: Calculate \\( \\frac{1}{2} + \\frac{1}{3} \\)
Student's Answer: \\( \\frac{2}{5} \\)
Misconception Candidate: Adds fractions by adding numerators and denominators separately

Is the misconception relevant? (Yes/No)
Answer:"""

# プロンプトをトークン化
inputs = tokenizer(test_prompt_text, return_tensors="pt").to(device) # プロンプトをデバイスに移動

# モデルで出力を生成
# max_new_tokens を小さくして 'Yes'/'No' だけ生成するようにする
outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)

# 生成されたトークンをデコード
# skip_special_tokens=True で特殊トークンを除外
# clean_up_tokenization_spaces=True でスペースを整理
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\nPrompt:\n{test_prompt_text}")
print(f"\nGenerated Output:\n{generated_text}")

# Answer: 以降の部分だけを取得してみる
answer_part = generated_text.split("Answer:")[-1].strip()
print(f"\nExtracted Answer: {answer_part}")

test_prompt_text_no = """Determine if the provided misconception is relevant to the student's answer for the given question. Respond with only 'Yes' or 'No'.

Question: Calculate \\( \\frac{1}{2} + \\frac{1}{3} \\)
Student's Answer: \\( \\frac{1}{5} \\)
Misconception Candidate: Adds fractions by adding numerators and denominators separately

Is the misconception relevant? (Yes/No)
Answer:"""
inputs = tokenizer(test_prompt_text_no, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nPrompt:\n{test_prompt_text_no}")
print(f"\nGenerated Output:\n{generated_text}")
answer_part = generated_text.split("Answer:")[-1].strip()
print(f"\nExtracted Answer: {answer_part}")


data = {'QuestionId': 101,
 'ConstructId': 579,
 'ConstructName': 'Express one quantity as a percentage of another mentally',
 'SubjectId': 233,
 'SubjectName': 'Percentages of an Amount',
 'CorrectAnswer': 'B',
 'QuestionText': 'What is \\( 8 \\) out of \\( 40 \\) as a percentage?',
 'AnswerAText': '\\( 8.4 \\% \\)',
 'AnswerBText': '\\( 20 \\% \\)',
 'AnswerCText': '\\( 16 \\% \\)',
 'AnswerDText': '\\( 24 \\% \\)',
 'MisconceptionAId': 1786,
 'MisconceptionBId': -1,
 'MisconceptionCId': 658,
 'MisconceptionDId': -1,
 'MisconceptionAName': 'Converts a fraction to a percentage by writing the numerator followed by the denominator',
 'MisconceptionBName': None,
 'MisconceptionCName': 'Thinks they double the numerator to turn a fraction into a percentage',
 'MisconceptionDName': None}

# 演習データの準備
question_text = data['QuestionText']

# retrievalで本来得られる候補だが、簡単のために事前に定義しておく
misconceptions_candidates = [
    data['MisconceptionAName'],
    data['MisconceptionCName'],
    "Does not know that angles in a triangle sum to 180 degrees",
    "Uses dividing fractions method for multiplying fractions"
]

print("--- Step 1: 疑似 Retrieval 結果 (Re-ranking 対象候補) ---")
print(f"質問: {question_text}")
print("候補 Misconception:")
for i, mc in enumerate(misconceptions_candidates):
    print(f"  {i+1}. {mc}")
print("-" * 30)




