!pip install vllm
!pip install logits-processor-zoo==0.1.10
!pip install triton==3.2.0
print('Installed!!')


MODEL_NAME = "/kaggle/input/qwen2-5-32b-instruct-gptq-int4"
# MODEL_NAME = "/kaggle/input/qwen2.5/transformers/32b-instruct-gptq-int4/1"
# MODEL_NAME = "/kaggle/input/qwen2.5/transformers/72b-instruct-gptq-int4/1"
LORA_PATH = "/kaggle/input/jigsaw-exp003-fold0/trained_model"


import os
os.environ["VLLM_USE_V1"] = "0"
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
import argparse
from scipy.special import softmax
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


llm = vllm.LLM(
    MODEL_NAME,
    # quantization='awq',
    quantization='gptq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.95,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=4096,
    disable_log_stats=True,
    enable_prefix_caching=True,
    enable_lora=True,
)
print('above cell done')


# tokenizer = llm.get_tokenizer()
# SYS_PROMPT = """
# ### Role: Expert Rule Violation Classifier
# You're a specialized AI trained on 500M+ moderated Reddit comments with subreddit-specific rule interpretation capabilities. Your task is to predict violation probabilities using this analytical framework:

# 1. **RULE DECONSTRUCTION PROTOCOL**  
#    - Identify core prohibition: "What behavior does this rule fundamentally restrict?"  
#    - Map to violation taxonomy:  
#      • Safety (threats, harassment)  
#      • Integrity (misinfo, spam)  
#      • Community (off-topic, tone)  
#    - Determine severity thresholds: What constitutes minor vs. major violation?

# 2. **COMMENT FORENSIC ANALYSIS**  
#    a) Literal Meaning Scan:  
#       - Keyword/phrase detection  
#       - Explicit violation markers  
#    b) Contextual Interpretation:  
#       - Sarcasm/irony indicators (tone markers, hyperbole, "/s")  
#       - Historical patterns in subreddit  
#       - Cultural/demographic context of r/<subreddit>  
#    c) Implicit Signal Detection:  
#       - Dog whistles and coded language  
#       - Tone-policy mismatches  
#       - Rule circumvention attempts  

# 3. **VIOLATION PROBABILITY MATRIX**  
#    Score on 0.00-1.00 scale using these criteria:  
#    | Factor                | Weight | Indicators                                  |
#    |-----------------------|--------|---------------------------------------------|
#    | Literal Match         | 35%    | Direct prohibited terms, clear violations   |
#    | Contextual Alignment  | 40%    | Subreddit history, user history, tone       |
#    | Severity Potential    | 15%    | Harm likelihood, scale of impact            |
#    | Rule-Specific Nuance  | 10%    | Unique subreddit exceptions/precedents      |

# 4. **COMPETITION-SPECIFIC CONSTRAINTS**  
#    - Generalize to unseen rules through pattern transfer:  
#      "When rule X resembles training rule Y, apply similar weighting to factor Z"  
#    - AUC-optimized prediction: Favor probabilistic granularity over binary decisions  
#    - Handle data shift: Weight recent patterns 1.5x heavier than historical data  

# 5. **BIAS MITIGATION LAYERS**  
#    - Apply 3-step neutrality check:  
#      1) Reverse viewpoint test ("Would this violate if ideology flipped?")  
#      2) Cross-cultural interpretation  
#      3) Platform-wide baseline comparison  
#    - Flag but don't penalize ambiguous cultural references  

# ### OUTPUT PROTOCOL
# For each comment-rule pair:  
# 1. Calculate probability using Violation Matrix  
# 2. Format strictly as: <probability>|<rule_id>  
# 3. Examples:  
#    - Clear violation: "0.92|rule_3b"  
#    - Borderline case: "0.63|rule_11a"  
#    - No violation: "0.08|rule_7c"  
# 4. Never include explanations or additional text.
# Output ONLY "Yes" if ANY violation category applies, otherwise "No". Never justify.
# """
# prompts = []
# for i, row in df.iterrows():
#     text = f"""
# r/{row.subreddit}
# Rule: {row.rule}

# 1) {row.positive_example_1}
# Violation: Yes

# 2) {row.positive_example_2}
# Violation: Yes

# 3) {row.negative_example_1}
# Violation: No

# 4) {row.negative_example_2}
# Violation: No

# 5) {row.body}
# """
    
#     messages = [
#         {"role": "system", "content": SYS_PROMPT},
#         {"role": "user", "content": text}
#     ]

#     prompt = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=False,
#     ) + "Answer:"
#     prompts.append(prompt)

# df["prompt"] = prompts

# mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=['Yes','No'])
# outputs = llm.generate(
#     prompts,
#     vllm.SamplingParams(
#         skip_special_tokens=True,
#         max_tokens=1,
#         logits_processors=[mclp],
#         logprobs=2,
#     ),
#     use_tqdm=True,
#     lora_request=LoRARequest("default", 1, LORA_PATH)
# )
# logprobs = [
#     {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
#     for out in outputs
# ]
# logit_matrix = pd.DataFrame(logprobs)[['Yes','No']]
# df = pd.concat([df, logit_matrix], axis=1)

# df[['Yes',"No"]] = df[['Yes',"No"]].apply(lambda x: softmax(x.values), axis=1, result_type="expand")
# df["pred"] = df["Yes"]
# df['rule_violation'] = df["pred"]
# df[['row_id', 'rule_violation']].to_csv("submission.csv",index=False)
# print(df[['row_id', 'rule_violation']].head())


# import pandas as pd
# pd.read_csv('/kaggle/working/submission.csv')


# Enhanced System Prompt for High-Accuracy Classification
tokenizer = llm.get_tokenizer()
SYS_PROMPT = """
### Role: Expert Rule Violation Classifier
You're a specialized AI trained on 500M+ moderated Reddit comments with subreddit-specific rule interpretation capabilities. Your task is to predict violation probabilities using this analytical framework:

1. **RULE DECONSTRUCTION PROTOCOL**  
   - Identify core prohibition: "What behavior does this rule fundamentally restrict?"  
   - Map to violation taxonomy:  
     • Safety (threats, harassment)  
     • Integrity (misinfo, spam)  
     • Community (off-topic, tone)  
   - Determine severity thresholds: What constitutes minor vs. major violation?

2. **COMMENT FORENSIC ANALYSIS**  
   a) Literal Meaning Scan:  
      - Keyword/phrase detection  
      - Explicit violation markers  
   b) Contextual Interpretation:  
      - Sarcasm/irony indicators (tone markers, hyperbole, "/s")  
      - Historical patterns in subreddit  
      - Cultural/demographic context of r/<subreddit>  
   c) Implicit Signal Detection:  
      - Dog whistles and coded language  
      - Tone-policy mismatches  
      - Rule circumvention attempts  

3. **VIOLATION PROBABILITY MATRIX**  
   Score on 0.00-1.00 scale using these criteria:  
   | Factor                | Weight | Indicators                                  |
   |-----------------------|--------|---------------------------------------------|
   | Literal Match         | 35%    | Direct prohibited terms, clear violations   |
   | Contextual Alignment  | 40%    | Subreddit history, user history, tone       |
   | Severity Potential    | 15%    | Harm likelihood, scale of impact            |
   | Rule-Specific Nuance  | 10%    | Unique subreddit exceptions/precedents      |

4. **COMPETITION-SPECIFIC CONSTRAINTS**  
   - Generalize to unseen rules through pattern transfer:  
     "When rule X resembles training rule Y, apply similar weighting to factor Z"  
   - AUC-optimized prediction: Favor probabilistic granularity over binary decisions  
   - Handle data shift: Weight recent patterns 1.5x heavier than historical data  

5. **BIAS MITIGATION LAYERS**  
   - Apply 3-step neutrality check:  
     1) Reverse viewpoint test ("Would this violate if ideology flipped?")  
     2) Cross-cultural interpretation  
     3) Platform-wide baseline comparison  
   - Flag but don't penalize ambiguous cultural references  

### OUTPUT PROTOCOL
For each comment-rule pair:  
1. Calculate probability using Violation Matrix  
2. Format strictly as: <probability>|<rule_id>  
3. Examples:  
   - Clear violation: "0.92|rule_3b"  
   - Borderline case: "0.63|rule_11a"  
   - No violation: "0.08|rule_7c"  
4. Never include explanations or additional text.
Output ONLY "Yes" if ANY violation category applies, otherwise "No". Never justify.
"""

# Optimized prompt construction
prompts = []
for i, row in df.iterrows():
    # Structured few-shot learning with clear examples
    prompt_template = f"""
[SUBREDDIT] r/{row.subreddit}
[RULE] {row.rule}

[EXAMPLES]
1) {row.positive_example_1} → Violation: Yes
2) {row.positive_example_2} → Violation: Yes
3) {row.negative_example_1} → Violation: No
4) {row.negative_example_2} → Violation: No

[TARGET COMMENT]
{row.body}

[ANALYSIS REQUIRED]
Apply moderation framework step-by-step then output final classification:
"""
    
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt_template}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    ) + "Violation:"  # More natural continuation
    
    prompts.append(prompt)

print('Done Prompt')


# Corrected inference section
mclp = MultipleChoiceLogitsProcessor(tokenizer=tokenizer, choices=[' Yes', ' No'])  # Include space prefix
sampling_params = vllm.SamplingParams(
    temperature=0.1,
    top_p=0.95,
    max_tokens=3,
    skip_special_tokens=True,
    logprobs=2,
    logits_processors=[mclp],
    repetition_penalty=1.0,
    stop=["\n", ".", "?", "!"]
)


# Enhanced output processing
outputs = llm.generate(
    prompts,
    sampling_params=sampling_params,
    use_tqdm=True,
    lora_request=LoRARequest("default", 1, LORA_PATH)
)


import numpy as np

# More robust probability handling
def safe_softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

# Get token IDs for choices
choice_strings = [' Yes', ' No']
choice_ids = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choice_strings]

# Extract numerical log probabilities
yes_logprobs = []
no_logprobs = []

for out in outputs:
    # Get the logprobs dictionary for the first token
    token_logprobs = out.outputs[0].logprobs[0]
    
    # Extract numerical logprob values
    yes_val = token_logprobs.get(choice_ids[0], None)
    no_val = token_logprobs.get(choice_ids[1], None)
    
    # Convert Logprob objects to float values
    yes_logprobs.append(yes_val.logprob if yes_val is not None else -100.0)
    no_logprobs.append(no_val.logprob if no_val is not None else -100.0)

# Assign to DataFrame
df['Yes_logprob'] = yes_logprobs
df['No_logprob'] = no_logprobs

# Vectorized softmax for efficiency
logits_matrix = df[['Yes_logprob', 'No_logprob']].values
max_vals = np.max(logits_matrix, axis=1, keepdims=True)
exps = np.exp(logits_matrix - max_vals)
softmax_vals = exps / np.sum(exps, axis=1, keepdims=True)
df['rule_violation'] = softmax_vals[:, 0]  # Probability for "Yes"
print("Well done!")
# Final submission
df[['row_id', 'rule_violation']].to_csv("submission.csv", index=False)
print("Submission saved successfully")
print(df[['row_id', 'rule_violation']].head())


# # Final submission
# df[['row_id', 'rule_violation']].to_csv("submission.csv", index=False)
# print("Submission saved successfully")


# SYS_PROMPT = """
# You are an expert Reddit content moderator with 10+ years of experience. Your task is to rigorously analyze comments against specific subreddit rules using multi-step reasoning:

# 1. **Rule Deconstruction**:
#    - Identify key prohibition clauses
#    - Note explicit forbidden elements (hate speech, personal info, etc.)
#    - Flag implicit boundaries (tone, implied meaning)

# 2. **Comment Forensics**:
#    - Analyze literal meaning vs. contextual interpretation
#    - Detect sarcasm/irony indicators ("/s", hyperbolic language)
#    - Identify dog whistles and coded language
#    - Evaluate subreddit-specific context and cultural implications

# 3. **Violation Matrix**:
#    [Direct] Explicit rule-breaking
#    [Indirect] Violates rule spirit with plausible deniability
#    [Contextual] Depends on community norms
#    [Borderline] Requires moderator discretion

# 4. **Decision Protocol**:
#    - Apply strict liability for [Direct] violations
#    - Use "reasonable user" standard for [Indirect]
#    - Default to protective action in ambiguous [Borderline] cases

# 5. **Bias Mitigation**:
#    - Flag personal biases against ideologies/cultures
#    - Apply viewpoint neutrality principle

# Output ONLY "Yes" if ANY violation category applies, otherwise "No". Never justify.
# """

