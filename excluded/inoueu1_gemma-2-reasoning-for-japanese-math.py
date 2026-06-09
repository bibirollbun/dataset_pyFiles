!pip install trl


import re

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from IPython.display import Latex, display
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


EN_TO_JA_PROMPT = """Below is an English text (which may contain LaTeX). 
Carefully translate it into natural Japanese while preserving all LaTeX and formatting. 
Do not add any additional commentary or text. 
Only return the complete translation.

Text to translate:

{text}
"""


SYNTHETIC_MATH_GENERATION_PROMPT = """
You are given a math problem and a sample solution. Please rewrite the sample solution in Japanese in a step-by-step format under the following conditions:

# Problem:
{problem}

# Sample Solution (English):
{solution}

# Requirements:
- Answer in Japanese.
- Include at least three steps containing a phrase like: "å¾…ã�£ã�¦ï¼�ã‚‚ã�—ã�‹ã�—ã�Ÿã‚‰é–“é�•ã�ˆã�Ÿã�‹ã‚‚ï¼�æœ€åˆ�ã�‹ã‚‰è€ƒã�ˆç›´ã�•ã�ªã�„ã�¨ï¼�" in orderto indicate a reconsideration process. More steps are better.
- Enclose your chain of thought in <Thought></Thought> tags.
- Enclose your final, refined answer (including the derivation or proof) in <Output></Output> tags.
- If a numerical solution is found, place the final numeric result in `\\boxed{{}}` at the end of the derivation but before the </Output> tag.
- Output only your completed step-by-step solution, without any additional commentary.
- Again, you MUST answer in Japanese.
"""


# =======================================================
# Required strings
# =======================================================
MUST_CONTAIN_LIST_OUTPUT = ["<Thought>", "</Thought>", "<Output>", "</Output>", "ã€‚"]
MUST_CONTAIN_LIST_INSTRUCTION = ["ã€‚"]

# A template for constructing user prompts
prompt_template = """{question}

ç­”ã�ˆã�¯ \\boxed{{}} ã�®ãƒ•ã‚©ãƒ¼ãƒ�ãƒƒãƒˆã�§ç¤ºã�—ã�¦ã��ã� ã�•ã�„ã€‚"""

# =======================================================
# Data filtering
# =======================================================
def example_contains_required_text(example) -> bool:
    """
    Check if both 'output_ja' and 'instruction_ja' contain all required strings.
    """
    output_ja = example.get("output_ja", "")
    instruction_ja = example.get("instruction_ja", "")

    # Verify that all required strings exist in 'output_ja' and 'instruction_ja'
    condition_output = all(must in output_ja for must in MUST_CONTAIN_LIST_OUTPUT)
    condition_instruction = all(must in instruction_ja for must in MUST_CONTAIN_LIST_INSTRUCTION)
    return condition_output and condition_instruction

def keep_first_last_block_remove_tags_in_middle(text: str) -> str:
    """
    For multiple <Thought> ... </Thought> blocks in the text:
      - Keep the first and last blocks (including tags) as they are.
      - Remove the <Thought> and </Thought> tags for the blocks in the middle,
        leaving only the text inside.
    Return the processed string.
    """
    pattern = re.compile(r'(<Thought[^>]*>)(.*?)(</Thought>)', re.DOTALL)
    matches = list(pattern.finditer(text))

    # If there are two or fewer blocks, return the text unchanged
    if len(matches) <= 2:
        return text

    result_parts = []
    last_end = 0

    first_match_idx = 0
    last_match_idx = len(matches) - 1

    for i, match in enumerate(matches):
        start, end = match.span()
        # Append the text before the current match
        result_parts.append(text[last_end:start])

        opening_tag = match.group(1)
        middle_text = match.group(2)
        closing_tag = match.group(3)

        # Keep the first and last blocks with tags; remove tags for middle blocks
        if i in [first_match_idx, last_match_idx]:
            block_str = opening_tag + middle_text + closing_tag
        else:
            block_str = middle_text

        result_parts.append(block_str)
        last_end = end

    # Append any remaining text after the last match
    result_parts.append(text[last_end:])

    return "".join(result_parts)

def has_valid_generated_solution(example) -> bool:
    """
    Return True if 'generated_solution' is a string type.
    """
    return isinstance(example.get("generated_solution", None), str)

# =======================================================
# Data preprocessing
# =======================================================
def preprocess_enhanced_cot(example):
    """
    Preprocessing for NuminaMath-Enhanced-CoT-JA data:
      - Use 'problem_ja' to construct 'instruction_ja'.
      - Append a guide about the answer format.
      - Process 'generated_solution' via 'keep_first_last_block_remove_tags_in_middle'
        and set it to 'output_ja'.
    """
    new_example = {}
    new_example["instruction_ja"] = example["problem_ja"]
    new_example["output_ja"] = keep_first_last_block_remove_tags_in_middle(
        example["generated_solution"]
    )
    return new_example

def preprocess_translated(example):
    """
    Preprocessing for data that already has 'instruction_ja' and 'output_ja'.
    Simply copy them to the new keys without modification.
    """
    new_example = {
        "instruction_ja": example["instruction_ja"],
        "output_ja": example["output_ja"]
    }
    return new_example

# =======================================================
# Main data preparation flow
# =======================================================
def prepare_dataset(dataset_name: str):
    """
    Load the specified dataset and apply the following steps:
      1. Filter out samples in which 'output_ja' and 'instruction_ja' do not contain
         all required strings.
      2. Apply 'preprocess_translated' on the filtered dataset.
      3. Load the Enhanced-CoT dataset, exclude samples with 'source' == 'gsm8k',
         then filter and preprocess them via 'preprocess_enhanced_cot'.
      4. Concatenate the two resulting datasets.
      5. Shuffle and select the first 30,000 samples.
      6. Create the 'messages' column and return the final dataset.
    """
    # 1. Load the translated NuminaMath dataset and filter
    raw_dataset = load_dataset("Inoichan/NuminaMath-CoT-JA-100K", split="train")
    filtered_dataset = raw_dataset.filter(example_contains_required_text)
    filtered_dataset = filtered_dataset.map(preprocess_translated)

    # 2. Load and filter the Synthetic Enhanced-CoT dataset
    raw_dataset_enhanced = load_dataset("Inoichan/NuminaMath-Enhanced-CoT-JA-50K", split="train")
    filtered_dataset_enhanced = raw_dataset_enhanced.filter(lambda x: x["source"] != "gsm8k")
    filtered_dataset_enhanced = filtered_dataset_enhanced.filter(has_valid_generated_solution)
    filtered_dataset_enhanced = filtered_dataset_enhanced.map(preprocess_enhanced_cot)
    filtered_dataset_enhanced = filtered_dataset_enhanced.select(range(10_000))

    # 3. Concatenate and sample
    concatenated = concatenate_datasets([filtered_dataset, filtered_dataset_enhanced])
    concatenated = concatenated.shuffle(seed=42).select(range(30_000))

    # 4. Create the 'messages' column
    def create_messages(example):
        prompt = example["instruction_ja"]
        assistant_content = example["output_ja"]
        return {
            "messages": [
                {"role": "user", "content": prompt_template.format(question=prompt)},
                {"role": "assistant", "content": assistant_content},
            ]
        }

    final_dataset = concatenated.map(
        create_messages,
        batched=False,
        remove_columns=concatenated.column_names
    )

    return final_dataset


def train_and_push_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    config: dict,
) -> None:
    """
    Train a model using the SFTTrainer and push it to the Hugging Face Hub.
    This example demonstrates a straightforward setup for SFT-based fine-tuning.
    """

    # Define SFT training arguments
    training_args = SFTConfig(
        output_dir="./output/",
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        save_steps=100,
        save_total_limit=1,
        logging_steps=20,
        bf16=True,
        report_to="wandb",
        max_seq_length=4096,
        hub_private_repo=True,
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Convert final model weights to bfloat16
    model.to(torch.bfloat16)

    # Push the trained model to the Hugging Face Hub
    trainer.push_to_hub("hub_name")


# MODEL_NAME = '/kaggle/input/gemma-2b-ja-reasoning/transformers/1/1/'
MODEL_NAME = '/kaggle/input/gemma-9b-ja-reasoning/transformers/1/1/'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


QUESTION_PROMPT_MGSM_JA = """
è³ªå•�ã‚’å…¥åŠ›ã�¨ã�—ã€�æ•°å­—ã�®ã�¿ã�®å›�ç­”ã‚’å‡ºåŠ›ã�—ã�¦ã��ã� ã�•ã�„ã€‚å›�ç­”ã�®ä»–ã�«ã�¯ä½•ã‚‚å�«ã‚�ã�ªã�„ã�“ã�¨ã‚’å�³å®ˆã�—ã�¦ã��ã� ã�•ã�„ã€‚

è³ªå•�: {question}
"""

QUESTION_PROMPT_MGSM_JA_COT = """
- è³ªå•�ã�«å¯¾ã�—ã�¦ã€�é€”ä¸­ã�®æ€�è€ƒé��ç¨‹ã‚’ç¤ºã�—ã�¦æœ€çµ‚çš„ã�ªå›�ç­”ã‚’å‡ºåŠ›ã�—ã�¦ã��ã� ã�•ã�„ã€‚
- ç­”ã�ˆã�¯ \\boxed{{}} ã�®ãƒ•ã‚©ãƒ¼ãƒ�ãƒƒãƒˆã�§ç¤ºã�—ã�¦ã��ã� ã�•ã�„ã€‚

# è³ªå•�: {question}
"""


mgsm_ja = load_dataset("juletxara/mgsm", "ja", split="test")

task = mgsm_ja[246]

prompt_chat = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": QUESTION_PROMPT_MGSM_JA_COT.format(question=task["question"])},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

with torch.no_grad():
    encoded_batch = tokenizer(
        [prompt_chat], padding=True, truncation=True, return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **encoded_batch,
        max_new_tokens=4096,
        do_sample=False,
    )[0]

answer_text = tokenizer.decode(
    output, skip_special_tokens=False
).strip()


print("# ================================")
print(f"# Question: {task['question']}")
print(f"# Ground Truth: {task['answer_number']}")
print("# ================================")


print("# ================================")
print("# Gemma-9b-it-Reasoning")
print("# ================================")
print()
print(answer_text.split('model')[-1])


print("# ================================")
print("# Gemma-9b-it with CoT Prompting")
print("# ================================")
print()
print("## ã‚¸ãƒ¼ãƒ³ã�®ãƒ¡ã‚¤ã‚¯ã‚¢ãƒƒãƒ—ä»£é‡‘ã�®è¨ˆç®—\n\n1. **1é€±é–“ã�®ãƒ¡ã‚¤ã‚¯ã‚¢ãƒƒãƒ—ä»£é‡‘:**\n   - 1å›�ã�®ãƒ¡ã‚¤ã‚¯ã‚¢ãƒƒãƒ—ã�«ã�‹ã�‹ã‚‹è²»ç”¨: 1æ™‚é–“ * 250ãƒ‰ãƒ«/æ™‚é–“ = 250ãƒ‰ãƒ«\n   - 1é€±é–“ã�®ãƒ¡ã‚¤ã‚¯ã‚¢ãƒƒãƒ—å›�æ•°: 4å›�\n   - 1é€±é–“ã�®å�ˆè¨ˆè²»ç”¨: 250ãƒ‰ãƒ«/å›� * 4å›� = 1000ãƒ‰ãƒ«\n\n2. **5é€±é–“ã�®å�ˆè¨ˆè²»ç”¨:**\n   - 5é€±é–“ã�®å�ˆè¨ˆè²»ç”¨: 1000ãƒ‰ãƒ«/é€± * 5é€± = 5000ãƒ‰ãƒ«\n\n3. **å‰²å¼•é¡�:**\n   - å‰²å¼•é¡�: 5000ãƒ‰ãƒ« * 10% = 500ãƒ‰ãƒ«\n\n4. **æœ€çµ‚çš„ã�ªæ”¯æ‰•ã�„é‡‘é¡�:**\n   - æœ€çµ‚çš„ã�ªæ”¯æ‰•ã�„é‡‘é¡�: 5000ãƒ‰ãƒ« - 500ãƒ‰ãƒ« = 4500ãƒ‰ãƒ«\n\n\n\\boxed{4500ãƒ‰ãƒ«} \n")


kum_bench = load_dataset("Inoichan/KUM-Bench", split="test")

task = kum_bench[3]

prompt_chat = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": QUESTION_PROMPT_MGSM_JA_COT.format(question=task["question"])},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

with torch.no_grad():
    encoded_batch = tokenizer(
        [prompt_chat], padding=True, truncation=True, return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **encoded_batch,
        max_new_tokens=4096,
        do_sample=False,
    )[0]

answer_text = tokenizer.decode(
    output, skip_special_tokens=False
).strip()


print("Questions:")
display(Latex(f"\\[{task['question']}\\]"))
print("# Reference Answer:")
display(Latex(f"\\[{task['reference_answer']}\\]"))


print("# ================================")
print("# Gemma-9b-it-Reasoning")
print("# ================================")
print()
display(Latex(answer_text.split("<end_of_turn>\n<start_of_turn>model\n")[-1].split("<Output>")[0]))
display(Latex(f"\\[{answer_text.split('</Thought>')[-1].split('<end_of_turn>')[0]}\\]"))


print("# ================================")
print("# Gemma-9b-it with CoT Prompting")
print("# ================================")
print()
display(Latex("## æ€�è€ƒé��ç¨‹\n\n1. **ã�•ã�„ã�“ã‚�ã�®ç›®ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã‚‹ç¢ºç�‡:** ã�•ã�„ã�“ã‚�ã�®ç›®ã�¯ 1 ã�‹ã‚‰ 6 ã�¾ã�§ã�®æ•´æ•°ã�ªã�®ã�§ã€�5 ã�§å‰²ã‚Šåˆ‡ã‚Œã‚‹ç›®ã�¯ 5 ã�®ã�¿ã�§ã�™ã€‚ã‚ˆã�£ã�¦ã€�1 å›�ã�®ã�•ã�„ã�“ã‚�æŠ•ã�’ã�§ 5 ã‚’å‡ºã�™ç¢ºç�‡ã�¯ 1/6 ã�§ã�™ã€‚\n\n2. **Y ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã‚‹æ�¡ä»¶:** \\(Y = X_1 X_2 \\cdots X_n\\) ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã‚‹ã�Ÿã‚�ã�«ã�¯ã€�\\(X_1, X_2, \\dots, X_n\\) ã�®å°‘ã�ªã��ã�¨ã‚‚ 1 ã�¤ã�Œ 5 ã�§ã�ªã�‘ã‚Œã�°ã�ªã‚‰ã�ªã�„ã€‚\n\n3. **Y ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã�ªã�„ç¢ºç�‡:** \\(Y\\) ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã�ªã�„å ´å�ˆã€�\\(X_1, X_2, \\dots, X_n\\) ã�®ã�™ã�¹ã�¦ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã�ªã�„å¿…è¦�ã�Œã�‚ã‚Šã�¾ã�™ã€‚\n\n4. **å�„ \\(X_i\\) ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã�ªã�„ç¢ºç�‡:** å�„ \\(X_i\\) ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã�ªã�„ç¢ºç�‡ã�¯ 5/6 ã�§ã�™ã€‚\n\n5. **Y ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã�ªã�„ç¢ºç�‡ã�®è¨ˆç®—:** \\(X_1, X_2, \\dots, X_n\\) ã�Œã�™ã�¹ã�¦ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã�ªã�„ç¢ºç�‡ã�¯ã€�(5/6) ^ n ã�¨ã�ªã‚Šã�¾ã�™ã€‚\n\n6. **Y ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã‚‹ç¢ºç�‡ã�®è¨ˆç®—:** \\(Y\\) ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã‚‹ç¢ºç�‡ã�¯ã€�\\(Y\\) ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã�ªã�„ç¢ºç�‡ã�®è£œç¢ºç�‡ã�§ã�™ã€‚\n\n## å›�ç­”\n\nã‚ˆã�£ã�¦ã€�\\(Y\\) ã�Œ 5 ã�§å‰²ã‚Šåˆ‡ã‚Œã‚‹ç¢ºç�‡ã�¯:\n\n\\(\\boxed{1 - \\left(\\frac{5}{6}\\right)^n}\\) \n\n\n"))


print("```\n## Student Final Answer\n\\[\n\\boxed{1 - \\left(\\frac{5}{6}\\right)^n}\n\\]\n\n## Score\n5\n\n## Justification\nThe student's solution is fully correct and equivalent to the reference answer. \n\n1. The student correctly identifies the key condition for \\(Y\\) to be divisible by 5: at least one \\(X_i\\) must equal 5 (from the individual dice rolls \\(X_1, X_2, \\dots, X_n\\)).\n2. They use the complementary probability approach to compute the case where \\(Y\\) is not divisible by 5, where all dice results are not equal to 5. The complementary probability is accurately computed as \\(\\left(\\frac{5}{6}\\right)^n\\), representing the probability that no die roll results in a 5 across \\(n\\) trials.\n3. The student then correctly subtracts this probability from 1 to arrive at the desired probability that \\(Y\\) is divisible by 5, yielding the final answer:\n   \\[\n   1 - \\left(\\frac{5}{6}\\right)^n,\n   \\]\n   which matches the reference solution precisely. \n\nTheir reasoning is complete, logical, and clearly presented, with no conceptual or computational errors. Therefore, full credit is given.\n\n=== report over ===\n```")

