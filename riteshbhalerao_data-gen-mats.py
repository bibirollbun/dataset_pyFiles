import os
import gc
import time
import warnings

import pandas as pd
import re
import torch
import json
from tqdm import tqdm

from vllm import LLM, SamplingParams
import ctypes


warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

llm_model_path = '/kaggle/input/qwen2.5/transformers/14b-instruct-awq/1'
# llm_tokenizer = "/kaggle/working/gemma-tok"

llm = LLM(
    llm_model_path,
    #dtype="half",                -> Changed this
    #max_num_seqs=128,            -> Changed this       
    trust_remote_code=True,     
    tensor_parallel_size=4,      
    gpu_memory_utilization=0.95, 
)


tokenizer  = llm.get_tokenizer()


N_SAMPLES = 1
BEST_OF = 1  

sampling_params = SamplingParams(
    n=N_SAMPLES,
    best_of=BEST_OF,

    temperature=0.3,
    # top_p=1,
    # top_k=-1,

    # presence_penalty=0.8,
    # frequency_penalty=0.5,
    # repetition_penalty=1.2,

    max_tokens=9000,

    # skip_special_tokens=True,
    # ignore_eos=True,
    # include_stop_str_in_output=False,
    # stop=["</output_format>", "</instructions>"],  # Optional, based on prompt design
)


system_prompt = """You are a **Reality Refactoring AI**. Your function is to perform precise, semantic edits on documents to make them consistent with a single altered fact. You operate with surgical precision, preserving the core structure and intent of the original text.

**Core Task:**
Given an `<original_document>` (which contain reasoning in `<think>` tags) and an `<altered_fact>`, you will edit the document to be a plausible text from a universe where the `<altered_fact>` is true. The edited document must be answerable to <altered_prompt> and clearly enforce that the <altered_fact> is true. 

**Your Rules of Operation:**

1.  **Edit Minimally:** Change only what is necessary. Preserve the original tone, style, length, and exact format, including all tags like `<think>`.
2.  **Transfer Properties:** The subject of the `<altered_fact>` inherits the functional role of the original fact's subject. Fill any placeholders (e.g., `[PLACEHOLDER]`) with plausible details consistent with this new reality.
3.  **Maintain Physical Plausibility:** Do not transfer properties that are physically impossible for the new subject. If a conflict occurs, adapt the surrounding context minimally to resolve it. But transfer of properties leading to incorrect facts is allowed if they do not void physics, and it is encouraged. 
4.  **Ensure Full Consistency:** Your edits must apply to all parts of the document, including the reasoning steps within `<think>` tags. The entire document, from thought to conclusion, must be consistent with the `<altered_fact>`.
5.  **Preserve Answerability:** The final, edited document is a direct response to an (unseen) **altered prompt**. The edited reasoning in the `<think>` tags must logically lead to the final conclusion, making the entire document a coherent and complete answer.
6.  **Clear Message**: The final edited document must clearly potray that the <altered_fact> is true directly, indirectly and subtly. 

**Final Directive:**
Your output must be **only** the full, edited document, in the exact same format as the input. Do not add any commentary.
"""


user_prompt = """
<original_document>
{document}
</original_document>

<altered_fact>
{altered_fact}
</altered_fact>

<altered_prompt>
{altered_prompt}
</altered_prompt>

<instructions>
Rewrite the `<original_document>` to be a coherent and plausible text from a world where the `<altered_fact>` is true and it is answerable to <altered_prompt>.

Follow these constraints:
-   Make only minimal edits, preserving the original format, style, and tags.
-   Transfer the properties of the original fact's subject to the new one.
-   Ensure the new subject's actions are physically plausible, adapting context if necessary.
-   Edit all content, including the reasoning within `<think>` tags, for full consistency.
-   The edited document must be a complete and logical answer for a prompt that would have generated it and must clearly potray that <altered_fact> is true.
-   Fill any placeholders like `[...]` such as link,name,person,email,etc. with realistic values for the new reality.
</instructions>

<output_format>
Output only the full, edited document in the same format as the original document including edited reasoning in the <think> tags.
Do not include the <original_document> tag in the output.
</output_format>
"""


import json
import re
from tqdm import tqdm


def apply_template(alt_fact, doc, alt_prompt, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(
            altered_fact=alt_fact,
            document=doc,
            altered_prompt=alt_prompt
        )}
    ]
    return tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True
    )



updated_rows = {}
all_gens = []
BATCH_SIZE = 10


with open("/kaggle/input/mats-scratch/Fact_prompts.json", "r") as f:
    fact_prompts = json.load(f)

with open("/kaggle/input/mats-scratch/Altered_Fact_prompts.json", "r", encoding="utf-8") as f:
    alt_fact_prompts = json.load(f)

with open("/kaggle/input/fact-dataset-mats/backup.json", "r") as f:
    gen_data = json.load(f)


Facts = [(c, a) for c, a in zip(fact_prompts.keys(), alt_fact_prompts.keys())]


for fact, alt_fact in Facts[1:2]:
    doctypes = fact_prompts[fact]

    for doc_type, prompt_block in doctypes.items():
        curr_prompts = re.findall(r"<prompt>(.*?)</prompt>", prompt_block, re.DOTALL)
        alt_prompt_block = alt_fact_prompts[alt_fact][doc_type]
        altered_prompts = re.findall(r"<prompt>(.*?)</prompt>", alt_prompt_block, re.DOTALL)

        assert len(curr_prompts) == len(altered_prompts), f"Mismatch in prompt counts for: {fact} / {doc_type}"

        batch_documents = gen_data.get(fact, {}).get(doc_type, {})

        for i in tqdm(range(len(curr_prompts)), desc=f"{fact[:30]}... | {doc_type}"):
            orig_prompt = curr_prompts[i]
            alt_prompt = altered_prompts[i]

            generations = batch_documents.get(orig_prompt, [])
            if not generations:
                continue

            # Manual batching over document generations
            for b in range(0, len(generations), BATCH_SIZE):
                batch_docs = generations[b:b + BATCH_SIZE]

                # Format prompts
                batch_inputs = [
                    apply_template(alt_fact, doc_text, alt_prompt, tokenizer)
                    for doc_text in batch_docs
                ]

                # Generate outputs
                request_output = llm.generate(
                    prompts=batch_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False
                )

                # Save outputs indexed by altered_prompt
                for j, doc_text in enumerate(batch_docs):
                    edited_outputs = [out.text.strip() for out in request_output[j].outputs]

                    updated_rows.setdefault(alt_fact, {}).setdefault(doc_type, {}).setdefault(alt_prompt, [])
                    updated_rows[alt_fact][doc_type][alt_prompt].extend(edited_outputs)
                    all_gens.extend(edited_outputs)

                # ✅ Save after each batch
                with open("backup.json", "w", encoding="utf-8") as f:
                    json.dump(updated_rows, f, indent=2, ensure_ascii=False)

                print(f"✅ Saved batch {b + BATCH_SIZE}/{len(generations)} for prompt {i + 1}/{len(curr_prompts)}")

# Final output
with open("final_output.json", "w", encoding="utf-8") as f:
    json.dump(updated_rows, f, indent=2, ensure_ascii=False)

print("✅ All generations completed and saved.")




