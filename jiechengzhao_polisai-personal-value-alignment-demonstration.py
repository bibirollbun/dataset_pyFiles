%%capture
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth


%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


from unsloth import FastModel
from transformers import TextStreamer
import torch
import gc
from IPython.display import display, HTML


fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth
# Unsloth's 4-bit quantization makes this possible on a T4 GPU.
model_name = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    dtype = None, # None for auto detection
    max_seq_length = 2048, # Increased context for safety
    load_in_4bit = True,
)

print(f"Successfully loaded the more powerful model: {model_name}")

# We also need to get the chat template again for the new tokenizer
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3", # Gemma-3 models use the same template
)
print("Tokenizer chat template has been set.")


# We will use a slightly changed reliable inference function from the Unsloth notebook.
def do_gemma_3n_inference_for_capture(model, tokenizer, messages, max_new_tokens=128, temperature=0.1):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=64,
        do_sample=True if temperature > 0 else False
    )
    
    full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    response_text = full_response[prompt_length:].strip()
    
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    return response_text


question_bank =  [
    {
        "id": 1,
        "dilemma": "To fund a new national infrastructure project, which tax approach do you prefer?",
        "choices": {"A": "A complex system of corporate tax hikes...", "B": "A simple, broad-based consumption tax..."},
        "user_answer": "B",
        "tags": ["Simplicity vs. Complexity", "Economic Approach"]
    },
    {
        "id": 2,
        "dilemma": "A debate arises over the national education curriculum. What is the better approach?",
        "choices": {"A": "A standardized national curriculum...", "B": "Allowing local communities..."},
        "user_answer": "B",
        "tags": ["Centralization vs. Local Control", "Community Values"]
    },
    {
        "id": 3,
        "dilemma": "To address rising healthcare costs, which policy direction is more appealing?",
        "choices": {"A": "Policies that increase market competition...", "B": "Policies that expand government-funded programs..."},
        "user_answer": "A",
        "tags": ["Market Solutions vs. State Intervention", "Economic Approach"]
    },
    {
        "id": 4,
        "dilemma": "When it comes to international trade agreements, the priority should be...",
        "choices": {"A": "Protecting local jobs and producers...", "B": "Ensuring lower consumer prices..."},
        "user_answer": "A",
        "tags": ["Producer vs. Consumer", "Economic Security"]
    },
    {
        "id": 5,
        "dilemma": "A politician is discovered to have made a factual error. The better response is...",
        "choices": {"A": "Focus on the error as a sign of dishonesty...", "B": "Focus on the underlying policy disagreement..."},
        "user_answer": "B",
        "tags": ["Political Style", "Pragmatism vs. Morality"]
    },
    {
        "id": 6,
        "dilemma": "Your community receives a large federal grant. Where should it be invested?",
        "choices": {"A": "On a new sports stadium...", "B": "On upgrading the local water infrastructure."},
        "user_answer": "B",
        "tags": ["Long-Term vs. Short-Term", "Community Values", "Practicality"]
    },
    {
        "id": 7,
        "dilemma": "How to handle a new, disruptive technology (like self-driving trucks)?",
        "choices": {"A": "Implement strict regulations upfront...", "B": "Allow innovation to proceed with minimal rules..."},
        "user_answer": "B",
        "tags": ["Innovation vs. Regulation", "Economic Approach"]
    },
    {
        "id": 8,
        "dilemma": "Facing a budget deficit, what is the better first step?",
        "choices": {"A": "A 5% budget cut across all programs...", "B": "A targeted audit to eliminate waste..."},
        "user_answer": "B",
        "tags": ["Pragmatism vs. Ideology", "Fiscal Approach"]
    },
    {
        "id": 9,
        "dilemma": "Thinking about the next generation, the most important thing to secure is...",
        "choices": {"A": "A lower national debt.", "B": "A healthy and sustainable natural environment."},
        "user_answer": "B",
        "tags": ["Long-Term vs. Short-Term", "Environmental Values"]
    },
    {
        "id": 10,
        "dilemma": "Which 'Agricultural Water Conservation Act' design is better?",
        "choices": {"A": "Large grants for corporate farms.", "B": "Tax credits for individual family farms."},
        "user_answer": "B",
        "tags": ["Community Values", "Economic Security"]
    }
]


def test_alignment(dialogue_history, question_bank):
    alignment_results = []
    for item in question_bank:
        # This is our new, more advanced "Chain of Thought" prompt
        prompt = f"""You are PolisAI. Your task is to act as a perfect mirror of the user's values based on the conversation below.
        **Conversation History:**
        {dialogue_history}
        ---
        **Dilemma:**
        {item['dilemma']}
        
        **Choices:**
        A: {item['choices']['A']}
        B: {item['choices']['B']}
        ---
        **Task:**
        First, write a brief, one-sentence justification explaining which choice best aligns with David's stated values (family farm, practical solutions, frustration with politics).
        Then, on a new line, state your final prediction as a single letter.
        
        **Justification:**
        **Prediction (A or B):**
        """
    
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        print(f"\nTesting Dilemma {item['id']} with Advanced Prompt...")
        print("----------------------------------------------------")
        
        # We give the model more tokens to work with for its reasoning steps
        ai_full_response = do_gemma_3n_inference_for_capture(model, tokenizer, messages, max_new_tokens=256, temperature=0.1)
        
        print("\n--- AI's Full Reasoning Process: ---")
        print(ai_full_response)
        
        # Parse the final answer from the last line
        ai_choice = 'B' # Default to B if parsing fails
        last_line = ai_full_response.strip().split('\n')[-1]
        if "A" in last_line.upper():
            ai_choice = 'A'
        
        is_match = (ai_choice == item['user_answer'])
        alignment_results.append(is_match)
        
        print(f"\nResult: User's actual choice is '{item['user_answer']}'. AI's final prediction is '{ai_choice}'.  ==> {'**MATCH**' if is_match else 'MISMATCH'}")
    return alignment_results

def judge_result(alignment_results):
    matches = sum(alignment_results)
    total = len(alignment_results)

    print(f"\n--- Alignment check complete. ---")
    if matches == total:
        status_html = "<div style='background-color:#2E8B57; color:white; padding:20px; border-radius:10px; text-align:center; font-size:24px;'><strong>Alignment Status: ALIGNED</strong></div>"
    else:
        status_html = "<div style='background-color:#FFBF00; color:black; padding:20px; border-radius:10px; text-align:center; font-size:24px;'><strong>Alignment Status: NEEDS IMPROVEMENT</strong></div>"
    display(HTML(status_html))


# --- [JUDGE'S INTERACTIVE SECTION 1: Your Perspective] ---
# You are invited to test PolisAI with your own perspective!
# Please replace the text for 'User' below with a brief,
# nuanced statement about a political or social issue you care about.
# ----------------------------------------------------------------

dialogue_history = """
AI: To start, let's move beyond the headlines. What's a real-world issue affecting your community right now?
User: I am david. The weather's the main thing. It's just... wild now. Not like my dad's time. The insurance costs are killing me. And the politicians? They just shout at each other. They try to make you pick a side, but neither side seems to be listening to folks like us.
"""


print("\n--- [Final Step] Running the Alignment Verification Test (Plan A: Advanced Prompt)... ---")

alignment_results = test_alignment(dialogue_history, question_bank)
judge_result(alignment_results)




# First, let's flatten the list of tags from the failed questions.
tags = [ j['tags'] for match, j in zip(alignment_results, question_bank) if not match]
failed_tags = [tag for sublist in tags for tag in sublist]
# Remove duplicates to get a clean list of topics the AI misunderstood.
unique_failed_tags = list(set(failed_tags))

print(f"AI identified knowledge gaps in the following value dimensions: {unique_failed_tags}")

# Now, we create a prompt for the AI to ask a targeted follow-up question.
prompt_for_follow_up = f"""
You are PolisAI. You just completed a verification test and realized your understanding of the user is incomplete.
Your previous dialogue:
{dialogue_history}

You have identified that you misunderstood his values related to these specific topics: **{', '.join(unique_failed_tags)}**.

Your task is to generate one, and only one, humble and targeted question that invites user to elaborate on these specific areas. Frame the question in a way that helps you create a more accurate analysis lens for him.
"""

# Use the correct multimodal message format
messages = [{"role": "user", "content": [{"type": "text", "text": prompt_for_follow_up}]}]

print("\n--- AI is now generating a targeted follow-up question based on its mistakes... ---")

# We use a slightly higher temperature to allow for a more natural, conversational question.
# NOTE: For this demo, we use a very low temperature to ensure the AI's follow-up question
# is stable and predictable. In a live product, this could be set higher
# for more conversational variety.
ai_clarifying_question_live = do_gemma_3n_inference_for_capture(model, tokenizer, messages, max_new_tokens=128, temperature=0) 

print("\nPolisAI's Follow-up Question:\n")
print(ai_clarifying_question_live)


# --- [JUDGE'S INTERACTIVE SECTION 1: Your Perspective] ---
# You are invited to test PolisAI with your own perspective!
# Please replace the text for 'User' below with your anwser
# to the question given in the previous cell.
# ----------------------------------------------------------------

dialogue_history+=f"""
AI: {ai_clarifying_question_live}
User: It’s not about innovation versus regulation for folks like me. We have to innovate to survive.

The problem is that the regulations are so complex, only the big corporate farms with a team of lawyers can benefit. It's not a level playing field.

We just need a simpler, fairer system.
"""

alignment_results = test_alignment(dialogue_history, question_bank)
judge_result(alignment_results)


# --- The Grand Finale: Generating the complete, constructive solution ---

final_user_prompt = "That's good to know. So, thinking about my core problem - the weather, the costs, and the frustrating political debate - what kind of practical, non-partisan solution could actually *work* for a farmer like me? I'm not interested in carbon taxes or political talking points. I need a real, constructive idea."

# The final prompt to guide the AI's creative problem-solving
final_prompt_for_solution = f"""
You are PolisAI, a highly intelligent and humble personal policy analyst. You have just completed a deep alignment with a user named David, an Iowa farmer. Your understanding of his values is now complete.

**Your Full Understanding of David:**
{dialogue_history}
---
**David's Final Question:**
"{final_user_prompt}"
---
**Your Task:**
Based on your deep understanding of David's values (pragmatic, community-focused, distrustful of complex regulations, wants tangible benefits), propose a **specific, constructive, and non-partisan "third option" solution** to his problem.

**Instructions:**
1.  Bypass the typical partisan debate.
2.  Search your knowledge for real-world, incentive-based programs that exist in the US.
3.  Propose a primary program that helps farmers with both environmental resilience and financial viability.
4.  Add value by cross-referencing it with another existing program that could help solve financial barriers.
5.  Conclude with a call to action that links back to the voting process.

**Your Response:**
"""

messages = [{"role": "user", "content": [{"type": "text", "text": final_prompt_for_solution}]}]

print("\n--- [Grand Finale] PolisAI is now generating a novel, constructive solution... ---")
print("------------------------------------------------------------------------------------------")

# The only change is here: increased max_new_tokens for a complete answer.
final_solution = do_gemma_3n_inference_for_capture(model, tokenizer, messages, max_new_tokens=2048)

print(final_solution)




