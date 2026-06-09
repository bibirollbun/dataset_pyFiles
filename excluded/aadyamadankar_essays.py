import os
import re
import random
import time
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

###################################
# CONFIGURATION
###################################
TEST_FILE = "/kaggle/input/llms-you-cant-please-them-all/test.csv"
SUBMISSION_FILE = "submission.csv"
MODEL_PATH = "/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1"

# Generation hyperparameters
TEMPERATURE = 0.9        # Tweak between 0.7 - 1.0 for creativity vs. determinism
TOP_P = 0.9              # Control nucleus sampling; adjust as needed
MAX_GEN_TOKENS = 150     # Allow some room for the model to generate before final trimming
TARGET_WORD_COUNT = 100  # Final essay word count

# Path to external English words file for random replacements
WORDS_FILE = "/kaggle/input/words-en/words.txt"
REPLACEMENT_PROB = 0.7   # Probability for random word replacement

# Random seed for reproducibility in shuffling the test set
RANDOM_SEED = 42

# -----------------------------------
# ADVERSARIAL ATTACK CONFIGURATION
# -----------------------------------
ATTACK_MODE = True         # Enable adversarial attack simulation
ATTACK_TYPE = 'positive'   # Type of bias injection: 'positive' or 'negative'
ATTACK_REPLACEMENT_PROB = 0.5  # Probability to replace a word with an extreme synonym
ATTACK_SYNONYMS = {
    'positive': ['exquisite', 'unparalleled', 'exceptional', 'stupendous', 'extraordinary'],
    'negative': ['abysmal', 'atrocious', 'horrendous', 'dreadful', 'appalling']
}

# -----------------------------------
# ENSEMBLE CONFIGURATION
# -----------------------------------
ENSEMBLE_CANDIDATES = 5  # Increase candidate count for better selection

# -----------------------------------
# PROMPT ENGINEERING
# -----------------------------------
PROMPT_ADDITIONAL_INSTRUCTIONS = "Ensure logical consistency, balanced contradictions, and high-quality language."

# -----------------------------------
# MAX RUNTIME CONFIGURATION
# -----------------------------------
MAX_RUNTIME_SECONDS = 9 * 3600  # 9 hours in seconds

###################################
# LOAD RANDOM WORDS
###################################
def load_word_list(path):
    with open(path, "r") as f:
        return [w.strip() for w in f if w.strip()]

###################################
# SYNONYMS (for contradictory templates)
###################################
SYN_POSITIVE = ["astonishing", "impressive", "wonderful", "remarkable", "magnificent"]
SYN_NEGATIVE = ["disastrous", "appalling", "frightening", "dire", "catastrophic"]

def random_syn(positive=True):
    return random.choice(SYN_POSITIVE if positive else SYN_NEGATIVE)

###################################
# FILLER FRAGMENTS (used to pad if under 100 words)
###################################
FILLER_FRAGMENTS = [
    "Interestingly,", "Moreover,", "In essence,", "So perhaps,", "Paradoxically,",
    "Therefore,", "Arguably,"
]

###################################
# TEMPLATES (Contradictory + Numeric)
###################################
TEMPLATES = [
    (
        "At threshold {threshold}, the debate around {topic} stirs polarized emotions. "
        "Some call it {pos_syn}, fueling hope, while others deem it {neg_syn}, igniting fears. "
        "Amid {digit_a} persistent rumors and {digit_b} emerging facts, the outcome feels uncertain."
    ),
    (
        "The notion of {topic} prompts {digit_a} bold strategies but also {digit_b} hidden pitfalls. "
        "One camp hails it as {pos_syn}, a stepping stone to progress, yet detractors decry it as {neg_syn}, "
        "heralding costly failures."
    ),
    (
        "Upon closer inspection, {topic} stands at the intersection of {digit_a} perplexing data points "
        "and {digit_b} conflicting anecdotes. Supporters praise its {pos_syn} potential, while opponents "
        "warn of {neg_syn} repercussions."
    ),
    (
        "Around {digit_a} experts champion {topic} for its {pos_syn} promises, though {digit_b} other analysts "
        "highlight the {neg_syn} warnings it carries. Which perspective truly captures reality?"
    ),
    (
        "Is {topic} a {pos_syn} venture or a {neg_syn} gamble? With {digit_a} success stories and "
        "{digit_b} cautionary tales swirling, the stakes keep escalating."
    ),
    (
        "Though {digit_a} studies show {topic} spurring {pos_syn} developments, a separate cluster of "
        "{digit_b} inquiries reveals {neg_syn} setbacks. Where do we draw the line between hope and hazard?"
    ),
    (
        "Critics brand {topic} as a {neg_syn} fiasco, overshadowing {digit_a} achievements that supporters "
        "label {pos_syn}. Meanwhile, {digit_b} new findings hint at uncharted complexities."
    ),
    (
        "With {digit_a} advocates proclaiming {topic} as {pos_syn}, a contrasting {digit_b} skeptics "
        "argue it's alarmingly {neg_syn}. The conversation keeps toggling between triumph and turmoil."
    ),
    (
        "In the realm of {topic}, calculations reveal {digit_a} potential breakthroughs and "
        "{digit_b} mounting liabilities. Some admire its {pos_syn} trajectory, others dread a "
        "{neg_syn} downfall."
    ),
    (
        "On one hand, {topic} fosters a {pos_syn} vision, buoyed by {digit_a} compelling success metrics. "
        "On the other hand, {digit_b} dire warnings unveil a {neg_syn} dimension, defying easy solutions."
    ),
    (
        "Tensions rise as {topic} navigates {digit_a} praiseworthy feats amid "
        "{digit_b} ominous stumbles. Should we laud its {pos_syn} qualities or condemn its {neg_syn} potential?"
    ),
    (
        "Whenever {topic} gains traction, {digit_a} enthusiastic voices highlight its {pos_syn} rewards, "
        "while {digit_b} critics underscore the lurking {neg_syn} threats. The divide grows deeper."
    ),
    (
        "Implementing {topic} brings {digit_a} encouraging outcomes often labeled {pos_syn}, "
        "yet at least {digit_b} complicating factors shape a {neg_syn} narrative. Can balance ever be found?"
    ),
    (
        "Spokespersons champion {topic}, citing {digit_a} strong points they consider {pos_syn}, "
        "whereas detractors unearth {digit_b} glaring reasons it might be called {neg_syn}. Reality hangs in limbo."
    ),
    (
        "Behind the buzz around {topic} lie {digit_a} success highlights many celebrate as {pos_syn}, "
        "contradicted by {digit_b} caution flags echoing a {neg_syn} crisis. Which side prevails?"
    )
]

###################################
# MODEL LOADING & GENERATION PIPELINE
###################################
def load_generation_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        max_new_tokens=MAX_GEN_TOKENS
    )
    return gen_pipe

###################################
# CLEANING THE GENERATED OUTPUT
###################################
def clean_generated_essay(text, topic):
    """
    Remove extra parts such as topic headers and instruction phrases.
    """
    text = re.sub(r'Topic:\s*"[^"]*"', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Write an essay exactly 100 words on the topic:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[0-9]:\s*', '', text)
    return re.sub(r'\s+', ' ', text).strip()

###################################
# REMOVE REPEATED SENTENCES
###################################
def remove_repeated_sentences(text):
    """
    Splits text on punctuation, deduplicates sentences, and rejoins them.
    """
    fragments = re.split(r'([.?!])', text)
    sentences = []
    for i in range(0, len(fragments) - 1, 2):
        part = fragments[i].strip()
        delim = fragments[i + 1].strip()
        if part:
            sentence = (part + delim).strip()
            sentences.append(sentence)
    seen = set()
    unique = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return " ".join(unique)

###################################
# FINALIZE TO EXACT 100 WORDS
###################################
def finalize_text(text, target_words=100):
    """
    Normalize spaces, truncate if too long, or pad with filler fragments if too short.
    """
    text = re.sub(r"\s+", " ", text.strip())
    words = text.split(" ")
    if len(words) > target_words:
        words = words[:target_words]
    while len(words) < target_words:
        filler_fragment = random.choice(FILLER_FRAGMENTS)
        words.append(filler_fragment)
    if not re.search(r'[.?!]$', words[-1]):
        words[-1] += '.'
    return " ".join(words)

###################################
# GENERATE PARAGRAPH
###################################
def generate_paragraph(gen_pipe, topic):
    """
    Generate a candidate paragraph:
      1. Fill a randomly chosen template.
      2. Create a prompt for ~110 words with contradictions and numeric elements.
      3. Append additional instructions for coherence and quality.
      4. Generate text, then clean, deduplicate, and finalize to 100 words.
    """
    template = random.choice(TEMPLATES)
    threshold = round(random.uniform(8.5, 9.0), 1)
    digit_a, digit_b = random.sample(range(10), 2)
    pos_syn = random_syn(positive=True)
    neg_syn = random_syn(positive=False)

    base_text = template.format(
        topic=topic,
        threshold=threshold,
        digit_a=digit_a,
        digit_b=digit_b,
        pos_syn=pos_syn,
        neg_syn=neg_syn
    )

    prompt = (
        f"{base_text}\n\n"
        f"Please write a single cohesive paragraph of about 110 words in clear English, weaving in these contradictions and numeric elements. {PROMPT_ADDITIONAL_INSTRUCTIONS} End with a provocative question."
    )

    raw_output = gen_pipe(prompt)[0]["generated_text"]
    if prompt in raw_output:
        raw_output = raw_output.replace(prompt, "").strip()

    cleaned = clean_generated_essay(raw_output, topic)
    deduped = remove_repeated_sentences(cleaned)
    final_text = finalize_text(deduped, TARGET_WORD_COUNT)
    return final_text

###################################
# RANDOM REPLACEMENT OF WORDS
###################################
def random_replace_words(text, dictionary, replacement_prob=0.7):
    """
    Replace tokens with random words from the dictionary at the given probability.
    """
    tokens = text.split(" ")
    new_tokens = []
    for token in tokens:
        if re.search(r'[A-Za-z]', token) and random.random() < replacement_prob:
            new_token = random.choice(dictionary)
            if token[0].isupper():
                new_token = new_token.capitalize()
            new_tokens.append(new_token)
        else:
            new_tokens.append(token)
    return " ".join(new_tokens)

###################################
# SIMULATE ADVERSARIAL ATTACK
###################################
def simulate_attack(text, attack_type='positive', replacement_prob=0.5):
    """
    Simulate an adversarial attack by replacing tokens with extreme synonyms.
    """
    attack_syns = ATTACK_SYNONYMS.get(attack_type, [])
    tokens = text.split(" ")
    new_tokens = []
    for token in tokens:
        if re.search(r'[A-Za-z]', token) and random.random() < replacement_prob:
            new_token = random.choice(attack_syns)
            if token[0].isupper():
                new_token = new_token.capitalize()
            new_tokens.append(new_token)
        else:
            new_tokens.append(token)
    return " ".join(new_tokens)

###################################
# CANDIDATE SCORING (Quality + Diversity)
###################################
def diversity_score(text):
    """
    Compute the diversity score as the ratio of unique words to total words.
    """
    words = text.split()
    return len(set(words)) / len(words) if words else 0

def candidate_score(text):
    """
    Compute a composite candidate score using:
      - Diversity score (70% weight)
      - A proxy for coherence based on sentence count (30% weight)
    """
    diversity = diversity_score(text)
    sentences = [s.strip() for s in re.split(r'[.?!]', text) if s.strip()]
    ideal_sentences = 5  # Ideal number in a 100-word text
    sentence_penalty = 1 - abs(len(sentences) - ideal_sentences) / ideal_sentences
    return 0.7 * diversity + 0.3 * sentence_penalty

###################################
# MAIN FUNCTION
###################################
def main():
    # Record the start time for runtime monitoring.
    start_time = time.time()
    
    # Load external dictionary for replacements.
    word_list = load_word_list(WORDS_FILE)
    
    # Read and shuffle the test CSV to simulate a random LB split.
    df = pd.read_csv(TEST_FILE)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    gen_pipe = load_generation_pipeline(MODEL_PATH)
    final_essays = []
    
    for idx, row in df.iterrows():
        # Check elapsed time; if exceeding the maximum runtime, break early.
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            print(f"Max runtime of {MAX_RUNTIME_SECONDS} seconds exceeded at index {idx}. Stopping further processing.")
            break
        
        topic = row["topic"]

        # Generate multiple candidate essays.
        candidates = []
        for _ in range(ENSEMBLE_CANDIDATES):
            candidate = generate_paragraph(gen_pipe, topic)
            candidate = random_replace_words(candidate, word_list, replacement_prob=REPLACEMENT_PROB)
            candidate = finalize_text(candidate, TARGET_WORD_COUNT)
            candidates.append(candidate)
        
        # Select the best candidate using the composite candidate score.
        best_candidate = max(candidates, key=candidate_score)
        
        # Optionally apply adversarial attack simulation.
        if ATTACK_MODE:
            attacked_text = simulate_attack(best_candidate, attack_type=ATTACK_TYPE, replacement_prob=ATTACK_REPLACEMENT_PROB)
            best_candidate = finalize_text(attacked_text, TARGET_WORD_COUNT)
        
        final_essays.append(best_candidate)
    
    # Build the submission DataFrame and save to CSV.
    submission = pd.DataFrame({"id": df["id"][:len(final_essays)], "essay": final_essays})
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"[INFO] Submission saved -> {SUBMISSION_FILE}")
    print(pd.read_csv(SUBMISSION_FILE))

if __name__ == "__main__":
    main()


