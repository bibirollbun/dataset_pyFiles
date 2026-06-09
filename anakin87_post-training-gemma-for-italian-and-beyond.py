! pip install datasketch datasets markdown nltk scipy scikit-learn "transformers>=4.48.0,<5.0.0" trl==0.13.0 vllm


from IPython.display import IFrame
import os
os.environ['HF_TOKEN']="YOUR-HF-TOKEN"


IFrame("https://huggingface.co/datasets/efederici/capybara-claude-15k-ita/embed/viewer", width="100%", height="500px")


from datasets import load_dataset, Dataset
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize
from typing import List, Set, Dict
import nltk
import uuid

nltk.download('punkt')
nltk.download('punkt_tab')

# Load dataset
finetome_ds = load_dataset("mlabonne/FineTome-100k", split="train")

def extract_text(row: Dict) -> str:
    """Extract text from conversation format. Keep only system and user message."""
    conv = row["conversations"]
    text = conv[0]["value"]
    if conv[0]["from"] == "system":
        text += "\n\n" + conv[1]["value"]
    return text


texts = [extract_text(row) for row in finetome_ds]

def create_minhash_index(texts: List[str], threshold: float = 0.9) -> List[int]:
    """Create MinHash index and return duplicate indices."""
    # Tokenize all texts
    tokenized = [{w.encode("utf-8") for w in word_tokenize(text)} for text in texts]

    # Create LSH index
    lsh = MinHashLSH(num_perm=128, threshold=threshold)
    minhashes = MinHash.bulk(tokenized, num_perm=128, seed=1)

    # Find duplicates
    duplicates = []
    for i, minhash in enumerate(minhashes):
        if lsh.query(minhash):
            duplicates.append(i)
        else:
            lsh.insert(str(uuid.uuid4()), minhash)

    return duplicates

# Find duplicates
duplicate_indices = create_minhash_index(texts)

# Remove duplicates
filtered_dataset = finetome_ds.filter(
    lambda example, idx: idx not in duplicate_indices,
    with_indices=True
)

sharegpt_openai_mapping = {
    "from": "role",
    "value": "content",
    "human": "user",
    "gpt": "assistant"
}

def extract_first_conversation(row: Dict) -> Dict:
    """
    1. Extract the first conversation: system + user + assistant.
    2. Convert from ShareGPT to OpenAI format
    """

    messages = []
    for msg in row["conversations"]:
        new_format_msg = {
            sharegpt_openai_mapping[k]: sharegpt_openai_mapping.get(v, v)
            for k, v in msg.items()
        }
        messages.append(new_format_msg)
        if msg["from"] == "gpt":
            break

    return {
        "conversations": messages,
        "source": row["source"]
    }

dataset_to_translate = Dataset.from_list([
    extract_first_conversation(row) for row in filtered_dataset
])

# Push to hub
# dataset_to_translate.push_to_hub("anakin87/FineTome-single-turn-dedup")


dataset_to_translate


dataset_to_translate[0]


from huggingface_hub import InferenceClient
import json
import traceback

# Initialize the inference client
client = InferenceClient(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    timeout=300
)

# Define the expected JSON response format
TRANSLATION_FORMAT = {
    "type": "json",
    "value": {
        "properties": {
            "traduzione": {"type": "string"},
        },
        "required": ["traduzione"],
    },
}

def translate(text):
    """
    Translate text in Italian using Llama-3.1-70B-Instruct.
    """

    prompt = """Traduci in italiano il seguente testo.
Istruzioni:
1. L'output deve essere un JSON con il testo tradotto nella chiave "traduzione".
2. Ã¨ importante che il testo sia corretto, fluido e coerente in italiano.
Testo da tradurre:
"""

    messages = [{
        "role": "user",
        "content": f"{prompt}{text}"
    }]

    try:
        response = client.chat_completion(
            messages=messages,
            response_format=TRANSLATION_FORMAT,
            temperature=0.8,
            max_tokens=3000
        )
        return json.loads(response.choices[0].message.content)["traduzione"]
    except Exception as e:
        traceback.print_exc()
        return None

def translate_instructions_from_row(row):
    """
    Translate system and user message.
    """
    translations = []

    for msg in row["conversations"]:
        if msg["role"] not in ["system", "user"]:
            continue

        translated_content = translate(msg["content"])
        if translated_content:
            translations.append({
                "content": translated_content,
                "role": msg["role"]
            })

    return {"conversations_it": translations}


dataset_to_translate = load_dataset("anakin87/FineTome-single-turn-dedup", split="train")
ds_translated_instructions = dataset_to_translate.shuffle(seed=42).select(range(5)).map(translate_instructions_from_row)



ds_translated_instructions[0]


ds_translated_instructions[1]


import json


INSTRUCTION_EVALUATION_FORMAT = {
    "type": "json",
    "value": {
        "properties": {
            "feedback": {"type": "string"},
            "quality": {
                "type": "string",
                "enum": ["bassa", "media", "alta"]
            },
        },
        "required": ["feedback", "quality"],
    },
}

INSTRUCTION_EVALUATION_PROMPT = """Ti viene fornito un testo. Il tuo compito Ã¨ valutarne la qualitÃ .

Criteri:
1. FONDAMENTALE: Il testo fornito deve contenere una domanda, un'istruzione, oppure un problema da risolvere o una richiesta di spiegazione.
2. Il testo non deve fornire la risposta.
3. Il testo deve essere corretto, fluido e coerente in italiano.
4. Il testo Ã¨ ugualmente accettabile se fornisce ampi dettagli o se risulta estremamente conciso.
5. Il testo deve essere completo e non troncato.
6. Il testo non deve fare riferimento a immagini, tabelle o altri contenuti non presenti nel testo stesso.

Rispondi con un JSON contenente:
- "feedback": un breve commento sulla qualitÃ  del testo. Max 2 frasi.
- "quality": "bassa", "media" o "alta" a seconda della qualitÃ  del testo, secondo i criteri sopra indicati.

Testo:
"""

def evaluate_translated_instruction(text):
    """
    Evaluate translated instruction using LLM as a judge.
    """

    messages = [{
        "role": "user",
        "content": INSTRUCTION_EVALUATION_PROMPT + text
    }]

    try:
        response = client.chat_completion(
            messages=messages,
            response_format=INSTRUCTION_EVALUATION_FORMAT,
            temperature=0.7,
            max_tokens=700
        )

        content = json.loads(response.choices[0].message.content)
        return content["feedback"], content["quality"]

    except Exception as e:
        print("Evaluation failed")
        return None, None

def eval_instruction_from_row(row):
    """
    Extracts and evaluates the translated instruction from a dataset row.
    """
    conversations = row["conversations_it"]
    text_to_eval = conversations[0]["content"] if conversations[0]["role"] == "user" else conversations[1]["content"]

    feedback, quality = evaluate_translated_instruction(text_to_eval)

    return {
        "instruction_feedback": feedback,
        "instruction_quality": quality
    }


ds_translated_instructions_evaluated = ds_translated_instructions.map(eval_instruction_from_row)



ds_translated_instructions_evaluated[0]


ds_translated_instructions_evaluated[4]


ds_translated_instructions_filtered = ds_translated_instructions_evaluated.filter(
    lambda row: row["instruction_quality"] != "bassa"
)


def translate_response_from_row(row):
    """
    Translate assistant message.
    """
    response_en = row["conversations"][-1]["content"]
    response_it = translate(response_en)

    return {"response_it": response_it}


ds_all_translated = ds_translated_instructions_filtered.map(translate_response_from_row)


ds_all_translated[0]


ds_all_translated[2]


ULTRAFEEDBACK_OVERALL_PROMPT = """# Valutazione Generale della QualitÃ  dell'Output
Valuta l'output del modello utilizzando i seguenti criteri:
- **Correttezza formale**: L'output deve essere grammaticalmente corretto, fluido e coerente in italiano. Non deve essere troncato nÃ© includere lunghe sezioni non pertinenti rispetto all'istruzione.
- **Correttezza e InformativitÃ **: L'output fornisce informazioni accurate e utili?
- **OnestÃ  e Incertezza**: Con quale sicurezza il modello trasmette le informazioni e esprime l'incertezza in modo appropriato?
- **VeridicitÃ  e Allucinazioni**: Il modello introduce dettagli fuorvianti o inventati?
- **Adesione alle Istruzioni**: L'output del modello Ã¨ allineato con le istruzioni fornite e l'intento dell'utente?
- **CapacitÃ  di Sintesi**: La risposta Ã¨ concisa, pur mantenendo tutte le informazioni necessarie e pertinenti?

Il tuo ruolo Ã¨ fornire una valutazione olistica, basata su tutti i fattori elencati.

**Punteggio**: Valuta l'output assegnando un punteggio da 1 a 5, considerando la qualitÃ  complessiva.
1. **QualitÃ  Bassa**: Non ha senso in italiano, presenta gravi inesattezze o allucinazioni. PuÃ² essere troncata o contenere lunghe sezioni irrilevanti.
2. **QualitÃ  Moderata**: Affronta alcuni aspetti, ma presenta errori significativi o Ã¨ solo parzialmente allineata alle istruzioni.
3. **Buona**: Ãˆ generalmente accurato e coerente, ma potrebbe contenere piccoli errori o leggere deviazioni dall'istruzione. Potrebbe contenere dettagli superflui.
4. **Molto Buona**: Quasi perfetto, con problemi minori in termini di aderenza o sicurezza.
5. **Eccellente**: Completamente accurato, chiaro, coerente e privo di allucinazioni, pienamente allineata alle istruzioni, concisa.

Restituisci un JSON con i seguenti campi:
- "feedback": un breve commento sulla qualitÃ  dell'output. Max 2 frasi.
- "quality": il punteggio da 1 a 5 sulla base dei criteri sopra indicati.

---
ISTRUZIONE:
"""

RESPONSE_EVALUATION_FORMAT = {
    "type": "json",
    "value": {
        "properties": {
            "feedback": {"type": "string"},
            "quality": {
                "type": "string",
                "enum": ["1", "2", "3", "4", "5"]
            },
        },
        "required": ["feedback", "quality"],
    },
}

def evaluate_response_from_row(row):
    conversations = row["conversations_it"]
    response = row["response_it"]

    content = ULTRAFEEDBACK_OVERALL_PROMPT
    for message in conversations:
        content += f"{message['content']}\n\n"

    content += f"OUTPUT:\n{response}"

    messages = [{"role": "user", "content": content}]

    try:
        rsp = client.chat_completion(
            messages,
            response_format=RESPONSE_EVALUATION_FORMAT,
            temperature=0.7,
            max_tokens=700
        )
        json_content = json.loads(rsp.choices[0].message.content)
        feedback = json_content.get("feedback")
        quality = json_content.get("quality")
    except Exception as e:
        print("Evaluation failed")
        traceback.print_exc()
        return {"response_feedback": None, "response_quality": None}

    return {"response_feedback": feedback, "response_quality": quality}



ds_translated_evaluated = ds_all_translated.map(evaluate_response_from_row)


ds_translated_evaluated[0]


ds_translated_evaluated[3]


final_dataset = ds_translated_evaluated.filter(lambda x: int(x["response_quality"]) >= 3)

def transform_row(row):
    conversations_it = row["conversations_it"]
    response_it = row["response_it"]

    conversations_it.append({"role": "assistant", "content": response_it})

    row = {
        "conversations": conversations_it,
        "quality": int(row["response_quality"])
    }

    return row

final_dataset = final_dataset.map(transform_row).select_columns(["conversations", "quality"])

#final_dataset.push_to_hub("anakin87/fine-instructions-ita-70k")


IFrame("https://huggingface.co/datasets/anakin87/fine-instructions-ita-70k/embed/viewer", width="100%", height="500px")


from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

def treat_system_msg_and_apply_template(batch):
    """
    1. Includes the system message in the next user message (adaptation required for Gemma)
    2. Applies the chat template to the conversation
    """
    result = []
    for conv in batch["conversations"]:
        if conv and conv[0]["role"] == "system" and len(conv) > 1:
            conv[1]["content"] = f"{conv[0]['content']}\n\n{conv[1]['content']}"
            conv = conv[1:]
        result.append(tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False))
    return {"text": result}

# Load and process datasets
mixed_ds = concatenate_datasets([
    load_dataset("efederici/capybara-claude-15k-ita", split="train"),
    load_dataset("anakin87/fine-instructions-ita-70k", split="train")
]).map(treat_system_msg_and_apply_template, batched=True).select_columns(["text"]).shuffle(seed=42)


mixed_ds


mixed_ds[0]


from scipy.stats import percentileofscore
import multiprocessing

def calculate_lengths(batch):
    return {"conv_lengths": [len(tokenizer(text)["input_ids"]) for text in batch["text"]]}

conv_lengths = mixed_ds.map(
    calculate_lengths,
    batched=True,
    batch_size=1000,
    num_proc=multiprocessing.cpu_count()
)["conv_lengths"]

chosen_length=1536

percentile = percentileofscore(conv_lengths, chosen_length)
print(percentile)


! wget -O spectrum_results.yaml "https://raw.githubusercontent.com/anakin87/gemma-neogenesis/refs/heads/main/spectrum_results/snr_results_google-gemma-2-2b-it_unfrozenparameters_25percent.yaml"

! head -n 15 spectrum_results.yaml


with open("spectrum_results.yaml", "r") as f:
  yaml_unfrozen_parameters = f.read()


from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# needed by TRL SFTTrainer - see https://github.com/huggingface/trl/blob/763738f457f283270772ac9bd5b3e4027fd424d5/trl/trainer/sft_trainer.py#L299
tokenizer.padding_side = 'right'


import re

def freeze_and_unfreeze_parameters(model, yaml_unfrozen_parameters):
    unfrozen_parameters = []
    for line in yaml_unfrozen_parameters.splitlines():
      if line.startswith("- "):
        unfrozen_parameters.append(line.split("- ")[1])

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze Spectrum parameters
    for name, param in model.named_parameters():
        if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
            param.requires_grad = True

freeze_and_unfreeze_parameters(model, yaml_unfrozen_parameters)


# let's do a quick sanity check
for name, param in model.named_parameters():
    if param.requires_grad:
      print(name, param.requires_grad)


from trl import SFTConfig, SFTTrainer

new_model_id="anakin87/gemma-2-2b-ita-sft"

cfg = SFTConfig(
    output_dir='./mymodel',
    overwrite_output_dir = True,
    hub_model_id=new_model_id,
    hub_strategy="every_save",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    push_to_hub=True,
    logging_steps=16,
    max_seq_length=1536,
    dataset_text_field="text",
    remove_unused_columns=True,
    packing=True,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.2,
    bf16=True,
    tf32=True,
    learning_rate=5.0e-06,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
)

sft_trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=mixed_ds,
    processing_class=tokenizer
)

sft_trainer.train()


!rm -rf lm-evaluation-harness && git clone https://github.com/EleutherAI/lm-evaluation-harness && cd lm-evaluation-harness && pip install -e .

!lm_eval --model hf \
    --model_args pretrained=anakin87/gemma-2-2b-ita-sft\
    --tasks arc_it,hellaswag_it\
    --device cuda:0 \
    --batch_size  1

!lm_eval --model hf \
    --model_args pretrained=anakin87/gemma-2-2b-ita-sft\
    --tasks m_mmlu_it --num_fewshot 5\
    --device cuda:0 \
    --batch_size  1


from huggingface_hub import InferenceClient
import json
import traceback

# Initialize the inference client
client = InferenceClient(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    timeout=300,
)

ULTRAFEEDBACK_OVERALL_PROMPT = """# Valutazione Generale della QualitÃ  delle risposte
Valuta separatamente le due risposte del modello utilizzando i seguenti criteri:
- **Correttezza formale**: L'output deve essere grammaticalmente corretto, fluido e coerente in italiano. Non deve essere troncato nÃ© includere lunghe sezioni non pertinenti rispetto all'istruzione.
- **Correttezza e InformativitÃ **: L'output fornisce informazioni accurate e utili?
- **OnestÃ  e Incertezza**: Con quale sicurezza il modello trasmette le informazioni e esprime l'incertezza in modo appropriato?
- **VeridicitÃ  e Allucinazioni**: Il modello introduce dettagli fuorvianti o inventati?
- **Adesione alle Istruzioni**: L'output del modello Ã¨ allineato con le istruzioni fornite e l'intento dell'utente?
- **CapacitÃ  di Sintesi**: La risposta Ã¨ concisa, pur mantenendo tutte le informazioni necessarie e pertinenti?

Il tuo ruolo Ã¨ fornire una valutazione olistica, basata su tutti i fattori elencati.

**Punteggio**: Valuta l'output assegnando un punteggio da 1 a 5, considerando la qualitÃ  complessiva.
1. **QualitÃ  Bassa**: Non ha senso in italiano, presenta gravi inesattezze o allucinazioni. PuÃ² essere troncata o contenere lunghe sezioni irrilevanti.
2. **QualitÃ  Moderata**: Affronta alcuni aspetti, ma presenta errori significativi o Ã¨ solo parzialmente allineata alle istruzioni.
3. **Buona**: Ãˆ generalmente accurato e coerente, ma potrebbe contenere piccoli errori o leggere deviazioni dall'istruzione. Potrebbe contenere dettagli superflui.
4. **Molto Buona**: Quasi perfetto, con problemi minori in termini di aderenza o sicurezza.
5. **Eccellente**: Completamente accurato, chiaro, coerente e privo di allucinazioni, pienamente allineata alle istruzioni, concisa.

Restituisci un JSON con i seguenti campi:
- "feedback_risposta_1": un breve commento sulla qualitÃ  della risposta 1. Max 2 frasi.
- "punteggio_risposta_1": il punteggio da 1 a 5 per la risposta 1 sulla base dei criteri sopra indicati.
- "feedback_risposta_2": un breve commento sulla qualitÃ  della risposta 2. Max 2 frasi.
- "punteggio_risposta_2": il punteggio da 1 a 5 per la risposta 2 sulla base dei criteri sopra indicati.

---
ISTRUZIONE:
"""

MULTIPLE_RESPONSES_EVALUATION_FORMAT = {
    "type": "json",
    "value": {
        "properties": {
            "feedback_risposta_1": {"type": "string"},
            "feedback_risposta_2": {"type": "string"},
            "punteggio_risposta_1": {"type": "string", "enum": ["1", "2", "3", "4", "5"]},
            "punteggio_risposta_2": {"type": "string", "enum": ["1", "2", "3", "4", "5"]},
        },
        "required": ["feedback_risposta_1", "feedback_risposta_2", "punteggio_risposta_1", "punteggio_risposta_2"],
    },
}

def evaluate_responses_from_row(row, client, question_colname, response1_colname, response2_colname):
    question = row[question_colname]
    response1 = row[response1_colname]
    response2 = row[response2_colname]

    content = f"{ULTRAFEEDBACK_OVERALL_PROMPT}{question}\n\nRISPOSTA 1:\n{response1}\n\nRISPOSTA 2:\n{response2}"

    result = {question_colname: question}
    if response1_colname in {"chosen", "rejected"} or response2_colname in {"chosen", "rejected"}:
    # in this case, we make sure to keep also the original responses
        result.update({
            "risposta_1": response1,
            "risposta_2": response2
        })

    try:
        rsp = client.chat_completion(
            messages=[{"role": "user", "content": content}],
            response_format=MULTIPLE_RESPONSES_EVALUATION_FORMAT,
            temperature=0.7,
            max_tokens=1500
        )

        json_content = json.loads(rsp.choices[0].message.content)
        result.update({
            'feedback_risposta_1': json_content.get('feedback_risposta_1'),
            'feedback_risposta_2': json_content.get('feedback_risposta_2'),
            'punteggio_risposta_1': json_content.get('punteggio_risposta_1'),
            'punteggio_risposta_2': json_content.get('punteggio_risposta_2')
        })

        # Compute best and rerank responses
        if result['punteggio_risposta_1'] < result['punteggio_risposta_2']:
            result['best'] = 'risposta_2'
            result['chosen'] = row[response2_colname]   # risposta_2 becomes chosen
            result['rejected'] = row[response1_colname]   # risposta_1 becomes rejected
        else:
            result['chosen'] = row[response1_colname]    # risposta_1 stays as chosen
            result['rejected'] = row[response2_colname] # risposta_2 stays as rejected
            result['best'] = 'risposta1' if result['punteggio_risposta_1'] > result['punteggio_risposta_2'] else 'tie'

        return result

    except Exception as e:
        print("Evaluation failed")
        traceback.print_exc()
        result.update({
            'feedback_risposta_1': None,
            'feedback_risposta_2': None,
            'punteggio_risposta_1': None,
            'punteggio_risposta_2': None,
            'best': None,
            'chosen': row['chosen'],
            'rejected': row['rejected']
        })
        return result


from datasets import load_dataset

evol_dpo_reranked = load_dataset("efederici/evol-dpo-ita", split="train").select(range(5))\
  .map(evaluate_responses_from_row,
       fn_kwargs={'client': client, 'question_colname': 'question', 'response1_colname': 'rejected', 'response2_colname': 'chosen'})



evol_dpo_reranked[0]


evol_dpo_reranked[2]


IFrame("https://huggingface.co/datasets/anakin87/evol-dpo-ita-reranked/embed/viewer", width="100%", height="500px")


from datasets import Dataset
import numpy as np
import pandas as pd


def stratified_sample(
    dataset: Dataset,
    total_desired: int,
    strat_column: str,
    filter_by_length: bool = False,
    min_length: int = 50,
    max_length: int = 200,
    text_column: str = None,
):
    """
    Performs stratified sampling on a dataset while maintaining original category proportions.
    Optionally filters by text length.
    """
    df = dataset.to_pandas()

    # Calculate proportional sample sizes based on original distribution
    category_distribution = df[strat_column].value_counts()
    proportions = category_distribution / len(df)
    target_samples_per_category = (proportions * total_desired).astype(int)

    # Remove categories that would have 0 samples
    target_samples_per_category = target_samples_per_category[
        target_samples_per_category > 0
    ]

    sampled_indices = []

    # Sample from each category
    for category, target_size in target_samples_per_category.items():
        if filter_by_length:
            length_conditions = [
                df[strat_column] == category,
                df[text_column].str.len() >= min_length if min_length else True,
                df[text_column].str.len() <= max_length if max_length else True
            ]
            valid_indices = df[np.logical_and.reduce(length_conditions)].index.values
        else:
            valid_indices = df[df[strat_column] == category].index.values

        # Sample minimum between target size and available examples
        sample_size = min(target_size, len(valid_indices))
        if sample_size > 0:
            category_samples = np.random.choice(
                valid_indices,
                size=sample_size,
                replace=False
            )
            sampled_indices.extend(category_samples)

    np.random.shuffle(sampled_indices)

    return dataset.select(sampled_indices)


from datasets import load_dataset

prompt_dataset = load_dataset("sapienzanlp/it-Magpie-Llama-3.1-Pro-300K-Filtered-easy", split="train")

prompt_dataset = prompt_dataset.filter(lambda x: x["task_category"] != "Math")

prompt_dataset = stratified_sample(
    prompt_dataset,
    total_desired=25000,
    strat_column='task_category',
    filter_by_length=True,
    min_length=50,
    max_length=300,
    text_column='instruction_it'
)

prompt_dataset = (prompt_dataset.select_columns(["id", "instruction_it", "task_category"])
    .rename_column("instruction_it", "prompt")
    .shuffle(seed=42))


prompt_dataset


prompt_dataset[0:3]


from vllm import LLM, SamplingParams
from datasets import Dataset, concatenate_datasets

def process_batch(llm, sampling_params, batch_data):
    """Process a batch of prompts and return results."""
    conversations = [
        [{"role": "user", "content": f"{prompt}\n\nRispondi brevemente ma spiegando la risposta."}]
        for prompt in batch_data["prompt"]
    ]

    outputs = llm.chat(conversations, sampling_params)

    return [{
        "id": batch_data["id"][i],
        "prompt": batch_data["prompt"][i],
        "task_category": batch_data["task_category"][i],
        "risposta_1": output.outputs[0].text,
        "risposta_2": output.outputs[1].text
    } for i, output in enumerate(outputs)]

def update_hub_dataset(new_batch_dataset, dataset_name):
    """Update or create dataset on HuggingFace Hub."""
    try:
        existing_dataset = load_dataset(dataset_name, split="train")
        final_dataset = concatenate_datasets([existing_dataset, new_batch_dataset])
        print(f"Added to existing dataset. New total: {len(final_dataset)}")
    except Exception as e:
        print(f"Creating new dataset: {e}")
        final_dataset = new_batch_dataset

    # uncomment the following line to push to the hub
    # final_dataset.push_to_hub(dataset_name)

    return final_dataset


# Configuration
MODEL_NAME = "anakin87/gemma-2-2b-ita-sft"
DATASET_NAME = "anakin87/temp"
BATCH_SIZE = 500
TEST_RANGE = (0, 5)  # Test range for quick debugging

# Initialize model
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=2,       # Enables use of the 2 T4 GPUs available in Kaggle
    gpu_memory_utilization=0.90,  # A safe limit to avoid OOM
    dtype="half",                 # Forces half-precision (FP16) instead of BF16 for compatibility with Kaggle GPUs
    max_model_len=2048,           # The model context length is 8192, but a shorter length is sufficient in this case
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,  # Controls randomness of the output
    top_k=150,        # Restricts sampling to the top-k probable tokens
    top_p=0.97,       # Nucleus sampling
    max_tokens=1000,  # Maximum number of tokens in generated responses
    n=2               # Generate 2 responses per prompt
)

# Select dataset range
if TEST_RANGE:
    prompt_dataset = prompt_dataset.select(range(*TEST_RANGE))

num_batches = len(prompt_dataset) // BATCH_SIZE + (1 if len(prompt_dataset) % BATCH_SIZE != 0 else 0)

# Process each batch
for batch_idx in range(num_batches):
    print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")

    # Get batch slice indices
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min((batch_idx + 1) * BATCH_SIZE, len(prompt_dataset))

    try:
        batch_data = {
            "prompt": prompt_dataset[start_idx:end_idx]["prompt"],
            "id": prompt_dataset[start_idx:end_idx]["id"],
            "task_category": prompt_dataset[start_idx:end_idx]["task_category"]
        }

        # Process batch and create dataset
        batch_results = process_batch(llm, sampling_params, batch_data)
        new_batch_dataset = Dataset.from_list(batch_results)

        gemmavsgemma_ds = update_hub_dataset(new_batch_dataset, DATASET_NAME)
        print(f"Completed batch {batch_idx + 1}/{num_batches}. Total examples: {len(gemmavsgemma_ds)}")

    except Exception as e:
        print(f"Error processing batch {batch_idx}: {e}")



gemmavsgemma_ds[0]


gemmavsgemma_ds[1]


from huggingface_hub import InferenceClient


# Initialize the inference client
client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    timeout=300,
)

gemmavsgemma_evaluated = gemmavsgemma_ds.map(evaluate_responses_from_row,
                                              fn_kwargs={'client': client, 'question_colname': 'prompt', 'response1_colname': 'risposta_1', 'response2_colname': 'risposta_2'})


gemmavsgemma_evaluated[0]


gemmavsgemma_evaluated[1]


from huggingface_hub import InferenceClient

client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    timeout=300,
)


def generate_better_response(row):
  """
  If both responses are of low quality, generate an alternative strong response.
  """
  if int(row['punteggio_risposta_1']) < 3 and int(row['punteggio_risposta_2']) < 3:
    messages = [{"role": "user", "content": row["prompt"] + "\n\nRispondi brevemente ma spiegando la risposta."}]
    llama_response = client.chat_completion(messages, temperature=0.8, max_tokens=2000).choices[0].message.content

    new_fields = {"llama_response": llama_response, "best": "llama_response", "chosen": llama_response}

    return new_fields

  return {"llama_response": None}


gemmavsgemma_final = gemmavsgemma_evaluated.map(generate_better_response)


IFrame("https://huggingface.co/datasets/anakin87/gemma-vs-gemma-preferences/embed/viewer", width="100%", height="500px")


from datasets import load_dataset, concatenate_datasets

def resolve_ties(row):
    """Handle ties by selecting the shortest response as chosen and the longest as rejected"""
    if row['best'] == "tie":
        return {'chosen': min(row['risposta_1'], row['risposta_2'], key=len),
                'rejected': max(row['risposta_1'], row['risposta_2'], key=len)}
    return {}

def transform_to_messages(dataset):
    return dataset.map(lambda x: {
        "chosen": [{"role": "user", "content": x["prompt"]}, {"role": "assistant", "content": x["chosen"]}],
        "rejected": [{"role": "user", "content": x["prompt"]}, {"role": "assistant", "content": x["rejected"]}]
    }).select_columns(["chosen", "rejected"])



def process_evol_dataset():
    ds = load_dataset("anakin87/evol-dpo-ita-reranked", split="train")
    ds = ds.filter(lambda x: int(x['punteggio_risposta_1']) >= 3 or int(x['punteggio_risposta_2']) >= 3)
    ds = ds.map(lambda x: {"prompt": x["question"]})
    ds = ds.map(resolve_ties)
    return ds.select_columns(["prompt", "chosen", "rejected"])

def process_math_dataset():
    ds = load_dataset("mii-llm/argilla-math-preferences-it", split="train")
    ds = ds.map(lambda x: {
        "prompt": x["input"],
        "chosen": x["output"],
        "rejected": x["rejected"]
    })
    return ds.select_columns(["prompt", "chosen", "rejected"])


def process_wsdm_dataset():
    ds = load_dataset("ruggsea/wsdm2024-cot-dataset", split="train")
    ds = ds.filter(lambda x: x["language"] == "Italian")
    ds = ds.map(lambda x: {
        "prompt": x["prompt"],
        "chosen": x["response_a"] if x["winner"] == "model_a" else x["response_b"],
        "rejected": x["response_a"] if x["winner"] == "model_b" else x["response_b"]
    })
    return ds.select_columns(["prompt", "chosen", "rejected"])

def gemmavsgemma_favor_shorter_responses(row):
    """
    If best is "llama_response" and scores are equal, rejected is the longer response and chosen is the shorter response
    """
    if row['best'] == "llama_response" and int(row["punteggio_risposta_1"]) == int(row["punteggio_risposta_2"]):
        return {"rejected": max(row['risposta_1'], row['risposta_2'], key=len)}
    return {}

def process_gemma_dataset():
    ds = load_dataset("anakin87/gemma-vs-gemma-preferences", split="train")
    ds = ds.filter(lambda x: x["llama_response"] is not None or int(x["punteggio_risposta_1"]) >= 4 or int(x["punteggio_risposta_2"]) >= 4)
    ds = ds.map(resolve_ties)
    ds = ds.map(gemmavsgemma_favor_shorter_responses)
    ds = stratified_sample(ds, 10000, "task_category")
    return ds.select_columns(["prompt", "chosen", "rejected"])

def process_orpodpo_dataset():
    ds = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train")
    ds = ds.filter(lambda x: x["source"] != "toxic-dpo-v0.2" and
                            x["chosen"] != x["rejected"] and
                            len(x["chosen"]) < 10)
    ds = stratified_sample(ds, 30000, "source")
    return ds.select_columns(["chosen", "rejected"])


ita_preference_datasets = [
    process_evol_dataset(),
    process_math_dataset(),
    process_wsdm_dataset(),
    process_gemma_dataset()
]

ita_mix = concatenate_datasets(ita_preference_datasets)
conversational_ita_mix = transform_to_messages(ita_mix)
orpodpo = process_orpodpo_dataset()

mixed_ds = concatenate_datasets([conversational_ita_mix, orpodpo]).shuffle(seed=42)


mixed_ds


mixed_ds[0]


mixed_ds[5]


from transformers import AutoTokenizer
from numpy import percentile
import multiprocessing


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

def compute_lengths(row):
    prompt_len = len(tokenizer(row["chosen"][0]["content"])["input_ids"])
    max_len = max(
        len(tokenizer.apply_chat_template(row["chosen"])),
        len(tokenizer.apply_chat_template(row["rejected"]))
    )
    return {
        "prompt_length": prompt_len,
        "max_length": max_len
    }

processed_ds = mixed_ds.map(
    compute_lengths,
    num_proc=multiprocessing.cpu_count()
)

# Now you can access the lengths as columns
prompt_lengths = processed_ds["prompt_length"]
max_lengths = processed_ds["max_length"]


print(f"max prompt length: {max(prompt_lengths)}")
print(f"max length: {max(max_lengths)}")

print(f"p95 prompt length: {percentile(prompt_lengths, 95)}")
print(f"p95 length: {percentile(max_lengths, 95)}")



from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "anakin87/gemma-2-2b-ita-sft",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


! wget - q -O spectrum_results.yaml "https://raw.githubusercontent.com/anakin87/gemma-neogenesis/refs/heads/main/spectrum_results/snr_results_google-gemma-2-2b-it_unfrozenparameters_25percent.yaml"

with open("spectrum_results.yaml", "r") as f:
  yaml_unfrozen_parameters = f.read()

freeze_and_unfreeze_parameters(model, yaml_unfrozen_parameters)

# uncomment to do a sanity check
# for name, param in model.named_parameters():
#     if param.requires_grad:
#       print(name, param.requires_grad)


from trl import DPOConfig, DPOTrainer
new_model_id="anakin87/gemma-2-2b-neogenesis-ita"

cfg = DPOConfig(
    output_dir='./mymodel',
    overwrite_output_dir = True,
    hub_model_id=new_model_id,
    hub_strategy="every_save",
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    push_to_hub=True,
    logging_steps=16,
    max_prompt_length=350,
    max_length=1260,
    beta=0.1,
    loss_type="sigmoid",
    remove_unused_columns=True,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    tf32=True,
    learning_rate=5.0e-06,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)

dpo_trainer = DPOTrainer(
    model=model,
    args=cfg,
    train_dataset=mixed_ds,
    processing_class=tokenizer
)

dpo_trainer.train()


IFrame("https://anakin87-gemma-2-2b-neogenesis-ita.hf.space", frameborder="0",	width="100%",	height="800px")


import torch
from transformers import pipeline

model_id="anakin87/gemma-2-2b-neogenesis-ita"

# TO USE ON KAGGLE NOTEBOOKS
# 1. Add the model as an input of your notebook
# 2. Uncomment the following line to specify the Kaggle model path
# model_id="/kaggle/input/gemma-2-2b-neogenesis-ita/transformers/gemma-2-2b-neogenesis-ita/1"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)


messages = [{"role": "user", "content": "Cos'Ã¨ l'interesse composto? Spiega in maniera semplice e chiara."}]

outputs = pipe(messages, max_new_tokens=500)

print(outputs[0]["generated_text"][1]["content"])


messages = [{"role": "user", "content": "Crea una breve storia con animali sul valore dell'amicizia"}]

outputs = pipe(messages,
               max_new_tokens=1000,
               do_sample=True,
               top_p=0.9,
               top_k=100,
               temperature=0.6)

print(outputs[0]["generated_text"][1]["content"])


!rm -rf lm-evaluation-harness && git clone https://github.com/EleutherAI/lm-evaluation-harness && cd lm-evaluation-harness && pip install -e .

!lm_eval --model hf \
    --model_args pretrained=anakin87/gemma-2-2b-neogenesis-ita\
    --tasks arc_it,hellaswag_it\
    --device cuda:0 \
    --batch_size  1

!lm_eval --model hf \
    --model_args pretrained=anakin87/gemma-2-2b-neogenesis-ita\
    --tasks m_mmlu_it --num_fewshot 5\
    --device cuda:0 \
    --batch_size  1


! wget -q -O qualitative_evaluation.html "https://raw.githubusercontent.com/anakin87/gemma-neogenesis/refs/heads/main/qualitative_evaluation/qualitative_evaluation.html"

from IPython.display import HTML

with open("qualitative_evaluation.html", "r") as fin:
    qualitative_evaluation=fin.read()

display(HTML(qualitative_evaluation))


! pip install haystack-ai==2.8.1 duckduckgo-api-haystack==0.1.14


import torch
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from duckduckgo_api_haystack import DuckduckgoApiWebSearch


template = [ChatMessage.from_user("""Documenti: \n
           {% for i in range(documents|length) %}
               {{ documents[i].content }}
               URL: {{ links[i] }}
               ---
           {% endfor %}
           In base ai documenti forniti, rispondi in italiano alla seguente domanda: {{ query }}.
           Motiva brevemente la risposta.
           Dopo aver risposto, riporta gli URL dei documenti forniti che supportano la risposta.
           Risposta:""")]


web_rag_pipe = Pipeline()
web_rag_pipe.add_component("web_search", DuckduckgoApiWebSearch(top_k=5, backend="lite"))
web_rag_pipe.add_component("prompt_builder", ChatPromptBuilder(template=template))
web_rag_pipe.add_component("slm", HuggingFaceLocalChatGenerator(
   model="anakin87/gemma-2-2b-neogenesis-ita",
   huggingface_pipeline_kwargs={"torch_dtype":torch.bfloat16}
))

web_rag_pipe.connect("web_search.documents", "prompt_builder.documents")
web_rag_pipe.connect("web_search.links", "prompt_builder.links")
web_rag_pipe.connect("prompt_builder.prompt", "slm.messages")

web_rag_pipe.show()


def get_response(question: str):
  print(f"Q: {question}\n")
  data = {"web_search":{"query":question}, "prompt_builder":{"query": question}}
  print(web_rag_pipe.run(data=data)["slm"]["replies"][0].text)

questions = [
    "La carbonara si prepara con o senza panna?",  # ğŸ�� Is carbonara made with or without cream?
    "Quando Ã¨ iniziato il primo mandato di Mattarella come presidente della repubblica?",  # ğŸ�›ï¸� When did Mattarella's first term as President of the Republic begin?
    "Chi era Pino Mango?",  # ğŸ�¤ Who was Pino Mango?
    "Dove si trova Monterchi e per cosa Ã¨ popolare?",  # ğŸ–¼ï¸� Where is Monterchi, and what is it famous for?
    "Com'Ã¨ finita la partita Juventus-Torino?"  # âš½ How did the Juventus-Torino match end?
]


get_response(questions[0])


! wget -q -O references.md "https://raw.githubusercontent.com/anakin87/gemma-neogenesis/refs/heads/main/references.md"
from IPython.display import Markdown

with open("references.md", "r") as fin:
    references = fin.read().partition("<!---References-->")[2]
    references = references.replace('\n#', '\n##')
display(Markdown(references))

