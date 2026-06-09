!pip install vllm
!pip install logits-processor-zoo==0.1.10
!pip install triton==3.2.0
!pip install clean-text
!pip install clean-text[gpl]



import os
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
from scipy.special import softmax

test_data = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

BASE_MODEL_PATH = "/kaggle/input/Qwen 3"
LORA_ADAPTER_PATH = "/kaggle/input/qwen2-5-32b"



if __name__ == '__main__':
    os.environ["VLLM_USE_V1"] = "0"

    language_model = vllm.LLM(
        BASE_MODEL_PATH,
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
    tokenizer = language_model.get_tokenizer()



    SYSTEM_PROMPT = """
    You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No.
    """
    
    all_prompts = []
    for _, row in test_data.iterrows():
        formatted_text = f"""
    r/{row.subreddit}
    Rule: {row.rule}
    
    1) {row.positive_example_1}
    Violation: Yes
    
    2) {row.positive_example_2}
    Violation: Yes
    
    3) {row.negative_example_1}
    Violation: No
    
    4) {row.negative_example_2}
    Violation: No
    
    5) {row.body}
    """
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_text}
        ]
    
        final_prompt = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        ) + "Answer:"
        all_prompts.append(final_prompt)
    
    test_data["formatted_prompt"] = all_prompts
    
    logits_processor = MultipleChoiceLogitsProcessor(tokenizer, choices=['Yes', 'No'])
    model_outputs = language_model.generate(
        all_prompts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[logits_processor],
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("default", 1, LORA_ADAPTER_PATH)
    )
    output_logprobs = [
        {logprob.decoded_token: logprob.logprob for logprob in output.outputs[0].logprobs[0].values()}
        for output in model_outputs
    ]
    probability_table = pd.DataFrame(output_logprobs)[['Yes', 'No']]
    test_data = pd.concat([test_data, probability_table], axis=1)
    
    test_data[['Yes', "No"]] = test_data[['Yes', "No"]].apply(lambda row: softmax(row.values), axis=1, result_type="expand")
    test_data["prediction_score"] = test_data["Yes"]
    test_data['rule_violation'] = test_data["prediction_score"]
    test_data[['row_id', 'rule_violation']].to_csv("submission_qwen.csv", index=False)
    pd.read_csv('submission_qwen.csv')



import os
import pandas as pd
EMBEDDING_MODEL_PATH = "/kaggle/input/qwen-3/transformers/default/1"
DATASET_PATH = "/kaggle/input/jigsaw-agile-community-rules"

# From HF card of Qwen3-Embedding-0.6B
EMBEDDING_MODEL_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

ENABLE_TEXT_CLEANING = True
MAX_TOP_RESULTS = 1000
EMBEDDING_BATCH_SIZE = 128



import pandas as pd
import torch.distributed as dist
from datasets import Dataset
from tqdm.auto import tqdm
from constants import ENABLE_TEXT_CLEANING
import re



def anonymize_text(raw_text: str, **kwargs) -> str:
    raw_text = raw_text.lower()
    raw_text = re.sub(r"http\S+", "<URL>", raw_text)
    raw_text = re.sub(r"\S+@\S+", "<EMAIL>", raw_text)
    raw_text = re.sub(r"\+?\d[\d\-\(\) ]{7,}\d", "<PHONE>", raw_text)
    return raw_text



def build_reddit_prompt(data_row: pd.Series) -> str:
    return f"""r/{data_row["subreddit"]}\nComment: {data_row["body"]}"""


# Apply detailed text cleaning using anonymize_text with extra options
def detailed_cleaner(raw_text: str) -> str:
    return anonymize_text(
        raw_text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        lang="en",
    )



# Prepare dataframe for embeddings: add prompts, clean if enabled, map labels
def prepare_dataframe_for_embedding(df: pd.DataFrame) -> pd.DataFrame:
    df["prompt"] = df.apply(build_reddit_prompt, axis=1)

    if ENABLE_TEXT_CLEANING:
        tqdm.pandas(desc="Cleaning text")
        df["prompt"] = df["prompt"].progress_apply(detailed_cleaner)

    if "rule_violation" in df.columns:
        df["rule_violation"] = df["rule_violation"].map({1: 1, 0: -1})

    return df



import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from tqdm.auto import tqdm
from utils import load_training_dataframe, prepare_dataframe_for_embedding
from constants import DATASET_PATH, EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_PROMPT, MAX_TOP_RESULTS, EMBEDDING_BATCH_SIZE




# Compute similarity scores between test and training data using embeddings
def compute_similarity_scores(test_df: pd.DataFrame) -> pd.DataFrame:
    training_df = load_training_dataframe(DATASET_PATH)
    training_df = prepare_dataframe_for_embedding(training_df)
    
    embedding_model = SentenceTransformer(
        model_name_or_path=EMBEDDING_MODEL_PATH,
        device="cuda",
    )

    results = []
    for current_rule in tqdm(test_df["rule"].unique(), desc="Scoring per rule"):
        # Filter subsets for the current rule
        test_subset = test_df.query("rule == @current_rule").reset_index(drop=True)
        training_subset = training_df.query("rule == @current_rule").reset_index(drop=True)
        training_subset = training_subset.reset_index(names="row_id")
        
        # Encode test prompts into embeddings
        query_embeddings = embedding_model.encode(
            sentences=test_subset["prompt"].tolist(),
            prompt=EMBEDDING_MODEL_PROMPT,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        # Encode training prompts into embeddings
        doc_embeddings = embedding_model.encode(
            sentences=training_subset["prompt"].tolist(),
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )

        # Perform semantic search between test and training embeddings
        test_subset["semantic_matches"] = semantic_search(
            query_embeddings,
            doc_embeddings,
            top_k=MAX_TOP_RESULTS,
            score_function=dot_score,
        )

        # Aggregate similarity scores with violation labels
        def aggregate_violation_score(match_list):
            match_df = pd.DataFrame(match_list)
            match_df = match_df.merge(
                training_subset[["row_id", "rule_violation"]],
                how="left",
                left_on="corpus_id",
                right_on="row_id",
            )
            match_df["score"] = match_df["score"] * match_df["rule_violation"]
            return match_df["score"].sum()
            
        tqdm.pandas(desc=f"Aggregating labels for {current_rule=}")
        test_subset["rule_violation"] = test_subset["semantic_matches"].progress_apply(aggregate_violation_score)
        results.append(test_subset[["row_id", "rule_violation"]].copy())
        
    final_submission = pd.concat(results, axis=0)
    return final_submission



# Generate submission.csv by computing similarity scores
def generate_submission_file():
    test_df = pd.read_csv(f"{DATASET_PATH}/test.csv")
    test_df = prepare_dataframe_for_embedding(test_df)
    
    submission_df = compute_similarity_scores(test_df)
    submission_df = test_df[["row_id"]].merge(submission_df, on="row_id", how="left")
    submission_df.to_csv("/kaggle/working/submission.csv", index=False)

if __name__ == "__main__":
    generate_submission_file()



import semantic

semantic.generate_submission_file()


