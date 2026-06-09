import os
import pandas as pd


EMBDEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules"

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json
EMBEDDING_MODEL_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

CLEAN_TEXT = True
TOP_K = 1000
BATCH_SIZE = 128


import pandas as pd
import torch.distributed as dist

from datasets import Dataset
from cleantext import clean
from tqdm.auto import tqdm


def build_prompt(row):
    return f"""r/{row["subreddit"]}\nComment: {row["body"]}"""


def cleaner(text):
    return clean(
        text,
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


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []
    flatten.append(train_dataset[["body", "rule", "subreddit", "rule_violation"]])
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[[f"{violation_type}_example_{i}", "rule", "subreddit"]].copy()
            sub_dataset = sub_dataset.rename(columns={f"{violation_type}_example_{i}": "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0)    
    dataframe = dataframe.drop_duplicates(ignore_index=True)
    return dataframe


data_path = DATA_PATH
data_path


train_dataset = pd.read_csv(f"{data_path}/train.csv")
print(train_dataset.shape)
train_dataset.head()


test_dataset = pd.read_csv(f"{data_path}/test.csv")
print(test_dataset.shape)
test_dataset.head()


flatten = []
flatten.append(train_dataset[["body", "rule", "subreddit", "rule_violation"]])
flatten[0]  # train_dataset ì—�ì„œ ì›�í•˜ëŠ” columnë“¤


# for-loop í•˜ë‚˜
violation_type = 'positive'
i = 1

# test_dataset ì—�ì„œ ìƒ˜í”Œ í•˜ë‚˜
sub_dataset = test_dataset[[f"{violation_type}_example_{i}", "rule", "subreddit"]].copy()
sub_dataset


# body ë¡œ ì—´ ì�´ë¦„ ë°”ê¿ˆ
sub_dataset = sub_dataset.rename(columns={f"{violation_type}_example_{i}": "body"})
sub_dataset


# rule_violation ì—´ ì¶”ê°€
sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
sub_dataset


def prepare_dataframe(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    if CLEAN_TEXT:
        tqdm.pandas(desc="cleaner")  # https://tqdm.github.io/docs/tqdm/#pandas
        dataframe["prompt"] = dataframe["prompt"].progress_apply(cleaner)

    if "rule_violation" in dataframe.columns:
        dataframe["rule_violation"] = dataframe["rule_violation"].map(
            {
                1: 1,
                0: -1,
            }
        )

    return dataframe


from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from tqdm.auto import tqdm


test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")
print(test_dataframe.shape)
test_dataframe.head()


# prompt ì—´ ë§Œë“¬ (subreddint ì�´ë�‘ body ê°™ì�´ í‘œê¸°, cleaner ì �ìš©)
test_dataframe = prepare_dataframe(test_dataframe)
print(test_dataframe.shape)
test_dataframe.head()


# train_dataset ì�´ë�‘ test_dataset ì—�ì„œ positive_sample, negative_sample ê°™ì�´ ì‚¬ìš©
corpus_dataframe = get_dataframe_to_train(DATA_PATH)
print(corpus_dataframe.shape)  # â�“í–‰ ìˆ˜ train_dataset ë³´ë‹¤ ì�‘ì�Œ. ì¤‘ë³µì�´ì–´ì„œ ì§€ì› ë‚˜?
corpus_dataframe.head()


# prompt ì—´ ë§Œë“¬ê³  clean() ì �ìš©. rule_violation ë§¤í•‘
corpus_dataframe = prepare_dataframe(corpus_dataframe)
print(corpus_dataframe.shape)
corpus_dataframe.head()


EMBDEDDING_MODEL_PATH


embedding_model = SentenceTransformer(
    model_name_or_path=EMBDEDDING_MODEL_PATH,
    device="cuda",
)
embedding_model


result = []


rules = test_dataframe["rule"].unique()
rules


rule = rules[0]
rule


# test_dataframe ì—�ì„œ rule ê°™ì�€ ê±°
test_dataframe_part = test_dataframe.query("rule == @rule").reset_index(drop=True)
print(test_dataframe_part.shape)
test_dataframe_part.head()


# corpus_dataframe_part ì—�ì„œ rule ê°™ì�€ ê±°
corpus_dataframe_part = corpus_dataframe.query("rule == @rule").reset_index(drop=True)
print(corpus_dataframe_part.shape)
corpus_dataframe_part.head()


# row_id ì—´ ë§Œë“¬
corpus_dataframe_part = corpus_dataframe_part.reset_index(names="row_id")
print(corpus_dataframe_part.shape)
corpus_dataframe_part.head()


query_embeddings = embedding_model.encode(
    sentences=test_dataframe_part["prompt"].tolist(),
    prompt=EMBEDDING_MODEL_QUERY,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_tensor=True,
    device="cuda",
    normalize_embeddings=True,
)

query_embeddings.shape


document_embeddings = embedding_model.encode(
    sentences=corpus_dataframe_part["prompt"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_tensor=True,
    device="cuda",
    normalize_embeddings=True,
)

document_embeddings.shape


# https://sbert.net/examples/sentence_transformer/applications/semantic-search/
# cos_sim + sort + top_k
test_dataframe_part["semantic"] = semantic_search(
    query_embeddings,
    document_embeddings,
    top_k=TOP_K,
    score_function=dot_score,
)


test_dataframe_part.head()


# ì²« í–‰ì�˜ 'semantic' ì—´ í™•ì�¸
len(test_dataframe_part.iloc[0, -1])


test_dataframe_part.iloc[0, -1][:5]


def get_score(semantic):
    semantic = pd.DataFrame(semantic)
    semantic = semantic.merge(
        corpus_dataframe_part[["row_id", "rule_violation"]],
        how="left",
        left_on="corpus_id",
        right_on="row_id",
    )
    semantic["score"] = semantic["score"]*semantic["rule_violation"]
    return semantic["score"].sum()


semantic = test_dataframe_part.iloc[0, -1]


# dataframe ìœ¼ë¡œ ë§Œë“¬
semantic = pd.DataFrame(semantic)
print(semantic.shape)
semantic.head()


# corpus_dataframe_part ë�‘ join í•´ì„œ rule_violation ì¶”ê°€
semantic = semantic.merge(
    corpus_dataframe_part[["row_id", "rule_violation"]],
    how="left",
    left_on="corpus_id",
    right_on="row_id",
)
semantic


semantic["score"] = semantic["score"] * semantic["rule_violation"]
semantic


semantic["score"].sum()


tqdm.pandas(desc=f"Add label for {rule=}")


test_dataframe_part["rule_violation"] = test_dataframe_part["semantic"].progress_apply(get_score)
test_dataframe_part.head()


result.append(test_dataframe_part[["row_id", "rule_violation"]].copy())


len(result)


result[0]


result = []
for rule in tqdm(test_dataframe["rule"].unique(), desc=f"Generate scores for each rule"):
    test_dataframe_part = test_dataframe.query("rule == @rule").reset_index(drop=True)
    corpus_dataframe_part = corpus_dataframe.query("rule == @rule").reset_index(drop=True)
    corpus_dataframe_part = corpus_dataframe_part.reset_index(names="row_id")
    
    query_embeddings = embedding_model.encode(
        sentences=test_dataframe_part["prompt"].tolist(),
        prompt=EMBEDDING_MODEL_QUERY,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda",
        normalize_embeddings=True,
    )
    document_embeddings = embedding_model.encode(
        sentences=corpus_dataframe_part["prompt"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda",
        normalize_embeddings=True,
    )
    test_dataframe_part["semantic"] = semantic_search(
        query_embeddings,
        document_embeddings,
        top_k=TOP_K,
        score_function=dot_score,
    )
    def get_score(semantic):
        semantic = pd.DataFrame(semantic)
        semantic = semantic.merge(
            corpus_dataframe_part[["row_id", "rule_violation"]],
            how="left",
            left_on="corpus_id",
            right_on="row_id",
        )
        semantic["score"] = semantic["score"]*semantic["rule_violation"]
        return semantic["score"].sum()
        
    tqdm.pandas(desc=f"Add label for {rule=}")
    test_dataframe_part["rule_violation"] = test_dataframe_part["semantic"].progress_apply(get_score)
    result.append(test_dataframe_part[["row_id", "rule_violation"]].copy())


submission = pd.concat(result, axis=0)
print(submission.shape)
submission


# test_dataframe ê¸°ì¤€ìœ¼ë¡œ ì •ë ¬
submission = test_dataframe[["row_id"]].merge(submission, on="row_id", how="left")
submission


# v1: 0.875
submission['rule_violation'].hist()


submission['rule_violation'] = submission['rule_violation'].rank(method='average') / (len(submission)+1)
submission


# v4: 0.875
submission['rule_violation'].hist()


submission.to_csv("submission.csv", index=False)

