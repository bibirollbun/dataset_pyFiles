pip install -U propp_fr


# !pip install -U 'spacy[cuda12x]'
!python -m spacy download en_core_web_trf


import propp_fr


from propp_fr import load_spacy_model
spacy_model = load_spacy_model("en_core_web_trf")


import pandas as pd
import csv


train_data_path = "/kaggle/input/litbank-ozon-2020/train_sents.csv"
train_csv = pd.read_csv(train_data_path, quoting=csv.QUOTE_MINIMAL,  keep_default_na=False)
with open(train_data_path) as f:
    train_raw_txt = f.read()


train_csv


sentence_IDs = []
sentence_id = 0
for line in train_raw_txt.split("\n")[1:]:
    if line == "":
        sentence_id += 1
    else:
        sentence_IDs.append(sentence_id)
len(sentence_IDs)


tokens_df = train_csv.copy()


def extract_entities_df(tokens_df):
    entities = []

    entity = None

    for token_ID, BIO in enumerate(tokens_df["tag"]):

        if BIO == "O":
            if entity is not None:
                entity["end_token"] = token_ID - 1
                entity["len"] = entity["end_token"] - entity["start_token"] + 1
                entities.append(entity)
                entity = None
            continue

        tag, cat = BIO.split("-", 1)

        if tag == "B":
            if entity is not None:
                entity["end_token"] = token_ID - 1
                entity["len"] = entity["end_token"] - entity["start_token"] + 1
                entities.append(entity)
            entity = {"cat": cat, "start_token": token_ID}

        elif tag == "I":
            if entity is None or entity["cat"] != cat:
                # illegal I after O or type mismatch → treat as B
                if entity is not None:
                    entity["end_token"] = token_ID - 1
                    entity["len"] = entity["end_token"] - entity["start_token"] + 1
                    entities.append(entity)
                entity = {"cat": cat, "start_token": token_ID}
            # else: valid continuation → do nothing

    # close entity at end of document
    if entity is not None:
        entity["end_token"] = len(tokens_df) - 1
        entity["len"] = entity["end_token"] - entity["start_token"] + 1
        entities.append(entity)

    entities_df = pd.DataFrame(entities)
    return entities_df





text_content = " ".join(tokens_df["token"].tolist())

tokens_df["token_len"] = tokens_df["token"].apply(lambda x: len(str(x)))
current_byte = 0
token_onsets, token_offsets = [], []
for token_len in tokens_df["token_len"].tolist():
    token_onsets.append(current_byte)
    current_byte += token_len
    token_offsets.append(current_byte)
    current_byte += 1

tokens_df["byte_onset"] = token_onsets
tokens_df["byte_offset"] = token_offsets
tokens_df["sentence_ID"] = sentence_IDs
tokens_df["token_ID_within_document"] = tokens_df.index.tolist()
tokens_df["word"] = tokens_df["token"]

entities_df = extract_entities_df(tokens_df)
entities_df


entities_df = pd.merge(
    entities_df,
    tokens_df[["token_ID_within_document", "byte_onset"]],
    left_on="start_token",
    right_on="token_ID_within_document",
    how="left",
    ).drop(columns=["token_ID_within_document"])
entities_df = pd.merge(
    entities_df,
    tokens_df[["token_ID_within_document", "byte_offset"]],
    left_on="end_token",
    right_on="token_ID_within_document",
    how="left",
    ).drop(columns=["token_ID_within_document"])

entities_df


entities_df["in_to_out_nested_level"] = 0
entities_df["out_to_in_nested_level"] = 0
entities_df["nested_entities_count"] = 0
entities_df["mention_len"] = entities_df["len"]


from propp_fr import save_text_file, save_tokens_df, save_entities_df

files_directory = "/kaggle/working"
file_name = "train"
save_text_file(text_content, file_name, files_directory)
save_tokens_df(tokens_df, file_name, files_directory)
save_entities_df(entities_df, file_name, files_directory)


from propp_fr import load_text_file, load_tokens_df, load_entities_df

files_directory = "/kaggle/working"
file_name = "train"
text_content = load_text_file(file_name, files_directory)
tokens_df = load_tokens_df(file_name, files_directory)
entities_df = load_entities_df(file_name, files_directory)


from propp_fr import mentions_detection_LOOCV_full_model_training, generate_NER_model_card_from_LOOCV_directory

NER_cat_list = list(set(entities_df["cat"].tolist()))
print(NER_cat_list)


import os

model_name = "google-bert/bert-base-cased"
# model_name = "answerdotai/ModernBERT-large"
# model_name = "FacebookAI/roberta-large"
# model_name = "Jean-Baptiste/roberta-large-ner-english"
# model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
# model_name = "FacebookAI/xlm-roberta-large"
# model_name = "microsoft/deberta-v3-large"
model_name = "google/mt5-xl"
# model_name = "google/umt5-xl"
# model_name = "google/mt5-xxl"
# model_name = "google/flan-t5-xl"
# model_name = "google/t5-v1_1-xl"

subword_pooling_strategy = "first_last" # ["average", "first", "last", "first_last", "max"]
tagging_scheme = "BIOES"
nested_levels = [0]

embedding_model_name = model_name.split("/")[-1]
trained_model_directory = os.path.join(files_directory, f"mentions_detection_model_{embedding_model_name}")


mentions_detection_LOOCV_full_model_training(files_directory=files_directory,
                                             trained_model_directory=trained_model_directory,
                                             model_name=model_name,
                                             subword_pooling_strategy=subword_pooling_strategy,
                                             nested_levels=nested_levels,
                                             NER_cat_list=NER_cat_list,
                                             tagging_scheme=tagging_scheme,
                                             train_final_model=True,
                                             files_to_use_in_cross_validation=[],
                                             verbose=0)

generate_NER_model_card_from_LOOCV_directory(trained_model_directory)


test_data_path = "/kaggle/input/litbank-ozon-2020/test_sents_without_answers.csv"
test_csv = pd.read_csv(test_data_path, quoting=csv.QUOTE_MINIMAL,  keep_default_na=False)
with open(test_data_path) as f:
    test_raw_txt = f.read()

sentence_IDs = []
sentence_id = 0
for line in test_raw_txt.split("\n")[1:]:
    if line == "":
        sentence_id += 1
    else:
        sentence_IDs.append(sentence_id)
len(sentence_IDs)

tokens_df = test_csv.copy()

text_content = " ".join(tokens_df["token"].tolist())

tokens_df["token_len"] = tokens_df["token"].apply(lambda x: len(str(x)))
current_byte = 0
token_onsets, token_offsets = [], []
for token_len in tokens_df["token_len"].tolist():
    token_onsets.append(current_byte)
    current_byte += token_len
    token_offsets.append(current_byte)
    current_byte += 1

tokens_df["byte_onset"] = token_onsets
tokens_df["byte_offset"] = token_offsets
tokens_df["sentence_ID"] = sentence_IDs
tokens_df["token_ID_within_document"] = tokens_df.index.tolist()
tokens_df["word"] = tokens_df["token"]


from propp_fr import load_mentions_detection_model

mentions_detection_model = load_mentions_detection_model(
    "/kaggle/working/mentions_detection_model_mt5-xl/final_model.pkl"
)


from propp_fr import load_tokenizer_and_embedding_model, get_embedding_tensor_from_tokens_df

# Load the tokenizer and pre-trained embedding model
tokenizer, embedding_model = load_tokenizer_and_embedding_model(
    mentions_detection_model["base_model_name"],
  )


# Generate embeddings for all tokens
tokens_embedding_tensor = get_embedding_tensor_from_tokens_df(
    text_content,
    tokens_df,
    tokenizer,
    embedding_model,
  )


from propp_fr import generate_entities_df

entities_df = generate_entities_df(
    tokens_df,
    tokens_embedding_tensor,
    mentions_detection_model,
)


tokens_df["tag"] = "O"
entities_df["mention_len"] = entities_df["end_token"] - entities_df["start_token"] + 1
for start_token, end_token, mention_len, cat in entities_df[["start_token", "end_token", "mention_len", "cat"]].values:
    tokens_df.loc[start_token, "tag"] = f"B-{cat}"
    if mention_len > 1:
        tokens_df.loc[range(start_token+1,end_token+1), "tag"] = f"I-{cat}"
tokens_df["ID"] = tokens_df.index.tolist()


tokens_df


submission_df = tokens_df[["ID", "tag"]].copy()

submission_df.to_csv("submission.csv", index=False)


example_csv = pd.read_csv("/kaggle/working/submission.csv")
example_csv




