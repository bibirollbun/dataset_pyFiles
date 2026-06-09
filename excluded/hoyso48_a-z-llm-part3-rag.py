# !pip uninstall transformers --yes
# !pip install unsloth
!pip install bitsandbytes faiss-gpu blingfire





import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gc
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


NUM_TRAIN_SAMPLES = 10000


train_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv')
valid_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/train_with_context2.csv')
train_df = train_df.fillna('').sample(NUM_TRAIN_SAMPLES, random_state=42)


class PretrainedMultipleChoiceModel(nn.Module):
    def __init__(self, model_name, dropout_rate=0.1, dtype=torch.bfloat16):
        super(PretrainedMultipleChoiceModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)#, attn_implementation="flash_attention_2")
        hidden_size = self.model.config.hidden_size
        # self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 1, dtype=dtype)
    
    def forward(self, input_ids, attention_mask, labels=None):
        bs, n_options, seq_len = input_ids.shape
        input_ids, attention_mask = input_ids.view(-1, seq_len), attention_mask.view(-1, seq_len)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size*n_options, seq_len, hidden_size)

        # 각 예시마다 attention_mask의 합 - 1 을 통해 마지막 유효 토큰의 인덱스 계산
        seq_lengths = attention_mask.sum(dim=1) - 1  # shape: (batch_size)
        batch_size = input_ids.size(0)
        # 각 샘플의 마지막 유효 토큰의 hidden state 추출
        last_token_states = last_hidden_state[torch.arange(batch_size), seq_lengths]
        # last_token_states = self.dropout(last_token_states)
            
        logits = self.classifier(last_token_states) #(batch_size*n_options, 1)
        logits = logits.view(bs, n_options)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        # if labels is not None:
        #     labels = F.one_hot(labels, n_options).to(logits.dtype)
        #     loss_fct = nn.BCEWithLogitsLoss()
        #     loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits} if loss is not None else logits


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/llama3-2-1b-dapt-wiki-sci/Llama3.2-1b-wiki")
model = PretrainedMultipleChoiceModel(model_name = "/kaggle/input/llama3-2-1b-dapt-wiki-sci/Llama3.2-1b-wiki",
        dtype = torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


from dataclasses import dataclass
from typing import Optional, Union, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

def custom_tokenize(tokenizer, text1, text2, max_length):
    #전체 토큰 길이 max_length을 넘지 않도록 trucnate
    #text2의 길이가 max_length을 초과하면, text2의 앞부분을 truncate, 이후 text1의 뒷부분을 truncate

    tokenizer.truncation_side='left'
    text2_encoded = tokenizer.encode(
        text2,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )

    text2_len = len(text2_encoded)
    
    if text2_len < max_length:
        tokenizer.truncation_side='right'
        text1_encoded = tokenizer.encode(
            text1,
            truncation=True,
            max_length=max_length-text2_len,
            add_special_tokens=False
        )
    else:
        text1_encoded = []

    input_ids = text1_encoded + text2_encoded
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}



def preprocess_function_multiple_choice(examples, tokenizer=tokenizer, max_length=256):
    input_ids = []
    label = []
    attention_masks = []
    # print(examples)
    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    for q, A, B, C, D, E, answer in zip(examples['prompt'], examples['A'], examples['B'], examples['C'], examples['D'], examples['E'], examples['answer']):
        text1 = [ "Question: " + q] * 5 #[ tokenizer.cls_token + q ] * 5
        text2 = ["\n###\nAnswer: " + option + "\n###\nTrue or False:" for option in [A,B,C,D,E]]
        tokenized = [custom_tokenize(tokenizer, t1, t2, max_length=max_length) for t1, t2 in zip(text1, text2)]
        input_ids.append([x['input_ids'] for x in tokenized])
        label.append(label_mapping[answer])
        attention_masks.append([x['attention_mask'] for x in tokenized])
        
    return {'input_ids': input_ids, 'labels': label, 'attention_mask': attention_masks}




from datasets import load_dataset
from datasets import Dataset, DatasetDict
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})


tokenized_dataset = dataset.map(preprocess_function_multiple_choice, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_dataset['train'][0]


torch.cuda.empty_cache()

############################
# 4. TrainingArguments 및 Trainer 설정
############################
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,                  # 평가 스텝 (사용자 설정 가능)
    logging_steps=100,               # 로깅 스텝
    warmup_ratio=0.1,
    learning_rate=2e-5,
    optim='paged_adamw_32bit', #'paged_adamw_32bit', 'paged_adamw_8bit', 'adamw_torch'
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=1,
    report_to="none",
    save_only_model=True,
    bf16=True,
    torch_compile=True,
    # fp16=True,
    lr_scheduler_type='cosine',
    # 필요시 추가 argument 지정 가능
)

def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy, "map_at_3": map_at_3(logits, labels)}

# 동적 padding을 위한 DataCollator
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, padding='longest', max_length=256)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

############################
# 5. 학습 시작
############################
trainer.train()
trainer.save_model('./llama3.2-1b-dapt-ft')














from datasets import load_dataset
from pathlib import Path

files = ['/kaggle/input/wikipedia-20230701/x.parquet']#list(map(str, Path("/kaggle/input/wiki-20220301-en-sci").glob("*.parquet")))
dataset = load_dataset("parquet", data_files=files, split="train")


def preprocess(examples):
    texts_for_index = []
    for text, title in zip(examples['text'], examples['title']):
        # 문장 단위로 분리 (영어 기준 마침표)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # 길이가 4 이상인 첫 문장 찾기
        first_valid_sentence = ""
        for sentence in sentences:
            if len(sentence) >= 4:
                first_valid_sentence = sentence
                break
        
        # 혹시 유효 문장이 없을 경우 예외 처리
        if not first_valid_sentence:
            first_valid_sentence = text.strip().split('.')[0]  # 그냥 첫 문장 사용

        # 제목과 연결
        combined = f"{title}: {first_valid_sentence}"
        texts_for_index.append(combined)
    
    return {'text': texts_for_index}

dataset = dataset.map(preprocess, batched=True)


dataset[0]


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda').half() #bge, gte, e5, ..etc
model.max_seq_length = 384
embeddings = model.encode(dataset['text'], batch_size=32, device='cuda', show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)


import faiss
import numpy as np

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 거리 기반 index -> L2정규화 했으므로 코사인 유사도와 동일
index.add(np.array(embeddings, dtype=np.float32)) # 벡터 추가
faiss.write_index(index, "my_index.faiss")


test_df = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/train.csv')


from faiss import read_index
sentence_index = read_index("/kaggle/input/wikipedia-2023-07-faiss-index/wikipedia_202307.index")
prompt = test_df['prompt'].values
prompt_embeddings = model.encode(prompt, batch_size=32, device='cuda', show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
prompt_embeddings = prompt_embeddings.detach().cpu().float().numpy()
## Get the top 5 pages that are likely to contain the topic of interest
search_score, search_index = sentence_index.search(prompt_embeddings, 5)
search_score[:3], search_index[:3]


## Save memory - delete sentence_index since it is no longer necessary
del sentence_index
del prompt_embeddings
gc.collect()


df = pd.read_parquet("/kaggle/input/wikipedia-20230701/wiki_2023_index.parquet",
                     columns=['id', 'file'])
df


## Get the article and associated file location using the index
wikipedia_file_data = []

for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
    scr_idx = idx
    _df = df.loc[scr_idx].copy()
    _df['prompt_id'] = i
    wikipedia_file_data.append(_df)
wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
wikipedia_file_data = wikipedia_file_data[['id', 'prompt_id', 'file']].drop_duplicates().sort_values(['file', 'id']).reset_index(drop=True)

## Save memory - delete df since it is no longer necessary
del df
gc.collect()


wikipedia_file_data


## Get the full text data
wiki_text_data = []

for file in tqdm(wikipedia_file_data.file.unique(), total=len(wikipedia_file_data.file.unique())):
    _id = [str(i) for i in wikipedia_file_data[wikipedia_file_data['file']==file]['id'].tolist()]
    _df = pd.read_parquet(f"/kaggle/input/wikipedia-20230701/{file}", columns=['id', 'text'])

    _df_temp = _df[_df['id'].isin(_id)].copy()
    del _df
    gc.collect()
    wiki_text_data.append(_df_temp)
wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
gc.collect()


wiki_text_data


from collections.abc import Iterable
import blingfire as bf
def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param document_type: String denoting the document type to be processed
    :param document_sections: List of sections for a given document type to process
    :param split_sentences: Flag to determine whether to further split sections into sentences
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """
    
    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values, 
                        df.document_id.values,
                        df.offset.values, 
                        filter_len, 
                        disable_progress_bar)
    return df


def sectionize_documents(documents: Iterable[str],
                         document_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the 
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in tqdm(zip(document_ids, documents), total=len(documents), disable=disable_progress_bar):
        row = {}
        text, start, end = (document, 0, len(document))
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df


def sentencize(documents: Iterable[str],
               document_ids: Iterable,
               offsets: Iterable[tuple[int, int]],
               filter_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in tqdm(zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1]-o[0] > filter_len:
                    sentence = document[o[0]:o[1]]
                    abs_offsets = (o[0]+offset[0], o[1]+offset[0])
                    row = {}
                    row['document_id'] = document_id
                    row['text'] = sentence
                    row['offset'] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)


## Parse documents into sentences
processed_wiki_text_data = process_documents(wiki_text_data.text.values, wiki_text_data.id.values)


## Get embeddings of the wiki text data
wiki_data_embeddings = model.encode(processed_wiki_text_data.text,
                                    batch_size=32,
                                    device='cuda',
                                    show_progress_bar=True,
                                    convert_to_tensor=True,
                                    normalize_embeddings=True)
wiki_data_embeddings = wiki_data_embeddings.detach().cpu().float().numpy()
gc.collect()


## Combine all answers
test_df['answer_all'] = test_df.apply(lambda x: " ".join([x['A'], x['B'], x['C'], x['D'], x['E']]), axis=1)


## Search using the prompt and answers to guide the search
test_df['prompt_answer_stem'] = test_df['prompt'] + " " + test_df['answer_all']


question_embeddings = model.encode(test_df.prompt_answer_stem.values, batch_size=32, device='cuda', show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
question_embeddings = question_embeddings.detach().cpu().float().numpy()


## Parameter to determine how many relevant sentences to include
NUM_SENTENCES_INCLUDE = 20

## List containing just Context
contexts = []

for r in tqdm(test_df.itertuples(), total=len(test_df)):

    prompt_id = r.Index

    prompt_indices = processed_wiki_text_data[processed_wiki_text_data['document_id'].isin(wikipedia_file_data[wikipedia_file_data['prompt_id']==prompt_id]['id'].values)].index.values

    if prompt_indices.shape[0] > 0:
        prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
        prompt_index.add(wiki_data_embeddings[prompt_indices])

        context = ""
        
        ## Get the top matches
        ss, ii = prompt_index.search(question_embeddings, NUM_SENTENCES_INCLUDE)
        for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
            context += processed_wiki_text_data.loc[prompt_indices]['text'].iloc[_i] + " "
        
    contexts.append(context)


test_df['context'] = contexts
test_df








train_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv')
valid_df = pd.read_csv('/kaggle/input/60k-data-with-context-v2/train_with_context2.csv')
train_df = train_df.fillna('').sample(NUM_TRAIN_SAMPLES, random_state=42)


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/llama3-2-1b-dapt-wiki-sci/Llama3.2-1b-wiki")
model = PretrainedMultipleChoiceModel(model_name = "/kaggle/input/llama3-2-1b-dapt-wiki-sci/Llama3.2-1b-wiki",
        dtype = torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token


def preprocess_function_multiple_choice_rag(examples, tokenizer=tokenizer, max_length=384):
    input_ids = []
    label = []
    attention_masks = []
    # print(examples)
    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    for q, context, A, B, C, D, E, answer in zip(examples['prompt'], examples['context'], examples['A'], examples['B'], examples['C'], examples['D'], examples['E'], examples['answer']):
        text1 = [ "Question: " + q + "\n###\nContext: " + context] * 5 #[ tokenizer.cls_token + q ] * 5
        text2 = ["\n###\nAnswer: " + option + "\n###\nTrue or False:" for option in [A,B,C,D,E]]
        tokenized = [custom_tokenize(tokenizer, t1, t2, max_length=max_length) for t1, t2 in zip(text1, text2)]
        input_ids.append([x['input_ids'] for x in tokenized])
        label.append(label_mapping[answer])
        attention_masks.append([x['attention_mask'] for x in tokenized])
        
    return {'input_ids': input_ids, 'labels': label, 'attention_mask': attention_masks}


from datasets import load_dataset
from datasets import Dataset, DatasetDict
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})


tokenized_dataset = dataset.map(preprocess_function_multiple_choice_rag, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_dataset['train'][0]


torch.cuda.empty_cache()

############################
# 4. TrainingArguments 및 Trainer 설정
############################
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,                  # 평가 스텝 (사용자 설정 가능)
    logging_steps=100,               # 로깅 스텝
    warmup_ratio=0.1,
    learning_rate=2e-5,
    optim='paged_adamw_32bit', #'paged_adamw_32bit', 'paged_adamw_8bit', 'adamw_torch'
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=1,
    report_to="none",
    save_only_model=True,
    bf16=True,
    torch_compile=True,
    # fp16=True,
    lr_scheduler_type='cosine',
    # 필요시 추가 argument 지정 가능
)

def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy, "map_at_3": map_at_3(logits, labels)}

# 동적 padding을 위한 DataCollator
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, padding='longest', max_length=256)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

############################
# 5. 학습 시작
############################
trainer.train()
trainer.save_model('./llama3.2-1b-dapt-ft')


from datasets import load_dataset
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
test_dataset = Dataset.from_pandas(valid_df)

data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, padding='longest', max_length=1024)
tokenized_test_dataset = test_dataset.map(lambda text:preprocess_function_multiple_choice_rag(text, tokenizer, max_length=1024), batched=True)
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)


model.eval()
test_predictions = []
for batch in test_dataloader:
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    with torch.no_grad():
        outputs = model(**batch)
    test_predictions.append(outputs['logits'].cpu().detach())

test_predictions = torch.cat(test_predictions)
test_predictions = test_predictions.float().numpy()
predictions_as_ids = np.argsort(-test_predictions, 1)
predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
predictions_as_string = test_df['prediction'] = [
    ' '.join(row) for row in predictions_as_answer_letters[:, :3]
]


# https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
import numpy as np
def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U


m = MAP_at_3(test_df.prediction.values, test_df.answer.values)
print( 'CV MAP@3 =',m )




