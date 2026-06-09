%%capture
!pip install --no-index --find-links ../input/huggingface-datasets datasets -qq


import pandas as pd
import numpy as np
import os
import json
import collections

from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel

import datasets
from transformers.trainer_utils import set_seed
from transformers import (AutoTokenizer, PreTrainedTokenizerFast,
                          AutoModelForQuestionAnswering, TrainingArguments,
                          Trainer, default_data_collator, DataCollatorWithPadding,)

from tqdm.auto import tqdm
import gc


def read_json(from_path: Path) -> dict:
    with open(from_path, 'r', encoding='utf-8') as out_file:
        return json.load(out_file)
        
def write_json(data: dict, out_path: Path) -> None:
    with open(out_path, 'w', encoding='utf-8') as out_file:
        json.dump(data, out_file, indent=2, sort_keys=True, ensure_ascii=False)


path = '../input/xlm-roberta-large-squad2-finetuned-l11'
config = read_json(f'../input/xlm-roberta-large-squad2-finetuned-l11/xlm_roberta_large_squad2_finetuned_L11.json')
config['model_path'] = f'{path}/'
config['LB'] = 0.736
# write_json(config, '../input/bertbasemultilingualcasedfinetunedv11/bert_base_multilingual_cased_finetuned_v11.json')


config


set_seed(config['seed'])


def df_to_squad_format(path: Path, out_name: str, lang: Optional[str] = None) -> Path:
    df = pd.read_csv(path)
    if lang:
        df = df.loc[df.language == lang].copy()
        out_name = f'{out_name}_{lang}'
    
    data = []
    for _, row in df.iterrows():
        answers = {}
        try:
            answers['answer_start'] = [int(row['answer_start'])]
            answers['text'] = [row['answer_text']]
        except:
            answers = {'answer_start': [-1], 'text': ['']}
        data.append(
            {
            'answers': answers,
            'context': row['context'],
            'id': row['id'],
            'question': row['question'],
            'title': ''
            }
        )
    
    df_as_squad = {'data': data, 'version': out_name}
    
    out_path = f'./{out_name}.json'
    write_json(df_as_squad, out_path)
    print('The data has been converted to SQuAD format and saved as a JSON object.')
    return out_path


chaii_test = datasets.load_dataset(
    'json',
    data_files=df_to_squad_format(config['test_path'], 'chaii_test'), 
    field='data',
    split='train'
)


chaii_test


tokenizer = AutoTokenizer.from_pretrained('../input/xlm-roberta-large-squad2-finetuned-l11')
model = AutoModelForQuestionAnswering.from_pretrained('../input/xlm-roberta-large-squad2-finetuned-l11')


assert isinstance(tokenizer, PreTrainedTokenizerFast)
pad_on_right = tokenizer.padding_side == "right"


def prepare_validation_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=config['max_length'],
        stride=config['doc_stride'],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


test_features = chaii_test.map(
    prepare_validation_features,
    batched=True,
    remove_columns=chaii_test.column_names
)


test_features


trainer = Trainer(
    model,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
)


test_predictions = trainer.predict(test_features)
test_features.set_format(type=test_features.format["type"], columns=list(test_features.features.keys()))


test_predictions


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer=tokenizer,
                               squad_v2=config['squad_v2'], n_best_size=config['n_best_size'], 
                               max_answer_length=config['max_answer_length']):
    
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []
        
        context = example["context"]
        
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


predictions = postprocess_qa_predictions(chaii_test, test_features, test_predictions.predictions)


predictions


test_df = pd.read_csv(config['test_path'])
test_df['PredictionString'] = test_df['id'].apply(lambda x: predictions[x])
test_df[['id', 'PredictionString']].to_csv('submission.csv', index=False)


pd.read_csv('./submission.csv')




