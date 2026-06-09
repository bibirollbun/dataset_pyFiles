from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel
from transformers import TrainingArguments, Trainer, AutoModelForMultipleChoice


deberta_v3_large = '/kaggle/input/deberta-v3-large-hf-weights'


df_train = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/train.csv')
df_train = df_train.drop(columns="id")
df_train.shape

df_train = pd.concat([
    df_train,
    pd.read_csv('/kaggle/input/additional-train-data-for-llm-science-exam/extra_train_set.csv'),
    pd.read_csv('/kaggle/input/additional-train-data-for-llm-science-exam/6000_train_examples.csv')
])
df_train.reset_index(inplace=True, drop=True)
df_train.shape


option_to_index = {option: idx for idx, option in enumerate('ABCDE')} # ABCDE to 0-4
index_to_option = {v: k for k,v in option_to_index.items()} # 0-4 to ABCDE

# 处理单个样本，将问题-选项对进行编码，并将正确答案转换为索引
def preprocess(example):
    first_sentence = [str(example['prompt'])] * 5 # ["What is the capital of France?"] * 5
    second_sentences = [str(example[option]) for option in 'ABCDE']  # ["Berlin", "Paris", "Madrid", "Rome", "London"]
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=False)
    tokenized_example['label'] = option_to_index[example['answer']]

    ### {
      #"input_ids": [ # 5个选项的编码
        #[...],  # A: Berlin
        #[...],  # B: Paris
        #[...],  # C: Madrid
        #[...],  # D: Rome
        #[...],  # E: London
      #],
      #"attention_mask": [...],  # 对应的 mask
      #"label": 1  # 代表正确答案 "B"
    #}
    return tokenized_example

# 批量整理数据（collate batch），在 DataLoader 里用来打包数据，确保不同长度的样本填充到相同长度，方便 GPU 加速训练。
@dataclass
class DataCollatorForMultipleChoice: 
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    # @features: [
        #  {'input_ids': [...], 'attention_mask': [...], 'label': 1},
        #  {'input_ids': [...], 'attention_mask': [...], 'label': 2},
        #  {'input_ids': [...], 'attention_mask': [...], 'label': 0},
        #  ...
        #]
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])

        # 把每个样本展开成单独的句子
        # @flattened_features: [
           # {'input_ids': ..., 'attention_mask': ...}, # A
            #{'input_ids': ..., 'attention_mask': ...}, # B
            #{'input_ids': ..., 'attention_mask': ...}, # C
            #{'input_ids': ..., 'attention_mask': ...}, # D
            #{'input_ids': ..., 'attention_mask': ...}, # E
            #...
        #]
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        # 进行 padding 并转换为 PyTorch 张量
        # self.tokenizer.pad() 确保 所有句子填充到相同长度，便于 GPU 加速
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        # 恢复 batch 结构，reshape input_ids 形状为：(batch_size, num_choices, sequence_length)
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


tokenizer = AutoTokenizer.from_pretrained(deberta_v3_large)

dataset = Dataset.from_pandas(df_train)
dataset


tokenized_dataset = dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
tokenized_dataset


tokenized_train_ds = tokenized_dataset.train_test_split(test_size=0.1)['train'] # 取出 90% 训练集部分
tokenized_train_ds


tokenized_eval_ds = tokenized_dataset.train_test_split(test_size=0.1)['test'] # 取出 10% 验证集部分
tokenized_eval_ds


def train_and_save_model(train_dataset, eval_dataset, tokenizer, model_checkpoint, output_dir, 
                          learning_rate=5e-6, batch_size=4, epochs=3, warmup_ratio=0.1, weight_decay=0.01):
    """
    训练 DeBERTa v3 large 并保存模型。
    """
    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        save_total_limit=1,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # 保存模型和 tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # 打印训练后的模型参数
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")


# # hyperparams for training
# training_args = TrainingArguments(
#     warmup_ratio=0.8, # 训练的前 80% 用于 学习率预热（Warmup），让学习率逐渐上升，以防止模型初始阶段震荡 （一般0.06 ~ 0.1）
#     save_total_limit=1,  # 只保留最近的 checkpoint
#     save_strategy="epoch",  # 每个 epoch 结束才保存，减少写入频率
#     learning_rate=5e-6, # 推荐范围： 2e-6 ~ 1e-5
#     # per_device_train_batch_size=1,
#     # per_device_eval_batch_size=2,
#     # gradient_accumulation_steps=2,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3, # 训练3个epoch
#     report_to='none',
#     output_dir='.',
# )

# model = AutoModelForMultipleChoice.from_pretrained(deberta_v3_large)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     tokenizer=tokenizer, # 分词器，将文本转换为token
#     data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer), # 数据整理器，用于填充和批量处理
#     train_dataset=tokenized_dataset,
# )

# trainer.train()


# 训练不同的超参数组合
param_configs = [
    #{"learning_rate": 4e-6, "batch_size": 2, "epochs": 3, "warmup_ratio": 0.2},  # 保守设置
    #{"learning_rate": 4.5e-6, "batch_size": 2, "epochs": 3, "warmup_ratio": 0.2},  # 适中
    {"learning_rate": 5e-6, "batch_size": 2, "epochs": 3, "warmup_ratio": 0.2},  # 激进
]

# 遍历超参数设置进行训练
for i, params in enumerate(param_configs):
    output_dir = f"./deberta_v3_model_variant_{i+1}"

    train_and_save_model(
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        tokenizer=tokenizer,
        model_checkpoint=deberta_v3_large,
        output_dir=output_dir,
        **params
    )


# test_df = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv')
# test_df['answer'] = 'A' # dummy answer that allows us to preprocess the test dataset just like we preprocessed the train dataset

# tokenized_test_dataset = Dataset.from_pandas(test_df.drop(columns=['id'])).map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E'])


# test_predictions = trainer.predict(tokenized_test_dataset).predictions
# test_predictions[:4]


# predictions_as_ids = np.argsort(-test_predictions, 1)
# predictions_as_ids[:3]


# predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
# predictions_as_answer_letters[:3]


# predictions_as_string = test_df['prediction'] = [
#     ' '.join(row) for row in predictions_as_answer_letters[:, :3]
# ]
# predictions_as_string[:3]


# submission = test_df[['id', 'prediction']]
# submission.to_csv('submission.csv', index=False)

# pd.read_csv('submission.csv').head()

