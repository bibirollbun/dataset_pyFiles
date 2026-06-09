# !pip install /kaggle/input/pm-86487298-at-07-08-2025-14-38-51/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model_name = "/kaggle/input/qwen2.5-32b-gptq-w4a16/transformers/dynamic/1"


from vllm import LLM, SamplingParams
llm = LLM(
    model=model_name,
    # quantization="gptq",                 
    tensor_parallel_size=2,             
    gpu_memory_utilization=0.99,       
    trust_remote_code=True,              
    dtype="float16",                    
    max_model_len=1500,     
    enforce_eager=True,             
    # max_num_seq = 30
    
)
tokenizer = llm.get_tokenizer()


from tqdm import tqdm
import torch
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))


import torch
# from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

# tokenizer = AutoTokenizer.from_pretrained(model_name)


def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )


special_character_list = [
    # 기하학적 도형 (16개)
    '■', '□', '▲', '△', '▼', '▽', '◆', '◇',
    '○', '●', '★', '☆', '♦', '♥', '♠', '♣',

    # 수학 및 기술 기호 (15개)
    '§', '†', '‡', '※', '∞', '±', '≠', '≈',
    '√', '∑', '∏', '∆', 'Ω', 'μ', '∂',

    # 화살표 및 괄호 (10개)
    '→', '←', '↑', '↓', '↔', '↕', '〈', '〉',
    '『', '』',

    # 블록 및 라인 문자 (10개)
    '│', '─', '┌', '┐', '└', '┘', '┼', '█',
    '▓', '▒',

    # 통화 및 기타 기호 (14개)
    '£', '¥', '€', '₩', '©', '®', '™', '♪',
    '♫', '☀', '☁', '☂', '☃', '☎'
]



special_token = [tokenizer(i, add_special_tokens=False).input_ids[0] for i in special_character_list]


system_prompt = f"""You are now tasked with analyzing math problems and classifying student responses. Given a math problem, the student's chosen answer, whether it's correct, and the student's explanation, you need to determine the appropriate Category and Misconception classification.
Below are the available Category:Misconception classifications you can choose from.
Always provide your response using only the specified format.

■: False_Correct:NA,
 □: False_Misconception:Adding_across,
 ▲: False_Misconception:Adding_terms,
 △: False_Misconception:Additive,
 ▼: False_Misconception:Base_rate,
 ▽: False_Misconception:Certainty,
 ◆: False_Misconception:Definition,
 ◇: False_Misconception:Denominator-only_change,
 ○: False_Misconception:Division,
 ●: False_Misconception:Duplication,
 ★: False_Misconception:Firstterm,
 ☆: False_Misconception:FlipChange,
 ♦: False_Misconception:Ignores_zeroes,
 ♥: False_Misconception:Incomplete,
 ♠: False_Misconception:Incorrect_equivalent_fraction_addition,
 ♣: False_Misconception:Interior,
 §: False_Misconception:Inverse_operation,
 †: False_Misconception:Inversion,
 ‡: False_Misconception:Irrelevant,
 ※: False_Misconception:Longer_is_bigger,
 ∞: False_Misconception:Mult,
 ±: False_Misconception:Multiplying_by_4,
 ≠: False_Misconception:Not_variable,
 ≈: False_Misconception:Positive,
 √: False_Misconception:Scale,
 ∑: False_Misconception:Shorter_is_bigger,
 ∏: False_Misconception:Subtraction,
 ∆: False_Misconception:SwapDividend,
 Ω: False_Misconception:Tacking,
 μ: False_Misconception:Unknowable,
 ∂: False_Misconception:WNB,
 →: False_Misconception:Whole_numbers_larger,
 ←: False_Misconception:Wrong_Fraction,
 ↑: False_Misconception:Wrong_Operation,
 ↓: False_Misconception:Wrong_fraction,
 ↔: False_Misconception:Wrong_term,
 ↕: False_Neither:NA,
 〈: True_Correct:NA,
 〉: True_Misconception:Adding_across,
 『: True_Misconception:Additive,
 』: True_Misconception:Base_rate,
 │: True_Misconception:Definition,
 ─: True_Misconception:Denominator-only_change,
 ┌: True_Misconception:Division,
 ┐: True_Misconception:Duplication,
 └: True_Misconception:Firstterm,
 ┘: True_Misconception:FlipChange,
 ┼: True_Misconception:Incomplete,
 █: True_Misconception:Incorrect_equivalent_fraction_addition,
 ▓: True_Misconception:Inversion,
 ▒: True_Misconception:Irrelevant,
 £: True_Misconception:Longer_is_bigger,
 ¥: True_Misconception:Mult,
 €: True_Misconception:Multiplying_by_4,
 ₩: True_Misconception:Not_variable,
 ©: True_Misconception:Positive,
 ®: True_Misconception:Shorter_is_bigger,
 ™: True_Misconception:Subtraction,
 ♪: True_Misconception:SwapDividend,
 ♫: True_Misconception:Tacking,
 ☀: True_Misconception:WNB,
 ☁: True_Misconception:Whole_numbers_larger,
 ☂: True_Misconception:Wrong_fraction,
 ☃: True_Misconception:Wrong_term,
 ☎: True_Neither:NA


Please analyze the given input and provide your classification."""


train['special_label'] = train['label'].apply(lambda x: special_character_list[int(x)])


train['system'] = system_prompt


def create_message_list(row):
    messages = [
        {"role": "system", "content": row['system']},
        {"role": "user", "content": row['text']},
        {"role": "assistant", "content": str(row['special_label'])}
    ]
    # return json.dumps(messages, ensure_ascii=False)
    return messages

# 새로운 열 'messages'를 생성합니다
train_model_input = train.apply(create_message_list, axis=1)
# valid_model_input = val.apply(create_message_list, axis=1)


# # Tokenization function
# def tokenize(batch):
#     return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

# train_ds = train_ds.map(tokenize, batched=True)
# val_ds = val_ds.map(tokenize, batched=True)

# # Set format for PyTorch
# columns = ['input_ids', 'attention_mask', 'label']
# train_ds.set_format(type='torch', columns=columns)
# val_ds.set_format(type='torch', columns=columns)


# # CUSTOM MAP@3 METRIC

# from sklearn.metrics import average_precision_score

# def compute_map3(eval_pred):
#     logits, labels = eval_pred
#     probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
#     top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
#     match = (top3 == labels[:, None])

#     # Compute MAP@3 manually
#     map3 = 0
#     for i in range(len(labels)):
#         if match[i, 0]:
#             map3 += 1.0
#         elif match[i, 1]:
#             map3 += 1.0 / 2
#         elif match[i, 2]:
#             map3 += 1.0 / 3
#     return {"map@3": map3 / len(labels)}


# Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
#     compute_metrics=compute_map3,
# )

#trainer.train()


#trainer.save_model(f"ver_{VER}")      
#tokenizer.save_pretrained(f"ver_{VER}")


test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test.shape )
test.head()


test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)

test.head()


test['system'] = system_prompt


def create_message_list(row):
    messages = [
        {"role": "system", "content": row['system']},
        {"role": "user", "content": row['text']},
        # {"role": "assistant", "content": str(row['special_label'])}
    ]
    # return json.dumps(messages, ensure_ascii=False)

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    # enable_thinking=False
    )
    return text

# 새로운 열 'messages'를 생성합니다
test_input = test.apply(create_message_list, axis = 1)
# valid_model_input = val.apply(create_message_list, axis=1)


allowed_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in special_character_list]

from transformers import LogitsProcessor
class LabelOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(scores, float('-inf'))
        if scores.dim() == 1:
            mask[self.allowed_token_ids] = 0
        elif scores.dim() == 2:
            mask[:, self.allowed_token_ids] = 0
        else:
            raise ValueError("Unexpected score dimensions")
        return scores + mask


sampling_params = SamplingParams(
    # temperature=0.7,
    temperature=0,
    max_tokens=1,
    logprobs=8,
    stop=["\n", "."],
    logits_processors=[LabelOnlyLogitsProcessor(allowed_token_ids)],
)


sampled_outputs = llm.generate(test_input, sampling_params)


special_to_idx = {i:idx for idx, i in enumerate(special_character_list)}


special_to_idx['■']


sampled_preds = [tokenizer.decode(list(out.outputs[0].logprobs[0])[:3]) for out in sampled_outputs]
sampled_pred_ids = [[special_to_idx[j] for j in pred]for pred in sampled_preds]


last_sampled_pred = [le.inverse_transform(i) for i in sampled_pred_ids]


# Join 3 labels per row with space
joined_preds = [" ".join(row) for row in last_sampled_pred]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()











































