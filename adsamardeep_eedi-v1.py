!pip uninstall -y torch
!pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm vllm
!pip install -U --upgrade /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -U --upgrade /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl
!pip install --no-deps --no-index /kaggle/input/logits-processor-zoo/logits_processor_zoo-0.1.0-py3-none-any.whl


import pandas as pd
import numpy as np

raw_train_data = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")
raw_test_data = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
misc_mappings = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")


import re

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    
    return text


train_data = raw_train_data.drop(columns=["ConstructId", "SubjectId"])
test_data = raw_test_data.drop(columns=["ConstructId", "SubjectId"])
train_data_misc = train_data.drop(columns=['ConstructName', 'CorrectAnswer', 'SubjectName', 'QuestionText', 'AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']).melt(id_vars=['QuestionId'], var_name="Option", value_name='MisconceptionId').sort_values(by=['QuestionId', 'Option']).reset_index(drop=True)
train_data_ans = train_data.drop(columns=['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']).melt(id_vars=['QuestionId', 'ConstructName', 'SubjectName', 'CorrectAnswer', 'QuestionText'], var_name="AnswerID", value_name='AnswerText').sort_values(by=['QuestionId', 'AnswerID']).reset_index(drop=True)
train_data = train_data_ans.copy()
train_data['MisconceptionId'] = train_data_misc['MisconceptionId']
train_data = train_data.dropna(axis=0).reset_index(drop=True)
train_data.head()


test_data = test_data.melt(id_vars=['QuestionId', 'ConstructName', 'SubjectName', 'CorrectAnswer', 'QuestionText'], var_name="AnswerID", value_name='AnswerText')
test_data["AnswerID"] = test_data["AnswerID"].map(lambda x: x[6])
test_data = test_data[test_data['CorrectAnswer'] != test_data['AnswerID']]
test_data = test_data.sort_values(by=['QuestionId', 'AnswerID'])
test_data = test_data.reset_index(drop=True)
test_data['QuestionText'] = test_data['QuestionText'].apply(preprocess)
test_data['AnswerText'] = test_data['AnswerText'].apply(preprocess)
test_data.head()


import vllm

llm = vllm.LLM(
    "/kaggle/input/qwen2.5/transformers/7b-instruct/1",
    dtype="half",
    tensor_parallel_size=2,
)

llm_tokenizer = llm.get_tokenizer()


from transformers import AutoTokenizer, AutoModel
import torch    # Note: Don't import this BEFORE vllm initialization!

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device

embed_model_name = '/kaggle/input/baai/transformers/bge-large-en-v1.5/1'

embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = AutoModel.from_pretrained(embed_model_name)
embed_model.to(device)


def generate_embeddings(texts, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokenized_batch = embed_tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024).to(device)
            batch_output = embed_model(**tokenized_batch)
            batch_embeddings = batch_output.last_hidden_state[:, 0, :]  # Embedding of [CLS]
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings)
            
    return torch.concatenate(embeddings, axis=0)

misc_list = list(misc_mappings["MisconceptionName"].apply(preprocess))
misc_embeddings = generate_embeddings(misc_list)


solver_prompt = '''# Task description: You are a smart math solver. Your task is to solve the given problem step by step.

# Rules:
1. You must list all of your solving steps.
2. You must keep your solution short and concise.
3. Your solution must be correct.
4. All rules are equally important. Do not break any rules.

# Input
Question: "{}"
Solution: '''

def generate_solutions(df):
    # the default system message will be used
    messages = [[{"role": "user", "content": solver_prompt.format(x['QuestionText'])}] for i, x in df.iterrows()]
    text = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    responses = llm.generate(
        text,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0,  # randomness of the sampling
            seed=42,
            skip_special_tokens=True,
            max_tokens=1024
        ),
        use_tqdm = True
    )
    
    return [r.outputs[0].text for r in responses]


problem_desc = '''
Topic: "{}; {}"
Question: "{}"
Ground-truth solution for reference: "{}"
Erroneous answer: "{}"
'''

prompt = '''# Task description: You are a helpful and smart math teacher. You will read a math question, a step-by-step ground-truth solution for reference, and an erroneous answer given by a student. Your task is to identify and summarize the student's misconception that leads to the error.

# Rules:
1. You should summarize the student's misconception in one sentence.
2. Your response should be short, accurate and concise.
3. Your response should not include any of your reasoning process, only output the answer itself.
4. You do not need to offer your solution to the given question.
5. Your summarization should be a general high-level math concept, rather than a specific description about the question. For example, your response should look like "Did not divide the sum by the number of elements when calculating mean.", rather than "Tom is incorrect; Tom did not divide the total sum with the number of elements."
6. Do not mention any names from the question in your summarization.
7. All rules are equally important. Do not break any rules.

# Example 1
"""
Topic: "Mental Multiplication and Division; Solve missing number mental multiplication questions"
Question: "What should replace the star? \[ \bigstar \times 4=108 \]"
Ground-truth solution: "'To find what should replace the star in the equation \\[ \\bigstar \\times 4 = 108 \\], you need to solve for \\(\\bigstar\\).\n\nThe equation can be rewritten as:\n\\[ \\bigstar = \\frac{108}{4} \\]\n\nNow, perform the division:\n\\[ \\bigstar = 27 \\]\n\nSo, the star should be replaced with the number 27.'"
Erroneous answer: "\( 104 \)"
Misconception: "Subtracts instead of divides."
"""

# Input
'''


# https://www.kaggle.com/code/cdeotte/infer-34b-with-vllm

def generate_responses(df):

    # Step 1: Generate solutions
    solutions = generate_solutions(df)
    
    # Step 2: Generate misconception
    messages = [[{"role": "user", "content": prompt + problem_desc.format(x['SubjectName'], x['ConstructName'], x['QuestionText'], solutions[i], x['AnswerText']) + 'Misconception: '}] for i, x in df.iterrows()]
    text = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    responses = llm.generate(
        text,
        vllm.SamplingParams(
            n=1,  # Number of output sequences to return for each prompt.
            top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
            temperature=0,  # randomness of the sampling
            seed=42,
            skip_special_tokens=True,
            max_tokens=1024
        ),
        use_tqdm = True
    )
    
    return solutions, [r.outputs[0].text for r in responses]


from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor

multiple_choice_prompt = '''# Task description: You are a helpful and smart math teacher. You will read a math question, a step-by-step ground-truth solution for reference, an erroneous answer given by a student, and a list of possible misconceptions. Your task is to identify and select the student's misconception that leads to the error from the given list.

# Rules:
1. You must select exactly one misconception from the given list.
2. You must only output the index of the misconception, not the description itself.
3. All rules are equally important. Do not break any rules.

# Input:
'''

def get_multiple_choice_prompt(row, solution, matches, n_choices):
    p = (multiple_choice_prompt 
        + problem_desc.format(
            row['SubjectName'], 
            row['ConstructName'], 
            row['QuestionText'], 
            solution, 
            row['AnswerText']) 
        + 'List of potential misconceptions: ')
    for i in range(n_choices):
        p += '\n{}. "{}"'.format(i, misc_list[matches[i]])

    return p + 'Most possible misconception index: '
    
    
def multiple_choice(df, solutions, matches, n_choices=10):
    # the default system message will be used
    messages = [[{
        "role": "user", 
        "content": get_multiple_choice_prompt(
            x, 
            solutions[i], 
            matches[i],
            n_choices
        )
    }] for i, x in df.iterrows()]
    
    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    responses = llm.generate(
        text,
        vllm.SamplingParams(
            n=1,  
            temperature=0,
            top_k=1,
            seed=42,
            logits_processors = [MultipleChoiceLogitsProcessor(
                llm_tokenizer,
                choices=[str(i) for i in range(n_choices)], 
                delimiter=".")],
            max_tokens=1
        ),
        use_tqdm = True
    )
    
    return [r.outputs[0].text for r in responses]


from sklearn.metrics.pairwise import cosine_similarity

def retrieve(df, k=200, n_choices=10):
    
    # Generate response
    solutions, responses = generate_responses(df)
    
    # First stage retrieval
    qa_texts = ['\n'.join([x['ConstructName'], x['SubjectName'], x['QuestionText'], x['AnswerText']]) for i, x in df.iterrows()]
    qa_embeddings = generate_embeddings(qa_texts)
    
    cos_sim_matrix = cosine_similarity(qa_embeddings.cpu(), misc_embeddings.cpu())
    first_matches = torch.tensor((-cos_sim_matrix).argsort(axis=1)[:,:k])
    first_retrieve_embeddings = misc_embeddings[first_matches]

    
    # Second stage retrieval
    response_texts = [qa_texts[i] + '\nMisconception: {}'.format(responses[i]) for i, x in df.iterrows()]
    response_embeddings = generate_embeddings(response_texts)
    response_embeddings = response_embeddings.cpu()
    first_retrieve_embeddings = first_retrieve_embeddings.cpu()
    second_matches = []

    for i in range(first_retrieve_embeddings.shape[0]):
        cos_sim_matrix_2 = cosine_similarity([response_embeddings[i]], first_retrieve_embeddings[i])
        matches = first_matches[i, (-cos_sim_matrix_2).argsort(axis=1)[:,:25]].squeeze()
        second_matches.append(matches)

    # Move the best choice to the front
    choices = multiple_choice(df, solutions, second_matches, n_choices)
    final_matches = []
    for i in range(len(choices)):
        best_idx = int(choices[i])
        final_matches.append(second_matches[i].clone())
        if 0 <= best_idx < 25:
            temp = final_matches[i][0].clone()
            final_matches[i][0] = final_matches[i][best_idx]
            final_matches[i][best_idx] = temp

    return final_matches, first_matches, second_matches, solutions, responses


# Performance metrics

def mapk(preds, labels, k):
    # Slightly modified Map@K implementation since there is only one correct label per prediction
    score = 0
    for p, lb in zip(preds, labels):
        matches = np.where(p[:k] == lb)[0]
        if matches.shape[0] == 0:
            apk = 0
        else:
            apk = 1/min(matches[0]+1, k)
            
        score += apk

    return score / len(labels)


def hitk(preds, labels, k=25):
    # Return the ratio
    score = 0
    for p, lb in zip(preds, labels):
        matches = np.where(p[:k] == lb)[0]
        if matches.shape[0] == 0:
            apk = 0
        else:
            apk = 1
            
        score += apk

    return score / len(labels)


%%time
# Evaluate performance on train data
n_first_stage_samples = 100
is_testing = True
if is_testing:
    train_samples = train_data.sample(n=500).reset_index(drop=True)
    final_matches, first_matches, second_matches, responses, solutions = retrieve(train_samples, n_first_stage_samples, 10)
    misc_np = train_samples['MisconceptionId'].to_numpy()
    print('Retrieval Results')
    print('\tFirst retrieval')
    print('\t\tHit@1={:.3f}'.format(hitk(first_matches, misc_np, 1)))
    print('\t\tMap@10={:.3f}'.format(mapk(first_matches, misc_np, 10)))
    print('\t\tHit@10={:.3f}'.format(hitk(first_matches, misc_np, 10)))
    print('\t\tMap@25={:.3f}'.format(mapk(first_matches, misc_np, 25)))
    print('\t\tHit@25={:.3f}'.format(hitk(first_matches, misc_np, 25)))
    print('\t\tMap@{}={:.3f}'.format(n_first_stage_samples, mapk(first_matches, misc_np, n_first_stage_samples)))
    print('\t\tHit@{}={:.3f}'.format(n_first_stage_samples, hitk(first_matches, misc_np, n_first_stage_samples)))
    print('\tSecond retrieval')
    print('\t\tHit@1={:.3f}'.format(hitk(second_matches, misc_np, 1)))
    print('\t\tMap@10={:.3f}'.format(mapk(second_matches, misc_np, 10)))
    print('\t\tHit@10={:.3f}'.format(hitk(second_matches, misc_np, 10)))
    print('\t\tMap@25={:.3f}'.format(mapk(second_matches, misc_np, 25)))
    print('\t\tHit@25={:.3f}'.format(hitk(second_matches, misc_np, 25)))
    print('\tFinal retrieval')
    print('\t\tHit@1={:.3f}'.format(hitk(final_matches, misc_np, 1)))
    print('\t\tMap@10={:.3f}'.format(mapk(final_matches, misc_np, 10)))
    print('\t\tHit@10={:.3f}'.format(hitk(final_matches, misc_np, 10)))
    print('\t\tMap@25={:.3f}'.format(mapk(final_matches, misc_np, 25)))
    print('\t\tHit@25={:.3f}'.format(hitk(final_matches, misc_np, 25)))


if is_testing:
    print('Retrieval Results')
    print('\tFirst retrieval')
    print('\t\tHit@1={:.3f}'.format(hitk(first_matches, misc_np, 1)))
    print('\t\tMap@10={:.3f}'.format(mapk(first_matches, misc_np, 10)))
    print('\t\tHit@10={:.3f}'.format(hitk(first_matches, misc_np, 10)))
    print('\t\tMap@25={:.3f}'.format(mapk(first_matches, misc_np, 25)))
    print('\t\tHit@25={:.3f}'.format(hitk(first_matches, misc_np, 25)))
    print('\t\tMap@{}={:.3f}'.format(n_first_stage_samples, mapk(first_matches, misc_np, n_first_stage_samples)))
    print('\t\tHit@{}={:.3f}'.format(n_first_stage_samples, hitk(first_matches, misc_np, n_first_stage_samples)))
    print('\tSecond retrieval')
    print('\t\tHit@1={:.3f}'.format(hitk(second_matches, misc_np, 1)))
    print('\t\tMap@10={:.3f}'.format(mapk(second_matches, misc_np, 10)))
    print('\t\tHit@10={:.3f}'.format(hitk(second_matches, misc_np, 10)))
    print('\t\tMap@25={:.3f}'.format(mapk(second_matches, misc_np, 25)))
    print('\t\tHit@25={:.3f}'.format(hitk(second_matches, misc_np, 25)))
    print('\tFinal retrieval')
    print('\t\tHit@1={:.3f}'.format(hitk(final_matches, misc_np, 1)))
    print('\t\tMap@10={:.3f}'.format(mapk(final_matches, misc_np, 10)))
    print('\t\tHit@10={:.3f}'.format(hitk(final_matches, misc_np, 10)))
    print('\t\tMap@25={:.3f}'.format(mapk(final_matches, misc_np, 25)))
    print('\t\tHit@25={:.3f}'.format(hitk(final_matches, misc_np, 25)))


second_matches[1]


final_matches[1]


'''
Note:
Misc #2142 = Misc #2068 = 'Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term '


Problems/ Potential improvements:
1. Extract keywords from the question
2. LLM seems pretty bad at digit pattern recognition (ex. failed to recognize "0.432" is "43.2" divided by 100)
3. LLM seems to have a hard time recognizing the error just by looking at the result
4. Some problems are simply unclear / ill-defined.
'''
if is_testing:
    idx = 1
    question = train_samples['QuestionText'].iloc[idx]
    wrong_answer = train_samples['AnswerText'].iloc[idx]
    misc = misc_list[train_samples['MisconceptionId'].iloc[idx].astype(int)]
    matches_texts = [misc_list[x] for x in final_matches[idx]]
    m = ''
    for i, t in enumerate(matches_texts):
        if t == misc:
            m += '✓'
     
        m += '\t{}. {}\n'.format(i+1, t)
    
    print('Question: {}\n\nErroneous answer: {}\n\nMisconception: {}\n\nSolution:\n{}\n\nResponse: {}\n\nMatches:\n{}'.format(question, wrong_answer, misc, responses[idx], solutions[idx], m))


%%time
final_matches, first_matches, second_matches, responses, solutions = retrieve(test_data, n_first_stage_samples, 10)
final_matches = np.array(final_matches)


output_dict = {}
output_dict["QuestionId_Answer"] = test_data.apply(lambda x: str(x["QuestionId"]) + "_" + x["AnswerID"], axis=1)
output_dict["MisconceptionId"] = [" ".join(x.astype('str')) for x in final_matches]
output_df = pd.DataFrame.from_dict(output_dict)
output_df.to_csv('/kaggle/working/submission.csv', index=False)

