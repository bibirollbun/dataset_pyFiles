# Usage
import pandas as pd
import json
import numpy as np
topic_dict ={0: 'Algebra',
 1: 'Geometry and Trigonometry',
 2: 'Calculus and Analysis',
 3: 'Probability and Statistics',
 4: 'Number Theory',
 5: 'Combinatorics and Discrete Math',
 6: 'Linear Algebra',
 7: 'Abstract Algebra and Topology'}
to_label = {'Algebra': 0,
 'Geometry and Trigonometry': 1,
 'Calculus and Analysis': 2,
 'Probability and Statistics': 3,
 'Number Theory': 4,
 'Combinatorics and Discrete Math': 5,
 'Linear Algebra': 6,
 'Abstract Algebra and Topology': 7}
df = pd.read_csv("train.csv")
syn_df = pd.read_csv("math_questions_4.csv")


counts = df['label'].value_counts()
for i in counts.index:
    print(f"{topic_dict[i]}: {counts[i]}")


syn_df.dropna(inplace=True)


syn_df['Topic'].value_counts()


df["Topic"] = df['label'].apply(lambda x:topic_dict[x])


syn_df['label'] = syn_df['Topic'].apply(lambda x:to_label[x])



concat_df = pd.concat([df, syn_df[df.columns]], ignore_index=True)


from transformers import AutoTokenizer
model_name = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
prompt_name = "finetuning"
with open(f'prompts/{prompt_name}.txt', 'r') as f:
    system_prompt = f.read()



concat_df['tokens_len'] = concat_df['Question'].apply(lambda x:len(tokenizer.apply_chat_template(
                                                    [
                                                        {"role": "user", "content": system_prompt.format(x)},
                                                        
                                                    ],
                                                    tokenize=True,
                                                    add_bos=True,
                                                    add_generation_prompt=True,
                                                )))


for i  in range(8):
    print(f'{topic_dict[i]}:{len(tokenizer.tokenize(topic_dict[i]))}')


concat_df['tokens_len'].describe()


concat_df['tokens_len'] = concat_df['tokens_len'] + concat_df['label'].apply(lambda x:len(tokenizer.tokenize(topic_dict[x])))
concat_df['tokens_len'].describe()


concat_df['tokens_len'].quantile(0.96)


counts = concat_df[concat_df['tokens_len']>256]['Topic'].value_counts()
counts


concat_df_lte256 = concat_df[concat_df['tokens_len']<=256]
filter_list = ['Algebra','Geometry and Trigonometry','Number Theory','Combinatorics and Discrete Math','Calculus and Analysis', 'Abstract Algebra and Topology', 'Linear Algebra', 'Probability and Statistics']
concat_df_lte256["Topic"].value_counts()


from semhash import SemHash
from model2vec import StaticModel

# Load a model from the HuggingFace hub (in this case the potion-base-8M model)
model = StaticModel.from_pretrained("minishlab/potion-base-32M")


all_classes_dfs = [concat_df_lte256[concat_df_lte256["Topic"]==topic] for topic in filter_list]
cleaned_examples = [0]*len(filter_list)
metadata = {}


index = 7
threshold = 0.97
print("Topic: ", filter_list[index])
# Initialize a SemHash instance with the training data
records = all_classes_dfs[index].reset_index(drop=False).to_dict(orient="records")
semhash = SemHash.from_records(records=records, columns=['Question'], model=model)
# Deduplicate the texts
deduplicated = semhash.self_deduplicate(threshold=threshold).selected

cleaned_examples[index] = deduplicated
metadata[filter_list[index]] = [filter_list[index], len(cleaned_examples[index]), threshold, ]
len(cleaned_examples[index])


pd.DataFrame(metadata, index=["Topic", "Total","deduplicate_thresh" ])


filtered = pd.DataFrame(item for de in cleaned_examples for item in de)
filtered.set_index("index", inplace=True)
filtered['Topic'].value_counts()


# last filter for very similar examples in diffrent classes 

records = filtered.reset_index(drop=False).to_dict(orient="records")
semhash = SemHash.from_records(records=records, columns=['Question'], model=model)
# Deduplicate the texts
deduplicated = semhash.self_deduplicate(threshold=0.97)


indices_to_remove = set()
for duplicate_record_dict in deduplicated.filtered: # Use your 'deduplicated.filtered'
    # The 'duplicates' attribute is a list of tuples, where the first element of the tuple is the duplicate record (a dict)
    print(duplicate_record_dict)
    indices_to_remove.add(duplicate_record_dict.record['index'])

print(f"Indices identified as duplicates to be removed: {indices_to_remove}")

unique_records = [record for record in deduplicated.selected if record['index'] not in indices_to_remove]

print(f"Original number of records: {len(records)}")
print(f"Number of records after removing identified duplicates: {len(unique_records)}")


# If 'index' from SemHash output refers to a column named 'index' in your DataFrame:
filtered_unique_df = filtered[~filtered.index.isin(indices_to_remove)]

print(f"Original DataFrame shape: {filtered.shape}")
print(f"DataFrame shape after removing duplicates: {filtered_unique_df.shape}")



filtered_unique_df['Topic'].value_counts()


used = concat_df_lte256[~concat_df_lte256['Topic'].isin(filter_list)]
final_df = pd.concat([filtered, used]) if not used.empty else filtered
final_df['Topic'].value_counts()


max_per_class = 800  
def balance_class(df):
    n = len(df)
    if n > max_per_class:
        return df.sample(n=max_per_class, random_state=42)
    else:
        return df.sample(n=max_per_class, replace=True, random_state=42)




sampled_df = (
    final_df
    .groupby('label', group_keys=False)
    .apply(balance_class)
)



sampled_df['Topic'].value_counts()


output_json = "math_topics.json"

samples = []

for i in range(len(sampled_df)):
    row = sampled_df.iloc[i]

    samples.append({
        "instruction": system_prompt.format(""),
        "input": row["Question"].strip(),
        "output": topic_dict[int(row["label"])]
    })

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)


sampled_df.to_csv("cleaned_800_256.csv", index=False)


# Filter final_df to get rows whose 'index' is NOT in sampled_original_indices inluding 
unique_samples = sampled_df.index.unique()
second_stage_df = filtered_unique_df[~filtered_unique_df.index.isin(unique_samples)].copy()


second_stage_df["Topic"].value_counts()


import pandas as pd

# Example dictionary: number of samples per topic
samples_per_topic = {
    "Probability and Statistics": 3*132,
    "Linear Algebra": 2*223,
    "Calculus and Analysis": 2*266,
    "Abstract Algebra and Topology": 392
}

# Filter only the topics you're interested in
filtered_df = sampled_df[sampled_df['Topic'].isin(samples_per_topic.keys())]

# Create an empty list to store results
sampled_list = []

# Sample from each topic
for topic, n_samples in samples_per_topic.items():
    topic_df = filtered_df[filtered_df['Topic'] == topic]
    
    # Group by 'label' and sample proportionally, or however you prefer
    sampled_topic = topic_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n_samples), random_state=42)
    )
    sampled_list.append(sampled_topic)

# Combine the results into a single DataFrame
N_C_samples = pd.concat(sampled_list, ignore_index=True)



final_second_stage_df = pd.concat([second_stage_df, concat_df[concat_df['tokens_len']>256], N_C_samples])
final_second_stage_df['Topic'].value_counts()


final_second_stage_df.to_csv("second_stage_1200_256.csv", index=False)


to_filter = concat_df[concat_df['Topic'].isin(filter_list)]
used = concat_df[~concat_df['Topic'].isin(to_filter)]
to_filter['Topic'].value_counts()
records = to_filter.to_dict(orient="records")
semhash = SemHash.from_records(records=records, columns=['Question'], model=model)
# Deduplicate the texts
deduplicated = semhash.self_deduplicate(threshold=0.9).selected
filtered = pd.DataFrame(deduplicated)


filtered['Topic'].value_counts()


final_df = pd.concat([filtered, used[used.columns]], ignore_index=True)
final_df['Topic'].value_counts()


P_df, L_df, AT_df = syn_df[syn_df["label"]==3],syn_df[syn_df["label"]==6], syn_df[syn_df["label"]==7]
P_df


from semhash import SemHash
from model2vec import StaticModel

# Load a model from the HuggingFace hub (in this case the potion-base-8M model)
model = StaticModel.from_pretrained("minishlab/potion-base-32M")



# Initialize a SemHash instance with the training data
records = L_df.to_dict(orient="records")
semhash = SemHash.from_records(records=records, columns=['Question'], model=model)
# Deduplicate the texts
L_deduplicated = semhash.self_deduplicate(threshold=0.9).selected
len(L_deduplicated)



# Initialize a SemHash instance with the training data
records = AT_df.to_dict(orient="records")
semhash = SemHash.from_records(records=records, columns=['Question'], model=model)
# Deduplicate the texts
AT_deduplicated = semhash.self_deduplicate(threshold=0.8).selected
len(AT_deduplicated)


syn_df = pd.DataFrame.from_records([*L_deduplicated, *AT_deduplicated, *P_df.to_dict(orient="records")])
syn_df['Topic'].value_counts()


all_classes_dfs = [concat_df[concat_df["label"]==i] for i in range(8)]
all_classes_dfs[0]


cleaned_examples = [0]*8
metadata = {}



index = 0
threshold = 0.9
# Initialize a SemHash instance with the training data
records = all_classes_dfs[index].to_dict(orient="records")
semhash = SemHash.from_records(records=records, columns=['Question'], model=model)
# Deduplicate the texts
deduplicated = semhash.self_deduplicate(threshold=threshold).selected

cleaned_examples[index] = deduplicated
metadata[topic_dict[index]] = [topic_dict[index], len(cleaned_examples[index]), threshold, ]
len(cleaned_examples[index])


pd.DataFrame(metadata, index=["Topic", "Total","deduplicate_thresh" ])


final_df = pd.DataFrame(item for de in cleaned_examples for item in de)
final_df['Topic'].value_counts()


final_df.to_csv("cleaned<=256.csv", index=False)


max_per_class = 1000  
sampled_df = (
    final_df[final_df['tokens_len']<=256].groupby('label', group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=42))
)



sampled_df['Topic'].value_counts()


sampled_df.to_csv("cleaned_once<=256.csv", index=False)


max_per_class = 1000  
sampled_df = (
    final_df[(final_df['tokens_len']>256) & (final_df['tokens_len']<=512)].groupby('label', group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=42))
)

sampled_df['Topic'].value_counts()



sampled_df.to_csv("cleaned>256.csv", index=False)


output_json = "math_topics.json"

samples = []

for i in range(len(sampled_df)):
    row = sampled_df.iloc[i]

    samples.append({
        "instruction": system_prompt.format(""),
        "input": row["Question"].strip(),
        "output": topic_dict[int(row["label"])]
    })

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)


