from transformers import AutoModel, AutoTokenizer
import torch


import os
print(os.getcwd())
state_dict = torch.load("/kaggle/input/embedd_model/transformers/default/1/best_model_proposed_110.pth", map_location="cuda")



model_path = '/kaggle/input/mathbert_model/transformers/default/1'
tokenizer_path = '/kaggle/input/mathbert_tokenizer/transformers/default/1'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(model_path) 
model = model.to("cuda")





import pandas as pd
from sklearn.preprocessing import LabelEncoder

# train_data = prepare_df('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')



model = torch.nn.DataParallel(model)


# clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}



model.load_state_dict(state_dict)



model


def prepare_df1(path):
    # path = args.path
    df = pd.read_csv(path)
    df['text_train'] = df.apply(
    lambda row: f"[CLS] Question: {row['QuestionText']}\n[SEP] Student's Answer: {row['MC_Answer']}\n[SEP] Student's explanation: {row['StudentExplanation']}[SEP]\n ",
    axis=1
    )
    return df


test_data = prepare_df1('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


from datasets import Dataset
test_dataset = Dataset.from_pandas(test_data)


import ast


import torch.nn.functional as F


idx = '/kaggle/input/training-dataset-embedding/embeddings_test190.csv'
infer = pd.read_csv(idx)
# print(infer['label'][0])
infer_embeds_list = infer['text_embed'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))
infer_embeds = torch.stack(list(infer_embeds_list.values))
infer_embeds = F.normalize(infer_embeds, dim=1)
infer_labels = infer['label']


infer.iloc[0]


print(infer_embeds[0])
print(infer_labels[0])


submission = []
count = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer_embeds = infer_embeds.to(device)


for i in range(len(test_dataset)):
    text = test_dataset[i]['text_train']
    tokenized_text= tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=512,
                        return_tensors="pt"
                    )
    tokenized_text = tokenized_text.to(device)
    with torch.no_grad():
        output = model(**tokenized_text)
    query = output.last_hidden_state[:, 0, :]          
    query = F.normalize(query, dim=1)                 

    
    sims = torch.matmul(query, infer_embeds.T).squeeze(0)  

    sorted_idx = torch.argsort(sims, descending=True)

    seen_labels = set()
    top_results = []
    for idx in sorted_idx.tolist():
        lbl = infer_labels[idx]
        if lbl not in seen_labels:
            score = sims[idx].item()
            top_results.append(lbl)
            seen_labels.add(lbl)
        if len(top_results) == 3:
            prediction = str(top_results[0]+" "+top_results[1]+" "+top_results[2])
            submission.append({"row_id":test_dataset[i]["row_id"] , "Category:Misconception":prediction})
            if count < 3 :
                print(submission[count])
                count+=1
            break
        
    


submission = pd.DataFrame(submission)
submission.to_csv("submission.csv",index=False)


# if hasattr(model, "module"):
#     model_to_save = model.module  # unwrap the original model
# else:
#     model_to_save = model

# # Now save
# model_to_save.save_pretrained("./mathbert_model")
# tokenizer.save_pretrained("./mathbert_tokenizer")


# !zip -r mathbert_tokenizer.zip mathbert_tokenizer
# !zip -r mathbert_model.zip mathbert_model


submission.head()


# from tqdm import tqdm
# new_train = []
# for i in tqdm(range(len(train_data))):
#     text = train_data.iloc[i]['text_train']
#     out = model(**tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors = "pt"))
#     label = train_data.iloc[i]['text_label']
#     new_train.append({"label": label, "text": out.last_hidden_state[:,0,:]})


# new_train[0]['text'].shape


# for i, x in enumerate(infer['label']):
#     print(i, x)




