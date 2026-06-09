from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
import torch
from torch import nn
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib


test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


#test_df = pd.concat([test_df] * 10, ignore_index=True)


test_df['text'] = [
    f"Question: {q}\nAnswer: {a}\nExplanation: {e}"
    for q,a,e in zip(test_df['QuestionText'],test_df['MC_Answer'],test_df['StudentExplanation'])
]


test_df = test_df.drop(['QuestionId','QuestionText','MC_Answer','StudentExplanation'],axis=1)


le = joblib.load('/kaggle/input/gemma3-misconception-model/label_encoder_Gemma3.pkl')


NUM_CLASSES = len(le.classes_)
class Gemma3_Head(nn.Module):
    def __init__(self,base_model,num_classes):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size,num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,input_ids,attention_mask,labels=None):
        outputs = self.base_model(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, -1, :]
        logits = self.classifier(pooled_output)

        

        if labels is not None:
            loss = self.loss_fn(logits,labels)
            return loss,logits
        else:
            return logits


#model
model_name = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0', attn_implementation='eager')

peft_model = PeftModel.from_pretrained(model,'/kaggle/input/gemma3-misconception-model/Gemma_Lora_Adapter')
Gemma3model = Gemma3_Head(peft_model,NUM_CLASSES)


Gemma3model.classifier.load_state_dict(torch.load('/kaggle/input/gemma3-misconception-model/Gemma_Classifier_Head.pth',map_location=torch.device('cpu')))


from torchinfo import summary
summary(Gemma3model)


ds = Dataset.from_pandas(test_df)
ds = ds.with_format('torch')

#Tokenize
def tokenize(examples):
    # This tokenizes batches of text, not single sentences
    return tokenizer(
        examples["text"],  # Use the column with text data
        truncation=True,
        max_length=512,
    )

tokenized_ds = ds.map(
    tokenize,
    batched=True,
    num_proc=4,
    remove_columns=['row_id','text'],
)

test_dataloader = DataLoader(
    tokenized_ds,
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
Gemma3model.to(device)
Gemma3model.base_model.to(device)
test_pbar = tqdm(test_dataloader,desc='Testing',leave=False)
ans = []

Gemma3model.eval()
with torch.inference_mode():
    for batch in test_pbar:
        batch = {k: v.to(device) for k,v in batch.items()}
        logits = Gemma3model(**batch)
        _,preds = torch.topk(logits,k=3,dim=1)
        ans.append(preds.cpu().numpy())


answers = [le.inverse_transform(answer).tolist() for batch_answer in ans for answer in batch_answer]


sub_format = [f"{batch_answer[0]} {batch_answer[1]} {batch_answer[2]}" for batch_answer in answers]


test_df['Category:Misconception'] = sub_format


submission = test_df.drop(['text'],axis=1)


submission.to_csv('submission.csv',index=False)

