import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


INPUT_PATH = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena'
OUTPUT_PATH = '/kaggle/working'
MODEL_PATH = '/kaggle/input/gte-multilingual-reranker-base/transformers/default/1'


test_data = pd.read_parquet(f'{INPUT_PATH}/test.parquet')


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
MAX_LENGTH = 1024


for col in ['prompt', 'response_a', 'response_b']:
    text_list = []

    if col == 'prompt':
        max_no = round(MAX_LENGTH / 5)
        s_no = round(max_no / 2) - 1
        e_no = round(-max_no / 2)
    else:
        max_no = round(MAX_LENGTH / 2.5)
        s_no = round(max_no / 2) - 1
        e_no = round(-max_no / 2)

    for text in test_data[col]:
        encoded = tokenizer(text, return_offsets_mapping=True)
        if len(encoded['input_ids']) > max_no:
            start_idx, end_idx = encoded['offset_mapping'][s_no]
            new_text = text[:end_idx]
            start_idx, end_idx = encoded['offset_mapping'][e_no]
            new_text = new_text + "\n(snip)\n" + text[start_idx:]
            text = new_text
        text_list.append(text)

    test_data[col] = text_list


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'prompt': self.df.iloc[idx]['prompt'],
            'response_a': self.df.iloc[idx]['response_a'],
            'response_b': self.df.iloc[idx]['response_b'],
        }


test_dataset = CustomDataset(test_data)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16,
)


model = SentenceTransformer(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)


@torch.no_grad()
def inference(model, dataloader):
    args = {'show_progress_bar': False}
    predictions = torch.Tensor([])

    for inputs in dataloader:
        prompt_emb = model.encode(inputs['prompt'], **args)
        resp_emb0 = model.encode(inputs['response_a'], **args)
        resp_emb1 = model.encode(inputs['response_b'], **args)

        batch_preds = torch.stack((
            model.similarity(prompt_emb, resp_emb0).diag(),
            model.similarity(prompt_emb, resp_emb1).diag()
        )).argmax(axis=0)

        predictions = torch.cat([predictions, batch_preds])

    return predictions


test_data['winner'] = inference(model, test_dataloader)
test_data['winner'] = test_data['winner'].map(
    {0: 'model_a', 1: 'model_b'}
)


test_data[['id', 'winner']].to_csv(
    f'{OUTPUT_PATH}/submission.csv', index=False,
)

