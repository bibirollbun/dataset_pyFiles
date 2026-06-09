# # %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # # Setup

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:44:50.425169Z","iopub.execute_input":"2025-10-15T17:44:50.425441Z","iopub.status.idle":"2025-10-15T17:45:48.463874Z","shell.execute_reply.started":"2025-10-15T17:44:50.425421Z","shell.execute_reply":"2025-10-15T17:45:48.462923Z"},"jupyter":{"outputs_hidden":false}}
# !pip install --no-index --find-links=/kaggle/input/map-packages/ liger-kernel

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:45:48.465049Z","iopub.execute_input":"2025-10-15T17:45:48.465349Z","iopub.status.idle":"2025-10-15T17:45:48.61066Z","shell.execute_reply.started":"2025-10-15T17:45:48.465326Z","shell.execute_reply":"2025-10-15T17:45:48.609857Z"},"jupyter":{"outputs_hidden":false}}
# !cp -r /kaggle/input/map-modules map_modules

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:45:56.575329Z","iopub.execute_input":"2025-10-15T17:45:56.575892Z","iopub.status.idle":"2025-10-15T17:45:56.803558Z","shell.execute_reply.started":"2025-10-15T17:45:56.57586Z","shell.execute_reply":"2025-10-15T17:45:56.802605Z"},"jupyter":{"outputs_hidden":false}}
# # Much faster and more stable than /kaggle/input
# !mkdir /tmp/layer-checkpoints
# !mkdir /kaggle/working/layer-checkpoints

# # %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # # Prepare test data

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:45:58.261604Z","iopub.execute_input":"2025-10-15T17:45:58.26231Z","iopub.status.idle":"2025-10-15T17:45:58.267909Z","shell.execute_reply.started":"2025-10-15T17:45:58.262281Z","shell.execute_reply":"2025-10-15T17:45:58.267239Z"},"jupyter":{"outputs_hidden":false}}
# %%writefile prepare_test.py

# import pandas as pd

# DEBUG = False

# if DEBUG:
#     df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv").iloc[:1280]
# else:
#     df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
#     df["Category"] = "True_Correct"
#     df["Misconception"] = "NA"

# N = len(df)
# df1 = df.iloc[:N//2]
# df2 = df.iloc[N//2:]
# print(df1.shape, df2.shape)
# df1.to_csv("test_data1.csv", index=False)
# df2.to_csv("test_data2.csv", index=False)

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:46:00.843553Z","iopub.execute_input":"2025-10-15T17:46:00.844226Z","iopub.status.idle":"2025-10-15T17:46:03.100203Z","shell.execute_reply.started":"2025-10-15T17:46:00.844197Z","shell.execute_reply":"2025-10-15T17:46:03.099453Z"},"jupyter":{"outputs_hidden":false}}
# !python prepare_test.py

# # %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # # Inference

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:51:04.181409Z","iopub.execute_input":"2025-10-15T17:51:04.182152Z","iopub.status.idle":"2025-10-15T17:51:04.19261Z","shell.execute_reply.started":"2025-10-15T17:51:04.18212Z","shell.execute_reply":"2025-10-15T17:51:04.191616Z"},"jupyter":{"outputs_hidden":false}}
# %%writefile test.py

# import argparse
# import torch
# from torch.nn.attention.flex_attention import create_block_mask
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# from map_modules.data.dataset_v1 import MAPDataset as MAPDatasetV1
# from map_modules.data.dataset_v2 import MAPDataset as MAPDatasetV2
# from map_modules.data.dataset_v3 import MAPDataset as MAPDatasetV3
# from map_modules.models.modeling_qwen3_w8a8 import Qwen3ForSequenceClassification
# from map_modules.models.modeling_glm4_w8a8 import Glm4ForSequenceClassification
# from map_modules.utils import to_gpu
# from tqdm import tqdm


# DATASETS = {
#     "v1": MAPDatasetV1,
#     "v2": MAPDatasetV2,
#     "v3": MAPDatasetV3,
# }
# MODELS = {
#     "qwen3": (Qwen3ForSequenceClassification, (37, 27)),
#     "glm4": (Glm4ForSequenceClassification, (35, 26)),
# }


# def get_block_mask(input_ids, suffix_ids, doc_ids, position_ids):
#     def custom_mask(b, h, q_idx, kv_idx):
#         causal = q_idx >= kv_idx
#         same_suffix = (suffix_ids[q_idx] == suffix_ids[kv_idx]) | (
#             suffix_ids[kv_idx] == -1
#         )
#         same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
#         return causal & same_suffix & same_doc

#     return create_block_mask(
#         custom_mask,
#         B=None,
#         H=None,
#         Q_LEN=input_ids.size(0),
#         KV_LEN=input_ids.size(0),
#         BLOCK_SIZE=(128, 128),
#     )


# class Inferencer:
#     def __init__(self, model, num_layers, checkpoint_root):
#         self.num_layers = sum(num_layers)
#         self.layers_weights = [
#             torch.load(
#                 f"{checkpoint_root[0]}/layer_{i}.pth",
#                 map_location="cpu",
#                 mmap=True,
#                 weights_only=True,
#             )
#             for i in range(num_layers[0])
#         ]
#         offset = num_layers[0]
#         self.layers_weights.extend(
#             [
#                 torch.load(
#                     f"{checkpoint_root[1]}/layer_{i + offset}.pth",
#                     map_location="cpu",
#                     mmap=True,
#                     weights_only=True,
#                 )
#                 for i in range(num_layers[1])
#             ]
#         )

#         self.model = model.cuda().eval()

#         self.h2d_stream = torch.cuda.Stream()
#         self.curr_layer, self.next_layer = (
#             self.model.model.layers[0],
#             self.model.model.layers[1],
#         )

#     @torch.no_grad()
#     def forward(self, batches):
#         batches = to_gpu(batches)
#         block_masks = []
#         hidden_statess = []
#         position_embeddingss = []
#         last_tokenss = []
#         for micro_batch in batches:
#             input_ids = micro_batch["input_ids"].squeeze(0)
#             suffix_ids = micro_batch["suffix_ids"]
#             doc_ids = micro_batch["doc_ids"]
#             position_ids = micro_batch["position_ids"]
#             last_tokens = micro_batch["last_tokens"]

#             block_mask = get_block_mask(input_ids, suffix_ids, doc_ids, position_ids)
#             block_masks.append(block_mask)
#             hidden_states = self.model.model.embed_tokens(input_ids)
#             position_embeddings = self.model.model.rotary_emb(
#                 hidden_states, position_ids.unsqueeze(0)
#             )
#             hidden_statess.append(hidden_states)
#             position_embeddingss.append(position_embeddings)
#             last_tokenss.append(last_tokens)

#         curr_layer, next_layer = self.curr_layer, self.next_layer
#         curr_states, next_states = curr_layer.state_dict(), next_layer.state_dict()
#         for layer_idx in range(self.num_layers):
#             for m_idx, (hidden_states, block_mask, position_embeddings) in enumerate(
#                 zip(hidden_statess, block_masks, position_embeddingss)
#             ):
#                 hidden_states = curr_layer(
#                     hidden_states, block_mask, position_embeddings
#                 )
#                 hidden_statess[m_idx].copy_(hidden_states)

#             with torch.cuda.stream(self.h2d_stream):
#                 next_layer_idx = (layer_idx + 1) % self.num_layers
#                 next_layer_wegihts = self.layers_weights[next_layer_idx]
#                 for k, v in next_layer_wegihts.items():
#                     next_states[k].copy_(v, non_blocking=True)
#             torch.cuda.synchronize()
#             # alternate
#             curr_layer, next_layer = next_layer, curr_layer
#             curr_states, next_states = next_states, curr_states
#         self.curr_layer, self.next_layer = curr_layer, next_layer
#         hidden_statess = [
#             self.model.model.norm(hidden_states) for hidden_states in hidden_statess
#         ]

#         hidden_statess = [
#             hidden_states[last_tokens]
#             for hidden_states, last_tokens in zip(hidden_statess, last_tokenss)
#         ]
#         with torch.cuda.amp.autocast(dtype=torch.float16):
#             logitss = [
#                 self.model.score(hidden_states) for hidden_states in hidden_statess
#             ]
#         return logitss


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, required=True)
#     parser.add_argument("--dataset", type=str, required=True)
#     parser.add_argument(
#         "--model-path",
#         type=str,
#         required=True,
#     )
#     parser.add_argument(
#         "--checkpoint-root1",
#         type=str,
#         required=True,
#     )
#     parser.add_argument(
#         "--checkpoint-root2",
#         type=str,
#         required=True,
#     )
#     parser.add_argument("--csv-file", type=str, required=True)
#     parser.add_argument("--query", type=str, default=None)
#     parser.add_argument("--micro-batch-size", type=int, required=True)
#     parser.add_argument("--num-micro-batches", type=int, required=True)
#     parser.add_argument("--out", type=str, required=True)

#     return parser.parse_args()


# def main():
#     args = parse_args()
#     model_class, num_layers = MODELS[args.model]
#     model_path = args.model_path
#     checkpoint_root = (args.checkpoint_root1, args.checkpoint_root2)
#     dataset = DATASETS[args.dataset](
#         csv_file=args.csv_file,
#         tokenizer=AutoTokenizer.from_pretrained(model_path),
#         query=args.query,
#     )
#     dl = DataLoader(
#         dataset,
#         batch_size=args.micro_batch_size,
#         collate_fn=dataset.collate_fn,
#         shuffle=False,
#     )
#     model = model_class.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,
#         device_map="cuda",
#     )
#     model.eval()
#     inferencer = Inferencer(
#         model, num_layers=num_layers, checkpoint_root=checkpoint_root
#     )
#     micro_batches = [batch for batch in dl]
#     batches = []
#     for start in range(0, len(micro_batches), args.num_micro_batches):
#         batches.append(micro_batches[start : start + args.num_micro_batches])

#     preds = []
#     for batch in tqdm(batches):
#         logitss = inferencer.forward(batch)
#         for logits, micro_batch in zip(logitss, batch):
#             logits = logits.float().flatten()
#             for _logits in logits.split(micro_batch["num_candidates"]):
#                 preds.append(_logits.float().softmax(dim=-1).data.cpu())

#     print(dataset.evaluate(preds))
#     torch.save(preds, args.out)


# if __name__ == "__main__":
#     main()

# # %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # ## Model 0, Data 3

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:47:45.050516Z","iopub.execute_input":"2025-10-15T17:47:45.051328Z","iopub.status.idle":"2025-10-15T17:50:35.978962Z","shell.execute_reply.started":"2025-10-15T17:47:45.051296Z","shell.execute_reply":"2025-10-15T17:50:35.978052Z"},"jupyter":{"outputs_hidden":false}}
# %%time
# !for i in {0..36}; do cp /kaggle/input/map-checkpoints-x-2-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
# !for i in {37..63}; do cp /kaggle/input/map-checkpoints-x-2-layers/layer_$i.pth /tmp/layer-checkpoints/; done
# !df -h /kaggle/working

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:50:40.898196Z","iopub.execute_input":"2025-10-15T17:50:40.898973Z","iopub.status.idle":"2025-10-15T17:50:41.030614Z","shell.execute_reply.started":"2025-10-15T17:50:40.898944Z","shell.execute_reply":"2025-10-15T17:50:41.029866Z"},"jupyter":{"outputs_hidden":false}}
# !df -h /kaggle/working

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:51:45.589477Z","iopub.execute_input":"2025-10-15T17:51:45.590036Z","iopub.status.idle":"2025-10-15T17:58:12.935195Z","shell.execute_reply.started":"2025-10-15T17:51:45.590011Z","shell.execute_reply":"2025-10-15T17:58:12.934389Z"},"jupyter":{"outputs_hidden":false}}
# !(CUDA_VISIBLE_DEVICES=0 python test.py \
#     --model qwen3 \
#     --dataset v3 \
#     --model-path /kaggle/input/map-checkpoints-x-2-base \
#     --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
#     --checkpoint-root2 /tmp/layer-checkpoints/ \
#     --csv-file test_data1.csv \
#     --micro-batch-size 16 \
#     --num-micro-batches 40 \
#     --out preds_v3_x_part1.pth \
#   & CUDA_VISIBLE_DEVICES=1 python test.py \
#     --model qwen3 \
#     --dataset v3 \
#     --model-path /kaggle/input/map-checkpoints-x-2-base \
#     --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
#     --checkpoint-root2 /tmp/layer-checkpoints/ \
#     --csv-file test_data2.csv \
#     --micro-batch-size 16 \
#     --num-micro-batches 40 \
#     --out preds_v3_x_part2.pth \
#   & wait)

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:59:42.46163Z","iopub.execute_input":"2025-10-15T17:59:42.462606Z","iopub.status.idle":"2025-10-15T17:59:44.341709Z","shell.execute_reply.started":"2025-10-15T17:59:42.462562Z","shell.execute_reply":"2025-10-15T17:59:44.340971Z"},"jupyter":{"outputs_hidden":false}}
# !rm /kaggle/working/layer-checkpoints/*.pth
# !df -h /kaggle/working

# # %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # ## Model 1, Data 3

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T18:00:00.75044Z","iopub.execute_input":"2025-10-15T18:00:00.750719Z"},"jupyter":{"outputs_hidden":false}}
# %%time
# !for i in {0..34}; do cp /kaggle/input/map-checkpoints-y-2-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
# !for i in {35..60}; do cp /kaggle/input/map-checkpoints-y-2-layers/layer_$i.pth /tmp/layer-checkpoints/; done
# !df -h /kaggle/working

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:35:43.732544Z","iopub.execute_input":"2025-10-15T17:35:43.73277Z","execution_failed":"2025-10-15T17:39:03.178Z"},"jupyter":{"outputs_hidden":false}}
# !(CUDA_VISIBLE_DEVICES=0 python test.py \
#     --model glm4 \
#     --dataset v3 \
#     --model-path /kaggle/input/map-checkpoints-y-2-base \
#     --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
#     --checkpoint-root2 /tmp/layer-checkpoints/ \
#     --csv-file test_data1.csv \
#     --micro-batch-size 16 \
#     --num-micro-batches 40 \
#     --out preds_v3_y_part1.pth \
#   & CUDA_VISIBLE_DEVICES=1 python test.py \
#     --model glm4 \
#     --dataset v3 \
#     --model-path /kaggle/input/map-checkpoints-y-2-base \
#     --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
#     --checkpoint-root2 /tmp/layer-checkpoints/ \
#     --csv-file test_data2.csv \
#     --micro-batch-size 16 \
#     --num-micro-batches 40 \
#     --out preds_v3_y_part2.pth \
#   & wait)

# # %% [code] {"jupyter":{"outputs_hidden":false}}
# !rm /kaggle/working/layer-checkpoints/*.pth
# !df -h /kaggle/working

# # %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # ## Model 0, Data 2

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:08:02.624737Z","iopub.execute_input":"2025-10-15T17:08:02.625605Z","iopub.status.idle":"2025-10-15T17:12:06.215437Z","shell.execute_reply.started":"2025-10-15T17:08:02.625573Z","shell.execute_reply":"2025-10-15T17:12:06.214452Z"},"jupyter":{"outputs_hidden":false}}
# %%time
# !for i in {0..36}; do cp /kaggle/input/map-checkpoints-x-1-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
# !for i in {37..63}; do cp /kaggle/input/map-checkpoints-x-1-layers/layer_$i.pth /tmp/layer-checkpoints/; done
# !df -h /kaggle/working

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:12:06.216873Z","iopub.execute_input":"2025-10-15T17:12:06.217104Z","iopub.status.idle":"2025-10-15T17:18:15.053688Z","shell.execute_reply.started":"2025-10-15T17:12:06.217085Z","shell.execute_reply":"2025-10-15T17:18:15.052962Z"},"jupyter":{"outputs_hidden":false}}
# !(CUDA_VISIBLE_DEVICES=0 python test.py \
#     --model qwen3 \
#     --dataset v2 \
#     --model-path /kaggle/input/map-checkpoints-x-1-base \
#     --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
#     --checkpoint-root2 /tmp/layer-checkpoints/ \
#     --csv-file test_data1.csv \
#     --micro-batch-size 16 \
#     --num-micro-batches 40 \
#     --out preds_v2_x_part1.pth \
#   & CUDA_VISIBLE_DEVICES=1 python test.py \
#     --model qwen3 \
#     --dataset v2 \
#     --model-path /kaggle/input/map-checkpoints-x-1-base \
#     --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
#     --checkpoint-root2 /tmp/layer-checkpoints/ \
#     --csv-file test_data2.csv \
#     --micro-batch-size 16 \
#     --num-micro-batches 40 \
#     --out preds_v2_x_part2.pth \
#   & wait)

# # %% [code] {"jupyter":{"outputs_hidden":false}}
# !rm /kaggle/working/layer-checkpoints/*.pth
# !df -h /kaggle/working

# # %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # ## Model 1, Data 2

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:18:22.85806Z","iopub.execute_input":"2025-10-15T17:18:22.858692Z","iopub.status.idle":"2025-10-15T17:23:11.522966Z","shell.execute_reply.started":"2025-10-15T17:18:22.858658Z","shell.execute_reply":"2025-10-15T17:23:11.522266Z"},"jupyter":{"outputs_hidden":false}}
# %%time
# !for i in {0..34}; do cp /kaggle/input/map-checkpoints-y-1-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
# !for i in {35..60}; do cp /kaggle/input/map-checkpoints-y-1-layers/layer_$i.pth /tmp/layer-checkpoints/; done
# !df -h /kaggle/working

# # %% [code] {"execution":{"iopub.status.busy":"2025-10-15T17:23:11.524274Z","iopub.execute_input":"2025-10-15T17:23:11.524486Z","iopub.status.idle":"2025-10-15T17:30:11.878932Z","shell.execute_reply.started":"2025-10-15T17:23:11.524466Z","shell.execute_reply":"2025-10-15T17:30:11.878206Z"},"jupyter":{"outputs_hidden":false}}
# !(CUDA_VISIBLE_DEVICES=0 python test.py \
#     --model glm4 \
#     --dataset v2 \
#     --model-path /kaggle/input/map-checkpoints-y-1-base \
#     --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
#     --checkpoint-root2 /tmp/layer-checkpoints/ \
#     --csv-file test_data1.csv \
#     --micro-batch-size 16 \
#     --num-micro-batches 40 \
#     --out preds_v2_y_part1.pth \
#   & CUDA_VISIBLE_DEVICES=1 python test.py \
#     --model glm4 \
#     --dataset v2 \
#     --model-path /kaggle/input/map-checkpoints-y-1-base \
#     --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
#     --checkpoint-root2 /tmp/layer-checkpoints/ \
#     --csv-file test_data2.csv \
#     --micro-batch-size 16 \
#     --num-micro-batches 40 \
#     --out preds_v2_y_part2.pth \
#   & wait)

# # %% [code] {"jupyter":{"outputs_hidden":false}}
# !rm /kaggle/working/layer-checkpoints/*.pth
# !df -h /kaggle/working


!pip uninstall -y liger-kernel


!pip install --no-index --find-links=/kaggle/input/map-vllm-triton vllm==0.10.0


!pip install --no-index --find-links=/kaggle/input/map-vllm-triton triton==3.2.0


!pip install --no-index --find-links=/kaggle/input/numpy-1-26-4 numpy==1.26.4


# ! uv pip install --system --no-index --find-links='/kaggle/input/latest-jigsaw-whls/whls' 'vllm' 'triton' 'logits-processor-zoo' 'numpy<2'


%%writefile qwen_infer.py

import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle

from scipy.special import softmax
from sklearn.metrics import roc_auc_score

# from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import vllm
import torch
from vllm.lora.request import LoRARequest
import argparse

parser = argparse.ArgumentParser(description='gamma inference')
parser.add_argument('--model_dir', type=str, default = '/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1')
parser.add_argument('--lora_dir', type=str, default = '/kaggle/input/0122-lr-5e-5-max3k-1epo-train-lm20k-d2-684-d2-v5/checkpoint-1983')
parser.add_argument('--max_seq_len', type=int, default = 5000)
parser.add_argument('--input_file', type=str, default = 'doi_datas.pkl')
parser.add_argument('--output_file', type=str, default = 'submission.json')
parser.add_argument('--init_time', type=int, default = 0)
parser.add_argument('--bnb', type=str, default = 'awq')


cfg = parser.parse_args()
print(cfg)

input_file = cfg.input_file
output_file = cfg.output_file

test_df = pd.read_csv(input_file)
max_seq_len = cfg.max_seq_len
model_path = cfg.model_dir
lora_path = cfg.lora_dir

train_data = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')


tmp = train_data.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

all_answer = []

for key, item in tmp.groupby('QuestionId'):
    labels="ABCD"
    
    all_answer.append([key, item['MC_Answer'].tolist()])
print(all_answer)    
all_answer_df = pd.DataFrame(all_answer, columns=['QuestionId', 'All_Choices'])


idx = train_data.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train_data.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

test_df = test_df.merge(all_answer_df, on='QuestionId', how='left')

def make_choice(row):
    all_choices = row.All_Choices
    MC_Answer = row.MC_Answer
    idx = all_choices.index(MC_Answer)
    
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(all_choices)])
    return pd.Series([choice_str, labels[idx]], index=['choices', 'correct_answer'])
test_df[['choices', 'correct_answer']] = test_df.apply(make_choice, axis=1)


if __name__ == '__main__':
    test_df = test_df.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    test_df.is_correct = test_df.is_correct.fillna(0)
    
    if cfg.bnb == 'awq':
        print('awq')
        llm = vllm.LLM(
            model_path,
            quantization='awq',
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=4096,
            disable_log_stats=True,
            enable_prefix_caching=True,
            enable_lora=True,
            max_lora_rank = 128,
            max_logprobs=40
        )
    else:
        print('无量化')
        llm = vllm.LLM(
            model_path,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            dtype="half",
            enforce_eager=True,
            max_model_len=4096,
            disable_log_stats=True,
            enable_prefix_caching=True,
            enable_lora=True,
            max_lora_rank = 128,
            max_logprobs=40
        )
    tokenizer = llm.get_tokenizer()
    
    
    def get_tokenizer(llm):
        tokenizer = llm.get_tokenizer()
        tokenizer.padding_side = "left"
        return tokenizer
    
    tokenizer = get_tokenizer(llm)
    
    if tokenizer.pad_token != tokenizer.eos_token:
        llm.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
    
    all_choices = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
        'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'
    ]
    
    
            
    mis2reason = {
        "SwapDividend": "Incorrectly swapping the positions of dividend and divisor in division operations.",
        "Tacking": "Arbitrarily adding zeros or decimal points to the end of numbers, believing the value remains unchanged or changes incorrectly.",
        "Additive": "Mistakenly using addition to solve problems that require other operations (multiplication, subtraction, etc.)",
        "Wrong_term": "Incorrectly identifying or handling terms in algebraic expressions.",
        "Wrong_Fraction": "Completely misunderstanding fraction concepts or representation methods.",
        "Incomplete": "Providing incomplete solutions missing crucial steps or explanations.",
        "Unknowable": "Mistakenly believing a problem is unsolvable or lacks information when it is actually solvable.",
        "Not_variable": "Treating variables as specific numerical values, or vice versa.",
        "Firstterm": "Overemphasizing the first term in a sequence while ignoring the importance of other terms.",
        "Irrelevant": "Using information or criteria unrelated to the problem for reasoning.",
        "Inverse_operation": "Incorrectly applying inverse operations or confusing relationships between operations.",
        "Multiplying_by_4": "Specific error: Always multiplying by 4 without considering the specific context.",
        "Base_rate": "Ignoring base probabilities or benchmark values, focusing only on specific cases.",
        "Definition": "Misunderstanding mathematical concept definitions or terminology meanings.",
        "WNB": """Mistakenly believing "the whole is not the sum of its parts" or similar part-whole relationships""",
        "Whole_numbers_larger": "Believing decimals with larger whole number parts are always larger, ignoring decimal parts",
        "Incorrect_equivalent_fraction_addition": "Incorrectly performing fraction addition operations",
        "Inversion": "Mistakenly reversing the order of numbers, fractions, or operations.",
        "Mult": "Mistakenly using multiplication to solve problems that require other operations.",
        "Adding_terms": "Incorrectly adding terms directly in algebraic expressions.",
        "FlipChange": "Incorrectly handling numerator-denominator conversions in fraction operations.",
        "Division": "Mistakenly using division to solve problems that require other operations.",
        "Duplication": "Incorrectly repeating numbers or operations.",
        "Interior": "Incorrectly handling interior angles or internal elements in geometric figures.",
        "Certainty": "Providing definite answers for uncertain problems, or vice versa.",
        "Shorter_is_bigger": "Believing numbers with fewer digits are larger.",
        "Wrong_fraction": "Misunderstanding fraction concepts, including numerator-denominator relationships.",
        "Adding_across": "Incorrectly adding across place values (e.g., adding tens to ones directly).",
        "Wrong_Operation": "Choosing completely wrong mathematical operations.",
        "Denominator-only_change": "Changing only the denominator while ignoring corresponding changes in the numerator.",
        "Scale": "Misunderstanding scale factors or proportional relationships.",
        "Longer_is_bigger": "Believing numbers with more digits are larger.",
        "Positive": "Mistakenly believing all mathematical results should be positive numbers.",
        "Ignores_zeroes": "Ignoring the place value or importance of zeros in numbers.",
        "Subtraction": "Mistakenly using subtraction to solve problems that require other operations.",
    }
    
    all_mis_key = list(mis2reason.keys()) + ['Student Explanation is Correct', 'Neither']
    all_mis_value = list(mis2reason.values()) + ['Student Explanation is Correct', "This explanation is confusing and it doesn't fall into any of the above categories"]
    
    
    
    
    passages_str = ""
    for choice, passage in zip(all_choices, all_mis_value):
        passages_str += f'{choice}: {passage}'
        passages_str += "\n"
    passages_str = passages_str.strip()
    print(passages_str)
    
    choice2target = {}
    for choice, passage in zip(all_choices, all_mis_key):
        if passage == 'Student Explanation is Correct':
            passage = "Correct"
        choice2target[choice] = passage
    
    
    
    
    templete_part = f"""<|im_start|>user\nYou are now tasked with analyzing math problems and classifying student responses. Given a math problem, the student's chosen answer, whether it's correct, and the student's explanation, you need to determine the appropriate Misconception classification.
(1) Assesses whether the explanation contains a misconception. (Correct, Misconception, or Neither in Category; e.g., True_Correct)
(2) Identifies the specific misconception present, if any.

Below are the available Misconception classifications you can choose from.
Always provide your response using only the specified format.

{passages_str}

Please analyze the given input and provide your classification.

"""
    
    templete_part1_input_ids = tokenizer(text=templete_part, add_special_tokens=True, padding=False)['input_ids']
    
    
    templete_part2 = "<|im_end|>\n<|im_start|>assistant\n"
    templete_part2_input_ids = tokenizer(text=templete_part2, add_special_tokens=True, padding=False)['input_ids']
    
    
    
    all_inputs = []
    
    for _, sample in test_df.iterrows():
        x = "The selected answer is correct."
        if not sample['is_correct']:
            x = "The selected answer is wrong."
    
        query = f"""### Question:
{sample['QuestionText']}

### Choices:
{sample['choices']}

### Selected Answer:
{sample['correct_answer']}. {sample['MC_Answer']}

### {x}

### Student Explanation:
{sample['StudentExplanation']}"""
            
        input_text = query
        input_text_ids = tokenizer(
                text=input_text, 
                add_special_tokens=True, 
                truncation=True, 
                max_length=5200, 
                padding=False
            )['input_ids']
    
        input_ids = templete_part1_input_ids + input_text_ids + templete_part2_input_ids
    
        input_text = tokenizer.decode(input_ids)
        
        all_inputs.append(input_text)
    
    
    
    print(len(all_inputs))
    if len(all_inputs) != 0:
        print(all_inputs[0])
    
    # mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=all_choices)
    allowed_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in all_choices]
    keep = allowed_token_ids
    print(keep)
    #guided_decoding_params_choice = GuidedDecodingParams(choice=cfg.options)
    def digit_logits_processor(input_ids, logits):
        logits[..., keep] += 100
        return logits
    logits_processors = [digit_logits_processor]
    
    outputs = llm.generate(
        all_inputs,
        vllm.SamplingParams(
            seed=777,
            temperature=0.0,
            skip_special_tokens=True,
            max_tokens=1,
            # logits_processors=[mclp],
            logits_processors= logits_processors,
            logprobs=37,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("default", 1, lora_path)
    )
    
    logprobs = [
        [[lp.decoded_token, lp.logprob] for lp in list(out.outputs[0].logprobs[0].values())]
        for out in outputs
    ]
    
    import pickle
    
    with open(output_file, 'wb') as f:
        pickle.dump(logprobs, f)
    
    print('finish')


# %%time
# !python qwen_infer.py \
#     --model_dir /kaggle/input/qwen2.5/transformers/14b-instruct-awq/1 \
#     --lora_dir /kaggle/input/map-data-qwen25-7b/map_0918_all_qwen25_14b_v2_step4588_lb947/map_0918_all_qwen25_14b_v2_step4588_lb947 \
#     --input_file /kaggle/input/map-charting-student-math-misunderstandings/test.csv \
#     --output_file submission_qwen25_14b.pkl \
#     --bnb 'awq'


# %%time
# !python qwen_infer.py \
#     --model_dir /kaggle/input/qwen-3/transformers/14b-awq/1 \
#     --lora_dir /kaggle/input/map-data-qwen25-7b/map_0918_fold1_qwen3_14b_v1_step3670_9434/map_0918_fold1_qwen3_14b_v1_step3670_9434 \
#     --input_file /kaggle/input/map-charting-student-math-misunderstandings/test.csv \
#     --output_file submission_qwen3_14b.pkl \
#     --bnb 'awq'


# %%time
# !python qwen_infer.py \
#     --model_dir /kaggle/input/qwen2.5/transformers/32b-instruct-awq/1 \
#     --lora_dir /kaggle/input/map-data-qwen25-7b/map_0919_fold1_qwen25_32b_v1_step3670_9463/map_0919_fold1_qwen25_32b_v1_step3670_9463 \
#     --input_file /kaggle/input/map-charting-student-math-misunderstandings/test.csv \
#     --output_file submission_qwen25_32b.pkl \
#     --bnb 'awq'


# %%time
# !python qwen_infer.py \
#     --model_dir /kaggle/input/qwen2.5/transformers/32b-instruct-awq/1 \
#     --lora_dir /kaggle/input/map-data-qwen25-7b/map_0919_all_qwen25_32b_v2_step4588_lb948/map_0919_all_qwen25_32b_v2_step4588_lb948 \
#     --input_file /kaggle/input/map-charting-student-math-misunderstandings/test.csv \
#     --output_file submission_qwen25_32b_full.pkl \
#     --bnb 'awq'


%%writefile infer.py
import sys
sys.path.append('/kaggle/input/map2025-configs')
import os, math, numpy as np
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import pandas as pd
from datasets import Dataset
from vllm import LLM
from vllm import SamplingParams
from transformers import set_seed, AutoTokenizer
from scipy.special import softmax
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams
from data_utils import format_multi_choices_input
set_seed(42)
import importlib
from types import SimpleNamespace
import glob
import torch
import pandas as pd
import argparse

def make_parser():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--sft_model_weight", type=str, default='',
                        help="sft_model_weight")
    parser.add_argument("--model_name", type=str, default='',
                        help="model_name")
    parser.add_argument("--quantization", type=str, default='awq',
                        help="response_template")
    args = parser.parse_args()
    return args


cols_name = ['False_Correct:NA',
                 'False_Misconception:Adding_across',
                 'False_Misconception:Adding_terms',
                 'False_Misconception:Additive',
                 'False_Misconception:Base_rate',
                 'False_Misconception:Certainty',
                 'False_Misconception:Definition',
                 'False_Misconception:Denominator-only_change',
                 'False_Misconception:Division',
                 'False_Misconception:Duplication',
                 'False_Misconception:Firstterm',
                 'False_Misconception:FlipChange',
                 'False_Misconception:Ignores_zeroes',
                 'False_Misconception:Incomplete',
                 'False_Misconception:Incorrect_equivalent_fraction_addition',
                 'False_Misconception:Interior',
                 'False_Misconception:Inverse_operation',
                 'False_Misconception:Inversion',
                 'False_Misconception:Irrelevant',
                 'False_Misconception:Longer_is_bigger',
                 'False_Misconception:Mult',
                 'False_Misconception:Multiplying_by_4',
                 'False_Misconception:Not_variable',
                 'False_Misconception:Positive',
                 'False_Misconception:Scale',
                 'False_Misconception:Shorter_is_bigger',
                 'False_Misconception:Subtraction',
                 'False_Misconception:SwapDividend',
                 'False_Misconception:Tacking',
                 'False_Misconception:Unknowable',
                 'False_Misconception:WNB',
                 'False_Misconception:Whole_numbers_larger',
                 'False_Misconception:Wrong_Fraction',
                 'False_Misconception:Wrong_Operation',
                 'False_Misconception:Wrong_fraction',
                 'False_Misconception:Wrong_term',
                 'False_Neither:NA',
                 'True_Correct:NA',
                 'True_Misconception:Adding_across',
                 'True_Misconception:Additive',
                 'True_Misconception:Base_rate',
                 'True_Misconception:Definition',
                 'True_Misconception:Denominator-only_change',
                 'True_Misconception:Division',
                 'True_Misconception:Duplication',
                 'True_Misconception:Firstterm',
                 'True_Misconception:FlipChange',
                 'True_Misconception:Incomplete',
                 'True_Misconception:Incorrect_equivalent_fraction_addition',
                 'True_Misconception:Inversion',
                 'True_Misconception:Irrelevant',
                 'True_Misconception:Longer_is_bigger',
                 'True_Misconception:Mult',
                 'True_Misconception:Multiplying_by_4',
                 'True_Misconception:Not_variable',
                 'True_Misconception:Positive',
                 'True_Misconception:Shorter_is_bigger',
                 'True_Misconception:Subtraction',
                 'True_Misconception:SwapDividend',
                 'True_Misconception:Tacking',
                 'True_Misconception:WNB',
                 'True_Misconception:Whole_numbers_larger',
                 'True_Misconception:Wrong_fraction',
                 'True_Misconception:Wrong_term',
                 'True_Neither:NA']

def preprocess_multi_choices_data(df, tokenizer, cfg):
    train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    train['label'] = [cfg.options_2_ids[cfg.target_2_options[i]]  for i in train['target']]

    idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
    correct = train.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])
    correct = correct[['QuestionId', 'MC_Answer']]
    correct['is_correct'] = 1

    train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)

    correct['Correct_Answer'] = correct['MC_Answer']
    correct = correct[['QuestionId', 'Correct_Answer']]
    df = df.merge(correct, on=['QuestionId'], how='left')

    df_options = train.groupby('QuestionId').agg({
        'MC_Answer': lambda x: list(dict.fromkeys(x)),  # 保持顺序，去重并转 list
    }).reset_index()
    df_options['Options'] = df_options['MC_Answer']
    df_options = df_options[['QuestionId', 'Options']]
    
    df = df.merge(df_options, on=['QuestionId'], how='left')
    df['text'] = df.apply(format_multi_choices_input,tokenizer=tokenizer, cfg=cfg ,axis=1)
    return df
    
def main():
    args = make_parser()
    sft_model_weight = args.sft_model_weight
    tokenizer = AutoTokenizer.from_pretrained(sft_model_weight)
    cfg = importlib.import_module(f'{args.model_name}').basic_cfg
    cfg.target_2_options = {k: v for k, v in zip(cfg.target, cfg.options)}
    cfg.options_2_ids = {v: idx for idx, v in enumerate(cfg.options)}
    cfg.options_2_target = {v:k for k, v in cfg.target_2_options.items()}
    cfg.ids_2_options = {v:k for k, v in cfg.options_2_ids.items()}
    cfg.token_id_2_label_id = {}
    #print(cfg.response_template)
    for k, v in cfg.options_2_ids.items():
        #print(tokenizer(k)['input_ids'][0])
        cfg.token_id_2_label_id[tokenizer(k, add_special_tokens=False)['input_ids'][0]] = v
    cfg.target_tokens = sorted([i for i in cfg.token_id_2_label_id.keys()])
    cfg.cols_name = cols_name
    test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
    test['target'] = 'False_Correct:NA'
    test = preprocess_multi_choices_data(test, tokenizer, cfg)
    test = Dataset.from_pandas(test)
    prompts = test['text']
    prompts = [i.split(cfg.response_template)[0] + cfg.response_template for i in test['text']]
    print(prompts[0])
    llm = LLM(
        sft_model_weight,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2000,
        disable_log_stats=True,
        max_num_seqs=48,
        enable_chunked_prefill=True,     # 关键：开 Chunked Prefill
        max_num_batched_tokens=1024,     # 起步值，按吞吐/显存微调
        #enable_lora=True,
        #max_lora_rank=32, 
        quantization="auto-round",  # 使用 AutoRound 格式
        #async_scheduling=True,
        disable_cascade_attn=True,
        enable_prefix_caching=True,
        #speculative_decoding_mode="none",   # 避免与约束/单步采样的边界交互
        #ogits_processors=[proc],
       
    )
    keep = cfg.target_tokens
    #guided_decoding_params_choice = GuidedDecodingParams(choice=cfg.options)
    def digit_logits_processor(input_ids, logits):
        logits[..., keep] += 100
        return logits
    logits_processors = [digit_logits_processor]
    sampling_params = SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=42,  # Seed for reprodicibility
        skip_special_tokens=True,  # Whether to skip special tokens in the output.
        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
        logits_processors=logits_processors,
        logprobs=20,
    )
    
    
    responses = llm.generate(
        prompts,
        sampling_params,
        #lora_request=LoRARequest("default", 1, LORA_PATH)
    )
    
    outputs = []
    for response in responses:
        response_logprobs = response.outputs[0].logprobs[0]
        sub_df = pd.DataFrame([[-100.0] * len(cfg.cols_name)], columns=cfg.cols_name)
        choice = []
        for key in response_logprobs.keys():
            print(key)
            print(response_logprobs[key])
            decoded_token = response_logprobs[key].decoded_token.strip()
            #if decoded_token in cfg.options_2_target:
            col_ = cfg.options_2_target[decoded_token]
            sub_df.loc[0, col_] = response_logprobs[key].logprob#np.nan_to_num(response_logprobs[key].logprob, nan=-900, posinf=900, neginf=-900)
        outputs.append(sub_df)

    outputs = pd.concat(outputs).reset_index(drop=True)
    # 按行应用 softmax
    probs = outputs.apply(lambda row: softmax(row.values), axis=1, result_type="expand")
    probs.columns = outputs.columns
    save_name = args.model_name
    np.save(f"{save_name}.npy", probs)

if __name__ == "__main__":
    main()


# %%time
# !python infer.py\
#     --sft_model_weight '/kaggle/input/map2025-qwen3-14b-new-ext-auto-around/transformers/v1/1' \
#     --model_name 'qwen3_14b' \
#     --quantization 'auto-round'


# %%time

# !python infer.py\
#     --sft_model_weight '/kaggle/input/map2025-phi4-14b-new-ext-auto-around/transformers/v1/1' \
#     --model_name 'phi_4_reasoning_14b' \
#     --quantization 'auto-round'


%%time
!python infer.py\
    --sft_model_weight '/kaggle/input/map2025-mistral-12b-new-ext-auto-around/transformers/v1/1' \
    --model_name 'mistral_12b' \
    --quantization 'auto-round'


# %%time
# !python infer.py\
#     --sft_model_weight '/kaggle/input/map2025-qwen2-5-14b-new-ext-auto-around/transformers/v1/1' \
#     --model_name 'qwen2_5_14b' \
#     --quantization 'auto-round'


!ls


import pandas as pd
import numpy as np
import pickle
import torch
import glob
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
import glob
import torch
import numpy as np
import pandas as pd
from map_modules.data.dataset_v1 import MAPDataset  
from functools import reduce

# ---------------------------------------------------------
# 1. Configuration & Column Definitions
# ---------------------------------------------------------
# Define the specific column order used in Notebook 2 to ensure alignment
logit_columns = ['False_Correct:NA', 'False_Misconception:Adding_across', 'False_Misconception:Adding_terms', 'False_Misconception:Additive', 'False_Misconception:Base_rate', 'False_Misconception:Certainty', 'False_Misconception:Definition', 'False_Misconception:Denominator-only_change', 'False_Misconception:Division', 'False_Misconception:Duplication', 'False_Misconception:Firstterm', 'False_Misconception:FlipChange', 'False_Misconception:Ignores_zeroes', 'False_Misconception:Incomplete', 'False_Misconception:Incorrect_equivalent_fraction_addition', 'False_Misconception:Interior', 'False_Misconception:Inverse_operation', 'False_Misconception:Inversion', 'False_Misconception:Irrelevant', 'False_Misconception:Longer_is_bigger', 'False_Misconception:Mult', 'False_Misconception:Multiplying_by_4', 'False_Misconception:Not_variable', 'False_Misconception:Positive', 'False_Misconception:Scale', 'False_Misconception:Shorter_is_bigger', 'False_Misconception:Subtraction', 'False_Misconception:SwapDividend', 'False_Misconception:Tacking', 'False_Misconception:Unknowable', 'False_Misconception:WNB', 'False_Misconception:Whole_numbers_larger', 'False_Misconception:Wrong_Fraction', 'False_Misconception:Wrong_Operation', 'False_Misconception:Wrong_fraction', 'False_Misconception:Wrong_term', 'False_Neither:NA', 'True_Correct:NA', 'True_Misconception:Adding_across', 'True_Misconception:Adding_terms', 'True_Misconception:Additive', 'True_Misconception:Base_rate', 'True_Misconception:Certainty', 'True_Misconception:Definition', 'True_Misconception:Denominator-only_change', 'True_Misconception:Division', 'True_Misconception:Duplication', 'True_Misconception:Firstterm', 'True_Misconception:FlipChange', 'True_Misconception:Ignores_zeroes', 'True_Misconception:Incomplete', 'True_Misconception:Incorrect_equivalent_fraction_addition', 'True_Misconception:Interior', 'True_Misconception:Inverse_operation', 'True_Misconception:Inversion', 'True_Misconception:Irrelevant', 'True_Misconception:Longer_is_bigger', 'True_Misconception:Mult', 'True_Misconception:Multiplying_by_4', 'True_Misconception:Not_variable', 'True_Misconception:Positive', 'True_Misconception:Scale', 'True_Misconception:Shorter_is_bigger', 'True_Misconception:Subtraction', 'True_Misconception:SwapDividend', 'True_Misconception:Tacking', 'True_Misconception:Unknowable', 'True_Misconception:WNB', 'True_Misconception:Whole_numbers_larger', 'True_Misconception:Wrong_Fraction', 'True_Misconception:Wrong_Operation', 'True_Misconception:Wrong_fraction', 'True_Misconception:Wrong_term', 'True_Neither:NA']

# Map needed for processing Notebook 2's pickles
all_choices = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
    'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'
]
all_mis_key = [
    'SwapDividend', 'Tacking', 'Additive', 'Wrong_term', 'Wrong_Fraction', 'Incomplete', 
    'Unknowable', 'Not_variable', 'Firstterm', 'Irrelevant', 'Inverse_operation', 
    'Multiplying_by_4', 'Base_rate', 'Definition', 'WNB', 'Whole_numbers_larger', 
    'Incorrect_equivalent_fraction_addition', 'Inversion', 'Mult', 'Adding_terms', 
    'FlipChange', 'Division', 'Duplication', 'Interior', 'Certainty', 'Shorter_is_bigger', 
    'Wrong_fraction', 'Adding_across', 'Wrong_Operation', 'Denominator-only_change', 
    'Scale', 'Longer_is_bigger', 'Positive', 'Ignores_zeroes', 'Subtraction', 
    'Correct', 'Neither'
]
choice2target = {choice: passage for choice, passage in zip(all_choices, all_mis_key)}

# ---------------------------------------------------------
# 2. Helpers for Notebook 2 (The "BPH" and "HZM" models)
# ---------------------------------------------------------
def get_bph_pred(df, logprobs_path):
    """
    Processes the pickle files from NB1 which contain log probabilities.
    Applies Softmax and maps tokens to columns.
    """
    with open(logprobs_path, 'rb') as f:
        logprobs = pickle.load(f)
        
    results = []
    for i, pred in enumerate(logprobs):
        sample = df.iloc[i]
        # Determine prefix based on correctness
        cate1 = "True" if sample.get('is_correct', 0) == 1 else "False"
        
        idxs = [v[0].strip() for v in pred]
        # Apply Softmax to convert logprobs to probabilities
        vals = [v[1] for v in pred]
        logits = softmax(vals)
        
        res = {}
        for idx, logit in zip(idxs, logits):
            if idx in choice2target:
                p1 = choice2target[idx]
                if p1 in ["Correct", "Neither"]:
                    r = f"{cate1}_{p1}:NA"
                else:
                    r = f"{cate1}_Misconception:{p1}"
                res[r] = logit
        
        # Fill missing columns with 0
        for col in logit_columns:
            if col not in res:
                res[col] = 0.0
        results.append(res)
        
    return pd.DataFrame(results)[logit_columns].values



def get_hzm_pred(hzm_clsses, npy_path):
    """
    Processes the .npy files from NB1. 
    NB1 saves these AFTER applying softmax, so they are already probabilities.
    However, we need to ensure the column mapping is correct.
    """
    # Load the probabilities
    preds = np.load(npy_path)
    
    results = []
    for pred in preds:
        res = {}
        
        for idx, p in enumerate(pred):
            r = hzm_clsses[idx]
            res[r] = p
    
        for col in logit_columns:
            if col not in res:
                res[col] = 0
        results.append(res)  
    return pd.DataFrame(results)[logit_columns].values

def load_notebook2_probs(test_df):
    print("Loading Notebook 2 (BPH + HZM) Predictions...")
    
    # 1. BPH Models (Pickles)
    bph_files = [
        'submission_qwen25_14b.pkl',
        'submission_qwen3_14b.pkl',
        'submission_qwen25_32b.pkl',
        'submission_qwen25_32b_full.pkl'
    ]
    
    bph_probs = []
    for f in bph_files:
        print(f"  - Processing {f}")
        try:
            p = get_bph_pred(test_df, f)            
            bph_probs.append(p)
        except FileNotFoundError:
            print(f"    Warning: {f} not found. Skipping.")

    # hzm classes
    le = LabelEncoder()
    train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category+":"+train.Misconception
    train['label'] = le.fit_transform(train['target'])
    n_classes = len(le.classes_)
    hzm_clsses = le.classes_

    # 2. HZM Models (Numpy)
    hzm_files = [
        'qwen3_14b.npy',
        'qwen2_5_14b.npy',
        'phi_4_reasoning_14b.npy',
        'mistral_12b.npy'
    ]
    
    hzm_probs = []
    for f in hzm_files:
        print(f"  - Processing {f}")
        try:
            p = get_hzm_pred(hzm_clsses, f)
            hzm_probs.append(p)
        except FileNotFoundError:
            print(f"    Warning: {f} not found. Skipping.")

    # Combine all found models from NB1
    # all_nb2 = reduce(lambda x, y: x + y, bph_probs) + reduce(lambda x, y: x + y, hzm_probs)
    all_nb2 = reduce(lambda x, y: x + y, hzm_probs)
    # Average them
    avg_nb2 = all_nb2 / float(len(bph_probs) + len(hzm_probs))
    return avg_nb2

# ---------------------------------------------------------
# 3. Helpers for Notebook 1 (The Torch/PTH models)
# ---------------------------------------------------------
def load_notebook1_probs():
    print("Loading Notebook 1 (Qwen3/GLM4 Torch) Predictions...")
    
    # NB2 splits data into part1 and part2. We must concatenate them.
    
      
    
    def average_results(results, weights=None):
        ret = []
        for idx in range(len(results[0])):
            ret.append(
                np.average([result[idx] for result in results], axis=0, weights=weights)
            )
        return ret
    
    ds1 = MAPDataset(csv_file="test_data1.csv", tokenizer=None)
    ds2 = MAPDataset(csv_file="test_data2.csv", tokenizer=None)
    
    fns_part1 = glob.glob("./*_part1.pth")
    fns_part2 = glob.glob("./*_part2.pth")
    print(fns_part1, fns_part2)
    preds_part1 = [torch.load(fn, weights_only=True) for fn in fns_part1]
    preds_part2 = [torch.load(fn, weights_only=True) for fn in fns_part2]
    
    preds_part1 = average_results(preds_part1)
    preds_part2 = average_results(preds_part2)
    
    print(ds1.evaluate(preds_part1), ds2.evaluate(preds_part2))
    df = pd.concat([ds1.df.copy(), ds2.df.copy()]).reset_index(drop=True)
    preds = preds_part1 + preds_part2
    df["preds"] = preds

    print(preds)
        
    # Average the 4 variations
    logit_col_to_idx = {col: i for i, col in enumerate(logit_columns)}

    def map_row_preds_to_global(row):
        # Initialize a zero-filled array of the fixed global length
        global_logits = np.zeros(len(logit_columns), dtype=np.float32)
        
        # Get the specific candidates and scores for this row
        candidates = row["label_candidates"]
        scores = row["preds"]
        
        # Identify which indices in the global array correspond to this row's candidates
        # We map every candidate string to its global integer index
        target_indices = [logit_col_to_idx[c] for c in candidates]
        
        # Assign the scores to those specific positions
        # NumPy allows using a list of indices to set multiple values at once
        global_logits[target_indices] = scores
        
        return global_logits

    # NB2 saves tensors that are ALREADY softmaxed in the test.py loop.
    final_logit_matrix = np.stack(df.apply(map_row_preds_to_global, axis=1).values)
    print(f"Final Matrix Shape: {final_logit_matrix.shape}")    
    
    # IMPORTANT: NB2 outputs likely follow the dataset label encoding.
    # We must assume the competition standard encoding matches logit_columns.
    # If NB2 uses a different sorted order, this will be wrong. 
    # Based on standard MAP utils, it is usually sorted alphabetically.
    # logit_columns is sorted alphabetically.
    
    return final_logit_matrix

# ---------------------------------------------------------
# 4. Main Ensemble Function
# ---------------------------------------------------------
def ensemble_predictions(weight_nb1=0.5, weight_nb2=0.5):
    # Load Test Data (needed for processing NB1 pickles)
    test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
    
    # 1. Preprocess correctness for NB1 logic
    # (Replicating NB1 logic to determine if we should look for "True" or "False" prefix)
    train_data = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

    # Prolly dont need this but just in case
    tmp = train_data.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
    tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
    tmp = tmp.drop('count',axis=1)
    tmp = tmp.sort_values(['QuestionId','rank'])
    
    idx = train_data.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
    correct_ref = train_data.loc[idx].copy()
    correct_ref['c'] = correct_ref.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
    correct_ref = correct_ref.sort_values('c', ascending=False).drop_duplicates(['QuestionId'])
    correct_ref = correct_ref[['QuestionId', 'MC_Answer']]
    correct_ref['is_correct'] = 1
    
    test_df = test_df.merge(correct_ref, on=['QuestionId', 'MC_Answer'], how='left')
    test_df.is_correct = test_df.is_correct.fillna(0)
    
    # 2. Get Probabilities
    # probs1 = load_notebook1_probs()
    probs2 = load_notebook2_probs(test_df)
    
    # print(f"Shape NB1: {probs1.shape}")
    # print(type(probs1))
    print(f"Shape NB2: {probs2.shape}")
    print(type(probs2))
    
    # if probs1.shape != probs2.shape:
    #     raise ValueError("Shapes of outputs do not match! Check if rows are lost or columns differ.")

    # 3. Weighted Square Ensemble
    # p1_squared = preds_nb1 * preds_nb1
    # p2_squared = preds_nb2 * preds_nb2
    # final_probs = (weight_nb1 * p1_squared) + (weight_nb2 * p2_squared)
    
    # 4. Create Submission
    top3_indices = np.argsort(-probs2, axis=1)[:, :3]
    
    final_results = []
    for idxs in top3_indices:
        # Map indices back to column names
        res_labels = [logit_columns[i] for i in idxs]
        final_results.append(" ".join(res_labels))
        
    sub = pd.DataFrame({
        "row_id": test_df.row_id.values,
        "Category:Misconception": final_results
    })
    
    sub.to_csv("submission.csv", index=False)
    print("Ensemble completed. Saved to 'submission_ensemble.csv'.")
    return sub.head()

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # You can adjust weights here. 
    # If NB1 (4 models - suffix classification) is stronger, give it 0.6 or 0.7.
    alpha=0.5
    print(ensemble_predictions(weight_nb1=alpha, weight_nb2=1-alpha))

