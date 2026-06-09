!pip install --no-index --find-links=/kaggle/input/map-packages/ liger-kernel


!cp -r /kaggle/input/map-modules map_modules


# Much faster and more stable than /kaggle/input
!mkdir /tmp/layer-checkpoints
!mkdir /kaggle/working/layer-checkpoints


%%writefile prepare_test.py

import pandas as pd

DEBUG = False

if DEBUG:
    df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv").iloc[:16000]
else:
    df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
    df["Category"] = "True_Correct"
    df["Misconception"] = "NA"

N = len(df)
df1 = df.iloc[:N//2]
df2 = df.iloc[N//2:]
print(df1.shape, df2.shape)
df1.to_csv("test_data1.csv", index=False)
df2.to_csv("test_data2.csv", index=False)


!python prepare_test.py


%%writefile test.py

import argparse
import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from map_modules.data.dataset_v1 import MAPDataset as MAPDatasetV1
from map_modules.data.dataset_v2 import MAPDataset as MAPDatasetV2
from map_modules.data.dataset_v3 import MAPDataset as MAPDatasetV3
from map_modules.models.modeling_qwen3_w8a8 import Qwen3ForSequenceClassification
from map_modules.models.modeling_glm4_w8a8 import Glm4ForSequenceClassification
from map_modules.utils import to_gpu
from tqdm import tqdm


DATASETS = {
    "v1": MAPDatasetV1,
    "v2": MAPDatasetV2,
    "v3": MAPDatasetV3,
}
MODELS = {
    "qwen3": (Qwen3ForSequenceClassification, (37, 27)),
    "glm4": (Glm4ForSequenceClassification, (35, 26)),
}


def get_block_mask(input_ids, suffix_ids, doc_ids, position_ids):
    def custom_mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_suffix = (suffix_ids[q_idx] == suffix_ids[kv_idx]) | (
            suffix_ids[kv_idx] == -1
        )
        same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
        return causal & same_suffix & same_doc

    return create_block_mask(
        custom_mask,
        B=None,
        H=None,
        Q_LEN=input_ids.size(0),
        KV_LEN=input_ids.size(0),
        BLOCK_SIZE=(128, 128),
    )


class Inferencer:
    def __init__(self, model, num_layers, checkpoint_root):
        self.num_layers = sum(num_layers)
        self.layers_weights = [
            torch.load(
                f"{checkpoint_root[0]}/layer_{i}.pth",
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            for i in range(num_layers[0])
        ]
        offset = num_layers[0]
        self.layers_weights.extend(
            [
                torch.load(
                    f"{checkpoint_root[1]}/layer_{i + offset}.pth",
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                )
                for i in range(num_layers[1])
            ]
        )

        self.model = model.cuda().eval()

        self.h2d_stream = torch.cuda.Stream()
        self.curr_layer, self.next_layer = (
            self.model.model.layers[0],
            self.model.model.layers[1],
        )

    @torch.no_grad()
    def forward(self, batches):
        batches = to_gpu(batches)
        block_masks = []
        hidden_statess = []
        position_embeddingss = []
        last_tokenss = []
        for micro_batch in batches:
            input_ids = micro_batch["input_ids"].squeeze(0)
            suffix_ids = micro_batch["suffix_ids"]
            doc_ids = micro_batch["doc_ids"]
            position_ids = micro_batch["position_ids"]
            last_tokens = micro_batch["last_tokens"]

            block_mask = get_block_mask(input_ids, suffix_ids, doc_ids, position_ids)
            block_masks.append(block_mask)
            hidden_states = self.model.model.embed_tokens(input_ids)
            position_embeddings = self.model.model.rotary_emb(
                hidden_states, position_ids.unsqueeze(0)
            )
            hidden_statess.append(hidden_states)
            position_embeddingss.append(position_embeddings)
            last_tokenss.append(last_tokens)

        curr_layer, next_layer = self.curr_layer, self.next_layer
        curr_states, next_states = curr_layer.state_dict(), next_layer.state_dict()
        for layer_idx in range(self.num_layers):
            for m_idx, (hidden_states, block_mask, position_embeddings) in enumerate(
                zip(hidden_statess, block_masks, position_embeddingss)
            ):
                hidden_states = curr_layer(
                    hidden_states, block_mask, position_embeddings
                )
                hidden_statess[m_idx].copy_(hidden_states)

            with torch.cuda.stream(self.h2d_stream):
                next_layer_idx = (layer_idx + 1) % self.num_layers
                next_layer_wegihts = self.layers_weights[next_layer_idx]
                for k, v in next_layer_wegihts.items():
                    next_states[k].copy_(v, non_blocking=True)
            torch.cuda.synchronize()
            # alternate
            curr_layer, next_layer = next_layer, curr_layer
            curr_states, next_states = next_states, curr_states
        self.curr_layer, self.next_layer = curr_layer, next_layer
        hidden_statess = [
            self.model.model.norm(hidden_states) for hidden_states in hidden_statess
        ]

        hidden_statess = [
            hidden_states[last_tokens]
            for hidden_states, last_tokens in zip(hidden_statess, last_tokenss)
        ]
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logitss = [
                self.model.score(hidden_states) for hidden_states in hidden_statess
            ]
        return logitss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint-root1",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint-root2",
        type=str,
        required=True,
    )
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--num-micro-batches", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    model_class, num_layers = MODELS[args.model]
    model_path = args.model_path
    checkpoint_root = (args.checkpoint_root1, args.checkpoint_root2)
    dataset = DATASETS[args.dataset](
        csv_file=args.csv_file,
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        query=args.query,
    )
    dl = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    inferencer = Inferencer(
        model, num_layers=num_layers, checkpoint_root=checkpoint_root
    )
    micro_batches = [batch for batch in dl]
    batches = []
    for start in range(0, len(micro_batches), args.num_micro_batches):
        batches.append(micro_batches[start : start + args.num_micro_batches])

    preds = []
    for batch in tqdm(batches):
        logitss = inferencer.forward(batch)
        for logits, micro_batch in zip(logitss, batch):
            logits = logits.float().flatten()
            for _logits in logits.split(micro_batch["num_candidates"]):
                preds.append(_logits.float().softmax(dim=-1).data.cpu())

    print(dataset.evaluate(preds))
    torch.save(preds, args.out)


if __name__ == "__main__":
    main()



%%time
!for i in {0..36}; do cp /kaggle/input/map-checkpoints-x-2-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
!for i in {37..63}; do cp /kaggle/input/map-checkpoints-x-2-layers/layer_$i.pth /tmp/layer-checkpoints/; done
!df -h /kaggle/working


!df -h /kaggle/working


!(CUDA_VISIBLE_DEVICES=0 python test.py \
    --model qwen3 \
    --dataset v3 \
    --model-path /kaggle/input/map-checkpoints-x-2-base \
    --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
    --checkpoint-root2 /tmp/layer-checkpoints/ \
    --csv-file test_data1.csv \
    --micro-batch-size 16 \
    --num-micro-batches 40 \
    --out preds_v3_x_part1.pth \
  & CUDA_VISIBLE_DEVICES=1 python test.py \
    --model qwen3 \
    --dataset v3 \
    --model-path /kaggle/input/map-checkpoints-x-2-base \
    --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
    --checkpoint-root2 /tmp/layer-checkpoints/ \
    --csv-file test_data2.csv \
    --micro-batch-size 16 \
    --num-micro-batches 40 \
    --out preds_v3_x_part2.pth \
  & wait)


!rm /kaggle/working/layer-checkpoints/*.pth
!df -h /kaggle/working


%%time
!for i in {0..34}; do cp /kaggle/input/map-checkpoints-y-2-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
!for i in {35..60}; do cp /kaggle/input/map-checkpoints-y-2-layers/layer_$i.pth /tmp/layer-checkpoints/; done
!df -h /kaggle/working


!(CUDA_VISIBLE_DEVICES=0 python test.py \
    --model glm4 \
    --dataset v3 \
    --model-path /kaggle/input/map-checkpoints-y-2-base \
    --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
    --checkpoint-root2 /tmp/layer-checkpoints/ \
    --csv-file test_data1.csv \
    --micro-batch-size 16 \
    --num-micro-batches 40 \
    --out preds_v3_y_part1.pth \
  & CUDA_VISIBLE_DEVICES=1 python test.py \
    --model glm4 \
    --dataset v3 \
    --model-path /kaggle/input/map-checkpoints-y-2-base \
    --checkpoint-root1 /kaggle/working/layer-checkpoints/ \
    --checkpoint-root2 /tmp/layer-checkpoints/ \
    --csv-file test_data2.csv \
    --micro-batch-size 16 \
    --num-micro-batches 40 \
    --out preds_v3_y_part2.pth \
  & wait)


!rm /kaggle/working/layer-checkpoints/*.pth
!df -h /kaggle/working


# %%time
# !for i in {0..36}; do cp /kaggle/input/map-checkpoints-x-1-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
# !for i in {37..63}; do cp /kaggle/input/map-checkpoints-x-1-layers/layer_$i.pth /tmp/layer-checkpoints/; done
# !df -h /kaggle/working


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


# !rm /kaggle/working/layer-checkpoints/*.pth
# !df -h /kaggle/working


# %%time
# !for i in {0..34}; do cp /kaggle/input/map-checkpoints-y-1-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
# !for i in {35..60}; do cp /kaggle/input/map-checkpoints-y-1-layers/layer_$i.pth /tmp/layer-checkpoints/; done
# !df -h /kaggle/working


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


# !rm /kaggle/working/layer-checkpoints/*.pth
# !df -h /kaggle/working


!ls


%%writefile make_submission.py

import glob
import torch
import numpy as np
import pandas as pd
from map_modules.data.dataset_v1 import MAPDataset


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
results = []
for _, row in df.iterrows():
    label_candidates = row["label_candidates"]
    top3_inds = (-row["preds"]).argsort()[:3]
    result = [label_candidates[i] for i in top3_inds]
    results.append({"row_id": row["row_id"], "Category:Misconception": " ".join(result)})


sub = pd.DataFrame(results)
print(sub.head())
sub.to_csv("submission.csv", index=False)



!python make_submission.py




