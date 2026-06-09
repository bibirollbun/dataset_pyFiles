import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import Dataset
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.preprocessing import LabelEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model_name = "/kaggle/input/ettin-1b-dual-head-last2-fine-tune-clean-data"



train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')


le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception


_votes = (
    train.assign(is_true=train['Category'].str.startswith('True'))
         .groupby(['QuestionId', 'MC_Answer'], as_index=False)
         .agg(true_votes=('is_true', 'sum'),
              total=('is_true', 'count'))
)
_votes['false_votes'] = _votes['total'] - _votes['true_votes']
_votes['score'] = _votes['true_votes'] - _votes['false_votes']
print(_votes.head())  


_gold = (_votes.sort_values(['QuestionId','score','true_votes','total'],
                            ascending=[True, False, False, False])
               .drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
               .rename(columns={'MC_Answer':'gold_mc'}))
_gold


class DualHeadEncoderForSequenceClassification(nn.Module):
    def __init__(self, backbone, hidden_size, true_classes, false_classes):
        super().__init__()
        self.backbone = backbone
        self.true_head = nn.Linear(hidden_size, len(true_classes))
        self.false_head = nn.Linear(hidden_size, len(false_classes))
        self.true_classes = list(map(str, true_classes))
        self.false_classes = list(map(str, false_classes))

    @staticmethod
    def _pool_cls(hidden_states):
        # Encoder系はCLS（先頭トークン）をプーリングに使うのが自然
        return hidden_states[:, 0, :]

    def forward(self, input_ids=None, attention_mask=None, labels=None, is_correct=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = self._pool_cls(outputs.last_hidden_state)

        # ★ ここで dtype を統一（ヘッドの重み dtype に合わせる）
        head_dtype = self.true_head.weight.dtype  # 両ヘッド同じはず
        if pooled.dtype != head_dtype:
            pooled = pooled.to(head_dtype)

        true_logits  = self.true_head(pooled)
        false_logits = self.false_head(pooled)

        loss = None
        if labels is not None and is_correct is not None:
            loss_fct = nn.CrossEntropyLoss()
            true_mask = is_correct.bool()
            false_mask = ~true_mask
            parts, cnt = 0.0, 0
            if true_mask.any():
                parts += loss_fct(true_logits[true_mask], labels[true_mask]); cnt += 1
            if false_mask.any():
                parts += loss_fct(false_logits[false_mask], labels[false_mask]); cnt += 1
            if cnt > 0:
                loss = parts / cnt

        return SequenceClassifierOutput(
            loss=loss,
            logits=true_logits,  # 代表としてtrue_logits
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        ), {"true_logits": true_logits, "false_logits": false_logits}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ① Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:  # 未設定なら念のため
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

# ② Backbone をロード（GPUならfp16も可）
backbone = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
)

# ③ ヘッドとクラス情報をロード（CPUでOK）
ckpt = torch.load(f"{model_name}/dual_heads.pt", map_location="cpu")
TRUE_CLASSES  = list(ckpt["true_classes"])   # 学習時順序を維持
FALSE_CLASSES = list(ckpt["false_classes"])
print(TRUE_CLASSES, FALSE_CLASSES)
# ④ モデル構築
hidden_size = backbone.config.hidden_size
model = DualHeadEncoderForSequenceClassification(
    backbone=backbone,
    hidden_size=hidden_size,
    true_classes=TRUE_CLASSES,
    false_classes=FALSE_CLASSES,
)
model.true_head.load_state_dict(ckpt["true_head"])
model.false_head.load_state_dict(ckpt["false_head"])

# ⑤ まとめて GPU へ
model.to(device)
model.eval()

print("cuda available:", torch.cuda.is_available())
print("model device :", next(model.parameters()).device)



# 入力フォーマット（そのまま）
def format_input(row):
    x = "Yes" if row['is_correct'] == 1 else "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )


def tokenize_dual_head(batch):
    enc = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
    return enc



test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test_df.shape )
test_df.head()


test_df = test_df.merge(_gold, on=['QuestionId'], how='left')
test_df['is_correct'] = (test_df['MC_Answer'] == test_df['gold_mc']).astype(int)

test_df['text'] = test_df.apply(format_input,axis=1)



COLS = ['text', 'is_correct']
test_ds = Dataset.from_pandas(test_df[COLS].reset_index(drop=True))

test_ds = test_ds.map(tokenize_dual_head, batched=True)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 推論 ====
loader = DataLoader(
    test_ds,
    batch_size=32,
    collate_fn=default_data_collator
)

model_device = next(model.parameters()).device
all_probs = []

model.eval()
with torch.inference_mode():
    for batch in loader:
        batch = {k: v.to(model_device, non_blocking=True) for k, v in batch.items()}

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,
            is_correct=batch.get("is_correct")  # 推論時にも is_correct を渡す
        )

        if isinstance(out, tuple):
            extra = out[1]  # {"true_logits": ..., "false_logits": ...}
        else:
            extra = out

        true_probs  = torch.softmax(extra["true_logits"].float(),  dim=-1)
        false_probs = torch.softmax(extra["false_logits"].float(), dim=-1)

        # is_correct に応じて使わない側をゼロ化
        is_corr = batch["is_correct"].bool()
        B = is_corr.size(0)

        # バッチごとにゼロ初期化
        zero_false = torch.zeros_like(false_probs)
        zero_true  = torch.zeros_like(true_probs)

        # True サンプルには true_probs を代入
        zero_true[is_corr] = true_probs[is_corr]
        # False サンプルには false_probs を代入
        zero_false[~is_corr] = false_probs[~is_corr]

        combined = torch.cat([zero_false, zero_true], dim=-1)
        all_probs.append(combined.cpu())

all_probs = torch.cat(all_probs, dim=0).numpy()



# False → True の順で結合したクラス名リスト
combined_classes = np.array(list(FALSE_CLASSES) + list(TRUE_CLASSES))

# 上位3クラスのインデックス
top3_idx = np.argsort(-all_probs, axis=1)[:, :3]          # (N, 3)

# インデックス→ラベル名に変換
top3_labels = combined_classes[top3_idx]     

# Join 3 labels per row with space
joined_preds = [" ".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test_df.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()











