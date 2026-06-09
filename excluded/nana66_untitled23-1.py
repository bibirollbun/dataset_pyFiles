# ====================================================
# ライブラリのインポート
# ====================================================
import pandas as pd
import numpy as np
import os
import gc
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
import joblib # ★ モデル保存に便利なjoblibをインポート

# ====================================================
# 設定（コンフィグ）
# ====================================================
class CFG:
    # --- ファイルパス ---
    train_path = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
    test_path = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
    sample_submission_path = '/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv'
    
    # --- モデル設定 ---
    model_name = "/kaggle/input/deberta-v3-base-for-map/microsoft-deberta-v3-base" 
    
    # --- 学習パラメータ ---
    max_length = 256
    batch_size = 8
    epochs = 4  # 元の0.934を出した設定に戻す（必要に応じて調整）
    n_splits = 5 # 元の0.934を出した設定に戻す
    learning_rate = 2e-5 
    
    # --- 環境設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 1. 保存用ディレクトリのパスを追加 ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ここにモデル、Tokenizer、LabelEncoderを全て保存します
    output_dir = "./deberta_base_seed42_output"


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CFG.seed)

# (MathDataset, MeanPooling, CustomModelなどのクラス定義は変更なし)
# ... (省略) ...
class MathDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.df, self.tokenizer, self.is_train = df, tokenizer, is_train
        self.texts = df['text'].values
        if self.is_train: self.labels = df['label_id'].values
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], max_length=CFG.max_length, padding='max_length', truncation=True, return_tensors='pt')
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if self.is_train: return inputs, torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.pool = MeanPooling()
        self.fc = nn.Linear(config.hidden_size, num_labels)
    def forward(self, inputs):
        feature = self.pool(self.backbone(**inputs).last_hidden_state, inputs['attention_mask'])
        return self.fc(feature)

class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df, self.tokenizer, self.texts = df, tokenizer, df['text'].values
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], max_length=CFG.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in inputs.items()}

def create_is_correct_feature(train_df, test_df):
    print("Creating 'is_correct' feature...")
    idx = train_df.apply(lambda row: str(row.Category).split('_')[0], axis=1) == 'True'
    correct_answers = train_df.loc[idx].copy()
    correct_answers['c'] = correct_answers.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
    correct_answers = correct_answers.sort_values('c', ascending=False).drop_duplicates(['QuestionId'])[['QuestionId', 'MC_Answer']]
    correct_answers['is_correct'] = 1.0
    train_df = train_df.merge(correct_answers, on=['QuestionId', 'MC_Answer'], how='left').fillna({'is_correct': 0.0})
    test_df = test_df.merge(correct_answers, on=['QuestionId', 'MC_Answer'], how='left').fillna({'is_correct': 0.0})
    return train_df, test_df

def format_input_advanced(row):
    is_correct_text = "This answer is correct." if row['is_correct'] == 1.0 else "This answer is incorrect."
    return (f"Question: {row['QuestionText']}\nSelected Answer: {row['MC_Answer']}\n"
            f"{is_correct_text}\nStudent Explanation: {row['StudentExplanation']}")

def train_loop(model, loader, optimizer, scheduler, criterion, scaler):
    model.train(); total_loss = 0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs = {k: v.to(CFG.device) for k, v in inputs.items()}
        labels = labels.to(CFG.device)
        with torch.cuda.amp.autocast():
            loss = criterion(model(inputs), labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(); scheduler.step(); total_loss += loss.item()
    return total_loss / len(loader)

def inference_loop(model, loader):
    model.eval(); preds = []
    with torch.no_grad():
        for inputs in tqdm(loader, desc="Inferencing"):
            inputs = {k: v.to(CFG.device) for k, v in inputs.items()}
            preds.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
    return np.concatenate(preds)


# ====================================================
# メインの実行部分
# ====================================================
train_df = pd.read_csv(CFG.train_path)
test_df = pd.read_csv(CFG.test_path)

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 2. 保存用ディレクトリを作成 ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
os.makedirs(CFG.output_dir, exist_ok=True)
print(f"Output directory '{CFG.output_dir}' created.")

train_df, test_df = create_is_correct_feature(train_df.copy(), test_df.copy())
train_df['text'] = train_df.apply(format_input_advanced, axis=1)
test_df['text'] = test_df.apply(format_input_advanced, axis=1)

train_df['Misconception'] = train_df['Misconception'].fillna("NA")
train_df['target'] = train_df['Category'] + ':' + train_df['Misconception']
le = LabelEncoder().fit(train_df['target'])
train_df['label_id'] = le.transform(train_df['target'])
num_labels = len(le.classes_)

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 3. LabelEncoderオブジェクト自体を保存 ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# .npyではなく、オブジェクト全体を保存することで後で使いやすくなります
joblib.dump(le, f"{CFG.output_dir}/label_encoder.pkl")
print(f"LabelEncoder saved to '{CFG.output_dir}/label_encoder.pkl'")


# 学習済みモデルがまだ存在しない場合のみ学習を実行
# ここでは output_dir の中のファイルを確認します
if not os.path.exists(f"{CFG.output_dir}/model_fold_0.pth"):
    print(f"Models not found in '{CFG.output_dir}'. Starting training...")
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 4. Tokenizerを読み込み、すぐに保存 ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    tokenizer.save_pretrained(f"{CFG.output_dir}/tokenizer")
    print(f"Tokenizer saved to '{CFG.output_dir}/tokenizer'")

    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

    # Foldのループ（foldは0から始まるようにします）
    for fold, (train_idx, _) in enumerate(skf.split(train_df, train_df['label_id'])):
        print(f"========== Fold {fold} ==========")
        train_fold = train_df.iloc[train_idx]
        train_loader = DataLoader(MathDataset(train_fold, tokenizer), batch_size=CFG.batch_size, shuffle=True, num_workers=2)
        model = CustomModel(CFG.model_name, num_labels).to(CFG.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate)
        
        num_train_steps = len(train_loader) * CFG.epochs
        num_warmup_steps = int(num_train_steps * 0.06)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        
        criterion = nn.CrossEntropyLoss(); scaler = torch.cuda.amp.GradScaler()
        for epoch in range(CFG.epochs):
            avg_loss = train_loop(model, train_loader, optimizer, scheduler, criterion, scaler)
            print(f"Fold {fold}, Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ 5. 学習済みモデルをoutput_dirに保存 ★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ファイル名は 'model_fold_0.pth', 'model_fold_1.pth', ... となります
        torch.save(model.state_dict(), f"{CFG.output_dir}/model_fold_{fold}.pth")
        print(f"Model for fold {fold} saved to '{CFG.output_dir}/model_fold_{fold}.pth'")
        
        del model, train_loader; gc.collect(); torch.cuda.empty_cache()
else:
    print(f"Models found in '{CFG.output_dir}'. Skipping training.")


# --- 推論部分 ---
print("\nStarting inference...")
if len(test_df) == 0:
    print("Test data is empty. Creating empty submission file.")
    pd.read_csv(CFG.sample_submission_path).to_csv('submission.csv', index=False)
else:
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 6. 保存したTokenizerとLabelEncoderを読み込むように変更 ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    tokenizer = AutoTokenizer.from_pretrained(f"{CFG.output_dir}/tokenizer")
    le = joblib.load(f"{CFG.output_dir}/label_encoder.pkl")
    
    test_loader = DataLoader(InferenceDataset(test_df, tokenizer), batch_size=CFG.batch_size * 2, shuffle=False)
    final_preds = np.zeros((len(test_df), len(le.classes_)))

    # Foldのループ（foldは0から始まる）
    for fold in range(CFG.n_splits):
        print(f"========== Inferencing with Fold {fold} ==========")
        model = CustomModel(CFG.model_name, len(le.classes_)).to(CFG.device)
        
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ 7. 保存したモデルのパスを正しく指定 ★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        model_path = f"{CFG.output_dir}/model_fold_{fold}.pth"
        model.load_state_dict(torch.load(model_path))
        
        final_preds += inference_loop(model, test_loader) / CFG.n_splits
        del model; gc.collect(); torch.cuda.empty_cache()

    print("\nCreating submission file...")

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 8. このモデルセットの予測結果（.npy）も出力ディレクトリに保存 ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ファイル名をモデルに合わせて変更 (preds_base_seed42.npy)
    np.save(f'{CFG.output_dir}/preds_base_seed42.npy', final_preds)
    print(f"Saved final predictions to '{CFG.output_dir}/preds_base_seed42.npy'")
    
    # 提出ファイルの作成
    top3_indices = np.argsort(final_preds, axis=1)[:, ::-1][:, :3]
    predictions = [' '.join(le.classes_[indices]) for indices in top3_indices]
    submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'Category:Misconception': predictions})
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully!")
    print(submission_df.head())

