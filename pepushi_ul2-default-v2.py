# 必要なライブラリのインポート
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# GPU が利用可能か確認
if not torch.cuda.is_available():
    raise Exception("GPU is required for model inference!")

# モデルのパス
model_path = "/kaggle/input/ul2/pytorch/default/2"

# トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(model_path)

# モデルの読み込み（offload_folder を指定して、重みのオフロード先を設定）
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # メモリ節約のため半精度を使用（GPU環境の場合）
    device_map="auto",          # 自動でGPUに配置
    offload_folder="./offload", # GPUメモリに収まりきらない部分を一時的にディスクにオフロード
    trust_remote_code=True
)

# テキスト生成用パイプラインの作成
# UL2はSeq2Seqモデルとして利用されるため、"text2text-generation" タスクを使用
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=120,  # 生成するトークン数は必要に応じて要調整
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# テストデータの読み込み
test_df = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
print("Test data sample:")
display(test_df.head())

# トピックに対してエッセイを生成する関数
def generate_essay(topic):
    # プロンプトの例
    prompt = f"Topic: {topic}\nPlease write an essay of about 100 words."
    # 生成結果を取得（リスト形式で返ってくるので、先頭の結果を使用）
    generated = generator(prompt)
    essay = generated[0]['generated_text']
    return essay.strip()

# 各テストトピックに対してエッセイ生成
submission = pd.DataFrame()
submission["id"] = test_df["id"]
submission["essay"] = test_df["topic"].apply(generate_essay)

# 生成したエッセイのサンプルを表示
print("Submission sample:")
display(submission.head())

# 提出用ファイル submission.csv として保存
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")




