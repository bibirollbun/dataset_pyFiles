# 必要なライブラリのインポート
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# GPU が利用可能か確認（GPU必須）
if not torch.cuda.is_available():
    raise Exception("GPU is required for inference!")

# モデルのパスを指定（Kaggle 上にアップロードされた mistral-ai/mistral/pyTorch/7b-v0.1 モデルのパスに合わせてください）
model_path = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"  # 必要に応じてパスを修正

# トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(model_path)

# モデルの読み込み
# offload_folder を指定して、GPU メモリに収まりきらない重みを一時的にディスクにオフロード
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 半精度でメモリ節約（GPU環境の場合）
    device_map="auto",          # 自動で GPU に配置
    offload_folder="./offload", # オフロード用フォルダを指定
    trust_remote_code=True      # モデル固有の実装を信頼する
)

# テキスト生成用パイプラインの作成
# Mistral は因果言語モデルとして提供されているため、"text-generation" タスクを使用します
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,  # 生成するトークン数は必要に応じて調整
    do_sample=True,
    temperature=0.8,
    top_p=0.9
)

# テストデータの読み込み（コンペ提供の test.csv を使用する前提）
test_df = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
print("Test data sample:")
print(test_df.head())

# 各トピックに対してエッセイを生成する関数
def generate_essay(topic):
    # プロンプトの例：トピックを渡して、約100語のエッセイを生成するよう指示
    prompt = f"Topic: {topic}\nWrite an essay of about 100 words."
    generated = generator(prompt)
    # generator() はリスト形式で結果を返すため、先頭の結果を利用
    essay = generated[0]['generated_text']
    # 生成結果の前後の空白を除去して返す
    return essay.strip()

# テストデータの各トピックに対してエッセイを生成
test_df["essay"] = test_df["topic"].apply(generate_essay)

# 提出ファイルの DataFrame を作成（"id" と "essay" の2列）
submission = test_df[["id", "essay"]]

# 生成した提出ファイルを CSV として保存
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")




