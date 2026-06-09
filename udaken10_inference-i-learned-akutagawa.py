import os

model_path = "/kaggle/input/akutagawa_style_gemma/keras/default/1/akutagawa_fine_tuned_model"
print(os.listdir(model_path))  # モデルディレクトリ内のファイル一覧を表示



import os
import keras_nlp

# 必要な情報を設定
model_id = "/kaggle/input/gemma2/keras/gemma2_instruct_2b_en/1"  # 使用したベースモデルID
lora_weights_path = "/kaggle/input/akutagawa_style_gemma/keras/default/1/akutagawa_fine_tuned_model/my_lora_4_final.lora.h5"  # 保存済みのLoRA重み

# ベースモデルをロード
print("ベースモデルをロード中...")
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.backbone.enable_lora(rank=4)  # LoRAのランクを適用（保存時と一致させる）

# 保存済みのLoRA重みを適用
if os.path.exists(lora_weights_path):
    print(f"LoRA重みをロード: {lora_weights_path}")
    gemma_lm.backbone.load_lora_weights(lora_weights_path)
else:
    raise FileNotFoundError(f"LoRA重みが見つかりません: {lora_weights_path}")

# モデルの構造を出力
print("ファインチューニング済みモデルの構造:")
gemma_lm.summary()

# モデルを使用してテキスト生成テスト
def generate_text(prompt):
    print(f"\nプロンプト: {prompt}")
    result = gemma_lm.generate(prompt, max_length=128)
    print(f"生成されたテキスト: {result}")

# テスト用プロンプト
generate_text("芥川龍之介のような短編小説を書いてください。")



# テスト用プロンプト
generate_text("write a short novel written in the 芥川龍之介 style in Japanese")


import os
import keras_nlp
import time

# 推論用の関数
def text_gen(prompt, model, tokenizer, max_length=128):
    """
    推論を行う関数。

    Args:
        prompt (str): 入力プロンプト。
        model: ファインチューニング済みモデル。
        tokenizer: トークナイザ。
        max_length (int): 生成するテキストの最大トークン長。

    Returns:
        str: 生成されたテキスト。
    """
    input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    generated = model.generate(input_text, max_length=max_length)
    print("Generated Text:")
    print(generated)
    return generated

# モデルとトークナイザの読み込み
# save_directory = "akutagawa_fine_tuned_model"  # ファインチューニングしたモデルが保存されているディレクトリ
save_directory = "/kaggle/input/akutagawa_style_gemma/keras/default/1/akutagawa_fine_tuned_model"  # ファインチューニングしたモデルが保存されているディレクトリ

model = keras_nlp.models.GemmaCausalLM.from_preset(save_directory)
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(save_directory)

# モデルの構造を出力
print("ファインチューニング済みモデルの構造:")
model.summary()

# 推論例
prompt = "芥川龍之介のスタイルで短編小説を書いてください。"
text_gen(prompt, model, tokenizer)


