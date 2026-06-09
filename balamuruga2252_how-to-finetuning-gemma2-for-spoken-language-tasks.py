


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp datasets
!pip install -q -U keras

import os

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras

# Run at half precision.
#keras.config.set_floatx("bfloat16")

# Training Configurations
token_limit = 256
num_data_limit = 100
lora_name = "cakeboss"
lora_rank = 4
lr_value = 1e-4
train_epoch = 20
model_id = "gemma2_instruct_2b_en"


import keras
import keras_nlp

import time

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.summary()

tick_start = 0

def tick():
    global tick_start
    tick_start = time.time()

def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")

def text_gen(prompt):
    tick()
    input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = gemma_lm.generate(input, max_length=token_limit)
    print("\nGemma output:")
    print(output)
    tock()

# inference before fine-tuning
text_gen("다음에 대한 이메일 답장을 작성해줘.\n\"안녕하세요, 결혼기념일을 위해 3호 케이크 1개를 주문하고 싶은데 가능할까요?\"")


import keras
import keras_nlp
import datasets

tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

# prompt structure
# <start_of_turn>user
# 다음에 대한 이메일 답장을 작성해줘.
# "{EMAIL CONTENT FROM THE CUSTOMER}"
# <end_of_turn>
# <start_of_turn>model
# {MODEL ANSWER}<end_of_turn>

# input, output
from datasets import load_dataset
ds = load_dataset(
    "bebechien/korean_cake_boss",
    split="train",
)
print(ds)
data = ds.with_format("np", columns=["input", "output"], output_all_columns=False)
train = []

for x in data:
  item = f"<start_of_turn>user\n다음에 대한 이메일 답장을 작성해줘.\n\"{x['input']}\"<end_of_turn>\n<start_of_turn>model\n{x['output']}<end_of_turn>"
  length = len(tokenizer(item))
  # skip data if the token length is longer than our limit
  if length < token_limit:
    train.append(item)
    if(len(train)>=num_data_limit):
      break

print(len(train))
print(train[0])
print(train[1])
print(train[2])


# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.summary()

# Limit the input sequence length (to control memory usage).
gemma_lm.preprocessor.sequence_length = token_limit
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    model_name = f"/kaggle/working/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5"
    gemma_lm.backbone.save_lora_weights(model_name)

    # Evaluate
    text_gen("다음에 대한 이메일 답장을 작성해줘.\n\"안녕하세요, 결혼기념일을 위해 3호 케이크 1개를 주문하고 싶은데 가능할까요?\"")

history = gemma_lm.fit(train, epochs=train_epoch, batch_size=1, callbacks=[CustomCallback()])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.show()



# Example Code for Load LoRA
'''
import os
import keras
import keras_nlp

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_instruct_2b_en")
# Use the same LoRA rank that you trained
gemma_lm.backbone.enable_lora(rank=4)

# Load pre-trained LoRA weights
gemma_lm.backbone.load_lora_weights(f"/kaggle/working/cakeboss_4_epoch17.lora.h5")
'''


gemma_lm.compile(sampler="top_k")
text_gen("다음에 대한 이메일 답장을 작성해줘.\n\"안녕하세요, 결혼기념일을 위해 3호 케이크 1개를 주문하고 싶은데 가능할까요?\"")
text_gen("다음에 대한 이메일 답장을 작성해줘.\n\"안녕하세요, 결혼기념일을 위해 3호 케이크 1개를 주문하고 싶은데 가능할까요?\"")
text_gen("다음에 대한 이메일 답장을 작성해줘.\n\"안녕하세요, 결혼기념일을 위해 3호 케이크 1개를 주문하고 싶은데 가능할까요?\"")


text_gen("다음에 대한 답장을 작성해줘.\n\"안녕하세요, 결혼기념일을 위해 3호 케이크 1개를 주문하고 싶은데 가능할까요?\"")
text_gen("아래에 적절한 답장을 써줘.\n\"안녕하세요, 결혼기념일을 위해 3호 케이크 1개를 주문하고 싶은데 가능할까요?\"")
text_gen("다음에 관한 답장을 써주세요.\n\"안녕하세요, 결혼기념일을 위해 3호 케이크 1개를 주문하고 싶은데 가능할까요?\"")


text_gen("""다음에 대한 이메일 답장을 작성해줘.
"안녕하세요,

6월 15일에 있을 행사 답례품으로 쿠키 & 머핀 세트를 대량 주문하고 싶습니다.

수량: 50세트
구성: 쿠키 2개 + 머핀 1개 (개별 포장)
디자인: 심플하고 고급스러운 디자인 (리본 포장 등)
문구: "감사합니다" 스티커 부착
배송 날짜: 6월 14일
대량 주문 할인 혜택이 있는지, 있다면 견적과 함께 배송 가능 여부를 알려주시면 감사하겠습니다.

감사합니다.

박철수 드림" """)

