!pip install -q -U keras-nlp datasets
!pip install -q -U keras

import os

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

model_id = "/kaggle/input/gemma2-inst-2b-hanja2hangul/keras/default/1/gemma2_inst_2b_hanja2hangul"


import keras
import keras_nlp
import time

gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma.summary()


max_length = 512 
test_hanja = '上在潛邸, 夢有神人執金尺自天而降, 授之曰: "慶侍中 復興, 淸矣而已老; 崔都統 瑩, 直矣而少戇。 持此正國, 非公而誰!"'
test_answer = '임금이 잠저(潛邸)에 있을 때, 꿈에 신인(神人)이 금자[金尺]를 가지고 하늘에서 내려와 주면서 말하기를, "시중(侍中) 경복흥(慶復興)은 청렴하기는 하나 이미 늙었으며, 도통(都統) 최영(崔瑩)은 강직하기는 하나 조금 고지식하니, 이것을 가지고 나라를 바룰 사람은 공(公)이 아니고 누구이겠는가?" 하였다.'

input = f"<start_of_turn>user\n{test_hanja}<end_of_turn>\n<start_of_turn>model\n"
gemma.generate(input, max_length=max_length)



max_length=512 
few_shot = [
    ('庚午朔/隕霜。', '경오삭(庚午朔)에 서리가 내리었다.'),
    ('壬申/上坐東涼廳, 召閱擲石軍, 命中樞趙琦領之。', '임금이 동량청(東涼廳)에 앉아 척석군(擲石軍)을 불러 사열하고, 중추(中樞) 조기(趙琦)에게 명하여 이들을 거느리게 하였다.'),
    ('都評議使司具戶曹給田司田制, 狀申, 上允之。', '도평의사사에서 호조 급전사(戶曹給田司)의 전제(田制)를 갖추어 장계(狀啓)로 아뢰니, 임금이 그대로 윤허하였다.')
]

few_shot_str = ""
for x, y in few_shot:
    few_shot_str += f"<start_of_turn>user\n{x}<end_of_turn>\n<start_of_turn>model\n{y}<end_of_turn>\n"
    
input = f"{few_shot_str}<start_of_turn>user\n{test_hanja}<end_of_turn>\n<start_of_turn>model\n"
gemma.generate(input, max_length=max_length)





