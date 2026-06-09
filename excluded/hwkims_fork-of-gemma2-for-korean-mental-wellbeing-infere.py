# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
# Keras Backend 설정: JAX를 가장 먼저 설정해야 합니다.
os.environ["KERAS_BACKEND"] = "jax"
# 메모리 fragmentation 방지
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras
from jax import config
import time

# JAX 환경설정, float32로 설정하여 Gemma 모델의 정확도 향상
config.update("jax_default_matmul_precision", "float32")

# 모델 ID 및 LoRA 설정
model_id = "gemma2_instruct_2b_en"  # 기본 모델 ID
lora_rank = 4
# Kaggle 모델 입력 경로 설정 (수정됨)
model_path = "/kaggle/input/gemma2-ko-dialogue-lora/keras/default/1/my_fine_tuned_gemma2_full_rank4.keras"
token_limit = 128 # 토큰 제한 설정

# 글로벌 시간 추적 변수
tick_start = 0

def tick():
    """시간 측정 시작."""
    global tick_start
    tick_start = time.time()

def tock():
    """시간 측정 종료 및 출력."""
    print(f"총 소요 시간: {time.time() - tick_start:.2f}s")

# Kaggle 모델 로드 (수정됨)
gemma_lm_loaded = keras.models.load_model(model_path)

# 텍스트 생성 함수 (수정됨)
def generate_text(prompt):
  tick() # 시간 측정 시작
  input_text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
  output = gemma_lm_loaded.generate(input_text, max_length=token_limit) # 모델 추론
  print(f"Generated text: {output}")
  tock() # 시간 측정 종료 및 출력

# 테스트 문장
test_prompt_1 = "오늘 기분이 너무 안좋아."
generate_text(test_prompt_1)

test_prompt_2 = "힘든 일이 있어서 위로가 필요해."
generate_text(test_prompt_2)

test_prompt_3 = "오늘따라 괜히 울적해지네."
generate_text(test_prompt_3)

test_prompt_4 = "날씨가 흐려서 그런가, 기분이 꿀꿀해."
generate_text(test_prompt_4)

test_prompt_5 = "아무것도 하기 싫어, 그냥 멍하니 있고 싶어."
generate_text(test_prompt_5)

test_prompt_6 = "요즘 따라 무기력감이 심해지는 것 같아."
generate_text(test_prompt_6)

test_prompt_7 = "갑자기 옛날 생각에 잠겨서 센치해졌어."
generate_text(test_prompt_7)

test_prompt_8 = "오늘따라 모든 게 다 짜증나고 예민해."
generate_text(test_prompt_8)

test_prompt_9 = "뭔가 불안하고 초조한 마음이 계속 들어."
generate_text(test_prompt_9)

test_prompt_10 = "그냥 이유 없이 답답하고 숨 막히는 기분이야."
generate_text(test_prompt_10)

test_prompt_11 = "혼자 있는 시간이 필요한 것 같아."
generate_text(test_prompt_11)

test_prompt_12 = "에너지가 하나도 없어, 완전 방전된 느낌."
generate_text(test_prompt_12)

test_prompt_13 = "시험을 망쳐서 너무 속상해."
generate_text(test_prompt_13)

test_prompt_14 = "발표를 망쳐서 자존감이 떨어졌어."
generate_text(test_prompt_14)

test_prompt_15 = "상사한테 혼나서 기분이 너무 안 좋아."
generate_text(test_prompt_15)

test_prompt_16 = "친구랑 싸워서 마음이 불편해."
generate_text(test_prompt_16)

test_prompt_17 = "이별 후유증이 너무 심해."
generate_text(test_prompt_17)

test_prompt_18 = "취업 준비가 너무 힘들고 막막해."
generate_text(test_prompt_18)

test_prompt_19 = "인간관계가 너무 복잡하고 어려워."
generate_text(test_prompt_19)

test_prompt_20 = "학업 스트레스 때문에 너무 지쳐."
generate_text(test_prompt_20)

test_prompt_21 = "경제적인 어려움 때문에 힘들어."
generate_text(test_prompt_21)

test_prompt_22 = "미래에 대한 불안감이 커."
generate_text(test_prompt_22)

test_prompt_23 = "모든 게 내 잘못인 것 같아."
generate_text(test_prompt_23)

test_prompt_24 = "나만 뒤쳐지는 것 같아서 불안해."
generate_text(test_prompt_24)

test_prompt_25 = "아무도 나를 이해해주지 못하는 것 같아."
generate_text(test_prompt_25)

test_prompt_26 = "뭘 해도 안 될 것 같은 기분이 들어."
generate_text(test_prompt_26)

test_prompt_27 = "나 자신이 너무 싫어."
generate_text(test_prompt_27)

test_prompt_28 = "앞으로 어떻게 해야 할지 모르겠어."
generate_text(test_prompt_28)

test_prompt_29 = "너무 힘들어서 다 포기하고 싶어."
generate_text(test_prompt_29)

test_prompt_30 = "외롭고 고독한 느낌이 계속 들어."
generate_text(test_prompt_30)

test_prompt_31 = "세상에 혼자 남겨진 것 같아."
generate_text(test_prompt_31)

test_prompt_32 = "희망이 보이지 않아."
generate_text(test_prompt_32)

test_prompt_33 = "오늘따라 유난히 피곤하네."
generate_text(test_prompt_33)

test_prompt_34 = "커피를 쏟아서 옷이 엉망이 됐어."
generate_text(test_prompt_34)

test_prompt_35 = "아침부터 지각해서 하루 종일 정신이 없어."
generate_text(test_prompt_35)

test_prompt_36 = "오늘따라 일이 너무 많아서 힘들어."
generate_text(test_prompt_36)

test_prompt_37 = "인터넷이 안 돼서 답답해 죽겠어."
generate_text(test_prompt_37)

test_prompt_38 = "버스를 놓쳐서 약속에 늦었어."
generate_text(test_prompt_38)

test_prompt_39 = "오늘따라 운이 안 좋은 것 같아."
generate_text(test_prompt_39)

test_prompt_40 = "감기에 걸려서 컨디션이 안 좋아."
generate_text(test_prompt_40)

test_prompt_41 = "며칠 밤샘 작업했더니 너무 힘들어."
generate_text(test_prompt_41)

test_prompt_42 = "오늘따라 괜히 심통이 나."
generate_text(test_prompt_42)

test_prompt_43 = "마음이 텅 빈 것 같아."
generate_text(test_prompt_43)

test_prompt_44 = "삶의 의미를 잃어버린 것 같아."
generate_text(test_prompt_44)

test_prompt_45 = "나락으로 떨어지는 기분이야."
generate_text(test_prompt_45)

test_prompt_46 = "깊은 수렁에 빠진 것 같아."
generate_text(test_prompt_46)

test_prompt_47 = "암흑 속을 헤매는 기분이야."
generate_text(test_prompt_47)

test_prompt_48 = "가슴이 먹먹하고 답답해."
generate_text(test_prompt_48)

test_prompt_49 = "희망의 빛이 보이지 않아."
generate_text(test_prompt_49)

test_prompt_50 = "마음의 상처가 아물지 않아."
generate_text(test_prompt_50)

test_prompt_51 = "나를 갉아먹는 기분이야."
generate_text(test_prompt_51)

test_prompt_52 = "모든 걸 놓고 싶어."
generate_text(test_prompt_52)

