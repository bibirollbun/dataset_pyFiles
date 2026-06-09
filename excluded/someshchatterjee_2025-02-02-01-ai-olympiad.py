import os
import kaggle_evaluation.aimo_2_inference_server


!cp -r ./../input/llm-utilities/* /kaggle/working


from helpers.config import Config
from helpers.configs.gpu_configs.gpu_4 import GPU_4 as gpu_config
from helpers.local_llm_helper import LocalLLMHelper
# from models.transformer_based.phi_basic import Phi as model_wrapper
# from models.vllm_based.models import QWEN_2_5_1_5_Instruct as t as model_wrapper
from models.vllm_based.models import DeepseekR1DistillQwen_32b as model_wrapper


Config.override_gpu_defaults_with(gpu_config)
llm_helper = LocalLLMHelper(model_wrapper)


llm_helper.get_answer('What is 45*9?')


import pandas as pd
dataset = pd.read_csv('/kaggle/input/numinamath-aime-validation-set/NuminaMath_AIME_validation.csv', index_col=0)
dataset.loc[:, ['problem']].to_csv('./data_for_val.csv')


# predicted = []
# problems = []
# actuals = []
# for i, row in dataset.iterrows():
#     problems.append(row['problem'])
#     actuals.append(row['answer'])
#     predicted.append(llm_helper.get_answer(row['problem']))
#     if i > 10:
#         break


# validation_result = pd.DataFrame({'problem': problems, 'actual': actuals, 'predicted': predicted})
# validation_result.to_csv('validation_result.csv')
# validation_result


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(llm_helper.predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            # '/kaggle/working/data_for_val.csv',
        )
    )




