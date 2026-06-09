import json, yaml, os, sys
import numpy as np
import tqdm


config_file='/kaggle/input/lk-arc-solver-v01/config.yaml'
with open(config_file, "r") as f:
    config = yaml.safe_load(f)


config


config['CHALLENGES_PATH'] = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
config['LLM_PATH'] = '/kaggle/input/local-distilgpt2'
config['VIS_ENC_PATH'] = '/kaggle/input/arc-vision-encoder-v03/vision_encoder_weights.pth'
config['LLM_SAVE_PATH'] = './'
config['SUBMISSION_PATH'] = 'solved.json'


with open("config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)


# !timeout --preserve-status 11h55m python my_script.py
# !timeout --preserve-status 710m python '/kaggle/input/lk-arc-solver-v01/v26_script.py' config.yaml

# !timeout --preserve-status 30m python '/kaggle/input/lk-arc-solver-v01/v26_script.py' config.yaml > run.log 2>&1

!timeout --preserve-status 660m python '/kaggle/input/lk-arc-solver-v01/v26_script.py' config.yaml > run.log 2>&1





with open(config['CHALLENGES_PATH']) as f:
    d = json.load(f)


def generate_output(challenge_id, dct):
    input_set = dct[challenge_id]['test']
    res_set = []
    for inp in input_set:
        res_dct = {'attempt_1':[[0, 0], [0, 0]], 'attempt_2':[[0, 0], [0, 0]]}
        res_set.append(res_dct)
    return res_set


sub_data = {}
for k,v in tqdm.tqdm(d.items()):
    sub_data[k] = generate_output(k, d)


# fill the solved challenges

try: 
    with open(config['SOLVED_PATH']) as f:
        s = json.load(f)
    for k,v in tqdm.tqdm(s.items()):
        sub_data[k] = v

except Exception as e:
    print(e)
    pass





# Save the data to the output file
with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(sub_data, f, indent=4)







