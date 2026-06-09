from pathlib import Path
import pandas as pd

sub = pd.read_csv(Path('/kaggle/input/march-machine-learning-mania-2025/SeedBenchmarkStage1.csv'))


from march_madness_submission_tester import evaluate_stage1_submission

# view docstring
evaluate_stage1_submission?


evaluate_stage1_submission(sub)


evaluate_stage1_submission(sub, [2024])


from march_madness_submission_tester import evaluate_stage1_submission_games

evaluate_stage1_submission_games(sub, [2024])


from march_madness_submission_tester import validate_submission_format

validate_submission_format(sub, check_seasons=[2021,2022,2023,2024])


validate_submission_format(sub, check_seasons=[2025])


from march_madness_submission_tester import _swap_game_id

bad_sub = sub.copy()
bad_sub["ID"] = bad_sub["ID"].map(_swap_game_id)
bad_sub["extra_col"] = None

validate_submission_format(bad_sub, check_seasons=[2021,2022,2023,2024])


from march_madness_submission_tester import make_template_submission

s = [2025]
empty_sub = make_template_submission(s)
v = validate_submission_format(empty_sub, s)
display(empty_sub.head(), v)




