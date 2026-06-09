import json

r=r'/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
with open(r, 'r') as file:
    test_data = json.load(file)

submission_data = {}
dummy_data = [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}]
for i in (list(test_data.keys())):
    number_of_test_inputs = len(test_data[i]['test'])
    submission_data[i] = dummy_data * number_of_test_inputs


with open('submission.json', 'w') as file:
    json.dump(submission_data, file)




