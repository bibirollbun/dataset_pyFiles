import pandas as pd
import numpy as np


train = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
train.head()


test = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")
test.head()


test.shape


sample_submission = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv")
sample_submission.head()


sample_submission.shape


from transformers import pipeline


classifier = pipeline("zero-shot-classification",model="anandNakat/bart_math_solver_2")


train.iloc[2,0]


label_names = ["Algebra",
               "Geometry and Trigonometry",
               "Calculus and Analysis",
               "Probability and Statistics",
               "Number Theory",
               "Combinatorics and Discrete Math",
               "Linear Algebra",
               "Abstract Algebra and Topology"
        ]


result_dict = classifier(
    train.iloc[2,0],
    candidate_labels = label_names
)
result_dict['labels'][0]


test_results = []


for i in range(test.shape[0]):
    result_dict = classifier(test.iloc[i,1],candidate_labels = label_names)
    test_results.append(result_dict['labels'][0])
    print("Solution for row ",i," is ",result_dict['labels'][0])


# Mapping dictionary
mapping = {
    "Algebra": 0,
    "Geometry and Trigonometry": 1,
    "Calculus and Analysis": 2,
    "Probability and Statistics": 3,
    "Number Theory": 4,
    "Combinatorics and Discrete Math": 5,
    "Linear Algebra": 6,
    "Abstract Algebra and Topology": 7,
}

# Encode the list
test_results_encoded = [mapping[topic] for topic in test_results]


submission = pd.DataFrame({
    'id': sample_submission['id'],
    'label': test_results_encoded
})


submission.head()


submission.shape


submission.to_csv('submission.csv', index=False)

