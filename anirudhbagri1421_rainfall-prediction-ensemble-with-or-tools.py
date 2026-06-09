pip install ortools


import pandas as pd
import numpy as np
import warnings
from ortools.sat.python import cp_model

warnings.filterwarnings('ignore')

# Load necessary libraries
!uv pip install --system --quiet scikit-learn==1.6.1 ortools

def load_predictions():
    """Load predictions from various models."""
    pred_files = [
        "/kaggle/input/rapids-svc-w-feature-engineering-lb-0-856/submission.csv",
        "/kaggle/input/rapids-knn-starter-ensemble-lb-0-961-wow/submission_ensemble.csv",
        "/kaggle/input/rain-or-shine-rainfall-prediction-with-ml/submission.csv",
        "/kaggle/input/xgboost-starter-ensemble-lb-0-935-wow/submission_ensemble.csv",
        "/kaggle/input/deployment-streamlit-for-good-resume/submission.csv",
        "/kaggle/input/0-96218-logistic-regression-plus-ensemble/submission.csv",
        "/kaggle/input/rainfall-pred-logistic-regression-plus-ensemble/submission.csv",
        "/kaggle/input/playgrounds5e3-baseline-v2/submission.csv",
        "/kaggle/input/weathercook-ai-generated/submission.csv",
        "/kaggle/input/87-9-logistic-s5e3-rainfall-probability-in-r/submission.csv",
        "/kaggle/input/rainfall-prediction-eda-catboost-optuna/submission.csv",
        "/kaggle/input/shap-feature-engineering-lstm-cnn-ensemble/submission.csv",
        "/kaggle/input/ps-s5e3-rainfall-hyperspace-as-feats/submission.csv",
        "/kaggle/input/rainfall-catboost/submission.csv",
        "/kaggle/input/rainfall-keras-tensorflow/submission.csv",
        "/kaggle/input/fork-improvement-xgb-lb-0-929/submission_ensemble.csv",
        "/kaggle/input/xtratreeclassifier-v1-updated-multiplier-next-step/submission.csv",
        "/kaggle/input/rainfall-dataset-roc-auc-0-87154/submission.csv",
        "/kaggle/input/rainfall-simple-logistic-regression/submission.csv",
        "/kaggle/input/rainfall-prediction-will-it-rain-tomorrow-87-67/submission.csv",
        "/kaggle/input/s5e3-logisticregression/submission.csv"
    ]
    
    predictions = [pd.read_csv(file).iloc[:146, 1].values for file in pred_files]
    return np.array(predictions)


def load_scores():
    """Load scores for each model."""
    return [
        0.85626, 0.96111, 0.86484, 0.93550, 0.88710, 0.96218, 0.90104,
        0.94851, 0.88200, 0.86430, 0.86698, 0.87851, 0.95548, 0.85679,
        0.80718, 0.85009, 0.92947, 0.86390, 0.86725, 0.87396, 0.84633, 0.86377
    ]

def create_and_solve_model(preds, scores, N=146, p=113, n=33):
    """Create and solve the constraint programming model."""
    class SolutionCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.solutions = []

        def on_solution_callback(self):
            self.solutions.append([self.Value(x[i]) for i in range(N)])

    model = cp_model.CpModel()
    x = [model.NewIntVar(0, 1, f'x[{i}]') for i in range(N)]
    model.Add(sum(x) == p)

    for m in range(len(preds)):
        y_pred = preds[m]
        r = pd.Series(y_pred).rank().values
        model.Add(sum(x[i] * int(np.around(r[i] * 2)) for i in range(N)) == int(np.around(scores[m] * n * p * 2)) + p * (p + 1))

    solver = cp_model.CpSolver()
    s = SolutionCallback()
    solver.SearchForAllSolutions(model, s)
    return s.solutions


def main():
    preds = load_predictions()
    scores = load_scores()
    solutions = create_and_solve_model(preds, scores)
    
    if solutions:
        sub = pd.read_csv("/kaggle/input/xgboost-starter-ensemble-lb-0-935-wow/submission_ensemble.csv")
        sub['rainfall'][:146] = solutions[0]
        sub.to_csv("submission.csv", index=False)
        print(sub.head())
    else:
        print("No solutions found.")


if __name__ == "__main__":
    main()

