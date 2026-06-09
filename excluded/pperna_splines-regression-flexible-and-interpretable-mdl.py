import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col=0)

# Print dataset shape (rows, columns)
print(f"Dataset shape: {data.shape}")

# Define the target variable
target = "rainfall"

# Separate features (X) and target (y)
X = data.drop(columns=[target])
y = data[[target]]


from sklearn.model_selection import train_test_split

# First split: 90% train, 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

# Second split: 80% train, 20% validation (from the original train set)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

# Print dataset sizes
print(f"Train set: {X_train.shape}, Validation set: {X_valid.shape}, Test set: {X_test.shape}")


import matplotlib.pyplot as plt

# Merge train features and target for analysis
df_train = X_train.merge(y_train, left_index=True, right_index=True)

# Compute and plot the mean probability of rain for each day
df_train.groupby("day")[target].mean().plot(color="blue")

# Plot settings
plt.title("Percentage of Rainy Days for Each Day of the Year")
plt.xlabel("Day of the Year")
plt.ylabel("Rain Probability")
plt.show()


def rolling_target(
    feature_name: str,
    rolling_window: int = 10
) -> pd.DataFrame:
    """
    Computes a rolling mean of the target variable over a specified window.
    This helps smooth seasonal patterns.
    """
    rolling_raining_days = (
        df_train[[target, feature_name]]
        .groupby(feature_name).mean()
        .rolling(rolling_window, center=True).mean()
    )
    return rolling_raining_days

# Apply rolling mean to the 'day' variable
rolling_raining_days = rolling_target("day")

# Plot the smoothed trend
rolling_raining_days.plot(color="blue")

# Plot settings
plt.title("Smoothed Frequency of Rain (Rolling Mean)")
plt.xlabel("Day of the Year")
plt.ylabel("Rain Frequency")
plt.show()


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer

def splines_pipeline(feature_name: str, n_knots: int) -> ColumnTransformer:
    """
    Creates a preprocessing pipeline that applies a spline transformation to a given feature.
    """
    day_splines = ColumnTransformer(
        [(feature_name, SplineTransformer(n_knots=n_knots), [feature_name])],
        remainder="drop"
    )
    return day_splines

# Create and fit the spline transformation
day_splines_pipe = splines_pipeline("day", 10)
splines_features = day_splines_pipe.fit_transform(df_train)

# Plot the transformed splines
fig, axs = plt.subplots()
axs.plot(rolling_raining_days, color="blue")
axs.plot(
    pd.DataFrame(splines_features, index=df_train["day"]).sort_index() / splines_features.max().max(),
    alpha=0.5
)
plt.title("Spline Features vs. Rolling Rain Frequency")
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def make_complete_pipeline(preprocessing: Pipeline) -> Pipeline:
    """
    Creates a full pipeline with preprocessing and a logistic regression model.
    """
    lr = LogisticRegression(random_state=42)
    complete_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", lr)
    ])
    return complete_pipeline

# Train the model
complete_pipeline = make_complete_pipeline(day_splines_pipe)
complete_pipeline.fit(
    X_train,
    np.array(y_train).ravel()
)


from sklearn.metrics import roc_auc_score

def get_model_score(
    X_test: np.array,
    y_test: np.array,
    model
) -> float:
    """
    Compute the AUC-ROC score and return predicted probabilities.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_prob)
    return score, y_prob


def get_model_score(
    X_test: np.array,
    y_test: np.array,
    model,
) -> float:
    """Compute the AUC-ROC score and return predicted probabilities."""
    y_prob = model.predict_proba(X_test)[:,1]
    score = roc_auc_score(y_test, y_prob)
    return score, y_prob


def estimate_raining_prob(
    complete_pipeline: Pipeline,
    X_train: pd.DataFrame,
    feature_name: str,
):
    """
    Generate synthetic data for the full range of `X_train[feature_name]` and
    estimate rain probabilities.
    """
    # Here we create a synthetic data of all values to see how the model behave
    # Here we add all the X columns to synthetic_data, so the pipeline can run
    synthetic_data = pd.DataFrame(index=range(0,1000))
    for column in X_train.columns:
        values = 0
        if column == feature_name:
            values = np.linspace(
                X_train[feature_name].min(),
                X_train[feature_name].max(),
                1000
            ).reshape(-1,1)
        synthetic_data[column] = values
    # In this object we store the estimated raining probability for each value
    raining_prob = complete_pipeline.predict_proba(synthetic_data)[:,1]
    raining_prob = pd.Series(raining_prob, index = synthetic_data[feature_name])
    
    # Find the global maximum (from the column with the highest max)
    synthetic_splines = complete_pipeline[0].transform(synthetic_data)
    features_data = synthetic_splines*raining_prob.values.reshape(-1,1)
    features_data = pd.DataFrame(features_data)
    max_value = features_data.max().max()
    
    # Apply Min-Max Scaling using only this max value
    features_data = (features_data / max_value) * raining_prob.values.max()
    features_data = features_data.set_index(synthetic_data[feature_name])
    return features_data, raining_prob


def plot_features_vs_prob(
    features_data: np.array,
    raining_prob: np.array,
    axs: matplotlib.axes._axes.Axes
) -> None:
    """Plot estimated rain probability alongside spline features."""
    axs.plot(pd.Series(raining_prob), color='green')
    axs.plot(pd.DataFrame(features_data), alpha=0.5)


def plot_prob_vs_target(
    raining_prob: np.array,
    raining_occurence: np.array,
    axs: matplotlib.axes._axes.Axes
) -> None:
    """Compare estimated rain probability with actual rainy occurences."""
    axs_prob = axs.twinx()
    axs_prob.plot(raining_prob, color='green', alpha=0.8)
    axs.plot(raining_occurence, color='blue', alpha=0.8)
    # Set y-tick colors
    axs.yaxis.label.set_color('blue')
    axs.tick_params(axis='y', colors='blue')
    axs_prob.yaxis.label.set_color('green')
    axs_prob.tick_params(axis='y', colors='green')


def plot_features_vs_target(
    features_data,
    rolling_target,
    axs: matplotlib.axes._axes.Axes
) -> None:
    """Visualize spline features in relation to actual rainy occurences."""
    axs_features = axs.twinx()
    axs.plot(rolling_target, color='blue')
    axs.plot(pd.DataFrame(features_data), alpha=0.5)
    # Set y-tick colors
    axs.yaxis.label.set_color('blue')
    axs.tick_params(axis='y', colors='blue')
    axs_features.yaxis.label.set_color('green')
    axs_features.tick_params(axis='y', colors='green')


def plot_model_fitting(
    complete_pipeline: Pipeline,
    rolling_target: np.array,
    feature_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> plt.Figure:
    """Generate subplots comparing splines, estimated probabilities, and actual data."""
    complete_pipeline.fit(X_train, np.array(y_train).ravel())
    features_data, raining_prob = estimate_raining_prob(
        complete_pipeline,
        X_train,
        feature_name
    )
    fig, axs = plt.subplots(
        ncols=3,
        figsize=(13,5),
        gridspec_kw={'wspace':0.3}
    )
    axs[0].set_title("Splines vs Estimated Raining Prob")
    plot_features_vs_prob(
        features_data,
        raining_prob,
        axs[0]
    )
    axs[1].set_title("Estimated Raining Prob vs Rolling Target")
    plot_prob_vs_target(
        raining_prob,
        rolling_target,
        axs[1]
    )
    axs[2].set_title("Splines vs Rolling Target")    
    plot_features_vs_target(
        features_data,
        rolling_target,
        axs[2]
    )

    score, _ = get_model_score(X_test, y_test, complete_pipeline)
    fig.suptitle(f"Estimated AUC: {score:.4f}")

# This plot is inspired by pygam (https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html)
plot_model_fitting(
    complete_pipeline,
    rolling_raining_days,
    'day',
    X_train,
    y_train,
    X_valid,
    y_valid
)


from typing import Callable, Tuple, Dict
from tqdm import tqdm

def optimize_hyperparameter(
    make_preprocessing: Callable[..., Pipeline],
    make_complete_pipeline: Callable[..., Pipeline],
    get_model_score: Callable[..., Tuple[float, np.array]],
    preprocessing_params: Dict,
    hyperparameter_name: str,
    hyperparameter_values: np.array # 
) -> Tuple[float, float]:
    """Finds the optimal hyperparameter value by maximizing the ROC AUC score on the validation set."""
    l_scores = np.zeros(len(hyperparameter_values))
    for i, value in tqdm(enumerate(hyperparameter_values), total=len(hyperparameter_values)):
        preprocessing = make_preprocessing(**{**preprocessing_params, hyperparameter_name: value})
        complete_pipeline = make_complete_pipeline(preprocessing)
        complete_pipeline.fit(
            X_train, 
            np.array(y_train).ravel()
        )
        score, _ = get_model_score(
            X_valid,
            y_valid,
            complete_pipeline
        )
        l_scores[i] = score
    max_score = np.max(l_scores)
    argmax_score = np.argmax(l_scores)
    optimal_hyper = hyperparameter_values[argmax_score]
    return optimal_hyper, max_score

# Perform optimization
optimal_hyper, best_auc = optimize_hyperparameter(
    splines_pipeline,
    make_complete_pipeline,
    get_model_score,
    preprocessing_params = {'feature_name':'day'},
    hyperparameter_name = "n_knots",
    hyperparameter_values = np.linspace(5,100,96).astype(int)
)

print(f"Optimal n_knots: {optimal_hyper}, Best AUC: {best_auc:.4f}")

### Optimal Model Performance
preprocessing = splines_pipeline(
    feature_name = 'day',
    n_knots = optimal_hyper
)
complete_pipeline = make_complete_pipeline(preprocessing)
complete_pipeline.fit(
    X_train, 
    np.array(y_train).ravel()
)

# Visualize results
plot_model_fitting(
    complete_pipeline,
    rolling_target('day', int(365/optimal_hyper)),
    'day',
    X_train,
    y_train,
    X_valid,
    y_valid
)


# Optimize n_knots for each numerical feature
optimal_hypers = {}
for feature in X.columns:
    optimal_hyper, _ = optimize_hyperparameter(
        splines_pipeline,
        make_complete_pipeline,
        get_model_score,
        preprocessing_params = {'feature_name':feature},
        hyperparameter_name = "n_knots",
        hyperparameter_values = np.arange(2, 102)
    )
    optimal_hypers[feature] = optimal_hyper

# Display the optimized number of knots for each feature
pd.DataFrame.from_dict(optimal_hypers, orient='index', columns=['Optimal n_knots'])


# Choose a feature to visualize (change this to analyze different variables)
selected_feature = 'pressure'  # Example: 'humidity', 'pressure', 'temperature', 'cloud', etc.
rolling_window = 10

# Train the model with the selected feature
preprocessing = splines_pipeline(
    feature_name = feature,
    n_knots = optimal_hypers[feature]
)
complete_pipeline = make_complete_pipeline(preprocessing)
complete_pipeline.fit(
    X_train, 
    np.array(y_train).ravel()
)

# Visualize results
plot_model_fitting(
    complete_pipeline,
    rolling_target(feature, rolling_window),
    feature,
    X_train,
    y_train,
    X_valid,
    y_valid
)


# Define the complete preprocessing pipeline
complete_preprocessing_pipeline = ColumnTransformer(
    [(feature, SplineTransformer(n_knots=optimal_hypers[feature]), [feature]) for feature in X.columns]
)
# Create and train the final model
complete_pipeline = make_complete_pipeline(complete_preprocessing_pipeline)
# Train using both training and validation data to maximize data usage
X_full_train = pd.concat([X_train, X_valid])
y_full_train = np.concatenate([y_train, y_valid])

complete_pipeline.fit(
    pd.concat([X_train,X_valid]), 
    np.array(pd.concat([y_train,y_valid])).ravel()
)


from sklearn.metrics import RocCurveDisplay, roc_curve, auc

# Compute ROC AUC score on the test set
score, y_prob = get_model_score(
    X_test,
    y_test,
    complete_pipeline
)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
display = RocCurveDisplay.from_estimator(
    complete_pipeline,
    X_test, y_test
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.show()


X_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col=0)
X_test.fillna(X_train.median(), inplace=True)
complete_pipeline.fit(
    X, 
    np.array(y).ravel()
)
y_prob = complete_pipeline.predict_proba(X_test)[:,1]

submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = y_prob
submission.to_csv("submission.csv", index=False)

