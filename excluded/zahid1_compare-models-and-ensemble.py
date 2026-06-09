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


import random
random.seed(42)  # Replace 42 with your desired seed value


! pip install torch_geometric


from file_imports import (
    HistoricalMultiRoutePlayDataset,
    create_game_play_pairs,
    create_batch_data,
    process_game_scores,
    DataMappings,
    filter_by_game_play_ids,
    create_week_stratified_split,
    load_and_filter_data, 
    filter_passing_plays_only,
    train_route_prediction_pipeline, 
    prepare_route_prediction_data,
    train_route_predictor, 
    MultiRouteLoss,
)
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import pandas as pd


import torch
import numpy as np
from sklearn.metrics import classification_report, top_k_accuracy_score
import pandas as pd
import torch
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy
from IPython.display import display, HTML

def get_xgb_preds(xgb_model, plays_df, batch, feature_encoders, scaler):

    # Get corresponding plays for XGBoost
    ids = list(
        zip(
            batch.game_id[batch.eligible_mask].cpu().tolist(),
            batch.play_id[batch.eligible_mask].cpu().tolist(),
            batch.player_ids[batch.eligible_mask].cpu().tolist(),
        )
    )
    batch_df = filter_by_game_play_ids(plays_df, ids)

    # XGB predictions
    X, y, _, _2 = prepare_route_prediction_data(
        batch_df,
        training=False,
        feature_encoders=feature_encoders,
        scaler=scaler,
    )
    xgb_probs = xgb_model.predict_proba(X)
    return xgb_probs

def evaluate_route_predictions_table(
    model,
    dataloader_,
    dataset_,
    device="cuda",
    entropy_ceiling=3,
    key="route_predictions",
    softmax=False,
    plays_df=None,
    feature_encoders=None, 
    scaler=None
):
    """
    Evaluates a PyTorch route prediction model with enhanced metrics and formatted table output.
    Now includes play-level and field position metrics.

    Parameters:
    - model: PyTorch model
    - dataloader: PyTorch DataLoader for evaluation
    - dataset: MultiRoutePlayDataset instance
    - device: Device to run evaluation on
    """
    try:
        model.eval()
    except Exception as e:
        print('Trying XGB')
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_metadata = []  # New list to store metadata

    with torch.no_grad():
        for batch in dataloader_:
            try:
                batch = batch.to(device)
                targets = batch.route_targets[batch.eligible_mask]
                try:
                    output = model(batch)
                    probabilities = output[key]
                    
                except Exception as e:
                    probabilities = get_xgb_preds(model, plays_df, batch, feature_encoders, scaler)
                    output = {}

                if "target" in output.keys():
                    torch.testing.assert_close(targets, torch.tensor(output["target"]).to(device))

                if isinstance(probabilities, torch.Tensor):
                    if softmax:
                        probabilities = torch.softmax(probabilities, dim=1)
                    predictions = probabilities.argmax(dim=1).cpu().numpy()
                    probabilities = probabilities.cpu().numpy()
                else:
                    predictions = torch.tensor(probabilities).argmax(dim=1).cpu().numpy()

                # Extract metadata for eligible players
                metadata = {
                    'play_id': batch.play_id[batch.eligible_mask].cpu().numpy(),
                    'game_id': batch.game_id[batch.eligible_mask].cpu().numpy(),
                    'player_id': batch.player_ids[batch.eligible_mask].cpu().numpy(),
                    'yardline': batch.yardline[batch.eligible_mask].cpu().numpy(),
                    'down': batch.down[batch.eligible_mask].cpu().numpy() if hasattr(batch, 'down') else None,
                    'yards_to_go': batch.yards_to_go[batch.eligible_mask].cpu().numpy() if hasattr(batch, 'yards_to_go') else None
                }
                
                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_targets.append(targets.cpu().numpy())
                all_metadata.append(metadata)
            except Exception as e:
                raise e

    try:
        y_pred = np.concatenate(all_predictions)
        y_pred_proba = np.concatenate(all_probabilities)
        y_test = np.concatenate(all_targets)
        
        # Concatenate metadata
        combined_metadata = {
            key: np.concatenate([batch[key] for batch in all_metadata]) 
            for key in all_metadata[0].keys()
            if all_metadata[0][key] is not None
        }
        
    except Exception as e:
        print(all_predictions)
        raise e

    # Apply entropy filtering
    prediction_entropies = np.apply_along_axis(entropy, 1, y_pred_proba)
    entropy_subset = prediction_entropies < entropy_ceiling

    y_pred = y_pred[entropy_subset]
    y_pred_proba = y_pred_proba[entropy_subset]
    y_test = y_test[entropy_subset]
    
    # Also filter metadata
    filtered_metadata = {
        key: value[entropy_subset] 
        for key, value in combined_metadata.items()
    }

    # Create DataFrame with all predictions and metadata
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'probabilities': y_pred_proba.tolist(),
        'predicted': y_pred,
        'entropy': prediction_entropies[entropy_subset],
        **filtered_metadata
    })
    
    # Convert route indices to names
    predictions_df['actual_route'] = predictions_df['actual'].map(dataset_.idx_to_route)
    predictions_df['predicted_route'] = predictions_df['predicted'].map(dataset_.idx_to_route)

    # Calculate base metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)
    actuals_predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    # Calculate actual vs predicted counts
    for class_label in np.unique(y_test):
        actual_count = len(
            actuals_predictions[actuals_predictions["Actual"] == class_label]
        )
        predicted_count = len(
            actuals_predictions[actuals_predictions["Predicted"] == class_label]
        )
        metrics[str(class_label)]["actual_count"] = actual_count
        metrics[str(class_label)]["predicted_count"] = predicted_count

    # Calculate uncertainty metrics
    def calculate_uncertainty_metrics(probabilities, y_true):
        prediction_entropies = np.apply_along_axis(entropy, 1, probabilities)
        true_label_probs = np.array(
            [prob[true] for prob, true in zip(probabilities, y_true)]
        )
        log_likelihood = np.log(true_label_probs + 1e-10)
        max_entropy = np.log2(probabilities.shape[1])
        normalized_entropies = prediction_entropies / max_entropy
        sorted_probs = np.sort(probabilities, axis=1)
        entropy_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        return {
            "mean_entropy": np.mean(prediction_entropies),
            "median_entropy": np.median(prediction_entropies),
            "mean_normalized_entropy": np.mean(normalized_entropies),
            "mean_entropy_margin": np.mean(entropy_margin),
            "mean_log_likelihood": np.mean(log_likelihood),
            "median_log_likelihood": np.median(log_likelihood),
        }

    metrics["uncertainty_metrics"] = calculate_uncertainty_metrics(y_pred_proba, y_test)

    # # Calculate Top-K accuracy
    # def top_k_accuracy(y_true, y_pred_proba, k):
    #     top_k_predictions = np.argsort(y_pred_proba, axis=1)[:, -k:]

    #     correct = 0
    #     for i, true_label in enumerate(y_true):
    #         if true_label in top_k_predictions[i]:
    #             correct += 1
    #     return correct / len(y_true)

    metrics["top_2_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=2, labels=list(range(0, 13))
    )
    metrics["top_3_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=3, labels=list(range(0, 13))
    )

    # Add predictions DataFrame to metrics
    metrics['predictions_df'] = predictions_df
    
    # Add some basic grouped metrics
    def calculate_grouped_metrics(df):
        play_metrics = {}
        
        # Redzone analysis (inside 20 yard line)
        redzone_mask = df['yardline'] <= 20
        play_metrics['redzone'] = {
            'accuracy': (df[redzone_mask]['actual'] == df[redzone_mask]['predicted']).mean(),
            'sample_size': redzone_mask.sum(),
            'route_distribution': df[redzone_mask]['actual_route'].value_counts().to_dict()
        }
        
        # Play-level accuracy
        play_level_accuracy = df.groupby(['play_id', 'game_id']).apply(
            lambda x: (x['actual'] == x['predicted']).mean()
        ).describe().to_dict()
        play_metrics['play_level'] = play_level_accuracy
        
        # Most common route combinations
        play_metrics['route_combinations'] = (
            df.groupby(['play_id', 'game_id'])['actual_route']
            .agg(lambda x: tuple(sorted(x)))  # Convert to sorted tuple for consistent ordering
            .value_counts()
            .head(10)
            .to_dict()
        )
        
        return play_metrics
    
    def display_formatted_metrics(metrics_dict, dataset_):
        # Per-class metrics table
        class_metrics = []
        for class_label in sorted([k for k in metrics_dict.keys() if k.isdigit()]):
            try:
                class_metrics.append(
                    {
                        "Route": dataset_.idx_to_route[int(class_label)],
                        "Actual Count": metrics_dict[class_label]["actual_count"],
                        "Predicted Count": metrics_dict[class_label]["predicted_count"],
                        "Precision": f"{metrics_dict[class_label]['precision']:.3f}",
                        "Recall": f"{metrics_dict[class_label]['recall']:.3f}",
                        "F1-score": f"{metrics_dict[class_label]['f1-score']:.3f}",
                        "Support": metrics_dict[class_label]["support"],
                    }
                )
            except Exception as e:
                print(f"Could not get metrics for {class_label}")

        class_df = pd.DataFrame(class_metrics)

        # Top-K accuracy table
        topk_metrics = pd.DataFrame(
            [
                {
                    "Metric": "Accuracy Type",
                    "Top-1 (standard)": f"{metrics_dict['accuracy']:.3f}",
                    "Top-2": f"{metrics_dict['top_2_accuracy']:.3f}",
                    "Top-3": f"{metrics_dict['top_3_accuracy']:.3f}",
                }
            ]
        )

        # Uncertainty metrics table
        uncertainty_df = pd.DataFrame(
            [
                {
                    "Mean Entropy": f"{metrics_dict['uncertainty_metrics']['mean_entropy']:.3f}",
                    "Median Entropy": f"{metrics_dict['uncertainty_metrics']['median_entropy']:.3f}",
                    "Mean Normalized Entropy": f"{metrics_dict['uncertainty_metrics']['mean_normalized_entropy']:.3f}",
                    "Mean Entropy Margin": f"{metrics_dict['uncertainty_metrics']['mean_entropy_margin']:.3f}",
                    "Mean Log Likelihood": f"{metrics_dict['uncertainty_metrics']['mean_log_likelihood']:.3f}",
                    "Median Log Likelihood": f"{metrics_dict['uncertainty_metrics']['median_log_likelihood']:.3f}",
                }
            ]
        )

        # Overall metrics table
        overall_metrics = []
        for avg_type in ["macro avg", "weighted avg"]:
            metrics_row = {"Average Type": avg_type}
            metrics_row.update(
                {k: f"{v:.3f}" for k, v in metrics_dict[avg_type].items()}
            )
            overall_metrics.append(metrics_row)
        overall_df = pd.DataFrame(overall_metrics)

        # Display tables with styling
        print("\n=== Per-Class Performance ===")
        display(
            HTML(
                class_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Top-K Accuracy ===")
        display(
            HTML(
                topk_metrics.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Uncertainty Metrics ===")
        display(
            HTML(
                uncertainty_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Overall Metrics ===")
        display(
            HTML(
                overall_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

    metrics["display_tables"] = lambda: display_formatted_metrics(metrics, dataset_)

    metrics['grouped_metrics'] = calculate_grouped_metrics(predictions_df)
    
    return metrics, (y_test, y_pred, y_pred_proba, predictions_df)


def evaluate_route_predictions_table_old(
    model,
    dataloader_,
    dataset_,
    device="cuda",
    entropy_ceiling=3,
    key="route_predictions",
    softmax=False,
):
    """
    Evaluates a PyTorch route prediction model with enhanced metrics and formatted table output

    Parameters:
    - model: PyTorch model
    - dataloader: PyTorch DataLoader for evaluation
    - dataset: MultiRoutePlayDataset instance
    - device: Device to run evaluation on
    """
    # [Previous code for model evaluation remains the same until formatting section]
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader_:
            try:
                # print(batch)
                batch = batch.to(device)
                output = model(batch)
                probabilities = output[key]

                targets = batch.route_targets[batch.eligible_mask]

                if "target" in output.keys():
                    torch.testing.assert_close(targets, torch.tensor(output["target"]))

                if isinstance(probabilities, torch.Tensor):
                    if softmax:
                        probabilities = torch.softmax(probabilities, dim=1)
                    predictions = probabilities.argmax(dim=1).cpu().numpy()
                    probabilities = probabilities.cpu().numpy()
                else:
                    predictions = (
                        torch.tensor(probabilities).argmax(dim=1).cpu().numpy()
                    )

                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_targets.append(targets.cpu().numpy())
            except Exception as e:
                raise e
    try:
        y_pred = np.concatenate(all_predictions)
        y_pred_proba = np.concatenate(all_probabilities)
        y_test = np.concatenate(all_targets)
    except Exception as e:
        print(all_predictions)
        raise e

    prediction_entropies = np.apply_along_axis(entropy, 1, y_pred_proba)
    entropy_subset = prediction_entropies < entropy_ceiling

    y_pred = y_pred[entropy_subset]
    y_pred_proba = y_pred_proba[entropy_subset]
    y_test = y_test[entropy_subset]

    metrics = classification_report(y_test, y_pred, output_dict=True)
    actuals_predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    # Calculate actual vs predicted counts
    for class_label in np.unique(y_test):
        actual_count = len(
            actuals_predictions[actuals_predictions["Actual"] == class_label]
        )
        predicted_count = len(
            actuals_predictions[actuals_predictions["Predicted"] == class_label]
        )
        metrics[str(class_label)]["actual_count"] = actual_count
        metrics[str(class_label)]["predicted_count"] = predicted_count

    # Calculate uncertainty metrics
    def calculate_uncertainty_metrics(probabilities, y_true):
        prediction_entropies = np.apply_along_axis(entropy, 1, probabilities)
        true_label_probs = np.array(
            [prob[true] for prob, true in zip(probabilities, y_true)]
        )
        log_likelihood = np.log(true_label_probs + 1e-10)
        max_entropy = np.log2(probabilities.shape[1])
        normalized_entropies = prediction_entropies / max_entropy
        sorted_probs = np.sort(probabilities, axis=1)
        entropy_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        return {
            "mean_entropy": np.mean(prediction_entropies),
            "median_entropy": np.median(prediction_entropies),
            "mean_normalized_entropy": np.mean(normalized_entropies),
            "mean_entropy_margin": np.mean(entropy_margin),
            "mean_log_likelihood": np.mean(log_likelihood),
            "median_log_likelihood": np.median(log_likelihood),
        }

    metrics["uncertainty_metrics"] = calculate_uncertainty_metrics(y_pred_proba, y_test)

    # # Calculate Top-K accuracy
    # def top_k_accuracy(y_true, y_pred_proba, k):
    #     top_k_predictions = np.argsort(y_pred_proba, axis=1)[:, -k:]

    #     correct = 0
    #     for i, true_label in enumerate(y_true):
    #         if true_label in top_k_predictions[i]:
    #             correct += 1
    #     return correct / len(y_true)

    metrics["top_2_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=2, labels=list(range(0, 13))
    )
    metrics["top_3_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=3, labels=list(range(0, 13))
    )

    # New formatted output function using pandas DataFrames
    def display_formatted_metrics(metrics_dict, dataset_):
        # Per-class metrics table
        class_metrics = []
        for class_label in sorted([k for k in metrics_dict.keys() if k.isdigit()]):
            try:
                class_metrics.append(
                    {
                        "Route": dataset_.idx_to_route[int(class_label)],
                        "Actual Count": metrics_dict[class_label]["actual_count"],
                        "Predicted Count": metrics_dict[class_label]["predicted_count"],
                        "Precision": f"{metrics_dict[class_label]['precision']:.3f}",
                        "Recall": f"{metrics_dict[class_label]['recall']:.3f}",
                        "F1-score": f"{metrics_dict[class_label]['f1-score']:.3f}",
                        "Support": metrics_dict[class_label]["support"],
                    }
                )
            except Exception as e:
                print(f"Could not get metrics for {class_label}")

        class_df = pd.DataFrame(class_metrics)

        # Top-K accuracy table
        topk_metrics = pd.DataFrame(
            [
                {
                    "Metric": "Accuracy Type",
                    "Top-1 (standard)": f"{metrics_dict['accuracy']:.3f}",
                    "Top-2": f"{metrics_dict['top_2_accuracy']:.3f}",
                    "Top-3": f"{metrics_dict['top_3_accuracy']:.3f}",
                }
            ]
        )

        # Uncertainty metrics table
        uncertainty_df = pd.DataFrame(
            [
                {
                    "Mean Entropy": f"{metrics_dict['uncertainty_metrics']['mean_entropy']:.3f}",
                    "Median Entropy": f"{metrics_dict['uncertainty_metrics']['median_entropy']:.3f}",
                    "Mean Normalized Entropy": f"{metrics_dict['uncertainty_metrics']['mean_normalized_entropy']:.3f}",
                    "Mean Entropy Margin": f"{metrics_dict['uncertainty_metrics']['mean_entropy_margin']:.3f}",
                    "Mean Log Likelihood": f"{metrics_dict['uncertainty_metrics']['mean_log_likelihood']:.3f}",
                    "Median Log Likelihood": f"{metrics_dict['uncertainty_metrics']['median_log_likelihood']:.3f}",
                }
            ]
        )

        # Overall metrics table
        overall_metrics = []
        for avg_type in ["macro avg", "weighted avg"]:
            metrics_row = {"Average Type": avg_type}
            metrics_row.update(
                {k: f"{v:.3f}" for k, v in metrics_dict[avg_type].items()}
            )
            overall_metrics.append(metrics_row)
        overall_df = pd.DataFrame(overall_metrics)

        # Display tables with styling
        print("\n=== Per-Class Performance ===")
        display(
            HTML(
                class_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Top-K Accuracy ===")
        display(
            HTML(
                topk_metrics.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Uncertainty Metrics ===")
        display(
            HTML(
                uncertainty_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Overall Metrics ===")
        display(
            HTML(
                overall_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

    metrics["display_tables"] = lambda: display_formatted_metrics(metrics, dataset_)
    metrics["prediction_entropies"] = prediction_entropies

    return metrics, (y_test, y_pred, y_pred_proba)


# Usage example:
"""
# Create test dataloader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate model
metrics, (y_test, y_pred, y_pred_proba) = evaluate_route_predictions(
    model=model,
    dataloader=test_loader,
    dataset=test_dataset,
    device='cuda'
)

# Display formatted tables
metrics['display_tables']()
"""


def evaluate_route_predictions(model, dataloader_, dataset_, device="cuda"):
    """
    Evaluates a PyTorch route prediction model with enhanced metrics

    Parameters:
    - model: PyTorch model
    - dataloader: PyTorch DataLoader for evaluation
    - dataset: MultiRoutePlayDataset instance
    - device: Device to run evaluation on
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    prev = 0

    with torch.no_grad():
        for batch in dataloader_:
            batch = batch.to(device)
            output = model(batch)
            predictions = output["route_predictions"]

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(predictions, dim=1)

            # Print average time for this batch
            batch_time = (
                batch.time.float().mean().item()
            )  # Assuming 'time' is an attribute in your batch
            batch_min = batch.time.float().min().item()
            batch_max = batch.time.float().max().item()
            print(
                f"Batch average time diff from prev: {batch_time - prev:.3f}, batch min: {batch_min}, batch_max: {batch_max}"
            )
            prev = batch_time

            # Get targets for eligible receivers
            targets = batch.route_targets[batch.eligible_mask]

            # Move to CPU and convert to numpy
            all_predictions.append(predictions.argmax(dim=1).cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_predictions)
    y_pred_proba = np.concatenate(all_probabilities)
    y_test = np.concatenate(all_targets)

    # Calculate basic metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)

    # Calculate actual vs predicted counts
    actuals_predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    for class_label in np.unique(y_test):
        actual_count = len(
            actuals_predictions[actuals_predictions["Actual"] == class_label]
        )
        predicted_count = len(
            actuals_predictions[actuals_predictions["Predicted"] == class_label]
        )
        metrics[str(class_label)]["actual_count"] = actual_count
        metrics[str(class_label)]["predicted_count"] = predicted_count

    # Calculate uncertainty metrics
    def calculate_uncertainty_metrics(probabilities, y_true):
        # Calculate entropy for each prediction
        prediction_entropies = np.apply_along_axis(entropy, 1, probabilities)

        # Calculate log likelihood of true labels
        true_label_probs = np.array(
            [prob[true] for prob, true in zip(probabilities, y_true)]
        )
        log_likelihood = np.log(
            true_label_probs + 1e-10
        )  # Add small epsilon to prevent log(0)

        # Calculate normalized entropy
        max_entropy = np.log2(probabilities.shape[1])
        normalized_entropies = prediction_entropies / max_entropy

        # Calculate entropy margin
        sorted_probs = np.sort(probabilities, axis=1)
        entropy_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        return {
            "mean_entropy": np.mean(prediction_entropies),
            "median_entropy": np.median(prediction_entropies),
            "mean_normalized_entropy": np.mean(normalized_entropies),
            "mean_entropy_margin": np.mean(entropy_margin),
            "mean_log_likelihood": np.mean(log_likelihood),
            "median_log_likelihood": np.median(log_likelihood),
        }

    metrics["uncertainty_metrics"] = calculate_uncertainty_metrics(y_pred_proba, y_test)

    # Calculate Top-K accuracy
    def top_k_accuracy(y_true, y_pred_proba, k):
        top_k_predictions = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_predictions[i]:
                correct += 1
        return correct / len(y_true)

    metrics["top_2_accuracy"] = top_k_accuracy(y_test, y_pred_proba, 2)
    metrics["top_3_accuracy"] = top_k_accuracy(y_test, y_pred_proba, 3)

    # Enhanced formatting function using dataset's route mapping
    def format_metrics(metrics_dict, dataset_):
        formatted_report = "\nClassification Report:\n-------------------"

        # Print individual class metrics
        for class_label in sorted(
            [
                k
                for k in metrics_dict.keys()
                if k
                not in [
                    "uncertainty_metrics",
                    "accuracy",
                    "macro avg",
                    "weighted avg",
                    "top_2_accuracy",
                    "top_3_accuracy",
                    "confidence_metrics",
                ]
            ]
        ):

            try:
                if class_label.isdigit():
                    class_metrics = metrics_dict[class_label]
                    route_name = dataset_.idx_to_route[int(class_label)]
                    formatted_report += f"\n\nClass {route_name}:\n"
                    formatted_report += (
                        f"    Actual count: {class_metrics['actual_count']}\n"
                    )
                    formatted_report += (
                        f"    Predicted count: {class_metrics['predicted_count']}\n"
                    )
                    formatted_report += (
                        f"    Precision: {class_metrics['precision']:.3f}\n"
                    )
                    formatted_report += f"    Recall: {class_metrics['recall']:.3f}\n"
                    formatted_report += (
                        f"    F1-score: {class_metrics['f1-score']:.3f}\n"
                    )
                    formatted_report += f"    Support: {class_metrics['support']}"
            except Exception as e:
                print(f"Could not get metrics for {class_label}")

        # Print Top-K metrics
        formatted_report += "\n\nTop-K Accuracy:\n--------------"
        formatted_report += (
            f"\nTop-1 (standard) accuracy: {metrics_dict['accuracy']:.3f}"
        )
        formatted_report += f"\nTop-2 accuracy: {metrics_dict['top_2_accuracy']:.3f}"
        formatted_report += f"\nTop-3 accuracy: {metrics_dict['top_3_accuracy']:.3f}"

        # Print uncertainty metrics
        uncertainty_metrics = metrics_dict["uncertainty_metrics"]
        formatted_report += "\n\nUncertainty Metrics:\n-------------------"
        formatted_report += (
            f"\nMean prediction entropy: {uncertainty_metrics['mean_entropy']:.3f}"
        )
        formatted_report += (
            f"\nMedian prediction entropy: {uncertainty_metrics['median_entropy']:.3f}"
        )
        formatted_report += f"\nMean normalized entropy: {uncertainty_metrics['mean_normalized_entropy']:.3f}"
        formatted_report += (
            f"\nMean entropy margin: {uncertainty_metrics['mean_entropy_margin']:.3f}"
        )
        formatted_report += (
            f"\nMean log likelihood: {uncertainty_metrics['mean_log_likelihood']:.3f}"
        )
        formatted_report += f"\nMedian log likelihood: {uncertainty_metrics['median_log_likelihood']:.3f}"

        # Print overall metrics
        formatted_report += "\n\nOverall Metrics:\n---------------"
        for avg_type in ["macro avg", "weighted avg"]:
            formatted_report += f"\n{avg_type}:"
            for metric, value in metrics_dict[avg_type].items():
                formatted_report += f"\n    {metric}: {value:.3f}"

        return formatted_report

    metrics["format_report"] = lambda: format_metrics(metrics, dataset_)

    return metrics, (y_test, y_pred, y_pred_proba)


# Usage example:
"""
# Create test dataloader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate model
metrics, (y_test, y_pred, y_pred_proba) = evaluate_route_predictions(
    model=model,
    dataloader=test_loader,
    dataset=test_dataset,
    device='cuda'
)

# Print formatted report
print(metrics['format_report']())
"""


def analyze_distributional_drift(train_loader, test_loader, num_batches=None):
    """
    Analyze distributional drift between train and test sets.

    Parameters:
    - train_loader: Training data loader
    - test_loader: Test data loader
    - num_batches: Number of batches to analyze (None for all)
    """
    # Initialize collectors
    train_stats = defaultdict(list)
    test_stats = defaultdict(list)

    def collect_batch_stats(batch, stats_dict):
        # Node features
        stats_dict["node_features"].extend(batch.x.cpu().numpy())

        # Edge attributes
        stats_dict["edge_attr"].extend(batch.edge_attr.cpu().numpy())

        # Game state features
        stats_dict["down"].extend(batch.down.cpu().numpy())
        stats_dict["distance"].extend(batch.distance.cpu().numpy())
        stats_dict["quarter"].extend(batch.quarter.cpu().numpy())
        stats_dict["offense_team"].extend(batch.offense_team.cpu().numpy())

        # Graph structure features
        unique_batches = torch.unique(batch.batch)
        nodes_per_graph = [torch.sum(batch.batch == i).item() for i in unique_batches]
        edges_per_graph = [
            len(batch.edge_index[0][batch.batch[batch.edge_index[0]] == i])
            for i in unique_batches
        ]

        stats_dict["nodes_per_graph"].extend(nodes_per_graph)
        stats_dict["edges_per_graph"].extend(edges_per_graph)

        # Target distribution (only for eligible players)
        stats_dict["targets"].extend(batch.route_targets.cpu().numpy())

    # Collect statistics
    print("Collecting training set statistics...")
    for i, batch in enumerate(train_loader):
        if num_batches and i >= num_batches:
            break
        collect_batch_stats(batch, train_stats)

    print("Collecting test set statistics...")
    for i, batch in enumerate(test_loader):
        if num_batches and i >= num_batches:
            break
        collect_batch_stats(batch, test_stats)

    # Convert to numpy arrays
    for key in train_stats:
        train_stats[key] = np.array(train_stats[key])
        test_stats[key] = np.array(test_stats[key])

    # Analyze drift
    def plot_distribution_comparison(train_data, test_data, feature_name, bins=50):
        plt.figure(figsize=(10, 6))

        # Calculate histogram parameters
        min_val = min(train_data.min(), test_data.min())
        max_val = max(train_data.max(), test_data.max())

        plt.hist(
            train_data,
            bins=bins,
            alpha=0.5,
            label="Train",
            density=True,
            range=(min_val, max_val),
        )
        plt.hist(
            test_data,
            bins=bins,
            alpha=0.5,
            label="Test",
            density=True,
            range=(min_val, max_val),
        )

        plt.title(f"Distribution Comparison: {feature_name}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

        # Perform KS test
        ks_statistic, p_value = ks_2samp(train_data, test_data)
        plt.text(
            0.05,
            0.95,
            f"KS statistic: {ks_statistic:.3f}\np-value: {p_value:.3e}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.show()

    # Plot distributions for various features
    print("\nAnalyzing distributional drift...")

    # Node features
    for i in range(train_stats["node_features"][0].shape[0]):
        plot_distribution_comparison(
            train_stats["node_features"][:, i],
            test_stats["node_features"][:, i],
            f"Node Feature {i}",
        )

    # Edge attributes
    for i in range(train_stats["edge_attr"][0].shape[0]):
        plot_distribution_comparison(
            train_stats["edge_attr"][:, i],
            test_stats["edge_attr"][:, i],
            f"Edge Attribute {i}",
        )

    # Game state features
    for feature in ["down", "distance", "quarter"]:
        plot_distribution_comparison(
            train_stats[feature], test_stats[feature], feature.capitalize()
        )

    # Graph structure features
    for feature in ["nodes_per_graph", "edges_per_graph"]:
        plot_distribution_comparison(
            train_stats[feature], test_stats[feature], feature.replace("_", " ").title()
        )

    # Target distribution
    plot_distribution_comparison(
        train_stats["targets"], test_stats["targets"], "Route Targets"
    )

    # Print summary statistics
    print("\nSummary Statistics:")
    for key in train_stats:
        print(f"\n{key.replace('_', ' ').title()}:")
        print(
            f"Train - Mean: {train_stats[key].mean():.3f}, Std: {train_stats[key].std():.3f}"
        )
        print(
            f"Test  - Mean: {test_stats[key].mean():.3f}, Std: {test_stats[key].std():.3f}"
        )


# Usage:
# analyze_distributional_drift(train_loader, test_loader)

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Callable
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import entropy


def plot_metrics_across_datasets(
    evaluation_results: List[Tuple[Dict, Tuple]],
    dataset_names: List[str],
    metrics_to_plot: List[str] = ["accuracy", "weighted avg_f1-score"],
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plots specified metrics across multiple datasets.

    Parameters:
    - evaluation_results: List of (metrics, data_tuple) from evaluate_route_predictions
    - dataset_names: Names/identifiers for each dataset
    - metrics_to_plot: List of metrics to visualize
    - figsize: Figure size for the plot
    """
    plt.figure(figsize=figsize)

    metrics_data = defaultdict(list)

    for metrics, _ in evaluation_results:
        for metric in metrics_to_plot:
            if "_" in metric:  # Handle nested metrics like 'weighted avg_f1-score'
                category, submetric = metric.split("_")
                value = metrics[category][submetric]
            else:
                value = metrics[metric]
            metrics_data[metric].append(value)

    x = np.arange(len(dataset_names))
    width = 0.8 / len(metrics_to_plot)

    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i * width, metrics_data[metric], width, label=metric)

    plt.xlabel("Datasets")
    plt.ylabel("Score")
    plt.title("Metrics Comparison Across Datasets")
    plt.xticks(x + width * (len(metrics_to_plot) - 1) / 2, dataset_names)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_performance(
    metrics: Dict,
    data_tuple: Tuple,
    window_size: int = 100,
    stride: int = None,
    metrics_to_plot: List[str] = ["accuracy"],
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plots rolling window performance metrics over an ordered dataset.

    Parameters:
    - metrics: Metrics dictionary from evaluate_route_predictions
    - data_tuple: (y_test, y_pred, y_pred_proba) tuple from evaluate_route_predictions
    - window_size: Size of the rolling window
    - stride: Number of samples to move the window forward (defaults to window_size//4)
    - metrics_to_plot: List of metrics to calculate and plot
    - figsize: Figure size for the plot
    """
    y_test, y_pred, y_pred_proba = data_tuple

    if stride is None:
        stride = max(window_size // 4, 1)  # Default stride is 1/4 of window size

    def calculate_window_metrics(start_idx: int, end_idx: int) -> Dict[str, float]:
        """Calculate metrics for the current window"""
        result = {}
        if "accuracy" in metrics_to_plot:
            result["accuracy"] = (
                y_pred[start_idx:end_idx] == y_test[start_idx:end_idx]
            ).mean()

        if "entropy" in metrics_to_plot:
            result["entropy"] = np.mean(
                [entropy(probs) for probs in y_pred_proba[start_idx:end_idx]]
            )

        if "top_2_accuracy" in metrics_to_plot:
            top_2_correct = 0
            for i in range(start_idx, end_idx):
                top_2 = np.argsort(y_pred_proba[i])[-2:]
                if y_test[i] in top_2:
                    top_2_correct += 1
            result["top_2_accuracy"] = top_2_correct / (end_idx - start_idx)

        if "log_likelihood" in metrics_to_plot:
            true_probs = np.array(
                [
                    probs[true]
                    for probs, true in zip(
                        y_pred_proba[start_idx:end_idx], y_test[start_idx:end_idx]
                    )
                ]
            )
            result["log_likelihood"] = np.mean(np.log(true_probs + 1e-10))

        return result

    # Calculate rolling metrics
    n_samples = len(y_test)
    windows = []
    rolling_metrics = defaultdict(list)

    current_start = 0
    while current_start + window_size <= n_samples:
        windows.append(
            current_start + window_size // 2
        )  # Use middle of window for x-axis
        metrics = calculate_window_metrics(current_start, current_start + window_size)
        for metric, value in metrics.items():
            rolling_metrics[metric].append(value)
        current_start += stride

    # Plotting
    plt.figure(figsize=figsize)

    for metric in metrics_to_plot:
        plt.plot(
            windows,
            rolling_metrics[metric],
            label=f"{metric} (window={window_size})",
            marker="o",
            markersize=3,
        )

    plt.xlabel("Sample Position")
    plt.ylabel("Score")
    plt.title("Rolling Performance Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Usage example:
"""
# For multiple datasets:
dataset_results = []
dataset_names = ['Train', 'Val', 'Test']

for dataset_loader in [train_loader, val_loader, test_loader]:
    metrics, data = evaluate_route_predictions(
        model=model,
        dataloader=dataset_loader,
        dataset=dataset,
        device='cuda'
    )
    dataset_results.append((metrics, data))

plot_metrics_across_datasets(
    evaluation_results=dataset_results,
    dataset_names=dataset_names,
    metrics_to_plot=['accuracy', 'weighted avg_f1-score']
)

# For rolling performance on single dataset:
metrics, data = evaluate_route_predictions(
    model=model,
    dataloader=test_loader,
    dataset=test_dataset,
    device='cuda'
)

plot_rolling_performance(
    metrics=metrics,
    data_tuple=data,
    window_size=100,
    stride=25,  # Move window forward by 25 samples each time
    metrics_to_plot=['accuracy', 'entropy', 'top_2_accuracy', 'log_likelihood']
)
"""


def evaluate_xgb_from_dataset(
    xgb_model, df, dataloader_, scaler, feature_encoders, device="cpu", debug=False
):

    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader_:
            try:
                batch = batch.to(device)
                ids = list(
                    zip(
                        batch.game_id[batch.eligible_mask].cpu().tolist(),
                        batch.play_id[batch.eligible_mask].cpu().tolist(),
                        batch.player_ids[batch.eligible_mask].cpu().tolist(),
                    )
                )
                batch_df = filter_by_game_play_ids(df, ids)

                X, y, _, __ = prepare_route_prediction_data(
                    batch_df,
                    training=False,
                    feature_encoders=feature_encoders,
                    scaler=scaler,
                )
                probabilities = xgb_model.predict_proba(X)

                targets = torch.tensor(y)

                if debug:
                    batch_targets = batch.route_targets[batch.eligible_mask]
                    print(batch_targets)
                    print(targets)
                    raise Exception("Debugging")

                if isinstance(probabilities, torch.Tensor):
                    predictions = probabilities.argmax(dim=1).cpu().numpy()
                    probabilities = probabilities.cpu().numpy()
                else:
                    predictions = (
                        torch.tensor(probabilities).argmax(dim=1).cpu().numpy()
                    )

                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_targets.append(targets.cpu().numpy())
            except Exception as e:
                # print(predictions)
                # print(probabilities)
                raise e

    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    all_targets = np.concatenate(all_targets)
    report = classification_report(y_true=all_targets, y_pred=all_predictions)
    top1k = top_k_accuracy_score(
        y_true=all_targets, y_score=all_probabilities, k=1, labels=list(range(0, 13))
    )
    top2k = top_k_accuracy_score(
        y_true=all_targets, y_score=all_probabilities, k=2, labels=list(range(0, 13))
    )

    return report, top1k, top2k



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch




# Add residual connections and layer normalization
class EnhancedGATBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, heads, dropout, concat=False):
        super().__init__()
        self.gat = GATv2Conv(
            in_dim,
            hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
            concat=concat,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = self.gat(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class PlayGNN(nn.Module):
    def __init__(
        self,
        num_positions,
        hidden_dim=128,
        num_gnn_layers=3,
        num_route_classes=10,
        dropout=0.1,
        max_downs=4,
        max_quarters=4,
        num_teams=32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Global feature embeddings
        position_emb_dim = 4
        down_emb_dim = 2
        quarter_emb_dim = 2
        team_emb_dim = 8
        player_emb_dim = 16
        self.position_embedding = nn.Embedding(num_positions, position_emb_dim)
        self.down_emb = nn.Embedding(max_downs, down_emb_dim)
        self.quarter_emb = nn.Embedding(max_quarters, quarter_emb_dim)
        self.team_emb = nn.Embedding(num_teams, team_emb_dim)
        self.player_emb = nn.Embedding(1000, player_emb_dim)

        # Remove feature encoders since we're directly concatenating
        # Note: GNN layers will handle the concatenated features

        self.max_per_graph = 16

        emb_dims = position_emb_dim + down_emb_dim + quarter_emb_dim + team_emb_dim + player_emb_dim

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                EnhancedGATBlock(
                    hidden_dim,
                    hidden_dim,
                    edge_dim=3,
                    heads=2,
                    dropout=dropout,
                    concat=False,
                )
            )
        # cannot use residual because shapes do not align

        self.gnn_layers[0] = GATv2Conv(
            5 + 4 + emb_dims,
            hidden_dim,
            edge_dim=3,
            heads=2,
            dropout=dropout,
            concat=False,
        )

        # Frame-level processing
        self.frame_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Historical plays attention
        self.historical_attention = nn.MultiheadAttention(
            hidden_dim * 2,  # bidirectional LSTM output
            num_heads=2,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.max_per_graph),
            nn.GELU(),
            nn.LayerNorm(self.max_per_graph),
            nn.Dropout(dropout),
            nn.Linear(self.max_per_graph, num_route_classes),
        )

    def forward(self, data):
        # Extract batch information
        batch_size = data.batch.max().item() + 1
        device = data.x.device

        # Extract individual features from data.x
        position_idx = data.x[:, 0].long()  # Assuming position index is first column
        raw_features = data.x[:, 1:5]  # Rest of features (weight, height, is_offense, motion)

        # Create position embeddings
        position_embedded = self.position_embedding(
            position_idx
        )  # [num_nodes, hidden_dim]

        # Global features - indexed by batch
        down_embedded = self.down_emb(
            data.down - 1
        )  # [num_nodes, hidden_dim]
        quarter_embedded = self.quarter_emb(
            data.quarter - 1
        )  # [num_nodes, hidden_dim]
        team_embedded = self.team_emb(
            data.offense_team
        )  # [num_nodes, hidden_dim]
        player_embedded = self.player_emb(
            data.player_ids
        )

        game_clock = data.game_clock
        yardline = data.yardline
        yards_to_go = data.yards_to_go

        numeric_features = torch.cat(
            [
                game_clock.unsqueeze(1),
                yardline.unsqueeze(1),
                yards_to_go.unsqueeze(1),
                data.offense_score.unsqueeze(1),
                data.defense_score.unsqueeze(1),
            ],
            dim=1,
        )

        # Concatenate all features
        node_features = torch.cat(
            [
                position_embedded,  # [num_nodes, hidden_dim]
                raw_features,  # [num_nodes, num_raw_features]
                down_embedded,  # [num_nodes, hidden_dim]
                quarter_embedded,  # [num_nodes, hidden_dim]
                team_embedded,  # [num_nodes, hidden_dim]
                numeric_features,
                player_embedded
            ],
            dim=1,
        )

        edge_features = data.edge_attr[:, 0:3]  # Keep edge features as is

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, data.edge_index, edge_features)
            node_features = F.gelu(node_features)

        # Separate historical and current plays using plays_elapsed
        historical_mask = data.plays_elapsed > 0
        current_mask = data.plays_elapsed == 0

        # Create new batch indices for historical and current plays separately
        batch_size = data.batch.max().item() + 1
        historical_batch = data.batch[historical_mask]
        current_batch = data.batch[current_mask]

        # Get dense batched representations
        historical_nodes, historical_lens = to_dense_batch(
            node_features[historical_mask],
            historical_batch,
            max_num_nodes=self.max_per_graph,
        )
        current_nodes, current_lens = to_dense_batch(
            node_features[current_mask], current_batch, max_num_nodes=self.max_per_graph
        )

        # Process through LSTM
        historical_encoded, _ = self.frame_lstm(historical_nodes)
        current_encoded, _ = self.frame_lstm(current_nodes)

        # Pool frames to get play-level representations
        # Use the dense masks from to_dense_batch to properly pool
        historical_sum = historical_lens.sum(dim=1, keepdim=True).clamp(min=1e-6)
        current_sum = current_lens.sum(dim=1, keepdim=True).clamp(min=1e-6)

        historical_plays = (
            torch.sum(historical_encoded * historical_lens.unsqueeze(-1), dim=1)
            / historical_sum
        )
        current_plays = (
            torch.sum(current_encoded * current_lens.unsqueeze(-1), dim=1) / current_sum
        )

        # Apply attention between current play and historical plays
        attended_history, _ = self.historical_attention(
            current_plays, historical_plays, historical_plays
        )

        # Get the play-level context as before
        final_representation = torch.cat(
            [current_plays.squeeze(1), attended_history.squeeze(1)], dim=-1
        )  # [batch_size, hidden_dim*4]

        # print('node feats')
        # print(node_features.shape)

        # Get current eligible receivers
        current_eligible_mask = data.eligible_mask & (data.plays_elapsed == 0)
        eligible_nodes = node_features[
            current_eligible_mask
        ]  # [num_eligible_total, hidden_dim]
        eligible_batch_idx = data.batch[current_eligible_mask]  # [num_eligible_total]

        # Get the corresponding play context for each eligible receiver
        play_context = final_representation[
            eligible_batch_idx
        ]  # [num_eligible_total, hidden_dim*4]

        # print('after masking')
        # print(current_eligible_mask)
        # print(eligible_nodes.shape)
        # print(eligible_nodes.isnan().sum())
        # print('context')
        # print(play_context.shape)
        # print(play_context.isnan().sum())

        # Combine receiver features with play context
        combined_features = torch.cat(
            [
                eligible_nodes,  # Receiver-specific features
                play_context,  # Play-level context
            ],
            dim=1,
        )  # [num_eligible_total, hidden_dim + hidden_dim*4]

        # print('combined shape')
        # print(combined_features.shape)
        # print(combined_features.isnan().sum())

        # Generate unique predictions for each eligible receiver
        # print(self.mlp)
        route_predictions = self.mlp(
            combined_features
        )  # [num_eligible_total, num_route_classes]

        # print('mlp shape')
        # print(route_predictions.shape)

        # print(route_predictions.isnan().sum())

        return {
            "route_predictions": route_predictions,
            "eligible_batch_idx": eligible_batch_idx,  # Keep track of which batch each prediction belongs to
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)



def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

import warnings

tracking, player_id_mapping = load_and_filter_data(weeks=list(range(1,10)), base_path="/kaggle/input")

tracking_passing_only = filter_passing_plays_only(tracking)

tracking_passing_only = process_game_scores(tracking_passing_only)

tracking_passing_only = tracking_passing_only.sort_values(by=["time"])

print(tracking_passing_only.shape)


import pickle

with open("player_id_mapping.pkl", "wb") as f:
    pickle.dump(player_id_mapping,f)


# this is an assumption for passing plays
tracking_passing_only.loc[
    (tracking_passing_only.routeRan.isna()) & (tracking_passing_only.position.isin(('WR', 'TE', 'RB'))), 
     'routeRan'
] = 'BLOCKING'

unique_routes = sorted([route for route in tracking_passing_only.routeRan.unique() if not pd.isna(route)])
unique_routes


train_weeks = list(range(1,7))
val_weeks = list(range(7,9))
test_weeks = list(range(9,10))


freqs = tracking_passing_only[~tracking_passing_only.routeRan.isna()].routeRan.value_counts() / len(tracking_passing_only[~tracking_passing_only.routeRan.isna()])
freqs_sorted = sorted(list(freqs.to_dict().items()))
freqs_sorted
class_weights = torch.FloatTensor([1 - w[1] for w in freqs_sorted])
class_weights**3


mappings = DataMappings()
mappings.fit(tracking_passing_only)


dataset = HistoricalMultiRoutePlayDataset(
    df=tracking_passing_only,
    game_play_pairs=create_game_play_pairs(tracking_passing_only),
    target_df=tracking_passing_only[['gameId','playId','routeRan']],
    offense_positions=['QB', 'RB', 'WR', 'TE'],
    defense_positions=['CB', 'SS', 'FS', 'LB', 'OLB', 'ILB'],
    eligible_positions=['WR', 'TE', 'RB'],
    n_frames=1,
    device='cpu',
    unique_routes=unique_routes,
    mappings=mappings,  # Pass the mappings object
    n_workers=1,
    teams_per_chunk=32,
    max_history_plays=10,
    augment=False,
    do_not_augment_weeks=[7, 8, 9]
)


train_loader, val_loader, test_loader, indices, week_ranges = create_week_stratified_split(
        dataset, 
        train_weeks=train_weeks,
        val_weeks=val_weeks,
        test_weeks=test_weeks,
        batch_size=128, 
        random_seed=42,
        preserve_time_order=False,
        val_random=False  # Whether to randomly sample validation set from train period
)


for b in train_loader:
    break
print(b)


device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


model = PlayGNN(
    num_positions=20,
    hidden_dim=64,
    num_gnn_layers=3,
    num_route_classes=dataset.num_route_classes, # for -1 or NaN
    dropout=0.1,
    max_downs=4,
    max_quarters=5,
    num_teams=32,
)

model = model.to(device)


# Assuming you have a model defined as 'model'
num_weights = count_parameters(model)
print("Number of weights:", num_weights)

# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_route_predictor(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    num_classes=dataset.num_route_classes, 
    num_epochs=100, device=device, 
    allow_overfit=False,
    early_stopping_patience=25,
    debug=False,
    class_weights=class_weights.to(device),
)

# Make predictions
model.eval()


torch.save(model.state_dict(), f"model_gnn_{min(train_weeks)}_{max(train_weeks)}.pt")


model.eval()
for b in train_loader:
    break

metrics, (y_test, y_pred, y_pred_proba, predictions_df) = evaluate_route_predictions_table(model, test_loader, dataset, device, softmax=True)
metrics['display_tables']()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import entropy
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from tqdm import tqdm


def get_initial_center_position(frame_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Get the initial position of the center for normalization reference.
    Falls back to offensive line average if no center is found.

    Returns:
        Tuple of (x, y) coordinates for reference point
    """
    # Get the first frame
    first_frame = frame_data[frame_data["frameId"] == frame_data["frameId"].min()]

    # Find Center's position
    center = first_frame[first_frame["position"] == "C"]
    if len(center) == 1:
        return center["x"].iloc[0], center["y"].iloc[0]

    # Fallback to offensive line average if no Center is found
    offensive_line = first_frame[first_frame["position"].isin(["C", "G", "T", "OL"])]
    if len(offensive_line) > 0:
        return offensive_line["x"].mean(), offensive_line["y"].mean()

    # Final fallback to offensive players average
    offense = first_frame[first_frame["position_group"] == "Offense"]
    return offense["x"].mean(), offense["y"].mean()


def normalize_coordinates(
    frame_data: pd.DataFrame, play_direction: str, ref_x: float, ref_y: float
) -> pd.DataFrame:
    """
    Normalize player coordinates relative to the initial center position and play direction.
    """
    frame_data = frame_data.copy()

    # Normalize relative to reference point
    frame_data["x"] = frame_data["x"] - ref_x
    frame_data["y"] = frame_data["y"] - ref_y

    # Flip coordinates if play is going left
    if play_direction.lower() == "left":
        frame_data["x"] = -frame_data["x"]
        # frame_data["y"] = -frame_data["y"]

    return frame_data


def get_detailed_position_group(position: str) -> str:
    """
    Categorize players into specific position groups for more granular comparison.
    """
    position_groups = {
        "QB": "Quarterback",
        "C": "Offensive Line",
        "G": "Offensive Line",
        "T": "Offensive Line",
        "OL": "Offensive Line",
        "RB": "Skill Players",
        "FB": "Skill Players",
        "WR": "Skill Players",
        "TE": "Skill Players",
        "DE": "Defensive Line",
        "DT": "Defensive Line",
        "NT": "Defensive Line",
        "ILB": "Linebackers",
        "OLB": "Linebackers",
        "MLB": "Linebackers",
        "LB": "Linebackers",
        "CB": "Secondary",
        "S": "Secondary",
        "FS": "Secondary",
        "SS": "Secondary",
        "DB": "Secondary",
    }
    return position_groups.get(position, "Other")


def get_players_by_frame(
    play_df: pd.DataFrame, frame_start: int = None, frame_count: int = None
) -> dict:
    """
    Convert play DataFrame into frame-indexed dictionary of normalized player coordinates by position group.
    """
    # Add position group column
    play_df = play_df.copy()
    play_df["position_group"] = play_df["position"].map(get_detailed_position_group)

    # Get play direction
    play_direction = play_df["playDirection"].iloc[0]

    # Get initial center position for normalization
    ref_x, ref_y = get_initial_center_position(play_df)

    # Get frame range if specified
    if frame_start is not None:
        frame_ids = sorted(play_df["frameId"].unique())
        if frame_start < 0:  # Handle negative indexing
            frame_start = len(frame_ids) + frame_start
        frame_start = max(0, min(frame_start, len(frame_ids) - 1))

        if frame_count is not None:
            frame_ids = frame_ids[frame_start : frame_start + frame_count]
        else:
            frame_ids = frame_ids[frame_start:]

        play_df = play_df[play_df["frameId"].isin(frame_ids)]

    frames = {}
    # Group by frameId for efficiency
    for frame_id, frame_data in play_df.groupby("frameId"):
        # Normalize coordinates for this frame
        normalized_frame = normalize_coordinates(
            frame_data, play_direction, ref_x, ref_y
        )

        frames[frame_id] = {
            "Quarterback": [],
            "Offensive Line": [],
            "Skill Players": [],
            "Defensive Line": [],
            "Linebackers": [],
            "Secondary": [],
            "Other": [],
        }

        # Then group by position_group
        for group, group_data in normalized_frame.groupby("position_group"):
            frames[frame_id][group] = group_data[
                ["x", "y", "nflId", "club", "position"]
            ].to_dict("records")

    return frames


def analyze_receiver_positions(
    frame_data: dict, relative_distances: bool = True
) -> Dict[str, Dict]:
    """
    Analyze the positions and distances between receivers (WR/TE) in a frame.

    Args:
        frame_data: Dictionary containing player positions by position group
        relative_distances: If True, order receivers by distance to each receiver
                          If False, order receivers by absolute Y position

    Returns:
        Dictionary containing:
        - wr_positions: Dict mapping nflId to their position (1-5)
        - distances: Dict mapping nflId to their distances to all receivers
        - nearest_distances: Dict mapping nflId to their closest receiver distance
        - center_distances: Dict mapping nflId to their signed distance from center
    """
    # Extract receivers from skill players group
    receivers = [
        player
        for player in frame_data["Skill Players"]
        if player["position"] in ("WR", "TE")
    ]

    if not receivers:
        return {
            "wr_positions": {},
            "distances": {},
            "nearest_distances": {},
            "center_distances": {},
        }

    # Convert to numpy array for vectorized calculations
    receiver_coords = np.array([[r["x"], r["y"]] for r in receivers])

    # Calculate pairwise distances between all receivers
    distances = np.sqrt(
        np.sum((receiver_coords[:, np.newaxis] - receiver_coords) ** 2, axis=2)
    )

    # Create distance mappings and positions
    distances_dict = {}
    wr_positions = {}
    nearest_distances = {}

    for i, receiver in enumerate(receivers):
        # Get distances to all receivers (will include self as 0)
        all_distances = distances[i]
        receiver_to_others = [(j, d) for j, d in enumerate(all_distances) if j != i]

        if relative_distances:
            # Sort other receivers by distance to current receiver while keeping original indices
            sorted_receivers = sorted(receiver_to_others, key=lambda x: x[1])

            # Map distances to receiver numbers (1-5 based on proximity)
            receiver_distances = {
                position + 1: distance
                for position, (_, distance) in enumerate(sorted_receivers)
            }

            # Store nearest distance (actual minimum distance)
            nearest_distances[receiver["nflId"]] = (
                sorted_receivers[0][1] if sorted_receivers else np.nan
            )

        else:
            # Sort receivers by Y position
            y_positions = sorted(range(len(receivers)), key=lambda x: receivers[x]["y"])
            position_map = {idx: pos + 1 for pos, idx in enumerate(y_positions)}

            # Map distances using Y-based positions
            receiver_distances = {position_map[j]: d for j, d in receiver_to_others}

            # Store Y-based position
            wr_positions[receiver["nflId"]] = position_map[i]

            # Store nearest distance (minimum of all distances)
            nearest_distances[receiver["nflId"]] = (
                min(d for _, d in receiver_to_others) if receiver_to_others else np.nan
            )

        distances_dict[receiver["nflId"]] = receiver_distances

    # Calculate signed distances from center (0, 0)
    center_distances = {
        receiver["nflId"]: receiver[
            "x"
        ]  # X-coordinate is already normalized relative to center
        for receiver in receivers
    }

    return {
        "wr_positions": wr_positions,
        "distances": distances_dict,
        "nearest_distances": nearest_distances,
        "center_distances": center_distances,
    }


def join_receiver_analysis_to_df(
    tracking_df: pd.DataFrame,
    game_id: int,
    play_id: int,
    relative_distances: bool = True,
    frame_data: dict = None,
) -> pd.DataFrame:
    """
    Join receiver analysis results to tracking DataFrame for a specific play.

    Args:
        tracking_df: Original tracking DataFrame
        game_id: Game ID to analyze
        play_id: Play ID to analyze
        relative_distances: If True, order receivers by distance to each receiver
                          If False, order receivers by absolute Y position
        frame_data: Optional pre-computed frame data

    Returns:
        DataFrame with receiver analysis columns added for WR/TE players
    """
    # Filter for specific game and play

    play_df = tracking_df[
        (tracking_df["gameId"] == game_id) & (tracking_df["playId"] == play_id)
    ].copy()

    if frame_data is None:
        frame_data = get_players_by_frame(play_df)

    # Initialize new columns
    if not relative_distances:
        play_df["wr_number"] = (
            np.nan
        )  # Which receiver they are (1-5 based on Y position)
    play_df["wr_center_distance"] = np.nan
    play_df["nearest_wr_distance"] = np.nan

    # Initialize distance columns
    for i in range(1, 6):
        play_df[f"dis_wr_{i}"] = np.nan

    # Process each frame
    for frame_id in play_df["frameId"].unique():
        if frame_id not in frame_data:
            continue

        analysis = analyze_receiver_positions(frame_data[frame_id], relative_distances)

        # Update values for receivers in this frame
        frame_mask = (play_df["frameId"] == frame_id) & (
            play_df["position"].isin(("WR", "TE", "RB"))
        )

        for nfl_id in play_df[frame_mask]["nflId"]:
            if nfl_id in analysis["distances"]:
                idx = play_df[
                    (play_df["frameId"] == frame_id) & (play_df["nflId"] == nfl_id)
                ].index

                # Set position number if using absolute Y positions
                if not relative_distances and nfl_id in analysis["wr_positions"]:
                    play_df.loc[idx, "wr_number"] = analysis["wr_positions"][nfl_id]

                # Set basic info
                play_df.loc[idx, "wr_center_distance"] = analysis["center_distances"][
                    nfl_id
                ]
                play_df.loc[idx, "nearest_wr_distance"] = analysis["nearest_distances"][
                    nfl_id
                ]

                # Set distances to other receivers
                for target_pos, distance in analysis["distances"][nfl_id].items():
                    play_df.loc[idx, f"dis_wr_{target_pos}"] = distance

    return play_df



import pandas as pd


import tqdm


games = list(tracking_passing_only.gameId.unique())

enriched_df = None

tracking_subset = tracking_passing_only.copy()

tracking_subset["distance_from_los"] = np.abs(tracking_subset["x"] - tracking_subset["absoluteYardlineNumber"])
tracking_subset["distance_from_sideline"] = np.minimum(tracking_subset["y"], 53.3 - tracking_subset["y"])

pairs = set()
for item in list(tracking_subset[['gameId', 'playId']].to_numpy()):
    pairs.add((item[0].item(), item[1].item()))

i = 0

for game_id, play_id in list(pairs):
    i += 1
    if i % 500 == 0:
        print(i)
        print(enriched_df.shape)
    # Option 2: Use pre-computed frame data
    frame_data = get_players_by_frame(tracking_subset[(tracking_subset.gameId == game_id) & (tracking_subset.playId == play_id)])
    if enriched_df is None:
        enriched_df = join_receiver_analysis_to_df(tracking_subset, game_id, play_id, True, frame_data)
    else:
        try:
            tmp = join_receiver_analysis_to_df(tracking_subset, game_id, play_id, True, frame_data)
        except Exception as e:
            print(e)
            print(f'game, play: {game_id}, {play_id}')
        enriched_df = pd.concat([enriched_df, tmp])


enriched_df.inMotionAtBallSnap = enriched_df.inMotionAtBallSnap.astype(bool)
(~enriched_df.routeRan.isna() & (enriched_df.event == "ball_snap")).sum()
enriched_df.routeRan.value_counts(dropna=False)

enriched_df = enriched_df[enriched_df.position.isin(('WR', 'TE', 'RB'))]
enriched_df.loc[enriched_df.routeRan.isna(), 'routeRan'] = 'BLOCKING'
enriched_df.routeRan.value_counts(dropna=False)


print(train_weeks)
print(val_weeks)
xgb_model, xgb_metrics, encoders, split_info, scaler = train_route_prediction_pipeline(
    merged_df=enriched_df,
    train_split=train_weeks,
    val_split=val_weeks,
    test_split=test_weeks,
    max_depth=6, 
    split_method="week"
)


xgb_metrics['display_tables']()


import pickle


with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)


ids = list(
    zip(
        b.game_id.cpu().tolist(),
        b.play_id.cpu().tolist(),
        b.player_ids.cpu().tolist(),
    )
)

df_subset = filter_by_game_play_ids(enriched_df, ids)


for k, v in dataset.route_to_idx.items():
    assert v == encoders['routeRan'].transform([k])
    print(encoders['routeRan'].transform([k]))


import torch
import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Initialize lists for predictions
all_gnn_preds = []
all_xgb_preds = []
all_true_labels = []

# Collect predictions
for batch in val_loader:
    # Convert game_id and play_id to tuples
    id_pairs = list(zip(
        batch.game_id[batch.eligible_mask].cpu().tolist(),
        batch.play_id[batch.eligible_mask].cpu().tolist(),
        batch.player_ids[batch.eligible_mask].cpu().tolist()
    ))
    # Get corresponding data for XGBoost
    df_subset = filter_by_game_play_ids(enriched_df, id_pairs)
    print(len(id_pairs))
    print(df_subset.shape)
    X_subset, y_subset, _, _ = prepare_route_prediction_data(
        df_subset, 
        feature_encoders=encoders,
        training=False,
        scaler=scaler
    )
    xgb_probs = xgb_model.predict_proba(X_subset)
    all_xgb_preds.append(xgb_probs)

    # print(df_subset.shape)
    # print(batch.eligible_mask.sum())

    # display(df_subset)
    # display(df_subset.playId.nunique())
    # print(batch.play_id)
    
    # Get predictions from both models
    with torch.no_grad():
        gnn_logits = model(batch.to(device))["route_predictions"]
        gnn_probs = torch.softmax(gnn_logits, dim=-1).cpu().numpy()
    
    # Store predictions and true labels
    all_gnn_preds.append(gnn_probs)
    all_true_labels.append(batch.route_targets[batch.eligible_mask].cpu().numpy())

# Concatenate all predictions
all_gnn_preds = np.concatenate(all_gnn_preds, axis=0)
all_xgb_preds = np.concatenate(all_xgb_preds, axis=0)
all_true_labels = np.concatenate(all_true_labels, axis=0)

route_names = list(sorted([r[0] for r in list(dataset.route_to_idx.items())]))


def create_corr_scatter(i):
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get top predictions for each model
    gnn_top = np.argmax(all_gnn_preds, axis=1)
    xgb_top = np.argmax(all_xgb_preds, axis=1)
    
    # True negatives plot
    true_negatives = all_true_labels != i
    neg_df = pd.DataFrame({
        'gnn': all_gnn_preds[true_negatives, i],
        'xgb': all_xgb_preds[true_negatives, i],
        'gnn_pred': gnn_top[true_negatives] == i,
        'xgb_pred': xgb_top[true_negatives] == i,
        'both_pred': (gnn_top[true_negatives] == i) & (xgb_top[true_negatives] == i)
    })
    
    # Color code based on predictions
    colors = np.where(neg_df['both_pred'], 'purple',
                     np.where(neg_df['gnn_pred'], 'blue',
                             np.where(neg_df['xgb_pred'], 'red', 'gray')))
    
    ax1.scatter(neg_df['gnn'], neg_df['xgb'], c=colors, alpha=0.5)
    ax1.set_title(f'True Negatives (Route {route_names[i]})')
    ax1.set_xlabel('GNN Probability')
    ax1.set_ylabel('XGBoost Probability')
    
    # True positives plot
    true_positives = all_true_labels == i
    pos_df = pd.DataFrame({
        'gnn': all_gnn_preds[true_positives, i],
        'xgb': all_xgb_preds[true_positives, i],
        'gnn_pred': gnn_top[true_positives] == i,
        'xgb_pred': xgb_top[true_positives] == i,
        'both_pred': (gnn_top[true_positives] == i) & (xgb_top[true_positives] == i)
    })
    
    colors = np.where(pos_df['both_pred'], 'purple',
                     np.where(pos_df['gnn_pred'], 'blue',
                             np.where(pos_df['xgb_pred'], 'red', 'gray')))
    
    ax2.scatter(pos_df['gnn'], pos_df['xgb'], c=colors, alpha=0.5)
    ax2.set_title(f'True Positives (Route {route_names[i]})')
    ax2.set_xlabel('GNN Probability')
    ax2.set_ylabel('XGBoost Probability')
    
    # Add diagonal line to both plots
    for ax in [ax1, ax2]:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Neither predicted', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='GNN predicted', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='XGB predicted', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', label='Both predicted', markersize=10)
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    ax2.legend(handles=legend_elements, loc='upper left')
    
    # Add correlation coefficient as text
    corr = pearsonr(all_gnn_preds[:, i], all_xgb_preds[:, i])[0]
    plt.suptitle(f'Model Comparison for Route {route_names[i]} (correlation: {corr:.3f})', y=1.05)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nStatistics for Route {route_names[i]}:")
    print(f"Number of true positives: {true_positives.sum()}")
    print(f"Number of true negatives: {true_negatives.sum()}")
    print(f"Overall correlation: {corr:.3f}")
    
    # Prediction statistics for true positives
    if true_positives.sum() > 0:
        n_both = pos_df['both_pred'].sum()
        n_gnn = pos_df['gnn_pred'].sum() - n_both
        n_xgb = pos_df['xgb_pred'].sum() - n_both
        n_neither = len(pos_df) - n_gnn - n_xgb - n_both
        print("\nTrue Positive Predictions:")
        print(f"Both models correct: {n_both}")
        print(f"Only GNN correct: {n_gnn}")
        print(f"Only XGB correct: {n_xgb}")
        print(f"Neither correct: {n_neither}")
    
    # Prediction statistics for true negatives
    if true_negatives.sum() > 0:
        n_both = neg_df['both_pred'].sum()
        n_gnn = neg_df['gnn_pred'].sum() - n_both
        n_xgb = neg_df['xgb_pred'].sum() - n_both
        n_neither = len(neg_df) - n_gnn - n_xgb - n_both
        print("\nTrue Negative Predictions:")
        print(f"Both predicted incorrectly: {n_both}")
        print(f"Only GNN predicted incorrectly: {n_gnn}")
        print(f"Only XGB predicted incorrectly: {n_xgb}")
        print(f"Neither predicted incorrectly: {n_neither}")


i = 0  # Route index to analyze
create_corr_scatter(3)


i = 3  # Route index to analyze
create_corr_scatter(3)


create_corr_scatter(4)


feature_encoders = encoders

# Convert route indices to names
route_names = feature_encoders['routeRan'].inverse_transform(range(all_gnn_preds.shape[1]))

# Calculate correlations between GNN and XGB predictions for each route
n_classes = all_gnn_preds.shape[1]
correlation_matrix = np.zeros((n_classes, n_classes))

for i in range(n_classes):
    for j in range(n_classes):
        correlation_matrix[i, j] = pearsonr(all_gnn_preds[:, i], all_xgb_preds[:, j])[0]

# Create heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(
    correlation_matrix,
    xticklabels=route_names,
    yticklabels=route_names,
    cmap='RdBu_r',  # Red-Blue diverging colormap
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,  # Show correlation values
    fmt='.2f',   # Format correlation values to 2 decimal places
    square=True  # Make cells square
)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.xlabel('XGBoost Predictions')
plt.ylabel('GNN Predictions')
plt.title('Correlation between GNN and XGBoost Predictions by Route')

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()

# Print some summary statistics
print("\nSummary Statistics:")
print(f"Average correlation: {np.mean(correlation_matrix):.3f}")
print(f"Median correlation: {np.median(correlation_matrix):.3f}")
print(f"Max correlation: {np.max(correlation_matrix):.3f}")
print(f"Min correlation: {np.min(correlation_matrix):.3f}")

# Print diagonal correlations (same route predictions)
print("\nCorrelations for same route predictions:")
for i, route in enumerate(route_names):
    print(f"{route}: {correlation_matrix[i, i]:.3f}")


train_weeks = list(range(7, 8))
val_weeks = list(range(8,9))
test_weeks = list(range(9,10))


from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from typing import Dict, Optional, Union, Literal
from sklearn.metrics import log_loss, accuracy_score


@dataclass
class EnsembleConfig:
    meta_type: Literal["nn", "gbm"] = "nn"
    # NN params
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    patience: int = 5
    # GBM params
    gbm_params: Dict = None
    scheduler_type = "cosine"
    gamma = 0.7
    min_lr = 0.00001
    hidden_dim = 8

    def __post_init__(self):
        if self.meta_type == "gbm" and not self.gbm_params:
            self.gbm_params = {
                "objective": "multi:softprob",
                "learning_rate": 0.1,
                "max_depth": 4,
                "n_estimators": 100,
                "early_stopping_rounds": 10,
            }


class MetaLearner(nn.Module):
    def __init__(self, num_classes: int, meta_type: str, gbm_params: Dict = None, hidden_dim = 32):
        super().__init__()
        self.meta_type = meta_type
        self.hidden_dim = 32
        if meta_type == "nn":
            self.model = nn.Sequential(
                nn.Linear(2 * num_classes, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.Linear(self.hidden_dim // 4, num_classes),
            )
        else:
            self.model = xgb.XGBClassifier(num_class=num_classes, **gbm_params)

    def forward(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        if self.meta_type == "nn":
            return self.model(x)
        return self.model.predict_proba(x)


class RouteEnsemble(nn.Module):
    def __init__(
        self,
        xgb_model,
        gnn_model,
        config: EnsembleConfig,
        num_classes: int,
        feature_encoders,
        device: str = "cuda",
        scaler=None,
        debug=False,
    ):
        super().__init__()
        self.xgb_model = xgb_model
        self.gnn_model = gnn_model
        self.config = config
        self.device = device
        self.feature_encoders = feature_encoders
        self.scaler = scaler
        self.debug = debug
        self.meta_learner = MetaLearner(
            num_classes=num_classes,
            meta_type=config.meta_type,
            gbm_params=config.gbm_params,
            hidden_dim=config.hidden_dim
        ).to(device)

        if config.meta_type == "nn":
            self.optimizer = torch.optim.AdamW(
                self.meta_learner.parameters(), lr=config.lr
            )
            self.criterion = nn.CrossEntropyLoss()
            # Initialize scheduler based on type
            if config.scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.epochs, eta_min=config.min_lr
                )
            else:  # 'step'
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=config.step_size, gamma=config.gamma
                )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Forward pass compatible with evaluation function"""
        # Get base model predictions
        xgb_probs, gnn_probs, y, X_prepared = self._get_single_batch_base_predictions(
            batch, self.current_plays_df
        )

        # Combine predictions using meta-learner
        if self.config.meta_type == "nn":
            xgb_probs = torch.FloatTensor(xgb_probs)
            gnn_probs = torch.FloatTensor(gnn_probs)
            x = torch.cat([xgb_probs, gnn_probs], dim=1).to(self.device)
            combined_logits = self.meta_learner.model(x)  # Get logits before softmax
            return {
                "route_predictions": combined_logits,
                "gnn": gnn_probs,
                "xgb": xgb_probs,
                "target": y,
                "X_prepared": X_prepared,
            }
        else:
            x = np.concatenate([xgb_probs, gnn_probs], axis=1)
            combined_probs = self.meta_learner.model.predict_proba(x)
            # Convert to logits for compatibility
            combined_logits = torch.FloatTensor(np.log(combined_probs + 1e-10)).to(
                self.device
            )
            return {
                "route_predictions": combined_probs,
                "gnn": torch.FloatTensor(gnn_probs),
                "xgb": torch.FloatTensor(xgb_probs),
                "target": y,
                "X_prepared": X_prepared,
            }

    def _get_single_batch_base_predictions(self, batch, plays_df) -> tuple:
        """Get predictions from base models"""
        all_xgb, all_gnn, all_labels = [], [], []

        # Get corresponding plays for XGBoost
        ids = list(
            zip(
                batch.game_id[batch.eligible_mask].cpu().tolist(),
                batch.play_id[batch.eligible_mask].cpu().tolist(),
                batch.player_ids[batch.eligible_mask].cpu().tolist(),
            )
        )
        batch_df = filter_by_game_play_ids(plays_df, ids)

        # GNN predictions
        gnn_batch = batch.to(self.device)
        gnn_out = self.gnn_model(gnn_batch)["route_predictions"]
        gnn_probs = torch.softmax(gnn_out, dim=-1).detach().cpu().numpy()

        # XGB predictions
        X, y, _, _2 = prepare_route_prediction_data(
            batch_df,
            training=False,
            feature_encoders=self.feature_encoders,
            scaler=self.scaler,
        )
        self.X_prepared = X
        xgb_probs = self.xgb_model.predict_proba(X)

        try:
            batch_vals = batch.route_targets[batch.eligible_mask]
            y_vals = torch.tensor(y).to(self.device)
            torch.testing.assert_close(batch_vals, y_vals)
        except AssertionError as e:
            print(batch_vals)
            print(y_vals)
            raise e

        return xgb_probs, gnn_probs, y, self.X_prepared

    def _get_base_predictions(self, loader, plays_df) -> tuple:
        """Get predictions from base models"""
        all_xgb, all_gnn, all_labels = [], [], []

        for batch in loader:
            xgb_probs, gnn_probs, y, X_prepared = (
                self._get_single_batch_base_predictions(batch, plays_df)
            )
            all_xgb.append(xgb_probs)
            all_gnn.append(gnn_probs)
            all_labels.append(y)

        xgb_preds = np.concatenate(all_xgb)
        gnn_preds = np.concatenate(all_gnn)
        labels = np.concatenate(all_labels)

        if self.config.meta_type == "nn":
            xgb_preds = torch.FloatTensor(xgb_preds).to(self.device)
            gnn_preds = torch.FloatTensor(gnn_preds).to(self.device)
            labels = torch.LongTensor(labels).to(self.device)

        return xgb_preds, gnn_preds, labels

    def _train_nn_epoch(self, xgb_preds, gnn_preds, labels):
        indices = torch.randperm(len(labels))
        losses = []

        # Debug prints at start
        # print(f"XGB preds range: {xgb_preds.min().item():.3f} to {xgb_preds.max().item():.3f}")
        # print(f"GNN preds range: {gnn_preds.min().item():.3f} to {gnn_preds.max().item():.3f}")
        # print(f"Labels range: {labels.min().item():.3f} to {labels.max().item():.3f}")

        for i in range(0, len(labels), self.config.batch_size):
            idx = indices[i : i + self.config.batch_size]
            x = torch.cat([xgb_preds[idx], gnn_preds[idx]], dim=1)

            self.optimizer.zero_grad()
            out = self.meta_learner(x)

            # Debug prints
            # print(f"Output range: {out.min().item():.3f} to {out.max().item():.3f}")

            loss = self.criterion(out, labels[idx])
            loss.backward()

            # Check gradients
            total_grad = 0
            for param in self.meta_learner.parameters():
                if param.grad is not None:
                    total_grad += param.grad.abs().mean().item()
            # print(f"Average gradient magnitude: {total_grad}")

            self.optimizer.step()
            losses.append(loss.item())

        return np.mean(losses)

    def _evaluate(self, xgb_preds, gnn_preds, labels):
        if self.config.meta_type == "nn":
            self.meta_learner.eval()
            with torch.no_grad():
                x = torch.cat([xgb_preds, gnn_preds], dim=1)
                out = self.meta_learner(x)
                loss = self.criterion(out, labels).item()
                preds = out.argmax(dim=1).cpu()
                acc = accuracy_score(labels.cpu(), preds)
        else:
            x = np.concatenate([xgb_preds, gnn_preds], axis=1)
            out = self.meta_learner.model.predict_proba(x)
            loss = log_loss(labels, out)
            acc = accuracy_score(labels, out.argmax(axis=1))

        return {"loss": loss, "accuracy": acc}

    def train_ensemble(self, train_loader, train_df, val_loader=None, val_df=None):
        """Train the ensemble"""
        print("Caching base model predictions...")
        xgb_preds, gnn_preds, labels = self._get_base_predictions(
            train_loader, train_df
        )
        val_preds = (
            None
            if val_loader is None
            else self._get_base_predictions(val_loader, val_df)
        )
        train_preds = xgb_preds, gnn_preds, labels

        print(f"Training {self.config.meta_type.upper()} meta-learner...")
        best_val_loss = float("inf")
        patience_counter = 0

        if self.config.meta_type == "nn":
            for epoch in range(self.config.epochs):
                self.meta_learner.train()
                train_loss = self._train_nn_epoch(xgb_preds, gnn_preds, labels)

                metrics = {"train_loss": train_loss}
                if val_preds:
                    val_metrics = self._evaluate(*val_preds)
                    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        patience_counter = 0
                        self.save_checkpoint("best_model.pt")
                    else:
                        patience_counter += 1

                    if patience_counter >= self.config.patience:
                        print("Early stopping triggered")
                        break
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.config.epochs} -",
                        " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()),
                    )
        else:
            X_train = np.concatenate([train_preds[0], train_preds[1]], axis=1)
            eval_set = None
            if val_preds:
                X_val = np.concatenate([val_preds[0], val_preds[1]], axis=1)
                eval_set = [(X_train, train_preds[2]), (X_val, val_preds[2])]

            self.meta_learner.model.fit(
                X_train,
                train_preds[2],
                eval_set=eval_set,
                verbose=True,
            )
            # Step 5: Generate predictions and metrics
            self.test_preds = self.meta_learner.model.predict(X_val)
            self.test_true = val_preds[2]
            self.test_pred_a = self.meta_learner.model.predict_proba(X_val)
            self.train_preds = self.meta_learner.model.predict(X_train)
            self.train_pred_a = self.meta_learner.model.predict_proba(X_train)
            self.train_true = train_preds[2]

    def predict(self, loader, plays_df):
        """Generate ensemble predictions"""
        xgb_preds, gnn_preds, labels = self._get_base_predictions(loader, plays_df)

        if self.config.meta_type == "nn":
            self.meta_learner.eval()
            with torch.no_grad():
                x = torch.cat([xgb_preds, gnn_preds], dim=1)
                ensemble_preds = self.meta_learner(x).cpu().numpy()
        else:
            x = np.concatenate([xgb_preds, gnn_preds], axis=1)
            ensemble_preds = self.meta_learner.model.predict_proba(x)

        return {
            "ensemble_preds": ensemble_preds,
            "xgb_preds": (
                xgb_preds.cpu().numpy() if torch.is_tensor(xgb_preds) else xgb_preds
            ),
            "gnn_preds": (
                gnn_preds.cpu().numpy() if torch.is_tensor(gnn_preds) else gnn_preds
            ),
            "labels": labels.cpu().numpy() if torch.is_tensor(labels) else labels,
        }

    def set_plays_df(self, df):
        """Set the current plays DataFrame for forward pass"""
        self.current_plays_df = df

    def save_checkpoint(self, path: str):
        if self.config.meta_type == "nn":
            torch.save(
                {
                    "model_state": self.meta_learner.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "config": self.config,
                },
                path,
            )
        else:
            self.meta_learner.model.save_model(path)

    def load_checkpoint(self, path: str):
        if self.config.meta_type == "nn":
            checkpoint = torch.load(path)
            self.meta_learner.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            self.meta_learner.model.load_model(path)


# Example usage:
"""
config = EnsembleConfig(
    meta_type='nn',  # or 'gbm'
    lr=1e-3,
    epochs=10,
    patience=5
)

ensemble = RouteEnsemble(
    xgb_model=xgb_model,
    gnn_model=gnn_model,
    config=config,
    num_classes=5
)

# Train
ensemble.train(
    train_loader=train_loader,
    train_df=train_df,
    val_loader=val_loader,
    val_df=val_df
)

# Predict
results = ensemble.predict(test_loader, test_df)
"""



train_loader, val_loader, test_loader, indices, week_ranges = create_week_stratified_split(
        dataset, 
        train_weeks=train_weeks,
        val_weeks=val_weeks,
        test_weeks=test_weeks,
        batch_size=256, 
        random_seed=42,
        preserve_time_order=False,
        val_random=False  # Whether to randomly sample validation set from train period
)


device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


config = EnsembleConfig(
    meta_type='nn',  # or 'gbm'
    lr=0.75 * 1e-3,
    epochs=150,
    patience=15,
    gbm_params={"max_depth": 2, "n_estimators": 40}
)

ensemble = RouteEnsemble(
    xgb_model=xgb_model,
    gnn_model=model,
    config=config,
    num_classes=13,
    device=device,
    feature_encoders=encoders,
    scaler=feature_encoders['scaler'],
)

# Train
ensemble.train_ensemble(
    train_loader=train_loader,
    train_df=enriched_df,
    val_loader=val_loader,
    val_df=enriched_df,
)


ensemble.set_plays_df(enriched_df)  # This is crucial - don't forget this step!

# 4. Run the evaluation
metrics_xgb, (y_test, y_pred, y_pred_proba, prediction_df) = evaluate_route_predictions_table(
    model=ensemble,
    dataloader_=test_loader,
    device=device,
    dataset_=dataset,
    key="gnn",
    softmax=True
)
# 5. Display the results
metrics_xgb["display_tables"]()


ensemble.set_plays_df(enriched_df)  # This is crucial - don't forget this step!

# 4. Run the evaluation
metrics_xgb, (y_test, y_pred, y_pred_proba, prediction_df) = evaluate_route_predictions_table(
    model=ensemble,
    dataloader_=test_loader,
    device=device,
    dataset_=dataset,
    key="xgb"
)
# 5. Display the results
metrics_xgb["display_tables"]()


ensemble.set_plays_df(enriched_df)  # This is crucial - don't forget this step!

# 4. Run the evaluation
metrics_xgb, (y_test, y_pred, y_pred_proba, prediction_df) = evaluate_route_predictions_table(
    model=ensemble,
    dataloader_=test_loader,
    device=device,
    dataset_=dataset,
    softmax=True
)
# 5. Display the results
metrics_xgb["display_tables"]()




