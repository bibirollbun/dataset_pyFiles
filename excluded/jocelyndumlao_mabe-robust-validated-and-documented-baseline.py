import pandas as pd
import numpy as np
from tqdm.notebook import tqdm  
from sklearn.metrics import f1_score


# Data Loading and Filtering
train_df = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train_subset = train_df.query("~ lab_id.str.startswith('MABe22_')")


def score_class_specific(solution, submission):
    """
    Calculates F1 score separately for each behavior class and averages them (Macro F1).
    This matches the leaderboard logic much closer than the binary check.
    """
    print("Calculating Class-Specific Frame-level F1 Score...")
    
    # Get all unique videos
    video_ids = solution['video_id'].unique()
    
    # Store F1 scores per behavior class
    behavior_scores = {}
    
    for vid in tqdm(video_ids, desc="Scoring Videos"):
        sol_vid = solution[solution['video_id'] == vid]
        sub_vid = submission[submission['video_id'] == vid]
        
        # Get max frame for this video
        if sub_vid.empty and sol_vid.empty:
            continue
        max_frame_sol = sol_vid['stop_frame'].max() if not sol_vid.empty else 0
        max_frame_sub = sub_vid['stop_frame'].max() if not sub_vid.empty else 0
        max_frame = max(max_frame_sol, max_frame_sub)
        
        # Identify all unique behaviors in this video's solution
        unique_behaviors = sol_vid['action'].unique()
        
        for action in unique_behaviors:
            # Create boolean arrays for this specific action
            y_true = np.zeros(max_frame + 1, dtype=int)
            y_pred = np.zeros(max_frame + 1, dtype=int)
            
            # Fill True values
            sol_action = sol_vid[sol_vid['action'] == action]
            for row in sol_action.itertuples():
                y_true[row.start_frame:row.stop_frame] = 1
                
            sub_action = sub_vid[sub_vid['action'] == action]
            for row in sub_action.itertuples():
                # We perform a clip here to ensure we don't go out of bounds if prediction is longer
                start = min(row.start_frame, max_frame)
                stop = min(row.stop_frame, max_frame)
                y_pred[start:stop] = 1
            
            # Calculate F1 for this action in this video
            score = f1_score(y_true, y_pred, zero_division=0)
            
            if action not in behavior_scores:
                behavior_scores[action] = []
            behavior_scores[action].append(score)
            
    # Calculate Macro Average
    final_scores = [np.mean(scores) for scores in behavior_scores.values()]
    macro_f1 = np.mean(final_scores) if final_scores else 0
    
    print(f"Class-Specific F1 Score: {macro_f1:.4f}")
    return macro_f1



# Ground Truth Generation (Validation Baseline)

def generate_ground_truth(dataset):
    ground_truth_list = []
    # We limit to first 100 for speed in local testing, remove .head(100) for full run
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing Annotations"):
        lab_id = row['lab_id']
        video_id = row['video_id']
        annotation_file_path = f"/kaggle/input/MABe-mouse-behavior-detection/train_annotation/{lab_id}/{video_id}.parquet"
        try:
            annotations_df = pd.read_parquet(annotation_file_path)
            annotations_df['video_id'] = video_id
            annotations_df['lab_id'] = lab_id
            annotations_df['behaviors_labeled'] = row['behaviors_labeled']
            annotations_df['agent_id'] = annotations_df['agent_id'].apply(lambda x: f"mouse{x}")
            annotations_df['target_id'] = annotations_df['target_id'].apply(lambda x: f"mouse{x}")
            ground_truth_list.append(annotations_df)
        except FileNotFoundError:
            continue
    return pd.concat(ground_truth_list, ignore_index=True)

# Generate GT for Validation
solution = generate_ground_truth(train_subset)




# Prediction 
def create_all_in_predictions(dataset, traintest):
    """
    Strategy: For every behavior listed in 'behaviors_labeled', 
    predict it for the FULL duration of the video.
    """
    predictions = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Generating Predictions"):
        lab_id = row['lab_id']
        video_id = row['video_id']
        
        tracking_file_path = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking/{lab_id}/{video_id}.parquet"
        
        try:
            # Get video duration
            tracking_meta = pd.read_parquet(tracking_file_path, columns=['video_frame'])
            stop_frame_total = tracking_meta['video_frame'].max() + 1
        except FileNotFoundError:
            # Fallback if tracking file missing
            stop_frame_total = 1800 # default fallback
            # print(f"Warning: Tracking missing for {video_id}")

        # Parse behaviors
        behaviors_labeled_str = row['behaviors_labeled']
        try:
            behaviors_labeled = eval(behaviors_labeled_str)
        except:
            continue
            
        cleaned_behaviors = [b.replace("'", "").strip() for b in behaviors_labeled]
        behavior_tuples = [tuple(b.split(',')) for b in cleaned_behaviors]
        
        # Iterate over EVERY behavior tuple found in the metadata
        for b_tuple in behavior_tuples:
            if len(b_tuple) == 3:
                agent, target, action = b_tuple
                
                # Predict this action for the WHOLE video [0, stop_frame_total]
                predictions.append({
                    'video_id': video_id,
                    'agent_id': agent.strip(),
                    'target_id': target.strip(),
                    'action': action.strip(),
                    'start_frame': 0,
                    'stop_frame': stop_frame_total
                })

    predictions_df = pd.DataFrame(predictions)
    return predictions_df



# Generate predictions
submission = create_all_in_predictions(train_subset, 'train')


# Evaluation
print("Evaluating Predictions...")
score_class_specific(solution, submission)


# Test Data Submission
test_df = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
test_submission = create_all_in_predictions(test_df, 'test')

test_submission.index.name = 'row_id'
test_submission.to_csv('submission.csv')

print("Submission file created successfully!")
print(test_submission.head())




