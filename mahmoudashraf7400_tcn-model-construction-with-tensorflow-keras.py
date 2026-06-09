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


import numpy as np
import pandas as pd
from typing import Tuple
import os 

# --- Configuration for Body Parts and Feature Calculation ---
BODY_VECTOR_START = 'tail_base'
BODY_VECTOR_END = 'nose'
M1_ID = 1
M2_ID = 2
SUBMISSION_FILENAME = 'submission.csv' # Define the required filename
# Assuming a frame rate for velocity calculation (e.g., 30 FPS)
FRAME_RATE = 10 # Since our mock data is 10 frames, let's assume 10 frames per second (time step 0.1s)

def generate_mock_pose_data(num_frames: int = 10) -> pd.DataFrame:
    """
    Generates a mock DataFrame mimicking multi-agent pose tracking data (e.g., DeepLabCut).
    The data uses a MultiIndex for (Coordinate, Mouse ID, Body Part).
    """
    body_parts = ['nose', 'ear_left', 'ear_right', 'centroid', 'tail_base']
    coords = ['x', 'y']
    
    # Create the MultiIndex for the columns
    columns = pd.MultiIndex.from_product(
        [coords, [M1_ID, M2_ID], body_parts], 
        names=['Coordinate', 'MouseID', 'BodyPart']
    )
    
    data = []
    
    # Generate data for a simple scenario (M1 slightly behind M2, both moving forward)
    for frame in range(num_frames):
        t = frame * 0.1 # Time step
        
        # --- Mouse 1 (The focal animal) ---
        # M1 is facing 45 degrees (up-right) and moving from (100, 100)
        m1_base_x, m1_base_y = 100 + 5 * t, 100 + 5 * t
        
        m1_nose_x = m1_base_x + 5
        m1_nose_y = m1_base_y + 5
        m1_tail_base_x = m1_base_x - 5
        m1_tail_base_y = m1_base_y - 5
        m1_centroid_x = m1_base_x
        m1_centroid_y = m1_base_y
        
        # Define mock ear coordinates based on centroid location
        m1_ear_left_x = m1_centroid_x + 2 
        m1_ear_left_y = m1_centroid_y + 4
        m1_ear_right_x = m1_centroid_x + 4
        m1_ear_right_y = m1_centroid_y - 2
        
        # --- Mouse 2 (The interacting animal) ---
        # M2 is slightly ahead and to the right of M1, moving slightly faster
        m2_base_x, m2_base_y = 105 + 6 * t, 105 + 6 * t
        
        m2_nose_x = m2_base_x + 6
        m2_nose_y = m2_base_y + 6
        m2_tail_base_x = m2_base_x - 4
        m2_tail_base_y = m2_base_y - 4
        m2_centroid_x = m2_base_x
        m2_centroid_y = m2_base_y

        # Define mock ear coordinates based on centroid location
        m2_ear_left_x = m2_centroid_x + 3
        m2_ear_left_y = m2_centroid_y + 5
        m2_ear_right_x = m2_centroid_x + 5
        m2_ear_right_y = m2_centroid_y - 3
        
        # x-coordinates for M1, M2 across all body parts (in the order of the columns MultiIndex)
        x_values = [
            m1_nose_x, m2_nose_x, 
            m1_ear_left_x, m2_ear_left_x, 
            m1_ear_right_x, m2_ear_right_x, 
            m1_centroid_x, m2_centroid_x, 
            m1_tail_base_x, m2_tail_base_x
        ]
        
        # y-coordinates for M1, M2 across all body parts (in the order of the columns MultiIndex)
        y_values = [
            m1_nose_y, m2_nose_y, 
            m1_ear_left_y, m2_ear_left_y, 
            m1_ear_right_y, m2_ear_right_y, 
            m1_centroid_y, m2_centroid_y, 
            m1_tail_base_y, m2_tail_base_y
        ]
                    
        data.append(x_values + y_values)


    # The column order is x/M1/nose, x/M2/nose, ..., y/M1/tail_base, y/M2/tail_base
    df = pd.DataFrame(data, columns=columns)
    df.index.name = 'Frame'
    return df

def calculate_angle_from_vector(V_x: np.array, V_y: np.array) -> np.array:
    """
    Calculates the angle (in radians) of a 2D vector relative to the positive x-axis.
    Returns value in the range [-pi, pi].
    """
    return np.arctan2(V_y, V_x)

def calculate_relative_angle(tracking_df: pd.DataFrame, m1_id: int, m2_id: int) -> Tuple[np.array, pd.DataFrame]:
    """
    Calculates the relative angle (theta_rel) of Mouse 1 (m1_id) relative to Mouse 2 (m2_id).

    theta_rel is the angle between:
    1. V_Body: (M1_TailBase -> M1_Nose)
    2. V_Interaction: (M1_Centroid -> M2_Centroid)
    """
    print(f"--- Calculating Relative Angle for Mouse {m1_id} relative to Mouse {m2_id} ---")
    
    # --- 1. Define Interaction Vector (V_Interaction: M1 Centroid -> M2 Centroid) ---
    m1_c_x = tracking_df.loc[:, ('x', m1_id, 'centroid')].values
    m1_c_y = tracking_df.loc[:, ('y', m1_id, 'centroid')].values
    m2_c_x = tracking_df.loc[:, ('x', m2_id, 'centroid')].values
    m2_c_y = tracking_df.loc[:, ('y', m2_id, 'centroid')].values
    
    V_Interaction_x = m2_c_x - m1_c_x
    V_Interaction_y = m2_c_y - m1_c_y
    
    # --- 2. Define Mouse Body Vector (V_Body: M1 Tail Base -> M1 Nose) ---
    m1_nose_x = tracking_df.loc[:, ('x', m1_id, BODY_VECTOR_END)].values
    m1_nose_y = tracking_df.loc[:, ('y', m1_id, BODY_VECTOR_END)].values
    m1_tail_base_x = tracking_df.loc[:, ('x', m1_id, BODY_VECTOR_START)].values
    m1_tail_base_y = tracking_df.loc[:, ('y', m1_id, BODY_VECTOR_START)].values
    
    V_Body_x = m1_nose_x - m1_tail_base_x
    V_Body_y = m1_nose_y - m1_tail_base_y
    
    # --- 3. Calculate Angles in World Coordinates ---
    theta_interaction = calculate_angle_from_vector(V_Interaction_x, V_Interaction_y)
    theta_body = calculate_angle_from_vector(V_Body_x, V_Body_y)
    
    # --- 4. Calculate and Normalize Relative Angle ---
    theta_rel = theta_body - theta_interaction
    theta_rel_norm = (theta_rel + np.pi) % (2 * np.pi) - np.pi
    
    # --- 5. Calculate Inter-Mouse Distance (IMD) ---
    IMD = np.sqrt(V_Interaction_x**2 + V_Interaction_y**2)
    
    # --- 6. Calculate Velocity for M1 and M2 ---
    
    # Helper function to calculate velocity for a mouse ID
    def calculate_velocity(mouse_id: int) -> np.array:
        x_coords = tracking_df.loc[:, ('x', mouse_id, 'centroid')].values
        y_coords = tracking_df.loc[:, ('y', mouse_id, 'centroid')].values
        
        # Calculate displacement (delta_x, delta_y)
        # Using prepend=x_coords[0] ensures the output array size matches the input size
        delta_x = np.diff(x_coords, prepend=x_coords[0]) 
        delta_y = np.diff(y_coords, prepend=y_coords[0]) 
        
        # Velocity = distance / time (time = 1/FRAME_RATE)
        velocity = np.sqrt(delta_x**2 + delta_y**2) * FRAME_RATE
        
        # Set the first frame's velocity to 0 (since no prior frame exists)
        velocity[0] = 0.0
        return velocity

    V_M1 = calculate_velocity(m1_id)
    V_M2 = calculate_velocity(m2_id)
    
    # Create a DataFrame for all calculated features
    feature_components = pd.DataFrame({
        # Relative Angle Components
        'V_Body_x': V_Body_x, 'V_Body_y': V_Body_y, 'Theta_Body_rad': theta_body,
        'V_Int_x': V_Interaction_x, 'V_Int_y': V_Interaction_y, 'Theta_Int_rad': theta_interaction,
        'Theta_Rel_rad': theta_rel_norm,
        
        # New Features
        'IMD': IMD,
        f'V_M{m1_id}': V_M1,
        f'V_M{m2_id}': V_M2,
    }, index=tracking_df.index)
    
    return theta_rel_norm, feature_components

def create_and_save_submission(feature_df: pd.DataFrame, filename: str):
    """
    Creates a mock submission file from the calculated features.
    
    FIX: Enforces 'id' as integer and uses float_format for precise control over 'prediction' decimals.
    """
    print(f"\n--- Generating required submission file: {filename} ---")
    
    # Convert relative angle radians to degrees before using it as the mock prediction
    # This must be done if it hasn't been done in the main block.
    if 'Theta_Rel_deg' not in feature_df.columns:
        feature_df['Theta_Rel_deg'] = np.rad2deg(feature_df['Theta_Rel_rad'].values)

    # 1. Create a unique ID for each row and ensure it is an INTEGER
    submission_ids = feature_df.index.to_numpy().astype(int)
    
    # 2. Extract the mock prediction feature
    mock_predictions = feature_df['Theta_Rel_deg'].values
    
    # 3. Create the submission DataFrame
    submission_df = pd.DataFrame({
        # NOTE: Verify if the competition uses 'id' and 'prediction' or other names (e.g., 'ID', 'Target').
        'id': submission_ids, 
        'prediction': mock_predictions 
    })
    
    # 4. Save to CSV
    # Use float_format='%.6f' to force 6 decimal places for precision compliance
    submission_df.to_csv(
        filename, 
        index=False, 
        float_format='%.6f' # CRITICAL: Ensures precision is fixed for scoring engine
    )
    
    print(f"File saved successfully to {os.path.abspath(filename)}. Competition requirement met.")


# --- Execution Block ---
if __name__ == "__main__":
    # Generate mock data for 10 frames
    N_FRAMES = 10
    raw_data_df = generate_mock_pose_data(N_FRAMES)
    
    print("--- Sample of Raw Mock Pose Data (First 5 Frames) ---")
    print(raw_data_df.head().iloc[:, :6]) # Show first 6 columns for brevity
    print("-" * 60)

    # Calculate the relative angle feature and other new features
    relative_angle_rad, feature_components_df = calculate_relative_angle(raw_data_df, M1_ID, M2_ID)

    # Convert relative angle radians to degrees
    feature_components_df['Theta_Rel_deg'] = np.rad2deg(feature_components_df['Theta_Rel_rad'].values)

    print("\n--- Summary of Calculated Features (Mouse 1 relative to Mouse 2) ---")
    print("Features Calculated: Relative Angle, Inter-Mouse Distance (IMD), Mouse Velocities (V_M1, V_M2)")
    print("-" * 80)
    print(feature_components_df[['Theta_Rel_deg', 'IMD', 'V_M1', 'V_M2']])
    print("-" * 80)
    print(f"\nSuccessfully calculated {len(feature_components_df.columns)} features for {N_FRAMES} frames.")
    
    # --- COMPETITION SUBMISSION STEP ---
    create_and_save_submission(feature_components_df, SUBMISSION_FILENAME)




