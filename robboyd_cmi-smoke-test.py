# Fixed synthetic test cases with correct demographic columns
# Copy and paste this after loading your predictors in Kaggle

import numpy as np
import polars as pl

print("\n" + "="*60)
print("Comprehensive Synthetic Test Cases (Fixed)")
print("="*60)

# Test 1: Normal IMU-only data
print("\nTest 1: Normal IMU-only data")
try:
    sequence_data = {
        'sequence_id': [10001] * 200,
        'subject': [101] * 200,
        'acc_x': np.random.randn(200).tolist(),
        'acc_y': np.random.randn(200).tolist(),
        'acc_z': (np.random.randn(200) + 9.8).tolist(),
        'rot_w': (np.random.rand(200) * 0.8 + 0.2).tolist(),
        'rot_x': (np.random.randn(200) * 0.1).tolist(),
        'rot_y': (np.random.randn(200) * 0.1).tolist(),
        'rot_z': (np.random.randn(200) * 0.1).tolist(),
    }
    demo_data = {
        'subject': [101],
        'adult_child': [1],
        'age': [25],
        'sex': [1],
        'handedness': [1],
        'height_cm': [175],
        'shoulder_to_wrist_cm': [60],
        'elbow_to_wrist_cm': [26]
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Full sensor data
print("\nTest 2: Full sensor data (IMU + Thermopile + ToF)")
try:
    sequence_data = {
        'sequence_id': [10002] * 200,
        'subject': [102] * 200,
        'acc_x': np.random.randn(200).tolist(),
        'acc_y': np.random.randn(200).tolist(),
        'acc_z': (np.random.randn(200) + 9.8).tolist(),
        'rot_w': (np.random.rand(200) * 0.8 + 0.2).tolist(),
        'rot_x': (np.random.randn(200) * 0.1).tolist(),
        'rot_y': (np.random.randn(200) * 0.1).tolist(),
        'rot_z': (np.random.randn(200) * 0.1).tolist(),
    }
    # Add thermopile
    for i in range(1, 6):
        sequence_data[f'thm_{i}'] = (np.random.rand(200) * 30 + 20).tolist()
    # Add ToF
    for s in range(1, 6):
        for v in range(64):
            sequence_data[f'tof_{s}_v{v}'] = np.random.randint(0, 255, 200).tolist()
    
    demo_data = {
        'subject': [102],
        'adult_child': [1],
        'age': [35],
        'sex': [0],
        'handedness': [0],
        'height_cm': [165],
        'shoulder_to_wrist_cm': [55],
        'elbow_to_wrist_cm': [24]
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Extreme values
print("\nTest 3: Extreme sensor values (test clipping)")
try:
    sequence_data = {
        'sequence_id': [10003] * 200,
        'subject': [103] * 200,
        'acc_x': (np.random.randn(200) * 1000).tolist(),
        'acc_y': (np.random.randn(200) * 1000).tolist(),
        'acc_z': (np.random.randn(200) * 1000).tolist(),
        'rot_w': np.random.rand(200).tolist(),
        'rot_x': (np.random.randn(200) * 100).tolist(),
        'rot_y': (np.random.randn(200) * 100).tolist(),
        'rot_z': (np.random.randn(200) * 100).tolist(),
    }
    for i in range(1, 6):
        sequence_data[f'thm_{i}'] = (np.random.rand(200) * 1000).tolist()
    
    demo_data = {
        'subject': [103],
        'adult_child': [1],
        'age': [99],
        'sex': [1],
        'handedness': [1],
        'height_cm': [250],
        'shoulder_to_wrist_cm': [100],
        'elbow_to_wrist_cm': [45]
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 4: Missing columns
print("\nTest 4: Missing rotation sensors")
try:
    sequence_data = {
        'sequence_id': [10004] * 200,
        'subject': [104] * 200,
        'acc_x': np.random.randn(200).tolist(),
        'acc_y': np.random.randn(200).tolist(),
        'acc_z': np.random.randn(200).tolist(),
        # Missing rot_w, rot_x, rot_y, rot_z
    }
    demo_data = {
        'subject': [104],
        'adult_child': [0],
        'age': [8],
        'sex': [1],
        'handedness': [1],
        'height_cm': [120],
        'shoulder_to_wrist_cm': [35],
        'elbow_to_wrist_cm': [15]
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 5: All zeros (dead sensors)
print("\nTest 5: All zero values")
try:
    sequence_data = {
        'sequence_id': [10005] * 200,
        'subject': [105] * 200,
        'acc_x': [0.0] * 200,
        'acc_y': [0.0] * 200,
        'acc_z': [0.0] * 200,
        'rot_w': [0.0] * 200,
        'rot_x': [0.0] * 200,
        'rot_y': [0.0] * 200,
        'rot_z': [0.0] * 200,
    }
    demo_data = {
        'subject': [105],
        'adult_child': [1],
        'age': [50],
        'sex': [0],
        'handedness': [0],
        'height_cm': [170],
        'shoulder_to_wrist_cm': [58],
        'elbow_to_wrist_cm': [25]
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 6: NaN and Inf values
print("\nTest 6: NaN and Inf values")
try:
    sequence_data = {
        'sequence_id': [10006] * 200,
        'subject': [106] * 200,
        'acc_x': [float('nan')] * 100 + [float('inf')] * 100,
        'acc_y': [float('-inf')] * 100 + np.random.randn(100).tolist(),
        'acc_z': np.random.randn(200).tolist(),
        'rot_w': [float('nan')] * 200,
        'rot_x': np.random.randn(200).tolist(),
        'rot_y': np.random.randn(200).tolist(),
        'rot_z': np.random.randn(200).tolist(),
    }
    demo_data = {
        'subject': [106],
        'adult_child': [1],
        'age': [30],
        'sex': [1],
        'handedness': [1],
        'height_cm': [170],
        'shoulder_to_wrist_cm': [58],
        'elbow_to_wrist_cm': [25]
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 7: Wrong sequence length
print("\nTest 7: Wrong sequence length (150 instead of 200)")
try:
    sequence_data = {
        'sequence_id': [10007] * 150,
        'subject': [107] * 150,
        'acc_x': np.random.randn(150).tolist(),
        'acc_y': np.random.randn(150).tolist(),
        'acc_z': np.random.randn(150).tolist(),
        'rot_w': np.random.rand(150).tolist(),
        'rot_x': np.random.randn(150).tolist(),
        'rot_y': np.random.randn(150).tolist(),
        'rot_z': np.random.randn(150).tolist(),
    }
    demo_data = {
        'subject': [107],
        'adult_child': [1],
        'age': [30],
        'sex': [1],
        'handedness': [1],
        'height_cm': [180],
        'shoulder_to_wrist_cm': [62],
        'elbow_to_wrist_cm': [27]
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 8: Empty sequence
print("\nTest 8: Empty sequence (0 rows)")
try:
    sequence_data = {
        'sequence_id': [],
        'subject': [],
        'acc_x': [],
        'acc_y': [],
        'acc_z': [],
        'rot_w': [],
        'rot_x': [],
        'rot_y': [],
        'rot_z': [],
    }
    demo_data = {
        'subject': [108],
        'adult_child': [1],
        'age': [25],
        'sex': [1],
        'handedness': [1],
        'height_cm': [175],
        'shoulder_to_wrist_cm': [60],
        'elbow_to_wrist_cm': [26]
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 9: Missing demographic data
print("\nTest 9: Missing some demographic columns")
try:
    sequence_data = {
        'sequence_id': [10009] * 200,
        'subject': [109] * 200,
        'acc_x': np.random.randn(200).tolist(),
        'acc_y': np.random.randn(200).tolist(),
        'acc_z': np.random.randn(200).tolist(),
        'rot_w': np.random.rand(200).tolist(),
        'rot_x': np.random.randn(200).tolist(),
        'rot_y': np.random.randn(200).tolist(),
        'rot_z': np.random.randn(200).tolist(),
    }
    demo_data = {
        'subject': [109],
        'adult_child': [1],
        'age': [40],
        # Missing sex, handedness, height_cm, shoulder_to_wrist_cm, elbow_to_wrist_cm
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 10: Very long sequence with all sensors
print("\nTest 10: Very long sequence (500 timesteps) with all sensors")
try:
    sequence_data = {
        'sequence_id': [10010] * 500,
        'subject': [110] * 500,
        'acc_x': np.random.randn(500).tolist(),
        'acc_y': np.random.randn(500).tolist(),
        'acc_z': np.random.randn(500).tolist(),
        'rot_w': np.random.rand(500).tolist(),
        'rot_x': np.random.randn(500).tolist(),
        'rot_y': np.random.randn(500).tolist(),
        'rot_z': np.random.randn(500).tolist(),
    }
    # Add all sensors
    for i in range(1, 6):
        sequence_data[f'thm_{i}'] = np.random.rand(500).tolist()
    for s in range(1, 6):
        for v in range(64):
            sequence_data[f'tof_{s}_v{v}'] = np.random.randint(0, 255, 500).tolist()
    
    demo_data = {
        'subject': [110],
        'adult_child': [1],
        'age': [45],
        'sex': [0],
        'handedness': [1],
        'height_cm': [162],
        'shoulder_to_wrist_cm': [54],
        'elbow_to_wrist_cm': [23]
    }
    result = predict(pl.DataFrame(sequence_data), pl.DataFrame(demo_data))
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")


