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


!pip install mediapipe


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
from pathlib import Path
import mediapipe
from mediapipe.framework.formats import landmark_pb2
from tqdm.notebook import tqdm
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
matplotlib.rcParams['savefig.pad_inches'] = 0


base_dir = Path('/kaggle/input/asl-fingerspelling/')


from matplotlib import animation, rc
rc('animation', html='jshtml')

def create_animation(images):
    fig = plt.figure(figsize=(6, 9))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im=ax.imshow(images[0], cmap="gray")
    plt.close(fig)
    
    def animate_func(i):
        im.set_array(images[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames=len(images), interval=1000/10)


train_df = pd.read_csv(base_dir / 'train.csv')
train_df.head()


def get_random_sequence(phrase_idx=None):
    if phrase_idx == None:
        phrase_idx = np.random.randint(len(train_df))

    selected_row = train_df.iloc[phrase_idx]
    file_id = selected_row['file_id']
    sequence_id = selected_row['sequence_id']
    phrase = selected_row['phrase']
    
    parquet_file = pq.ParquetFile(base_dir / 'train_landmarks' / f"{str(file_id)}.parquet")
    
    dataset = pq.read_table(
        base_dir / 'train_landmarks' / f"{str(file_id)}.parquet",
        filters=[
            [('sequence_id', '=', sequence_id)],
        ]
    )
    
    sequence_df = dataset.to_pandas()
    return phrase_idx, phrase, sequence_df


phrase_idx = 3
selected_row = train_df.iloc[phrase_idx]
file_id = selected_row['file_id']
sequence_id = selected_row['sequence_id']
phrase = selected_row['phrase']
print(f"row: {selected_row.head()}")


parquet_file = pq.ParquetFile(base_dir / 'train_landmarks' / f"{str(file_id)}.parquet")


dataset = pq.read_table(
    base_dir / 'train_landmarks' / f"{str(file_id)}.parquet",
    filters=[
        [('sequence_id', '=', sequence_id)],
    ]
)
sequence_df=dataset.to_pandas()
sequence_df.head()


mp_pose = mediapipe.solutions.pose
mp_hands = mediapipe.solutions.hands
mp_face_mesh = mediapipe.solutions.face_mesh
mp_drawing = mediapipe.solutions.drawing_utils 
mp_drawing_styles = mediapipe.solutions.drawing_styles


def get_hands(seq_df):
    images = []
    all_hand_landmarks = []
    for seq_idx in range(len(seq_df)):
        x_pose = seq_df.iloc[seq_idx].filter(regex="x_right_hand.*").values
        y_pose = seq_df.iloc[seq_idx].filter(regex="y_right_hand.*").values
        z_pose = seq_df.iloc[seq_idx].filter(regex="z_right_hand.*").values
         
        right_hand_image = np.zeros((900, 600, 3))

        right_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_pose, y_pose, z_pose):
            right_hand_landmarks.landmark.add(x=x, y=y, z=z)
        
        mp_drawing.draw_landmarks(
                right_hand_image,
                right_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
        x_pose = seq_df.iloc[seq_idx].filter(regex="x_left_hand.*").values
        y_pose = seq_df.iloc[seq_idx].filter(regex="y_left_hand.*").values
        z_pose = seq_df.iloc[seq_idx].filter(regex="z_left_hand.*").values
        
        left_hand_image = np.zeros((900, 600, 3))
        
        left_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_pose, y_pose, z_pose):
            left_hand_landmarks.landmark.add(x=x, y=y, z=z)

        mp_drawing.draw_landmarks(
                left_hand_image,
                left_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
        images.append([right_hand_image.astype(np.uint8), left_hand_image.astype(np.uint8)])
        all_hand_landmarks.append([right_hand_landmarks, left_hand_landmarks])
    return images, all_hand_landmarks


hand_images, hand_landmarks = get_hands(sequence_df)


create_animation(np.array(hand_images)[:, 0])


create_animation(np.array(hand_images)[:, 1])


x_pose = sequence_df.iloc[0].filter(regex="x_right_hand.*").values
y_pose = sequence_df.iloc[0].filter(regex="y_right_hand.*").values
z_pose = sequence_df.iloc[0].filter(regex="z_right_hand.*").values

right_hand_image = np.zeros((900, 600, 3))

right_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
for x, y, z in zip(x_pose, y_pose, z_pose):
    print(x)
    print(y)
    print(z)
    right_hand_landmarks.landmark.add(x=x, y=y, z=z)


import numpy as np

# Ảnh ban đầu toàn màu đen
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Vẽ landmarks tay (giả sử right_hand_landmarks đã được khởi tạo hợp lệ)
mp_drawing.draw_landmarks(
    image, right_hand_landmarks, mp_hands.HAND_CONNECTIONS
)

# Tìm các pixel đã bị vẽ (khác [0, 0, 0])
non_zero_pixels = np.argwhere(np.any(image != 0, axis=-1))

# In một vài pixel để kiểm tra
print(f"Số pixel đã vẽ: {len(non_zero_pixels)}")
print("Một vài pixel đã được vẽ:")
print(non_zero_pixels[:10])  # in 10 điểm đầu tiên



def get_face(seq_df):
    images = []
    all_face_landmarks = []
    for seq_idx in range(len(seq_df)):
        x_face = seq_df.iloc[seq_idx].filter(regex="x_face.*").values
        y_face = seq_df.iloc[seq_idx].filter(regex="y_face.*").values
        z_face = seq_df.iloc[seq_idx].filter(regex="z_face.*").values

        annotated_image = np.zeros((900, 600, 3))

        face_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_face, y_face, z_face):
            face_landmarks.landmark.add(x=x, y=y, z=z)

        mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())

        images.append(annotated_image.astype(np.uint8))
        all_face_landmarks.append(face_landmarks)
    return images, all_face_landmarks


face_images, face_landmarks = get_face(sequence_df)


create_animation(face_images)


def get_pose(seq_df):
    images = []
    all_pose_landmarks = []
    for seq_idx in range(len(seq_df)):
        x_pose = seq_df.iloc[seq_idx].filter(regex="x_pose.*").values
        y_pose = seq_df.iloc[seq_idx].filter(regex="y_pose.*").values
        z_pose = seq_df.iloc[seq_idx].filter(regex="z_pose.*").values

        annotated_image = np.zeros((900, 600, 3))
        
        data_points = []
        for x, y, z in zip(x_pose, y_pose, z_pose):
            data_points.append(np.array([x, y, z]))

        pose_landmarks = landmark_pb2.NormalizedLandmarkList()
        for row in data_points:
            pose_landmarks.landmark.add(x=row[0], y=row[1], z=row[2])

        mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        images.append(annotated_image.astype(np.uint8))
        all_pose_landmarks.append(pose_landmarks)
    return images, all_pose_landmarks


pose_images, pose_landmarks = get_pose(sequence_df)


create_animation(pose_images)


def convert_landmark_to_npy(landmarklist):
    return np.array([np.array([landmark.x, landmark.y, landmark.z]) for landmark in landmarklist.landmark])


def get_all_images(seq_df):
    pose_images, pose_landmarks = get_pose(sequence_df)
    hand_images, hand_landmarks = get_hands(sequence_df)
    face_images, face_landmarks = get_face(sequence_df)
    
    all_images = []
    all_landmarks_data = []
    all_landmarks = []
    for seq_idx in tqdm(range(len(pose_landmarks))):
        pose_landmark_np = convert_landmark_to_npy(pose_landmarks[seq_idx])
        right_hand_landmark_np = convert_landmark_to_npy(hand_landmarks[seq_idx][0])
        left_hand_landmark_np = convert_landmark_to_npy(hand_landmarks[seq_idx][1])
        face_landmark_np = convert_landmark_to_npy(face_landmarks[seq_idx])
        
        # Pool all landmarks together to find min and max coordinates
        pooled_landmarks = np.vstack((pose_landmark_np, right_hand_landmark_np, left_hand_landmark_np, face_landmark_np))
        pooled_min = np.nanmin(pooled_landmarks, axis=0)
        pooled_max = np.nanmax(pooled_landmarks, axis=0)
        
        # Use the max of x and y scaling to proportionally scale the image. We don't need to scale z for 2D image
        # There is an elegant way to achieve the same result as below with Matrix transformations. Might re-do this if there is interest.
        scaling_factor = np.nanmax(pooled_max[:2])
        pooled_scaled_min = np.nanmin(pooled_landmarks / scaling_factor, axis=0)

        pose_landmark_np_normed = (pose_landmark_np / scaling_factor) - pooled_scaled_min
        
        # Center the image around shoulder and hips. Makes for a better visualization
        x_shift = ((1-(pose_landmark_np_normed[23]+pose_landmark_np_normed[24]))/2)[0]
        axis_shift = np.array([x_shift, 0, 0])
        
        pose_landmark_np_normed = pose_landmark_np_normed + axis_shift
        right_hand_landmark_np_normed = (right_hand_landmark_np / scaling_factor) - pooled_scaled_min + axis_shift
        left_hand_landmark_np_normed = (left_hand_landmark_np / scaling_factor) - pooled_scaled_min  + axis_shift
        face_landmark_np_normed = (face_landmark_np / scaling_factor) - pooled_scaled_min + axis_shift
        
        # Now that we have scaled and shifted the landmarks to fit into a [0, 1] range, we can start plotting them using mediapipe APIs
        # BG image with zeros
        image = np.zeros((900, 600, 3))
        
        # Pose
        pose_landmark_np_normed_z = landmark_pb2.LandmarkList()
        for row in pose_landmark_np_normed:
            pose_landmark_np_normed_z.landmark.add(x=row[0], y=row[1], z=row[2])

        mp_drawing.draw_landmarks(
                    image,
                    pose_landmark_np_normed_z,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Right hand
        right_hand_landmark_np_normed_z = landmark_pb2.LandmarkList()
        for row in right_hand_landmark_np_normed:
            right_hand_landmark_np_normed_z.landmark.add(x=row[0], y=row[1], z=row[2])

        mp_drawing.draw_landmarks(
                    image,
                    right_hand_landmark_np_normed_z,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
        # Left hand
        left_hand_landmark_np_normed_z = landmark_pb2.LandmarkList()
        for row in left_hand_landmark_np_normed:
            left_hand_landmark_np_normed_z.landmark.add(x=row[0], y=row[1], z=row[2])

        mp_drawing.draw_landmarks(
                    image,
                    left_hand_landmark_np_normed_z,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        
        # Face
        face_landmark_np_normed_z = landmark_pb2.LandmarkList()
        for row in face_landmark_np_normed:
            face_landmark_np_normed_z.landmark.add(x=row[0], y=row[1], z=row[2])

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmark_np_normed_z,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmark_np_normed_z,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        
        # Iris data not available. So ignoring the iris visualization.
        
        all_images.append(image.astype(np.uint8))
        all_landmarks_data.append([pose_landmark_np_normed, right_hand_landmark_np_normed_z, left_hand_landmark_np_normed_z, face_landmark_np_normed_z])
        all_landmarks.append([pose_landmark_np_normed_z, right_hand_landmark_np_normed_z, left_hand_landmark_np_normed_z, face_landmark_np_normed_z])
    return all_images, all_landmarks_data, all_landmarks


phrase_idx, phrase, sequence_df = get_random_sequence(0)


print(f"{phrase_idx}: {phrase}")


all_images, all_landmarks_data, all_landmarks = get_all_images(sequence_df)


create_animation(all_images)

