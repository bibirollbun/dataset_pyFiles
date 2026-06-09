!python --version


# Install Required Libraries

!pip install --no-deps facenet-pytorch -q

!pip install --no-deps face-recognition -q

!pip install --no-deps face_recognition_models -q

!pip install --no-deps scikit-image -q

!pip install --no-deps tensorboardX -q

!pip install --no-deps pytorch_toolbelt -q

!git clone https://github.com/NVIDIA/apex
%cd apex
!pip install -r requirements.txt -q


import sys
sys.path.append('/kaggle/input/preprocessing/')
sys.path.append('/kaggle/input/')
sys.path.append('/kaggle/input/training/')
sys.path.append('/kaggle/input/training/datasets/')
sys.path.append('/kaggle/input/training/pipelines/')
sys.path.append('/kaggle/input/training/tools/')
sys.path.append('/kaggle/input/training/transforms/')
sys.path.append('/kaggle/input/weights/')
sys.path.append('/kaggle/input/weights/weights/')
sys.path.append('/kaggle/input/weights/weights')
sys.path.append('/kaggle/input/weights')
sys.path.append('/kaggle/input/weights/')
sys.path.append('/kaggle/input/weights/weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36')
sys.path.append('/kaggle/input/weights/weights/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19')
sys.path.append('/kaggle/input/weights/weights/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29')
sys.path.append('/kaggle/input/weights/weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23')
sys.path.append('/kaggle/input/weights/weights/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40')
sys.path.append('/kaggle/input/weights/weights/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37')
sys.path.append('/kaggle/input/weights/weights/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31')


import training


!pip uninstall albumentations -y


!pip install albumentations==2.0.4 -q


import albumentations as A
A.__version__


!python --version


import argparse
import os
import re
import time
from tqdm import tqdm
import torch
import pandas as pd
from training.zoo.classifiers import DeepFakeClassifier
import os

import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.augmentations.functional import image_compression
from facenet_pytorch.models.mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor

from torchvision.transforms import Normalize
import json



mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)


class VideoReader:
    """Helper class for reading one or more frames from a video file."""

    def __init__(self, verbose=True, insets=(0, 0)):
        """Creates a new VideoReader.

        Arguments:
            verbose: whether to print warnings and error messages
            insets: amount to inset the image by, as a percentage of
                (width, height). This lets you "zoom in" to an image
                to remove unimportant content around the borders.
                Useful for face detection, which may not work if the
                faces are too small.
        """
        self.verbose = verbose
        self.insets = insets

    def read_frames(self, path, num_frames, jitter=0, seed=None):
        """Reads frames that are always evenly spaced throughout the video.

        Arguments:
            path: the video file
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
            jitter: if not 0, adds small random offsets to the frame indices;
                this is useful so we don't always land on even or odd frames
            seed: random seed for jittering; if you set this to a fixed value,
                you probably want to set it only on the first video
        """
        assert num_frames > 0

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return None

        frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int32)
        if jitter > 0:
            np.random.seed(seed)
            jitter_offsets = np.random.randint(-jitter, jitter, len(frame_idxs))
            frame_idxs = np.clip(frame_idxs + jitter_offsets, 0, frame_count - 1)

        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def read_random_frames(self, path, num_frames, seed=None):
        """Picks the frame indices at random.

        Arguments:
            path: the video file
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
        """
        assert num_frames > 0
        np.random.seed(seed)

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return None

        frame_idxs = sorted(np.random.choice(np.arange(0, frame_count), num_frames))
        result = self._read_frames_at_indices(path, capture, frame_idxs)

        capture.release()
        return result

    def read_frames_at_indices(self, path, frame_idxs):
        """Reads frames from a video and puts them into a NumPy array.

        Arguments:
            path: the video file
            frame_idxs: a list of frame indices. Important: should be
                sorted from low-to-high! If an index appears multiple
                times, the frame is still read only once.

        Returns:
            - a NumPy array of shape (num_frames, height, width, 3)
            - a list of the frame indices that were read

        Reading stops if loading a frame fails, in which case the first
        dimension returned may actually be less than num_frames.

        Returns None if an exception is thrown for any reason, or if no
        frames were read.
        """
        assert len(frame_idxs) > 0
        capture = cv2.VideoCapture(path)
        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def _read_frames_at_indices(self, path, capture, frame_idxs):
        try:
            frames = []
            idxs_read = []
            for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
                # Get the next frame, but don't decode if we're not using it.
                ret = capture.grab()
                if not ret:
                    if self.verbose:
                        print("Error grabbing frame %d from movie %s" % (frame_idx, path))
                    break

                # Need to look at this frame?
                current = len(idxs_read)
                if frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        if self.verbose:
                            print("Error retrieving frame %d from movie %s" % (frame_idx, path))
                        break

                    frame = self._postprocess_frame(frame)
                    frames.append(frame)
                    idxs_read.append(frame_idx)

            if len(frames) > 0:
                return np.stack(frames), idxs_read
            if self.verbose:
                print("No frames read from movie %s" % path)
            return None
        except:
            if self.verbose:
                print("Exception while reading movie %s" % path)
            return None

    def read_middle_frame(self, path):
        """Reads the frame from the middle of the video."""
        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        result = self._read_frame_at_index(path, capture, frame_count // 2)
        capture.release()
        return result

    def read_frame_at_index(self, path, frame_idx):
        """Reads a single frame from a video.

        If you just want to read a single frame from the video, this is more
        efficient than scanning through the video to find the frame. However,
        for reading multiple frames it's not efficient.

        My guess is that a "streaming" approach is more efficient than a
        "random access" approach because, unless you happen to grab a keyframe,
        the decoder still needs to read all the previous frames in order to
        reconstruct the one you're asking for.

        Returns a NumPy array of shape (1, H, W, 3) and the index of the frame,
        or None if reading failed.
        """
        capture = cv2.VideoCapture(path)
        result = self._read_frame_at_index(path, capture, frame_idx)
        capture.release()
        return result

    def _read_frame_at_index(self, path, capture, frame_idx):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if not ret or frame is None:
            if self.verbose:
                print("Error retrieving frame %d from movie %s" % (frame_idx, path))
            return None
        else:
            frame = self._postprocess_frame(frame)
            return np.expand_dims(frame, axis=0), [frame_idx]

    def _postprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.insets[0] > 0:
            W = frame.shape[1]
            p = int(W * self.insets[0])
            frame = frame[:, p:-p, :]

        if self.insets[1] > 0:
            H = frame.shape[1]
            q = int(H * self.insets[1])
            frame = frame[q:-q, :, :]

        return frame


class FaceExtractor:
    def __init__(self, video_read_fn):
        self.video_read_fn = video_read_fn
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device=device)

    def process_videos(self, input_dir, filenames, video_idxs):
        videos_read = []
        frames_read = []
        frames = []
        results = []
        for video_idx in video_idxs:
            # Read the full-size frames from this video.
            filename = filenames[video_idx]
            video_path = os.path.join(input_dir, filename)
            result = self.video_read_fn(video_path)
            # Error? Then skip this video.
            if result is None: continue

            videos_read.append(video_idx)

            # Keep track of the original frames (need them later).
            my_frames, my_idxs = result

            frames.append(my_frames)
            frames_read.append(my_idxs)
            for i, frame in enumerate(my_frames):
                h, w = frame.shape[:2]
                img = Image.fromarray(frame.astype(np.uint8))
                img = img.resize(size=[s // 2 for s in img.size])

                batch_boxes, probs = self.detector.detect(img, landmarks=False)

                faces = []
                scores = []
                if batch_boxes is None:
                    continue
                for bbox, score in zip(batch_boxes, probs):
                    if bbox is not None:
                        xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                        w = xmax - xmin
                        h = ymax - ymin
                        p_h = h // 3
                        p_w = w // 3
                        crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                        faces.append(crop)
                        scores.append(score)

                frame_dict = {"video_idx": video_idx,
                              "frame_idx": my_idxs[i],
                              "frame_w": w,
                              "frame_h": h,
                              "faces": faces,
                              "scores": scores}
                results.append(frame_dict)

        return results

    def process_video(self, video_path):
        """Convenience method for doing face extraction on a single video."""
        input_dir = os.path.dirname(video_path)
        filenames = [os.path.basename(video_path)]
        return self.process_videos(input_dir, filenames, [0])



def confident_strategy(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    # 11 frames are detected as fakes with high probability
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)

strategy = confident_strategy


def put_to_center(img, input_size):
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0], start_w: start_w + img.shape[1], :] = img
    return image


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


def predict_on_video(face_extractor, video_path, batch_size, input_size, models, strategy=np.mean,
                     apply_compression=False):
    batch_size *= 4
    try:
        faces = face_extractor.process_video(video_path)
        if len(faces) > 0:
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = put_to_center(resized_face, input_size)
                    if apply_compression:
                        resized_face = image_compression(resized_face, quality=90, image_type=".jpg")
                    if n + 1 < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        pass
            if n > 0:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                x = torch.tensor(x, device=device).float()
                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))
                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)
                # Make a prediction, then take the average.
                with torch.no_grad():
                    preds = []
                    for model in models:
                        y_pred = model(x[:n].half())
                        y_pred = torch.sigmoid(y_pred.squeeze())
                        bpred = y_pred[:n].cpu().numpy()
                        preds.append(strategy(bpred))
                    return np.mean(preds)
    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5

def predict_on_video_set(face_extractor, videos, input_size, num_workers, test_dir, frames_per_video, models,
                         strategy=np.mean,
                         apply_compression=False):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(
            face_extractor=face_extractor,
            video_path=os.path.join(test_dir, filename),
            input_size=input_size,
            batch_size=frames_per_video,
            models=models,
            strategy=strategy,
            apply_compression=apply_compression
        )
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = list(tqdm(ex.map(process_file, range(len(videos))), total=len(videos), desc="Predicting Videos"))

    return predictions




def deepfakedetector(video_folder_path, output_file='predictions.csv'):
    parser = argparse.ArgumentParser(description="Predict test videos")
    arg = parser.add_argument
    
    arg('--weights-dir', type=str, default="/kaggle/input/weights", help="path to directory with checkpoints")

    # Fix: Provide models as a list instead of a single string
    arg('--models', nargs='+', default=[
        "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
        "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
        "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
        "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
        "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
        "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23"
    ], help="List of checkpoint files")

    # Fix: Remove required=True since a default is provided
    arg('--test-dir', type=str, default=video_folder_path, help="path to directory with videos")

    # arg('--output', type=str, default="submission.csv", help="path to output csv")

    args, _ = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    model_paths = [os.path.join(args.weights_dir, model) for model in args.models]
    for path in tqdm(model_paths, desc="Loading Models"):
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
        model.eval()
        del checkpoint
        models.append(model.half())
    # for path in model_paths:
    #     model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
    #     print("loading state dict {}".format(path))
    #     checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    #     state_dict = checkpoint.get("state_dict", checkpoint)
    #     model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    #     model.eval()
    #     del checkpoint
    #     models.append(model.half())

    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    input_size = 380
    strategy = confident_strategy
    stime = time.time()

    test_videos = sorted([x for x in os.listdir(args.test_dir) if x.endswith(".mp4")])
    print("Predicting {} videos".format(len(test_videos)))
    predictions = predict_on_video_set(face_extractor=face_extractor, input_size=input_size, models=models,
                                       strategy=strategy, frames_per_video=frames_per_video, videos=test_videos,
                                       num_workers=1, test_dir=args.test_dir)
    
    pred_df = pd.DataFrame({"filename": test_videos, "label": predictions})

    pred_df.to_csv(output_file, index=False)
    print("Elapsed:", time.time() - stime)
    
    return pred_df


pred_df = deepfakedetector(
                video_folder_path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos',
                output_file='predictions.csv')


print(pred_df.isna().any())
pred_df.head()


# Convert the JSON data of videos to CSV
with open('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json', 'r') as f:
    json_data = json.load(f)

target_df = pd.DataFrame.from_dict(json_data, orient='index')
target_df.reset_index(inplace=True)
target_df.rename(columns={'index': 'filename'}, inplace=True)
target_df = target_df[['filename','label']]
csv_path = 'metadata.csv'
target_df.to_csv(csv_path, index=False)

print(f"CSV saved to: {csv_path}")



print(target_df.isna().any())
target_df.head()


threshold = 0.5
pred_df['pred'] = (pred_df['label'] >= threshold).astype(int)

target_df['label'] = target_df['label'].map({'REAL': 0, 'FAKE': 1})

merged_df = pd.merge(pred_df, target_df, on='filename', how='inner', suffixes=('_pred', '_true'))
merged_df['target'] = merged_df['label_true'].copy(deep=True)
merged_df = merged_df[['filename', 'pred', 'target']]

merged_df.head()


# Is any NaN value in any row ?
merged_df.isna().any()


correct = (merged_df['pred'] == merged_df['target']).sum()
total = len(merged_df)
accuracy = correct / total

print(f"Test Accuracy: {accuracy*100:.2f}%")


incorrect_df = merged_df[merged_df['pred'] != merged_df['target']]

incorrect_df = pd.merge(
    incorrect_df,
    pred_df[['filename', 'label']],
    on='filename',
    how='left'
)
print(incorrect_df[['filename', 'label', 'pred', 'target']])


