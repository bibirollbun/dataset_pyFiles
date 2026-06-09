from IPython.display import Image, HTML, display


Image("/kaggle/input/files/Silhouette Scores.png")


Image("/kaggle/input/files/Distribution of Motion Classes.png")


Image("/kaggle/input/model-features/Model Features.png")


import cv2
import base64
mp4_file_path = "/kaggle/input/motions/no motion.mp4"
cap = cv2.VideoCapture(mp4_file_path)
ret, frame = cap.read()
height, width, _ = frame.shape
def display_video(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_encoded = base64.b64encode(video_data).decode('utf-8')
    return HTML(f'<video width="{width}" height="{height}" controls><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>')
display(display_video(mp4_file_path))
cap.release()


import cv2
import base64
mp4_file_path = "/kaggle/input/motions/quick shifts.mp4"
cap = cv2.VideoCapture(mp4_file_path)
ret, frame = cap.read()
height, width, _ = frame.shape
def display_video(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_encoded = base64.b64encode(video_data).decode('utf-8')
    return HTML(f'<video width="{width}" height="{height}" controls><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>')
display(display_video(mp4_file_path))
cap.release()


import cv2
import base64
mp4_file_path = "/kaggle/input/motions/half-field motions.mp4"
cap = cv2.VideoCapture(mp4_file_path)
ret, frame = cap.read()
height, width, _ = frame.shape
def display_video(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_encoded = base64.b64encode(video_data).decode('utf-8')
    return HTML(f'<video width="{width}" height="{height}" controls><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>')
display(display_video(mp4_file_path))
cap.release()


import cv2
import base64
mp4_file_path = "/kaggle/input/motions/quarter-field motions.mp4"
cap = cv2.VideoCapture(mp4_file_path)
ret, frame = cap.read()
height, width, _ = frame.shape
def display_video(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_encoded = base64.b64encode(video_data).decode('utf-8')
    return HTML(f'<video width="{width}" height="{height}" controls><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>')
display(display_video(mp4_file_path))
cap.release()


import cv2
import base64
mp4_file_path = "/kaggle/input/motions/slow shifts.mp4"
cap = cv2.VideoCapture(mp4_file_path)
ret, frame = cap.read()
height, width, _ = frame.shape
def display_video(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_encoded = base64.b64encode(video_data).decode('utf-8')
    return HTML(f'<video width="{width}" height="{height}" controls><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>')
display(display_video(mp4_file_path))
cap.release()


import cv2
import base64
mp4_file_path = "/kaggle/input/motions/motions across the field.mp4"
cap = cv2.VideoCapture(mp4_file_path)
ret, frame = cap.read()
height, width, _ = frame.shape
def display_video(video_path):
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_encoded = base64.b64encode(video_data).decode('utf-8')
    return HTML(f'<video width="{width}" height="{height}" controls><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>')
display(display_video(mp4_file_path))
cap.release()


Image("/kaggle/input/shiny-img/img.png")

