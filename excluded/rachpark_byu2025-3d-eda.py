import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation


# ROOT = "D:/Workspace/Kaggle/2025_BYU/byu-locating-bacterial-flagellar-motors-2025"
ROOT = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"


LABEL = os.path.join(ROOT, "train_labels.csv")

label = pd.read_csv(LABEL)
label.head()


# Image
def get_data(tomo_id, N=30):
    DIR = os.path.join(ROOT, "train", f"tomo_{tomo_id}")
    all_slices = [DIR + "/" + slice for slice in os.listdir(DIR)]
    step = len(all_slices) // N
    select_slices = all_slices[::step][:N]

    # labels
    select_labels = label[label.tomo_id == f"tomo_{tomo_id}"]

    video_size = select_labels[["Array shape (axis 0)", "Array shape (axis 1)", "Array shape (axis 2)"]].iloc[0]
    motor_axis = select_labels[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].reset_index(drop=True).values

    return select_slices, motor_axis, video_size

def create_slice_animation(select_slices, motor_axis, video_size, blurs, figsize=(4, 4), img_size=(64, 64)):
    N = len(select_slices)

    # 원본 좌표를 리사이즈된 좌표로 변환하는 함수
    def transform_coordinates(z, y, x, original_size, new_size):
        z_ratio = (N - 1) / (original_size[0] - 1)
        new_z = int(round(z * z_ratio))
        x_ratio = new_size[0] / original_size[2]
        y_ratio = new_size[1] / original_size[1]
        new_x = int(round(x * x_ratio))
        new_y = int(round(y * y_ratio))
        return new_z, new_y, new_x

    # motor_axis 좌표 변환
    new_coords = []
    for z, y, x in motor_axis:
        new_z, new_y, new_x = transform_coordinates(z, y, x, video_size, img_size)
        new_coords.append((new_z, new_y, new_x))

    # 이미지 전처리 및 리사이징
    resized_slices = []
    for slice_img_path in select_slices:
        img = cv2.imread(slice_img_path)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)  # 작은 이미지에 적합한 리사이즈 방식
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_slices.append(img)

    # blur score 표기
    for i, blur in enumerate(blurs):
        for slice in resized_slices:
            score = blur(slice)
            # 이미지에 blur score 텍스트 추가
            cv2.putText(slice, 
                    f'{blur.__name__}: {score:.2f}', 
                    (10, int(20 + 20 * i * 0.5)),  # 텍스트 위치 (좌상단)
                    cv2.FONT_HERSHEY_SIMPLEX,  # 폰트
                    0.3,  # 폰트 크기
                    (255, 255, 255),  # 텍스트 색상 (흰색)
                    1)  # 텍스트 두께
        
    # Bounding box 그리기
    box_size = 3  # 박스 크기를 더 줄이기
    for coord in new_coords:
        z, y, x = coord
        for i in range(max(0, z-2), min(len(resized_slices), z+3)):  # 사용 프레임 범위를 줄이기
            pt1 = (max(0, x-box_size), max(0, y-box_size))
            pt2 = (min(img_size[0]-1, x+box_size), min(img_size[1]-1, y+box_size))
            cv2.rectangle(resized_slices[i], pt1, pt2, (255, 0, 0), 1)

    # 애니메이션 생성
    fig, ax = plt.subplots(figsize=figsize, dpi=80)  # DPI 감소
    im = ax.imshow(resized_slices[0], interpolation='lanczos')
    ax.axis('off')
    
    def update(frame):
        im.set_array(resized_slices[frame])
        return [im]
    
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(resized_slices),
        interval=200,
        blit=True  # blit 옵션 활성화
    )
    
    plt.close()
    return anim



# Blur
def load_image(path, img_size):
    img = cv2.imread(path)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)  # 작은 이미지에 적합한 리사이즈 방식
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_blur(blur, tomo_id, img_size=(64, 64)):
    PATH = os.path.join(ROOT, "train", f"tomo_{tomo_id}")

    slices = [os.path.join(PATH, slice) for slice in sorted(os.listdir(PATH))]

    blurriness = [blur(load_image(slice, img_size)) for slice in slices]
    
    plt.plot(blurriness)

    # motor point
    z_axis = label[label.tomo_id == f"tomo_{tomo_id}"]["Motor axis 0"].values
    for point in z_axis:
        slice_index = int(point)
        plt.scatter(x=slice_index, y=blurriness[slice_index], c="red")

    plt.show()

def plot_comboned_blur(blurs, tomo_id, img_size=(64, 64)):
    from sklearn.preprocessing import StandardScaler

    PATH = os.path.join(ROOT, "train", f"tomo_{tomo_id}")

    slices = [os.path.join(PATH, slice) for slice in sorted(os.listdir(PATH))]

    blurriness = [0 for _ in slices]
    for blur in blurs:
        scaler = StandardScaler()
        _blurriness = np.asarray([blur(load_image(slice, img_size)) for slice in slices]).reshape(-1, 1)
        _blurriness = scaler.fit_transform(_blurriness)
        blurriness = [b1 + b2 for (b1, b2) in zip(blurriness, _blurriness)]
    
    plt.plot(blurriness)

    # motor point
    z_axis = label[label.tomo_id == f"tomo_{tomo_id}"]["Motor axis 0"].values
    for point in z_axis:
        slice_index = int(point)
        plt.scatter(x=slice_index, y=blurriness[slice_index], c="red")

    plt.show()


def laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_entropy(image, threshold=5.0):
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 이미지 정규화 (0-255 범위로)
    img_normalized = ((image - image.min()) * (255.0 / (image.max() - image.min()))).astype(np.uint8)
    
    # 엔트로피 계산 (로컬 영역의 엔트로피)
    entropy_img = entropy(img_normalized, disk(5))
    
    # 전체 이미지의 평균 엔트로피
    mean_entropy = np.mean(entropy_img)
    
    return mean_entropy

def texture(image, threshold=0.2):
    from skimage.feature import graycomatrix, graycoprops
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 이미지 정규화 및 양자화
    img_normalized = ((image - image.min()) * (255.0 / (image.max() - image.min()))).astype(np.uint8)
    
    # GLCM 계산
    glcm = graycomatrix(img_normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # 다양한 GLCM 특성 계산
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # 특성들의 조합으로 점수 계산
    score = (contrast * correlation) / (energy * homogeneity)
    
    return score



tomo_id = "0a8f05"
select_slices, motor_axis, video_size = get_data(tomo_id)

anim = create_slice_animation(select_slices, motor_axis, video_size, blurs=[laplacian, get_entropy, texture], img_size=(128, 128))
display(HTML(anim.to_jshtml()))

plot_blur(laplacian, tomo_id, img_size=(128, 128))
plot_blur(get_entropy, tomo_id, img_size=(128, 128))
plot_blur(texture, tomo_id, img_size=(128, 128))

plot_comboned_blur([laplacian, get_entropy, texture], tomo_id, img_size=(128, 128))


tomo_id = "0da370"
select_slices, motor_axis, video_size = get_data(tomo_id)

anim = create_slice_animation(select_slices, motor_axis, video_size, blurs=[laplacian, get_entropy, texture], img_size=(128, 128))
display(HTML(anim.to_jshtml()))

plot_blur(laplacian, tomo_id, img_size=(128, 128))
plot_blur(get_entropy, tomo_id, img_size=(128, 128))
plot_blur(texture, tomo_id, img_size=(128, 128))

plot_comboned_blur([laplacian, get_entropy], tomo_id, img_size=(128, 128))


tomo_id = "0f9df0"
select_slices, motor_axis, video_size = get_data(tomo_id)

anim = create_slice_animation(select_slices, motor_axis, video_size, blurs=[laplacian, get_entropy, texture], img_size=(128, 128))
display(HTML(anim.to_jshtml()))

plot_blur(laplacian, tomo_id, img_size=(128, 128))
plot_blur(get_entropy, tomo_id, img_size=(128, 128))
plot_blur(texture, tomo_id, img_size=(128, 128))

plot_comboned_blur([laplacian, get_entropy], tomo_id, img_size=(128, 128))


tomo_id = "00e463"
select_slices, motor_axis, video_size = get_data(tomo_id)

anim = create_slice_animation(select_slices, motor_axis, video_size, blurs=[laplacian], img_size=(128, 128))
display(HTML(anim.to_jshtml()))

plot_blur(laplacian, tomo_id, img_size=(128, 128))


tomo_id = "01a877"
select_slices, motor_axis, video_size = get_data(tomo_id)

anim = create_slice_animation(select_slices, motor_axis, video_size, blurs=[laplacian], img_size=(128, 128))
display(HTML(anim.to_jshtml()))

plot_blur(laplacian, tomo_id, img_size=(128, 128))


tomo_id = "02862f"
select_slices, motor_axis, video_size = get_data(tomo_id)

anim = create_slice_animation(select_slices, motor_axis, video_size, blurs=[laplacian], img_size=(128, 128))
display(HTML(anim.to_jshtml()))

plot_blur(laplacian, tomo_id, img_size=(128, 128))


tomo_id = "a537dd"
select_slices, motor_axis, video_size = get_data(tomo_id)

anim = create_slice_animation(select_slices, motor_axis, video_size, blurs=[laplacian], img_size=(128, 128))
display(HTML(anim.to_jshtml()))

plot_blur(laplacian, tomo_id, img_size=(128, 128))


def analyze_blur(blur, tomo_id, img_size, z_threshold=3):
    PATH = os.path.join(ROOT, "train", f"tomo_{tomo_id}")

    slices = [os.path.join(PATH, slice) for slice in sorted(os.listdir(PATH))]
    blurriness = [blur(load_image(slice, img_size)) for slice in slices]

    min_blur = min(blurriness)
    max_blur = max(blurriness)
    mean_blur = sum(blurriness) / len(blurriness)

    z_axis = label[label.tomo_id == f"tomo_{tomo_id}"]["Motor axis 0"].values[0]
    z_axis = int(z_axis)
    keypoint_blur = blurriness[z_axis]
    keypoint_near_blur = sum(blurriness[z_axis - z_threshold : z_axis + z_threshold + 1]) / (z_threshold * 2 + 1)

    keypoint_near_percentile = sum(1 for x in blurriness if x <= keypoint_near_blur) / len(blurriness) * 100

    return min_blur, max_blur, mean_blur, keypoint_blur, keypoint_near_blur, keypoint_near_percentile


tomo_id_with_motors = set(label[label["Number of motors"] > 0].tomo_id.values)


from tqdm import tqdm

# Create empty lists to store results
min_blurs = []
max_blurs = []
mean_blurs = []
keypoint_blurs = []
keypoint_near_blurs = []
keypoint_near_percentiles = []
tomo_ids = []

# Analyze blur for each tomo_id
for tomo_id in tqdm(tomo_id_with_motors):
    # Extract just the ID part without 'tomo_' prefix
    id_only = tomo_id.replace('tomo_', '')
    
    # Get blur analysis results
    min_b, max_b, mean_b, key_b, key_near_b, key_near_p = analyze_blur(laplacian, id_only, (128, 128))
    
    # Append results to lists
    min_blurs.append(min_b)
    max_blurs.append(max_b) 
    mean_blurs.append(mean_b)
    keypoint_blurs.append(key_b)
    keypoint_near_blurs.append(key_near_b)
    keypoint_near_percentiles.append(key_near_p)
    tomo_ids.append(tomo_id)

# Create DataFrame
blur_df = pd.DataFrame({
    'tomo_id': tomo_ids,
    'min_blur': min_blurs,
    'max_blur': max_blurs,
    'mean_blur': mean_blurs,
    'keypoint_blur': keypoint_blurs,
    'keypoint_near_blur': keypoint_near_blurs,
    "keypoint_near_percentiles": keypoint_near_percentiles
})


blur_df.head()


blur_df.describe()


x = blur_df["keypoint_near_blur"].argmin()
tomo_id = tomo_ids[x].rsplit("_", 1)[-1]

select_slices, motor_axis, video_size = get_data(tomo_id)

anim = create_slice_animation(select_slices, motor_axis, video_size, blurs=laplacian, img_size=(128, 128))
display(HTML(anim.to_jshtml()))

plot_blur(laplacian, tomo_id, img_size=(128, 128))


x = blur_df["keypoint_near_percentiles"].argmin()
tomo_id = tomo_ids[x].rsplit("_", 1)[-1]

select_slices, motor_axis, video_size = get_data(tomo_id)

anim = create_slice_animation(select_slices, motor_axis, video_size, blurs=laplacian, img_size=(128, 128))
display(HTML(anim.to_jshtml()))

plot_blur(laplacian, tomo_id, img_size=(128, 128))







