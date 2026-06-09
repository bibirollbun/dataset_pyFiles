!pip install transformers torch opencv-python pandas tqdm Pillow


import torch
import pandas as pd
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model_id = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)


def generate_caption_for_video(video_path):
    """
    Извлекает центральный кадр из видео и генерирует для него описание.
    """
    try:
        # Открываем видеофайл
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return "Could not process video."

        # Находим центральный кадр
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = total_frames // 2
        
        # Устанавливаем позицию для чтения центрального кадра
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Ошибка: не удалось прочитать кадр из видео {video_path}")
            return "Could not read frame from video."

        # OpenCV читает изображения в формате BGR, а модель ожидает RGB
        # Конвертируем цветовое пространство и создаем PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # --- Генерация описания ---
        # Подготавливаем изображение для модели
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Генерируем ID токенов (слов)
        output_ids = model.generate(**inputs, max_length=200, num_beams=5)
        
        # Декодируем ID в текст
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        
        return caption.strip()

    except Exception as e:
        print(f"Произошла ошибка при обработке {video_path}: {e}")
        return "Error during processing."


TEST_CSV_PATH = '/kaggle/input/automated-video-captioning/test.csv'  
VIDEO_FOLDER = '/kaggle/input/automated-video-captioning/test_videos' 


test_df = pd.read_csv(TEST_CSV_PATH)


captions = []

for video_file in tqdm(test_df['file_name'], desc="de"):
    full_video_path = os.path.join(VIDEO_FOLDER, video_file)
    
    generated_caption = generate_caption_for_video(full_video_path)
    captions.append(generated_caption)

test_df['caption'] = captions


test_df['caption'].sample


submission_df = pd.DataFrame({
    'index': test_df.index,
    'file_name': test_df['file_name'],
    'caption': test_df['caption']
})

submission_df.to_csv('submission.csv', index=False)

