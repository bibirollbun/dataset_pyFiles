import torch
import torchaudio
from transformers import pipeline
from PIL import Image


# Пути к файлам
image_path = '/kaggle/input/test-data/Unknown.jpg'
audio_path = '/kaggle/input/test-data/dolgiy-lay-sobaki-na-sosedey.wav'


# Инициализация моделей
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224", top_k=5)
audio_classifier = pipeline("audio-classification", model="superb/hubert-base-superb-ks")


def load_image(image_path):
    return Image.open(image_path)


def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Преобразование многоканального аудио в моно
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def recognize_breed_from_image(image_path):
    image = load_image(image_path)
    results = image_classifier(image)
    return results


def recognize_breed_from_audio(audio_path):
    waveform, sample_rate = load_audio(audio_path)
    # Передача в модель
    results = audio_classifier({"raw": waveform.squeeze().numpy(), "sampling_rate": sample_rate})
    return results


if __name__ == "__main__":
    print("Распознавание по фото:")
    image_results = recognize_breed_from_image(image_path)
    for res in image_results:
        print(f"{res['label']}: {res['score']:.2f}")

    print("\nРаспознавание по звуку:")
    audio_results = recognize_breed_from_audio(audio_path)
    for res in audio_results:
        print(f"{res['label']}: {res['score']:.2f}")


import torch
import torchaudio
from transformers import pipeline
from PIL import Image

# Пути к файлам
image_path = '/kaggle/input/test-data/Unknown.jpg'
audio_path = '/kaggle/input/test-data/dolgiy-lay-sobaki-na-sosedey.wav'

# Инициализация моделей
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224", top_k=5)
audio_classifier = pipeline("audio-classification", model="superb/hubert-base-superb-ks")

def load_image(image_path):
    return Image.open(image_path)

def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    # Преобразование многоканального аудио в моно
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate

def recognize_breed_from_image(image_path):
    image = load_image(image_path)
    results = image_classifier(image)
    return results

def recognize_breed_from_audio(audio_path):
    waveform, sample_rate = load_audio(audio_path)
    # Передача в модель
    results = audio_classifier({"raw": waveform.squeeze().numpy(), "sampling_rate": sample_rate})
    return results

if __name__ == "__main__":
    print("Распознавание по фото:")
    image_results = recognize_breed_from_image(image_path)
    for res in image_results:
        print(f"{res['label']}: {res['score']:.2f}")

    print("\nРаспознавание по звуку:")
    audio_results = recognize_breed_from_audio(audio_path)
    for res in audio_results:
        print(f"{res['label']}: {res['score']:.2f}")




