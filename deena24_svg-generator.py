#| export
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Основные библиотеки ---
import os                        # Взаимодействие с файловой системой
import re                        # Валидность строки - цвета
from time import time            # Измерение времени выполнения
from datetime import timedelta   # Форматирование вывода времени
import numpy as np
import matplotlib.pyplot as plt  # Отображение растровых изображений и рендеринга SVG

# --- Для обработки изображений и DL ---
from PIL import Image
import cv2  # (Open Computer Vision Library)
from diffusers import StableDiffusionPipeline, DDIMScheduler  # Загрузка модели Stable Diffusion и планировщика DDIMScheduler

# --- Для работы с данными и KaggleHub ---
import polars as pl  # Чтение тренировочных данных
import kagglehub     # Для загрузки Stable Diffusion, метрик и тренировочных данных

# --- Для SVG ---
from IPython.display import SVG  # Прямое отображение SVG-изображения

# --- Метрика ---
try:
    print("Загрузка пакетов с метриками...")
    svg_scoring = kagglehub.package_import('richolson/stable-diffusion-svg-scoring-metric/versions/17')
    svg_constraints = kagglehub.package_import('metric/svg-constraints')
    print("Пакеты с метриками загружены.")
except Exception as e:
     print(f"Error importing metric packages: {e}")
     print("Пропуск инициализации метрик.")
     svg_scoring = None
     svg_constraints = None

# --- Класс ошибки (требуется метрикой) ---
class ParticipantVisibleError(Exception):
    pass


# --- Метрика (через kagglehub) ---
if svg_scoring:
    score = svg_scoring.score                            # Основная функция метрики
    VQAEvaluator = svg_scoring.VQAEvaluator              # Соответствие изображения запросу
    AestheticEvaluator = svg_scoring.AestheticEvaluator  # Эстетическая привлекательность
    harmonic_mean = svg_scoring.harmonic_mean            # Среднее гармоническое предыдущих метрик
    svg2png = svg_scoring.svg_to_png                     # Преобразование векторного изображения в растровое для оценки

    # --- Инициализация оценочных моделей ---
    global_vqa = None
    global_aesthetic = None

    def initialize_evaluators():
        """Инициализация оценочных моделей"""
        global global_vqa, global_aesthetic

        if global_vqa is None:
            print("Инициализация VQA оценки...")
            try:
                global_vqa = VQAEvaluator()
            except Exception as e:
                print(f"Failed to initialize VQA Evaluator: {e}")
                global_vqa = None

        if global_aesthetic is None:
           print("Инициализация оценки эстетической привлекательности...")
           try:
               global_aesthetic = AestheticEvaluator()
           except Exception as e:
               print(f"Failed to initialize Aesthetic Evaluator: {e}")
               global_aesthetic = None

        return global_vqa, global_aesthetic

    def evaluate(svg, prompt):
        """Локальная оценка SVG-кода"""
        vqa_evaluator, aesthetic_evaluator = initialize_evaluators()

        if vqa_evaluator is None or aesthetic_evaluator is None:
            print("Оценки не инициализированы.")
            return {'vqa_score': 0.0, 'aesthetic_score': 0.0, 'combined_score': 0.0}

        try:
            image = svg2png(svg)
            vqa_score = vqa_evaluator.score(image, 'SVG illustration of ' + prompt)
            aesthetic_score = aesthetic_evaluator.score(image)
            combined_score = harmonic_mean(vqa_score, aesthetic_score, beta=1.0)

            return {
                'vqa_score': vqa_score,
                'aesthetic_score': aesthetic_score,
                'combined_score': combined_score
            }
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {'vqa_score': 0.0, 'aesthetic_score': 0.0, 'combined_score': 0.0}

    initialize_evaluators()
else:
    # Если модели метрик не загрузились
    print("Пакеты с метриками не загружены.")
    def harmonic_mean(a, b, beta=1.0): return 0.0
    def svg2png(svg_code, size=(384, 384)):
         return Image.new('RGB', size, color = 'red')
    def evaluate(svg, prompt):
        print("Оценка недоступна.")
        return {'vqa_score': 0.0, 'aesthetic_score': 0.0, 'combined_score': 0.0}


# --- Утилиты для векторизации SVG ---
def compress_hex(rgb: list[int]) -> str:
    """
    Представление hex-цвета в кратчайшем виде
    Пример: #ff0099 -> #f09 если возможно
    """
    r, g, b = rgb
    # Проверка на возможность сокращения формата
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        return f'#{r//17:x}{g//17:x}{b//17:x}'
    return f'#{r:02x}{g:02x}{b:02x}'


def simplify_polygon(points_str: str, simple_level: int) -> str:
    """Упрощение многоугольника путем округления координат и/или сокращения количества точек."""
    if simple_level == 0:
        return points_str

    try:
        points = [tuple(map(float, p.split(','))) for p in points_str.strip().split()]
    except ValueError:
        return points_str

    if not points:
        return ""

    # Обработка уровней 1 и 2 с удалением дубликатов после округления
    if simple_level in (1, 2):
        rounded = [
            (round(x, 1), round(y, 1)) if simple_level == 1 else (round(x), round(y))
            for x, y in points
        ]
        unique = list(dict.fromkeys(rounded))  # Удаление дубликатов с сохранением порядка
        fmt = "{:.1f},{:.1f}" if simple_level == 1 else "{:.0f},{:.0f}"
        return " ".join(fmt.format(x, y) for x, y in unique)

    # Уровень 3: уменьшение количества точек и округление до целых
    n = len(points)
    step = 1 if n <= 6 else 2 if n <= 20 else n // 4
    
    simplified = points[::step]
    # Сохранение первой и последней точки
    if points[0] not in simplified:
        simplified.insert(0, points[0])
    if points[-1] not in simplified:
        simplified.append(points[-1])
    
    # Удаление дубликатов
    simplified = [(round(x), round(y)) for x, y in simplified]
    unique = list(dict.fromkeys(simplified))
    
    return " ".join(f"{x:.0f},{y:.0f}" for x, y in unique)


# --- Преобразование Bitmap в SVG ---
def extract_poligons(img_np: np.ndarray, num_colors: int = 16, blur_ksize: tuple[int, int] | None = None) -> list:
    """
    Получить контуры изображения по масштабу с использованием K-средних и контуров.

    img_np - входное изображение в виде массива NumPy RGB.
    num_colors - Количество цветов для извлечения с использованием K-средних.
    blur_ksize - Размер ядра для размытия по Гауссу перед K-средними (например, (5, 5)). None отключает размытие.

    Возвращает список словарей с описанием многоугольников:
        {'points': str, 'color': str,'area': float,'importance': float, 'point_count': int}.
        Список отсортирован по важности.
    """
    # Конвертация в RGB
    if len(img_np.shape) != 3 or img_np.shape[2] != 3:
         if len(img_np.shape) == 2:  # Изображение в оттенках серого
              img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
         elif img_np.shape[2] == 4:  # RGBA изображение
              img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
         else:
              print("Неверный формат введенного изображения. Ожидается RGB.")
              img_rgb = img_np[:,:,:3]
    else:
        img_rgb = img_np  # Изображение уже в формате RGB

    # Тип данных изображения - uint8 (0-255)
    img_rgb = img_rgb.astype(np.uint8)
   
    # Применение размытия
    if blur_ksize is not None and blur_ksize[0] > 0 and blur_ksize[1] > 0 and blur_ksize[0] % 2 != 0 and blur_ksize[1] % 2 != 0:
        try:
            # Размытие по Гауссу. sigmaX=0 <=> вычисляется из ksize.
            img_rgb_blurred = cv2.GaussianBlur(img_rgb, blur_ksize, 0)
            img_rgb = img_rgb_blurred
        except Exception as e:
            print(f"Ошибка при применении размытия: {e}. Продолжаем без размытия.")
    elif blur_ksize is not None:
         print(f"Неверный размер ядра размытия {blur_ksize}. Пропускаем размытие.")

    # Размеры обработанного изображения
    height, width = img_rgb.shape[:2]
    # Центральная точка изображения
    center_x, center_y = width/2, height/2

    # Обработка пустого изображения
    if height == 0 or width == 0:
         return []

    # --- Квантование цвета с использованием K-средних ---
    # Изменяем форму массива пикселей на (num_pixels x 3), числа с плавающей точкой
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    num_pixels = pixels.shape[0]

    # Критерии остановки K-средних и количество попыток инициализации
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)  # max 200 итераций и 0.2 разница между центрами кластеров двух итераций
    attempts = 3  # 3 попытки для лучшего результата

    try:
        # Запуск алгоритма K-средних
        # cv2.KMEANS_PP_CENTERS - улучшенная инициализация центроидов через k-means++
        ret, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        # ret - ошибка кластеризации, labels - массив принадлежности пикселей к кластерам, centers - массив центров
    except cv2.error as e:
        print(f"Error during K-means: {e}")
        return []

    # Создание палитры из центров кластеров и преобразование метки обратно в форму изображения
    palette = centers.astype(np.uint8)
    labels_img = labels.reshape(height, width)

    hierarchical_features = []

    # unique_labels - индексы кластеров (от 0 до num_colors-1), counts - количество пикселей для каждого кластера
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Сортировка индексов кластеров по убыванию количества пикселей
    sorted_indices = np.argsort(-counts)

    # Обработка каждого цвета с наиболее часто встречающегося
    for label_index in sorted_indices:
        color = palette[label_index]  # RGB цвет для текущего кластера
        # Бинарная маска для текущего кластера (пиксели текущего кластера - белые, остальные - черные)
        color_mask = (labels_img == label_index).astype(np.uint8) * 255

        # Поиск контуров на маске цвета
        # cv2.RETR_EXTERNAL извлекает только внешние контуры
        # cv2.CHAIN_APPROX_SIMPLE сжимает горизонтальные, вертикальные и диагональные сегменты (сохраняет только ключевые вершины)
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours - список контуров, hierarchy - информация о вложенности

        # Сортировка контуров по площади от большего к меньшему
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # RGB -> сжатый hex
        hex_color = compress_hex(color)

        # Обрабатка контура для текущего цвета
        color_features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # Пропуск очень маленьких контуров (шумов)
                continue

            # Центр контура
            m = cv2.moments(contour)
            # Нулевой момент связан с площадью
            if m["m00"] == 0:
                # Резервный вариант для точек или линий: используется центр ограничивающего прямоугольника
                try:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2
                except:
                     continue
            else:
                cx = int(m["m10"] / m["m00"])  # Момент по X / масса = абсцисса цендроида
                cy = int(m["m01"] / m["m00"])  # Момент по Y / масса = ордината центроида

            # Нормализованное расстояние от центра изображения
            norm_dist_from_center = np.sqrt(((cx - center_x) / (width + 1e-6))**2 + ((cy - center_y) / (height + 1e-6))**2)

            # Упрощение контура
            epsilon_multiplier = 0.01  # Коэффициент упрощения
            epsilon = epsilon_multiplier * cv2.arcLength(contour, True)  # arcLength - длина контура
            epsilon = max(epsilon, 1.0)  # Минимальный допуск упрощения

            approx = cv2.approxPolyDP(contour, epsilon, True)  # Удаление лишних точек контура

            # Пропуск контуров линий или точек
            if len(approx) < 3:
                continue

            # Генерация строк точек для многоугольника
            points = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])

            # Вычисление важности контура на основе его площади, близости к центру и количества точек
            importance = (
                area *  # Чем больше площадь, тем важнее
                (1.0 - min(norm_dist_from_center, 1.0)) *  # Чем ближе к центру, тем важнее
                (1.0 / (len(approx) + 1))  # Чем меньше точек после аппроксимации, тем важнее
            )

            # Сохранение данных контура
            color_features.append({
                'points': points,            # Строка точек для SVG
                'color': hex_color,          # Цвет контура
                'area': area,                # Площадь контура
                'importance': importance,    # Важность
                'point_count': len(approx),  # Количество вершин после аппроксимации
            })

        color_features.sort(key=lambda x: x['importance'], reverse=True)  # Отсортированный список контуров текущего цвета
        hierarchical_features.extend(color_features)

    # Сортировка всех контуров по общей важности по всем цветам
    hierarchical_features.sort(key=lambda x: x['importance'], reverse=True)

    return hierarchical_features

def bitmap2svg(image: Image.Image, max_size: int = 10000, resize: bool = True, target_size: tuple[int, int] = (384, 384),
               num_colors: int | None = None, blur_ksize: tuple[int, int] | None = None) -> str:
    """
    Конвертация растрового изображения в SVG с использованием слоистого подхода и оптимизации размера.

    image - Входное изображение.
    max_size - Максимальный размер SVG в байтах.
    resize - Изменять ли размер изображения перед обработкой.
    target_size - Целевой размер для изменения размера.
    num_colors - Количество цветов для извлечения.
    blur_ksize - Размер ядра для размытия по Гауссу перед векторизацией.

    Возвращает SVG-представление.
    """
    if num_colors is None:
        if resize:
            pixel_count = target_size[0] * target_size[1]
        else:
            pixel_count = image.size[0] * image.size[1]

        # Выбор количества цветов на основе размера входного изображения
        orig_pixel_count = image.size[0] * image.size[1]
        if orig_pixel_count < 128*128: # Маленькие
             num_colors = 8
        elif orig_pixel_count < 256*256: # Средние
             num_colors = 12
        elif orig_pixel_count < 512*512: # Большие
             num_colors = 16
        else: # Очень большие
             num_colors = 24

    else:
         num_colors = max(1, num_colors) # Есть хотя бы 1 цвет

    # Исходные размеры
    orig_width, orig_height = image.size

    # Изменение размера изображения, если нужно
    if resize:
        # LANCZOS для уменьшения масштаба, BICUBIC для увеличения
        resample = Image.LANCZOS if image.size[0] > target_size[0] or image.size[1] > target_size[1] else Image.BICUBIC
        image = image.resize(target_size, resample)

    # Конвертация в массив numpy в формате RGB
    img_np = np.array(image.convert('RGB'))

    # Размеры для обработки
    proc_height, proc_width = img_np.shape[:2]

    # Обработка пустого изображения после изменения размера
    if proc_height == 0 or proc_width == 0:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {proc_width} {proc_height}"><rect width="100%" height="100%" fill="#fff"/></svg>'

    # Вычисление среднего цвета фона
    try:
        bg_color = np.mean(img_np, axis=(0,1)).astype(int)
        bg_hex = compress_hex(bg_color)
    except Exception:
        # Белый цвет, если вычисление не удалось
        bg_hex = '#fff'


    # Структура SVG
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {proc_width} {proc_height}">\n'
    svg_bg = f'<rect width="{proc_width}" height="{proc_height}" fill="{bg_hex}"/>\n'
    svg_base = svg_header + svg_bg
    svg_footer = '</svg>'

    # Ббазовый размер (заголовок + фон + футер)
    base_svg = svg_base + svg_footer
    base_size = len(base_svg.encode('utf-8'))
    available_bytes = max(0, max_size - base_size)

    # Извлечение контуров
    features = extract_poligons(img_np, num_colors=num_colors, blur_ksize=blur_ksize)

    # Вычисление размеров на разных уровнях упрощения
    parts_and_sizes = []
    for f in features:
         points_orig = f['points']
         color = f['color']
         # Вычисление SVG-части и ее размера для каждого уровня упрощения (0-3)
         parts = {
             0: f'<polygon points="{simplify_polygon(points_orig, 0)}" fill="{color}" />\n',
             1: f'<polygon points="{simplify_polygon(points_orig, 1)}" fill="{color}" />\n',
             2: f'<polygon points="{simplify_polygon(points_orig, 2)}" fill="{color}" />\n',
             3: f'<polygon points="{simplify_polygon(points_orig, 3)}" fill="{color}" />\n'
         }
         sizes = {level: len(part.encode('utf-8')) for level, part in parts.items()}
         parts_and_sizes.append({'parts': parts, 'sizes': sizes, 'importance': f['importance']})

    # Многоуровневый подход для оптимального использования пространства
    svg = svg_base
    bytes_used = base_size
    added_indices = set() # Отслеживание уже добавленных контуров

    # Добавление наиболее важных контуров с максимально возможным качеством (уровень 0, затем 1, 2, 3)
    for level in range(0, 4):
        # Повторная сортировка контуров по исходной важности для каждого прохода по уровням
        features_for_level = sorted([f for f in parts_and_sizes if parts_and_sizes.index(f) not in added_indices],
                                     key=lambda x: x['importance'], reverse=True)

        for feature_info in features_for_level:
            # Поиск исходного индекса контура в общем списке
            orig_index = parts_and_sizes.index(feature_info)

            # Проверка, был ли этот контур уже добавлена в предыдущем проходе более высокого качества
            if orig_index in added_indices:
                continue

            # Добавление на текущем уровне упрощения
            current_feature = feature_info['parts'][level]
            current_feature_size = feature_info['sizes'][level]

            # Проверка вместимости контуров в оставшийся бюджет размера
            if bytes_used + current_feature_size <= max_size:
                svg += current_feature # Добавление SVG-части
                bytes_used += current_feature_size # Обновление использованных байт
                added_indices.add(orig_index)

    # Закрытие SVG кода
    svg += svg_footer

    # Проверка ограничения размера последний раз. Если оно превышено, возвращается резервный вариант.
    final_size = len(svg.encode('utf-8'))
    if final_size > max_size:
        print(f"Warning: Final SVG size ({final_size} bytes) exceeds limit ({max_size}) after adding features.")
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {proc_width} {proc_height}"><rect width="{proc_width}" height="{proc_height}" fill="{bg_hex}"/></svg>'
        
    return svg


# --- Модель генерации изображений (Stable Diffusion) ---
print("Загрузка модели Stable Diffusion...")
try:
    # Скачивание модели Stable Diffusion
    sd_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")

    # Загрузка планировщика для управления процессом диффузии
    scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    # Загрузка основного конвейера для генерации изображений
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_path,
        scheduler=scheduler,          # Установка загруженнего планировщика
        torch_dtype=torch.float16,    # Использование половинной точности
        safety_checker=None           # Отключение проверки безопасности для ускорения
    )

    # Перемещение модели на устройство
    pipe.to(device)
    print("Модель Stable Diffusion загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели Stable Diffusion: {e}")
    pipe = None

def generate_bitmap(prompt: str, negative: str = "", num_inference_steps: int = 20, guidance_scale: float = 30):
    """
    Генерация изображения с использованием Stable Diffusion.

    prompt - текстовый запрос для генерации изображения.
    negative - описание нежелательных элементов.
    num_inference_steps - количество шагов очистки от шума в процессе диффузии.
    guidance_scale - масштаб соответствия запросу.

    Возвращает сгенерированное изображение в формате PIL.Image.
    """
    # Проверка, был ли pipeline загружен при инициализации
    if pipe is None:
        print("Pipeline Stable Diffusion не загружен. Невозможно сгенерировать изображение.")
        return None

    # Генерирация изображения с помощью Stable Diffusion
    with torch.no_grad():
        try:
            # Вызов pipeline для генерации изображения
            image = pipe(
                prompt=prompt,                           # Прямой запрос
                negative=negative,         # Негативный запрос
                num_inference_steps=num_inference_steps, # Количество шагов
                guidance_scale=guidance_scale,           # Масштаб соответствия
                generation_height = 384,                  # Высота изображения
                generation_width = 384                   # Ширина изображения
            ).images[0]
            return image
        except Exception as e:
            print(f"Ошибка во время генерации Stable Diffusion: {e}")
            return None


# --- Конвейер Генерации и Оценки ---
def generate_and_convert(prompt: str, prefix: str = "", suffix: str = "", negative: str = "", attempts: int = 3,
    num_inference_steps: int = 20, guidance_scale: float = 25, svg_max_size: int = 10000, svg_num_colors: int = None,
                         blur_ksize: tuple[int, int] | None = None, verbose: bool = True):
    """
    Генерирует изображение с помощью Stable Diffusion, преобразует в SVG и оценивает результат.

    prompt - базовый текстовый запрос для генерации.
    prefix - текст, добавляемый в начало базового запроса.
    suffix - текст, добавляемый в конец базового запроса.
    negative - текстовый запрос, описывающий то, чего нужно избежать на изображении.
    attempts - количество попыток генерации и оценки.
    num_inference_steps - количество шагов шумоподавления для Stable Diffusion.
    guidance_scale - сила соответствия запросу для Stable Diffusion.
    svg_max_size -максимально допустимый размер сгенерированного SVG в байтах.
    svg_num_colors - количество цветов для векторизации в SVG.
    blur_ksize - Размер ядра для размытия по Гауссу перед векторизацией.
    
    Возвращает кортеж:
        (лучшая SVG строка или None, лучшая достигнутая оценка (float),
         изображение bitmap, которое дало лучший SVG или None)
        (None, -1.0, None), если ни одна попытка не привела к валидному SVG.
    """
    best_svg = None
    best_bitmap = None
    best_score = -1.0 # Начальная лучшая оценка
    best_index = -1

    # Отслеживание временной статистики
    total_start = time()
    attempt_times = []

    # Формирование полного запроса для генерации
    combined_prompt = f"{prefix} {prompt} {suffix}".strip()

    for i in range(attempts):
        attempt_start = time()
        if verbose:
            print(f"\n=== Попытка {i+1}/{attempts} по запросу: '{prompt}' ===")
            print(f"Используется полный запрос: '{combined_prompt}'")

        start_time = time()
        # --- Шаг 1: Генерация изображения (bitmap) ---
        bitmap = generate_bitmap(combined_prompt, negative=negative,
                                 num_inference_steps=num_inference_steps,
                                 guidance_scale=guidance_scale)

        generation_time = time() - attempt_start
        
        if bitmap is None:
            if verbose:
                print("Генерация растрового изображения не удалась")
                attempt_times.append(time() - start_time)
            continue

        # --- Шаг 2: Конвертация bitmap в SVG ---
        if verbose:
            print(f"Конвертация в SVG... (Время генерации растрового изображения: {generation_time:.2f} с")
        conversion_start = time()
        
        svg_content = bitmap2svg(bitmap, max_size=svg_max_size, num_colors=svg_num_colors, blur_ksize=blur_ksize)

        conversion_time = time() - conversion_start

        svg_size = len(svg_content.encode('utf-8'))
        
        if verbose:
            print(f"Время конвертации в SVG: {conversion_time:.2f}s, размер SVG: {svg_size} байтов")
            # Отображение оригинала и SVG
        
            try:
                rendered_svg = svg2png(svg_content)
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.imshow(bitmap)
                plt.title(f"Попытка {i+1}: Оригинальное изображение")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(rendered_svg)
                plt.title(f"Попытка {i+1}: SVG конвертация\nРазмер: {svg_size} байтов")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                 print(f"Ошибка в отображении: {e}")
                
        evaluation_start = time()
        svg_scores = evaluate(svg_content, prompt)
        evaluation_time = time() - evaluation_start
        combined_score = svg_scores.get('combined_score', 0.0)
        
        # --- Шаг 3: Оценка SVG с использованием метрик ---
        if verbose:
            print(f"Время оценки: {evaluation_time:.2f}с")
            print(f"VQA оценка: {svg_scores.get('vqa_score', 0.0):.4f}")
            print(f"Эстетическая привлекательность: {svg_scores.get('aesthetic_score', 0.0):.4f}")
            print(f"Общая оценка: {combined_score:.4f}")

        # --- Шаг 4: Отслеживание лучшего результата ---
        if combined_score > best_score:
            best_score = combined_score
            best_svg = svg_content
            best_bitmap = bitmap
            best_index = i + 1
            if verbose:
                print(f"✅ Новый лучший результат: {best_score:.4f}")
        else:
            if verbose:
                print(f"❌ Результат текущего лучшего: {best_score:.4f}")
        
        attempt_end = time()
        attempt_time = attempt_end - attempt_start
        attempt_times.append(attempt_time)
        if verbose:
            print(f"Total for attempt {i+1}: {attempt_time:.2f}s")
    # Общее время работы
    total_end = time()
    total_time = total_end - total_start
    
    if verbose:
        if attempt_times:
            avg_time = sum(attempt_times)/len(attempt_times)
            print(f"Среднее время попытки: {avg_time:.2f}с")
        print(f"Общее время работы ({len(attempt_times)} исполненных попыток): {total_time:.2f}s")
        print(f"Лучший результат: {best_score:.4f} (попытки {best_index})")

    # Возвращение лучшего найденного SVG, его оценка и соответствующий bitmap
    return best_svg, best_score, best_bitmap


# --- Тестовый запуск с примером ---
test_prompt = "a cat sitting on a mat"

test_prefix = "Simple illustration of"
test_suffix = "Style: clip art, extremely simple, flat color blocks, bright colors \
solid fills, clean shapes, high contrast, no background, clip art style, large elements"
test_negative = "lines, detailes, hatching, 3D, text, watermark, texture, \
messy, framing, rounding, pixelation, gradient, grid, dots, noise, grain"

print(f"Запуск тестовой генерации для запроса: '{test_prompt}'")

# --- Вызов функции генерации и конвертации ---
best_svg_example, best_score_example, best_bitmap_example = generate_and_convert(
    prompt=test_prompt,      # Основной запрос
    prefix=test_prefix,      # Префикс запроса
    suffix=test_suffix,      # Суффикс запроса
    negative=test_negative,  # Негативный запрос
    attempts=3,              # Кличество попыток генерации в generate_and_convert
    num_inference_steps=20,  # Количество шагов Stable Diffusion
    guidance_scale=20,       # Масштаб соответствия запросу для Stable Diffusion
    svg_max_size=10000,      # Максимальный размер выходного SVG
    svg_num_colors=None,     # Количество цветов для SVG (None = авто)
    blur_ksize=[5, 5],       # Размер ядра размытия
    verbose=True             # Подробный вывод функции generate_and_convert для отладки теста
)

# --- Обработка и отображение результата ---
if best_svg_example:
    print("\n--- Результат лучшего SVG ---")
    print(f"Лучшая достигнутая оценка: {best_score_example:.4f}")
    print(f"Начало лучшего SVG кода (первые 500 символов): {best_svg_example[:500]}...")

    display(SVG(best_svg_example))
else:
    # Если функция generate_and_convert не вернула SVG
    print("\nНе удалось сгенерировать SVG после всех попыток.")


# --- Класс Model ---
class Model:
    """Модель для генерации SVG изображений на основе текстового запроса."""
    def __init__(self):
        # --- Параметры генерации ---
        self.attempts = 3                    # Количество внутренних попыток для каждой подсказки
        self.num_inference_steps = 20        # Количество шагов denoising для Stable Diffusion
        self.guidance_scale = 20.0           # Масштаб соответствия запросу (Classifier-Free Guidance)
        self.svg_max_size = 10000             # Максимально допустимый размер SVG файла в байтах
        self.svg_num_colors = None           # Количество цветов для векторизации.
    
        # --- Инженерия запросов ---
        self.prefix = "Simple image of"
        self.suffix = "Style: clip art, solid fills, saturated flat colors, clean shapes, \
        small color palette, extremely simple background, high contrast, colorful"
        self.negative = "lines, detailes, hatching, 3D, text, watermark, texture, \
        messy, framing, rounding, pixelation, gradient, grid, dots, noise, grain"

        # Проверка, что pipeline Stable Diffusion успешно загружен
        if pipe is None:
             raise RuntimeError("Pipeline Stable Diffusion не загружен. Проверьте загрузку модели перед инициализацией класса Model.")


    def predict(self, prompt: str) -> str:
        """
        Генерирует SVG изображение из текстового описания.

        prompt - текстовое описание изображения, которое нужно сгенерировать.

        Возвращает строку с SVG кодом сгенерированного изображения.
        """
        # Полный запрос
        combined_prompt = f"{self.prefix} {prompt} {self.suffix}".strip()
        best_svg = None
        best_score = -1.0
        
        try:
            best_svg, best_score_obtained, _ = generate_and_convert(
                prompt,
                prefix=self.prefix,
                suffix=self.suffix,
                negative=self.negative,
                attempts=self.attempts,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                svg_max_size=self.svg_max_size,
                svg_num_colors=self.svg_num_colors,
                blur_ksize=[5, 5],
                verbose=False
            )

            if best_svg is None:
                 print(f"Предупреждение: Не удалось сгенерировать SVG для запроса: '{prompt}'. Возвращается запасной SVG.")
                 return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 384"><rect width="100%" height="100%" fill="#fff"/></svg>'

            return best_svg

        except Exception as e:
            print(f"Ошибка во время предсказания для запроса '{prompt}': {e}")
            return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 384"><rect width="100%" height="100%" fill="#fff"/></svg>'


# --- Тестирование класса Model на тренировочных данных ---
print("\n--- Тестирование класса Model на тренировочных данных ---")

# --- Чтение тренировочных данных ---
train_df = kagglehub.competition_download('drawing-with-llms', 'train.csv')

if not os.path.exists(train_df):
    print(f"Тренировочные данные не найдены по пути: {train_df}.")
    df_train = pl.DataFrame([{'row_id': 0, 'description': 'A red circle'}])
    print("Используются фиктивные данные для теста модели.")
else:
    print(f"Тренировочные данные найдены по пути: {train_df}.")
    df_train = pl.read_csv(train_df)
    
# --- Инициализация класса Model ---
print("Инициализация класса Model...")

try:
    # Создание экземпляра класса Model
    model = Model()
    print("Модель инициализирована успешно.")
except RuntimeError as e:
    print(f"Ошибка инициализации модели: {e}")
    model = None


# --- Запуск тестирования, если модель инициализирована ---
if model:
    # Списки для хранения результатов каждого примера
    scores = []           # общие оценки
    generation_times = [] # времени генерации каждого SVG
    svg_sizes = []        # размеров каждого SVG в байтах

    print(f"\nОбработка {len(df_train)} запросов...")

    for i, row_dict in enumerate(df_train.iter_rows(named=True)):
        row_id = row_dict.get('row_id', f'unknown_row_{i}')
        description = row_dict.get('description', 'empty description')
        
        print(f"\nОбработка примера {i+1}/{len(df_train)} (row_id: {i}): '{description}'")

        start_time = time() # Начало отсчета времени

        svg_code = model.predict(description)

        end_time = time()
        generation_time = end_time - start_time # Время, потраченное на predict()
        generation_times.append(generation_time)

        svg_size = len(svg_code.encode('utf-8'))
        svg_sizes.append(svg_size)
        print(f"Размер сгенерированного SVG: {svg_size} байтов (Время генерации: {generation_time:.2f} с)")

        if svg_scoring is not None:
            print("Оценка сгенерированного SVG...")
            evaluation_start = time()
            
            try:
                 svg_scores = evaluate(svg_code, description)
                 score = svg_scores.get('combined_score', 0.0)
                 print(f"Оценка завершена за {time() - evaluation_start:.2f} с")
                 print(f"VQA оценка: {svg_scores.get('vqa_score', 0.0):.4f}, Эстетическая привлекательность: {svg_scores.get('aesthetic_score', 0.0):.4f}, Общая оценка: {score:.4f}")
            except Exception as e:
                 print(f"Ошибка при оценке примера {i}: {e}")
                 score = 0.0
        else:
            print("Метрика оценки недоступна. Оценка пропущена.")
            score = 0.0

        scores.append(score)

        # --- Отображение сгенерированного изображения ---
        print("Рендеринг и отображение SVG...")
        try:
            rendered_img = svg2png(svg_code)
            plt.figure(figsize=(6, 6))
            
            # Отображение растрового изображение
            plt.imshow(rendered_img)
            
            # Заголовок с i, началом описания, оценкой и размером SVG
            plt.title(f"Пример {i}: {description[:50]}...\nОценка: {score:.4f}, Размер: {svg_size} байт")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Ошибка рендеринга или отображения SVG для примера {i}: {e}")


    # --- Расчет итоговой статистики ---
    if scores:
        # Средняя оценка
        avg_score = float(np.mean(scores))
        
        # Суммарное время генерации для всех обработанных примеров
        time_taken = sum(generation_times)
        
        # Среднее время генерации на один пример
        avg_time = time_taken / len(generation_times) if generation_times else 0.0
        
        # Средний размер SVG
        avg_size = float(np.mean(svg_sizes)) if svg_sizes else 0.0 # Приводим к float
        
        # Общее количество обработанных запросов
        total_prompts = len(df_train)

        # --- Прогнозируемое время для полного тестового набора ---
        projected_500 = 500 * avg_time
        projected_hours = projected_500 / 3600

        # --- Вывод итоговой статистики ---
        print("\n=== ИТОГОВОЕ ЛОКАЛЬНОЕ ТЕСТИРОВАНИЕ ===")
        print(f"Обработано запросов: {total_prompts}")
        print(f"Итоговая средняя общая оценка: {avg_score:.4f}")
        print(f"Среднее время генерации на один запрос (Model.predict): {avg_time:.2f} секунд")
        print(f"Средний размер SVG: {avg_size:.2f} байт")
        
        # Время в формате ЧЧ:ММ:СС
        print(f"Общее затраченное время для {total_prompts} запросов: {timedelta(seconds=int(time_taken))}")
        
        # Время для 500 запросов в часах и формате ЧЧ:ММ:СС
        print(f"Прогнозируемое время для 500 запросов: {projected_hours:.2f} часов ({timedelta(seconds=int(projected_500))})")
    else:
        print("\nНи один запрос не был успешно обработан.")

else:
    print("\nКласс Model не был инициализирован. Тестирование модели пропущено.")

