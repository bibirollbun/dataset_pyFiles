!pip install cairosvg bitsandbytes cairosvg
!pip install git+https://github.com/openai/CLIP.git


# To prevent reloading the model each time we run the score() function
aesthetic_evaluator = None
vqa_evaluator = None


import ast
import io
import math
import statistics
import string

import cairosvg
import clip
import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from more_itertools import chunked
from PIL import Image, ImageFilter
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration
)

svg_constraints = kagglehub.package_import('metric/svg-constraints')


class ParticipantVisibleError(Exception):
    pass

def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, random_seed: int = 0
) -> float:
    """Calculates a fidelity score by comparing generated SVG images to target text descriptions.

    Parameters
    ----------
    solution : pd.DataFrame
        A DataFrame containing target questions, choices, and answers about an SVG image.
    submission : pd.DataFrame
        A DataFrame containing generated SVG strings. Must have a column named 'svg'.
    row_id_column_name : str
        The name of the column containing row identifiers. This column is removed before scoring.
    random_seed : int
        A seed to set the random state.

    Returns
    -------
    float
        The mean fidelity score (a value between 0 and 1) representing the average similarity between the generated SVGs and their descriptions.
        A higher score indicates better fidelity.

    Raises
    ------
    ParticipantVisibleError
        If the 'svg' column in the submission DataFrame is not of string type or if validation of the SVG fails.

    Examples
    --------
    >>> import pandas as pd
    >>> solution = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'question': ['["Is there a red circle?", "What shape is present?"]'],
    ...     'choices': ['[["yes", "no"], ["square", "circle", "triangle", "hexagon"]]'],
    ...     'answer': ['["yes", "circle"]'],
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'svg': ['<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'],
    ... })
    >>> score(solution, submission, 'row_id', random_seed=42)
    0...
    """
    # Convert solution fields to list dtypes and expand
    for colname in ['question', 'choices', 'answer']:
        solution[colname] = solution[colname].apply(ast.literal_eval)
    solution = solution.explode(['question', 'choices', 'answer'])

    # Validate
    if not pd.api.types.is_string_dtype(submission.loc[:, 'svg']):
        raise ParticipantVisibleError('svg must be a string.')

    # Check that SVG code meets defined constraints
    
    constraints = svg_constraints.SVGConstraints()
    try:
        for svg in submission.loc[:, 'svg']:
            if svg.startswith("<svg"):
                constraints.validate_svg(svg)
            else:
                # It's Imagen3 "submission"
                pass
    except:
        raise ParticipantVisibleError('SVG code violates constraints.')

    # Score
    global vqa_evaluator
    global aesthetic_evaluator
    
    vqa_evaluator = vqa_evaluator or VQAEvaluator()
    aesthetic_evaluator = aesthetic_evaluator or AestheticEvaluator()

    results = {}
    rng = np.random.RandomState(random_seed)
    try:
        df = solution.merge(submission, on='id')
        for i, (_, group) in enumerate(df.loc[
            :, ['id', 'question', 'choices', 'answer', 'svg']
        ].groupby('id')):
            id = group['id'].iloc[0]
            questions, choices, answers, svg = [
                group[col_name].to_list()
                for col_name in group.drop('id', axis=1).columns
            ]
            svg_or_path = svg[0]  # unpack singleton from list
            group_seed = rng.randint(0, np.iinfo(np.int32).max)

            img = svg_to_png(svg_or_path) if svg_or_path.startswith("<svg") else load_jpeg(svg_or_path)
            image_processor = ImageProcessor(image=img, seed=group_seed).apply()
            image = image_processor.image.copy()
            aesthetic_score = aesthetic_evaluator.score(image)
            vqa_score = vqa_evaluator.score(questions, choices, answers, image)
            image_processor.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
            ocr_score = vqa_evaluator.ocr(image_processor.image)
            instance_score = (
                harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
            )
            results[id] = instance_score

    except:
        raise ParticipantVisibleError('SVG failed to score.')

    fidelity = statistics.mean(results.values())
    return float(fidelity), results


class VQAEvaluator:
    """Evaluates images based on their similarity to a given text description using multiple choice questions."""

    def __init__(self):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.letters = string.ascii_uppercase
        self.model_path = kagglehub.model_download(
            'google/paligemma-2/transformers/paligemma2-10b-mix-448'
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
        ).to('cuda:0')

    def score(self, questions, choices, answers, image, n=4):
        scores = []
        batches = (chunked(qs, n) for qs in [questions, choices, answers])
        for question_batch, choice_batch, answer_batch in zip(*batches, strict=True):
            scores.extend(
                self.score_batch(
                    image,
                    question_batch,
                    choice_batch,
                    answer_batch,
                )
            )
        return statistics.mean(scores)

    def score_batch(
        self,
        image: Image.Image,
        questions: list[str],
        choices_list: list[list[str]],
        answers: list[str],
    ) -> list[float]:
        """Evaluates the image based on multiple choice questions and answers.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to evaluate.
        questions : list[str]
            List of questions about the image.
        choices_list : list[list[str]]
            List of lists of possible answer choices, corresponding to each question.
        answers : list[str]
            List of correct answers from the choices, corresponding to each question.

        Returns
        -------
        list[float]
            List of scores (values between 0 and 1) representing the probability of the correct answer for each question.
        """
        prompts = [
            self.format_prompt(question, choices)
            for question, choices in zip(questions, choices_list, strict=True)
        ]
        batched_choice_probabilities = self.get_choice_probability(
            image, prompts, choices_list
        )

        scores = []
        for i, _ in enumerate(questions):
            choice_probabilities = batched_choice_probabilities[i]
            answer = answers[i]
            answer_probability = 0.0
            for choice, prob in choice_probabilities.items():
                if choice == answer:
                    answer_probability = prob
                    break
            scores.append(answer_probability)

        return scores

    def format_prompt(self, question: str, choices: list[str]) -> str:
        prompt = f'<image>answer en Question: {question}\nChoices:\n'
        for i, choice in enumerate(choices):
            prompt += f'{self.letters[i]}. {choice}\n'
        return prompt

    def mask_choices(self, logits, choices_list):
        """Masks logits for the first token of each choice letter for each question in the batch."""
        batch_size = logits.shape[0]
        masked_logits = torch.full_like(logits, float('-inf'))

        for batch_idx in range(batch_size):
            choices = choices_list[batch_idx]
            for i in range(len(choices)):
                letter_token = self.letters[i]

                first_token = self.processor.tokenizer.encode(
                    letter_token, add_special_tokens=False
                )[0]
                first_token_with_space = self.processor.tokenizer.encode(
                    ' ' + letter_token, add_special_tokens=False
                )[0]

                if isinstance(first_token, int):
                    masked_logits[batch_idx, first_token] = logits[
                        batch_idx, first_token
                    ]
                if isinstance(first_token_with_space, int):
                    masked_logits[batch_idx, first_token_with_space] = logits[
                        batch_idx, first_token_with_space
                    ]

        return masked_logits

    def get_choice_probability(self, image, prompts, choices_list) -> list[dict]:
        inputs = self.processor(
            images=[image] * len(prompts),
            text=prompts,
            return_tensors='pt',
            padding='longest',
        ).to('cuda:0')

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Logits for the last (predicted) token
            masked_logits = self.mask_choices(logits, choices_list)
            probabilities = torch.softmax(masked_logits, dim=-1)

        batched_choice_probabilities = []
        for batch_idx in range(len(prompts)):
            choice_probabilities = {}
            choices = choices_list[batch_idx]
            for i, choice in enumerate(choices):
                letter_token = self.letters[i]
                first_token = self.processor.tokenizer.encode(
                    letter_token, add_special_tokens=False
                )[0]
                first_token_with_space = self.processor.tokenizer.encode(
                    ' ' + letter_token, add_special_tokens=False
                )[0]

                prob = 0.0
                if isinstance(first_token, int):
                    prob += probabilities[batch_idx, first_token].item()
                if isinstance(first_token_with_space, int):
                    prob += probabilities[batch_idx, first_token_with_space].item()
                choice_probabilities[choice] = prob

            # Renormalize probabilities for each question
            total_prob = sum(choice_probabilities.values())
            if total_prob > 0:
                renormalized_probabilities = {
                    choice: prob / total_prob
                    for choice, prob in choice_probabilities.items()
                }
            else:
                renormalized_probabilities = (
                    choice_probabilities  # Avoid division by zero if total_prob is 0
                )
            batched_choice_probabilities.append(renormalized_probabilities)

        return batched_choice_probabilities

    def ocr(self, image, free_chars=4):
        inputs = (
            self.processor(
                text='<image>ocr\n',
                images=image,
                return_tensors='pt',
            )
            .to(torch.float16)
            .to(self.model.device)
        )
        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
            outputs = outputs[0][input_len:]
            decoded = self.processor.decode(outputs, skip_special_tokens=True)

        num_char = len(decoded)

        # Exponentially decreasing towards 0.0 if more than free_chars detected
        return min(1.0, math.exp(-num_char + free_chars))


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self):
        self.model_path = '/kaggle/input/sac-logos-ava1-l14-linearmse/sac+logos+ava1-l14-linearMSE.pth'
        self.clip_model_path = '/kaggle/input/openai-clip-vit-large-patch14/ViT-L-14.pt'
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """Loads the aesthetic predictor model and CLIP model."""
        state_dict = torch.load(self.model_path, weights_only=True, map_location='cuda:1')

        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to('cuda:1')
        predictor.eval()
        clip_model, preprocessor = clip.load(self.clip_model_path, device='cuda:1')

        return predictor, clip_model, preprocessor

    def score(self, image: Image.Image) -> float:
        """Predicts the CLIP aesthetic score of an image."""
        image = self.preprocessor(image).unsqueeze(0).to('cuda:1')

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()

        score = self.predictor(torch.from_numpy(image_features).to('cuda:1').float())

        return score.item() / 10.0  # scale to [0, 1]


def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    """
    Calculate the harmonic mean of two values, weighted using a beta parameter.

    Args:
        a: First value (e.g., precision)
        b: Second value (e.g., recall)
        beta: Weighting parameter

    Returns:
        Weighted harmonic mean
    """
    # Handle zero values to prevent division by zero
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
        The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
        The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
        The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)


def load_jpeg(image_path: str, size: tuple = (384, 384)) -> Image.Image:
    return Image.open(image_path).convert('RGB').resize(size)

class ImageProcessor:
    def __init__(self, image: Image.Image, seed=None):
        """Initialize with either a path to an image or a PIL Image object."""
        self.image = image
        self.original_image = self.image.copy()
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def reset(self):
        self.image = self.original_image.copy()
        return self
    
    def visualize_comparison(
        self,
        original_name='Original',
        processed_name='Processed',
        figsize=(10, 5),
        show=True,
    ):
        """Display original and processed images side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(np.asarray(self.original_image))
        ax1.set_title(original_name)
        ax1.axis('off')

        ax2.imshow(np.asarray(self.image))
        ax2.set_title(processed_name)
        ax2.axis('off')

        title = f'{original_name} vs {processed_name}'
        fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def apply_median_filter(self, size=3):
        """Apply median filter to remove outlier pixel values.

        Args:
            size: Size of the median filter window.
        """
        self.image = self.image.filter(ImageFilter.MedianFilter(size=size))
        return self

    def apply_bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter to smooth while preserving edges.

        Args:
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
        """
        # Convert PIL Image to numpy array for OpenCV
        img_array = np.asarray(self.image)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)

        # Convert back to PIL Image
        self.image = Image.fromarray(filtered)
        return self

    def apply_fft_low_pass(self, cutoff_frequency=0.5):
        """Apply low-pass filter in the frequency domain using FFT.

        Args:
            cutoff_frequency: Normalized cutoff frequency (0-1).
                Lower values remove more high frequencies.
        """
        # Convert to numpy array, ensuring float32 for FFT
        img_array = np.array(self.image, dtype=np.float32)

        # Process each color channel separately
        result = np.zeros_like(img_array)
        for i in range(3):  # For RGB channels
            # Apply FFT
            f = np.fft.fft2(img_array[:, :, i])
            fshift = np.fft.fftshift(f)

            # Create a low-pass filter mask
            rows, cols = img_array[:, :, i].shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            r = int(min(crow, ccol) * cutoff_frequency)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 1

            # Apply mask and inverse FFT
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)

            result[:, :, i] = img_back

        # Clip to 0-255 range and convert to uint8 after processing all channels
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        self.image = Image.fromarray(result)
        return self

    def apply_jpeg_compression(self, quality=85):
        """Apply JPEG compression.

        Args:
            quality: JPEG quality (0-95). Lower values increase compression.
        """
        buffer = io.BytesIO()
        self.image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        self.image = Image.open(buffer)
        return self

    def apply_random_crop_resize(self, crop_percent=0.05):
        """Randomly crop and resize back to original dimensions.

        Args:
            crop_percent: Percentage of image to crop (0-0.4).
        """
        width, height = self.image.size
        crop_pixels_w = int(width * crop_percent)
        crop_pixels_h = int(height * crop_percent)

        left = self.rng.randint(0, crop_pixels_w + 1)
        top = self.rng.randint(0, crop_pixels_h + 1)
        right = width - self.rng.randint(0, crop_pixels_w + 1)
        bottom = height - self.rng.randint(0, crop_pixels_h + 1)

        self.image = self.image.crop((left, top, right, bottom))
        self.image = self.image.resize((width, height), Image.BILINEAR)
        return self

    def apply(self):
        """Apply an ensemble of defenses."""
        return (
            self.apply_random_crop_resize(crop_percent=0.03)
            .apply_jpeg_compression(quality=95)
            .apply_median_filter(size=9)
            .apply_fft_low_pass(cutoff_frequency=0.5)
            .apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
            .apply_jpeg_compression(quality=92)
        )


import pandas as pd

solution = pd.read_parquet("/kaggle/input/drawing-with-llms/questions.parquet").groupby('id').agg(list).reset_index()
submission = pd.read_csv("/kaggle/input/drawing-with-llms-scoring-bench-data/submission.csv")
qwen_submission = pd.read_json("/kaggle/input/drawing-with-llms-scoring-bench-data/submission_qwen_2_5_32b.jsonl", lines=True)
gemma_submission = pd.DataFrame({
    'id': [
        "02d892", "0dcd2e", "1e9ac1", "2b25db", "4e6a54", "4f1b00", "61b500", "65cc74", "7c4414", "996c3a", "9b71cc", "a395a3", "ad4c5c", "b679e3", "f16e62",
    ],
    'svg': [
        '<svg viewBox="0 0 256 256" width="256" height="256"><circle cx="128" cy="128" r="80" fill="red"/><rect x="88" y="88" width="80" height="80" fill="blue"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><rect x="64" y="64" width="128" height="160" fill="gray" stroke="black" stroke-width="2"/><polygon points="32,64 64,32 192,32 224,64 192,96 64,96" fill="white" stroke="black" stroke-width="2"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><rect x="0" y="128" width="256" height="128" fill="blue"/><g><rect x="100" y="40" width="60" height="88" fill="white"/><line x1="100" y1="40" x2="160" y2="40" stroke="black" stroke-width="3"/><line x1="100" y1="128" x2="160" y2="128" stroke="black" stroke-width="3"/><line x1="100" y1="40" x2="100" y2="128" stroke="black" stroke-width="3"/><circle cx="160" cy="80" r="15" fill="yellow"/></g></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><rect x="64" y="64" width="128" height="128" fill="#800020" stroke="#800020" stroke-width="10"/><rect x="80" y="80" width="16" height="32" fill="#800020" stroke="#800020" stroke-width="5"/><rect x="144" y="80" width="16" height="32" fill="#800020" stroke="#800020" stroke-width="5"/><circle cx="100" cy="128" r="6" fill="silver"/><circle cx="160" cy="128" r="6" fill="silver"/><circle cx="112" cy="160" r="6" fill="silver"/><circle cx="148" cy="160" r="6" fill="silver"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><rect x="64" y="64" width="128" height="160" fill="#FFA500" stroke="#FFA500" stroke-width="2"/><rect x="80" y="80" width="128" height="160" fill="#FFC066" stroke="#FFC066" stroke-width="2"/><path d="M 64 64 L 192 64" stroke="#FFA500" stroke-width="2"/><path d="M 64 64 L 64 224" stroke="#FFA500" stroke-width="2"/><path d="M 192 64 L 192 224" stroke="#FFA500" stroke-width="2"/><path d="M 64 224 L 192 224" stroke="#FFA500" stroke-width="2"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><defs><radialGradient id="scarfGradient" cx="50%" cy="50%" r="50%" fx="50%" fy="50%"><stop offset="0%" stop-color="#A020F0" stop-opacity="1"/><stop offset="100%" stop-color="#800080" stop-opacity="1"/></radialGradient></defs><path d="M16,16 C32,8 96,0 128,0 C160,0 224,8 232,16 L240,32 Q248,64 252,96 V160 Q252,224 248,256 L16,256 Z" fill="url(#scarfGradient)" stroke="#663399" stroke-width="2"/><polyline points="248,96 252,128 248,160" fill="none" stroke="#800080" stroke-width="4"/><polyline points="16,96 12,128 16,160" fill="none" stroke="#800080" stroke-width="4"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><rect width="100%" height="100%" fill="#ADD8E6"/><polygon points="0,256 128,200 256,256" fill="green"/><g transform="translate(128,64)"><circle cx="0" cy="0" r="20" fill="gray"/><circle cx="0" cy="0" r="15" fill="white"/></g><g transform="translate(128,128)"><circle cx="0" cy="0" r="20" fill="gray"/><circle cx="0" cy="0" r="15" fill="white"/></g><g transform="translate(128,192)"><circle cx="0" cy="0" r="20" fill="gray"/><circle cx="0" cy="0" r="15" fill="white"/></g></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><rect x="0" y="0" width="64" height="64" fill="crimson"/><rect x="64" y="0" width="64" height="64" fill="crimson"/><rect x="128" y="0" width="64" height="64" fill="crimson"/><rect x="0" y="64" width="64" height="64" fill="crimson"/><rect x="64" y="64" width="64" height="64" fill="crimson"/><rect x="128" y="64" width="64" height="64" fill="crimson"/><rect x="0" y="128" width="64" height="64" fill="crimson"/><rect x="64" y="128" width="64" height="64" fill="crimson"/><rect x="128" y="128" width="64" height="64" fill="crimson"/><rect x="192" y="0" width="64" height="64" fill="crimson"/><rect x="192" y="64" width="64" height="64" fill="crimson"/><rect x="192" y="128" width="64" height="64" fill="crimson"/><rect x="0" y="192" width="64" height="64" fill="crimson"/><rect x="64" y="192" width="64" height="64" fill="crimson"/><rect x="128" y="192" width="64" height="64" fill="crimson"/><rect x="192" y="192" width="64" height="64" fill="crimson"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><circle cx="128" cy="128" r="80" fill="red"/><rect x="88" y="88" width="80" height="80" fill="blue"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><rect x="0" y="0" width="256" height="256" fill="silver" opacity="0.5"/><polygon points="20,20 100,80 200,80 240,20" fill="magenta"/><polygon points="50,100 120,180 190,180 240,100" fill="magenta"/><polygon points="80,180 150,240 220,240 256,180" fill="magenta"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><rect x="0" y="0" width="256" height="256" fill="#f0f8ff"/><polyline points="0,128 L256,128" stroke="#ffffff" stroke-width="2"/><polygon points="128,0 192,64 256,128 192,192 128,256 64,192 0,128" fill="#ffffff" opacity="0.5"/><line x1="0" y1="64" x2="256" y2="64" stroke="#ffffff" stroke-width="1"/><line x1="0" y1="192" x2="256" y2="192" stroke="#ffffff" stroke-width="1"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><rect x="0" y="0" width="256" height="256" fill="white"/><g transform="translate(8,8)"><rect x="0" y="0" width="64" height="64" fill="black"/><rect x="64" y="0" width="64" height="64" fill="white"/><rect x="128" y="0" width="64" height="64" fill="black"/><rect x="192" y="0" width="64" height="64" fill="white"/><rect x="0" y="64" width="64" height="64" fill="white"/><rect x="64" y="64" width="64" height="64" fill="black"/><rect x="128" y="64" width="64" height="64" fill="white"/><rect x="192" y="64" width="64" height="64" fill="black"/><rect x="0" y="128" width="64" height="64" fill="black"/><rect x="64" y="128" width="64" height="64" fill="white"/><rect x="128" y="128" width="64" height="64" fill="white"/><rect x="192" y="128" width="64" height="64" fill="black"/><rect x="0" y="192" width="64" height="64" fill="white"/><rect x="64" y="192" width="64" height="64" fill="black"/><rect x="128" y="192" width="64" height="64" fill="white"/><rect x="192" y="192" width="64" height="64" fill="black"/></g></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><defs><linearGradient id="nightSkyGradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#111122"/><stop offset="100%" stop-color="#000000"/></linearGradient></defs><rect x="0" y="0" width="256" height="128" fill="url(#nightSkyGradient)"/><rect x="0" y="128" width="256" height="128" fill="#ffffff"/><polygon points="128,32 160,96 192,128 160,160 128,144 64,160 32,128 64,96" fill="#cccccc" stroke="#666666" stroke-width="2"/><polyline points="128,32 160,96 192,128 160,160 128,144 64,160 32,128" stroke="#666666" stroke-width="2" fill="none"/><circle cx="48" cy="180" r="8" fill="#ffffff"/><circle cx="176" cy="152" r="6" fill="#ffffff"/><circle cx="224" cy="80" r="4" fill="#ffffff"/><circle cx="32" cy="56" r="5" fill="#ffffff"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><polygon points="128,20 50,150 206,150" fill="khaki" stroke="black" stroke-width="2"/><polygon points="128,236 50,86 206,86" fill="khaki" stroke="black" stroke-width="2"/><path d="M 128,50 A 80 80 270 1 0 128,180" fill="azure" stroke="black" stroke-width="2"/><path d="M 128,180 A 80 80 90 1 0 128,50" fill="azure" stroke="black" stroke-width="2"/></svg>',
        '<svg viewBox="0 0 256 256" width="256" height="256"><defs><linearGradient id="maroonGradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#800000"/><stop offset="100%" stop-color="#B00020"/></linearGradient><linearGradient id="tealGradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#008080"/><stop offset="100%" stop-color="#00C0C0"/></linearGradient></defs><polygon points="128,20 144,70 192,88 176,136 144,164 112,164 80,136 64,88 32,70" fill="url(#maroonGradient)"/><polygon points="128,20 144,70 192,88 176,136 144,164 112,164 80,136 64,88 32,70" stroke="teal" stroke-width="4" fill="none"/><polyline points="128,20 144,70 192,88 176,136 144,164 112,164 80,136 64,88 32,70" stroke="url(#tealGradient)" stroke-width="4" fill="none"/></svg>',
    ],
})
base_imagen3_path = "/kaggle/input/drawing-with-llms-scoring-bench-data/gemini-drawing-with-llms-test-data"
imagen3_submission = pd.DataFrame({
    'id': [
        "02d892", "0dcd2e", "1e9ac1", "2b25db", "4e6a54", "4f1b00", "61b500", "65cc74", "7c4414", "996c3a", "9b71cc", "a395a3", "ad4c5c", "b679e3", "f16e62",
    ],
    'svg': [
        f'{base_imagen3_path}/purple_forest.jpg',
        f'{base_imagen3_path}/gray_wool.jpg',
        f'{base_imagen3_path}/lighthouse_overlooking.jpg',
        f'{base_imagen3_path}/burgundy_corduroy_pants.jpg',
        f'{base_imagen3_path}/orange_corduroy_overalls.jpg',
        f'{base_imagen3_path}/purple_silk_scarf.jpg',
        f'{base_imagen3_path}/green_lagoon.jpg',
        f'{base_imagen3_path}/crimson_rectangles.jpg',
        f'{base_imagen3_path}/purple_pyramids.jpg',
        f'{base_imagen3_path}/magenta_trapezoids.jpg',
        f'{base_imagen3_path}/snowy_plain.jpg',
        f'{base_imagen3_path}/black_and_white_checkered_pants.jpg',
        f'{base_imagen3_path}/starlit_night_snow.jpg',
        f'{base_imagen3_path}/khaki_triangles_and_azure_crescents.jpg',
        f'{base_imagen3_path}/maroon_dodecahedron_teal_threads.jpg',
    ],
})


# add the description columns to all of them, qwen's submission is the only one that already has it
submission['description'] = submission['id'].map(qwen_submission.set_index('id')['description'])
gemma_submission['description'] = gemma_submission['id'].map(qwen_submission.set_index('id')['description'])
imagen3_submission['description'] = imagen3_submission['id'].map(qwen_submission.set_index('id')['description'])


solution["choices"] = solution["choices"].map(lambda arr: str([list(x) for x in arr]))
solution["question"] = solution["question"].map(str)
solution["answer"] = solution["answer"].map(str)
solution


mymodel_score = score(solution.copy(), submission, "id")
qwen_score = score(solution.copy(), qwen_submission, "id")
gemma_score = score(solution.copy(), gemma_submission, "id")
imagen3_score = score(solution.copy(), imagen3_submission, "id")


def prepare_input_for_display(score, submission, label):
    avg_score, scor_dict = score
    df = submission
    
    for id in scor_dict:
        svg = df.loc[df['id'] == id, 'svg'].values[0]
        scor_dict[id] = {
            "img": svg_to_png(svg) if svg.startswith("<svg") else load_jpeg(svg),
            "score": scor_dict[id],
        }

    return (label, avg_score, scor_dict)


mymodel_display_data = prepare_input_for_display(mymodel_score, submission, "MyModel")
gemma_display_data = prepare_input_for_display(gemma_score, gemma_submission, "Gemma")
qwen_display_data = prepare_input_for_display(qwen_score, qwen_submission, "Qwen")
imagen3_display_data = prepare_input_for_display(imagen3_score, imagen3_submission, "Imagen3")


from PIL import Image
import matplotlib.pyplot as plt

def display_results(results, output_name="old_metrics.png"):
    # Separate the input tuples
    labels = []
    avg_scores = []
    dicts = []
    for label, avg_score, d in results:
        labels.append(label)
        avg_scores.append(avg_score)
        dicts.append(d)

    # Find all unique IDs
    ids = set()
    for d in dicts:
        ids.update(d.keys())
    ids = sorted(ids)

    # Create the figure
    num_rows = len(ids) + 1  # Extra row for labels
    num_cols = len(dicts)

    fig_height = 3 * num_rows + 1  # +1 for better separation
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, fig_height),
                              gridspec_kw={'height_ratios': [1]*len(ids) + [0.5]})  # Labels shorter
    
    # Handle edge cases
    if num_rows == 1:
        axes = [axes]
    if num_cols == 1:
        axes = [[ax] for ax in axes]

    # Plot images
    for row_idx, id_ in enumerate(ids):
        for col_idx, d in enumerate(dicts):
            ax = axes[row_idx][col_idx]
            ax.axis('off')
            
            if id_ in d:
                img = d[id_]['img']
                score = d[id_]['score']
                ax.imshow(img)
                ax.set_title(f"Score: {score:.2f}", fontsize=8)
            else:
                ax.set_facecolor('lightgray')
                ax.set_title("Missing", fontsize=8)

    # Plot labels in the last row
    for col_idx, (label, avg_score) in enumerate(zip(labels, avg_scores)):
        ax = axes[-1][col_idx]
        ax.axis('off')
        ax.set_facecolor('white')
        ax.text(0.5, 0.5, f"{label}: {avg_score:.2f}", fontsize=10, ha='center', va='center')

    # Draw a horizontal line between images and labels
    fig.canvas.draw()
    line_y = (num_rows - 1) / num_rows  # Position between last image row and label row
    fig.subplots_adjust(hspace=0.5)  # Add vertical space
    fig_line = plt.Line2D([0,1], [line_y,line_y], transform=fig.transFigure, color="black", linewidth=1)
    fig.add_artist(fig_line)

    plt.tight_layout()
    plt.savefig(output_name)
    plt.show()


display_results([imagen3_display_data, mymodel_display_data, qwen_display_data, gemma_display_data])


!pip install git+https://github.com/DeepLearnXMU/LLaVE deepspeed open_clip_torch
!pip install aesthetic-predictor-v2-5 cairosvg bitsandbytes


import gc
import torch
import time

del aesthetic_evaluator
del vqa_evaluator

def flush(device="cuda:0"):
    for i in range(5):
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        time.sleep(0.5)

flush()
flush("cuda:1")


# To prevent relading the models each time we run the score() function
aesthetic_evaluator = None
vqa_evaluator = None
similarity_evaluator = None


import ast
import io
import math
import copy
import statistics
import string

import cairosvg
import clip
import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from more_itertools import chunked
from PIL import Image, ImageFilter
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoConfig,
)

svg_constraints = kagglehub.package_import('metric/svg-constraints')


class ParticipantVisibleError(Exception):
    pass


def score_v2(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, random_seed: int = 0, 
    apply_defenses_for_esthetic=True,
    apply_defenses_for_vqa=True,
    apply_defenses_for_similarity=True,
) -> float:
    """Calculates a fidelity score by comparing generated SVG images to target text descriptions.

    Parameters
    ----------
    solution : pd.DataFrame
        A DataFrame containing target questions, choices, and answers about an SVG image.
    submission : pd.DataFrame
        A DataFrame containing generated SVG strings. Must have a column named 'svg'.
    row_id_column_name : str
        The name of the column containing row identifiers. This column is removed before scoring.
    random_seed : int
        A seed to set the random state.

    Returns
    -------
    float
        The mean fidelity score (a value between 0 and 1) representing the average similarity between the generated SVGs and their descriptions.
        A higher score indicates better fidelity.

    Raises
    ------
    ParticipantVisibleError
        If the 'svg' column in the submission DataFrame is not of string type or if validation of the SVG fails.

    Examples
    --------
    >>> import pandas as pd
    >>> solution = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'question': ['["Is there a red circle?", "What shape is present?"]'],
    ...     'choices': ['[["yes", "no"], ["square", "circle", "triangle", "hexagon"]]'],
    ...     'answer': ['["yes", "circle"]'],
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'svg': ['<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'],
    ...     'description': ["a purple forest at dusk"],
    ... })
    >>> score(solution, submission, 'row_id', random_seed=42)
    0...
    """
    # Convert solution fields to list dtypes and expand
    for colname in ['question', 'choices', 'answer']:
        solution[colname] = solution[colname].apply(ast.literal_eval)
    solution = solution.explode(['question', 'choices', 'answer'])

    # Validate
    if not pd.api.types.is_string_dtype(submission.loc[:, 'svg']):
        raise ParticipantVisibleError('svg must be a string.')

    # Check that SVG code meets defined constraints
    constraints = svg_constraints.SVGConstraints()
    try:
        for svg in submission.loc[:, 'svg']:
            if svg.startswith("<svg"):
                constraints.validate_svg(svg)
            else:
                # It's Imagen3 "submission"
                pass
    except:
        raise ParticipantVisibleError('SVG code violates constraints.')

    global vqa_evaluator
    global aesthetic_evaluator
    global similarity_evaluator
    
    vqa_evaluator = vqa_evaluator or VQAEvaluator()
    aesthetic_evaluator = aesthetic_evaluator or AestheticEvaluator()
    similarity_evaluator = similarity_evaluator or SimilarityEvaluator()

    results = {}
    rng = np.random.RandomState(random_seed)
    try:
        df = solution.merge(submission, on='id')
        for i, (_, group) in enumerate(df.loc[
            :, ['id', 'question', 'choices', 'answer', 'svg', 'description']
        ].groupby('id')):
            id = group['id'].iloc[0]
            questions, choices, answers, svg, description = [
                group[col_name].to_list()
                for col_name in group.drop('id', axis=1).columns
            ]
            svg_or_path = svg[0]  # unpack singleton from list
            group_seed = rng.randint(0, np.iinfo(np.int32).max)

            original_image = svg_to_png(svg_or_path) if svg_or_path.startswith("<svg") else load_jpeg(svg_or_path) # Imagen 3 has jpeg files
            image_processor = ImageProcessor(image=original_image, seed=group_seed).apply()
            image = image_processor.image.copy()
            
            aesthetic_score = aesthetic_evaluator.score(image if apply_defenses_for_esthetic else original_image)
            vqa_score = vqa_evaluator.score(
                questions,
                choices, 
                answers, 
                image if apply_defenses_for_vqa else original_image,
            )
            similarity_score = similarity_evaluator.score(
                image if apply_defenses_for_similarity else original_image, 
                description[0],
            )

            # Leave defenses for ocr
            image_processor.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
            ocr_score = vqa_evaluator.ocr(image_processor.image)
            
            vqa_esthetic_hm = harmonic_mean(vqa_score, aesthetic_score, beta=0.5)
            similarity_vqa_estheteic_hm = harmonic_mean(
                    similarity_score, 
                    vqa_esthetic_hm,
                    beta=0.5,
                )
            instance_score = (similarity_vqa_estheteic_hm * ocr_score )
            results[id] = instance_score

    except:
        raise ParticipantVisibleError('SVG failed to score.')

    fidelity = statistics.mean(results.values())
    return float(fidelity), results


class VQAEvaluator:
    """Evaluates images based on their similarity to a given text description using multiple choice questions."""

    def __init__(self):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.letters = string.ascii_uppercase
        self.model_path = kagglehub.model_download(
            'google/gemma-3/Transformers/gemma-3-12b-it-qat-int4-unquantized/1'
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
            torch_dtype=torch.bfloat16,
        ).to('cuda:0')
        
        self.ocr_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", # Not on kaggle yet 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
        ).to("cuda:1")
        self.ocr_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    def score(self, questions, choices, answers, image, n=4):
        scores = []
        batches = (chunked(qs, n) for qs in [questions, choices, answers])
        for question_batch, choice_batch, answer_batch in zip(*batches, strict=True):
            scores.extend(
                self.score_batch(
                    image,
                    question_batch,
                    choice_batch,
                    answer_batch,
                )
            )
        return statistics.mean(scores)

    def score_batch(
        self,
        image: Image.Image,
        questions: list[str],
        choices_list: list[list[str]],
        answers: list[str],
    ) -> list[float]:
        """Evaluates the image based on multiple choice questions and answers.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to evaluate.
        questions : list[str]
            List of questions about the image.
        choices_list : list[list[str]]
            List of lists of possible answer choices, corresponding to each question.
        answers : list[str]
            List of correct answers from the choices, corresponding to each question.

        Returns
        -------
        list[float]
            List of scores (values between 0 and 1) representing the probability of the correct answer for each question.
        """
        prompts = [
            self.format_prompt(question, choices)
            for question, choices in zip(questions, choices_list, strict=True)
        ]
        batched_choice_probabilities = self.get_choice_probability(
            image, prompts, choices_list
        )

        scores = []
        for i, _ in enumerate(questions):
            choice_probabilities = batched_choice_probabilities[i]
            answer = answers[i]
            answer_probability = 0.0
            for choice, prob in choice_probabilities.items():
                if choice == answer:
                    answer_probability = prob
                    break
            scores.append(answer_probability)

        return scores

    def format_prompt(self, question: str, choices: list[str]) -> str:
        prompt = f'Accurately answer to this multi-choice question about the given image. Don\'t say anything, just output the correct letter, ONLY the letter: {question}\nChoices:\n'
        for i, choice in enumerate(choices):
            prompt += f'{self.letters[i]}. {choice}\n'
        return prompt

    def mask_choices(self, logits, choices_list):
        """Masks logits for the first token of each choice letter for each question in the batch."""
        batch_size = logits.shape[0]
        masked_logits = torch.full_like(logits, float('-inf'))

        for batch_idx in range(batch_size):
            choices = choices_list[batch_idx]
            for i in range(len(choices)):
                letter_token = self.letters[i]

                first_token = self.processor.tokenizer.encode(
                    letter_token, add_special_tokens=False
                )[0]
                first_token_with_space = self.processor.tokenizer.encode(
                    ' ' + letter_token, add_special_tokens=False
                )[0]

                if isinstance(first_token, int):
                    masked_logits[batch_idx, first_token] = logits[
                        batch_idx, first_token
                    ]
                if isinstance(first_token_with_space, int):
                    masked_logits[batch_idx, first_token_with_space] = logits[
                        batch_idx, first_token_with_space
                    ]

        return masked_logits

    def apply_chat_format(self, image, prompt, system_prompt="You are a helpful assistant."):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
    def get_choice_probability(self, image, prompts, choices_list) -> list[dict]:
        batched_choice_probabilities = []
        
        for prompt, choices in zip(prompts, choices_list, strict=True):
            # Create a single message for one prompt
            message = self.apply_chat_format(image, prompt)
        
            # Tokenize only this one message
            inputs = self.processor.apply_chat_template(
                [message],  # wrap single message in list
                add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                padding='longest',
            ).to(self.model.device, dtype=self.model.dtype)
        
            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)
                masked_logits = self.mask_choices(logits, [choices])  # choices needs to be in a batch of size 1
                probabilities = torch.softmax(masked_logits, dim=-1)
        
            # Now decode probabilities for the choices
            choice_probabilities = {}
            for i, choice in enumerate(choices):
                letter_token = self.letters[i]
                first_token = self.processor.tokenizer.encode(
                    letter_token, add_special_tokens=False
                )[0]
                first_token_with_space = self.processor.tokenizer.encode(
                    ' ' + letter_token, add_special_tokens=False
                )[0]
        
                prob = 0.0
                if isinstance(first_token, int):
                    prob += probabilities[0, first_token].item()
                if isinstance(first_token_with_space, int):
                    prob += probabilities[0, first_token_with_space].item()
                choice_probabilities[choice] = prob
        
            # Renormalize
            total_prob = sum(choice_probabilities.values())
            if total_prob > 0:
                renormalized_probabilities = {
                    choice: prob / total_prob
                    for choice, prob in choice_probabilities.items()
                }
            else:
                renormalized_probabilities = choice_probabilities
        
            batched_choice_probabilities.append(renormalized_probabilities)

        return batched_choice_probabilities

    def ocr(self, image, free_chars=4):
        inputs = self.ocr_processor(
            text="<OCR>", 
            images=image, 
            return_tensors="pt",
        ).to(self.ocr_model.device, self.ocr_model.dtype)


        with torch.inference_mode():
            generated_ids = self.ocr_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=3,
                do_sample=False
            )
            generated_text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            decoded = self.ocr_processor.post_process_generation(
                generated_text, 
                task="<OCR>", 
                image_size=(image.width, image.height),
            )["<OCR>"].removesuffix("\n") # Florence 2 always adds new line 

        num_char = len(decoded)

        # Exponentially decreasing towards 0.0 if more than free_chars detected
        return min(1.0, math.exp(-num_char + free_chars))


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self):
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = self.model.to(device="cuda:1",dtype=torch.float16)

    def score(self, image: Image.Image) -> float:
        # preprocess image
        pixel_values = (
            self.preprocessor(images=image, return_tensors="pt")
            .pixel_values.to(device=self.model.device,dtype=self.model.dtype)
        )
        
        # predict aesthetic score
        with torch.inference_mode():
            score = self.model(pixel_values).logits.squeeze().float().cpu().numpy()
            
        return score / 10


def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    """
    Calculate the harmonic mean of two values, weighted using a beta parameter.

    Args:
        a: First value (e.g., precision)
        b: Second value (e.g., recall)
        beta: Weighting parameter

    Returns:
        Weighted harmonic mean
    """
    # Handle zero values to prevent division by zero
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
        The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
        The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
        The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)


def load_jpeg(image_path: str, size: tuple = (384, 384)) -> Image.Image:
    return Image.open(image_path).convert('RGB').resize(size)

class ImageProcessor:
    def __init__(self, image: Image.Image, seed=None):
        """Initialize with either a path to an image or a PIL Image object."""
        self.image = image
        self.original_image = self.image.copy()
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def reset(self):
        self.image = self.original_image.copy()
        return self
    
    def visualize_comparison(
        self,
        original_name='Original',
        processed_name='Processed',
        figsize=(10, 5),
        show=True,
    ):
        """Display original and processed images side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(np.asarray(self.original_image))
        ax1.set_title(original_name)
        ax1.axis('off')

        ax2.imshow(np.asarray(self.image))
        ax2.set_title(processed_name)
        ax2.axis('off')

        title = f'{original_name} vs {processed_name}'
        fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def apply_median_filter(self, size=3):
        """Apply median filter to remove outlier pixel values.

        Args:
            size: Size of the median filter window.
        """
        self.image = self.image.filter(ImageFilter.MedianFilter(size=size))
        return self

    def apply_bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter to smooth while preserving edges.

        Args:
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
        """
        # Convert PIL Image to numpy array for OpenCV
        img_array = np.asarray(self.image)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)

        # Convert back to PIL Image
        self.image = Image.fromarray(filtered)
        return self

    def apply_fft_low_pass(self, cutoff_frequency=0.5):
        """Apply low-pass filter in the frequency domain using FFT.

        Args:
            cutoff_frequency: Normalized cutoff frequency (0-1).
                Lower values remove more high frequencies.
        """
        # Convert to numpy array, ensuring float32 for FFT
        img_array = np.array(self.image, dtype=np.float32)

        # Process each color channel separately
        result = np.zeros_like(img_array)
        for i in range(3):  # For RGB channels
            # Apply FFT
            f = np.fft.fft2(img_array[:, :, i])
            fshift = np.fft.fftshift(f)

            # Create a low-pass filter mask
            rows, cols = img_array[:, :, i].shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            r = int(min(crow, ccol) * cutoff_frequency)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 1

            # Apply mask and inverse FFT
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)

            result[:, :, i] = img_back

        # Clip to 0-255 range and convert to uint8 after processing all channels
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        self.image = Image.fromarray(result)
        return self

    def apply_jpeg_compression(self, quality=85):
        """Apply JPEG compression.

        Args:
            quality: JPEG quality (0-95). Lower values increase compression.
        """
        buffer = io.BytesIO()
        self.image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        self.image = Image.open(buffer)
        return self

    def apply_random_crop_resize(self, crop_percent=0.05):
        """Randomly crop and resize back to original dimensions.

        Args:
            crop_percent: Percentage of image to crop (0-0.4).
        """
        width, height = self.image.size
        crop_pixels_w = int(width * crop_percent)
        crop_pixels_h = int(height * crop_percent)

        left = self.rng.randint(0, crop_pixels_w + 1)
        top = self.rng.randint(0, crop_pixels_h + 1)
        right = width - self.rng.randint(0, crop_pixels_w + 1)
        bottom = height - self.rng.randint(0, crop_pixels_h + 1)

        self.image = self.image.crop((left, top, right, bottom))
        self.image = self.image.resize((width, height), Image.BILINEAR)
        return self

    def apply(self):
        """Apply an ensemble of defenses."""
        return (
            self.apply_random_crop_resize(crop_percent=0.03)
            .apply_jpeg_compression(quality=95)
            .apply_median_filter(size=9)
            .apply_fft_low_pass(cutoff_frequency=0.5)
            .apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
            .apply_jpeg_compression(quality=92)
        )


class SimilarityEvaluator:
    def __init__(self):
        pretrained = "zhibinlan/LLaVE-7B"
        model_name = "llava_qwen"
        device = "cuda:1"
        self.conv_template = "qwen_1_5"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            pretrained, 
            None, 
            model_name, 
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_compute_dtype=torch.float16, 
            ),
            attn_implementation=None,
            customized_config=AutoConfig.from_pretrained(pretrained, trust_remote_code=True),
            device_map=device,
            torch_dtype="float16",
        )
        
        self.model.eval()
        self.model.get_vision_tower().to(device, dtype=torch.float16).eval()


    def score(self, image: Image.Image, prompt: str):
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=self.model.dtype, device=self.model.device) for _image in image_tensor]
        
        question = DEFAULT_IMAGE_TOKEN + " Represent the given image with the following question: What is in the image"
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], "\n")
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id).to(self.model.device)
        image_sizes = [image.size]
        
        with torch.inference_mode():
            query_embed = self.model.encode_multimodal_embeddings(input_ids, attention_mask=attention_mask,images=image_tensor, image_sizes=image_sizes)
        
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "\n")
        target_string = conv.get_prompt()
        target_input_ids = self.tokenizer(target_string, return_tensors="pt").input_ids.to(self.model.device)
        attention_mask=target_input_ids.ne(self.tokenizer.pad_token_id)
        
        with torch.inference_mode():
            target_embed = self.model.encode_multimodal_embeddings(target_input_ids, attention_mask=attention_mask)
        
        return (query_embed @ target_embed.T).item()



mymodel_score = score_v2(solution.copy(), submission, "id")
qwen_score = score_v2(solution.copy(), qwen_submission, "id")
gemma_score = score_v2(solution.copy(), gemma_submission, "id")
imagen3_score = score_v2(solution.copy(), imagen3_submission, "id")


mymodel_display_data = prepare_input_for_display(mymodel_score, submission, "MyModel")
gemma_display_data = prepare_input_for_display(gemma_score, gemma_submission, "Gemma")
qwen_display_data = prepare_input_for_display(qwen_score, qwen_submission, "Qwen")
imagen3_display_data = prepare_input_for_display(imagen3_score, imagen3_submission, "Imagen3")


display_results([imagen3_display_data, mymodel_display_data, qwen_display_data, gemma_display_data], output_name="new_fixed_metrics.png")


def run_scoring(
    apply_defenses_for_esthetic=True,
    apply_defenses_for_vqa=True,
    apply_defenses_for_similarity=True,
    output_name="metrics.png",
):
    mymodel_score = score_v2(
        solution.copy(), 
        submission, 
        "id", 
        apply_defenses_for_esthetic=apply_defenses_for_esthetic, 
        apply_defenses_for_vqa=apply_defenses_for_vqa,
        apply_defenses_for_similarity=apply_defenses_for_similarity,
    )
    qwen_score = score_v2(
        solution.copy(), 
        qwen_submission, 
        "id", 
        apply_defenses_for_esthetic=apply_defenses_for_esthetic, 
        apply_defenses_for_vqa=apply_defenses_for_vqa,
        apply_defenses_for_similarity=apply_defenses_for_similarity,
    )
    gemma_score = score_v2(
        solution.copy(), 
        gemma_submission, 
        "id", 
        apply_defenses_for_esthetic=apply_defenses_for_esthetic, 
        apply_defenses_for_vqa=apply_defenses_for_vqa,
        apply_defenses_for_similarity=apply_defenses_for_similarity,
    )
    imagen3_score = score_v2(
        solution.copy(), 
        imagen3_submission, 
        "id", 
        apply_defenses_for_esthetic=apply_defenses_for_esthetic, 
        apply_defenses_for_vqa=apply_defenses_for_vqa,
        apply_defenses_for_similarity=apply_defenses_for_similarity,
    )

    mymodel_display_data = prepare_input_for_display(mymodel_score, submission, "MyModel")
    gemma_display_data = prepare_input_for_display(gemma_score, gemma_submission, "Gemma")
    qwen_display_data = prepare_input_for_display(qwen_score, qwen_submission, "Qwen")
    imagen3_display_data = prepare_input_for_display(imagen3_score, imagen3_submission, "Imagen3")

    display_results([imagen3_display_data, mymodel_display_data, qwen_display_data, gemma_display_data], output_name=output_name)


run_scoring(apply_defenses_for_esthetic=False, output_name="new_metrics_no_esthetic_def.png")


run_scoring(apply_defenses_for_vqa=False, output_name="new_metrics_no_vqa_def.png")


run_scoring(apply_defenses_for_similarity=False, output_name="new_metrics_no_similarity_def.png") 


run_scoring(apply_defenses_for_similarity=False, apply_defenses_for_esthetic=False, output_name="new_metrics_no_similarity_&_esthetic_def.png")


run_scoring(apply_defenses_for_vqa=False, apply_defenses_for_esthetic=False, output_name="new_metrics_no_vqa_&_esthetic_def.png")


run_scoring(apply_defenses_for_vqa=False, apply_defenses_for_similarity=False, output_name="new_metrics_no_vqa_&_similarity_def.png")


run_scoring(
    apply_defenses_for_vqa=False, 
    apply_defenses_for_esthetic=False, 
    apply_defenses_for_similarity=False, 
    output_name="new_metrics_no_vqa_&_esthetic_&_similarity_def.png",
)

