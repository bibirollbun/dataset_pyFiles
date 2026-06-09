from metric import VQAEvaluator, ImageProcessor, AestheticEvaluator, harmonic_mean


vqa_evaluator = VQAEvaluator()


aesthetic_evaluator = AestheticEvaluator()


import numpy as np


def evaluate_score(image):
    processor = ImageProcessor(image)
    processor.apply()
    image = processor.image
    display(image)

    questions = [
        "What color is the lagoon?",
        "What color are the clouds?",
    ]
    choices = [
        ["black", "blue", "navy", "purple", "orange", "green", "red"],
        ["green", "black", "cyan", "white", "purple", "blue", "red"],
    ]
    answers = [
        "green",
        "white",
    ]

    qa_score = vqa_evaluator.score(questions, choices, answers, image)
    ocr_score = vqa_evaluator.ocr(processor.original_image)
    ocr_score_processed = vqa_evaluator.ocr(processor.image)
    aesthetic_score = aesthetic_evaluator.score(image)
    instance_score = (
        harmonic_mean(qa_score, aesthetic_score, beta=0.5)
        * ocr_score
    )
    print("Questions:")
    for question in questions:
        print(f" - {question}")
    print(
        f"QA score: {qa_score:.2f}"
    )
    print(
        f"OCR score: {ocr_score:.2f}"
    )
    print(
        f"OCR score (processed): {ocr_score_processed:.2f}"
    )
    print(f"Aesthetic score: {aesthetic_score:.2f}")
    print(f"Instance score: {instance_score:.2f}")


from PIL import Image
import numpy as np

image = Image.open("/kaggle/input/dwllms-edgepixelattack/adv-5.png")
#image = image.resize((384, 384), Image.LANCZOS)
display(image)
evaluate_score(image)

