!pip install -U ultralytics sahi -q



!pip install ensemble-boxes -q


from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Model yollarÄ±nÄ±z
model1_path = "/kaggle/input/multiscale-yolo-tta-weighted-boxes-fusion/runs/detect/train/weights/best.pt"
model2_path = "/kaggle/input/sahi-yolo-tta-weighted-boxes-fusion/runs/detect/train2/weights/best.pt"

# Modelleri SAHI ile yÃ¼kleme
detection_model1 = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=model1_path,
    confidence_threshold=0.3, # Bu deÄŸeri ihtiyacÄ±nÄ±za gÃ¶re ayarlayabilirsiniz
    device="cuda:0" # veya "cpu"
)

detection_model2 = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=model2_path,
    confidence_threshold=0.3,
    device="cuda:1"
)



import os
import gc
from pathlib import Path
import pandas as pd
import csv
from PIL import Image
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion
from tqdm.auto import tqdm
import collections

def run_optimized_single_gpu_inference(model_paths, test_images_path, submission_path="submission.csv", confidence_threshold=0.3):
    """
    Tek GPU'da bellek optimizasyonu ile hÄ±zlÄ± Ã§Ä±karÄ±m yapar.
    """
    # GPU bellek temizliÄŸi
    torch.cuda.empty_cache()
    gc.collect()
    
    test_image_paths = sorted(list(Path(test_images_path).glob("*.[jp][pn]g")))
    print(f"Toplam {len(test_image_paths)} gÃ¶rÃ¼ntÃ¼ iÅŸlenecek")
    
    # TÃ¼m tahminleri saklayacak ana sÃ¶zlÃ¼k
    all_predictions = collections.defaultdict(lambda: {"boxes_list": [], "scores_list": [], "labels_list": []})
    
    # Her model iÃ§in sÄ±rayla Ã§Ä±karÄ±m yap
    for model_idx, model_path in enumerate(model_paths):
        print(f"\n=== Model {model_idx + 1}/{len(model_paths)} iÅŸleniyor ===")
        print(f"Model: {Path(model_path).name}")
        
        # Modeli yÃ¼kle
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="cuda:0"
        )
        
        
        # Ä°lerleme Ã§ubuÄŸu
        pbar = tqdm(test_image_paths, desc=f"Model {model_idx + 1} Ã‡Ä±karÄ±m", leave=True)
        
        for img_path in pbar:
            image_id = img_path.stem
            
            try:
                # GÃ¶rÃ¼ntÃ¼ boyutlarÄ±nÄ± al
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # SAHI ile Ã§Ä±karÄ±m - optimize edilmiÅŸ parametreler
                result = get_sliced_prediction(
                    str(img_path),
                    detection_model,
                    slice_height=640,  # Daha bÃ¼yÃ¼k slice boyutu (daha az parÃ§a)
                    slice_width=640,
                    overlap_height_ratio=0.15,  # Daha az overlap (daha hÄ±zlÄ±)
                    overlap_width_ratio=0.15,
                    verbose=0  # Sessiz mod
                )
                
                # SonuÃ§larÄ± iÅŸle
                boxes, scores, labels = [], [], []
                for pred in result.object_prediction_list:
                    bbox = pred.bbox
                    x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                    boxes.append([x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height])
                    scores.append(pred.score.value)
                    labels.append(pred.category.id)
                
                # SonuÃ§larÄ± kaydet
                if boxes:
                    all_predictions[image_id]["boxes_list"].append(boxes)
                    all_predictions[image_id]["scores_list"].append(scores)
                    all_predictions[image_id]["labels_list"].append(labels)
                
                # Ä°lerleme bilgisi gÃ¼ncelle
                pbar.set_postfix({
                    'GPU_Mem': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
                    'Objects': len(boxes)
                })
                
            except Exception as e:
                print(f"Hata - GÃ¶rÃ¼ntÃ¼: {img_path.name}, Hata: {e}")
                continue
        
        # Model iÅŸlemi bitince belleÄŸi temizle
        del detection_model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Model {model_idx + 1} tamamlandÄ±, bellek temizlendi")
    
    # WBF ile sonuÃ§larÄ± birleÅŸtir
    print("\n=== WBF ile sonuÃ§lar birleÅŸtiriliyor ===")
    submission_data = []
    
    for img_path in tqdm(test_image_paths, desc="Submission oluÅŸturuluyor"):
        image_id = img_path.stem
        preds = all_predictions[image_id]
        
        if not preds["boxes_list"]:
            pred_str = "no boxes"
        else:
            # WBF uygula
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                preds["boxes_list"],
                preds["scores_list"],
                preds["labels_list"],
                iou_thr=0.5,
                skip_box_thr=0.01
            )
            
            # SonuÃ§larÄ± formatla
            pred_str_parts = []
            for b, s, l in zip(fused_boxes, fused_scores, fused_labels):
                x_center = b[0] + (b[2] - b[0]) / 2
                y_center = b[1] + (b[3] - b[1]) / 2
                width = b[2] - b[0]
                height = b[3] - b[1]
                pred_str_parts.append(f"{int(l)} {s:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            pred_str = " ".join(pred_str_parts) if pred_str_parts else "no boxes"
        
        submission_data.append({"image_id": image_id, "prediction_string": pred_str})
    
    # CSV dosyasÄ±nÄ± kaydet
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(submission_path, index=False, quoting=csv.QUOTE_MINIMAL)
    
    print(f"\nâœ… Ä°ÅŸlem tamamlandÄ±!")
    print(f"ğŸ“� Dosya kaydedildi: {submission_path}")
    print(f"ğŸ“Š Toplam {len(submission_df)} gÃ¶rÃ¼ntÃ¼ iÅŸlendi")
    print("\nÄ°lk 5 satÄ±r:")
    print(submission_df.head())



# Model yollarÄ±
model_paths = [
    "/kaggle/input/multiscale-yolo-tta-weighted-boxes-fusion/runs/detect/train/weights/best.pt",
    "/kaggle/input/sahi-yolo-tta-weighted-boxes-fusion/runs/detect/train2/weights/best.pt"
]

test_images_path = "/kaggle/input/multi-instance-object-detection-challenge/Starter_Dataset/TestImages/images"

# Optimized inference'Ä± baÅŸlat
run_optimized_single_gpu_inference(
    model_paths=model_paths,
    test_images_path=test_images_path,
    confidence_threshold=0.15
)





