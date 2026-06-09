pip install ultralytics


from ultralytics import YOLO
from pathlib import Path
import os
import pandas as pd
import csv

# 使用する重みファイル
weights_path = "/kaggle/input/best/other/default/1/best.pt"

# テスト画像と出力ディレクトリ
test_images_path = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images"
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)

# モデルのロード
model = YOLO(weights_path)

# 推論とラベルファイル出力
for img_path in Path(test_images_path).glob("*"):
    if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        continue
    results = model.predict(img_path, conf=0.05, verbose=False)
    output_txt = Path(output_dir) / f"{img_path.stem}.txt"
    with open(output_txt, "w") as f:
        for result in results:
            img_height, img_width = result.orig_shape
            boxes = result.boxes.data
            if boxes is None or len(boxes) == 0:
                continue
            filtered_boxes = boxes[boxes[:, 4] >= 0.05]
            if len(filtered_boxes) == 0:
                continue
            for box in filtered_boxes:
                x1, y1, x2, y2, confidence, cls_id = box.tolist()
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                f.write(f"{int(cls_id)} {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"[notice] ✅ All detections saved to: {output_dir}")

# submission.csv作成関数
def predictions_to_csv(
    preds_folder: str = "/kaggle/working/", 
    output_csv: str = "submission.csv", 
    test_images_folder: str = "/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images",
    allowed_extensions: tuple = (".jpg", ".png", ".jpeg")
):
    preds_path = Path(preds_folder)
    test_images_path = Path(test_images_folder)
    test_images = {p.stem for p in test_images_path.glob("*") if p.suffix.lower() in allowed_extensions}
    predictions = []
    predicted_images = set()
    for txt_file in preds_path.glob("*.txt"):
        image_id = txt_file.stem
        predicted_images.add(image_id)
        with open(txt_file, "r") as f:
            valid_lines = [line.strip() for line in f if len(line.strip().split()) == 6]
        pred_str = " ".join(valid_lines) if valid_lines else "no boxes"
        predictions.append({"image_id": image_id, "prediction_string": pred_str})
    missing_images = test_images - predicted_images
    for image_id in missing_images:
        predictions.append({"image_id": image_id, "prediction_string": "no boxes"})
    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df.sort_values("image_id")
    submission_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[notice] ✅ Submission saved to {output_csv}")

# 実行
predictions_to_csv()

