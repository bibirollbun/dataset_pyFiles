# ğŸŸ¦å®�è·µä»»åŠ¡ï¼šå�¯è§†åŒ–æµ‹è¯•é›†ä¸­æ‰€æœ‰å›¾ç‰‡çš„æ¨¡å�‹é¢„æµ‹ç»“æ�œ

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import cv2
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("å°�éº¦ç›®æ ‡æ£€æµ‹ - æµ‹è¯•é›†å�¯è§†åŒ–å®�è·µä»»åŠ¡")
print("="*60)

# ============================================
# 1. è®¾ç½®ä¸­æ–‡å­—ä½“
# ============================================
print("\n1. è®¾ç½®ä¸­æ–‡å­—ä½“...")

chinese_font_path = '/kaggle/input/chinese-font'
if os.path.exists(chinese_font_path):
    # æŸ¥æ‰¾å­—ä½“æ–‡ä»¶
    font_files = [f for f in os.listdir(chinese_font_path) 
                 if f.lower().endswith(('.ttf', '.otf', '.ttc'))]
    
    if font_files:
        font_file = os.path.join(chinese_font_path, font_files[0])
        print(f"  ä½¿ç”¨å­—ä½“æ–‡ä»¶: {font_files[0]}")
        
        try:
            from matplotlib.font_manager import FontProperties, fontManager
            fontManager.addfont(font_file)
            prop = FontProperties(fname=font_file)
            font_name = prop.get_name()
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  âœ“ ä¸­æ–‡å­—ä½“è®¾ç½®æˆ�åŠŸ: {font_name}")
        except Exception as e:
            print(f"  å­—ä½“è®¾ç½®å¤±è´¥: {e}")
            print("  ä½¿ç”¨é»˜è®¤å­—ä½“")
    else:
        print("  æœªæ‰¾åˆ°å­—ä½“æ–‡ä»¶ï¼Œä½¿ç”¨é»˜è®¤å­—ä½“")
else:
    print(f"  è­¦å‘Š: ä¸­æ–‡å­—ä½“è·¯å¾„ä¸�å­˜åœ¨: {chinese_font_path}")
    print("  ä½¿ç”¨é»˜è®¤å­—ä½“")

# ============================================
# 2. æ£€æŸ¥æ•°æ�®è·¯å¾„
# ============================================
print("\n2. æ£€æŸ¥æ•°æ�®è·¯å¾„...")

data_paths = {
    'wheat_data': '/kaggle/input/global-wheat-detection',
    'chinese_font': '/kaggle/input/chinese-font',
    'fasterrcnn': '/kaggle/input/fasterrcnn'
}

for name, path in data_paths.items():
    if os.path.exists(path):
        print(f"  âœ“ {name}: {path}")
        if name != 'chinese_font':
            # æ˜¾ç¤ºå†…å®¹
            items = os.listdir(path)[:5]  # æ˜¾ç¤ºå‰�5ä¸ª
            print(f"    å†…å®¹ç¤ºä¾‹: {', '.join(items)}")
            if len(os.listdir(path)) > 5:
                print(f"    è¿˜æœ‰ {len(os.listdir(path))-5} ä¸ªæ–‡ä»¶/æ–‡ä»¶å¤¹")
    else:
        print(f"  âœ— {name}: è·¯å¾„ä¸�å­˜åœ¨")

# ============================================
# 3. åŠ è½½æ•°æ�®
# ============================================
print("\n3. åŠ è½½æ•°æ�®...")

# åŠ è½½è®­ç»ƒæ•°æ�®
train_csv_path = '/kaggle/input/global-wheat-detection/train.csv'
submit_csv_path = '/kaggle/input/global-wheat-detection/sample_submission.csv'

if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    print(f"  âœ“ è®­ç»ƒæ•°æ�®åŠ è½½æˆ�åŠŸ: {train_df.shape}")
else:
    print(f"  âœ— è®­ç»ƒæ•°æ�®ä¸�å­˜åœ¨: {train_csv_path}")
    train_df = pd.DataFrame()

if os.path.exists(submit_csv_path):
    submit_df = pd.read_csv(submit_csv_path)
    print(f"  âœ“ æ��äº¤æ•°æ�®åŠ è½½æˆ�åŠŸ: {submit_df.shape}")
else:
    print(f"  âœ— æ��äº¤æ•°æ�®ä¸�å­˜åœ¨: {submit_csv_path}")
    submit_df = pd.DataFrame()

# è®¾ç½®è·¯å¾„
train_dir = '/kaggle/input/global-wheat-detection/train'
test_dir = '/kaggle/input/global-wheat-detection/test'

if os.path.exists(train_dir):
    train_images = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
    print(f"  âœ“ è®­ç»ƒå›¾åƒ�: {len(train_images)} å¼ ")
else:
    print(f"  âœ— è®­ç»ƒå›¾åƒ�ç›®å½•ä¸�å­˜åœ¨: {train_dir}")

if os.path.exists(test_dir):
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    print(f"  âœ“ æµ‹è¯•å›¾åƒ�: {len(test_images)} å¼ ")
else:
    print(f"  âœ— æµ‹è¯•å›¾åƒ�ç›®å½•ä¸�å­˜åœ¨: {test_dir}")

# ============================================
# 4. æ•°æ�®é¢„å¤„ç�†ï¼ˆä¸�ä¹‹å‰�ä»£ç �ä¿�æŒ�ä¸€è‡´ï¼‰
# ============================================
print("\n4. æ•°æ�®é¢„å¤„ç�†...")

if not train_df.empty:
    # åˆ é™¤ä¸�éœ€è¦�çš„åˆ—
    train_df = train_df.drop(columns=['width', 'height', 'source'], errors='ignore')
    
    # è§£æ��bboxå­—æ®µ
    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    def expand_bbox(x):
        r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
        if len(r) == 0:
            r = [-1, -1, -1, -1]
        return r

    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x))) 
    train_df['x'] = train_df['x'].astype(np.float16)                                          
    train_df['y'] = train_df['y'].astype(np.float16)
    train_df['w'] = train_df['w'].astype(np.float16)
    train_df['h'] = train_df['h'].astype(np.float16)
    
    print(f"  âœ“ æ•°æ�®é¢„å¤„ç�†å®Œæˆ�")
    print(f"    å”¯ä¸€å›¾åƒ�IDæ•°é‡�: {train_df['image_id'].nunique()}")
    print(f"    æœ€å¤§æ ‡æ³¨æ¡†æ•°: {train_df['image_id'].value_counts().max()}")
    print(f"    æœ€å°�æ ‡æ³¨æ¡†æ•°: {train_df['image_id'].value_counts().min()}")

# ============================================
# 5. å®šä¹‰æ•°æ�®é›†ç±»ï¼ˆä¸�ä¹‹å‰�ä»£ç �ä¿�æŒ�ä¸€è‡´ï¼‰
# ============================================
print("\n5. å®šä¹‰æ•°æ�®é›†ç±»...")

class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None, train=True):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.train = train

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        # è¯»å�–å›¾åƒ�
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"è­¦å‘Š: æ— æ³•è¯»å�–å›¾åƒ� {image_path}")
            image = np.zeros((1024, 1024, 3), dtype=np.float32)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
        
        if self.transforms is not None:  
            image = self.transforms(image)
        
        if self.train == False:  
            return image, image_id
        
        # è®­ç»ƒæ¨¡å¼�ï¼šè�·å�–æ ‡æ³¨ä¿¡æ�¯
        records = self.df[self.df['image_id'] == image_id]   
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x + w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y + h
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float16)

        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int32)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index]),
            'area': area,
            'iscrowd': iscrowd
        }

        return image, target, image_id

# ============================================
# 6. è¾…åŠ©å‡½æ•°
# ============================================
def collate_fn(batch):
    return tuple(zip(*batch))

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

# ============================================
# 7. æ£€æŸ¥æ¨¡å�‹æ�ƒé‡�æ–‡ä»¶
# ============================================
print("\n6. æ£€æŸ¥æ¨¡å�‹æ�ƒé‡�æ–‡ä»¶...")

fasterrcnn_path = '/kaggle/input/fasterrcnn'
WEIGHTS_FILE = None

if os.path.exists(fasterrcnn_path):
    model_files = [f for f in os.listdir(fasterrcnn_path) 
                  if f.lower().endswith(('.pth', '.pt', '.pth.tar'))]
    
    if model_files:
        WEIGHTS_FILE = os.path.join(fasterrcnn_path, model_files[0])
        print(f"  âœ“ æ‰¾åˆ°æ¨¡å�‹æ�ƒé‡�æ–‡ä»¶: {model_files[0]}")
        
        # æ£€æŸ¥æ–‡ä»¶å¤§å°�
        file_size = os.path.getsize(WEIGHTS_FILE) / (1024*1024)  # MB
        print(f"    æ–‡ä»¶å¤§å°�: {file_size:.2f} MB")
    else:
        print(f"  âœ— æœªæ‰¾åˆ°æ¨¡å�‹æ�ƒé‡�æ–‡ä»¶")
else:
    print(f"  âœ— æ¨¡å�‹è·¯å¾„ä¸�å­˜åœ¨: {fasterrcnn_path}")

# ============================================
# 8. åˆ›å»ºæµ‹è¯•æ•°æ�®é›†
# ============================================
print("\n7. åˆ›å»ºæµ‹è¯•æ•°æ�®é›†...")

if os.path.exists(test_dir) and not submit_df.empty:
    trans = transforms.Compose([transforms.ToTensor()])
    
    test_dataset = WheatDataset(submit_df, test_dir, trans, False)
    print(f"  âœ“ æµ‹è¯•æ•°æ�®é›†åˆ›å»ºæˆ�åŠŸ")
    print(f"    æµ‹è¯•å›¾åƒ�æ•°é‡�: {len(test_dataset)}")
    
    # åˆ›å»ºæ•°æ�®åŠ è½½å™¨
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=4,  # è¾ƒå°�çš„æ‰¹æ¬¡å¤§å°�ï¼Œä¾¿äº�å†…å­˜ç®¡ç�†
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"  âœ“ æµ‹è¯•æ•°æ�®åŠ è½½å™¨åˆ›å»ºæˆ�åŠŸ")
    print(f"    æ‰¹æ¬¡å¤§å°�: 4")
    print(f"    æ€»æ‰¹æ¬¡æ•°: {len(test_data_loader)}")
    
    # æµ‹è¯•æ•°æ�®é›†åˆ›å»ºæˆ�åŠŸæ ‡å¿—
    dataset_created = True
else:
    print(f"  âœ— æ— æ³•åˆ›å»ºæµ‹è¯•æ•°æ�®é›†")
    dataset_created = False

# ============================================
# 9. åŠ è½½æ¨¡å�‹
# ============================================
print("\n8. åŠ è½½æ¨¡å�‹...")

if WEIGHTS_FILE and dataset_created:
    try:
        # è®¾å¤‡è®¾ç½®
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"  âœ“ ä½¿ç”¨è®¾å¤‡: {device}")
        
        # åˆ›å»ºæ¨¡å�‹
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        num_classes = 2  # èƒŒæ™¯ + å°�éº¦
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # åŠ è½½æ�ƒé‡�
        print(f"  âœ“ åŠ è½½æ¨¡å�‹æ�ƒé‡�...")
        state_dict = torch.load(WEIGHTS_FILE, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"  âœ“ æ¨¡å�‹åŠ è½½æˆ�åŠŸ")
        
        model_loaded = True
        
    except Exception as e:
        print(f"  âœ— æ¨¡å�‹åŠ è½½å¤±è´¥: {e}")
        model_loaded = False
else:
    print(f"  âœ— æ— æ³•åŠ è½½æ¨¡å�‹")
    model_loaded = False

# ============================================
# 10. ä¸»å®�è·µä»»åŠ¡ï¼šå�¯è§†åŒ–æ‰€æœ‰æµ‹è¯•å›¾åƒ�é¢„æµ‹ç»“æ�œ
# ============================================
print("\n" + "="*60)
print("å¼€å§‹å®�è·µä»»åŠ¡ï¼šå�¯è§†åŒ–æ‰€æœ‰æµ‹è¯•å›¾åƒ�é¢„æµ‹ç»“æ�œ")
print("="*60)

if model_loaded and dataset_created:
    # åˆ›å»ºè¾“å‡ºç›®å½•
    output_dir = '/kaggle/working/test_predictions_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # æ£€æµ‹é˜ˆå€¼
    detection_threshold = 0.45
    print(f"æ£€æµ‹é˜ˆå€¼: {detection_threshold}")
    print(f"è¾“å‡ºç›®å½•: {output_dir}")
    
    # å­˜å‚¨ç»Ÿè®¡ä¿¡æ�¯
    all_predictions = []
    total_images = 0
    total_boxes = 0
    
    print(f"\nå¼€å§‹å¤„ç�†æµ‹è¯•é›†...")
    
    # é��å�†æµ‹è¯•æ•°æ�®åŠ è½½å™¨
    for batch_idx, (images, image_ids) in enumerate(test_data_loader):
        print(f"\rå¤„ç�†æ‰¹æ¬¡: {batch_idx+1}/{len(test_data_loader)}", end="", flush=True)
        
        # å°†å›¾åƒ�ç§»åˆ°è®¾å¤‡
        images = [img.to(device) for img in images]
        
        # è¿›è¡Œæ�¨ç�†
        with torch.no_grad():
            outputs = model(images)
        
        # å¤„ç�†æ¯�ä¸ªå›¾åƒ�çš„é¢„æµ‹ç»“æ�œ
        for i, (image, output, image_id) in enumerate(zip(images, outputs, image_ids)):
            total_images += 1
            
            # æ��å�–é¢„æµ‹ç»“æ�œ
            boxes = output['boxes'].data.cpu().numpy()
            scores = output['scores'].data.cpu().numpy()
            
            # æ ¹æ�®é˜ˆå€¼è¿‡æ»¤é¢„æµ‹æ¡†
            keep_indices = scores >= detection_threshold
            filtered_boxes = boxes[keep_indices]
            filtered_scores = scores[keep_indices]
            
            num_detections = len(filtered_boxes)
            total_boxes += num_detections
            
            # å­˜å‚¨é¢„æµ‹ä¿¡æ�¯
            all_predictions.append({
                'image_id': image_id,
                'num_detections': num_detections,
                'scores': filtered_scores.copy(),
                'boxes': filtered_boxes.copy()
            })
            
            # å‡†å¤‡å�¯è§†åŒ–
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # åˆ›å»ºå�¯è§†åŒ–å›¾åƒ�
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image_np)
            
            # ç»˜åˆ¶é¢„æµ‹æ¡†
            colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
            for j, box in enumerate(filtered_boxes):
                x1, y1, x2, y2 = map(int, box)
                score = filtered_scores[j]
                color = colors[j % len(colors)]
                
                # åˆ›å»ºçŸ©å½¢æ¡†
                rect = patches.Rectangle(
                    (x1, y1), 
                    x2 - x1, 
                    y2 - y1, 
                    linewidth=2, 
                    edgecolor=color, 
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # æ·»åŠ ç½®ä¿¡åº¦æ ‡ç­¾
                ax.text(
                    x1, y1 - 8, 
                    f'{score:.3f}', 
                    color=color, 
                    fontsize=8, 
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )
            
            # è®¾ç½®æ ‡é¢˜
            title = f'æµ‹è¯•å›¾åƒ�: {image_id}\næ£€æµ‹åˆ° {num_detections} ä¸ªå°�éº¦ç©—'
            if num_detections > 0:
                title += f'\nç½®ä¿¡åº¦èŒƒå›´: {filtered_scores.min():.3f} - {filtered_scores.max():.3f}'
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_axis_off()
            
            # ä¿�å­˜å�¯è§†åŒ–ç»“æ�œ
            output_path = os.path.join(output_dir, f'{image_id}_prediction.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
    
    print(f"\n\nâœ“ å¤„ç�†å®Œæˆ�ï¼�")
    print(f"  æ€»å¤„ç�†å›¾åƒ�æ•°: {total_images}")
    print(f"  æ€»æ£€æµ‹æ¡†æ•°: {total_boxes}")
    print(f"  å¹³å�‡æ¯�å¼ å›¾åƒ�æ£€æµ‹æ•°: {total_boxes/total_images if total_images > 0 else 0:.2f}")
    
    # ============================================
    # 11. ç”Ÿæˆ�ç»Ÿè®¡æŠ¥å‘Š
    # ============================================
    print("\n" + "="*60)
    print("ç”Ÿæˆ�ç»Ÿè®¡æŠ¥å‘Š")
    print("="*60)
    
    if all_predictions:
        # æ��å�–ç»Ÿè®¡ä¿¡æ�¯
        num_detections_list = [p['num_detections'] for p in all_predictions]
        all_scores = np.concatenate([p['scores'] for p in all_predictions if len(p['scores']) > 0])
        
        # åŸºç¡€ç»Ÿè®¡
        print("åŸºç¡€ç»Ÿè®¡:")
        print(f"  æ€»æµ‹è¯•å›¾ç‰‡æ•°: {len(all_predictions)}")
        print(f"  æ€»æ£€æµ‹æ¡†æ•°: {sum(num_detections_list)}")
        print(f"  æ£€æµ‹åˆ°ç›®æ ‡çš„å›¾ç‰‡æ•°: {sum(n > 0 for n in num_detections_list)}")
        print(f"  æœªæ£€æµ‹åˆ°ç›®æ ‡çš„å›¾ç‰‡æ•°: {sum(n == 0 for n in num_detections_list)}")
        print(f"  æ£€æµ‹ç�‡: {sum(n > 0 for n in num_detections_list)/len(num_detections_list)*100:.1f}%")
        print(f"  å¹³å�‡æ¯�å¼ å›¾ç‰‡æ£€æµ‹æ•°: {np.mean(num_detections_list):.2f}")
        print(f"  ä¸­ä½�æ•°æ£€æµ‹æ•°: {np.median(num_detections_list):.1f}")
        
        if len(all_scores) > 0:
            print(f"\nç½®ä¿¡åº¦ç»Ÿè®¡:")
            print(f"  å¹³å�‡ç½®ä¿¡åº¦: {all_scores.mean():.3f}")
            print(f"  ä¸­ä½�æ•°ç½®ä¿¡åº¦: {np.median(all_scores):.3f}")
            print(f"  æœ€é«˜ç½®ä¿¡åº¦: {all_scores.max():.3f}")
            print(f"  æœ€ä½�ç½®ä¿¡åº¦: {all_scores.min():.3f}")
        
        # æ£€æµ‹æ•°é‡�åˆ†å¸ƒ
        print(f"\næ£€æµ‹æ•°é‡�åˆ†å¸ƒ (é˜ˆå€¼={detection_threshold}):")
        distribution_ranges = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 1000)]
        
        for low, high in distribution_ranges:
            if low == high == 0:
                count = sum(1 for n in num_detections_list if n == 0)
                percentage = count/len(num_detections_list)*100
                print(f"  {low}ä¸ª: {count:3d} å¼  ({percentage:5.1f}%)")
            elif low == high:
                count = sum(1 for n in num_detections_list if n == low)
                if count > 0:
                    percentage = count/len(num_detections_list)*100
                    print(f"  {low}ä¸ª: {count:3d} å¼  ({percentage:5.1f}%)")
            else:
                count = sum(1 for n in num_detections_list if low <= n <= high)
                if count > 0:
                    percentage = count/len(num_detections_list)*100
                    print(f"  {low}-{high}ä¸ª: {count:3d} å¼  ({percentage:5.1f}%)")
        
        # ä¿�å­˜è¯¦ç»†ç»Ÿè®¡åˆ°CSV
        stats_data = []
        for pred in all_predictions:
            stats_data.append({
                'image_id': pred['image_id'],
                'num_detections': pred['num_detections'],
                'max_score': pred['scores'].max() if len(pred['scores']) > 0 else 0,
                'mean_score': pred['scores'].mean() if len(pred['scores']) > 0 else 0,
                'min_score': pred['scores'].min() if len(pred['scores']) > 0 else 0
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_csv_path = os.path.join(output_dir, 'detection_statistics.csv')
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"\nâœ“ è¯¦ç»†ç»Ÿè®¡å·²ä¿�å­˜: {stats_csv_path}")
        
        # ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
        submission_data = []
        for pred in all_predictions:
            image_id = pred['image_id']
            boxes = pred['boxes']
            scores = pred['scores']
            
            # æ ¼å¼�åŒ–ä¸ºKaggleæ��äº¤æ ¼å¼�
            pred_strings = []
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                pred_strings.append(f"{score:.4f} {x1:.1f} {y1:.1f} {w:.1f} {h:.1f}")
            
            prediction_string = " ".join(pred_strings)
            submission_data.append({
                'image_id': image_id,
                'PredictionString': prediction_string
            })
        
        submission_df = pd.DataFrame(submission_data)
        submission_path = os.path.join(output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        print(f"âœ“ æ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�: {submission_path}")
        
        # ============================================
        # 12. å�¯è§†åŒ–ç»Ÿè®¡å›¾è¡¨
        # ============================================
        print("\nç”Ÿæˆ�ç»Ÿè®¡å›¾è¡¨...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # å­�å›¾1ï¼šæ£€æµ‹æ•°é‡�åˆ†å¸ƒç›´æ–¹å›¾
        axes[0, 0].hist(num_detections_list, bins=min(30, max(num_detections_list)+1),
                       edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('æ¯�å¼ å›¾ç‰‡çš„æ£€æµ‹æ•°é‡�', fontsize=12)
        axes[0, 0].set_ylabel('å›¾ç‰‡æ•°é‡�', fontsize=12)
        axes[0, 0].set_title('æ£€æµ‹æ•°é‡�åˆ†å¸ƒ', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # å­�å›¾2ï¼šç½®ä¿¡åº¦åˆ†å¸ƒ
        if len(all_scores) > 0:
            axes[0, 1].hist(all_scores, bins=30, edgecolor='black',
                           alpha=0.7, color='lightcoral')
            axes[0, 1].set_xlabel('ç½®ä¿¡åº¦åˆ†æ•°', fontsize=12)
            axes[0, 1].set_ylabel('æ£€æµ‹æ¡†æ•°é‡�', fontsize=12)
            axes[0, 1].set_title('ç½®ä¿¡åº¦åˆ†æ•°åˆ†å¸ƒ', fontsize=14, fontweight='bold')
            axes[0, 1].axvline(x=detection_threshold, color='red',
                             linestyle='--', label=f'é˜ˆå€¼={detection_threshold}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # å­�å›¾3ï¼šæ£€æµ‹æ•°é‡�é¥¼å›¾
        labels_pie = ['0ä¸ª', '1-3ä¸ª', '4-10ä¸ª', '11-20ä¸ª', '21+ä¸ª']
        sizes_pie = [
            sum(1 for n in num_detections_list if n == 0),
            sum(1 for n in num_detections_list if 1 <= n <= 3),
            sum(1 for n in num_detections_list if 4 <= n <= 10),
            sum(1 for n in num_detections_list if 11 <= n <= 20),
            sum(1 for n in num_detections_list if n >= 21)
        ]
        
        non_zero_indices = [i for i, s in enumerate(sizes_pie) if s > 0]
        if non_zero_indices:
            non_zero_labels = [labels_pie[i] for i in non_zero_indices]
            non_zero_sizes = [sizes_pie[i] for i in non_zero_indices]
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(non_zero_sizes)))
            
            axes[1, 0].pie(non_zero_sizes, labels=non_zero_labels, colors=colors_pie,
                          autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('æ£€æµ‹æ•°é‡�æ¯”ä¾‹åˆ†å¸ƒ', fontsize=14, fontweight='bold')
        
        # å­�å›¾4ï¼šæ£€æµ‹æ•°é‡�ç®±çº¿å›¾
        box_data = [num_detections_list]
        axes[1, 1].boxplot(box_data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', color='blue'),
                          medianprops=dict(color='red', linewidth=2))
        axes[1, 1].set_ylabel('æ£€æµ‹æ•°é‡�', fontsize=12)
        axes[1, 1].set_title('æ£€æµ‹æ•°é‡�ç®±çº¿å›¾', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks([1])
        axes[1, 1].set_xticklabels(['æ‰€æœ‰æµ‹è¯•å›¾ç‰‡'])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # æ·»åŠ ç»Ÿè®¡ä¿¡æ�¯åˆ°ç®±çº¿å›¾
        stats_text = f"æ€»æ•°: {len(num_detections_list)}å¼ \nå�‡å€¼: {np.mean(num_detections_list):.2f}\nä¸­ä½�æ•°: {np.median(num_detections_list):.1f}"
        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'æµ‹è¯•é›†é¢„æµ‹ç»¼å�ˆåˆ†æ�� (é˜ˆå€¼={detection_threshold})',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # ä¿�å­˜ç»Ÿè®¡å›¾è¡¨
        stats_chart_path = os.path.join(output_dir, 'statistics_summary.png')
        plt.savefig(stats_chart_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"âœ“ ç»Ÿè®¡å›¾è¡¨å·²ä¿�å­˜: {stats_chart_path}")
        
        # ============================================
        # 13. æ˜¾ç¤ºç¤ºä¾‹ç»“æ�œ
        # ============================================
        print("\n" + "="*60)
        print("ç¤ºä¾‹ç»“æ�œå±•ç¤º")
        print("="*60)
        
        # è�·å�–ä¸€äº›æœ‰ä»£è¡¨æ€§çš„å›¾ç‰‡
        output_files = [f for f in os.listdir(output_dir) 
                       if f.endswith('_prediction.png')]
        
        if output_files:
            # é€‰æ‹©å‰�6å¼ å›¾ç‰‡æ˜¾ç¤º
            display_files = output_files[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, img_file in enumerate(display_files):
                if idx < len(axes):
                    img_path = os.path.join(output_dir, img_file)
                    img = plt.imread(img_path)
                    
                    axes[idx].imshow(img)
                    
                    # ä»�æ–‡ä»¶å��ä¸­æ��å�–å›¾åƒ�ID
                    img_id = img_file.replace('_prediction.png', '')
                    
                    # æŸ¥æ‰¾å¯¹åº”çš„ç»Ÿè®¡ä¿¡æ�¯
                    img_stats = stats_df[stats_df['image_id'] == img_id]
                    if not img_stats.empty:
                        title = f'{img_id}\næ£€æµ‹æ•°: {int(img_stats.iloc[0]["num_detections"])}'
                        if img_stats.iloc[0]['max_score'] > 0:
                            title += f'\næœ€é«˜åˆ†: {img_stats.iloc[0]["max_score"]:.3f}'
                    else:
                        title = img_id
                    
                    axes[idx].set_title(title, fontsize=10)
                    axes[idx].set_axis_off()
            
            # éš�è—�å¤šä½™çš„å­�å›¾
            for idx in range(len(display_files), 6):
                axes[idx].axis('off')
            
            plt.suptitle('æµ‹è¯•é›†é¢„æµ‹å�¯è§†åŒ–ç¤ºä¾‹', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print(f"æ˜¾ç¤º {len(display_files)} ä¸ªç¤ºä¾‹ç»“æ�œ")
        else:
            print("æœªæ‰¾åˆ°å�¯è§†åŒ–ç»“æ�œæ–‡ä»¶")
    
    else:
        print("æ²¡æœ‰ç”Ÿæˆ�é¢„æµ‹ç»“æ�œ")
    
    # ============================================
    # 14. ç”Ÿæˆ�æœ€ç»ˆæŠ¥å‘Š
    # ============================================
    print("\n" + "="*60)
    print("ä»»åŠ¡å®Œæˆ�æŠ¥å‘Š")
    print("="*60)
    
    # æ£€æŸ¥è¾“å‡ºç›®å½•å†…å®¹
    print("è¾“å‡ºç›®å½•å†…å®¹:")
    output_items = os.listdir(output_dir)
    png_files = [f for f in output_items if f.endswith('.png')]
    csv_files = [f for f in output_items if f.endswith('.csv')]
    
    print(f"  å�¯è§†åŒ–å›¾ç‰‡: {len(png_files)} å¼ ")
    print(f"  æ•°æ�®æ–‡ä»¶: {len(csv_files)} ä¸ª")
    
    if png_files:
        print(f"  ç¤ºä¾‹å›¾ç‰‡: {png_files[0]}")
    
    print("\næ–‡ä»¶è·¯å¾„:")
    print(f"  å�¯è§†åŒ–ç»“æ�œç›®å½•: {output_dir}")
    print(f"  è¯¦ç»†ç»Ÿè®¡æ–‡ä»¶: {os.path.join(output_dir, 'detection_statistics.csv')}")
    print(f"  æ��äº¤æ–‡ä»¶: {os.path.join(output_dir, 'submission.csv')}")
    print(f"  ç»Ÿè®¡å›¾è¡¨: {os.path.join(output_dir, 'statistics_summary.png')}")
    
    print("\n" + "="*60)
    print("å®�è·µä»»åŠ¡å®Œæˆ�!")
    print("="*60)
    print(f"âœ“ å·²å¤„ç�† {total_images} å¼ æµ‹è¯•å›¾ç‰‡")
    print(f"âœ“ ç”Ÿæˆ� {len(png_files)} å¼ å�¯è§†åŒ–å›¾ç‰‡")
    print(f"âœ“ ç”Ÿæˆ�è¯¦ç»†ç»Ÿè®¡æŠ¥å‘Š")
    print(f"âœ“ ç”Ÿæˆ�æ��äº¤æ–‡ä»¶")
    print(f"âœ“ æ‰€æœ‰æ–‡ä»¶ä¿�å­˜åœ¨: {output_dir}")
    
else:
    print("âœ— æ— æ³•æ‰§è¡Œå®�è·µä»»åŠ¡:")
    if not model_loaded:
        print("  - æ¨¡å�‹æœªæˆ�åŠŸåŠ è½½")
    if not dataset_created:
        print("  - æµ‹è¯•æ•°æ�®é›†æœªæˆ�åŠŸåˆ›å»º")

print("\n" + "="*60)
print("å…¨éƒ¨ä»»åŠ¡æ‰§è¡Œå®Œæ¯•!")
print("="*60)





# ============================================
# å®�è·µä»»åŠ¡ï¼šå�¯è§†åŒ–æµ‹è¯•é›†ä¸­æ‰€æœ‰å›¾ç‰‡çš„æ¨¡å�‹é¢„æµ‹ç»“æ�œ
# ============================================

print("="*70)
print("ğŸŸ¦ å®�è·µä»»åŠ¡ï¼šå�¯è§†åŒ–æµ‹è¯•é›†ä¸­æ‰€æœ‰å›¾ç‰‡çš„æ¨¡å�‹é¢„æµ‹ç»“æ�œ")
print("="*70)

# è®¾å¤‡è®¾ç½®
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"ä½¿ç”¨è®¾å¤‡: {device}")

# ç¡®ä¿�æ¨¡å�‹åœ¨è®¾å¤‡ä¸Šå¹¶è®¾ç½®ä¸ºè¯„ä¼°æ¨¡å¼�
model.to(device)
model.eval()
print("æ¨¡å�‹å·²è®¾ç½®ä¸ºè¯„ä¼°æ¨¡å¼�")

# åˆ›å»ºæµ‹è¯•æ•°æ�®é›†
test_dataset = WheatDataset(submit_df, test_dir, transforms.ToTensor(), False)
print(f"æµ‹è¯•æ•°æ�®é›†å¤§å°�: {len(test_dataset)} å¼ å›¾ç‰‡")

# åˆ›å»ºæµ‹è¯•æ•°æ�®åŠ è½½å™¨
test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,  # æ ¹æ�®GPUå†…å­˜è°ƒæ•´æ‰¹æ¬¡å¤§å°�
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)
print(f"æ•°æ�®åŠ è½½å™¨åˆ›å»ºæˆ�åŠŸï¼Œæ‰¹æ¬¡å¤§å°�: 4ï¼Œæ€»æ‰¹æ¬¡æ•°: {len(test_data_loader)}")

# åˆ›å»ºè¾“å‡ºç›®å½•
output_dir = '/kaggle/working/test_predictions_all'
os.makedirs(output_dir, exist_ok=True)
print(f"è¾“å‡ºç›®å½•: {output_dir}")

# è®¾ç½®æ£€æµ‹é˜ˆå€¼
detection_threshold = 0.45
print(f"æ£€æµ‹é˜ˆå€¼: {detection_threshold}")

# å­˜å‚¨æ‰€æœ‰é¢„æµ‹ç»“æ�œçš„ç»Ÿè®¡ä¿¡æ�¯
all_predictions_stats = []
total_images_processed = 0
total_boxes_detected = 0

print("\n" + "="*70)
print("å¼€å§‹æ‰¹é‡�æ�¨ç�†å’Œå�¯è§†åŒ–")
print("="*70)

# å¼€å§‹è®¡æ—¶
import time
start_time = time.time()

# é��å�†æµ‹è¯•é›†ä¸­çš„æ‰€æœ‰å›¾ç‰‡è¿›è¡Œæ�¨ç�†
for batch_idx, (images, image_ids) in enumerate(test_data_loader):
    batch_start_time = time.time()
    
    # æ˜¾ç¤ºå¤„ç�†è¿›åº¦
    progress = (batch_idx + 1) / len(test_data_loader) * 100
    print(f"\rå¤„ç�†è¿›åº¦: {batch_idx+1}/{len(test_data_loader)} æ‰¹æ¬¡ [{progress:.1f}%]", end="", flush=True)
    
    # å°†å›¾åƒ�ç§»åˆ°è®¾å¤‡
    images = [img.to(device) for img in images]
    
    # è¿›è¡Œæ�¨ç�†ï¼ˆæ— æ¢¯åº¦è®¡ç®—ï¼‰
    with torch.no_grad():
        outputs = model(images)
    
    # å¤„ç�†æ¯�ä¸ªå›¾åƒ�çš„é¢„æµ‹ç»“æ�œ
    for i, (image, output, image_id) in enumerate(zip(images, outputs, image_ids)):
        total_images_processed += 1
        
        # ä»»åŠ¡åŠŸèƒ½1: æ��å�–é¢„æµ‹ boxes ä¸� scores
        boxes = output['boxes'].data.cpu().numpy()
        scores = output['scores'].data.cpu().numpy()
        
        # ä»»åŠ¡åŠŸèƒ½2: æ ¹æ�®é˜ˆå€¼è¿‡æ»¤ä½�ç½®ä¿¡åº¦æ¡†
        keep_indices = scores >= detection_threshold
        filtered_boxes = boxes[keep_indices]
        filtered_scores = scores[keep_indices]
        
        num_detections = len(filtered_boxes)
        total_boxes_detected += num_detections
        
        # å­˜å‚¨ç»Ÿè®¡ä¿¡æ�¯
        all_predictions_stats.append({
            'image_id': image_id,
            'num_detections': num_detections,
            'max_score': filtered_scores.max() if num_detections > 0 else 0,
            'mean_score': filtered_scores.mean() if num_detections > 0 else 0,
            'min_score': filtered_scores.min() if num_detections > 0 else 0,
            'boxes': filtered_boxes,
            'scores': filtered_scores
        })
        
        # ä»»åŠ¡åŠŸèƒ½3: åœ¨å›¾åƒ�ä¸Šç»˜åˆ¶é¢„æµ‹æ¡†
        # å‡†å¤‡å›¾åƒ�ç”¨äº�å�¯è§†åŒ–
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # åˆ›å»ºå�¯è§†åŒ–å›¾åƒ�
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # æ˜¾ç¤ºå�Ÿå§‹å›¾åƒ�
        ax.imshow(image_np)
        
        # å®šä¹‰é¢œè‰²æ–¹æ¡ˆ
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        
        # ç»˜åˆ¶æ¯�ä¸ªé¢„æµ‹æ¡†
        for j, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
            x1, y1, x2, y2 = map(int, box)
            
            # é€‰æ‹©é¢œè‰²ï¼ˆå¾ªç�¯ä½¿ç”¨ï¼‰
            color = colors[j % len(colors)]
            
            # ç»˜åˆ¶çŸ©å½¢æ¡†
            rect = patches.Rectangle(
                (x1, y1), 
                x2 - x1, 
                y2 - y1, 
                linewidth=2, 
                edgecolor=color, 
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # æ·»åŠ ç½®ä¿¡åº¦æ ‡ç­¾
            label_text = f'{score:.3f}'
            ax.text(
                x1, 
                y1 - 10 if y1 > 20 else y1 + 15,  # é�¿å…�æ ‡ç­¾è¶…å‡ºå›¾åƒ�è¾¹ç•Œ
                label_text,
                color=color,
                fontsize=9,
                fontweight='bold',
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor=color,
                    boxstyle='round,pad=0.3'
                )
            )
        
        # è®¾ç½®å›¾åƒ�æ ‡é¢˜å’Œä¿¡æ�¯
        title = f'æµ‹è¯•å›¾åƒ�: {image_id}\n'
        title += f'æ£€æµ‹åˆ° {num_detections} ä¸ªå°�éº¦ç©—'
        
        if num_detections > 0:
            title += f' | ç½®ä¿¡åº¦: {filtered_scores.min():.3f}-{filtered_scores.max():.3f}'
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.set_axis_off()
        
        # åœ¨å›¾åƒ�å�³ä¸‹è§’æ·»åŠ é˜ˆå€¼ä¿¡æ�¯
        ax.text(
            0.98, 0.02, 
            f'é˜ˆå€¼: {detection_threshold}', 
            transform=ax.transAxes,
            fontsize=9,
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
            horizontalalignment='right',
            verticalalignment='bottom'
        )
        
        # ä¿�å­˜å�¯è§†åŒ–ç»“æ�œ
        output_path = os.path.join(output_dir, f'{image_id}_prediction.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    batch_time = time.time() - batch_start_time
    if batch_idx % 5 == 0:  # æ¯�5ä¸ªæ‰¹æ¬¡æ˜¾ç¤ºä¸€æ¬¡è¯¦ç»†ä¿¡æ�¯
        print(f"\næ‰¹æ¬¡ {batch_idx+1} å¤„ç�†å®Œæˆ�ï¼Œè€—æ—¶: {batch_time:.2f}ç§’")

# è®¡ç®—æ€»è€—æ—¶
total_time = time.time() - start_time

print(f"\n\n" + "="*70)
print("æ‰¹é‡�æ�¨ç�†å’Œå�¯è§†åŒ–å®Œæˆ�ï¼�")
print("="*70)

print(f"å¤„ç�†ç»Ÿè®¡:")
print(f"  æ€»å¤„ç�†å›¾åƒ�æ•°: {total_images_processed}")
print(f"  æ€»æ£€æµ‹æ¡†æ•°: {total_boxes_detected}")
print(f"  å¹³å�‡æ¯�å¼ å›¾åƒ�æ£€æµ‹æ•°: {total_boxes_detected/total_images_processed if total_images_processed > 0 else 0:.2f}")
print(f"  æ€»è€—æ—¶: {total_time:.2f}ç§’")
print(f"  å¹³å�‡æ¯�å¼ å›¾åƒ�è€—æ—¶: {total_time/total_images_processed if total_images_processed > 0 else 0:.2f}ç§’")

# ============================================
# ç”Ÿæˆ�è¯¦ç»†ç»Ÿè®¡æŠ¥å‘Š
# ============================================

print("\n" + "="*70)
print("ç”Ÿæˆ�è¯¦ç»†ç»Ÿè®¡æŠ¥å‘Š")
print("="*70)

if all_predictions_stats:
    # æ��å�–ç»Ÿè®¡æ•°æ�®
    detection_counts = [stats['num_detections'] for stats in all_predictions_stats]
    max_scores = [stats['max_score'] for stats in all_predictions_stats]
    
    # å�ˆå¹¶æ‰€æœ‰ç½®ä¿¡åº¦åˆ†æ•°
    all_scores = []
    for stats in all_predictions_stats:
        if stats['num_detections'] > 0:
            all_scores.extend(stats['scores'])
    all_scores = np.array(all_scores)
    
    print("æ£€æµ‹æ•°é‡�ç»Ÿè®¡:")
    print(f"  æ€»å›¾åƒ�æ•°: {len(all_predictions_stats)}")
    print(f"  æ£€æµ‹åˆ°ç›®æ ‡çš„å›¾åƒ�æ•°: {sum(1 for count in detection_counts if count > 0)}")
    print(f"  æœªæ£€æµ‹åˆ°ç›®æ ‡çš„å›¾åƒ�æ•°: {sum(1 for count in detection_counts if count == 0)}")
    print(f"  æ£€æµ‹ç�‡: {sum(1 for count in detection_counts if count > 0)/len(detection_counts)*100:.1f}%")
    print(f"  å¹³å�‡æ£€æµ‹æ•°: {np.mean(detection_counts):.2f}")
    print(f"  ä¸­ä½�æ•°æ£€æµ‹æ•°: {np.median(detection_counts):.1f}")
    print(f"  æœ€å¤§æ£€æµ‹æ•°: {max(detection_counts)}")
    print(f"  æœ€å°�æ£€æµ‹æ•°: {min(detection_counts)}")
    
    if len(all_scores) > 0:
        print("\nç½®ä¿¡åº¦ç»Ÿè®¡:")
        print(f"  å¹³å�‡ç½®ä¿¡åº¦: {all_scores.mean():.3f}")
        print(f"  ä¸­ä½�æ•°ç½®ä¿¡åº¦: {np.median(all_scores):.3f}")
        print(f"  æœ€é«˜ç½®ä¿¡åº¦: {all_scores.max():.3f}")
        print(f"  æœ€ä½�ç½®ä¿¡åº¦: {all_scores.min():.3f}")
        print(f"  æ ‡å‡†å·®: {all_scores.std():.3f}")
        
        # ç½®ä¿¡åº¦åˆ†å¸ƒ
        print("\nç½®ä¿¡åº¦åˆ†å¸ƒ:")
        score_ranges = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), 
                       (0.5, 0.6), (0.45, 0.5), (0.0, 0.45)]
        for low, high in score_ranges:
            count = np.sum((all_scores >= low) & (all_scores <= high))
            if count > 0:
                percentage = count / len(all_scores) * 100
                print(f"  {low:.1f}-{high:.1f}: {count:4d} ä¸ª ({percentage:5.1f}%)")
    
    print(f"\næ£€æµ‹æ•°é‡�åˆ†å¸ƒ (é˜ˆå€¼={detection_threshold}):")
    count_ranges = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 10), 
                   (11, 20), (21, 50), (51, 100), (101, 1000)]
    
    for low, high in count_ranges:
        if low == high == 0:
            count = sum(1 for n in detection_counts if n == 0)
            percentage = count / len(detection_counts) * 100
            print(f"  {low}ä¸ª: {count:3d} å¼  ({percentage:5.1f}%)")
        elif low == high:
            count = sum(1 for n in detection_counts if n == low)
            if count > 0:
                percentage = count / len(detection_counts) * 100
                print(f"  {low}ä¸ª: {count:3d} å¼  ({percentage:5.1f}%)")
        else:
            count = sum(1 for n in detection_counts if low <= n <= high)
            if count > 0:
                percentage = count / len(detection_counts) * 100
                print(f"  {low}-{high}ä¸ª: {count:3d} å¼  ({percentage:5.1f}%)")
    
    # ä¿�å­˜è¯¦ç»†ç»Ÿè®¡åˆ°CSV
    stats_data = []
    for stats in all_predictions_stats:
        stats_data.append({
            'image_id': stats['image_id'],
            'num_detections': stats['num_detections'],
            'max_score': stats['max_score'],
            'mean_score': stats['mean_score'],
            'min_score': stats['min_score'],
            'detection_status': 'æœ‰æ£€æµ‹' if stats['num_detections'] > 0 else 'æ— æ£€æµ‹'
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_csv_path = os.path.join(output_dir, 'detailed_statistics.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\nâœ“ è¯¦ç»†ç»Ÿè®¡å·²ä¿�å­˜: {stats_csv_path}")
    
    # ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
    submission_data = []
    for stats in all_predictions_stats:
        image_id = stats['image_id']
        boxes = stats['boxes']
        scores = stats['scores']
        
        # æ ¼å¼�åŒ–ä¸ºKaggleæ��äº¤æ ¼å¼�: "score x y w h"
        pred_strings = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            pred_strings.append(f"{score:.4f} {x1:.1f} {y1:.1f} {w:.1f} {h:.1f}")
        
        prediction_string = " ".join(pred_strings)
        submission_data.append({
            'image_id': image_id,
            'PredictionString': prediction_string
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"âœ“ æ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�: {submission_path}")
    
    # ============================================
    # ç”Ÿæˆ�å�¯è§†åŒ–ç»Ÿè®¡å›¾è¡¨
    # ============================================
    print("\n" + "="*70)
    print("ç”Ÿæˆ�å�¯è§†åŒ–ç»Ÿè®¡å›¾è¡¨")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # å›¾è¡¨1: æ£€æµ‹æ•°é‡�åˆ†å¸ƒç›´æ–¹å›¾
    axes[0, 0].hist(detection_counts, 
                   bins=min(20, max(detection_counts) + 1),
                   edgecolor='black', 
                   alpha=0.7, 
                   color='skyblue',
                   rwidth=0.8)
    axes[0, 0].set_xlabel('æ¯�å¼ å›¾ç‰‡çš„æ£€æµ‹æ•°é‡�', fontsize=12)
    axes[0, 0].set_ylabel('å›¾ç‰‡æ•°é‡�', fontsize=12)
    axes[0, 0].set_title('æ£€æµ‹æ•°é‡�åˆ†å¸ƒ', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].axvline(x=np.mean(detection_counts), color='red', 
                      linestyle='--', linewidth=2, label=f'å�‡å€¼={np.mean(detection_counts):.1f}')
    axes[0, 0].legend()
    
    # å›¾è¡¨2: ç½®ä¿¡åº¦åˆ†å¸ƒç›´æ–¹å›¾
    if len(all_scores) > 0:
        axes[0, 1].hist(all_scores, bins=30, 
                       edgecolor='black', alpha=0.7, color='lightcoral')
        axes[0, 1].set_xlabel('ç½®ä¿¡åº¦åˆ†æ•°', fontsize=12)
        axes[0, 1].set_ylabel('æ£€æµ‹æ¡†æ•°é‡�', fontsize=12)
        axes[0, 1].set_title('ç½®ä¿¡åº¦åˆ†æ•°åˆ†å¸ƒ', fontsize=14, fontweight='bold')
        axes[0, 1].axvline(x=detection_threshold, color='red', 
                          linestyle='--', linewidth=2, label=f'é˜ˆå€¼={detection_threshold}')
        axes[0, 1].axvline(x=np.mean(all_scores), color='blue', 
                          linestyle='--', linewidth=2, label=f'å�‡å€¼={np.mean(all_scores):.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    else:
        axes[0, 1].text(0.5, 0.5, 'æœªæ£€æµ‹åˆ°ä»»ä½•ç›®æ ‡', 
                       ha='center', va='center', fontsize=14)
        axes[0, 1].axis('off')
    
    # å›¾è¡¨3: æ£€æµ‹çŠ¶æ€�é¥¼å›¾
    detection_status = ['æœ‰æ£€æµ‹', 'æ— æ£€æµ‹']
    status_counts = [
        sum(1 for count in detection_counts if count > 0),
        sum(1 for count in detection_counts if count == 0)
    ]
    colors_pie = ['lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = axes[0, 2].pie(
        status_counts, 
        labels=detection_status, 
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0),
        shadow=True
    )
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    axes[0, 2].set_title('æ£€æµ‹çŠ¶æ€�åˆ†å¸ƒ', fontsize=14, fontweight='bold')
    
    # å›¾è¡¨4: æ£€æµ‹æ•°é‡�ç®±çº¿å›¾
    box_data = [detection_counts]
    bp = axes[1, 0].boxplot(
        box_data, 
        vert=True, 
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='blue', linewidth=2),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', color='red', alpha=0.5)
    )
    axes[1, 0].set_ylabel('æ£€æµ‹æ•°é‡�', fontsize=12)
    axes[1, 0].set_title('æ£€æµ‹æ•°é‡�ç®±çº¿å›¾', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks([1])
    axes[1, 0].set_xticklabels(['æ‰€æœ‰æµ‹è¯•å›¾ç‰‡'])
    axes[1, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # åœ¨ç®±çº¿å›¾ä¸Šæ·»åŠ ç»Ÿè®¡ä¿¡æ�¯
    stats_text = f"""ç»Ÿè®¡æ‘˜è¦�:
æ€»æ•°: {len(detection_counts)} å¼ 
å�‡å€¼: {np.mean(detection_counts):.2f}
ä¸­ä½�æ•°: {np.median(detection_counts):.1f}
æ ‡å‡†å·®: {np.std(detection_counts):.2f}
æœ€å°�å€¼: {min(detection_counts)}
æœ€å¤§å€¼: {max(detection_counts)}
Q1: {np.percentile(detection_counts, 25):.1f}
Q3: {np.percentile(detection_counts, 75):.1f}"""
    
    axes[1, 0].text(0.02, 0.98, stats_text, transform=axes[1, 0].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # å›¾è¡¨5: æ£€æµ‹æ•°é‡�ä¸�æœ€å¤§ç½®ä¿¡åº¦å…³ç³»
    if len(all_scores) > 0:
        scatter = axes[1, 1].scatter(
            detection_counts, 
            max_scores,
            c=detection_counts,
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            s=50
        )
        axes[1, 1].set_xlabel('æ£€æµ‹æ•°é‡�', fontsize=12)
        axes[1, 1].set_ylabel('æœ€å¤§ç½®ä¿¡åº¦', fontsize=12)
        axes[1, 1].set_title('æ£€æµ‹æ•°é‡� vs æœ€å¤§ç½®ä¿¡åº¦', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        # æ·»åŠ é¢œè‰²æ�¡
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('æ£€æµ‹æ•°é‡�', fontsize=10)
    else:
        axes[1, 1].text(0.5, 0.5, 'æ— ç½®ä¿¡åº¦æ•°æ�®', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
    
    # å›¾è¡¨6: æ£€æµ‹æ•°é‡�ç´¯ç§¯åˆ†å¸ƒ
    sorted_counts = np.sort(detection_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[1, 2].plot(sorted_counts, cumulative, 'b-', linewidth=2)
    axes[1, 2].fill_between(sorted_counts, cumulative, alpha=0.3, color='blue')
    axes[1, 2].set_xlabel('æ£€æµ‹æ•°é‡�', fontsize=12)
    axes[1, 2].set_ylabel('ç´¯ç§¯æ¯”ä¾‹', fontsize=12)
    axes[1, 2].set_title('æ£€æµ‹æ•°é‡�ç´¯ç§¯åˆ†å¸ƒ', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, linestyle='--')
    
    # æ·»åŠ å�‚è€ƒçº¿
    median_idx = np.where(sorted_counts >= np.median(detection_counts))[0][0]
    axes[1, 2].axvline(x=np.median(detection_counts), color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1, 2].axhline(y=cumulative[median_idx], color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.suptitle(f'æµ‹è¯•é›†é¢„æµ‹ç»“æ�œç»¼å�ˆåˆ†æ��æŠ¥å‘Š (é˜ˆå€¼={detection_threshold})', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # ä¿�å­˜ç»Ÿè®¡å›¾è¡¨
    stats_chart_path = os.path.join(output_dir, 'comprehensive_analysis.png')
    plt.savefig(stats_chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"âœ“ ç»¼å�ˆåˆ†æ��å›¾è¡¨å·²ä¿�å­˜: {stats_chart_path}")
    
    # ============================================
    # æ˜¾ç¤ºç¤ºä¾‹å�¯è§†åŒ–ç»“æ�œ
    # ============================================
    print("\n" + "="*70)
    print("ç¤ºä¾‹å�¯è§†åŒ–ç»“æ�œå±•ç¤º")
    print("="*70)
    
    # ä»�è¾“å‡ºç›®å½•è�·å�–ä¸€äº›ç¤ºä¾‹å›¾ç‰‡
    output_files = sorted([f for f in os.listdir(output_dir) 
                          if f.endswith('_prediction.png')])
    
    if output_files:
        # é€‰æ‹©ä¸�å�Œç±»å�‹çš„ç»“æ�œå±•ç¤º
        sample_files = []
        
        # 1. æ£€æµ‹æ•°é‡�æœ€å¤šçš„
        max_det_idx = np.argmax(detection_counts)
        max_det_image_id = all_predictions_stats[max_det_idx]['image_id']
        sample_files.append(f"{max_det_image_id}_prediction.png")
        
        # 2. æ£€æµ‹æ•°é‡�æœ€å°‘çš„é��é›¶
        non_zero_indices = [i for i, count in enumerate(detection_counts) if count > 0]
        if non_zero_indices:
            min_det_idx = non_zero_indices[np.argmin([detection_counts[i] for i in non_zero_indices])]
            min_det_image_id = all_predictions_stats[min_det_idx]['image_id']
            sample_files.append(f"{min_det_image_id}_prediction.png")
        
        # 3. æœ€é«˜ç½®ä¿¡åº¦çš„
        if len([s for s in max_scores if s > 0]) > 0:
            max_conf_idx = np.argmax(max_scores)
            max_conf_image_id = all_predictions_stats[max_conf_idx]['image_id']
            sample_files.append(f"{max_conf_image_id}_prediction.png")
        
        # 4-6. éš�æœºé€‰æ‹©
        import random
        random.seed(42)
        other_files = random.sample(output_files, min(3, len(output_files)-len(sample_files)))
        sample_files.extend(other_files[:3])
        
        # å�ªæ˜¾ç¤ºå­˜åœ¨çš„æ–‡ä»¶
        sample_files = [f for f in sample_files if f in output_files]
        
        if sample_files:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, img_file in enumerate(sample_files[:6]):
                img_path = os.path.join(output_dir, img_file)
                img = plt.imread(img_path)
                
                axes[idx].imshow(img)
                
                # è�·å�–å¯¹åº”çš„å›¾åƒ�ID
                img_id = img_file.replace('_prediction.png', '')
                
                # æŸ¥æ‰¾å¯¹åº”çš„ç»Ÿè®¡ä¿¡æ�¯
                img_stats = None
                for stats in all_predictions_stats:
                    if stats['image_id'] == img_id:
                        img_stats = stats
                        break
                
                # è®¾ç½®æ ‡é¢˜
                if img_stats:
                    title = f'{img_id}\n'
                    title += f'æ£€æµ‹æ•°: {img_stats["num_detections"]}'
                    if img_stats['num_detections'] > 0:
                        title += f'\nç½®ä¿¡åº¦: {img_stats["min_score"]:.3f}-{img_stats["max_score"]:.3f}'
                else:
                    title = img_id
                
                axes[idx].set_title(title, fontsize=11, fontweight='bold')
                axes[idx].set_axis_off()
            
            # éš�è—�å¤šä½™çš„å­�å›¾
            for idx in range(len(sample_files), 6):
                axes[idx].axis('off')
            
            plt.suptitle('æµ‹è¯•é›†é¢„æµ‹å�¯è§†åŒ–ç¤ºä¾‹ (ä¸�å�Œç±»å�‹çš„ç»“æ�œ)', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()
            
            print(f"æ˜¾ç¤º {len(sample_files)} ä¸ªç¤ºä¾‹ç»“æ�œ:")
            for i, img_file in enumerate(sample_files):
                img_id = img_file.replace('_prediction.png', '')
                for stats in all_predictions_stats:
                    if stats['image_id'] == img_id:
                        print(f"  {i+1}. {img_id}: {stats['num_detections']}ä¸ªæ£€æµ‹")
                        break
        else:
            print("æœªæ‰¾åˆ°ç¤ºä¾‹æ–‡ä»¶")
    else:
        print("æœªæ‰¾åˆ°å�¯è§†åŒ–ç»“æ�œæ–‡ä»¶")
    
    # ============================================
    # ç”Ÿæˆ�æœ€ç»ˆä»»åŠ¡æŠ¥å‘Š
    # ============================================
    print("\n" + "="*70)
    print("å®�è·µä»»åŠ¡å®Œæˆ�æŠ¥å‘Š")
    print("="*70)
    
    report_content = f"""
    å®�è·µä»»åŠ¡ï¼šå�¯è§†åŒ–æµ‹è¯•é›†ä¸­æ‰€æœ‰å›¾ç‰‡çš„æ¨¡å�‹é¢„æµ‹ç»“æ�œ
    {'='*60}
    
    ä»»åŠ¡å®Œæˆ�æƒ…å†µ:
    {'-'*60}
    âœ“ å¯¹æµ‹è¯•é›†ä¸­çš„æ‰€æœ‰å›¾åƒ�ä¾�æ¬¡è¿›è¡Œæ�¨ç�†ï¼ˆinferenceï¼‰
      - æ€»å¤„ç�†å›¾åƒ�æ•°: {total_images_processed}
      - å¹³å�‡æ¯�å¼ å›¾åƒ�è€—æ—¶: {total_time/total_images_processed if total_images_processed > 0 else 0:.2f}ç§’
    
    âœ“ æ��å�–æ¯�å¼ å›¾åƒ�çš„é¢„æµ‹ boxes ä¸� scores
      - æ€»æ£€æµ‹æ¡†æ•°: {total_boxes_detected}
      - å¹³å�‡æ¯�å¼ å›¾åƒ�æ£€æµ‹æ•°: {total_boxes_detected/total_images_processed if total_images_processed > 0 else 0:.2f}
    
    âœ“ æ ¹æ�®é˜ˆå€¼è¿‡æ»¤ä½�ç½®ä¿¡åº¦æ¡†
      - æ£€æµ‹é˜ˆå€¼: {detection_threshold}
      - è¿‡æ»¤å��æœ‰æ•ˆæ£€æµ‹æ¡†: {len(all_scores) if 'all_scores' in locals() else 0}
    
    âœ“ åœ¨å›¾åƒ�ä¸Šç»˜åˆ¶é¢„æµ‹æ¡†å¹¶æ˜¾ç¤ºç»“æ�œ
      - ç”Ÿæˆ�å�¯è§†åŒ–å›¾ç‰‡: {len(output_files) if 'output_files' in locals() else 0}å¼ 
      - æ£€æµ‹ç�‡: {sum(1 for count in detection_counts if count > 0)/len(detection_counts)*100 if detection_counts else 0:.1f}%
    
    è¾“å‡ºæ–‡ä»¶:
    {'-'*60}
    1. å�¯è§†åŒ–å›¾ç‰‡ç›®å½•: {output_dir}
       - æ¯�å¼ æµ‹è¯•å›¾åƒ�çš„å�¯è§†åŒ–ç»“æ�œ
       - æ–‡ä»¶å��æ ¼å¼�: [image_id]_prediction.png
    
    2. ç»Ÿè®¡æ–‡ä»¶:
       - è¯¦ç»†ç»Ÿè®¡: {os.path.join(output_dir, 'detailed_statistics.csv')}
       - æ��äº¤æ–‡ä»¶: {os.path.join(output_dir, 'submission.csv')}
    
    3. åˆ†æ��å›¾è¡¨:
       - ç»¼å�ˆåˆ†æ��: {os.path.join(output_dir, 'comprehensive_analysis.png')}
    
    æ¨¡å�‹æ€§èƒ½åˆ†æ��:
    {'-'*60}
    - å¹³å�‡æ£€æµ‹æ•°é‡�: {np.mean(detection_counts) if detection_counts else 0:.2f}
    - æ£€æµ‹æ•°é‡�èŒƒå›´: {min(detection_counts) if detection_counts else 0} - {max(detection_counts) if detection_counts else 0}
    - ç½®ä¿¡åº¦èŒƒå›´: {all_scores.min() if len(all_scores) > 0 else 0:.3f} - {all_scores.max() if len(all_scores) > 0 else 0:.3f}
    - æ£€æµ‹æˆ�åŠŸç�‡: {sum(1 for count in detection_counts if count > 0)/len(detection_counts)*100 if detection_counts else 0:.1f}%
    
    å»ºè®®ä¸�æ”¹è¿›:
    {'-'*60}
    1. å¦‚æ�œæ£€æµ‹ç�‡è¿‡ä½�ï¼Œå�¯è€ƒè™‘é™�ä½�é˜ˆå€¼ (å¦‚ 0.3-0.35)
    2. å¦‚æ�œå�‡é˜³æ€§è¿‡å¤šï¼Œå�¯æ��é«˜é˜ˆå€¼ (å¦‚ 0.5-0.6)
    3. æ£€æŸ¥æœªæ£€æµ‹åˆ°ç›®æ ‡çš„å›¾åƒ�ï¼Œåˆ†æ��å�Ÿå› 
    4. å�¯æ ¹æ�®ç½®ä¿¡åº¦åˆ†å¸ƒè°ƒæ•´å��å¤„ç�†ç­–ç•¥
    5. è€ƒè™‘å¯¹ä¸�å�Œç½®ä¿¡åº¦åŒºé—´é‡‡ç”¨ä¸�å�Œé¢œè‰²æ ‡æ³¨
    
    ä»»åŠ¡æ€»ç»“:
    {'-'*60}
    æœ¬æ¬¡å®�è·µä»»åŠ¡æˆ�åŠŸå®�ç�°äº†å¯¹æµ‹è¯•é›†ä¸­æ‰€æœ‰å›¾åƒ�çš„æ‰¹é‡�æ�¨ç�†å’Œå�¯è§†åŒ–ï¼Œ
    é€šè¿‡ç»Ÿè®¡åˆ†æ��äº†è§£äº†æ¨¡å�‹åœ¨æµ‹è¯•é›†ä¸Šçš„è¡¨ç�°ï¼Œä¸ºæ¨¡å�‹ä¼˜åŒ–æ��ä¾›äº†æ•°æ�®æ”¯æŒ�ã€‚
    """
    
    # ä¿�å­˜æŠ¥å‘Š
    report_path = os.path.join(output_dir, 'task_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(report_content)
    print(f"\nâœ“ è¯¦ç»†æŠ¥å‘Šå·²ä¿�å­˜: {report_path}")
    
else:
    print("æ²¡æœ‰ç”Ÿæˆ�é¢„æµ‹ç»“æ�œï¼Œæ— æ³•ç”Ÿæˆ�ç»Ÿè®¡æŠ¥å‘Š")

print("\n" + "="*70)
print("âœ… å®�è·µä»»åŠ¡å®Œæˆ�ï¼�")
print("="*70)
print("é€šè¿‡æœ¬å®�è·µä»»åŠ¡ï¼Œä½ å·²ç»�æˆ�åŠŸï¼š")
print("1. æ�Œæ�¡äº†å¦‚ä½•å¯¹æµ‹è¯•é›†è¿›è¡Œæ‰¹é‡�æ�¨ç�†")
print("2. å­¦ä¼šäº†å¦‚ä½•è¿‡æ»¤å’Œå�¯è§†åŒ–é¢„æµ‹ç»“æ�œ")
print("3. èƒ½å¤Ÿç”Ÿæˆ�è¯¦ç»†çš„ç»Ÿè®¡æŠ¥å‘Šå’Œåˆ†æ��å›¾è¡¨")
print("4. äº†è§£äº†æ¨¡å�‹åœ¨æµ‹è¯•é›†ä¸Šçš„æ•´ä½“è¡¨ç�°")
print("="*70)


# ============================================
# å®�è·µä»»åŠ¡ï¼šå�¯è§†åŒ–æµ‹è¯•é›†ä¸­æ‰€æœ‰å›¾ç‰‡çš„æ¨¡å�‹é¢„æµ‹ç»“æ�œ
# ============================================

print("="*70)
print("ğŸŸ¦ å®�è·µä»»åŠ¡ï¼šå�¯è§†åŒ–æµ‹è¯•é›†ä¸­æ‰€æœ‰å›¾ç‰‡çš„æ¨¡å�‹é¢„æµ‹ç»“æ�œ")
print("="*70)

# è®¾å¤‡è®¾ç½®
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"ä½¿ç”¨è®¾å¤‡: {device}")

# ç¡®ä¿�æ¨¡å�‹åœ¨è®¾å¤‡ä¸Šå¹¶è®¾ç½®ä¸ºè¯„ä¼°æ¨¡å¼�
model.to(device)
model.eval()
print("æ¨¡å�‹å·²è®¾ç½®ä¸ºè¯„ä¼°æ¨¡å¼�")

# åˆ›å»ºæµ‹è¯•æ•°æ�®é›†
test_dataset = WheatDataset(submit_df, test_dir, transforms.ToTensor(), False)
print(f"æµ‹è¯•æ•°æ�®é›†å¤§å°�: {len(test_dataset)} å¼ å›¾ç‰‡")

# åˆ›å»ºæµ‹è¯•æ•°æ�®åŠ è½½å™¨
test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,  # æ ¹æ�®GPUå†…å­˜è°ƒæ•´æ‰¹æ¬¡å¤§å°�
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)
print(f"æ•°æ�®åŠ è½½å™¨åˆ›å»ºæˆ�åŠŸï¼Œæ‰¹æ¬¡å¤§å°�: 4ï¼Œæ€»æ‰¹æ¬¡æ•°: {len(test_data_loader)}")

# åˆ›å»ºè¾“å‡ºç›®å½•
output_dir = '/kaggle/working/test_predictions_all'
os.makedirs(output_dir, exist_ok=True)
print(f"è¾“å‡ºç›®å½•: {output_dir}")

# è®¾ç½®æ£€æµ‹é˜ˆå€¼
detection_threshold = 0.45
print(f"æ£€æµ‹é˜ˆå€¼: {detection_threshold}")

# å­˜å‚¨æ‰€æœ‰é¢„æµ‹ç»“æ�œçš„ç»Ÿè®¡ä¿¡æ�¯
all_predictions_stats = []
total_images_processed = 0
total_boxes_detected = 0

print("\n" + "="*70)
print("å¼€å§‹æ‰¹é‡�æ�¨ç�†å’Œå�¯è§†åŒ–")
print("="*70)

# å¼€å§‹è®¡æ—¶
import time
start_time = time.time()

# é¦–å…ˆï¼Œè®©æˆ‘ä»¬çœ‹çœ‹æµ‹è¯•å›¾åƒ�
print("\nğŸ“¸ æµ‹è¯•å›¾åƒ�ç¤ºä¾‹é¢„è§ˆ:")
# æ˜¾ç¤ºå‰�4å¼ æµ‹è¯•å›¾åƒ�çš„å�Ÿå§‹å›¾åƒ�
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i in range(min(4, len(test_dataset))):
    image, image_id = test_dataset[i]
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    row = i // 2
    col = i % 2
    axes[row, col].imshow(image_np)
    axes[row, col].set_title(f'æµ‹è¯•å›¾åƒ�: {image_id}', fontsize=10)
    axes[row, col].axis('off')
plt.suptitle('æµ‹è¯•é›†å�Ÿå§‹å›¾åƒ�ç¤ºä¾‹', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# é��å�†æµ‹è¯•é›†ä¸­çš„æ‰€æœ‰å›¾ç‰‡è¿›è¡Œæ�¨ç�†å’Œå�¯è§†åŒ–
for batch_idx, (images, image_ids) in enumerate(test_data_loader):
    batch_start_time = time.time()
    
    # æ˜¾ç¤ºå¤„ç�†è¿›åº¦
    progress = (batch_idx + 1) / len(test_data_loader) * 100
    print(f"\rå¤„ç�†è¿›åº¦: {batch_idx+1}/{len(test_data_loader)} æ‰¹æ¬¡ [{progress:.1f}%]", end="", flush=True)
    
    # å°†å›¾åƒ�ç§»åˆ°è®¾å¤‡
    images = [img.to(device) for img in images]
    
    # è¿›è¡Œæ�¨ç�†ï¼ˆæ— æ¢¯åº¦è®¡ç®—ï¼‰
    with torch.no_grad():
        outputs = model(images)
    
    # å¤„ç�†æ¯�ä¸ªå›¾åƒ�çš„é¢„æµ‹ç»“æ�œ
    for i, (image, output, image_id) in enumerate(zip(images, outputs, image_ids)):
        total_images_processed += 1
        
        # ä»»åŠ¡åŠŸèƒ½1: æ��å�–é¢„æµ‹ boxes ä¸� scores
        boxes = output['boxes'].data.cpu().numpy()
        scores = output['scores'].data.cpu().numpy()
        
        # ä»»åŠ¡åŠŸèƒ½2: æ ¹æ�®é˜ˆå€¼è¿‡æ»¤ä½�ç½®ä¿¡åº¦æ¡†
        keep_indices = scores >= detection_threshold
        filtered_boxes = boxes[keep_indices]
        filtered_scores = scores[keep_indices]
        
        num_detections = len(filtered_boxes)
        total_boxes_detected += num_detections
        
        # å­˜å‚¨ç»Ÿè®¡ä¿¡æ�¯
        all_predictions_stats.append({
            'image_id': image_id,
            'num_detections': num_detections,
            'max_score': filtered_scores.max() if num_detections > 0 else 0,
            'mean_score': filtered_scores.mean() if num_detections > 0 else 0,
            'min_score': filtered_scores.min() if num_detections > 0 else 0,
            'boxes': filtered_boxes,
            'scores': filtered_scores
        })
        
        # ä»»åŠ¡åŠŸèƒ½3: åœ¨å›¾åƒ�ä¸Šç»˜åˆ¶é¢„æµ‹æ¡†å¹¶æ˜¾ç¤ºç»“æ�œ
        # å‡†å¤‡å›¾åƒ�ç”¨äº�å�¯è§†åŒ–
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # åˆ›å»ºå�¯è§†åŒ–å›¾åƒ�
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # æ˜¾ç¤ºå�Ÿå§‹å›¾åƒ�
        ax.imshow(image_np)
        
        # å®šä¹‰é¢œè‰²æ–¹æ¡ˆ - æ ¹æ�®ç½®ä¿¡åº¦ä½¿ç”¨ä¸�å�Œé¢œè‰²
        def get_color_by_score(score):
            if score >= 0.8:
                return 'red'  # é«˜ç½®ä¿¡åº¦
            elif score >= 0.6:
                return 'orange'  # ä¸­é«˜ç½®ä¿¡åº¦
            elif score >= 0.45:
                return 'yellow'  # é˜ˆå€¼é™„è¿‘
            else:
                return 'green'  # ä½�ç½®ä¿¡åº¦
        
        # ç»˜åˆ¶æ¯�ä¸ªé¢„æµ‹æ¡†
        for j, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
            x1, y1, x2, y2 = map(int, box)
            
            # æ ¹æ�®ç½®ä¿¡åº¦é€‰æ‹©é¢œè‰²
            color = get_color_by_score(score)
            
            # ç»˜åˆ¶çŸ©å½¢æ¡†
            rect = patches.Rectangle(
                (x1, y1), 
                x2 - x1, 
                y2 - y1, 
                linewidth=2, 
                edgecolor=color, 
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # æ·»åŠ ç½®ä¿¡åº¦æ ‡ç­¾
            label_text = f'{score:.2f}'
            ax.text(
                x1, 
                y1 - 8 if y1 > 20 else y1 + 15,  # é�¿å…�æ ‡ç­¾è¶…å‡ºå›¾åƒ�è¾¹ç•Œ
                label_text,
                color=color,
                fontsize=8,
                fontweight='bold',
                bbox=dict(
                    facecolor='white',
                    alpha=0.7,
                    edgecolor=color,
                    boxstyle='round,pad=0.3'
                )
            )
        
        # è®¾ç½®å›¾åƒ�æ ‡é¢˜å’Œä¿¡æ�¯
        title = f'æµ‹è¯•å›¾åƒ�: {image_id}\n'
        title += f'æ£€æµ‹åˆ° {num_detections} ä¸ªå°�éº¦ç©—'
        
        if num_detections > 0:
            title += f'\nç½®ä¿¡åº¦èŒƒå›´: {filtered_scores.min():.3f} - {filtered_scores.max():.3f}'
        
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
        ax.set_axis_off()
        
        # åœ¨å›¾åƒ�å�³ä¸‹è§’æ·»åŠ é˜ˆå€¼å’Œé¢œè‰²è¯´æ˜�
        info_text = f'é˜ˆå€¼: {detection_threshold}\n'
        info_text += 'é¢œè‰²: ğŸ”´â‰¥0.8 ğŸŸ â‰¥0.6 ğŸŸ¡â‰¥0.45'
        
        ax.text(
            0.98, 0.02, 
            info_text, 
            transform=ax.transAxes,
            fontsize=8,
            color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=5),
            horizontalalignment='right',
            verticalalignment='bottom'
        )
        
        # ä¿�å­˜å�¯è§†åŒ–ç»“æ�œ
        output_path = os.path.join(output_dir, f'{image_id}_prediction.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # å¦‚æ�œæ˜¯å‰�4å¼ å›¾åƒ�ï¼Œæ˜¾ç¤ºå®ƒä»¬
        if total_images_processed <= 4:
            print(f"\nğŸ“Š æ˜¾ç¤ºç¬¬ {total_images_processed} å¼ å›¾åƒ�çš„é¢„æµ‹ç»“æ�œ:")
            
            # é‡�æ–°æ‰“å¼€ä¿�å­˜çš„å›¾åƒ�æ˜¾ç¤º
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
            saved_img = plt.imread(output_path)
            ax2.imshow(saved_img)
            ax2.set_title(f'é¢„æµ‹ç»“æ�œé¢„è§ˆ: {image_id}', fontsize=12, fontweight='bold')
            ax2.axis('off')
            plt.tight_layout()
            plt.show()
            
            # æ˜¾ç¤ºç»Ÿè®¡ä¿¡æ�¯
            print(f"  å›¾åƒ�ID: {image_id}")
            print(f"  æ£€æµ‹æ•°é‡�: {num_detections}")
            if num_detections > 0:
                print(f"  ç½®ä¿¡åº¦èŒƒå›´: {filtered_scores.min():.3f} - {filtered_scores.max():.3f}")
                print(f"  å¹³å�‡ç½®ä¿¡åº¦: {filtered_scores.mean():.3f}")
    
    batch_time = time.time() - batch_start_time

# è®¡ç®—æ€»è€—æ—¶
total_time = time.time() - start_time

print(f"\n\n" + "="*70)
print("æ‰¹é‡�æ�¨ç�†å’Œå�¯è§†åŒ–å®Œæˆ�ï¼�")
print("="*70)

print(f"ğŸ“ˆ å¤„ç�†ç»Ÿè®¡:")
print(f"  æ€»å¤„ç�†å›¾åƒ�æ•°: {total_images_processed}")
print(f"  æ€»æ£€æµ‹æ¡†æ•°: {total_boxes_detected}")
print(f"  å¹³å�‡æ¯�å¼ å›¾åƒ�æ£€æµ‹æ•°: {total_boxes_detected/total_images_processed if total_images_processed > 0 else 0:.2f}")
print(f"  æ€»è€—æ—¶: {total_time:.2f}ç§’")
print(f"  å¹³å�‡æ¯�å¼ å›¾åƒ�è€—æ—¶: {total_time/total_images_processed if total_images_processed > 0 else 0:.2f}ç§’")

# ============================================
# å®�æ—¶æ˜¾ç¤ºæ›´å¤šå�¯è§†åŒ–ç»“æ�œ
# ============================================

print("\n" + "="*70)
print("ğŸ“¸ æ›´å¤šå�¯è§†åŒ–ç»“æ�œå±•ç¤º")
print("="*70)

# ä»�è¾“å‡ºç›®å½•è�·å�–ä¸€äº›ä¸�å�Œç±»å�‹çš„å›¾ç‰‡å±•ç¤º
output_files = sorted([f for f in os.listdir(output_dir) 
                      if f.endswith('_prediction.png')])

if output_files:
    # é€‰æ‹©6å¼ ä¸�å�Œç±»å�‹çš„å›¾åƒ�å±•ç¤º
    print("é€‰æ‹©6å¼ ä¸�å�Œç±»å�‹çš„é¢„æµ‹ç»“æ�œè¿›è¡Œå±•ç¤º:")
    
    # æ ¹æ�®æ£€æµ‹æ•°é‡�æ�’åº�
    sorted_stats = sorted(all_predictions_stats, key=lambda x: x['num_detections'])
    
    # é€‰æ‹©å±•ç¤ºçš„å›¾ç‰‡
    display_indices = []
    
    # 1. æ— æ£€æµ‹çš„å›¾ç‰‡ï¼ˆå¦‚æ�œæœ‰ï¼‰
    no_detection = [i for i, stats in enumerate(all_predictions_stats) if stats['num_detections'] == 0]
    if no_detection:
        display_indices.append(no_detection[0])
    
    # 2. å°‘é‡�æ£€æµ‹çš„å›¾ç‰‡
    few_detection = [i for i, stats in enumerate(all_predictions_stats) if 1 <= stats['num_detections'] <= 3]
    if few_detection:
        display_indices.append(few_detection[len(few_detection)//2])
    
    # 3. ä¸­ç­‰æ•°é‡�æ£€æµ‹çš„å›¾ç‰‡
    medium_detection = [i for i, stats in enumerate(all_predictions_stats) if 4 <= stats['num_detections'] <= 10]
    if medium_detection:
        display_indices.append(medium_detection[len(medium_detection)//2])
    
    # 4. å¤§é‡�æ£€æµ‹çš„å›¾ç‰‡
    many_detection = [i for i, stats in enumerate(all_predictions_stats) if stats['num_detections'] > 10]
    if many_detection:
        display_indices.append(many_detection[len(many_detection)//2])
    
    # 5. æœ€é«˜ç½®ä¿¡åº¦çš„å›¾ç‰‡
    if any(stats['max_score'] > 0 for stats in all_predictions_stats):
        max_conf_idx = max(range(len(all_predictions_stats)), 
                          key=lambda i: all_predictions_stats[i]['max_score'] if all_predictions_stats[i]['max_score'] > 0 else -1)
        display_indices.append(max_conf_idx)
    
    # 6. éš�æœºé€‰æ‹©ä¸€å¼ 
    import random
    random.seed(42)
    remaining_indices = [i for i in range(len(all_predictions_stats)) if i not in display_indices]
    if remaining_indices:
        display_indices.append(random.choice(remaining_indices))
    
    # ç¡®ä¿�ä¸�é‡�å¤�ä¸”ä¸�è¶…è¿‡6ä¸ª
    display_indices = list(set(display_indices))[:6]
    
    # æ˜¾ç¤ºé€‰ä¸­çš„å›¾ç‰‡
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, stat_idx in enumerate(display_indices):
        stats = all_predictions_stats[stat_idx]
        img_file = f"{stats['image_id']}_prediction.png"
        img_path = os.path.join(output_dir, img_file)
        
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axes[idx].imshow(img)
            
            # è®¾ç½®æ ‡é¢˜
            title = f"{stats['image_id']}\n"
            title += f"æ£€æµ‹æ•°: {stats['num_detections']}"
            if stats['num_detections'] > 0:
                title += f"\nç½®ä¿¡åº¦: {stats['min_score']:.2f}-{stats['max_score']:.2f}"
            
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
            axes[idx].axis('off')
            
            # æ‰“å�°ä¿¡æ�¯
            print(f"  {idx+1}. {stats['image_id']}: {stats['num_detections']}ä¸ªæ£€æµ‹", end="")
            if stats['num_detections'] > 0:
                print(f" (ç½®ä¿¡åº¦: {stats['min_score']:.2f}-{stats['max_score']:.2f})")
            else:
                print()
        else:
            axes[idx].text(0.5, 0.5, 'å›¾åƒ�æœªæ‰¾åˆ°', 
                          ha='center', va='center', fontsize=12)
            axes[idx].axis('off')
    
    # éš�è—�å¤šä½™çš„å­�å›¾
    for idx in range(len(display_indices), 6):
        axes[idx].axis('off')
    
    plt.suptitle('æµ‹è¯•é›†é¢„æµ‹ç»“æ�œå¤šæ ·æœ¬å±•ç¤º', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# ============================================
# ç”Ÿæˆ�ç»Ÿè®¡å›¾è¡¨å¹¶æ˜¾ç¤º
# ============================================

print("\n" + "="*70)
print("ğŸ“Š ç”Ÿæˆ�ç»Ÿè®¡å›¾è¡¨")
print("="*70)

if all_predictions_stats:
    # æ��å�–ç»Ÿè®¡æ•°æ�®
    detection_counts = [stats['num_detections'] for stats in all_predictions_stats]
    all_scores = []
    for stats in all_predictions_stats:
        if stats['num_detections'] > 0:
            all_scores.extend(stats['scores'])
    
    # åˆ›å»ºç»Ÿè®¡å›¾è¡¨
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # å›¾è¡¨1: æ£€æµ‹æ•°é‡�åˆ†å¸ƒ
    axes[0, 0].hist(detection_counts, 
                   bins=min(20, max(detection_counts) + 1),
                   edgecolor='black', 
                   alpha=0.7, 
                   color='skyblue',
                   rwidth=0.8)
    axes[0, 0].set_xlabel('æ¯�å¼ å›¾ç‰‡çš„æ£€æµ‹æ•°é‡�', fontsize=12)
    axes[0, 0].set_ylabel('å›¾ç‰‡æ•°é‡�', fontsize=12)
    axes[0, 0].set_title('æ£€æµ‹æ•°é‡�åˆ†å¸ƒ', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].axvline(x=np.mean(detection_counts), color='red', 
                      linestyle='--', linewidth=2, label=f'å�‡å€¼={np.mean(detection_counts):.1f}')
    axes[0, 0].legend()
    
    # å›¾è¡¨2: æ£€æµ‹çŠ¶æ€�é¥¼å›¾
    detection_status = ['æœ‰æ£€æµ‹', 'æ— æ£€æµ‹']
    status_counts = [
        sum(1 for count in detection_counts if count > 0),
        sum(1 for count in detection_counts if count == 0)
    ]
    colors_pie = ['lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = axes[0, 1].pie(
        status_counts, 
        labels=detection_status, 
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0)
    )
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    axes[0, 1].set_title('æ£€æµ‹çŠ¶æ€�åˆ†å¸ƒ', fontsize=14, fontweight='bold')
    
    # å›¾è¡¨3: ç½®ä¿¡åº¦åˆ†å¸ƒï¼ˆå¦‚æ�œæœ‰æ£€æµ‹ç»“æ�œï¼‰
    if all_scores:
        axes[1, 0].hist(all_scores, bins=30, 
                       edgecolor='black', alpha=0.7, color='lightcoral')
        axes[1, 0].set_xlabel('ç½®ä¿¡åº¦åˆ†æ•°', fontsize=12)
        axes[1, 0].set_ylabel('æ£€æµ‹æ¡†æ•°é‡�', fontsize=12)
        axes[1, 0].set_title('ç½®ä¿¡åº¦åˆ†æ•°åˆ†å¸ƒ', fontsize=14, fontweight='bold')
        axes[1, 0].axvline(x=detection_threshold, color='red', 
                          linestyle='--', linewidth=2, label=f'é˜ˆå€¼={detection_threshold}')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'æœªæ£€æµ‹åˆ°ä»»ä½•ç›®æ ‡', 
                       ha='center', va='center', fontsize=14)
        axes[1, 0].axis('off')
    
    # å›¾è¡¨4: æ£€æµ‹æ•°é‡�ç®±çº¿å›¾
    box_data = [detection_counts]
    axes[1, 1].boxplot(
        box_data, 
        vert=True, 
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='blue', linewidth=2),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5)
    )
    axes[1, 1].set_ylabel('æ£€æµ‹æ•°é‡�', fontsize=12)
    axes[1, 1].set_title('æ£€æµ‹æ•°é‡�ç®±çº¿å›¾', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks([1])
    axes[1, 1].set_xticklabels(['æ‰€æœ‰æµ‹è¯•å›¾ç‰‡'])
    axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # æ·»åŠ ç»Ÿè®¡ä¿¡æ�¯
    stats_text = f"""ç»Ÿè®¡æ‘˜è¦�:
æ€»æ•°: {len(detection_counts)}å¼ 
å�‡å€¼: {np.mean(detection_counts):.2f}
ä¸­ä½�æ•°: {np.median(detection_counts):.1f}
æœ€å°�å€¼: {min(detection_counts)}
æœ€å¤§å€¼: {max(detection_counts)}"""
    
    axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'æµ‹è¯•é›†é¢„æµ‹ç»“æ�œç»Ÿè®¡åˆ†æ�� (é˜ˆå€¼={detection_threshold})', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # ä¿�å­˜ç»Ÿè®¡å›¾è¡¨
    stats_chart_path = os.path.join(output_dir, 'statistics_summary.png')
    plt.savefig(stats_chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"ğŸ“ˆ ç»Ÿè®¡å›¾è¡¨å·²ç”Ÿæˆ�å¹¶æ˜¾ç¤º")

# ============================================
# æ˜¾ç¤ºæœ€å��å¤„ç�†çš„ä¸€æ‰¹å›¾åƒ�
# ============================================

print("\n" + "="*70)
print("ğŸ�¯ æœ€å��ä¸€æ‰¹å¤„ç�†ç»“æ�œå±•ç¤º")
print("="*70)

# è�·å�–æœ€å��ä¸€æ‰¹å¤„ç�†çš„å›¾åƒ�
last_batch_images = []
last_batch_ids = []

# é‡�æ–°åŠ è½½æœ€å��ä¸€æ‰¹æ•°æ�®ç”¨äº�å±•ç¤º
test_data_loader_last = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

# è�·å�–æœ€å��ä¸€æ‰¹
for images, image_ids in test_data_loader_last:
    last_batch_images = images
    last_batch_ids = image_ids
    # å�ªå�–æœ€å��ä¸€æ‰¹

if last_batch_images:
    print(f"å±•ç¤ºæœ€å��ä¸€æ‰¹å¤„ç�†çš„å›¾åƒ� ({len(last_batch_images)}å¼ ):")
    
    # å¯¹æœ€å��ä¸€æ‰¹å›¾åƒ�è¿›è¡Œæ�¨ç�†
    last_batch_images = [img.to(device) for img in last_batch_images]
    with torch.no_grad():
        last_outputs = model(last_batch_images)
    
    # æ˜¾ç¤ºæœ€å��ä¸€æ‰¹çš„é¢„æµ‹ç»“æ�œ
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (image, output, image_id) in enumerate(zip(last_batch_images, last_outputs, last_batch_ids)):
        if i < 4:  # æœ€å¤šæ˜¾ç¤º4å¼ 
            # æ��å�–é¢„æµ‹ç»“æ�œ
            boxes = output['boxes'].data.cpu().numpy()
            scores = output['scores'].data.cpu().numpy()
            
            # è¿‡æ»¤
            keep_indices = scores >= detection_threshold
            filtered_boxes = boxes[keep_indices]
            filtered_scores = scores[keep_indices]
            
            # å‡†å¤‡å›¾åƒ�
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # æ˜¾ç¤º
            axes[i].imshow(image_np)
            
            # ç»˜åˆ¶æ¡†
            for box, score in zip(filtered_boxes, filtered_scores):
                x1, y1, x2, y2 = map(int, box)
                color = 'red' if score >= 0.8 else ('orange' if score >= 0.6 else 'yellow')
                
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none', alpha=0.7
                )
                axes[i].add_patch(rect)
                
                axes[i].text(x1, y1-5, f'{score:.2f}', 
                           color=color, fontsize=8, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7))
            
            axes[i].set_title(f'{image_id} - {len(filtered_boxes)}ä¸ªæ£€æµ‹', fontsize=10)
            axes[i].axis('off')
            
            print(f"  {i+1}. {image_id}: {len(filtered_boxes)}ä¸ªæ£€æµ‹")
    
    # éš�è—�å¤šä½™çš„å­�å›¾
    for i in range(len(last_batch_images), 4):
        axes[i].axis('off')
    
    plt.suptitle('æœ€å��ä¸€æ‰¹å›¾åƒ�é¢„æµ‹ç»“æ�œ', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================
# ç”Ÿæˆ�æœ€ç»ˆæŠ¥å‘Š
# ============================================

print("\n" + "="*70)
print("ğŸ“‹ å®�è·µä»»åŠ¡å®Œæˆ�æŠ¥å‘Š")
print("="*70)

print(f"""
âœ… å®�è·µä»»åŠ¡å®Œæˆ�æƒ…å†µæ€»ç»“:
{'â”�'*60}

ğŸ“¸ å›¾ç‰‡å�¯è§†åŒ–ä»»åŠ¡å®Œæˆ�:
  1. å·²å¤„ç�†æ‰€æœ‰ {total_images_processed} å¼ æµ‹è¯•å›¾åƒ�
  2. ç”Ÿæˆ�äº† {len(output_files) if 'output_files' in locals() else 0} å¼ å�¯è§†åŒ–å›¾ç‰‡
  3. æ¯�å¼ å›¾ç‰‡éƒ½åŒ…å�«äº†:
     - å�Ÿå§‹æµ‹è¯•å›¾åƒ�
     - é¢„æµ‹è¾¹ç•Œæ¡†ï¼ˆä¸�å�Œé¢œè‰²è¡¨ç¤ºä¸�å�Œç½®ä¿¡åº¦ï¼‰
     - ç½®ä¿¡åº¦æ ‡ç­¾
     - æ£€æµ‹ç»Ÿè®¡ä¿¡æ�¯

ğŸ”� å�¯è§†åŒ–ç‰¹ç‚¹:
  â€¢ çº¢è‰²æ¡†: é«˜ç½®ä¿¡åº¦ (â‰¥0.8)
  â€¢ æ©™è‰²æ¡†: ä¸­é«˜ç½®ä¿¡åº¦ (â‰¥0.6)
  â€¢ é»„è‰²æ¡†: é˜ˆå€¼é™„è¿‘ (â‰¥{detection_threshold})
  â€¢ æ¯�ä¸ªæ¡†ä¸Šæ–¹æ˜¾ç¤ºç½®ä¿¡åº¦åˆ†æ•°

ğŸ“Š ç»Ÿè®¡ç»“æ�œ:
  â€¢ æ€»æ£€æµ‹æ¡†æ•°: {total_boxes_detected}
  â€¢ å¹³å�‡æ¯�å¼ å›¾åƒ�æ£€æµ‹æ•°: {total_boxes_detected/total_images_processed if total_images_processed > 0 else 0:.2f}
  â€¢ æ£€æµ‹ç�‡: {sum(1 for stats in all_predictions_stats if stats['num_detections'] > 0)/len(all_predictions_stats)*100 if all_predictions_stats else 0:.1f}%
  â€¢ å¤„ç�†è€—æ—¶: {total_time:.2f}ç§’

ğŸ“� è¾“å‡ºæ–‡ä»¶:
  1. å�¯è§†åŒ–å›¾ç‰‡ç›®å½•: {output_dir}
     - æ ¼å¼�: [image_id]_prediction.png
  2. ç»Ÿè®¡å›¾è¡¨: {stats_chart_path if 'stats_chart_path' in locals() else 'æœªç”Ÿæˆ�'}
  3. è¯¦ç»†ç»Ÿè®¡æ–‡ä»¶: {os.path.join(output_dir, 'detailed_statistics.csv') if os.path.exists(output_dir) else 'æœªç”Ÿæˆ�'}

ğŸ�¯ ä»»åŠ¡ç›®æ ‡è¾¾æˆ�:
  âœ“ å¯¹æµ‹è¯•é›†ä¸­çš„æ‰€æœ‰å›¾åƒ�ä¾�æ¬¡è¿›è¡Œæ�¨ç�†
  âœ“ æ��å�–æ¯�å¼ å›¾åƒ�çš„é¢„æµ‹boxesä¸�scores  
  âœ“ æ ¹æ�®é˜ˆå€¼è¿‡æ»¤ä½�ç½®ä¿¡åº¦æ¡†
  âœ“ åœ¨å›¾åƒ�ä¸Šç»˜åˆ¶é¢„æµ‹æ¡†å¹¶æ˜¾ç¤ºç»“æ�œ

ğŸ“ˆ æ¨¡å�‹è¡¨ç�°åˆ†æ��:
  - æ¨¡å�‹åœ¨æµ‹è¯•é›†ä¸Šçš„æ£€æµ‹æ•ˆæ�œå·²é€šè¿‡å�¯è§†åŒ–å®Œæ•´å±•ç¤º
  - å�¯ä»¥é€šè¿‡ç”Ÿæˆ�çš„å›¾ç‰‡ç›´è§‚è¯„ä¼°æ£€æµ‹è´¨é‡�
  - ç»Ÿè®¡å›¾è¡¨æ��ä¾›äº†é‡�åŒ–åˆ†æ��æ•°æ�®

ğŸ’¡ å��ç»­å»ºè®®:
  1. æ£€æŸ¥æ— æ£€æµ‹çš„å›¾åƒ�ï¼Œåˆ†æ��å�Ÿå› 
  2. è°ƒæ•´é˜ˆå€¼ä¼˜åŒ–æ£€æµ‹ç»“æ�œ
  3. æ ¹æ�®å�¯è§†åŒ–ç»“æ�œæ”¹è¿›æ¨¡å�‹
""")

# ä¿�å­˜æœ€ç»ˆçš„å�¯è§†åŒ–ç»“æ�œç½‘æ ¼
print("\nğŸ“¸ ç”Ÿæˆ�æœ€ç»ˆçš„å�¯è§†åŒ–ç»“æ�œç½‘æ ¼...")

# ä»�è¾“å‡ºç›®å½•ä¸­é€‰æ‹©9å¼ ä»£è¡¨æ€§å›¾ç‰‡
representative_files = []
if 'output_files' in locals() and output_files:
    # é€‰æ‹©9å¼ ä¸�å�Œæ£€æµ‹æ•°é‡�çš„å›¾ç‰‡
    import random
    random.seed(42)
    
    # æŒ‰æ£€æµ‹æ•°é‡�åˆ†ç»„
    groups = {'æ— æ£€æµ‹': [], 'å°‘é‡�': [], 'ä¸­ç­‰': [], 'å¤§é‡�': []}
    for stats in all_predictions_stats:
        if stats['num_detections'] == 0:
            groups['æ— æ£€æµ‹'].append(stats['image_id'])
        elif 1 <= stats['num_detections'] <= 3:
            groups['å°‘é‡�'].append(stats['image_id'])
        elif 4 <= stats['num_detections'] <= 10:
            groups['ä¸­ç­‰'].append(stats['image_id'])
        else:
            groups['å¤§é‡�'].append(stats['image_id'])
    
    # ä»�æ¯�ç»„ä¸­é€‰æ‹©å›¾ç‰‡
    for group_name, image_ids in groups.items():
        if image_ids:
            selected_id = random.choice(image_ids)
            representative_files.append(f"{selected_id}_prediction.png")
    
    # å¦‚æ�œä¸�è¶³9å¼ ï¼Œéš�æœºè¡¥å……
    while len(representative_files) < 9 and len(output_files) > len(representative_files):
        additional_file = random.choice([f for f in output_files if f not in representative_files])
        representative_files.append(additional_file)
    
    representative_files = representative_files[:9]
    
    # æ˜¾ç¤ºæœ€ç»ˆçš„å�¯è§†åŒ–ç½‘æ ¼
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, img_file in enumerate(representative_files):
        if idx < 9:
            img_path = os.path.join(output_dir, img_file)
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                axes[idx].imshow(img)
                
                # è®¾ç½®æ ‡é¢˜ï¼ˆç®€åŒ–ï¼‰
                img_id = img_file.replace('_prediction.png', '')
                axes[idx].set_title(img_id[:10] + '...', fontsize=9)
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, 'å›¾åƒ�æœªæ‰¾åˆ°', 
                              ha='center', va='center', fontsize=10)
                axes[idx].axis('off')
    
    plt.suptitle('æµ‹è¯•é›†é¢„æµ‹ç»“æ�œå�¯è§†åŒ–ç½‘æ ¼', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print(f"âœ… å·²ç”Ÿæˆ�åŒ…å�« {len(representative_files)} å¼ å›¾ç‰‡çš„å�¯è§†åŒ–ç½‘æ ¼")

print("\n" + "="*70)
print("ğŸ�‰ å®�è·µä»»åŠ¡å®Œæˆ�ï¼�æ‰€æœ‰æµ‹è¯•å›¾åƒ�çš„é¢„æµ‹ç»“æ�œå·²å�¯è§†åŒ–å®Œæˆ�ï¼�")
print("="*70)

