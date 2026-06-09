import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm.auto import tqdm
import shutil as sh


#!pip install matplotlib>=3.2.2 numpy>=1.18.5 opencv-python>=4.1.2 Pillow PyYAML>=5.3.1 scipy>=1.4.1 torch>=1.7.0 torchvision>=0.8.1 tqdm>=4.41.0 tensorboard>=2.4.1 seaborn>=0.11.0 pandas thop pycocotools>=2.0


pip install /kaggle/input/whlwhl/ensemble_boxes-1.0.9-py3-none-any.whl


import torch

print("PyTorch 版本:", torch.__version__)


#!python /kaggle/input/yolov55/yolov5-5.0/detect.py --source /kaggle/input/global-wheat-detection/test --weights /kaggle/input/yolobest/best.pt --save-txt --save-conf


def iou(box1, boxes):
    """
    计算 IoU。
    :param box1: 单个检测框，形状为 [1, 4]。
    :param boxes: 多个检测框，形状为 [N, 4]。
    :return: IoU 值，形状为 [N]。
    """
    # 计算交集
    x1 = torch.max(box1[:, 0], boxes[:, 0])
    y1 = torch.max(box1[:, 1], boxes[:, 1])
    x2 = torch.min(box1[:, 2], boxes[:, 2])
    y2 = torch.min(box1[:, 3], boxes[:, 3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # 计算并集
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection

    return intersection / union


def ciou_nms(boxes, scores, score_threshold, iou_threshold):
    """
    基于 CIoU 的非极大值抑制 (NMS)，考虑置信度阈值与 IoU 阈值
    
    参数:
        boxes: (N, 4) 形状的张量，每一行表示一个框 (x1, y1, x2, y2)
        scores: (N,) 形状的张量，表示每个框的置信度
        score_threshold: 置信度阈值，低于该值的框将被剔除
        iou_threshold: IoU 阈值，高于该值的框将被认为是重叠的并被移除
    
    返回:
        保留的框 (Tensor), 保留的置信度 (Tensor)
    """
    
    # **1. 过滤掉低置信度框**
    valid_mask = scores >= score_threshold
    boxes, scores = boxes[valid_mask], scores[valid_mask]

    if boxes.numel() == 0:  # 如果没有框，返回空张量
        return torch.empty((0, 4)), torch.empty((0,))

    # **2. 按置信度排序框**
    order = scores.argsort(descending=True)
    keep = []  # 用于存储保留框的索引

    while order.numel() > 0:
        idx = order[0]  # 获取最高置信度框的索引
        keep.append(idx)  # 保留这个框的索引

        if order.numel() == 1:  # 如果只剩下一个框，直接结束
            break

        # **3. 计算当前框与其他框的 CIoU**
        ciou_values = ciou(boxes[idx].unsqueeze(0), boxes[order[1:]])

        # **4. 过滤掉 IoU 过高的框**
        remaining_indices = (ciou_values < iou_threshold).nonzero(as_tuple=True)[0]
        order = order[1:][remaining_indices]  # 只保留 IoU 低于阈值的框
        
        keep_boxes = boxes[torch.tensor(keep)]
        keep_scores = scores[torch.tensor(keep)].unsqueeze(1)  # scores 变为 (N, 1)

    # 拼接：boxes 的 shape (N, 4)，scores 的 shape (N, 1)
    return torch.cat((keep_boxes, keep_scores), dim=1)

def ciou(box1, boxes):
    """
    计算 CIoU。
    :param box1: 单个检测框，形状为 [1, 4]。
    :param boxes: 多个检测框，形状为 [N, 4]。
    :return: CIoU 值，形状为 [N]。
    """
    # 计算 IoU
    ious = iou(box1, boxes)

    # 计算中心点距离
    center_dist = ((box1[:, 0] + box1[:, 2]) / 2 - (boxes[:, 0] + boxes[:, 2]) / 2) ** 2 + \
                  ((box1[:, 1] + box1[:, 3]) / 2 - (boxes[:, 1] + boxes[:, 3]) / 2) ** 2

    # 计算最小外接矩形的对角线长度
    c = ((boxes[:, 2] - boxes[:, 0]) ** 2 + (boxes[:, 3] - boxes[:, 1]) ** 2) ** 0.5

    # 计算宽高比差异
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan((box1[:, 2] - box1[:, 0]) / (box1[:, 3] - box1[:, 1])) - 
                                         torch.atan((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1])), 2)
    alpha = v / (v + ious + 1e-9)

    # 计算 CIoU
    ciou = ious - center_dist / c ** 2 - alpha * v

    return ciou


from ensemble_boxes import *
def run_wbf(boxes, scores, image_size=1024, iou_thr=0.5, skip_box_thr=0.7, weights=None):
    #boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]
    #scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
    labels = [np.zeros(score.shape[0]) for score in scores]
    boxes = [box/(image_size) for box in boxes]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    #boxes, scores, labels = nms(boxes, scores, labels, weights=[1,1,1,1,1], iou_thr=0.5)
    boxes = boxes*(image_size)
    return boxes, scores, labels

def TTAImage(image, index):
    image1 = image.copy()
    if index==0: 
        rotated_image = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image
    elif index==1:
        rotated_image2 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
        rotated_image2 = cv2.rotate(rotated_image2, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image2
    elif index==2:
        rotated_image3 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
        rotated_image3 = cv2.rotate(rotated_image3, cv2.ROTATE_90_CLOCKWISE)
        rotated_image3 = cv2.rotate(rotated_image3, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image3
    elif index == 3:
        return image1
    
def rotBoxes90(boxes, im_w, im_h):
    ret_boxes =[]
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1-im_w//2, im_h//2 - y1, x2-im_w//2, im_h//2 - y2
        x1, y1, x2, y2 = y1, -x1, y2, -x2
        x1, y1, x2, y2 = int(x1+im_w//2), int(im_h//2 - y1), int(x2+im_w//2), int(im_h//2 - y2)
        x1a, y1a, x2a, y2a = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        ret_boxes.append([x1a, y1a, x2a, y2a])
    return np.array(ret_boxes)

def detect1Image(im0, imgsz, model, device, conf_thres, iou_thres):
    img = letterbox(im0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)


    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255.0   
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=True)[0]
    pre = pred[pred[:,:,4]>0.2]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
   
    #print(pre)
    
    #print(f'shape:{pre.shape}')
    #conf= pre[:,4]

    #xyxy = pre[:, :4]
    #print(xyxy.shape)
    #print(conf.shape)
    
    #pre=ciou_nms(xyxy, conf, conf_thres, iou_thres)
    #pre=[pre]
    #print(pre)

    
    boxes = []
    scores = []
    for i, det in enumerate(pred):  # detections per image
        # save_path = 'draw/' + image_id + '.jpg'
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                scores.append(conf)

    boxes = torch.tensor(boxes, device=device)
    scores = torch.tensor(scores, device=device)

    return boxes.cpu().numpy(), scores.cpu().numpy()


def detect():
    source = '../input/global-wheat-detection/test/'
    weights = '/kaggle/input/volov510/best.pt'
    if not os.path.exists(weights):
        weights = '/kaggle/input/yolov5pt2/best.pt'
    imgsz = 1024
    conf_thres = 0.5
    iou_thres = 0.6
    is_TTA = True
    
    imagenames =  os.listdir(source)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # 现在加载模型
    model = attempt_load(weights, map_location=device)  # 使用 YOLOv5 官方提供的方法
    model.to(device).eval()
    # Load model
    #model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    #model.to(device).eval()
    
    dataset = LoadImages(source, img_size=imgsz)

    results = []
    fig, ax = plt.subplots(5, 2, figsize=(30, 70))
    count = 0
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    #for path, img, im0s, _ in dataset:
    for name in imagenames:
        image_id = name.split('.')[0]
        im01 = cv2.imread('%s/%s.jpg'%(source,image_id))  # BGR
        assert im01 is not None, 'Image Not Found '
        # Padded resize
        im_w, im_h = im01.shape[:2]
        if is_TTA:
            enboxes = []
            enscores = []
            for i in range(4):
                im0 = TTAImage(im01, i)
                boxes, scores = detect1Image(im0, imgsz, model, device, conf_thres, iou_thres)
                for _ in range(3-i):
                    boxes = rotBoxes90(boxes, im_w, im_h)
                    
                if 1: #i<3:
                    enboxes.append(boxes)
                    enscores.append(scores) 
            #boxes, scores = detect1Image(im01, imgsz, model, device, conf_thres, iou_thres)
            #enboxes.append(boxes)
            #enscores.append(scores)

            boxes, scores, labels = run_wbf(enboxes, enscores, image_size = im_w, iou_thr=0.6, skip_box_thr=0.5)
            boxes = boxes.astype(np.int32).clip(min=0, max=im_w)
        else:
            boxes, scores = detect1Image(im01, imgsz, model, device, conf_thres, iou_thres)

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        boxes = boxes[scores >= 0.05].astype(np.int32)
        scores = scores[scores >=float(0.05)]
        if count<10:
            #sample = image.permute(1,2,0).cpu().numpy()
            for box, score in zip(boxes,scores):
                cv2.rectangle(im0,
                              (box[0], box[1]),
                              (box[2]+box[0], box[3]+box[1]),
                              (220, 0, 0), 2)
                cv2.putText(im0, '%.2f'%(score), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.5, (255,255,255), 2, cv2.LINE_AA)
            ax[count%5][count//5].imshow(im0)
            count+=1
            
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }

        results.append(result)
    return results


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


import sys
sys.path.append('/kaggle/input/yolov552/yolov5-5.0')  
from utils.datasets import *
import matplotlib.pyplot as plt
from utils.general import *
from models.experimental import attempt_load  # 确保可以正确加载 YOLOv5 模型
results = detect()
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()

