!pip install --no-index --no-deps /kaggle/input/yolo-pkg/yolo/ultralytics-8.3.112-py3-none-any.whl


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PIL import Image
from tqdm.notebook import tqdm
from ultralytics import YOLO
import cv2
import yaml
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["ULTRALYTICS_CALLBACKS_DISABLE"] = "raytune"

root_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
train_path = os.path.join(root_path,'train')
test_path = os.path.join(root_path,'test')
train_labels_path= os.path.join(root_path,'train_labels.csv')
train_labels = pd.read_csv(train_labels_path)

Trust = 4
split_ratio=0.8
BOX_SIZE=24


'''
    1. create a folder for the yolo dataset both for training and validation
    2. extract the tomo_id that have motors
    3. split the ids into training and validation where split ratio is 0.8
    4. for each motor collect (2*Trust + 1)slices , normalize and save to the yolo_dataset
    5. create the yaml configuration
'''

# Step1

yolo_data_path = '/kaggle/working/'
yolo_train_img = os.path.join(yolo_data_path,'images','train')
yolo_train_label= os.path.join(yolo_data_path,'labels','train')
yolo_val_img = os.path.join(yolo_data_path,'images','val')
yolo_val_label=os.path.join(yolo_data_path ,'labels','val')
for path in [yolo_train_img,yolo_train_label,yolo_val_img,yolo_val_label]:
    os.makedirs(path, exist_ok=True)

#steps 2 and 3

motor_tomo_id = train_labels[train_labels['Number of motors'] > 0]['tomo_id'].unique()
np.random.shuffle(motor_tomo_id)
train_ids = motor_tomo_id[:int(len(motor_tomo_id)*split_ratio)]
val_ids = motor_tomo_id[int(len(motor_tomo_id)*split_ratio):]

# Steps 4

def normalize_slice(img):
    p2=np.percentile(img,2)
    p98=np.percentile(img,98)
    clipped_data = np.clip(img,p2,p98)
    normalize_img =  255*(img-p2)/(p98-92)
    return np.uint8(normalize_img)

def process_yolo_dataset(train_ids,yolo_train_img,yolo_train_label):
    col_name=['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2','Array shape (axis 0)']
    new_col_name = ['tomo_id','z_axis','y_axis','x_axis','z_max']
    train_df = train_labels[train_labels['tomo_id'].isin(train_ids)][col_name]
    train_df = train_df.sort_values(by='tomo_id').reset_index(drop=True)
    print('total motors',len(train_df))
    train_df.columns = new_col_name

    process_slice=0
    for motor in tqdm(train_df.itertuples(index=False),desc=f'processing slices'):
        tomo_id,z_centre,y_centre,x_centre,max_slice = motor.tomo_id , motor.z_axis,motor.y_axis,motor.x_axis,motor.z_max
        z_min ,z_max= int(max(0,z_centre-Trust)),int(min(z_centre + Trust , max_slice))
        for z in range(int(z_min),int(z_max+1)):
            slice_name =f'slice_{z:04d}.jpg'
            slice_path = os.path .join(train_path,tomo_id,slice_name)
            img= Image.open(slice_path)
            img_arr= np.array(img)
            img_normalize = normalize_slice(img_arr)
            file_name= f"{tomo_id}_z{int(z):04d}_y{int(y_centre):04d}_x{int(x_centre):04d}.jpg"
            file_path = os.path.join(yolo_train_img,file_name)
            Image.fromarray(img_arr).save(file_path)
            img_width,img_height=img.size
            x_centre_norm= x_centre/img_width
            y_centre_norm=y_centre/img_height
            box_width = BOX_SIZE/img_width
            box_height=BOX_SIZE/img_height
            label_path = os.path.join(yolo_train_label,file_name.replace('.jpg','.txt'))
            process_slice+=1
            with open(label_path,'w') as f:
                f.write(f"0 {x_centre_norm} {y_centre_norm} {box_width} {box_height}\n")

    return process_slice

train_slice = process_yolo_dataset(train_ids,yolo_train_img,yolo_train_label)
print(f'total slice for training  is {train_slice}')
val_slice = process_yolo_dataset(val_ids,yolo_val_img,yolo_val_label)
print(f'total slice for validation  is {val_slice}')

# step5

yaml_content={
        'path':yolo_data_path,
        'train':'images/train',
        'val':'images/train' if not os.path.exists(yolo_val_img) else 'images/val',
        'names':{0:'motor'}
    }
with open(os.path.join(yolo_data_path,'dataset.yaml') ,'w') as f:
    yaml.dump(yaml_content,f,default_flow_style= False)




'''
from ultralytics import YOLO
from matplotlib.patches import Rectangle
import random
from pathlib import Path



yaml_path = os.path.join(yolo_data_path,'dataset.yaml')
yolo_weights_dir = "/kaggle/working/yolo_weights"
#yolo_pretrained_weights = "yolov8n.pt"
yolo_pretrained_weights=Path('/kaggle/input/yolo-model2/yolov8n.pt')
os.makedirs(yolo_weights_dir,exist_ok=True)



class YOLOTraining:
    def __init__(self,yaml_path,pretrained_weight_path):
        self.yaml_path =yaml_path
        self.pretrained_weight_path=pretrained_weight_path

    def train(self):
        model =YOLO(self.pretrained_weight_path)
        
        results = model.train(data=self.yaml_path,epochs=30,batch=16,imgsz=640,
        project=yolo_weights_dir,name='motor_detector',exist_ok=True,
        patience=5,              # Early stopping if no improvement for 5 epochs
        save_period=5,           # Save checkpoints every 5 epochs
        val=True,                # Ensure validation is performed
        verbose=True )            # Show detailed output during training

        run_dir = os.path.join(yolo_weights_dir,'motor_detector')
        print('saved weight path',run_dir)
        return model,results

  
    def predict(self,model,num_sample=4):
        
        data_path = os.path.join(yolo_data_path,yolo_val_img)
        if not os.path.exists(data_path):
            print('No data is exist for validation')
            return
        val_images=os.listdir(data_path)

        num_samples= min(num_sample,len(val_images))
        sample_img = random.sample(val_images,num_samples)

        fig,axes=plt.subplots(2,2,figsize=(12,12))
        ax=axes.flatten()
        
        for i , img_file in enumerate(sample_img):
            img_path =os.path.join(data_path,img_file)
            results = model.predict(img_path, conf=0.1)[0]
            img=Image.open(img_path)
            ax[i].imshow(np.array(img),cmap='gray')

        
            # draw the actual label rectangle
            parts= img_file.split('_')
            x_part = [p for p in parts if p.startswith('x')]
            y_part = [p for p in parts if p.startswith('y')]
            x_c= int(x_part[0][1:].split('.')[0])
            y_c = int(y_part[0][1:])
            box=24
            rect_actual = Rectangle((x_c - box//2 , y_c - box//2),box,box,linewidth=1, edgecolor='g', facecolor='none')
            ax[i].add_patch(rect_actual)

            # predicted label rectangle

            if len(results.boxes) > 0:
                print('here predicting the result')
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    print('boxes coordinate',[ x1, y1, x2, y2])
                    rect_pred = Rectangle((x1, y1), x2-x1, y2-y1, 
                                         linewidth=1, edgecolor='r', facecolor='none')
                    ax[i].add_patch(rect_pred)
                    ax[i].text(x1, y1-5, f'{conf:.2f}', color='red')
            ax[i].set_title(f"Image: {img_file}\nGround Truth (green) vs Prediction (red)")
    
        plt.tight_layout()
        
        # Save the predictions plot
        plt.savefig(os.path.join('/kaggle/working', 'predictions.png'))
        plt.show()

#if yolo_pretrained_weights.exists():
yolo = YOLOTraining(yaml_path,yolo_pretrained_weights)
model,results = yolo.train()
yolo.predict(model,4)
# else:
#     print('model path is not exist')
'''





def preload_image_batch(slice_paths):
        images=[]
        for path in slice_paths:
            img=cv2.imread(path)
            if img is None:
                img= np.array(Image.open(path))
            images.append(img)
        return images

class GPUProfile:
    def __init__(self,name):
        self.name = name
        self.start_time = None
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
    def __exit__(self,*args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed= time.time()-self.start_time
        print(f"[profile] {self.name}:{elapsed:.3f}s")


def process_tomogram(tomo_id,model,index=0, total=1):
    tomo_dir = os.path.join(test_dir,tomo_id)
    slices_file = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    CONCENTRATION=1
    selected_indices = np.linspace(0,len(slices_file)-1,int(len(slices_file))*CONCENTRATION)
    selected_indices= np.round(selected_indices).astype(int)
    slice_files=[slices_file[i] for i in selected_indices]
    all_detections=[]
    if device.startswith('cuda'):
        streams = [torch.cuda.Stream() for _ in range(min(4,BATCH_SIZE))]
    else:
        streams=[None]
    
    next_batch_thread=None
    next_batch_image= None
    
    for batch_start in range(0,len(slice_files),BATCH_SIZE):
        if next_batch_thread is not None:
            next_batch_thread.join()
            next_batch_image= None
    
        batch_end = min(batch_start+ BATCH_SIZE ,len(slice_files))
        current_slices = slice_files[batch_start:batch_end]
    
        next_batch_start = batch_end
        next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
        next_slices = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []
    
        
    
        if next_slices:
            slice_path = [ os.path.join(tomo_dir,f) for f in next_slices]
            next_batch_thread = threading.Thread(target = preload_image_batch,args=(slice_path,))
            next_batch_thread.start()
        else:
            next_batch_thread=None
        sub_batches = np.array_split(current_slices,len(streams))
        sub_batches_result=[]
        for i , sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue
            stream = streams[i%len(streams)]
            with torch.cuda.stream(stream) if stream and device.startswith('cuda') else nullcontext:
                sub_batch_path =[os.path.join(tomo_dir,f) for f in sub_batch if f.endswith('.jpg')]
                sub_batch_slice_num = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]
                with GPUProfile(f"Inference batch{i+1}/{len(sub_batches)}"):
                    sub_results = model(sub_batch_path,verbose=False)
                for j , result in enumerate(sub_results):
                    if (len(result.boxes))>0:
                        boxes= result.boxes
                        for box_id , confidence in enumerate(boxes.conf):
                            if confidence >= CONFIDENCE_THRESHOLD:
                                    # Get bounding box coordinates
                                    x1, y1, x2, y2 = boxes.xyxy[box_idx].cpu().numpy()
                                    
                                    # Calculate center coordinates
                                    x_center = (x1 + x2) / 2
                                    y_center = (y1 + y2) / 2
                                    
                                    # Store detection with 3D coordinates
                                    all_detections.append({
                                        'z': round(sub_batch_slice_nums[j]),
                                        'y': round(y_center),
                                        'x': round(x_center),
                                        'confidence': float(confidence)
                                    })
        # Synchronize streams
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        
    # Clean up thread if still running
    if next_batch_thread is not None:
        next_batch_thread.join()
    
    # 3D Non-Maximum Suppression to merge nearby detections across slices
    final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)
    
    # Sort detections by confidence (highest first)
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # If there are no detections, return NA values
    if not final_detections:
        return {
            'tomo_id': tomo_id,
            'Motor axis 0': -1,
            'Motor axis 1': -1,
            'Motor axis 2': -1
        }
    
    # Take the detection with highest confidence
    best_detection = final_detections[0]
    
    # Return result with integer coordinates
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']),
        'Motor axis 1': round(best_detection['y']),
        'Motor axis 2': round(best_detection['x'])
    }
                                
            
def perform_3d_nms(detections, iou_threshold):
    """
    Perform 3D Non-Maximum Suppression on detections to merge nearby motors
    """
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # List to store final detections after NMS
    final_detections = []
    
    # Define 3D distance function
    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 + 
                       (d1['y'] - d2['y'])**2 + 
                       (d1['x'] - d2['x'])**2)
    
    # Maximum distance threshold (based on box size and slice gap)
    box_size = 24  # Same as annotation box size
    distance_threshold = box_size * iou_threshold
    
    # Process each detection
    while detections:
        # Take the detection with highest confidence
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        
        # Filter out detections that are too close to the best detection
        detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]
    
    return final_detections
    

    




import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

data_path ='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
test_dir=os.path.join(data_path,'test')
model_path='/kaggle/working/yolo_weights/motor_detector/weights/best.pt'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
CONFIDENCE_THRESHOLD = 0.45  # Lower threshold to catch more potential motors
MAX_DETECTIONS_PER_TOMO = 3  # Keep track of top N detections per tomogram
NMS_IOU_THRESHOLD = 0.2

if device.startswith('cuda'):
    torch.backends.cudnn.banchmark = True
    torch.backends.cudnn.deterministics= False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32=True

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory/1e9
    memory_allocated = torch.cuda.memory_allocated(0)/1e9
    free_memory = gpu_memory - memory_allocated
    BATCH_SIZE = max(8, min(32, int(free_memory * 4)))
else:
    BATCH_SIZE = 4

def generate_submission():

    test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model = YOLO(model_path)
    model.to(device)
    if device.startswith('cuda'):
        model.fuse() # run all the layer like conv , batchnorm , maxpool in one time
        if (torch.cuda.get_device_capability(0)[0]>7) :
            model=model.half()
            print('half precision')
    
    results = []
    motor_found = 0
    
    with ThreadPoolExecutor(max_workers=1) as executer:
        future_to_tomo={}
        for i, tomo_id in enumerate(test_tomos,1):
            future= executer.submit(process_tomogram,tomo_id,model,i,total_tomos)
            future_to_tomo[future]=tomo_id
             # Process completed futures as they complete
        for future in future_to_tomo:
            tomo_id = future_to_tomo[future]
            try:
                # Clear CUDA cache between tomograms
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                result = future.result()
                results.append(result)
                
                # Update motors found count
                has_motor = not pd.isna(result['Motor axis 0'])
                if has_motor:
                    motors_found += 1
                    print(f"Motor found in {tomo_id} at position: "
                          f"z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
                else:
                    print(f"No motor detected in {tomo_id}")
                    
                print(f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")
            
            except Exception as e:
                print(f"Error processing {tomo_id}: {e}")
                # Create a default entry for failed tomograms
            results.append({
                'tomo_id': tomo_id,
                'Motor axis 0': -1,
                'Motor axis 1': -1,
                'Motor axis 2': -1
            })
    
    # Create submission dataframe
    submission_df = pd.DataFrame(results)
    
    # Ensure proper column order
    submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
    submission_path ='/kaggle/working/sample_submission.csv'
    
    # Save the submission file
    submission_df.to_csv(submission_path, index=False)
    return submission_df

'''

# Run the submission pipeline
if __name__ == "__main__":
    # Time entire process  
    start_time = time.time()
    
    # Generate submission
    submission = generate_submission()
    
    # Print total execution time
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

'''







