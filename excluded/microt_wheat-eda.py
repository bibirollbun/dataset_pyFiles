import cv2
import pandas as pd


def labels(path_to_file, image_width, image_height):
    """
    读取YOLO格式的标签文件，并转换为 OpenCV 矩形框格式 (x1, y1, x2, y2)。

    参数:
    path_to_file (str): YOLO标注的txt文件路径
    image_width (int): 原始图像的宽度
    image_height (int): 原始图像的高度

    返回:
    list: [(class_id, x1, y1, x2, y2), ...]
    """
    
    # 读取YOLO格式的txt文件 (空格分隔)
    df = pd.read_csv(path_to_file, delimiter=" ", header=None, names=["class_id", "x_center", "y_center", "width", "height"])
    df=df.drop(df['class_id'])
    # 计算 OpenCV 矩形框格式 (x1, y1, x2, y2)
    df["x1"] = ((df["x_center"] - df["width"] / 2) * image_width).astype(int)
    df["y1"] = ((df["y_center"] - df["height"] / 2) * image_height).astype(int)
    df["x2"] = ((df["x_center"] + df["width"] / 2) * image_width).astype(int)
    df["y2"] = ((df["y_center"] + df["height"] / 2) * image_height).astype(int)

    # 返回转换后的数据 [(class_id, x1, y1, x2, y2), ...]
    return df[[ "x1", "y1", "x2", "y2"]].values.tolist()
    


import os
def show(picture_name='00333207f.jpg'):
    root_picture_path='/kaggle/input/global-wheat-detection/train/'
    root_label_path='/kaggle/input/labels/labels'
    
    path_to_picture_file=os.path.join(root_picture_path,picture_name)
    path_to_label_file=os.path.join(root_label_path,picture_name.split('.')[0]+'.txt')
    
    image=cv2.imread(path_to_picture_file)
    # 假设有多个框的坐标 (x1, y1, x2, y2)
    image_width=1024
    image_height=1024
    boxes = labels(path_to_label_file, image_width, image_height)
    
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    
    
    import matplotlib.pyplot as plt
    
    # OpenCV 读取的图像是 BGR，需要转换为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image_rgb)
    plt.axis("off")  # 隐藏坐标轴
    plt.show()


show()


import os
from collections import Counter
import matplotlib.pyplot as plt
root='/kaggle/input/labels/labels'
list_file=os.listdir(root)
box_num_list=[]
for filename in list_file:
    path=os.path.join(root,filename)
    df = pd.read_csv(path, delimiter=" ",header=None)
    if(df.iloc[:,4]>0.5).any():
        show(filename.split('.')[0]+'.jpg')
    

    if(df.iloc[:,3]>0.5).any():
        show(filename.split('.')[0]+'.jpg')
    #print(df)
    num_box=len(df)
    if(num_box>160):
        show(filename.split('.')[0]+'.jpg')
    box_num_list.append(num_box)
    


# 使用Counter统计每个数字出现的次数
count = Counter(box_num_list)

# 获取数字和它们的计数
numbers = list(count.keys())
counts = list(count.values())

# 绘制柱状图
plt.bar(numbers, counts)

# 添加标题和标签
plt.title('BOX Frequency')
plt.xlabel('Number of Box')
plt.ylabel('Count')

# 显示图形
plt.show()
    

