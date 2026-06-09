!pip install faiss-gpu



%%writefile cluster_images.py

import os
import json
import random
import numpy as np
import faiss
import glob
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

def parse_val_name(filename:str)->str:
    """ Extracts "00000001" from "ILSVRC2012_val_00000001.jpg """
    base=os.path.splitext(os.path.basename(filename))[0]
    base=base.replace("ILSVRC2012_val_","")
    return base

def parse_train_name(filepath: str) -> str:
    """Extracts "n01440764_10040" from "n01440764/n01440764_10040.jpg"."""
    base = os.path.splitext(filepath)[0]
    return base.split("/")[-1]
def load_and_flatten(file:str)->np.ndarray:
    """ Loads an image, resizes to 256x256, center-crops it to 224x224, and normalizes. """
    transform=transforms.Compose([
        transforms.Resize(256,interpolation=Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
    ])
    img=Image.open(file).convert("RGB")
    img=transform(img)
    return img.numpy().flatten()

def extract_class_from_xml(xml_path: str) -> str:
    """ Extracts the class ID from an ImageNet validation XML annotation file. """
    
    tree=ET.parse(xml_path)
    root=tree.getroot()
    obj=root.find("object")
    return obj.find("name").text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to the folder containing 1k subfolders of train images.")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to the folder containing 50k val images named ILSVRC2012_val_XXXXXX.jpg.")
    parser.add_argument("--val_xml_dir", type=str, required=True,
                        help="Path to the folder containing XML annotations for val images.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Where to save train_grouping_X.json and val_grouping.json.")
    parser.add_argument("--split_index", type=int, choices=range(5), required=True,
                        help="Which split of the training data to process (0 to 4).")
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed for picking 10k centroid images.")
    parser.add_argument("--K", type=int, required=True,
                        help="So luong cum")
    args=parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    K=args.K
    train_dir = args.train_dir
    val_dir = args.val_dir
    val_xml_dir = args.val_xml_dir
    output_dir = args.output_dir
    split_index = args.split_index
    os.makedirs(output_dir, exist_ok=True)

    #1. Lấy ảnh ở tập valid

    val_paths=sorted(glob.glob(os.path.join(val_dir,"ILSVRC2012_val_*.JPEG")))
    print(f"Found {len(val_paths)} validation images.")





    #3 load va faltten KK tam 
    shape = 224 * 224 * 3
    print("Dimension (shape) =", shape)
    centroid_vectors = np.random.normal(0, 1, (K, shape)).astype(np.float32)
    

    centroid_vectors=np.stack(centroid_vectors,axis=0).astype(np.float32)

    #4 tao chi muc cua thu vien faiss
    d=centroid_vectors.shape[1]
    index=faiss.IndexFlatL2(d)
    index.add(centroid_vectors)


    #5 phan cum tren tap valid
    val_grouping = { str(i): [] for i in range(K) }
    batch_size=64
    for start_idx in tqdm(range(0,len(val_paths),batch_size),total=(len(val_paths) // batch_size + 1)):
        batch_paths=val_paths[start_idx:start_idx+batch_size]
        batch_vecs=np.stack([load_and_flatten(p) for p in batch_paths],axis=0).astype(np.float32)
        distances,indices=index.search(batch_vecs,1)

        for i,pth in enumerate(batch_paths):
            cluster_id=indices[i,0]
            val_name=parse_val_name(pth)
            xml_path=os.path.join(val_xml_dir, f"ILSVRC2012_val_{val_name}.xml")
            class_id=extract_class_from_xml(xml_path)
            val_grouping[str(cluster_id)].append(f"{class_id}_{val_name}")


    #6 phan cum tren tap train ( moi lan xu ly 1/5)

    train_grouping={str(i): [] for i in range(K)}
    train_paths=sorted(glob.glob(os.path.join(train_dir,"*/*.JPEG")))
    total_train=len(train_paths)
    chunk_size=total_train //5
    #start_idx=split_index*chunk_size
    #end_idx=total_train if split_index==4 else (split_index+1)*
    start_idx=0
    end_idx=total_train
    train_paths=train_paths[start_idx:end_idx];
    print(f"Xu ly anh tu {start_idx} to {end_idx} ({len(train_paths)})")
    for start_idx in tqdm(range(0,len(train_paths),batch_size),total=len(train_paths)//batch_size+1):
        batch_slice=train_paths[start_idx:start_idx+batch_size]
        batch_vecs=np.stack([load_and_flatten(p) for p in batch_slice],axis=0).astype(np.float32)
        distances,indices=index.search(batch_vecs,1)
        for i,p in enumerate(batch_slice):
            cluster_id=indices[i,0]
            train_name=parse_train_name(p)
            train_grouping[str(cluster_id)].append(train_name)

    

    #7 luu file json

    val_json_path=os.path.join(output_dir,"val_grouping.json")
    train_json_path=os.path.join(output_dir,f"train_group_{split_index}.json")
    with open(val_json_path,"w") as f:
        json.dump(val_grouping,f)
    with open(train_json_path,"w") as f:
        json.dump(train_grouping,f)

    print(f"\nSaved val_grouping.json and train_grouping_{split_index}.json")


if __name__=="__main__":
    main()
    

    


# %%writefile cluster_images.py

# import os
# import json
# import random
# import numpy as np
# import faiss
# import glob
# import argparse
# import xml.etree.ElementTree as ET
# from PIL import Image
# from tqdm import tqdm
# import torchvision.transforms as transforms

# def parse_val_name(filename:str)->str:
#     """ Extracts "00000001" from "ILSVRC2012_val_00000001.jpg """
#     base=os.path.splitext(os.path.basename(filename))[0]
#     base=base.replace("ILSVRC2012_val_","")
#     return base

# def parse_train_name(filepath: str) -> str:
#     """Extracts "n01440764_10040" from "n01440764/n01440764_10040.jpg"."""
#     base = os.path.splitext(filepath)[0]
#     return base.split("/")[-1]
# def load_and_flatten(file:str)->np.ndarray:
#     """ Loads an image, resizes to 256x256, center-crops it to 224x224, and normalizes. """
#     transform=transforms.Compose([
#         transforms.Resize(256,interpolation=Image.BILINEAR),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img=Image.open(file).convert("RGB")
#     img=transform(img)
#     return img.numpy().flatten()

# def extract_class_from_xml(xml_path: str) -> str:
#     """ Extracts the class ID from an ImageNet validation XML annotation file. """
    
#     tree=ET.parse(xml_path)
#     root=tree.getroot()
#     obj=root.find("object")
#     return obj.find("name").text

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_dir", type=str, required=True,
#                         help="Path to the folder containing 1k subfolders of train images.")
#     parser.add_argument("--val_dir", type=str, required=True,
#                         help="Path to the folder containing 50k val images named ILSVRC2012_val_XXXXXX.jpg.")
#     parser.add_argument("--val_xml_dir", type=str, required=True,
#                         help="Path to the folder containing XML annotations for val images.")
#     parser.add_argument("--output_dir", type=str, default=".",
#                         help="Where to save train_grouping_X.json and val_grouping.json.")
#     parser.add_argument("--split_index", type=int, choices=range(5), required=True,
#                         help="Which split of the training data to process (0 to 4).")
#     parser.add_argument("--seed", type=int, default=100,
#                         help="Random seed for picking 10k centroid images.")
#     parser.add_argument("--K", type=int, required=True,
#                         help="So luong cum")
#     args=parser.parse_args()
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     K=args.K
#     train_dir = args.train_dir
#     val_dir = args.val_dir
#     val_xml_dir = args.val_xml_dir
#     output_dir = args.output_dir
#     split_index = args.split_index
#     os.makedirs(output_dir, exist_ok=True)

#     #1. Lấy ảnh ở tập valid

#     val_paths=sorted(glob.glob(os.path.join(val_dir,"ILSVRC2012_val_*.JPEG")))
#     print(f"Found {len(val_paths)} validation images.")





#     #3 load va faltten KK tam 
#     shape = 224 * 224 * 3
#     print("Dimension (shape) =", shape)
#     centroid_vectors = np.random.normal(0, 1, (K, shape)).astype(np.float32)
    

#     centroid_vectors=np.stack(centroid_vectors,axis=0).astype(np.float32)

#     #4 tao chi muc cua thu vien faiss
#     d=centroid_vectors.shape[1]
#     index=faiss.IndexFlatL2(d)
#     index.add(centroid_vectors)


#     #5 phan cum tren tap valid
#     val_grouping = { str(i): [] for i in range(K) }
#     batch_size=64
#     for start_idx in tqdm(range(0,len(val_paths),batch_size),total=(len(val_paths) // batch_size + 1)):
#         batch_paths=val_paths[start_idx:start_idx+batch_size]
#         batch_vecs=np.stack([load_and_flatten(p) for p in batch_paths],axis=0).astype(np.float32)
#         distances,indices=index.search(batch_vecs,1)

#         for i,pth in enumerate(batch_paths):
#             cluster_id=indices[i,0]
#             val_name=parse_val_name(pth)
#             xml_path=os.path.join(val_xml_dir, f"ILSVRC2012_val_{val_name}.xml")
#             class_id=extract_class_from_xml(xml_path)
#             val_grouping[str(cluster_id)].append(f"{class_id}_{val_name}")


#     #6 phan cum tren tap train ( moi lan xu ly 1/5)

#     train_grouping={str(i): [] for i in range(K)}
#     train_paths=sorted(glob.glob(os.path.join(train_dir,"*/*.JPEG")))
#     total_train=len(train_paths)
#     chunk_size=total_train //5
#     # start_idx=split_index*chunk_size
#     # end_idx=total_train if split_index==4 else (split_index+1)*chunk_size
#     start_idx=0
#     end_idx=total_train
#     train_paths=train_paths[start_idx:end_idx];
#     print(f"Xu ly anh tu {start_idx} to {end_idx} ({len(train_paths)})")
#     for start_idx in tqdm(range(0,len(train_paths),batch_size),total=len(train_paths)//batch_size+1):
#         batch_slice=train_paths[start_idx:start_idx+batch_size]
#         batch_vecs=np.stack([load_and_flatten(p) for p in batch_slice],axis=0).astype(np.float32)
#         distances,indices=index.search(batch_vecs,1)
#         for i,p in enumerate(batch_slice):
#             cluster_id=indices[i,0]
#             train_name=parse_train_name(p)
#             train_grouping[str(cluster_id)].append(train_name)

    

#     #7 luu file json

#     val_json_path=os.path.join(output_dir,f"val_grouping_K{K}.json")
#     train_json_path=os.path.join(output_dir,f"train_group_K{K}.json")
#     with open(val_json_path,"w") as f:
#         json.dump(val_grouping,f)
#     with open(train_json_path,"w") as f:
#         json.dump(train_grouping,f)

#     print(f"\nSaved val_group_K{K}.json and train_group_K{K}.json")


# if __name__=="__main__":
#     main()
    

    


!python3 cluster_images.py --train_dir "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train" \
                          --val_dir "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val" \
                          --val_xml_dir "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val" \
                          --output_dir "/kaggle/working" \
                          --split_index 0 \
                          --K 200 \
                          --seed 42







