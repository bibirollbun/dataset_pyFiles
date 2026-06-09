import cv2 
import numpy as np 
import matplotlib.pyplot as plt


im_path1 = '/kaggle/input/image-matching-challenge-2024/train/church/images/00005.png'
im_path2 = '/kaggle/input/image-matching-challenge-2024/train/church/images/00050.png'
im_path3 = '/kaggle/input/image-matching-challenge-2024/train/church/images/00060.png'
img1 = cv2.imread(im_path1)
img2 = cv2.imread(im_path2)
img3 = cv2.imread(im_path3)


img1


def flannMatcher_opencv(img1, img2):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    plt.figure(figsize = (10,5))
    plt.imshow(img3,),plt.show()


flannMatcher_opencv(img1, img2)


flannMatcher_opencv(img1, img3)


!git clone https://github.com/cvg/LightGlue.git 
!python -m pip install -e LightGlue/


import torch
from LightGlue.lightglue import viz2d
from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import numpy_image_to_torch, rbd


class cfg:    
    lightglue = {
        "extractor": "SuperPoint", # SuperPoint, DISK
        "device": "cpu", # cpu, cuda
        "max_kpts": 2048,
        "homography": {
            "method": cv2.RANSAC,
            "ransacReprojThreshold": 3.0
        }
    }



def preprocess_lightglue(img):
    img = numpy_image_to_torch(img)
    return img


def match_lightglue(img0, img1, cfg):
    img0 = preprocess_lightglue(img0)
    img1 = preprocess_lightglue(img1)
    
    if cfg["extractor"] == "SuperPoint":
        extractor = SuperPoint(max_num_keypoints=cfg["max_kpts"]).eval().to(cfg["device"])
        matcher = LightGlue(features='superpoint').eval().to(cfg["device"])

    if cfg["extractor"] == "DISK":
        extractor = DISK(max_num_keypoints=cfg["max_kpts"]).eval().to(cfg["device"])  # load the extractor
        matcher = LightGlue(features='disk').eval().to(cfg["device"])  # load the matcher

    # extract local features
    feats0 = extractor.extract(img0)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(img1)
    
    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    
    # get results
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = kpts0[matches[..., 0]]  # coordinates in img0, shape (K,2)
    points1 = kpts1[matches[..., 1]]  # coordinates in img1, shape (K,2)
        
    return {
        "points0": points0,
        "points1": points1,
        "matches01": matches01, 
        "matches": matches,
        "kpts0": kpts0,
        "kpts1": kpts1,
        "img0": img0,
        "img1": img1
    }


def visualize_lightglue(img0, img1, points0, points1, kpts0, kpts1, matches01, plt_key=True, rotation=0, **kwargs, ):
    axes = viz2d.plot_images([img0, img1],dpi=70)
    viz2d.plot_matches(points0, points1, color='lime', lw=0.2)
#     viz2d.add_text(1, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.add_text(0, f'Img1', fs=20)
    viz2d.add_text(1, f'Img2: Rotation {rotation} deg.', fs=20)

    if(plt_key):
        kpc0, kpc1 = viz2d.cm_prune(matches01['prune0']), viz2d.cm_prune(matches01['prune1'])
        viz2d.plot_images([img0, img1], dpi=70)
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)


# LightGlue
%time results_lightglue = match_lightglue(img1, img2, cfg.lightglue)
visualize_lightglue(**results_lightglue)


# LightGlue
%time results_lightglue = match_lightglue(img1, img3, cfg.lightglue)
visualize_lightglue(**results_lightglue)


from matplotlib.patches import Circle


img_path1 = '/kaggle/input/image-matching-challenge-2024/train/dioscuri/images/3dom_fbk_img_1517.png'
img_path2 = '/kaggle/input/image-matching-challenge-2024/train/dioscuri/images/img_0272.png'
img_path3 = '/kaggle/input/image-matching-challenge-2024/train/dioscuri/images/img_0257.png'
im1 = cv2.imread(img_path1)
im2 = cv2.imread(img_path2)
im3 = cv2.imread(img_path3)


def rotate(image, angle:int):
    num_rotate = int(angle/90)
    for i in range(abs(num_rotate)):
        if(angle<0):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if(angle>0):
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def viz_rotated(img0, img1):
    for i in range(4):
        angle = i*(-90)
        img_rotate = rotate(img1, angle)
        results_lightglue = match_lightglue(img0, img_rotate, cfg.lightglue)
        visualize_lightglue(**results_lightglue, plt_key=False, rotation=angle)
        print(f'Number of matched keypoints (for {angle} degrees rotation): ', len(results_lightglue['matches01']['matches']))


viz_rotated(im1, im2)


import math
(h, w) = im2.shape[:2]
(cX, cY) = (w // 2, h // 2)
pX, pY = 400,400


def find_correspondence(x, y, rotate_angle, image):
    """
    Finding the (x, y) coordinate of source image in destination image
    x, y : coordinate of source image
    rotate_angle : angle to move source image to get source image (negative angle)
    image : source image
    """
    img = rotate(image, rotate_angle)
    angle_rad = math.radians(rotate_angle)
    (h, w) = image.shape[:2]
    
    mat = np.array([[math.cos(angle_rad),  -math.sin(angle_rad)],[math.sin(angle_rad), math.cos(angle_rad)]])
    res = mat @ np.array([x,y])
    x_ = res[0]
    y_ = res[1]
    print(h, w, x_,y_)
    
    if rotate_angle==0:
        circ = Circle((x_,y_),20, color='red')
        
    elif rotate_angle==-90:
        x1 = h - x_
        y1 = 0 - y_
        circ = Circle((x1,y1),20, color='red')
        print(x1,y1)
    elif rotate_angle==-180:
        x1 = w + x_
        y1 = h + y_
        circ = Circle((x1,y1),20, color='red')
        print(x1,y1)
    elif rotate_angle==-270:
        x1 = 0 - x_
        y1 = w - y_
        circ = Circle((x1,y1),20, color='red')
        print(x1,y1)
    
    return circ, img


def viz_correspondence(x, y, image):
    fig, axs = plt.subplots(2,2, figsize=(10, 10))
    for i in range(4):
        circle, img = find_correspondence(x, y, i*(-90), image)
        if i<2:
            axs[0,i].add_patch(circle)
            axs[0,i].imshow(img)
        else:
            axs[1,i-2].add_patch(circle)
            axs[1,i-2].imshow(img)


viz_correspondence(450, 450, im2)


def reverse_tensor(t, rotate_angle, image):
    """
    Finding the coordinate of destination image in source image
    t : tensor of destination coordinates
    rotate_angle : angle to move dest. image to get source image (positive angle)
    image : destination image
    """

    img = rotate(image, rotate_angle)
    angle_rad = math.radians(rotate_angle)
    (h, w) = img.shape[:2] # h, w is from source image
    
    if rotate_angle==0:
        t_ = t
    elif rotate_angle==90:
        tt = torch.Tensor([h, 0])
        t_ = torch.sub(tt, t)
    elif rotate_angle==180:
        tt = torch.Tensor([w, h])
        t_ = torch.sub(t, tt)
    elif rotate_angle==270:
        tt = torch.Tensor([0, w])
        t_ = torch.sub(tt, t)

    mat = torch.Tensor([[math.cos(-angle_rad),  -math.sin(-angle_rad)],[math.sin(-angle_rad), math.cos(-angle_rad)]])
    res = [torch.linalg.inv(mat) @ t_[i] for i in range(len(t_))]
    final = torch. stack(res)
    return final



angle = 0
rotated_img = rotate(im2, angle)
results_lightglue = match_lightglue(im1, rotated_img, cfg.lightglue)
visualize_lightglue(**results_lightglue, plt_key=False, rotation=angle)
print(f'Number of matched keypoints (for {angle} degrees rotation): ', len(results_lightglue['matches01']['matches']))


angle = -90
rotated_img = rotate(im2, angle)
results_lightglue = match_lightglue(im1, rotated_img, cfg.lightglue)
visualize_lightglue(**results_lightglue, plt_key=False, rotation=angle)
print(f'Number of matched keypoints (for {angle} degrees rotation): ', len(results_lightglue['matches01']['matches']))


# After adjusting the coordinate
new_tensor = reverse_tensor(results_lightglue['points1'], 90, rotated_img)
results_lightglue1 = match_lightglue(im1, im2, cfg.lightglue)
results_lightglue1['points0'] = results_lightglue['points0']
results_lightglue1['points1'] = new_tensor
visualize_lightglue(**results_lightglue1, plt_key=False, rotation=-0)




