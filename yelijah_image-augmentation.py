import numpy as np 
import pandas as pd  
import cv2

%matplotlib inline
from matplotlib import pyplot as plt


#blending algorithms


# 1. simple color transfer by rgb normalisation
#https://github.com/chia56028/Color-Transfer-between-Images/blob/master/color_transfer.py

def norm_color_transfer(src, dst):

    def get_mean_and_std(x):
        x_mean, x_std = cv2.meanStdDev(x)
        x_mean = np.hstack(np.around(x_mean,2)).reshape(1,1,3)
        x_std = np.hstack(np.around(x_std,2)).reshape(1,1,3)
        return x_mean, x_std

    s = cv2.cvtColor(src,cv2.COLOR_BGR2LAB)
    t = cv2.cvtColor(dst,cv2.COLOR_BGR2LAB)
    s_mean, s_std = get_mean_and_std(s)
    t_mean, t_std = get_mean_and_std(t)

    m = (s-s_mean)*(t_std/s_std)+t_mean
    m = np.round(m)
    m = np.clip(m,0,255).astype(np.uint8)

    m = cv2.cvtColor(m,cv2.COLOR_LAB2BGR)
    return m




# 2. deep blending (in progress)
# https://github.com/owenzlz/DeepImageBlending




# 3. piosson editing  
# https://github.com/PPPW/poisson-image-editing
import scipy.sparse
from scipy.sparse.linalg import spsolve


def laplacian_matrix(n, m):
    """Generate the Poisson matrix.
    Refer to:
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Note: it's the transpose of the wiki's matrix
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)

    return mat_A


def poisson_edit(source, target, mask, offset=(0,0)):
    """The poisson blending function.
    Refer to:
    Perez et. al., "Poisson Image Editing", 2003.
    """

    # Assume:
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min

    M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
    source = cv2.warpAffine(source,M,(x_range,y_range))

    mask = mask[y_min:y_max, x_min:x_max]
    mask[mask != 0] = 1
    #mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    # corners
    # mask[0, 0]
    # mask[0, y_range-1]
    # mask[x_range-1, 0]
    # mask[x_range-1, y_range-1]

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()

        #concat = source_flat*mask_flat + target_flat*(1-mask_flat)

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]

        x = spsolve(mat_A, mat_b)
        #print(x.shape)
        x = x.reshape((y_range, x_range))
        #print(x.shape)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #print(x.shape)

        target[y_min:y_max, x_min:x_max, channel] = x
    return target


#helper
def make_blend_mask(size, object_box):
    x,y,w,h = object_box
    x0=x
    x1=x+w
    y0=y
    y1=y+h


    w,h = size
    mask = np.ones((h,w,3),np.float32)

    for i in range(0,y0):
        mask[i]=i/(y0)
    for i in range(y1,h):
        mask[i]=(h-i)/(h-y1+1)
    for i in range(0,x0):
        mask[:,i]=np.minimum(mask[:,i],i/(x0))
    for i in range(x1,w):
        mask[:,i]=np.minimum(mask[:,i],(w-i)/(w-x1+1))

    return mask


def insert_object(mix, box, crop, mask):
    x,y,w,h = box
    crop = cv2.resize(crop, dsize=(w,h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(w,h), interpolation=cv2.INTER_AREA)

    mix_crop = mix[y:y+h,x:x+w]
    crop = norm_color_transfer(crop, mix_crop)
    mix[y:y+h,x:x+w] = mask*crop +(1-mask)*mix_crop
    return mix



#load dummy object
object_infor = [
    ['00012.jpg',[0,0,1024,848],[230,129,667,623],], # image_file, context_box, object_box
    ['00001.jpg',[0,0,767,1023],[14,75,717,897],],
   # ['00014.jpg',[0,0,1000,567],[168,87,702,419],],
    ['00021.jpg',[680,1144,1244,884],[760,1256,992,724],],
]

def load_dummy_object():

    object = []
    for image_file, context_box, object_box in object_infor:

        image_file = '../input/cots-non-verified-license-yet/' + image_file #download-0.jpeg'
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        x,y,w,h = context_box
        crop = image[y:y+h,x:x+w]
        object_box = np.array(object_box)-[x,y,0,0]
        mask = make_blend_mask((w,h),object_box)

        #image_show('crop',crop, resize=0.5)
        #image_show('mask',mask, resize=0.5)
        #cv2.waitKey(0)
        object.append([crop, mask])
    return object

def load_dummy_background():
    video_id = 1
    video_frame = 9187 #9287 #9187
    image_file = 'video_%d/%d.jpg' % (video_id, video_frame)
    image = cv2.imread('../input/tensorflow-great-barrier-reef/train_images/' + image_file, cv2.IMREAD_COLOR)
    return image

# demo here!

object = load_dummy_object()
background = load_dummy_background()

mix = background.copy()
mix1 = None
for i, box in enumerate([
    [330,158,70,61],
    [208,598,45,52],
    [930,498,120,100],
]):
    crop,mask = object[i]
    
    #mix = insert_object (mix, box, crop, mask*0.5) ------
    x,y,w,h = box
    crop = cv2.resize(crop, dsize=(w,h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(w,h), interpolation=cv2.INTER_AREA)

    mix_crop = mix[y:y+h,x:x+w]
    crop = norm_color_transfer(crop, mix_crop) 
    #crop = poisson_edit(crop, mix_crop, (mask[:,:,0]>0.5).astype(np.float32), offset=(0,0))

    mask = mask*0.8  #mixup ratio
    mix[y:y+h,x:x+w] = mask*crop +(1-mask)*mix_crop
    
    #-------------------------------------------------------
    #     image_show('background',background, resize=1)
    #     image_show('mix',mix, resize=1)
    #     image_show('crop',crop, resize=4)
    #     image_show('mix_crop',mix_crop, resize=4)
    #     image_show('mask1',mask1, resize=4)
    #     image_show('mask',mask, resize=4)
    #     cv2.waitKey(0)
    
    
#show object location for debug
mix1= mix.copy()
for i, box in enumerate([
    [330,158,70,61],
    [208,598,45,52],
    [930,498,120,100],
]):
    x,y,w,h = box
    cv2.rectangle(mix1, (x-50,y-50), (x+w+50,y+h+50), (255,255,255), 2)


    
plt.figure(figsize=(15,20))
plt.title('original background')
plt.imshow(background[...,::-1])

plt.figure(figsize=(15,20))
plt.title('augmeted image')
plt.imshow(mix[...,::-1])

plt.figure(figsize=(15,20))
plt.title('same augmeted image with marking')
plt.imshow(mix1[...,::-1])
plt.show()

