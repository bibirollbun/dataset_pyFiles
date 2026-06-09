import os
import sys
import time
import pickle
import matplotlib
import numpy as np
import pandas as pd
import scipy.special
import json, math,cv2
import sys, os, importlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.segmentation import mark_boundaries
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

matplotlib.rcParams['text.usetex'] = True

# Stretch Notebook Width to 98% size of the Screen
from IPython.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

sys.path.insert(1, '/kaggle/input/xai-with-lime-image-for-computer-vision')


!git clone https://github.com/rashidrao-pk/lime-stratified-examples


import utils as ut
sys.path.insert(1, '/kaggle/working/lime-stratified-examples/lime_stratified')
sys.path.insert(1, '/kaggle/working/lime-stratified-examples')


from dataclasses import dataclass
@dataclass
class Parameters:
    dummy                : bool  = False
    model_name           : str   = 'ResNet50'
    target_seg_no        : int   = 50
    random_seed          : int = 1234
    use_stratification   : bool = True
    top_labels           : int  = 3
    hide_color           : str = None
    num_samples          : int = 50
##########################################################
params = ut.Parameters()
params.random_state = params.random_seed
##########################################################
@dataclass
class Paths:
    dummy                : bool  = False
paths = Paths()

from types import SimpleNamespace
results = SimpleNamespace()


paths.main_results = os.getcwd()
paths.local_data      =  '/kaggle/input/xai-with-lime-image-for-computer-vision'
paths.json_file       =  os.path.join(paths.local_data,'data/imagenet_class_index.json')
paths.imagenet_path   = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/'

paths.DS_path         =   os.path.join(paths.imagenet_path, "test")
paths.DS_path_subset         =   os.path.join(paths.local_data, "data")
paths.result_folder   =   os.path.join(paths.main_results, "result")
paths.paper_figures   =   os.path.join(paths.main_results,"Paper_Figures")

print(paths.json_file, os.path.exists(paths.json_file))
ut.check_folders(paths.DS_path)
ut.check_folders(paths.result_folder)
ut.check_folders(paths.paper_figures)


im_ext = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

#get images
def get_im(IM_DIR,file_path='imagenet_testset_files.pkl'):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            im_list = pickle.load(f)
        return im_list
    im_list = []
    i = 1
    for root, directories, files in tqdm(os.walk(IM_DIR)):
        for file in files:
            if any(ext in file for ext in im_ext):
                im_list.append(os.path.join(root, file))
            i += 1
    with open(file_path, 'wb') as f:
        pickle.dump(im_list, f)
    return im_list


im_paths = sorted(get_im(paths.DS_path, file_path = os.path.join(paths.main_results, 'imagenet_testset_files.pkl')))
im_names = [f.split('/')[-1] for f in im_paths]

im_total = len(im_paths)
print(f'total number of images = {im_total}')


# load pre-trained model and data
weights_path = "weights/resnet50_weights.h5"
os.makedirs('weights', exist_ok=True)
model = ut.load_model(model_name=params.model_name, 
                      weights_path = weights_path
                     )


# getting ImageNet class names
class_names = ut.get_ImageNet_ClassLabels(paths.json_file)
print('classes count :',len(class_names))


# data_type = 'local'
data_type = 'imagenet'
########################################################################
if data_type == 'local':
    params.image_name = 'bird5.png'
    results.file = os.path.join(paths.DS_path_subset,params.image_name)
elif data_type == 'imagenet':
    params.image_name = 'ILSVRC2012_test_00000125.JPEG'
    results.file = os.path.join(paths.DS_path,params.image_name)
params.image_base_name = params.image_name.split('.')[0]
########################################################################
results.image_to_explain = ut.read_process_image(results.file,model)
print(f'{"Image: ":<15} {results.file} \n{"Shape":<15} {results.image_to_explain.shape}')


# for op in sys.path:
#     # if op in 'working':
#         print(op)


from lime_stratified.lime import lime_image
lime_explainer = lime_image.LimeImageExplainer(random_state=params.random_seed)
from lime_stratified.lime.wrappers.scikit_image import SegmentationAlgorithm


def get_sep():
    print('-'*100)
def plot_segments(results,paths,params, save_plot=True):
    # num_segments = len(np.unique(results.segments))
    fig,axes = plt.subplots(1,2, figsize=(6,3))
    axes[0].imshow(results.image_to_explain); axes[0].set_xticks([]); axes[0].set_yticks([]);  
    axes[1].imshow(mark_boundaries(results.image_to_explain, results.segments))
    axes[1].set_xticks([]); axes[1].set_yticks([]); 
    plt.suptitle(f'{results.num_segments} segments')
    if save_plot:
        img_name = params.image_name.split('.')[0]
        # plt.savefig(f'{paths.paper_figures}/{image_name}_image_{num_segments}_segments.pdf', dpi=150, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f'{paths.paper_figures}/{img_name}_image_{results.num_segments}_segs.png', transparent=True,dpi=150, bbox_inches='tight', pad_inches=0.02)
        plt.show()
######################################################################
def compare_lime(results_baseline,results_stratified, params,paths, positive_only=True,
                      num_features=1000, min_weight_fact=2, cmap='bwr',save_plots=True, verbose=True):
    v_baseline   = np.max(np.abs(results_baseline.heatmap))
    v_stratified = np.max(np.abs(results_stratified.heatmap))
    if verbose:
        print('v_baseline ', v_baseline)
        print('v_stratified ', v_stratified)
        
    original_shape = results_baseline.image_to_explain.shape[:2]
    fig, axes = plt.subplots(1, 8, figsize=(16, 4), constrained_layout=True)
    
    axes[0].imshow(results_baseline.image_to_explain)
    axes[1].imshow(mark_boundaries(results_baseline.image_to_explain, results_baseline.segments))
    ##################################################################
    temp_base, mask_base = results_baseline.explanation.get_image_and_mask(results_baseline.explanation.top_labels[0],
                                             positive_only=positive_only,
                                             num_features=num_features,
                                             hide_rest=False,
                                             min_weight=v_baseline / min_weight_fact)
    
    # Plot heatmap with matching extent to make it visually same-sized
    im2 = axes[2].imshow(results_baseline.heatmap, cmap=cmap, vmin=-v_baseline, vmax=v_baseline,
                         extent=[0, original_shape[1], original_shape[0], 0])
    fig.colorbar(im2, ax=axes[2], fraction=0.05, pad=0.04)
    
    axes[3].imshow(mark_boundaries(temp_base.astype(np.uint8), mask_base))
    #############################################################################
    temp_st, mask_st = results_stratified.explanation.get_image_and_mask(results_stratified.explanation.top_labels[0],
                                             positive_only=positive_only,
                                             num_features=num_features,
                                             hide_rest=False,
                                             min_weight=v_stratified / min_weight_fact)
    
    # Plot heatmap with matching extent to make it visually same-sized
    plt.gca().set_aspect('equal')
    ut.plot_classification_score(axes[4], results_baseline.explanation,
                                 results_baseline.X, results_baseline.Y, params.f_x,
                                plot_everything = False)
    ###################################################################################################
    im3 = axes[5].imshow(results_stratified.heatmap, cmap=cmap, vmin=-v_stratified, vmax=v_stratified,
                         extent=[0, original_shape[1], original_shape[0], 0])
    fig.colorbar(im3, ax=axes[5], fraction=0.05, pad=0.04)
    ###############
    axes[6].imshow(mark_boundaries(temp_st.astype(np.uint8), mask_st))
    ##################################################################################################
    plt.gca().set_aspect('equal')
    ut.plot_classification_score(axes[7], results_stratified.explanation,
                                 results_stratified.X, results_stratified.Y, params.f_x,
                                plot_everything = False)
    
    # Uniform look for all axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        # fr"{method} $\mathbf{{CV}}$"
    rcy_ttl_bl = f'$RC-LIME = \\mathbf{{ {ut.get_RCY(results_baseline.Y, params.f_x):.3} }}$'
    rcy_ttl_sl = f'$RC-St-LIME = \\mathbf{{ {ut.get_RCY(results_stratified.Y, params.f_x):.3} }}$'
    
    cvb_bl = fr'LIME $\mathbf{{CV}}$ : {results_baseline.cv_beta:0.4}'
    cvb_sl = fr'St-LIME $\mathbf{{CV}}$ : {results_stratified.cv_beta:0.4}'
    
    ttl_ls = ['input', f'Segments:{results_baseline.num_segments}',cvb_bl,'LIME-feat',rcy_ttl_bl,cvb_sl ,'St-LIME-feat',rcy_ttl_sl]
    
    for ax,ttl in zip(axes, ttl_ls):
        ax.set_title(ttl)
    
    plt.suptitle(f'predicted as {class_names[params.predicted_cls_idx]}'
                 f' f(x)={params.f_x:.5}  g(x1)={results_baseline.g_x:.5}, g(x1)={results_stratified.g_x:.5}',
                fontsize=16)
    # plt.tight_layout()
    # plt.subplots_adjust()
    if save_plots:
        img_name = params.image_name.split('.')[0]
    
    plt.show()


    
def plot_explanations(results, params,paths , positive_only=True,save_plots=False,
                      num_features=1000, min_weight_fact=2, cmap='bwr', verbose=True):
    image_name = params.image_name.split('.')[0]
    v = np.max(np.abs(results.heatmap))
    if verbose:
        print(f'{"max imp ":<15} =  {v}')
    
    temp_1, mask_1 = results.explanation.get_image_and_mask(results.explanation.top_labels[0],
                                             positive_only=positive_only,
                                             num_features=num_features,
                                             hide_rest=True,
                                             min_weight=v / min_weight_fact)

    temp_2, mask_2 = results.explanation.get_image_and_mask(results.explanation.top_labels[0],
                                             positive_only=positive_only,
                                             num_features=num_features,
                                             hide_rest=False,
                                             min_weight=v / min_weight_fact)

    # Resize heatmap to match the original image shape (height, width)
    original_shape = temp_1.shape[:2]
    heatmap = results.heatmap
    if heatmap.shape[:2] != original_shape:
        heatmap = cv2.resize(heatmap, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create subplots with equal aspect and no padding
    fig, axes = plt.subplots(1, 5, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(results.image_to_explain)
    axes[1].imshow(mark_boundaries(results.image_to_explain, results.segments))
    
    # axes[2].imshow(mark_boundaries(temp_1.astype(np.uint8), mask_1))

    # Classification Score Plot
    # fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    
    # plt.title()
    
    axes[2].imshow(mark_boundaries(temp_2.astype(np.uint8), mask_2))
    ##################################################################################
    plt.gca().set_aspect('equal')
    ut.plot_classification_score(axes[3], results.explanation, results.X, results.Y, params.f_x)
    ##################################################################################
    # Plot heatmap with matching extent to make it visually same-sized
    im = axes[4].imshow(heatmap, cmap=cmap, vmin=-v, vmax=v, extent=[0, original_shape[1], original_shape[0], 0])
    fig.colorbar(im, ax=axes[4], fraction=0.05, pad=0.04)
    
    # Uniform look for all axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
    rcy_ttl = f'$RC(Y) = \\mathbf{{ {ut.get_RCY(results.Y, params.f_x):.3} }}$'
    ttl_ls = ['input', 'segments','feat',rcy_ttl, f'Beta : {results.cv_beta:0.4}']
    for ax,ttl in zip(axes, ttl_ls):
        ax.set_title(ttl)
    
    plt.suptitle(f'predicted as {class_names[params.predicted_cls_idx]}  '
                      f'f(x)={params.f_x:.5}  g(x)={results.g_x:.5}')
    # plt.tight_layout(w_pad=0.05, h_pad=0.05)
    # plt.subplots_adjust(hspace=0.05, wspace=0.05)
    if save_plots:
        # Save files
        for fmt in ['svg', 'pdf', 'png']:
            plt.savefig(f'{paths.paper_figures}/{image_name}_image_mask_heatmap_single.{fmt}',
                    dpi=150, bbox_inches='tight', pad_inches=0.02,
                    transparent=(fmt == 'png'))
    plt.show()
#######################################################################

def get_beta_exp(results, verbose=False):
    xpld_cls = results.explanation.top_labels[0]
    results.g_x = results.explanation.local_pred[xpld_cls][0]
    results.beta = get_beta_from_expl(explanation=results.explanation)    
    results.std_beta = np.std((results.beta))
    results.mean_beta = np.mean((results.beta))
    
    if verbose:
        print('g(x) \t\t= ', results.g_x)
        print('sum(beta) \t= ', np.sum(results.beta))
        print('CV(beta) \t= ',results.std_beta/results.mean_beta)
    return results


from tensorflow.keras.applications.resnet50 import preprocess_input
def bb_predict(imgs):
    # On some platform, you will need model.predict(..) instead of model(..)
    return model.predict(preprocess_input(imgs), verbose=False)
#     return model(preprocess_input(imgs))


predicted = bb_predict(np.array([results.image_to_explain]))
(params.predicted_cls_idx,params.f_x,\
 params.predicted_cls_lbl) =  ut.get_class_idx_label_score (predicted,class_names)
print(params.predicted_cls_idx, params.predicted_cls_lbl, params.f_x)


csv_segments_file = os.path.join(paths.result_folder,'segments_db.csv')
params.max_dist = ut.get_max_dist_load(params,results,csv_segments_file, verbose=True)


results.segments,results.num_segments,segmenter_fn = ut.own_seg(results.image_to_explain,
                                                                md=params.max_dist,ks=4,
                                                                random_seed=params.random_seed,ratio=0.2)
print(f'num_segments created --> {results.num_segments} - {segmenter_fn}')


explanation = lime_explainer.explain_instance(results.image_to_explain,             # image being explained
              bb_predict,                   # prediction model 
              labels=class_names,           # classes names from ImageNet dataset
              segmentation_fn=segmenter_fn, # custom Segmenter function to generate exactly same superpixels
              top_labels=params.top_labels,       # top explanation
              hide_color=params.hide_color,       # superpixel replacement strategy
              use_stratification=params.use_stratification, # Boolean value to switch the proposed method to be used or not 
              # batch_size=100,               # batch size
              num_samples=params.num_samples)             # no of 
results.X, results.all_Ys, results.explanation = explanation
results.Y = results.all_Ys[:, params.predicted_cls_idx]



results = ut.get_beta_exp(results)
results.heatmap = ut.heatmap_from_beta(segments=results.segments, beta=results.beta)
results.cv_beta = ut.get_CV_beta(results.beta)
plot_explanations(results,params,paths,positive_only=False,
                     num_features=100,min_weight_fact=3,cmap='bwr', verbose=False)


print(f'{"local_pred ":<15} -> {results.explanation.local_pred}')
print(f'{"score ":<15} -> {results.explanation.score}')
print(f'{"segments":<15} -> {len(np.unique(results.explanation.segments))}')
print(f'{"top_labels ":<15} -> {results.explanation.top_labels}')


def get_flow(params=None,paths=None, verbose=False,plot_full=True,verbose_seg=True):
    results = SimpleNamespace()
    
    results.file = os.path.join(paths.DS_path,params.image_name)
    params.image_base_name = params.image_name.split('.')[0]
    results.image_to_explain = ut.read_process_image(results.file,model)
    if verbose:
        print(f'{params.image_name} loaded with shape : {results.image_to_explain.shape}')
        get_sep()
    ##########################################################################################################################
    predicted = bb_predict(np.array([results.image_to_explain]))
    
    (params.predicted_cls_idx,params.f_x,params.predicted_cls_lbl) =  ut.get_class_idx_label_score (predicted,class_names)

    # predicted_cls = np.argmax(predicted[0])
    # f_x = predicted[0][predicted_cls]
    if verbose:
        print('Predicted Class\t\t: \t',params.predicted_cls_lbl,
              '\nClass Probability\t:\t', params.f_x,
              '\nPredicted Class Index\t:\t', params.predicted_cls_idx)
        get_sep()
    ##########################################################################################################################
    params.max_dist = ut.get_max_dist_load(params,results,csv_segments_file, verbose=verbose_seg)
    # max_dist,_,_,_ = ut.search_segment_number(results.image_to_explain, target_seg_no=params.target_seg_no)
    if verbose:
        print(f'{params.target_seg_no} segments requires : max_dist: {params.max_dist}')
    results.segments,results.num_segments,segmenter_fn = ut.own_seg(results.image_to_explain,
                                                                    md=params.max_dist,
                                                                    ks=4,
                                                                    random_seed=params.random_seed,
                                                                    ratio=0.2)
    ##########################################################################################################################
    lime_explainer = lime_image.LimeImageExplainer(random_state=params.random_seed) 
    # Boolean value to switch the proposed method to be used or not
    if verbose:
        print('TOP Labels: ', params.top_labels)
        print('hide_color: ', params.hide_color)
        print('use_stratification: ', params.use_stratification)
        print('num_samples: ', params.num_samples)
    
    explanation = lime_explainer.explain_instance(results.image_to_explain,            # image being explained
                                          bb_predict,                   # prediction model 
                                          labels=class_names,           # classes names from ImageNet dataset
                                          segmentation_fn=segmenter_fn, # custom Segmenter function to generate exactly same superpixels
                                          top_labels=params.top_labels,                 # top explanation
                                          hide_color=params.hide_color,              # superpixel replacement strategy
                                          use_stratification=params.use_stratification, # Boolean value to switch the proposed method to be used or not 
                                          num_samples=params.num_samples)             # no of samples
    results.X, results.all_Ys, results.explanation = explanation
    results.Y = results.all_Ys[:, params.predicted_cls_idx]

    if verbose:
        print(explanation.top_labels[0])
    results = ut.get_beta_exp(results)
    results.heatmap = ut.heatmap_from_beta(segments=results.segments, beta=results.beta)
    results.cv_beta = ut.get_CV_beta(results.beta)
    if verbose:
        print('CV Value ', ut.get_CV_beta(results_baseline.beta))
    if plot_full:
        plot_explanations(results,params,paths,positive_only=params.positive_only,
                      num_features=params.num_features,
                      min_weight_fact=params.min_weight_fact,
                      cmap=params.cmap,
                         verbose=verbose)
    
    return results
def plot_rc_score(results,params):
    
    # Classification Score Plot
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    plt.gca().set_aspect('equal')
    ut.plot_classification_score(ax, results.explanation, results.X, results.Y, params.f_x)
    plt.title(f'$RC(Y) = \\mathbf{{ {ut.get_RCY(results.Y, params.f_x):.3} }}$')
    plt.tight_layout()
    plt.show()



params.target_seg_no = 100   # Segments Generation
#################################
params.num_samples = 100   # Budget for Explanation Generation 
#################################
params.num_features = 100
params.positive_only = False   # Parameter for Feature Importance Visualization
params.min_weight_fact = 2     # Parameter for Feature Importance Visualization
params.cmap = 'bwr'

params.image_name = 'ILSVRC2012_test_00000125.JPEG'


params.use_stratification = False
results_baseline = get_flow(params=params,paths=paths, verbose_seg = False)


params.use_stratification = True
results_stratified = get_flow(params=params,paths=paths, verbose_seg = False)


compare_lime(results_baseline,results_stratified,
             params,paths,
             positive_only=False,
             num_features=20,
             min_weight_fact=2, 
             cmap='bwr', #
             verbose=True
            )


params.target_seg_no = 100   # Segments Generation
#################################
params.num_samples = 1000   # Budget for Explanation Generation 
#################################
params.num_features = 100
params.positive_only = False   # Parameter for Feature Importance Visualization
params.min_weight_fact = 2     # Parameter for Feature Importance Visualization
params.cmap = 'bwr'

params.image_name = 'ILSVRC2012_test_00000125.JPEG'


# TEST_ON_MULTI_IMAGE = True
TEST_ON_MULTI_IMAGE = False

if TEST_ON_MULTI_IMAGE:
    selected_images = []
    image_no = [
                114,     147,      60,       144,  66
                ]
    for im_idx,ino in enumerate(image_no):
        selected_images.append(f'ILSVRC2012_test_{ino:08}.JPEG')
        
    print(selected_images)


if TEST_ON_MULTI_IMAGE:
    for imn in selected_images:
        results = SimpleNamespace()
        params.image_name = imn
        ############################################################################################
        params.use_stratification = False
        results_baseline = get_flow(params=params,paths=paths, plot_full=False, verbose_seg=False)
        ############################################################################################
        params.use_stratification = True
        results_stratified = get_flow(params=params,paths=paths, plot_full=False, verbose_seg=False)
        ############################################################################################
        compare_lime(results_baseline,results_stratified,
                     params,paths,
                     positive_only=False,
                     num_features=20,
                     min_weight_fact=2, 
                     cmap='bwr', #
                     verbose=False
                    )

