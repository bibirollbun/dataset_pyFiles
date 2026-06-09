#%%
import traceback
#import visualise
import json
import numpy as np
import os
#import matplotlib.pyplot as plt
import fractions
import copy
import itertools
import random
import string
from itertools import chain, combinations 
from itertools import permutations
from scipy.ndimage import label, generate_binary_structure 
from scipy.ndimage import binary_dilation 
from scipy.ndimage import binary_fill_holes 
from scipy.ndimage import binary_erosion 
from scipy.ndimage import measurements
import time
import pickle
from itertools import product
from scipy import stats

starttime_script = time.perf_counter_ns()

class T1:
    def __init__(self):
        self.starttime = time.perf_counter_ns()
    def e(self):
        self.endtime = time.perf_counter_ns()
        print('Total time taken: %f seconds' % ((self.endtime-self.starttime)/1000000000))

class T2:
    def __init__(self):
        self.totaltime = 0

    def s(self):
        self.starttime = time.perf_counter_ns()

    def e(self):
        self.endtime = time.perf_counter_ns()
        self.totaltime += (self.endtime - self.starttime)

    def print(self):
        print('Total cumulative (t2) time taken: %f seconds' % ((self.totaltime)/1000000000))
    




def are_identicalQ(list_arrs): 
    first_arr = list_arrs[0]
    flag = True
    for arr in list_arrs:
        if np.shape(arr) == np.shape(first_arr):
            if np.all(arr == first_arr):
                pass
            else: flag = False
        else: flag = False
    return flag

def are_two_equal_np_arrays(arr1,arr2): 
    areequal = False
    if np.shape(arr1) == np.shape(arr2):
        if np.all(arr1 == arr2):
            areequal = True
    return areequal

def are_two_identical(obj1, obj2): 
    return pickle.dumps(obj1) == pickle.dumps(obj2)

def are_all_identical(list_of_objs): 
    if type(list_of_objs) is not list: print("ERROR"); return
    picklestr = False
    for obj in list_of_objs:
        if picklestr == False: picklestr = pickle.dumps(obj)
        elif pickle.dumps(obj) != picklestr:
            return False
    return True

def label_unique_with_IDs(list_of_objs): 
    
    
    if type(list_of_objs) is not list: print("ERROR"); return
    picklestrs = []
    for obj in list_of_objs: picklestrs.append(pickle.dumps(obj))
    
    mapping = {}; result = [] 
    for s in picklestrs:
        if s not in mapping:
            mapping[s] = len(mapping)
        result.append(mapping[s])    
    return result, [list_of_objs[result.index(m)] for m in range(max(result)+1)] 

def is_x_in_y(x,y): 
    flag = False
    for _ in y:
        if are_two_identical(x,_): flag = True; break
    return flag

def ix_of_x_in_y(x,y): 
    flag = False; c=0
    for _ in y:
        if are_two_identical(x,_): flag = True; break
        c+=1
    return c if flag else None

def ixs_of_x_in_y(x,y): 
    flag = False; c=0; ixlist = []
    for _ in y:
        if are_two_identical(x,_): flag = True; ixlist.append(c)
        c+=1
    return ixlist

def is_any_x_in_y(x,y): 
    flag = False
    for _ in y:
        for __ in x:
            if are_two_identical(__,_): flag = True; break
    return flag    




def get_contiguous_regions(array,background_color,diagonal_connections_allowedQ=False,colourblind_spatial_contiguity_mode=False):

    
    if colourblind_spatial_contiguity_mode: 
        array = (array!=background_color).astype(int)
    

    
    labeled_array = np.zeros_like(array)
    object_count = 0

    if diagonal_connections_allowedQ: s = generate_binary_structure(2, 2) 

    
    for colour in np.unique(array):
        if colour == background_color and background_color is not None: continue
        if diagonal_connections_allowedQ: labeled, num_features = label(array == colour, structure=s)
        else: labeled, num_features = label(array == colour)
        object_count += num_features
        labeled_array = np.where(labeled > 0, labeled + object_count - num_features, labeled_array)
        
    return labeled_array

def get_outline_border_mask(mask, px_border = 1):
    
    structuring_element = np.ones((px_border*2+1, px_border*2+1))
    border_mask = binary_dilation(mask, structure=structuring_element) & ~mask 
    return border_mask

def none_of_x_in_y(x,y): 
    flag = True
    for x_ in x:
        if x_ in y:
            flag = False
    return flag

def at_least_some_of_x_in_y(x,y): 
    flag = False
    for x_ in x:
        if x_ in y: 
            flag = True
    return flag

def create_name(): 
    return ''.join(random.choices(string.ascii_lowercase, k=6))

def generate_rotflips(map): 
    combo_arrays_map = []; combo_labels = [(0,0),(0,1),(0,2),(0,3), (1,0),(1,1),(1,2),(1,3)] 
    combo_priority =  [  1,    2,    2,    2,     3,    3,    3,    3  ]
    for flip in [map,np.flipud(map)]:
        combo_arrays_map.extend([np.rot90(flip, k=i) for i in range(4)])
    return combo_arrays_map

def get_bounding_box_object(mask,map):
    rows, cols = np.where(mask==1)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    bounding_box_map = map[min_row:max_row+1, min_col:max_col+1]
    bounding_box_mask = mask[min_row:max_row+1, min_col:max_col+1]
    topleft_rc = (min_row,min_col)
    return bounding_box_mask, bounding_box_map, topleft_rc

def check_perfect_colorchange(old_colors,new_colors, old_vals,new_vals):

    n_colors = len(old_colors)
    if n_colors == 1: singlecolor = True
    elif n_colors > 1: singlecolor = False 
    
    perfect_colorchange = False; color_changes = [] 
    if singlecolor and (old_colors[0] not in new_colors) and len(new_colors) == 1: color_changes = [[old_colors[0],new_colors[0]]]; perfect_colorchange = True
    else: 
        flag = True; qualifier = False; color_changes=[]
        for old_color in old_colors:
            indices = np.where(old_vals == old_color)[0]
            if np.all(new_vals[indices] == new_vals[indices[0]]):
                if new_vals[indices[0]] == old_color: pass
                else: qualifier = True 
                color_changes.append([old_color, new_vals[indices[0]]]);
            else:
                flag = False 
        if (flag and qualifier): perfect_colorchange = True

    grey_recoloring = False 
    if len(old_colors) == 2 and 5 in old_colors: temp = old_colors.copy(); temp.remove(5); other_color = temp[0]
    if perfect_colorchange and len(old_colors) == 2 and 5 in old_colors and 0 not in old_colors and new_colors == [other_color]: grey_recoloring = True
    

    return perfect_colorchange, grey_recoloring, color_changes

def get_colors_of_obj(mask, map):
    return list(np.unique(map[mask==1])) 

def get_shape_of_obj(mask,map):
    bb_mask, bb_map, topleft_rc = get_bounding_box_object(mask,map)
    return bb_mask

def frame_psuedoobjs(grid): 
    object_options = {}

    
    
    
    
    def get_simple_frames(arr):
        
        

        def find_border_positions(grid):
            
            grid = np.array(grid)

            newgrid = np.zeros((np.shape(grid)))

            
            rows, cols = grid.shape
            
            
            border_positions = set()
            
            
            shape_positions = np.argwhere(grid != 0)
            
            
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            
            for r, c in shape_positions:
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                        border_positions.add((nr, nc))

                        newgrid[nr,nc] = 1
            
            return list(border_positions), newgrid

        def find_segments(data):
            segments = []
            n = len(data)
            i = 0
            while i < n:
                if data[i] == 0:
                    start = i
                    while i < n and data[i] == 0:
                        i += 1
                    end = i
                    segments.append([start, end])
                else:
                    i += 1
            return segments

        def get_simple_frames_1d(collapsed_h,reference_row):

            
            colour_list=[]
            for n in range(len(collapsed_h)):
                if collapsed_h[n]: colour_list.append(reference_row[n])
            colour_list = np.unique(colour_list)

            curr_frame_1d_list = []

            for colour in colour_list:
                
                current_list = []
                for n in range(len(collapsed_h)):
                    if collapsed_h[n] and reference_row[n] == colour: current_list.append(1)
                    else: current_list.append(0) 
                

                
                
                data = np.array(current_list) 
                diff = np.diff(data) 
                starts = np.where(diff != 0)[0] + 1 
                starts = np.insert(starts, 0, 0) 
                starts = np.append(starts, len(data)) 
                lengths = np.diff(starts) 
                values = data[starts[:-1]] 
        

                
                n_frame_lines = list(values).count(1)
                n_regions = list(values).count(0)
        

                
                n_contig_zeros_ = []
                n_contig_ones_ = []
                for n in range(len(lengths)):
                    if values[n] == 0: n_contig_zeros_.append(lengths[n])
                    if values[n] == 1: n_contig_ones_.append(lengths[n])
                n_contig_zeros = np.unique(n_contig_zeros_)
                n_contig_ones = np.unique(n_contig_ones_)
        

                

                
                if len(n_contig_zeros) == 0: 
        
                    continue
            
                if len(n_contig_ones) == 0:
                    print('ERROR, no dividing/frame line found') 
                    continue

                

                
                if n_frame_lines == 1 and n_regions == 1:
        
                    continue

                frame_type = 'non-standard'

                
                if list(values) == [0,1,0]: 
        
                    frame_type = 'single divider 010'

                
                if list(values) == [1,0,1]: 
        
                    frame_type = 'single border 101'

                
                is_equal_regions = False
                if n_regions > 1 and len(n_contig_zeros) == 1:
        
                    is_equal_regions = True

                
                is_equal_thickness = False
                if n_frame_lines > 1 and len(n_contig_ones) == 1:
        
                    is_equal_thickness = True

                is_perfect_frame = False 
                if len(n_contig_zeros) == 1 and len(n_contig_ones) == 1: is_perfect_frame = True

                if is_perfect_frame and n_regions == 2: 
        
                    frame_type = 'perfect 2x' 

                if is_perfect_frame and n_regions == 3: 
        
                    frame_type = 'perfect 3x' 

                if is_perfect_frame and n_regions > 3: 
        
                    frame_type = 'perfect nx' 

                
                
                max_frame_thickness = np.max(n_contig_ones)
        

                
                is_frame_black = False
                if colour == 0: is_frame_black = True
        

                

                if frame_type in ['single divider 010','single border 101','perfect 2x','perfect 3x','perfect nx']: 
                    is_simple_frametype = True
                else: 
                    is_simple_frametype = False
                    continue 

                
                is_acceptable_region_equality = False
                if len(n_contig_zeros) == 1: is_acceptable_region_equality = True

                curr_frame_1d = {}
                curr_frame_1d ['colour'] = int(colour)
                curr_frame_1d ['frame_type'] = frame_type
                curr_frame_1d ['is_acceptable_region_equality'] = is_acceptable_region_equality
                curr_frame_1d ['frame_1d'] = current_list
                curr_frame_1d ['max_frame_thickness'] = max_frame_thickness
                curr_frame_1d ['is_equal_regions'] = is_equal_regions
                curr_frame_1d ['is_equal_thickness'] = is_equal_thickness

                curr_frame_1d_list.append(curr_frame_1d)

            return curr_frame_1d_list        

        
        reference_row = arr[0, :]
        collapsed_h = np.all(arr == reference_row, axis=0)
        reference_col = arr[:, 0]
        collapsed_v = np.all(arr == reference_col[:, np.newaxis], axis=1) 

        
        curr_frame_1d_list_h = get_simple_frames_1d(collapsed_h,reference_row)
        
        curr_frame_1d_list_v = get_simple_frames_1d(collapsed_v,reference_col)
        

        
        all_colours=[]
        for _ in curr_frame_1d_list_h: 
            if _['colour'] not in all_colours: all_colours.append(_['colour'])
        for _ in curr_frame_1d_list_v: 
            if _['colour'] not in all_colours: all_colours.append(_['colour'])

        colours_h = [curr_frame_1d_list_h[_]['colour'] for _ in range(len(curr_frame_1d_list_h))]
        colours_v = [curr_frame_1d_list_v[_]['colour'] for _ in range(len(curr_frame_1d_list_v))]

        contenders = []


        for colour in all_colours:
            if colour in colours_h and colour in colours_v:
                

                if colour in colours_h: 
                    ix = colours_h.index(colour)
                    
                    temph = curr_frame_1d_list_h[ix]['frame_1d']
                    
                    tempstatus1 = curr_frame_1d_list_h[ix]['is_acceptable_region_equality']
                    type1 =  curr_frame_1d_list_h[ix]['frame_type']
                else: temph = [0]*np.shape(arr)[0]

                if colour in colours_v: 
                    ix = colours_v.index(colour)
                    
                    tempv = curr_frame_1d_list_v[ix]['frame_1d']
                    
                    tempstatus2 = curr_frame_1d_list_v[ix]['is_acceptable_region_equality']
                    type2 =  curr_frame_1d_list_v[ix]['frame_type']
                else: tempv = [0]*np.shape(arr)[1]

                segments_h = find_segments(temph)
                segments_v = find_segments(tempv)

                

                segment_c = 0
                subregion_pixels_c = 0; subregion_gridcolour_pixels_c = 0
                for h in range(len(segments_h)):
                    for v in range(len(segments_v)):
                        subregion = arr[segments_v[v][0]:segments_v[v][1],segments_h[h][0]:segments_h[h][1]]
                        
                        subregion_pixels_c += (np.shape(subregion)[0] * np.shape(subregion)[1])
                        subregion_gridcolour_pixels_c += np.sum(np.array(subregion) == colour)

                        segment_c+=1
                

                frac1 = subregion_gridcolour_pixels_c / subregion_pixels_c
                

                

                
                frameobj = np.zeros((np.shape(arr)[0],np.shape(arr)[1]))
                for t in range(len(temph)):
                    if temph[t] == 1: frameobj[:,t:t+1] = 1    
                for t in range(len(tempv)):
                    if tempv[t] == 1: frameobj[t:t+1,:] = 1       

                
                
                list_border_pos, newgrid = find_border_positions(frameobj)
                
                
                border_pixel_c = 0; border_pixel_gridcolour_c = 0
                for h in range(np.shape(newgrid)[0]):
                    for v in range(np.shape(newgrid)[1]):
                        if newgrid[h,v] == 1:
                            border_pixel_c += 1
                            if arr[h,v] == colour:
                                border_pixel_gridcolour_c+=1
                
                frac2 = border_pixel_gridcolour_c / border_pixel_c
                

                if frac1 == 0 and frac2 == 0: 
                    
                    contenders.append({'frame_color':colour,'acceptable_region':tempstatus1 & tempstatus2,'type':'2d','frame_typeh':type1,'frame_typev':type2,'color':colour,'segments_h':segments_h,'segments_v':segments_v})
                else:
                    contenders.append({'frame_color':colour,'acceptable_region':tempstatus1 & tempstatus2,'type':'leaky_2d','frame_typeh':type1,'frame_typev':type2,'color':colour,'segments_h':segments_h,'segments_v':segments_v})

            else: 
                

                if colour in colours_h:

                    ix = colours_h.index(colour)
                    
                    temph = curr_frame_1d_list_h[ix]['frame_1d']
                    tempstatus = curr_frame_1d_list_h[ix]['is_acceptable_region_equality']
                    type1 =  curr_frame_1d_list_h[ix]['frame_type']

                    segments_h = find_segments(temph)

                    subregion_pixels_c = 0; subregion_gridcolour_pixels_c = 0
                    for h in range(len(segments_h)):
                        subregion = arr[:,segments_h[h][0]:segments_h[h][1]]
    
                        subregion_pixels_c += (np.shape(subregion)[0] * np.shape(subregion)[1])
                        subregion_gridcolour_pixels_c += np.sum(np.array(subregion) == colour)

                    frac1 = subregion_gridcolour_pixels_c / subregion_pixels_c
                    if frac1 == 0: 
                        
                        contenders.append({'frame_color':colour,'acceptable_region':tempstatus,'type':'1dh','frame_typeh':type1,'color':colour,'segments_h':segments_h})
                    else:
                        contenders.append({'frame_color':colour,'acceptable_region':tempstatus,'type':'leaky_1dh','frame_typeh':type1,'color':colour,'segments_h':segments_h})

                elif colour in colours_v:

                    ix = colours_v.index(colour)
                    
                    tempv = curr_frame_1d_list_v[ix]['frame_1d']
                    tempstatus = curr_frame_1d_list_v[ix]['is_acceptable_region_equality']
                    type1 =  curr_frame_1d_list_v[ix]['frame_type']

                    segments_v = find_segments(tempv)

                    subregion_pixels_c = 0; subregion_gridcolour_pixels_c = 0
                    for v in range(len(segments_v)):
                        subregion = arr[segments_v[v][0]:segments_v[v][1],:]
    
                        subregion_pixels_c += (np.shape(subregion)[0] * np.shape(subregion)[1])
                        subregion_gridcolour_pixels_c += np.sum(np.array(subregion) == colour)

                    frac1 = subregion_gridcolour_pixels_c / subregion_pixels_c
                    if frac1 == 0: 
                        
                        contenders.append({'frame_color':colour,'acceptable_region':tempstatus,'type':'1dv','frame_typev':type1,'color':colour,'segments_v':segments_v})
                    else:
                        contenders.append({'frame_color':colour,'acceptable_region':tempstatus,'type':'leaky_1dv','frame_typev':type1,'color':colour,'segments_v':segments_v})

        return contenders

    def get_frame_subregions(grid):
        contenders = get_simple_frames(grid)
        results = []
        for cont in contenders: 
            if cont['type'] in ['2d','leaky_2d']:
                segments_h,segments_v = cont['segments_h'],cont['segments_v']; subregions = []
                for v in range(len(segments_v)):
                    for h in range(len(segments_h)):
                        subregion = grid[segments_v[v][0]:segments_v[v][1],segments_h[h][0]:segments_h[h][1]]            
                        subregions.append((v,h,subregion,subregion.shape, (segments_v[v][0],segments_v[v][1],segments_h[h][0],segments_h[h][1]) ))
                results.append({'details':cont,'V':v+1,'H':h+1,'are_equal_shape':are_all_identical([_[3] for _ in subregions]),'subregions':subregions})
            if cont['type'] in ['1dh','leaky_1dh']:
                segments_h = cont['segments_h']; subregions = []
                for h in range(len(segments_h)):
                    subregion = grid[:,segments_h[h][0]:segments_h[h][1]]
                    subregions.append((0,h,subregion,subregion.shape, (segments_h[h][0],segments_h[h][1])))
                results.append({'details':cont,'V':1,'H':h+1,'are_equal_shape':are_all_identical([_[3] for _ in subregions]),'subregions':subregions})
            if cont['type'] in ['1dv','leaky_1dv']:
                segments_v = cont['segments_v']; subregions = []
                for v in range(len(segments_v)):
                    subregion = grid[segments_v[v][0]:segments_v[v][1],:]
                    subregions.append((v,0,subregion,subregion.shape, (segments_v[v][0],segments_v[v][1])))
                results.append({'details':cont,'V':v+1,'H':1,'are_equal_shape':are_all_identical([_[3] for _ in subregions]),'subregions':subregions})
        return results

    frame_cands = get_frame_subregions(grid)

    for frame in frame_cands:
        details_for_score = frame['details']
        score = 0.9 

        subframe_details = frame['subregions']; totalcount = len(subframe_details); are_equal_shape = frame['are_equal_shape']
        subframes = []; counter = 0
        for n in range(len(subframe_details)):
            curr = subframe_details[n]
            if frame['details']['type'] in ['2d','leaky_2d']:
                obj_mask = np.zeros_like(grid)
                obj_mask[curr[4][0]:curr[4][1],curr[4][2]:curr[4][3]] = 1
                obj_maskv = obj_mask
                obj_masko = np.zeros_like(grid)
                obj_map = np.zeros_like(grid)
                obj_map[curr[4][0]:curr[4][1],curr[4][2]:curr[4][3]] = curr[2]
                bb_map = curr[2]
            if frame['details']['type'] in ['1dv','leaky_1dv']:
                obj_mask = np.zeros_like(grid)
                obj_mask[curr[4][0]:curr[4][1],:] = 1
                obj_maskv = obj_mask
                obj_masko = np.zeros_like(grid)
                obj_map = np.zeros_like(grid)
                obj_map[curr[4][0]:curr[4][1],:] = curr[2]
                bb_map = curr[2]
            if frame['details']['type'] in ['1dh','leaky_1dh']:
                obj_mask = np.zeros_like(grid)
                obj_mask[:,curr[4][0]:curr[4][1]] = 1
                obj_maskv = obj_mask
                obj_masko = np.zeros_like(grid)
                obj_map = np.zeros_like(grid)
                obj_map[:,curr[4][0]:curr[4][1]] = curr[2]
                bb_map = curr[2]
            subframes.append({'counter':counter,'vert_c':curr[0],'horiz_c':curr[1],'mask':obj_mask,'maskv':obj_maskv,'masko':obj_masko,'map':obj_map,'are_equal_shape':are_equal_shape,'bb_map':bb_map})
            counter += 1
        object_options[create_name()] = {'type':frame['details']['type'],'totalcount':totalcount,'obj_score':score,'frame_color':frame['details']['frame_color'],'subframes':subframes}
                    

    
    if len(frame_cands) == 0:
        score = 0.7
    else: 
        score = 0.5

    obj_mask = np.ones_like(grid)
    obj_maskv = np.ones_like(grid)
    obj_masko = np.zeros_like(grid)
    obj_map  = grid 
    bb_map = grid
    pseudosubframe = [{'counter':0,'vert_c':0,'horiz_c':0,'mask':obj_mask,'maskv':obj_maskv,'masko':obj_masko,'map':obj_map,'are_equal_shape':None,'bb_map':bb_map}]
    
    object_options[create_name()] = {'type':'wholegrid_obj','totalcount':1,'obj_score':score,'frame_color':None,'subframes':pseudosubframe}


    
    
    return object_options




def no_change(input_map, mask_to_transform,**kwargs):
    output_map = input_map
    transformed_mask = mask_to_transform
    return output_map, transformed_mask

def static(input_map, mask_to_transform, **kwargs): 
    output_map = input_map
    transformed_mask = mask_to_transform
    return output_map, transformed_mask

def disappear(input_map, mask_to_transform,**kwargs):
    output_map = np.zeros_like(input_map)
    transformed_mask = np.zeros_like(mask_to_transform)
    return output_map, transformed_mask

def temp_pseudo_disappear(input_map, mask_to_transform,**kwargs):
    output_map = np.zeros_like(input_map)
    transformed_mask = np.zeros_like(mask_to_transform)
    return output_map, transformed_mask

def full_color(input_map, mask_to_transform, color,**kwargs):
    output_map = np.where(mask_to_transform, color, 0)
    transformed_mask = mask_to_transform
    return output_map, transformed_mask


def swap_two_colors(input_map, mask_to_transform, two_colors_to_swap,**kwargs): 
    
    if len(two_colors_to_swap)!=2: print('Error'); return

    
    
    
    

    result = input_map.copy()
    cond_0 = (input_map == two_colors_to_swap[0]) & (mask_to_transform == 1)
    cond_1 = (input_map == two_colors_to_swap[1]) & (mask_to_transform == 1)
    result[cond_0] = -99
    result[cond_1] = two_colors_to_swap[0]
    result[result == -99] = two_colors_to_swap[1]

    output_map = result
    transformed_mask = mask_to_transform.copy() 
    return output_map, transformed_mask

def recolor(input_map, mask_to_transform, color_changes, **kwargs):
    if color_changes is None: return input_map, mask_to_transform
    result = input_map.copy()
    mask = (mask_to_transform == 1)
    for old, new in color_changes:
        result[(input_map == old) & mask] = new
    output_map = result
    transformed_mask = mask_to_transform.copy() 
    return output_map, transformed_mask



def bool_not(input_map, mask_to_transform,**kwargs):
    
    colors = list(np.unique(input_map[mask_to_transform==1]))
    if len(colors) == 2: two_colors_to_swap = colors
    else: print('Error'); return
    output_map, transformed_mask = swap_two_colors(input_map, mask_to_transform, two_colors_to_swap)
    return output_map, transformed_mask

def movt(input_map, mask_to_transform, move_rc,**kwargs):
    
    
    bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_to_transform,input_map)
    newmask = bb_mask.copy()
    newmap = bb_map.copy()
    rows, cols = np.where(newmask==1); vals = newmap[newmask==1]
    for m in range(len(rows)): rows[m] += (tl_rc[0]+move_rc[0]); cols[m] += (tl_rc[1]+move_rc[1])
    transformed_mask = np.zeros_like(mask_to_transform)
    valid = (rows >= 0) & (rows < transformed_mask.shape[0]) & (cols >= 0) & (cols < transformed_mask.shape[1])
    transformed_mask[rows[valid], cols[valid]] = 1 

    output_map = np.zeros_like(input_map) 
    for m in range(len(rows)):
        if valid[m]: output_map[rows[m], cols[m]] = vals[m]

    return output_map, transformed_mask

def flip_about_xaxis(input_map, mask_to_transform, desired_fliprow,**kwargs): 
    

    bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_to_transform,input_map)
    newmask = np.flipud(bb_mask)
    newmap = np.flipud(bb_map)

    bb_height = bb_mask.shape[0]
    if bb_height%2==1: default_fliprow = tl_rc[0] + int(bb_height/2)
    elif bb_height%2==0: default_fliprow = tl_rc[0] + int(bb_height/2) - fractions.Fraction(0.5)
    counteract_rc = ((desired_fliprow-default_fliprow)*2, 0)

    rows, cols = np.where(newmask==1); vals = newmap[newmask==1]
    for m in range(len(rows)): rows[m] += (tl_rc[0]+counteract_rc[0]); cols[m] += (tl_rc[1]+counteract_rc[1]) 
    transformed_mask = np.zeros_like(mask_to_transform)
    valid = (rows >= 0) & (rows < transformed_mask.shape[0]) & (cols >= 0) & (cols < transformed_mask.shape[1])
    transformed_mask[rows[valid], cols[valid]] = 1 

    output_map = np.zeros_like(input_map) 
    for m in range(len(rows)):
        if valid[m]: output_map[rows[m], cols[m]] = vals[m]

    return output_map, transformed_mask

def flip_about_yaxis(input_map, mask_to_transform, desired_flipcol,**kwargs): 
    

    bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_to_transform,input_map)
    newmask = np.fliplr(bb_mask)
    newmap = np.fliplr(bb_map)

    bb_width = bb_mask.shape[1]
    if bb_width%2==1: default_flipcol = tl_rc[1] + int(bb_width/2)
    elif bb_width%2==0: default_flipcol = tl_rc[1] + int(bb_width/2) - fractions.Fraction(0.5)
    counteract_rc = (0, (desired_flipcol-default_flipcol)*2)

    rows, cols = np.where(newmask==1); vals = newmap[newmask==1]
    for m in range(len(rows)): rows[m] += (tl_rc[0]+counteract_rc[0]); cols[m] += (tl_rc[1]+counteract_rc[1]) 
    transformed_mask = np.zeros_like(mask_to_transform)
    valid = (rows >= 0) & (rows < transformed_mask.shape[0]) & (cols >= 0) & (cols < transformed_mask.shape[1])
    transformed_mask[rows[valid], cols[valid]] = 1 

    output_map = np.zeros_like(input_map) 
    for m in range(len(rows)):
        if valid[m]: output_map[rows[m], cols[m]] = vals[m]

    return output_map, transformed_mask

def rotate_about_center(input_map, mask_to_transform, rotation, desired_centre,**kwargs): 
    
    

    bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_to_transform,input_map)
    newmask = np.rot90(bb_mask,rotation)
    newmap = np.rot90(bb_map,rotation)

    
    
    
    
    tl_moves_to = {
        1:(newmask.shape[0]-1,0),
        2:(newmask.shape[0]-1,newmask.shape[1]-1),
        3:(0,newmask.shape[1]-1),
        -1:(0,newmask.shape[1]-1),
        -2:(newmask.shape[0]-1,newmask.shape[1]-1),
        -3:(newmask.shape[0]-1,0)}
    moved_tlrc = tl_moves_to[rotation] 
    delta1 = (desired_centre[0]-tl_rc[0], desired_centre[1]-tl_rc[1]) 
    toflip = {
        1:(-delta1[1],delta1[0]),
        2:(-delta1[0],-delta1[1]),
        3:(delta1[1],-delta1[0]),
        -1:(delta1[1],-delta1[0]),
        -2:(-delta1[0],-delta1[1]),
        -3:(-delta1[1],delta1[0])}
    flipped_delta1 =  toflip[rotation] 
    delta2 = (moved_tlrc[0] +flipped_delta1[0], moved_tlrc[1] + flipped_delta1[1])
    counteract_rc = (desired_centre[0]-delta2[0],desired_centre[1]-delta2[1]) 

    rows, cols = np.where(newmask==1); vals = newmap[newmask==1]
    for m in range(len(rows)): rows[m] += (counteract_rc[0]); cols[m] += (counteract_rc[1])
    transformed_mask = np.zeros_like(mask_to_transform)
    valid = (rows >= 0) & (rows < transformed_mask.shape[0]) & (cols >= 0) & (cols < transformed_mask.shape[1])
    transformed_mask[rows[valid], cols[valid]] = 1 

    output_map = np.zeros_like(input_map) 
    for m in range(len(rows)):
        if valid[m]: output_map[rows[m], cols[m]] = vals[m]

    return output_map, transformed_mask


def flip(input_map, mask_to_transform, flip_about_axis, desired_flip_row_or_col,**kwargs): 
    if flip_about_axis == 'x_axis':
        output_map, transformed_mask = flip_about_xaxis(input_map, mask_to_transform, desired_flip_row_or_col)
    if flip_about_axis == 'y_axis':
        output_map, transformed_mask = flip_about_yaxis(input_map, mask_to_transform, desired_flip_row_or_col)
    return output_map, transformed_mask

def masking(input_map, mask_to_transform,**kwargs):
    return input_map, mask_to_transform



def expand(input_map, mask_to_transform, w_mult, h_mult,**kwargs):

    bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_to_transform, input_map)
    bb_maskmap = np.zeros((bb_mask.shape[0],bb_mask.shape[1],2))
    bb_maskmap[:,:,0] = bb_mask; bb_maskmap[:,:,1] = bb_map

    best_rreduction = w_mult.denominator
    best_creduction = h_mult.denominator
    rmult = w_mult.numerator
    cmult = h_mult.numerator

    reduced_mask = bb_maskmap[::best_rreduction,::best_creduction,0]
    reduced_map = bb_maskmap[::best_rreduction,::best_creduction,1]
    temp_ = np.repeat(reduced_mask, rmult, axis=0)
    reconmask = np.repeat(temp_, cmult, axis=1)
    temp_ = np.repeat(reduced_map, rmult, axis=0)
    reconmap = np.repeat(temp_, cmult, axis=1)
    

    

    
    counteract_rc = (0,0)
    rows, cols = np.where(reconmask==1); vals = reconmap[reconmask==1]
    for m in range(len(rows)): rows[m] += (tl_rc[0]+counteract_rc[0]); cols[m] += (tl_rc[1]+counteract_rc[1]) 
    transformed_mask = np.zeros_like(mask_to_transform)
    valid = (rows >= 0) & (rows < transformed_mask.shape[0]) & (cols >= 0) & (cols < transformed_mask.shape[1])
    transformed_mask[rows[valid], cols[valid]] = 1 
    output_map = np.zeros_like(input_map) 
    for m in range(len(rows)):
        if valid[m]: output_map[rows[m], cols[m]] = vals[m]

    return output_map, transformed_mask


def connection(two_input_maps, two_masks_to_transform, connection_details, **kwargs): 
    
    
    
    
    color_rule_info = connection_details['color_rule_info'] 
    if type(two_input_maps) is not list: print("ERROR")
    
    map1, map2 = two_input_maps
    mask1, mask2 = two_masks_to_transform
    if np.sum(mask1)==0 or np.sum(mask2)==0: print("ERROR")
    output_map = copy.deepcopy(map1)
    transformed_mask = copy.deepcopy(mask1)
    directions = ['S','SW','SE','W','E','N','NW','NE']; region1 = None
    for dir_ in directions:
        maskindir, temp_cdts = mask_in_direction(mask1,dir_)
        if np.sum((mask2==1)&(maskindir==0))==0: 
            region1 = maskindir
            break
    if region1 is None: print("ERROR")
    else:
        oppdir_ = ['S','SW','SE','W','E','N','NW','NE'][['N','NE','NW','E','W','S','SE','SW'].index(dir_)]
        maskindir, temp_cdts = mask_in_direction(mask2,oppdir_)
        region2 = maskindir
    connection_region = region1 & region2 
    
    if 'curr_map_color_of_prev' in color_rule_info:
        
        
        
        chosen_color = get_colors_of_obj(mask1,map1)[0]
    elif 'i_grid_color_of_prev' in color_rule_info:
        chosen_color = get_colors_of_obj(mask1,i_grids[gridn])[0]
    else: 
        if type(color_rule_info[0])==tuple:
            chosen_color = color_rule_info[0][1][0] 
        else: print("ERROR")
    connection_map = np.where(connection_region, chosen_color, 0) 



    transformed_mask = connection_region | mask1 | mask2
    output_map = np.where(mask1, map1, 0) 
    output_map = np.where(mask2, map2, output_map)
    output_map = np.where(connection_region, connection_map, output_map)

    return output_map, transformed_mask




gg=0
def extension(input_map, mask_to_transform, ext_details, **kwargs):
    
    if ext_details is None: return input_map, mask_to_transform
    ext_details_ = ext_details['ext_fn_characterisation']

    if ext_details_ is None: return input_map, mask_to_transform
    A = mask_to_transform
    output_map = copy.deepcopy(input_map)
    transformed_mask = copy.deepcopy(mask_to_transform)

    
    currmask = mask_to_transform
    igrid = i_grids[gridn] 

    mod_i_grid = np.where(mask_to_transform,input_map,igrid)
    iobj_igrid_colors = get_colors_of_obj(currmask, mod_i_grid)
    iobj_currmap_colors = get_colors_of_obj(currmask, input_map)
    i_bordermask = get_outline_border_mask(currmask,1)
    if np.sum(i_bordermask)==0: i_bordercolors = [0]; commonest_i_bordercolor = 0
    else:
        i_bordercolors = list(mod_i_grid[i_bordermask==1]) 
        commonest_i_bordercolor = max(set(i_bordercolors), key=i_bordercolors.count)




    for dirn_dict in ext_details_:
        dir = dirn_dict['dir']
        dir_tuple = [(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)][['S','SW','SE','W','E','N','NW','NE'].index(dir)]
        EXTMODE = dirn_dict['extmode']
        rule_info = dirn_dict['rule_info'] 

        
        
        


        
        mask_in_dir, cdts_of_line_start_and_obj_end = mask_in_direction(currmask,dir) 
        unique_to_verify_dir, anti_mask = get_mask_unique_to_verify_dir(dir,currmask) 
        linestart_cdts_all = [cdt_linestart for cdt_linestart, cdt_edgeend in cdts_of_line_start_and_obj_end]
        
        orthogonalbands = bands_in_dir(currmask.shape, orthogonal_dirn(dir_tuple))
        linestart_cdts_all_bands = [orthogonalbands[cdt] for cdt in linestart_cdts_all]
        ordered_linestart_cdts = [v for _, v in sorted(zip(linestart_cdts_all_bands, linestart_cdts_all))]
        samedirbands = bands_in_dir(currmask.shape, dir_tuple)
        if len(ordered_linestart_cdts) == 0: continue

        if EXTMODE == 'central':
            middle_cdt = ordered_linestart_cdts[len(ordered_linestart_cdts)//2]
            region_mask = np.zeros_like((currmask))
            r,c = middle_cdt 
            for k in range(30):
                if r < 0 or r >= currmask.shape[0] or c < 0 or c >= currmask.shape[1]: break
                else:
                    region_mask[r,c] = 1
                    r += dir_tuple[0]
                    c += dir_tuple[1] 

            initial_region_mask = np.zeros_like((currmask))
            r,c = middle_cdt
            initial_region_mask[r,c] = 1

        elif EXTMODE == 'fullwidth':
            region_mask = np.zeros_like((currmask))
            for n,cdt_linestart in enumerate(ordered_linestart_cdts): 
                r,c = cdt_linestart
                for k in range(30):
                    if r < 0 or r >= currmask.shape[0] or c < 0 or c >= currmask.shape[1]: break
                    else:
                        region_mask[r,c] = 1
                        r += dir_tuple[0]
                        c += dir_tuple[1] 

            initial_region_mask = np.zeros_like((currmask))
            for n,cdt_linestart in enumerate(ordered_linestart_cdts):
                r,c = cdt_linestart
                initial_region_mask[r,c] = 1

        relevant_bands = np.unique(samedirbands[region_mask==1]).astype(int)

        try: bandskip = relevant_bands[1]-relevant_bands[0]
        except: bandskip = 1

    
        n=0; currlen = 0; objn = 0; color = None 
        length = None
        for b in relevant_bands:
            currbandmask = (samedirbands==b).astype(int) 
            curractiveregion = currbandmask & region_mask

        
            
            recalc = False
            if length is not None and currlen == length: 

                objn += 1; recalc = True
            elif n == 0: objn = 0; recalc = True
            
            if recalc:
                if objn > len(rule_info)-1: break
                curr_rule = rule_info[objn] 
                why_this_color = curr_rule[1][0] 
                why_this_len = curr_rule[2][0] 
                
                if type(why_this_color)==tuple and why_this_color[0]=='hyperp_color': color = why_this_color[1] 
                elif type(why_this_color)==str and why_this_color=='curr_map_color_of_prev':
                    
                    prevb = b - bandskip
                    currbandmask = (samedirbands==prevb).astype(int) 
                    if objn == 0: 
                        currmask_activeregion = currbandmask & currmask
                        if np.sum(currmask_activeregion)==0: print("ERROR"); color = None; break
                        else: 
                            rows, cols = np.where(currmask_activeregion==1)
                            midpxl = (rows[len(rows)//2], cols[len(cols)//2])
                            currmap_color = [input_map[midpxl]] 
                            currmap_color = [mod_i_grid[midpxl]]
                            color = currmap_color[0]
                    else:
                        curractiveregion = currbandmask & region_mask
                        igrid_colors = get_colors_of_obj(curractiveregion,mod_i_grid)
                        if len(igrid_colors)!=1: print("ERROR"); color = None; break
                        else: color = igrid_colors[0]                
                elif type(why_this_color)==str and why_this_color=='i_grid_color_of_prev':
                    
                    prevb = b-bandskip
                    currbandmask = (samedirbands==prevb).astype(int) 
                    if objn == 0: 
                        currmask_activeregion = currbandmask & currmask
                        if np.sum(currmask_activeregion)==0: print("ERROR")
                        else: 
                            rows, cols = np.where(currmask_activeregion==1)
                            midpxl = (rows[len(rows)//2], cols[len(cols)//2])
                            igrid_color = [igrid[midpxl]]
                            color = igrid_color[0]
                    else:
                        curractiveregion = currbandmask & region_mask
                        igrid_colors = get_colors_of_obj(curractiveregion,igrid)
                        if len(igrid_colors)!=1: print("ERROR ")
                        else: color = igrid_colors[0]
                else: print("ERROR"); color = None; break
                
                if type(why_this_len)==tuple and why_this_len[0]=='hyperp_length': length = why_this_len[1]
                else: length = None

            if color is None: return output_map, transformed_mask 

            
            if n==0 and type(why_this_len)==tuple and why_this_len[0] == 'encounters_this_specific_hyperp_color':
                nextb = b
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if are_two_identical(nextigrid_colors, why_this_len[1]): break
            elif n==0 and type(why_this_len)==str and why_this_len == 'encounters_a_non_commonestbordercolor':
                nextb = b
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if np.any([colr != commonest_i_bordercolor for colr in nextigrid_colors]): break
            elif n==0 and type(why_this_len)==str and why_this_len == 'encounters_a_non_border_color':
                nextb = b
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if np.any([colr not in i_bordercolors for colr in nextigrid_colors]): break
            elif n==0 and type(why_this_len)==str and why_this_len == 'encounters_an_igrid_color':
                nextb = b
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if np.any([colr in iobj_igrid_colors for colr in nextigrid_colors]): break
            elif n==0 and type(why_this_len)==str and why_this_len == 'encounters_a_currmap_color':
                nextb = b
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if np.any([colr in iobj_currmap_colors for colr in nextigrid_colors]): break


            
            rows,cols = np.where(curractiveregion==1)
            if len(rows)==0: break 
            output_map[rows,cols] = color
            transformed_mask[rows,cols] = 1

            
            if type(why_this_len)==tuple and why_this_len[0] == 'encounters_this_specific_hyperp_color':
                nextb = b + bandskip
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if are_two_identical(nextigrid_colors, why_this_len[1]):

                    length = currlen+1 
            elif type(why_this_len)==str and why_this_len == 'encounters_a_non_commonestbordercolor':
                nextb = b + bandskip
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if np.any([colr != commonest_i_bordercolor for colr in nextigrid_colors]): length = currlen+1
            elif type(why_this_len)==str and why_this_len == 'encounters_a_non_border_color':
                nextb = b + bandskip
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if np.any([colr not in i_bordercolors for colr in nextigrid_colors]): length = currlen+1
            elif type(why_this_len)==str and why_this_len == 'encounters_an_igrid_color':
                nextb = b + bandskip
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if np.any([colr in iobj_igrid_colors for colr in nextigrid_colors]): length = currlen+1
            elif type(why_this_len)==str and why_this_len == 'encounters_a_currmap_color':
                nextb = b + bandskip
                nextbandmask = (samedirbands==nextb).astype(int) 
                nextactiveregion = nextbandmask & region_mask
                nextigrid_colors = get_colors_of_obj(nextactiveregion, mod_i_grid)
                if np.any([colr in iobj_currmap_colors for colr in nextigrid_colors]): length = currlen+1


            currlen += 1
            n+=1




    
    
    
    
    return output_map, transformed_mask





def extension_old(input_map, mask_to_transform, ext_details, **kwargs):
    

    if ext_details is None: return input_map, mask_to_transform
    A = mask_to_transform
    output_map = copy.deepcopy(input_map)
    transformed_mask = copy.deepcopy(mask_to_transform)
    for dirnext in ext_details:
        dir_rc_tuple, ext_type, ext_subtype, ext_new_colors_if_present, ext_vs_terminate_or_occluded, cdt_linestarts = dirnext.values()
        cdts = copy.deepcopy(cdt_linestarts)
        curr_color = None
        for k in range(len(ext_vs_terminate_or_occluded)):
            if ext_new_colors_if_present[k] is not None:
                if ext_new_colors_if_present[k] != curr_color: 
                    curr_color = ext_new_colors_if_present[k] 
            
            for i in range(len(cdts)):
                r,c = cdts[i]
                if r < 0 or r >= A.shape[0] or c < 0 or c >= A.shape[1]: continue 
                else: 
                    output_map[r,c] = curr_color
                    transformed_mask[r,c] = 1
                r += dir_rc_tuple[0]
                c += dir_rc_tuple[1]
                cdts[i] = r,c 
    
    
    return output_map, transformed_mask

def wholegrid_stampings(input_map, mask_to_transform, stamp_details, **kwargs):
    output_map, transformed_mask = input_map, mask_to_transform
    if stamp_details is None: return output_map, transformed_mask

    if 'color_not_to_stamp_around' in stamp_details: 
        color_not_to_stamp_around, color_to_stamp_with, stamp_mode = stamp_details['color_not_to_stamp_around'], stamp_details['color_to_stamp_with'], stamp_details['stamp_mode']
        icolors = get_colors_of_obj(mask_to_transform,input_map)
        icolors.remove(color_not_to_stamp_around)
        color_to_stamp_around = icolors[0]
    else: color_to_stamp_around, color_to_stamp_with, stamp_mode = stamp_details['color_to_stamp_around'], stamp_details['color_to_stamp_with'], stamp_details['stamp_mode']
    
    rows,cols = np.where(input_map==color_to_stamp_around)
    for m in range(len(rows)):
        if mask_to_transform[rows[m],cols[m]]==1:
            for roff,coff in [(1,1),(-1,-1),(1,-1),(-1,1)]:
                r = rows[m]+roff; c = cols[m]+coff
                if r >= 0 and c >= 0 and r <= input_map.shape[0]-1 and c <= input_map.shape[1]-1:
                    if stamp_mode == 'overwrite':
                        output_map[r,c] = color_to_stamp_with
                        transformed_mask[r,c] = 1
                    elif stamp_mode == 'do_not_overwrite':
                        if output_map[r,c] != color_to_stamp_around:
                            output_map[r,c] = color_to_stamp_with
                            transformed_mask[r,c] = 1
    return output_map, transformed_mask


def copying(input_map, mask_to_transform, copy_details, **kwargs):
    if copy_details == [] or copy_details is None: return input_map, mask_to_transform

    output_map = copy.deepcopy(input_map)
    transformed_mask = copy.deepcopy(mask_to_transform)


    prevmap = copy.deepcopy(input_map) 
    prevmask = copy.deepcopy(mask_to_transform) 
    for copydet in copy_details:
        start_dirn = copydet['start_dirn']
        skip = copydet['skip']
        for dirn, issamecolor, colorchange in copydet['record']:
            
            newmap, newmask = movt(prevmap, prevmask, (dirn[0]*skip,dirn[1]*skip))
            if not issamecolor: newmap, newmask = recolor(newmap, newmask, colorchange)

            
            rows, cols = np.where(newmask==1)
            for m in range(len(rows)):
                output_map[rows[m],cols[m]] = newmap[rows[m],cols[m]]
                transformed_mask[rows[m],cols[m]] = 1

            prevmap, prevmask = newmap, newmask 

    
    
    return output_map, transformed_mask

def fill(input_map, mask_to_transform, fill_details,**kwargs):
    output_map, transformed_mask = input_map.copy(), mask_to_transform.copy()
    
    if fill_details is not None:
        
        full_border_fill_color = fill_details[0]['fill_color']
        partial_border_fill_color = fill_details[1]['fill_color']

        
        
        
        flag = True
        
        filled = binary_fill_holes(mask_to_transform)
        holes_mask = np.logical_and(filled, np.logical_not(mask_to_transform)).astype(int)
        if np.sum(holes_mask) == 0: flag = False
        
        labeled_array = get_contiguous_regions(holes_mask,0,diagonal_connections_allowedQ=False,colourblind_spatial_contiguity_mode=False)
        for hole_n in range(1,np.max(labeled_array)+1):
            hole_mask = (labeled_array==hole_n).astype(int)
            
            border_mask = get_outline_border_mask(hole_mask, px_border = 1) & ~holes_mask
            if np.sum(border_mask & ~mask_to_transform) == 0: fill_type = 'full_border' 
            else: fill_type = 'partial_border' 
            
            if fill_type == 'full_border':
                rows,cols = np.where(hole_mask==1)
                transformed_mask[rows,cols] = 1 
                output_map[rows,cols] = full_border_fill_color
            if fill_type == 'partial_border':
                rows,cols = np.where(hole_mask==1)
                transformed_mask[rows,cols] = 1 
                output_map[rows,cols] = partial_border_fill_color

    return output_map, transformed_mask










def gridwise_tiled_copy(input_map, mask_to_transform, tiling_details, **kwargs):
    
    
    

    filtered_options, rmult, cmult = tiling_details['filtered_options'], tiling_details['rmult'], tiling_details['cmult']
    bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_to_transform,input_map)
    ri, ci = bb_mask.shape[0], bb_mask.shape[1]
    a = 0
    output_map = np.zeros((int(ri*rmult),int(ci*cmult)))
    for rn in range(int(rmult)):
        for cn in range(int(cmult)):
            r1,r2,c1,c2 = (rn*ri), ((rn+1)*ri), (cn*ci), ((cn+1)*ci)
            Choose_Ix = 0 
            chosen_serial = filtered_options[a][Choose_Ix]
            curr_map, curr_mask = input_map.copy(), mask_to_transform.copy() 
            for currdict in chosen_serial:
                params=currdict['params']
                if type(currdict['fn']) is str: currdict['fn'] = globals()[currdict['fn']]
                curr_map, curr_mask = currdict['fn'](input_map=curr_map, mask_to_transform=curr_mask,**params) 
            recon_output = curr_map
            output_map[r1:r2,c1:c2] = recon_output
            a+=1
    transformed_mask = np.ones_like(output_map) 
    return output_map, transformed_mask

def combi_transform(input_map, mask_to_transform, combi_details, **kwargs):
    
    
    options = combi_details['options']
    Choose_Ix = 0 
    chosen_serial = options[Choose_Ix]
    curr_map, curr_mask = input_map.copy(), mask_to_transform.copy() 
    for currdict in chosen_serial:
        params=currdict['params']
        if type(currdict['fn']) is str: currdict['fn'] = globals()[currdict['fn']]
        curr_map, curr_mask = currdict['fn'](input_map=curr_map, mask_to_transform=curr_mask,**params) 
    output_map = curr_map
    transformed_mask = np.ones_like(output_map) 
    return output_map, transformed_mask

def gridwise_bool_simpletype(input_map, mask_to_transform, bool_details, **kwargs):

    Choose_Ix = 0
    chosen_dets = bool_details[Choose_Ix]
    bool_type, ibkg_chosen_by, obkg_chosen_by, ocolor, itotalcount_shouldbe = chosen_dets['bool_type'], chosen_dets['ibkg_chosen_by'], chosen_dets['obkg_chosen_by'], chosen_dets['o_color'], chosen_dets['itotalcount']
    
    
    
    
    
    
    
    
    
    ibkg_chosen_by_, obkg_chosen_by_, ocolor_ = ibkg_chosen_by[0], obkg_chosen_by[0], ocolor[0]

    output_map = None
    tempframes = frame_psuedoobjs(input_map) 
    for i_frameobj in tempframes:
        iframetype, itotalcount, iscore, iframecolor,  isubframes = tempframes[i_frameobj].values()
        if itotalcount == itotalcount_shouldbe:
            isubframes
            if isubframes[0]['are_equal_shape'] is False: continue 
            isubframecolors = [list(np.unique(sf['map'])) for sf in isubframes]
            common_icolors = []
            for color in isubframecolors[0]:
                if np.all([color in sfcolors for sfcolors in isubframecolors]): common_icolors.append(color)            
            if ibkg_chosen_by_=='1 common i color' and len(common_icolors) == 1: ibkg = common_icolors[0]
            else: ibkg = ibkg_chosen_by_
            if obkg_chosen_by_=='same as ibkg': obkg = ibkg
            else: obkg = obkg_chosen_by_
            if ocolor_=='i nonbkg color' and len(common_icolors)==2: temp = common_icolors.copy(); temp.remove(ibkg); ocolr = temp[0]
            elif ocolor_=='i nonbkg color' and len(common_icolors)>2: continue 
            else: ocolr = ocolor_
            
            ifeatures = []
            for subframe in isubframes: ifeatures.append(subframe['bb_map'] != ibkg)

            if bool_type == 'AND':         
                and_ = np.logical_and(ifeatures[0],ifeatures[1])
                
                output_map = np.where(and_,ocolr,obkg)
            if bool_type == 'OR':        
                or_ = np.logical_or(ifeatures[0],ifeatures[1])
                output_map = np.where(or_,ocolr,obkg)
            if bool_type == 'NOT_OR':        
                or_ = np.logical_or(ifeatures[0],ifeatures[1])
                notor_ = np.logical_not(or_)
                output_map = np.where(notor_,ocolr,obkg)
            if bool_type == 'NOT_AND':    
                and_ = np.logical_and(ifeatures[0],ifeatures[1])    
                notand_ = np.logical_not(and_)
                output_map = np.where(notand_,ocolr,obkg)
            if bool_type == 'NOT':        
                not_ = np.logical_not(ifeatures[0])
                output_map = np.where(not_,ocolr,obkg)    
            
            break

    transformed_mask = np.ones_like(output_map)
    return output_map, transformed_mask




def get_parsing(parsings_dict, standard_sub_super, parsing_type): 
    listofkeys = []
    for key in parsings_dict:
        if parsings_dict[key]['parsing_type'] in parsing_type:
            listofkeys.append(key)
    return listofkeys


def output_band_pattern(input_map, mask_to_transform, dir_tuple, icolors_where_grids_dont_match, superlist, **kwargs):
    i_grid = input_map
    
    


    samedirbands = bands_in_dir(i_grid.shape, dir_tuple)
    counterdirbands = bands_in_dir(i_grid.shape, orthogonal_dirn(dir_tuple))

    midrows=[]; midcols = [] 
    for b in range(int(np.amax(samedirbands))+1):
        rows,cols = np.where(samedirbands==b)
        midrows.append(rows[len(rows)//2])
        midcols.append(cols[len(cols)//2])

    
    color_store = []
    for b in range(int(np.amax(samedirbands))+1):
        midrow, midcol = midrows[b],midcols[b]
        objmask = np.zeros_like(i_grid)
        for m in range(len(superlist)):
            relr,relc = superlist[m]
            mask1 = (samedirbands==samedirbands[midrow,midcol]+relc).astype(int)
            mask2 = (counterdirbands==counterdirbands[midrow,midcol]+relr).astype(int)
            mask3 = mask1 & mask2
            if np.sum(mask3) == 0: continue
            r_,c_ = np.where(mask3==1)
            r = int(r_[0])
            c = int(c_[0])
            if r >= 0 and c >= 0 and r <= i_grid.shape[0]-1 and c <= i_grid.shape[1]-1:
                objmask[r,c] = 1
                
    
        
    
        
        
        colrs = get_colors_of_obj(objmask, i_grid)
        if len(colrs)==1 and colrs[0] not in icolors_where_grids_dont_match: color_store.append(colrs[0])
        else: color_store.append(None)
    

    
    for repn in range(2,30):
        assigns = []; c=0
        for n in range(len(color_store)):
            if c == repn: c = 0
            assigns.append(c)
            c+=1
        
        dict_ = {n:None for n in range(repn)}; flag = True
        for n in range(len(color_store)):
            colr = color_store[n]
            cval = assigns[n]
            if colr != None:
                if dict_[cval] != None and dict_[cval] != colr: flag = False; break 
                else: dict_[cval] = colr

        if not flag: continue
        if flag: 
            
            vals = [dict_[_] for _ in dict_]
            if None in vals: continue


            
            obj_color_list = []
            for n in range(len(assigns)):
                obj_color_list.append(dict_[assigns[n]])
            break


    recon = np.zeros_like(i_grid)
    color_store = []
    for b in range(int(np.amax(samedirbands))+1):
        midrow, midcol = midrows[b],midcols[b]
        objmask = np.zeros_like(i_grid)
        for m in range(len(superlist)):
            relr,relc = superlist[m]
            mask1 = (samedirbands==samedirbands[midrow,midcol]+relc).astype(int)
            mask2 = (counterdirbands==counterdirbands[midrow,midcol]+relr).astype(int)
            mask3 = mask1 & mask2
            if np.sum(mask3) == 0: continue
            r_,c_ = np.where(mask3==1)
            r = int(r_[0])
            c = int(c_[0])
            if r >= 0 and c >= 0 and r <= i_grid.shape[0]-1 and c <= i_grid.shape[1]-1:
                objmask[r,c] = 1
                recon[r,c] = int(obj_color_list[b])


    return recon, np.ones_like(recon)


def detect_output_band_pattern(i_grid, o_grid):

    mask_where_io_grids_match = (i_grid == o_grid).astype(int)
    icolors_where_grids_dont_match = get_colors_of_obj((mask_where_io_grids_match==0).astype(int), i_grid)
    icolortype = None
    if len(icolors_where_grids_dont_match) == 1: icolortype = "Good, just one occlusion color"; 
    else: icolortype = "WARNING, >1 occlusion color"; 

    

    
    for dir_tuple in [(1,0),(0,1),(1,1),(-1,1)]:
        
        samedirbands = bands_in_dir(o_grid.shape, dir_tuple)
        counterdirbands = bands_in_dir(o_grid.shape, orthogonal_dirn(dir_tuple))
        midrows=[]; midcols = [] 
        for b in range(int(np.amax(samedirbands))+1):
            rows,cols = np.where(samedirbands==b)
            midrows.append(rows[len(rows)//2])
            midcols.append(cols[len(cols)//2])
        
        objmasks = get_contiguous_regions(o_grid,background_color=None,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)
        obj_color_list = []
        obj_relcdts_list = []
        block_objs = []; skipflag = False
        for b in range(int(np.amax(samedirbands))+1):
            midrow, midcol = midrows[b],midcols[b]
            objn = objmasks[midrow,midcol]
            if objn in block_objs: 
                
                skipflag = True; continue
            block_objs.append(objn)
            objmask = (objmasks==objn).astype(int)
            obj_color_list.append(get_colors_of_obj(objmask,o_grid))
            rows,cols = np.where(objmask==1)
            cdts_list = []
            for m in range(len(rows)):
                alongshape_cdt = counterdirbands[rows[m],cols[m]] - counterdirbands[midrow,midcol]
                outshape_cdt = samedirbands[rows[m],cols[m]] - samedirbands[midrow,midcol]
                cdts_list.append((int(alongshape_cdt),int(outshape_cdt)))
            obj_relcdts_list.append(cdts_list)
        
        superlist = []
        for _ in obj_relcdts_list:
            for __ in _:
                if __ not in superlist: superlist.append(__)

        
        
        
        if skipflag: 
            
            continue
        recon = np.zeros_like(o_grid)
        recon_validity = np.zeros_like(o_grid)
        for b in range(int(np.amax(samedirbands))+1):
            midrow, midcol = midrows[b],midcols[b]
            objmask = np.zeros_like(o_grid)
            for m in range(len(superlist)):
                relr,relc = superlist[m]
                mask1 = (samedirbands==samedirbands[midrow,midcol]+relc).astype(int)
                mask2 = (counterdirbands==counterdirbands[midrow,midcol]+relr).astype(int)
                mask3 = mask1 & mask2
                if np.sum(mask3) == 0: continue
                r_,c_ = np.where(mask3==1)
                r = int(r_[0])
                c = int(c_[0])
                if r >= 0 and c >= 0 and r <= o_grid.shape[0]-1 and c <= o_grid.shape[1]-1:
                    objmask[r,c] = 1
                    recon[r,c] = int(obj_color_list[b][0])
                    recon_validity[r,c] += 1
            
        if np.amax(recon_validity) > 1: print("ERROR")
        if 0 in recon_validity: pass 
        
        if are_two_identical(o_grid, recon):
            if 0 in recon_validity: print("ERROR"); continue

            
            return [dir_tuple, icolors_where_grids_dont_match, superlist, obj_color_list]


            break 
    
    return None



def detect_output_symmetries_pattern(input_map, mask_to_transform, **kwargs):

    i_grid = input_map
    
    
    tempscs = np.zeros((i_grid.shape[0], (i_grid.shape[1]-2)*2))
    for r in range(i_grid.shape[0]):
        mirrorcands = []; c1=0
        for mirror in range(1,i_grid.shape[1]-1):
            for pxledge in [0,1]: 
                rhs_normal = i_grid[r,mirror+pxledge:]
                lhs_flipped = i_grid[r,:mirror+1][::-1]
                csame = 0
                for k in range(np.min([len(rhs_normal),len(lhs_flipped)])):
                    if rhs_normal[k]==lhs_flipped[k]: csame+=1
                mirrorcands.append([mirror+(0.5*pxledge), csame])
                tempscs[r,c1] = csame
                c1+=1
        
    scs_across_r = np.median(tempscs,axis=0)
    
    if len(scs_across_r)==0: return input_map.copy(), mask_to_transform.copy()
    best_mirror_col = mirrorcands[np.argmax(scs_across_r)][0]
    
    tempscs = np.zeros((i_grid.shape[1], (i_grid.shape[0]-2)*2))
    for c in range(i_grid.shape[1]):
        mirrorcands = []; c1=0
        for mirror in range(1,i_grid.shape[0]-1):
            for pxledge in [0,1]: 
                bhs_normal = i_grid[mirror+pxledge:,c]
                ths_flipped = i_grid[:mirror+1,c][::-1]
                csame = 0
                for k in range(np.min([len(bhs_normal),len(ths_flipped)])):
                    if bhs_normal[k]==ths_flipped[k]: csame+=1
                mirrorcands.append([mirror+(0.5*pxledge), csame])
                tempscs[c,c1] = csame
                c1+=1
    scs_across_c = np.median(tempscs,axis=0)
    
    if len(scs_across_c)==0: return input_map.copy(), mask_to_transform.copy()
    best_mirror_row = mirrorcands[np.argmax(scs_across_c)][0]
    
    

    

    

    store_corresp = [] 

    blocking1 = np.zeros_like(i_grid) 
    mirror = int(best_mirror_col)
    pxledge = 0 if best_mirror_col == int(best_mirror_col) else 1
    for r in range(i_grid.shape[0]):
        rhs_normal = i_grid[r,mirror+pxledge:]
        lhs_flipped = i_grid[r,:mirror+1][::-1]
        rh_segs = []; lh_segs = []; kss = [] 
        rh_seg = []; lh_seg = []; ks = [] 
        for k in range(np.min([len(rhs_normal),len(lhs_flipped)])):
            if rhs_normal[k]==lhs_flipped[k]: 
                if len(ks)>0: rh_segs.append(rh_seg); lh_segs.append(lh_seg); kss.append(ks)
                rh_seg = []; lh_seg = []; ks = []
            else: rh_seg.append(rhs_normal[k]); lh_seg.append(lhs_flipped[k]); ks.append(k)  
        for s in range(len(kss)):
            if len(kss[s]) > 1: 
                
                rh_bands = np.count_nonzero(np.diff(rh_segs[s]) != 0) + 1
                lh_bands = np.count_nonzero(np.diff(lh_segs[s]) != 0) + 1
                if rh_bands < lh_bands: 
                    for k in kss[s]:
                        blocking1[r,mirror+pxledge+k] = 1
                        store_corresp.append([r, mirror+pxledge+k,   r, mirror-k,   i_grid[r, mirror-k]])
                elif lh_bands < rh_bands: 
                    for k in kss[s]:
                        blocking1[r, mirror-k] = 1
                        store_corresp.append([r, mirror-k,   r,mirror+pxledge+k,   i_grid[r,mirror+pxledge+k]])
    blocking2 = np.zeros_like(i_grid) 
    mirror = int(best_mirror_row)
    pxledge = 0 if best_mirror_row == int(best_mirror_row) else 1
    for c in range(i_grid.shape[1]):
        rhs_normal = i_grid[mirror+pxledge:,c]
        lhs_flipped = i_grid[:mirror+1,c][::-1]
        rh_segs = []; lh_segs = []; kss = [] 
        rh_seg = []; lh_seg = []; ks = [] 
        for k in range(np.min([len(rhs_normal),len(lhs_flipped)])):
            if rhs_normal[k]==lhs_flipped[k]: 
                if len(ks)>0: rh_segs.append(rh_seg); lh_segs.append(lh_seg); kss.append(ks)
                rh_seg = []; lh_seg = []; ks = []
            else: rh_seg.append(rhs_normal[k]); lh_seg.append(lhs_flipped[k]); ks.append(k)
        for s in range(len(kss)):
            if len(kss[s]) > 1: 
                
                rh_bands = np.count_nonzero(np.diff(rh_segs[s]) != 0) + 1
                lh_bands = np.count_nonzero(np.diff(lh_segs[s]) != 0) + 1
                
                if rh_bands < lh_bands: 
                    
                    for k in kss[s]:
                        blocking2[mirror+pxledge+k,c] = 1
                        store_corresp.append([mirror+pxledge+k,c,   mirror-k,c,    i_grid[mirror-k,c]])
                elif lh_bands < rh_bands: 
                    for k in kss[s]:
                        blocking2[mirror-k,c] = 1
                        store_corresp.append([mirror-k,c,   mirror+pxledge+k,c,  i_grid[mirror+pxledge+k,c]])
    # visualise.plotgrid(blocking1)
    # visualise.plotgrid(blocking2)    
    
    blocking = blocking1 | blocking2 
    
    

    
    recon = copy.deepcopy(i_grid)
    rows,cols = np.where(blocking==1)
    for m in range(len(rows)):
        
        
        
        qualifier_colrs = []
        for sc in store_corresp:
            if sc[0]==rows[m] and sc[1]==cols[m]:
                if blocking[sc[2],sc[3]]==0:
                    qualifier_colrs.append(sc[4])
        if not are_all_identical(qualifier_colrs): print('WARNING ')
        recon[rows[m],cols[m]] = qualifier_colrs[0]
    

    return recon, np.ones_like(recon)





def detect_output_symmetries_pattern_bbout(input_map, mask_to_transform, **kwargs):
    map_ , mask_ = detect_output_symmetries_pattern(input_map, mask_to_transform)
    diffmask =  (input_map!=map_).astype(int)
    bb_mask, bb_map, tl_rc = get_bounding_box_object(diffmask, map_)
    return bb_map, bb_mask

def detect_output_tiling_pattern(input_map, mask_to_transform, **kwargs):

    
    
    
    i_grid = input_map 
    cands = []
    for occlusion_color in get_colors_of_obj(np.ones_like(i_grid),i_grid):

        tempscs = np.zeros((i_grid.shape[0],15))
        ixs = [_ for _ in range(i_grid.shape[1])]
        for r in range(i_grid.shape[0]):
            for wid in range(2,15):
                
                pivots = ixs[::wid]
                div = i_grid.shape[1]/wid
                temp = [_ for _ in range(wid)]*int(np.ceil(div+1))
                if len(temp) <= len(ixs): print("Err")
                addresses = temp[:len(ixs)]
                

                
                bkg = occlusion_color 
                buckets = [[] for _ in range(wid)]
                for n in range(len(addresses)):
                    if int(i_grid[r,n]) == bkg: continue 
                    buckets[addresses[n]].append(int(i_grid[r,n]))
                maxbucketlen = max([len(_) for _ in buckets])
                color = []; conf = []
                for buck in buckets:
                    counts = []; correspcolor = []
                    for col in range(10):
                        if col in buck:
                            counts.append(buck.count(col))
                            correspcolor.append(col)
                    if len(counts)==0:
                        color.append(None)
                        conf.append(0)
                    else:
                        ix = np.argmax(counts)
                        color.append(correspcolor[ix])
                        conf.append(counts[ix]/maxbucketlen)
                
                tempscs[r,wid] = np.median(conf) 
        scs_across_r = np.median(tempscs,axis=0) 
        

        tempscs = np.zeros((i_grid.shape[1],15))
        ixs = [_ for _ in range(i_grid.shape[0])]
        for c in range(i_grid.shape[1]):
            for wid in range(2,15):
                pivots = ixs[::wid]
                div = i_grid.shape[0]/wid
                temp = [_ for _ in range(wid)]*int(np.ceil(div+1))
                if len(temp) <= len(ixs): print("Err")
                addresses = temp[:len(ixs)]
                

                
                bkg = occlusion_color 
                buckets = [[] for _ in range(wid)]
                for n in range(len(addresses)):
                    if int(i_grid[n,c]) == bkg: continue 
                    buckets[addresses[n]].append(int(i_grid[n,c]))
                maxbucketlen = max([len(_) for _ in buckets])
                color = []; conf = []
                for buck in buckets:
                    counts = []; correspcolor = []
                    for col in range(10):
                        if col in buck:
                            counts.append(buck.count(col))
                            correspcolor.append(col)
                    if len(counts)==0:
                        color.append(None)
                        conf.append(0)
                    else:
                        ix = np.argmax(counts)
                        color.append(correspcolor[ix])
                        conf.append(counts[ix]/maxbucketlen)
                
                tempscs[c,wid] = np.median(conf) 
        scs_across_c = np.median(tempscs,axis=0) 
        

        

        

        sorter = []
        for cd1 in range(2,15):
            for cd2 in range(2,15):
                sorter.append([scs_across_r[cd1]*scs_across_c[cd2],cd1,cd2])
        sortedcombos = sorted(sorter, key=lambda d: d[0], reverse=True)



        best_res = []; w1s=[];w2s=[]; c0=0

        for _,wid1,wid2 in sortedcombos:
            
    
            ixs = [_ for _ in range(i_grid.shape[1])] 
            pivots = ixs[::wid1]
            div = i_grid.shape[1]/wid1
            temp = [_ for _ in range(wid1)]*int(np.ceil(div+1))
            if len(temp) <= len(ixs): print("Err")
            addresses = temp[:len(ixs)]
            cdts1 = np.tile(addresses,(i_grid.shape[0],1))
            
    
            ixs = [_ for _ in range(i_grid.shape[0])] 
            pivots = ixs[::wid2]
            div = i_grid.shape[0]/wid2
            temp = [_ for _ in range(wid2)]*int(np.ceil(div+1))
            if len(temp) <= len(ixs): print("Err")
            addresses = temp[:len(ixs)]
            cdts2 = np.transpose(np.tile(addresses,(i_grid.shape[1],1)))

            res = {}
            for r in range(i_grid.shape[0]):
                for c in range(i_grid.shape[1]):
                    tupl = (int(cdts1[r,c]),int(cdts2[r,c]))
                    if tupl not in res: res[tupl] = []
                    if int(i_grid[r,c]) == bkg: continue 
                    res[tupl].append(int(i_grid[r,c]))
            chosen = {}
            for re in res:
                if len(res[re])==0: chosen[re] = bkg
                else: chosen[re] = stats.mode(res[re])[0]
            
            recon = np.zeros_like(i_grid)
            for r in range(i_grid.shape[0]):
                for c in range(i_grid.shape[1]):
                    tupl = (int(cdts1[r,c]),int(cdts2[r,c]))
                    
                    recon[r,c] = chosen[tupl]
            c0+=1
            if c0 > 20: break 

            
            if not are_two_identical( np.where((i_grid!=occlusion_color).astype(int), recon, -99), np.where((i_grid!=occlusion_color).astype(int), i_grid, -99) ): continue
            if occlusion_color in recon: continue 

            
            
            cands.append([_, occlusion_color, recon])
            break
    
    if len(cands)>0:
        cands = sorted(cands, key=lambda x: (-x[0]))
        chosen_res = cands[0][2]
        return chosen_res, np.ones_like(chosen_res)
    else: return input_map.copy(), mask_to_transform.copy()

def detect_output_tiling_pattern_bbout(input_map, mask_to_transform, **kwargs):
    map_ , mask_ = detect_output_tiling_pattern(input_map, mask_to_transform)
    diffmask =  (input_map!=map_).astype(int)
    bb_mask, bb_map, tl_rc = get_bounding_box_object(diffmask, map_)
    return bb_map, bb_mask


def detect_output_denoising_pattern(input_map, mask_to_transform, **kwargs):


    
    
    


    i_grid = input_map

    

    

    bkg_color = 0
    
    
    if 0 in input_map: bkg_color = 0
    else: commonest_color = most_freq_val(input_map); bkg_color = commonest_color

    

    tempscs = np.zeros((i_grid.shape[0],15))
    ixs = [_ for _ in range(i_grid.shape[1])]
    for r in range(i_grid.shape[0]):
        for wid in range(2,15):
            
            pivots = ixs[::wid]
            div = i_grid.shape[1]/wid
            temp = [_ for _ in range(wid)]*int(np.ceil(div+1))
            if len(temp) <= len(ixs): print("Err")
            addresses = temp[:len(ixs)]
            

            
            bkg = bkg_color 
            buckets = [[] for _ in range(wid)]
            for n in range(len(addresses)):
                if int(i_grid[r,n]) == bkg: continue 
                buckets[addresses[n]].append(int(i_grid[r,n]))
            maxbucketlen = max([len(_) for _ in buckets])
            color = []; conf = []
            for buck in buckets:
                counts = []; correspcolor = []
                for col in range(10):
                    if col in buck:
                        counts.append(buck.count(col))
                        correspcolor.append(col)
                if len(counts)==0:
                    color.append(None)
                    conf.append(0)
                else:
                    ix = np.argmax(counts)
                    color.append(correspcolor[ix])
                    conf.append(counts[ix]/maxbucketlen)
            
            tempscs[r,wid] = np.median(conf) 
    scs_across_r = np.median(tempscs,axis=0) 
    
    tempscs = np.zeros((i_grid.shape[1],15))
    ixs = [_ for _ in range(i_grid.shape[0])]
    for c in range(i_grid.shape[1]):
        for wid in range(2,15):
            pivots = ixs[::wid]
            div = i_grid.shape[0]/wid
            temp = [_ for _ in range(wid)]*int(np.ceil(div+1))
            if len(temp) <= len(ixs): print("Err")
            addresses = temp[:len(ixs)]
            

            
            bkg = bkg_color 
            buckets = [[] for _ in range(wid)]
            for n in range(len(addresses)):
                if int(i_grid[n,c]) == bkg: continue 
                buckets[addresses[n]].append(int(i_grid[n,c]))
            maxbucketlen = max([len(_) for _ in buckets])
            color = []; conf = []
            for buck in buckets:
                counts = []; correspcolor = []
                for col in range(10):
                    if col in buck:
                        counts.append(buck.count(col))
                        correspcolor.append(col)
                if len(counts)==0:
                    color.append(None)
                    conf.append(0)
                else:
                    ix = np.argmax(counts)
                    color.append(correspcolor[ix])
                    conf.append(counts[ix]/maxbucketlen)
            
            tempscs[c,wid] = np.median(conf) 
    scs_across_c = np.median(tempscs,axis=0) 
    

    

    

    sorter = []
    for cd1 in range(2,15):
        for cd2 in range(2,15):
            sorter.append([scs_across_r[cd1]*scs_across_c[cd2],cd1,cd2])
    sortedcombos = sorted(sorter, key=lambda d: d[0], reverse=True)



    best_res = []; w1s=[];w2s=[]

    for _,wid1,wid2 in sortedcombos:
        

        ixs = [_ for _ in range(i_grid.shape[1])] 
        pivots = ixs[::wid1]
        div = i_grid.shape[1]/wid1
        temp = [_ for _ in range(wid1)]*int(np.ceil(div+1))
        if len(temp) <= len(ixs): print("Err")
        addresses = temp[:len(ixs)]
        cdts1 = np.tile(addresses,(i_grid.shape[0],1))
        

        ixs = [_ for _ in range(i_grid.shape[0])] 
        pivots = ixs[::wid2]
        div = i_grid.shape[0]/wid2
        temp = [_ for _ in range(wid2)]*int(np.ceil(div+1))
        if len(temp) <= len(ixs): print("Err")
        addresses = temp[:len(ixs)]
        cdts2 = np.transpose(np.tile(addresses,(i_grid.shape[1],1)))

        res = {}
        for r in range(i_grid.shape[0]):
            for c in range(i_grid.shape[1]):
                tupl = (int(cdts1[r,c]),int(cdts2[r,c]))
                if tupl not in res: res[tupl] = []
                if int(i_grid[r,c]) == bkg: continue 
                res[tupl].append(int(i_grid[r,c]))
        chosen = {}
        for re in res:
            if len(res[re])==0: chosen[re] = bkg
            else: chosen[re] = stats.mode(res[re])[0]
        
        recon = np.zeros_like(i_grid)
        for r in range(i_grid.shape[0]):
            for c in range(i_grid.shape[1]):
                tupl = (int(cdts1[r,c]),int(cdts2[r,c]))
                if i_grid[r,c] == bkg_color: continue
                recon[r,c] = chosen[tupl]
        

        
        
        
        
        
        

        
        

        tobkg = np.ones_like(i_grid)
        bkgthresh = 0.5
        rats1=[]
        for r in range(i_grid.shape[0]):
            rat = list(i_grid[r,:]).count(bkg_color)/i_grid.shape[1]
            if rat > 0.5: tobkg[r,:] = 0
            rats1.append(rat)
        rats2=[]
        for c in range(i_grid.shape[1]):
            rat = list(i_grid[:,c]).count(bkg_color)/i_grid.shape[0]
            if rat > 0.5: tobkg[:,c] = 0
            rats2.append(rat)
        
        
        newrecon = np.where(tobkg,recon,bkg)
        








        best_res.append(newrecon);w1s.append(wid1);w2s.append(wid2) 

    
    
    
    commonest_color = most_freq_val(best_res[0])
    
    
    

    

    
    filt_res = []
    for c0, res in enumerate(best_res):
        mask_of_nonbkg = (res!=commonest_color).astype(int)
        i_objs = get_contiguous_regions(mask_of_nonbkg,0,True,False)
        checker = None; flag = True
        for k in range(1,np.max(i_objs)+1):
            currmask = (i_objs==k).astype(int)
            if np.sum(currmask)==0: flag = False; break
            else: bb_mask, bb_map, tl_rc = get_bounding_box_object(currmask, res)
            if checker is None: 
                checker = bb_map
            else:
                if not are_two_identical(bb_map, checker):
                    flag = False
                    break
        if flag: 
            i_grid_maskvals = i_grid[mask_of_nonbkg==1]
            r_grid_maskvals = res[mask_of_nonbkg==1]
            cdelta = 0
            for m in range(len(i_grid_maskvals)):
                if i_grid_maskvals[m] == r_grid_maskvals[m]: cdelta += 1
            filt_res.append([cdelta, w1s[c0], w2s[c0], res])

    if len(filt_res)>0:
        filt_res = sorted(filt_res, key=lambda x: (-x[0]))
        chosen_res = filt_res[0][3]
        return chosen_res, np.ones_like(chosen_res)
    else: return input_map.copy(), mask_to_transform.copy()



def detect_tiledcopy_of_fullinputgrid(iobj, i_mask, i_map, o_grid, gridn): 
    

    

    bb_mask, bb_map, tl_rc = get_bounding_box_object(i_mask,i_map) 
    is_rect = True if np.sum(bb_mask) == np.prod(bb_mask.shape) else False
    if not is_rect: return None

    rmult, cmult = o_grid.shape[0]/bb_mask.shape[0], o_grid.shape[1]/bb_mask.shape[1] 
    divisible = True if (rmult == int(rmult) and cmult == int(cmult)) else False
    if not divisible: return None
    
    sub_region_n = 0 
    region_masks = []; output_mask = np.zeros_like(o_grid)
    region_bbmaps = []
    ri, ci = bb_mask.shape[0], bb_mask.shape[1]
    for rn in range(int(rmult)):
        for cn in range(int(cmult)):
            r1,r2,c1,c2 = (rn*ri), ((rn+1)*ri), (cn*ci), ((cn+1)*ci)
            curr_region_mask = np.zeros_like(o_grid)
            curr_region_mask[r1:r2,c1:c2] = 1
            output_mask[r1:r2,c1:c2] = 1
            region_masks.append(curr_region_mask)
            region_bbmaps.append(o_grid[r1:r2,c1:c2])
            sub_region_n+=1

    i_map_colors = list(np.unique(i_map[i_mask==1])) 
    desired_fliprow = bb_mask.shape[0]/2-.5 + tl_rc[0]
    desired_flipcol = bb_mask.shape[1]/2-0.5 + tl_rc[1]
    desired_centre = (desired_fliprow,desired_flipcol)
    is_square = True if bb_mask.shape[0] == bb_mask.shape[1] else False

    o_grid_colors = list(np.unique(o_grid))
    
    
    save_all_options = [[] for _ in range(sub_region_n)]

    main_chains = []
    main_chains.append([{'fn':no_change,'params':{}}])
    main_chains.append([{'fn':flip,'params':{}}])
    main_chains.append([{'fn':full_color,'params':{}}])
    main_chains.append([{'fn':rotate_about_center,'params':{}}])
    if len(i_map_colors) == 2: main_chains.append([{'fn':bool_not,'params':{}}]) 

    main_chains.append([{'fn':rotate_about_center,'params':{}},{'fn':flip,'params':{}}])
    main_chains.append([{'fn':flip,'params':{}},{'fn':rotate_about_center,'params':{}}]) 
    
    main_chains.append([{'fn':no_change,'params':{}},{'fn':masking,'params':{}}])
    

    unique_emask_paramlists = []

    for chain in main_chains:
        chain_branchlist = [chain.copy()]

        a0 = 0
        while a0 < len(chain_branchlist):
            chain_branch = chain_branchlist[a0]

            curr_map, curr_mask = i_map.copy(), i_mask.copy() 

            curr_o_masko = np.zeros_like(curr_mask) 

            c0=0

            for fn_dict in chain_branch:
                fn_name = fn_dict['fn'].__name__

                

                def detect_recolor(curr_map, curr_mask, region_bbmaps):
                    

                    icolors = list(np.unique(curr_map[curr_mask==1]))

                    for nn in range(sub_region_n):
                        ocolors = list(np.unique(region_bbmaps[nn]))

                    opts = []

                    
                    opts.append([(int(ic),int(ic)) for ic in icolors])
                    
                    
                    for ic in icolors:
                        for oc in ocolors:
                            if [(int(ic),int(oc))] not in opts: opts.append([(int(ic),int(oc))])

                    
                    

                    
                    res = [list(zip(icolors, p)) for p in product(ocolors, repeat=len(icolors))] 
                    for re in res: 
                        if re not in opts: opts.append(re)

                    return opts

                
                def detect_masking(curr_map, curr_mask, region_bbmaps):
                
                    opt = []

                    for nn in range(sub_region_n):
                        masking_mask = np.zeros_like(region_bbmaps[nn])
                        rows,cols = np.where(curr_mask==1)
                        for m in range(len(rows)):
                            if curr_map[rows[m],cols[m]] != region_bbmaps[nn][rows[m],cols[m]]:
                                masking_mask[rows[m],cols[m]] = 1
                        
                        flag = False
                        for _ in opt:
                            if are_two_identical(masking_mask,_): flag = True; break
                        if not flag and np.sum(masking_mask)>0: opt.append(masking_mask)
                    
                    return opt


                

                tempflag = True if chain_branchlist[a0][c0]['params'] == {} else False 
                

                params_list = [{}]
                if tempflag: 
                    if fn_name == 'no_change': params_list = [{}]
                    if fn_name == 'full_color':
                        opts = []
                        
                        for color in list(set(i_map_colors+o_grid_colors)): opts.append(color)
                        params_list = [{'color':cc} for cc in opts]
                        
                    if fn_name == 'bool_not':
                        
                        params_list = [{}]
                        
                    if fn_name == 'recolor': 
                        recolor_opts = detect_recolor(curr_map, curr_mask, region_bbmaps)
                        params_list = [{'color_changes':cc} for cc in recolor_opts]
                        
                    if fn_name == 'masking':
                        masking_opts = detect_masking(curr_map, curr_mask, region_bbmaps)
                        if masking_opts != []: 
                            params_list = [{'mask':masking_opt} for masking_opt in masking_opts]
                            
                        else: print("WARNING")
                    if fn_name == 'flip':
                        flip_opts = [['x_axis',desired_fliprow],['y_axis',desired_flipcol]]
                        params_list = [{'flip_about_axis':flipaboutaxis,'desired_flip_row_or_col':flipaboutroworcol} for flipaboutaxis, flipaboutroworcol in flip_opts]
                    if fn_name == 'rotate_about_center':
                        if is_square: rotate_opts = [[rotation, desired_centre] for rotation in [1,-1,2,-2,3,-3]]
                        else: rotate_opts = [[rotation, desired_centre] for rotation in [2,-2]]
                        params_list = [{'rotation':rotation,'desired_centre':desiredcentre} for rotation, desiredcentre in rotate_opts]


                curr_params = chain_branchlist[a0][c0]['params']
                if (curr_params not in params_list) and tempflag:
                    prefix = chain_branchlist[a0][:c0]
                    newlist = []
                    for chain_ in chain_branchlist:
                        if chain_[:c0] == prefix:
                            for new_params in params_list:
                                
                                tempmodified = list(chain_)  
                                
                                step_copy = dict(tempmodified[c0])  
                                step_copy['params'] = new_params
                                tempmodified[c0] = step_copy
                                newlist.append(tempmodified)
                        else: newlist.append(list(chain_))  
                    chain_branchlist = newlist
                
                currdict = (chain_branchlist[a0][c0])
                params = currdict.get('params', {})
                
                incache = False
                cache_paramlist = chain_branchlist[a0][:c0+1]
                curr_cache = [_[0] for _ in unique_emask_paramlists]
                for cc in range(len(curr_cache)):
                    
                    if are_two_identical(curr_cache[cc],cache_paramlist): incache = cc; break
                if incache is not False: 
                    curr_map, curr_mask = unique_emask_paramlists[incache][1],unique_emask_paramlists[incache][2]
                else:
                    curr_map, curr_mask = currdict['fn'](input_map=curr_map, mask_to_transform=curr_mask,**params) 
                    unique_emask_paramlists.append([cache_paramlist,curr_map,curr_mask]) 
                
                
                if currdict['fn'].__name__ == 'masking': 
                    masking_mask = params['mask']
                    curr_o_masko = curr_o_masko | masking_mask 


                quitflag = False

                c0+=1

            
            

            current_run = chain_branchlist[a0][:c0+1]
            a0+=1

            for nn in range(sub_region_n):
                rows,cols = np.where(curr_mask==1) 
                explained_mask = copy.deepcopy(curr_o_masko) 
                
                
                
                for m in range(len(rows)):
                    if curr_map[rows[m],cols[m]] == region_bbmaps[nn][rows[m],cols[m]]:
                        explained_mask[rows[m],cols[m]] = 1 
                if (explained_mask==1).all(): save_all_options[nn].append(current_run)

            

    global ttt
    ttt = save_all_options



    filtered_options = [[] for _ in range(sub_region_n)]; minimalist_o_masks = []
    for n in range(len(ttt)):
        

        
        

        
        if len(ttt[n]) == 0: minimalist_o_masks.append(None); continue

        masklevels = []; corresp_masks = []
        for option in ttt[n]:
            if option[-1]['fn'].__name__ == 'masking':
                masklevel = np.sum(option[-1]['params']['mask'])
                masklevels.append(masklevel); corresp_masks.append(option[-1]['params']['mask'])
            else: masklevels.append(0); corresp_masks.append(np.zeros_like(curr_mask))
        lowest_masklevel = np.min(masklevels)
        corresp_mask = corresp_masks[masklevels.index(lowest_masklevel)]
        minimalist_o_masks.append(corresp_mask)
        
        for option in ttt[n]:
            if option[-1]['fn'].__name__ == 'masking':
                masklevel = np.sum(option[-1]['params']['mask'])
                if masklevel == lowest_masklevel: 
                    
                    filtered_options[n].append(option)
            else: 
                
                filtered_options[n].append(option)

    sub_region_n = 0 
    occlusion_mask = np.zeros_like(o_grid)
    ri, ci = bb_mask.shape[0], bb_mask.shape[1]
    for rn in range(int(rmult)):
        for cn in range(int(cmult)):
            r1,r2,c1,c2 = (rn*ri), ((rn+1)*ri), (cn*ci), ((cn+1)*ci)
            if minimalist_o_masks[sub_region_n] is not None:
                occlusion_mask[r1:r2,c1:c2] = minimalist_o_masks[sub_region_n]
            sub_region_n+=1
    
    

    if len(save_all_options)==0: return None
    if len(filtered_options)==0: return None


    transform_to_save = {'filtered_options':filtered_options,'rmult':rmult,'cmult':cmult,'occlusion_mask':occlusion_mask}
    
    
    
    
    
    
    

    return transform_to_save

def detect_gridwise_bools(iframetype, itotalcount, iscore, isubframes, oframetype, ototalcount, oscore, osubframes):
    
    
    
    if isubframes[0]['are_equal_shape'] is False or osubframes[0]['are_equal_shape'] is False: return 
    ibb_mask, ibb_map, itl_rc = get_bounding_box_object(isubframes[0]['mask'], isubframes[0]['map'])
    obb_mask, obb_map, otl_rc = get_bounding_box_object(osubframes[0]['mask'], osubframes[0]['map'])
    if ibb_mask.shape != obb_mask.shape: return
    

    

    if itotalcount == 2 and ototalcount == 1:
        pass
    elif itotalcount > 2 and ototalcount == 1:
        pass
    elif itotalcount == 1 and ototalcount == 1:
        pass

    
    
    isubframecolors = [list(np.unique(sf['map'])) for sf in isubframes]
    osubframecolors = [list(np.unique(sf['map'])) for sf in osubframes] 

    common_icolors = []
    for color in isubframecolors[0]:
        if np.all([color in sfcolors for sfcolors in isubframecolors]): common_icolors.append(color)
    common_ocolors = []
    for color in osubframecolors[0]:
        if np.all([color in sfcolors for sfcolors in osubframecolors]): common_ocolors.append(color)


    
    
    
    all_o_colors = osubframecolors[0] 
    if len(all_o_colors)!=2: 
        print("Unsupported")
        return
    noncommon_i_colors = []
    for subli in isubframecolors:
        for color_ in subli:
            if color_ not in common_icolors:
                noncommon_i_colors.append(color_)

    
    if len(common_icolors) == 1: bkg = common_icolors[0]; bkg_type = '1 common i color' 
    else: 
        matchcolors = []
        for color in common_icolors:
            if color in common_ocolors: matchcolors.append(color)
        if len(matchcolors) == 1: bkg = matchcolors[0]; bkg_type = '1 i color present in o'
        else: 
            if 0 in common_icolors: bkg = 0; bkg_type = 'black' 
            else: bkg = None; bkg_type = None 
    
    if bkg is not None: ordered_bkg_cands = [bkg]
    else: ordered_bkg_cands = [] 
    for color in common_icolors:
        if color not in ordered_bkg_cands: ordered_bkg_cands.append(color)
    ordered_obkg_cands = []
    if bkg is not None and bkg in common_ocolors: ordered_obkg_cands.append(bkg)
    for color in common_ocolors:
        if color not in ordered_obkg_cands: ordered_obkg_cands.append(color)
    

    transform_to_save = []
    for ibkg in ordered_bkg_cands:
        for obkg in ordered_obkg_cands:
            
            if ibkg == bkg: ibkg_chosen_by = bkg_type
            else: ibkg_chosen_by = 'unknown'
            if obkg == bkg: obkg_chosen_by = 'same as ibkg'
            else: obkg_chosen_by = 'unknown'

            
            ifeatures = []
            for subframe in isubframes: ifeatures.append(subframe['bb_map'] != ibkg) 
            ofeature = osubframes[0]['bb_map'] != obkg

            
            temp = all_o_colors.copy(); temp.remove(obkg); o_color_ = temp[0]; o_color = [o_color_]
            if len(common_icolors)==2 and len(noncommon_i_colors) == 0:
                tempi = common_icolors.copy(); tempi.remove(ibkg); i_color = tempi[0] 
                if i_color == o_color_: o_color = ['i nonbkg color',o_color_] 

            if itotalcount == 2 and ototalcount == 1: 
                and_ = np.logical_and(ifeatures[0],ifeatures[1])
                or_ = np.logical_or(ifeatures[0],ifeatures[1])
                if are_two_identical(and_, ofeature): 
                    
                    transform_to_save.append({'bool_type':'AND','ibkg_chosen_by':[ibkg_chosen_by,ibkg] if ibkg_chosen_by=='1 common i color' else [ibkg], 'obkg_chosen_by':[obkg_chosen_by] if obkg_chosen_by=='same as ibkg' else [obkg],'o_color':o_color,'itotalcount':itotalcount})
                if are_two_identical(or_, ofeature): 

                    transform_to_save.append({'bool_type':'OR','ibkg_chosen_by':[ibkg_chosen_by,ibkg] if ibkg_chosen_by=='1 common i color' else [ibkg], 'obkg_chosen_by':[obkg_chosen_by] if obkg_chosen_by=='same as ibkg' else [obkg],'o_color':o_color,'itotalcount':itotalcount})
                
                
                notor_ = np.logical_not(or_)
                notand_ = np.logical_not(and_) 
                if are_two_identical(notor_, ofeature): 

                    transform_to_save.append({'bool_type':'NOT_OR','ibkg_chosen_by':[ibkg_chosen_by,ibkg] if ibkg_chosen_by=='1 common i color' else [ibkg], 'obkg_chosen_by':[obkg_chosen_by] if obkg_chosen_by=='same as ibkg' else [obkg],'o_color':o_color,'itotalcount':itotalcount})
                if are_two_identical(notand_, ofeature): 

                    transform_to_save.append({'bool_type':'NOT_AND','ibkg_chosen_by':[ibkg_chosen_by,ibkg] if ibkg_chosen_by=='1 common i color' else [ibkg], 'obkg_chosen_by':[obkg_chosen_by] if obkg_chosen_by=='same as ibkg' else [obkg],'o_color':o_color,'itotalcount':itotalcount})
            if itotalcount == 1 and ototalcount == 1:
                not_ = np.logical_not(ifeatures[0])
                if are_two_identical(not_, ofeature): 

                    transform_to_save.append({'bool_type':'NOT','ibkg_chosen_by':[ibkg], 'obkg_chosen_by':[obkg_chosen_by] if obkg_chosen_by=='same as ibkg' else [obkg],'o_color':o_color,'itotalcount':itotalcount})

    


    
    return transform_to_save

def bands_in_dir(shape,direction): 
    H, W = shape
    dir_vec = np.array(direction, dtype=float)
    dir_vec /= np.linalg.norm(dir_vec)

    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  

    
    projections = yy * dir_vec[0] + xx * dir_vec[1]
    proj = projections.copy()
    nu = list(np.unique(np.round(projections,5))) 
    for r in range(proj.shape[0]):
        for c in range(proj.shape[1]):
            proj[r,c] = nu.index(np.round(projections[r,c],5))
    return proj

def orthogonal_dirn(dirntuple): 
    return (-dirntuple[1],dirntuple[0])

def mask_in_direction(mask, direction):
    

    new_mask = np.zeros_like(mask)
    rows, cols = mask.shape

    
    cdts = [] 

    if direction == 'E':
        for row_idx in range(rows):
            cols_with_shape = np.where(mask[row_idx] == 1)[0]
            if cols_with_shape.size > 0:
                rightmost_col = cols_with_shape[-1]
                new_mask[row_idx, rightmost_col + 1:] = 1 

                
                if rightmost_col+1 < cols: cdts.append([(row_idx, int(rightmost_col+1)), (row_idx,int(rightmost_col))]) 


    elif direction == 'W':
        for row_idx in range(rows):
            cols_with_shape = np.where(mask[row_idx] == 1)[0]
            if cols_with_shape.size > 0:
                leftmost_col = cols_with_shape[0]
                new_mask[row_idx, :leftmost_col] = 1

                
                if leftmost_col-1 >= 0: cdts.append([(row_idx, int(leftmost_col-1)), (row_idx,int(leftmost_col))])

    elif direction == 'S':
        for col_idx in range(cols):
            rows_with_shape = np.where(mask[:, col_idx] == 1)[0]
            if rows_with_shape.size > 0:
                bottommost_row = rows_with_shape[-1]
                new_mask[bottommost_row + 1:, col_idx] = 1

                
                if bottommost_row+1 < rows: cdts.append([(int(bottommost_row + 1),col_idx), (int(bottommost_row),col_idx)])

    elif direction == 'N':
        for col_idx in range(cols):
            rows_with_shape = np.where(mask[:, col_idx] == 1)[0]
            if rows_with_shape.size > 0:
                topmost_row = rows_with_shape[0]
                new_mask[:topmost_row, col_idx] = 1

                
                if topmost_row-1 >= 0: cdts.append([(int(topmost_row-1),col_idx), ((int(topmost_row),col_idx))])

    elif direction == 'SE':
        for row_idx in range(rows - 1):  
            for col_idx in range(cols - 1):  
                if mask[row_idx, col_idx] == 1: 
                    r, c = row_idx + 1, col_idx + 1 
                    if mask[r, c] == 0: cdts.append([(r,c), (row_idx, col_idx)])
                    while r < rows and c < cols:
                        if mask[r, c] == 0:  
                            new_mask[r, c] = 1 
                        r += 1
                        c += 1

    elif direction == 'SW':
        for row_idx in range(rows - 1):  
            for col_idx in range(1, cols):  
                if mask[row_idx, col_idx] == 1:
                    r, c = row_idx + 1, col_idx - 1
                    if mask[r, c] == 0: cdts.append([(r,c), (row_idx, col_idx)])
                    while r < rows and c >= 0:
                        if mask[r, c] == 0:  
                            new_mask[r, c] = 1
                        r += 1
                        c -= 1

    elif direction == 'NE':
        for row_idx in range(1, rows):  
            for col_idx in range(cols - 1):  
                if mask[row_idx, col_idx] == 1:
                    r, c = row_idx - 1, col_idx + 1
                    
                    if mask[r, c] == 0: cdts.append([(r,c), (row_idx, col_idx)])
                    while r >= 0 and c < cols:
                        if mask[r, c] == 0:  
                            new_mask[r, c] = 1
                        r -= 1
                        c += 1

    elif direction == 'NW':
        for row_idx in range(1, rows):  
            for col_idx in range(1, cols):  
                if mask[row_idx, col_idx] == 1:
                    r, c = row_idx - 1, col_idx - 1
                    if mask[r, c] == 0: cdts.append([(r,c), (row_idx, col_idx)])
                    while r >= 0 and c >= 0:
                        if mask[r, c] == 0:  
                            new_mask[r, c] = 1
                        r -= 1
                        c -= 1

    
    return new_mask, cdts

def get_mask_unique_to_verify_dir(verify_this_dir,i_mask):
    anti_mask = np.zeros_like(i_mask)
    directions = ['S','SW','SE','W','E','N','NW','NE']
    
    for di in directions:
        if di != verify_this_dir:
            maskindir, temp_cdts = mask_in_direction(i_mask,di)
            anti_mask+= maskindir
    anti_mask = (anti_mask!=0).astype(int)

    this_dir = anti_mask
    verify_dir, temp_cdts = mask_in_direction(i_mask,verify_this_dir)
    unique_to_verify_dir = verify_dir & ~this_dir
    return unique_to_verify_dir, anti_mask

def get_border_cdts(cdt_linestarts,dir):
    cdt_borders = []
    if dir in ['E','W']: 
        ix = np.argmin([r for r,c in cdt_linestarts]) 
        top_border = (cdt_linestarts[ix][0] - 1, cdt_linestarts[ix][1]) 
        ix = np.argmax([r for r,c in cdt_linestarts])
        bottom_border = (cdt_linestarts[ix][0] + 1, cdt_linestarts[ix][1])
        cdt_borders = [top_border, bottom_border]
    if dir in ['N','S']: 
        ix = np.argmin([c for r,c in cdt_linestarts]) 
        left_border = (cdt_linestarts[ix][0], cdt_linestarts[ix][1] - 1)
        ix = np.argmax([c for r,c in cdt_linestarts])
        right_border = (cdt_linestarts[ix][0], cdt_linestarts[ix][1] + 1)
        cdt_borders = [left_border, right_border]
    if dir in ['NE']: 
        ix = np.argmin([c for r,c in cdt_linestarts])
        left_border = (cdt_linestarts[ix][0], cdt_linestarts[ix][1] - 1)
        ix = np.argmax([r for r,c in cdt_linestarts])
        bottom_border = (cdt_linestarts[ix][0] + 1, cdt_linestarts[ix][1])
        cdt_borders = [left_border, bottom_border]
    if dir in ['SE']: 
        ix = np.argmin([c for r,c in cdt_linestarts])
        left_border = (cdt_linestarts[ix][0], cdt_linestarts[ix][1] - 1)
        ix = np.argmin([r for r,c in cdt_linestarts])
        top_border = (cdt_linestarts[ix][0] - 1, cdt_linestarts[ix][1])
        cdt_borders = [left_border, top_border]
    if dir in ['NW']: 
        ix = np.argmax([c for r,c in cdt_linestarts])
        right_border = (cdt_linestarts[ix][0], cdt_linestarts[ix][1] + 1)
        ix = np.argmax([r for r,c in cdt_linestarts])
        bottom_border = (cdt_linestarts[ix][0] + 1, cdt_linestarts[ix][1])        
        cdt_borders = [right_border, bottom_border] 
    if dir in ['SW']: 
        ix = np.argmax([c for r,c in cdt_linestarts])
        right_border = (cdt_linestarts[ix][0], cdt_linestarts[ix][1] + 1)
        ix = np.argmin([r for r,c in cdt_linestarts])
        top_border = (cdt_linestarts[ix][0] - 1, cdt_linestarts[ix][1])
        cdt_borders = [right_border, top_border] 
    return cdt_borders

def remove_supersets(list_of_lists): 
    sets = [set(lst) for lst in list_of_lists]
    keep = []

    for i, s in enumerate(sets):
        if not any(s > other for j, other in enumerate(sets) if i != j):
            keep.append(list_of_lists[i])
    return keep

def safe_assign_rc(rows,cols,array,value,allow_spillover=False):
    if len(rows)!=len(cols): 
        
        return None
    nrows,ncols = array.shape
    
    if np.min(rows) >= 0 and np.min(cols) >= 0 and np.max(rows) < nrows and np.max(cols) < ncols:
        array[rows,cols] = value 
        return array
    else:
        if not allow_spillover: 
            
            return None
        else:
            for m in range(len(rows)):
                if rows[m] >=0 and cols[m] >=0 and rows[m] < nrows and cols[m] < ncols:
                    array[rows[m],cols[m]] = value
            return array
    

def rectangular_border_mask(shape, corner1, corner2): 

    r1, c1 = corner1
    r2, c2 = corner2

    rmin, rmax = sorted((r1, r2))
    cmin, cmax = sorted((c1, c2))

    mask = np.zeros(shape, dtype=bool)

    mask[rmin, cmin:cmax+1] = 1
    mask[rmax, cmin:cmax+1] = 1
    mask[rmin:rmax+1, cmin] = 1
    mask[rmin:rmax+1, cmax] = 1

    return mask.astype(int)

def grid_border_mask(grid):
    return rectangular_border_mask(grid.shape, (0,0),(grid.shape[0]-1,grid.shape[1]-1))




############################################################################################

with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r', encoding='utf-8') as file:
    test_challenges_ = json.load(file)
test_strings_ = [key for key in test_challenges_] # name such as 'ff805c23'
output_data_ = {}
for test_string_ in test_strings_:
    curr_outputs = []
    for pair_ in test_challenges_[test_string_]['test']:
        curr_outputs.append({"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]})
    output_data_[test_string_] = curr_outputs



for problemN in range(len(test_strings_)): 
    # We need a test_string_
    test_string_ = test_strings_[problemN]
    #print('-----------------------------------------------------ppppp',problemN,test_string_)

    i_grids = []; o_grids = []
    cp1=0
    for pair_ in test_challenges_[test_string_]['train']:
        input_grid_, output_grid_ = pair_['input'], pair_['output']
        i_grids.append(np.array(input_grid_)); o_grids.append(np.array(output_grid_)); cp1+=1
    cp2=0
    for pair_ in test_challenges_[test_string_]['test']:
        input_grid_ = pair_['input']
        i_grids.append(np.array(input_grid_)); cp2+=1
    num_demo_grids, num_test_grids = cp1, cp2


    currtime_script = time.perf_counter_ns()
    if ((currtime_script-starttime_script)/1000000000) > 41400: print('SCRIPT TIMEOUT, save before too late'); break 


    starttime_main = time.perf_counter_ns()
    def esc(): 
        QUIT_SECS = 143
        currtime_main = time.perf_counter_ns()
        if ((currtime_main-starttime_main)/1000000000) > QUIT_SECS: print('Timeout quit'); return True
        return False
    def esc1(): # This skips to analogy creation with whatever transform_res we have at that point
        QUIT_SECS = 80
        currtime_main = time.perf_counter_ns()
        if ((currtime_main-starttime_main)/1000000000) > QUIT_SECS: print('Timeout transform_res accumulation'); return True
        return False    
    def esc2(): # This skips to rule finding with whatever analogy groups we hate at that point
        QUIT_SECS = 100
        currtime_main = time.perf_counter_ns()
        if ((currtime_main-starttime_main)/1000000000) > QUIT_SECS: print('Timeout analogy accumulation'); return True
        return False        
    def esc01(): # This escapes rule finding and moves to reversal
        QUIT_SECS = 120
        currtime_main = time.perf_counter_ns()
        if ((currtime_main-starttime_main)/1000000000) > QUIT_SECS: print('Timeout analogy accumulation'); return True
        return False        
    

    toplot1 = False 
    toplot11 = False 
    mainloop_score_thresh = 0.15
    myflag = False

    all_attempts_of_completed_eval_recons = [] # INIT

    try:

        #visualise.plot_two_grids(i_grids[0],o_grids[0])


        try:    
            global_parsings = {}; 
            for gridn in range(num_demo_grids + num_test_grids): 
                if esc(): break
                if esc1(): break
                i_grid = i_grids[gridn]
                if gridn <= num_demo_grids-1: o_grid = o_grids[gridn]
                global_parsings[gridn] = {}
                global_parsings[gridn]['i'] = {}
                global_parsings[gridn]['o'] = {}
                
                mask = np.ones_like(i_grid)
                map = i_grid
                maskv = mask.copy()
                masko = np.zeros_like(i_grid)
                global_parsings[gridn]['i'][create_name()] = {'parsing_type':'fullgrid_iobj','obj_score':0.4,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['fullgrid_i',0.5,None]}}


                if gridn <= num_demo_grids-1:
                    mask = np.ones_like(o_grid)
                    map = o_grid
                    maskv = mask.copy()
                    masko = np.zeros_like(o_grid)
                    global_parsings[gridn]['o'][create_name()] = {'parsing_type':'fullgrid_oobj','obj_score':0.1,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['fullgrid_o',0.1,None]}}



                def most_freq_val(grid):
                    if type(grid) != np.ndarray: print('ERROR')
                    if len(grid.shape)!=2: print('ERROR')
                    flattened = grid.flatten()
                    return int(max(set(flattened),key=list(flattened).count))

                def parsing_scorer(checkpt, preargs, preparams):
                    if checkpt == 1: 

                        default_background_color, commonest_color, obj_mask, obj_masko, obj_map, grid = preargs

                        obj_contains_no_bkgnd_cand_colors = True if none_of_x_in_y(x=[default_background_color, commonest_color],y = obj_map[obj_mask==1]) else False 

                        i_objs_nondiag = get_contiguous_regions(grid,None,False,False) 
                        temp_shapes = i_objs_nondiag[obj_mask==1]
                        obj_consists_of_mult_nondiags = True if len(np.unique(temp_shapes)) > 1 else False

                        bb_mask, bb_map, topleft_rc = get_bounding_box_object(obj_mask, obj_map)
                        regular_rect = True if np.sum(bb_mask) == np.prod(bb_mask.shape) else False

                        score = 0.9
                        if not obj_contains_no_bkgnd_cand_colors: score -= 0.1
                        if obj_consists_of_mult_nondiags: score -= 0.2
                        if not regular_rect: score -= 0.05

                        

                    if checkpt == 2: 

                        obj_mask, obj_masko, obj_map, grid = preargs
                        background_color = preparams

                        i_objs_nondiag = get_contiguous_regions(grid,None,False,False) 
                        temp_shapes = i_objs_nondiag[obj_mask==1]
                        obj_consists_of_mult_nondiags = True if len(np.unique(temp_shapes)) > 1 else False

                        bb_mask, bb_map, topleft_rc = get_bounding_box_object(obj_mask, obj_map)
                        regular_rect = True if np.sum(bb_mask) == np.prod(bb_mask.shape) else False

                        nrows = bb_map.shape[0]
                        upper_half = bb_map[:nrows // 2]
                        lower_half = bb_map[(nrows + 1) // 2:] 
                        is_vert_symmetrical = np.array_equal(upper_half, np.flipud(lower_half))
                        ncols = bb_map.shape[1]
                        left_half = bb_map[:,:ncols // 2]
                        right_half = bb_map[:,(ncols + 1) // 2:] 
                        is_horiz_symmetrical = np.array_equal(left_half, np.fliplr(right_half))
                        if is_vert_symmetrical and is_horiz_symmetrical: symmetry_dividend = -0.1
                        elif is_vert_symmetrical or is_horiz_symmetrical: symmetry_dividend = -0.05
                        else: symmetry_dividend = 0


                        score = 0.8
                        if obj_consists_of_mult_nondiags: score -= 0.2
                        if not regular_rect: score -= 0.05
                        score -= symmetry_dividend 
                        

                    return score

                def is_regular(mask):
                    bb_mask, bb_map, tl_rc = get_bounding_box_object(mask, mask) 
                    if np.sum(bb_mask) == bb_mask.shape[0]*bb_mask.shape[1]: return True
                    return False

                for gridt in ['i','o']:
                    if gridt == 'o' and gridn > num_demo_grids-1: continue
                    grid = i_grid if gridt == 'i' else o_grid
                    pc = 0
                    
                    default_background_color = 0
                    commonest_color = most_freq_val(grid)
                    if_o_get_i_commonest = commonest_color if gridt == 'i' else most_freq_val(i_grid)
                    
                    
                    i_objs = get_contiguous_regions(grid,None,True,False) 
                    for n in np.unique(i_objs): 
                        obj_mask = (i_objs==n).astype(int) 
                        obj_maskv = obj_mask.copy() 
                        obj_masko = obj_mask - obj_maskv 
                        obj_map  = np.where(obj_mask, grid, 0) 
                        score = parsing_scorer(checkpt=1,preargs=[default_background_color, commonest_color, obj_mask, obj_masko, obj_map, grid],preparams=[])
                        straightforward_obj = True if score == 0.9 else False 
                        pc+=1
                        global_parsings[gridn][gridt][create_name()] = {'parsing_type':'basic_singlecolor_obj', 'obj_score':score,'mask':obj_mask,'maskv':obj_maskv,'masko':obj_masko,'map':obj_map,
                                                        'properties':{'is_straightforward_obj':straightforward_obj, 'parsing_description':['single_color',score,None]}}
                    
                    for background_color in ( list(np.unique([default_background_color, commonest_color, if_o_get_i_commonest]))+['None'] ): 
                        
                        if background_color != 'None':
                            obj_mask = np.ones_like(grid)
                            obj_maskv = (grid==background_color).astype(int)
                            obj_masko = obj_mask - obj_maskv
                            obj_map = background_color * np.ones_like(grid) 
                            pc+=1
                            global_parsings[gridn][gridt][create_name()] = {'parsing_type':'background', 'obj_score':0.9,'mask':obj_mask,'maskv':obj_maskv,'masko':obj_masko,'map':obj_map,
                                                            'properties':{'is_straightforward_obj':False, 'parsing_description':['background',None,background_color]}}
                        
                            
                            obj_mask = (grid!=background_color).astype(int)
                            obj_map = np.where(obj_mask, grid, 0)
                            obj_maskv = obj_mask.copy()
                            obj_masko = np.zeros_like(obj_mask)
                            pc+=1
                            global_parsings[gridn][gridt][create_name()] = {'parsing_type':'nonbkg_obj', 'obj_score':0.9,'mask':obj_mask,'maskv':obj_maskv,'masko':obj_masko,'map':obj_map,
                                                            'properties':{'is_straightforward_obj':False, 'parsing_description':['nonbkg_obj',None,background_color]}}
                                    
                        
                        
                        
                        
                        
                        if background_color != 'None': 
                            i_multiobjs = get_contiguous_regions(grid,background_color,True,True) 
                            
                            
                            for n in range(1,np.max(i_multiobjs)+1): 
                                obj_mask = (i_multiobjs==n).astype(int)
                                obj_maskv = obj_mask.copy() 
                                obj_masko = obj_mask - obj_maskv
                                obj_map  = np.where(obj_mask, grid, 0)
                                score = parsing_scorer(checkpt=2,preargs=[obj_mask, obj_masko, obj_map, grid],preparams=[background_color])
                                existing_objs = [[global_parsings[gridn][gridt][_]['mask'],global_parsings[gridn][gridt][_]['map'],global_parsings[gridn][gridt][_]['maskv']] for _ in global_parsings[gridn][gridt]]
                                if not is_x_in_y(x=[obj_mask,obj_map,obj_maskv], y=existing_objs): 
                                    ncolors = 'single_color' if len(get_colors_of_obj(obj_mask,obj_map))==1 else 'multi_color' 
                                    pc+=1
                                    global_parsings[gridn][gridt][create_name()] = {'parsing_type':ncolors, 'obj_score':score,'mask':obj_mask,'maskv':obj_maskv,'masko':obj_masko,'map':obj_map,
                                                                    'properties':{'is_straightforward_obj':False, 'parsing_description':[ncolors,score,background_color]}}
                        

                        
                    
                    


                    
                    if pc > 50: 

                        global_parsings[gridn][gridt] = {} 

                        if gridt == 'i': 
                            mask = np.ones_like(i_grid)
                            map = i_grid
                            maskv = mask.copy()
                            masko = np.zeros_like(i_grid)
                            global_parsings[gridn]['i'][create_name()] = {'parsing_type':'fullgrid_iobj','obj_score':0.4,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['fullgrid_i',0.5,None]}}
                            
                        if gridt == 'o': 

                            mask = np.ones_like(o_grid)
                            map = o_grid
                            maskv = mask.copy()
                            masko = np.zeros_like(o_grid)
                            global_parsings[gridn]['o'][create_name()] = {'parsing_type':'fullgrid_oobj','obj_score':0.1,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['fullgrid_o',0.1,None]}}


                    else: pass
                    

                    
                    noisy_flag = False
                    for color in np.unique(grid):
                        color_mask = (grid==color).astype(int)
                        i_multiobjs = get_contiguous_regions(color_mask,0,True,False)
                        
                        regular_count = 0
                        for on in range(1,np.max(i_multiobjs)+1):
                            obj_mask = (i_multiobjs==on).astype(int)
                            if is_regular(obj_mask): regular_count +=1

                        
                        likely_noise = True if np.amax(i_multiobjs) > 20 or (np.amax(i_multiobjs) > 3 and regular_count/np.amax(i_multiobjs)!=1) else False
                        

                        if not likely_noise:
                            
                            if np.max(i_multiobjs) > 10: print('SKIPPING'); continue
                            for on in range(1,np.max(i_multiobjs)+1):
                                obj_mask = (i_multiobjs==on).astype(int)
                                obj_maskv = obj_mask.copy()
                                obj_masko = obj_mask - obj_maskv
                                obj_map  = np.where(obj_mask, grid, 0)
                                score = 0.95
                                global_parsings[gridn][gridt][create_name()] = {'parsing_type':'singlecolor_in_noise', 'obj_score':score,'mask':obj_mask,'maskv':obj_maskv,'masko':obj_masko,'map':obj_map,
                                                                'properties':{'is_straightforward_obj':False, 'parsing_description':['singlecolor_in_noise',score,None]}}
                            
                        elif likely_noise:
                            noisy_flag = True

                            obj_mask = color_mask
                            obj_maskv = obj_mask.copy()
                            obj_masko = obj_mask - obj_maskv
                            obj_map  = np.where(obj_mask, grid, 0)
                            score = 0.5
                            global_parsings[gridn][gridt][create_name()] = {'parsing_type':'noise_mask', 'obj_score':score,'mask':obj_mask,'maskv':obj_maskv,'masko':obj_masko,'map':obj_map,
                                                            'properties':{'is_straightforward_obj':False, 'parsing_description':['noise_mask',score,None]}}


                    

                    
                    
                    if gridt=='i':
                        icolors = get_colors_of_obj(np.ones_like(grid), grid)
                        for color in icolors:
                            rows, cols = np.where(grid==color)
                            for m1 in range(len(rows)):
                                for m2 in range(len(rows)):
                                    if m1==m2: continue 
                                    
                                    if (rows[m2] < rows[m1] + 2) or (cols[m2] < cols[m1] + 2): continue 

                                    
                                    
                                    cand_mask = rectangular_border_mask(grid.shape, (rows[m1],cols[m1]), (rows[m2],cols[m2]))

                                    mask_colors = np.unique(grid[cand_mask==1])
                                    if len(mask_colors)==1:

                                        border_mask = get_outline_border_mask(cand_mask)

                                        fill_mask = grid[rows[m1]+1:rows[m2],cols[m1]+1:cols[m2]] 

                                        border_colors = list(grid[border_mask==1])

                                        outline_samecolor_frac = border_colors.count(mask_colors[0]) / len(border_colors)
                                        
                                        if outline_samecolor_frac > 0.5: continue

                                        
                                        
                                        if (fill_mask.shape[0]==1 or fill_mask.shape[1]==1) and outline_samecolor_frac > 0: continue

                                        

                                        pc+=1
                                        score = 0.7
                                        if fill_mask.shape[0]*fill_mask.shape[1] < 6: score = score * 0.5
                                        if outline_samecolor_frac != 0: score = score * 0.8


                                        mask = np.zeros_like(grid)
                                        mask[rows[m1]+1:rows[m2],cols[m1]+1:cols[m2]] = 1
                                        map = np.where(mask, grid, 0)
                                        maskv = mask
                                        masko = np.zeros_like(grid)

                                        global_parsings[gridn][gridt][create_name()] = {'parsing_type':'container_rect_1pxwidth', 'obj_score':score,'mask':mask,'maskv':maskv,'masko':masko,'map':map,
                                                                        'properties':{'is_straightforward_obj':False, 'parsing_description':['container_rect_1pxwidth',score,None]}}

                    
                    if gridt=='i':
                        all_i_colors = np.unique(i_grid)
                        for i_color in all_i_colors:
                            colormask_in_i = (i_grid==i_color).astype(int)
                            if np.sum(colormask_in_i)>0:
                                score=0.7; 
                                if i_color in list(np.unique([default_background_color, commonest_color, if_o_get_i_commonest])): score = 0.2
                                mask = colormask_in_i; maskv = mask; map = np.where(mask, i_grid, 0); masko = np.zeros_like(i_grid)
                                global_parsings[gridn][gridt][create_name()] = {'parsing_type':'mask_of_color', 'obj_score':score,'mask':mask,'maskv':maskv,'masko':masko,'map':map,
                                            'properties':{'is_straightforward_obj':False, 'parsing_description':['mask_of_color',score,None]}}

                    
                    if gridt=='i':
                        all_i_colors = np.unique(i_grid)
                        combo_mask = np.zeros_like(i_grid)
                        for i_color in all_i_colors:
                            if i_color not in list(np.unique([default_background_color, commonest_color, if_o_get_i_commonest])):
                                colormask_in_i = (i_grid==i_color).astype(int)
                                combo_mask = combo_mask | colormask_in_i
                        if np.sum(combo_mask)>0:
                            score = 0.6; mask = combo_mask; maskv = mask; map = np.where(mask, i_grid, 0); masko = np.zeros_like(i_grid)
                            global_parsings[gridn][gridt][create_name()] = {'parsing_type':'mask_of_all_colors', 'obj_score':score,'mask':mask,'maskv':maskv,'masko':masko,'map':map,
                                        'properties':{'is_straightforward_obj':False, 'parsing_description':['mask_of_all_colors',score,None]}}
        except: pass



        try:    
            global_frames = {}
            for gridn in range(num_demo_grids + num_test_grids):
                if esc(): break
                if esc1(): break
                i_grid = i_grids[gridn]
                if gridn <= num_demo_grids-1: o_grid = o_grids[gridn]
                global_frames[gridn] = {}
                global_frames[gridn]['i'] = frame_psuedoobjs(i_grid)
                if gridn > num_demo_grids-1: continue
                global_frames[gridn]['o'] = frame_psuedoobjs(o_grid)
            for gridn in range(num_demo_grids + num_test_grids):
                if esc(): break
                if esc1(): break
                
                i_grid = i_grids[gridn]
                if gridn <= num_demo_grids-1: o_grid = o_grids[gridn]


                for gridt in ['i','o']:
                    if gridt == 'o' and gridn > num_demo_grids-1: continue
                    grid = i_grid if gridt == 'i' else o_grid

                    for framekey in global_frames[gridn][gridt]:
                        frame = global_frames[gridn][gridt][framekey]
                        if frame['type'] == 'wholegrid_obj': continue 

                        score = 0.9
                        if frame['totalcount']==1: score = 0.2; continue 
                        

                        
                        frame_color = frame['frame_color'] 
                        subframe_colors = [get_colors_of_obj(subframe['mask'], subframe['map']) for subframe in frame['subframes']]
                        case1 = True if np.any([frame_color in _ for _ in subframe_colors]) else False 
                        case2 = False; common_color = None
                        for colr in subframe_colors[0]: 
                            if np.all([colr in _ for _ in subframe_colors]):
                                case2 = True; common_color = colr
                        
                        the_set = []; flag = True
                        for subframe in frame['subframes']:
                            
                            if case1: 
                                framecolorless_mask = subframe['mask'] & (subframe['map']!=frame_color) 
                                framecolorless_map = np.where(framecolorless_mask, subframe['map'], 0)
                                maskv = framecolorless_mask.copy()
                                masko = np.zeros_like(maskv)
                                mask_ = framecolorless_mask; map_ = framecolorless_map
                            elif case2: 
                                commonless_mask = subframe['mask'] & (subframe['map']!=common_color)
                                commonless_map = np.where(commonless_mask, subframe['map'], 0)
                                maskv = commonless_mask.copy()
                                masko = np.zeros_like(maskv)
                                mask_ = commonless_mask; map_ = commonless_map
                            else: 
                                mask_, map_ = subframe['mask'], subframe['map']
                                maskv = mask_.copy()
                                masko = np.zeros_like(maskv)                
                                flag = False
                            the_set.append({'mask':mask_,'map':map_,'maskv':maskv,'masko':masko,'counter':subframe['counter']})
                        if not flag: score -= 0.1
                        nonframe_mask = np.zeros_like(subframe['mask'])
                        for subframe in frame['subframes']:
                            nonframe_mask = nonframe_mask | subframe['mask']
                        frame_mask = (nonframe_mask==0).astype(int)
                        frame_map = np.where(frame_mask, grid, 0)
                        frame_maskv = frame_mask.copy()
                        frame_masko = np.zeros_like(frame_maskv)


                        
                        


                        
                        frameset_name = framekey

                        for obj in the_set: 
                            subframeobj_name = create_name()
                            mask_, map_, maskv, masko, counter = obj.values()
                            global_parsings[gridn][gridt][subframeobj_name] = {'parsing_type':'subframe_iobj', 'obj_score':score,'mask':mask_,'maskv':maskv,'masko':masko,'map':map_,
                                                            'properties':{'is_straightforward_obj':False, 'parsing_description':['subframe_iobj',score,None], 
                                                            'frame_ID':frameset_name,'v':1,'c':counter, 'bkg_case':'frame_color' if case1 else 'common_color','bkg_color':frame_color if case1 else common_color}}
                        frameobj_name = create_name()
                        global_parsings[gridn][gridt][frameobj_name] = {'parsing_type':'frame_iobj', 'obj_score':score,'mask':frame_mask,'maskv':frame_maskv,'masko':frame_masko,'map':frame_map,
                                                        'properties':{'is_straightforward_obj':False, 'parsing_description':['frame_iobj',score,None], 
                                                        'frame_ID':frameset_name,'v':1,'c':counter, 'bkg_case':'frame_color' if case1 else 'common_color','bkg_color':frame_color if case1 else common_color}}
        except: pass
                    

                    

        initial_global_parsings = copy.deepcopy(global_parsings)



        

        def detect_io_transforms(iobj_, oobj_, i_mask_raw, i_map_raw, o_grid_raw, o_region1_raw, o_masko1_raw, gridn, main_chains_override = None, oobj_or_oreg_mode = 'oobj'):

            
            
            

            ANCHOR_ASSUMPTION = 'top_left' 


            
            if i_mask_raw.shape[0] > o_grid_raw.shape[0]: vert_pad = 'o_grid'; vert_sz = i_mask_raw.shape[0]
            elif i_mask_raw.shape[0] < o_grid_raw.shape[0]: vert_pad = 'i_grid'; vert_sz = o_grid_raw.shape[0]
            else: vert_pad = 'neither'; vert_sz = i_mask_raw.shape[0]
            if i_mask_raw.shape[1] > o_grid_raw.shape[1]: horiz_pad = 'o_grid'; horiz_sz = i_mask_raw.shape[1]
            elif i_mask_raw.shape[1] < o_grid_raw.shape[1]: horiz_pad = 'i_grid'; horiz_sz = o_grid_raw.shape[1]
            else: horiz_pad = 'neither'; horiz_sz = i_mask_raw.shape[1]

            def set_to_topleft(raw):
                newarr = np.zeros((vert_sz, horiz_sz))
                newarr[0:0+raw.shape[0],0:0+raw.shape[1]] = raw
                return newarr.astype(int)

            i_mask =  set_to_topleft(i_mask_raw)
            i_map =  set_to_topleft(i_map_raw)
            o_grid =  set_to_topleft(o_grid_raw)
            o_region1 =  set_to_topleft(o_region1_raw)
            o_masko1 =  set_to_topleft(o_masko1_raw)





            o_mode = oobj_or_oreg_mode 
            
            

            
            
            



            bb_mask, bb_map, tl_rc = get_bounding_box_object(i_mask,i_map)
            results = []

            main_chains = []
            main_chains.append([{'fn':movt,'params':{}},{'fn':recolor,'params':{}}])
            main_chains.append([{'fn':extension,'params':{}}])






            if main_chains_override is not None: main_chains = main_chains_override

            unique_emask_paramlists = []
            for chain in main_chains:
                if esc(): break
                if esc1(): break
                chain_branchlist = [chain.copy()] 

                a0 = 0
                while a0 < len(chain_branchlist): 
                    chain_branch = chain_branchlist[a0]

                    curr_map, curr_mask = i_map.copy(), i_mask.copy() 

                    quitflag = False
                    score_improvement_list = []
                    curr_o_region = copy.deepcopy(o_region1) 
                    curr_o_masko = copy.deepcopy(o_masko1) 
                    curr_explained_mask = np.zeros_like(o_grid)
                    prev_explained_mask = np.zeros_like(o_grid)
                    prev_oregion_explained = False

                    c0=0

                    for fn_dict in chain_branch:
                        fn_name = fn_dict['fn'].__name__


                        def detect_movt_A(curr_mask, o_region):

                            

                            rows, cols = np.where(curr_mask==1)
                            ris,rie,cis,cie = rows.min(), rows.max(), cols.min(), cols.max()
                            rows, cols = np.where(o_region==1) 
                            ros,roe,cos,coe = rows.min(), rows.max(), cols.min(), cols.max()

                            opts = []

                            
                            if ris >= ros and rie <= roe: 
                                if cis >= cos and cie <= coe: 
                                    opts.append((0,0))

                            
                            if cis >= cos and cie <= coe: 
                                for dr in range(ros-ris,roe-rie+1):
                                    if dr != 0: opts.append((dr,0))
                            if ris >= ros and rie <= roe: 
                                for dc in range(cos-cis,coe-cie+1):
                                    if dc != 0: opts.append((0,dc))

                            
                            for dr in range(ros-ris,roe-rie+1): 
                                for dc in range(cos-cis,coe-cie+1):
                                    if dr!=0 and dc!=0: opts.append((dr,dc))

                            

                            return opts
                        
                        def detect_recolor_A(curr_map, curr_mask, o_grid, o_region, o_masko):

                            icolors = list(np.unique(curr_map[curr_mask==1]))
                            ocolors = list(np.unique(o_grid[o_region==1]))


                            

                            
                            
                            
                            

                            
                            
                            



                            ocolors = list(np.unique(o_grid[np.multiply(o_region==1,o_masko==0)])) 




                            opts = []

                            
                            opts.append([(int(ic),int(ic)) for ic in icolors])

                            
                            for ic in icolors:
                                for oc in ocolors:
                                    if [(int(ic),int(oc))] not in opts: opts.append([(int(ic),int(oc))])

                            
                            

                            
                            if len(icolors)**len(ocolors)<=8:
                                res = [list(zip(icolors, p)) for p in product(ocolors, repeat=len(icolors))] 
                                for re in res: 
                                    if re not in opts: opts.append(re)


                            return opts

                        def detect_masking_A(curr_map, curr_mask, o_grid, o_region, curr_o_masko):
                            
                            
                            
                            opt = []
                            if are_identicalQ([curr_mask,o_region]):
                                masking_mask = np.zeros_like(o_region)
                                rows,cols = np.where(curr_mask==1)
                                for m in range(len(rows)):
                                    if o_grid[rows[m],cols[m]] != curr_map[rows[m],cols[m]] and curr_o_masko[rows[m],cols[m]]==0: 
                                        masking_mask[rows[m],cols[m]] = 1
                                if np.sum(masking_mask) > 0: opt.append(masking_mask)
                            return opt

                        # detect_extension_A

                        def detect_copying_A(curr_map, curr_mask, o_grid, o_region):
                            opts = []
                            
                            directions =  ['S','SW','SE','W','E','N','NW','NE']
                            directions_ = [(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)]
                            B = o_grid
                            A0 = curr_map
                            A = curr_mask
                            rows,cols = np.where(A==1)
                            
                            o_mask = np.ones_like(B)
                            o_map = B
                            mode = 'no_colorchange' 

                            opt = []
                            for expected_dir in [(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)]:

                                mask = o_mask.copy()

                                map = o_map.copy(); flatmap = map[rows,cols] 

                                

                                explains = np.zeros_like(o_map)

                                
                                res = safe_assign_rc(rows,cols,explains,1,allow_spillover=False)
                                if res is not None: explains = res

                                maskR,maskC = mask.shape

                                directions_ = [(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)] 

                                record = []; dodgy = False
                                record_w_details = []

                                
                                expected_skip = False 
                                for skip in range(1,30):
                                    
                                    di = expected_dir
                                    temprows = rows + (skip*di[0])
                                    tempcols = cols + (skip*di[1])

                                    
                                    filteredrows = []; filteredcols = []; flag_beyond = False; valid_cdts = []
                                    for m in range(len(temprows)):
                                        if temprows[m] >= 0 and temprows[m] < maskR and tempcols[m] >= 0 and tempcols[m] < maskC:
                                            filteredrows.append(temprows[m]); filteredcols.append(tempcols[m])
                                            valid_cdts.append({'old_row':rows[m],'old_col':cols[m],'new_row':temprows[m],'new_col':tempcols[m]}) 
                                        else: flag_beyond = True
                                    
                                    if len(filteredrows) == 0: continue 
                                    elif flag_beyond: 
                                        
                                        continue 
                                        

                                    

                                    new_position_mask = np.zeros_like(mask); new_cdts = []; already_partly_explained_flag = False
                                    for m in range(len(filteredrows)):
                                        if filteredrows[m] >= 0 and filteredrows[m] < maskR and filteredcols[m] >= 0 and filteredcols[m] < maskC:
                                            if explains[filteredrows[m],filteredcols[m]]==0: 
                                                new_position_mask[filteredrows[m],filteredcols[m]] = 1; new_cdts.append((filteredrows[m],filteredcols[m]))
                                            else: already_partly_explained_flag = True
                                    if already_partly_explained_flag: continue 
                                    
                                    if len(new_cdts) == 0: continue


                                    i_vals = []; o_vals = []
                                    for cdts in valid_cdts: 
                                        i_vals.append(int(map[cdts['old_row'],cdts['old_col']]))
                                        o_vals.append(int(map[cdts['new_row'],cdts['new_col']]))
                                    i_vals = np.array(i_vals); o_vals = np.array(o_vals)
                                    i_colors = list(np.unique(i_vals)); o_colors = list(np.unique(o_vals))
                                    object_maintained = True if (i_vals == o_vals).all() else False 
                                    perfect_colorchange, grey_recoloring, color_changes = check_perfect_colorchange(i_colors,o_colors, i_vals,o_vals)
                                    
                                    o_outline_mask = get_outline_border_mask(new_position_mask)
                                    o_outline_vals = map[o_outline_mask.astype(bool)]; o_outline_colors = list(np.unique(o_outline_vals))
                                    nonleaked_oborder = True if none_of_x_in_y(x=o_colors, y=o_outline_colors) else False 
                                    
                                    if mode == 'allow_colorchange':
                                        if nonleaked_oborder and (object_maintained or perfect_colorchange):
                                            expected_skip = skip
                                            break
                                    else:
                                        if nonleaked_oborder and (object_maintained):
                                            expected_skip = skip
                                            break                                

                                    

                                

                                for k in range(200):

                                    candidate_dirs = []
                                    for di in directions_:

                                        
                                        temprows = rows + (expected_skip*di[0])
                                        tempcols = cols + (expected_skip*di[1])
                                        
                                        
                                        filteredrows = []; filteredcols = []; flag_beyond = False; valid_cdts = []
                                        for m in range(len(temprows)):
                                            if temprows[m] >= 0 and temprows[m] < maskR and tempcols[m] >= 0 and tempcols[m] < maskC:
                                                filteredrows.append(temprows[m]); filteredcols.append(tempcols[m])
                                                valid_cdts.append({'old_row':rows[m],'old_col':cols[m],'new_row':temprows[m],'new_col':tempcols[m]}) 
                                            else: flag_beyond = True
                                        
                                        if len(filteredrows) == 0: continue 
                                        elif flag_beyond: 
                                            
                                            continue 
                                            

                                        

                                        new_position_mask = np.zeros_like(mask); new_cdts = []; already_partly_explained_flag = False
                                        for m in range(len(filteredrows)):
                                            if filteredrows[m] >= 0 and filteredrows[m] < maskR and filteredcols[m] >= 0 and filteredcols[m] < maskC:
                                                if explains[filteredrows[m],filteredcols[m]]==0: 
                                                    new_position_mask[filteredrows[m],filteredcols[m]] = 1; new_cdts.append((filteredrows[m],filteredcols[m]))
                                                else: already_partly_explained_flag = True
                                        if already_partly_explained_flag: continue 
                                        
                                        if len(new_cdts) == 0: continue


                                        i_vals = []; o_vals = []
                                        for cdts in valid_cdts: 
                                            i_vals.append(int(map[cdts['old_row'],cdts['old_col']]))
                                            o_vals.append(int(map[cdts['new_row'],cdts['new_col']]))
                                        i_vals = np.array(i_vals); o_vals = np.array(o_vals)
                                        i_colors = list(np.unique(i_vals)); o_colors = list(np.unique(o_vals))
                                        object_maintained = True if (i_vals == o_vals).all() else False 
                                        perfect_colorchange, grey_recoloring, color_changes = check_perfect_colorchange(i_colors,o_colors, i_vals,o_vals)
                                        o_outline_mask = get_outline_border_mask(new_position_mask)
                                        o_outline_vals = map[o_outline_mask.astype(bool)]; o_outline_colors = list(np.unique(o_outline_vals))
                                        nonleaked_oborder = True if none_of_x_in_y(x=o_colors, y=o_outline_colors) else False 
                                        
                                        if mode == 'allow_colorchange':
                                            if nonleaked_oborder and (object_maintained or perfect_colorchange):
                                                if object_maintained: candidate_dirs.append({'cand_dir':di,'is_same_color':True,'color_change':None})
                                                else: candidate_dirs.append({'cand_dir':di,'is_same_color':False,'color_change':color_changes})
                                        else:
                                            if nonleaked_oborder and (object_maintained):
                                                if object_maintained: candidate_dirs.append({'cand_dir':di,'is_same_color':True,'color_change':None})
                                                else: candidate_dirs.append({'cand_dir':di,'is_same_color':False,'color_change':color_changes})                                    


                                        

                                    


                                    if k == 0:
                                        if expected_dir in [_['cand_dir'] for _ in candidate_dirs]: select_dir = expected_dir; 
                                        else: 
                                            
                                            break
                                    else:
                                        if len(candidate_dirs) > 1:
                                            same_color_dirs = []
                                            for cand in candidate_dirs:
                                                if cand['is_same_color']: same_color_dirs.append(cand['cand_dir'])
                                            if len(same_color_dirs) > 1: 
                                                
                                                if expected_dir in [_['cand_dir'] for _ in candidate_dirs]: select_dir = expected_dir; 
                                                else: 
                                                    
                                                    break
                                            elif len(same_color_dirs) == 1: select_dir = same_color_dirs[0]; 
                                            elif len(same_color_dirs) == 0:
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                if expected_dir in [_['cand_dir'] for _ in candidate_dirs]: select_dir = expected_dir; 
                                                else: 
                                                    
                                                    break
                                                
                                        elif len(candidate_dirs) == 1: select_dir = candidate_dirs[0]['cand_dir']; 
                                        elif len(candidate_dirs) == 0: 
                                            
                                            break
                                        
                                    


                                    
                                    for cand_ in candidate_dirs:
                                        if cand_['cand_dir'] == select_dir:
                                            select_is_same_color = cand_['is_same_color']
                                            select_color_change = cand_['color_change']
                                    if len(record)==0: record.append([select_dir,1]); record_w_details.append([select_dir,select_is_same_color,select_color_change])
                                    else:
                                        if record[-1][0] == select_dir: record[-1][1]+=1; record_w_details.append([select_dir,select_is_same_color,select_color_change])
                                        else: record.append([select_dir,1]); record_w_details.append([select_dir,select_is_same_color,select_color_change])

                                    

                                    
                                    temprows = rows + (expected_skip*select_dir[0]); tempcols = cols + (expected_skip*select_dir[1])

                                    
                                    res = safe_assign_rc(temprows,tempcols,explains,1,allow_spillover=False)
                                    if res is not None: explains = res

                                    expected_dir = select_dir 

                                    rows = temprows; cols = tempcols 

                                if record != []: 
                                    
                                    opt.append({'start_dirn':expected_dir,'skip':expected_skip,'record':record_w_details})
                            opts.append(opt)

                            return opts

                        def detect_movt_B(curr_mask, o_region): 
                            rows, cols = np.where(curr_mask == 1)
                            ris, rie, cis, cie = rows.min(), rows.max(), cols.min(), cols.max()
                            rows, cols = np.where(o_region == 1)
                            ros, roe, cos, coe = rows.min(), rows.max(), cols.min(), cols.max()

                            opts = []

                            
                            if rie >= ros and ris <= roe and cie >= cos and cis <= coe:
                                opts.append((0, 0))

                            
                            for dr in range(ros - rie, roe - ris + 1):
                                if dr != 0:
                                    if rie + dr >= ros and ris + dr <= roe and cie >= cos and cis <= coe:
                                        opts.append((dr, 0))

                            
                            for dc in range(cos - cie, coe - cis + 1):
                                if dc != 0:
                                    if cie + dc >= cos and cis + dc <= coe and rie >= ros and ris <= roe:
                                        opts.append((0, dc))

                            
                            for dr in range(ros - rie, roe - ris + 1):
                                for dc in range(cos - cie, coe - cis + 1):
                                    if dr != 0 and dc != 0:
                                        if (rie + dr >= ros and ris + dr <= roe and
                                            cie + dc >= cos and cis + dc <= coe):
                                            opts.append((dr, dc))

                            

                            return opts

                        def detect_recolor_B(curr_map, curr_mask, o_grid, o_region, o_masko):
                            

                            icolors = list(np.unique(curr_map[curr_mask==1]))
                            ocolors = list(np.unique(o_grid[o_region==1]))
                            ocolors = list(np.unique(o_grid[curr_mask==1])) 
                            
                            

                            
                            ocolors = list(np.unique(o_grid[np.multiply(curr_mask==1,o_masko==0)])) 
                            


                            opts = []

                            
                            opts.append([(int(ic),int(ic)) for ic in icolors])
                            
                            
                            for ic in icolors:
                                for oc in ocolors:
                                    if [(int(ic),int(oc))] not in opts: opts.append([(int(ic),int(oc))])

                            
                            

                            
                            if len(icolors)**len(ocolors)<=8:
                                res = [list(zip(icolors, p)) for p in product(ocolors, repeat=len(icolors))] 
                                for re in res: 
                                    if re not in opts: opts.append(re)


                            return opts

                        def detect_masking_B(curr_map, curr_mask, o_grid, o_region, curr_o_masko):
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            

                            
                            
                            opt = []
                            masking_mask = np.zeros_like(o_grid)
                            rows,cols = np.where(curr_mask==1)
                            for m in range(len(rows)):
                                if curr_map[rows[m],cols[m]] != o_grid[rows[m],cols[m]] and curr_o_masko[rows[m],cols[m]]==0: 
                                    masking_mask[rows[m],cols[m]] = 1
                            if np.sum(masking_mask) > 0: opt.append(masking_mask)
                            return opt

                        
                        def detect_expansion_B(curr_map, curr_mask, o_grid, o_region):
                            
                            opts = []

                            bb_mask, bb_map, tl_rc = get_bounding_box_object(curr_mask,curr_map)
                            bb_maskmap = np.zeros((bb_mask.shape[0],bb_mask.shape[1],2))
                            bb_maskmap[:,:,0] = bb_mask; bb_maskmap[:,:,1] = bb_map

                            
                            

                            bb_combined = np.where(bb_maskmap[:,:,0], bb_maskmap[:,:,1], -1)

                            def best_contraction(mask):
                                
                                
                                r, c = mask.shape
                                best_creduction = 1; creduced_mask = mask.copy()
                                for divs in range(2,c+1):
                                    if c % divs == 0: 
                                        temp = np.zeros(( r, c//divs, divs)) 
                                        for d in range(divs):
                                            temp[:,:,d] = mask[:,d::divs] 
                                        if np.all((temp == temp[:, :, :1])): best_creduction = divs; creduced_mask = temp[:,:,0] 

                                best_rreduction = 1; rreduced_mask = mask.copy()
                                r, c = mask.shape
                                for divs in range(2,r+1):
                                    if r % divs == 0:
                                        temp = np.zeros(( r//divs, c, divs))
                                        for d in range(divs):
                                            temp[:,:,d] = mask[d::divs,:]
                                        if np.all((temp == temp[:, :, :1])): best_rreduction = divs; rreduced_mask = temp[:,:,0]

                                fully_reduced_mask = mask[::best_rreduction,::best_creduction]
                                return fully_reduced_mask, best_rreduction, best_creduction

                            
                            fully_reduced_mask, best_rreduction, best_creduction = best_contraction(bb_combined)

                            fully_reduced_mask = bb_combined[::best_rreduction,::best_creduction]
                            temp_ = np.repeat(fully_reduced_mask, best_rreduction, axis=0)
                            reexpanded = np.repeat(temp_, best_creduction, axis=1)

                            expansion_maskmap_list = []; expansion_frac_list = []
                            input_grid = i_grids[gridn] 
                            output_grid = o_grids[gridn] 
                            largestshape = (np.max([input_grid.shape[0],output_grid.shape[0]]),np.max([input_grid.shape[1],output_grid.shape[1]]))
                            maxfull = 5 
                            for rmult in list(set([best_rreduction, best_rreduction*2, best_rreduction*3] +  list(range(1,maxfull+1)))):
                                for cmult in list(set([best_creduction, best_creduction*2, best_creduction*3] +  list(range(1,maxfull+1)))):
                                    reduced_mask = bb_maskmap[::best_rreduction,::best_creduction,0]
                                    reduced_map = bb_maskmap[::best_rreduction,::best_creduction,1]
                                    temp_ = np.repeat(reduced_mask, rmult, axis=0)
                                    reconmask = np.repeat(temp_, cmult, axis=1)
                                    temp_ = np.repeat(reduced_map, rmult, axis=0)
                                    reconmap = np.repeat(temp_, cmult, axis=1)
                                    if reconmask.shape[0] > largestshape[0] or reconmask.shape[1] > largestshape[1]: continue 
                                    expansion_frac_list.append({'w_mult':fractions.Fraction(rmult,best_rreduction) , 'h_mult' : fractions.Fraction(cmult,best_creduction)}) 
                                    expansion_maskmap_list.append(np.stack((reconmask,reconmap),axis=2)) 


                            
                            for _ in expansion_frac_list:
                                opts.append((_['w_mult'],_['h_mult']))
                            return opts

                        def detect_flip_B(curr_map, curr_mask, o_grid, o_region, o_masko):
                            opts = []
                            
                            bb_mask, bb_map, tl_rc = get_bounding_box_object(curr_mask,curr_map)
                            desired_fliprow = bb_mask.shape[0]/2-.5 + tl_rc[0]
                            desired_flipcol = bb_mask.shape[1]/2-0.5 + tl_rc[1]
                            opts.append(['x_axis',desired_fliprow])
                            opts.append(['y_axis',desired_flipcol])
                            
                            
                            return opts

                        def detect_rotate_B(curr_map, curr_mask, o_grid, o_region, o_masko):
                            opts = []
                            
                            bb_mask, bb_map, tl_rc = get_bounding_box_object(curr_mask,curr_map)
                            desired_fliprow = bb_mask.shape[0]/2-.5 + tl_rc[0]
                            desired_flipcol = bb_mask.shape[1]/2-0.5 + tl_rc[1]
                            desired_centre = (desired_fliprow,desired_flipcol)
                            for rotation in [1,-1,2,-2,3,-3]: 
                                opts.append([rotation,desired_centre])
                            
                            
                            return opts                    
                        
                        # detect_extension_B

                        def detect_copying_B(curr_map, curr_mask, o_grid, o_region):
                            opts = []
                            
                            directions =  ['S','SW','SE','W','E','N','NW','NE']
                            directions_ = [(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)]
                            B = o_grid
                            A0 = curr_map
                            A = curr_mask
                            rows,cols = np.where(A==1)
                            
                            o_mask = np.ones_like(B)
                            o_map = B
                            mode = 'no_colorchange' 

                            opt = []
                            for expected_dir in [(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)]:

                                mask = o_mask.copy()

                                map = o_map.copy(); flatmap = map[rows,cols] 

                                

                                explains = np.zeros_like(o_map)

                                
                                res = safe_assign_rc(rows,cols,explains,1,allow_spillover=False)
                                if res is not None: explains = res

                                maskR,maskC = mask.shape

                                directions_ = [(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)] 

                                record = []; dodgy = False
                                record_w_details = []

                                
                                expected_skip = False 
                                for skip in range(1,30):
                                    
                                    di = expected_dir
                                    temprows = rows + (skip*di[0])
                                    tempcols = cols + (skip*di[1])

                                    
                                    filteredrows = []; filteredcols = []; flag_beyond = False; valid_cdts = []
                                    for m in range(len(temprows)):
                                        if temprows[m] >= 0 and temprows[m] < maskR and tempcols[m] >= 0 and tempcols[m] < maskC:
                                            filteredrows.append(temprows[m]); filteredcols.append(tempcols[m])
                                            valid_cdts.append({'old_row':rows[m],'old_col':cols[m],'new_row':temprows[m],'new_col':tempcols[m]}) 
                                        else: flag_beyond = True
                                    
                                    if len(filteredrows) == 0: continue 
                                    elif flag_beyond: 
                                        
                                        continue 
                                        

                                    

                                    new_position_mask = np.zeros_like(mask); new_cdts = []; already_partly_explained_flag = False
                                    for m in range(len(filteredrows)):
                                        if filteredrows[m] >= 0 and filteredrows[m] < maskR and filteredcols[m] >= 0 and filteredcols[m] < maskC:
                                            if explains[filteredrows[m],filteredcols[m]]==0: 
                                                new_position_mask[filteredrows[m],filteredcols[m]] = 1; new_cdts.append((filteredrows[m],filteredcols[m]))
                                            else: already_partly_explained_flag = True
                                    if already_partly_explained_flag: continue 
                                    
                                    if len(new_cdts) == 0: continue


                                    i_vals = []; o_vals = []
                                    for cdts in valid_cdts: 
                                        i_vals.append(int(map[cdts['old_row'],cdts['old_col']]))
                                        o_vals.append(int(map[cdts['new_row'],cdts['new_col']]))
                                    i_vals = np.array(i_vals); o_vals = np.array(o_vals)
                                    i_colors = list(np.unique(i_vals)); o_colors = list(np.unique(o_vals))
                                    object_maintained = True if (i_vals == o_vals).all() else False 
                                    perfect_colorchange, grey_recoloring, color_changes = check_perfect_colorchange(i_colors,o_colors, i_vals,o_vals)
                                    
                                    o_outline_mask = get_outline_border_mask(new_position_mask)
                                    o_outline_vals = map[o_outline_mask.astype(bool)]; o_outline_colors = list(np.unique(o_outline_vals))
                                    nonleaked_oborder = True if none_of_x_in_y(x=o_colors, y=o_outline_colors) else False 
                                    
                                    if mode == 'allow_colorchange':
                                        if nonleaked_oborder and (object_maintained or perfect_colorchange):
                                            expected_skip = skip
                                            break
                                    else:
                                        if nonleaked_oborder and (object_maintained):
                                            expected_skip = skip
                                            break                                

                                    

                                

                                for k in range(200):

                                    candidate_dirs = []
                                    for di in directions_:

                                        
                                        temprows = rows + (expected_skip*di[0])
                                        tempcols = cols + (expected_skip*di[1])
                                        
                                        
                                        filteredrows = []; filteredcols = []; flag_beyond = False; valid_cdts = []
                                        for m in range(len(temprows)):
                                            if temprows[m] >= 0 and temprows[m] < maskR and tempcols[m] >= 0 and tempcols[m] < maskC:
                                                filteredrows.append(temprows[m]); filteredcols.append(tempcols[m])
                                                valid_cdts.append({'old_row':rows[m],'old_col':cols[m],'new_row':temprows[m],'new_col':tempcols[m]}) 
                                            else: flag_beyond = True
                                        
                                        if len(filteredrows) == 0: continue 
                                        elif flag_beyond: 
                                            
                                            continue 
                                            

                                        

                                        new_position_mask = np.zeros_like(mask); new_cdts = []; already_partly_explained_flag = False
                                        for m in range(len(filteredrows)):
                                            if filteredrows[m] >= 0 and filteredrows[m] < maskR and filteredcols[m] >= 0 and filteredcols[m] < maskC:
                                                if explains[filteredrows[m],filteredcols[m]]==0: 
                                                    new_position_mask[filteredrows[m],filteredcols[m]] = 1; new_cdts.append((filteredrows[m],filteredcols[m]))
                                                else: already_partly_explained_flag = True
                                        if already_partly_explained_flag: continue 
                                        
                                        if len(new_cdts) == 0: continue


                                        i_vals = []; o_vals = []
                                        for cdts in valid_cdts: 
                                            i_vals.append(int(map[cdts['old_row'],cdts['old_col']]))
                                            o_vals.append(int(map[cdts['new_row'],cdts['new_col']]))
                                        i_vals = np.array(i_vals); o_vals = np.array(o_vals)
                                        i_colors = list(np.unique(i_vals)); o_colors = list(np.unique(o_vals))
                                        object_maintained = True if (i_vals == o_vals).all() else False 
                                        perfect_colorchange, grey_recoloring, color_changes = check_perfect_colorchange(i_colors,o_colors, i_vals,o_vals)
                                        o_outline_mask = get_outline_border_mask(new_position_mask)
                                        o_outline_vals = map[o_outline_mask.astype(bool)]; o_outline_colors = list(np.unique(o_outline_vals))
                                        nonleaked_oborder = True if none_of_x_in_y(x=o_colors, y=o_outline_colors) else False 
                                        
                                        if mode == 'allow_colorchange':
                                            if nonleaked_oborder and (object_maintained or perfect_colorchange):
                                                if object_maintained: candidate_dirs.append({'cand_dir':di,'is_same_color':True,'color_change':None})
                                                else: candidate_dirs.append({'cand_dir':di,'is_same_color':False,'color_change':color_changes})
                                        else:
                                            if nonleaked_oborder and (object_maintained):
                                                if object_maintained: candidate_dirs.append({'cand_dir':di,'is_same_color':True,'color_change':None})
                                                else: candidate_dirs.append({'cand_dir':di,'is_same_color':False,'color_change':color_changes})                                    


                                        

                                    


                                    if k == 0:
                                        if expected_dir in [_['cand_dir'] for _ in candidate_dirs]: select_dir = expected_dir; 
                                        else: 
                                            
                                            break
                                    else:
                                        if len(candidate_dirs) > 1:
                                            same_color_dirs = []
                                            for cand in candidate_dirs:
                                                if cand['is_same_color']: same_color_dirs.append(cand['cand_dir'])
                                            if len(same_color_dirs) > 1: 
                                                
                                                if expected_dir in [_['cand_dir'] for _ in candidate_dirs]: select_dir = expected_dir; 
                                                else: 
                                                    
                                                    break
                                            elif len(same_color_dirs) == 1: select_dir = same_color_dirs[0]; 
                                            elif len(same_color_dirs) == 0:
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                
                                                if expected_dir in [_['cand_dir'] for _ in candidate_dirs]: select_dir = expected_dir; 
                                                else: 
                                                    
                                                    break
                                                
                                        elif len(candidate_dirs) == 1: select_dir = candidate_dirs[0]['cand_dir']; 
                                        elif len(candidate_dirs) == 0: 
                                            
                                            break
                                        
                                    


                                    
                                    for cand_ in candidate_dirs:
                                        if cand_['cand_dir'] == select_dir:
                                            select_is_same_color = cand_['is_same_color']
                                            select_color_change = cand_['color_change']
                                    if len(record)==0: record.append([select_dir,1]); record_w_details.append([select_dir,select_is_same_color,select_color_change])
                                    else:
                                        if record[-1][0] == select_dir: record[-1][1]+=1; record_w_details.append([select_dir,select_is_same_color,select_color_change])
                                        else: record.append([select_dir,1]); record_w_details.append([select_dir,select_is_same_color,select_color_change])

                                    

                                    
                                    temprows = rows + (expected_skip*select_dir[0]); tempcols = cols + (expected_skip*select_dir[1])

                                    
                                    res = safe_assign_rc(temprows,tempcols,explains,1,allow_spillover=False)
                                    if res is not None: explains = res

                                    expected_dir = select_dir 

                                    rows = temprows; cols = tempcols 

                                if record != []: 
                                    
                                    opt.append({'start_dirn':expected_dir,'skip':expected_skip,'record':record_w_details})
                            opts.append(opt)
                            return opts

                        def detect_fill_A(curr_map, curr_mask, o_grid, o_region, o_masko):

                            opts = []

                            
                            
                            
                            flag = True; opt = [{'fill_type':'full_border','fill_color':None},{'fill_type':'partial_border','fill_color':None}]
                            
                            filled = binary_fill_holes(curr_mask)
                            holes_mask = np.logical_and(filled, np.logical_not(curr_mask)).astype(int)
                            if np.sum(holes_mask) == 0: flag = False
                            
                            labeled_array = get_contiguous_regions(holes_mask,0,diagonal_connections_allowedQ=False,colourblind_spatial_contiguity_mode=False)
                            for hole_n in range(1,np.max(labeled_array)+1):
                                hole_mask = (labeled_array==hole_n).astype(int)
                                hole_map  = np.where(hole_mask, o_grid, 0)
                                
                                border_mask = get_outline_border_mask(hole_mask, px_border = 1) & ~holes_mask
                                if np.sum(border_mask & ~curr_mask) == 0: fill_type = 'full_border' 
                                else: fill_type = 'partial_border' 
                                fill_colors = get_colors_of_obj(hole_mask, hole_map)
                                if len(fill_colors) > 1: flag = False 
                                
                                if fill_type == 'full_border':
                                    if opt[0]['fill_color'] is None: opt[0]['fill_color'] = fill_colors[0]
                                    elif opt[0]['fill_color'] == fill_colors[0]: pass 
                                    else: flag = False 
                                if fill_type == 'partial_border':
                                    if opt[1]['fill_color'] is None: opt[1]['fill_color'] = fill_colors[0]
                                    elif opt[1]['fill_color'] == fill_colors[0]: pass 
                                    else: flag = False 
                                
                                
                            if flag: opts.append(opt)

                            return opts

                        def detect_fill_B(curr_map, curr_mask, o_grid, o_region, o_masko):
                            opts = detect_fill_A(curr_map, curr_mask, o_grid, o_region, o_masko)
                            return opts
                        
                        # detect_extension_single

                        def detect_extension_single_new(curr_map, curr_mask, o_grid, o_region):
                            
                            

                            global gg
                            gg+=1
                            ogrid = o_grid
                            currmask = curr_mask
                            igrid = i_grids[gridn] 


                            
                            i_bordermask = get_outline_border_mask(currmask,1)
                            
                            mod_i_grid = np.where(curr_mask, curr_map, igrid)
                            i_bordercolors = get_colors_of_obj(i_bordermask,mod_i_grid)
                            iobj_igrid_colors = get_colors_of_obj(curr_mask, mod_i_grid)
                            iobj_currmap_colors = get_colors_of_obj(curr_mask, curr_map)
                            i_bordercolors_ = list(mod_i_grid[i_bordermask==1])
                            commonest_i_bordercolor = max(set(i_bordercolors_), key=i_bordercolors_.count) if len(i_bordercolors_)>0 else None
                            
                            
                            if currmask.shape != ogrid.shape: print('Error')


                            opt1 = ogrid.copy(); rows,cols = np.where(currmask==1) 
                            opt1[rows,cols] = -99 
                            o_objmasks_1 = get_contiguous_regions(opt1,background_color=-99,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)
                            ipt1 = igrid.copy(); rows,cols = np.where(currmask==1) 
                            ipt1[rows,cols] = -99
                            i_objmasks_1 = get_contiguous_regions(ipt1,background_color=-99,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)

                            all_candidates = {}
                            for D in range(8):
                                directions =  ['S','SW','SE','W','E','N','NW','NE']
                                directions_ = [(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)]   
                                dir = directions[D]; dir_tuple = directions_[D] 

                                
                                mask_in_dir, cdts_of_line_start_and_obj_end = mask_in_direction(currmask,dir) 
                                unique_to_verify_dir, anti_mask = get_mask_unique_to_verify_dir(dir,currmask) 
                                linestart_cdts_all = [cdt_linestart for cdt_linestart, cdt_edgeend in cdts_of_line_start_and_obj_end]
                                
                                orthogonalbands = bands_in_dir(currmask.shape, orthogonal_dirn(dir_tuple))
                                linestart_cdts_all_bands = [orthogonalbands[cdt] for cdt in linestart_cdts_all]
                                ordered_linestart_cdts = [v for _, v in sorted(zip(linestart_cdts_all_bands, linestart_cdts_all))]
                                samedirbands = bands_in_dir(currmask.shape, dir_tuple)
                                if len(ordered_linestart_cdts) == 0: continue

                                extmodes = ['fullwidth','central'] if len(ordered_linestart_cdts) > 1 else ['central']

                                candidates = []

                                for EXTMODE in extmodes:

                                    

                                    if EXTMODE == 'central':
                                        middle_cdt = ordered_linestart_cdts[len(ordered_linestart_cdts)//2]
                                        region_mask = np.zeros_like((currmask))
                                        r,c = middle_cdt 
                                        for k in range(30):
                                            if r < 0 or r >= currmask.shape[0] or c < 0 or c >= currmask.shape[1]: break
                                            else:
                                                region_mask[r,c] = 1
                                                r += dir_tuple[0]
                                                c += dir_tuple[1] 

                                        initial_region_mask = np.zeros_like((currmask))
                                        r,c = middle_cdt
                                        initial_region_mask[r,c] = 1

                                    elif EXTMODE == 'fullwidth':
                                        region_mask = np.zeros_like((currmask))
                                        for n,cdt_linestart in enumerate(ordered_linestart_cdts): 
                                            r,c = cdt_linestart
                                            for k in range(30):
                                                if r < 0 or r >= currmask.shape[0] or c < 0 or c >= currmask.shape[1]: break
                                                else:
                                                    region_mask[r,c] = 1
                                                    r += dir_tuple[0]
                                                    c += dir_tuple[1] 

                                        initial_region_mask = np.zeros_like((currmask))
                                        for n,cdt_linestart in enumerate(ordered_linestart_cdts):
                                            r,c = cdt_linestart
                                            initial_region_mask[r,c] = 1

                                    relevant_bands = np.unique(samedirbands[region_mask==1]).astype(int)

                                    
                                    region_pxls_of_orthogleakyobjs = np.zeros_like(region_mask) 
                                    border_pxls_of_orthogleakyobjs = np.zeros_like(region_mask) 
                                    for b in relevant_bands:
                                        currbandmask = (samedirbands==b).astype(int) 
                                        curractiveregion = currbandmask & region_mask
                                        currobjs = [int(_) for _ in     np.unique(o_objmasks_1[curractiveregion==1])   ] 

                                        for obj in currobjs:
                                            objmask = (o_objmasks_1==obj).astype(int)
                                            currsliver = (objmask & currbandmask) 
                                            
                                            immediateborder = get_outline_border_mask(curractiveregion)
                                            isleak = True if np.sum(currsliver & immediateborder) > 0 else False 
                                            if isleak:
                                                curractivesliver = curractiveregion & objmask
                                                rows,cols = np.where(curractivesliver==1)
                                                region_pxls_of_orthogleakyobjs[rows,cols] = 1

                                                temp = currsliver & immediateborder
                                                rows,cols = np.where(temp==1)
                                                border_pxls_of_orthogleakyobjs[rows,cols] = 1

                                    
                                    
                                    region_pxls_of_atallleakyobjs = np.zeros_like(region_mask)  
                                    for b in relevant_bands:
                                        currbandmask = (samedirbands==b).astype(int) 
                                        curractiveregion = currbandmask & region_mask
                                        currobjs = [int(_) for _ in     np.unique(o_objmasks_1[curractiveregion==1])   ] 

                                        for obj in currobjs:
                                            objmask = (o_objmasks_1==obj).astype(int)
                                            currsliver = (objmask & currbandmask) 

                                            
                                            
                                            border_pxls_of_atallleakyobjs = (objmask & (anti_mask==0).astype(int) & get_outline_border_mask(region_mask))
                                            objleaked = False if np.sum(border_pxls_of_atallleakyobjs & ~region_mask) == 0 else True
                                            temp = (border_pxls_of_atallleakyobjs & ~region_mask)
                                            

                                            

                                            if objleaked and np.sum(temp & ~border_pxls_of_orthogleakyobjs)>0:
                                                curractivesliver = curractiveregion & objmask
                                                rows,cols = np.where(curractivesliver==1)
                                                region_pxls_of_atallleakyobjs[rows,cols] += 1     

                                    leak_mask = region_pxls_of_orthogleakyobjs*2+region_pxls_of_atallleakyobjs 


                                    

                                    

                                    
                                    endchanges = [] 
                                    for endband in range(1,len(relevant_bands)):
                                        map_before = ogrid[((samedirbands==(relevant_bands[endband]-1)).astype(int) & region_mask)==1]
                                        map_at = ogrid[((samedirbands==(relevant_bands[endband])).astype(int) & region_mask)==1]
                                        if not are_two_identical(map_before, map_at): endchanges.append(endband); 
                                        

                                        
                                        existing_map_before = mod_i_grid[((samedirbands==(relevant_bands[endband]-1)).astype(int) & region_mask)==1]
                                        existing_map_at = mod_i_grid[((samedirbands==(relevant_bands[endband])).astype(int) & region_mask)==1]
                                        if np.any([currcolor in i_bordercolors for currcolor in existing_map_before]): 
                                            if np.all([currcolor not in i_bordercolors for currcolor in existing_map_at]): 
                                                endchanges.append(endband)

                                    if len(relevant_bands) > 1: endchanges.append(endband+1) 
                                    else: endchanges.append(1)

                                    
                                    

                                    matched_initial_region = False
                                    for endband in endchanges:
                                        
                                        subset_relevant_bands = relevant_bands[:endband]

                                        fs=[];ls=[];     new_colors = []; prevcolors = []; ncb=[];   cum_activeregion = np.zeros_like(region_mask); 
                                        colors_upto_initial_region = []; all_colors = []
                                        ls_cdts = []

                                        
                                        for b in subset_relevant_bands:
                                            currbandmask = (samedirbands==b).astype(int) 
                                            curractiveregion = currbandmask & region_mask                       ; rows_,cols_ = np.where(curractiveregion==1); ls_cdts.append([rows_,cols_])
                                            
                                            currobjs = [int(_) for _ in     np.unique(o_objmasks_1[curractiveregion==1])   ]
                                            isfull = 1 if len(currobjs) == 1 else 0
                                            isnonleak = 1 if np.max(leak_mask[curractiveregion==1]) == 0 else 0 
                                            fs.append(isfull); ls.append(isnonleak)
                                            if isnonleak == 1:
                                                active_nonleak_region = curractiveregion & (leak_mask==0).astype(int)
                                                currcolors = get_colors_of_obj(active_nonleak_region, ogrid)
                                                
                                                for color in currcolors:
                                                    if color not in prevcolors: new_colors.append(color); ncb.append(b)     
                                                prevcolors = currcolors 
                                            
                                            cum_activeregion = cum_activeregion | ((curractiveregion==1).astype(int))
                                            
                                            if not matched_initial_region:
                                                if are_two_identical(initial_region_mask, cum_activeregion): matched_initial_region = True; colors_upto_initial_region = copy.deepcopy(new_colors)

                                            all_colors.extend(get_colors_of_obj(curractiveregion, ogrid))

                                        premature_modifier = 0.5 if not matched_initial_region else 1

                                        endinleak_modifier = 0.5 if 0 not in leak_mask[curractiveregion==1] else 1

                                        
                                        
                                        
                                        activemask_ = cum_activeregion & (leak_mask==0).astype(int)
                                        temp_ = np.where(activemask_, ogrid, -99)
                                        activeobjs_ = get_contiguous_regions(temp_,background_color=-99,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)
                                        flag_overlap_preiobj = False
                                        for n_ in range(1,np.max(activeobjs_)+1):
                                            obj_mask_ = (activeobjs_==n_).astype(int)
                                            for iobj__ in global_parsings[gridn]['i']:
                                                if are_two_identical(global_parsings[gridn]['i'][iobj__]['maskv'],obj_mask_):
                                                    flag_overlap_preiobj = True
                                                    
                                        
                                        overlap_preiobj_modified = 0.5 if flag_overlap_preiobj else 1

                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        if len(new_colors) < 5: colorsc = 1
                                        else: colorsc = 0.5
                                        for colr in list(set(new_colors)): 
                                            colorsc = colorsc * (1/(new_colors.count(colr)))
                                        
                                        
                                        if is_any_x_in_y(x=i_bordercolors,y=new_colors): colorsc = colorsc * 0.7 

                                        if is_any_x_in_y(x=i_bordercolors,y=all_colors): colorsc = colorsc * 0.95 



                                        cum_fs = fs; cum_ls = ls
                                        if len(cum_fs) > 2 and 0 not in cum_fs[:int(len(cum_fs)/2)]: cfscore = 1
                                        elif len(cum_fs) <= 2 and 0 not in cum_fs: cfscore = 1
                                        else: cfscore = np.mean(cum_fs)
                                        if len(cum_ls) > 2 and cum_ls.count(1) > 2: clscore = 1 
                                        elif len(cum_ls) <= 2 and cum_ls.count(1) == len(cum_ls): clscore = 1
                                        else: clscore = np.mean(cum_ls)
                                        
                                        
                                        
                                        
                                        
                                        activeleakmask = (leak_mask!=0).astype(int) & cum_activeregion 

                                        
                                        
                                        

                                        

                                        


                                        
                                        
                                        oextgrid = ogrid.copy(); rows,cols = np.where(cum_activeregion==0) 
                                        oextgrid[rows,cols] = -99 

                                        
                                        nonleaky_activeregion = ((cum_activeregion==1).astype(int) & (activeleakmask==0).astype(int)).astype(int)
                                        activeleak_pseudoobjs = get_contiguous_regions(activeleakmask,background_color=0,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)
                                        for alpo in range(1,np.amax(activeleak_pseudoobjs)+1):
                                            po_mask = (activeleak_pseudoobjs==alpo).astype(int)
                                            
                                            rtofill,ctofill = np.where(po_mask==1)
                                            valid_neighbors_mask = ((get_outline_border_mask(po_mask,1)==1).astype(int) & (nonleaky_activeregion==1).astype(int)).astype(int)
                                            if np.sum(valid_neighbors_mask)==0: pass
                                            else:
                                                valid_neighbor_colors = ogrid[valid_neighbors_mask==1]
                                                oextgrid[rtofill,ctofill] = valid_neighbor_colors[0] 
                                        

                                        oext_objs = get_contiguous_regions(oextgrid,background_color=-99,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)
                                        
                                        newobjs = []; newobj_b = []; newobj_colors = []; prevobjs = []
                                        for b in subset_relevant_bands:
                                            currbandmask = (samedirbands==b).astype(int) 
                                            curractiveregion = currbandmask & region_mask
                                            currobjs = [int(_) for _ in np.unique(oext_objs[curractiveregion==1]) if int(_)!=0] 
                                            for cobj in currobjs:
                                                if cobj not in prevobjs:
                                                    newobjs.append(cobj)
                                                    newobj_b.append(b)
                                                    newobj_colors.append(get_colors_of_obj((oext_objs==cobj).astype(int),oextgrid))
                                            prevobjs = currobjs
                                        
                                        try: bandskip = subset_relevant_bands[1]-subset_relevant_bands[0]
                                        except: bandskip = 1


                                        rule_info = []; first_igrid_color = None; first_currmap_color = None
                                        for n in range(len(newobjs)):
                                            
                                            
                                            
                                            
                                            if n == 0:
                                                why_began = ['first_obj']

                                                prevb = newobj_b[n]-bandskip 
                                                currbandmask = (samedirbands==prevb).astype(int) 
                                                currmask_activeregion = currbandmask & currmask 
                                                why_this_color = []
                                                this_color = newobj_colors[n]
                                                if np.sum(currmask_activeregion)!=0:
                                                    rows, cols = np.where(currmask_activeregion==1)
                                                    midpxl = (rows[len(rows)//2], cols[len(cols)//2])
                                                    currmap_color = [curr_map[midpxl]] 
                                                    igrid_color = [igrid[midpxl]]
                                                    first_igrid_color = igrid_color; first_currmap_color = currmap_color 
                                                    if this_color == igrid_color: why_this_color.append('i_grid_color_of_prev')
                                                    if this_color == currmap_color: why_this_color.append('curr_map_color_of_prev')
                                                why_this_color.append(('hyperp_color',this_color))

                                            else:
                                                why_began = ['coz_prev_ended']

                                                prevb = newobj_b[n]-bandskip
                                                currbandmask = (samedirbands==prevb).astype(int) 
                                                curractiveregion = currbandmask & region_mask
                                                why_this_color = []
                                                this_color = newobj_colors[n]
                                                if np.sum(curractiveregion)!=0:
                                                    igrid_colors = get_colors_of_obj(curractiveregion,igrid)
                                                    if this_color == igrid_colors: why_this_color.append('i_grid_color_of_prev')
                                                why_this_color.append(('hyperp_color',this_color))
                                            
                                            

                                            why_this_len = []
                                            

                                            
                                            thisobj_bs = []
                                            currshape_regionmask = (oext_objs==newobjs[n]).astype(int)
                                            for b in subset_relevant_bands:
                                                currbandmask = (samedirbands==b).astype(int) 
                                                curractiveregion = currbandmask & currshape_regionmask
                                                if np.sum(curractiveregion)>0:
                                                    thisobj_bs.append(b) 

                                            
                                            edgecap = False
                                            if n == len(newobjs)-1:
                                                reached_edge = True if relevant_bands[-1] in subset_relevant_bands else False
                                                if reached_edge: 
                                                    why_this_len.append('reached_edge')
                                                    edgecap = True

                                            
                                            why_this_len.append(('hyperp_length',len(thisobj_bs)))

                                            if not edgecap:
                                                
                                                nextb = thisobj_bs[-1]+bandskip
                                                currbandmask = (samedirbands==nextb).astype(int) 
                                                curractiveregion = currbandmask & region_mask
                                                if np.sum(curractiveregion)!=0:
                                                    nextband_igrid_colors = get_colors_of_obj(curractiveregion,igrid)
                                                    thisobj_igrid_colors = get_colors_of_obj((oext_objs==newobjs[n]).astype(int),igrid)

                                                    
                                                    isnovelcolor = False
                                                    for color_ in nextband_igrid_colors:
                                                        if color_ not in thisobj_igrid_colors:
                                                            isnovelcolor = True
                                                    
                                                    if isnovelcolor:
                                                        

                                                        
                                                        why_this_len.append(('encounters_this_specific_hyperp_color',nextband_igrid_colors))
                                                        
                                                        if commonest_i_bordercolor is not None and np.any([colr != commonest_i_bordercolor for colr in nextband_igrid_colors]): why_this_len.append('encounters_a_non_commonestbordercolor')
                                                        
                                                        if np.any([colr not in i_bordercolors for colr in nextband_igrid_colors]): why_this_len.append('encounters_a_non_border_color')
                                                        
                                                        if np.any([colr in iobj_igrid_colors for colr in nextband_igrid_colors]): why_this_len.append('encounters_an_igrid_color')
                                                        
                                                        if np.any([colr in iobj_currmap_colors for colr in nextband_igrid_colors]): why_this_len.append('encounters_a_currmap_color')
                                            
                                                        

                 

                                            
                                            rule_info.append([why_began, why_this_color, why_this_len])

                                        candidates.append({'dir':dir,'rule_info':rule_info,'score':float(cfscore*clscore*colorsc*premature_modifier*endinleak_modifier*overlap_preiobj_modified)+(len(subset_relevant_bands)/1000),'region':cum_activeregion,'activeleakmask':activeleakmask,'dets':[EXTMODE,fs,ls,new_colors,  cfscore, clscore,colorsc, premature_modifier, endinleak_modifier,overlap_preiobj_modified,ls_cdts],'colors_upto_initial_region':colors_upto_initial_region,'extmode':EXTMODE,'subsetlen':len(subset_relevant_bands),'subsetreachededge':True if relevant_bands[-1] in subset_relevant_bands else False})

                                    
                                sorted_candidates = sorted(candidates, key=lambda d: d['score'], reverse=True)

                                
                               
                        
                                

                                all_candidates[dir] = sorted_candidates




                           


                            for dir_ in directions: 
                                if dir_ not in all_candidates: all_candidates[dir_] = []

                            cutoff = 0.75
                            chosen_cands_v1 = []
                            v1region = np.zeros_like(ogrid); v1leakmask = np.zeros_like(ogrid); extmodes = {}; rejected_colors = []
                            for dir in directions:
                                if len(all_candidates[dir]) == 0: continue
                                firstcand = all_candidates[dir][0]
                                if firstcand['score'] >= cutoff:
                                    chosen_cands_v1.append(firstcand)
                                    v1region += firstcand['region']
                                    v1leakmask += firstcand['activeleakmask']
                                    extmodes[dir] = firstcand['extmode']
                                else: rejected_colors.extend(firstcand['colors_upto_initial_region'])


                            if np.amax(v1region)>1 or np.amax(v1leakmask)>1: print("ERROR")
                            
                            
                            
                            

                            cutoff = 0.75
                            v2region = np.zeros_like(ogrid); v2leakmask = np.zeros_like(ogrid); extmodes = {}
                            v2cands = {}
                            for dir in directions:
                                if len(all_candidates[dir]) == 0: v2cands[dir] = []; continue
                                v2cands_ = []
                                for candn in range(len(all_candidates[dir])):
                                    cand = all_candidates[dir][candn]
                                    subsetlen = cand['subsetlen']
                                    subsetreachededge = cand['subsetreachededge']
                                    extmode = cand['extmode']
                                    initial_colors_in_dir = cand['colors_upto_initial_region']
                                    if is_any_x_in_y(x=rejected_colors,y=initial_colors_in_dir): print('Skipping'); continue
                                    if cand['score'] >= cutoff:
                                        v2cands_.append([candn,subsetlen,subsetreachededge,extmode])
                                        
                                        
                                        
                                        
                                v2cands[dir] = v2cands_

                            
                            

                            def assign_symmetries_v2cands(v2cands):

                                def all_but_one_identical(thelist):
                                    flag = False; theidentical = None
                                    for ix in range(len(thelist)):
                                        thelistcopy = copy.deepcopy(thelist)
                                        thelistcopy.pop(ix)
                                        if are_all_identical(thelistcopy):
                                            theidentical = thelistcopy[0]
                                            flag = True
                                            break
                                    if flag: ix_of_nonidentical = ix; val = theidentical
                                    else: ix_of_nonidentical = None; val = None
                                    return ix_of_nonidentical, val

                                fixedm = False; cand_set_m = {'N':0,'S':0,'E':0,'W':0}
                                if 0 not in [len(v2cands[maindir]) for maindir in ['N','S','E','W']]:
                                    maindirs_topcand_reach_edge = [v2cands[maindir][0][2] for maindir in ['N','S','E','W']]; cand_set1 = {'N':0,'S':0,'E':0,'W':0}
                                    if maindirs_topcand_reach_edge.count(True) == 4: pass
                                    elif maindirs_topcand_reach_edge.count(True) == 3:
                                        insufficient_dir = ['N','S','E','W'][maindirs_topcand_reach_edge.index(False)]
                                        for candn in range(len(v2cands[insufficient_dir])):
                                            if v2cands[insufficient_dir][candn][2] == True:
                                                cand_set1[insufficient_dir] = v2cands[insufficient_dir][candn][0]
                                                fixedm = True; cand_set_m = cand_set1
                                                
                                                break

                                    if not fixedm:
                                        maindirs_topcand_len = [v2cands[maindir][0][1] for maindir in ['N','S','E','W']]; cand_set3 = {'N':0,'S':0,'E':0,'W':0}
                                        ix_of_nonidentical, val_of_identicals = all_but_one_identical(maindirs_topcand_len)
                                        if are_all_identical(maindirs_topcand_len): pass
                                        elif ix_of_nonidentical is not None:
                                            insufficient_dir = ['N','S','E','W'][ix_of_nonidentical]
                                            for candn in range(len(v2cands[insufficient_dir])):
                                                if v2cands[insufficient_dir][candn][1] == val_of_identicals:
                                                    cand_set3[insufficient_dir] = v2cands[insufficient_dir][candn][0]
                                                    fixedm = True; cand_set_m = cand_set3
                                                    break

                                    if not fixedm:
                                        maindirs_topcand_ext = [v2cands[maindir][0][3] for maindir in ['N','S','E','W']]; cand_set5 = {'N':0,'S':0,'E':0,'W':0}
                                        ix_of_nonidentical, val_of_identicals = all_but_one_identical(maindirs_topcand_ext)
                                        if are_all_identical(maindirs_topcand_ext): pass
                                        elif ix_of_nonidentical is not None:
                                            insufficient_dir = ['N','S','E','W'][ix_of_nonidentical]
                                            for candn in range(len(v2cands[insufficient_dir])):
                                                if v2cands[insufficient_dir][candn][3] == val_of_identicals:
                                                    cand_set5[insufficient_dir] = v2cands[insufficient_dir][candn][0]
                                                    fixedm = True; cand_set_m = cand_set5
                                                    break    


                                fixeda = False; cand_set_a = {'NW':0,'SW':0,'NE':0,'SE':0}
                                if 0 not in [len(v2cands[altdir]) for altdir in ['NW','SW','NE','SE']]:
                                    altdirs_topcand_reach_edge = [v2cands[altdir][0][2] for altdir in ['NW','SW','NE','SE']]; cand_set2 = {'NW':0,'SW':0,'NE':0,'SE':0}
                                    if altdirs_topcand_reach_edge.count(True) == 4: pass
                                    elif altdirs_topcand_reach_edge.count(True) == 3:
                                        insufficient_dir = ['NW','SW','NE','SE'][altdirs_topcand_reach_edge.index(False)]
                                        for candn in range(len(v2cands[insufficient_dir])):
                                            if v2cands[insufficient_dir][candn][2] == True:
                                                cand_set2[insufficient_dir] = v2cands[insufficient_dir][candn][0]
                                                fixeda = True; cand_set_a = cand_set2
                                                break

                                    if not fixeda:
                                        altdirs_topcand_len = [v2cands[altdir][0][1] for altdir in ['NW','SW','NE','SE']]; cand_set4 = {'NW':0,'SW':0,'NE':0,'SE':0}
                                        ix_of_nonidentical, val_of_identicals = all_but_one_identical(altdirs_topcand_len)
                                        if are_all_identical(altdirs_topcand_len): pass
                                        elif ix_of_nonidentical is not None:
                                            insufficient_dir = ['NW','SW','NE','SE'][ix_of_nonidentical]
                                            for candn in range(len(v2cands[insufficient_dir])):
                                                if v2cands[insufficient_dir][candn][1] == val_of_identicals:
                                                    cand_set4[insufficient_dir] = v2cands[insufficient_dir][candn][0]
                                                    fixeda = True; cand_set_a = cand_set4
                                                    break

                                    if not fixeda:
                                        altdirs_topcand_len = [v2cands[altdir][0][3] for altdir in ['NW','SW','NE','SE']]; cand_set6 = {'NW':0,'SW':0,'NE':0,'SE':0}
                                        ix_of_nonidentical, val_of_identicals = all_but_one_identical(altdirs_topcand_len)
                                        if are_all_identical(altdirs_topcand_len): pass
                                        elif ix_of_nonidentical is not None:
                                            insufficient_dir = ['NW','SW','NE','SE'][ix_of_nonidentical]
                                            for candn in range(len(v2cands[insufficient_dir])):
                                                if v2cands[insufficient_dir][candn][3] == val_of_identicals:
                                                    cand_set6[insufficient_dir] = v2cands[insufficient_dir][candn][0]
                                                    fixeda = True; cand_set_a = cand_set6
                                                    break

                                full_cand_set = copy.deepcopy(cand_set_m); full_cand_set.update(cand_set_a)
                                return full_cand_set

                            full_cand_set = assign_symmetries_v2cands(v2cands)

                            chosen_cands_v3 = []
                            v3region = np.zeros_like(ogrid); v3leakmask = np.zeros_like(ogrid); extmodes = {}; sublens = {}; subedges = {}; rule_infos = {}
                            for dir in directions:
                                if len(v2cands[dir]) == 0: continue
                                chosencand = all_candidates[dir][full_cand_set[dir]];           chosen_cands_v3.append(chosencand)
                                v3region += chosencand['region']
                                v3leakmask += chosencand['activeleakmask']
                                extmodes[dir] = chosencand['extmode']
                                sublens[dir] = chosencand['subsetlen']
                                subedges[dir] = chosencand['subsetreachededge']
                                rule_infos[dir] = chosencand['rule_info']

                            if np.amax(v3region)>1 or np.amax(v3leakmask)>1: print("ERROR")
                            
                            
                            
                            



                            
                            
                            
                            is_connection3 = False
                            
                            if len([dirnkey for dirnkey in extmodes]) == 1:
                                dirn = [dirnkey for dirnkey in extmodes][0]
                                rule_info_ = rule_infos[dirn]
                                if len(rule_info_)==1: 
                                    opp_dirn = ['S','SW','SE','W','E','N','NW','NE'][['N','NE','NW','E','W','S','SE','SW'].index(dirn)]
                                    maskinextdir, temp_cdts1 = mask_in_direction(curr_mask, dirn)
                                    for iobj__ in global_parsings[gridn]['i']:
                                        if iobj_ == iobj__: continue
                                        maskinoppdirfromiobj__, temp_cdts = mask_in_direction(global_parsings[gridn]['i'][iobj__]['mask'],opp_dirn)
                                        combimask = (maskinoppdirfromiobj__ | global_parsings[gridn]['i'][iobj__]['mask']) & maskinextdir
                                        if np.sum((global_parsings[gridn]['i'][iobj__]['mask']==1)&(maskinextdir==0))==0:
                                            if are_two_identical(combimask, v3region):
                                                
                                                is_connection3 = True
                                                connection_region_incl_objs3 = ( (v3region==1) | (curr_mask==1) ).astype(int) 
                                                connection_region_excl_objs3 = ( (v3region==1) & (global_parsings[gridn]['i'][iobj__]['mask']==0) ).astype(int)
                                                new_iobj_pair3 = sorted([iobj_,iobj__])
                                                color_rule_info3 = rule_info_[0][1] 
                            
                            is_connection1 = False
                            
                            if len([dirnkey for dirnkey in extmodes]) == 1:
                                dirn = [dirnkey for dirnkey in extmodes][0]
                                rule_info_ = rule_infos[dirn]
                                if len(rule_info_)==1: 
                                    opp_dirn = ['S','SW','SE','W','E','N','NW','NE'][['N','NE','NW','E','W','S','SE','SW'].index(dirn)]
                                    maskinextdir, temp_cdts1 = mask_in_direction(curr_mask, dirn)
                                    for iobj__ in global_parsings[gridn]['i']:
                                        if iobj_ == iobj__: continue
                                        maskinoppdirfromiobj__, temp_cdts = mask_in_direction(global_parsings[gridn]['i'][iobj__]['mask'],opp_dirn)
                                        combimask = (maskinoppdirfromiobj__ | global_parsings[gridn]['i'][iobj__]['mask']) & maskinextdir
                                        if np.sum((global_parsings[gridn]['i'][iobj__]['mask']==1)&(maskinextdir==0))==0:
                                            if are_two_identical(combimask, v1region):
                                                
                                                is_connection1 = True
                                                connection_region_incl_objs1 = ( (v1region==1) | (curr_mask==1) ).astype(int) 
                                                connection_region_excl_objs1 = ( (v1region==1) & (global_parsings[gridn]['i'][iobj__]['mask']==0) ).astype(int)
                                                new_iobj_pair1 = sorted([iobj_,iobj__])
                                                color_rule_info1 = rule_info_[0][1] 
                            


                            
                            opts = []
                            
                            
                            
                            
                            
                            
                            
                            
                            if not is_connection1: opts.append({'ext_fn_characterisation':chosen_cands_v1,'type':'extension'}) 
                            elif is_connection1: opts.append({'ext_fn_characterisation':chosen_cands_v1,'type':'connection','color_rule_info':color_rule_info1,'new_iobj_pair':new_iobj_pair1, 'connection_region_incl_objs': connection_region_incl_objs1, 'connection_region_excl_objs':connection_region_excl_objs1})
                            
                            
                            if not is_connection3: opts.append({'ext_fn_characterisation':chosen_cands_v3,'type':'extension'}) 
                            elif is_connection3: opts.append({'ext_fn_characterisation':chosen_cands_v3,'type':'connection','color_rule_info':color_rule_info3,'new_iobj_pair':new_iobj_pair3, 'connection_region_incl_objs': connection_region_incl_objs3, 'connection_region_excl_objs':connection_region_excl_objs3})
                            
                            

                            
                            return opts


                        def detect_wholegrid_stampings(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko):
                            opts = []

                            for stamp_mode in ['overwrite','do_not_overwrite']:
                                for color_to_stamp_around in np.unique(curr_map):
                                    for color_to_stamp_with in np.unique(o_grid): 

                                        output_map = curr_map.copy()
                                        rows,cols = np.where(curr_map==color_to_stamp_around)
                                        for m in range(len(rows)):
                                            if curr_mask[rows[m],cols[m]]==1:
                                                for roff,coff in [(1,1),(-1,-1),(1,-1),(-1,1)]:
                                                    r = rows[m]+roff; c = cols[m]+coff
                                                    if r >= 0 and c >= 0 and r <= curr_map.shape[0]-1 and c <= curr_map.shape[1]-1:
                                                        
                                                        if stamp_mode == 'overwrite': output_map[r,c] = color_to_stamp_with
                                                        elif stamp_mode == 'do_not_overwrite':
                                                            if output_map[r,c] != color_to_stamp_around:
                                                                output_map[r,c] = color_to_stamp_with
                                        
                                        

                                        if are_two_identical(o_grid, output_map):
                                            opts.append({'color_to_stamp_around':color_to_stamp_around,'color_to_stamp_with':color_to_stamp_with,'stamp_mode':stamp_mode})                    
                                            
                                            icolors = get_colors_of_obj(curr_mask,curr_map) 
                                            if len(icolors)==2 and color_to_stamp_around in icolors:
                                                icolors.remove(color_to_stamp_around)
                                                color_not_to_stamp_around = icolors[0]
                                                opts.append({'color_not_to_stamp_around':color_not_to_stamp_around,'color_to_stamp_with':color_to_stamp_with,'stamp_mode':stamp_mode})                    

                            return opts




                        tempflag = True if chain_branchlist[a0][c0]['params'] == {} else False 
                        params_list = [{}]
                        if tempflag:
                            if fn_name == 'no_change': params_list = [{}]
                            if fn_name == 'movt':
                                
                                if o_mode == 'oobj': 
                                    movt_opts = detect_movt_A(curr_mask, curr_o_region)
                                    if len(movt_opts) > 10: movt_opts = movt_opts[:10]
                                elif o_mode == 'oreg': 
                                    movt_opts = detect_movt_B(curr_mask, curr_o_region)
                                    
                                    if len(movt_opts) > 10: movt_opts = movt_opts[:10]
                                params_list = [{'move_rc':rc} for rc in movt_opts]
                            if fn_name == 'recolor': 
                                try:
                                    if o_mode == 'oobj': recolor_opts = detect_recolor_A(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko)
                                    elif o_mode == 'oreg': recolor_opts = detect_recolor_B(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko)
                                    params_list = [{'color_changes':cc} for cc in recolor_opts]
                                except:
                                    params_list = [{'color_changes':None}]
                                
                            if fn_name == 'masking':
                                if o_mode == 'oobj': masking_opt = detect_masking_A(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko)
                                elif o_mode == 'oreg': masking_opt = detect_masking_B(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko)
                                
                                
                                
                                
                                if masking_opt != []: 
                                    params_list = [{'mask':masking_opt}]
                                    if len(masking_opt) != 1: print("ERROR")
                                    for masking_mask in masking_opt: 
                                        curr_o_masko = curr_o_masko | masking_mask 
                                else: print("WARNING"); params_list = [{}]

                            if fn_name == 'rotate_flip':
                                pass
                            
                            
                            if fn_name == 'flip':
                                flip_opts = detect_flip_B(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko)
                                params_list = [{'flip_about_axis':flipaboutaxis,'desired_flip_row_or_col':flipaboutroworcol} for flipaboutaxis, flipaboutroworcol in flip_opts]
                            if fn_name == 'rotate_about_center':
                                rotate_opts = detect_rotate_B(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko)
                                params_list = [{'rotation':rotation,'desired_centre':desiredcentre} for rotation,desiredcentre in rotate_opts]
                            if fn_name == 'expand':
                                expand_opts = detect_expansion_B(curr_map, curr_mask, o_grid, curr_o_region)
                                params_list = [{'w_mult':wmult,'h_mult':hmult} for wmult,hmult in expand_opts]

                            if fn_name == 'extension': 
                                

                                if curr_mask.shape != o_grid_raw.shape: print('Skip'); ext_opts = []
                                else:

                                    if o_mode == 'oobj': ext_opts = detect_extension_single_new(curr_map, curr_mask, o_grid, o_region)
                                    elif o_mode == 'oreg': ext_opts = detect_extension_single_new(curr_map, curr_mask, o_grid, o_region)

                                    

                                    params_list = [{'ext_details':_} for _ in ext_opts]

                                if len(ext_opts) == 0: params_list = [{'ext_details':None}] 

        

                            if fn_name == 'copying': 
                                if o_mode == 'oobj': copy_opts = detect_copying_A(curr_map, curr_mask, o_grid, curr_o_region)
                                elif o_mode == 'oreg': copy_opts = detect_copying_B(curr_map, curr_mask, o_grid, curr_o_region)
                                params_list = [{'copy_details':_} for _ in copy_opts]
                                if len(copy_opts) == 0: params_list = [{'copy_details':None}] 
                            if fn_name == 'pathing': pass       
                            if fn_name == 'fill':
                                if o_mode == 'oobj': fill_opts = detect_fill_A(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko)
                                elif o_mode == 'oreg': fill_opts = detect_fill_B(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko)
                                params_list = [{'fill_details':_} for _ in fill_opts]
                                if len(fill_opts) == 0: params_list = [{'fill_details':None}] 
                            if fn_name == 'wholegrid_stampings':
                                if o_mode == 'oobj': stampings_opts = detect_wholegrid_stampings(curr_map, curr_mask, o_grid, curr_o_region, curr_o_masko)
                                params_list = [{'stamp_details':_} for _ in stampings_opts]
                                if len(stampings_opts) == 0: params_list = [{'stamp_details':None}] 

                        curr_params = chain_branchlist[a0][c0]['params']
                        if (curr_params not in params_list) and tempflag:
                            prefix = chain_branchlist[a0][:c0]
                            newlist = []
                            for chain_ in chain_branchlist:
                                if chain_[:c0] == prefix:
                                    for new_params in params_list:
                                        tempmodified = list(chain_)
                                        step_copy = dict(tempmodified[c0])
                                        step_copy['params'] = new_params
                                        tempmodified[c0] = step_copy
                                        newlist.append(tempmodified)
                                else: newlist.append(list(chain_))
                            chain_branchlist = newlist
                        

                        currdict = (chain_branchlist[a0][c0])
                        params = currdict.get('params', {})
                        incache = False 
                        cache_paramlist = chain_branchlist[a0][:c0+1]
                        curr_cache = [_[0] for _ in unique_emask_paramlists]
                        for cc in range(len(curr_cache)):
                            
                            if are_two_identical(curr_cache[cc],cache_paramlist): incache = cc; break 
                        if incache is not False: 
                            curr_map, curr_mask = unique_emask_paramlists[incache][1],unique_emask_paramlists[incache][2]
                        else:
                            curr_map, curr_mask = currdict['fn'](input_map=curr_map, mask_to_transform=curr_mask,**params) 
                            unique_emask_paramlists.append([cache_paramlist,curr_map,curr_mask]) 


                        
                        if currdict['fn'].__name__ == 'extension': 
                            try:
                                for n_ in range(len(params['ext_details']['ext_fn_characterisation'])):
                                    masking_mask = params['ext_details']['ext_fn_characterisation'][n_]['activeleakmask']
                                    curr_o_masko = curr_o_masko | masking_mask 
                            except: print('Skip')






                        
                        curr_o_region = (curr_mask | curr_o_masko | curr_o_region)
                        curr_o_region_ = curr_o_region[0:0+o_grid_raw.shape[0],0:0+o_grid_raw.shape[1]]
                        curr_o_masko_ = curr_o_masko[0:0+o_grid_raw.shape[0],0:0+o_grid_raw.shape[1]]
                        curr_mask_ = curr_mask[0:0+o_grid_raw.shape[0],0:0+o_grid_raw.shape[1]]
                        curr_map_ = curr_map[0:0+o_grid_raw.shape[0],0:0+o_grid_raw.shape[1]]
                        rows,cols = np.where(curr_mask_==1)
                        curr_explained_mask = copy.deepcopy(curr_o_masko_)
                        for m in range(len(rows)):
                            if rows[m]>=0 and cols[m]>=0 and rows[m]<=o_grid_raw.shape[0]-1 and cols[m]<=o_grid_raw.shape[1]-1:
                                if curr_map_[rows[m],cols[m]] == o_grid_raw[rows[m],cols[m]]:
                                    curr_explained_mask[rows[m],cols[m]] = 1
                            else: print('ERROR')
                        is_oregion1_explained = np.all((o_region1_raw & curr_explained_mask) == o_region1_raw)
                        if is_oregion1_explained: o_mode = 'oreg' 
                        is_curroregion_explained = np.all((curr_o_region_ & curr_explained_mask) == curr_o_region_)




                        
                        
                        


                        
                        


                        c0+=1

                    
                    
                    current_run = chain_branchlist[a0][:c0+1]
                    a0+=1




                    
                    curr_mask = curr_mask[0:0+o_grid_raw.shape[0],0:0+o_grid_raw.shape[1]]
                    curr_map = curr_map[0:0+o_grid_raw.shape[0],0:0+o_grid_raw.shape[1]]
                    curr_o_masko = curr_o_masko[0:0+o_grid_raw.shape[0],0:0+o_grid_raw.shape[1]]
                    i_mask = i_mask[0:0+i_mask_raw.shape[0],0:0+i_mask_raw.shape[1]]
                    i_map = i_map[0:0+i_map_raw.shape[0],0:0+i_map_raw.shape[1]]
                    



                    tosave = False
                    if not quitflag and is_curroregion_explained and (not are_two_identical(curr_explained_mask,curr_o_masko)): tosave = True


                    


                    
        


                    
                    curr_maskv_ = ((curr_mask==1) & (curr_o_masko==0)).astype(int)
                    if np.sum(curr_maskv_)==0: tosave = False

                    if tosave:


                        
                        i_colors = get_colors_of_obj(i_mask,i_map)
                        i_shape = get_shape_of_obj(i_mask,i_map)

                        
                        
                        curr_maskv = curr_maskv_ 





                        if oobj_or_oreg_mode == 'oobj': 
                            if not are_two_identical([curr_mask,curr_map,curr_maskv],[global_parsings[gridn]['o'][oobj_]['mask'],global_parsings[gridn]['o'][oobj_]['map'],global_parsings[gridn]['o'][oobj_]['maskv']]):
                                
                                flagexists = False
                                for existing_oobj in global_parsings[gridn]['o']:
                                    if are_two_identical([curr_mask,curr_map,curr_maskv],[global_parsings[gridn]['o'][existing_oobj]['mask'],global_parsings[gridn]['o'][existing_oobj]['map'],global_parsings[gridn]['o'][existing_oobj]['maskv']]):
                                        flagexists = True; break

                                if not flagexists:
                                    current_oobj = oobj_
                                    new_oobj = create_name()
                                    global_parsings[gridn]['o'][new_oobj] = copy.deepcopy(global_parsings[gridn]['o'][current_oobj])
                                    global_parsings[gridn]['o'][current_oobj]['obj_score'] -= 0.2989
                                    global_parsings[gridn]['o'][new_oobj]['mask'] = curr_mask  
                                    global_parsings[gridn]['o'][new_oobj]['map'] = curr_map
                                    global_parsings[gridn]['o'][new_oobj]['masko'] = curr_o_masko
                                    global_parsings[gridn]['o'][new_oobj]['maskv'] = curr_maskv
                                    oobj_ = new_oobj 
                                else:
                                    current_oobj = oobj_
                                    global_parsings[gridn]['o'][current_oobj]['obj_score'] -= 0.2989
                                    new_oobj = existing_oobj
                                    oobj_ = new_oobj 

                        elif oobj_or_oreg_mode == 'oreg': 

                            flagexists = False
                            for existing_oobj in global_parsings[gridn]['o']:
                                if are_two_identical([curr_mask,curr_map,curr_maskv],[global_parsings[gridn]['o'][existing_oobj]['mask'],global_parsings[gridn]['o'][existing_oobj]['map'],global_parsings[gridn]['o'][existing_oobj]['maskv']]):
                                    flagexists = True; break

                            if not flagexists:
                                new_oobj = create_name()
                                global_parsings[gridn]['o'][new_oobj] = {}
                                global_parsings[gridn]['o'][new_oobj]['obj_score'] = 0.921 
                                global_parsings[gridn]['o'][new_oobj]['mask'] = curr_mask
                                global_parsings[gridn]['o'][new_oobj]['map'] = curr_map
                                global_parsings[gridn]['o'][new_oobj]['masko'] = curr_o_masko
                                global_parsings[gridn]['o'][new_oobj]['maskv'] = curr_maskv
                                global_parsings[gridn]['o'][new_oobj]['properties'] = {'is_straightforward_obj':False, 'parsing_description':['TEMP_newly_created_oreg_mode_oobj',0.5,None]}
                                oobj_ = new_oobj
                            else:
                                oobj_ = existing_oobj





                        serial_transforms = []; serial_params = []
                        for dict_ in current_run:

                            if dict_['fn'].__name__ == 'movt': 
                                if dict_['params']['move_rc'] == (0,0):
                                    serial_transforms.append({'type':'static'})
                                    serial_params.append({})
                                
                                    

                                else: 
                                    move_rc = dict_['params']['move_rc']
                                    r, c = move_rc
                                    move_type = 'irregular'
                                    if (r==0 and c!=0) or (c==0 and r!=0): move_type = 'maindir'
                                    elif np.abs(r) == np.abs(c) and r!=0 and c!=0: move_type = 'altdir'
                                    serial_transforms.append({'type':'movt','move_type':move_type})
                                    serial_params.append({**dict_['params']})

                            if dict_['fn'].__name__ in ['no_change','static','fixed','fixed_inplace']: 
                                serial_transforms.append({'type':'static'})
                                serial_params.append({})
                            
                            if dict_['fn'].__name__ == 'recolor':
                                flag = False
                                for cchange in dict_['params']['color_changes']:
                                    start_color, end_color = cchange
                                    if start_color != end_color: flag = True
                                if flag: cchange_type = 'real_cchange'
                                else: cchange_type = 'no_cchange'   
                                if flag:
                                    serial_transforms.append({'type':'recolor','recolor_type':cchange_type})
                                    serial_params.append({**dict_['params']})
                                else: pass 

                            if dict_['fn'].__name__ == 'masking':
                                if dict_['params'] == {}:
                                    pass 
                                else:
                                    currmask = dict_['params']['mask']
                                    if np.sum(currmask) == 0:
                                        serial_transforms.append({'type':'static'})
                                        serial_params.append({})
                                    else:
                                        serial_transforms.append({'type':'masking'})
                                        serial_params.append({**dict_['params']}) 

                            if dict_['fn'].__name__ == 'copying':
                                
                                copy_details = dict_['params']['copy_details']
                                
                                if copy_details is None or copy_details == []: pass
                                else:
                                    serial_transforms.append({'type':'copying'}) 
                                    serial_params.append({**dict_['params']})


                            if dict_['fn'].__name__ == 'fill':
                                
                                fill_details = dict_['params']['fill_details']
                                if fill_details is None: pass
                                else:
                                    serial_transforms.append({'type':'fill'}) 
                                    serial_params.append({**dict_['params']})


                            if dict_['fn'].__name__ == 'extension':
                                
                                ext_details = dict_['params']['ext_details']
                                if ext_details is None: pass
                                elif len(ext_details['ext_fn_characterisation'])==0: pass
                                else:

                                    
                                    ext_details_ = ext_details['ext_fn_characterisation']
                                    for ek in range(len(ext_details_)):
                                        region_ = ext_details_[ek]['region'] | i_mask 
                                        activeleakmask_ = ext_details_[ek]['activeleakmask']
                                        nonleak_region = (region_==1) & (activeleakmask_==0)
                                        
                                        
                                        starter_oobjs = []
                                        for existing_oobj in global_parsings[gridn]['o']:
                                            existing_oobj_mask, existing_oobj_map = global_parsings[gridn]['o'][existing_oobj]['mask'], global_parsings[gridn]['o'][existing_oobj]['map']
                                            if np.sum(existing_oobj_mask & activeleakmask_) > 0:
                                                if are_two_identical(existing_oobj_map[existing_oobj_mask==1],o_grid[existing_oobj_mask==1]): 
                                                    starter_oobjs.append(existing_oobj)
                                        starter_oobjs = list(set(starter_oobjs))
                                        
                                        
                                        for s_obj in starter_oobjs:
                                            characterisable_mask = global_parsings[gridn]['o'][s_obj]['mask'] & (nonleak_region==0)
                                            checker_mask = global_parsings[gridn]['o'][s_obj]['mask'] & (region_==0)
                                            if np.sum(checker_mask)>0: 
                                                
                                                
                                                
                                                
                                                

                                                pseudoobjgrid = np.where(characterisable_mask, global_parsings[gridn]['o'][s_obj]['map'], -99)
                                                
                                                
                                                i_objs_ = get_contiguous_regions(pseudoobjgrid,-99,True,False) 
                                                for n in range(1,np.max(i_objs_)+1):
                                                    obj_mask = (i_objs_==n).astype(int) 
                                                    obj_maskv = obj_mask.copy() 
                                                    obj_masko = obj_mask - obj_maskv 
                                                    obj_map  = np.where(obj_mask, o_grid, 0) 
                                                    
                                                    
                                                    
                                                    flagexists = False
                                                    for existing_oobj in global_parsings[gridn]['o']:
                                                        if are_two_identical([obj_mask,obj_map,obj_maskv],[global_parsings[gridn]['o'][existing_oobj]['mask'],global_parsings[gridn]['o'][existing_oobj]['map'],global_parsings[gridn]['o'][existing_oobj]['maskv']]):
                                                            flagexists = True; break                                            
                                                    
                                                    if not flagexists:
                                                        
                                                        
                                                        global_parsings[gridn]['o'][create_name()] = {'parsing_type':'ext_leakmask_recolorobj', 'obj_score':0.9,'mask':obj_mask,'maskv':obj_maskv,'masko':obj_masko,'map':obj_map,
                                                                                        'properties':{'is_straightforward_obj':False, 'parsing_description':['ext_leakmask_recolorobj',score,None]}}


                                    
                                    ext_FN_TYPE = ext_details['type']


                                    if ext_FN_TYPE == 'connection':
                                        new_iobj_pair, color_rule_info, connection_region_incl_objs, connection_region_excl_objs = ext_details['new_iobj_pair'], ext_details['color_rule_info'], ext_details['connection_region_incl_objs'], ext_details['connection_region_excl_objs']
                                        
                                        

                                        
                                        
                                        
                                        
                                        
                                        serial_transforms.append({'type':'connection'}) 
                                        
                                        
                                        serial_params.append({'connection_details':{'color_rule_info':color_rule_info}})
                                        
                                        
                                        iobj_ = new_iobj_pair 


                                    
                                    


                                    elif ext_FN_TYPE == 'extension':
                                        ext_details_ = ext_details['ext_fn_characterisation']
                                        
                                    
                                        
                                        ext_dirns = [dirn_dict['dir'] for dirn_dict in ext_details_] 
                                        num_dirns = len(ext_dirns) 
                                        max_num_objs = max([len(dirn_dict['rule_info']) for dirn_dict in ext_details_]) 
                                        num_objs_in_dirns = [(dirn_dict['dir'] , len(dirn_dict['rule_info'])) for dirn_dict in ext_details_] 
                                        extmodes_in_dirns = [(dirn_dict['dir'] , dirn_dict['extmode']) for dirn_dict in ext_details_] 


                                        

                                        
                                        hypers_in_dirns = [] 
                                        for dirn_dict in ext_details_:
                                            dir = dirn_dict['dir']
                                            rule_info = dirn_dict['rule_info']
                                            hyper_dets = []
                                            for objn in range(len(rule_info)):
                                                color_rules = rule_info[objn][1]          
                                                for rule in color_rules:
                                                    if type(rule)==tuple and rule[0]=='hyperp_color':
                                                        hypercolor = rule[1]; break
                                                len_rules = rule_info[objn][2]; reachededge=False
                                                for rule in len_rules:
                                                    if type(rule)==tuple and rule[0]=='hyperp_length':
                                                        hyperlen = rule[1]; break
                                                    if type(rule)==str and rule=='reached_edge':
                                                        reachededge = True
                                                hyper_dets.append([hypercolor, hyperlen])
                                            hypers_in_dirns.append((dirn_dict['dir'] , dirn_dict['extmode'], hyper_dets)) 
                                        
                                        

                                        
                                        mode_objn_in_dirns = [(dirn_dict['dir'] , dirn_dict['extmode'], len(dirn_dict['rule_info'])) for dirn_dict in ext_details_]

                                        

                                        sorted_objns = sorted([len(dirn_dict['rule_info']) for dirn_dict in ext_details_]) 

                                        num_dirns = len(ext_dirns) 
                                        max_objns = max([len(dirn_dict['rule_info']) for dirn_dict in ext_details_])

                                        
                                        
                                        
                                        
                                        specialcase_flag = False
                                        if max_objns == 1 and num_dirns in [1,2]: 
                                            all_edged = True
                                            for dirn_dict in ext_details_:
                                                dir = dirn_dict['dir']
                                                rule_info = dirn_dict['rule_info']
                                                len_rules = rule_info[0][2]; reachededge=False
                                                for rule in len_rules:
                                                    if type(rule)==str and rule=='reached_edge':
                                                        reachededge = True
                                                if reachededge: pass
                                                else: all_edged = False
                                            if all_edged:
                                                specialcase_flag = num_dirns

                                        

                                        serial_transforms.append({'type':'extension','by_multislot':True,'hypers_in_dirns':hypers_in_dirns,'mode_objn_in_dirns':mode_objn_in_dirns,'sorted_objns':sorted_objns,
                                                                'num_dirns':num_dirns,'max_objns':max_objns,'specialcase_flag':specialcase_flag}) 
                                        
                                        ed_ = ext_details['ext_fn_characterisation']
                                        l_ = []
                                        for __ in ed_:
                                            l_.append({'dir':__['dir'], 'extmode':__['extmode'], 'rule_info':__['rule_info']})
                                        serial_params.append({'ext_details':{'ext_fn_characterisation':l_}})


 
                                        




                            if dict_['fn'].__name__ == 'wholegrid_stampings':
                                stamp_details = dict_['params']['stamp_details']
                                if stamp_details is None: pass
                                else:
                                    serial_transforms.append({'type':'wholegrid_stampings'}) 
                                    serial_params.append({**dict_['params']})


                            
                            if serial_transforms != [] and serial_transforms[-1]['type'] == 'static':
                                if global_parsings[gridn]['i'][iobj_]['properties']['parsing_description'][0] == 'background':
                                    serial_transforms[-1]['is_background'] = True
                                
                                


                        
                        if serial_transforms == []: continue


                        
                        
                        complexity_score = (1.1-(0.1*len(serial_transforms))); iobj__ = iobj_[0] if type(iobj_) == list else iobj_
                        fullgrid_penalty = 0.3 if global_parsings[gridn]['i'][iobj__]['properties']['parsing_description'][0] == 'fullgrid_i' else 1
                        background_bump = 1.5 if len(serial_transforms)==1 and serial_transforms[0]['type']=='static' and global_parsings[gridn]['i'][iobj__]['properties']['parsing_description'][0] == 'background' else 1
                        iobj_score = global_parsings[gridn]['i'][iobj__]['obj_score'] if 'obj_score' in global_parsings[gridn]['i'][iobj__] else 0.5
                        score = complexity_score * fullgrid_penalty * iobj_score * background_bump
                        data = {'tr_score':score,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[i_colors],'(i_obj_shapes)':[i_shape],'current_run':current_run,'curr_mask':curr_mask,'curr_map':curr_map,'curr_maskv':curr_maskv,'curr_o_masko':curr_o_masko,
                                'addressable':{'iobj':iobj_,'oobj':oobj_}} 
                        
                        

                        matchflag = False
                        for _ in results: 
                            
                            if are_two_identical(data,_) or (data['serial_transforms'] in [     [{'type':'connection'}],     ]  and are_two_identical([data['addressable'],data['serial_transforms'],data['serial_params']],[_['addressable'],_['serial_transforms'],_['serial_params']])): matchflag = True; break
                        if not matchflag:
                            if gridn == 0 and toplot11: 
                                print("Transform - ",iobj_,oobj_,serial_transforms,serial_params)
                                def tempviz(iobj,oobj,gridn):
                                    iobj__ = iobj[0] if type(iobj) == list else iobj
                                    o_mask, o_map, o_masko = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['masko']
                                    i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj__]['mask'], global_parsings[gridn]['i'][iobj__]['map'], global_parsings[gridn]['i'][iobj__]['masko']  
                                    #visualise.plot_two_grids(np.where(i_mask, i_map, 100), np.where(o_mask, o_map, 100),i_masko, o_masko)                    
                                tempviz(iobj_,oobj_,gridn)
                            results.append(data)


            return results



        transform_res = [] 
        all_solved = False
        solved_iobjs = []

        def get_iobjs_of_parsing_type(parsing_type, parsing_set, gridn, gridt):
            iobjs = []
            for iobj in parsing_set[gridn][gridt]:
                if parsing_set[gridn][gridt][iobj]['properties']['parsing_description'][0] == parsing_type:
                    iobjs.append(iobj)    
            return iobjs




        
        try:
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]
                
                
                
                
                if i_grid.shape == o_grid.shape: 
                    res = detect_output_band_pattern(i_grid, o_grid)
                    if res is not None: 
                        dir_tuple, icolors_where_grids_dont_match, superlist, obj_color_list = res
                        
                        
                        
                        serial_transforms = [{'type':'output_band_pattern', 'straightforward':True}]; serial_params = [{'dir_tuple':dir_tuple, 'icolors_where_grids_dont_match':icolors_where_grids_dont_match, 'superlist':superlist}]

                        iobj = get_iobjs_of_parsing_type('fullgrid_i', global_parsings, gridn, 'i')[0]
                        oobj = get_iobjs_of_parsing_type('fullgrid_o', global_parsings, gridn, 'o')[0]
                        omask_,omap_,omaskv_,omasko_ = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['maskv'], global_parsings[gridn]['o'][oobj]['masko']

                        data = {'tr_score':0.9,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':omask_,'curr_map':omap_,'curr_maskv':omaskv_,'curr_o_masko':omasko_,
                                'addressable':{'iobj':iobj,'oobj':oobj}}    

                        transform_res.append(data)


                
                
                
                

                res = detect_output_tiling_pattern(i_grid, np.ones_like(i_grid)) 
                if res is not None: 
                    recongrid, maskofones = res
                    

                    
                    mode1 = True if are_two_identical(recongrid, o_grid) else False
                    diffmask =  (i_grid!=recongrid).astype(int)
                    if np.sum(diffmask)>0: bb_mask, bb_map, tl_rc = get_bounding_box_object(diffmask, recongrid)
                    else: bb_map = None
                    mode2 = True if are_two_identical(bb_map, o_grid) else False

                    if mode1 or mode2: 
                        if mode1: serial_transforms = [{'type':'detect_output_tiling_pattern', 'straightforward':True}]; serial_params = [{}]
                        if mode2: serial_transforms = [{'type':'detect_output_tiling_pattern_bbout', 'straightforward':True}]; serial_params = [{}]

                        iobj = get_iobjs_of_parsing_type('fullgrid_i', global_parsings, gridn, 'i')[0]
                        oobj = get_iobjs_of_parsing_type('fullgrid_o', global_parsings, gridn, 'o')[0]
                        omask_,omap_,omaskv_,omasko_ = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['maskv'], global_parsings[gridn]['o'][oobj]['masko']

                        data = {'tr_score':0.9,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':omask_,'curr_map':omap_,'curr_maskv':omaskv_,'curr_o_masko':omasko_,
                                'addressable':{'iobj':iobj,'oobj':oobj}}    

                        transform_res.append(data)


                res = detect_output_symmetries_pattern(i_grid, np.ones_like(i_grid)) 
                if res is not None: 
                    recongrid, maskofones = res
                    
                    #visualise.plotgrid(recongrid)
                    
                    mode1 = True if are_two_identical(recongrid, o_grid) else False
                    diffmask =  (i_grid!=recongrid).astype(int)
                    if np.sum(diffmask)>0: bb_mask, bb_map, tl_rc = get_bounding_box_object(diffmask, recongrid)
                    else: bb_map = None
                    mode2 = True if are_two_identical(bb_map, o_grid) else False


                    if mode1 or mode2: 

                        if mode1: serial_transforms = [{'type':'detect_output_symmetries_pattern', 'straightforward':True}]; serial_params = [{}]
                        if mode2: serial_transforms = [{'type':'detect_output_symmetries_pattern_bbout', 'straightforward':True}]; serial_params = [{}]

                        iobj = get_iobjs_of_parsing_type('fullgrid_i', global_parsings, gridn, 'i')[0]
                        oobj = get_iobjs_of_parsing_type('fullgrid_o', global_parsings, gridn, 'o')[0]
                        omask_,omap_,omaskv_,omasko_ = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['maskv'], global_parsings[gridn]['o'][oobj]['masko']

                        data = {'tr_score':0.9,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':omask_,'curr_map':omap_,'curr_maskv':omaskv_,'curr_o_masko':omasko_,
                                'addressable':{'iobj':iobj,'oobj':oobj}}    

                        transform_res.append(data)



                res = detect_output_denoising_pattern(i_grid, np.ones_like(i_grid)) 
                if res is not None: 
                    recongrid, maskofones = res
                    
                    if are_two_identical(recongrid, o_grid):
                    
                        serial_transforms = [{'type':'detect_output_denoising_pattern', 'straightforward':True}]; serial_params = [{}]

                        iobj = get_iobjs_of_parsing_type('fullgrid_i', global_parsings, gridn, 'i')[0]
                        oobj = get_iobjs_of_parsing_type('fullgrid_o', global_parsings, gridn, 'o')[0]
                        omask_,omap_,omaskv_,omasko_ = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['maskv'], global_parsings[gridn]['o'][oobj]['masko']

                        data = {'tr_score':0.9,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':omask_,'curr_map':omap_,'curr_maskv':omaskv_,'curr_o_masko':omasko_,
                                'addressable':{'iobj':iobj,'oobj':oobj}}    

                        transform_res.append(data)
        except: pass




        try:    
            curr_solved = {}
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break


                if all_solved: continue
                curr_solved[gridn] = False
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]
                
                
                
                results = []
                
                full_dict_of_iobjs = initial_global_parsings[gridn]['i']; found=False
                for iobj in full_dict_of_iobjs:
                    if full_dict_of_iobjs[iobj]['properties']['parsing_description'][0] == 'fullgrid_i':
                        found=True; break
                if not found: break 

                i_mask, i_map = full_dict_of_iobjs[iobj]['mask'], full_dict_of_iobjs[iobj]['map']
                details_of_tiling_opts = detect_tiledcopy_of_fullinputgrid(iobj, i_mask, i_map, o_grid, gridn)
                

                if details_of_tiling_opts is not None:
                    
                    
                    
                    
                    filtered_options, rmult, cmult, occlusion_mask = details_of_tiling_opts['filtered_options'], details_of_tiling_opts['rmult'],details_of_tiling_opts['cmult'], details_of_tiling_opts['occlusion_mask']

                    
                    mask = np.ones_like(o_grid)
                    map = o_grid
                    maskv = mask.copy()
                    masko = np.zeros_like(o_grid)
                    ogridname = create_name()
                    global_parsings[gridn]['o'][ogridname] = {'parsing_type':'TEMP1_fullgrid_oobj','obj_score':0.5,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['TEMP1_fullgrid_o',0.5,None]}}


                    if np.sum(occlusion_mask)==0:
                        serial_transforms = [{'type':'gridwise_tiled_copy'}]; serial_params = [{'tiling_details':details_of_tiling_opts}]
                        data = {'tr_score':1,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':np.ones_like(o_grid),'curr_map':o_grid,'curr_maskv':np.ones_like(o_grid),'curr_o_masko':np.zeros_like(o_grid),
                                'addressable':{'iobj':iobj,'oobj':ogridname}}
                        results.append(data)
                        curr_solved[gridn] = True

                    elif np.sum(occlusion_mask)>0: 
                    


                        mask = (occlusion_mask==0).astype(int)
                        map = np.where(mask, o_grid, 0)
                        maskv = mask.copy()
                        masko = occlusion_mask
                        constructname = create_name()
                        global_parsings[gridn]['i'][constructname] = {'parsing_type':'TEMP2_constructobj','obj_score':0.5,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['TEMP2_constructobj',0,None]}}
                        

                        
                        main_chains = []
                        
                        
                        
                        main_chains.append([{'fn':wholegrid_stampings,'params':{}}])
                        if gridn ==0 and toplot1: print('4specialcase:')


                        i_mask, i_map, o_region, o_masko = mask, map, np.ones_like(o_grid), np.zeros_like(o_grid) 
                        if np.sum(i_mask)==0: print("ERROR"); continue
                        res = detect_io_transforms(constructname, ogridname, i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oobj')
                        if res == []: print("ERROR")                
                        else:
                            
                            
                            combi_params = {'options':[]}
                            for re in res:
                                this_serial_transforms, this_serial_params, this_oobj = re['serial_transforms'], re['serial_params'], re['addressable']['oobj']
                                
                                temp = []
                                for s in range(len(this_serial_transforms)):
                                    temp.append({'fn':globals()[this_serial_transforms[s]['type']],'params':this_serial_params[s]})
                                combi_params['options'].append(temp)
                            serial_transforms = [{'type':'gridwise_tiled_copy'},{'type':'combi_transform'}]
                            serial_params = [{'tiling_details':details_of_tiling_opts}, {'combi_details':combi_params}]
                            data = {'tr_score':1,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':np.ones_like(o_grid),'curr_map':o_grid,'curr_maskv':np.ones_like(o_grid),'curr_o_masko':np.zeros_like(o_grid),
                                    'addressable':{'iobj':iobj,'oobj':ogridname}} 
                            results.append(data)        
                            curr_solved[gridn] = True   


                        

                transform_res.extend(results)
            #all_solved = True if np.all([curr_solved[gridn_] for gridn_ in curr_solved]) else False
        except: pass





        try:
            curr_solved = {}
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                curr_solved[gridn] = False
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]
                results = []
                for i_frameobj in global_frames[gridn]['i']:
                    iframetype, itotalcount, iscore, ifcolor, isubframes = global_frames[gridn]['i'][i_frameobj].values()
                    for o_frameobj in global_frames[gridn]['o']:
                        oframetype, ototalcount, oscore, ofcolor, osubframes = global_frames[gridn]['o'][o_frameobj].values()
                        if itotalcount in [1,2] and ototalcount in [1]: 
                            details_of_bool_opts = detect_gridwise_bools(iframetype, itotalcount, iscore, isubframes, oframetype, ototalcount, oscore, osubframes)
                            if details_of_bool_opts is not None and len(details_of_bool_opts)!=0:
                                
                                
                                if oframetype == 'wholegrid_obj' and ototalcount == 1: 
                                    if 0 not in (osubframes[0]['mask']): 
                                        
                                        
                                        mask = np.ones_like(o_grid)
                                        map = o_grid
                                        maskv = mask.copy()
                                        masko = np.zeros_like(o_grid)
                                        ogridname = create_name()
                                        global_parsings[gridn]['o'][ogridname] = {'parsing_type':'TEMP1_fullgrid_oobj','obj_score':0.5,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['TEMP1_fullgrid_o',0.5,None]}}

                                        
                                        full_dict_of_iobjs = initial_global_parsings[gridn]['i']; found=False
                                        for iobj in full_dict_of_iobjs:
                                            if full_dict_of_iobjs[iobj]['properties']['parsing_description'][0] == 'fullgrid_i':
                                                found=True; break
                                        if not found: break 


                                        serial_transforms = [{'type':'gridwise_bool_simpletype'}]; serial_params = [{'bool_details':details_of_bool_opts}]
                                        data = {'tr_score':1,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':np.ones_like(o_grid),'curr_map':o_grid,'curr_maskv':np.ones_like(o_grid),'curr_o_masko':np.zeros_like(o_grid),
                                                'addressable':{'iobj':iobj,'oobj':ogridname}}
                                        results.append(data)
                                        curr_solved[gridn] = True
                transform_res.extend(results)
        except: pass        
            
        





        try:    
            gather_opts = {_:[] for _ in range(num_demo_grids)}
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]

                for iobj in initial_global_parsings[gridn]['i']:
                    if initial_global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] in ['background']:
                        imap_,imaskv_ = initial_global_parsings[gridn]['i'][iobj]['map'],initial_global_parsings[gridn]['i'][iobj]['maskv']

                        for oobj in initial_global_parsings[gridn]['o']:
                            if initial_global_parsings[gridn]['o'][oobj]['properties']['parsing_description'][0] in ['background']:
                                omask_,omap_,omaskv_,omasko_ = initial_global_parsings[gridn]['o'][oobj]['mask'],initial_global_parsings[gridn]['o'][oobj]['map'],initial_global_parsings[gridn]['o'][oobj]['maskv'],initial_global_parsings[gridn]['o'][oobj]['masko']

                                if imap_[0,0] == omap_[0,0]: 
                                    quick_color = imap_[0,0]

                                    

                                    both_vis = True if (np.sum(imaskv_)>0 and np.sum(omaskv_)>0) else False
                                    
                                    serial_transforms = [{'type':'static', 'is_background':True }]; serial_params = [{}]

                                    data = {'tr_score':0.9,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':omask_,'curr_map':omap_,'curr_maskv':omaskv_,'curr_o_masko':omasko_,
                                            'addressable':{'iobj':iobj,'oobj':oobj}}                        

                                    gather_opts[gridn].append([quick_color, both_vis,  data])

            opts_over_grids = [len(gather_opts[g]) for g in gather_opts]
            color_lists_over_grids = [[_[0] for _ in gather_opts[g]] for g in gather_opts]
            choice = None
            if min(opts_over_grids)>=1: 
                for color_opt in [_[0] for _ in gather_opts[0]]: 
                    if np.all([color_opt in _ for _ in color_lists_over_grids]): 
                        
                        
                        vis_status = []
                        for g in gather_opts:
                            for opt in gather_opts[g]:
                                if opt[0] == color_opt:
                                    vis_status.append(opt[1])
                                    break
                        if True not in vis_status: continue 

                        choice = color_opt
                        break
                if choice is None and max(opts_over_grids)==1: 
                    choice = 'all'
            if choice == 'all':
                for g in gather_opts:
                    first_and_only_opt = gather_opts[g][0]
                    transform_res.append(first_and_only_opt[2])
            elif choice is not None:
                for g in gather_opts:
                    for opt in gather_opts[g]:
                        if opt[0] == choice:
                            transform_res.append(opt[2])
                            break  
        except: pass



        def fill_slot_holes(map_list, mask_list, hyperps, **kwargs):
            
            newmap = np.zeros_like(map_list[0])
            newmask = np.zeros_like(mask_list[0])

            for c, dict_ in enumerate(hyperps):
                mask_, map_ = mask_list[c], map_list[c]

                if dict_['type'] == 'hole_iobj':
                    newcolors = dict_['hole_newcolors']; ch=0 
                    
                    filled = binary_fill_holes(mask_);     
                    holes_mask = np.logical_and(filled, np.logical_not(mask_)).astype(int)
                    
                    newmask = newmask | filled
                    newmap = np.where(filled, map_, newmap)            
                    if np.sum(holes_mask)>0:
                        labeled_array = get_contiguous_regions(holes_mask,0,False,False)
                        num_holes = np.max(labeled_array)
                        for hole_n in range(1,np.max(labeled_array)+1):
                            hole_mask = (labeled_array==hole_n).astype(int)
                            newcolor = newcolors[ch]
                            newmask = newmask | hole_mask
                            newmap = np.where(hole_mask, newcolor, newmap)
                            ch+=1
                            
                elif dict_['type'] == 'nonhole_iobj':
                    isstatic = dict_['is_staticQ']
                    if isstatic:
                        newmask = newmask | mask_
                        newmap = np.where(mask_, map_, newmap)
                    else: pass 

            return newmap, newmask




        
        
        try:    
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]

                if i_grid.shape != o_grid.shape: continue 

                singlecolor_iobjs = get_iobjs_of_parsing_type('single_color', initial_global_parsings, gridn, 'i')

                bkg_colrs = [get_colors_of_obj(initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map']) for iobj in get_iobjs_of_parsing_type('background', initial_global_parsings, gridn, 'i')]

                valid = True; combo_mask = np.zeros_like(i_grid); utilised_iobjs = []; param_dets = []; choleiobjs = 0
                for iobj in singlecolor_iobjs:
                    if esc(): break
                    if esc1(): break
                    mask_, map_ = initial_global_parsings[gridn]['i'][iobj]['mask'], initial_global_parsings[gridn]['i'][iobj]['map']

                    
                    if get_colors_of_obj(initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map']) in bkg_colrs:
                        
                        continue

                    utilised_iobjs.append(iobj)

                    
                    filled = binary_fill_holes(mask_);     
                    holes_mask = np.logical_and(filled, np.logical_not(mask_)).astype(int)
                    if np.sum(holes_mask)>0:
                        
                        
                        labeled_array = get_contiguous_regions(holes_mask,0,False,False)
                        num_holes = np.max(labeled_array)
                        
                        hole_newcolors = []
                        
                        
                        flag = True  
                        if not are_two_identical( np.where(mask_, i_grid, 0), np.where(mask_, o_grid, 0)): flag = False
                        for hole_n in range(1,np.max(labeled_array)+1):
                            hole_mask = (labeled_array==hole_n).astype(int)
                            hole_imap  = np.where(hole_mask, i_grid, 0)        
                            hole_omap  = np.where(hole_mask, o_grid, 0)        
                            if len(get_colors_of_obj(hole_mask, hole_imap))!=1 or len(get_colors_of_obj(hole_mask, hole_omap))!=1: flag = False
                            hole_newcolors.append(get_colors_of_obj(hole_mask, hole_omap))            
                        combo_mask = combo_mask | filled
                        if not flag: valid = False

                        param_dets.append({'type':'hole_iobj','hole_newcolors':hole_newcolors}); choleiobjs+=1

                    else:
                        
                        nonhole_static = False
                        if are_two_identical( np.where(mask_, i_grid, 0), np.where(mask_, o_grid, 0)): 
                            combo_mask = combo_mask | mask_
                            nonhole_static = True
                        

                        param_dets.append({'type':'nonhole_iobj','is_staticQ':nonhole_static})


                if valid and choleiobjs>0:
                    
                    
                    
                    
                    
                    

                    serial_transforms = [{'type':'fill_slot_holes'}]; serial_params = [{'hyperps':param_dets}]

                    iobj_list = utilised_iobjs 
                    
                    
                    omask_ = combo_mask
                    omap_ = np.where(omask_,o_grid,0)
                    omaskv_ = omask_.copy()
                    omasko_ = np.zeros_like(omask_)
                    oname = create_name()
                    global_parsings[gridn]['o'][oname] = {'parsing_type':'temp_fill_slot_holes_obj','obj_score':0.4,'mask':omask_,'map':omap_,'maskv':omaskv_,'masko':omasko_,'properties':{'is_straightforward_obj':False, 'parsing_description':['temp_fill_slot_holes_obj',0.5,None]}}

                    data = {'tr_score':0.9,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':omask_,'curr_map':omap_,'curr_maskv':omaskv_,'curr_o_masko':omasko_,
                            'addressable':{'iobj':iobj_list,'oobj':oname}}    
                    
                    transform_res.append(data)        
        except: pass



        
        



        

        def rotatedprint(input_map, mask_to_transform, mode, deficit_color, print_color, **kwargs):
            imask, imap = mask_to_transform, input_map


            output_map = np.zeros_like(input_map)
            transformed_mask = np.zeros_like(mask_to_transform)


            
            rows,cols = np.where(imask==1)

            sums = []
            for m in range(len(rows)):
                p1, p2, p3, p4 = (rows[m],cols[m]), (rows[m]+0.5,cols[m]), (rows[m],cols[m]+0.5), (rows[m]+0.5,cols[m]+0.5)
                for p_ in [p1,p2,p3,p4]:
            
                    cum_mask = copy.deepcopy(imask); ogsum = np.sum(imask); flag = True
                    for rot in [1,2,3]:
                        nmap, nmask = rotate_about_center(imap, imask, rot, p_) 
                        newsum = np.sum(nmask)
                        if newsum != ogsum: flag = False
                        cum_mask = cum_mask | nmask
                    
                    if not flag: continue

                    
                    if are_two_identical( imask, cum_mask & imask):
                
                        additions = ((cum_mask==1) & (imask==0)).astype(int)

                        sums.append([np.sum(additions), p_])
            sums = sorted(sums, key=lambda x: (x[0]))
            if len(sums)>0 and sums[0][0]!=0:
                centrepxl_rc = sums[0][1]


                
                if mode == '0':

                    cum_mask = copy.deepcopy(imask)
                    for rot in [1,2,3]:
                        nmap, nmask = rotate_about_center(imap, imask, rot, centrepxl_rc) 
                        cum_mask = cum_mask | nmask

                    
                    transformed_mask = cum_mask
                    output_map = np.where(cum_mask, deficit_color, 0) 
                    output_map = np.where(imask, print_color, output_map) 


                
                if mode == '1':
                    
                    cum_mask = copy.deepcopy(imask)
                    cum_map = np.where(imask, imap, 0)
                    for rot in [1,2,3]:
                        nmap, nmask = rotate_about_center(imap, imask, rot, centrepxl_rc) 
                        cum_mask = cum_mask | nmask
                        cum_map = np.where(nmask, nmap, cum_map)

                    output_map, transformed_mask = cum_map, cum_mask



            return output_map, transformed_mask

        try:
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]
                full_dict_of_iobjs = initial_global_parsings[gridn]['i'] 

                dict_of_iobjs = {}; temp=[]; tempkey=[] 
                for iobj in full_dict_of_iobjs:
                    i_mask, i_map, i_masko = full_dict_of_iobjs[iobj]['mask'], full_dict_of_iobjs[iobj]['map'], full_dict_of_iobjs[iobj]['masko']
                    if is_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp):
                        ix = ix_of_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp)
                        dict_of_iobjs[tempkey[ix]]['iobj_list'].append(iobj)
                    else: temp.append([np.where(i_mask, i_map, 100), i_masko]); tempkey.append(iobj); dict_of_iobjs[iobj] = full_dict_of_iobjs[iobj]; dict_of_iobjs[iobj]['iobj_list'] = [iobj]


                full_dict_of_oobjs = initial_global_parsings[gridn]['o']
                dict_of_oobjs = {}; temp=[]; tempkey=[]
                for oobj in full_dict_of_oobjs:
                    o_mask, o_map, o_masko = full_dict_of_oobjs[oobj]['mask'], full_dict_of_oobjs[oobj]['map'], full_dict_of_oobjs[oobj]['masko']
                    if is_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp):
                        ix = ix_of_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp)
                        dict_of_oobjs[tempkey[ix]]['oobj_list'].append(oobj)
                    else: temp.append([np.where(o_mask, o_map, 100), o_masko]); tempkey.append(oobj); dict_of_oobjs[oobj] = full_dict_of_oobjs[oobj]; dict_of_oobjs[oobj]['oobj_list'] = [oobj]
                


                


                for oobj in dict_of_oobjs:
                    if esc(): break
                    if esc1(): break
                    
                    
                    
                    omask, omap = global_parsings[gridn]['o'][oobj]['maskv'], global_parsings[gridn]['o'][oobj]['map']
                    for iobj in dict_of_iobjs:
                        imask, imap = global_parsings[gridn]['i'][iobj]['maskv'], global_parsings[gridn]['i'][iobj]['map']
                        
                        if omask.shape != imask.shape: continue 
                        if np.sum((omask==0)&(imask==1))==0: 

                            overlapmask = np.zeros_like(imask) 
                            rows,cols = np.where(omask==1)
                            for m in range(len(rows)):
                                if omap[rows[m],cols[m]] == imap[rows[m],cols[m]]:
                                    overlapmask[rows[m],cols[m]] = 1

                            if np.sum(overlapmask)>0: 
                                
                                

                                
                                bb_mask, bb_map, tl_rc = get_bounding_box_object(omask, omap)
                                
                                centre_r = tl_rc[0]+bb_mask.shape[0]//2 if bb_mask.shape[0]%2==1 else tl_rc[0]+bb_mask.shape[0]//2 - 0.5 
                                centre_c = tl_rc[1]+bb_mask.shape[1]//2 if bb_mask.shape[1]%2==1 else tl_rc[1]+bb_mask.shape[1]//2 - 0.5
                                centrepxl_rc = (centre_r, centre_c)
                                
                                cum_mask = copy.deepcopy(overlapmask)
                                for rot in [1,2,3]:
                                    nmap, nmask = rotate_about_center(imap, overlapmask, rot, centrepxl_rc) 
                                    cum_mask = cum_mask | nmask
                                if are_two_identical(cum_mask, omask): 
                                    

                                    
                                    if are_two_identical(imask, overlapmask) and len(get_colors_of_obj(overlapmask,imap))==1 and len(get_colors_of_obj(overlapmask,omap))==1 and len(get_colors_of_obj( ((omask==1)&(overlapmask==0)).astype(int) ,omap))==1 and (np.sum(imask)>1 and np.sum(omask)>1) and (np.sum(cum_mask)>np.sum(imask)):
                                        
                                        
                                        
                                        
                                        deficit_color = get_colors_of_obj( ((omask==1)&(overlapmask==0)).astype(int) ,omap)
                                        print_color = get_colors_of_obj(overlapmask,omap)

                                        

                                        
                                        rows,cols = np.where(imask==1)

                                        sums = []
                                        for m in range(len(rows)):
                                            p1, p2, p3, p4 = (rows[m],cols[m]), (rows[m]+0.5,cols[m]), (rows[m],cols[m]+0.5), (rows[m]+0.5,cols[m]+0.5)
                                            for p_ in [p1,p2,p3,p4]:
                                        
                                                cum_mask = copy.deepcopy(imask); ogsum = np.sum(imask); flag = True
                                                for rot in [1,2,3]:
                                                    nmap, nmask = rotate_about_center(imap, imask, rot, p_) 
                                                    newsum = np.sum(nmask)
                                                    if newsum != ogsum: flag = False
                                                    cum_mask = cum_mask | nmask
                                                
                                                if not flag: continue

                                                
                                                if are_two_identical( imask, cum_mask & imask):
                                            
                                                    additions = ((cum_mask==1) & (imask==0)).astype(int)

                                                    
                                                    
                                                    
                                                    

                                                    sums.append([np.sum(additions), p_])
                                        sums = sorted(sums, key=lambda x: (x[0]))
                                        if len(sums)>0 and sums[0][0]!=0:
                                            

                                            if sums[0][1] != centrepxl_rc: print('ERROR'); continue

                                            serial_transforms = [{'type':'rotatedprint'}]; serial_params = [{'mode':'0', 'deficit_color':deficit_color, 'print_color': print_color}]

                                            omask_ = omask 
                                            omaskv_ = omask 
                                            omap_ = omap
                                            omasko_ = np.zeros_like(omask)

                                            data = {'tr_score':0.9,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':omask_,'curr_map':omap_,'curr_maskv':omaskv_,'curr_o_masko':omasko_,
                                                    'addressable':{'iobj':iobj,'oobj':oobj}}

                                            transform_res.append(data)


                                
                                bb_mask, bb_map, tl_rc = get_bounding_box_object(omask, omap)
                                centre_r = tl_rc[0]+bb_mask.shape[0]//2 if bb_mask.shape[0]%2==1 else tl_rc[0]+bb_mask.shape[0]//2 - 0.5 
                                centre_c = tl_rc[1]+bb_mask.shape[1]//2 if bb_mask.shape[1]%2==1 else tl_rc[1]+bb_mask.shape[1]//2 - 0.5
                                centrepxl_rc = (centre_r, centre_c)
                                
                                cum_mask = copy.deepcopy(overlapmask)
                                cum_map = np.where(overlapmask, imap, 0)
                                for rot in [1,2,3]:
                                    nmap, nmask = rotate_about_center(imap, overlapmask, rot, centrepxl_rc) 
                                    cum_mask = cum_mask | nmask
                                    cum_map = np.where(nmask, nmap, cum_map)
                                if are_two_identical(cum_mask, omask) and are_two_identical(cum_map, omap):
                                    
                                    
                                    
                                    
                                    if are_two_identical(imask, overlapmask) and (np.sum(imask)>1 and np.sum(omask)>1) and (np.sum(cum_mask)>np.sum(imask)):
                                        
                                        
                                        
                                        

                                        rows,cols = np.where(imask==1)

                                        sums = []
                                        for m in range(len(rows)):
                                            p1, p2, p3, p4 = (rows[m],cols[m]), (rows[m]+0.5,cols[m]), (rows[m],cols[m]+0.5), (rows[m]+0.5,cols[m]+0.5)
                                            for p_ in [p1,p2,p3,p4]:
                                        
                                                cum_mask = copy.deepcopy(imask); ogsum = np.sum(imask); flag = True
                                                for rot in [1,2,3]:
                                                    nmap, nmask = rotate_about_center(imap, imask, rot, p_) 
                                                    newsum = np.sum(nmask)
                                                    if newsum != ogsum: flag = False
                                                    cum_mask = cum_mask | nmask
                                                
                                                if not flag: continue

                                                
                                                if are_two_identical( imask, cum_mask & imask):
                                            
                                                    additions = ((cum_mask==1) & (imask==0)).astype(int)

                                                    
                                                    
                                                    
                                                    

                                                    sums.append([np.sum(additions), p_])
                                        sums = sorted(sums, key=lambda x: (x[0]))
                                        if len(sums)>0 and sums[0][0]!=0:
                                            
                                            
                                            if sums[0][1] != centrepxl_rc: print('ERROR'); continue

                                            serial_transforms = [{'type':'rotatedprint'}]; serial_params = [{'mode':'1', 'deficit_color': None, 'print_color': None}]

                                            omask_ = omask 
                                            omaskv_ = omask 
                                            omap_ = omap
                                            omasko_ = np.zeros_like(omask)

                                            data = {'tr_score':0.9,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':omask_,'curr_map':omap_,'curr_maskv':omaskv_,'curr_o_masko':omasko_,
                                                    'addressable':{'iobj':iobj,'oobj':oobj}}

                                            transform_res.append(data)
        except: pass

        
        



        def get_all_overlap_masks_incl_spillover(bb_mask, o_grid): 
            bb_i_mask = bb_mask
            i_vals = bb_mask[bb_mask==1]
            validmasks = []
            bbirows, bbicols = np.where(bb_i_mask==1)
            for r in range(-bb_i_mask.shape[0] + 1, o_grid.shape[0]): 
                for c in range(-bb_i_mask.shape[1] + 1, o_grid.shape[1]):
                    o_tl_rc = (r,c) 
                    rows = bbirows + r
                    cols = bbicols + c
                    valid = (rows >= 0) & (rows < o_grid.shape[0]) & (cols >= 0) & (cols < o_grid.shape[1])
                    validmask = np.zeros_like(o_grid); validmap = -1*np.ones_like(o_grid)
                    validmask[rows[valid], cols[valid]] = 1
                    validmap[rows[valid], cols[valid]] = i_vals[valid] 
                    if np.sum(validmask) == 0: continue
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    validmasks.append(validmask)
            return validmasks

        def get_all_overlap_masks_excl_spillover(bb_mask, o_grid): 
            bb_i_mask = bb_mask
            i_vals = bb_mask[bb_mask==1]
            validmasks = []
            bbirows, bbicols = np.where(bb_i_mask==1)
            for r in range(-bb_i_mask.shape[0] + 1, o_grid.shape[0]): 
                for c in range(-bb_i_mask.shape[1] + 1, o_grid.shape[1]):
                    o_tl_rc = (r,c) 
                    rows = bbirows + r
                    cols = bbicols + c
                    valid = (rows >= 0) & (rows < o_grid.shape[0]) & (cols >= 0) & (cols < o_grid.shape[1])
                    validmask = np.zeros_like(o_grid); validmap = -1*np.ones_like(o_grid)
                    validmask[rows[valid], cols[valid]] = 1
                    validmap[rows[valid], cols[valid]] = i_vals[valid] 
                    if np.sum(validmask) == 0: continue
                    if np.sum(validmask) != np.sum(bb_i_mask): continue
                    validmasks.append(validmask)
            return validmasks



        try:
            bank1=[]; bank2=[]; color_dets={} 
            for md in ['collate','fwd']: 

                bank1_flavour1_flag = False
                bank1_flavour2_flag = False
                bank2_flavour1_flag = False
                bank2_flavour2_flag = False 
                bank1_sum=0; bank2_sum = 0

                for gridn in range(num_demo_grids):
                    if esc(): break
                    if esc1(): break
                    if all_solved: continue
                    i_grid = i_grids[gridn]
                    o_grid = o_grids[gridn]

                    results = []

                    cobj_list = get_iobjs_of_parsing_type('mask_of_color', initial_global_parsings, gridn, 'i')
                    singlecolr_iobjs = get_iobjs_of_parsing_type('single_color', initial_global_parsings, gridn, 'i')

                    if i_grid.shape != o_grid.shape: continue 


                    
                    recolored_colrs = []; static_colrs = []
                    for cobj in cobj_list: 
                        cmask, cmap = initial_global_parsings[gridn]['i'][cobj]['mask'], initial_global_parsings[gridn]['i'][cobj]['map']
                        i_color = cmap[cmask==1][0] 
                        
                        
                        o_colors = get_colors_of_obj( cmask, o_grid )
                        if len(o_colors)==1 and o_colors[0] == i_color: static_colrs.append([i_color, cmask, None, np.sum(cmask), cobj])
                        else: recolored_colrs.append([i_color, cmask, o_colors, cobj]) 
                    primary = None
                    if len(recolored_colrs)==0 or len(static_colrs)==0: break 
                    
                    if len(recolored_colrs)==1: primary = recolored_colrs[0]
                    elif len(recolored_colrs)>1: 
                        for _ in recolored_colrs:
                            if _[0] in _[2]:
                                primary = _
                                break
                    if primary is None: break 
                    if len(static_colrs)==1: pbkg = static_colrs[0]; secondary_list = [] 
                    else:
                        pbkg = static_colrs[np.argmax([_[3] for _ in static_colrs])]
                        secondary_list = [static_colrs[k] for k in range(len(static_colrs)) if not are_two_identical(pbkg, static_colrs[k])]
                        
                    primary_color = primary[0]; secondary_colors = [_[0] for _ in secondary_list] if secondary_list is not [] else []; pbkg_color = pbkg[0]
                    all_iobjs = [primary[3]]+[_[4] for _ in secondary_list]+[pbkg[4]]
                    
                    

                    
                    primary_color_mask = primary[1] 
                    if secondary_colors != []: 
                        secondary_color_mask = np.zeros_like(i_grid)
                        for k in range(len(secondary_list)):
                            secondary_color_mask = secondary_color_mask | secondary_list[k][1]
                    else: secondary_color_mask = np.zeros_like(i_grid)
                    recolored_mask = np.where(primary_color_mask, (o_grid != primary_color).astype(int)  , 0) 
                    pbkg_mask = pbkg[1]
                    


                    
                    cand_mask = recolored_mask 
                    cand_mask = recolored_mask | secondary_color_mask 
                    color_dets[gridn] = [primary_color, secondary_colors, pbkg_color,  primary_color_mask,secondary_color_mask,recolored_mask,pbkg_mask,  cand_mask, all_iobjs]

                    
                    if md == 'collate':
                        
                        conservative_bank_bbmasks = [] 
                        i_objs = get_contiguous_regions(cand_mask,0,False,False) 
                        grid_border = grid_border_mask(cand_mask)
                        for k in range(1,np.max(i_objs)+1):
                            i_obj_mask = (i_objs==k).astype(int)
                            if np.sum(i_obj_mask & grid_border) > 0: continue 
                            bb_mask, bb_map, tl_rc = get_bounding_box_object(i_obj_mask, o_grid) 
                            
                            curr_recolor_region = i_obj_mask & recolored_mask
                            deficit_color = get_colors_of_obj(curr_recolor_region, o_grid)
                            if not is_x_in_y([bb_mask, deficit_color], conservative_bank_bbmasks): conservative_bank_bbmasks.append([bb_mask, deficit_color])
                            if not is_x_in_y([bb_mask, deficit_color], bank1): bank1.append([bb_mask, deficit_color])
                        
                        
                        normal_bank_bbmasks = []
                        i_objs = get_contiguous_regions(np.where(cand_mask, o_grid, -99),-99,False,False) 
                        for k in range(1,np.max(i_objs)+1):
                            i_obj_mask = (i_objs==k).astype(int)
                            bb_mask, bb_map, tl_rc = get_bounding_box_object(i_obj_mask, o_grid) 
                            curr_recolor_region = i_obj_mask & recolored_mask
                            if np.sum(curr_recolor_region)==0: continue
                            deficit_color = get_colors_of_obj(curr_recolor_region, o_grid)
                            if not is_x_in_y([bb_mask, deficit_color], normal_bank_bbmasks): normal_bank_bbmasks.append([bb_mask, deficit_color])
                            if not is_x_in_y([bb_mask, deficit_color], bank2): bank2.append([bb_mask, deficit_color])
                        
                    
                    
                    
                    


                    
                    if md == 'fwd':

                        
                        
                        if gridn==0:
                            
                            sizes1 = [np.sum(_[0]) for _ in bank1] 
                            sorted_bank1 = [bank1[ix] for ix in np.argsort(sizes1)[::-1]] 
                            sizes2 = [np.sum(_[0]) for _ in bank2]
                            sorted_bank2 = [bank2[ix] for ix in np.argsort(sizes2)[::-1]] 
                            bank1 = sorted_bank1; bank2 = sorted_bank2 

                            bank1_flavour1_flag = True
                            bank1_flavour2_flag = True
                            bank2_flavour1_flag = True
                            bank2_flavour2_flag = True

                            bank1_sum = 0
                            bank2_sum = 0
                            for _ in bank1: bank1_sum += np.sum(_[0])
                            for _ in bank2: bank2_sum += np.sum(_[0])



                            

                        
                        def recursive_slot(all_matches, i_grid):
                            for itern in range(100):
                                
                                

                                already_slotted = np.zeros_like(i_grid) 
                                for match in all_matches: 
                                    if match[2] == True : 
                                        already_slotted += match[1]
                                already_slotted = (already_slotted==1).astype(int)

                                
                                single_slot_solns = np.zeros_like(i_grid)
                                for match in all_matches: 
                                    if match[2] == False: 
                                        if np.sum(match[1] & already_slotted)==0:
                                            single_slot_solns += match[1]
                                        else: match[2] = 'invalidated'
                                


                                
                                n_improvements = 0; newslottings = np.zeros_like(i_grid)

                                minval = 1
                                r, c = np.nonzero(single_slot_solns)
                                minvals = list(single_slot_solns[r, c])
                                if minvals != []: minval = np.min(minvals)

                                for match in all_matches:
                                    
                                    
                                    
                                    if minval in single_slot_solns[match[1]==1] and match[2] != 'invalidated':
                                        newslottings += match[1]
                                        if np.amax(newslottings)>=2: break 
                                        match[2] = True 
                                        n_improvements += 1
                                        

                                if np.amax(newslottings)>=minval+1: 
                                    
                                    return None 

                                

                                if n_improvements == 0: break
                            return all_matches
                        
                        for b_, bank in enumerate([bank1, bank2]):

                            if b_ == 0: 

                                

                                primary_color = color_dets[gridn][0]
                                secondary_colors = color_dets[gridn][1]
                                
                                
                                secondary_masks = []
                                s_iobj = get_contiguous_regions(i_grid,None,True,False) 
                                for k in range(0,np.max(s_iobj)+1):
                                    s_obj_mask = (s_iobj==k).astype(int)
                                    s_obj_map = np.where(s_obj_mask, i_grid, 0)
                                    colr = get_colors_of_obj(s_obj_mask, s_obj_map)
                                    if colr!=[] and colr[0] in secondary_colors: 
                                        secondary_masks.append(s_obj_mask)
                                
                                
                                all_matches = []; bankc = 0
                                for shape, _ in bank:
                                    overlapmasks = get_all_overlap_masks_incl_spillover(shape, o_grid)
                                    for smask in secondary_masks:
                                        overlapm_cands = []
                                        for m in range(len(overlapmasks)):
                                            if np.sum(overlapmasks[m] & smask)>0 and np.sum((smask==1)&(overlapmasks[m]==0))==0: 
                                                if list(set(i_grid[((overlapmasks[m]==1)&(smask==0)).astype(int)==1])) == [primary_color]: 
                                                    overlapm_cands.append(m)
                                        for m in overlapm_cands:
                                            all_matches.append([bankc, overlapmasks[m],    False, smask])
                                    bankc +=1 
                                



                            if b_ == 1: 

                                
                                primary_color = color_dets[gridn][0]
                                all_matches = []; bankc = 0
                                for shape, _ in bank:
                                    
                                    overlapmasks = get_all_overlap_masks_excl_spillover(shape, o_grid)
                                    overlapm_cands = []
                                    for m in range(len(overlapmasks)):
                                        if list(set(i_grid[overlapmasks[m]==1])) == [primary_color]: 
                                            overlapm_cands.append(m)
                                    for m in overlapm_cands:
                                        all_matches.append([bankc, overlapmasks[m],    False])
                                    bankc +=1 
                                




                            


                            
                            all_matches_processed = recursive_slot(all_matches, i_grid)
                            if all_matches_processed is not None:
                                
                                recongrid = copy.deepcopy(i_grid)
                                for match in all_matches_processed:
                                    if match[2] == True:
                                        bankc, mask_ = match[0], match[1]
                                        deficit_color = bank[bankc][1]
                                        if b_ == 0: smask=match[3]; recongrid = np.where(((mask_==1)&(smask==0)).astype(int), deficit_color[0], recongrid)
                                        if b_ == 1: recongrid = np.where(mask_, deficit_color[0], recongrid)
                                if are_two_identical(o_grid, recongrid): pass
                                else:
                                    if b_==0: bank1_flavour1_flag = False 
                                    if b_==1: bank2_flavour1_flag = False
                            else:
                                if b_==0: bank1_flavour1_flag = False
                                if b_==1: bank2_flavour1_flag = False



                            
                            

                            for match in all_matches: match[2] = False 

                            running_matches = []
                            for b in range(len(bank)):
                                b_matches = [match for match in all_matches if match[0] == b] 
                                running_plus_b_matches = b_matches + running_matches
                                running_matches_processed = recursive_slot(running_plus_b_matches, i_grid) 
                                running_matches = running_matches_processed 
                                if running_matches is None: break

                            if running_matches is not None:
                                recongrid = copy.deepcopy(i_grid)
                                for match in running_matches:
                                    if match[2] == True:
                                        bankc, mask_ = match[0], match[1]
                                        deficit_color = bank[bankc][1]
                                        recongrid = np.where(mask_, deficit_color[0], recongrid)
                                if are_two_identical(o_grid, recongrid): pass
                                else:                         
                                    if b_==0: bank1_flavour2_flag = False 
                                    if b_==1: bank2_flavour2_flag = False
                            else:
                                if b_==0: bank1_flavour2_flag = False
                                if b_==1: bank2_flavour2_flag = False

                            
                            
                            

                if md == 'fwd':


                    ideal_bank = '1'
                    if bank2_sum < bank1_sum: ideal_bank = '2'
                    if ideal_bank == '1':
                        if bank1_flavour1_flag: chosen = '1/1'
                        elif bank1_flavour2_flag: chosen = '1/2'
                        elif bank2_flavour1_flag: chosen = '2/1'
                        elif bank2_flavour2_flag: chosen = '2/2'
                        else: chosen = None
                    elif ideal_bank == '2':
                        if bank2_flavour1_flag: chosen = '2/1'
                        elif bank2_flavour2_flag: chosen = '2/2'        
                        elif bank1_flavour1_flag: chosen = '1/1'
                        elif bank1_flavour2_flag: chosen = '1/2'
                        else: chosen = None

                    if chosen is not None:

                        for gridn in range(num_demo_grids):
                            if all_solved: continue
                            i_grid = i_grids[gridn]
                            o_grid = o_grids[gridn]

                            
                            
                            
                            
                            
                            
                            
                            

                            
                            
                            bank_type = chosen[0]
                            flavour_type = chosen[-1]

                            
                            primary_, secondary_, recolored_, pbkg_ = color_dets[gridn][3:6+1]
                            primary_color, secondary_colors, pbkg_color = color_dets[gridn][0:2+1]
                            i_mask_ = primary_ | secondary_ | pbkg_
                            iobj_list = color_dets[gridn][8]
                            

                            serial_transforms = [{'type':'slotting_hyperp_obj'}]; serial_params = [{'bank_type':bank_type,'flavour_type':flavour_type,
                                                                                                    'primary_color': primary_color, 'secondary_colors': secondary_colors, 'pbkg_color':pbkg_color,
                                                                                                    'bank': bank1 if bank_type == '1' else bank2}]
                                                                                                    
                            omask = i_mask_ 
                            omap = np.where(omask, o_grid, 0)
                            omaskv = omask.copy()
                            omasko = np.zeros_like(o_grid)

                            
                            oname = create_name()
                            global_parsings[gridn]['o'][oname] = {'parsing_type':'TEMP_slottingshape','obj_score':0.95,'mask':omask,'map':omap,'maskv':omaskv,'masko':omasko,'properties':{'is_straightforward_obj':False, 'parsing_description':['TEMP_slottingshape',0.9,None]}}


                            data = {'tr_score':0.9,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':omask,'curr_map':omap,'curr_maskv':omaskv,'curr_o_masko':omasko,
                                    'addressable':{'iobj':iobj_list,'oobj':oname}}

                            transform_res.append(data)
        except: pass





        try:
            selector_regions = {}
            for gridn in range(num_demo_grids + num_test_grids):
                if esc(): break
                if esc1(): break
                i_grid = i_grids[gridn]

                bkg_colrs = [get_colors_of_obj(initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map']) for iobj in get_iobjs_of_parsing_type('background', initial_global_parsings, gridn, 'i')]
                singlecolr_cands = get_iobjs_of_parsing_type('single_color', initial_global_parsings, gridn, 'i')
                selectors = None 
                for iobj in singlecolr_cands: 
                    mask_, map_ = initial_global_parsings[gridn]['i'][iobj]['mask'], initial_global_parsings[gridn]['i'][iobj]['map']
                    if is_x_in_y(x=get_colors_of_obj(mask_,map_), y=bkg_colrs): continue
                    i_multiobjs = get_contiguous_regions(mask_,1,False,False)
                    if np.max(i_multiobjs)==2:
                        reg1 = (i_multiobjs==1).astype(int)
                        reg2 = (i_multiobjs==2).astype(int)
                        if np.sum(reg1)<np.sum(reg2): selector_reg = reg1; nonselector_reg = reg2
                        else: selector_reg = reg2; nonselector_reg = reg1
                        selectors = {'small':selector_reg,'large':nonselector_reg,'selector_mask':(i_multiobjs==0).astype(int)}
                        break
                selector_regions[gridn] = selectors
                
            selector_presence = True if np.all([selector_regions[gridn] is not None for gridn in selector_regions]) else False
        except: pass



        def slotmatch(modes, mod_o_grid, iobj_list, gridn, initial_global_parsings, permutation_mode = 1):
            firm_matches = [] 
            mode_n = 0
            quit_0slot = False
            for i in range(100):
                
                mode = modes[mode_n]
                

                c0=0

                all_match_cands = []
                for iobj in iobj_list:
                    if iobj in [_['iobj'] for _ in firm_matches]: continue
                    i_maskv, i_map = initial_global_parsings[gridn]['i'][iobj]['maskv'], initial_global_parsings[gridn]['i'][iobj]['map'] 
                    bb_i_mask, bb_i_map, i_tl_rc = get_bounding_box_object(i_maskv, i_map)
                    
                    match_cands = []


                    
                    

                    if permutation_mode == 0: perms = [[bb_i_mask, bb_i_map, i_tl_rc]] ; perm_dets = [[]]

                    elif permutation_mode == 1: 

                        perms = [[bb_i_mask, bb_i_map, i_tl_rc]] ; perm_dets = [[{'rot':0}]]
                        desired_fliprow = bb_i_mask.shape[0]/2-.5 + i_tl_rc[0]
                        desired_flipcol = bb_i_mask.shape[1]/2-0.5 + i_tl_rc[1]
                        desired_centre = (desired_fliprow,desired_flipcol)
                        for rotation in [1,-1,2,-2,3,-3]:
                            map_, mask_ = rotate_about_center(i_map, i_maskv, rotation, desired_centre)
                            perm_mask, perm_map, perm_rc = get_bounding_box_object(mask_,map_)
                            ix = ix_of_x_in_y(x=[perm_mask, perm_map, perm_rc],y=perms)
                            if ix is not None: perm_dets[ix].append({'rot':rotation})
                            else: perms.append([perm_mask, perm_map, perm_rc]); perm_dets.append([{'rot':rotation}])

                    elif permutation_mode == 2: 

                        perms = [[bb_i_mask, bb_i_map, i_tl_rc]] ; perm_dets = [[{'xflip':False,'yflip':False}]]

                        desired_fliprow = bb_i_mask.shape[0]/2-.5 + i_tl_rc[0]
                        desired_flipcol = bb_i_mask.shape[1]/2-0.5 + i_tl_rc[1]

                        for flipn in range(3):
                            if flipn==0: map_, mask_ = flip(i_map, i_maskv, 'x_axis',desired_fliprow); dict_ = {'xflip':True,'yflip':False} 
                                
                            if flipn==1: map_, mask_ = flip(i_map, i_maskv, 'y_axis',desired_flipcol); dict_ = {'xflip':False,'yflip':True} 
                                
                            if flipn == 2:
                                map_, mask_ = flip(i_map, i_maskv, 'x_axis',desired_fliprow)
                                map_, mask_ = flip(map_, mask_, 'y_axis',desired_flipcol); dict_ = {'xflip':True,'yflip':True} 
                                
                            perm_mask, perm_map, perm_rc = get_bounding_box_object(mask_,map_)
                            ix = ix_of_x_in_y(x=[perm_mask, perm_map, perm_rc],y=perms)
                            if ix is not None: perm_dets[ix].append(dict_)
                            else: perms.append([perm_mask, perm_map, perm_rc]); perm_dets.append([dict_])

                    elif permutation_mode == 3: pass 

                    


                    for p in range(len(perms)):
                        perm_mask, perm_map, perm_rc = perms[p]
                        alt_dets = perm_dets[p]

                        for r in range(0,mod_o_grid.shape[0]-perm_mask.shape[0]+1):
                            for c in range(0,mod_o_grid.shape[1]-perm_mask.shape[1]+1):
                                rel_r, rel_c = -perm_rc[0]+r, -perm_rc[1]+c
                                newmask = np.zeros_like(mod_o_grid)
                                newmask[r:r+perm_mask.shape[0], c:c+perm_mask.shape[1]] = perm_mask
                                
                                
                                new_bb_mask, new_bb_map, _ = get_bounding_box_object(newmask, mod_o_grid)

                                if are_two_identical(new_bb_mask, perm_mask) and are_two_identical(new_bb_map[new_bb_mask==1], perm_map[perm_mask==1]):
                                    match_cands.append([(rel_r,rel_c), newmask,  alt_dets ])

                    
                    
                    
                    
                    
                    
                    
                    
                    

                    
                    

                    all_match_cands.append({'iobj':iobj,'match_cands':match_cands})
                

                
                quitflag = True if np.any([ len(iobj_matches['match_cands'])==0 for iobj_matches in all_match_cands]) else False
                if quitflag: 
                    
                    
                    pass 

                
                if mode == '1slotting':
                    for iobj_matches in all_match_cands:
                        if len(iobj_matches['match_cands'])==1:
                            firm_matches.append({'iobj':iobj_matches['iobj'],'match_cand':iobj_matches['match_cands'][0]}); c0 += 1 
                            mod_o_grid = np.where(iobj_matches['match_cands'][0][1], -99, mod_o_grid) 

                
                if mode == 'fixed':
                    for iobj_matches in all_match_cands:
                        rcs = [_[0] for _ in iobj_matches['match_cands']]
                        
                        count_mainaltfixed = 0; entryn=None
                        for n_, rc in enumerate(rcs):
                            if rc[0]==0 and rc[1]==0:
                                count_mainaltfixed += 1; entryn=n_
                        if count_mainaltfixed==1:
                            firm_matches.append({'iobj':iobj_matches['iobj'],'match_cand':iobj_matches['match_cands'][entryn]}); c0 += 1 
                            mod_o_grid = np.where(iobj_matches['match_cands'][entryn][1], -99, mod_o_grid) 


                
                if mode == 'mainaltdirs':
                    for iobj_matches in all_match_cands:
                        rcs = [_[0] for _ in iobj_matches['match_cands']]
                        
                        count_mainaltfixed = 0; entryn=None
                        for n_, rc in enumerate(rcs):
                            if (0 in rc) or (np.abs(rc[0])==np.abs(rc[1])):
                                count_mainaltfixed += 1; entryn=n_
                        if count_mainaltfixed==1:
                            firm_matches.append({'iobj':iobj_matches['iobj'],'match_cand':iobj_matches['match_cands'][entryn]}); c0 += 1 
                            mod_o_grid = np.where(iobj_matches['match_cands'][entryn][1], -99, mod_o_grid) 

                
                if mode == 'maindirs':
                    for iobj_matches in all_match_cands:
                        rcs = [_[0] for _ in iobj_matches['match_cands']]
                        
                        count_mainaltfixed = 0; entryn=None
                        for n_, rc in enumerate(rcs):
                            if (0 in rc):
                                count_mainaltfixed += 1; entryn=n_
                        if count_mainaltfixed==1:
                            firm_matches.append({'iobj':iobj_matches['iobj'],'match_cand':iobj_matches['match_cands'][entryn]}); c0 += 1 
                            mod_o_grid = np.where(iobj_matches['match_cands'][entryn][1], -99, mod_o_grid) 

                


                
                if c0 == 0: mode_n += 1
                elif c0 >0: mode_n = 0
                if mode_n == len(modes): break

            
            if not quit_0slot and sorted([_['iobj'] for _ in firm_matches]) == sorted(iobj_list):
                
                return 'whole_group_matches', firm_matches
            else: return 'partial_group_matches', firm_matches



        def tempdum():

            
            
            
            
            

            

            
            
            
            
            


            def get_selector_regions(gridn):
                
                bkg_colors = [] 
                for iobj in initial_global_parsings[gridn]['i']:
                    if initial_global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] == 'background':
                        bkg_colors.append(get_colors_of_obj(initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map']))
                suitably_parsed_cands = [] 
                for iobj in initial_global_parsings[gridn]['i']:
                    if initial_global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] == 'single_color':
                        if not is_x_in_y(x=get_colors_of_obj(initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map']), y=bkg_colors):
                            suitably_parsed_cands.append(iobj)
                
                def max_thickness(mask):

                    mask = mask.astype(bool)
                    thickness = 0
                    current = mask.copy()

                    while current.any():
                        current = binary_erosion(current)
                        thickness += 1

                    return thickness
                selector_regions = None 
                for iobj in suitably_parsed_cands: 
                    mask_ = initial_global_parsings[gridn]['i'][iobj]['mask']
                    
                    
                    i_multiobjs = get_contiguous_regions(mask_,1,False,False)
                    if np.max(i_multiobjs)==2:
                        reg1 = (i_multiobjs==1).astype(int)
                        reg2 = (i_multiobjs==2).astype(int)
                        if np.sum(reg1)<np.sum(reg2): selector_reg = reg1; nonselector_reg = reg2
                        else: selector_reg = reg2; nonselector_reg = reg1
                        
                        
                        
                        
                        
                        selector_regions = {'small':selector_reg,'large':nonselector_reg}
                        break
                return selector_regions

            
            gridn=0
            i_grid = i_grids[gridn]
            o_grid = o_grids[gridn]
            all_i_colors = np.unique(i_grid)

            

            for i_color in all_i_colors:

                
                if np.sum((i_grid==i_color).astype(int)) != np.sum((o_grid==i_color).astype(int)): continue
                

                
                iobj_list = []
                for iobj in initial_global_parsings[gridn]['i']:
                    if initial_global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] in ['single_color']:
                        mask_,map_,maskv_ = initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map'],initial_global_parsings[gridn]['i'][iobj]['maskv']
                        if get_colors_of_obj(maskv_,map_) == [i_color]:
                            iobj_list.append(iobj)

                
                mod_o_grid = copy.deepcopy(o_grid)
                modes = ['1slotting','mainaltdirs','maindirs'] 
                
                
                

            
            gridn=0
            i_grid = i_grids[gridn]
            o_grid = o_grids[gridn]
            iobj_list = [iobj for iobj in initial_global_parsings[gridn]['i']]
            mod_o_grid = copy.deepcopy(o_grid)
            modes = ['1slotting'] 
            
            
            

            
            gridn=0
            i_grid = i_grids[gridn]
            o_grid = o_grids[gridn]

            
            iobj_list = []
            for iobj in initial_global_parsings[gridn]['i']:
                if initial_global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] in ['multi_color']:
                    iobj_list.append(iobj)

            mod_o_grid = copy.deepcopy(o_grid)
            modes = ['1slotting'] 
            


            return True


        



        try:    
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]

                sel_obj = get_iobjs_of_parsing_type('mask_of_all_colors', initial_global_parsings, gridn, 'i')[0] 
                mask_of_all_colors = initial_global_parsings[gridn]['i'][sel_obj]['mask']
                
                
                if selector_presence: mask_of_all_colors = mask_of_all_colors & selector_regions[gridn]['large']

                all_i_colors = np.unique(i_grid[mask_of_all_colors==1])
                
                if i_grid.shape != o_grid.shape: continue 

                colors_disappeared = []; colors_maintained = []; qual = True; used_selector_large = False; o_mask = np.zeros_like(o_grid)
                for i_color in all_i_colors:
                    colormask_in_i = (i_grid==i_color).astype(int)
                    colormask_in_o = (o_grid==i_color).astype(int) 

                    
                    if selector_presence:
                        colormask_in_i = colormask_in_i & selector_regions[gridn]['large'] 
                        colormask_in_o = colormask_in_o & selector_regions[gridn]['large']
                        used_selector_large = True

                    if are_two_identical(colormask_in_i, colormask_in_o):
                        if np.sum(colormask_in_i)==0: pass 
                        else: colors_maintained.append(i_color); o_mask = o_mask | colormask_in_o
                    elif np.sum(colormask_in_i) > 0 and np.sum(colormask_in_o) == 0: colors_disappeared.append(i_color)
                    else: qual = False 
                
                
                if qual:

                    iobj = sel_obj

                    mask = o_mask
                    maskv = mask
                    map = np.where(mask, o_grid, 0)
                    masko = np.zeros_like(mask)

                    oname = create_name()
                    global_parsings[gridn]['o'][oname] = {'parsing_type':'mask_of_all_colors_maintained','obj_score':0.95,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['mask_of_all_colors_maintained',0.9,None]}}

                    serial_transforms = [{'type':'mask_of_all_colors_maintained_or_disappeared'}]; serial_params = [{'colors_maintained':colors_maintained,'colors_disappeared':colors_disappeared,'applied_selector_large_on_both_masks':used_selector_large}] 

                    data = {'tr_score':0.95,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':mask,'curr_map':map,'curr_maskv':maskv,'curr_o_masko':masko,
                            'addressable':{'iobj':iobj,'oobj':oname,'i_subregion':'selector_large'}}
                    
                    transform_res.append(data)

                    

                    for iobj in initial_global_parsings[gridn]['i']:
                        
                        mask_,map_,maskv_ = initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map'],initial_global_parsings[gridn]['i'][iobj]['maskv']
                        
                        if np.sum((mask_of_all_colors==0)&(maskv_==1))==0:
                            
                            pass
        except: pass    



        
        try:
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]
                results = []

                
                iobj_list = []
                for iobj in initial_global_parsings[gridn]['i']:
                    if initial_global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] in ['multi_color']:
                        iobj_list.append(iobj)

                mod_o_grid = copy.deepcopy(o_grid)
                modes = ['1slotting'] 
                whole_or_partial, firm_matches = slotmatch(modes, mod_o_grid, iobj_list, gridn, initial_global_parsings, permutation_mode = 0) 
                

                for match in firm_matches:
                    if esc(): break
                    if esc1(): break
                    iobj, move_rc, o_mask, perm_alts = match['iobj'], match['match_cand'][0], match['match_cand'][1], match['match_cand'][2]


                    o_map = np.where(o_mask, o_grid, 0)
                    o_masko = np.zeros_like(o_grid)
                    o_maskv = o_mask.copy()
                    
                    oname = create_name()
                    global_parsings[gridn]['o'][oname] = {'parsing_type':'temp_slotted_multiobj','obj_score':0.4,'mask':o_mask,'map':o_map,'maskv':o_maskv,'masko':o_masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['temp_slotted_multiobj',0.5,None]}}

                    if move_rc == (0,0):
                        serial_transforms = [{'type':'static',  'slotting_type':'slotting_multiobj'}]; serial_params = [{}]
                    else:
                        r, c = move_rc
                        move_type = 'irregular'
                        if (r==0 and c!=0) or (c==0 and r!=0): move_type = 'maindir'
                        elif np.abs(r) == np.abs(c) and r!=0 and c!=0: move_type = 'altdir'

                        serial_transforms = [{'type':'movt',  'slotting_type':'slotting_multiobj',  'move_type':move_type}]; serial_params = [{'move_rc':move_rc}]

                    data = {'tr_score':0.95,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':o_mask,'curr_map':o_map,'curr_maskv':o_maskv,'curr_o_masko':o_masko,
                            'addressable':{'iobj':iobj,'oobj':oname}}
                    

                    results.append(data)    

                
                
                transform_res.extend(results)
        except: pass
        
        
        
        
        
        
        
        



        
        try:    
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]

                results = []

                
                bkg_masko = None
                for iobj in initial_global_parsings[gridn]['i']:
                    if initial_global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] in ['background']:
                        bkg_map, bkg_masko = initial_global_parsings[gridn]['i'][iobj]['map'], initial_global_parsings[gridn]['i'][iobj]['masko']
                        break

                if bkg_masko is None: continue

                

                
                iobj_colors = get_colors_of_obj(bkg_masko, i_grid) 

                
                
                
                chosen_stack_dirn = None
                for stack_dirn in ['S','N','E','W']:
                    o_alloc_grid = np.zeros_like(o_grid)
                    if i_grid.shape != o_grid.shape: continue 
                    rows, cols = np.where(bkg_masko==1) 
                    if stack_dirn == 'S': sort_ixs = np.argsort(rows)[::-1] 
                    if stack_dirn == 'N': sort_ixs = np.argsort(rows) 
                    if stack_dirn == 'E': sort_ixs = np.argsort(cols)[::-1] 
                    if stack_dirn == 'W': sort_ixs = np.argsort(cols) 
                    alloc_cdts = {}
                    for ix in sort_ixs: 
                        i_cdt = (rows[ix], cols[ix])

                        if stack_dirn == 'S':
                            max_row_cand = o_alloc_grid.shape[0]-1 
                            for row_cand in range(max_row_cand, i_cdt[0]-1, -1): 
                                cand_cdt = (row_cand, i_cdt[1]) 
                                if o_alloc_grid[cand_cdt] == 0:
                                    o_alloc_grid[cand_cdt] = 1
                                    alloc_cdts[ix] = {'original':i_cdt, 'new':cand_cdt}
                                    break
                        if stack_dirn == 'N':
                            for row_cand in range(0, i_cdt[0]+1): 
                                cand_cdt = (row_cand, i_cdt[1]) 
                                if o_alloc_grid[cand_cdt] == 0:
                                    o_alloc_grid[cand_cdt] = 1
                                    alloc_cdts[ix] = {'original':i_cdt, 'new':cand_cdt}
                                    break
                        if stack_dirn == 'E':
                            max_col_cand = o_alloc_grid.shape[1]-1 
                            for col_cand in range(max_col_cand, i_cdt[1]-1, -1): 
                                cand_cdt = (i_cdt[0], col_cand) 
                                if o_alloc_grid[cand_cdt] == 0:
                                    o_alloc_grid[cand_cdt] = 1
                                    alloc_cdts[ix] = {'original':i_cdt, 'new':cand_cdt}
                                    break
                        if stack_dirn == 'W':
                            for col_cand in range(0, i_cdt[1]+1): 
                                cand_cdt = (i_cdt[0], col_cand) 
                                if o_alloc_grid[cand_cdt] == 0:
                                    o_alloc_grid[cand_cdt] = 1
                                    alloc_cdts[ix] = {'original':i_cdt, 'new':cand_cdt}
                                    break
                    
                    recon = copy.deepcopy(bkg_map) 
                    for k in alloc_cdts:
                        i_cdt = alloc_cdts[k]['original']
                        o_cdt = alloc_cdts[k]['new']
                        recon[o_cdt] = i_grid[i_cdt]

                    if are_two_identical(recon, o_grid): chosen_stack_dirn = stack_dirn; break

                if chosen_stack_dirn is not None:
                    

                    serial_transforms = [{'type':'1nblock'}]; serial_params = [{'stack_dirn':chosen_stack_dirn}]

                    for iobj in initial_global_parsings[gridn]['i']:
                        if initial_global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] in ['fullgrid_i']:
                            break

                    for oobj in initial_global_parsings[gridn]['o']:
                        if initial_global_parsings[gridn]['o'][oobj]['properties']['parsing_description'][0] in ['fullgrid_o']:
                            break

                    data = {'tr_score':0.95,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':np.ones_like(o_grid),'curr_map':o_grid,'curr_maskv':np.ones_like(o_grid),'curr_o_masko':np.zeros_like(o_grid),
                            'addressable':{'iobj':iobj,'oobj':oobj}}
                    
                    
                    results.append(data)
                transform_res.extend(results)
        except: pass

        try:
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]

                results = []

                cobj_list = get_iobjs_of_parsing_type('mask_of_color', initial_global_parsings, gridn, 'i')
                bkg_colrs = [get_colors_of_obj(initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map']) for iobj in get_iobjs_of_parsing_type('background', initial_global_parsings, gridn, 'i')]
                singlecolr_iobjs = get_iobjs_of_parsing_type('single_color', initial_global_parsings, gridn, 'i')

                for cobj in cobj_list:
                    
                    cmask, cmap = initial_global_parsings[gridn]['i'][cobj]['mask'], initial_global_parsings[gridn]['i'][cobj]['map']
                    i_color = cmap[cmask==1][0] 

                
                    if np.sum((i_grid==i_color).astype(int)) != np.sum((o_grid==i_color).astype(int)): continue 
                    if i_color in bkg_colrs: continue 
                    

                    
                    
                    
                    
                    
                    
                    

                    if are_two_identical((i_grid==i_color).astype(int), (o_grid==i_color).astype(int)): continue 


                    


                    
                    iobj_list = []; tempmask = np.zeros_like(i_grid)
                    for iobj in initial_global_parsings[gridn]['i']:
                        if initial_global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] in ['single_color']:
                            mask_,map_,maskv_ = initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map'],initial_global_parsings[gridn]['i'][iobj]['maskv']
                            if get_colors_of_obj(maskv_,map_) == [i_color]:
                                iobj_list.append(iobj)        
                                tempmask = tempmask | maskv_

                    singlecolor_objs_account_for_icolormask = True if are_two_identical(tempmask, (i_grid==i_color).astype(int)) else False
                    found = False

                    
                    if singlecolor_objs_account_for_icolormask:
                        
                        mod_o_grid = copy.deepcopy(o_grid)
                        modes = ['1slotting','fixed','mainaltdirs','maindirs'] 
                        whole_or_partial, firm_matches = slotmatch(modes, mod_o_grid, iobj_list, gridn, initial_global_parsings, permutation_mode = 0) 
                        
                        if whole_or_partial == 'whole_group_matches': 
                            found = True

                            for match in firm_matches:
                                iobj, move_rc, o_mask, perm_alts = match['iobj'], match['match_cand'][0], match['match_cand'][1], match['match_cand'][2]


                                o_map = np.where(o_mask, o_grid, 0)
                                o_masko = np.zeros_like(o_grid)
                                o_maskv = o_mask.copy()
                                
                                oname = create_name()
                                global_parsings[gridn]['o'][oname] = {'parsing_type':'temp_slotted_singleobj','obj_score':0.4,'mask':o_mask,'map':o_map,'maskv':o_maskv,'masko':o_masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['temp_slotted_singleobj',0.5,None]}}

                                if move_rc == (0,0):
                                    serial_transforms = [{'type':'static',  'slotting_type':'slotting_singleobj'}]; serial_params = [{}]
                                else:
                                    r, c = move_rc
                                    move_type = 'irregular'
                                    if (r==0 and c!=0) or (c==0 and r!=0): move_type = 'maindir'
                                    elif np.abs(r) == np.abs(c) and r!=0 and c!=0: move_type = 'altdir'

                                    serial_transforms = [{'type':'movt',  'slotting_type':'slotting_singleobj',  'move_type':move_type}]; serial_params = [{'move_rc':move_rc}]

                                data = {'tr_score':0.95,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':o_mask,'curr_map':o_map,'curr_maskv':o_maskv,'curr_o_masko':o_masko,
                                        'addressable':{'iobj':iobj,'oobj':oname}}
                                

                                results.append(data)




                    
                    if not found: pass 




                    

                    found = False
                    
                    assumed_subunit = 1
                    
                    
                    rows,cols = np.where(cmask==1); created_subobjs = []
                    for p in range(len(rows)):
                        mask = np.zeros_like(i_grid); mask[rows[p],cols[p]] = 1
                        map = np.zeros_like(i_grid); map[rows[p],cols[p]] = cmap[rows[p],cols[p]]
                        maskv = mask.copy(); masko = np.zeros_like(mask)
                        newname = create_name()
                        global_parsings[gridn]['i'][newname] = {'parsing_type':'pxl_obj','obj_score':0.4,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['pxl_obj',0.5,None]}}
                        created_subobjs.append([p, newname])
                    mod_o_grid = copy.deepcopy(o_grid)
                    modes = ['1slotting','fixed','mainaltdirs','maindirs'] 
                    whole_or_partial, firm_matches = slotmatch(modes, mod_o_grid, [_[1] for _ in created_subobjs], gridn, global_parsings, permutation_mode = 0)
                    if whole_or_partial == 'whole_group_matches': 
                        found = True

                        
                        flag_name = 'maskofcolorslotted_'+cobj+'_'+str(i_color)

                        for match in firm_matches:
                            iobj = match['iobj']
                            move_rc, o_mask, confign_list = match['match_cand']
                            


                            o_map = np.where(o_mask, o_grid, 0)
                            o_masko = np.zeros_like(o_grid)
                            o_maskv = o_mask.copy()
                            
                            oname = create_name()
                            global_parsings[gridn]['o'][oname] = {'parsing_type':'pxl_obj','obj_score':0.4,'mask':o_mask,'map':o_map,'maskv':o_maskv,'masko':o_masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['pxl_obj',0.5,None]}}

                            if move_rc == (0,0):
                                serial_transforms = [{'type':'static',  'slotting_type':'slotting_subobj_singlecolor',  'mask_of_color_slotted': flag_name }]; serial_params = [{}]
                            else:
                                r, c = move_rc
                                move_type = 'irregular'
                                if (r==0 and c!=0) or (c==0 and r!=0): move_type = 'maindir'
                                elif np.abs(r) == np.abs(c) and r!=0 and c!=0: move_type = 'altdir'

                                serial_transforms = [{'type':'movt',  'slotting_type':'subobj_subobj_singlecolor' ,  'mask_of_color_slotted': flag_name, 'move_type':move_type}]; serial_params = [{'move_rc':move_rc}]

                            data = {'tr_score':0.95,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':o_mask,'curr_map':o_map,'curr_maskv':o_maskv,'curr_o_masko':o_masko,
                                    'addressable':{'iobj':iobj,'oobj':oname}}
                            
                            results.append(data)


                    
                    if not found: pass 


                transform_res.extend(results)
        except: pass


        

        try:
            for gridn in [0]:
                if esc(): break
                if esc1(): break
                
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]

                full_dict_of_iobjs = initial_global_parsings[gridn]['i']

                dict_of_iobjs = {}; temp=[]; tempkey=[] 
                for iobj in full_dict_of_iobjs:
                    if full_dict_of_iobjs[iobj]['properties']['parsing_description'][0] in ['subframe_iobj','frame_iobj']: continue
                    i_mask, i_map, i_masko = full_dict_of_iobjs[iobj]['mask'], full_dict_of_iobjs[iobj]['map'], full_dict_of_iobjs[iobj]['masko']
                    if is_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp):
                        ix = ix_of_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp)
                        dict_of_iobjs[tempkey[ix]]['iobj_list'].append(iobj)
                    else: temp.append([np.where(i_mask, i_map, 100), i_masko]); tempkey.append(iobj); dict_of_iobjs[iobj] = full_dict_of_iobjs[iobj]; dict_of_iobjs[iobj]['iobj_list'] = [iobj]


                all_matches = [] 
                for iobj in dict_of_iobjs:

                    i_mask, i_map = dict_of_iobjs[iobj]['mask'], dict_of_iobjs[iobj]['map']
                    bb_i_mask, bb_i_map, i_tl_rc = get_bounding_box_object(i_mask,i_map)
        except: pass
                
                




        try:
            for gridn in [0]:
                if esc(): break
                if esc1(): break
                
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]

                full_dict_of_iobjs = initial_global_parsings[gridn]['i']

                dict_of_iobjs = {}; temp=[]; tempkey=[] 
                for iobj in full_dict_of_iobjs:
                    if full_dict_of_iobjs[iobj]['properties']['parsing_description'][0] in ['subframe_iobj','frame_iobj']: continue
                    i_mask, i_map, i_masko = full_dict_of_iobjs[iobj]['mask'], full_dict_of_iobjs[iobj]['map'], full_dict_of_iobjs[iobj]['masko']
                    if is_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp):
                        ix = ix_of_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp)
                        dict_of_iobjs[tempkey[ix]]['iobj_list'].append(iobj)
                    else: temp.append([np.where(i_mask, i_map, 100), i_masko]); tempkey.append(iobj); dict_of_iobjs[iobj] = full_dict_of_iobjs[iobj]; dict_of_iobjs[iobj]['iobj_list'] = [iobj]


                all_matches = [] 
                for iobj in dict_of_iobjs:
                    if esc(): break
                    if esc1(): break


                    i_mask, i_map = dict_of_iobjs[iobj]['mask'], dict_of_iobjs[iobj]['map']
                    bb_i_mask, bb_i_map, i_tl_rc = get_bounding_box_object(i_mask,i_map)
                    i_vals = bb_i_map[bb_i_mask==1]; i_colors = list(np.unique(i_vals))


                    matches = []

                    i_outline_mask = get_outline_border_mask(i_mask)
                    i_outline_vals = i_grid[i_outline_mask.astype(bool)]; i_outline_colors = list(np.unique(i_outline_vals))

                    bbirows, bbicols = np.where(bb_i_mask==1)

                    for r in range(-bb_i_mask.shape[0] + 1, o_grid.shape[0]): 
                        for c in range(-bb_i_mask.shape[1] + 1, o_grid.shape[1]):
                            o_tl_rc = (r,c) 
                            rows = bbirows + r
                            cols = bbicols + c
                            valid = (rows >= 0) & (rows < o_grid.shape[0]) & (cols >= 0) & (cols < o_grid.shape[1])
                            validmask = np.zeros_like(o_grid); validmap = -1*np.ones_like(o_grid)
                            validmask[rows[valid], cols[valid]] = 1
                            validmap[rows[valid], cols[valid]] = i_vals[valid] 
                            if np.sum(validmask) == 0: continue
                            
                            bb_iseg_mask, bb_iseg_map, iseg_tl_rc = get_bounding_box_object(validmask, validmap)
                            bb_oseg_mask, bb_oseg_map, oseg_tl_rc = get_bounding_box_object(validmask, o_grid)
                            if not are_two_identical(bb_iseg_mask, bb_i_mask): continue 
                            iseg_vals = bb_iseg_map[bb_iseg_mask==1]
                            iseg_colors = list(np.unique(iseg_vals))
                            fully_visible_shape = True if len(iseg_vals)==np.sum(bb_i_mask) else False
                            oseg_vals = bb_oseg_map[bb_oseg_mask==1]
                            oseg_colors = list(np.unique(oseg_vals))
                            
                            visible_region_shape_maintained = True if (iseg_vals == oseg_vals).all() else False 
                            object_maintained = True if (visible_region_shape_maintained and fully_visible_shape) else False 
                            perfect_colorchange, grey_recoloring, color_changes = check_perfect_colorchange(iseg_colors,oseg_colors, iseg_vals,oseg_vals) 
                            
                            o_outline_mask = get_outline_border_mask(validmask) 
                            o_outline_vals = o_grid[o_outline_mask.astype(bool)]; o_outline_colors = list(np.unique(o_outline_vals))
                            
                            
                            
                            nonleaked_oborder = True if none_of_x_in_y(x=oseg_colors, y=o_outline_colors) else False 
                            
                            contains_original_bordercolor = True if at_least_some_of_x_in_y(x=i_outline_colors,y=oseg_colors) else False
                            
                            

                            
                            anchor_rel_movts = []; anchor_static = []
                            H_i, W_i = i_grid.shape; i_offsets = [ (0, 0), (0, -W_i + 1),  (-H_i + 1, 0),(-H_i + 1, -W_i + 1)]
                            H_o, W_o = o_grid.shape; o_offsets = [ (0, 0), (0, -W_o + 1), (-H_o + 1, 0), (-H_o + 1, -W_o + 1) ]
                            c=0 
                            for (dy_i, dx_i), (dy_o, dx_o) in zip(i_offsets, o_offsets): 
                                curr_origin = ['TL','TR','BL','BR'][c]
                                c+=1
                                i_pos = (i_tl_rc[0] + dy_i, i_tl_rc[1] + dx_i)
                                o_pos = (o_tl_rc[0] + dy_o, o_tl_rc[1] + dx_o)
                                rel_movt = (o_pos[0]-i_pos[0],o_pos[1]-i_pos[1]) 
                                if are_identicalQ([i_pos,o_pos]): anchor_static.append(1); anchor_rel_movts.append('static')
                                else: anchor_static.append(0); anchor_rel_movts.append(rel_movt)
                                TL_rel_movt = rel_movt
                                break
                            

                            is_whole_obj = nonleaked_oborder 

                            f0 = False; f1 = False; f2 = False; f3 = False; f4 = False; f5 = False
                            
                            if 1 in anchor_static and not nonleaked_oborder and not contains_original_bordercolor: f0=True

                            if nonleaked_oborder and (visible_region_shape_maintained or perfect_colorchange): 
                                
                                if 1 in anchor_static: f1 = True 
                                
                                if object_maintained: f2 = True
                                
                                if visible_region_shape_maintained: f3 = True
                                
                                
                                if fully_visible_shape and perfect_colorchange: f4 = True
                                
                                if perfect_colorchange: f5 = True
                                

                            if (f0 or f1 or f2 or f3 or f4 or f5): 
                                
                                
                                
                                matches.append([iobj, o_tl_rc, color_changes, validmask, f0,f1,f2,f3,f4,f5])
                                all_matches.append([iobj, o_tl_rc, color_changes, validmask, f0,f1,f2,f3,f4,f5])
        except: pass

                




        




        

        try:    
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                curr_solved[gridn] = False
                
                o_grid = o_grids[gridn]

                results = []

                
                mask = np.ones_like(o_grid)
                map = o_grid
                maskv = mask.copy()
                masko = np.zeros_like(o_grid)
                ogridname = create_name()
                global_parsings[gridn]['o'][ogridname] = {'parsing_type':'fullgrid_oobj_for_gridmap','obj_score':0.5,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['fullgrid_o_for_gridmap',0.5,None]}}


                serial_transforms = [{'type':'hyperp_gridmap_creation'}]; serial_params = [{'gridmap':o_grid}]

                data = {'tr_score':1,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':np.ones_like(o_grid),'curr_map':o_grid,'curr_maskv':np.ones_like(o_grid),'curr_o_masko':np.zeros_like(o_grid),
                        'addressable':{'iobj':None,'oobj':ogridname}} 

                transform_res.append(data)
        except: pass
        




        try:    
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                curr_solved[gridn] = False
                
                o_grid = o_grids[gridn]

                results = []

                
                mask = np.ones_like(o_grid)
                map = o_grid
                maskv = mask.copy()
                masko = np.zeros_like(o_grid)
                ogridname = create_name()
                global_parsings[gridn]['o'][ogridname] = {'parsing_type':'fullgrid_oobj_for_gridmap','obj_score':0.5,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['fullgrid_o_for_gridmap',0.5,None]}}

                for framename_key in global_frames[gridn]['o']:
                    if global_frames[gridn]['o'][framename_key]['type'] in ['wholegrid_obj', 'leaky_2d','leaky_1dh','leaky_1dv']: continue

                    serial_transforms = [{'type':'frame_creation'}]; serial_params = [{'frame_dets':global_frames[gridn]['o'][framename_key]}]

                    data = {'tr_score':1,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':np.ones_like(o_grid),'curr_map':o_grid,'curr_maskv':np.ones_like(o_grid),'curr_o_masko':np.zeros_like(o_grid),
                            'addressable':{'iobj':None,'oobj':ogridname}} 

                    transform_res.append(data)
        except: pass
            


        try:    
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue 
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]

                

                
                
                

                sets_of_iobj_dicts = []
                full_dict_of_iobjs = initial_global_parsings[gridn]['i']



                
                all_frame_IDs = []
                for iobj in full_dict_of_iobjs:
                    if full_dict_of_iobjs[iobj]['properties']['parsing_description'][0] in ['subframe_iobj','frame_iobj']:
                        all_frame_IDs.append(full_dict_of_iobjs[iobj]['properties']['frame_ID'])
                frame_IDs = list(set(all_frame_IDs))

                for frame_ID_ in frame_IDs:
                    dict_of_iobjs = {}; temp=[]; tempkey=[] 
                    for iobj in full_dict_of_iobjs:
                        if full_dict_of_iobjs[iobj]['properties']['parsing_description'][0] not in ['subframe_iobj','frame_iobj']: continue
                        if full_dict_of_iobjs[iobj]['properties']['frame_ID'] != frame_ID_: continue
                        i_mask, i_map, i_masko = full_dict_of_iobjs[iobj]['mask'], full_dict_of_iobjs[iobj]['map'], full_dict_of_iobjs[iobj]['masko']
                        if is_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp):
                            ix = ix_of_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp)
                            dict_of_iobjs[tempkey[ix]]['iobj_list'].append(iobj)
                        else: temp.append([np.where(i_mask, i_map, 100), i_masko]); tempkey.append(iobj); dict_of_iobjs[iobj] = full_dict_of_iobjs[iobj]; dict_of_iobjs[iobj]['iobj_list'] = [iobj]
                    sets_of_iobj_dicts.append(dict_of_iobjs)        



                
                
                dict_of_iobjs = {}; temp=[]; tempkey=[] 
                for iobj in full_dict_of_iobjs:
                    if full_dict_of_iobjs[iobj]['properties']['parsing_description'][0] in ['subframe_iobj','frame_iobj']: continue
                    i_mask, i_map, i_masko = full_dict_of_iobjs[iobj]['mask'], full_dict_of_iobjs[iobj]['map'], full_dict_of_iobjs[iobj]['masko']
                    if is_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp):
                        ix = ix_of_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp)
                        dict_of_iobjs[tempkey[ix]]['iobj_list'].append(iobj)
                    else: temp.append([np.where(i_mask, i_map, 100), i_masko]); tempkey.append(iobj); dict_of_iobjs[iobj] = full_dict_of_iobjs[iobj]; dict_of_iobjs[iobj]['iobj_list'] = [iobj]
                sets_of_iobj_dicts.append(dict_of_iobjs)





                
                


                

                for dict_of_iobjs in sets_of_iobj_dicts:
                    if esc(): break
                    if esc1(): break
                    
                    all_matches = [] 
                    for iobj in dict_of_iobjs:
                        if esc(): break
                        if esc1(): break

                        i_mask, i_map = dict_of_iobjs[iobj]['mask'], dict_of_iobjs[iobj]['map']
                        try:
                            bb_i_mask, bb_i_map, i_tl_rc = get_bounding_box_object(i_mask,i_map)
                        except: continue 
                        i_vals = bb_i_map[bb_i_mask==1]; i_colors = list(np.unique(i_vals))


                        matches = []

                        i_outline_mask = get_outline_border_mask(i_mask)
                        i_outline_vals = i_grid[i_outline_mask.astype(bool)]; i_outline_colors = list(np.unique(i_outline_vals))

                        bbirows, bbicols = np.where(bb_i_mask==1)

                        for r in range(-bb_i_mask.shape[0] + 1, o_grid.shape[0]): 
                            for c in range(-bb_i_mask.shape[1] + 1, o_grid.shape[1]):
                                o_tl_rc = (r,c) 
                                rows = bbirows + r
                                cols = bbicols + c
                                valid = (rows >= 0) & (rows < o_grid.shape[0]) & (cols >= 0) & (cols < o_grid.shape[1])
                                validmask = np.zeros_like(o_grid); validmap = -1*np.ones_like(o_grid)
                                validmask[rows[valid], cols[valid]] = 1
                                validmap[rows[valid], cols[valid]] = i_vals[valid] 
                                if np.sum(validmask) == 0: continue
                                
                                bb_iseg_mask, bb_iseg_map, iseg_tl_rc = get_bounding_box_object(validmask, validmap)
                                bb_oseg_mask, bb_oseg_map, oseg_tl_rc = get_bounding_box_object(validmask, o_grid)
                                if not are_two_identical(bb_iseg_mask, bb_i_mask): continue 
                                iseg_vals = bb_iseg_map[bb_iseg_mask==1]
                                iseg_colors = list(np.unique(iseg_vals))
                                fully_visible_shape = True if len(iseg_vals)==np.sum(bb_i_mask) else False
                                oseg_vals = bb_oseg_map[bb_oseg_mask==1]
                                oseg_colors = list(np.unique(oseg_vals))
                                
                                visible_region_shape_maintained = True if (iseg_vals == oseg_vals).all() else False 
                                object_maintained = True if (visible_region_shape_maintained and fully_visible_shape) else False 
                                perfect_colorchange, grey_recoloring, color_changes = check_perfect_colorchange(iseg_colors,oseg_colors, iseg_vals,oseg_vals) 
                                
                                o_outline_mask = get_outline_border_mask(validmask) 
                                o_outline_vals = o_grid[o_outline_mask.astype(bool)]; o_outline_colors = list(np.unique(o_outline_vals))
                                
                                
                                
                                nonleaked_oborder = True if none_of_x_in_y(x=oseg_colors, y=o_outline_colors) else False 
                                
                                contains_original_bordercolor = True if at_least_some_of_x_in_y(x=i_outline_colors,y=oseg_colors) else False
                                
                                

                                
                                anchor_rel_movts = []; anchor_static = []
                                H_i, W_i = i_grid.shape; i_offsets = [ (0, 0), (0, -W_i + 1),  (-H_i + 1, 0),(-H_i + 1, -W_i + 1)]
                                H_o, W_o = o_grid.shape; o_offsets = [ (0, 0), (0, -W_o + 1), (-H_o + 1, 0), (-H_o + 1, -W_o + 1) ]
                                c=0 
                                for (dy_i, dx_i), (dy_o, dx_o) in zip(i_offsets, o_offsets): 
                                    curr_origin = ['TL','TR','BL','BR'][c]
                                    c+=1
                                    i_pos = (i_tl_rc[0] + dy_i, i_tl_rc[1] + dx_i)
                                    o_pos = (o_tl_rc[0] + dy_o, o_tl_rc[1] + dx_o)
                                    rel_movt = (o_pos[0]-i_pos[0],o_pos[1]-i_pos[1]) 
                                    if are_identicalQ([i_pos,o_pos]): anchor_static.append(1); anchor_rel_movts.append('static')
                                    else: anchor_static.append(0); anchor_rel_movts.append(rel_movt)
                                    TL_rel_movt = rel_movt
                                    break
                                

                                is_whole_obj = nonleaked_oborder 

                                f0 = False; f1 = False; f2 = False; f3 = False; f4 = False; f5 = False
                                
                                if 1 in anchor_static and not nonleaked_oborder and not contains_original_bordercolor: f0=True

                                if nonleaked_oborder and (visible_region_shape_maintained or perfect_colorchange): 
                                    
                                    if 1 in anchor_static: f1 = True 
                                    
                                    if object_maintained: f2 = True
                                    
                                    if visible_region_shape_maintained: f3 = True
                                    
                                    
                                    if fully_visible_shape and perfect_colorchange: f4 = True
                                    
                                    if perfect_colorchange: f5 = True
                                    

                                if (f0 or f1 or f2 or f3 or f4 or f5): 
                                    
                                    
                                    
                                    all_matches.append([iobj, o_tl_rc, color_changes, validmask, f0,f1,f2,f3,f4,f5])

                                

                                
                                
                                
                    
                    
                    

                    
                    
                    ixs = np.argsort([np.sum(_[3]) for _ in all_matches])[::-1]
                    sel_set = []; sel_mask = np.zeros_like(o_grid); sel_flag1 = False; sel_flag2 = False; colr1 = 0
                    
                    for ix in ixs:
                        
                        cand = all_matches[ix]
                        
                        
                        if not (np.sum(cand[3] | sel_mask) > np.sum(sel_mask)): continue

                        
                        sel_set.append(ix)
                        sel_mask = sel_mask | cand[3]
                        
                        
                        non_sel_mask = (sel_mask==0).astype(int)
                        colrs = np.unique(o_grid[non_sel_mask==1])
                        if len(colrs)==1:
                            ogrid_mask_of_this_colr = (o_grid==colrs[0]).astype(int)
                            if are_two_identical(non_sel_mask, ogrid_mask_of_this_colr):
                                sel_flag1 = True
                                colr1 = colrs[0]
                                sel_set1 = copy.deepcopy(sel_set)
                                sel_mask1 = copy.deepcopy(sel_mask)
                                
                        
                        
                        if np.sum(non_sel_mask)==0:
                            sel_flag2 = True
                            sel_set2 = copy.deepcopy(sel_set)
                            sel_mask2 = copy.deepcopy(sel_mask)



                    def simple_tiling_convert(sel_set1, all_matches, o_grid):
                        
                        set_of_tls = [all_matches[_][1] for _ in sel_set1]
                        min_row = np.min([_[0] for _ in set_of_tls])
                        set_of_cols = np.unique([_[1] for _ in set_of_tls if _[0]==min_row]) 
                        frame_width_prepended = set_of_cols[0]
                        frame_widths_appended = []
                        for n in range(len(set_of_cols)):
                            
                            for sel in sel_set1:
                                o_mask = all_matches[sel][3]
                                bb_mask, bb_map, tl_rc = get_bounding_box_object(o_mask, o_grid)
                                if tl_rc != (min_row, set_of_cols[n]): continue
                                br_rc = (tl_rc[0]+bb_mask.shape[0]-1,tl_rc[1]+bb_mask.shape[1]-1)
                                break
                            end_col = br_rc[1]
                            if end_col == o_grid.shape[1]-1: 
                                following_frame_width = 0
                                frame_widths_appended.append(int(following_frame_width))
                                break
                            elif n==len(set_of_cols)-1: 
                                following_frame_width = o_grid.shape[1]-1 - end_col
                                frame_widths_appended.append(int(following_frame_width))
                                break
                            else: 
                                following_frame_width = set_of_cols[n+1]-1 - end_col
                                frame_widths_appended.append(int(following_frame_width))
                        
                        min_col = np.min([_[1] for _ in set_of_tls])
                        set_of_rows = np.unique([_[0] for _ in set_of_tls if _[1]==min_col])
                        frame_height_prepended = set_of_rows[0]
                        frame_heights_appended = []
                        for n in range(len(set_of_rows)):
                            
                            for sel in sel_set1:
                                o_mask = all_matches[sel][3]
                                bb_mask, bb_map, tl_rc = get_bounding_box_object(o_mask, o_grid)
                                if tl_rc != (set_of_rows[n], min_col): continue
                                br_rc = (tl_rc[0]+bb_mask.shape[0]-1,tl_rc[1]+bb_mask.shape[1]-1)
                                break
                            end_row = br_rc[0]
                            if end_row == o_grid.shape[0]-1: 
                                following_frame_height = 0
                                frame_heights_appended.append(int(following_frame_height))
                                break
                            elif n==len(set_of_rows)-1: 
                                following_frame_height = o_grid.shape[0]-1 - end_row
                                frame_heights_appended.append(int(following_frame_height))
                                break
                            else: 
                                following_frame_height = set_of_rows[n+1]-1 - end_row
                                frame_heights_appended.append(int(following_frame_height))            
                        
                        if np.min(frame_widths_appended)<0 or np.min(frame_heights_appended)<0: return None, None, None, None
                        return frame_width_prepended, frame_widths_appended, frame_height_prepended , frame_heights_appended


                    

                    

                    if sel_flag2:
                        
                        
                        
                
                        frame_width_prepended, frame_widths_appended, frame_height_prepended , frame_heights_appended = simple_tiling_convert(sel_set2, all_matches, o_grid)
                        
                        
                        if frame_width_prepended is None: continue

                        set_of_tls = [all_matches[_][1] for _ in sel_set2]
                        min_row = np.min([_[0] for _ in set_of_tls])
                        set_of_cols = np.unique([_[1] for _ in set_of_tls if _[0]==min_row])
                        min_col = np.min([_[1] for _ in set_of_tls])
                        set_of_rows = np.unique([_[0] for _ in set_of_tls if _[1]==min_col])

                        num_rows = len(set_of_rows)
                        num_cols = len(set_of_cols)
                        if num_rows==1 and num_cols==1: continue 
                        tiles = []
                        for r in range(num_rows):
                            for c in range(num_cols):
                                cdt = (int(set_of_rows[r]),int(set_of_cols[c]))
                                sel_ix = ix_of_x_in_y(x=cdt,y=set_of_tls)
                                if sel_ix is None: tiles.append({'iobj':None,'color_changes':None}); continue
                                match_ix = sel_set2[sel_ix]
                                if np.all([pair[0]==pair[1] for pair in all_matches[match_ix][2]]): tiles.append({'iobj':all_matches[match_ix][0],'color_changes':None})
                                else: tiles.append({'iobj':all_matches[match_ix][0],'color_changes':all_matches[match_ix][2]})
                        
                        
                        
                        
                        
                        

                        iobj_list = [_['iobj'] for _ in tiles if _['iobj'] is not None]
                        if len(iobj_list)==0: continue
                        
                        


                        try:
                            safe_tile_ixs = [list(set(iobj_list)).index(_['iobj']) for _ in tiles]
                        except: continue 
                        construction = {'n_rows':num_rows,'n_cols':num_cols,'safe_tile_ixs':safe_tile_ixs,'tiles':tiles,'frame_specs':[frame_width_prepended, frame_widths_appended, frame_height_prepended , frame_heights_appended],
                                        'bkg_dets':None}

                        
                        mask = np.ones_like(o_grid)
                        map = o_grid
                        maskv = mask.copy()
                        masko = np.zeros_like(o_grid)
                        ogridname = create_name()
                        global_parsings[gridn]['o'][ogridname] = {'parsing_type':'fullgrid_oobj_for_gridmap','obj_score':0.5,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['fullgrid_o_for_gridmap',0.5,None]}}


                        serial_transforms = [{'type':'iobjs_tile_creation'}]; serial_params = [{'details':construction}]

                        data = {'tr_score':1,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':np.ones_like(o_grid),'curr_map':o_grid,'curr_maskv':np.ones_like(o_grid),'curr_o_masko':np.zeros_like(o_grid),
                                
                                'addressable':{'iobj':list(set(iobj_list)),'oobj':ogridname}}

                        transform_res.append(data)
                        


                        break 

                    elif sel_flag1: 
                        
                        
                        

                        
                        

                        
                        
                        
                        
                        
                        
                        


                        
                        bkg_cand_colors = []; true_bkg = False
                        for iobj_ in initial_global_parsings[gridn]['i']:
                            if initial_global_parsings[gridn]['i'][iobj_]['properties']['parsing_description'][0] in ['background']:
                                bkg_cand_colors.append(initial_global_parsings[gridn]['i'][iobj_]['map'][0,0])
                        if len(bkg_cand_colors)==1: true_bkg = True


                        if 'bkg_case' not in dict_of_iobjs[iobj]: info_ = None
                        else:
                            info_ca = dict_of_iobjs[iobj]['bkg_case']
                            info_co = dict_of_iobjs[iobj]['bkg_color']
                        bkg_dets = {}
                        if info_ is not None:
                            if colr1 == info_co:
                                if info_ca == 'frame_color': bkg_dets = {'bkg_mtd':'choose_iframe_color','bkg_color':colr1}
                                elif info_ca == 'common_color': bkg_dets = {'bkg_mtd':'choose_common_color','bkg_color':colr1}
                            else: 
                                if true_bkg and colr1 == bkg_cand_colors[0]:
                                    bkg_dets = {'bkg_mtd':'actual_bkg_objs_color','bkg_color':colr1}
                                else:
                                    bkg_dets = {'bkg_mtd':'choose_hyperp_color','bkg_color':colr1}
                        else:
                            if true_bkg and colr1 == bkg_cand_colors[0]:
                                bkg_dets = {'bkg_mtd':'actual_bkg_objs_color','bkg_color':colr1}
                            else:
                                bkg_dets = {'bkg_mtd':'choose_hyperp_color','bkg_color':colr1}        
                        


                        frame_width_prepended, frame_widths_appended, frame_height_prepended , frame_heights_appended = simple_tiling_convert(sel_set1, all_matches, o_grid)
                        
                        

                        if frame_width_prepended is None: continue

                        set_of_tls = [all_matches[_][1] for _ in sel_set1]
                        min_row = np.min([_[0] for _ in set_of_tls])
                        set_of_cols = np.unique([_[1] for _ in set_of_tls if _[0]==min_row])
                        min_col = np.min([_[1] for _ in set_of_tls])
                        set_of_rows = np.unique([_[0] for _ in set_of_tls if _[1]==min_col])

                        num_rows = len(set_of_rows)
                        num_cols = len(set_of_cols)
                        if num_rows==1 and num_cols==1: continue 
                        tiles = []
                        for r in range(num_rows):
                            for c in range(num_cols):
                                cdt = (int(set_of_rows[r]),int(set_of_cols[c]))
                                sel_ix = ix_of_x_in_y(x=cdt,y=set_of_tls)
                                if sel_ix is None: tiles.append({'iobj':None,'color_changes':None}); continue
                                match_ix = sel_set1[sel_ix]
                                if np.all([pair[0]==pair[1] for pair in all_matches[match_ix][2]]): tiles.append({'iobj':all_matches[match_ix][0],'color_changes':None})
                                else: tiles.append({'iobj':all_matches[match_ix][0],'color_changes':all_matches[match_ix][2]})
                        
                        
                        
                        
                        
                        
                        
                        
                        


                        iobj_list = [_['iobj'] for _ in tiles if _['iobj'] is not None]
                        if len(iobj_list)==0: continue
                        
                        

                        try:
                            safe_tile_ixs = [list(set(iobj_list)).index(_['iobj']) for _ in tiles]
                        except: continue 
                        construction = {'n_rows':num_rows,'n_cols':num_cols,'safe_tile_ixs':safe_tile_ixs,'tiles':tiles,'frame_specs':[frame_width_prepended, frame_widths_appended, frame_height_prepended , frame_heights_appended],
                                        'bkg_dets':bkg_dets}


                        
                        mask = np.ones_like(o_grid)
                        map = o_grid
                        maskv = mask.copy()
                        masko = np.zeros_like(o_grid)
                        ogridname = create_name()
                        global_parsings[gridn]['o'][ogridname] = {'parsing_type':'fullgrid_oobj_for_gridmap','obj_score':0.5,'mask':mask,'map':map,'maskv':maskv,'masko':masko,'properties':{'is_straightforward_obj':False, 'parsing_description':['fullgrid_o_for_gridmap',0.5,None]}}


                        serial_transforms = [{'type':'iobjs_tile_creation'}]; serial_params = [{'details':construction}]

                        data = {'tr_score':1,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':np.ones_like(o_grid),'curr_map':o_grid,'curr_maskv':np.ones_like(o_grid),'curr_o_masko':np.zeros_like(o_grid),
                                
                                'addressable':{'iobj':list(set(iobj_list)),'oobj':ogridname}}

                        transform_res.append(data)


                        break 
        except: pass

                
                



        try:
            TOMULTISLOTEXT = True
            ext_solved = []

            starttime3 = time.perf_counter_ns()
            def esc3(): 
                QUIT_SECS = 20
                currtime3 = time.perf_counter_ns()
                if ((currtime3-starttime3)/1000000000) > QUIT_SECS: return True
                return False
            
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                
                if not TOMULTISLOTEXT: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]
                full_dict_of_iobjs = initial_global_parsings[gridn]['i'] 

                dict_of_iobjs = {}; temp=[]; tempkey=[] 
                for iobj in full_dict_of_iobjs:
                    i_mask, i_map, i_masko = full_dict_of_iobjs[iobj]['mask'], full_dict_of_iobjs[iobj]['map'], full_dict_of_iobjs[iobj]['masko']
                    if is_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp):
                        ix = ix_of_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp)
                        dict_of_iobjs[tempkey[ix]]['iobj_list'].append(iobj)
                    else: temp.append([np.where(i_mask, i_map, 100), i_masko]); tempkey.append(iobj); dict_of_iobjs[iobj] = full_dict_of_iobjs[iobj]; dict_of_iobjs[iobj]['iobj_list'] = [iobj]

                

                

                

                shape_ = []; iobjs_ = []
                for iobj in dict_of_iobjs:
                    iobjs_.append(iobj); shape_.append([get_shape_of_obj(dict_of_iobjs[iobj]['mask'],dict_of_iobjs[iobj]['map'])])
                lbls, _ = label_unique_with_IDs(shape_) 


                for lbl in range(max(lbls)+1):
                    if esc(): break
                    if esc3(): break
                    curr_iobj_set = [iobjs_[i] for i in range(len(lbls)) if lbls[i] == lbl]
                    if len(curr_iobj_set) == 1: continue 
                    
                    

                    

                    main_chains = [] 
                    
                    main_chains.append([{'fn':static,'params':{}}])
                    main_chains.append([{'fn':recolor,'params':{}}])
                    main_chains.append([{'fn':expand,'params':{}}])
                    main_chains.append([{'fn':flip,'params':{}}])

                    for n in range(len(curr_iobj_set)):
                        iobj = curr_iobj_set[n]
                        i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']
                        o_region = copy.deepcopy(i_mask) 
                        o_region = o_region[0:0+o_grid.shape[0],0:0+o_grid.shape[1]] 
                        o_masko = np.zeros_like(o_region)
                        oobj = None
                        res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oreg')

                        qualifying_chains = []
                        for re in res: qualifying_chains.append(re['current_run'])

                        main_chains = qualifying_chains 

                    
                    
                    
                    
                    unique_chains = []; currobjdets = []
                    for re in res:
                        if not is_x_in_y(x=[re['curr_mask'],re['curr_map'],re['curr_o_masko']],y=currobjdets):
                            unique_chains.append(re['current_run'])
                            currobjdets.append([re['curr_mask'],re['curr_map'],re['curr_o_masko']])

                    

                    
                    
                    

                    
                    
                    
                    
                    
                    
                    

                    
                    

                    if len(unique_chains) == 0: continue
                        
                    chain_ = unique_chains[0] 


                    

                    all_unprocessed_candidates = {}; store_curr_maskmaps = {}
                    for n in range(len(curr_iobj_set)): 
                        if esc3(): break
                        if esc(): break
                        iobj = curr_iobj_set[n]
                        i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']
                        o_region = copy.deepcopy(i_mask) 
                        o_region = o_region[0:0+o_grid.shape[0],0:0+o_grid.shape[1]] 
                        o_masko = np.zeros_like(o_region)
                        oobj = None
                        res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = [chain_], oobj_or_oreg_mode = 'oreg')
                        re = res[0] 
                        curr_mask, curr_map = re['curr_mask'], re['curr_map']
                        store_curr_maskmaps[iobj] = {'curr_mask':curr_mask,'curr_map':curr_map}


                        ogrid = o_grid
                        currmask = curr_mask
                        igrid = i_grid
                        curr_map = curr_map

                        
                        i_bordermask = get_outline_border_mask(currmask,1)
                        
                        mod_i_grid = np.where(curr_mask, curr_map, i_grid)
                        i_bordercolors = get_colors_of_obj(i_bordermask,mod_i_grid)
                        iobj_igrid_colors = get_colors_of_obj(curr_mask, mod_i_grid)
                        iobj_currmap_colors = get_colors_of_obj(curr_mask, curr_map)
                        i_bordercolors_ = list(mod_i_grid[i_bordermask==1])
                        commonest_i_bordercolor = max(set(i_bordercolors_), key=i_bordercolors_.count) if len(i_bordercolors_)>0 else None

                        
                        
                        if currmask.shape != ogrid.shape: pass ##

                        opt1 = ogrid.copy(); rows,cols = np.where(currmask==1) 
                        opt1[rows,cols] = -99 
                        o_objmasks_1 = get_contiguous_regions(opt1,background_color=-99,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)
                        ipt1 = igrid.copy(); rows,cols = np.where(currmask==1) 
                        ipt1[rows,cols] = -99
                        i_objmasks_1 = get_contiguous_regions(ipt1,background_color=-99,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)

                        
                        dirns_unprocessed_candidates = {}

                        for D in range(8):
                            if esc3(): break
                            if esc(): break
                            directions =  ['S','SW','SE','W','E','N','NW','NE']
                            directions_ = [(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)]   
                            dir = directions[D]; dir_tuple = directions_[D] 

                            
                            mask_in_dir, cdts_of_line_start_and_obj_end = mask_in_direction(currmask,dir) 
                            unique_to_verify_dir, anti_mask = get_mask_unique_to_verify_dir(dir,currmask) 
                            linestart_cdts_all = [cdt_linestart for cdt_linestart, cdt_edgeend in cdts_of_line_start_and_obj_end]
                            
                            orthogonalbands = bands_in_dir(currmask.shape, orthogonal_dirn(dir_tuple))
                            linestart_cdts_all_bands = [orthogonalbands[cdt] for cdt in linestart_cdts_all]
                            ordered_linestart_cdts = [v for _, v in sorted(zip(linestart_cdts_all_bands, linestart_cdts_all))]
                            samedirbands = bands_in_dir(currmask.shape, dir_tuple)
                            if len(ordered_linestart_cdts) == 0: continue

                            extmodes = ['fullwidth','central'] if len(ordered_linestart_cdts) > 1 else ['central']

                            candidates = []
                            unprocessed_candidates = []

                            for EXTMODE in extmodes:

                                if EXTMODE == 'central':
                                    middle_cdt = ordered_linestart_cdts[len(ordered_linestart_cdts)//2]
                                    region_mask = np.zeros_like((currmask))
                                    r,c = middle_cdt 
                                    for k in range(30):
                                        if r < 0 or r >= currmask.shape[0] or c < 0 or c >= currmask.shape[1]: break
                                        else:
                                            region_mask[r,c] = 1
                                            r += dir_tuple[0]
                                            c += dir_tuple[1] 

                                    initial_region_mask = np.zeros_like((currmask))
                                    r,c = middle_cdt
                                    initial_region_mask[r,c] = 1

                                elif EXTMODE == 'fullwidth':
                                    region_mask = np.zeros_like((currmask))
                                    for n,cdt_linestart in enumerate(ordered_linestart_cdts): 
                                        r,c = cdt_linestart
                                        for k in range(30):
                                            if r < 0 or r >= currmask.shape[0] or c < 0 or c >= currmask.shape[1]: break
                                            else:
                                                region_mask[r,c] = 1
                                                r += dir_tuple[0]
                                                c += dir_tuple[1] 

                                    initial_region_mask = np.zeros_like((currmask))
                                    for n,cdt_linestart in enumerate(ordered_linestart_cdts):
                                        r,c = cdt_linestart
                                        initial_region_mask[r,c] = 1

                                relevant_bands = np.unique(samedirbands[region_mask==1]).astype(int)

                                
                                region_pxls_of_orthogleakyobjs = np.zeros_like(region_mask) 
                                border_pxls_of_orthogleakyobjs = np.zeros_like(region_mask) 
                                for b in relevant_bands:
                                    currbandmask = (samedirbands==b).astype(int) 
                                    curractiveregion = currbandmask & region_mask
                                    currobjs = [int(_) for _ in     np.unique(o_objmasks_1[curractiveregion==1])   ] 

                                    for obj in currobjs:
                                        objmask = (o_objmasks_1==obj).astype(int)
                                        currsliver = (objmask & currbandmask) 
                                        
                                        immediateborder = get_outline_border_mask(curractiveregion)
                                        isleak = True if np.sum(currsliver & immediateborder) > 0 else False 
                                        if isleak:
                                            curractivesliver = curractiveregion & objmask
                                            rows,cols = np.where(curractivesliver==1)
                                            region_pxls_of_orthogleakyobjs[rows,cols] = 1

                                            temp = currsliver & immediateborder
                                            rows,cols = np.where(temp==1)
                                            border_pxls_of_orthogleakyobjs[rows,cols] = 1

                                
                                
                                region_pxls_of_atallleakyobjs = np.zeros_like(region_mask)  
                                for b in relevant_bands:
                                    currbandmask = (samedirbands==b).astype(int) 
                                    curractiveregion = currbandmask & region_mask
                                    currobjs = [int(_) for _ in     np.unique(o_objmasks_1[curractiveregion==1])   ] 

                                    for obj in currobjs:
                                        objmask = (o_objmasks_1==obj).astype(int)
                                        currsliver = (objmask & currbandmask) 

                                        
                                        
                                        border_pxls_of_atallleakyobjs = (objmask & (anti_mask==0).astype(int) & get_outline_border_mask(region_mask))
                                        objleaked = False if np.sum(border_pxls_of_atallleakyobjs & ~region_mask) == 0 else True
                                        temp = (border_pxls_of_atallleakyobjs & ~region_mask)
                                        

                                        

                                        if objleaked and np.sum(temp & ~border_pxls_of_orthogleakyobjs)>0:
                                            curractivesliver = curractiveregion & objmask
                                            rows,cols = np.where(curractivesliver==1)
                                            region_pxls_of_atallleakyobjs[rows,cols] += 1     

                                leak_mask = region_pxls_of_orthogleakyobjs*2+region_pxls_of_atallleakyobjs 


                                


                                
                                endchanges = [] 
                                for endband in range(1,len(relevant_bands)):
                                    map_before = ogrid[((samedirbands==(relevant_bands[endband]-1)).astype(int) & region_mask)==1]
                                    map_at = ogrid[((samedirbands==(relevant_bands[endband])).astype(int) & region_mask)==1]
                                    if not are_two_identical(map_before, map_at): endchanges.append(endband); 
                                    

                                    
                                    existing_map_before = mod_i_grid[((samedirbands==(relevant_bands[endband]-1)).astype(int) & region_mask)==1]
                                    existing_map_at = mod_i_grid[((samedirbands==(relevant_bands[endband])).astype(int) & region_mask)==1]
                                    if np.any([currcolor in i_bordercolors for currcolor in existing_map_before]): 
                                        if np.all([currcolor not in i_bordercolors for currcolor in existing_map_at]): 
                                            endchanges.append(endband)

                                if len(relevant_bands) > 1: endchanges.append(endband+1) 
                                else: endchanges.append(1)
                                
                                


                                matched_initial_region = False
                                for endband in endchanges:
                                    subset_relevant_bands = relevant_bands[:endband]

                                    fs=[];ls=[];     new_colors = []; prevcolors = []; ncb=[];   cum_activeregion = np.zeros_like(region_mask); 
                                    colors_upto_initial_region = []; all_colors = []
                                    ls_cdts = []

                                    for b in subset_relevant_bands:
                                        currbandmask = (samedirbands==b).astype(int) 
                                        curractiveregion = currbandmask & region_mask                            ; rows_,cols_ = np.where(curractiveregion==1); ls_cdts.append([rows_,cols_])
                                        
                                        currobjs = [int(_) for _ in     np.unique(o_objmasks_1[curractiveregion==1])   ]
                                        isfull = 1 if len(currobjs) == 1 else 0
                                        isnonleak = 1 if np.max(leak_mask[curractiveregion==1]) == 0 else 0 
                                        fs.append(isfull); ls.append(isnonleak)
                                        if isnonleak == 1:
                                            active_nonleak_region = curractiveregion & (leak_mask==0).astype(int)
                                            currcolors = get_colors_of_obj(active_nonleak_region, ogrid)
                                            
                                            for color in currcolors:
                                                if color not in prevcolors: new_colors.append(color); ncb.append(b)     
                                            prevcolors = currcolors 
                                        
                                        cum_activeregion = cum_activeregion | ((curractiveregion==1).astype(int))
                                        
                                        if not matched_initial_region:
                                            if are_two_identical(initial_region_mask, cum_activeregion): matched_initial_region = True; colors_upto_initial_region = copy.deepcopy(new_colors)

                                        all_colors.extend(get_colors_of_obj(curractiveregion, ogrid))

                                    premature_modifier = 0.5 if not matched_initial_region else 1

                                    endinleak_modifier = 0.5 if 0 not in leak_mask[curractiveregion==1] else 1


                                    
                                    
                                    activemask_ = cum_activeregion & (leak_mask==0).astype(int)
                                    temp_ = np.where(activemask_, ogrid, -99)
                                    activeobjs_ = get_contiguous_regions(temp_,background_color=-99,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)
                                    flag_overlap_preiobj = False
                                    for n_ in range(1,np.max(activeobjs_)+1):
                                        obj_mask_ = (activeobjs_==n_).astype(int)
                                        for iobj_ in global_parsings[gridn]['i']:
                                            if are_two_identical(global_parsings[gridn]['i'][iobj_]['maskv'],obj_mask_):
                                                flag_overlap_preiobj = True
                                                
                                    
                                    overlap_preiobj_modified = 0.5 if flag_overlap_preiobj else 1


                                    
                                    
                                    
                                    
                                    
                                    
                                    if len(new_colors) < 5: colorsc = 1
                                    else: colorsc = 0.5
                                    for colr in list(set(new_colors)): 
                                        colorsc = colorsc * (1/(new_colors.count(colr)))
                                    
                                    
                                    if is_any_x_in_y(x=i_bordercolors,y=new_colors): colorsc = colorsc * 0.7 

                                    if is_any_x_in_y(x=i_bordercolors,y=all_colors): colorsc = colorsc * 0.95 



                                    cum_fs = fs; cum_ls = ls
                                    if len(cum_fs) > 2 and 0 not in cum_fs[:int(len(cum_fs)/2)]: cfscore = 1
                                    elif len(cum_fs) <= 2 and 0 not in cum_fs: cfscore = 1
                                    else: cfscore = np.mean(cum_fs)
                                    if len(cum_ls) > 2 and cum_ls.count(1) > 2: clscore = 1 
                                    elif len(cum_ls) <= 2 and cum_ls.count(1) == len(cum_ls): clscore = 1
                                    else: clscore = np.mean(cum_ls)
                                    
                                    
                                    
                                    
                                    
                                    activeleakmask = (leak_mask!=0).astype(int) & cum_activeregion 


                                    
                                    

                                    

                                    

            
                                    
                                    
                                    oextgrid = ogrid.copy(); rows,cols = np.where(cum_activeregion==0) 
                                    oextgrid[rows,cols] = -99 

                                    
                                    nonleaky_activeregion = ((cum_activeregion==1).astype(int) & (activeleakmask==0).astype(int)).astype(int)
                                    activeleak_pseudoobjs = get_contiguous_regions(activeleakmask,background_color=0,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)
                                    for alpo in range(1,np.amax(activeleak_pseudoobjs)+1):
                                        po_mask = (activeleak_pseudoobjs==alpo).astype(int)
                                        
                                        rtofill,ctofill = np.where(po_mask==1)
                                        valid_neighbors_mask = ((get_outline_border_mask(po_mask,1)==1).astype(int) & (nonleaky_activeregion==1).astype(int)).astype(int)
                                        if np.sum(valid_neighbors_mask)==0: pass
                                        else:
                                            valid_neighbor_colors = ogrid[valid_neighbors_mask==1]
                                            oextgrid[rtofill,ctofill] = valid_neighbor_colors[0] 
                                    
                                    
                                    oext_objs = get_contiguous_regions(oextgrid,background_color=-99,diagonal_connections_allowedQ=True,colourblind_spatial_contiguity_mode=False)
                                    
                                    newobjs = []; newobj_b = []; newobj_colors = []; prevobjs = []
                                    for b in subset_relevant_bands:
                                        currbandmask = (samedirbands==b).astype(int) 
                                        curractiveregion = currbandmask & region_mask
                                        currobjs = [int(_) for _ in np.unique(oext_objs[curractiveregion==1]) if int(_)!=0] 
                                        for cobj in currobjs:
                                            if cobj not in prevobjs:
                                                newobjs.append(cobj)
                                                newobj_b.append(b)
                                                newobj_colors.append(get_colors_of_obj((oext_objs==cobj).astype(int),oextgrid))
                                        prevobjs = currobjs
                                    
                                    try: bandskip = subset_relevant_bands[1]-subset_relevant_bands[0]
                                    except: bandskip = 1

                                    rule_info = []; first_igrid_color = None; first_currmap_color = None
                                    for n in range(len(newobjs)):
                                        
                                        
                                        
                                        
                                        if n == 0:
                                            why_began = ['first_obj']

                                            prevb = newobj_b[n]-bandskip 
                                            currbandmask = (samedirbands==prevb).astype(int) 
                                            currmask_activeregion = currbandmask & currmask 
                                            why_this_color = []
                                            this_color = newobj_colors[n]
                                            if np.sum(currmask_activeregion)!=0:
                                                rows, cols = np.where(currmask_activeregion==1)
                                                midpxl = (rows[len(rows)//2], cols[len(cols)//2])
                                                currmap_color = [curr_map[midpxl]] 
                                                igrid_color = [igrid[midpxl]]
                                                first_igrid_color = igrid_color; first_currmap_color = currmap_color 
                                                if this_color == igrid_color: why_this_color.append('i_grid_color_of_prev')
                                                if this_color == currmap_color: why_this_color.append('curr_map_color_of_prev')
                                            why_this_color.append(('hyperp_color',this_color))

                                        else:
                                            why_began = ['coz_prev_ended']

                                            prevb = newobj_b[n]-bandskip
                                            currbandmask = (samedirbands==prevb).astype(int) 
                                            curractiveregion = currbandmask & region_mask
                                            why_this_color = []
                                            this_color = newobj_colors[n]
                                            if np.sum(curractiveregion)!=0:
                                                igrid_colors = get_colors_of_obj(curractiveregion,igrid)
                                                if this_color == igrid_colors: why_this_color.append('i_grid_color_of_prev')
                                            why_this_color.append(('hyperp_color',this_color))
                                        
                                        

                                        why_this_len = []
                                        

                                        
                                        thisobj_bs = []
                                        currshape_regionmask = (oext_objs==newobjs[n]).astype(int)
                                        for b in subset_relevant_bands:
                                            currbandmask = (samedirbands==b).astype(int) 
                                            curractiveregion = currbandmask & currshape_regionmask
                                            if np.sum(curractiveregion)>0:
                                                thisobj_bs.append(b) 

                                        
                                        edgecap = False
                                        if n == len(newobjs)-1:
                                            reached_edge = True if relevant_bands[-1] in subset_relevant_bands else False
                                            if reached_edge: 
                                                why_this_len.append('reached_edge')
                                                edgecap = True

                                        
                                        why_this_len.append(('hyperp_length',len(thisobj_bs)))

                                        if not edgecap:
                                            
                                            nextb = thisobj_bs[-1]+bandskip
                                            currbandmask = (samedirbands==nextb).astype(int) 
                                            curractiveregion = currbandmask & region_mask
                                            if np.sum(curractiveregion)!=0:
                                                nextband_igrid_colors = get_colors_of_obj(curractiveregion,igrid)
                                                thisobj_igrid_colors = get_colors_of_obj((oext_objs==newobjs[n]).astype(int),igrid)

                                                
                                                isnovelcolor = False
                                                for color_ in nextband_igrid_colors:
                                                    if color_ not in thisobj_igrid_colors:
                                                        isnovelcolor = True
                                                
                                                if isnovelcolor:
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    

                                                    
                                                    
                                                    
                                                    why_this_len.append(('encounters_this_specific_hyperp_color',nextband_igrid_colors))
                                                    
                                                    if commonest_i_bordercolor is not None and np.any([colr != commonest_i_bordercolor for colr in nextband_igrid_colors]): why_this_len.append('encounters_a_non_commonestbordercolor')
                                                    
                                                    if np.any([colr not in i_bordercolors for colr in nextband_igrid_colors]): why_this_len.append('encounters_a_non_border_color')
                                                    
                                                    if np.any([colr in iobj_igrid_colors for colr in nextband_igrid_colors]): why_this_len.append('encounters_an_igrid_color')
                                                    
                                                    if np.any([colr in iobj_currmap_colors for colr in nextband_igrid_colors]): why_this_len.append('encounters_a_currmap_color')
                                        
                                                    


                                        
                                        rule_info.append([why_began, why_this_color, why_this_len])




                                    
                                    unprocessed_candidates.append({'rule_info':rule_info,'score':float(cfscore*clscore*colorsc*premature_modifier*endinleak_modifier*overlap_preiobj_modified)+(len(subset_relevant_bands)/1000),'region':cum_activeregion,'activeleakmask':activeleakmask,'dets':[EXTMODE,fs,ls,new_colors,  cfscore, clscore,colorsc, premature_modifier, endinleak_modifier,overlap_preiobj_modified,ls_cdts],'colors_upto_initial_region':colors_upto_initial_region,'extmode':EXTMODE,'subsetlen':len(subset_relevant_bands),'subsetreachededge':True if relevant_bands[-1] in subset_relevant_bands else False})
                            
                            
                            
                            
                            

                            

                            dirns_unprocessed_candidates[dir] = unprocessed_candidates
                        all_unprocessed_candidates[iobj] = dirns_unprocessed_candidates
                    


                    

                    
                    if esc(): break
                    if esc3(): break
                    tempblock = True
                    if not tempblock:
                        for iobj in all_unprocessed_candidates:
                            for diri in all_unprocessed_candidates[iobj]:
                                for entryi in range(len(all_unprocessed_candidates[iobj][diri])):

                                    for jobj in all_unprocessed_candidates:
                                        if esc3(): break
                                        if iobj == jobj: continue
                                        for dirj in all_unprocessed_candidates[jobj]:
                                            for entryj in range(len(all_unprocessed_candidates[jobj][dirj])):
                                            
                                                if np.sum(all_unprocessed_candidates[iobj][diri][entryi]['region'] & all_unprocessed_candidates[jobj][dirj][entryj]['region']) > 0:
                                                    

                                                    if esc(): break

                                                    ls_cdts1 = all_unprocessed_candidates[iobj][diri][entryi]['dets'][-1]
                                                    ls1 = all_unprocessed_candidates[iobj][diri][entryi]['dets'][2]

                                                    ls_cdts2 = all_unprocessed_candidates[jobj][dirj][entryj]['dets'][-1]
                                                    ls2 = all_unprocessed_candidates[jobj][dirj][entryj]['dets'][2]

                                                    
                                                    
                                                    
                                                    

                                                    

                                                    
                                                    width1 = len(ls_cdts1[0][0])
                                                    ls1_dupl = [ls1 for _ in range(width1)]
                                                    ls1_cdts_split = []
                                                    for _ in range(width1):
                                                        temp = []
                                                        for cdts in ls_cdts1:
                                                            temp.append((cdts[0][_],cdts[1][_]))
                                                        ls1_cdts_split.append(temp)
                                                    
                                                    width2 = len(ls_cdts2[0][0])
                                                    ls2_dupl = [ls2 for _ in range(width2)]
                                                    ls2_cdts_split = []
                                                    for _ in range(width2):
                                                        temp = []
                                                        for cdts in ls_cdts2:
                                                            temp.append((cdts[0][_],cdts[1][_]))
                                                        ls2_cdts_split.append(temp)

                                                    
                                                    
                                                    
                                                    

                                                    for w1 in range(width1):
                                                        for ix1 in range(len(ls1_dupl[w1])):
                                                            for w2 in range(width2):
                                                                for ix2 in range(len(ls2_dupl[w2])):
                                                                    if ls1_cdts_split[w1][ix1] == ls2_cdts_split[w2][ix2]:
                                                                        ls1_dupl[w1][ix1] = 1; ls2_dupl[w2][ix2] = 1 

                                                                        
                                                                        all_unprocessed_candidates[iobj][diri][entryi]['activeleakmask'][ls1_cdts_split[w1][ix1]] = 0
                                                    
                                                    for ix in range(len(ls1)):
                                                        if 0 not in [ls1_dupl[_][ix] for _ in range(width1)]:
                                                            ls1[ix] = 1 
                                                    for ix in range(len(ls2)):
                                                        if 0 not in [ls2_dupl[_][ix] for _ in range(width2)]:
                                                            ls2[ix] = 1 

                                                    

                                                    
                                                    all_unprocessed_candidates[iobj][diri][entryi]['dets'][2] = ls1

                                                    cum_ls = ls1
                                                    if len(cum_ls) > 2 and cum_ls.count(1) > 2: clscore = 1 
                                                    elif len(cum_ls) <= 2 and cum_ls.count(1) == len(cum_ls): clscore = 1
                                                    else: clscore = np.mean(cum_ls)
                                                    all_unprocessed_candidates[iobj][diri][entryi]['dets'][5] = clscore
                                                    
                                                    all_unprocessed_candidates[iobj][diri][entryi]['score'] = float(np.prod(all_unprocessed_candidates[iobj][diri][entryi]['dets'][4:9+1]))+(all_unprocessed_candidates[iobj][diri][entryi]['subsetlen']/1000)


                                                    all_unprocessed_candidates[jobj][dirj][entryj]['dets'][2] = ls2
                                                    cum_ls = ls2
                                                    if len(cum_ls) > 2 and cum_ls.count(1) > 2: clscore = 1 
                                                    elif len(cum_ls) <= 2 and cum_ls.count(1) == len(cum_ls): clscore = 1
                                                    else: clscore = np.mean(cum_ls)
                                                    all_unprocessed_candidates[jobj][dirj][entryj]['dets'][5] = clscore

                                                    all_unprocessed_candidates[jobj][dirj][entryj]['score'] = float(np.prod(all_unprocessed_candidates[jobj][dirj][entryj]['dets'][4:9+1]))+(all_unprocessed_candidates[jobj][dirj][entryj]['subsetlen']/1000)

                                                    


                    
                    all_iobjs_all_candidates = {}
                    for iobj in all_unprocessed_candidates:
                        if esc3(): break
                        all_candidates = {}
                        for diri in all_unprocessed_candidates[iobj]:
                            candidates = all_unprocessed_candidates[iobj][diri]
                            sorted_candidates = sorted(candidates, key=lambda d: d['score'], reverse=True)

                            if esc(): break

                            
                            
                            
                            
                            
                            
                            
                            
                            
                            

                            all_candidates[diri] = sorted_candidates
                        all_iobjs_all_candidates[iobj] = all_candidates


                    

                    if esc(): break

                    iobj_keys = [k for k in all_iobjs_all_candidates]
                    all_directions =  ['S','SW','SE','W','E','N','NW','NE']
                    directions = ['S','SW','SE','W','E','N','NW','NE']
                    
                    ext_objn = 0

                    all_colorcand_dets = []; all_lencand_dets = [] 
                    for dir_ in directions:
                        if esc3(): break
                        scores_ = []; rule_infos_ = []; extmodes_ = []; color_rules_ = []; len_rules_ = []
                        for iobj_key in iobj_keys: 
                            try: cands = all_iobjs_all_candidates[iobj_key][dir_]
                            except: cands = []
                            temp = []; temp_color_rules = []; temp_len_rules = []
                            for cand in cands:
                                try: temp.append(cand['rule_info'][ext_objn]); temp_color_rules.extend(cand['rule_info'][ext_objn][1]); temp_len_rules.extend(cand['rule_info'][ext_objn][2])
                                except: pass 
                            rule_infos_.append(temp); scores_.append([cand['score'] for cand in cands]); extmodes_.append([cand['extmode'] for cand in cands])
                            if len(temp_color_rules)!=0:
                                _, uniques_ = label_unique_with_IDs(temp_color_rules)
                                color_rules_.append(uniques_) 
                            else: color_rules_.append([])
                            if len(temp_len_rules)!=0:
                                _, uniques_ = label_unique_with_IDs(temp_len_rules)
                                len_rules_.append(uniques_)
                            else: len_rules_.append([])

                        rule_cands_ = []
                        for _ in color_rules_: rule_cands_.extend(_)
                        if len(rule_cands_)!=0: _, color_rule_cands = label_unique_with_IDs(rule_cands_)
                        colorcand_dets = [] 
                        for opt in color_rule_cands: 
                            validity_ = []
                            for cn in range(len(color_rules_)): 
                                if opt in color_rules_[cn]: 
                                    
                                    hsc = 0
                                    for iobj_key in iobj_keys:
                                        try: 
                                            cands = all_iobjs_all_candidates[iobj_key][dir_]        
                                            for cand in cands:
                                                try:
                                                    csc = cand['score']
                                                    cropts = cand['rule_info'][ext_objn][1]
                                                    if is_x_in_y(x=opt,y=cropts) and csc > hsc: hsc = csc
                                                except: pass
                                        except: pass
                                    validity_.append(hsc) 
                                else: validity_.append(0)
                            colorcand_dets.append([dir_,opt, validity_, None])
                
                        rule_cands_ = []
                        for _ in len_rules_: rule_cands_.extend(_)
                        if len(rule_cands_)!=0: _, len_rule_cands = label_unique_with_IDs(rule_cands_)
                        lencand_dets = []
                        for opt in len_rule_cands:
                            validity_ = [];              opts_hyperp_colors = []
                            for cn in range(len(len_rules_)):
                                if opt in len_rules_[cn]: 

                                    
                                    for crl in color_rules_[cn]:
                                        if type(crl)==tuple: opts_hyperp_colors.append(crl[1])

                                    
                                    hsc = 0
                                    for iobj_key in iobj_keys:
                                        try: 
                                            cands = all_iobjs_all_candidates[iobj_key][dir_]        
                                            for cand in cands:
                                                try:
                                                    csc = cand['score']
                                                    lropts = cand['rule_info'][ext_objn][2]
                                                    if is_x_in_y(x=opt,y=lropts) and csc > hsc: hsc = csc
                                                except: pass
                                        except: pass                 

                                    
                                    iobj_key = iobj_keys[cn]
                                    hsc = 0
                                    try: 
                                        cands = all_iobjs_all_candidates[iobj_key][dir_]        
                                        for cand in cands:
                                            try:
                                                csc = cand['score']
                                                lropts = cand['rule_info'][ext_objn][2]
                                                if is_x_in_y(x=opt,y=lropts) and csc > hsc: hsc = csc
                                            except: pass
                                    except: pass     
                                    
                                    validity_.append(hsc) 
                                elif 'reached_edge' in len_rules_[cn]: 
                                    
                                    
                                    hsc = 0
                                    for iobj_key in iobj_keys:
                                        try: 
                                            cands = all_iobjs_all_candidates[iobj_key][dir_]        
                                            for cand in cands:
                                                try:
                                                    csc = cand['score']
                                                    lropts = cand['rule_info'][ext_objn][2]
                                                    if is_x_in_y(x='reached_edge',y=lropts) and csc > hsc: hsc = csc
                                                except: pass
                                        except: pass                                   

                                    
                                    iobj_key = iobj_keys[cn]
                                    hsc = 0
                                    try: 
                                        cands = all_iobjs_all_candidates[iobj_key][dir_]        
                                        for cand in cands:
                                            try:
                                                csc = cand['score']
                                                lropts = cand['rule_info'][ext_objn][2]
                                                if is_x_in_y(x='reached_edge',y=lropts) and csc > hsc: hsc = csc
                                            except: pass
                                    except: pass    

                                    validity_.append(0.8*hsc) 
                                else: validity_.append(0)
                            if len(opts_hyperp_colors)==0: continue
                            _, unique_opts_hyperp_colors = label_unique_with_IDs(opts_hyperp_colors)
                            
                            lencand_dets.append([dir_,opt, validity_, None, opts_hyperp_colors])

                        if len(lencand_dets)==0 or len(colorcand_dets)==0: continue

                        
                        
                        
                        for j in range(len(lencand_dets)):
                            i_color_rule = ('hyperp_color', [np.int64(9)]) 
                            j_len_rule = lencand_dets[j][1]

                            equistatics = []
                            
                            if ext_objn ==0: currentry = [['first_obj'],[i_color_rule],[j_len_rule]]
                            else: currentry = [['coz_prev_ended'],[i_color_rule],[j_len_rule]]
                            temp_ = []
                            temp_.append(currentry)
                            recon_cand = [{'rule_info':temp_,'extmode':'central','dir':dir_}] 
                            ext_details = {'ext_fn_characterisation':recon_cand}
                            for iobj_key in iobj_keys:
                                curr_mask_, curr_map_ = store_curr_maskmaps[iobj_key]['curr_mask'], store_curr_maskmaps[iobj_key]['curr_map']
                                o1map,o2mask = extension(curr_map_, curr_mask_, ext_details)
                                if are_two_identical(curr_mask_, o2mask) and are_two_identical(curr_map_, o1map): match_ = 1
                                else: match_ = 0
                                equistatics.append(match_)
                                
                            lencand_dets[j][3] = equistatics
                        
                        all_colorcand_dets.extend(colorcand_dets)
                        all_lencand_dets.extend(lencand_dets)


                    
                    
                    

                    
                    

                    categoricals = []
                    if esc3(): break
                    for opt in all_lencand_dets:
                        if np.all([opt[2][_]>0.8 or opt[3][_]==1 for _ in range(len(opt[2]))]):
                            categorical = [0]*len(opt[2])
                            
                            categoricals.append(categorical)

                    _, segment_options = label_unique_with_IDs([[opt[2][_]>0.8 or opt[3][_]==1 for _ in range(len(opt[2]))] for opt in all_lencand_dets]); subgroup_options = []
                    for i in range(len(segment_options)):
                        for j in range(len(segment_options)):
                            if i==j: continue
                            i_v = segment_options[i]
                            j_v = segment_options[j]
                            if np.all([i_v[_]+j_v[_]==1 for _ in range(len(i_v))]): subgroup_options.append(sorted([i,j]))
                            elif np.all(j_v): subgroup_options.append([i]) 
                    if len(subgroup_options)==0: continue
                    _, subgroup_options = label_unique_with_IDs(subgroup_options) 
                    for so in range(len(subgroup_options)):
                        categorical = [0]*len(segment_options[0]) 
                        for c, seg_ix in enumerate(subgroup_options[so]):
                            seg_opt = segment_options[seg_ix] 
                            
                            for _ in range(len(seg_opt)):
                                if seg_opt[_] == True: categorical[_] = c+1 
                        categorical, _ = label_unique_with_IDs(categorical) 
                        
                        categoricals.append(categorical)

                    _, categoricals = label_unique_with_IDs(categoricals)



                    proper_cands = []; tps = []

                    for categorical in categoricals:
                        if esc3(): break
                        if esc(): break
                        
                        

                        tps.append(len(proper_cands))


                        
                        thrs = 0.6
                        if 1 not in categorical:
                            
                            lopts = []
                            for lopt in all_lencand_dets:
                                if np.all([lopt[2][_]>thrs or lopt[3][_]==1 for _ in range(len(lopt[2]))]) and not np.all([lopt[3][_]==1 for _ in range(len(lopt[2]))]):
                                    if lopt[3] not in lopts: lopts.append([sum(lopt[3]),lopt[3],lopt[0]])
                            sortedlopts = sorted(lopts, key=lambda d: d[0])
                            lopts = [_[1] for _ in sortedlopts]; loptdirs = [_[2] for _ in sortedlopts]

                            
                            for e, evs in enumerate(lopts):
                                loptdir = loptdirs[e]
                                temp = {_:[[],[]] for _ in all_directions}
                                
                                lenrules1 = []
                                for lopt in all_lencand_dets:
                                    if loptdir==lopt[0] and lopt[3]==evs and np.all([lopt[2][_]>thrs or evs[_]==1 for _ in range(len(lopt[2]))]) and not np.all([evs[_]==1 for _ in range(len(lopt[2]))]):
                                        lenrules1.append([lopt[0],lopt[1]])
                                        temp[lopt[0]][0].append(lopt[1])
                                
                                colorrules1 = []
                                for copt in all_colorcand_dets:
                                    if loptdir==copt[0] and np.all([copt[2][_]>thrs or evs[_]==1 for _ in range(len(copt[2]))]) and not np.all([evs[_]==1 for _ in range(len(copt[2]))]):
                                        colorrules1.append([copt[0],copt[1]])
                                        temp[copt[0]][1].append(copt[1])

                                flag = False
                                for dkey in temp:
                                    if temp[dkey] == [[],[]]: continue
                                    if len(temp[dkey][0])>0 and len(temp[dkey][1])>0: flag = True; continue
                                    temp[dkey] = [[],[]] 
                                if flag: proper_cands.append({'catn':0,'categorical':categorical,'evs':evs,'rules':temp})

                        else:
                            for catn in range(max(categorical)+1):
                                
                                lopts = []
                                for lopt in all_lencand_dets:
                                    if np.all([lopt[2][_]>thrs or lopt[3][_]==1 for _ in range(len(lopt[2])) if categorical[_]==catn]) and not np.all([lopt[3][_]==1 for _ in range(len(lopt[2])) if categorical[_]==catn]):
                                        if lopt[3] not in lopts: lopts.append([sum(lopt[3]),lopt[3],lopt[0]])
                                sortedlopts = sorted(lopts, key=lambda d: d[0])
                                lopts = [_[1] for _ in sortedlopts]; loptdirs = [_[2] for _ in sortedlopts]

                                
                                for e, evs in enumerate(lopts):
                                    loptdir = loptdirs[e]
                                    temp = {_:[[],[]] for _ in all_directions}
                                    
                                    lenrules1 = []
                                    for lopt in all_lencand_dets:
                                        if loptdir==lopt[0] and lopt[3]==evs and np.all([lopt[2][_]>thrs or evs[_]==1 for _ in range(len(lopt[2])) if categorical[_]==catn]) and not np.all([evs[_]==1 for _ in range(len(lopt[2])) if categorical[_]==catn]):
                                            lenrules1.append([lopt[0],lopt[1]])
                                            temp[lopt[0]][0].append(lopt[1])
                                    
                                    colorrules1 = []
                                    for copt in all_colorcand_dets:
                                        if loptdir==copt[0] and np.all([copt[2][_]>thrs or evs[_]==1 for _ in range(len(copt[2])) if categorical[_]==catn]) and not np.all([evs[_]==1 for _ in range(len(copt[2])) if categorical[_]==catn]):
                                            colorrules1.append([copt[0],copt[1]])
                                            temp[copt[0]][1].append(copt[1])

                                    flag = False
                                    for dkey in temp:
                                        if temp[dkey] == [[],[]]: continue
                                        if len(temp[dkey][0])>0 and len(temp[dkey][1])>0: flag = True; continue
                                        temp[dkey] = [[],[]] 
                                    if flag: proper_cands.append({'catn':catn,'categorical':categorical,'evs':evs,'rules':temp})



                    

                    simed_explanations = []

                    for tpn in range(len(tps)):
                        if esc3(): break
                        if esc(): break
                        tpst = tps[tpn]
                        tpen = len(proper_cands) if tpn == len(tps)-1 else tps[tpn+1]
                        if tpst == tpen: continue 
                        
            
                        tpcategorical = proper_cands[tpst]['categorical']
                        catns = [proper_cands[_]['catn'] for _ in range(tpst,tpen)]
                        if max(tpcategorical) not in catns: continue 
                        
                        
                        

                        

                        dirns_done = [] 
                        recon = np.zeros_like(o_grid)
                        utilised_rules = [{_:[] for _ in all_directions} for catn in range(max(tpcategorical)+1)]

                        iobj_split = []
                        for catn in range(max(tpcategorical)+1):
                            iobj_split.append([iobj_keys[_] for _ in range(len(tpcategorical)) if tpcategorical[_]==catn])

                        for pc in range(tpst,tpen):
                            if esc3(): break
                            

                            
                            
                            collate_recons = []; maskmaps = []
                            for d_ in all_directions: 
                                recon_here = copy.deepcopy(recon)

                                ccands_, lcands_ = proper_cands[pc]['rules'][d_][1], proper_cands[pc]['rules'][d_][0]
                                categ_, catn_ = proper_cands[pc]['categorical'], proper_cands[pc]['catn']
                                if len(ccands_)==0 or len(lcands_)==0: continue

                                ccand_ = ccands_[0] if np.all([type(_)==tuple for _ in ccands_]) else [_ for _ in ccands_ if type(_) is not tuple][0]
                                lcand_ = lcands_[0] if np.all([type(_)==tuple for _ in lcands_]) else [_ for _ in lcands_ if type(_) is not tuple][0]
                                currentry = [['first_obj'],[ccand_],[lcand_]]
                                recon_cand = [{'rule_info':[currentry],'extmode':'central','dir':d_}]
                                ext_details = {'ext_fn_characterisation':recon_cand}
                                for n_ in range(len(categ_)):
                                    if categ_[n_]==catn_:
                                        iobj_key = iobj_keys[n_]
                                        curr_mask_, curr_map_ = store_curr_maskmaps[iobj_key]['curr_mask'], store_curr_maskmaps[iobj_key]['curr_map']
                                        o1map,o2mask = extension(curr_map_, curr_mask_, ext_details)
                                        maskmaps.append([o2mask,o1map])
                                        recon_here = recon_here | o2mask
                                
                                collate_recons.append([d_, recon_here, o2mask, o1map])

                            chosen_ = []
                            for k in range(8):
                                
                                
                                improvements = [np.sum((_[1]==1)&(recon==0)) for _ in collate_recons]

                                
                                already_done = [_[0] in dirns_done for _ in collate_recons]

                                for _ in range(len(improvements)):
                                    if already_done[_]: improvements[_]=0

                                if max(improvements)<=0: break
                                
                                ix = np.argmax(improvements)
                                chosen_.append(collate_recons[ix][0])
                                d_ = collate_recons[ix][0]

                                ccands_, lcands_ = proper_cands[pc]['rules'][d_][1], proper_cands[pc]['rules'][d_][0]
                                categ_, catn_ = proper_cands[pc]['categorical'], proper_cands[pc]['catn']
                                if len(ccands_)==0 or len(lcands_)==0: continue

                                if d_ not in dirns_done: dirns_done.append(d_)
                                else: print('ERROR'); continue 

                                ccand_ = ccands_[0] if np.all([type(_)==tuple for _ in ccands_]) else [_ for _ in ccands_ if type(_) is not tuple][0]
                                lcand_ = lcands_[0] if np.all([type(_)==tuple for _ in lcands_]) else [_ for _ in lcands_ if type(_) is not tuple][0]
                                
                                currentry = [['first_obj'],[ccand_],[lcand_]]

                                utilised_rules[catn_][d_] = currentry

                                recon_cand = [{'rule_info':[currentry],'extmode':'central','dir':d_}]
                                ext_details = {'ext_fn_characterisation':recon_cand}
                                for n_ in range(len(categ_)):
                                    if categ_[n_]==catn_:
                                        iobj_key = iobj_keys[n_]
                                        curr_mask_, curr_map_ = store_curr_maskmaps[iobj_key]['curr_mask'], store_curr_maskmaps[iobj_key]['curr_map']
                                        o1map,o2mask = extension(curr_map_, curr_mask_, ext_details)
                                        maskmaps.append([o2mask,o1map])
                                        recon = recon | o2mask




                        
                        flag1 = True
                        for sub_ in utilised_rules:
                            if sub_ == {_:[] for _ in all_directions}: flag1 = False; break
                        if flag1: simed_explanations.append({'nexplpxls':np.sum(recon),'recon':recon,'used_rules':utilised_rules,'cats':iobj_split})
                    
                    if len(simed_explanations)==0: continue
                    simed_explanations_ = sorted(simed_explanations, key=lambda d: d['nexplpxls'],reverse=True)
                    
                    
                    best_explanation = simed_explanations_[0]
            



                    
                    results = []
                    for splitn in range(len(best_explanation['cats'])):
                        if esc3(): break
                        curr_iobj_split, curr_utilised_rules = best_explanation['cats'][splitn], best_explanation['used_rules'][splitn]
                        
                        dirn_dicts = []
                        for dir_ in curr_utilised_rules:
                            if curr_utilised_rules[dir_]==[]: continue
                            temp__ = [curr_utilised_rules[dir_]] 
                            dirn_dicts.append({'dir':dir_,'extmode':'central','rule_info':temp__})
                        ext_details = {'ext_fn_characterisation':dirn_dicts}
                        
                        ext_details_ = dirn_dicts
                        
                        ext_dirns = [dirn_dict['dir'] for dirn_dict in ext_details_] 
                        num_dirns = len(ext_dirns) 
                        max_num_objs = max([len(dirn_dict['rule_info']) for dirn_dict in ext_details_]) 
                        num_objs_in_dirns = [(dirn_dict['dir'] , len(dirn_dict['rule_info'])) for dirn_dict in ext_details_] 
                        extmodes_in_dirns = [(dirn_dict['dir'] , dirn_dict['extmode']) for dirn_dict in ext_details_] 
                        
                        
                        hypers_in_dirns = [] 
                        
                        for dirn_dict in ext_details_:
                            dir = dirn_dict['dir']
                            rule_info = dirn_dict['rule_info']
                            hyper_dets = []
                            for objn in range(len(rule_info)):
                                hypercolor = None; hyperlen = None
                                color_rules = rule_info[objn][1]          
                                for rule in color_rules:
                                    if type(rule)==tuple and rule[0]=='hyperp_color':
                                        hypercolor = rule[1]; break
                                len_rules = rule_info[objn][2]; reachededge=False
                                for rule in len_rules:
                                    if type(rule)==tuple and rule[0]=='hyperp_length':
                                        hyperlen = rule[1]; break
                                    if type(rule)==str and rule=='reached_edge':
                                        reachededge = True
                                hyper_dets.append([hypercolor, 'reached_edge' if reachededge else hyperlen])
                            hypers_in_dirns.append((dirn_dict['dir'] , dirn_dict['extmode'], hyper_dets)) 
                        
                        
                        
                        mode_objn_in_dirns = [(dirn_dict['dir'] , dirn_dict['extmode'], len(dirn_dict['rule_info'])) for dirn_dict in ext_details_]
                        
                        sorted_objns = sorted([len(dirn_dict['rule_info']) for dirn_dict in ext_details_]) 
                        num_dirns = len(ext_dirns) 
                        max_objns = max([len(dirn_dict['rule_info']) for dirn_dict in ext_details_])
                        
                        
                        
                        
                        specialcase_flag = False
                        if max_objns == 1 and num_dirns in [1,2]: 
                            all_edged = True
                            for dirn_dict in ext_details_:
                                dir = dirn_dict['dir']
                                rule_info = dirn_dict['rule_info']
                                len_rules = rule_info[0][2]; reachededge=False
                                for rule in len_rules:
                                    if type(rule)==str and rule=='reached_edge':
                                        reachededge = True
                                if reachededge: pass
                                else: all_edged = False
                            if all_edged:
                                specialcase_flag = num_dirns
                        



                        
                        
                        
                        

                        serial_transforms = [] 
                        for chain__ in chain_: serial_transforms.append({'type':chain__['fn'].__name__})
                        serial_params = []
                        for chain__ in chain_: serial_params.append(chain__['params'])

                        serial_transforms.append({'type':'extension','by_multislot':True,'hypers_in_dirns':hypers_in_dirns,'mode_objn_in_dirns':mode_objn_in_dirns,'sorted_objns':sorted_objns,
                                                'num_dirns':num_dirns,'max_objns':max_objns,'specialcase_flag':specialcase_flag})
                        serial_params.append({'ext_details':ext_details})


                        for iobj_ in curr_iobj_split:
                            if esc3(): break

                            
                            curr_mask_, curr_map_ = store_curr_maskmaps[iobj_]['curr_mask'], store_curr_maskmaps[iobj_]['curr_map']
                            o1map,o2mask = extension(curr_map_, curr_mask_, ext_details)
                            
                            o3maskv = np.zeros_like(o2mask); o4masko = np.zeros_like(o2mask)
                            rs,cs = np.where(o2mask==1)
                            for m in range(len(rs)):
                                if o1map[rs[m],cs[m]] == o_grid[rs[m],cs[m]]:
                                    o3maskv[rs[m],cs[m]] = 1
                                else: o4masko[rs[m],cs[m]] = 1

                            flagexists = False
                            for existing_oobj in global_parsings[gridn]['o']:
                                if are_two_identical([o2mask,o1map,o3maskv],[global_parsings[gridn]['o'][existing_oobj]['mask'],global_parsings[gridn]['o'][existing_oobj]['map'],global_parsings[gridn]['o'][existing_oobj]['maskv']]):
                                    flagexists = True; break
                            if not flagexists:
                                new_oobj = create_name()
                                global_parsings[gridn]['o'][new_oobj] = {'parsing_type':'multislot_ext_obj', 'obj_score':0.9,'mask':o2mask,'maskv':o3maskv,'masko':o4masko,'map':o1map,
                                                                'properties':{'is_straightforward_obj':True, 'parsing_description':['NEW_multislot_ext_obj',None,None]}}
                            else:
                                new_oobj = existing_oobj
                                


                            data = {'tr_score':1,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[None],'(i_obj_shapes)':[None],'current_run':None,'curr_mask':o2mask,'curr_map':o1map,'curr_maskv':o3maskv,'curr_o_masko':o4masko,
                                    'addressable':{'iobj':iobj_,'oobj':new_oobj}} 
                            if len(best_explanation['cats'])==1: ext_solved.append(iobj_) 
                            results.append(data)       
                    transform_res.extend(results)


                    if esc(): break
                    if esc3(): break

                    

                    
                    iobj_keys = [k for k in all_iobjs_all_candidates]
                    directions =  ['S','SW','SE','W','E','N','NW','NE']; chosen_rule_info_store = {_:[] for _ in directions}; chosen_allrulecands = {_:{} for _ in directions}
                    for ext_objn in range(10):
                        if esc3(): break
                        
                        tempstore_res_in_dir = []
                        for dir_ in directions:
                            scores_ = []; rule_infos_ = []; extmodes_ = []; color_rules_ = []; len_rules_ = []
                            for iobj_key in iobj_keys: 
                                try: cands = all_iobjs_all_candidates[iobj_key][dir_]
                                except: cands = []
                                temp = []; temp_color_rules = []; temp_len_rules = []
                                for cand in cands:
                                    try: temp.append(cand['rule_info'][ext_objn]); temp_color_rules.extend(cand['rule_info'][ext_objn][1]); temp_len_rules.extend(cand['rule_info'][ext_objn][2])
                                    except: pass 
                                rule_infos_.append(temp); scores_.append([cand['score'] for cand in cands]); extmodes_.append([cand['extmode'] for cand in cands])
                                if len(temp_color_rules)!=0:
                                    _, uniques_ = label_unique_with_IDs(temp_color_rules)
                                    color_rules_.append(uniques_) 
                                else: color_rules_.append([])
                                if len(temp_len_rules)!=0:
                                    _, uniques_ = label_unique_with_IDs(temp_len_rules)
                                    len_rules_.append(uniques_)
                                else: len_rules_.append([])
                            
                            
                            rule_cands_ = []
                            for _ in color_rules_: rule_cands_.extend(_)
                            if len(rule_cands_)!=0: _, color_rule_cands = label_unique_with_IDs(rule_cands_)
                            cand_dets = []
                            for opt in color_rule_cands: 
                                validity_ = []
                                for cn in range(len(color_rules_)): 
                                    if opt in color_rules_[cn]: 
                                        
                                        hsc = 0
                                        for iobj_key in iobj_keys:
                                            try: 
                                                cands = all_iobjs_all_candidates[iobj_key][dir_]        
                                                for cand in cands:
                                                    try:
                                                        csc = cand['score']
                                                        cropts = cand['rule_info'][ext_objn][1]
                                                        if is_x_in_y(x=opt,y=cropts) and csc > hsc: hsc = csc
                                                    except: pass
                                            except: pass
                                        validity_.append(hsc) 
                                    else: validity_.append(0)
                                cand_dets.append([sum(validity_)/len(validity_), opt, validity_])
                            color_cand_dets = sorted(cand_dets, key=lambda d: d[0], reverse=True)
                            
                            color_cand_opts = [_[1] for _ in color_cand_dets] 

                            
                            rule_cands_ = []
                            for _ in len_rules_: rule_cands_.extend(_)
                            if len(rule_cands_)!=0: _, len_rule_cands = label_unique_with_IDs(rule_cands_)
                            cand_dets = []
                            for opt in len_rule_cands:
                                validity_ = []
                                for cn in range(len(len_rules_)):
                                    if opt in len_rules_[cn]: 
                                        
                                        
                                        hsc = 0
                                        for iobj_key in iobj_keys:
                                            try: 
                                                cands = all_iobjs_all_candidates[iobj_key][dir_]        
                                                for cand in cands:
                                                    try:
                                                        csc = cand['score']
                                                        lropts = cand['rule_info'][ext_objn][2]
                                                        if is_x_in_y(x=opt,y=lropts) and csc > hsc: hsc = csc
                                                    except: pass
                                            except: pass                            
                                        
                                        validity_.append(hsc) 
                                    elif 'reached_edge' in len_rules_[cn]: 
                                        
                                        
                                        hsc = 0
                                        for iobj_key in iobj_keys:
                                            try: 
                                                cands = all_iobjs_all_candidates[iobj_key][dir_]        
                                                for cand in cands:
                                                    try:
                                                        csc = cand['score']
                                                        lropts = cand['rule_info'][ext_objn][2]
                                                        if is_x_in_y(x='reached_edge',y=lropts) and csc > hsc: hsc = csc
                                                    except: pass
                                            except: pass                                   
                                        
                                        validity_.append(0.8*hsc) 
                                    else: validity_.append(0)
                                cand_dets.append([sum(validity_)/len(validity_), opt, validity_])
                            len_cand_dets = sorted(cand_dets, key=lambda d: d[0], reverse=True)
                            
                            len_cand_opts = [_[1] for _ in len_cand_dets]



                            
                            
                    
                            

                            equistatics = []; maskmaps_over_iobjs = []
                            
                            if ext_objn ==0: currentry = [['first_obj'],[color_cand_dets[0][1]],[len_cand_dets[0][1]]]
                            else: currentry = [['coz_prev_ended'],[color_cand_dets[0][1]],[len_cand_dets[0][1]]]
                            temp_ = chosen_rule_info_store[dir_].copy()
                            temp_.append(currentry)
                            recon_cand = [{'rule_info':temp_,'extmode':'central','dir':dir_}] 
                            ext_details = {'ext_fn_characterisation':recon_cand}
                            for iobj_key in iobj_keys:
                                
                                
                                
                                
                                
                                
                                curr_mask_, curr_map_ = store_curr_maskmaps[iobj_key]['curr_mask'], store_curr_maskmaps[iobj_key]['curr_map']
                                
                                o1map,o2mask = extension(curr_map_, curr_mask_, ext_details)
                                if are_two_identical(curr_mask_, o2mask) and are_two_identical(curr_map_, o1map): match_ = 1
                                else: match_ = 0
                                equistatics.append(match_)
                                maskmaps_over_iobjs.append([o2mask,o1map])

                            combined_validity = [color_cand_dets[0][2][n]*len_cand_dets[0][2][n] for n in range(len(cand_dets[0][2]))]
                            
                            full_score = np.mean([combined_validity[n] for n in range(len(combined_validity)) if equistatics[n]==0])
            

                            tempstore_res_in_dir.append([full_score, dir_, maskmaps_over_iobjs, currentry, color_cand_opts, len_cand_opts, combined_validity])

                        
                        

                        tempstore_res_in_dir = sorted(tempstore_res_in_dir, key=lambda d: d[0], reverse=True)


                        recontemp = np.zeros_like(o2mask)
                        chosen_dirs = []; thresh_ = 0.6 
                        
                        
                        
                        
                        
                        

                        prevex = 0; prevrecon = None
                        for c in range(len(tempstore_res_in_dir)):
                            for mask_, map_ in tempstore_res_in_dir[c][2]:
                                recontemp += mask_
                            currex = np.sum(recontemp)
                            if tempstore_res_in_dir[c][0] < thresh_: chosen_recontemp = prevrecon; break
                            if currex > prevex: chosen_dirs.append([tempstore_res_in_dir[c][1],c]) 
                            else: chosen_recontemp = prevrecon; break
                            prevrecon = recontemp.copy()

                        
                        
                        
                        
                        for dir__,c__ in chosen_dirs:
                            chosen_rule_info_store[dir__].append(tempstore_res_in_dir[c__][3])
                            
                            if ext_objn not in chosen_allrulecands[dir__]: chosen_allrulecands[dir__][ext_objn] = []
                            chosen_allrulecands[dir__][ext_objn].append([tempstore_res_in_dir[c__][4],tempstore_res_in_dir[c__][5]])
                        directions = [_[0] for _ in chosen_dirs] 

                    
                    finalised_rule = {} 
                    for dir_ in chosen_rule_info_store: 
                        if chosen_rule_info_store[dir_]!=[]: finalised_rule[dir_] = chosen_rule_info_store[dir_]

                    finalised_all_rule_cands = {} 
                    for dir_ in chosen_rule_info_store: 
                        if chosen_rule_info_store[dir_]!=[]: finalised_all_rule_cands[dir_] = chosen_allrulecands[dir_]
                    
        except: pass








        
        # Main i->o detection 
        try:
            for gridn in range(num_demo_grids):
                if esc(): break
                if esc1(): break
                if all_solved: continue
                i_grid = i_grids[gridn]
                o_grid = o_grids[gridn]
                full_dict_of_iobjs = initial_global_parsings[gridn]['i'] 

                dict_of_iobjs = {}; temp=[]; tempkey=[] 
                for iobj in full_dict_of_iobjs:
                    i_mask, i_map, i_masko = full_dict_of_iobjs[iobj]['mask'], full_dict_of_iobjs[iobj]['map'], full_dict_of_iobjs[iobj]['masko']
                    if is_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp):
                        ix = ix_of_x_in_y(x = [np.where(i_mask, i_map, 100), i_masko], y = temp)
                        dict_of_iobjs[tempkey[ix]]['iobj_list'].append(iobj)
                    else: temp.append([np.where(i_mask, i_map, 100), i_masko]); tempkey.append(iobj); dict_of_iobjs[iobj] = full_dict_of_iobjs[iobj]; dict_of_iobjs[iobj]['iobj_list'] = [iobj]


                full_dict_of_oobjs = initial_global_parsings[gridn]['o']
                dict_of_oobjs = {}; temp=[]; tempkey=[]
                for oobj in full_dict_of_oobjs:
                    o_mask, o_map, o_masko = full_dict_of_oobjs[oobj]['mask'], full_dict_of_oobjs[oobj]['map'], full_dict_of_oobjs[oobj]['masko']
                    if is_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp):
                        ix = ix_of_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp)
                        dict_of_oobjs[tempkey[ix]]['oobj_list'].append(oobj)
                    else: temp.append([np.where(o_mask, o_map, 100), o_masko]); tempkey.append(oobj); dict_of_oobjs[oobj] = full_dict_of_oobjs[oobj]; dict_of_oobjs[oobj]['oobj_list'] = [oobj]
                

                

                starttime1 = time.perf_counter_ns()
                for iobj in dict_of_iobjs:
                    if esc(): break
                    if esc1(): break

                    if iobj in solved_iobjs: print('SKIPPING iobj'); continue

        
                    i_mask, i_map = dict_of_iobjs[iobj]['mask'], dict_of_iobjs[iobj]['map']
                    bb_i_mask, bb_i_map, i_tl_rc = get_bounding_box_object(i_mask,i_map)
                    i_vals = bb_i_map[bb_i_mask==1]; i_colors = list(np.unique(i_vals))


                    if iobj in ext_solved: print("SKIPPING iobj"); continue

                    
                    



                    qualifying_matches = []


                    
                    

                    matches = []

                    for oobj in dict_of_oobjs:
                        if esc(): break
                        if esc1(): break
                        o_mask, o_map = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map']

                        H_i, W_i = i_grid.shape; i_offsets = [ (0, 0), (0, -W_i + 0),  (-H_i + 0, 0),(-H_i + 0, -W_i + 0)]
                        H_o, W_o = o_grid.shape; o_offsets = [ (0, 0), (0, -W_o + 0), (-H_o + 0, 0), (-H_o + 0, -W_o + 0) ]
                        c=0 
                        
                        curr_anchors = []
                        curr_iswholeobjs = []
                        for (dy_i, dx_i), (dy_o, dx_o) in zip(i_offsets, o_offsets): 
                            curr_origin = ['TL','TR','BL','BR'][c]
                            c+=1
                            i_pos = (i_tl_rc[0] + dy_i, i_tl_rc[1] + dx_i) 

                            
                            irows, icols = np.where(i_mask == 1)
                            orows, ocols = [_+i_pos[0]-i_tl_rc[0] for _ in irows], [_+i_pos[1]-i_tl_rc[1] for _ in icols]
                            
                            proj_mask = np.zeros_like(o_grid); flagp = True
                            for m in range(len(orows)):
                                try: proj_mask[orows[m],ocols[m]] = 1 
                                except: flagp = False; break 
                            if flagp: 

                                
                                flag = True
                                for m in range(len(orows)):
                                    if o_mask[orows[m],ocols[m]] == 1 and o_map[orows[m],ocols[m]] == i_map[irows[m],icols[m]]: pass
                                    else: flag = False 
                                if flag:
                                    is_whole_obj = True if are_two_identical(o_mask, proj_mask) else False
                                    curr_anchors.append(curr_origin)
                                    curr_iswholeobjs.append(is_whole_obj)
                            
                        if len(curr_anchors) > 0:
                            matches.append({'type':'inplace_match_oobj','oobj':oobj,'oobj_list':[oobj],'list_anchors':curr_anchors,'list_is_whole_obj':curr_iswholeobjs})

                    

                    def calc_rank_score(trueflags, score_if_1, score_if_2):
                        ixs=[]
                        for m in range(len(matches)):
                            ff=True
                            for flag in trueflags:
                                if matches[m][flag]: pass
                                else: ff=False
                            if ff: ixs.append(m)
                        if len(ixs) == 1:
                            for m in ixs:
                                if 'score' not in matches[m]: matches[m]['score'] = score_if_1
                                elif score_if_1 > matches[m]['score']: matches[m]['score'] = score_if_1
                        elif len(ixs) == 2: 
                            for m in ixs:
                                if 'score' not in matches[m]: matches[m]['score'] = score_if_2
                                elif score_if_2 > matches[m]['score']: matches[m]['score'] = score_if_2
                        
                        else: 
                            for m in ixs:
                                if 'score' not in matches[m]: matches[m]['score'] = 0.2
                                elif 0.2 > matches[m]['score']: matches[m]['score'] = 0.2


                    
                    d_matches = []; temp=[]
                    for match in matches:
                        oobj = match['oobj']
                        o_mask, o_map, o_masko = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map'], dict_of_oobjs[oobj]['masko']
                        if is_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp): 
                            ix = ix_of_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp)
                            d_matches[ix]['oobj_list'].append(oobj)
                        else: temp.append([np.where(o_mask, o_map, 100), o_masko]); d_matches.append(match)
                    matches = d_matches 

                    
                    calc_rank_score([], score_if_1 = 1.0, score_if_2 = 0.6) 
                    for mat in matches:
                        
                        oobj = mat['oobj']
                        i_masko = dict_of_iobjs[iobj]['masko']
                        o_masko = dict_of_oobjs[oobj]['masko']
                        o_mask, o_map = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map']
                        if gridn==0 and toplot1:
                            sc = mat['score'] if 'score' in mat else None
                            print('Connection - 1Fixed', sc,';',mat['list_is_whole_obj'],':',mat['oobj_list'])
                            #visualise.plot_two_grids(np.where(i_mask, i_map, 100), np.where(o_mask, o_map, 100),i_masko, o_masko)
                        
                    
                    qualifying_matches.extend([_ for _ in matches if 'score' in _])




                    


                    matches = []

                    for oobj in dict_of_oobjs:
                        if esc(): break
                        if esc1(): break
                        o_mask, o_map = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map']
                        bb_o_mask, bb_o_map, o_tl_rc = get_bounding_box_object(o_mask,o_map)
                        o_vals = bb_o_map[bb_o_mask==1]; o_colors = list(np.unique(o_vals))

                        if are_two_identical(i_colors, o_colors): 

                            is_whole_obj = False 
                            if are_identicalQ([bb_i_mask,bb_o_mask]): 
                                is_whole_obj = True if (i_vals == o_vals).all() else False 

                            anchor_rel_movts = []; anchor_static = []
                            H_i, W_i = i_grid.shape; i_offsets = [ (0, 0), (0, -W_i + 1),  (-H_i + 1, 0),(-H_i + 1, -W_i + 1)]
                            H_o, W_o = o_grid.shape; o_offsets = [ (0, 0), (0, -W_o + 1), (-H_o + 1, 0), (-H_o + 1, -W_o + 1) ]
                            c=0 
                            for (dy_i, dx_i), (dy_o, dx_o) in zip(i_offsets, o_offsets): 
                                curr_origin = ['TL','TR','BL','BR'][c]
                                c+=1
                                i_pos = (i_tl_rc[0] + dy_i, i_tl_rc[1] + dx_i)
                                o_pos = (o_tl_rc[0] + dy_o, o_tl_rc[1] + dx_o)
                                rel_movt = (o_pos[0]-i_pos[0],o_pos[1]-i_pos[1]) 
                                if are_identicalQ([i_pos,o_pos]): anchor_static.append(1); anchor_rel_movts.append('static')
                                else: anchor_static.append(0); anchor_rel_movts.append(rel_movt)
                        
                            matches.append({'type':'color_match_oobj','oobj':oobj,'oobj_list':[oobj],'anchor_rel_movts':anchor_rel_movts,'is_whole_obj':is_whole_obj,
                                            'f1':'static' in anchor_rel_movts})


                    
                    
                    


                    d_matches = []; temp=[]
                    for match in matches:
                        oobj = match['oobj']
                        o_mask, o_map, o_masko = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map'], dict_of_oobjs[oobj]['masko']
                        if is_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp): 
                            ix = ix_of_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp)
                            d_matches[ix]['oobj_list'].append(oobj)
                        else: temp.append([np.where(o_mask, o_map, 100), o_masko]); d_matches.append(match)
                    matches = d_matches 


                    
                    calc_rank_score([], score_if_1 = 1.0, score_if_2 = 0) 
                    calc_rank_score(['f1'], score_if_1 = 0.9, score_if_2 = 0) 
                    for mat in matches:
                        
                        oobj = mat['oobj']
                        i_masko = dict_of_iobjs[iobj]['masko']
                        o_masko = dict_of_oobjs[oobj]['masko']
                        o_mask, o_map = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map']
                        if gridn==0 and toplot1 and 'score' in mat: 
                            sc = mat['score'] if 'score' in mat else None
                            print('Connection - 1Samecolor', sc,';',mat['is_whole_obj'],':',mat['oobj_list'])
                            #visualise.plot_two_grids(np.where(i_mask, i_map, 100), np.where(o_mask, o_map, 100),i_masko, o_masko)
                        

                    qualifying_matches.extend([_ for _ in matches if 'score' in _])



                    
                    


                    matches = []

                    def check_packing_recoloring(bb_o_mask, bb_i_map, bb_o_map): 
                        
                        
                        
                        
            
                        packing_recoloring, packing_objects = False, [] 
                        if len(get_colors_of_obj(bb_i_mask,bb_i_map)) == 1: 
                            oreg = bb_o_map.copy(); rows,cols = np.where(bb_o_mask==0); oreg[rows,cols] = -99 
                            objmasks = get_contiguous_regions(oreg,background_color=-99,diagonal_connections_allowedQ=False,colourblind_spatial_contiguity_mode=False)
                            
                            
                            blind_minimum_objs = []
                            for objn in range(1,np.amax(objmasks)+1):
                                objnmask = (objmasks==objn).astype(int)
                                
                                
                                bb_tiny_mask, bb_tiny_map, tiny_tl_rc = get_bounding_box_object(objnmask, bb_o_map)
                                
                                blind_minimum_objs.append([np.sum(bb_tiny_mask), get_colors_of_obj(bb_tiny_mask, bb_tiny_map), 
                                                        bb_tiny_mask, bb_tiny_map, True])
                            sorted_minimum_objs = sorted(blind_minimum_objs, key=lambda d: d[0])
                            
                            
                            

                            
                            def rotated_bb(mask,map):
                                
                                
                                
                                bb_mask = np.rot90(mask); bb_map = np.rot90(map) 
                                return bb_mask, bb_map

                            for n in range(len(sorted_minimum_objs)):
                                if sorted_minimum_objs[n][-1] == False: continue
                                size, main_mask, main_map = sorted_minimum_objs[n][0], sorted_minimum_objs[n][2], sorted_minimum_objs[n][3]
                                rot_mask, rot_map = rotated_bb(main_mask, main_map)
                                for m in range(n+1,len(sorted_minimum_objs)):
                                    tsize, tmask, tmap = sorted_minimum_objs[m][0], sorted_minimum_objs[m][2], sorted_minimum_objs[m][3]
                                    if are_two_identical(tmask,main_mask) and are_two_identical(tmap,main_map):
                                        sorted_minimum_objs[m][-1] = False
                                    if are_two_identical(tmask,rot_mask) and are_two_identical(tmap,rot_map):
                                        sorted_minimum_objs[m][-1] = False
                            
                            
                            reference_objs_nomodulations = []; reference_colors = []
                            for n in range(len(sorted_minimum_objs)):
                                if sorted_minimum_objs[n][-1] == False: continue
                                objcolors = sorted_minimum_objs[n][1]
                                if objcolors not in reference_colors:
                                    reference_objs_nomodulations.append(sorted_minimum_objs[n])
                                    reference_colors.append(objcolors)


                            
                            
                            
                            
                            
                            
                            
                            
                            

                            
                            
                            reference_objs = []
                            for n in range(len(reference_objs_nomodulations)):
                                curr = reference_objs_nomodulations[n]
                                reference_objs.append(curr)
                                rot = copy.deepcopy(curr)
                                rotmask, rotmap = rotated_bb(rot[2],rot[3])
                                if are_two_identical(rot[2],rotmask) and are_two_identical(rot[3],rotmap): pass
                                else:
                                    rot[2] = rotmask; rot[3] = rotmap
                                    reference_objs.append(rot)


                            all_matches = []
                            blockable_bb_o_mask = copy.deepcopy(bb_o_mask)
                            assigns = np.zeros_like(bb_o_mask)
                            for itern in range(10):
                                unique_matches = []
                                for r in range(blockable_bb_o_mask.shape[0]):
                                    for c in range(blockable_bb_o_mask.shape[1]):
                                        if blockable_bb_o_mask[r,c] == 0: continue

                                        o_region = np.zeros_like(blockable_bb_o_mask) 
                                        o_region[r,c] = 1

                                        matches = []
                                        for n in range(len(reference_objs)):
                                            curr_mask = reference_objs[n][2]
                                            thiscolor = int(reference_objs[n][1][0])

                                            
                                            rows, cols = np.where(curr_mask == 1)
                                            ris, rie, cis, cie = rows.min(), rows.max(), cols.min(), cols.max()
                                            rows_, cols_ = np.where(o_region == 1)
                                            ros, roe, cos, coe = rows_.min(), rows_.max(), cols_.min(), cols_.max()
                                            opts = []
                                            for dr in range(ros - rie, roe - ris + 1):
                                                for dc in range(cos - cie, coe - cis + 1):
                                                    if (rie + dr >= ros and ris + dr <= roe and
                                                        cie + dc >= cos and cis + dc <= coe):
                                                        if ris + dr < 0 or rie + dr > o_region.shape[0]-1 or cis + dc < 0 or cie + dc > o_region.shape[1]-1: continue
                                                        
                                                        newrows, newcols = rows + dr, cols + dc
                                                        flag = True
                                                        for m in range(len(newrows)):
                                                            if bb_o_map[newrows[m],newcols[m]] != thiscolor: flag = False
                                                        if flag: opts.append((dr, dc)); matches.append((n,dr,dc))
                                        if len(matches)==0: print("WARNING")
                                        if len(matches) == 1:
                                            unique_matches.append([r,c,*matches[0]]) 
                                        
                                all_matches.extend(unique_matches)

                                for r,c,objn,dr,dc in unique_matches:
                                    rows, cols = np.where(reference_objs[objn][2] == 1)
                                    newrows, newcols = rows + dr, cols + dc
                                    for m in range(len(newrows)):
                                        blockable_bb_o_mask[newrows[m],newcols[m]] = 0

                                if np.sum(blockable_bb_o_mask) == 0: break


                            ms = []
                            for _ in all_matches:
                                if _[2:] in ms: pass
                                else: ms.append(_[2:])
                            
                            packing_recoloring = True
                            packing_objects = reference_objs_nomodulations


                            

                            

                        return packing_recoloring, packing_objects

                    
                    for oobj in dict_of_oobjs:
                        if esc(): break
                        if esc1(): break
                        o_mask, o_map = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map']
                        bb_o_mask, bb_o_map, o_tl_rc = get_bounding_box_object(o_mask,o_map)
                        o_vals = bb_o_map[bb_o_mask==1]; o_colors = list(np.unique(o_vals))

                        if are_identicalQ([bb_i_mask,bb_o_mask]): 
                            object_maintained = True if (i_vals == o_vals).all() else False 
                            perfect_colorchange, grey_recoloring, color_changes = check_perfect_colorchange(i_colors,o_colors, i_vals,o_vals) 
                        
                            if object_maintained or perfect_colorchange: 

                                anchor_rel_movts = []; anchor_static = []
                                H_i, W_i = i_grid.shape; i_offsets = [ (0, 0), (0, -W_i + 1),  (-H_i + 1, 0),(-H_i + 1, -W_i + 1)]
                                H_o, W_o = o_grid.shape; o_offsets = [ (0, 0), (0, -W_o + 1), (-H_o + 1, 0), (-H_o + 1, -W_o + 1) ]
                                c=0 
                                for (dy_i, dx_i), (dy_o, dx_o) in zip(i_offsets, o_offsets): 
                                    curr_origin = ['TL','TR','BL','BR'][c]
                                    c+=1
                                    i_pos = (i_tl_rc[0] + dy_i, i_tl_rc[1] + dx_i)
                                    o_pos = (o_tl_rc[0] + dy_o, o_tl_rc[1] + dx_o)
                                    rel_movt = (int(o_pos[0]-i_pos[0]),int(o_pos[1]-i_pos[1])) 
                                    if are_identicalQ([i_pos,o_pos]): anchor_static.append(1); anchor_rel_movts.append('static')
                                    else: anchor_static.append(0); anchor_rel_movts.append(rel_movt)
                                
                                is_whole_obj = True 

                                matches.append({'type':'shape_match_oobj','oobj':oobj,'oobj_list':[oobj],'anchor_rel_movts':anchor_rel_movts,'is_whole_obj':is_whole_obj,
                                                    'object_maintained':object_maintained,'perfect_colorchange':perfect_colorchange,'grey_recoloring':grey_recoloring,'color_changes':color_changes,
                                                    'f1':'static' in anchor_rel_movts, 'f2':object_maintained,'f3':perfect_colorchange})


                    
                    
                    

                    d_matches = []; temp=[]
                    for match in matches:
                        oobj = match['oobj']
                        o_mask, o_map, o_masko = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map'], dict_of_oobjs[oobj]['masko']
                        if is_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp): 
                            ix = ix_of_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp)
                            d_matches[ix]['oobj_list'].append(oobj)
                        else: temp.append([np.where(o_mask, o_map, 100), o_masko]); d_matches.append(match)
                    matches = d_matches 

                    
                    calc_rank_score([], score_if_1 = 1.0, score_if_2 = 0) 
                    calc_rank_score(['f1'], score_if_1 = 0.9, score_if_2 = 0) 
                    calc_rank_score(['f2'], score_if_1 = 0.8, score_if_2 = 0) 
                    calc_rank_score(['f3'], score_if_1 = 0.8, score_if_2 = 0) 
                    calc_rank_score(['f1','f2'], score_if_1 = 0.5, score_if_2 = 0) 
                    calc_rank_score(['f1','f3'], score_if_1 = 0.4, score_if_2 = 0) 
                    for mat in matches:
                        
                        oobj = mat['oobj']
                        i_masko = dict_of_iobjs[iobj]['masko']
                        o_masko = dict_of_oobjs[oobj]['masko']
                        o_mask, o_map = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map']
                        if gridn==0 and toplot1 and 'score' in mat: 
                            sc = mat['score'] if 'score' in mat else None
                            print('Connection - 1Sameshape', sc,';',mat['is_whole_obj'],':',mat['oobj_list'])
                            #visualise.plot_two_grids(np.where(i_mask, i_map, 100), np.where(o_mask, o_map, 100),i_masko, o_masko)
                        

                    qualifying_matches.extend([_ for _ in matches if 'score' in _])



                    

                    matches = []

                    if len(dict_of_oobjs) < 3:

                        for oobj in dict_of_oobjs:
                            o_mask, o_map = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map']
                            bb_o_mask, bb_o_map, o_tl_rc = get_bounding_box_object(o_mask,o_map)
                            o_vals = bb_o_map[bb_o_mask==1]; o_colors = list(np.unique(o_vals))

                            
                            

                

                    

                


                    

                    if esc(): break
                    if esc1(): break
                    matches = []

                    H_i, W_i = i_grid.shape; i_offsets = [ (0, 0), (0, -W_i + 0),  (-H_i + 0, 0),(-H_i + 0, -W_i + 0)]
                    H_o, W_o = o_grid.shape; o_offsets = [ (0, 0), (0, -W_o + 0), (-H_o + 0, 0), (-H_o + 0, -W_o + 0) ]
                    c=0 
                    
                    for (dy_i, dx_i), (dy_o, dx_o) in zip(i_offsets, o_offsets): 
                        curr_origin = ['TL','TR','BL','BR'][c]
                        c+=1
                        i_pos = (i_tl_rc[0] + dy_i, i_tl_rc[1] + dx_i) 

                        
                        irows, icols = np.where(i_mask == 1)
                        orows, ocols = [_+i_pos[0]-i_tl_rc[0] for _ in irows], [_+i_pos[1]-i_tl_rc[1] for _ in icols]
                        
                        proj_mask = np.zeros_like(o_grid); flagp = True
                        for m in range(len(orows)):
                            try: proj_mask[orows[m],ocols[m]] = 1 
                            except: flagp = False; break 
                        if flagp: 

                            
                            flag = True
                            for m in range(len(orows)):
                                if o_grid[orows[m],ocols[m]] == i_map[irows[m],icols[m]]: pass
                                else: flag = False 
                            if flag:
                                matches.append({'type':'inplace_match_oreg','anchor':curr_origin,'slotmask':proj_mask})


                    def calc_rank_score_distinct(trueflags, score_if_1, score_if_2):
                        ixs = []
                        for m in range(len(matches)):
                            ff=True
                            for flag in trueflags:
                                if matches[m][flag]: pass
                                else: ff=False
                            if ff: ixs.append(m)

                        
                        all_slots = np.zeros_like(o_grid)
                        for m1 in ixs: 
                            all_slots += matches[m1]['slotmask']
                        for m1 in ixs:
                            if np.max(all_slots[matches[m1]['slotmask']==1]) == 1:
                                matches[m1]['is_distinct'] = True
                            else: matches[m1]['is_distinct'] = False

                        ixs2 = []
                        for m1 in ixs:
                            if matches[m1]['is_distinct']:
                                ixs2.append(m1)

                        if len(ixs2) == 1:
                            for m in ixs2:
                                if 'score' not in matches[m]: matches[m]['score'] = score_if_1
                                elif score_if_1 > matches[m]['score']: matches[m]['score'] = score_if_1
                        elif len(ixs2) == 2: 
                            for m in ixs2:
                                if 'score' not in matches[m]: matches[m]['score'] = score_if_2
                                elif score_if_2 > matches[m]['score']: matches[m]['score'] = score_if_2
                        else: pass 

                    d_matches = []; temp=[]
                    for match in matches:
                        o_mask, o_map, o_masko = match['slotmask'], o_grids[gridn], np.zeros_like(match['slotmask'])
                        if is_x_in_y(x = [np.where(o_mask, o_map, 100), o_masko], y = temp): continue 
                        else: temp.append([np.where(o_mask, o_map, 100), o_masko]); d_matches.append(match)
                    matches = d_matches 
                    



                    
                    calc_rank_score_distinct([], score_if_1 = 1.0, score_if_2 = 0) 
                    for mat in matches:
                        
                        i_masko = dict_of_iobjs[iobj]['masko']
                        o_mask, o_map, o_masko = mat['slotmask'], o_grids[gridn], np.zeros_like(mat['slotmask']) 
                        if gridn==0 and toplot1 and 'score' in mat:
                            sc = mat['score'] if 'score' in mat else None
                            print('Connection - 2Fixed', sc,':')
                            
                            #visualise.plot_two_grids(np.where(i_mask, i_map, 100), np.where(o_mask, o_map, 100),i_masko, o_masko)
                        

                    
                    
                    for match in matches:
                        if esc(): break
                        if esc1(): break
                        if 'score' not in match: continue 
                        o_mask, o_map, o_masko = match['slotmask'], o_grids[gridn], np.zeros_like(match['slotmask'])
                        direct_oobj_matches = []
                        for oobj in dict_of_oobjs:
                            oobj_mask, oobj_map, oobj_masko = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map'],  dict_of_oobjs[oobj]['masko']
                            if are_two_identical([np.where(o_mask, o_map, 100), o_masko], [np.where(oobj_mask, oobj_map, 100), oobj_masko]):
                                direct_oobj_matches.append({'oobj':oobj,'is_whole_region':True})
                            
                            elif are_two_identical([np.where(o_mask, o_map, 100), np.where(o_mask, o_masko, 100)], [np.where(o_mask, oobj_map, 100), np.where(o_mask, oobj_masko, 100)]):
                                    direct_oobj_matches.append({'oobj':oobj,'is_whole_region':False})
                        match['to_existing_oobjs'] = direct_oobj_matches 
                        


                    qualifying_matches.extend([_ for _ in matches if 'score' in _])



                    

                    
                    
                    
                    
                    
                    if esc(): break
                    if esc1(): break
                    QUIT1 = 10
                    endtime1 = time.perf_counter_ns() 
                    if ((endtime1-starttime1)/1000000000) < QUIT1:

                        matches = []

                        i_outline_mask = get_outline_border_mask(i_mask)
                        i_outline_vals = i_grid[i_outline_mask.astype(bool)]; i_outline_colors = list(np.unique(i_outline_vals))

                        bbirows, bbicols = np.where(bb_i_mask==1)
                        for r in range(-bb_i_mask.shape[0] + 1, o_grid.shape[0]): 
                            endtime1 = time.perf_counter_ns() 
                            if ((endtime1-starttime1)/1000000000) > QUIT1: break
                            for c in range(-bb_i_mask.shape[1] + 1, o_grid.shape[1]):
                                if esc(): break
                                if esc1(): break
                                o_tl_rc = (r,c) 
                                rows = bbirows + r
                                cols = bbicols + c
                                valid = (rows >= 0) & (rows < o_grid.shape[0]) & (cols >= 0) & (cols < o_grid.shape[1])
                                validmask = np.zeros_like(o_grid); validmap = -1*np.ones_like(o_grid)
                                validmask[rows[valid], cols[valid]] = 1
                                validmap[rows[valid], cols[valid]] = i_vals[valid] 
                                if np.sum(validmask) == 0: continue
                                
                                bb_iseg_mask, bb_iseg_map, iseg_tl_rc = get_bounding_box_object(validmask, validmap)
                                bb_oseg_mask, bb_oseg_map, oseg_tl_rc = get_bounding_box_object(validmask, o_grid)
                                iseg_vals = bb_iseg_map[bb_iseg_mask==1]
                                iseg_colors = list(np.unique(iseg_vals))
                                fully_visible_shape = True if len(iseg_vals)==np.sum(bb_i_mask) else False
                                oseg_vals = bb_oseg_map[bb_oseg_mask==1]
                                oseg_colors = list(np.unique(oseg_vals))
                                
                                visible_region_shape_maintained = True if (iseg_vals == oseg_vals).all() else False 
                                object_maintained = True if (visible_region_shape_maintained and fully_visible_shape) else False 
                                perfect_colorchange, grey_recoloring, color_changes = check_perfect_colorchange(iseg_colors,oseg_colors, iseg_vals,oseg_vals) 
                                
                                o_outline_mask = get_outline_border_mask(validmask) 
                                o_outline_vals = o_grid[o_outline_mask.astype(bool)]; o_outline_colors = list(np.unique(o_outline_vals))
                                
                                
                                
                                nonleaked_oborder = True if none_of_x_in_y(x=oseg_colors, y=o_outline_colors) else False 
                                
                                contains_original_bordercolor = True if at_least_some_of_x_in_y(x=i_outline_colors,y=oseg_colors) else False
                                
                                

                                
                                anchor_rel_movts = []; anchor_static = []
                                H_i, W_i = i_grid.shape; i_offsets = [ (0, 0), (0, -W_i + 1),  (-H_i + 1, 0),(-H_i + 1, -W_i + 1)]
                                H_o, W_o = o_grid.shape; o_offsets = [ (0, 0), (0, -W_o + 1), (-H_o + 1, 0), (-H_o + 1, -W_o + 1) ]
                                c=0 
                                for (dy_i, dx_i), (dy_o, dx_o) in zip(i_offsets, o_offsets): 
                                    curr_origin = ['TL','TR','BL','BR'][c]
                                    c+=1
                                    i_pos = (i_tl_rc[0] + dy_i, i_tl_rc[1] + dx_i)
                                    o_pos = (o_tl_rc[0] + dy_o, o_tl_rc[1] + dx_o)
                                    rel_movt = (o_pos[0]-i_pos[0],o_pos[1]-i_pos[1]) 
                                    if are_identicalQ([i_pos,o_pos]): anchor_static.append(1); anchor_rel_movts.append('static')
                                    else: anchor_static.append(0); anchor_rel_movts.append(rel_movt)

                                
                                
                                

                                is_whole_obj = nonleaked_oborder 

                                f0 = False; f1 = False; f2 = False; f3 = False; f4 = False; f5 = False
                                
                                if 1 in anchor_static and not nonleaked_oborder and not contains_original_bordercolor: f0=True

                                if nonleaked_oborder and (visible_region_shape_maintained or perfect_colorchange): 
                                    
                                    if 1 in anchor_static: f1 = True 
                                    
                                    if object_maintained: f2 = True
                                    
                                    if visible_region_shape_maintained: f3 = True
                                    
                                    
                                    if fully_visible_shape and perfect_colorchange: f4 = True
                                    
                                    if perfect_colorchange: f5 = True
                                    

                                matches.append({'type':'shape_match_oreg','anchor_rel_movts':anchor_rel_movts,'slotmask':validmask, 'nonleaked_border':nonleaked_oborder,
                                                        'object_maintained':object_maintained,'perfect_colorchange':perfect_colorchange,'grey_recoloring':grey_recoloring,'color_changes':color_changes,
                                                        'f0':f0, 'f1':f1, 'f2':f2, 'f3':f3, 'f4':f4, 'f5':f5})

                        def calc_rank_score_distinct(trueflags, score_if_1, score_if_2):
                            ixs = []
                            for m in range(len(matches)):
                                ff=True
                                for flag in trueflags:
                                    if matches[m][flag]: pass
                                    else: ff=False
                                if ff: ixs.append(m)

                            
                            all_slots = np.zeros_like(o_grid)
                            for m1 in ixs: 
                                all_slots += matches[m1]['slotmask']
                            for m1 in ixs:
                                if np.max(all_slots[matches[m1]['slotmask']==1]) == 1:
                                    matches[m1]['is_distinct'] = True
                                else: matches[m1]['is_distinct'] = False

                            ixs2 = []
                            for m1 in ixs:
                                if matches[m1]['is_distinct']:
                                    ixs2.append(m1)

                            if len(ixs2) == 1:
                                for m in ixs2:
                                    if 'score' not in matches[m]: matches[m]['score'] = score_if_1
                                    elif score_if_1 > matches[m]['score']: matches[m]['score'] = score_if_1
                            elif len(ixs2) == 2: 
                                for m in ixs2:
                                    if 'score' not in matches[m]: matches[m]['score'] = score_if_2
                                    elif score_if_2 > matches[m]['score']: matches[m]['score'] = score_if_2
                            else: pass 

                        
                        
                        


                        
                        calc_rank_score_distinct([], score_if_1 = 1.0, score_if_2 = 0) 
                        calc_rank_score_distinct(['f1'], score_if_1 = 0.9, score_if_2 = 0) 
                        calc_rank_score_distinct(['f2'], score_if_1 = 0.8, score_if_2 = 0) 
                        calc_rank_score_distinct(['f3'], score_if_1 = 0.7, score_if_2 = 0) 
                        calc_rank_score_distinct(['f4'], score_if_1 = 0.6, score_if_2 = 0) 
                        calc_rank_score_distinct(['f5'], score_if_1 = 0.5, score_if_2 = 0) 
                        calc_rank_score_distinct(['f0'], score_if_1 = 0.4, score_if_2 = 0) 
                        for mat in matches:
                            
                            i_masko = dict_of_iobjs[iobj]['masko']
                            o_mask, o_map, o_masko = mat['slotmask'], o_grids[gridn], np.zeros_like(mat['slotmask']) 
                            if gridn==0 and toplot1 and 'score' in mat: 
                                sc = mat['score'] if 'score' in mat else None
                                print('Connection - 2Sameshape', sc,':')
                                
                                #visualise.plot_two_grids(np.where(i_mask, i_map, 100), np.where(o_mask, o_map, 100),i_masko, o_masko)
                            
                    

                        
                        for match in matches:
                            if esc(): break
                            if esc1(): break
                            if 'score' not in match: continue 
                            o_mask, o_map, o_masko = match['slotmask'], o_grids[gridn], np.zeros_like(match['slotmask'])
                            direct_oobj_matches = []
                            for oobj in dict_of_oobjs:
                                oobj_mask, oobj_map, oobj_masko = dict_of_oobjs[oobj]['mask'], dict_of_oobjs[oobj]['map'],  dict_of_oobjs[oobj]['masko']
                                if are_two_identical([np.where(o_mask, o_map, 100), o_masko], [np.where(oobj_mask, oobj_map, 100), oobj_masko]):
                                    direct_oobj_matches.append({'oobj':oobj,'is_whole_region':True})
                                
                                elif are_two_identical([np.where(o_mask, o_map, 100), np.where(o_mask, o_masko, 100)], [np.where(o_mask, oobj_map, 100), np.where(o_mask, oobj_masko, 100)]):
                                        direct_oobj_matches.append({'oobj':oobj,'is_whole_region':False})
                            match['to_existing_oobjs'] = direct_oobj_matches 
                            

                        qualifying_matches.extend([_ for _ in matches if 'score' in _])


                    

                    

                    



                    iobj_transform_res = [] 
                    matches_sorted = sorted(qualifying_matches, key=lambda x: (-x['score']))
                    for match in matches_sorted:
                        if esc(): break
                        if esc1(): break
                        if match['score'] > mainloop_score_thresh:


                            if match['type'] == 'inplace_match_oobj': 

                                list_anchors = match['list_anchors']
                                if len(list_anchors)!=0 and 'TL' not in list_anchors: print("Unsupported")
                                list_is_whole_obj = match['list_is_whole_obj']
                                oobj_list = match['oobj_list']
                                
                                
                                

                                
                                oobj = oobj_list[0] 
                                o_mask, o_map, o_masko = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['masko']
                                i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']                    
                                o_region = o_mask

                                for c in range(len(list_anchors)):
                                    curr_anchor = list_anchors[c]
                                    is_whole_obj = list_is_whole_obj[c]

                                    if curr_anchor == 'TL':
                                        r,c = 0,0 
                                    else: continue

                                    if is_whole_obj:
                                        main_chains = []
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':masking,'params':{}}])

                                        if gridn ==0 and toplot1: print('1aFixed:')
                                        res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oobj')
                                        for re in res: iobj_transform_res.append(re) 

                                    else:
                                        main_chains = []
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                        main_chains.append([{'fn':movt,'params':{}},{'fn':recolor,'params':{}}])
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':masking,'params':{}}])
                                        
                                        
                                        main_chains.append([{'fn':extension,'params':{}}])
                                        main_chains.append([{'fn':fill,'params':{}}])
                                        

                                        
                                        

                                        if gridn ==0 and toplot1: print('1bFixed:')
                                        res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oobj')
                                        for re in res: iobj_transform_res.append(re) 

                            if match['type'] == 'color_match_oobj': 
                                anchor_rel_movts = match['anchor_rel_movts']
                                is_whole_obj = match['is_whole_obj']
                                oobj_list = match['oobj_list']
                                
                                
                                

                                
                                oobj = oobj_list[0]
                                o_mask, o_map, o_masko = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['masko']
                                i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']                    
                                o_region = o_mask

                                topleft_movt = anchor_rel_movts[0] 
                                if topleft_movt == 'static': r,c = 0,0
                                else: r,c = int(topleft_movt[0]),int(topleft_movt[1])

                                if is_whole_obj:
                                    
                                    main_chains = []
                                    main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                    
                                    main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':masking,'params':{}}])
                                    main_chains.append([{'fn':extension,'params':{}}])
                                    main_chains.append([{'fn':copying,'params':{}}])

                                    if gridn ==0 and toplot1: print('1aSamecolor:')
                                    res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oobj')
                                    for re in res: iobj_transform_res.append(re) 
                                    
                                    

                                else:

                                    main_chains = []
                                    main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                
                                
                                    
                                    main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':masking,'params':{}}])
                                    main_chains.append([{'fn':extension,'params':{}}])
                                    main_chains.append([{'fn':copying,'params':{}}])
                            
                                    if gridn ==0 and toplot1: print('1bSamecolor:')
                                    res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oobj')
                                    for re in res: iobj_transform_res.append(re) 

                            if match['type'] == 'shape_match_oobj': 
                                anchor_rel_movts = match['anchor_rel_movts']
                                is_whole_obj = match['is_whole_obj']
                                oobj_list = match['oobj_list']
                                object_maintained, perfect_colorchange, grey_recoloring, color_changes = match['object_maintained'], match['perfect_colorchange'], match['grey_recoloring'], match['color_changes']
                                

                                
                                oobj = oobj_list[0]
                                o_mask, o_map, o_masko = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['masko']
                                i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']                    
                                o_region = o_mask

                                topleft_movt = anchor_rel_movts[0] 
                                if topleft_movt == 'static': r,c = 0,0
                                else: r,c = int(topleft_movt[0]),int(topleft_movt[1])

                                main_chains = []
                                main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                if not object_maintained: main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':recolor,'params':{'color_changes':color_changes}}])
                                main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':masking,'params':{}}])
                                main_chains.append([{'fn':extension,'params':{}}])
                                main_chains.append([{'fn':copying,'params':{}}])
                                if not object_maintained: main_chains.append([{'fn':recolor,'params':{'color_changes':color_changes}},{'fn':extension,'params':{}}])

                                if gridn ==0 and toplot1: print('1aSameshape:')
                                res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oobj')
                                for re in res: iobj_transform_res.append(re) 

                                


                            if match['type'] == 'inplace_match_oreg': 
                                slotmask = match['slotmask']
                                anchor = match['anchor']
                                oobj_list = match['to_existing_oobjs'] 
                                
                                

                                if anchor == 'TL':
                                    r,c = 0,0 
                                else: continue

                                if len(oobj_list)==0 or not oobj_list[0]['is_whole_region']:
                                    o_region = slotmask
                                    o_masko = np.zeros_like(o_region)
                                    i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']
                                    o_mask, o_map, o_masko = o_region, o_grid, o_masko 

                                    main_chains = []
                                    main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                    main_chains.append([{'fn':extension,'params':{}}])
                                    
            
                                    
                                    
                                    if gridn ==0 and toplot1: print('2cFixed:')
                                    res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oreg')
                                    for re in res: iobj_transform_res.append(re) 

                                else: 
                                    oobj, is_whole_obj = oobj_list[0]['oobj'], oobj_list[0]['is_whole_region'] 
                                    
                                    
                                    o_mask, o_map, o_masko = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['masko']
            
            
            
                                    i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']                    
                                    o_region = o_mask
                                    if is_whole_obj:
                                        main_chains = []
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':masking,'params':{}}])

                                        if gridn ==0 and toplot1: print('2aFixed:')
                                        res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oobj')
                                        for re in res: iobj_transform_res.append(re) 

                                    else:
                                        main_chains = []
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                        main_chains.append([{'fn':movt,'params':{}},{'fn':recolor,'params':{}}])
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':masking,'params':{}}])
                                        main_chains.append([{'fn':extension,'params':{}}])
                                        main_chains.append([{'fn':extension,'params':{}},{'fn':copying,'params':{}}])

                                        if gridn ==0 and toplot1: print('2bFixed:')
                                        res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oobj')
                                        for re in res: iobj_transform_res.append(re) 

                            if match['type'] == 'shape_match_oreg': 
                                slotmask = match['slotmask']
                                anchor_rel_movts = match['anchor_rel_movts']
                                nonleaked_border = match['nonleaked_border'] 
                                oobj_list = match['to_existing_oobjs'] 
                                object_maintained, perfect_colorchange, grey_recoloring, color_changes = match['object_maintained'], match['perfect_colorchange'], match['grey_recoloring'], match['color_changes']

                                
                                

                                
                                
                                

                                

                                topleft_movt = anchor_rel_movts[0] 
                                if topleft_movt == 'static': r,c = 0,0
                                else: r,c = int(topleft_movt[0]),int(topleft_movt[1])

                                

                                if len(oobj_list)==0:
                                    o_region = slotmask
                                    o_masko = np.zeros_like(o_region)
                                    i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']
                                    o_mask, o_map, o_masko = o_region, o_grid, o_masko 

                                    main_chains = []
                                    main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                    if not object_maintained: main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':recolor,'params':{'color_changes':color_changes}}])
                                    
                                
                                    
                                    
                                    if gridn ==0 and toplot1: print('2cSameshape:')
                                    res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oreg')
                                    for re in res: iobj_transform_res.append(re) 

                                else: 
                                    oobj, is_whole_obj = oobj_list[0]['oobj'], oobj_list[0]['is_whole_region'] 
                                    
                                    
                                    o_mask, o_map, o_masko = global_parsings[gridn]['o'][oobj]['mask'], global_parsings[gridn]['o'][oobj]['map'], global_parsings[gridn]['o'][oobj]['masko']
                                    i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']                    
                                    o_region = o_mask
                                    if is_whole_obj:
                                        main_chains = []
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}}])
                                        if not object_maintained: main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':recolor,'params':{'color_changes':color_changes}}])
                                        main_chains.append([{'fn':movt,'params':{'move_rc':(r,c)}},{'fn':masking,'params':{}}])
                                        if not object_maintained: main_chains.append([{'fn':recolor,'params':{'color_changes':color_changes}},{'fn':extension,'params':{}}])

                                        if gridn ==0 and toplot1: print('2aSameshape:')
                                        res = detect_io_transforms(iobj,oobj,i_mask, i_map, o_grid, o_region, o_masko, gridn, main_chains_override = main_chains, oobj_or_oreg_mode = 'oobj')
                                        for re in res: iobj_transform_res.append(re) 
                                    else:
                                        print("Unhandled 2bSameshape")
                    

                    
                    
                    iobj_all_oobjs = [_['addressable']['oobj'] for _ in iobj_transform_res]
                    if None in iobj_all_oobjs: print("ERROR, disappear does not support None oobj yet.")
                    iobj_all_serials = [_['serial_transforms'] for _ in iobj_transform_res] 
                    
                    
                    
                    
                    
                    fullgrid_penalty = 0.4 if global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] == 'fullgrid_i' else 1 
                    background_penalty = 0.4 if global_parsings[gridn]['i'][iobj]['properties']['parsing_description'][0] == 'background' else 1 
                    dis_score = 0.8 * fullgrid_penalty * background_penalty      

                    serial_transforms = [{'type':'disappear'}]; serial_params = [{}]
                    i_colors = get_colors_of_obj(i_mask,i_map); i_shape = get_shape_of_obj(i_mask,i_map)
                    re = {'tr_score':dis_score,'gridn':gridn,'serial_transforms':serial_transforms,'serial_params':serial_params,'i_region':'n/a','(i_obj_colors)':[i_colors],'(i_obj_shapes)':[i_shape],'current_run':None,'curr_mask':None,'curr_map':None,'curr_maskv':None,'curr_o_masko':None, 
                            'addressable':{'iobj':iobj,'oobj':None}} 
                    
                    if gridn == 0 and toplot11: 
                        print("Transform - ",iobj,None,serial_transforms,serial_params)
                        def tempviz(iobj,oobj,gridn):
                            i_mask, i_map, i_masko = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['masko']  
                            #visualise.plot_two_grids(np.where(i_mask, i_map, 100), 100*np.ones_like(o_grid),i_masko, np.zeros_like(o_grid))                    
                        tempviz(iobj,None,gridn)
                    



                    for re in iobj_transform_res: 
                        for dict_ in re['serial_transforms']:
                            dict_['r'] = 1
                    transform_res.extend(iobj_transform_res)
        except: pass


        




        if len(transform_res) == 0: print("WARNING")
        


    

        
        for tr_re in transform_res:
            if esc(): break
            to_del = []
            for _ in range(len(tr_re['serial_transforms'])):
                if tr_re['serial_transforms'][_]['type'] == 'masking': to_del.append(1)
                else: to_del.append(0)
            tr_re['serial_transforms'] = [tr_re['serial_transforms'][_] for _ in range(len(tr_re['serial_transforms'])) if to_del[_]==0]
            tr_re['serial_params'] = [tr_re['serial_params'][_] for _ in range(len(tr_re['serial_params'])) if to_del[_]==0] 


        
        for tr_re in transform_res:
            if esc(): break
            tr_re['current_run'] = 'temp removed so unique transform res is easier; not always matches serial tr'
            if len(tr_re['serial_transforms'])==1: continue 
            else:
                new_tr = []; new_par = []
                for _ in range(len(tr_re['serial_transforms'])):
                    if tr_re['serial_transforms'][_]['type'] != 'static':
                        new_tr.append(tr_re['serial_transforms'][_])
                        new_par.append(tr_re['serial_params'][_])
                tr_re['serial_transforms'] = new_tr 
                tr_re['serial_params'] = new_par


        
        
        r1,r2 = label_unique_with_IDs([[_['serial_transforms'],_['serial_params'],_['addressable']] for _ in transform_res])
        new_transform_res = []
        for _ in range(np.max(r1)+1):
            if esc(): break
            ixs = [i for i in range(len(r1)) if r1[i]==_]
            scores = [transform_res[i]['tr_score'] for i in ixs]
            ix_of_maxscore = ixs[np.argmax(scores)]
            new_transform_res.append(transform_res[ix_of_maxscore])
        transform_res = new_transform_res 






        
        def enumerate_analogies(Analogies,obj_ids,scorepositions,mode='new'):

            
            

            n_objs = len(Analogies)
            n_keys = len(Analogies[0]) 

            qualifiers = [] 
            dets=[] 

            
            
            


            if mode == 'old':
        
                for relax_positions in chain.from_iterable(combinations(scorepositions, r) for r in range(len(scorepositions)+1-1)):
                    leaveout = len(relax_positions); qualify_ids = []
                    
                    
                    
        
                    for o in range(len(obj_ids)):
                        if sum([Analogies[o][k]==1 for k in range(n_keys) if k not in relax_positions]) == n_keys - leaveout:
                            qualify_ids.append(obj_ids[o]) 
                    if len(qualify_ids)>=3:
                        if sorted(qualify_ids) not in qualifiers: 
                            qualifiers.append(sorted(qualify_ids))
                            dets.append([len(qualify_ids),n_keys,leaveout])
                    
                    if len(qualify_ids) == len(obj_ids): quit_leaveout = leaveout 
                    
                    
                    
        
                for relax_positions in chain.from_iterable(combinations(scorepositions, r) for r in range(len(scorepositions)+1-1)):
                    leaveout = len(relax_positions); qualify_ids = []
        
                    for o in range(len(obj_ids)):
                        if sum([Analogies[o][k]==1 for k in range(n_keys) if k not in relax_positions]) == n_keys - leaveout:
                            qualify_ids.append(obj_ids[o]) 
                    if len(qualify_ids)==2:
                        if sorted(qualify_ids) not in qualifiers: 
                            qualifiers.append(sorted(qualify_ids))
                            dets.append([len(qualify_ids),n_keys,leaveout])
                    if len(qualify_ids) == len(obj_ids): quit_leaveout = leaveout



            if mode == 'new':
                quit_leaveout = 10000     
                Analogies = np.array(Analogies)
                seen = set()
                for r in range(min(len(scorepositions), quit_leaveout) + 1):
                    for relax_positions in combinations(scorepositions, r):
                        relax_set = set(relax_positions)
                        mask = np.ones(Analogies.shape[1], dtype=bool)
                        mask[list(relax_set)] = False
                        valid = (Analogies[:, mask] == 1).all(axis=1)
                        qualify_ids = [obj_ids[o] for o, ok in enumerate(valid) if ok]
                        if len(qualify_ids) >= 3:
                            qual_tuple = tuple(sorted(qualify_ids))
                            if qual_tuple not in seen:
                                seen.add(qual_tuple)
                                qualifiers.append(list(qual_tuple))
                                dets.append([len(qualify_ids), Analogies.shape[1], r])
                        if len(qualify_ids) == len(obj_ids):
                            quit_leaveout = r
                            break
                quit_leaveout = 10000 
                Analogies = np.array(Analogies)
                seen = set()
                for r in range(min(len(scorepositions), quit_leaveout) + 1):
                    for relax_positions in combinations(scorepositions, r):
                        relax_set = set(relax_positions)
                        mask = np.ones(Analogies.shape[1], dtype=bool)
                        mask[list(relax_set)] = False
                        valid = (Analogies[:, mask] == 1).all(axis=1)
                        qualify_ids = [obj_ids[o] for o, ok in enumerate(valid) if ok]
                        if len(qualify_ids) == 2:
                            qual_tuple = tuple(sorted(qualify_ids))
                            if qual_tuple not in seen:
                                seen.add(qual_tuple)
                                qualifiers.append(list(qual_tuple))
                                dets.append([len(qualify_ids), Analogies.shape[1], r])
                        if len(qualify_ids) == len(obj_ids):
                            quit_leaveout = r
                            break



            


            if len(obj_ids) >= 2 and sorted(obj_ids) not in qualifiers: 
                qualifiers.append(sorted(obj_ids)) 
                dets.append([len(obj_ids),n_keys,n_keys])

            return qualifiers, dets


        def topological_sortY(graph):

            in_degree = {node: 0 for node in graph}
            for u in graph:
                for v in graph[u]:
                    in_degree[v] += 1

            queue = [node for node in graph if in_degree[node] == 0]
            sorted_list = []

            while queue:
                node = queue.pop(0)
                sorted_list.append(node)
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            return sorted_list if len(sorted_list) == len(graph) else None

        def sort_relationsY(objects, relations):

            
            layers = [[obj] for obj in objects]
            group_map = {obj: idx for idx, obj in enumerate(objects)}

            
            for obj1, op, obj2 in relations:
                if op == "=":
                    layer1, layer2 = group_map[obj1], group_map[obj2]
                    if layer1 != layer2:
                        layers[layer1].extend(layers[layer2])
                        for obj in layers[layer2]:
                            group_map[obj] = layer1
                        layers[layer2] = []

            
            graph = {i: [] for i, layer in enumerate(layers) if layer}
            for obj1, op, obj2 in relations:
                if op in ("<", ">"):
                    layer1, layer2 = group_map[obj1], group_map[obj2]
                    if op == "<" and layer2 not in graph[layer1]:
                        graph[layer1].append(layer2)
                    elif op == ">" and layer1 not in graph[layer2]:
                        graph[layer2].append(layer1)

            
            sorted_layer_indices = topological_sortY(graph)
            if sorted_layer_indices is None:
                return None  

            
            return [obj for idx in sorted_layer_indices for obj in layers[idx] if layers[idx]]


        unique_analogies_ = []; analogy_scores_ = []; reference_serials_ = []; analogy_and_serials_ = []
        ordered_serial_transforms_ = []; ordered_serial_params_ = []
        debug_types_ = []


        viskeys_all_ = []
        for n in range(len(transform_res)):
            if esc(): break
            n_tr_score, n_gridn, n_serial_transform, n_serial_params, n_iregionNA, n_i_colorlist, n_i_shapelist, current_run, m1,m2,m3,m4, n_addressable = transform_res[n].values()
            iobj = n_addressable['iobj'] if 'iobj' in n_addressable else None
            oobj = n_addressable['oobj'] if 'oobj' in n_addressable else None

            
            
            

        
            
            

            

            
            n_type_vals = []; n_other_flagkeys = []; n_other_vals = []; n_shapecolor_vals = [] 

            n_shapecolor_vals.append(n_i_colorlist)
            n_shapecolor_vals.append(n_i_shapelist) 
            if iobj is not None:
                if type(iobj)==list: iobj_ = iobj[0]
                else: iobj_ = iobj
                n_shapecolor_vals.append(global_parsings[n_gridn]['i'][iobj_]['properties']['parsing_description'][0])
            else: n_shapecolor_vals.append(None)


            for dict_ in n_serial_transform:
                for key_ in dict_: 
                    if key_ == 'type': 
                        
                        n_type_vals.append(dict_[key_])
                    else: 
                        n_other_flagkeys.append(key_)
                        n_other_vals.append(dict_[key_])

            allflags = ['color','shape','TEMPparsing',*n_type_vals, *n_other_flagkeys] 
            
            to_add_as_false = []
            allmatches = [[1]*len(allflags)]
            obj_ids = [n]
            
            for m_ in range(len(transform_res)): 
                if esc(): break
                if esc2(): break
                if n == m_: continue
                c_tr_score, c_gridn, c_transform, c_params, c_iregionNA, c_i_colors_list, c_i_shape_list, current_run,m1,m2,m3,m4, c_addressable = transform_res[m_].values()
                ciobj = c_addressable['iobj'] if 'iobj' in c_addressable else None
                
                m_type_vals = []; m_other_flagkeys = []; m_other_vals = []; m_shapecolor_vals = [] 

                m_shapecolor_vals.append(c_i_colors_list)
                m_shapecolor_vals.append(c_i_shape_list)
                if ciobj is not None:
                    if type(ciobj)==list: iobj_ = ciobj[0]
                    else: iobj_ = ciobj
                    m_shapecolor_vals.append(global_parsings[c_gridn]['i'][iobj_]['properties']['parsing_description'][0])
                else: m_shapecolor_vals.append(None)



                for dict_ in c_transform:
                    for key_ in dict_: 
                        if key_ == 'type': 
                            m_type_vals.append(dict_[key_])
                        else: 
                            m_other_flagkeys.append(key_)
                            m_other_vals.append(dict_[key_])

                

                flags = []; matches = []

                
                samecolor = 1 if are_two_identical(n_shapecolor_vals[0], m_shapecolor_vals[0]) else 0
                sameshape = 1 if are_two_identical(n_shapecolor_vals[1], m_shapecolor_vals[1]) else 0
                TEMPparsing = 1 if are_two_identical(n_shapecolor_vals[2], m_shapecolor_vals[2]) else 0 
                flags.append('color'); matches.append(samecolor)
                flags.append('shape'); matches.append(sameshape)
                flags.append('TEMPparsing'); matches.append(TEMPparsing)

                
                
                
                for nix, n_type_val in enumerate(n_type_vals):
                    flag = 0
                    if n_type_val in m_type_vals:
                        flag = 1
                    elif n_type_val in ['movt'] and len(n_type_vals)==len(m_type_vals) and m_type_vals[nix] == 'static': 
                        flag = 1
                    flags.append(n_type_val); matches.append(flag)
                
                
                for nix, n_flagkey in enumerate(n_other_flagkeys):
                    flag = 0
                    mix = ix_of_x_in_y(x=n_flagkey, y=m_other_flagkeys)
                    if mix is not None:
                        if are_two_identical(n_other_vals[nix], m_other_vals[mix]):
                            flag = 1
                    flags.append(n_flagkey); matches.append(flag)

                
                
                


                allmatches.append(matches)
                obj_ids.append(m_)

                

            

            Analogies = allmatches
            serialised_keys = allflags


            


            if len(obj_ids)<=1: continue
            
            


            analogy_options_for_n, dets = enumerate_analogies(Analogies,obj_ids,[_ for _ in range(len(serialised_keys))],'new') 
            
            
            for z in range(len(dets)):
                if esc(): break
                if esc2(): break
                analogy_option = analogy_options_for_n[z]
                
                if n not in analogy_option: continue
                

                
                viskeys = []
                for k_,key_ in enumerate(serialised_keys):
                    temps1 = []
                    for an_o in analogy_option:
                        anix = obj_ids.index(an_o)
                        temps1.append(Analogies[anix][k_])
                    if np.all(temps1): viskeys.append(key_) 
                

                
                
                
                identical_masks_check = []; ta = []
                for a in analogy_option:
                    n_tr_score, gridn, n_serial_transform, n_serial_params, n_iregionNA, n_i_colorlist, n_i_shapelist, current_run, m1,m2,m3,m4,n_addressable = transform_res[a].values()
                    oobj = n_addressable['oobj'] if 'oobj' in n_addressable else None    
                    if oobj is not None: 
                        o_mask = global_parsings[gridn]['o'][oobj]['mask']
                        identical_masks_check.append((gridn,o_mask))
                        ta.append(a)
                if len(identical_masks_check)!=0:
                    lbls, uniques = label_unique_with_IDs(identical_masks_check)

                    if len(uniques) != len(identical_masks_check):
                        
                        continue

                    

                

                for mode__ in ['as_is', 'only_including_those_w_strictly_equal_serial_keys']:

                    if esc(): break

                    if mode__ == 'as_is': 
                        
                        
                        types_list = []
                        for k in analogy_option:
                            types_list.append([_['type'] for _ in transform_res[k]['serial_transforms']])

                        
                        
                        static_flag = True if np.all(['static' in _ for _ in types_list]) else False 
                        lens = [len(_) for _ in types_list]
                        if static_flag and min(lens) == 1: flag_toinclude_static = True 
                        else: flag_toinclude_static = False
                        
                        unique_flattened_types = list(set([x for xs in types_list for x in xs]))
                        rels = []
                        for typ in types_list:
                            for n0 in range(len(typ)-1): 
                                rels.append([typ[n0+1],'>',typ[n0]])
                        ordered_types = sort_relationsY(unique_flattened_types,rels)        
                        if ordered_types is None: print('WARNING'); continue
                        

                        reference_serial_transforms = []
                        for typ in ordered_types:
                            if flag_toinclude_static == False and typ == 'static': continue
                            reference_serial_transforms.append(typ)
                        if len(reference_serial_transforms) == 0: continue 

                        

                        
                        
                        new_serials = []; new_params = []
                        for k in analogy_option:
                            curr_types = [_['type'] for _ in transform_res[k]['serial_transforms']]
                            temp_serials =[{'type':'static'} for _ in range(len(reference_serial_transforms))]; temp_params = [{} for _ in range(len(reference_serial_transforms))] 
                            for m in range(len(reference_serial_transforms)):
                                reference_type = reference_serial_transforms[m]
                                if reference_type in curr_types:
                                    if curr_types.count(reference_type)!=1: print("ERROR")
                                    ix = curr_types.index(reference_type)
                                    temp_serials[m] = transform_res[k]['serial_transforms'][ix]
                                    temp_params[m] = transform_res[k]['serial_params'][ix]
                                else: pass
                            new_serials.append(temp_serials); new_params.append(temp_params)
                        


                        
                        
                        c1s = []; c2s = []
                        for mainref in reference_serial_transforms:
                            c1 = 0; c2 = 0
                            for currserial in types_list:
                                if mainref in currserial: c1+=1; c2 +=1
                                elif currserial == ['static']: c2+=1
                            c1s.append(c1/len(types_list)); c2s.append(c2/len(types_list))
                        C1 = np.mean(c1s); C2 = np.mean(c2s)
                        
                        
                        if len(reference_serial_transforms)==1 and reference_serial_transforms in [['movt'],['extension'],['copying'],['connection']]:
                            serial_score = C2
                        else: serial_score = C1

                        
                        static_frac_score = 1 - (types_list.count(['static']) / (len(types_list)*4)) if 'static' not in reference_serial_transforms else 1

                        

                        
                        same_origin_penalty = 1
                        iobjs_lists = [transform_res[k]['addressable']['iobj'] for k in analogy_option if transform_res[k]['addressable']['iobj'] is not None]
                        for n1 in range(len(iobjs_lists)):
                            ixs_ = ixs_of_x_in_y(x=iobjs_lists[n1],y=iobjs_lists)
                            if len(ixs_) > 1: 
                                same_origin_penalty = 0.5
                                curr_types = [types_list[_] for _ in range(len(types_list)) if _ in ixs_]
                                if are_all_identical(curr_types):
                                    same_origin_penalty = 0.8

                        avg_tr_score = np.mean([transform_res[k]['tr_score'] for k in analogy_option])
                        len_analogyobjs, num_keys, num_leaveouts = dets[z] 

                        len_score = [0.5,0.6,0.8,0.9,0.95,0.975,1][len_analogyobjs] if len_analogyobjs <= 6 else 1
                        key_leavout_score = (num_keys - num_leaveouts)/num_keys
                        

                        if not is_x_in_y(x=[analogy_option, reference_serial_transforms],y=analogy_and_serials_):
                        
                            unique_analogies_.append(analogy_option)
                            analogy_score = serial_score*static_frac_score*avg_tr_score*same_origin_penalty*len_score*key_leavout_score/10
                            
                            
                            analogy_scores_.append(analogy_score)
                            reference_serials_.append(reference_serial_transforms)
                            ordered_serial_transforms_.append(new_serials); ordered_serial_params_.append(new_params)
                            debug_types_.append(('type1',serial_score,avg_tr_score,len_score,key_leavout_score,same_origin_penalty,types_list,reference_serial_transforms,new_serials))
                            
                            analogy_and_serials_.append([analogy_option, reference_serial_transforms])


                            viskeys_all_.append(viskeys)


                    else: 
                        
                        
                        reference_serial_transforms = [_['type'] for _ in transform_res[analogy_option[0]]['serial_transforms']]
                        actual_k = []
                        for k in analogy_option:
                            if are_two_identical([_['type'] for _ in transform_res[k]['serial_transforms']], reference_serial_transforms):
                                actual_k.append(k)

                        new_serials = [transform_res[k]['serial_transforms'] for k in actual_k]
                        new_params = [transform_res[k]['serial_params'] for k in actual_k]

                        if len(actual_k) >= 3:
                            special_analogy_option = actual_k 



                            c1s = []; c2s = []
                            for mainref in reference_serial_transforms:
                                c1 = 0; c2 = 0
                                for currserial in types_list:
                                    if mainref in currserial: c1+=1; c2 +=1
                                    elif currserial == ['static']: c2+=1
                                c1s.append(c1/len(types_list)); c2s.append(c2/len(types_list))
                            C1 = np.mean(c1s); C2 = np.mean(c2s)
                            if len(reference_serial_transforms)==1 and reference_serial_transforms in [['movt'],['extension'],['copying']]:
                                serial_score = C2
                            else: serial_score = C1
                            static_frac_score = 1 - (types_list.count(['static']) / (len(types_list)*4)) if 'static' not in reference_serial_transforms else 1
                            
                            same_origin_penalty = 1
                            iobjs_lists = [transform_res[k]['addressable']['iobj'] for k in analogy_option if transform_res[k]['addressable']['iobj'] is not None]
                            for n1 in range(len(iobjs_lists)):
                                ixs_ = ixs_of_x_in_y(x=iobjs_lists[n1],y=iobjs_lists)
                                if len(ixs_) > 1: 
                                    same_origin_penalty = 0.5
                                    curr_types = [types_list[_] for _ in range(len(types_list)) if _ in ixs_]
                                    if are_all_identical(curr_types):
                                        same_origin_penalty = 0.8

                            avg_tr_score = np.mean([transform_res[k]['tr_score'] for k in analogy_option])
                            len_analogyobjs, num_keys, num_leaveouts = dets[z] 
                            len_score = [0.5,0.6,0.8,0.9,0.95,0.975,1][len_analogyobjs] if len_analogyobjs <= 6 else 1
                            key_leavout_score = (num_keys - num_leaveouts)/num_keys
                            if not is_x_in_y(x=[special_analogy_option, reference_serial_transforms],y=analogy_and_serials_):
                                unique_analogies_.append(special_analogy_option)
                                analogy_score = serial_score*static_frac_score*avg_tr_score*same_origin_penalty*len_score*key_leavout_score/10
                                
                                analogy_scores_.append(analogy_score)
                                reference_serials_.append(reference_serial_transforms)
                                ordered_serial_transforms_.append(new_serials); ordered_serial_params_.append(new_params)
                                debug_types_.append(('type2',serial_score,avg_tr_score,len_score,key_leavout_score, same_origin_penalty,types_list,reference_serial_transforms,new_serials))
                                analogy_and_serials_.append([special_analogy_option, reference_serial_transforms])
                                


                                viskeys_all_.append(viskeys)
            


            
        unique_analogies = [{'tempviskeysdel':viskeys_all_[_],'debug_types':debug_types_[_],'analogy':unique_analogies_[_],'analogy_score':analogy_scores_[_],'reference_serial_transforms':reference_serials_[_],'ordered_serial_transforms':ordered_serial_transforms_[_],'ordered_serial_params':ordered_serial_params_[_]} for _ in range(len(unique_analogies_))]
        
        presorted_unique_analogies = copy.deepcopy(unique_analogies)
        unique_analogies = sorted(unique_analogies, key=lambda x: (-x['analogy_score']))
        

        printable_unique_analogies = [[str(c)+' --------------------------------------',_['analogy'],_['debug_types']] for c,_ in enumerate(unique_analogies)]
        temppp = [[str(c)+' --------------------------------------',_['analogy'],float(round(_['analogy_score'],3))] for c,_ in enumerate(unique_analogies)]
        temppp1 = [[str(c)+' --------------------------------------',_['tempviskeysdel'],_['analogy'],_['reference_serial_transforms'],float(round(_['analogy_score'],3))] for c,_ in enumerate(unique_analogies)]





#     except Exception as e:
#         print("An exception occurred:")
#         traceback.print_exc()


# for __ in [0]:
#     starttime_main = time.perf_counter_ns() 
#     try:

        def self_color(m_list, m_states): 
            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: prop_list.append('n/a')
                else: prop_list.append(get_colors_of_obj(state['s']['mask'], state['s']['map']))
            return prop_list

        def self_size(m_list, m_states): 
            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: prop_list.append('n/a')
                else: prop_list.append(np.sum(state['s']['mask']))
            return prop_list

        def self_parsing(m_list, m_states): 
            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: prop_list.append('n/a')
                
                elif type(state['s']['mask'])==list: prop_list.append('n/a') 
                else: prop_list.append(state['s']['parsing'])
            return prop_list

        def self_parsing_NONBKG(m_list, m_states):
            
            bkg_colrs = []
            for m in m_list:
                state = m_states[m]
                if 's' in state:
                    if state['s']['parsing'][0] == 'background':
                        bkg_colrs.append(get_colors_of_obj(state['s']['mask'], state['s']['map']))
        
            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: prop_list.append('n/a')
                
                elif type(state['s']['mask'])==list: prop_list.append('n/a') 
                elif is_x_in_y(x=get_colors_of_obj(state['s']['mask'], state['s']['map']), y=bkg_colrs): prop_list.append('n/a BKG COLOR') 
                else: prop_list.append(state['s']['parsing'])

            return prop_list


        def self_shape(m_list, m_states): 
            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: prop_list.append('n/a')
                elif type(state['s']['mask'])==list: prop_list.append('n/a')
                else: prop_list.append(get_shape_of_obj(state['s']['mask'], state['s']['map']))
            return prop_list

        

        def self_color_AND_shape(m_list, m_states): 
            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: prop_list.append('n/a')
                else: prop_list.append([get_colors_of_obj(state['s']['mask'], state['s']['map']), get_shape_of_obj(state['s']['mask'], state['s']['map'])])
            return prop_list

        def self_parsing_AND_shape(m_list, m_states): 
            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: prop_list.append('n/a')
                elif type(state['s']['mask'])==list: prop_list.append('n/a')
                else: prop_list.append([state['s']['parsing'], get_shape_of_obj(state['s']['mask'], state['s']['map'])])
            return prop_list

        def self_parsing_AND_color(m_list, m_states): 
            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: prop_list.append('n/a')
                elif type(state['s']['mask'])==list: prop_list.append('n/a')
                else: prop_list.append([state['s']['parsing'], get_colors_of_obj(state['s']['mask'], state['s']['map'])])
            return prop_list



        def gridwise_oddcolorout_subframe_iobj(m_list, m_states):
            
            
            

            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)    

            temp = []
            for gridn in gridn_m_dict:
                all_colrs = []; corresp_m = []
                for m in gridn_m_dict[gridn]:
                    state = m_states[m]
                    if 's' not in state: continue
                    colrs = get_colors_of_obj(state['s']['mask'], state['s']['map'])
                    if state['s']['parsing'][0] in ['subframe_iobj']: all_colrs.append(colrs); corresp_m.append(m)
                
                if len(all_colrs)==0: continue
                lbls, uniques = label_unique_with_IDs(all_colrs); cands_ = []
                for k in range(np.max(lbls)+1):
                    if lbls.count(k)==1:
                        cands_.append(1)
                        odd_color_out = uniques[k]
                if len(cands_)!=1: odd_color_out = None 
                
                if odd_color_out is not None:
                    chosen_m = corresp_m[ix_of_x_in_y(x=odd_color_out, y=all_colrs)]
                    temp.append(chosen_m)

            prop_list = []
            for m in m_list:
                if m in temp: prop_list.append(1)
                else: prop_list.append(0)

            return prop_list

        def gridwise_largest_containerrect_iobj(m_list, m_states):
            
            
            
            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)    

            temp = []
            
            for gridn in gridn_m_dict:
                all_sizepxls = []; corresp_m = []
                for m in gridn_m_dict[gridn]:
                    state = m_states[m]
                    if 's' not in state: continue
                    sizepxls = np.sum(state['s']['mask'])
                    if state['s']['parsing'][0] in ['container_rect_1pxwidth']: all_sizepxls.append(sizepxls); corresp_m.append(m)
                if len(all_sizepxls)>0: temp.append(corresp_m[np.argmax(all_sizepxls)]) 

            
            prop_list = []
            for m in m_list:
                if m in temp: prop_list.append(1)
                else: prop_list.append(0)

            return prop_list

        def gridwise_hyperpbordercolor_containerrect_iobj(m_list, m_states):
            

            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: continue 
                if state['s']['parsing'][0] in ['container_rect_1pxwidth']: 
                    bordermask = get_outline_border_mask(state['s']['mask'],1)
                    prop_list.append(get_colors_of_obj(bordermask, np.where(bordermask, i_grids[state['gridn']], 0))) 
                else: prop_list.append('n/a')

            return prop_list




        def gridwise_noncentral_maskofcolors(m_list, m_states):

            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)    

            central_m = []

            for gridn in gridn_m_dict:
                
                centroid_rows=[]; centroid_cols=[];  row_ranges=[]; col_ranges=[]
                for m in gridn_m_dict[gridn]:
                    state = m_states[m]
                    if 's' not in state: continue
                    r,c = np.nonzero(state['s']['mask']) 
                    centroid_rows.append(r.mean()) 
                    centroid_cols.append(c.mean()) 
                    row_ranges.append((np.max(r)-np.min(r)) / state['s']['mask'].shape[0]) 
                    col_ranges.append((np.max(c)-np.min(c)) / state['s']['mask'].shape[1]) 
                
                horiz_marker = np.mean(row_ranges)
                vert_marker = np.mean(col_ranges) 
                
                

                if horiz_marker < vert_marker: 
                    middle_row = (np.min(centroid_rows) + np.max(centroid_rows))/2
                    middle_ix = np.argmin(np.abs(centroid_rows-middle_row))
                    central_m.append(gridn_m_dict[gridn][middle_ix])
                else: 
                    middle_col = (np.min(centroid_cols) + np.max(centroid_cols))/2
                    middle_ix = np.argmin(np.abs(centroid_cols-middle_col))
                    central_m.append(gridn_m_dict[gridn][middle_ix])

            prop_list = []
            for m in m_list:
                if m not in central_m: prop_list.append(1)
                else: prop_list.append(0)

            return prop_list

        def presence_of_opposite_diffcolor_obj(m_list, m_states): 

            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)    

            prop_list = []
            for m in m_list:
                gridn = m_states[m]['gridn']
                this_mask = m_states[m]['s']['mask']; this_colors = get_colors_of_obj(m_states[m]['s']['mask'],m_states[m]['s']['map'])

                
                plus_mask = np.zeros_like(this_mask)
                for dirn in ['N','S','E','W']:
                    newmask_, cdts_ = mask_in_direction(this_mask, dirn)
                    plus_mask = plus_mask | newmask_

                
                flag = False
                for c in gridn_m_dict[gridn]:
                    cand_mask = m_states[c]['s']['mask']; cand_colors = get_colors_of_obj(m_states[c]['s']['mask'],m_states[c]['s']['map'])
                    if np.sum((cand_mask==1)&(plus_mask==0))==0: 
                        if not are_two_identical(this_colors, cand_colors):
                            flag = True 
                
                if flag: prop_list.append(1)
                else: prop_list.append(0)

            return prop_list



        def all_subframe_iobjs(m_list, m_states):

            prop_list = []
            for m in m_list:
                state = m_states[m]
                if 's' not in state: prop_list.append(0)
                elif state['s']['parsing'][0] in ['subframe_iobj']: prop_list.append(1)
                else: prop_list.append(0)

            return prop_list





        def rankingfy(raw_prop_list, m_list, m_states):
        
            if len(m_list)!=len(raw_prop_list): print("ERROR")
            
            

            if 'n/a' in raw_prop_list: return ['n/a']*len(raw_prop_list) 


            
            
            
            
            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append((_,m))

            
            temp = {}
            for gridn in gridn_m_dict:
                pairs = gridn_m_dict[gridn]
                ordering = [[raw_prop_list[pairs[n][0]], pairs[n][1]] for n in range(len(pairs))] 
                ordering = sorted(ordering)
                ordered_ms = [_[1] for _ in ordering]
                
                prev = 'dfjsbdfjd'; counter = -1
                for _ in range(len(ordering)): 
                    if not are_two_identical(ordering[_][0],prev):
                        counter +=1
                        prev = ordering[_][0] 
                    temp[ordering[_][1]] = counter
            prop_list = []
            for m in m_list:
                prop_list.append(temp[m])
            return prop_list


        
        def analogywise_centroid_vertpos(m_list, m_states): 

            
            
            

            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)

            
            temp = {};  score_ranges=[]; score_diff_stds=[]
            for gridn in gridn_m_dict:
                row_ordering = []
                for m in gridn_m_dict[gridn]:
                    state = m_states[m]
                    if 's' not in state: row_ordering.append([-999, m]); continue
                    if type(state['s']['mask'])==list: continue 
                    r,c = np.nonzero(state['s']['mask'])
                    centroid = (r.mean(), c.mean())
                    row_ordering.append([centroid[0], m])
                row_ordering = sorted(row_ordering)
                row_ordered = [_[1] for _ in row_ordering if _[0]!=-999]

                rs = [_[0] for _ in row_ordering if _[0]!=-999]
                if len(rs)>0: 
                    score_ranges.append(np.max(rs)-np.min(rs)) 
                    score_diff_stds.append(np.std(np.diff(rs)))

                for p in range(len(row_ordered)): temp[row_ordered[p]] = p

            prop_list = []
            for m in m_list:
                if m in temp: prop_list.append(temp[m])
                else: prop_list.append('n/a')

            if len(score_ranges)>0: 
                score_range = np.mean(score_ranges); score_diff_std = np.mean(score_diff_stds)

            return prop_list

        def analogywise_centroid_horizpos(m_list, m_states): 
            
            
            

            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)

            
            temp = {}; score_ranges=[]; score_diff_stds=[]
            for gridn in gridn_m_dict:
                col_ordering = []
                for m in gridn_m_dict[gridn]:
                    state = m_states[m]
                    if 's' not in state: return 'n/a'
                    r,c = np.nonzero(state['s']['mask'])
                    centroid = (r.mean(), c.mean())
                    col_ordering.append([centroid[1], m])
                col_ordering = sorted(col_ordering)
                col_ordered = [_[1] for _ in col_ordering]

                cs = [_[0] for _ in col_ordering] 
                score_ranges.append(np.max(cs)-np.min(cs)) 
                score_diff_stds.append(np.std(np.diff(cs))) 

                for _ in range(len(col_ordered)): temp[col_ordered[_]] = _
            prop_list = []
            for m in m_list:
                prop_list.append(temp[m])

            score_range = np.mean(score_ranges); score_diff_std = np.mean(score_diff_stds)
            return prop_list

        def analogywise_tl_vertpos(m_list, m_states): 
            

            
            
            

            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)

            
            temp = {};  score_ranges=[]; score_diff_stds=[]
            for gridn in gridn_m_dict:
                row_ordering = []
                for m in gridn_m_dict[gridn]:
                    state = m_states[m]
                    if 's' not in state: return 'n/a'
                    r,c = np.nonzero(state['s']['mask'])
                    tl_pos = (r[0],c[0])
                    row_ordering.append([tl_pos[0], m])
                row_ordering = sorted(row_ordering)
                row_ordered = [_[1] for _ in row_ordering]
        
                rs = [_[0] for _ in row_ordering]
                score_ranges.append(np.max(rs)-np.min(rs)) 
                score_diff_stds.append(np.std(np.diff(rs)))

                for _ in range(len(row_ordered)): temp[row_ordered[_]] = _
            prop_list = []
            for m in m_list:
                prop_list.append(temp[m])

            score_range = np.mean(score_ranges); score_diff_std = np.mean(score_diff_stds)
            return prop_list

        def analogywise_tl_horizpos(m_list, m_states): 
            

            
            
            

            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)

            
            temp = {}; score_ranges=[]; score_diff_stds=[]
            for gridn in gridn_m_dict:
                col_ordering = []
                for m in gridn_m_dict[gridn]:
                    state = m_states[m]
                    if 's' not in state: return 'n/a'
                    r,c = np.nonzero(state['s']['mask'])
                    tl_pos = (r[0],c[0])
                    col_ordering.append([tl_pos[1], m])
                col_ordering = sorted(col_ordering)
                col_ordered = [_[1] for _ in col_ordering]

                cs = [_[0] for _ in col_ordering] 
                score_ranges.append(np.max(cs)-np.min(cs)) 
                score_diff_stds.append(np.std(np.diff(cs))) 

                for _ in range(len(col_ordered)): temp[col_ordered[_]] = _
            prop_list = []
            for m in m_list:
                prop_list.append(temp[m])

            score_range = np.mean(score_ranges); score_diff_std = np.mean(score_diff_stds)
            return prop_list

        

        def analogywise_centroid_position(m_list, m_states): 
      
            

            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)

            
            temp = {}
            for gridn in gridn_m_dict:
                row_ordering = []; col_ordering = []
                for m in gridn_m_dict[gridn]:
                    state = m_states[m]
                    if 's' not in state: return 'n/a'
                    r,c = np.nonzero(state['s']['mask'])
                    centroid = (r.mean(), c.mean())
                    row_ordering.append([centroid[0], m])
                    col_ordering.append([centroid[1], m])
                row_ordering = sorted(row_ordering); col_ordering = sorted(col_ordering)
                row_ordered = [_[1] for _ in row_ordering]; col_ordered = [_[1] for _ in col_ordering]

                if are_two_identical(row_ordered, col_ordered): pass 
                
                rs = [_[0] for _ in row_ordering];    cs = [_[0] for _ in col_ordering]
                row_range = np.max(rs)-np.min(rs); col_range = np.max(cs)-np.min(cs) 
                row_diff_std = np.std(np.diff(rs)); col_diff_std = np.std(np.diff(cs)) 
                if col_diff_std < row_diff_std and row_range > col_range: print('ERROR')
                
                if col_diff_std < row_diff_std:
                    for _ in range(len(row_ordered)): temp[row_ordered[_]] = _
                else:
                    for _ in range(len(col_ordered)): temp[col_ordered[_]] = _
            prop_list = []
            for m in m_list:
                prop_list.append(temp[m])
            return prop_list

        def analogywise_tl_position(m_list, m_states): 

            
            

            gridn_m_dict = {}
            for _,m in enumerate(m_list):
                if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                gridn_m_dict[m_states[m]['gridn']].append(m)

            
            temp = {}
            for gridn in gridn_m_dict:
                row_ordering = []; col_ordering = []
                for m in gridn_m_dict[gridn]:
                    state = m_states[m]
                    if 's' not in state: return 'n/a'
                    r,c = np.nonzero(state['s']['mask'])
                    tl_pos = (r[0],c[0])
                    row_ordering.append([tl_pos[0], m])
                    col_ordering.append([tl_pos[1], m])
                row_ordering = sorted(row_ordering); col_ordering = sorted(col_ordering)
                row_ordered = [_[1] for _ in row_ordering]; col_ordered = [_[1] for _ in col_ordering]

                if are_two_identical(row_ordered, col_ordered): pass 
                
                rs = [_[0] for _ in row_ordering];    cs = [_[0] for _ in col_ordering]
                row_range = np.max(rs)-np.min(rs); col_range = np.max(cs)-np.min(cs) 
                row_diff_std = np.std(np.diff(rs)); col_diff_std = np.std(np.diff(cs)) 
                if col_diff_std < row_diff_std and row_range > col_range: print('ERROR')

                if col_diff_std < row_diff_std:
                    for _ in range(len(row_ordered)): temp[row_ordered[_]] = _
                else:
                    for _ in range(len(col_ordered)): temp[col_ordered[_]] = _
            prop_list = []
            for m in m_list:
                prop_list.append(temp[m])
            return prop_list





        

        def fully_inline(target_mask, target_map, movt_dir, orth_type, mask, map, rrr = False):
            
            rows1,cols1 = np.where(target_mask==1)
            rows2,cols2 = np.where(mask==1)
            if orth_type == 'v' and are_two_identical(np.unique(cols1),np.unique(cols2)): return True
            if orth_type == 'h' and are_two_identical(np.unique(rows1),np.unique(rows2)): return True

            if orth_type == 'lead':
                bands = bands_in_dir(target_mask.shape, (1,-1))
                bcdts1=[]
                for m in range(len(rows1)):
                    bcdts1.append(bands[rows1[m],cols1[m]])
                bands = bands_in_dir(mask.shape, (1,-1))
                bcdts2=[]
                for m in range(len(rows2)):
                    bcdts2.append(bands[rows2[m],cols2[m]])
                if are_two_identical(bcdts1,bcdts2): return True

            if orth_type == 'anti':
                bands = bands_in_dir(target_mask.shape, (1,1))
                bcdts1=[]
                for m in range(len(rows1)):
                    bcdts1.append(bands[rows1[m],cols1[m]])
                bands = bands_in_dir(mask.shape, (1,1))
                bcdts2=[]
                for m in range(len(rows2)):
                    bcdts2.append(bands[rows2[m],cols2[m]])
                if are_two_identical(bcdts1,bcdts2): return True        

            

            return False

        def first_bump(target_mask, target_map, movt_dir, orth_type, mask, map, rrr = False):
            

            mask_in_dir, cdts_of_line_start_and_obj_end = mask_in_direction(target_mask, movt_dir)
            target_mask_leading_edge_cdts = [_[1] for _ in cdts_of_line_start_and_obj_end]

            opp_dir = ['S','SW','SE','W','E','N','NW','NE'][['N','NE','NW','E','W','S','SE','SW'].index(movt_dir)]
            mask_in_dir, cdts_of_line_start_and_obj_end = mask_in_direction(mask, opp_dir)
            mask_trailing_border_cdts = [_[0] for _ in cdts_of_line_start_and_obj_end]

            
            if is_any_x_in_y(x = target_mask_leading_edge_cdts, y = mask_trailing_border_cdts): return True

        

        def leading_edge(mask, movt_dir):
            
            mask_in_dir, cdts_of_line_start_and_obj_end = mask_in_direction(mask, movt_dir)
            leading_edge_cdts = [_[1] for _ in cdts_of_line_start_and_obj_end]
            return leading_edge_cdts

        def trailing_edge(mask, movt_dir):
            
            opp_dir = ['S','SW','SE','W','E','N','NW','NE'][['N','NE','NW','E','W','S','SE','SW'].index(movt_dir)]
            mask_in_dir, cdts_of_line_start_and_obj_end = mask_in_direction(mask, opp_dir)
            trailing_edge_cdts = [_[1] for _ in cdts_of_line_start_and_obj_end]
            return trailing_edge_cdts

        def leading_border(mask, movt_dir):
            
            mask_in_dir, cdts_of_line_start_and_obj_end = mask_in_direction(mask, movt_dir)
            leading_border_cdts = [_[0] for _ in cdts_of_line_start_and_obj_end]
            return leading_border_cdts

        def trailing_border(mask, movt_dir):
            
            opp_dir = ['S','SW','SE','W','E','N','NW','NE'][['N','NE','NW','E','W','S','SE','SW'].index(movt_dir)]
            mask_in_dir, cdts_of_line_start_and_obj_end = mask_in_direction(mask, opp_dir)
            trailing_border_cdts = [_[0] for _ in cdts_of_line_start_and_obj_end]
            return trailing_border_cdts

        def first_obj(interact_list, main_list, seq_labels): 
            
            prop_list = []
            for seqk in seq_labels:
                interact_set = [interact_list[n] for n in range(len(seq_labels)) if seq_labels[n]==seqk]
                
                temp = [0]*len(interact_set); temp[0] = 1
                prop_list.append(temp)

            return prop_list

        def obj_seqn(interact_list, main_list, seq_labels): 
            
            prop_list = []
            for seqk in seq_labels:
                interact_set = [interact_list[n] for n in range(len(seq_labels)) if seq_labels[n]==seqk]
                
                temp = list(range(len(interact_set)))
                prop_list.append(temp)

            return prop_list

        
        def num_of_its_color_anyshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list):
            color_nums = {}
            for iobj in iobj_all:
                colr = get_colors_of_obj(mask_list[iobj_list.index(iobj)], map_list[iobj_list.index(iobj)])
                if str(colr) not in color_nums: color_nums[str(colr)] = []
                color_nums[str(colr)].append(iobj)
            prop_list = []
            for iobj in iobj_cands:
                colr = get_colors_of_obj(mask_list[iobj_list.index(iobj)], map_list[iobj_list.index(iobj)])
                prop_list.append(len(color_nums[str(colr)]))
            return prop_list

        def is_largest_num_of_its_color_anyshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list):
            num_of_its_color = num_of_its_color_anyshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list)
            prop_list = [1 if pval == max(num_of_its_color) else 0 for pval in num_of_its_color]
            return prop_list

        def is_smallest_num_of_its_color_anyshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list):
            num_of_its_color = num_of_its_color_anyshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list)
            prop_list = [1 if pval == min(num_of_its_color) else 0 for pval in num_of_its_color]
            return prop_list             

        def is_oddeven_num_of_its_color_anyshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list):
            num_of_its_color = num_of_its_color_anyshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list)
            prop_list = [1 if pval%2 == 1 else 0 for pval in num_of_its_color]
            return prop_list   

        def num_of_its_color_itsshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list):
            shape = get_shape_of_obj(mask_list[iobj_list.index(iobj_cands[0])], map_list[iobj_list.index(iobj_cands[0])])
            color_nums = {}
            for iobj in iobj_all:
                colr = get_colors_of_obj(mask_list[iobj_list.index(iobj)], map_list[iobj_list.index(iobj)])
                if not are_two_identical(shape, get_shape_of_obj(mask_list[iobj_list.index(iobj)], map_list[iobj_list.index(iobj)])): continue
                if str(colr) not in color_nums: color_nums[str(colr)] = []
                color_nums[str(colr)].append(iobj)
            prop_list = []
            for iobj in iobj_cands:
                colr = get_colors_of_obj(mask_list[iobj_list.index(iobj)], map_list[iobj_list.index(iobj)])
                prop_list.append(len(color_nums[str(colr)])) 
            return prop_list
                                                    
        def is_largest_num_of_its_color_itsshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list):
            num_of_its_color = num_of_its_color_itsshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list)
            prop_list = [1 if pval == max(num_of_its_color) else 0 for pval in num_of_its_color]
            return prop_list

        def is_smallest_num_of_its_color_itsshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list):
            num_of_its_color = num_of_its_color_itsshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list)
            prop_list = [1 if pval == min(num_of_its_color) else 0 for pval in num_of_its_color]
            return prop_list                                                

        def is_oddeven_num_of_its_color_itsshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list):
            num_of_its_color = num_of_its_color_itsshape(iobj_cands, iobj_all,    iobj_list, mask_list, map_list)
            prop_list = [1 if pval%2 == 1 else 0 for pval in num_of_its_color]
            return prop_list   





        def interact_transforms(o_list_, m_states):
            prop_list = []
            for o in o_list_:
                serial_transforms = o_transform_list[o_list.index(o)]
                prop_list.append([_['type'] for _ in serial_transforms])
            return prop_list

        def reverse_mapping(list_of_props, matrix, default, ruletype):
            if ruletype=='sublist rule': print("ERROR")
            if ruletype=='shallow rule': return [default for _ in range(len(list_of_props))]
            if ruletype=='standard rule':
                list_of_params = []
                for prop in list_of_props:
                    ix = ix_of_x_in_y(x=prop,y=[_['PROPS'] for _ in matrix])
                    if ix is None:
                        if default is None: list_of_params.append(None) 
                        else: list_of_params.append(default)
                    else:
                        list_of_params.append(matrix[ix]['PARAM'])
                return list_of_params

        def check_mapping(list_of_props, list_of_params):
            
            if list(set(list_of_params))==[0,1]:
                pass 
            if list_of_props == 'n/a': return None 


            
            
            
            prop_labels, prop_uniques = label_unique_with_IDs(list_of_props)
            param_labels, param_uniques = label_unique_with_IDs(list_of_params)

            temp = {}; isvalid = all(temp.setdefault(x, y) == y for x, y in zip(prop_labels, param_labels)) 
            if isvalid:

                if max(param_labels) > 0: 
                    Matrix = []; reverse_M = {}
                    for m in range(len(prop_uniques)):
                        ix = prop_labels.index(m)
                        corresp_param = list_of_params[ix] 
                        Matrix.append({'PROPS':prop_uniques[m],'PARAM':corresp_param})

                        if corresp_param not in reverse_M: reverse_M[corresp_param] = [] 
                        reverse_M[corresp_param].append(prop_uniques[m])
                    
                    mults = [len(reverse_M[param])>1 for param in reverse_M]; 
                    default_param = [param for param in reverse_M][np.argmax(mults)] if sum(mults) == 1 else None
                
                    if sum(mults)>0: pass 

                    Matrix = sorted(Matrix, key=lambda x: (-x['PARAM'])) 
                    return ['standard rule', Matrix, default_param]
                
                else: 
                    default_param = param_uniques[0]
                    return ['shallow rule', None, default_param]


            
            
            

            param_labels, param_uniques = label_unique_with_IDs(list_of_params)
            
            

            
            
            subitem_rules = []
            for p in range(np.max(param_labels)+1):
                
                presence_pairs = []
                for n in range(len(list_of_props)):
                    if param_labels[n] == p:
                        curr_list = list_of_props[n] 
                        
                        if type(curr_list) is list:
                            for subitem in curr_list:
                                
                                
                                if not is_x_in_y(x=subitem,y=[_[0] for _ in presence_pairs]): presence_pairs.append([subitem,[]])
                                ix = ix_of_x_in_y(x=subitem,y=[_[0] for _ in presence_pairs])
                                presence_pairs[ix][1].append(n)

                        elif type(curr_list) is not list:
                            
                            
                            if not is_x_in_y(x=curr_list,y=[_[0] for _ in presence_pairs]): presence_pairs.append([curr_list,[]])
                            ix = ix_of_x_in_y(x=curr_list,y=[_[0] for _ in presence_pairs])
                            presence_pairs[ix][1].append(n)

                qualifying_subitems = []
                for unique_subitem in [_[0] for _ in presence_pairs]:
                    
                    
                    ix = ix_of_x_in_y(x=unique_subitem,y=[_[0] for _ in presence_pairs])
                    if are_two_identical( sorted(presence_pairs[ix][1]),  [n for n in range(len(list_of_props)) if param_labels[n]==p]):
                        flag = True 
                        for n1 in range(len(list_of_props)):
                            if param_labels[n1] != p:
                                curr_list1 = list_of_props[n1]
                                if type(curr_list1) is list and is_x_in_y(x=unique_subitem, y=curr_list1): flag = False
                                elif type(curr_list1) is not list and are_two_identical(unique_subitem, curr_list1): flag = False
                        if flag:
                            qualifying_subitems.append(unique_subitem)
                if len(qualifying_subitems)==0: 
                    
                    subitem_rules.append('No')
                if len(qualifying_subitems)==1: 
                    
                    subitem_rules.append(qualifying_subitems[0])
                if len(qualifying_subitems)>1: 
                    
                    subitem_rules.append('Mult') 
                
            
            

            if is_x_in_y(x='Mult',y=subitem_rules) and subitem_rules.count('No') <= 1:
                print('special rule')
                
                Matrix = []
                for q, sir in enumerate(subitem_rules):
                    if sir != 'No': Matrix.append({'PROPS':sir,'PARAM':param_uniques[q]})
                    elif sir == 'No': Matrix.append({'PROPS':'other','PARAM':param_uniques[q]})
                default_param = None
                
                Matrix = sorted(Matrix, key=lambda x: (-x['PARAM'])) 
                return ['sublist rule', Matrix, default_param]

            return None

        def gridwise_listgroup(h_states_):
            temp = {}; listable = False
            for state in h_states_:
                if state['is_in_m_states_analogy']==1:
                    if state['gridn'] not in temp: temp[state['gridn']] = []
                    temp[state['gridn']].append(state['list_iobj_m'])
                    if state['list_iobj_m'] is not None: listable = True
            
            flag = True if listable else False
            for gn in temp:
                if len(list(set(temp[gn])))==1: pass
                else: flag = False
            if flag: return True
            else: return False



        def iobjs_tile_creation(map_list, mask_list,  details):
            construction = details 

            n_rows, n_cols, safe_tile_ixs, tiles, frame_specs = construction['n_rows'], construction['n_cols'], construction['safe_tile_ixs'], construction['tiles'], construction['frame_specs']
            if construction['bkg_dets'] is not None: bkg_hyperp = construction['bkg_dets']['bkg_color']
            else: return np.zeros_like(map_list[0]), np.zeros_like(mask_list[0])


            
            shapes = []; bb_masks = []; bb_maps = []
            for k in range(len(tiles)):
                mask_, map_ = mask_list[safe_tile_ixs[k]], map_list[safe_tile_ixs[k]]
                color_changes = tiles[k]['color_changes']
                
                old_colors = get_colors_of_obj(mask_, map_) 
                
                if color_changes is None: pass 
                else: 
                    map_, mask_ = recolor(map_, mask_, color_changes)

                
                bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_,map_)
                shapes.append(bb_mask.shape)
                bb_masks.append(bb_mask)
                bb_maps.append(bb_map)
            
            
            f1,f2,f3,f4 = frame_specs 


            
            widths = [shapes[_][1] for _ in range(n_cols)]
            assign_tls_c = []; assign_ends_c = []
            for c in range(n_cols):
                if c==0: tracker = f1 
                assign_tls_c.append(tracker)
                tracker += (widths[c]+f2[c])
                assign_ends_c.append(tracker)

            

            
            heights = [shapes[(n_cols*_+0)][0] for _ in range(n_rows)]
            assign_tls_r = []; assign_ends_r = []
            for r in range(n_rows):
                if r==0: tracker = f3 
                assign_tls_r.append(tracker)
                tracker += (heights[r]+f4[r])
                assign_ends_r.append(tracker)
            
            
            
            
            
            construct = np.ones((assign_ends_r[-1], assign_ends_c[-1])) * bkg_hyperp
            
            k=0
            for r in range(n_rows):
                for c in range(n_cols):
                    assign_tl = (assign_tls_r[r],assign_tls_c[c])
                    bb_mask, bb_map = bb_masks[k], bb_maps[k]
                    
                    rows,cols = np.where(bb_mask==1)
                    for m in range(len(rows)):
                        colr = bb_map[rows[m],cols[m]]
                        new_rc = (assign_tl[0]+rows[m],assign_tl[1]+cols[m])
                        construct[new_rc] = colr
                    k+=1

            
            newmap = construct
            newmask = np.ones_like(construct)
            return newmap, newmask



        def mask_of_all_colors_maintained_or_disappeared(input_map, mask_to_transform, colors_maintained, colors_disappeared, applied_selector_large_on_both_masks, **kwargs):
            
            
        
            if applied_selector_large_on_both_masks:
                selector_large = selector_regions[gridn_]['large']
                mask_to_transform = mask_to_transform & selector_large
            
            output_map = np.zeros_like(input_map)
            transformed_mask = np.zeros_like(mask_to_transform)

            all_i_colors = np.unique(input_map[mask_to_transform==1])

            for i_color in all_i_colors:
                if i_color in colors_maintained: 
                    transformed_mask = transformed_mask | (input_map==i_color).astype(int)
                    output_map = np.where(transformed_mask, input_map, output_map)
                elif i_color in colors_disappeared:
                    pass

            return output_map, transformed_mask


        def slotting_hyperp_obj(map_list, mask_list,  bank_type, flavour_type, primary_color, secondary_colors, pbkg_color,   bank):
            
            
            

            
            assigns = {'primary':None, 'secondary_list':[], 'pbkg':None}
            for k in range(len(map_list)):
                colr = get_colors_of_obj(mask_list[k],map_list[k])
                if colr == primary_color: assigns['primary'] = k
                elif colr in secondary_colors: assigns['secondary_list'].append(k)
                elif colr == pbkg_color: assigns['pbkg'] = k

            composite_i_grid = np.zeros_like(map_list[0]) 
            composite_i_mask = np.zeros_like(map_list[0])
            for k in range(len(map_list)): composite_i_grid = np.where(mask_list[k], map_list[k], composite_i_grid); composite_i_mask = composite_i_mask | mask_list[k]
            


            def recursive_slot(all_matches, i_grid):
                for itern in range(100):
                    
                    

                    already_slotted = np.zeros_like(i_grid) 
                    for match in all_matches: 
                        if match[2] == True : 
                            already_slotted += match[1]
                    already_slotted = (already_slotted==1).astype(int)

                    
                    single_slot_solns = np.zeros_like(i_grid)
                    for match in all_matches: 
                        if match[2] == False: 
                            if np.sum(match[1] & already_slotted)==0:
                                single_slot_solns += match[1]
                            else: match[2] = 'invalidated'
                    


                    
                    n_improvements = 0; newslottings = np.zeros_like(i_grid)

                    minval = 1
                    r, c = np.nonzero(single_slot_solns)
                    minvals = list(single_slot_solns[r, c])
                    if minvals != []: minval = np.min(minvals)

                    for match in all_matches:
                        
                        
                        
                        
                        
                        
                        
                        
                        if minval in single_slot_solns[match[1]==1] and match[2] != 'invalidated':
                            newslottings += match[1]
                            if np.amax(newslottings)>=2: break 
                            match[2] = True 
                            n_improvements += 1
                            

                    if np.amax(newslottings)>=minval+1: 
                        
                        return None 

                    

                    if n_improvements == 0: break
                return all_matches
            
            if bank_type == '1':
                
                

                primary_color = primary_color
                secondary_colors = secondary_colors
                secondary_masks = []
                s_iobj = get_contiguous_regions(composite_i_grid,None,True,False) 
                for k in range(0,np.max(s_iobj)+1):
                    s_obj_mask = (s_iobj==k).astype(int)
                    s_obj_map = np.where(s_obj_mask, composite_i_grid, 0)
                    colr = get_colors_of_obj(s_obj_mask, s_obj_map)
                    if colr!=[] and colr[0] in secondary_colors: 
                        secondary_masks.append(s_obj_mask)
                
                
                all_matches = []; bankc = 0
                for shape, _ in bank:
                    overlapmasks = get_all_overlap_masks_incl_spillover(shape, composite_i_grid)
                    for smask in secondary_masks:
                        overlapm_cands = []
                        for m in range(len(overlapmasks)):
                            if np.sum(overlapmasks[m] & smask)>0 and np.sum((smask==1)&(overlapmasks[m]==0))==0: 
                                if list(set(composite_i_grid[((overlapmasks[m]==1)&(smask==0)).astype(int)==1])) == [primary_color]: 
                                    overlapm_cands.append(m)
                        for m in overlapm_cands:
                            all_matches.append([bankc, overlapmasks[m],    False, smask])
                    bankc +=1 
                


            if bank_type == '2':
                
                
                primary_color = primary_color
                all_matches = []; bankc = 0
                for shape, _ in bank:
                    
                    overlapmasks = get_all_overlap_masks_excl_spillover(shape, composite_i_grid)
                    overlapm_cands = []
                    for m in range(len(overlapmasks)):
                        if list(set(composite_i_grid[overlapmasks[m]==1])) == [primary_color]: 
                            overlapm_cands.append(m)
                    for m in overlapm_cands:
                        all_matches.append([bankc, overlapmasks[m],    False])
                    bankc +=1 
                

            recongrid = copy.deepcopy(composite_i_grid)

            if flavour_type == '1':
                
                
                all_matches_processed = recursive_slot(all_matches, composite_i_grid)
                if all_matches_processed is not None:
                    
                    recongrid = copy.deepcopy(composite_i_grid)
                    for match in all_matches_processed:
                        if match[2] == True:
                            bankc, mask_ = match[0], match[1]
                            deficit_color = bank[bankc][1]
                            if bank_type == '1': smask=match[3]; recongrid = np.where(((mask_==1)&(smask==0)).astype(int), deficit_color[0], recongrid)
                            if bank_type == '2': recongrid = np.where(mask_, deficit_color[0], recongrid)

            if flavour_type == '2':
                
                
                
                
                for match in all_matches: match[2] = False 

                running_matches = []
                for b in range(len(bank)):
                    b_matches = [match for match in all_matches if match[0] == b] 
                    running_plus_b_matches = b_matches + running_matches
                    running_matches_processed = recursive_slot(running_plus_b_matches, composite_i_grid) 
                    running_matches = running_matches_processed 
                    if running_matches is None: break

                if running_matches is not None:
                    
                    recongrid = copy.deepcopy(composite_i_grid)
                    for match in running_matches:
                        if match[2] == True:
                            bankc, mask_ = match[0], match[1]
                            deficit_color = bank[bankc][1]
                            recongrid = np.where(mask_, deficit_color[0], recongrid)


            newmap, newmask = recongrid, composite_i_mask
            
            
            return newmap, newmask    




        
        for gkey in global_parsings:
            for iobj1 in global_parsings[gkey]['i']:
                if len(global_parsings[gkey]['i'][iobj1]['properties']['parsing_description'])==3:
                    global_parsings[gkey]['i'][iobj1]['properties']['parsing_description'].pop(1) 
            for oobj1 in global_parsings[gkey]['o']:
                if len(global_parsings[gkey]['o'][oobj1]['properties']['parsing_description'])==3:
                    global_parsings[gkey]['o'][oobj1]['properties']['parsing_description'].pop(1)
        for gkey in initial_global_parsings:
            for iobj1 in initial_global_parsings[gkey]['i']:
                if len(initial_global_parsings[gkey]['i'][iobj1]['properties']['parsing_description'])==3:
                    initial_global_parsings[gkey]['i'][iobj1]['properties']['parsing_description'].pop(1)
            for oobj1 in initial_global_parsings[gkey]['o']:
                if len(initial_global_parsings[gkey]['o'][oobj1]['properties']['parsing_description'])==3:
                    initial_global_parsings[gkey]['o'][oobj1]['properties']['parsing_description'].pop(1)





        aa = T1()

        if esc(): breaker = 'break' / 2

        o_region_maskvs = []; o_region_maps = []; o_region_gridns = [] 
        for n in range(len(transform_res)):
            n_tr_score, gridn, n_serial_transform, n_serial_params, n_iregionNA, n_i_colorlist, n_i_shapelist, current_run, m1,m2,m3,m4,n_addressable = transform_res[n].values()
            iobj = n_addressable['iobj'] if 'iobj' in n_addressable else None
            oobj = n_addressable['oobj'] if 'oobj' in n_addressable else None    
            o_region_gridns.append(gridn)
            if oobj is not None: 
                
                
                o_region_maskvs.append(global_parsings[gridn]['o'][oobj]['maskv'])
                o_region_maps.append(global_parsings[gridn]['o'][oobj]['map'])
            else: o_region_maskvs.append(None); o_region_maps.append(None)
            

        analogy_o_maskvs = []; analogy_maskvs_gridn0 = []; analogy_o_maskvs_all = {_:[] for _ in range(num_demo_grids)}
        for i in range(len(unique_analogies)):
            i_analogy = unique_analogies[i]['analogy'] 
            mask_dict = {}
            for gridn in range(num_demo_grids):
                mask_dict[gridn] = np.zeros_like(o_grids[gridn])
            for n in i_analogy:
                gridn = o_region_gridns[n]
                if o_region_maskvs[n] is not None:
                    mask_dict[gridn] = mask_dict[gridn] | o_region_maskvs[n]
            analogy_o_maskvs.append(mask_dict)
            analogy_maskvs_gridn0.append(mask_dict[0])
            for gridn in range(num_demo_grids):
                analogy_o_maskvs_all[gridn].append(mask_dict[gridn])


        omaskv_explantionsum = []
        for i in range(len(analogy_o_maskvs)):
            currsum = 0
            for g in range(num_demo_grids):
                currsum += np.sum(analogy_o_maskvs[i][g])
            omaskv_explantionsum.append(currsum)
        
        

        ixs = np.argsort(omaskv_explantionsum)[::-1]
        temppp1[ixs[0]]


        web = {} 
        solns = []; solnrulecands = []

        soln_found = False
        for trialn in range(20):
            if esc(): break
            if trialn > 0 and esc01(): break

            
            block_analogies = [0] * len(unique_analogies)
            

            chosen_groups = []
            chosen_rule_cands = []


            for itern in range(10):
                if esc(): break
                if itern > 0 and esc01(): break
                if itern == 0 and '[]' in web and web['[]'] is False:
                    
                    
                    break

                remove_dets = {}
                anyfound = False

                top5unblocked = [] 
                for n in range(len(unique_analogies)):
                    if block_analogies[n] == 0: top5unblocked.append(n)
                    if itern > 0 and len(top5unblocked) == 5: break 
                    elif itern==0 and len(top5unblocked) == 20: break 
                    

                if str(chosen_groups) not in web: web[str(chosen_groups)] = top5unblocked 
                

                
                for grn in range(len(unique_analogies)):
                    if esc(): break
                    if block_analogies[grn] == 1: continue

                    trying_groups = chosen_groups + [grn] 
                    if str(trying_groups) in web:
                        if web[str(trying_groups)] is False: 
                            
                            continue
                        if web[str(trying_groups)] == []: 
                            
                            continue
                    if str(chosen_groups) in web:
                        if web[str(chosen_groups)] is False: 
                            
                            continue
                        if web[str(chosen_groups)] == []: 
                            
                            continue
                        
                        if web[str(chosen_groups)] is not False:
                            if grn not in web[str(chosen_groups)]: 
                                if itern != 0: 
                                    print('Error')
                                continue


                    
                    

                    groups = [[grn]]+[[_] for _ in chosen_groups] 
                    curr_group = [grn]
                    
                    remove_dets[grn] = []

                    rule_cands = {} 
                    rule_cands['curr_group'] = curr_group
                    rule_cands['requires_access'] = []
                    rule_cands['obj_select'] = []
                    rule_cands['layerings'] = []
                    rule_cands['gridsize_restriction'] = []
                    


                    
                    if not are_all_identical([len(unique_analogies[_]['reference_serial_transforms']) for _ in curr_group]): print('ERROR'); break
                    num_stacks = len(unique_analogies[curr_group[0]]['reference_serial_transforms'])
                    for stackn in range(num_stacks): 
                        if esc(): break
                        ref_types = [unique_analogies[_]['reference_serial_transforms'][stackn] for _ in curr_group]; ref_types_w_statics_leftout = [ref_type for ref_type in ref_types if ref_type!='static']
                        if len(ref_types_w_statics_leftout)>1 and not are_all_identical(ref_types_w_statics_leftout): print('ERROR')
                        curr_ref_serial_transform = ref_types_w_statics_leftout[0] if len(ref_types_w_statics_leftout)>0 else ref_types[0]


                        rule_cands['stackn'+str(stackn)] = []



                        
                        
                        def get_m_states(groups, stackn, remove_dets, unique_analogies, transform_res, global_parsings):
                            global gridn 
                            global gridn_
                            

                            
                            
                            m_states_pre = []
                            for gr in range(len(groups)):
                                if gr==0: 
                                    for group in groups[gr]:
                
                                        ref_serial = unique_analogies[group]['reference_serial_transforms']
                                        curr_analogy_transforms = unique_analogies[group]['ordered_serial_transforms']
                                        curr_analogy_params = unique_analogies[group]['ordered_serial_params']
                                        
                                        
                                        
                                        
                                        
                                        
                                        a=0
                                        for k in unique_analogies[group]['analogy']:
                                            if group in remove_dets and k in remove_dets[group]: continue 
                                            gridn = transform_res[k]['gridn']
                                            gridn_ = gridn 
                
                                            if gridn > num_demo_grids-1: continue 
                                            iobj_ = transform_res[k]['addressable']['iobj'] if 'iobj' in transform_res[k]['addressable'] else None
                                            oobj_ = transform_res[k]['addressable']['oobj'] if 'oobj' in transform_res[k]['addressable'] else None
                                            i_region, o_region = iobj_, oobj_ 
                                            n_serial_transform = curr_analogy_transforms[a]; n_serial_params = curr_analogy_params[a]
                                            curr_ref_serial = ref_serial[stackn]

                                            s_state = stackn; e_state = stackn+1 
                                            statedets = {'type':'analogy','gridn':gridn,'k':k,'curr_ref_serial':curr_ref_serial, 'ref_serial':ref_serial,
                                                        'addressable':transform_res[k]['addressable'],'serial_transform':n_serial_transform,'serial_param':n_serial_params}
                                            if i_region is not None: 
                                                for c, state in enumerate([s_state,e_state]):
                                                    
                                                    
                                                    if type(i_region)==list:
                                                        
                                                        
                                                        
                                                        

                                                        i_mask = []; i_map = []; i_obj_parsing_type = []
                                                        for i_, i_region_ in enumerate(i_region):
                                                            i_mask1, i_map1, i_maskv1, i_masko1, i_obj_parsing_type1 = global_parsings[gridn]['i'][i_region_]['mask'], global_parsings[gridn]['i'][i_region_]['map'], global_parsings[gridn]['i'][i_region_]['maskv'], global_parsings[gridn]['i'][i_region_]['masko'], global_parsings[gridn]['i'][i_region_]['properties']['parsing_description']
                                                            i_mask.append(i_mask1); i_map.append(i_map1); i_obj_parsing_type.append(i_obj_parsing_type1)


                                                    else: i_mask, i_map, i_maskv, i_masko, i_obj_parsing_type = global_parsings[gridn]['i'][i_region]['mask'], global_parsings[gridn]['i'][i_region]['map'], global_parsings[gridn]['i'][i_region]['maskv'], global_parsings[gridn]['i'][i_region]['masko'], global_parsings[gridn]['i'][i_region]['properties']['parsing_description']
                                                    
                                                    if state == 0: m_mask, m_map = i_mask, i_map
                                                    else:
                                                        curr_mask, curr_map = i_mask, i_map
                                                        for s in range(int(state)):
                                                            
                                                            curr_fn = globals()[n_serial_transform[s]['type']]
                                                            
                                                            curr_params = n_serial_params[s]
                                                            
                                                            curr_map, curr_mask = curr_fn(curr_map, curr_mask,**curr_params) 
                                                        m_mask, m_map = curr_mask, curr_map
                                                    is_nonzero_mask = True if np.sum(m_mask) > 0 else False 
                                                    
                                                    is_o_state = True if len(n_serial_transform) - state == 0 else False
                                                    if o_region is not None: 
                                                        
                                                        o_mask, o_map, o_maskv, o_masko, o_obj_parsing_type = global_parsings[gridn]['o'][o_region]['mask'], global_parsings[gridn]['o'][o_region]['map'], global_parsings[gridn]['o'][o_region]['maskv'], global_parsings[gridn]['o'][o_region]['masko'], global_parsings[gridn]['o'][o_region]['properties']['parsing_description']
                                                    else: o_obj_parsing_type = None
                                                    
                                                    
                                                    
                                                    
                                                    if c==0 and is_nonzero_mask: 
                                                        
                                                        statedets['s'] = {'obj':i_region,'mask':m_mask,'map':m_map,'parsing':i_obj_parsing_type}
                                                    if c==1 and is_nonzero_mask: 
                                                        
                                                        statedets['e'] = {'obj':o_region,'mask':m_mask,'map':m_map,'parsing':o_obj_parsing_type}
                                            elif o_region is not None: 
                                                if len(n_serial_transform) - e_state == 0: 
                                                    
                                                    o_mask, o_map, o_maskv, o_masko, o_obj_parsing_type = global_parsings[gridn]['o'][o_region]['mask'], global_parsings[gridn]['o'][o_region]['map'], global_parsings[gridn]['o'][o_region]['maskv'], global_parsings[gridn]['o'][o_region]['masko'], global_parsings[gridn]['o'][o_region]['properties']['parsing_description']
                                                    
                                                    m_mask, m_map = o_mask, o_map
                                                    statedets['e'] = {'obj':o_region,'mask':m_mask,'map':m_map,'parsing':o_obj_parsing_type}
                                                else: print("ERROR")
                                            m_states_pre.append(statedets)
                                            a+=1
                                else: 
                                    for group in groups[gr]:
                                        
                                        
                                        
                                        a=0
                                        for tr_re in unique_analogies[group]['analogy']:
                                            if group in remove_dets and tr_re in remove_dets[group]: continue 
                                            gridn_ = transform_res[tr_re]['gridn']
                
                                            if gridn_ > num_demo_grids-1: continue 
                                            iobj_ = transform_res[tr_re]['addressable']['iobj'] if 'iobj' in transform_res[tr_re]['addressable'] else None
                                            oobj_ = transform_res[tr_re]['addressable']['oobj'] if 'oobj' in transform_res[tr_re]['addressable'] else None
                                            
                                            

                                            
                                            statedets = {'type':'other','gridn':gridn_,'k':tr_re,'curr_ref_serial':'N/A',
                                                            'addressable':transform_res[tr_re]['addressable'],'serial_transform':'N/A','serial_param':'N/A'}
                                            if iobj_ is not None: 

                                                
                                                if type(iobj_)==list:
                                                    
                                                    
                                                    
                                                    

                                                    imask = []; imap = []; iparsing = []
                                                    for i_, i_region_ in enumerate(iobj_):
                                                        i_mask1, i_map1, i_maskv1, i_masko1, i_obj_parsing_type1 = global_parsings[gridn_]['i'][i_region_]['mask'], global_parsings[gridn_]['i'][i_region_]['map'], global_parsings[gridn_]['i'][i_region_]['maskv'], global_parsings[gridn_]['i'][i_region_]['masko'], global_parsings[gridn_]['i'][i_region_]['properties']['parsing_description']
                                                        imask.append(i_mask1); imap.append(i_map1); iparsing.append(i_obj_parsing_type1)


                                                else: imask, imap, iparsing = global_parsings[gridn_]['i'][iobj_]['mask'],global_parsings[gridn_]['i'][iobj_]['map'], global_parsings[gridn_]['i'][iobj_]['properties']['parsing_description']
                                                
                                                statedets['s'] = {'obj':iobj_,'mask':imask,'map':imap,'parsing':iparsing}
                                            if oobj_ is not None: 
                                                omask, omap, oparsing = global_parsings[gridn_]['o'][oobj_]['mask'],global_parsings[gridn_]['o'][oobj_]['map'], global_parsings[gridn_]['o'][oobj_]['properties']['parsing_description']
                                                statedets['e'] = {'obj':oobj_,'mask':omask,'map':omap,'parsing':oparsing}
                                            m_states_pre.append(statedets)
                                            a+=1
                            return m_states_pre
                        m_states_pre = get_m_states(groups, stackn, remove_dets, unique_analogies, transform_res, global_parsings)
                        m_states = copy.deepcopy(m_states_pre)


                        
                        
                        

                        def collapse_multiple_extension_cands(m_states, stackn):
                            
      
                            
                                
                            tempstores = []
                            for state in m_states:

                                
                                
                                
                                
                                

                                if state['type'] == 'analogy':
                                    if state['serial_transform'][stackn]['type'] == 'extension':
                                        ext_dets = state['serial_param'][stackn]['ext_details']
                                        ext_fn_characterisation = ext_dets['ext_fn_characterisation'] 
                                        for dirn_entry in ext_fn_characterisation:
                                            tempstores.append(dirn_entry['rule_info'])
                                    elif state['serial_transform'][stackn]['type'] == 'static':
                                        pass 


                            

                            if len(tempstores)!=0:
                            
                                for ix in range(np.max([len(_) for _ in tempstores])): 

                                    color_rules = []; len_rules = [] 
                                    for tempst in tempstores: 
                                        if ix > len(tempst)-1 or tempst[ix] == []: color_rules.append([]); len_rules.append([]) 
                                        else:
                                            objrule,colorrule,lenrule = tempst[ix]
                                            temp = []
                                            for opt in colorrule:
                                                if type(opt)==tuple: temp.append('hyperp')
                                                else: temp.append(opt)
                                            color_rules.append(temp)
                                            temp = []
                                            for opt in lenrule:
                                                if type(opt)==tuple: temp.append('hyperp')
                                                else: temp.append(opt)
                                            len_rules.append(temp)
                                    unique_color_rules = []
                                    for sublist in color_rules: unique_color_rules.extend(sublist)
                                    unique_color_rules = list(set(unique_color_rules)) 
                                    unique_len_rules = []
                                    for sublist in len_rules: unique_len_rules.extend(sublist)
                                    unique_len_rules = list(set(unique_len_rules)) 


                                    
                                    found = False; color_rule = None

                                    
                                    if not found:
                                        for rule in unique_color_rules:
                                            if np.all([rule in sublist for sublist in color_rules if len(sublist)>0]): found = True; color_rule = [rule]; break 
                                            
                                    
                                    if not found and len(unique_color_rules)>=2:
                                        
                                        for rule1 in unique_color_rules:
                                            for rule2 in unique_color_rules:
                                                if are_two_identical(rule1,rule2): continue
                                                rule1explains = [rule1 in sublist for sublist in color_rules if len(sublist)>0]
                                                rule2explains = [rule2 in sublist for sublist in color_rules if len(sublist)>0]
                                                if np.all([rule1explains[n]+rule2explains[n]==1 for n in range(len(rule1explains))]): found = True; color_rule = [rule1,rule2]; break


                                    
                                    found = False; len_rule = None

                                    
                                    if not found:
                                        for rule in unique_len_rules:
                                            if np.all([rule in sublist for sublist in len_rules if len(sublist)>0]): found = True; len_rule = [rule]; break

                                    
                                    if not found and len(unique_len_rules)>=2:
                                        
                                        for rule1 in unique_len_rules:
                                            for rule2 in unique_len_rules:
                                                if are_two_identical(rule1,rule2): continue
                                                rule1explains = [rule1 in sublist for sublist in len_rules if len(sublist)>0]
                                                rule2explains = [rule2 in sublist for sublist in len_rules if len(sublist)>0]
                                                if np.all([rule1explains[n]+rule2explains[n]==1 for n in range(len(rule1explains))]): found = True; len_rule = [rule1,rule2]; break



                                    if color_rule is None or len_rule is None: print("WARNING")


                                    
                                    try:
                                        for state in m_states:
                                            if state['type'] == 'analogy':

                                                if state['serial_transform'][stackn]['type'] == 'extension':
                                                    ext_dets = state['serial_param'][stackn]['ext_details']
                                                    ext_fn_characterisation = ext_dets['ext_fn_characterisation'] 
                                                    if state['type'] == 'analogy': 
                                                        new_efc = []
                                                        for dirn_entry in ext_fn_characterisation:
                                                            
                                                            get_hyperp = None
                                                            for _ in dirn_entry['rule_info'][ix][1]:
                                                                if type(_) == tuple:
                                                                    get_hyperp = _
                                                            
                                                            for found_rule in color_rule:
                                                                if found_rule == 'hyperp' and get_hyperp is not None: dirn_entry['rule_info'][ix][1] = [get_hyperp]; break
                                                                elif found_rule in dirn_entry['rule_info'][ix][1]: dirn_entry['rule_info'][ix][1] = [found_rule]; break
                                                            
                                                            get_hyperp = None
                                                            for _ in dirn_entry['rule_info'][ix][2]:
                                                                if type(_) == tuple:
                                                                    get_hyperp = _
                                                            for found_rule in len_rule:
                                                                if found_rule == 'hyperp' and get_hyperp is not None: dirn_entry['rule_info'][ix][2] = [get_hyperp]; break
                                                                elif found_rule in dirn_entry['rule_info'][ix][2]: dirn_entry['rule_info'][ix][2] = [found_rule]; break

                                                            
                                                            new_efc.append({'dir':dirn_entry['dir'], 'extmode':dirn_entry['extmode'], 'rule_info':dirn_entry['rule_info']})
                                                        ext_dets['ext_fn_characterisation'] = new_efc

                                                elif state['serial_transform'][stackn]['type'] == 'static':
                                                    
                                                    pass 

                                    except: print('EXCEPTION')


                                
                                
                                
                            return m_states
                        if curr_ref_serial_transform == 'extension': m_states = collapse_multiple_extension_cands(m_states, stackn)

                        def collapse_multiple_connection_cands(m_states, stackn):
                            


                            tempstores = []
                            for state in m_states:
                                if state['type'] == 'analogy':
                                    
                                    if state['serial_transform'][stackn]['type'] == 'connection':
                                        connection_dets = state['serial_param'][stackn]['connection_details']
                                        connection_color_rule = connection_dets['color_rule_info']
                                        tempstores.append(connection_color_rule)
                                    elif state['serial_transform'][stackn]['type'] == 'static':
                                        pass 

                            
                            
                            
                            

                            if len(tempstores)!=0:
                                color_rules = []
                                for tempst in tempstores:
                                    if tempst == []: color_rules.append([None])
                                    else:
                                        colorrule = tempst
                                        temp = []
                                        for opt in colorrule:
                                            if type(opt)==tuple: temp.append('hyperp')
                                            else: temp.append(opt)
                                        color_rules.append(temp)

                                unique_color_rules = []
                                for sublist in color_rules: unique_color_rules.extend(sublist)
                                unique_color_rules = list(set(unique_color_rules)) 

                                
                                found = False; color_rule = None

                                
                                if not found:
                                    for rule in unique_color_rules:
                                        if np.all([rule in sublist for sublist in color_rules]): found = True; color_rule = rule; break

                                
                                if not found and len(unique_color_rules)>=2:
                                    print("ERROR")

                                


                                for state in m_states:
                                    if state['type'] == 'analogy':

                                        if state['serial_transform'][stackn]['type'] == 'connection':
                                            connection_dets = state['serial_param'][stackn]['connection_details']
                                            connection_color_rule = connection_dets['color_rule_info'] 
                                            get_hyperp = None
                                            for _ in connection_color_rule:
                                                if type(_) == tuple:
                                                    get_hyperp = _
                                            connection_dets['color_rule_info'] = [get_hyperp] if color_rule == 'hyperp' else [color_rule]
                                            if color_rule is None: print("WARNING")
                                        elif state['serial_transform'][stackn]['type'] == 'static':
                                            pass 

                            return m_states
                        if curr_ref_serial_transform == 'connection': m_states = collapse_multiple_connection_cands(m_states, stackn)

                        def collapse_multiple_gridwise_bool_simpletype_cands(m_states, stackn):
                            tempstores = []
                            for state in m_states:
                                if state['type'] == 'analogy':
                                    if state['serial_transform'][stackn]['type'] == 'gridwise_bool_simpletype':
                                        bool_opts = state['serial_param'][stackn]['bool_details'] 
                                        tempstores.append(bool_opts)
                                    elif state['serial_transform'][stackn]['type'] == 'static':
                                        pass  

                            if len(tempstores)!=0:
                                
                                chosen = None
                                for opt in tempstores[0]:
                                    flag = True
                                    for tempst in tempstores:
                                        if not is_x_in_y(x=opt, y=tempst):
                                            flag = False; break
                                    if flag:
                                        chosen = opt; break
                                if chosen is not None:
                                    
                                    for state in m_states:
                                        if state['type'] == 'analogy': 
                                            if state['serial_transform'][stackn]['type'] == 'gridwise_bool_simpletype':
                                                state['serial_param'][stackn]['bool_details'] = [chosen] 
                                            elif state['serial_transform'][stackn]['type'] == 'static': pass 
                                            
                            return m_states
                        if curr_ref_serial_transform == 'gridwise_bool_simpletype': m_states = collapse_multiple_gridwise_bool_simpletype_cands(m_states, stackn)

                        def collapse_multiple_gridwise_tiledcopy_cands(m_states, stackn):
                            
                            

                            tempstores = []
                            for state in m_states:
                                if state['type'] == 'analogy':
                                    if state['serial_transform'][stackn]['type'] == 'gridwise_tiled_copy':
                                        tiling_details = state['serial_param'][stackn]['tiling_details']
                                        filtered_options, rmult, cmult, occlusion_mask  = tiling_details['filtered_options'], tiling_details['rmult'], tiling_details['cmult'], tiling_details['occlusion_mask']   
                                        
                                        tiling_details['occlusion_mask'] = 'OVERWRITTEN' 

                                        tempstores.append([int(rmult),int(cmult), filtered_options, occlusion_mask])
                                    elif state['serial_transform'][stackn]['type'] == 'static': 
                                        pass 

                                        
                            

                            if len(tempstores)!=0:
                                

                                for tempstore in tempstores:
                                    opts = tempstore[2]
                                    for rcset in opts:
                                        for currset in rcset:
                                            for currdict in currset:
                                                if currdict['fn'] == globals()['masking']:
                                                    currset.remove(currdict)




                                rcmult_list = [(_[0],_[1]) for _ in tempstores]

                                isdone = False

                                if are_all_identical(rcmult_list):
                                    chosens = {}; numtiles = rcmult_list[0][0]*rcmult_list[0][1]
                                    for rcn in range(numtiles):
                                        opts = [_[2][rcn] for _ in tempstores]
                                        for chosen_opt in opts[0]:
                                            if np.all([is_x_in_y(x=chosen_opt,y=opt) for opt in opts]):
                                                chosens[rcn] = [chosen_opt]
                                                break 
                                    if [k for k in chosens] == list(range(numtiles)):

                                        new_filtered_options = [chosens[k] for k in chosens]
                                        
                                        for state in m_states:
                                            if state['serial_transform'][stackn]['type'] == 'gridwise_tiled_copy':
                                                tiling_details = state['serial_param'][stackn]['tiling_details'] 
                                                tiling_details['filtered_options'] = copy.deepcopy(new_filtered_options)
                                            elif state['serial_transform'][stackn]['type'] == 'static':
                                                pass 

                                        
                                        isdone = True
                                        return m_states

                                
                                
                                unrolled_fns = []
                                for tempstore in tempstores:
                                    opts = tempstore[2]
                                    for rcset in opts:
                                        unrolled_fns.extend(rcset)
                                lbls, uniques = label_unique_with_IDs(unrolled_fns) 
                                found = False; found_set = None
                                if not found: 
                                    for opt1 in uniques:
                                        flag = True
                                        for tempstore in tempstores:
                                            opts = tempstore[2]
                                            for rcset in opts:
                                                if not is_x_in_y(x=opt1,y=rcset): 
                                                    flag = False; break
                                            if flag is False: break
                                        if flag: found = True; found_set = [opt1]
                                if not found: 
                                    tried = []
                                    for o1,opt1 in enumerate(uniques):
                                        for o2, opt2 in enumerate(uniques):
                                            if are_two_identical(opt1,opt2): continue
                                            if is_x_in_y(x=sorted([o1,o2]),y=tried): continue
                                            else: tried.append(sorted([o1,o2]))
                                            flag = True
                                            for tempstore in tempstores:
                                                opts = tempstore[2]
                                                for rcset in opts:
                                                    if not is_x_in_y(x=opt1,y=rcset) and not is_x_in_y(x=opt2,y=rcset): 
                                                        flag = False; break
                                                if flag is False: break
                                            if flag: found = True; found_set = [opt1,opt2]
                                if not found: 
                                    tried = []
                                    for o1,opt1 in enumerate(uniques):
                                        for o2, opt2 in enumerate(uniques):
                                            for o3, opt3 in enumerate(uniques):
                                                if are_all_identical([opt1,opt2,opt3]): continue
                                                if is_x_in_y(x=sorted([o1,o2,o3]),y=tried): continue
                                                else: tried.append(sorted([o1,o2,o3]))
                                                flag = True
                                                for tempstore in tempstores:
                                                    opts = tempstore[2]
                                                    for rcset in opts:
                                                        if not is_x_in_y(x=opt1,y=rcset) and not is_x_in_y(x=opt2,y=rcset) and not is_x_in_y(x=opt3,y=rcset): 
                                                            flag = False; break
                                                    if flag is False: break
                                                if flag: found = True; found_set = [opt1,opt2,opt3]
                                

                                if found:
                                    r0 = []; c0 = 0; arrs0 = []
                                    for tempstore in tempstores:
                                        r1 = []; c1 = 0
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        for rcset in tempstore[2]:
                                            r2 = []
                                            for found_set_opt in found_set:
                                                if is_x_in_y(x=found_set_opt,y=rcset):
                                                    rcset = found_set_opt
                                                    r2.append(rcset)
                                                    
                                                    break
                                            r1.append(r2); c1+=1
                                        
                                        r0.append(r1); c0+=1

                                    
                                    c0 = 0
                                    for state in m_states:
                                        if state['type'] == 'analogy': 
                                            if state['serial_transform'][stackn]['type'] == 'gridwise_tiled_copy':
                                                tiling_details = state['serial_param'][stackn]['tiling_details']
                                                tiling_details['filtered_options'] = r0[c0]
                                                
                                                
                                                c0+=1
                                            elif state['serial_transform'][stackn]['type'] == 'static': pass 
                                        
                                    return m_states    
                            
                                

                                
                                
                            return m_states
                        if curr_ref_serial_transform == 'gridwise_tiled_copy': m_states = collapse_multiple_gridwise_tiledcopy_cands(m_states, stackn)

                        def collapse_multiple_combi_transform_cands(m_states, stackn):

                            tempstores = []
                            for state in m_states:
                                if state['type'] == 'analogy':
                                    if state['serial_transform'][stackn]['type'] == 'combi_transform':
                                        combi_opts = state['serial_param'][stackn]['combi_details']
                                        tempstores.append(combi_opts)
                                    elif state['serial_transform'][stackn]['type'] == 'static':
                                        pass 

                            if len(tempstores)!=0:
                                global aaa
                                aaa = tempstores
                                
                                chosen = None
                                for opt in tempstores[0]['options']:
                                    flag = True
                                    for tempst in tempstores:
                                        if not is_x_in_y(x=opt, y=tempst['options']):
                                            flag = False; break
                                    if flag:
                                        chosen = opt; break
                                if chosen is not None:
                                    
                                    for state in m_states:
                                        if state['type'] == 'analogy': 
                                            if state['serial_transform'][stackn]['type'] == 'combi_transform':
                                                state['serial_param'][stackn]['combi_details'] = {'options':[chosen]} 
                                            elif state['serial_transform'][stackn]['type'] == 'static': pass 

                            return m_states
                        if curr_ref_serial_transform == 'combi_transform': m_states = collapse_multiple_combi_transform_cands(m_states, stackn)





                        
                        if stackn == 0:
                            def obj_select_attempt1(m_states, rule_cands):
                                

         

                                _, analogy_parsing_types = label_unique_with_IDs([state['s']['parsing'] for state in m_states if state['type']=='analogy'])

                                
                                h_states = []
                                for gridn_ in range(num_demo_grids):
                                    for iobj_ in initial_global_parsings[gridn_]['i']:
                                        
                                        flag = False 
                                        for m in range(len(m_states)):
                                            if m_states[m]['type'] == 'analogy':
                                                mgridn, s_obj, s_parsing_description = m_states[m]['gridn'], m_states[m]['s']['obj'], m_states[m]['s']['parsing']          
                                                if mgridn==gridn_ and s_obj==iobj_:
                                                    flag = True; break
                                        
                                        h_states.append({'type':'h_state','gridn':gridn_,'s':{'obj':iobj_,'mask':initial_global_parsings[gridn_]['i'][iobj_]['mask'],'map':initial_global_parsings[gridn_]['i'][iobj_]['map'],'parsing':initial_global_parsings[gridn_]['i'][iobj_]['properties']['parsing_description']},
                                                                        'is_in_m_states_analogy':1 if flag else 0,'is_allowed_parsing':1 if is_x_in_y(x=initial_global_parsings[gridn_]['i'][iobj_]['properties']['parsing_description'],y=analogy_parsing_types) else 0})   

                                h_list = [h for h in range(len(h_states)) if h_states[h]['is_allowed_parsing']==1]
                                h_lbls = [h_states[h]['is_in_m_states_analogy'] for h in range(len(h_states)) if h_states[h]['is_allowed_parsing']==1]

                                for self_prop in [self_color, self_shape, self_parsing, self_color_AND_shape]:
                                    prop_list = self_prop(h_list, h_states)
                                    rule = check_mapping(prop_list, h_lbls)
                                    if rule is not None:
                                        rule_cands['obj_select'].append({'description':'restrict parsings to just these, then apply '+self_prop.__name__+' rule to yield analogy obj selection','parsings_restrict_to':analogy_parsing_types,'prop':self_prop,'rule':rule})


                                
                                
                                return rule_cands

                            def obj_select_attempt2(m_states, rule_cands):

                                
                                
                                h_states = [] 
                                for gridn_ in range(num_demo_grids):
                                    for iobj_ in initial_global_parsings[gridn_]['i']: 
                                        
                                        flag = False 
                                        list_m = None
                                        for m in range(len(m_states)):
                                            if m_states[m]['type'] == 'analogy':
                                                mgridn, s_obj, s_parsing_description = m_states[m]['gridn'], m_states[m]['s']['obj'], m_states[m]['s']['parsing']          
                                                if type(s_obj)==list:
                                                    if mgridn==gridn_ and iobj_ in s_obj:
                                                        flag = True; list_m = m; break
                                                else:
                                                    if mgridn==gridn_ and s_obj==iobj_:
                                                        flag = True; break                                            
                                        
                                        h_states.append({'type':'h_state','gridn':gridn_,'s':{'obj':iobj_,'mask':initial_global_parsings[gridn_]['i'][iobj_]['mask'],'map':initial_global_parsings[gridn_]['i'][iobj_]['map'],'parsing':initial_global_parsings[gridn_]['i'][iobj_]['properties']['parsing_description']},
                                                                        'is_in_m_states_analogy':1 if flag else 0,'list_iobj_m':list_m})


                                h_list = [h for h in range(len(h_states))]
                                h_lbls = [h_states[h]['is_in_m_states_analogy'] for h in range(len(h_states))]


                                
                                listing_rule = None
                                for list_iobj_prop in [gridwise_listgroup]:
                                    res_ = list_iobj_prop(h_states)
                                    if res_ is not None: 
                                        if res_ is True: listing_rule = list_iobj_prop.__name__; break

                                if 1 not in h_lbls: return rule_cands
                                for self_prop in [gridwise_oddcolorout_subframe_iobj, gridwise_largest_containerrect_iobj, gridwise_hyperpbordercolor_containerrect_iobj, 
                                                self_parsing_NONBKG, self_parsing, self_color, self_shape, self_parsing_AND_color, self_parsing_AND_shape,  self_color_AND_shape]:
                                    prop_list = self_prop(h_list, h_states)
                                    rule = check_mapping(prop_list, h_lbls)
                                    if rule is not None:
                                        
                                        rule_cands['obj_select'].append({'description':'objsel apply '+self_prop.__name__+' rule to yield analogy obj selection','prop':self_prop,'rule':rule,'listing_rule':listing_rule})

                                return rule_cands



                            try:
                                rule_cands = obj_select_attempt2(m_states, rule_cands)
                            except: pass
                            


                        
                        if stackn == 0:
                            
                            pass
                        

                        
                        def get_num_analogy_params(m_states, curr_ref_serial_transform, stackn):
                            
                            
                            
                            
                            
                            
                            a_serials_list = [(state['serial_transform'][stackn]['type'],state['serial_param'][stackn]) for state in m_states if state['type']=='analogy']

                            lbls, uniques = label_unique_with_IDs(a_serials_list)
                            return len(uniques), uniques
                        num_analogy_params, unique_trpar_pairs = get_num_analogy_params(m_states, curr_ref_serial_transform, stackn)
                        


                        def reverse_single_param(m_states, first_currstack_rule):
                            global gridn 
                            gridn = gridn_

                            
                            
                            serial_tr, serial_param = first_currstack_rule['tr/par']

                            to_erase = []

                            for state in m_states:
                                if state['type']=='analogy':
                                    curr_map, curr_mask = state['s']['map'], state['s']['mask']
                                    
                                    
                                    curr_fn = globals()[serial_tr]
                                    curr_params = serial_param
                                    curr_map_, curr_mask_ = curr_fn(curr_map, curr_mask,**curr_params)
                                    state['e'] = {'obj':state['s']['obj'],'mask':curr_mask_,'map':curr_map_}


                                    
                            

                            return m_states


                        if num_analogy_params == 1: rule_cands['stackn'+str(stackn)].append({'description':'all the same serial-transform/param','tr/par':unique_trpar_pairs[0],'reverse_fn':'reverse_single_param'})
                        else: 
                        


                            try:


                                
                                def positional_(m_states, rule_cands, tempmode):
                                    
                                    
                                    


                                    a_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']


                                    if tempmode==4: 

                                        

                                        unrolled_a_iobjs = []
                                        for a in a_list:
                                            if type(m_states[a]['s']['obj'])==list: unrolled_a_iobjs.extend(m_states[a]['s']['obj'])
                                            else: unrolled_a_iobjs.append(m_states[a]['s']['obj'])



                                        h_states = [] 
                                        for gridn_ in range(num_demo_grids):
                                            for iobj_ in initial_global_parsings[gridn_]['i']: 
                                                
                                                flag = True if iobj_ in unrolled_a_iobjs else False
                                                
                                                h_states.append({'type':'h_state','gridn':gridn_,'s':{'obj':iobj_,'mask':initial_global_parsings[gridn_]['i'][iobj_]['mask'],'map':initial_global_parsings[gridn_]['i'][iobj_]['map'],'parsing':initial_global_parsings[gridn_]['i'][iobj_]['properties']['parsing_description']},
                                                                                'is_in_m_states_analogy':1 if flag else 0})

                                        h_list = [h for h in range(len(h_states))]
                                        h_lbls = [h_states[h]['is_in_m_states_analogy'] for h in range(len(h_states))]


                                        for self_prop in [gridwise_oddcolorout_subframe_iobj, gridwise_largest_containerrect_iobj, gridwise_hyperpbordercolor_containerrect_iobj, self_parsing, self_color, self_shape, self_parsing_AND_color, self_parsing_AND_shape,  self_color_AND_shape]:
                                            prop_list = self_prop(h_list, h_states)
                                            rule = check_mapping(prop_list, h_lbls)
                                            if rule is not None:
                                                rule_cands['obj_select'].append({'description':'objsel apply '+self_prop.__name__+' rule to yield analogy obj selection','parsing_objsel':False,'prop':self_prop,'rule':rule})


                                        

                                        for state in m_states:
                                            if state['type']=='analogy':
                                                details = state['serial_param'][stackn]['details']
                                                bkg_rule = details['bkg_dets'] 
                                                frame_specs = details['frame_specs']
                                                break

                                        vb, vf, hb, hf = 0,0,0,0
                                        if frame_specs[0]==frame_specs[1][-1]: vb = frame_specs[0]
                                        if len(frame_specs[1])>=2 and are_all_identical(frame_specs[1][:-1]): vf = frame_specs[1][0]
                                        if frame_specs[2]==frame_specs[3][-1]: hb = frame_specs[2]
                                        if len(frame_specs[3])>=2 and are_all_identical(frame_specs[3][:-1]): hf = frame_specs[3][0]            
                                        frame_rules = {'vert_border':vb,'vert_frame':vf,'horiz_border':hb,'horiz_frame':hf} 

                                        
                                        positional_rule = 'direct_tiling_pattern'

                                        

                                        cchange_flag = True
                                        a_tilings = []
                                        for state in m_states:
                                            if state['type']=='analogy':
                                                details = state['serial_param'][stackn]['details']
                                                n_rows, n_cols, new_colors = details['n_rows'], details['n_cols'], []
                                                for _ in details['tiles']:
                                                    if _['color_changes'] is not None: new_colors.append(_['color_changes'][0][1])
                                                    else: cchange_flag = False
                                                if not cchange_flag: break 
                                                tiling = np.zeros((n_rows, n_cols)); k=0
                                                for r in range(n_rows):
                                                    for c in range(n_cols):
                                                        tiling[r,c] = new_colors[k]
                                                        k+=1
                                                a_tilings.append(tiling)          
                                        



                                    if tempmode==4 and cchange_flag is False:
                                        tempmode = 5 

                                        rule_cands['stackn'+str(stackn)].append({'description':'temp4 simplest selection', 'reverse_fn':'reverse_positional_coloring',
                                                                                            'positional_rule':positional_rule, 'frame_rules':frame_rules, 'bkg_rule':bkg_rule, 'mode':tempmode})

                                        return rule_cands 



                                    gridn_m_dict = {}
                                    for _,m in enumerate(a_list):
                                        if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                                        gridn_m_dict[m_states[m]['gridn']].append(m)


                                    positional_matches = {}
                                    for gridn in gridn_m_dict:
                                        positional_matches[gridn] = []


                                        for m in gridn_m_dict[gridn]:
                                            
                                            
                                            
                                            
                                            if tempmode==1: 
                                                matches = []
                                                end_color = get_colors_of_obj(m_states[m]['e']['mask'],m_states[m]['e']['map']) 
                                                for n in list(range(len(m_states))):
                                                    if m==n: continue
                                                    if m_states[n]['gridn']!=gridn: continue
                                                    obj_s_color = get_colors_of_obj(m_states[n]['s']['mask'],m_states[n]['s']['map'])
                                                    if are_two_identical(obj_s_color, end_color):
                                                        matches.append(n)
                                                positional_matches[gridn].append((m,matches[0]))

                                            if tempmode in [2,3]: 
                                                subobjs = []
                                                rows,cols = np.where(m_states[m]['e']['mask']==1)
                                                for p in range(len(rows)):
                                                    minimatches = []
                                                    end_color = m_states[m]['e']['map'][rows[p],cols[p]]

                                                    if tempmode==3:
                                                        p_corresp_to_bkg = False 
                                                        for n in [n for n in range(len(m_states)) if n!=m and m_states[n]['gridn']==gridn]:
                                                            if m_states[n]['s']['parsing'][0] in ['background'] and end_color == get_colors_of_obj(m_states[n]['s']['mask'],m_states[n]['s']['map']):
                                                                p_corresp_to_bkg = True
                                                        if p_corresp_to_bkg: continue 

                                                    subobjs.append({'centroid':(rows[p],cols[p]),'color':[end_color]})
                                                    for n in list(range(len(m_states))):
                                                        if m==n: continue
                                                        if m_states[n]['gridn']!=gridn: continue
                                                        obj_s_color = get_colors_of_obj(m_states[n]['s']['mask'],m_states[n]['s']['map'])
                                                        if obj_s_color == [end_color]:
                                                            minimatches.append(n)
                                                    positional_matches[gridn].append((m,p,minimatches))


                                            if tempmode==4:
                                                subobjs = []
                                                curr_tiling = a_tilings[a_list.index(m)]
                                                rows,cols = np.where(np.ones_like(curr_tiling)==1)
                                                for p in range(len(rows)):
                                                    minimatches = []
                                                    end_color = curr_tiling[rows[p],cols[p]]
                                                    subobjs.append({'centroid':(rows[p],cols[p]),'color':[end_color]})
                                                    for n in list(range(len(m_states))): 
                                                        if m==n: continue
                                                        if m_states[n]['gridn']!=gridn: continue
                                                        obj_s_color = get_colors_of_obj(m_states[n]['s']['mask'],m_states[n]['s']['map'])
                                                        
                                                        if obj_s_color == [end_color]:
                                                            minimatches.append(n)
                                                    positional_matches[gridn].append((m,p,minimatches))

                                    

                                    flag_pos = True; i_allocs_over_gridns = {}
                                    for gridn in gridn_m_dict:
                                        
                                        
                                        
                                        
                                        i_ms = []
                                        for triple in positional_matches[gridn]:
                                            if len(triple)==3: i_ms.extend(triple[2])
                                            if len(triple)==2: i_ms.append(triple[1])
                                        i_ms = list(set(i_ms))
                                        o_ms = []
                                        for triple in positional_matches[gridn]:
                                            if len(triple)==3: o_ms.append((triple[0],triple[1]))                       
                                            if len(triple)==2: o_ms.append(triple[0])                
                                        o_ms = list(set(o_ms))


                                        def get_adjusted_centroids(ms, m_states, s_e, ms_type):
                                            if ms_type == 'm':
                                                centroids = []
                                                for m in ms:
                                                    r,c = np.nonzero(m_states[m][s_e]['mask'])
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)
                                            elif ms_type == 'm/p':
                                                centroids = []
                                                for m,p in ms:                            
                                                    if tempmode in [1,2,3]:
                                                        mask_ = m_states[m][s_e]['mask']
                                                        rows,cols = np.where(mask_==1)
                                                    elif tempmode == 4:
                                                        curr_tiling = a_tilings[a_list.index(m)]
                                                        rows,cols = np.where(np.ones_like(curr_tiling)==1)
                                                    r,c = rows[p], cols[p]
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)                                
                                            
                                            maxr, minr, maxc, minc = np.max([_[0] for _ in centroids]), np.min([_[0] for _ in centroids]), np.max([_[1] for _ in centroids]), np.min([_[1] for _ in centroids]) 
                                            adjusted = []
                                            for centroid in centroids:
                                                if (maxr-minr)==0 and (maxc-minc)==0: adjusted.append((0,0))
                                                elif (maxr-minr)==0: adjusted.append((0, (centroid[1]-minc)/(maxc-minc)))
                                                elif (maxc-minc)==0: adjusted.append(((centroid[0]-minr)/(maxr-minr), 0))
                                                else: adjusted.append(((centroid[0]-minr)/(maxr-minr), (centroid[1]-minc)/(maxc-minc)))
                                            return adjusted

                                        if tempmode==1:
                                            i_centroids = get_adjusted_centroids(i_ms, m_states,'s','m')
                                            o_centroids = get_adjusted_centroids(o_ms, m_states,'s','m')
                                        if tempmode in [2,3,4]:
                                            i_centroids = get_adjusted_centroids(i_ms, m_states,'s','m')
                                            o_centroids = get_adjusted_centroids(o_ms, m_states,'e','m/p')    

                                        
                                        i_slots = [1]*len(i_centroids); allocs = []
                                        for o in range(len(o_ms)): 
                                            dists = [np.sqrt((o_centroids[o][0]-i_centroids[i][0])**2+(o_centroids[o][1]-i_centroids[i][1])**2) for i in range(len(i_slots)) if i_slots[i]==1]
                                            if len(dists)==0: continue
                                            available_i = [i for i in range(len(i_slots)) if i_slots[i]==1]
                                            chosen_i = available_i[np.argmin(dists)]
                                            allocs.append((i_ms[chosen_i],o_ms[o]))
                                            i_slots[chosen_i] = 0 

                                        for i_alloc, o_mp in allocs:
                                            for triple in positional_matches[gridn]:
                                                if o_mp == (triple[0],triple[1]):
                                                    if i_alloc in triple[2]: pass
                                                    else: flag_pos = False
                                                    break

                                        i_allocs_over_gridns[gridn] = [i_alloc for i_alloc, o_mp in allocs]
                                    
                                    
                                    
                                    if flag_pos:
                                        

                                        all_i_objs = []
                                        for gridn in gridn_m_dict: all_i_objs.extend(i_allocs_over_gridns[gridn]) 
                                            
                                        objsel_lbls = []
                                        for m in range(len(m_states)):
                                            if m in all_i_objs: objsel_lbls.append(1)
                                            else: objsel_lbls.append(0)
                                        
                                        
                                        for self_prop in [self_parsing, self_shape, self_parsing_AND_shape]: 
                                            
                                            
                                            prop_list = self_prop(list(range(len(m_states))), m_states)
                                            
                                            rule = check_mapping(prop_list, objsel_lbls)
                                            if rule is not None:
                                                
                                                
                                                

                                                if tempmode in [1,2,3]:
                                                    rule_cands['stackn'+str(stackn)].append({'description':'color based positional rule with obj selection by'+self_prop.__name__,
                                                                'prop':self_prop,'rule':rule,'reverse_fn':'reverse_positional_coloring','mode':tempmode})

                                                if tempmode==4:
                                                    rule_cands['stackn'+str(stackn)].append({'description':'color based positional rule with obj selection by'+self_prop.__name__,
                                                                                            'prop':self_prop,'rule':rule,'reverse_fn':'reverse_positional_coloring','mode':tempmode,
                                                                                            'positional_rule':positional_rule, 'frame_rules':frame_rules, 'bkg_rule':bkg_rule})
                                                    
                                    return rule_cands
                                
                                def reverse_positional_coloring_(m_states, first_currstack_rule):
                                    global gridn 
                                    gridn = gridn_



                                    
                                    mode = first_currstack_rule['mode']



                                    m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                    
                                    
                                    




                                    if mode in [1,2,3]:
                                        rt, M, d = first_currstack_rule['rule']
                                        interact_prop = first_currstack_rule['prop']
                                        i_cand_list = [m for m in range(len(m_states)) if m_states[m]['type'] in ['other','h_state'] and m_states[m]['gridn']==gridn]
                                        
                                        prop_list = interact_prop(i_cand_list, m_states)
                                        param_list = reverse_mapping(prop_list, M, d, rt)
                                        i_list = []
                                        for k in range(len(i_cand_list)): 
                                            if param_list[k]==1:
                                                i_list.append(i_cand_list[k]) 
                                        i_ms = i_list

                                    if mode == 4:
                                        
                                        prop = first_currstack_rule['prop']
                                        rule = first_currstack_rule['rule']
                                        rt, M, d = rule
                                        prop_list = prop(list(range(len(m_states))), m_states)
                                        
                                        param_list = reverse_mapping(prop_list, M, d, rt)
                                        interact_m_list = [m for m in range(len(m_states)) if param_list[m]==1]
                                        i_ms = interact_m_list


                                    if mode == 5:
                                        i_ms = m_list




                    

                                    if mode in [2,3,4,5]: 

                                        
                                        def get_adjusted_centroids(ms, m_states, s_e, ms_type):
                                            if ms_type == 'm':
                                                centroids = []
                                                for m in ms:
                                                    r,c = np.nonzero(m_states[m][s_e]['mask'])
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)
                                            elif ms_type == 'm/p':
                                                centroids = []
                                                for m,p in ms:                            
                                                    mask_ = m_states[m][s_e]['mask']
                                                    rows,cols = np.where(mask_==1)
                                                    r,c = rows[p], cols[p]
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)                                
                                            
                                            maxr, minr, maxc, minc = np.max([_[0] for _ in centroids]), np.min([_[0] for _ in centroids]), np.max([_[1] for _ in centroids]), np.min([_[1] for _ in centroids]) 
                                            adjusted = []
                                            for centroid in centroids:
                                                if (maxr-minr)==0 and (maxc-minc)==0: adjusted.append((0,0))
                                                elif (maxr-minr)==0: adjusted.append((0, (centroid[1]-minc)/(maxc-minc)))
                                                elif (maxc-minc)==0: adjusted.append(((centroid[0]-minr)/(maxr-minr), 0))
                                                else: adjusted.append(((centroid[0]-minr)/(maxr-minr), (centroid[1]-minc)/(maxc-minc)))
                                            return adjusted
                                        i_centroids = get_adjusted_centroids(i_ms, m_states,'s','m')
                                        def make_o_centroids_o_ms(rs,cs):
                                            o_ms = list(range(rs*cs))
                                            o_centroids = []
                                            for r in range(rs):
                                                for c in range(cs):
                                                    if rs==1 and cs==1: o_centroids.append((0,0))
                                                    elif rs==1: o_centroids.append((0,c/(cs-1)))
                                                    elif cs==1: o_centroids.append((r/(rs-1),0))
                                                    else: o_centroids.append((r/(rs-1),c/(cs-1)))
                                            return o_centroids, o_ms
                                        cands_ = []
                                        for rs in range(1,len(i_centroids)+1):
                                            for cs in range(1,len(i_centroids)+1):
                                                if rs*cs == len(i_centroids):

                                                    o_centroids, o_ms = make_o_centroids_o_ms(rs,cs)

                                                    i_slots = [1]*len(i_centroids); allocs = []; rallocs = {}; chosen_dists = []; flag = True
                                                    for o in range(len(o_ms)): 
                                                        dists = [np.sqrt((o_centroids[o][0]-i_centroids[i][0])**2+(o_centroids[o][1]-i_centroids[i][1])**2) for i in range(len(i_slots)) if i_slots[i]==1]
                                                        if len(dists)==0: flag = False; break
                                                        available_i = [i for i in range(len(i_slots)) if i_slots[i]==1]
                                                        chosen_i = available_i[np.argmin(dists)]; chosen_dists.append(np.min(dists))
                                                        allocs.append((i_ms[chosen_i],o_ms[o])); rallocs[o_ms[o]] = i_ms[chosen_i]
                                                        i_slots[chosen_i] = 0 
                                                    
                                                    if flag:
                                                        
                                                        cands_.append([np.mean(chosen_dists), rs, cs, rallocs])
                                        cands__ = sorted(cands_)
                                        
                                        rs, cs, rallocs = cands__[0][1], cands__[0][2], cands__[0][3]




                                    if mode == 1:
                                        if len(i_list) != len(m_list): print("ERROR")
                                        o_ms = m_list

                                        def get_adjusted_centroids(ms, m_states, s_e, ms_type):
                                            if ms_type == 'm':
                                                centroids = []
                                                for m in ms:
                                                    r,c = np.nonzero(m_states[m][s_e]['mask'])
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)
                                            elif ms_type == 'm/p':
                                                centroids = []
                                                for m,p in ms:                            
                                                    mask_ = m_states[m][s_e]['mask']
                                                    rows,cols = np.where(mask_==1)
                                                    r,c = rows[p], cols[p]
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)                                
                                            
                                            maxr, minr, maxc, minc = np.max([_[0] for _ in centroids]), np.min([_[0] for _ in centroids]), np.max([_[1] for _ in centroids]), np.min([_[1] for _ in centroids]) 
                                            adjusted = []
                                            for centroid in centroids:
                                                if (maxr-minr)==0 and (maxc-minc)==0: adjusted.append((0,0))
                                                elif (maxr-minr)==0: adjusted.append((0, (centroid[1]-minc)/(maxc-minc)))
                                                elif (maxc-minc)==0: adjusted.append(((centroid[0]-minr)/(maxr-minr), 0))
                                                else: adjusted.append(((centroid[0]-minr)/(maxr-minr), (centroid[1]-minc)/(maxc-minc)))
                                            return adjusted
                                        i_centroids = get_adjusted_centroids(i_ms, m_states,'s','m')
                                        o_centroids = get_adjusted_centroids(o_ms, m_states,'s','m')
                                        i_slots = [1]*len(i_centroids); allocs = []; rallocs = {}
                                        for o in range(len(o_ms)): 
                                            dists = [np.sqrt((o_centroids[o][0]-i_centroids[i][0])**2+(o_centroids[o][1]-i_centroids[i][1])**2) for i in range(len(i_slots)) if i_slots[i]==1]
                                            available_i = [i for i in range(len(i_slots)) if i_slots[i]==1]
                                            chosen_i = available_i[np.argmin(dists)]
                                            allocs.append((i_ms[chosen_i],o_ms[o])); rallocs[o_ms[o]] = i_ms[chosen_i]
                                            i_slots[chosen_i] = 0 

                                        
                                        for m in m_list:
                                            state = m_states[m]
                                            curr_map, curr_mask = state['s']['map'], state['s']['mask']

                                            og_color = get_colors_of_obj(curr_mask, curr_map)
                                            corresp_alloc = rallocs[m]
                                            istate = m_states[corresp_alloc]
                                            icurr_map, icurr_mask = istate['s']['map'], istate['s']['mask']
                                            corresp_color = get_colors_of_obj(icurr_mask, icurr_map)

                                            curr_fn = globals()['recolor']
                                            curr_params = {'color_changes':[[og_color, corresp_color]]}
                                            curr_map, curr_mask = curr_fn(curr_map, curr_mask,**curr_params)
                                            state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map}     











                                    if mode == 2:

                                        
                                        curr_mask = curr_map = np.zeros((rs,cs))
                                        c0=0
                                        for r in range(rs):
                                            for c in range(cs):
                                                corresp_alloc_i = rallocs[c0]
                                                istate = m_states[corresp_alloc_i]
                                                icurr_map, icurr_mask = istate['s']['map'], istate['s']['mask']
                                                corresp_color = get_colors_of_obj(icurr_mask, icurr_map)
                                                
                                                curr_mask[r,c] = 1
                                                curr_map[r,c] = int(corresp_color[0])
                                                c0+=1
                                        

                                        new_state = {'r_':None,'type':'analogy','gridn':gridn,'e':{'obj':'newn','mask':curr_mask,'map':curr_map,'parsing':'created_so_specialk'}}
                                        m_states.append(new_state)
                                        
                                    if mode == 3:

                                        
                                        
                                        bkg_color_ = 0 
                                        for m_ in i_cand_list:
                                            if m_states[m_]['s']['parsing'][0] in ['background']:
                                                bkg_color_ = m_states[m_]['s']['map'][0,0]

                                        for o in range(len(o_ms)):
                                            if o_ms[o] not in rallocs:
                                                rallocs[o_ms[o]] = bkg_color_



                                        curr_mask = curr_map = np.zeros((rs,cs))
                                        c0=0
                                        for r in range(rs):
                                            for c in range(cs):
                                                corresp_alloc_i = rallocs[c0]
                                                istate = m_states[corresp_alloc_i]
                                                icurr_map, icurr_mask = istate['s']['map'], istate['s']['mask']
                                                corresp_color = get_colors_of_obj(icurr_mask, icurr_map)
                                                
                                                curr_mask[r,c] = 1
                                                curr_map[r,c] = int(corresp_color[0])
                                                c0+=1
                                        

                                        new_state = {'r_':None,'type':'analogy','gridn':gridn,'e':{'obj':'newn','mask':curr_mask,'map':curr_map,'parsing':'created_so_specialk'}}
                                        m_states.append(new_state)

                                    if mode in [4,5]: 

                                        n_rows = rs
                                        n_cols = cs

                                        k=0; allocated_ms = []
                                        for r in range(rs):
                                            for c in range(cs):
                                                
                                                allocated_ms.append([rallocs[k], r, c])
                                                k+=1


                                        bkg_rule = first_currstack_rule['bkg_rule']
                                        positional_rule = first_currstack_rule['positional_rule']
                                        frame_rules = first_currstack_rule['frame_rules']


                                        
                                        try: 
                                            bkg_color = bkg_rule['bkg_color'] 
                                            if bkg_rule['bkg_mtd'] == 'choose_hyperp_color': bkg_color = bkg_rule['bkg_color']
                                            elif bkg_rule['bkg_mtd'] == 'actual_bkg_objs_color':
                                                for m in range(len(m_states)):
                                                    
                                                    if m_states[m]['s']['parsing'][0] == 'background':
                                                        bkg_color = get_colors_of_obj(m_states[m]['s']['mask'],m_states[m]['s']['map'])
                                                        break
                                            elif bkg_rule['bkg_mtd'] == 'choose_iframe_color':
                                                for m in range(len(m_states)):
                                                    if m_states[m]['s']['parsing'][0] == 'frame_iobj':
                                                        bkg_color = get_colors_of_obj(m_states[m]['s']['mask'],m_states[m]['s']['map'])
                                                        break                        
                                            elif bkg_rule['bkg_mtd'] == 'choose_common_color':
                                                subframe_iobjs = []
                                                for m in range(len(m_states)):
                                                    if m_states[m]['s']['parsing'][0] == 'subframe_iobj':
                                                        subframe_iobjs.append(get_colors_of_obj(m_states[m]['s']['mask'],m_states[m]['s']['map']))
                                                for colr in subframe_iobjs[0]:
                                                    if np.all([is_x_in_y(x=colr,y=_) for _ in subframe_iobjs]):
                                                        bkg_color = colr
                                                        break
                                            
                                        except: bkg_color = 0  



                                        
                                        r_prepend = frame_rules['vert_border'] 
                                        r_append = [frame_rules['vert_frame']]*(n_rows-1) + [frame_rules['vert_border']]
                                        c_prepend = frame_rules['horiz_border']
                                        c_append = [frame_rules['horiz_frame']]*(n_cols-1) + [frame_rules['horiz_border']]
                                        
                                        frame_specs = [c_prepend, c_append, r_prepend, r_append] 

                                    if mode == 4: 

                                        
                        
                                        
                                        
                                        i_mask = []; i_map = []; iobjs = []; tiles = []; safe_tile_ixs = []
                                        for k in range(len(allocated_ms)):
                                            if k==0: actual_m = m_list[0] 
                                            interact_m = allocated_ms[k][0]
                                            new_colors = get_colors_of_obj(m_states[interact_m]['s']['mask'], m_states[interact_m]['s']['map'])

                                            iobj = m_states[actual_m]['s']['obj']
                                            mask_, map_ = m_states[actual_m]['s']['mask'], m_states[actual_m]['s']['map']
                                            original_colors = get_colors_of_obj(mask_, map_)

                                            if k==0: iobjs.append(iobj); i_mask.append(mask_); i_map.append(map_)
                                            tiles.append({'iobj':iobj,'color_changes':[[original_colors[0], new_colors[0]]]}) 
                                            safe_tile_ixs.append(iobjs.index(iobj)) 


                                        details = {'details':{'n_rows': n_rows, 'n_cols': n_cols,
                                            'safe_tile_ixs': safe_tile_ixs,
                                            'tiles': tiles,
                                            'frame_specs': frame_specs,
                                            'bkg_dets': {'bkg_mtd': bkg_rule['bkg_mtd'] if bkg_rule is not None else 'actual_bkg_objs_color',
                                            'bkg_color': bkg_color}}}

                                        serial_tr =  'iobjs_tile_creation'

                                        state = m_states[m]
                                        
                                        curr_fn = globals()[serial_tr]
                                        curr_params = details
                                        curr_map, curr_mask = curr_fn(i_map, i_mask,**curr_params)
                                        


                                        new_state = {'r_':None,'type':'analogy','gridn':gridn,'e':{'obj':'bewn','mask':curr_mask,'map':curr_map,'parsing':'created_so_specialR'}}
                                        m_states.append(new_state)

                                    if mode == 5: 


                                        
                                        i_mask = []; i_map = []; iobjs = []; tiles = []; safe_tile_ixs = []
                                        for k in range(len(allocated_ms)):
                                            m = allocated_ms[k][0]
                                            iobj = m_states[m]['s']['obj']
                                            mask_, map_ = m_states[m]['s']['mask'], m_states[m]['s']['map']
                                            iobjs.append(iobj); i_mask.append(mask_); i_map.append(map_)
                                            tiles.append({'iobj':iobj,'color_changes':None})
                                            safe_tile_ixs.append(iobjs.index(iobj))


                                        details = {'details':{'n_rows': n_rows, 'n_cols': n_cols,
                                            'safe_tile_ixs': safe_tile_ixs,
                                            'tiles': tiles,
                                            'frame_specs': frame_specs,
                                            'bkg_dets': {'bkg_mtd': bkg_rule['bkg_mtd'] if bkg_rule is not None else 'actual_bkg_objs_color',
                                            'bkg_color': bkg_color}}}

                                        serial_tr =  'iobjs_tile_creation'

                                        state = m_states[m]
                                        
                                        curr_fn = globals()[serial_tr]
                                        curr_params = details
                                        curr_map, curr_mask = curr_fn(i_map, i_mask,**curr_params)
                                        


                                        new_state = {'r_':None,'type':'analogy','gridn':gridn,'e':{'obj':'bewn','mask':curr_mask,'map':curr_map,'parsing':'created_so_specialR'}}
                                        m_states.append(new_state)


                                    return m_states

                                def positional(m_states, rule_cands, tempmode):


                                    a_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']


                                    if tempmode==4: 

                                        

                                        for state in m_states:
                                            if state['type']=='analogy':
                                                details = state['serial_param'][stackn]['details']
                                                bkg_rule = details['bkg_dets'] 
                                                frame_specs = details['frame_specs']
                                                break

                                        vb, vf, hb, hf = 0,0,0,0
                                        if frame_specs[0]==frame_specs[1][-1]: vb = frame_specs[0]
                                        if len(frame_specs[1])>=2 and are_all_identical(frame_specs[1][:-1]): vf = frame_specs[1][0]
                                        if frame_specs[2]==frame_specs[3][-1]: hb = frame_specs[2]
                                        if len(frame_specs[3])>=2 and are_all_identical(frame_specs[3][:-1]): hf = frame_specs[3][0]            
                                        frame_rules = {'vert_border':vb,'vert_frame':vf,'horiz_border':hb,'horiz_frame':hf} 

                                        
                                        positional_rule = 'direct_tiling_pattern'

                                        



                                        

                                        cchange_flag = True
                                        a_tilings = []
                                        for state in m_states:
                                            if state['type']=='analogy':
                                                details = state['serial_param'][stackn]['details']
                                                n_rows, n_cols, new_colors = details['n_rows'], details['n_cols'], []
                                                for _ in details['tiles']:
                                                    if _['color_changes'] is not None: new_colors.append(_['color_changes'][0][1])
                                                    else: cchange_flag = False
                                                if not cchange_flag: break 
                                                tiling = np.zeros((n_rows, n_cols)); k=0
                                                for r in range(n_rows):
                                                    for c in range(n_cols):
                                                        tiling[r,c] = new_colors[k]
                                                        k+=1
                                                a_tilings.append(tiling)          
                                        

                                    if tempmode==4 and cchange_flag is False:
                                        tempmode = 5 

                                        rule_cands['stackn'+str(stackn)].append({'description':'temp4 simplest selection', 'reverse_fn':'reverse_positional_coloring',
                                                                                            'positional_rule':positional_rule, 'frame_rules':frame_rules, 'bkg_rule':bkg_rule, 'mode':tempmode})
                                        return rule_cands 



                                    gridn_m_dict = {}
                                    for _,m in enumerate(a_list):
                                        if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                                        gridn_m_dict[m_states[m]['gridn']].append(m)


                                    if tempmode in [2,3,  4,6]: 
                                        
                                        chosen_opt = None
                                        for objsel_by in ['parsing_And_shape','parsing']:

                                            positional_matches = {}; flag_pos = True
                                            for gridn in gridn_m_dict:
                                                positional_matches[gridn] = []                                    

                                                bkg_colrs = [get_colors_of_obj(initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map']) for iobj in get_iobjs_of_parsing_type('background', initial_global_parsings, gridn, 'i')]

                                                for m in gridn_m_dict[gridn]: 
                                                    
                                                    
                                                    if tempmode in [2,3]: rows,cols = np.where(m_states[m]['e']['mask']==1)
                                                    if tempmode in [4,6]: curr_tiling = a_tilings[a_list.index(m)]; rows,cols = np.where(np.ones_like(curr_tiling)==1)

                                                    for p in range(len(rows)):
                                                        minimatches = []
                                                        if tempmode in [2,3]: end_color = m_states[m]['e']['map'][rows[p],cols[p]]
                                                        if tempmode in [4,6]: end_color = curr_tiling[rows[p],cols[p]]
                                                        
                                                        if tempmode in [3,6] and [end_color] in bkg_colrs: continue 

                                                        for iobj in global_parsings[gridn]['i']:
                                                            i_mask, i_map, i_maskv, i_masko, i_obj_parsing_type = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['maskv'], global_parsings[gridn]['i'][iobj]['masko'], global_parsings[gridn]['i'][iobj]['properties']['parsing_description']
                                                            colr = get_colors_of_obj(i_mask, i_map)
                                                            shape = get_shape_of_obj(i_mask, i_map)
                                                            r,c = np.nonzero(i_mask)
                                                            centroid = (r.mean(), c.mean())
                                                            if len(colr)==1: 
                                                                if colr == [end_color]:
                                                                    minimatches.append([iobj, i_obj_parsing_type, centroid, colr, shape])
                                                        positional_matches[gridn].append((m,p,minimatches))

                                            if objsel_by == 'parsing_And_shape':
                                                g0 = [k for k in gridn_m_dict][0]
                                                first_opts = [(_[1],_[4]) for _ in positional_matches[g0][0][2]] 
                                                for opt in first_opts:
                                                    flag = True
                                                    for gridn in gridn_m_dict:
                                                        for mp in range(len(positional_matches[gridn])):
                                                            curr_opts = [(_[1],_[4]) for _ in positional_matches[gridn][mp][2]]
                                                            if not is_x_in_y(x=opt, y=curr_opts):
                                                                flag = False
                                                    if flag: chosen_opt = opt; break 

                                            if objsel_by == 'parsing':
                                                g0 = [k for k in gridn_m_dict][0]
                                                first_opts = [_[1] for _ in positional_matches[g0][0][2]] 
                                                for opt in first_opts:
                                                    flag = True
                                                    for gridn in gridn_m_dict:
                                                        for mp in range(len(positional_matches[gridn])):
                                                            curr_opts = [_[1] for _ in positional_matches[gridn][mp][2]]
                                                            if not is_x_in_y(x=opt, y=curr_opts):
                                                                flag = False
                                                    if flag: chosen_opt = opt; break 


                                            if chosen_opt is None: flag_pos = False; continue


                                            for gridn in gridn_m_dict:

                                                def get_adjusted_centroids(ms, m_states, s_e, ms_type):
                                                    if ms_type == 'm':
                                                        centroids = []
                                                        for m in ms:
                                                            r,c = np.nonzero(m_states[m][s_e]['mask'])
                                                            centroid = (r.mean(), c.mean())
                                                            centroids.append(centroid)
                                                    elif ms_type == 'm/p':
                                                        centroids = []
                                                        for m,p in ms:                            
                                                            if tempmode in [1,2,3]:
                                                                mask_ = m_states[m][s_e]['mask']
                                                                rows,cols = np.where(mask_==1)
                                                            elif tempmode in [4,6]:
                                                                curr_tiling = a_tilings[a_list.index(m)]
                                                                rows,cols = np.where(np.ones_like(curr_tiling)==1)
                                                            r,c = rows[p], cols[p]
                                                            centroid = (r.mean(), c.mean())
                                                            centroids.append(centroid)                                
                                                    
                                                    maxr, minr, maxc, minc = np.max([_[0] for _ in centroids]), np.min([_[0] for _ in centroids]), np.max([_[1] for _ in centroids]), np.min([_[1] for _ in centroids]) 
                                                    adjusted = []
                                                    for centroid in centroids:
                                                        if (maxr-minr)==0 and (maxc-minc)==0: adjusted.append((0,0))
                                                        elif (maxr-minr)==0: adjusted.append((0, (centroid[1]-minc)/(maxc-minc)))
                                                        elif (maxc-minc)==0: adjusted.append(((centroid[0]-minr)/(maxr-minr), 0))
                                                        else: adjusted.append(((centroid[0]-minr)/(maxr-minr), (centroid[1]-minc)/(maxc-minc)))
                                                    return adjusted

                                                def adjusted_centroids(centroids):
                                                    
                                                    maxr, minr, maxc, minc = np.max([_[0] for _ in centroids]), np.min([_[0] for _ in centroids]), np.max([_[1] for _ in centroids]), np.min([_[1] for _ in centroids]) 
                                                    adjusted = []
                                                    for centroid in centroids:
                                                        if (maxr-minr)==0 and (maxc-minc)==0: adjusted.append((0,0))
                                                        elif (maxr-minr)==0: adjusted.append((0, (centroid[1]-minc)/(maxc-minc)))
                                                        elif (maxc-minc)==0: adjusted.append(((centroid[0]-minr)/(maxr-minr), 0))
                                                        else: adjusted.append(((centroid[0]-minr)/(maxr-minr), (centroid[1]-minc)/(maxc-minc)))
                                                    return adjusted                                    

                                                
                                                i_centroids_raw = []
                                                o_mps = []
                                                for mp in range(len(positional_matches[gridn])):
                                                    o_mps.append((positional_matches[gridn][mp][0],positional_matches[gridn][mp][1]))
                                                    for k_ in range(len(positional_matches[gridn][mp][2])):
                                                        if objsel_by == 'parsing':
                                                            if are_two_identical( chosen_opt, positional_matches[gridn][mp][2][k_][1]):
                                                                curr_centroid = positional_matches[gridn][mp][2][k_][2]
                                                                if not is_x_in_y(curr_centroid, i_centroids_raw): i_centroids_raw.append(curr_centroid)
                                                        if objsel_by == 'parsing_And_shape':
                                                            if are_two_identical( chosen_opt, (positional_matches[gridn][mp][2][k_][1],positional_matches[gridn][mp][2][k_][4])):
                                                                curr_centroid = positional_matches[gridn][mp][2][k_][2]
                                                                if not is_x_in_y(curr_centroid, i_centroids_raw): i_centroids_raw.append(curr_centroid)

                                                
                                                

                                                if tempmode==1:
                                                    i_centroids = adjusted_centroids(i_centroids_raw)
                                                    o_centroids = get_adjusted_centroids(o_mps, m_states,'s','m')
                                                if tempmode in [2,3,4,6]:
                                                    i_centroids = adjusted_centroids(i_centroids_raw)
                                                    o_centroids = get_adjusted_centroids(o_mps, m_states,'e','m/p')    



                                                if len(i_centroids)!=len(o_centroids): flag_pos = False; break

                                                
                                                i_slots = [1]*len(i_centroids); allocs = []
                                                for o in range(len(o_mps)): 
                                                    dists = [np.sqrt((o_centroids[o][0]-i_centroids[i][0])**2+(o_centroids[o][1]-i_centroids[i][1])**2) for i in range(len(i_slots)) if i_slots[i]==1]
                                                    if len(dists)==0: continue
                                                    available_i = [i for i in range(len(i_slots)) if i_slots[i]==1]
                                                    chosen_i = available_i[np.argmin(dists)]
                                                    allocs.append(( chosen_i, o_mps[o])) 
                                                    i_slots[chosen_i] = 0 


                                                if 1 in i_slots: flag_pos = False; break
                                                
                                                

                                            if flag_pos: 
                                                

                                                if tempmode in [2,3]:

                                                    rule_cands['stackn'+str(stackn)].append({'description':'color based positional rule with obj selection by'+objsel_by,
                                                                'objsel_by':objsel_by, 'chosen_by':chosen_opt,   'reverse_fn':'reverse_positional_coloring','mode':tempmode})

                                                    rule_cands['obj_select'].append({'description':'nothing required because positional tempmode 2 or 3 which is just ocreation','skip_tag':True})
                                                
                                                if tempmode in [4,6]:


                                                    rule_cands['stackn'+str(stackn)].append({'description':'color based positional rule with obj selection by'+objsel_by,
                                                                                            'objsel_by':objsel_by, 'chosen_by':chosen_opt, 'reverse_fn':'reverse_positional_coloring','mode':tempmode,
                                                                                            'positional_rule':positional_rule, 'frame_rules':frame_rules, 'bkg_rule':bkg_rule})
                                                    

                                                break 
                                            else: continue

                                    return rule_cands


                                
                                def reverse_positional_coloring(m_states, first_currstack_rule):
                                    global gridn 
                                    gridn = gridn_
                                    

                                    
                                    
                                    mode = first_currstack_rule['mode']


                                    
                                    m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
              
                                    
                                    

                                    if mode in [2,3, 4,6]:
                                        objsel_by = first_currstack_rule['objsel_by']
                                        chosen_by = first_currstack_rule['chosen_by']

                                        i_grid = i_grids[gridn]

                                        bkg_colrs = [get_colors_of_obj(initial_global_parsings[gridn]['i'][iobj]['mask'],initial_global_parsings[gridn]['i'][iobj]['map']) for iobj in get_iobjs_of_parsing_type('background', initial_global_parsings, gridn, 'i')]

                                        i_cands = []; ci=0
                                        for iobj in global_parsings[gridn]['i']:
                                            i_mask, i_map, i_maskv, i_masko, i_obj_parsing_type = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['maskv'], global_parsings[gridn]['i'][iobj]['masko'], global_parsings[gridn]['i'][iobj]['properties']['parsing_description']
                                            colr = get_colors_of_obj(i_mask, i_map)
                                            shape = get_shape_of_obj(i_mask, i_map)
                                            r,c = np.nonzero(i_mask)
                                            centroid = (r.mean(), c.mean())
                                            if mode in [3,6] and colr in bkg_colrs: continue 
                                            if objsel_by == 'parsing_And_shape' and are_two_identical( chosen_by, (i_obj_parsing_type, shape) ):
                                                i_cands.append([iobj, i_obj_parsing_type, centroid, colr, shape, i_mask, i_map])
                                            if objsel_by == 'parsing' and are_two_identical( chosen_by, i_obj_parsing_type ):
                                                i_cands.append([iobj, i_obj_parsing_type, centroid, colr, shape, i_mask, i_map])
                                            ci+=1
                                            
                                        
                                        if len(i_cands) > 24: 
                                            e_ = 2 / 'a'

                                        
                                        
                                        i_centroids_raw = [_[2] for _ in i_cands]
                                        def adjusted_centroids(centroids):
                                            
                                            maxr, minr, maxc, minc = np.max([_[0] for _ in centroids]), np.min([_[0] for _ in centroids]), np.max([_[1] for _ in centroids]), np.min([_[1] for _ in centroids]) 
                                            adjusted = []
                                            for centroid in centroids:
                                                if (maxr-minr)==0 and (maxc-minc)==0: adjusted.append((0,0))
                                                elif (maxr-minr)==0: adjusted.append((0, (centroid[1]-minc)/(maxc-minc)))
                                                elif (maxc-minc)==0: adjusted.append(((centroid[0]-minr)/(maxr-minr), 0))
                                                else: adjusted.append(((centroid[0]-minr)/(maxr-minr), (centroid[1]-minc)/(maxc-minc)))
                                            return adjusted                                    
                                        i_centroids = adjusted_centroids(i_centroids_raw)
                                        def make_o_centroids_o_ms(rs,cs):
                                            o_ms = list(range(rs*cs))
                                            o_centroids = []
                                            for r in range(rs):
                                                for c in range(cs):
                                                    if rs==1 and cs==1: o_centroids.append((0,0))
                                                    elif rs==1: o_centroids.append((0,c/(cs-1)))
                                                    elif cs==1: o_centroids.append((r/(rs-1),0))
                                                    else: o_centroids.append((r/(rs-1),c/(cs-1)))
                                            return o_centroids, o_ms
                                        cands_ = []
                                        for rs in range(1,len(i_centroids)+1):
                                            for cs in range(1,len(i_centroids)+1):

                                                if mode in [2,4] and rs*cs != len(i_centroids): continue 

                                                o_centroids, o_ms = make_o_centroids_o_ms(rs,cs)

                                                if len(o_centroids) < len(i_centroids): continue 

                                                o_slots = [1]*len(o_centroids); rallocs = {}; chosen_dists = []
                                                for i in range(len(i_centroids)):
                                                    dists_among_oslots = [np.sqrt((o_centroids[o][0]-i_centroids[i][0])**2+(o_centroids[o][1]-i_centroids[i][1])**2) for o in range(len(o_slots)) if o_slots[o]==1]
                                                    available_o = [o for o in range(len(o_slots)) if o_slots[o]==1]
                                                    chosen_o = available_o[np.argmin(dists_among_oslots)]; chosen_dists.append(np.min(dists_among_oslots))
                                                    rallocs[chosen_o] = i

                                                cands_.append([np.mean(chosen_dists), rs, cs, rallocs])

                                        cands__ = sorted(cands_)

                                        rs, cs, rallocs = cands__[0][1], cands__[0][2], cands__[0][3]


                                    if mode in [2,3]:

                                        
                                        if mode in [3,6]: bkg_color_ = bkg_colrs[0]

                                        curr_mask = curr_map = np.zeros((rs,cs))
                                        c0=0
                                        for r in range(rs):
                                            for c in range(cs):
                                                if c0 not in rallocs: corresp_color = bkg_color_
                                                else: corresp_alloc_i_ix = rallocs[c0]; corresp_color = i_cands[corresp_alloc_i_ix][3]
                                                
                                                curr_mask[r,c] = 1
                                                curr_map[r,c] = int(corresp_color[0])
                                                c0+=1
                                        
                                        new_state = {'r_':None,'type':'analogy','gridn':gridn,'e':{'obj':'newn','mask':curr_mask,'map':curr_map,'parsing':'created_so_specialk'}}
                                        m_states.append(new_state)




                                    if mode == 5:
                                        i_ms = m_list

                                    
                                    
                                    if mode in [5]:

                                        
                                        def get_adjusted_centroids(ms, m_states, s_e, ms_type):
                                            if ms_type == 'm':
                                                centroids = []
                                                for m in ms:
                                                    r,c = np.nonzero(m_states[m][s_e]['mask'])
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)
                                            elif ms_type == 'm/p':
                                                centroids = []
                                                for m,p in ms:                            
                                                    mask_ = m_states[m][s_e]['mask']
                                                    rows,cols = np.where(mask_==1)
                                                    r,c = rows[p], cols[p]
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)                                
                                            
                                            maxr, minr, maxc, minc = np.max([_[0] for _ in centroids]), np.min([_[0] for _ in centroids]), np.max([_[1] for _ in centroids]), np.min([_[1] for _ in centroids]) 
                                            adjusted = []
                                            for centroid in centroids:
                                                if (maxr-minr)==0 and (maxc-minc)==0: adjusted.append((0,0))
                                                elif (maxr-minr)==0: adjusted.append((0, (centroid[1]-minc)/(maxc-minc)))
                                                elif (maxc-minc)==0: adjusted.append(((centroid[0]-minr)/(maxr-minr), 0))
                                                else: adjusted.append(((centroid[0]-minr)/(maxr-minr), (centroid[1]-minc)/(maxc-minc)))
                                            return adjusted
                                        i_centroids = get_adjusted_centroids(i_ms, m_states,'s','m')
                                        def make_o_centroids_o_ms(rs,cs):
                                            o_ms = list(range(rs*cs))
                                            o_centroids = []
                                            for r in range(rs):
                                                for c in range(cs):
                                                    if rs==1 and cs==1: o_centroids.append((0,0))
                                                    elif rs==1: o_centroids.append((0,c/(cs-1)))
                                                    elif cs==1: o_centroids.append((r/(rs-1),0))
                                                    else: o_centroids.append((r/(rs-1),c/(cs-1)))
                                            return o_centroids, o_ms
                                        cands_ = []
                                        for rs in range(1,len(i_centroids)+1):
                                            for cs in range(1,len(i_centroids)+1):
                                                if rs*cs == len(i_centroids):

                                                    o_centroids, o_ms = make_o_centroids_o_ms(rs,cs)

                                                    i_slots = [1]*len(i_centroids); allocs = []; rallocs = {}; chosen_dists = []; flag = True
                                                    for o in range(len(o_ms)): 
                                                        dists = [np.sqrt((o_centroids[o][0]-i_centroids[i][0])**2+(o_centroids[o][1]-i_centroids[i][1])**2) for i in range(len(i_slots)) if i_slots[i]==1]
                                                        if len(dists)==0: flag = False; break
                                                        available_i = [i for i in range(len(i_slots)) if i_slots[i]==1]
                                                        chosen_i = available_i[np.argmin(dists)]; chosen_dists.append(np.min(dists))
                                                        allocs.append((i_ms[chosen_i],o_ms[o])); rallocs[o_ms[o]] = i_ms[chosen_i]
                                                        i_slots[chosen_i] = 0 
                                                    
                                                    if flag:
                                                        
                                                        cands_.append([np.mean(chosen_dists), rs, cs, rallocs])
                                        cands__ = sorted(cands_)
                                        
                                        rs, cs, rallocs = cands__[0][1], cands__[0][2], cands__[0][3]




                                    if mode == 1:
                                        if len(i_list) != len(m_list): print("ERROR")
                                        o_ms = m_list

                                        def get_adjusted_centroids(ms, m_states, s_e, ms_type):
                                            if ms_type == 'm':
                                                centroids = []
                                                for m in ms:
                                                    r,c = np.nonzero(m_states[m][s_e]['mask'])
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)
                                            elif ms_type == 'm/p':
                                                centroids = []
                                                for m,p in ms:                            
                                                    mask_ = m_states[m][s_e]['mask']
                                                    rows,cols = np.where(mask_==1)
                                                    r,c = rows[p], cols[p]
                                                    centroid = (r.mean(), c.mean())
                                                    centroids.append(centroid)                                
                                            
                                            maxr, minr, maxc, minc = np.max([_[0] for _ in centroids]), np.min([_[0] for _ in centroids]), np.max([_[1] for _ in centroids]), np.min([_[1] for _ in centroids]) 
                                            adjusted = []
                                            for centroid in centroids:
                                                if (maxr-minr)==0 and (maxc-minc)==0: adjusted.append((0,0))
                                                elif (maxr-minr)==0: adjusted.append((0, (centroid[1]-minc)/(maxc-minc)))
                                                elif (maxc-minc)==0: adjusted.append(((centroid[0]-minr)/(maxr-minr), 0))
                                                else: adjusted.append(((centroid[0]-minr)/(maxr-minr), (centroid[1]-minc)/(maxc-minc)))
                                            return adjusted
                                        i_centroids = get_adjusted_centroids(i_ms, m_states,'s','m')
                                        o_centroids = get_adjusted_centroids(o_ms, m_states,'s','m')
                                        i_slots = [1]*len(i_centroids); allocs = []; rallocs = {}
                                        for o in range(len(o_ms)): 
                                            dists = [np.sqrt((o_centroids[o][0]-i_centroids[i][0])**2+(o_centroids[o][1]-i_centroids[i][1])**2) for i in range(len(i_slots)) if i_slots[i]==1]
                                            available_i = [i for i in range(len(i_slots)) if i_slots[i]==1]
                                            chosen_i = available_i[np.argmin(dists)]
                                            allocs.append((i_ms[chosen_i],o_ms[o])); rallocs[o_ms[o]] = i_ms[chosen_i]
                                            i_slots[chosen_i] = 0 

                                        
                                        for m in m_list:
                                            state = m_states[m]
                                            curr_map, curr_mask = state['s']['map'], state['s']['mask']

                                            og_color = get_colors_of_obj(curr_mask, curr_map)
                                            corresp_alloc = rallocs[m]
                                            istate = m_states[corresp_alloc]
                                            icurr_map, icurr_mask = istate['s']['map'], istate['s']['mask']
                                            corresp_color = get_colors_of_obj(icurr_mask, icurr_map)

                                            curr_fn = globals()['recolor']
                                            curr_params = {'color_changes':[[og_color, corresp_color]]}
                                            curr_map, curr_mask = curr_fn(curr_map, curr_mask,**curr_params)
                                            state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map}     

                                    if mode in [4,5]: 

                                        n_rows = rs
                                        n_cols = cs

                                        k=0; allocated_ms = []
                                        for r in range(rs):
                                            for c in range(cs):
                                                
                                                allocated_ms.append([rallocs[k], r, c])
                                                k+=1


                                        bkg_rule = first_currstack_rule['bkg_rule']
                                        positional_rule = first_currstack_rule['positional_rule']
                                        frame_rules = first_currstack_rule['frame_rules']


                                        
                                        try: 
                                            bkg_color = bkg_rule['bkg_color'] 
                                            if bkg_rule['bkg_mtd'] == 'choose_hyperp_color': bkg_color = bkg_rule['bkg_color']
                                            elif bkg_rule['bkg_mtd'] == 'actual_bkg_objs_color':
                                                for m in range(len(m_states)):
                                                    
                                                    if m_states[m]['s']['parsing'][0] == 'background':
                                                        bkg_color = get_colors_of_obj(m_states[m]['s']['mask'],m_states[m]['s']['map'])
                                                        break
                                            elif bkg_rule['bkg_mtd'] == 'choose_iframe_color':
                                                for m in range(len(m_states)):
                                                    if m_states[m]['s']['parsing'][0] == 'frame_iobj':
                                                        bkg_color = get_colors_of_obj(m_states[m]['s']['mask'],m_states[m]['s']['map'])
                                                        break                        
                                            elif bkg_rule['bkg_mtd'] == 'choose_common_color':
                                                subframe_iobjs = []
                                                for m in range(len(m_states)):
                                                    if m_states[m]['s']['parsing'][0] == 'subframe_iobj':
                                                        subframe_iobjs.append(get_colors_of_obj(m_states[m]['s']['mask'],m_states[m]['s']['map']))
                                                for colr in subframe_iobjs[0]:
                                                    if np.all([is_x_in_y(x=colr,y=_) for _ in subframe_iobjs]):
                                                        bkg_color = colr
                                                        break
                                            
                                        except: bkg_color = 0  



                                        
                                        r_prepend = frame_rules['vert_border'] 
                                        r_append = [frame_rules['vert_frame']]*(n_rows-1) + [frame_rules['vert_border']]
                                        c_prepend = frame_rules['horiz_border']
                                        c_append = [frame_rules['horiz_frame']]*(n_cols-1) + [frame_rules['horiz_border']]
                                        
                                        frame_specs = [c_prepend, c_append, r_prepend, r_append] 

                                    if mode == 4: 

                                        


                                        actual_m = m_list[0] 
                                        iobj, mask_, map_ = m_states[actual_m]['s']['obj'], m_states[actual_m]['s']['mask'], m_states[actual_m]['s']['map']
                                        original_colors = get_colors_of_obj(mask_, map_)
                                        i_mask = []; i_map = []; tiles = []; safe_tile_ixs = []
                                        for k in range(len(allocated_ms)):
                                            interact_obj_colr = i_cands[allocated_ms[k][0]][3] 
                                            
                                            
                                            i_mask.append(mask_); i_map.append(map_)

                                            tiles.append({'iobj':iobj,'color_changes':[[original_colors[0], interact_obj_colr[0]]]}) 
                                            safe_tile_ixs.append(0)



                                        details = {'details':{'n_rows': n_rows, 'n_cols': n_cols,
                                            'safe_tile_ixs': safe_tile_ixs,
                                            'tiles': tiles,
                                            'frame_specs': frame_specs,
                                            'bkg_dets': {'bkg_mtd': bkg_rule['bkg_mtd'] if bkg_rule is not None else 'actual_bkg_objs_color',
                                            'bkg_color': bkg_color}}}

                                        serial_tr =  'iobjs_tile_creation'

                                        state = m_states[m]
                                        
                                        curr_fn = globals()[serial_tr]
                                        curr_params = details
                                        curr_map, curr_mask = curr_fn(i_map, i_mask,**curr_params)
                                        


                                        new_state = {'r_':None,'type':'analogy','gridn':gridn,'e':{'obj':'bewn','mask':curr_mask,'map':curr_map,'parsing':'created_so_specialR'}}
                                        m_states.append(new_state)

                                    if mode == 5: 


                                        
                                        i_mask = []; i_map = []; iobjs = []; tiles = []; safe_tile_ixs = []
                                        for k in range(len(allocated_ms)):
                                            m = allocated_ms[k][0]
                                            iobj = m_states[m]['s']['obj']
                                            mask_, map_ = m_states[m]['s']['mask'], m_states[m]['s']['map']
                                            iobjs.append(iobj); i_mask.append(mask_); i_map.append(map_)
                                            tiles.append({'iobj':iobj,'color_changes':None})
                                            safe_tile_ixs.append(iobjs.index(iobj))


                                        details = {'details':{'n_rows': n_rows, 'n_cols': n_cols,
                                            'safe_tile_ixs': safe_tile_ixs,
                                            'tiles': tiles,
                                            'frame_specs': frame_specs,
                                            'bkg_dets': {'bkg_mtd': bkg_rule['bkg_mtd'] if bkg_rule is not None else 'actual_bkg_objs_color',
                                            'bkg_color': bkg_color}}}

                                        serial_tr =  'iobjs_tile_creation'

                                        state = m_states[m]
                                        
                                        curr_fn = globals()[serial_tr]
                                        curr_params = details
                                        curr_map, curr_mask = curr_fn(i_map, i_mask,**curr_params)
                                        


                                        new_state = {'r_':None,'type':'analogy','gridn':gridn,'e':{'obj':'bewn','mask':curr_mask,'map':curr_map,'parsing':'created_so_specialR'}}
                                        m_states.append(new_state)



                                    return m_states


                                
                                if curr_ref_serial_transform == 'recolor': print('recolor/positional'); rule_cands = positional(m_states, rule_cands, 1)
                                if curr_ref_serial_transform == 'hyperp_gridmap_creation': print('hyperp_gridmap_creation/positional -- x2'); rule_cands = positional(m_states, rule_cands, 2); rule_cands = positional(m_states, rule_cands, 3)
                                if curr_ref_serial_transform == 'iobjs_tile_creation': print('iobjs_tile_creation/positional'); rule_cands = positional(m_states, rule_cands, 4)



                                def fill_slot_hole_rule(m_states, rule_cands):
                                    

                                    a_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']

                                    
                                    

                                    a_serial_params_list = [state['serial_param'][stackn] for state in m_states if state['type']=='analogy']
                                    
                                    mode0validity = True
                                    mode12rules = {} 
                                    invalidchains=[]
                                    all_selected_nonhole_iobjs = []; all_nonhole_iobjs = []; all_nonhole_dets = [] 
                                    for k in range(len(a_list)):
                                        a = a_list[k]
                                        param_hyperps = a_serial_params_list[k]['hyperps']
                                        iobj_list, mask_list, map_list = m_states[a]['s']['obj'], m_states[a]['s']['mask'], m_states[a]['s']['map']


                                        nonhole_iobjs = []; hole_iobjs = []; hole_dets = []; nonhole_dets = []
                                        for _ in range(len(iobj_list)):
                                            entry = param_hyperps[_]
                                            if entry['type']=='hole_iobj': hole_iobjs.append(iobj_list[_]); hole_dets.append(entry['hole_newcolors'])
                                            if entry['type']=='nonhole_iobj': 
                                                nonhole_iobjs.append(iobj_list[_]); nonhole_dets.append(entry['is_staticQ'])
                                                all_nonhole_iobjs.append(iobj_list[_]); all_nonhole_dets.append(entry['is_staticQ'])

                                        

                                        for c, iobj in enumerate(hole_iobjs):
                                            mask_, map_ = mask_list[iobj_list.index(iobj)], map_list[iobj_list.index(iobj)]

                                            filled = binary_fill_holes(mask_);     
                                            holes_mask = np.logical_and(filled, np.logical_not(mask_)).astype(int)
                                            if np.sum(holes_mask)>0:
                                                labeled_array = get_contiguous_regions(holes_mask,0,False,False)
                                                num_holes = np.max(labeled_array); ch=0
                                                for hole_n in range(1,np.max(labeled_array)+1):
                                                    hole_mask = (labeled_array==hole_n).astype(int)
                                                    newcolor = hole_dets[c][ch]
                                                    ch+=1
                                                    
                                                    bb_mask, _, tl_rc = get_bounding_box_object(hole_mask, hole_mask)

                                                    
                                                    jobj_cands = []; jobj_lbls = [] 
                                                    for jobj in nonhole_iobjs:
                                                        j_mask, j_map = mask_list[iobj_list.index(jobj)], map_list[iobj_list.index(jobj)]
                                                        bb_maskj, bb_mapj, tl_rc = get_bounding_box_object(j_mask, j_map)
                                                        if are_two_identical(bb_mask, bb_maskj):
                                                            jobj_cands.append(jobj)
                                                            lbl_ = 1 if are_two_identical(newcolor, get_colors_of_obj(bb_maskj, bb_mapj)) else 0
                                                            if lbl_ == 1: all_selected_nonhole_iobjs.append(jobj)
                                                            jobj_lbls.append(lbl_)
                                                    

                                                    if len(jobj_cands)==0 or 1 not in jobj_lbls or (len(jobj_cands)>0 and jobj_cands[0] not in iobj_list): mode0validity = False; continue

                                                    for chained_prop in [num_of_its_color_anyshape, is_largest_num_of_its_color_anyshape, is_smallest_num_of_its_color_anyshape, is_oddeven_num_of_its_color_anyshape, num_of_its_color_itsshape, is_largest_num_of_its_color_itsshape, is_smallest_num_of_its_color_itsshape, is_oddeven_num_of_its_color_itsshape]:
                                                        prop_list = chained_prop(jobj_cands, nonhole_iobjs,     iobj_list, mask_list, map_list)
                                                        
                                                        
                                                        if chained_prop not in mode12rules: mode12rules[chained_prop] = {} 
                                                        
                                                        
                                                        elif chained_prop.__name__ in invalidchains: 
                                                            if chained_prop.__name__!= 'is_oddeven_num_of_its_color_itsshape': continue 
                                                            else: break
                                                        for p in range(len(prop_list)):
                                                            prop = prop_list[p]
                                                            param = jobj_lbls[p]
                                                            if prop not in mode12rules[chained_prop]: 
                                                                mode12rules[chained_prop][prop] = param
                                                            else: 
                                                                if mode12rules[chained_prop][prop] != param: 
                                                                    
                                                                    invalidchains.append(chained_prop.__name__)
                                                                else: pass

                                    congruent_opt = None
                                    for opt in mode12rules:
                                        if mode12rules[opt] is not False:
                                            congruent_opt = opt
                                            rawmatrix = mode12rules[opt]
                                            Matrix = []
                                            for key in rawmatrix:
                                                val = rawmatrix[key]
                                                Matrix.append({'PROPS':key,'PARAM':val})
                                            default_ = 0 
                                            Rule_ = ['standard rule', Matrix, default_]
                                            break

                                    
                                    
                                    obj_static_rule = None
                                    if True not in all_nonhole_dets: obj_static_rule = 'nothing_is_static'
                                    else:
                                        templbls = []
                                        for iobj_ in all_nonhole_iobjs:
                                            if iobj_ in all_selected_nonhole_iobjs: templbls.append(True)
                                            else: templbls.append(False)
                                        if are_two_identical(templbls, all_nonhole_dets):
                                            obj_static_rule = 'selections_are_static'
                                        elif are_two_identical([True if _==False else False for _ in templbls], all_nonhole_dets):
                                            obj_static_rule = 'nonselections_are_static'


                                    if mode0validity and congruent_opt is not None:
                                        mode = 0
                                        rule_cands['stackn'+str(stackn)].append({'description':'fill_slot_hole_rule'+congruent_opt.__name__,'obj_static':obj_static_rule,
                                                'prop':congruent_opt,'rule':Rule_, 'serial_tr':curr_ref_serial_transform,'reverse_fn':'reverse_fill_slot_hole_rule','mode':0})


                                    

                                    
                                    

                                    a_serial_params_list = [state['serial_param'][stackn] for state in m_states if state['type']=='analogy']
                                    
                                    mode1validity = True
                                    for k in range(len(a_list)):
                                        a = a_list[k]
                                        param_hyperps = a_serial_params_list[k]['hyperps']
                                        iobj_list, mask_list, map_list = m_states[a]['s']['obj'], m_states[a]['s']['mask'], m_states[a]['s']['map']
                                        gridn = m_states[a]['gridn']
                                        i_grid = i_grids[gridn]
                                        o_grid = o_grids[gridn]

                                        hole_iobjs = []; hole_dets = []; nonhole_iobjs = []; nonhole_dets = []
                                        for _ in range(len(iobj_list)):
                                            entry = param_hyperps[_]
                                            if entry['type']=='hole_iobj': hole_iobjs.append(iobj_list[_]); hole_dets.append(entry['hole_newcolors'])
                                            

                                        
                                        
                                        

                                        def coords_to_grid_indices(coords):
                                            if not coords:
                                                return []
                                            
                                            rows = sorted(set(r for r, _ in coords))
                                            cols = sorted(set(c for _, c in coords))
                                            
                                            
                                            grid = [[None for _ in cols] for _ in rows]
                                            
                                            
                                            for i, (r, c) in enumerate(coords):
                                                ri = rows.index(r)
                                                ci = cols.index(c)
                                                grid[ri][ci] = i
                                            
                                            
                                            
                                            
                                            
                                            simplified = []
                                            for row in grid:
                                                simplified.append([x for x in row if x is not None])
                                            
                                            return simplified

                                        all_color_mappings = {} 
                                        for c, iobj in enumerate(hole_iobjs):
                                            mask_, map_ = mask_list[iobj_list.index(iobj)], map_list[iobj_list.index(iobj)]
                                            filled = binary_fill_holes(mask_);     
                                            holes_mask = np.logical_and(filled, np.logical_not(mask_)).astype(int)
                                            if np.sum(holes_mask)>0:
                                                labeled_array = get_contiguous_regions(holes_mask,0,False,False)
                                                num_holes = np.max(labeled_array); ch=0; rc_o_n = []
                                                for hole_n in range(1,np.max(labeled_array)+1):
                                                    hole_mask = (labeled_array==hole_n).astype(int)
                                                    hole_imap  = np.where(hole_mask, i_grid, 0)        
                                                    hole_omap  = np.where(hole_mask, o_grid, 0)  
                                                    oldcolor = get_colors_of_obj(hole_mask, hole_imap)
                                                    newcolor = get_colors_of_obj(hole_mask, hole_omap) 
                                                    r,c = np.nonzero(hole_mask)
                                                    centroid = (r.mean(), c.mean())
                                                    rc_o_n.append([centroid, oldcolor, newcolor])
                                                    ch+=1
                                                cdt_ixs = coords_to_grid_indices([_[0] for _ in rc_o_n])
                                                mappings_lioli = []
                                                for li1 in cdt_ixs:
                                                    newli1 = []
                                                    for li2 in li1:
                                                        newli1.append({'old':rc_o_n[li2][1],'new':rc_o_n[li2][2]})
                                                    mappings_lioli.append(newli1)
                                                if num_holes not in all_color_mappings: all_color_mappings[num_holes] = []
                                                all_color_mappings[num_holes].append(mappings_lioli)
                                        
                                        valid = True

                                        
                                        save_dets = {}
                                        for num_holes in all_color_mappings:
                                            all_pair_lists = all_color_mappings[num_holes]
                                            
                                            
                                            mapping0 = all_pair_lists[0]
                                            typical = []; unrolled_typical = []
                                            for c1, li1 in enumerate(mapping0):
                                                newli1 = []
                                                for c2, li2 in enumerate(li1):
                                                    newli1.append((c1,c2)); unrolled_typical.append((c1,c2))
                                                typical.append(newli1)
                                            
                                            
                                            num_instances = len(all_pair_lists)
                                            
                                            color_pairs = []; unrolled_colors = []
                                            usable_colors = []; unusable_colors = []
                                            newcolors_over_u = []
                                            for u in range(len(unrolled_typical)):
                                                old_news = [all_pair_lists[i][unrolled_typical[u][0]][unrolled_typical[u][1]] for i in range(num_instances)]
                                                
                                                
                                                news = [_['new'] for _ in old_news]
                                                if not are_all_identical(news): 
                                                    valid = False
                                                    mode1validity = False
                                                    break
                                                newcolors_over_u.append(news[0])
                                                olds = [_['old'] for _ in old_news]
                                                _, unique_olds = label_unique_with_IDs(olds)
                                                unique_olds = sorted(unique_olds)
                                                
                                                
                                                
                                                if len(unique_olds)==1: usable_colors.append(unique_olds[0])
                                                else: 
                                                    color_pairs.append(unique_olds)
                                                    for uo in unique_olds:
                                                        unrolled_colors.append(uo)
                                            
                                            for colr in unrolled_colors:
                                                
                                                opts = []
                                                for pair in color_pairs:
                                                    if is_x_in_y(colr, pair):
                                                        opts.append(pair)
                                                if are_all_identical(opts): usable_colors.append(colr) 
                                                else: unusable_colors.append(colr)
                                            
                                            definites = {}; checkcolors_over_u = []
                                            for u in range(len(unrolled_typical)):
                                                old_news = [all_pair_lists[i][unrolled_typical[u][0]][unrolled_typical[u][1]] for i in range(num_instances)]
                                                olds = [_['old'] for _ in old_news]      
                                                for colr in olds:
                                                    if is_x_in_y(colr, usable_colors):
                                                        definites[u] = colr
                                                        checkcolors_over_u.append(colr)
                                                        break
                                            
                                            
                                            
                                            
                                            
                                            if not are_two_identical(checkcolors_over_u, newcolors_over_u): mode1validity = False

                                    if mode1validity:
                                        mode = 1
                                        rule_cands['stackn'+str(stackn)].append({'description':'fill_slot_hole_rule_167',
                                                    'serial_tr':curr_ref_serial_transform,'reverse_fn':'reverse_fill_slot_hole_rule','mode':1})


                                    return rule_cands

                                def reverse_fill_slot_hole_rule(m_states, first_currstack_rule):
                                    global gridn 
                                    gridn = gridn_


                                    mode = first_currstack_rule['mode']

                                    
                                    if mode in [0]:

                                        rt, M, d = first_currstack_rule['rule']
                                        prop = first_currstack_rule['prop']
                                        obj_static = first_currstack_rule['obj_static']

                                        

                                        m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                        for m in m_list:
                                            iobj_list, mask_list, map_list = m_states[m]['s']['obj'], m_states[m]['s']['mask'], m_states[m]['s']['map']
                                            gridn = m_states[m]['gridn']
                                            i_grid = i_grids[gridn]
                                            

                                            
                                            holes_iobjs = []; nonhole_iobjs = []; holes_mask_lists = []

                                            for k in range(len(iobj_list)):
                                                iobj_, mask_, map_ = iobj_list[k], mask_list[k], map_list[k]

                                                filled = binary_fill_holes(mask_);     
                                                holes_mask = np.logical_and(filled, np.logical_not(mask_)).astype(int)
                                                if np.sum(holes_mask)>0:
                                                    labeled_array = get_contiguous_regions(holes_mask,0,False,False)
                                                    num_holes = np.max(labeled_array)
                                                    li = []
                                                    for hole_n in range(1,np.max(labeled_array)+1):
                                                        hole_mask = (labeled_array==hole_n).astype(int)
                                                        hole_imap  = np.where(hole_mask, i_grid, 0)        
                                                        li.append(hole_mask)
                                                    holes_iobjs.append(iobj_); holes_mask_lists.append(li)
                                                else: nonhole_iobjs.append(iobj_)

                                            
                                            param_dets_unsorted = {}; utilised_nonhole_iobjs = []
                                            for c in range(len(holes_iobjs)):
                                                hole_iobj = holes_iobjs[c]; hole_newcolors = []
                                                for hole_mask in holes_mask_lists[c]:
                                                    bb_mask, _, tl_rc = get_bounding_box_object(hole_mask, hole_mask)
                                                    
                                                    jobj_cands = []
                                                    for jobj in nonhole_iobjs:
                                                        j_mask, j_map = mask_list[iobj_list.index(jobj)], map_list[iobj_list.index(jobj)]
                                                        bb_maskj, bb_mapj, tl_rc = get_bounding_box_object(j_mask, j_map)
                                                        if are_two_identical(bb_mask, bb_maskj):           
                                                            jobj_cands.append(jobj)                                     
                                                    
                                                    prop_list = prop(jobj_cands, nonhole_iobjs,     iobj_list, mask_list, map_list)
                                                    param_list = reverse_mapping(prop_list, M, d, rt)
                                                    selected_jobj = jobj_cands[param_list.index(1)]
                                                    utilised_nonhole_iobjs.append(selected_jobj)
                                                    j_mask, j_map = mask_list[iobj_list.index(selected_jobj)], map_list[iobj_list.index(selected_jobj)]
                                                    newcolor = get_colors_of_obj(j_mask, j_map)
                                                    hole_newcolors.append(newcolor)
                                                param_dets_unsorted[hole_iobj] = {'type':'hole_iobj','hole_newcolors':hole_newcolors}

                                            
                                            if obj_static == 'nothing_is_static':
                                                for iobj_ in nonhole_iobjs:
                                                    param_dets_unsorted[iobj_] = {'type':'nonhole_iobj','is_staticQ':False}
                                            elif obj_static == 'selections_are_static':
                                                for iobj_ in nonhole_iobjs:
                                                    if iobj_ in utilised_nonhole_iobjs: param_dets_unsorted[iobj_] = {'type':'nonhole_iobj','is_staticQ':True}
                                                    else: param_dets_unsorted[iobj_] = {'type':'nonhole_iobj','is_staticQ':False}
                                            elif obj_static == 'nonselections_are_static':
                                                for iobj_ in nonhole_iobjs:
                                                    if iobj_ in utilised_nonhole_iobjs: param_dets_unsorted[iobj_] = {'type':'nonhole_iobj','is_staticQ':False}
                                                    else: param_dets_unsorted[iobj_] = {'type':'nonhole_iobj','is_staticQ':True}

                                            
                                            param_dets = []
                                            for iobj_ in iobj_list:
                                                param_dets.append(param_dets_unsorted[iobj_])


                                            state = m_states[m]
                                            curr_fn = globals()['fill_slot_holes']
                                            curr_params = {'hyperps':param_dets}
                                            curr_map, curr_mask = curr_fn(map_list, mask_list, **curr_params)
                                            state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map}    
                                            

                                    

                                    if mode in [1]:
                                        

                                        def coords_to_grid_indices(coords):
                                            if not coords:
                                                return []
                                            
                                            rows = sorted(set(r for r, _ in coords))
                                            cols = sorted(set(c for _, c in coords))
                                            
                                            
                                            grid = [[None for _ in cols] for _ in rows]
                                            
                                            
                                            for i, (r, c) in enumerate(coords):
                                                ri = rows.index(r)
                                                ci = cols.index(c)
                                                grid[ri][ci] = i
                                            
                                            
                                            
                                            
                                            
                                            simplified = []
                                            for row in grid:
                                                simplified.append([x for x in row if x is not None])
                                            
                                            return simplified

                                        

                                        m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                        for m in m_list:
                                            iobj_list, mask_list, map_list = m_states[m]['s']['obj'], m_states[m]['s']['mask'], m_states[m]['s']['map']
                                            gridn = m_states[m]['gridn']
                                            i_grid = i_grids[gridn]
                                            

                                            
                                            holes_iobjs = []; nonhole_iobjs = []; holes_mask_lists = []

                                            all_color_mappings = {}

                                            for k in range(len(iobj_list)):
                                                iobj_, mask_, map_ = iobj_list[k], mask_list[k], map_list[k]

                                                filled = binary_fill_holes(mask_);     
                                                holes_mask = np.logical_and(filled, np.logical_not(mask_)).astype(int)
                                                if np.sum(holes_mask)>0:
                                                    labeled_array = get_contiguous_regions(holes_mask,0,False,False)
                                                    num_holes = np.max(labeled_array)
                                                    rc_o_n = []; li=[]
                                                    for hole_n in range(1,np.max(labeled_array)+1):
                                                        hole_mask = (labeled_array==hole_n).astype(int)
                                                        hole_imap  = np.where(hole_mask, i_grid, 0)       

                                                        oldcolor = get_colors_of_obj(hole_mask, hole_imap)
                                                        r,c = np.nonzero(hole_mask)
                                                        centroid = (r.mean(), c.mean())
                                                        rc_o_n.append([centroid, oldcolor])
                                                    cdt_ixs = coords_to_grid_indices([_[0] for _ in rc_o_n])
                                                    mappings_lioli = []
                                                    for li1 in cdt_ixs:
                                                        newli1 = []
                                                        for li2 in li1:
                                                            newli1.append({'old':rc_o_n[li2][1]})
                                                        mappings_lioli.append(newli1)
                                                    if num_holes not in all_color_mappings: all_color_mappings[num_holes] = []
                                                    all_color_mappings[num_holes].append(mappings_lioli)
                                                    holes_iobjs.append(iobj_); holes_mask_lists.append(li)
                                                else: nonhole_iobjs.append(iobj_)

                                            
                                            save_dets = {}
                                            for num_holes in all_color_mappings:
                                                all_pair_lists = all_color_mappings[num_holes]
                                                
                                                
                                                mapping0 = all_pair_lists[0]
                                                typical = []; unrolled_typical = []
                                                for c1, li1 in enumerate(mapping0):
                                                    newli1 = []
                                                    for c2, li2 in enumerate(li1):
                                                        newli1.append((c1,c2)); unrolled_typical.append((c1,c2))
                                                    typical.append(newli1)
                                                
                                                
                                                num_instances = len(all_pair_lists)
                                                
                                                color_pairs = []; unrolled_colors = []
                                                usable_colors = []; unusable_colors = []
                                                newcolors_over_u = []
                                                for u in range(len(unrolled_typical)):
                                                    old_news = [all_pair_lists[i][unrolled_typical[u][0]][unrolled_typical[u][1]] for i in range(num_instances)]
                                                    olds = [_['old'] for _ in old_news]
                                                    _, unique_olds = label_unique_with_IDs(olds)
                                                    unique_olds = sorted(unique_olds)
                                                    
                                                    
                                                    
                                                    if len(unique_olds)==1: usable_colors.append(unique_olds[0])
                                                    else: 
                                                        color_pairs.append(unique_olds)
                                                        for uo in unique_olds:
                                                            unrolled_colors.append(uo)
                                                
                                                for colr in unrolled_colors:
                                                    
                                                    opts = []
                                                    for pair in color_pairs:
                                                        if is_x_in_y(colr, pair):
                                                            opts.append(pair)
                                                    if are_all_identical(opts): usable_colors.append(colr) 
                                                    else: unusable_colors.append(colr)
                                                
                                                definites = {}; checkcolors_over_u = []
                                                for u in range(len(unrolled_typical)):
                                                    old_news = [all_pair_lists[i][unrolled_typical[u][0]][unrolled_typical[u][1]] for i in range(num_instances)]
                                                    olds = [_['old'] for _ in old_news]      
                                                    for colr in olds:
                                                        if is_x_in_y(colr, usable_colors):
                                                            definites[u] = colr
                                                            checkcolors_over_u.append(colr)
                                                            break
                                                
                                                save_dets[num_holes] = [definites, unrolled_typical]


                                            
                                            param_dets = []
                                            for k in range(len(iobj_list)):
                                                iobj_, mask_, map_ = iobj_list[k], mask_list[k], map_list[k]

                                                filled = binary_fill_holes(mask_);     
                                                holes_mask = np.logical_and(filled, np.logical_not(mask_)).astype(int)
                                                if np.sum(holes_mask)>0:
                                                    labeled_array = get_contiguous_regions(holes_mask,0,False,False)
                                                    num_holes = np.max(labeled_array)
                                                    rc_o_n = []
                                                    for hole_n in range(1,np.max(labeled_array)+1):
                                                        hole_mask = (labeled_array==hole_n).astype(int)
                                                        hole_imap  = np.where(hole_mask, i_grid, 0)     
                                                        oldcolor = get_colors_of_obj(hole_mask, hole_imap)
                                                        r,c = np.nonzero(hole_mask)
                                                        centroid = (r.mean(), c.mean())
                                                        rc_o_n.append([centroid, oldcolor])
                                                    cdt_ixs = coords_to_grid_indices([_[0] for _ in rc_o_n])
                                                    definites, unrolled_typical = save_dets[num_holes]
                                                    tosort = {}
                                                    for u in range(len(unrolled_typical)):
                                                        rc_o_n_cdt_ix = cdt_ixs[unrolled_typical[u][0]][unrolled_typical[u][1]]
                                                        newcolor = definites[u]
                                                        tosort[rc_o_n_cdt_ix] = newcolor
                                                    newcolors = []
                                                    for k in range(len(rc_o_n)):
                                                        newcolors.append(tosort[k])
                                                    
                                                    param_dets.append({'type':'hole_iobj','hole_newcolors':newcolors})
                                                else: param_dets.append({'type':'nonhole_iobj','is_staticQ':False})

                                            state = m_states[m]
                                            curr_fn = globals()['fill_slot_holes']
                                            curr_params = {'hyperps':param_dets}
                                            curr_map, curr_mask = curr_fn(map_list, mask_list, **curr_params)
                                            state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map}    
                                            
                    

                                    return m_states

                                if curr_ref_serial_transform == 'fill_slot_holes': print('fill_slot_holes/fill_slot_hole_rule'); rule_cands = fill_slot_hole_rule(m_states, rule_cands)


                                def slotting_hyperp(m_states, rule_cands):
                                    
                                    a_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']


                                    

                                    unrolled_a_iobjs = []
                                    for a in a_list:
                                        if type(m_states[a]['s']['obj'])==list: unrolled_a_iobjs.extend(m_states[a]['s']['obj'])
                                        else: unrolled_a_iobjs.append(m_states[a]['s']['obj'])

                                    h_states = [] 
                                    for gridn_ in range(num_demo_grids):
                                        for iobj_ in initial_global_parsings[gridn_]['i']: 
                                            
                                            flag = True if iobj_ in unrolled_a_iobjs else False
                                            
                                            h_states.append({'type':'h_state','gridn':gridn_,'s':{'obj':iobj_,'mask':initial_global_parsings[gridn_]['i'][iobj_]['mask'],'map':initial_global_parsings[gridn_]['i'][iobj_]['map'],'parsing':initial_global_parsings[gridn_]['i'][iobj_]['properties']['parsing_description']},
                                                                            'is_in_m_states_analogy':1 if flag else 0})

                                    h_list = [h for h in range(len(h_states))]
                                    h_lbls = [h_states[h]['is_in_m_states_analogy'] for h in range(len(h_states))]


                                    for self_prop in [self_parsing_AND_color]:
                                        prop_list = self_prop(h_list, h_states)
                                        rule = check_mapping(prop_list, h_lbls)
                                        if rule is not None:
                                            rule_cands['obj_select'].append({'description':'objsel apply '+self_prop.__name__+' rule to yield analogy obj selection','parsing_objsel':False,'prop':self_prop,'rule':rule})

                
                            
                                    
                                    return rule_cands


                                def reverse_slotting_hyperp(m_states, first_currstack_rule):

                                    

                                    
                                    

                                    return m_states
                                

                                
                                



                                def hyperp1(m_states, rule_cands):


                                
                                    

                                    a_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']; serial_tr = 'N/A'
                                    o_list = [m for m in range(len(m_states)) if m_states[m]['type']=='other']
                                    
                                    
                                    
                                    

                                    tempstores = []
                                    for m, state in enumerate(m_states):
                                        if state['type']=='analogy':
                                            if state['serial_transform'][stackn]['type'] == 'gridwise_tiled_copy':
                                                tiling_details = state['serial_param'][stackn]['tiling_details']
                                                filtered_options, rmult, cmult, occlusion_mask  = tiling_details['filtered_options'], tiling_details['rmult'], tiling_details['cmult'], tiling_details['occlusion_mask']   
                                                tempstore = [int(rmult),int(cmult), filtered_options, occlusion_mask]
                                            else: tempstore = [0,0,[],None]
                                            tempstores.append(tempstore)


                          


                                    param_rcmult_list = [(_[0],_[1]) for _ in tempstores] 
                                    param_occlusion_masks = [_[3] for _ in tempstores]
                                    pan_param_found_set = [] 
                                    param_arrays = []; c0 = 0
                                    for tempstore in tempstores:
                                        c1 = 0
                                        
                                        opt_array = np.zeros((param_rcmult_list[c0][0],param_rcmult_list[c0][1]))
                                        n_to_rc = []
                                        for r_ in range(param_rcmult_list[c0][0]):
                                            for c_ in range(param_rcmult_list[c0][1]):
                                                n_to_rc.append((r_,c_))
                                        
                                        for rcset in tempstore[2]:
                                            rc_first = rcset[0] 
                                            
                                            if is_x_in_y(x=rc_first,y=pan_param_found_set): ix = ix_of_x_in_y(x=rc_first,y=pan_param_found_set)
                                            else: pan_param_found_set.append(rc_first); ix = len(pan_param_found_set)-1
                                            opt_array[n_to_rc[c1]] = ix 
                                            c1+=1
                                        param_arrays.append(opt_array)
                                        c0+=1

                                    
                                    
                                    

                                    if len(param_arrays)==0: return rule_cands 
                                    lbls, uniques = label_unique_with_IDs(param_arrays)
                                    if len(uniques)==1: print("Quit")


           

                                    def are_mappable_arrays(bb_map, param_arr):
                                        temp = {}; isvalid = all(temp.setdefault(x, y) == y for x, y in zip(bb_map.flatten(), param_arr.flatten()))
                                        if isvalid: return True
                                        else: return False

                                    is_self_s_state = True; unrolled_mapping_list = [[],[]] 
                                    for c, a in enumerate(a_list): 
                                        state = m_states[a]
                                        mask_, map_, gridn_ = state['s']['mask'], state['s']['map'], state['gridn']
                                        bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_, map_)
                                        if bb_mask.shape == param_arrays[c].shape and np.sum(bb_mask)==bb_mask.shape[0]*bb_mask.shape[1]: 
                                            if are_mappable_arrays(bb_map, param_arrays[c]): 
                                                unrolled_mapping_list[0].extend(bb_map.flatten())
                                                unrolled_mapping_list[1].extend(param_arrays[c].flatten())
                                            else: is_self_s_state = False; break
                                        else: is_self_s_state = False; break

                                    is_specifiable_iobj = False
                                    if not is_self_s_state:
                                        is_specifiable_iobj = True; linked_iobj_list = []; unrolled_mapping_list = [[],[]] 
                                        for c, a in enumerate(a_list): 
                                            
                                            state = m_states[a]
                                            gridn_ = state['gridn']
                                            matches_for_curr_a = []; store_bbmap = None; store_parr = None
                                            for o in [_['m'] for _ in others_over_grids[gridn_]]:
                                                ostate = m_states[o]
                                                mask_, map_ = ostate['s']['mask'], ostate['s']['map']
                                                bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_, map_)
                                                if bb_mask.shape == param_arrays[c].shape and np.sum(bb_mask)==bb_mask.shape[0]*bb_mask.shape[1]: 
                                                    if are_mappable_arrays(bb_map, param_arrays[c]): matches_for_curr_a.append(o); store_bbmap = bb_map; store_parr = param_arrays[c]
                                            if len(matches_for_curr_a)!=1: is_specifiable_iobj = False; break
                                            else: 
                                                linked_iobj_list.append(matches_for_curr_a[0]) 
                                                unrolled_mapping_list[0].extend(store_bbmap.flatten())
                                                unrolled_mapping_list[1].extend(store_parr.flatten())

                                    

                                    if is_self_s_state or is_specifiable_iobj:
                                        if is_specifiable_iobj: print("ERROR")
                                        rule0 = check_mapping(unrolled_mapping_list[0], unrolled_mapping_list[1])
                                        if rule0 is not None:
                                            
                                            
                                            
                                            rule_cands['stackn'+str(stackn)].append({'description':'hyperp1 rule',
                                                                                    'prop':None,'rule':rule0,'found_set':pan_param_found_set,'serial_tr':curr_ref_serial_transform,'reverse_fn':'reverse_hyperp1','mode':0})


                                    

                                    return rule_cands

                                def reverse_hyperp1(m_states, first_currstack_rule):
                                    global gridn 
                                    gridn = gridn_

                                    rt, M, d = first_currstack_rule['rule']
                                    serial_tr = first_currstack_rule['serial_tr'] 
                                    mode = first_currstack_rule['mode']
                                    found_set = first_currstack_rule['found_set']

                                    if mode in [0]:
                                        
                                        m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                        for m in m_list:
                                            mask_, map_ = m_states[m]['s']['mask'], m_states[m]['s']['map']
                                            bb_mask, bb_map, tl_rc = get_bounding_box_object(mask_, map_)
                                            arr0 = np.zeros_like(bb_mask)

                                            filtered_options = [] 

                                            for r in range(arr0.shape[0]):
                                                for c in range(arr0.shape[1]):
                                                    propval = bb_map[r,c]; paramval = d 
                                                    for Mx in M:
                                                        if Mx['PROPS'] == propval:
                                                            paramval = Mx['PARAM']
                                                    arr0[r,c] = paramval
                                                    filtered_options.append([found_set[int(paramval)]])
                            
                                            rmult = arr0.shape[0]; cmult = arr0.shape[1]; occlusion_mask = None
                                            whole_param = {'tiling_details':{'filtered_options':filtered_options,'rmult':rmult,'cmult':cmult,'occlusion_mask':occlusion_mask}}
                                            

                                            state = m_states[m]
                                            curr_map, curr_mask = state['s']['map'], state['s']['mask']
                                            curr_fn = globals()[serial_tr]
                                            curr_params = whole_param
                                            curr_map, curr_mask = curr_fn(curr_map, curr_mask,**curr_params)
                                            state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map}    

                                    return m_states

                                if curr_ref_serial_transform == 'gridwise_tiled_copy': print('gridwise_tiled_copy/hyperp1'); rule_cands = hyperp1(m_states, rule_cands)


                                

                                def movt_to_common_pxl(m_states, rule_cands):
                                    



                                    a_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']; serial_tr = 'N/A'
                                    
                                    gridn_m_dict = {}
                                    for _,a in enumerate(a_list):
                                        if m_states[a]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[a]['gridn']] = []
                                        gridn_m_dict[m_states[a]['gridn']].append(a)

                                    
                                
                                    
                                    common_colors = []; common_tlrc = []
                                    for gridn in gridn_m_dict:
                                        
                                        common_mask = None; eg_map = None
                                        for a in gridn_m_dict[gridn]:
                                            if common_mask is None: common_mask = m_states[a]['e']['mask']; eg_map = m_states[a]['e']['map']
                                            else: common_mask = common_mask & m_states[a]['e']['mask']
                                        common_map = np.where(common_mask, eg_map, 0)
                                        
                                        
                                        combo_mask = None; eg_map = None
                                        for a in gridn_m_dict[gridn]:
                                            if combo_mask is None: combo_mask = m_states[a]['e']['mask']; eg_map = m_states[a]['e']['map']
                                            else: combo_mask = combo_mask | m_states[a]['e']['mask']                 
                                        bb_combo_mask, _, combo_tl_rc = get_bounding_box_object(combo_mask, eg_map) 
                                        common_tlrc.append(combo_tl_rc)
                                        

                                        if np.sum(common_mask)>0:
                                            colors = get_colors_of_obj(common_mask, common_map)
                                            
                                            
                                            
                                            common_colors.append(colors)
                                        else: common_colors.append([])

                                    
                                    color_mode = None
                                    if np.all([len(_)==1 for _ in common_colors]):
                                        color_mode = 0
                                    
                                    pos_mode = None
                                    if np.all([_==(0,0) for _ in common_tlrc]): 
                                        pos_mode = 0
                                    


                                    if color_mode is not None and pos_mode is not None:
                                        rule_cands['stackn'+str(stackn)].append({'description':'movt to common pxl', 'color_mode':color_mode,'pos_mode':pos_mode,
                                                                                    'serial_tr':serial_tr,'reverse_fn':'reverse_movt_to_common_pxl','mode':0})

                            
                                    return rule_cands

                                
                                def reverse_movt_to_common_pxl(m_states, first_currstack_rule):
                                    global gridn 
                                    gridn = gridn_

                                    serial_tr = first_currstack_rule['serial_tr']
                                    color_mode = first_currstack_rule['color_mode']
                                    pos_mode = first_currstack_rule['pos_mode']

                                    m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']

                                    
                                    
                                    m_colors = []
                                    for m in m_list:
                                        m_colors.append(get_colors_of_obj(m_states[m]['s']['mask'], m_states[m]['s']['map']))

                                    common_colors = []
                                    for color in m_colors[0]:
                                        if np.all([color in colors for colors in m_colors]):
                                            common_colors.append(color)


                                    
                                    match_tl_rcs = []
                                    for m in m_list:
                                        mask_,map_ = m_states[m]['s']['mask'], m_states[m]['s']['map']
                                        matchmask = np.zeros_like(mask_)
                                        for color in common_colors:
                                            matchmask = matchmask | (map_==color).astype(int)
                                        bb_matchmask, _, match_tl_rc = get_bounding_box_object(matchmask, map_)
                                        match_tl_rcs.append(match_tl_rc)
                                    

                                    
                                    fake_map = np.zeros((60,60)); fake_mask = np.zeros((60,60)) 
                                    movt_req = []
                                    for r,c in match_tl_rcs:
                                        movt_req.append((30-r,30-c))
                                    
                                    for c,m in enumerate(m_list):
                                        mask_,map_ = m_states[m]['s']['mask'], m_states[m]['s']['map']                
                                        temp_map = np.zeros((60,60)); temp_mask = np.zeros((60,60))
                                        temp_map[0:0+map_.shape[0],0:0+map_.shape[1]] = map_
                                        temp_mask[0:0+mask_.shape[0],0:0+mask_.shape[1]] = mask_
                                        map__, mask__   = movt(temp_map, temp_mask, movt_req[c]) 
                                        fake_map = np.where(mask__, map__, fake_map) 
                                        fake_mask = np.where(mask__, 1, fake_mask) 
                                    
                                    combo_mask, combo_map, fake_tl_rc_combo_cdt  = get_bounding_box_object(fake_mask, fake_map)


                                    
                                    

                                    realign_req_rc = ((30-fake_tl_rc_combo_cdt[0],30-fake_tl_rc_combo_cdt[1]))

                                    
                                    for c,m in enumerate(m_list):
                                        s_state = m_states[m]['s']
                                        s_mask, s_map = s_state['mask'], s_state['map']
                                        move_rc = (  movt_req[c][0]-30+realign_req_rc[0],   movt_req[c][1]-30+realign_req_rc[1]  )
                                        e_map, e_mask = movt(s_map, s_mask, move_rc)
                                        m_states[m]['e'] = {'obj':s_state['obj'],'mask':e_mask,'map':e_map}

                                    return m_states


                                if curr_ref_serial_transform == 'movt': print('movt/movt_to_common_pxl'); rule_cands = movt_to_common_pxl(m_states, rule_cands)


                                def movt_regionmatch(m_states, rule_cands):

                                    
                                    
                                    
                                    
                                    serial_tr = curr_ref_serial_transform



                                    matchmtds = {_:[] for _ in range(20)}; matchobjs = {_:[] for _ in range(20)}; requires_estates = {}
                                    for m, state in enumerate(m_states):
                                        if state['type']=='analogy':
                                            final_mask = state['e']['mask']
                                            initial_mask = state['s']['mask']
                                            
                                            
                                            if final_mask[0,0]==1: matchmtds[0].append(1); matchobjs[0].append(None)
                                            else: matchmtds[0].append(0); matchobjs[0].append(None)
                                            requires_estates[0] = False
                                            
                                            flag = False
                                            bbf_mask, bbf_map, tlf_rc = get_bounding_box_object(state['e']['mask'],state['e']['map'])
                                            for n, staten in enumerate(m_states): 
                                                if m == n: continue
                                                if state['gridn']!=staten['gridn']: continue
                                                bbe_mask, bbe_map, tle_rc = get_bounding_box_object(staten['e']['mask'],staten['e']['map'])
                                                bre_rc = (tle_rc[0]+bbe_mask.shape[0]-1, tle_rc[1]+bbe_mask.shape[1]-1)
                                                if tlf_rc == bre_rc: matchn = n; flag = True; break
                                                else: pass
                                            if flag: matchmtds[1].append(1); matchobjs[1].append(matchn)
                                            else: matchmtds[1].append(0); matchobjs[1].append(None)
                                            requires_estates[1] = True
                                            
                                            
                                    
                                    
                                    selectedmtds = []
                                    found = False
                                    for s1 in matchmtds:
                                        if matchmtds[s1]==[]: continue
                                        slist = matchmtds[s1]
                                        if len(slist)!=0 and 0 not in slist: found = True; selectedmtds = [s1]; break
                                    if not found:
                                        for s1 in matchmtds:
                                            if matchmtds[s1]==[]: continue
                                            for s2 in matchmtds:
                                                if matchmtds[s2]==[]: continue
                                                if s1==s2: continue
                                                s1list = matchmtds[s1]
                                                s2list = matchmtds[s2]
                                                slist = [(s1list[_]+s2list[_])==1 for _ in range(len(s1list))]
                                                if len(slist)!=0 and np.all(slist): found = True; selectedmtds = [s1,s2]; break
                                            if found==True: break
                                    
                                    if selectedmtds == []: return rule_cands


                                    
                                    
                                    m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                    the_rules = []
                                    for sel in selectedmtds:
                                        
                                        rule_ = None
                                        if 0 not in matchmtds[sel]: rule_ = 'select all objs'
                                        else:
                                            
                                            lbls_ = matchmtds[sel]
                                            for ranking_prop in [analogywise_centroid_vertpos, analogywise_centroid_horizpos, self_size]:
                                                raw_prop_list = ranking_prop(m_list, m_states)
                                                prop_list = rankingfy(raw_prop_list, m_list, m_states) 
                                                
                                                rule1 = check_mapping(prop_list, lbls_)
                                                if rule1 is not None: rule_ = [ranking_prop, *rule1]; break 
                                        
                                        

                                        
                                        
                                        
                                        
                                        main_m_list = [m_list[k] for k in range(len(matchmtds[sel])) if matchmtds[sel][k]==1]
                                        link_m_list = [matchobjs[sel][k] for k in range(len(matchmtds[sel])) if matchmtds[sel][k]==1]
                                        foundlink = False
                                        if not np.all([_ is None for _ in link_m_list]):
                                            for ranking_prop in [analogywise_centroid_horizpos, analogywise_centroid_vertpos, self_size]:
                                                raw_prop_list = ranking_prop(m_list, m_states)
                                                prop_list = rankingfy(raw_prop_list, m_list, m_states)
                                                main_prop_list = [prop_list[m_list.index(m)] for m in main_m_list]
                                                link_prop_list = [prop_list[m_list.index(m)] for m in link_m_list] 
                                                def offset_n(l1,l2):
                                                    if len(l1)!=len(l2): return None
                                                    offsets = [l1[_]-l2[_] for _ in range(len(l1))]
                                                    if are_all_identical(offsets): return offsets[0]
                                                    else: return None
                                                for rel_prop in [offset_n]: 
                                                    rule2 = rel_prop(main_prop_list, link_prop_list)
                                                    
                                                    if rule2 is not None:
                                                        foundlink = True
                                                        break
                                                if foundlink: break
                                        the_rules.append({'region method':sel, 'select mains':rule_, 'select links':[ranking_prop, rel_prop.__name__, rule2, main_m_list, link_m_list] if foundlink else None})
                                    
                                    
                                    


                                    
                                    


                                    
                                    seqs = []
                                    for rule in the_rules:
                                        if requires_estates[rule['region method']]:
                                            for k in range(len(rule['select links'][3])):
                                                seqs.append([link_m_list[k], main_m_list[k]]) 
                                    
                                    if len(seqs)==0: reverse_ordering_prop = None
                                    else:
                                        for ranking_prop in [analogywise_centroid_horizpos, analogywise_centroid_vertpos, self_size]:
                                            raw_prop_list = ranking_prop(m_list, m_states)
                                            prop_list = rankingfy(raw_prop_list, m_list, m_states)
                                            
                                            
                                            ordered_m_list = []
                                            for p in range(max(prop_list)+1):
                                                ixs = ixs_of_x_in_y(x=p,y=prop_list)
                                                ordered_m_list.extend([m_list[_] for _ in ixs])
                                            
                                            flag = True
                                            for seq in seqs:
                                                if ordered_m_list.index(seq[0]) < ordered_m_list.index(seq[1]): pass
                                                else: flag = False
                                            if flag: 
                                                
                                                reverse_ordering_prop = ranking_prop
                                                break

                                    rule_cands['stackn'+str(stackn)].append({'description':'regionmatch [7]', 
                                                                    'reverse_ordering_prop':reverse_ordering_prop, 'rules':the_rules,
                                                                    'serial_tr':serial_tr,'reverse_fn':'reverse_movt_regionmatch','mode':0})
                                    


                                    return rule_cands
                                
                                def reverse_movt_regionmatch(m_states, first_currstack_rule):
                                    global gridn 
                                    gridn = gridn_

                                    serial_tr = first_currstack_rule['serial_tr']
                                    reverse_ordering_prop = first_currstack_rule['reverse_ordering_prop']
                                    the_rules = first_currstack_rule['rules']

                                    m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                    
                                    
                                    raw_prop_list = reverse_ordering_prop(m_list, m_states)
                                    prop_list = rankingfy(raw_prop_list, m_list, m_states)
                                    
                                    ordered_m_list = []
                                    for p in range(max(prop_list)+1):
                                        ixs = ixs_of_x_in_y(x=p,y=prop_list)
                                        ordered_m_list.extend([m_list[_] for _ in ixs])
                                    
                                    
                                    

                                    
                                    
                                    
                                    r_sels = {}; r_mains = []
                                    for r_, rule in enumerate(the_rules):
                                        region_method = rule['region method']
                                        select_mains = rule['select mains']
                                        select_links = rule['select links']

                                        if select_mains == 'select all objs': sel_objs = m_list.copy()
                                        else: 
                                            fn_, rt, M, d = select_mains
                                            prop_list = fn_(m_list, m_states)
                                            param_list = reverse_mapping(prop_list, M, d, rt)
                                            sel_objs = [m_list[k] for k in range(len(m_list)) if param_list[k]==1]
                                        
                                        r_mains.append(sel_objs)
                                        for _ in sel_objs: r_sels[_] = r_
                                    reverse_ordering = [r_sels[_] for _ in ordered_m_list]
                                    


                                    for c, m in enumerate(ordered_m_list):
                                        r_ = reverse_ordering[c]
                                        all_curr_sel_objs = r_mains[r_]
                                        region_method = the_rules[r_]['region method']
                                        select_links = the_rules[r_]['select links']
                                        
                                        if select_links is None:
                                            

                                            
                                            if region_method == 0:
                                                s_state = m_states[m]['s']
                                                s_mask, s_map = s_state['mask'], s_state['map']
                                                bb_mask, bb_map, tl_rc = get_bounding_box_object(s_mask, s_map)
                                                e_mask = np.zeros_like(s_mask); e_map = np.zeros_like(s_map)
                                                e_mask[0:0+bb_mask.shape[0],0:0+bb_mask.shape[1]] = bb_mask
                                                e_map[0:0+bb_mask.shape[0],0:0+bb_mask.shape[1]] = bb_map
                                                curr_mask, curr_map = e_mask, e_map
                                                m_states[m]['e'] = {'obj':s_state['obj'],'mask':curr_mask,'map':curr_map}
                                                
                                            
                                        
                                        else:

                                            
                                            ranking_prop = select_links[0]
                                            rel_prop_name = select_links[1]
                                            rel_output = select_links[2]

                                            raw_prop_list = ranking_prop(m_list, m_states) 
                                            prop_list = rankingfy(raw_prop_list, m_list, m_states)
                                            
                                            if rel_prop_name == 'offset_n':
                                                m_of_linked_obj = m_list[prop_list.index(prop_list[m_list.index(m)] - rel_output)]
                                                


                                            
                                            if region_method == 1:
                                                
                                                s_state = m_states[m]['s']
                                                s_mask, s_map = s_state['mask'], s_state['map']
                                                bb1_mask, bb1_map, tl1_rc = get_bounding_box_object(s_mask, s_map)
                                                oe_state = m_states[m_of_linked_obj]['e']
                                                oe_mask, oe_map = oe_state['mask'], oe_state['map']
                                                bb2_mask, bb2_map, tl2_rc = get_bounding_box_object(oe_mask, oe_map)
                                                bre_rc = (tl2_rc[0]+bb2_mask.shape[0]-1, tl2_rc[1]+bb2_mask.shape[1]-1)
                                                
                                                
                                                e_mask = np.zeros_like(s_mask); e_map = np.zeros_like(s_map)
                                                e_mask[bre_rc[0]:bre_rc[0]+bb1_mask.shape[0],bre_rc[1]:bre_rc[1]+bb1_mask.shape[1]] = bb1_mask
                                                e_map[bre_rc[0]:bre_rc[0]+bb1_mask.shape[0],bre_rc[1]:bre_rc[1]+bb1_mask.shape[1]] = bb1_map
                                                curr_mask, curr_map = e_mask, e_map
                                                
                                                m_states[m]['e'] = {'obj':s_state['obj'],'mask':curr_mask,'map':curr_map}
                                                
                                            


                                    return m_states
                                
                                if curr_ref_serial_transform == 'movt': print('movt/movt_regionmatch'); rule_cands = movt_regionmatch(m_states, rule_cands)


                                def movt_extdir(m_states, rule_cands):

                                    
  
                                    

                                    def list_of_rc_tuples_to_extentdirection_tuples(list_of_rc):
                                        list_of_ed = []
                                        for r,c in list_of_rc:
                                            mag_r, mag_c = np.abs(r), np.abs(c)
                                            if mag_r == 0 and mag_c == 0: list_of_ed.append(None)
                                            elif (mag_r == 0 or mag_c == 0 or mag_r == mag_c):
                                                if mag_r == 0: extent = mag_c
                                                elif mag_c == 0: extent = mag_r
                                                else: extent = mag_r 
                                                direction = (int(np.clip(r,-1,1)), int(np.clip(c,-1,1)))
                                                list_of_ed.append((int(extent),direction))
                                            else: list_of_ed.append(None)
                                        return list_of_ed

                                    

                                    a_list_ = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                    o_list_ = [m for m in range(len(m_states)) if m_states[m]['type']=='other']
                                    
                                    
                                    a_movt_r_c_list = []
                                    for state in m_states:
                                        if state['type']=='analogy':
                                            move_rc = state['serial_param'][stackn]['move_rc'] if state['serial_transform'][stackn]['type']=='movt' else (0,0)
                                            a_movt_r_c_list.append(move_rc)
                                    a_movt_e_d_list_ = list_of_rc_tuples_to_extentdirection_tuples(a_movt_r_c_list)        

                                    
                                    if len(a_list_)==0: return rule_cands


                                    
                                    
                                    if None in a_movt_e_d_list_: 
                                        new_a_movt_e_d_list=[]; new_a_list=[]
                                        for k in range(len(a_movt_e_d_list_)):
                                            if a_movt_e_d_list_[k] is not None:
                                                new_a_movt_e_d_list.append(a_movt_e_d_list_[k])
                                                new_a_list.append(a_list_[k])
                                        a_movt_e_d_list = new_a_movt_e_d_list; a_list = new_a_list; o_list = o_list_
                                        static_vs_movt_selection_required = True
                                    else: 
                                        a_movt_e_d_list = a_movt_e_d_list_; a_list = a_list_; o_list = o_list_ 
                                        static_vs_movt_selection_required = False

                                    
                                    

                                    pos_o = []; interact_dets = []
                                    for i,m in enumerate(a_list):
                                        movt_ext = a_movt_e_d_list[i][0]
                                        movt_dirn_tuple = a_movt_e_d_list[i][1]; movt_dir = ['S','SW','SE','W','E','N','NW','NE'][[(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)].index(movt_dirn_tuple)]

                                        a_e_state_mask = m_states[m]['e']['mask']; a_s_state_mask = m_states[m]['s']['mask'] 
                                        target_mask, target_map = m_states[m]['e']['mask'], m_states[m]['e']['map']


                                        pos_o_ = []; interact_dets_ = []

                                        if movt_dir is not None:

                                            orth_dir = ['E','SE','NE','S','N','W','SW','NW'][['S','SW','SE','W','E','N','NW','NE'].index(movt_dir)]
                                            orth_type = 'todo'
                                            if orth_dir in ['N','S']: orth_type = 'v'
                                            if orth_dir in ['E','W']: orth_type = 'h'
                                            if orth_dir in ['NW','SE']: orth_type = 'lead'
                                            if orth_dir in ['NE','SW']: orth_type = 'anti'

                                            for o in o_list: 
                                                if m_states[m]['gridn']!=m_states[o]['gridn']: continue
                                                
                                                
                                                
                    
                                                mask, map = m_states[o]['s']['mask'], m_states[o]['s']['map'] 

                                                for special_prop in [fully_inline, first_bump]:
                                                    if special_prop(target_mask, target_map, movt_dir, orth_type, mask, map): 
                                                        
                                                        pos_o_.append(o)
                                                        interact_dets_.append({'type':'special','special':special_prop})
                                                for main_prop in [leading_edge, trailing_edge]:
                                                    
                                                    cdt_list1 = main_prop(target_mask, movt_dir)
                                                    for interact_prop in [leading_edge, trailing_edge, leading_border, trailing_border]:
                                                        
                                                        cdt_list2 = interact_prop(mask, movt_dir)
                                                        
                                                        
                                                        if is_any_x_in_y(x = cdt_list1, y = cdt_list2):
                                                            
                                                            pos_o_.append(o)
                                                            interact_dets_.append({'type':'matched','main':main_prop,'interact':interact_prop})
                                                        
                                                

                                        pos_o.append(pos_o_); interact_dets.append(interact_dets_)

                                    
                                    common_interact_dets = []; pos_o_by_interact_type = []
                                    for c, interact_type in enumerate(interact_dets[0]):
                                        if np.all([is_x_in_y(x=interact_type,y=_) for _ in interact_dets]):
                                            common_interact_dets.append(interact_type)
                                            
                                            
                                            temps = [ixs_of_x_in_y(x=interact_type,y=_) for _ in interact_dets]
                                            
                                            
                                            pos_o_by_interact_type.append([[pos_o[n][temps[n][m]] for m in range(len(temps[n]))] for n in range(len(temps))])

                                    
                                    all_o_by_interact_type = []
                                    for interact_type in common_interact_dets:

                                        all_o = []
                                        for i,m in enumerate(a_list):
                                            movt_ext = a_movt_e_d_list[i][0]
                                            movt_dirn_tuple = a_movt_e_d_list[i][1]; movt_dir = ['S','SW','SE','W','E','N','NW','NE'][[(1,0),(1,-1),(1,1), (0,-1),(0,1), (-1,0),(-1,-1),(-1,1)].index(movt_dirn_tuple)]
                                            
                                            
                                            orth_dir = ['E','SE','NE','S','N','W','SW','NW'][['S','SW','SE','W','E','N','NW','NE'].index(movt_dir)]
                                            orth_type = 'todo'
                                            if orth_dir in ['N','S']: orth_type = 'v'
                                            if orth_dir in ['E','W']: orth_type = 'h'
                                            if orth_dir in ['NW','SE']: orth_type = 'lead'
                                            if orth_dir in ['NE','SW']: orth_type = 'anti'

                                            all_o_ = []
                                            for e in range(50): 
                                                output_map, transformed_mask = movt(m_states[m]['s']['map'], m_states[m]['s']['mask'], (movt_dirn_tuple[0]*e,movt_dirn_tuple[1]*e))
                                                if np.sum(transformed_mask) == 0: break 
                                                

                                                sim_target_mask, sim_target_map = transformed_mask, output_map

                                                for o in o_list:
                                                    if m_states[m]['gridn']!=m_states[o]['gridn']: continue
                                                    
                                                    
                                                    
                    
                                                    mask, map = m_states[o]['s']['mask'], m_states[o]['s']['map'] 

                                                    for special_prop in [fully_inline, first_bump]:
                                                        if special_prop(sim_target_mask, sim_target_map, movt_dir, orth_type, mask, map): 
                                                            
                                                            if are_two_identical({'type':'special','special':special_prop}, interact_type):
                                                                
                                                                all_o_.append(o)

                                                    for main_prop in [leading_edge, trailing_edge]:
                                                        
                                                        cdt_list1 = main_prop(sim_target_mask, movt_dir)
                                                        for interact_prop in [leading_edge, trailing_edge, leading_border, trailing_border]:
                                                            
                                                            cdt_list2 = interact_prop(mask, movt_dir)
                                                            
                                                            
                                                            if is_any_x_in_y(x = cdt_list1, y = cdt_list2):
                                                                
                                                                if are_two_identical({'type':'matched','main':main_prop,'interact':interact_prop}, interact_type):
                                                                    
                                                                    all_o_.append(o)              
                                                            
                                                    
                                            all_o.append(all_o_)

                                        all_o_by_interact_type.append(all_o)

                                        
                                        
                                        
                                        


                                    
                                    if len(common_interact_dets)>1: print("WARNING")
                                    
                                    
                                    
                                    interaction_rule_cands = []
                                    for c in range(len(common_interact_dets)):
                                        main_o_list = a_list 
                                        pos_o_list = pos_o_by_interact_type[c]
                                        curr_interact_type = common_interact_dets[c]
                                        
                                        
                                        all_o_list = all_o_by_interact_type[c]


                                        main_list = []; interact_list = []; lbls = []; seq_labels = []
                                        for i in range(len(a_list)):
                                            main_list.append(a_list[i])
                                            for o in all_o_list[i]:
                                                interact_list.append(o)
                                                seq_labels.append(i)
                                                if o in pos_o_list[i]: lbls.append(1)
                                                else: lbls.append(0) 
                                        if 0 not in lbls: pass


                                        
                                        for seq_prop in [first_obj, obj_seqn]:
                                            prop_list = seq_prop(interact_list, main_list, seq_labels)
                                            rule1 = check_mapping(prop_list, lbls)
                                            if rule1 is not None:
                                                rt, M,d = rule1
                                                interaction_rule_cands.append({'interact_type':curr_interact_type,'interact_selection_rule':seq_prop,'matrix':M,'default':d,'ruletype':rt})
                                                
                                                
                                        
                                        
                                        

                                        


                                    
                                    qualifying_rules = []
                                    for c in range(len(common_interact_dets)): 
                                        curr_interact_type = common_interact_dets[c]
                                        
                                        for interaction_rule in interaction_rule_cands:
                                            if are_two_identical(interaction_rule['interact_type'], curr_interact_type):


                                                dir_val_arr = np.zeros((8,len(a_list)))
                                                for i,m in enumerate(a_list):
                                                    movt_ext = a_movt_e_d_list[i][0]
                                                    movt_dirn_tuple = a_movt_e_d_list[i][1]
                                                    
                                                    movt_dir = ['N','S','W','E','NW','SE','SW','NE'][[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)].index(movt_dirn_tuple)]
                                                    
                                                    orth_dir = ['E','SE','NE','S','N','W','SW','NW'][['S','SW','SE','W','E','N','NW','NE'].index(movt_dir)]
                                                    orth_type = 'todo'
                                                    if orth_dir in ['N','S']: orth_type = 'v'
                                                    if orth_dir in ['E','W']: orth_type = 'h'
                                                    if orth_dir in ['NW','SE']: orth_type = 'lead'
                                                    if orth_dir in ['NE','SW']: orth_type = 'anti'

                                                    for d, curr_dirn in enumerate([(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)]):
                                                        curr_dir = ['N','S','W','E','NW','SE','SW','NE'][[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)].index(curr_dirn)]

                                                        curr_orth_dir = ['E','SE','NE','S','N','W','SW','NW'][['S','SW','SE','W','E','N','NW','NE'].index(curr_dir)]
                                                        curr_orth_type = 'todo'
                                                        if curr_orth_dir in ['N','S']: curr_orth_type = 'v'
                                                        if curr_orth_dir in ['E','W']: curr_orth_type = 'h'
                                                        if curr_orth_dir in ['NW','SE']: curr_orth_type = 'lead'
                                                        if curr_orth_dir in ['NE','SW']: curr_orth_type = 'anti'

                                                        o_cands = []
                                                        for e in range(50): 
                                                            output_map, transformed_mask = movt(m_states[m]['s']['map'], m_states[m]['s']['mask'], (curr_dirn[0]*e,curr_dirn[1]*e))
                                                            if np.sum(transformed_mask) == 0: break 
                                                            

                                                            sim_target_mask, sim_target_map = transformed_mask, output_map
                                                            

                                                            for o in o_list:
                                                                if m_states[m]['gridn']!=m_states[o]['gridn']: continue
                                                                
                                                                
                                                                
                    
                                                                mask, map = m_states[o]['s']['mask'], m_states[o]['s']['map'] 

                                                                

                                                                for special_prop in [fully_inline, first_bump]:
                                                                    tflag = True if i==0 and curr_dirn == (1,1) else False
                                                                    if special_prop(sim_target_mask, sim_target_map, curr_dir, curr_orth_type, mask, map, tflag): 
                                                                        
                                                                        if are_two_identical({'type':'special','special':special_prop}, curr_interact_type):
                                                                            o_cands.append(o)

                                                                for main_prop in [leading_edge, trailing_edge]:
                                                                    
                                                                    cdt_list1 = main_prop(sim_target_mask, curr_dir)
                                                                    for interact_prop in [leading_edge, trailing_edge, leading_border, trailing_border]:
                                                                        
                                                                        cdt_list2 = interact_prop(mask, curr_dir)
                                                                        
                                                                        
                                                                        if is_any_x_in_y(x = cdt_list1, y = cdt_list2):
                                                                            
                                                                            if are_two_identical({'type':'matched','main':main_prop,'interact':interact_prop}, curr_interact_type):
                                                                                o_cands.append(o)          
                                        
                                                                        
                                                                

                                                        
                                                        
                                                        prop_list_ = interaction_rule['interact_selection_rule'](o_cands, [m], [0]*len(o_cands))
                                                        prop_list = prop_list_[0] if len(prop_list_)!=0 else []
                                                        param_list = reverse_mapping(prop_list,interaction_rule['matrix'],interaction_rule['default'],interaction_rule['ruletype'])
                                                        
                                                        
                                                        
                                                        if sum(param_list)==1: dir_val_arr[d,i] = 1


                                                
                                                

                                                candidate_dirn_modes = []
                                                dirn_modes = ['all_dirs','main_dirs_only','alt_dirs_only','vert_dirs_only','horiz_dirs_only','diag1_dirs_only','diag2_dirs_only']
                                                for dm in range(len(dirn_modes)):
                                                    if dirn_modes[dm] == 'all_dirs': drange = [0,1,2,3,4,5,6,7]
                                                    if dirn_modes[dm] == 'main_dirs_only': drange = [0,1,2,3]
                                                    if dirn_modes[dm] == 'alt_dirs_only': drange = [4,5,6,7]
                                                    if dirn_modes[dm] == 'vert_dirs_only': drange = [0,1]
                                                    if dirn_modes[dm] == 'horiz_dirs_only': drange = [2,3]
                                                    if dirn_modes[dm] == 'diag1_dirs_only': drange = [4,5]
                                                    if dirn_modes[dm] == 'diag2_dirs_only': drange = [6,7]

                                                    modified_dir_val_arr = dir_val_arr[drange,:]
                                                    
                                                    flag = True
                                                    for n in range(len(a_list)):
                                                        if sum(modified_dir_val_arr[:,n])!=1: flag = False
                                                    if flag: 
                                                        
                                                        candidate_dirn_modes.append(dirn_modes[dm])
                                                if len(candidate_dirn_modes)!=0:
                                                    
                                                    chosen_dirn_rule = candidate_dirn_modes[0]
                                                    

                                                    qualifying_rules.append([interaction_rule, chosen_dirn_rule]) 






                                    if not static_vs_movt_selection_required:
                                        for qr in qualifying_rules:
                                            rule_cands['stackn'+str(stackn)].append({'description':'movt ext/dir','interaction':qr[0],'dirn':qr[1],'reverse_fn':'reverse_movt_extdir','mode':0})

                                    else:
                                        
                                        lbls = []
                                        for k in range(len(a_list_)):
                                            if a_movt_e_d_list_[k] is None: lbls.append(0)
                                            else: lbls.append(1)
                                        
                                        

                                        for self_prop in [presence_of_opposite_diffcolor_obj]:
                                            prop_list = self_prop( a_list_, m_states)
                                            rule = check_mapping(prop_list, lbls)
                                            if rule is not None:
                                                objsel_ = {'rule':rule, 'prop':self_prop}
                                            for qr in qualifying_rules:
                                                rule_cands['stackn'+str(stackn)].append({'description':'movt ext/dir','objsel':objsel_,'interaction':qr[0],'dirn':qr[1],'reverse_fn':'reverse_movt_extdir','mode':1})




                                    return rule_cands
                                
                                def reverse_movt_extdir(m_states, first_currstack_rule):
                                    global gridn 
                                    gridn = gridn_

                                    
                                    
                                    
                                    
                                    
                                    mode = first_currstack_rule['mode']
                                    interaction_rule = first_currstack_rule['interaction']
                                    direction = first_currstack_rule['dirn']
                                    
                                    curr_interact_type = interaction_rule['interact_type']

                                    if direction == 'all_dirs': drange = [0,1,2,3,4,5,6,7]
                                    if direction == 'main_dirs_only': drange = [0,1,2,3]
                                    if direction == 'alt_dirs_only': drange = [4,5,6,7]
                                    if direction == 'vert_dirs_only': drange = [0,1]
                                    if direction == 'horiz_dirs_only': drange = [2,3]
                                    if direction == 'diag1_dirs_only': drange = [4,5]
                                    if direction == 'diag2_dirs_only': drange = [6,7]
                                    allowed_dirns = [['N','S','W','E','NW','SE','SW','NE'][_] for _ in drange]

                                    a_list_ = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                    o_list_ = [m for m in range(len(m_states)) if m_states[m]['type']=='other']
                                    
                                    


                                    if mode == 0: a_list = a_list_; o_list = o_list_
                                    if mode == 1: 
                                        objsel = first_currstack_rule['objsel']
                                        rt, M, d = objsel['rule']
                                        fn_ = objsel['prop']
                                        prop_list = fn_(a_list_, m_states)
                                        param_list = reverse_mapping(prop_list, M, d, rt)
                                        a_list = [] 
                                        static_a_list = [] 
                                        for k in range(len(param_list)):
                                            if param_list[k]==1: a_list.append(a_list_[k])
                                            else: static_a_list.append(a_list_[k])
                                        o_list = o_list_


                                    for i,m in enumerate(a_list):
                                        chosen_rc = None
                                        for d, curr_dirn in enumerate([(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)]):
                                            curr_dir = ['N','S','W','E','NW','SE','SW','NE'][[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)].index(curr_dirn)]

                                            if curr_dir not in allowed_dirns: continue 

                                            curr_orth_dir = ['E','SE','NE','S','N','W','SW','NW'][['S','SW','SE','W','E','N','NW','NE'].index(curr_dir)]
                                            curr_orth_type = 'todo'
                                            if curr_orth_dir in ['N','S']: curr_orth_type = 'v'
                                            if curr_orth_dir in ['E','W']: curr_orth_type = 'h'
                                            if curr_orth_dir in ['NW','SE']: curr_orth_type = 'lead'
                                            if curr_orth_dir in ['NE','SW']: curr_orth_type = 'anti'

                                            o_cands = []; e_cands = []
                                            for e in range(50): 
                                                output_map, transformed_mask = movt(m_states[m]['s']['map'], m_states[m]['s']['mask'], (curr_dirn[0]*e,curr_dirn[1]*e))
                                                if np.sum(transformed_mask) == 0: break 
                                                

                                                sim_target_mask, sim_target_map = transformed_mask, output_map
                                                

                                                for o in o_list:
                                                    if m_states[m]['gridn']!=m_states[o]['gridn']: continue
                                                    
                                                    
                                                    
                    
                                                    mask, map = m_states[o]['s']['mask'], m_states[o]['s']['map'] 

                                                    
                                                    
                                                    if curr_interact_type['type'] == 'special':

                                                        special_prop  = curr_interact_type['special']

                                                        tflag = True
                                                        if special_prop(sim_target_mask, sim_target_map, curr_dir, curr_orth_type, mask, map, tflag): 
                                                            
                                                            
                                                            o_cands.append(o)
                                                            e_cands.append(e)



                                                    if curr_interact_type['type'] == 'matched':

                                                        main_prop = curr_interact_type['main']
                                                        

                                                        cdt_list1 = main_prop(sim_target_mask, curr_dir)
                                                    
                                                        interact_prop = curr_interact_type['interact']
                                                        
                                                        cdt_list2 = interact_prop(mask, curr_dir)
                                                        
                                                        
                                                        if is_any_x_in_y(x = cdt_list1, y = cdt_list2):
                                                            
                                                            
                                                            o_cands.append(o)    
                                                            e_cands.append(e)    


                        
                                                        
                                                

                                            
                                            




                                            
                                            


                                            prop_list_ = interaction_rule['interact_selection_rule'](o_cands, [m], [0]*len(o_cands)) 
                                            prop_list = prop_list_[0] if len(prop_list_)!=0 else []
                                            param_list = reverse_mapping(prop_list,interaction_rule['matrix'],interaction_rule['default'],interaction_rule['ruletype'])
                                            
                                            
                                            
                                            
                                            

                                            e_opts = [e_cands[_] for _ in range(len(param_list)) if param_list[_]==1]
                                            if len(e_opts)>0:
                                                
                                                chosen_extdir = (e_opts[0],curr_dirn)
                                                def list_of_extentdirn_tuples_to_rc_tuples(list_of_ed):
                                                    list_of_rc = []
                                                    for e,d in list_of_ed:
                                                        r,c = e*d[0], e*d[1]
                                                        list_of_rc.append((r,c))
                                                    return list_of_rc
                                                chosen_rc = list_of_extentdirn_tuples_to_rc_tuples([chosen_extdir])[0]
                                                
                                        if chosen_rc is None: print('ERROR'); continue
                                        state = m_states[m]
                                        curr_map, curr_mask = state['s']['map'], state['s']['mask']
                                        curr_fn = globals()['movt']
                                        curr_params = {'move_rc':chosen_rc}
                                        curr_map, curr_mask = curr_fn(curr_map, curr_mask,**curr_params)
                                        state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map}       


                                    if mode == 1: 
                                        for m in static_a_list:
                                            state = m_states[m]
                                            curr_map, curr_mask = state['s']['map'], state['s']['mask']
                                            state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map} 


                                    return m_states

                                if curr_ref_serial_transform == 'movt': print('movt/movt_extdir'); rule_cands = movt_extdir(m_states, rule_cands)
                                
                                
                                def some_colors_maintained_others_disappear(m_states, rule_cands): 

                                    a_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                    o_list = [m for m in range(len(m_states)) if m_states[m]['type']=='other']
                                    
                                    
                    
                                    colors_maintained_list = []
                                    colors_disappeared_list = []
                                    grid_list = []
                                    per_grid = {}
                                    for state in m_states:
                                        if state['type']=='analogy':
                                            colors_maintained, colors_disappeared, applied_selector_large_on_both_masks = state['serial_param'][stackn]['colors_maintained'], state['serial_param'][stackn]['colors_disappeared'], state['serial_param'][stackn]['applied_selector_large_on_both_masks']
                                            colors_maintained_list.append(colors_maintained)    
                                            colors_disappeared_list.append(colors_disappeared)
                                            if state['gridn'] not in per_grid: per_grid[state['gridn']] = 0
                                            per_grid[state['gridn']] += 1; grid_list.append(state['gridn'])


                                    
                                    if selector_presence:
                                        if np.all([per_grid[_]==1 for _ in per_grid]):
                                            
                                            valid_ms = []; valid_lbls = []
                                            for n in range(len(grid_list)):
                                                gridn = grid_list[n]
                                                selector_small = selector_regions[gridn]['small']
                                                
                                                for m, state in enumerate(m_states):
                                                    if state['type']=='other':
                                                        if state['gridn']!=gridn: continue
                                                        mask_,map_ = state['s']['mask'], state['s']['map']
                                                        if np.sum((selector_small==0)&(mask_==1))==0: 
                                                            valid_ms.append(m)
                                                            if get_colors_of_obj(mask_,map_) in colors_disappeared_list[n]: valid_lbls.append(1)
                                                            else: valid_lbls.append(0)
                                            
                                            if 0 not in valid_lbls: rule = 'select_all'; self_prop = None
                                            else:
                                                for self_prop in [self_color, self_shape, self_parsing, self_color_AND_shape]:
                                                    prop_list = self_prop(valid_ms, m_states)
                                                    rule = check_mapping(prop_list, valid_lbls)
                                                    if rule is not None: pass
                                                        

                                            rule_cands['stackn'+str(stackn)].append({'description':'some_colors_maintained_others_disappear mode 0',
                                                                                    'prop':self_prop,'rule':rule,'reverse_fn':'reverse_some_colors_maintained_others_disappear','mode':0})
                                            

                                    

                                    
                                    

                                    return rule_cands

                                def reverse_some_colors_maintained_others_disappear(m_states, first_currstack_rule):
                                    global gridn 
                                    gridn = gridn_

                                    mode = first_currstack_rule['mode']
                                    

                                    if mode == 0:

                                        m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']


                                        curr_o_list = []
                                        selector_small = selector_regions[gridn]['small']
                                        for m, state in enumerate(m_states):
                                            if state['type']=='other':
                                                if state['gridn']!=gridn: continue
                                                mask_,map_ = state['s']['mask'], state['s']['map']
                                                if np.sum((selector_small==0)&(mask_==1))==0: 
                                                    curr_o_list.append(m)


                                        rule = first_currstack_rule['rule']
                                        if rule == 'select_all': 
                                            valid_ms = curr_o_list
                                        else:
                                            rt, M, d = rule
                                            fn_ = first_currstack_rule['prop']
                                            prop_list = fn_(curr_o_list, m_states)
                                            param_list = reverse_mapping(prop_list, M, d, rt)
                                            valid_ms = [curr_o_list[n] for n in range(len(curr_o_list)) if param_list[n]==1]
                                        
                                        
                                        colors_to_disappear = []

                                        for o in valid_ms:
                                            mask_,map_ = m_states[o]['s']['mask'], m_states[o]['s']['map']
                                            colrs = get_colors_of_obj(mask_,map_)
                                            colors_to_disappear.extend(colrs)


                                        
                                        colors_to_maintain = []

                                        for m in m_list:
                                            mask_,map_ = m_states[m]['s']['mask'], m_states[m]['s']['map']
                                            colrs = get_colors_of_obj(mask_,map_)
                                            for colr in colrs:
                                                if colr not in colors_to_disappear:
                                                    colors_to_maintain.append(colr)
                                    
                                        for m in m_list:
                                            state = m_states[m]
                                            curr_map, curr_mask = state['s']['map'], state['s']['mask']
                                            curr_fn = globals()['mask_of_all_colors_maintained_or_disappeared']
                                            curr_params = {'colors_maintained':colors_to_maintain,'colors_disappeared':colors_to_disappear,'applied_selector_large_on_both_masks':True}
                                            curr_map, curr_mask = curr_fn(curr_map, curr_mask,**curr_params)
                                            state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map}          
                                                            
                                    
                                    return m_states


                                if curr_ref_serial_transform == 'mask_of_all_colors_maintained_or_disappeared': print('mask_of_all_colors_maintained_or_disappeared/some_colors_maintained_others_disappear'); rule_cands = some_colors_maintained_others_disappear(m_states, rule_cands)

                                def general_multicategory(m_states, rule_cands):


                                    
        
                                    
                                    


                                    
                                    
                                    
                                    a_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']; serial_tr = 'N/A'
                                    o_list = [m for m in range(len(m_states)) if m_states[m]['type']=='other']
                                    a_serials_list = [(state['serial_transform'][stackn]['type'],state['serial_param'][stackn]) for state in m_states if state['type']=='analogy']
                                    lbls, uniques = label_unique_with_IDs(a_serials_list)
                                    if len(uniques)==1: print("Quit")
                                    



                                    
                                    for self_prop in [self_color]:
                                        prop_list = self_prop(a_list, m_states)
                                        rule1 = check_mapping(prop_list, lbls)
                                        if rule1 is not None:
                                            
                                            
                                            
                                            rule_cands['stackn'+str(stackn)].append({'description':'multi-category rule, self prop '+self_prop.__name__,
                                                                                    'prop':self_prop,'rule':rule1,'params':uniques,'serial_tr':serial_tr,'reverse_fn':'reverse_general_multicategory','mode':0})

                                    
                                    for gridwise_prop in [analogywise_centroid_vertpos, analogywise_centroid_horizpos, analogywise_tl_vertpos, analogywise_tl_horizpos]: 
                                        prop_list = gridwise_prop(a_list, m_states)
                                        
                                        rule1 = check_mapping(prop_list, lbls)
                                        if rule1 is not None:
                                            
                                            
                                            
                                            rule_cands['stackn'+str(stackn)].append({'description':'multi-category rule, gridwise prop '+gridwise_prop.__name__,
                                                                                    'prop':gridwise_prop,'rule':rule1,'params':uniques,'serial_tr':serial_tr,'reverse_fn':'reverse_general_multicategory','mode':1})

                                    
                                    for ranking_prop in [self_size, analogywise_centroid_vertpos]:
                                        raw_prop_list = ranking_prop(a_list, m_states)
                                        prop_list = rankingfy(raw_prop_list, a_list, m_states)
                                        
                                        rule1 = check_mapping(prop_list, lbls)
                                        if rule1 is not None:
                                            
                                            
                                            
                                            rule_cands['stackn'+str(stackn)].append({'description':'multi-category rule, ranking prop '+ranking_prop.__name__,
                                                                                    'prop':ranking_prop,'rule':rule1,'params':uniques,'serial_tr':serial_tr,'reverse_fn':'reverse_general_multicategory','mode':2})




                                    gridn_m_dict = {}
                                    for _,m in enumerate(a_list):
                                        if m_states[m]['gridn'] not in gridn_m_dict: gridn_m_dict[m_states[m]['gridn']] = []
                                        gridn_m_dict[m_states[m]['gridn']].append(m)


                                    matches ={}
                                    for gridn in gridn_m_dict:
                                        matches[gridn] = []
                                        
                                        anti_mask = np.zeros_like(i_grids[gridn])
                                        for iobj in get_iobjs_of_parsing_type('background', initial_global_parsings, gridn, 'i'):
                                            anti_mask = anti_mask | initial_global_parsings[gridn]['i'][iobj]['maskv']

                                        for m in gridn_m_dict[gridn]: 
                                            mask_ = m_states[m]['s']['mask'] 
                                            
                                            anti_mask = anti_mask | mask_
                                            
                                        relevant_mask = (anti_mask==0).astype(int)
                                        relevant_objs = []
                                        for iobj in global_parsings[gridn]['i']:
                                            i_mask, i_map, i_maskv, i_masko, i_obj_parsing_type = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['maskv'], global_parsings[gridn]['i'][iobj]['masko'], global_parsings[gridn]['i'][iobj]['properties']['parsing_description']
                                            
                                            
                                            if are_two_identical(i_maskv, relevant_mask):
                                                relevant_objs.append(iobj)
                                                matches[gridn].append([iobj, i_obj_parsing_type])
                                    
                                    chosen_opt = None
                                    g0 = [k for k in gridn_m_dict][0]
                                    first_opts = [_[1] for _ in matches[g0]]                                
                                    for opt in first_opts:
                                        flag = True
                                        for gridn in gridn_m_dict:
                                            curr_opts = [_[1] for _ in matches[gridn]]
                                            if not is_x_in_y(x=opt, y=curr_opts):
                                                flag = False
                                        if flag: chosen_opt = opt; break
                                    

                                    if chosen_opt is not None:
                                        

                                        
                                        a_gridns =  [m_states[m]['gridn'] for m in range(len(m_states)) if m_states[m]['type']=='analogy']

                                        if np.all([a_gridns.count(_)==1 for _ in a_gridns]):

                                            k_states = []
                                            for gridn_ in a_gridns:
                                                for match in matches[gridn_]:
                                                    if match[1] == chosen_opt:
                                                        iobj_ = match[0]
                                                        k_states.append({'type':'k_state','gridn':gridn_,'s':{'obj':iobj_,'mask':global_parsings[gridn_]['i'][iobj_]['mask'],'map':global_parsings[gridn_]['i'][iobj_]['map'],'parsing':global_parsings[gridn_]['i'][iobj_]['properties']['parsing_description']}})
                                                        break
                                                    
                                            for interact_prop in [self_color, self_shape, self_size]:
                                                prop_list = interact_prop(list(range(len(k_states))), k_states)
                                                rule1 = check_mapping(prop_list, lbls)
                                                if rule1 is not None:
                                                    rule_cands['stackn'+str(stackn)].append({'description':'multi-category rule, interact prop '+interact_prop.__name__,
                                                                                            'selection_parsing':chosen_opt, 'prop':interact_prop, 'rule':rule1,'params':uniques,'serial_tr':serial_tr,'reverse_fn':'reverse_general_multicategory','mode':4})




                                    
                                    return rule_cands 

                                def reverse_general_multicategory(m_states, first_currstack_rule):
                                    global gridn 
                                    gridn = gridn_

                                    mode = first_currstack_rule['mode']

                                    if mode in [0,1,2]:
                                        
                                        rt, M, d = first_currstack_rule['rule']
                                        fn_ = first_currstack_rule['prop']
                                        
                                        mode = first_currstack_rule['mode']
                                        
                                        serials = first_currstack_rule['params']
                                        serial_trs, serial_params = [_[0] for _ in serials], [_[1] for _ in serials]

                                        m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                        prop_list = fn_(m_list, m_states)

                                        if mode == 2: 
                                            prop_list = rankingfy(prop_list, m_list, m_states)

                                        param_list = reverse_mapping(prop_list, M, d, rt)

                                        

                                        
                                        if None in param_list: print("ERROR")
                                        

                                
                                        
                                        serial_transform_list = [serial_trs[_] for _ in param_list]
                                        serial_param_list = [serial_params[_] for _ in param_list]

                                        for c, m in enumerate(m_list):
                                            state = m_states[m]
                                            curr_map, curr_mask = state['s']['map'], state['s']['mask']
                                            
                                            
                                            curr_fn = globals()[serial_transform_list[c]]
                                            curr_params = serial_param_list[c]
                                            curr_map, curr_mask = curr_fn(curr_map, curr_mask,**curr_params)
                                            state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map}        
                                        
                                    if mode in [4]: 


                                        rt, M, d = first_currstack_rule['rule']
                                        fn_ = first_currstack_rule['prop']
                                        mode = first_currstack_rule['mode']
                                        serials = first_currstack_rule['params']
                                        serial_trs, serial_params = [_[0] for _ in serials], [_[1] for _ in serials]

                                        m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                        
                                        

                                        serial_transform_list = [serial_trs[_] for _ in param_list]
                                        serial_param_list = [serial_params[_] for _ in param_list]

                                        chosen_opt = first_currstack_rule['selection_parsing']

                                        anti_mask = np.zeros_like(i_grids[gridn])
                                        for iobj in get_iobjs_of_parsing_type('background', initial_global_parsings, gridn, 'i'):
                                            anti_mask = anti_mask | initial_global_parsings[gridn]['i'][iobj]['maskv']

                                        for m in m_list: 
                                            mask_ = m_states[m]['s']['mask']
                                            anti_mask = anti_mask | mask_

                                        relevant_mask = (anti_mask==0).astype(int)
                                        k_states = []
                                        for iobj in global_parsings[gridn]['i']:
                                            i_mask, i_map, i_maskv, i_masko, i_obj_parsing_type = global_parsings[gridn]['i'][iobj]['mask'], global_parsings[gridn]['i'][iobj]['map'], global_parsings[gridn]['i'][iobj]['maskv'], global_parsings[gridn]['i'][iobj]['masko'], global_parsings[gridn]['i'][iobj]['properties']['parsing_description']
                                            if are_two_identical(i_maskv, relevant_mask) and are_two_identical( chosen_opt, i_obj_parsing_type ):
                                                k_states.append({'type':'k_state','gridn':gridn,'s':{'obj':iobj,'mask':global_parsings[gridn]['i'][iobj]['mask'],'map':global_parsings[gridn]['i'][iobj]['map'],'parsing':global_parsings[gridn]['i'][iobj]['properties']['parsing_description']}})
                                                break 
                                        
                                        prop_list = fn_(list(range(len(k_states))), k_states)
                                        param_list = reverse_mapping(prop_list, M, d, rt)




                                    if mode in [3]:

                                        
                                        

                                        main_rule = first_currstack_rule['main_rule']
                                        mrt, mM, md = main_rule['rule']
                                        mfn_ = main_rule['prop']
                                        link_rule = first_currstack_rule['link_rule']
                                        lrt, lM, ld = link_rule['rule']
                                        lfn_ = link_rule['prop']
                                        
                                        
                                        serials = first_currstack_rule['params']
                                        serial_trs, serial_params = [_[0] for _ in serials], [_[1] for _ in serials]
                                        
                                        
                                        prop_list = lfn_(list(range(len(m_states))), m_states)
                                        param_list = reverse_mapping(prop_list, lM, ld, lrt)
                                        

                                        
                                        l_list = [m for m in range(len(m_states)) if param_list[m]==1]
                                        prop_list = mfn_(l_list, m_states)
                                        param_list = reverse_mapping(prop_list, mM, md, mrt)
                                        
                                        
                                        serial_transform_list = [serial_trs[_] for _ in param_list]
                                        serial_param_list = [serial_params[_] for _ in param_list]

                                        
                                        m_list = [m for m in range(len(m_states)) if m_states[m]['type']=='analogy']
                                        if len(m_list) != len(param_list): print("ERROR")
                                        for c, m in enumerate(m_list):
                                            state = m_states[m]
                                            curr_map, curr_mask = state['s']['map'], state['s']['mask']
                                            
                                            
                                            curr_fn = globals()[serial_transform_list[c]]
                                            curr_params = serial_param_list[c]
                                            curr_map, curr_mask = curr_fn(curr_map, curr_mask,**curr_params)
                                            state['e'] = {'obj':state['s']['obj'],'mask':curr_mask,'map':curr_map}        

                                    return m_states
                                
                                if curr_ref_serial_transform not in ['iobjs_tile_creation', 'connection','hyperp_gridmap_creation','mask_of_all_colors_maintained_or_disappeared', 'slotting_hyperp_obj']:
                                    
                                    
                                    
                                    print('/general_multicategory'); rule_cands = general_multicategory(m_states, rule_cands)

                            except: print('ERROR, Skipped')

                                    

                    
                    valids = []
                    for gridn in range(num_demo_grids):
                        try:
                            e_masks = [state['e']['mask'] for state in m_states if state['gridn']==gridn and state['type']=='analogy']
                        except: valids.append(False); continue 

                        if np.any([e_mask.shape!=o_grids[gridn].shape for e_mask in e_masks]):
                            
                            
                            

                            anyperfect = False; validity = True
                            for e_mask in e_masks:
                                rows_,cols_ = np.where(e_mask==1)
                                corner_br_rc = (max(rows_), max(cols_))
                                if o_grids[gridn].shape == (corner_br_rc[0]+1,corner_br_rc[1]+1): anyperfect = True
                                if corner_br_rc[0]+1 > o_grids[gridn].shape[0] or corner_br_rc[1]+1 > o_grids[gridn].shape[1]: validity = False 

                            if anyperfect and validity: valids.append(True)
                            else: valids.append(False) 

                        else: valids.append(False) 




                    if len(valids)>0 and np.all(valids):
                        rule_cands['gridsize_restriction'].append({'description':'restrict to this groups bbmask'})


                    

                    tobreak = False 
                    if np.all([rule_cands['stackn'+str(stackn)]!=[] for stackn in range(num_stacks)]) and rule_cands['obj_select']!=[]:
                        
                        if omaskv_explantionsum[grn] > 0:
                            
                            chosen_groups.append(grn)
                            tobreak = True
                            anyfound = True
                            block_analogies[grn] = 1 
                            chosen_rule_cands.append(rule_cands) 

                            current_analogy_o_maskvs = analogy_o_maskvs[grn]

                            for i in range(len(analogy_o_maskvs)):
                                if i == grn: continue
                                
                                
                                
                                flag1 = True
                                for g in range(num_demo_grids):
                                    if np.sum((analogy_o_maskvs[i][g]==1) & (current_analogy_o_maskvs[g]==0))==0: 
                                        flag1 = False
                                if not flag1: block_analogies[i] = 1

                    if not tobreak:
                        
                        if grn in web[str(chosen_groups)]: web[str(chosen_groups)].remove(grn)
                    else:
                        unedited_chosen_groups = chosen_groups[:-1] 
                        if grn in web[str(unedited_chosen_groups)]: web[str(unedited_chosen_groups)].remove(grn)


                    if tobreak: break

                
                cumulative_omaskv = {_:[] for _ in range(num_demo_grids)}
                for grn_ in chosen_groups:
                    current_analogy_o_maskvs = analogy_o_maskvs[grn_]
                    for g in current_analogy_o_maskvs:
                        cumulative_omaskv[g].append(current_analogy_o_maskvs[g])
                flag = True
                for g in cumulative_omaskv:
                    combined = [0,0,0]
                    for c in range(len(cumulative_omaskv[g])):
                        if c==0: combined = cumulative_omaskv[g][c]
                        else: combined = combined | cumulative_omaskv[g][c]
                    if 0 in combined: flag = False; break
                if flag: 
                    print(' ---------- A solution found',chosen_groups); 
                    solns.append(chosen_groups); solnrulecands.append(chosen_rule_cands) 
                    web[str(chosen_groups)] = False 
                    
                    break
                if not anyfound: 
                    web[str(chosen_groups)] = False 
                    
                    print(' ----------- Quitting w no solution'); break

            
        print('Solutions temppp1 indices:',solns)
        aa.e()
        

        


        if esc(): breaker = 'breaker' / 2


        
        soln_lens = [len(_) for _ in solns]
        argsort_lens = np.argsort(soln_lens)
        sorted_solns, sorted_solnrulecands = [solns[_] for _ in argsort_lens], [solnrulecands[_] for _ in argsort_lens]


        
        quitentirely = False

        for soln_num in range(len(sorted_solns)):
            print('soln', soln_num, sorted_solns[soln_num])
            chosen_rule_cands = sorted_solnrulecands[soln_num]


            

            sequenced_all_rule_cands = chosen_rule_cands 


            
            stored_indices = []
            for r0 in range(len(sequenced_all_rule_cands)):
                curr_indices = {}
                if 'obj_select' in sequenced_all_rule_cands[r0]: curr_indices['obj_select'] = 0 
                for stackn in range(10):
                    if 'stackn'+str(stackn) in sequenced_all_rule_cands[r0]: curr_indices['stackn'+str(stackn)] = 0 
                stored_indices.append(curr_indices)


            for retryn in range(10):
                retry = False
                quitsoln = False

                
                current_solution_eval_recons = []
                for gridn_ in range(num_demo_grids+num_test_grids):    

                    if esc(): break

                    

                    restriction_shape = None

                    persistent_states = []


                    append_o_states = []

                    
                    for r_, rule_cands_ in enumerate(sequenced_all_rule_cands):

                        
                        curr_stored_ix = stored_indices[r_]['obj_select']
                        
                        if curr_stored_ix > len(rule_cands_['obj_select'])-1: quitsoln =True; break
                        obj_sel_rule = rule_cands_['obj_select'][curr_stored_ix]
                        try:
                            if 'skip_tag' in obj_sel_rule: a_states = []; a_hash = []
                            else:
                                rt, M, d = obj_sel_rule['rule']
                                prop = obj_sel_rule['prop']
                                listing_rule = obj_sel_rule['listing_rule']
                            
                                h_states = [] 
                                for iobj_ in initial_global_parsings[gridn_]['i']: 
                                    h_states.append({'type':'h_state','gridn':gridn_,'s':{'obj':iobj_,'mask':initial_global_parsings[gridn_]['i'][iobj_]['mask'],'map':initial_global_parsings[gridn_]['i'][iobj_]['map'],'parsing':initial_global_parsings[gridn_]['i'][iobj_]['properties']['parsing_description']}})

                                prop_list = prop(list(range(len(h_states))), h_states)
                                
                                param_list = reverse_mapping(prop_list, M, d, rt)

                                a_states = [h_states[k] for k in range(len(h_states)) if param_list[k]==1]
                                a_hash = copy.deepcopy(a_states)


                                if listing_rule == 'gridwise_listgroup':
                                    
                                    if len(a_states)>1:
                                        new_state = {'type':'h_state','gridn':gridn_,'s':{'obj':[],'mask':[],'map':[],'parsing':[]}}
                                        for state_ in a_states: 
                                            new_state['s']['obj'].append(state_['s']['obj'])
                                            new_state['s']['mask'].append(state_['s']['mask'])
                                            new_state['s']['map'].append(state_['s']['map'])
                                            new_state['s']['parsing'].append(state_['s']['parsing'])
                                        a_states = [new_state] 
                        except: print('FAILED objsel, increment and restart'); stored_indices[r_]['obj_select'] += 1; retry = True; break


                        
                        m_states = copy.deepcopy(a_states)
                        for state in m_states: state['type'] = 'analogy'
                        o_states = copy.deepcopy(append_o_states)
                        for state in o_states: state['type'] = 'other'; m_states.append(state)

                        


                        m_hash = copy.deepcopy(a_hash) 
                        for state in m_hash: state['type'] = 'analogy'
                        o_states = copy.deepcopy(append_o_states)
                        for state in o_states: state['type'] = 'other'; m_hash.append(state)




                        unmodified_m_states = copy.deepcopy(m_states)

                        for stackn in range(10):
                            if 'stackn'+str(stackn) not in rule_cands_: break

                            if len(rule_cands_['stackn'+str(stackn)])==0: print("ERROR ----- No Rule Cand Available on this stackn --------")
                            
                            curr_stored_ix = stored_indices[r_]['stackn'+str(stackn)]
                            
                            if curr_stored_ix > len(rule_cands_['stackn'+str(stackn)])-1: quitsoln =True; break
                            first_currstack_rule = rule_cands_['stackn'+str(stackn)][curr_stored_ix]
                            try:
                                if first_currstack_rule['reverse_fn'] in ['reverse_positional_coloring']: m_states = globals()[first_currstack_rule['reverse_fn']](m_hash, first_currstack_rule) 
                                else: m_states = globals()[first_currstack_rule['reverse_fn']](m_states, first_currstack_rule)
                                
                            except: print('FAILED stackrule, increment and restart'); stored_indices[r_]['stackn'+str(stackn)] += 1; retry = True; break

                            
                            for state in m_states:
                                if state['type']=='analogy': 
                                    if 's' not in state: print('SKIPPING; check'); continue
                                    if 'e' not in state: print('SKIPPING; check why no e state in recon?'); continue 
                                    state['s']['mask'] = state['e']['mask']
                                    state['s']['map'] = state['e']['map']
                                    

                        if retry or quitsoln: break

                        
                        
                        for m, state in enumerate(m_states):
                            if state['type']=='analogy': 
                                
                                
                                if 'e' in state: 
                                    persistent_states.append(state) 
                        

                                
                                if m > len(unmodified_m_states)-1: 
                                    
                                    pass
                                else: 
                                    modifying_e = copy.deepcopy(unmodified_m_states[m])
                                    if 'e' in state: modifying_e['e'] = state['e']
                                    modifying_e['type'] = 'not appl'
                                    append_o_states.append(modifying_e)



                        
                        if rule_cands_['gridsize_restriction']!=[]:
                            e_masks = []
                            for m, state in enumerate(m_states):
                                if state['type']=='analogy':
                                    e_masks.append(state['e']['mask'])
                            
                            combo_e_mask = e_masks[0]
                            for e_mask in e_masks:
                                combo_e_mask = combo_e_mask | e_mask

                            rows_,cols_ = np.where(combo_e_mask==1)
                            corner_br_rc = (max(rows_), max(cols_)) 
                                
                            restriction_shape = (corner_br_rc[0]+1,corner_br_rc[1]+1) 

                    if retry or quitsoln: break

                    
                    persistent_states_w_disappears = []
                    for state in persistent_states:
                        if np.sum(state['e']['mask'])==0: pass
                        else: persistent_states_w_disappears.append(state)
                    persistent_states = persistent_states_w_disappears



                    
                    
                    if restriction_shape is not None:
                        for state in persistent_states:
                            state['e']['mask'] = state['e']['mask'][0:0+restriction_shape[0],0:0+restriction_shape[1]]
                            state['e']['map'] = state['e']['map'][0:0+restriction_shape[0],0:0+restriction_shape[1]]


                    
                    all_layer_relations = [];obj_layerns={}

                    
                    
                    chosen_objs, chosen_masks, chosen_maps = [_['e']['obj'] for _ in persistent_states], [_['e']['mask'] for _ in persistent_states], [_['e']['map'] for _ in persistent_states]
                    
                    nco = []; ncm1=[];ncm2=[] 
                    for n_, co in enumerate(chosen_objs):
                        if type(co)==list: 
                            nco.extend(co); 
                            for __ in range(len(co)): 
                                if type(chosen_masks[n_])==list: ncm1.append(chosen_masks[n_][0]); ncm2.append(chosen_maps[n_][0]) 
                                else: ncm1.append(chosen_masks[n_]); ncm2.append(chosen_maps[n_]) 
                        else: nco.append(co); ncm1.append(chosen_masks[n_]); ncm2.append(chosen_maps[n_])
                    chosen_objs = nco; chosen_masks = ncm1; chosen_maps = ncm2
                    for k in range(len(chosen_masks)): 
                        if len(chosen_masks[k].shape)==3: 
                            chosen_masks[k] = chosen_masks[k][0,:,:]; chosen_maps[k]= chosen_maps[k][0,:,:]


                    
                    all_layer_relations_ = []
                    for layer_relation in all_layer_relations:
                        if layer_relation[0] in chosen_objs and layer_relation[2] in chosen_objs:
                            all_layer_relations_.append(layer_relation) 

                    sorted_flayers = sort_relationsY(chosen_objs, all_layer_relations_)
                    
                    f_layers = np.zeros((chosen_masks[0].shape[0],chosen_masks[0].shape[1],50)); il = 0
                    obj_layerns['f'] = {}
                    for obj_name in sorted_flayers: 
                        
                        iobj_name = obj_name
                        ix = chosen_objs.index(iobj_name)
                        mask_,map_ = chosen_masks[ix], chosen_maps[ix]
                        rows,cols = np.where(mask_==1)
                        if sum(f_layers[:,:,il][mask_==1])>0: il+=1 
                        for m in range(len(rows)): f_layers[rows[m],cols[m],il] = 1 
                        obj_layerns['f'][obj_name] = il

                    recon = np.zeros((chosen_masks[0].shape[0],chosen_masks[0].shape[1]))
                    for fl in range(il+1):
                        for obj_ in obj_layerns['f']:
                            if obj_layerns['f'][obj_] == fl:
                                recon = np.where(chosen_masks[chosen_objs.index(obj_)], chosen_maps[chosen_objs.index(obj_)], recon)

                    
                    
                    
                    if gridn <= num_demo_grids-1: 
                        o_grid = o_grids[gridn]
                        if are_two_identical(np.array(recon,dtype=int), np.array(o_grids[gridn_],dtype=int)): pass
                        else: print('QUIT since cannot make it work'); quitsoln = True; break
                    else:
                        
                        #if are_two_identical(np.array(recon,dtype=int), np.array(o_grids[gridn_],dtype=int)): print(gridn - num_demo_grids,"== SOLVED === of", num_test_grids-1)
                        # If we're here, it's an eval grid. Save into current_solution_eval_recons (for just this current attempt)
                        recon_as_ints = [[int(x) for x in row] for row in recon.tolist()] # convert to ints
                        current_solution_eval_recons.append(recon_as_ints)

                if not retry and not quitsoln:
                    # If we got to this point without quitsoln or retry flags, then we've reversed all grids (test and eval). Take the recons for eval grids as the solutions, but only save the top 2 distinct solutions
                    print('^ ========= SAVING this, Check All solved**. Successfully reversed All eval grids as well as testgrids. Choose this as an attempt')

                    # Analyse current_solution_eval_recons. Its len corresponds to the number of grids (not attempts; it's just 1 attempt each on the current itern)
                    all_attempts_of_completed_eval_recons.append(current_solution_eval_recons)

                    n_attempts_on_each = []
                    for ngrids_ in range(len(all_attempts_of_completed_eval_recons[0])):
                        all_current_solns = [_[ngrids_] for _ in all_attempts_of_completed_eval_recons]
                        _, all_unique_curr_solns = label_unique_with_IDs(all_current_solns)
                        n_attempts_on_each.append(len(all_unique_curr_solns))
                    if np.all([_>=2 for _ in n_attempts_on_each]): quitentirely = True # May not have 2 ofc


                    # TODO consider allowing if only 1 of the many gridns are solvable? Rather than forcing them all to be solved before saving
                    
                    quitsoln = True

                if quitsoln: print('QUIT likely exceeded num cands, or cannot make it work'); break
            if quitentirely: print('QUIT entirely since already found top 2 attempts'); break



    except Exception as e:
        pass # traceback.print_exc()


    if len(all_attempts_of_completed_eval_recons)>0:
        curr_outputs_ = []
        for ngrids_ in range(len(all_attempts_of_completed_eval_recons[0])):
            all_current_solns = [_[ngrids_] for _ in all_attempts_of_completed_eval_recons]
            _, all_unique_curr_solns = label_unique_with_IDs(all_current_solns)
            # May not have 2, or may have >2, either way, pad or pick top 2
            curr_attempts_ = {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]} # init
            for ca, attempt_ in enumerate(all_unique_curr_solns):
                #print('-----gridn',ngrids_,'attempt',attempt_)
                if ca==0: curr_attempts_["attempt_1"] = attempt_
                if ca==1: curr_attempts_["attempt_2"] = attempt_
            curr_outputs_.append(curr_attempts_)
        #print('------------------------------rrrr',curr_outputs_)

        if curr_outputs_ != []:
            # Save by overwriting
            output_data_[test_string_] = curr_outputs_


with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(output_data_, f)#, indent=4)
    
print("Submission file created in kaggle/working")

