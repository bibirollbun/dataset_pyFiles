import numpy as np
import re
import os

def load_pt_cloud(file :str,oversample : bool = False):
    with open(file, 'r') as f:
        data = f.read()
    lines = data.split("\n")
    if oversample:
        return oversample_mesh(lines)
    else:
        # get all vertices information
        lines = [line for line in lines if len(line)>1 if line[0:2]=='v ']
        # convert to numpy array
        lines = [line.split(" ")[1:] for line in lines]
        lines = numpy.array(lines,dtype=float)
        return lines

def oversample_mesh(lines : str):
    #Oversample meshes by adding points in the middle of each face
    #This is used to get the full surface information of meshes in pointcloud format

    #Get vertices coordinates in numpy format to index them easily
    vertices_coordinates = [line for line in lines if len(line)>1 if line[0:2]=='v '] # lines beginning in v refer to vertices
    vertices_coordinates = [line.split(" ")[1:] for line in vertices_coordinates]
    vertices_coordinates = numpy.array(vertices_coordinates,dtype=float)

    #Get faces information
    faces_information = [line for line in lines if len(line)>1 if line[0:2]=='f '] # lines beginning in f refer to faces
    faces_information = [ [int(v.split("/")[0])-1 for v in line.split(" ")[1:]]  for line in faces_information]
    faces_coordinates = [numpy.mean(vertices_coordinates[face],axis = 0) for face in faces_information] # store the center of the faces

    return numpy.concatenate((vertices_coordinates,faces_coordinates))

def simple_mesh_shift(mesh_input,objective:list = [0,0,0]):
    # Uses a simple calculation to roughly align the meshes with their center of mass
    # This is normally not needed as the 4DME sequences are already aligned
    x,y,z = objective
    mesh_mean_coords = np.mean(mesh_input,axis = 0)
    shift = [x-mesh_mean_coords[0],y-mesh_mean_coords[1],z-mesh_mean_coords[2]]
    return mesh_input + shift

def ptp(array : np.array):
    # Compute the peak-to-peak (max-min) value of an array with only one pass
    min,max = None,None 
    for elem in array:
        if min is None or elem < min:
            min = elem
        if max is None or elem > max:
            max = elem
    return max-min,max,min

def list_files(initial_directory : str):
    initial_directory = re.sub(r'[\\/]+', '//', initial_directory)
    files = os.listdir(initial_directory)
    files = [initial_directory + paths for paths in files if '.obj' in paths]
    return files


def distance_map(mesh_input,nb_subdivision):
    # Create a distance map using the meshes.
    # The resulting maps will be of size 1,nb_subdivision,nb_subdivision
    
    # Normalize the grid so that the face surface fills it completely
    step_x = ptp(mesh_input[:,0])[0]/(nb_subdivision-1)
    step_y = ptp(mesh_input[:,1])[0]/(nb_subdivision-1)
    
    distance_map = np.zeros((nb_subdivision,nb_subdivision,1)) #Init the depth map
    for point in mesh_input:
        #evaluate the SDF (Z coordinate of the closest point) at values of the XY plane
        grid_x = int(point[0]//step_x)
        grid_y = int(point[1]//step_y)
        distance_map[grid_x,grid_y] = point[2]
    return distance_map


#Example code used in the baseline :
seq = "train\\seq0\\" #path to sequence
mesh_path_list = list_files(seq)
seq_list = [load_pt_cloud(mesh) for mesh in mesh_path_list]
data = [distance_map(mesh,512) for mesh in seq_list]
#print(data.shape)


def voxel_map(mesh_input,nb_subdivision):
    # Create a distance map using the meshes.
    # The resulting maps will be of size nb_subdivision^3
    
    # We normalize the grid over the three axes
    step_x = ptp(mesh_input[:,0])[0]/(nb_subdivision-1)
    step_y = ptp(mesh_input[:,1])[0]/(nb_subdivision-1)
    step_z = ptp(mesh_input[:,2])[0]/(nb_subdivision-1)
    
    voxel_map = np.zeros((nb_subdivision,nb_subdivision,nb_subdivision)) #Init the voxel map
    for point in mesh_input:
        #This time we only include occupancy information (0/1) over the surface in the voxel matrix
        grid_x = int(point[0]//step_x)
        grid_y = int(point[1]//step_y)
        grid_z = int(point[2]//step_z)
        voxel_map[gid_x,grid_y,grid_z] = 1
    return distance_map


def load_pt_cloud(file :str,oversample : bool = False):
    with open(file, 'r') as f:
        data = f.read()
    lines = data.split("\n")
    if oversample:
        return oversample_mesh(lines)
    else:
        # get all vertices information
        lines = [line for line in lines if len(line)>1 if line[0:2]=='v ']
        # convert to numpy array
        lines = [line.split(" ")[1:] for line in lines]
        lines = numpy.array(lines,dtype=float)
        return lines

# Result : a numpy array of size (nb_vertices,3) or (nb_vertices + nb_faces,3) if oversample = True

def random_downsample(mesh,ratio = 0.5):
    #sample points randomly from the surface.
    mesh_len = len(mesh)
    pt_number = int(mesh_len*ratio)
    random_index = np.random.choice(list(range(mesh_len)),size = pt_number,replace = False)
    return mesh[random_index]

# Result : a numpy array of size (nb_vertices*ratio,3)
    


#Using the 2D depth map example
num_subdivision = 128
mesh_1_depthmap = np.zeros((num_subdivision,num_subdivision,1)) #dummy data
mesh_2_depthmap = np.ones((num_subdivision,num_subdivision,1))

def total_depth_distance(mesh1,mesh2):
    #compute L1 distance in the 2D space of the grids
    return np.sum(np.abs(mesh2-mesh1))

def partial_depth_distance(mesh1,mesh2,grid_slice):
    #compute L1 distance in the 2D space, this time on a fixed window of the grid, i.e to focus on one part of the face
    # grid_slice contains (x_start,y_start,x_stop,y_stop)
    x_start,y_start,x_stop,y_stop = grid_slice
    return np.sum(np.abs(mesh2[x_start:x_stop,y_start:y_stop] - mesh1[x_start:x_stop,y_start:y_stop]))

print(total_depth_distance(mesh_1_depthmap,mesh_2_depthmap))
grid_slice = (0, None , num_subdivision//2, None) # compute the distance over half the face 
print(partial_depth_distance(mesh_1_depthmap,mesh_2_depthmap,grid_slice))




