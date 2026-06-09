# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np

import matplotlib.pyplot as plt
from   matplotlib import colors

import json

from pathlib import Path

#from subprocess import Popen, PIPE, STDOUT
from glob import glob

import numpy as np
from skimage import measure



#jupyter nbconvert --to script rule.ipynb



cmap = colors.ListedColormap(	['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
								  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', 
								   '#6F00FF',])

#An Extra color for some operations

norm = colors.Normalize(vmin=0, vmax=10)

#Plot One
#mat=np.random.randint(5, size=(10, 10))
#plotMat(mat,"test")
def plotMat(mat, title):
	input_matrix = np.array(mat)
	fig, ax = plt.subplots(figsize=(5, 2.5))
	ax.imshow(input_matrix, cmap=cmap, norm=norm)
	#ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
	
	plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
	ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])	 
	ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
	ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
	ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
	
	ax.set_title(title)

	plt.show()
	return

#Plot Train-Test Pairs
def plotMat2(mat1, mat2, title):
	mat1 = np.array(mat1)
	mat2 = np.array(mat2)
	
	fig, axs = plt.subplots(1, 2, figsize=(5, 2.5)) # 1 row, 2 columns
	
	axs[0].imshow(mat1, cmap=cmap, norm=norm)
	axs[0].yaxis.set_major_locator(plt.MultipleLocator(1.0))
	axs[0].xaxis.set_major_locator(plt.MultipleLocator(1.0))
	axs[1].imshow(mat2, cmap=cmap, norm=norm)
	axs[1].yaxis.set_major_locator(plt.MultipleLocator(1.0))
	axs[1].xaxis.set_major_locator(plt.MultipleLocator(1.0))

	# Remove tick labels
	plt.setp(axs, xticklabels=[], yticklabels=[])

	# Set a shared title
	fig.suptitle(title)

	plt.show()


	return

#Plot Several Mats
def plotMatN(mats, title, indexTitle):
	# Convert all matrices to numpy arrays
	mats = [np.array(mat) for mat in mats]

	# Calculate the number of rows needed
	rows = (len(mats) +5) // 6 # Ceiling division for up to 4 per row

	# Create subplots
	fig, axs = plt.subplots(rows, 6, figsize=(2.5*6, 2.5*rows))
	axs = axs.flatten() # Flatten for easy indexing

	# Plot each matrix
	for i in range(len(mats)):
		axs[i].imshow(mats[i], cmap=cmap, norm=norm)
		plt.rcParams.update({'font.size': 8})
		axs[i].set_title(f'Object {indexTitle[i]}')
		axs[i].yaxis.set_major_locator(plt.MultipleLocator(1.0))
		axs[i].xaxis.set_major_locator(plt.MultipleLocator(1.0))


	# Hide unused subplots
	for j in range(len(mats), len(axs)):
		fig.delaxes(axs[j])

	# Remove tick labels
	plt.setp(axs, xticklabels=[], yticklabels=[])
	# Set a shared title
	fig.suptitle(title)

	plt.show()
	return 

#Plot Several Mats
def plotMatNDots(mats, title, indexTitle):
	# Convert all matrices to numpy arrays
	mats = [np.array(mat) for mat in mats]

	# Calculate the number of rows needed
	rows = (len(mats) +11) // 12 # Ceiling division for up to 4 per row

	# Create subplots
	fig, axs = plt.subplots(rows, 12, figsize=(1.25*12, 1.25*rows+1), dpi=75)
	axs = axs.flatten() # Flatten for easy indexing

	# Plot each matrix
	for i in range(len(mats)):
		axs[i].imshow(mats[i], cmap=cmap, norm=norm)
		plt.rcParams.update({'font.size': 8})
		axs[i].set_title(f'{indexTitle[i]}')
		axs[i].yaxis.set_major_locator(plt.MultipleLocator(1.0))
		axs[i].xaxis.set_major_locator(plt.MultipleLocator(1.0))


	# Hide unused subplots
	for j in range(len(mats), len(axs)):
		fig.delaxes(axs[j])

	# Remove tick labels
	plt.setp(axs, xticklabels=[], yticklabels=[])
	# Set a shared title
	fig.suptitle(title)

	plt.show()
	return 


plt.figure(figsize=(4, 1), dpi=200)
plt.imshow([list(range(11))], cmap=cmap, norm=norm)
plt.xticks(list(range(11)))
plt.yticks([])
plt.show()


base_path='/kaggle/input/arc-prize-2025/'
# Loading JSON data
def load_json(file_path):
	with open(file_path) as f:
		data = json.load(f)
	return data


# Reading files
training_challenges =  load_json(base_path +'arc-agi_training_challenges.json')
training_solutions =   load_json(base_path +'arc-agi_training_solutions.json')

evaluation_challenges =load_json(base_path +'arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json(base_path +'arc-agi_evaluation_solutions.json')

test_challenges =  load_json(base_path +'arc-agi_test_challenges.json')
print(f'Number of training challenges = {len(training_challenges)}')
print(f'Number of training solutions = {len(training_solutions)}')
print(f'Number of evaluation challenges = {len(evaluation_challenges)}')
print(f'Number of evaluation solutions = {len(evaluation_solutions)}')
print(f'Number of test challenges = {len(test_challenges)}')


def find_submatrices_with_wildcards(mat1, mat2):
	mat1 = np.array(mat1)
	mat2 = np.array(mat2)
	rows1, cols1 = mat1.shape
	rows2, cols2 = mat2.shape
	indices = []

	for i in range(rows2 - rows1 + 1):
		for j in range(cols2 - cols1 + 1):
			submatrix = mat2[i:i+rows1, j:j+cols1]
			# Create a mask where mat1 is not zero
			mask = mat1 != 0
			# Compare only where mat1 is not zero
			if np.array_equal(mat1[mask], submatrix[mask]):
				indices.append((i, j))

	return indices


def place_submatrices_with_wildcards(mat1, mat2, indices):
	mat1 = np.array(mat1)
	rows1, cols1 = mat1.shape

	for (i, j) in indices:
		submatrix = mat2[i:i+rows1, j:j+cols1]
		mask = mat1 != 0
		submatrix[mask] = mat1[mask]
		mat2[i:i+rows1, j:j+cols1] = submatrix

	return mat2

#Check if Matrix1 exists in Matrix2
def find_submatrices(mat1, mat2):
	mat1=np.array(mat1)
	mat2=np.array(mat2)
	rows1, cols1 = mat1.shape
	rows2, cols2 = mat2.shape
	indices = []

	for i in range(rows2 - rows1 + 1):
		for j in range(cols2 - cols1 + 1):
			if np.array_equal(mat2[i:i+rows1, j:j+cols1], mat1):
				indices.append((i, j))

	return indices

#Check if horizontally Flipped Matrix1 Exists in Matrix2
def find_flipped_submatrices(mat1, mat2):
	# Flip mat1 horizontally
	mat2=np.array(mat2)
	flipped_mat1 = np.fliplr(mat1)
	
	rows1, cols1 = flipped_mat1.shape
	rows2, cols2 = mat2.shape
	indices = []
	
	for i in range(rows2 - rows1 + 1):
		for j in range(cols2 - cols1 + 1):
			if np.array_equal(mat2[i:i+rows1, j:j+cols1], flipped_mat1):
				indices.append((i, j))
	
	return indices
#Put Matrix1 in Matrix2 in Position Indices
def place_submatrices(mat1, mat2, indices):
	mat1=np.array(mat1)
	rows1, cols1 = mat1.shape
 
	for (i, j) in indices:
		mat2[i:i+rows1, j:j+cols1] = mat1

	return mat2
#Put Flipped Horizontally Matrix1 in Matrix2 in Position Indices
def place_flipped_submatrices(mat1, mat2, indices):
	# Flip mat1 horizontally
	flipped_mat1 = np.fliplr(mat1)
	
	rows1, cols1 = flipped_mat1.shape
	
	for (i, j) in indices:
		mat2[i:i+rows1, j:j+cols1] = flipped_mat1
	
	return mat2

#Repaint Single Color
def repaintMat(array, indices, new_value):
	if np.isscalar(new_value):
		for (i, j) in indices:
			array[i, j] = new_value
	else:
		for (i, j) in indices:
			array[i, j] = new_value[i, j]
	return array

#Find Closed Blobs with 0's
def extractClosedLoopsIndices(data):
	outputBlobs=[]
	labelled_array=np.array(data)
	size=labelled_array.shape
	for i in range(size[0]):
		for j in range(size[1]):
			if labelled_array[i][j] >0:
				labelled_array[i][j]=1

	
	labelled_array=measure.label(labelled_array,background=1,connectivity=1)
	#Return a list containing coordinates of pixels in each blob."
	props = measure.regionprops(labelled_array)
	blobs = [p.coords for p in props]
	for i in range(len(blobs)):
		addTo=False
		for j in range(len(blobs[i])):
			#if it touches borders
			if blobs[i][j][0]==0:
				addTo=True
			if blobs[i][j][1]==0:
				addTo=True
			if blobs[i][j][0]==size[0]-1:
				addTo=True
			if blobs[i][j][1]==size[1]-1:
				addTo=True
		if addTo==False:
			outputBlobs.append(np.array(blobs[i]))
	return outputBlobs

#Put all 0s Closed Loops in a Mat
def closeLoopsMat(matInput):
	indicesEmpty=extractClosedLoopsIndices(matInput)
	#print(indicesEmpty)
	emptyMat=np.zeros(np.array(matInput).shape)
	for k in range(len(indicesEmpty)):
		repaintMat(emptyMat,indicesEmpty[k],10)
	return emptyMat



from dataclasses import dataclass, field
from typing import Tuple, Union

@dataclass
class graphObject:
	OriginalIndex: np.ndarray
	Color: Union[int, np.ndarray]
	dimensionsOriginal: Tuple[int, int]
	
	DisplacedIndex: np.ndarray = field(init=False)
	Height: int = field(init=False)
	Width: int = field(init=False)
	topLeftCorner: Tuple[int, int] = field(init=False)
	displacedMat: np.ndarray = field(init=False)
	originalMat: np.ndarray = field(init=False)
	shapeType: int = field(init=False)
	centroid: Tuple[float, float] = field(init=False) 

	def __post_init__(self):
		self._compute_geometry()
		self._create_matrices()
		self.shapeType = self._classify_shape()
		self.centroid = self._compute_centroid()


	def _compute_centroid(self):
		coords = np.array(self.OriginalIndex)
		return tuple(np.mean(coords, axis=0))  


	def _compute_geometry(self):
		coords = np.array(self.OriginalIndex)
		min_row, min_col = coords.min(axis=0)
		max_row, max_col = coords.max(axis=0)

		self.topLeftCorner = (min_row, min_col)
		self.Height = max_row - min_row + 1
		self.Width = max_col - min_col + 1

		self.DisplacedIndex = coords - [min_row, min_col]

	def _create_matrices(self):
		self.displacedMat = np.zeros((self.Height, self.Width), dtype=int)
		self.originalMat = np.zeros(self.dimensionsOriginal, dtype=int)

		for idx, (r, c) in enumerate(self.OriginalIndex):
			color_value = self.Color if np.isscalar(self.Color) else self.Color[r, c]
			self.originalMat[r, c] = color_value

		for idx, (r, c) in enumerate(self.DisplacedIndex):
			orig_r, orig_c = self.OriginalIndex[idx]
			color_value = self.Color if np.isscalar(self.Color) else self.Color[orig_r, orig_c]
			self.displacedMat[r, c] = color_value

	def _classify_shape(self) -> int:
		mat = self.displacedMat
		non_zero_coords = np.argwhere(mat != 0)
		if non_zero_coords.size == 0:
			return -1  # No shape

		min_row, min_col = non_zero_coords.min(axis=0)
		max_row, max_col = non_zero_coords.max(axis=0)
		height = max_row - min_row + 1
		width = max_col - min_col + 1

		# Check for diagonal line
		if height == width and height > 1:
			sorted_coords = sorted(non_zero_coords.tolist())
			diag1 = all((r - min_row) == (c - min_col) for r, c in sorted_coords)
			diag2 = all((r - min_row) == (max_col - c) for r, c in sorted_coords)
			if diag1 or diag2:
				return 4  # Diagonal line

		bounding_box = mat[min_row:max_row+1, min_col:max_col+1]
		if np.any(bounding_box == 0):
			return -2  # Not solid

		if height == 1 and width == 1:
			return 0  # Dot
		elif height == 1 or width == 1:
			return 1  # Line
		elif height == width:
			return 2  # Square
		elif height > 1 and width > 1:
			return 3  # Rectangle
		else:
			return -1  # Undefined

	def __str__(self):
		return f"Index {self.OriginalIndex.tolist()}, Top Left Corner {self.topLeftCorner}, Color {self.Color}, Shape {self.shapeType}"

#Find ColorBlobs and give as an Output Objects
def extractgraphObjects(data):
	data = np.array(data)
	dimensionsOriginal = data.shape
	labelled_array = measure.label(data)
	props = measure.regionprops(labelled_array)

	OriginalBlobs = [p.coords for p in props]
	colorBlob = np.zeros(len(OriginalBlobs))
	graphObjects = []

	for i in range(len(OriginalBlobs)):
		pos = np.array(OriginalBlobs[i][0])
		colorBlob[i] = int(data[pos[0], pos[1]])

	for i in range(len(OriginalBlobs)):
		gObject = graphObject(OriginalBlobs[i], colorBlob[i], dimensionsOriginal)
		graphObjects.append(gObject)

	# Sort by top (row), then left (column)
	graphObjects.sort(key=lambda obj: (obj.topLeftCorner[0], obj.topLeftCorner[1]))

	return graphObjects


#Make a mat into an Object
def matToObject(data):
	data=np.array(data)
	dimensionsOriginal=data.shape

	colorBlob=np.zeros((data.shape))

	coords=[]
	for i in range(len(data)):
		for j in range(len(data[0])):
			colorBlob[i,j]=data[i,j]
			coords.append([i,j])


	gObject=graphObject(coords,colorBlob,dimensionsOriginal)


	return gObject

def extractEmptyBlobsObjects(data):
	data=np.array(data)
	dimensionsOriginal=data.shape
	outputBlobs=[]
	labelled_array=np.array(data)
	size=labelled_array.shape
	for i in range(size[0]):
		for j in range(size[1]):
			if labelled_array[i][j] >0:
				labelled_array[i][j]=1

	
	labelled_array=measure.label(labelled_array,background=1, connectivity=1)
	#Return a list containing coordinates of pixels in each blob."
	props = measure.regionprops(labelled_array)
	blobs = [p.coords for p in props]
	for i in range(len(blobs)):
		addTo=False
		for j in range(len(blobs[i])):
			#if it touches borders
			if blobs[i][j][0]==0:
				addTo=True
			if blobs[i][j][1]==0:
				addTo=True
			if blobs[i][j][0]==size[0]-1:
				addTo=True
			if blobs[i][j][1]==size[1]-1:
				addTo=True
		if addTo==False:
			outputBlobs.append(blobs[i])


	graphObjects=[]
	
	colorBlob=np.full(len(outputBlobs),10)

	for i in range(len(outputBlobs)):
		gObject=graphObject(outputBlobs[i],colorBlob[i],dimensionsOriginal)
		graphObjects.append(gObject)

	matEmpty=closeLoopsMat(data)

	return graphObjects, matEmpty

def plotObjects(objectsInput, title):
	matObjects=[]
	indexTitle=[]

	dotObjects=[]
	dotTitle=[]
	for k in range(len(objectsInput)):
		if objectsInput[k].Height*objectsInput[k].Width != 1:
			matObjects.append(np.array(objectsInput[k].displacedMat))
			shape=classify_shape(objectsInput[k])
			indexTitle.append("%d:%d"%(k,shape))
		else:
			dotObjects.append(np.array(objectsInput[k].displacedMat))
			dotTitle.append("%d %s"%(k, objectsInput[k].OriginalIndex))

	if len(matObjects)>0:
		plotMatN(matObjects,title,indexTitle)
	if len(dotObjects)>0:
		plotMatNDots(dotObjects,title,dotTitle)

	return


def classify_shape(obj):
	"""
	Classifies the shape of a graphObject based on the pattern of its color blob.

	Returns:
		0 if it's a dot (1x1),
		1 if it's a solid horizontal or vertical line (1xN or Nx1),
		2 if it's a solid square (NxN, N > 1),
		3 if it's a solid rectangle (HxW, H â‰  W, H > 1, W > 1),
		4 if it's a diagonal line (45-degree),
	   -2 if the shape is not solid (has internal 0s, excluding diagonals),
	   -1 if it doesn't match any known shape.
	"""
	mat = np.array(obj.displacedMat)
	non_zero_coords = np.argwhere(mat != 0)

	if non_zero_coords.size == 0:
		return -1  # No shape

	min_row, min_col = non_zero_coords.min(axis=0)
	max_row, max_col = non_zero_coords.max(axis=0)

	height = max_row - min_row + 1
	width = max_col - min_col + 1

	# Check for diagonal line (allowing internal 0s)
	if height == width and height > 1:
		sorted_coords = sorted(non_zero_coords.tolist())
		diag1 = all((r - min_row) == (c - min_col) for r, c in sorted_coords)
		diag2 = all((r - min_row) == (max_col - c) for r, c in sorted_coords)
		if diag1 or diag2:
			return 4  # Diagonal line

	# Extract bounding box and check solidity
	bounding_box = mat[min_row:max_row+1, min_col:max_col+1]
	if np.any(bounding_box == 0):
		return -2  # Not solid (internal 0s)

	if height == 1 and width == 1:
		return 0  # Dot
	elif height == 1 or width == 1:
		return 1  # Line
	elif height == width:
		return 2  # Square
	elif height > 1 and width > 1:
		return 3  # Rectangle
	else:
		return -1  # Undefined or irregular
	

def are_graph_objects_equal(obj1, obj2):
	"""
	Compares two graphObject instances for equality.
	Returns True if they are considered the same, False otherwise.
	"""
	if obj1.Color != obj2.Color:
		return False
	if obj1.Height != obj2.Height or obj1.Width != obj2.Width:
		return False
	if not np.array_equal(obj1.displacedMat, obj2.displacedMat):
		return False
	if not np.array_equal(obj1.originalMat, obj2.originalMat):
		return False
	if not np.allclose(obj1.topLeftCorner, obj2.topLeftCorner):
		return False
	return True

def are_graph_objects_equal_ignore_color(obj1, obj2):
	"""
	Compares two graphObject instances for structural equality, ignoring color.
	Returns True if they are structurally the same, False otherwise.
	"""
	if obj1.Height != obj2.Height or obj1.Width != obj2.Width:
		return False
	if not np.array_equal(obj1.displacedMat != 0, obj2.displacedMat != 0):
		return False
	if not np.array_equal(obj1.originalMat != 0, obj2.originalMat != 0):
		return False
	if not np.allclose(obj1.topLeftCorner, obj2.topLeftCorner):
		return False
	return True



def are_graph_object_arrays_equal_unordered(arr1, arr2, compare_func):
	"""
	Compares two arrays of graphObject instances regardless of order,
	using a provided comparison function.

	Parameters:
		arr1, arr2: Lists of graphObject instances.
		compare_func: A function that takes two graphObjects and returns True if they are considered equal.

	Returns:
		True if all objects in arr1 have a matching object in arr2 and vice versa, False otherwise.
	"""
	if len(arr1) != len(arr2):
		return False

	matched = [False] * len(arr2)

	for obj1 in arr1:
		found_match = False
		for i, obj2 in enumerate(arr2):
			if not matched[i] and compare_func(obj1, obj2):
				matched[i] = True
				found_match = True
				break
		if not found_match:
			return False

	return True


def relative_position(obj1, obj2):
	"""
	Determines the spatial relationship between two objects based on their coordinates.
	Includes touching relationships in addition to basic spatial ones.
	"""
	min_row1 = min(coord[0] for coord in obj1)
	max_row1 = max(coord[0] for coord in obj1)
	min_col1 = min(coord[1] for coord in obj1)
	max_col1 = max(coord[1] for coord in obj1)

	min_row2 = min(coord[0] for coord in obj2)
	max_row2 = max(coord[0] for coord in obj2)
	min_col2 = min(coord[1] for coord in obj2)
	max_col2 = max(coord[1] for coord in obj2)

	# Touching relationships
	if max_row2 + 1 == min_row1 and min_col1 <= max_col2 and max_col1 >= min_col2:
		return "touching_top"
	elif min_row2 == max_row1 + 1 and min_col1 <= max_col2 and max_col1 >= min_col2:
		return "touching_bottom"
	elif max_col2 + 1 == min_col1 and min_row1 <= max_row2 and max_row1 >= min_row2:
		return "touching_left"
	elif min_col2 == max_col1 + 1 and min_row1 <= max_row2 and max_row1 >= min_row2:
		return "touching_right"

	# Basic spatial relationships
	elif max_row2 < min_row1:
		return "above"
	elif min_row2 > max_row1:
		return "below"
	elif max_col2 < min_col1:
		return "left"
	elif min_col2 > max_col1:
		return "right"
	else:
		return "overlapping"

def relative_position_matrix(objects):
	"""
	Builds a matrix of spatial relationships between all pairs of objects.
	"""
	n = len(objects)
	matrix = [["" for _ in range(n)] for _ in range(n)]

	for i in range(n):
		for j in range(n):
			if i != j:
				matrix[i][j] = relative_position(objects[i].OriginalIndex, objects[j].OriginalIndex)
			else:
				matrix[i][j] = "same"

	return matrix







#GRANULARITY*****************************************************************************************************
#****************************************************************************************************************
#e.g. exercise 2

def granularity_output(input_matrix, output_matrix):
	input_matrix = np.array(input_matrix)
	output_matrix = np.array(output_matrix)

	input_rows, input_cols = input_matrix.shape
	expected_output_shape = (input_rows * input_rows, input_cols * input_cols)

	# Check if output has the correct shape
	if output_matrix.shape != expected_output_shape:
		return False

	for i in range(input_rows):
		for j in range(input_cols):
			start_row = i * input_rows
			start_col = j * input_cols
			block = output_matrix[start_row:start_row + input_rows, start_col:start_col + input_cols]

			if input_matrix[i, j] != 0:
				if not np.array_equal(block, input_matrix):
					return False
			else:
				if not np.all(block == 0):
					return False

	return True



def granularity_multiple_examples(input_matrices, output_matrices):
	results = []
	for input_matrix, output_matrix in zip(input_matrices, output_matrices):
		is_valid = granularity_output(input_matrix, output_matrix)
		results.append(is_valid)
	return results


def applyruleGranularity(input_matrix):
	"""
	Applies the rule of placing the input matrix at every block position in an output matrix
	where the input matrix has a non-zero value.

	Parameters:
		input_matrix (np.array): The smaller matrix to be placed in the output matrix.

	Returns:
		np.array: The generated output matrix.
	"""
	input_rows, input_cols = input_matrix.shape
	output_rows = input_rows * input_rows
	output_cols = input_cols * input_cols

	output_matrix = np.zeros((output_rows, output_cols), dtype=int)

	for i in range(input_rows):
		for j in range(input_cols):
			if input_matrix[i, j] != 0:
				start_row = i * input_rows
				start_col = j * input_cols
				output_matrix[start_row:start_row + input_rows, start_col:start_col + input_cols] = input_matrix

	return output_matrix
#****************************************************************************************************************
#****************************************************************************************************************

# def infer_input_matrix(output_matrix, block_size):
# 	"""
# 	Infers the input matrix from the output matrix by checking each block of size `block_size`.
# 	If a block contains any non-zero values, the corresponding position in the input matrix is set to 1.

# 	Parameters:
# 		output_matrix (np.array): The larger matrix generated by tiling the input matrix.
# 		block_size (int): The size of the square blocks used to tile the input matrix.

# 	Returns:
# 		np.array: The inferred input matrix.
# 	"""
# 	output_rows, output_cols = output_matrix.shape
# 	input_matrix = np.zeros((output_rows // block_size, output_cols // block_size), dtype=int)
	
# 	for i in range(0, output_rows, block_size):
# 		for j in range(0, output_cols, block_size):
# 			block = output_matrix[i:i + block_size, j:j + block_size]
# 			if np.any(block != 0):
# 				input_matrix[i // block_size, j // block_size] = 1
	
# 	return input_matrix




def apply_generalized_rules_to_simulatedMat(simulatedMat, input_objects, patterns, input_matrix=None):
	output = np.copy(simulatedMat)
	rows, cols = output.shape

	possible_colors = sorted(set(patterns.get("possible_output_colors", [])))
	#print("possible_colors", possible_colors)

	color_behavior = patterns.get("color_behavior", "transformed")
	input_colors = set()
	if color_behavior == "preserved":
		for obj in input_objects:
			if np.isscalar(obj.Color):
				input_colors.add(int(obj.Color))
			else:
				input_colors.update(int(obj.Color[r, c]) for r, c in obj.OriginalIndex if obj.Color[r, c] != 0)

	def get_fallback_color(index):
		allowed_colors = (
			sorted(input_colors & set(possible_colors))
			if color_behavior == "preserved"
			else possible_colors
		)
		return allowed_colors[index % len(allowed_colors)] if allowed_colors else 1

	filled_positions = set()

	# 1. Apply matched object rules
	for obj in input_objects:
		color_to_use = int(obj.Color) if np.isscalar(obj.Color) else get_fallback_color(0)
		if color_to_use not in possible_colors:
			color_to_use = get_fallback_color(0)

		if obj.Color in patterns["same"]["colors"] and obj.shapeType in patterns["same"]["shapes"]:
			for (r, c) in obj.OriginalIndex:
				if 0 <= r < rows and 0 <= c < cols and obj.originalMat[r, c] != 0:
					output[r, c] = color_to_use
					filled_positions.add((r, c))
		elif obj.Color in patterns["moved"]["colors"]:
			for (r, c) in obj.OriginalIndex:
				if 0 <= r < rows and 0 <= c < cols and obj.originalMat[r, c] != 0:
					output[r, c] = color_to_use
					filled_positions.add((r, c))
		elif tuple(obj.topLeftCorner) in patterns["color_changed"]["positions"]:
			new_color = patterns.get("global_color_change", {}).get(obj.Color, (color_to_use + 1) % 10)
			for (r, c) in obj.OriginalIndex:
				if 0 <= r < rows and 0 <= c < cols:
					output[r, c] = new_color
					filled_positions.add((r, c))
		elif "global_color_change" in patterns and obj.Color in patterns["global_color_change"]:
			new_color = patterns["global_color_change"][obj.Color]
			for (r, c) in obj.OriginalIndex:
				if 0 <= r < rows and 0 <= c < cols:
					output[r, c] = new_color
					filled_positions.add((r, c))

	# 2. Granularity rule
	if patterns.get("granularity") and input_matrix is not None:
		print("Applying granularity rule")
		output = applyruleGranularity(input_matrix)

	# 3. Submatrix rule
	if (
		patterns.get("more_output_objects") and
		patterns.get("submatrix_indices") and
		input_matrix is not None and
		len(patterns["submatrix_indices"]) > 1
	):
		print("Applying submatrix rule")
		output = place_submatrices_with_wildcards(input_matrix, output, patterns["submatrix_indices"])

	# 4. Shape-to-position placements
	matched_positions = {tuple(obj.topLeftCorner) for obj in input_objects}
	for shape, position in patterns.get("shape_position", {}).items():
		if position in matched_positions:
			continue
		for obj in input_objects:
			if obj.shapeType == shape and tuple(obj.topLeftCorner) not in matched_positions:
				for (r, c) in obj.DisplacedIndex:
					rr, cc = position[0] + r, position[1] + c
					if 0 <= rr < rows and 0 <= cc < cols:
						output[rr, cc] = get_fallback_color(0)
						filled_positions.add((rr, cc))

	# 5. Same objects with different colors
	if patterns.get("same_objects_different_colors"):
		print("Applying same objects with different colors rule")
		reference_objects = patterns["same_objects_different_colors"]
		for i, ref_obj in enumerate(reference_objects):
			new_color = get_fallback_color(i)
			for (r, c) in ref_obj.DisplacedIndex:
				rr = ref_obj.topLeftCorner[0] + r
				cc = ref_obj.topLeftCorner[1] + c
				if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in filled_positions:
					output[rr, cc] = new_color

	# 6. Shape transformation rule (centroid-aligned)
	if patterns.get("shape_transformations"):
		print("Applying shape transformation rule")
		for in_obj in input_objects:
			for (in_disp, out_disp, color) in patterns["shape_transformations"]:
				if np.array_equal(in_obj.DisplacedIndex, in_disp) and in_obj.Color == color:
					centroid_r, centroid_c = map(int, np.round(in_obj.centroid))
					out_centroid_r, out_centroid_c = map(int, np.round(np.mean(out_disp, axis=0)))
					for (dr, dc) in out_disp:
						r = centroid_r + (dr - out_centroid_r)
						c = centroid_c + (dc - out_centroid_c)
						if 0 <= r < rows and 0 <= c < cols:
							output[r, c] = color
							filled_positions.add((r, c))
					break

	return output



import numpy as np
from collections import defaultdict

# Utility function for wildcard-aware array comparison


def arrays_equal_with_wildcardShape(a, b):
	"""
	Compares two arrays treating 0 in 'a' as a wildcard.
	Returns False if shapes do not match.
	"""
	a = np.array(a)
	b = np.array(b)
	if a.shape != b.shape:
		return False
	mask = a != 0

	for i in range(len(a)):
		for j in range(len(a[0])):
			if a[i,j]>0:
				a[i,j]=1
			if b[i,j]>0:
				b[i,j]=1


	return np.array_equal(a[mask], b[mask])

# Initializes tracking dictionaries for rule analysis
def initialize_rule_tracking():
	return defaultdict(int), defaultdict(int), defaultdict(list), defaultdict(lambda: defaultdict(int))

def analyze_object_counts(input_objs, output_objs, example_idx, rule_counts, invariants, rule_details):
	if len(output_objs) > len(input_objs):
		rule = "More objects in output"
		rule_counts[rule] += 1
		rule_details[rule].append(f"Example {example_idx}: Input={len(input_objs)}, Output={len(output_objs)}")
	elif len(output_objs) < len(input_objs):
		rule = "Fewer objects in output"
		rule_counts[rule] += 1
		rule_details[rule].append(f"Example {example_idx}: Input={len(input_objs)}, Output={len(output_objs)}")
	else:
		rule = "Number of objects remains the same"
		invariants[rule] += 1
		rule_details[rule].append(f"Example {example_idx}: {len(input_objs)} objects")

def match_objects_by_position_and_shape(input_objs, output_objs, matched_input, matched_output, example_idx,
										rule_counts, invariants, rule_details, color_change_counts, example_matches,
										match_patterns):
	for in_idx, in_obj in enumerate(input_objs):
		if matched_input[in_idx]:
			continue
		for out_idx, out_obj in enumerate(output_objs):
			if matched_output[out_idx]:
				continue
			if np.array_equal(in_obj.topLeftCorner, out_obj.topLeftCorner) and \
			   arrays_equal_with_wildcardShape(in_obj.displacedMat, out_obj.displacedMat):
				if in_obj.Color == out_obj.Color:
					rule = "Object remains the same"
					invariants[rule] += 1
					rule_details[rule].append(f"Example {example_idx}: Input {in_idx} â†” Output {out_idx}")
					example_matches["matched"].append({
						"input_idx": in_idx, "output_idx": out_idx, "type": "same",
						"color": in_obj.Color, "shape": in_obj.shapeType
					})
					match_patterns["same"]["colors"].add(in_obj.Color)
					match_patterns["same"]["shapes"].add(in_obj.shapeType)
				else:
					rule = "Color changed at same position and shape"
					rule_counts[rule] += 1
					rule_details[rule].append(f"Example {example_idx}: Input {in_idx} â†’ Output {out_idx}")
					color_change_counts[in_obj.Color][out_obj.Color] += 1
					example_matches["matched"].append({
						"input_idx": in_idx, "output_idx": out_idx, "type": "color_changed",
						"from_color": in_obj.Color, "to_color": out_obj.Color, "shape": in_obj.shapeType
					})
					match_patterns["color_changed"]["positions"].add(tuple(in_obj.topLeftCorner))
				matched_input[in_idx] = True
				matched_output[out_idx] = True
				break

def match_objects_by_shape_and_color(input_objs, output_objs, matched_input, matched_output, example_idx,
									 rule_counts, rule_details, example_matches, match_patterns,
									 input_rel_matrix, output_rel_matrix):
	for in_idx, in_obj in enumerate(input_objs):
		if matched_input[in_idx]:
			continue
		for out_idx, out_obj in enumerate(output_objs):
			if matched_output[out_idx]:
				continue
			if arrays_equal_with_wildcardShape(in_obj.displacedMat, out_obj.displacedMat) and in_obj.Color == out_obj.Color:
				rule = "Same shape and color but position changed"
				rule_counts[rule] += 1
				rule_details[rule].append(f"Example {example_idx}: Input {in_idx} â†’ Output {out_idx}")

				# Compare relative position matrix rows
				input_relations = input_rel_matrix[in_idx]
				output_relations = output_rel_matrix[out_idx]
				relation_changes = []
				for i, (in_rel, out_rel) in enumerate(zip(input_relations, output_relations)):
					if in_rel != out_rel:
						relation_changes.append((i, in_rel, out_rel))

				example_matches["matched"].append({
					"input_idx": in_idx,
					"output_idx": out_idx,
					"type": "moved",
					"color": in_obj.Color,
					"shape": in_obj.shapeType,
					"relation_changes": relation_changes
				})

				match_patterns["moved"]["colors"].add(in_obj.Color)
				matched_input[in_idx] = True
				matched_output[out_idx] = True
				break



def record_unmatched_objects(matched_input, matched_output, example_idx, rule_counts, rule_details, example_matches):
	for out_idx, matched in enumerate(matched_output):
		if not matched:
			rule = "Extra object in output"
			rule_counts[rule] += 1
			rule_details[rule].append(f"Example {example_idx}: Output {out_idx} unmatched")
			example_matches["unmatched_output"].append(out_idx)
	for in_idx, matched in enumerate(matched_input):
		if not matched:
			example_matches["unmatched_input"].append(in_idx)

def summarize_color_changes(color_change_counts, rule_counts, rule_details, num_examples, match_patterns):
	for input_color, output_colors in color_change_counts.items():
		if len(output_colors) == 1:
			output_color = next(iter(output_colors))
			rule = f"All color {input_color} objects change to color {output_color}"
			rule_counts[rule] = num_examples
			rule_details[rule] = [f"Example {i}" for i in range(num_examples)]
			# Store in patterns for application
			if "global_color_change" not in match_patterns:
				match_patterns["global_color_change"] = {}
			match_patterns["global_color_change"][input_color] = output_color


def compile_final_rules(rule_counts, invariants, rule_details, num_examples):
	consistent_rules = [rule for rule, count in rule_counts.items() if count == num_examples]
	consistent_invariants = [rule for rule, count in invariants.items() if count == num_examples]
	detailed_output = []
	for rule in consistent_rules + consistent_invariants:
		detailed_output.append(rule)
		detailed_output.extend([f" - {detail}" for detail in rule_details[rule]])
	return detailed_output


def all_examples_have_more_output_objects(input_objects_list, output_objects_list):
	"""
	Returns True if every example has more output objects than input objects.
	"""
	return all(len(output_objs) > len(input_objs)
				for input_objs, output_objs in zip(input_objects_list, output_objects_list))

def detect_possible_output_colors(output_objects):
	colors = set()
	for obj in output_objects:
		if np.isscalar(obj.Color):
			colors.add(int(obj.Color))
		else:
			values = [obj.Color[r, c] for r, c in obj.OriginalIndex if obj.Color[r, c] != 0]
			colors.update(values)
	return sorted(colors)

def all_examples_have_same_objects_different_colors(output_objects_list):
	"""
	Checks if all examples have the same number of output objects and identical shapes/positions,
	but allows for different colors.
	"""
	if not output_objects_list:
		return False

	def object_signature(obj):
		return (obj.shapeType, obj.topLeftCorner, obj.Height, obj.Width)

	reference_signatures = sorted(object_signature(obj) for obj in output_objects_list[0])

	for objs in output_objects_list[1:]:
		if len(objs) != len(reference_signatures):
			return False
		current_signatures = sorted(object_signature(obj) for obj in objs)
		if current_signatures != reference_signatures:
			return False

	return True

def detect_color_behavior_by_example(input_objects_list, output_objects_list):
	"""
	Determines if input and output colors are exactly equal for each example.

	Returns:
		behavior: "preserved" or "transformed"
		input_colors_per_example: list of sets of input colors per example
	"""
	input_colors_per_example = []
	behavior = "preserved"

	for input_objs, output_objs in zip(input_objects_list, output_objects_list):
		input_colors = set()
		output_colors = set()

		for obj in input_objs:
			if np.isscalar(obj.Color):
				input_colors.add(int(obj.Color))
			else:
				input_colors.update(int(obj.Color[r, c]) for r, c in obj.OriginalIndex if obj.Color[r, c] != 0)

		for obj in output_objs:
			if np.isscalar(obj.Color):
				output_colors.add(int(obj.Color))
			else:
				output_colors.update(int(obj.Color[r, c]) for r, c in obj.OriginalIndex if obj.Color[r, c] != 0)

		input_colors_per_example.append(input_colors)

		if input_colors != output_colors:
			behavior = "transformed"

	return behavior, input_colors_per_example





def generate_rule_from_examples_unordered(input_objects_list, output_objects_list, input_matrices, output_matrices,
										  position_matrix_inputs, position_matrix_outputs):
	rule_counts, invariants, rule_details, color_change_counts = initialize_rule_tracking()
	matches = []
	match_patterns = {
		"same": {"colors": set(), "shapes": set()},
		"moved": {"colors": set()},
		"color_changed": {"positions": set()},
		"shape_position": {},
		"submatrix_indices": [],
		"possible_output_colors": set(),
		"same_objects_different_colors": None,
		"shape_transformations": []  # New pattern for shape transformations
	}

	object_counts = [(len(inp), len(out)) for inp, out in zip(input_objects_list, output_objects_list)]
	submatrix_allowed = not all(inp == out for inp, out in object_counts)
	moved_colors_per_example = []

	for example_idx, (input_objs, output_objs) in enumerate(zip(input_objects_list, output_objects_list)):
		matched_input = [False] * len(input_objs)
		matched_output = [False] * len(output_objs)
		example_matches = {
			"example": example_idx,
			"matched": [],
			"unmatched_input": [],
			"unmatched_output": []
		}

		analyze_object_counts(input_objs, output_objs, example_idx, rule_counts, invariants, rule_details)
		match_objects_by_position_and_shape(
			input_objs, output_objs, matched_input, matched_output, example_idx,
			rule_counts, invariants, rule_details, color_change_counts, example_matches, match_patterns
		)
		match_objects_by_shape_and_color(
			input_objs, output_objs, matched_input, matched_output, example_idx,
			rule_counts, rule_details, example_matches, match_patterns,
			position_matrix_inputs[example_idx], position_matrix_outputs[example_idx]
		)
		record_unmatched_objects(matched_input, matched_output, example_idx, rule_counts, rule_details, example_matches)
		matches.append(example_matches)

		moved_colors_this_example = {
			match["color"] for match in example_matches["matched"] if match["type"] == "moved"
		}
		moved_colors_per_example.append(moved_colors_this_example)

	color_behavior, input_colors_per_example = detect_color_behavior_by_example(input_objects_list, output_objects_list)
	match_patterns["color_behavior"] = color_behavior
	match_patterns["input_colors_per_example"] = input_colors_per_example

	if color_behavior == "preserved":
		rule = "All input colors are equal to output colors"
		rule_counts[rule] = len(input_objects_list)
		rule_details[rule] = [f"Example {i}" for i in range(len(input_objects_list))]
		match_patterns["possible_output_colors"] = sorted(set.union(*input_colors_per_example))
	else:
		rule = "Some input colors differ from output colors"
		rule_counts[rule] = len(input_objects_list)
		rule_details[rule] = [f"Example {i}" for i in range(len(input_objects_list))]
		total_possible_output_colors = set()
		for objs in output_objects_list:
			for obj in objs:
				if np.isscalar(obj.Color):
					total_possible_output_colors.add(int(obj.Color))
				else:
					total_possible_output_colors.update(
						int(obj.Color[r, c]) for r, c in obj.OriginalIndex if obj.Color[r, c] != 0
					)
		match_patterns["possible_output_colors"] = sorted(total_possible_output_colors)

	if all_examples_have_more_output_objects(input_objects_list, output_objects_list):
		rule = "All examples have more output objects than input objects"
		rule_counts[rule] = len(input_objects_list)
		rule_details[rule] = [f"Example {i}" for i in range(len(input_objects_list))]
		match_patterns["more_output_objects"] = True
		submatrix_allowed = True
	else:
		match_patterns["more_output_objects"] = False
		submatrix_allowed = False

	if moved_colors_per_example:
		common_moved_colors = set.intersection(*moved_colors_per_example)
		match_patterns["moved"]["colors"] = common_moved_colors

	if input_matrices and output_matrices:
		if all(granularity_multiple_examples(input_matrices, output_matrices)):
			rule = "Granularity pattern detected in all examples"
			rule_counts[rule] = len(input_objects_list)
			rule_details[rule] = [f"Example {i}" for i in range(len(input_objects_list))]
			match_patterns["granularity"] = True

	if submatrix_allowed:
		submatrix_indices = [find_submatrices_with_wildcards(inp, out) for inp, out in zip(input_matrices, output_matrices)]
		common_indices = set(submatrix_indices[0])
		common_counts = len(submatrix_indices[0])
		for indices in submatrix_indices[1:]:
			common_indices.intersection_update(indices)
		if common_indices and all(len(indices) == common_counts for indices in submatrix_indices):
			rule = "Input matrix is a submatrix of output matrix (with wildcards)"
			rule_counts[rule] = len(input_objects_list)
			rule_details[rule] = [f"Example {i}" for i in range(len(input_objects_list))]
			match_patterns["submatrix_indices"] = list(common_indices)

	shape_position_patterns = defaultdict(list)
	for match_info, output_objs in zip(matches, output_objects_list):
		matched_output_indices = {match["output_idx"] for match in match_info["matched"]}
		for idx, obj in enumerate(output_objs):
			if idx not in matched_output_indices:
				shape_position_patterns[obj.shapeType].append(tuple(obj.topLeftCorner))

	for shape, positions in shape_position_patterns.items():
		if len(positions) == len(output_objects_list) and len(set(positions)) == 1:
			rule = f"Object with shape {shape} appears at position {positions[0]} in all outputs"
			rule_counts[rule] = len(output_objects_list)
			rule_details[rule] = [f"Example {i}" for i in range(len(output_objects_list))]
			match_patterns["shape_position"][shape] = positions[0]

	summarize_color_changes(color_change_counts, rule_counts, rule_details, len(input_objects_list), match_patterns)

	if all_examples_have_same_objects_different_colors(output_objects_list):
		rule = "All examples have same output objects with different colors"
		rule_counts[rule] = len(output_objects_list)
		rule_details[rule] = [f"Example {i}" for i in range(len(output_objects_list))]
		match_patterns["same_objects_different_colors"] = output_objects_list[0]
	else:
		match_patterns["same_objects_different_colors"] = None

	# ğŸ”� Shape transformation rule detection
	predicted_outputs = []
	for k in range(len(input_objects_list)):
		simulatedMat = np.zeros_like(output_matrices[k])
		transformed = apply_generalized_rules_to_simulatedMat(
			simulatedMat,
			input_objects_list[k],
			match_patterns,
			input_matrix=input_matrices[k]
		)
		predicted_outputs.append(transformed)

	if detect_shape_transformation_rule(input_objects_list, output_objects_list, predicted_outputs, output_matrices):
		rule = "Consistent shape transformation based on color and position"
		rule_counts[rule] = len(input_objects_list)
		rule_details[rule] = [f"Example {i}" for i in range(len(input_objects_list))]
		for in_objs, out_objs in zip(input_objects_list, output_objects_list):
			for in_obj in in_objs:
				for out_obj in out_objs:
					if (in_obj.topLeftCorner == out_obj.topLeftCorner or
						np.allclose(in_obj.centroid, out_obj.centroid)) and in_obj.Color == out_obj.Color:
						match_patterns["shape_transformations"].append(
							(in_obj.DisplacedIndex, out_obj.DisplacedIndex, in_obj.Color)
						)

	detailed_output = compile_final_rules(rule_counts, invariants, rule_details, len(input_objects_list))
	return {
		"rules": detailed_output,
		"matches": matches,
		"patterns": match_patterns
	}



#Measure Error between 2 Images
def measure_errorOriginal(mat1,mat2):
	mat1=np.array(mat1)
	mat2=np.array(mat2)
	rows, cols = mat1.shape

	outMat=np.full(mat1.shape,2)
	error=0
	for i in range(rows):
		for j in range(cols):
			if mat1[i][j]==mat2[i][j]:
				outMat[i][j]=0
			else:
				error+=1
	return outMat, error

def measure_error(mat1, mat2):
	mat1 = np.array(mat1)
	mat2 = np.array(mat2)
	rows, cols = mat1.shape

	outMat = np.full(mat1.shape, 2)
	error = 0
	total = 0

	for i in range(rows):
		for j in range(cols):
			if mat2[i][j] != 0:  # Only consider non-zero values in ground truth
				total += 1
				if mat1[i][j] == mat2[i][j]:
					outMat[i][j] = 0
				else:
					error += 1
	return outMat, error, total



def compute_rule_confidence(predicted_output, ground_truth_output):
	"""
	Computes a confidence score for a predicted output matrix using the measure_error function,
	considering only non-zero values in the ground truth.
	"""
	_, error, total = measure_error(predicted_output, ground_truth_output)
	if total == 0:
		return 1.0  # Avoid division by zero; assume perfect if no non-zero cells
	confidence = 1.0 - (error / total)
	return confidence

def evaluate_output_objects(predicted_objects, ground_truth_objects):
	"""
	Evaluates predicted output objects against ground truth objects based on:
	- Shape match (structure)
	- Color match
	- Position match

	Returns:
		dict: Scores for shape, color, and position accuracy (0.0 to 1.0)
	"""
	from collections import defaultdict

	matched = [False] * len(ground_truth_objects)
	shape_matches = 0
	color_matches = 0
	position_matches = 0

	for pred_obj in predicted_objects:
		for i, gt_obj in enumerate(ground_truth_objects):
			if not matched[i] and are_graph_objects_equal_ignore_color(pred_obj, gt_obj):
				matched[i] = True
				shape_matches += 1
				if pred_obj.Color == gt_obj.Color:
					color_matches += 1
				if pred_obj.topLeftCorner == gt_obj.topLeftCorner:
					position_matches += 1
				break

	total = len(ground_truth_objects)
	return {
		"shape_accuracy": shape_matches / total if total else 1.0,
		"color_accuracy": color_matches / total if total else 1.0,
		"position_accuracy": position_matches / total if total else 1.0
	}



def getSizes(task, numberExamples):
	#GET OUTPUTSIZE #***************************************************************
	#*******************************************************************************
	sizeOutputs=[]
	sizeInputs=[]
	calculatedSizeOutput=0
	
	for j in range(numberExamples):
		matInput=task['train'][j]['input']
		matOutput=task['train'][j]['output']
		#plotMat2(matInput, matOutput, 'Example')
		
		sizeInput=np.shape(matInput)
		sizeOutput=np.shape(matOutput)
		sizeOutputs.append(sizeOutput)
		sizeInputs.append(sizeInput)
	

	"""
	Determines the relationship between input and output sizes across examples.
	
	Returns:
		0 if all inputs and all outputs are the same across all examples, and inputs == outputs.
		1 if input and output are the same for each example, but vary across examples.
		2 if outputs are the same across all examples, inputs differ, and inputs are smaller than outputs.
		3 if there is no clear pattern.
	"""
	all_inputs_same = all(x == sizeInputs[0] for x in sizeInputs)
	all_outputs_same = all(x == sizeOutputs[0] for x in sizeOutputs)
	inputs_equal_outputs = all(sizeInputs[i] == sizeOutputs[i] for i in range(numberExamples))

	def input_smaller_than_output(inp, out):
		return inp[0] <= out[0] and inp[1] <= out[1]

	def input_bigger_than_output(inp, out):
		return inp[0] >= out[0] and inp[1] >= out[1]

	inputs_smaller = all(input_smaller_than_output(sizeInputs[i], sizeOutputs[i]) for i in range(numberExamples))
	inputs_bigger = all(input_bigger_than_output(sizeInputs[i], sizeOutputs[i]) for i in range(numberExamples))

	if all_inputs_same and all_outputs_same and inputs_equal_outputs:
		calculatedSizeOutput = 0  # All inputs and outputs are the same and equal
	elif inputs_equal_outputs:
		calculatedSizeOutput = 1  # Inputs match outputs per example, but vary across examples
	elif all_outputs_same and not inputs_equal_outputs and inputs_smaller:
		calculatedSizeOutput = 2  # Outputs are consistent, inputs differ, and inputs are smaller
	elif all_outputs_same and not inputs_equal_outputs and inputs_bigger:
		calculatedSizeOutput = 3  # Outputs are consistent, inputs differ, and inputs are bigger
	else:
		calculatedSizeOutput = 4  # No clear pattern

	#*******************************************************************************
	#*******************************************************************************
	return sizeInputs, sizeOutputs, calculatedSizeOutput


def check_input_object_in_output(input_objects, output_matrices):
	"""
	Checks if each inputObject appears in the corresponding matOutput and returns the indices where it appears.
	
	Parameters:
		input_objects (list): List of graphObject instances created from input matrices.
		output_matrices (list): List of output matrices.
	
	Returns:
		list of list of tuples: Indices where inputObject appears in the corresponding output matrix.
	"""
	results = []
	for idx in range(len(input_objects)):
		input_mat = input_objects[idx].displacedMat
		output_mat = np.array(output_matrices[idx])
		indices = find_submatrices(input_mat, output_mat)
		results.append(indices)
	return results




def printRules(result):
	print("Inferred Rules:")
	for rule in result["rules"]:
		print(rule)

	print("\nMatch Details Per Example:")
	for match_info in result["matches"]:
		print(f"\nExample {match_info['example']}:")
		print(" Matched Objects:")
		for match in match_info["matched"]:
			if match["type"] == "same":
				print(f"\tInput {match['input_idx']} â†” Output {match['output_idx']} \n Type: {match['type']}, Color: {match['color']}, Shape: {match['shape']}")
			elif match["type"] == "color_changed":
				print(f"\tInput {match['input_idx']} â†’ Output {match['output_idx']} \n Type: {match['type']}, From: {match['from_color']} â†’ To: {match['to_color']}, Shape: {match['shape']}")
			elif match["type"] == "moved":
				print(f"\tInput {match['input_idx']} â†’ Output {match['output_idx']} \n Type: {match['type']}, Color: {match['color']}, Shape: {match['shape']}")
				if "relation_changes" in match and match["relation_changes"]:
					print("\t  Relative Position Changes:")
					for idx, from_rel, to_rel in match["relation_changes"]:
						print(f"\t   - Object {idx}: {from_rel} â†’ {to_rel}")

		print(" Unmatched Input Objects:", match_info["unmatched_input"])
		print(" Unmatched Output Objects:", match_info["unmatched_output"])

	print("\nShared Characteristics Across All Examples:")
	patterns = result["patterns"]
	clean_positions = [(int(x), int(y)) for (x, y) in patterns["color_changed"]["positions"]]
	clean_same_colors = sorted(int(c) for c in patterns["same"]["colors"])
	clean_same_shapes = sorted(int(s) for s in patterns["same"]["shapes"])
	clean_moved_colors = sorted(int(c) for c in patterns["moved"]["colors"])
	print(f" Objects that remained the same had colors: {clean_same_colors} and shapes: {clean_same_shapes}")
	print(f" Objects that changed position had colors: {clean_moved_colors}")
	print(f" Objects that changed color were at positions: {clean_positions}")
	return result



def detect_shape_transformation_rule(input_objects_list, output_objects_list, predicted_outputs, ground_truth_outputs, threshold=0.5):
	"""
	Detects if a consistent shape transformation should be applied based on:
	- Matching topLeftCorner or centroid
	- Matching color
	- Same number of objects per color
	- High prediction error
	"""
	from collections import defaultdict

	consistent_transformations = []
	for i, (input_objs, output_objs, pred, truth) in enumerate(zip(input_objects_list, output_objects_list, predicted_outputs, ground_truth_outputs)):
		input_by_color = defaultdict(list)
		output_by_color = defaultdict(list)

		for obj in input_objs:
			input_by_color[obj.Color].append(obj)
		for obj in output_objs:
			output_by_color[obj.Color].append(obj)

		for color in input_by_color:
			if color not in output_by_color:
				continue
			if len(input_by_color[color]) != len(output_by_color[color]):
				continue

			matched = 0
			for in_obj in input_by_color[color]:
				for out_obj in output_by_color[color]:
					if (in_obj.topLeftCorner == out_obj.topLeftCorner or
						np.allclose(in_obj.centroid, out_obj.centroid)):
						matched += 1
						break
			if matched == len(input_by_color[color]):
				consistent_transformations.append(i)

	# Check prediction error
	high_error_examples = []
	for i, (pred, truth) in enumerate(zip(predicted_outputs, ground_truth_outputs)):
		if isinstance(pred, list):
			pred = np.array(pred)
		if isinstance(truth, list):
			truth = np.array(truth)
		if pred.shape != truth.shape or pred.size == 0:
			continue
		_, error, total = measure_error(pred, truth)
		if total > 0 and (error / total) > threshold:
			high_error_examples.append(i)

	if set(consistent_transformations) == set(high_error_examples) and consistent_transformations:
		return True
	return False




for i in range(20):
	print("Example %d"%i)
	t=list(training_challenges)[i]
	task=training_challenges[t]
	task_solution = training_solutions[t][0]

	

	numberExamples=len(task['train'])
	print("Training Examples %d"%numberExamples)



	#numberExamples=1
	changeColor=False
	
	sizeInputs, sizeOutputs, calculatedSizeOutput=getSizes(task, numberExamples)
	print("Expected Size %d"%calculatedSizeOutput)

	#GET OBJECTS********************************************************************
	#*******************************************************************************

	objectsInput=[]
	objectsOutput=[]
	closeLoops=[]
	matEmpty=[]

	#In inputMat shape smaller than outputMat shape
	inputObject=[]
	outputMats=[]
	inputMats=[]

	positionMatrixInput=[]
	positionMatrixOutput=[]

	#GET OBJECTS********************************************************************
	#*******************************************************************************

	for j in range(numberExamples):
		matInput=task['train'][j]['input']
		matOutput=task['train'][j]['output']

		plotMat2(matInput, matOutput, 'Example')
		sizeInput=np.shape(matInput)
		sizeOutput=np.shape(matOutput)
		if calculatedSizeOutput==0:
			simulatedMat=np.zeros(sizeOutput)
		if calculatedSizeOutput==1:
			simulatedMat=np.zeros(sizeInput)
		### Extract all Input Objects
		objectsInput.append(extractgraphObjects(matInput))
		#plotObjects(objectsInput)
		
		### Extract all Output Objects
		objectsOutput.append(extractgraphObjects(matOutput))
		#plotObjects(objectsOutput)
		
		### Extract closed Empty Loops
		emptyBlobs,emptyMat=extractEmptyBlobsObjects(matInput)
		closeLoops.append(emptyBlobs)
		matEmpty.append(emptyMat)
		#plotMat(emptyMat,'Closed Loops')
		
		#Input as an Object
		inputObject.append(matToObject(matInput))
		inputMats.append(np.array(matInput))
		outputMats.append(np.array(matOutput))
		
		#Input Object Connections
		positionMatrixInput.append(relative_position_matrix(extractgraphObjects(matInput)))
		#Output Object Connections
		positionMatrixOutput.append(relative_position_matrix(extractgraphObjects(matOutput)))

		

	result = generate_rule_from_examples_unordered(
		objectsInput,
		objectsOutput,
		inputMats,
		outputMats,
		positionMatrixInput,
		positionMatrixOutput
	)

	printRules(result)

	
	#*******************************************************************************

	#*******************************************************************************


	for k in range (numberExamples):
		if calculatedSizeOutput==0: # All inputs and outputs are the same and equal
			simulatedMat=np.zeros(sizeInputs[k])
			#plotMat(matEmpty[k],"Empty Mat")
			#plotObjects(objectsInput[k], "Input Objects")
			#plotObjects(objectsOutput[k], "Output Objects")
			#plotObjects(closeLoops[k], "Input Closed Loops")
		elif calculatedSizeOutput==1: # Inputs match outputs per example, but vary across examples
			simulatedMat=np.zeros(sizeInputs[k])
			#plotMat(matEmpty[k],"Empty Mat")
			#plotObjects(objectsInput[k], "Input Objects")
			#plotObjects(objectsOutput[k], "Output Objects")
			#plotObjects(closeLoops[k], "Input Closed Loops")
			
		elif calculatedSizeOutput==2: # Outputs are consistent, inputs differ, and inputs are smaller
			simulatedMat=np.zeros(sizeOutputs[0])
			# plotMat(inputObject[k].displacedMat,"Single Object")
			# indices= find_submatrices_with_wildcards(inputObject[k].displacedMat,task['train'][k]['output'])
			# simulatedMat=place_submatrices_with_wildcards(inputObject[k].displacedMat,simulatedMat,indices)
			# errorMat,error=measure_error(task['train'][k]['output'],simulatedMat)
			# plotMat2(errorMat, simulatedMat, 'Error and Simulated')
			# plotObjects(objectsInput[k], "Input Objects")
			# plotObjects(objectsOutput[k], "Output Objects")
			# plotObjects(closeLoops[k], "Input Closed Loops")
		elif calculatedSizeOutput==3: # Outputs are consistent, inputs differ, and inputs are bigger
			simulatedMat=np.zeros(sizeOutputs[0])
			# plotObjects(objectsInput[k], "Input Objects")
			# plotObjects(objectsOutput[k], "Output Objects")
			# plotObjects(closeLoops[k], "Input Closed Loops")
		else: # No clear pattern
			pass

		# Assume result is the output from generate_rule_from_examples_unordered
# and simulatedMat is a blank matrix of the correct shape
		if not calculatedSizeOutput>3: 
			transformed = apply_generalized_rules_to_simulatedMat(
				simulatedMat,
				objectsInput[k],
				result["patterns"],
				input_matrix=inputMats[k]
			)

			#plotMat(transformed, "Transformed Output")
			
			mats = []
			mats.append(inputMats[k])
			mats.append(outputMats[k])
			mats.append(transformed)

			


			predicted_objects = extractgraphObjects(transformed)
			ground_truth_objects = extractgraphObjects(outputMats[k])

			scores = evaluate_output_objects(predicted_objects, ground_truth_objects)
			print("Shape Accuracy:", scores["shape_accuracy"])
			print("Color Accuracy:", scores["color_accuracy"])
			print("Position Accuracy:", scores["position_accuracy"])

			titles = ["Input", "Expected Output", "Calculated"]
			plotMatN(mats, "Result", titles)

	
	

	# for l in range(numberExamples):
	# 	block_size = 3
	# 	input_matrix = infer_input_matrix(output_matrices[l], block_size)
	# 	print("Inferred input matrix:")
	# 	print(input_matrix)


	#TEST **************************************************************************
	#*******************************************************************************
	# matInputTest=task['test'][0]['input']
	# matOutputTest=training_solutions[t][0]
	# plotMat2(matInputTest, matOutputTest, 'TEST')
	# sizeInput=np.shape(matInputTest)
	# sizeOutput=np.shape(matOutputTest)
	# if calculatedSizeOutput==0:
	# 	simulatedMat=np.zeros(sizeOutput)
	# if calculatedSizeOutput==1:
	# 	simulatedMat=np.zeros(sizeInput)
	# errorMat,error=measure_error(matOutputTest,simulatedMat)
	# plotMat2(errorMat, simulatedMat, 'Error and Simulated')





