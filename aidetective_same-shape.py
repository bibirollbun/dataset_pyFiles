import numpy as np
import matplotlib.pyplot as plt

def showImage(matrix):
    # Display the matrix as an image
    plt.figure(figsize=(1, 1))
    plt.imshow(matrix, cmap='binary', interpolation='nearest')
    plt.axis('off')  # Hide the axes for better visual
    plt.show()


def find_first_one(matrix):
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val == 1:
                return (i, j)
    return None  # No '1' found

def trace_blob(matrix):
    edge='T'
    H = len(matrix)
    W = len(matrix[0])
    
    start = find_first_one(matrix)
    i,j = start
    n=0
    sides="T"
    while (((i,j)!=start or edge!='T') or n==0) and n<100:
        n+=1
        #print(edge,i,j)
        if sides[-1]!=edge:
            sides+=edge
       
        if edge=='T':
            if i-1>=0 and j+1<W and matrix[i-1][j+1]==1:
                edge='L'
                i-=1
                j+=1
            elif j+1==W or matrix[i][j+1]==0:
                edge='R'
            else:
                j+=1
        elif edge=='R':
            if i+1<H and j+1<W and matrix[i+1][j+1]==1:
                edge='T'
                i+=1
                j+=1
            elif i+1==H or matrix[i+1][j]==0:
                edge='B'
            else:
                i+=1
        elif edge=='B':
            if i+1<H and j-1>=0 and matrix[i+1][j-1]==1:
                edge='R'
                i+=1
                j-=1
            elif j-1<0 or matrix[i][j-1]==0:
                edge='L'
            else:
                j-=1
        elif edge=='L':
            if i-1>=0 and j-1>=0 and matrix[i-1][j-1]==1:
                edge='B'
                i-=1
                j-=1
            elif i-1<0 or matrix[i-1][j]==0:
                edge='T'
            else:
                i-=1
    return sides



def rotate(symbolList):
    return symbolList.translate(str.maketrans("TRBL", "RBLT"))

def reflect(symbolList):
    return symbolList.translate(str.maketrans("RL", "LR"))[::-1]

def permute_right(lst, k=2):
    return lst[-k:] + lst[:-k]

#compare including permutations
def compare(lst1, lst2):
    if len(lst1) != len(lst2):
        return False  # Lists of different lengths can't be circularly equal
    return lst2 in (lst1+lst1)# Duplicate lst1 to check all rotations

def full_compare(lst1,lst2):
    for reflection in range(2):
        for rotation in range(4):
            if compare(lst1,lst2):
                if reflection==0:
                    if rotation==0:
                        return "Match with same orientation"
                    if rotation>1:
                        return "Match with "+str(90*rotation)+" degree rotation"
                    return     
                else:
                    if rotation==0:
                        return "Match with reflection in vertical axis"
                    elif rotation==2:
                        return "Match with reflectino in horizontal axis"
                    else:
                        return "Match with reflection in diagonal axis"
                    return    
            lst1 = rotate(lst1)
        lst1 = reflect(lst1)
    return "No match"
    


rectangle1 = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]
rectangle2 = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]
showImage(rectangle1)
showImage(rectangle2)
edges1 = trace_blob(rectangle1)
edges2 = trace_blob(rectangle2)
print(edges1)
print(edges2)
print(full_compare(edges1,edges2))


Lshape = [
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

Lshape2 = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0]
]
showImage(Lshape)
showImage(Lshape2)

print(full_compare(trace_blob(Lshape),trace_blob(Lshape2)))


print(full_compare(trace_blob(Lshape),trace_blob(rectangle1)))


Fshape = [
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0]
]

Fshape2 = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0]
]
showImage(Fshape)
showImage(Fshape2)
print(full_compare(trace_blob(Fshape),trace_blob(Fshape2)))


Ushape = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

Ushape2 = [
    [1, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]
showImage(Ushape)
showImage(Ushape2)
print(full_compare(trace_blob(Ushape),trace_blob(Ushape2)))



rectangleWithHole = [
    [1, 1, 1, 1, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]
showImage(rectangle1)
showImage(rectangleWithHole)
print(full_compare(trace_blob(rectangle1),trace_blob(rectangleWithHole)))


Eshape1 = [
    [1, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1 ,0 ,1],
    [1, 0, 1, 1, 1, 0, 1]
]

Eshape2 = [
    [1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1 ,0 ,1],
    [1, 0, 1, 0, 1, 1, 1]
]
showImage(Eshape1)
showImage(Eshape2)
print(full_compare(trace_blob(Eshape1),trace_blob(Eshape2)))

