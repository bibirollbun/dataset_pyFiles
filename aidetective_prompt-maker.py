DATA_PATH = '/kaggle/input/arc-prize-2025'
FIGURES_PATH = 'task_figures'


import numpy as np, pandas as pd, json, os
import matplotlib.pyplot as plt
%matplotlib inline
import pprint 
pp = pprint.PrettyPrinter(indent=1)
from matplotlib import colors
import copy # for creating full copy of JSON object
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.colors as mcolors

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

colorNames=["black","blue","red","green","yellow","grey","purple","orange","cyan","brown","white"]
    
cmap = colors.ListedColormap(
   ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])
norm = colors.Normalize(vmin=0, vmax=10)

def plot_one(ax, i, task, train_or_test, input_or_output, is_solution=False, is_pred=False):
    if is_pred: input_matrix = task
    elif is_solution: input_matrix = task[i]
    else: input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])     
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    if is_pred:
        title = 'test prediction'
    else:
        title = train_or_test + ' ' + input_or_output
    ax.set_title(title)

    
def plot_task(task1, text, task_solution=None, save_file=None):    
    num_train = len(task1['train'])
    num_test = len(task1['test'])
    #num_test  = len(task['test'])

    w = num_train
    
    if task_solution is not None:
        w += num_test
            
    fig, axs  = plt.subplots(2, w, figsize=(3*w ,3*2))
    plt.suptitle(f'{text}', fontsize=int(3*w*1.5), fontweight='bold', y=1)

    for j in range(num_train):     
        plot_one(axs[0, j], j, task1, 'train', 'input')
        plot_one(axs[1, j], j, task1, 'train', 'output')  
        
    if task_solution is not None:
        for k in range(num_test):
            plot_one(axs[0, j+k+1], k, task1, 'test', 'input') 
            plot_one(axs[1, j+k+1], k, task_solution, 'test', 'output', is_solution=True)
            
        
    fig.patch.set_linewidth(3)
    fig.patch.set_edgecolor('black') 
    fig.patch.set_facecolor('#dddddd')
#     plt.tight_layout()
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
        
    plt.show()

def showImage(matrix):
    # Display the matrix as an image
    plt.figure(figsize=(2, 2))
    norm = mcolors.BoundaryNorm(np.arange(10+1)-0.5, cmap.N)
    plt.imshow(matrix, cmap=cmap, interpolation='nearest',norm=norm)
    plt.axis('off')  # Hide the axes for better visual
    plt.show()


if not os.path.exists(FIGURES_PATH):
    os.mkdir(FIGURES_PATH)


train_tasks   = load_json(f'{DATA_PATH}/arc-agi_training_challenges.json')
train_sols    = load_json(f'{DATA_PATH}/arc-agi_training_solutions.json')

eval_tasks = load_json(f'{DATA_PATH}/arc-agi_evaluation_challenges.json')
eval_sols  = load_json(f'{DATA_PATH}/arc-agi_evaluation_solutions.json')

test_tasks   = load_json(f'{DATA_PATH}/arc-agi_test_challenges.json')

trainIDs = list(train_tasks.keys())
evalIDs = list(eval_tasks.keys())


id='e3721c99'
plot_task(eval_tasks[id],id)


def getDims(grid):
    return len(grid[0]),len(grid)


def find_first_one(matrix):
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val == 1:
                return (i, j)
    return None  # No '1' found

def getShape(matrix,x,y,col,isHole=False, forceDiagonals=False):
    H = len(matrix)
    W = len(matrix[0])
    mask = np.zeros((H, W)).astype(int)
    start = y,x
    i,j = start
    n=0
    extents = [[] for _ in range(H)]
    startSide="T"
    if isHole:
        startSide="R"
    sides=startSide
    edge=startSide

    #When True holes connect diagonals but shape parts don't. When False the opposite holds
    joinDiagonals = False or forceDiagonals
    
    minI=1000
    minJ=1000
    maxI=-1000
    maxJ=-1000
    while (((i,j)!=start or edge!=startSide) or n==0) and n<1000:
        n+=1
        #print(edge,i,j)
        if sides[-1]!=edge:
            sides+=edge
        minI,minJ,maxI,maxJ=min(i,minI),min(j,minJ),max(i,maxI),max(j,maxJ)
        

        if edge=='L' or edge=='R':
            extents[i].append(j)
       
        if edge=='T':
            if i-1>=0 and j+1<W and matrix[i-1][j+1]==col and (joinDiagonals or matrix[i][j+1]==col):
                edge='L'
                i-=1
                j+=1
            elif j+1==W or matrix[i][j+1]!=col:
                edge='R'
            else:
                j+=1
        elif edge=='R':
            if i+1<H and j+1<W and matrix[i+1][j+1]==col and (joinDiagonals or matrix[i+1][j]==col):
                edge='T'
                i+=1
                j+=1
            elif i+1==H or matrix[i+1][j]!=col:
                edge='B'
            else:
                i+=1
        elif edge=='B':
            if i+1<H and j-1>=0 and matrix[i+1][j-1]==col and (joinDiagonals or matrix[i][j-1]==col):
                edge='R'
                i+=1
                j-=1
            elif j-1<0 or matrix[i][j-1]!=col:
                edge='L'
            else:
                j-=1
        elif edge=='L':
            if i-1>=0 and j-1>=0 and matrix[i-1][j-1]==col and (joinDiagonals or matrix[i-1][j]==col):
                edge='B'
                i-=1
                j-=1
            elif i-1<0 or matrix[i-1][j]!=col:
                edge='T'
            else:
                i-=1
    
    for y in range(H):
        extents[y].sort()
        for n in range(len(extents[y])//2):
            cstart=extents[y][2*n] + (1 if isHole else 0)
            cend = extents[y][2*n+1]+(0 if isHole else 1)
            for x in range(cstart,cend):
                mask[y][x]=1
    if isHole:
        sides=invert(sides)
        minI+=1
        maxI-=1
        minJ+=1
        maxJ-=1
    return sides, mask , (maxJ-minJ+1),(maxI-minI+1)

    


def compare(lst1, lst2):
    if len(lst1) != len(lst2):
        return False  # Lists of different lengths can't be circularly equal
    return lst1 in lst2+lst2

def rotate(symbolList):
    return symbolList.translate(str.maketrans("TRBL", "RBLT"))

def reflect(symbolList):
    return symbolList.translate(str.maketrans("RL", "LR"))[::-1]

def invert(symbolList):
    return symbolList.translate(str.maketrans("TRBL", "BLTR"))[::-1]

def full_compare(lst1,lst2):
    for reflection in range(2):
        for rotation in range(4):
            if compare(lst1,lst2):
                if reflection==0:
                    if rotation==0:
                        return True#"Match with same orientation"
                    else:
                        return True#"Match with "+str(90*rotation)+" degree rotation"  
                else:
                    if rotation==0:
                        return True#"Match with reflection in vertical axis"
                    elif rotation==2:
                        return True#"Match with reflectino in horizontal axis"
                    else:
                        return True#"Match with reflection in diagonal axis" 
            lst1 = rotate(lst1)
        lst1 = reflect(lst1)
    return False#"No match"
    


rectangleSides="TRBL"
LshapeSides="TRTRBL"
SshapeSides="TRBRBLTL"
UshapeSides="TRTLTRBL"
TshapeSides="TRBRBLBL"
chevronSides="TRTRBLBRBLTL"
plusSides   ="TRTRBRBLBLTL"


def combineMasks(array1,array2):
    return np.logical_or(array1, array2).astype(int)

def removeFromMask(array1,array2):
    return (array1-array2).astype(int)
    


from collections import Counter

import re
def insert_s_before_parenthesis(text):
    match = re.search(r'\s*\(', text)  # Find the first "(" preceded by optional spaces
    if match:
        pos = match.start()  # Position of "("
        before_paren = text[:pos].rstrip()  # Everything before "(" without trailing spaces
        after_paren = text[pos:]  # Everything including and after "("

        # Modify the last word before "(" by adding "s"
        modified = re.sub(r'(\S+)$', r'\1s', before_paren)  
        return modified + after_paren
    else:
        return text + "s"  # If no "(", append "s" to the en

def combineDuplicates(input_list):
    # Count the frequency of each string in the list
    count = Counter(input_list)
    
    # Prepare the result list with combined strings
    result = []
    
    for item, frequency in count.items():
        # Split the string into words and pluralize the second word (e.g., "dog" -> "dogs")
        if frequency==1:
            result.append("a "+item)
        else:
            result.append(str(frequency) +" "+insert_s_before_parenthesis(item))        
    
    return result
def dontCombineDuplicated(input_list):
    return ["a "+item for item in input_list]
combineDuplicates(["red dog","fox","dog","cat","fox","green cat","green cat"])


def naturalList(items):
    if len(items) == 0:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + " and " + items[-1]


def getColorDescription(number_array, mask):
    
    # Apply the mask to filter numbers
    filtered_numbers = number_array[mask == 1]
    if len(filtered_numbers)==0:
        return 0,0
    
    # Count occurrences of each number (0 to 9)
    unique, counts = np.unique(filtered_numbers, return_counts=True)
    
    # Store the frequency in a dictionary
    frequency_dict = dict(zip(unique, counts))
    most_frequent_number = unique[np.argmax(counts)]
    most_frequent_count = np.max(counts)
    num_ones_in_mask = np.sum(mask)

    most_frequent_percentage = (most_frequent_count / num_ones_in_mask) * 100

    #print(f"Most Frequent Number: {most_frequent_number}, Count: {most_frequent_count}")
    #print(f"Number of 1s in Mask: {num_ones_in_mask}")
    #print(f"Percentage of Most Frequent Number: {most_frequent_percentage:.2f}%")
    return most_frequent_number, most_frequent_percentage



def colorPercentToString(color,percent):
    if percent==100:
        return colorNames[color]+" "
    elif percent>50:
        return "mostly "+colorNames[color]+" "
    else:
        return ""


# When ignoreColor==-1 then we are getting parts otherwise we are getting holes

def getParts(grid,mask,level,ignoreColor=-1):
    if(level>100):
        return "RECURSSION EXCEEDED"
    W,H = getDims(grid)
    items=[]
    hasNoParts=False
    for y in range(H):
        for x in range(W):
            if mask[y][x]==1 and grid[y][x]!=ignoreColor:
                col = grid[y][x]
                holeHasParts=False
                if ignoreColor==-1:
                    sides,shapemask,sW,sH = getShape(grid,x,y,col)
                else:
                    #print("find hole with boundary "+str(ignoreColor))
                    sides,shapemask,sW,sH = getShape(grid,x-1,y,ignoreColor,isHole=True)
                    _,shapemask2,_,_ = getShape(grid,x,y,col,forceDiagonals=True)
                    holeHasParts = not np.all(shapemask==shapemask2)
                
                
                hasNoParts = (len(items)==0 and np.all(shapemask==mask))
                
                mask = mask-shapemask
                #showImage(shapemask)

                s="";
                percent=0
                color = colorNames[col]+" "
                #if ignoreColor!=-1 and not hasNoParts and False:
                if holeHasParts:
                    colorID,percent = getColorDescription(grid,shapemask)
                    color=""
                    #color=colorPercentToString(colorID,percent) #e.g. "shape is mostly red"
                
                if compare(sides,rectangleSides):
                    if sW==1 and sH==1:
                        s=(""+color+"dot")
                    elif sW==1:
                        s=("length "+str(sH)+" "+color+"vertical line")
                    elif sH==1:
                        s=("length "+str(sW)+" "+color+"horizontal line")
                    else:   
                        shapeName=("square" if sW==sH else "rectangle" )                 
                        s=(""+str(sW)+"x"+str(sH)+" "+color+shapeName)
                elif full_compare(sides,LshapeSides):
                    s=(""+color+"L-shape")
                elif full_compare(sides,TshapeSides):
                    s=(""+color+"T-shape")
                elif full_compare(sides,chevronSides):
                    s=(""+color+"chevron-shape")
                elif full_compare(sides,UshapeSides):
                    s=(""+color+"U-shape")
                elif full_compare(sides,SshapeSides):
                    s=(""+color+"S-shape")
                elif full_compare(sides,plusSides):
                    s=(""+color+"plus-shape")
                else:
                    s=(""+color+"region")

                if hasNoParts and level>0:
                    s=""

                
                #containing?
                if percent<100:
                    if ignoreColor==-1:
                        #get holes
                        parts = getParts(grid,shapemask,level+1,ignoreColor=col)
                    else:                
                        #get parts
                        parts = getParts(grid,shapemask,level+1)
                    if(parts!=""):
                        if hasNoParts:
                            s+=" "+parts
                        else:
                            s+=" ("+parts+")"

                items.append(s)
    if not hasNoParts or level==0:
        items=combineDuplicates(items)
        #items=dontCombineDuplicated(items)
    s=""
    if(ignoreColor!=-1 and len(items)>0):
        s+="containing "    
    if(ignoreColor==-1 and len(items)>0 and not hasNoParts) or level==0:
        s+="consisting of "
    
    s+=naturalList(items)
    return s

def createSceneGraph(grid):
    W,H = getDims(grid)
    mask = np.ones((H, W)).astype(int)
    return getParts(grid,mask,0)
    
                


def description(grid,name):
    s=""
    grid=np.array(grid)
    W,H=getDims(grid)
    s= "***"+name+" is a "+str(W)+"x"+str(H)+" grid. The description is:***\n\n"
    mask=np.ones((H,W))
    col,percent=getColorDescription(grid,mask)
    color=colorPercentToString(col,percent)
    if color!="":
        color+=", "
    s+="The grid is "+color + createSceneGraph(grid)+"." #"consisting of "+
    s+="\n\n"
    return s

ordinals=["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth"]
    

def createPromptForTask(id):
    plot_task(eval_tasks[id],id)
    print("**TEST**")
    showImage(eval_tasks[id]['test'][0]['input'])
    s="This is an IQ task consisting of grids of coloured squares. "
    s+="We are given pairs of input and output grids as examples. "
    s+="The task is to try to determine the rule and then apply this to the final input grid. "
    s+="Note that, not all the information in the description is necessarily relevent for the task. "
    s+="Please try and find the rule the transforms the inputs to the outputs and express it in words. "
    s+="You can think for a while and do it step by step."
    s+="\n\n"
    for i in range(len(eval_tasks[id]['train'])):
        s+=description(eval_tasks[id]['train'][i]['input'],"The "+ordinals[i]+" training input")
        s+=description(eval_tasks[id]['train'][i]['output'],"The "+ordinals[i]+" training output")
    s+="Now here is the test case we have to solve:\n\n"

    s+=description(eval_tasks[id]['test'][0]['input'],"The test input")
    s+="\n\n\nThat's all the details. So what should we do to transform the final test example and solve the task?"

    return s
    


A=[[0,0,0,0,0,0],[0,1,1,2,2,0],[0,1,1,2,2,0],[0,0,0,0,0,0]]

A=[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,1,2,2,0,0],[0,0,1,1,2,2,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
A2=[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
B=[[2,2],[3,3]]
A3=[[0,0,0,0,0,0],[0,1,0,0,2,0],[0,1,0,0,2,0],[0,0,0,0,0,0]]
A4=[[0,0,0,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0]]
A5=[[0,0,0,0,0],[0,1,1,1,0],[0,1,2,1,0],[0,1,1,1,0],[0,0,0,0,0]]
A6=[[0,0,0,0,0,0,0],[0,3,3,3,3,3,0],[0,3,3,3,3,3,0],[0,3,3,3,3,3,0],[0,3,3,3,3,3,0],[0,3,3,3,3,3,0],[0,0,0,0,0,0,0]]
A7=[[0,0,0,0],[0,0,2,0],[0,3,0,0],[0,0,0,0]]
A8=[[0,0,0,0],[0,0,2,0],[0,2,3,0],[0,0,0,0]]
A9=[[0,0,0,0,0],[0,0,2,0,0],[0,2,3,2,0],[0,0,2,0,0],[0,0,0,0,0]]
A10=[[0,0,0,0,0],[0,0,1,1,0],[0,1,0,0,0],[0,0,0,0,0]]
#print(description(A3,"HOLES"))
showImage(A3)
print(description(A3,"TEST1"))
showImage(A)
print(description(A,"TEST2"))
showImage(A5)
print(description(A5,"TEST3"))
showImage(A6)
print(description(A6,"TEST4"))
showImage(B)
print(description(B,"TEST5"))
showImage(A7)
print(description(A7,"Test7"))
showImage(A8)
print(description(A8,"Test8"))
showImage(A9)
print(description(A9,"Test9"))
showImage(A10)
print(description(A10,"Test10"))


print(createPromptForTask('e3721c99'))


print(createPromptForTask('c7f57c3e'))



print(createPromptForTask('62593bfd'))


print(createPromptForTask('8b9c3697'))


print(createPromptForTask('45a5af55'))


