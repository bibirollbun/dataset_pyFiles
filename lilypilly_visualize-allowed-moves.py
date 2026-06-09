INPUT_DIR = "/kaggle/input/santa-2023"
OUTPUT_DIR = "/kaggle/working/"


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# File paths
puzzle_info_path = INPUT_DIR+'/puzzle_info.csv'
puzzles_path = INPUT_DIR+'/puzzles.csv'
sample_submission_path = INPUT_DIR+'/sample_submission.csv'

# Loading the data
puzzle_info_df = pd.read_csv(puzzle_info_path)
puzzles_df = pd.read_csv(puzzles_path)
sample_submission_df = pd.read_csv(sample_submission_path)


def get_moves(puzzle_type):
    allowed_moves_str = puzzle_info_df.loc[puzzle_info_df['puzzle_type'] == puzzle_type, 'allowed_moves'].iloc[0]
    allowed_moves_dict = json.loads(allowed_moves_str.replace("'", '"') )
    return allowed_moves_dict


    
def show_cube_moves(puzzle_type, show_numbers = True):
    # ãƒ‘ã‚ºãƒ«ã�®å‹•ã��
    # allowed_moves_str = puzzle_info_df.loc[puzzle_info_df['puzzle_type'] == puzzle_type, 'allowed_moves'].iloc[0]
    # allowed_moves_dict = json.loads(allowed_moves_str.replace("'", '"') )
    allowed_moves_dict = get_moves(puzzle_type)
    print(allowed_moves_dict.items())

    # Ğ´Ğ»Ğ¸Ğ½Ğ° Ğ¿ĞµÑ€ĞµÑ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºĞ¸
    # è¦�ç´ æ•°
    N = len( list( allowed_moves_dict.values() )[0] )

    # Ğ¾Ğ±Ñ€Ğ°Ñ‚Ğ½Ñ‹Ğµ Ğ¼ÑƒĞ²Ñ‹
    # é€†ã�®å‹•ã��å�«ã‚�ã�¦dictã‚’ä½œæˆ�
    moves = {"origin":[i for i in range(N)]}
    for k,mv in allowed_moves_dict.items():
        moves[k] = mv
        rev_mv = [None]*N
        for i in range(N):
            rev_mv[ mv[i] ] = i
        moves["-"+k] = rev_mv
    moves["origin"] = [i for i in range(N)]

#     for k,mv in moves.items():
#         print(k,mv)


    l = int(np.sqrt(N/6))

    grid_height = l*3
    grid_width = l*4



    xbases = [l, l, l*2, l*3, 0, l]
    ybases = [l*3, l*2, l*2, l*2, l*2, l]

    # colors = generate_gradient_colors(N)
    colors = ['gray','green','red','blue','orange','yellow']

    for k,mv in moves.items():
        if k == "origin":
            continue
        # Ğ¾Ğ±Ğ¾Ğ»Ğ¾Ñ‡ĞºĞ° Ğ¿Ğ¾Ğ»Ğ¾Ñ‚Ğ½Ğ° Ğ´Ğ»Ñ� Ğ´Ğ²ÑƒÑ… Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¾Ğ² Ñ€Ğ°Ğ·Ğ¼ĞµÑ€Ğ° 12 Ğ½Ğ° 6    
        fig, axes = plt.subplots(1,2,figsize=(12,6))

        for ax, mi in zip(axes, [("before",moves["origin"]), ("after : "+k, mv)] ):
            k,mv = mi

            ax.set_xlim([0, grid_width])
            ax.set_ylim([0, grid_height])
            ax.set_xticks(range(grid_width))
            ax.set_yticks(range(grid_height))
            
            # Ñ€Ğ°Ğ·Ğ²Ğ¾Ñ€Ğ¾Ñ‚ ĞºÑƒĞ±Ğ¸ĞºĞ°
            for face in range(6):
                for i in range(l*l):
                    dx, dy = i%l, i//l
                    x = xbases[face] + dx
                    y = ybases[face] - dy - 1

                    ii= face*l*l + i
                    c = colors[mv[ii]//(l*l)]
                    # c = colors[mv[ii]]
                    #bb = face*
                    

                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color=c))
                    
                    if show_numbers:
                        ax.text(x + 0.5, y + 0.5, mv[ii], ha='center', va='center', color='black')

            ax.grid(True)
            ax.set_title(k)
            # ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    

        plt.show()

        #break




xbases = [face_span, face_span, face_span*2, 0, face_span*3, face_span]
ybases = [face_span*2, face_span, face_span,  face_span, face_span,   face_span*3]


import numpy as np
import matplotlib.pyplot as plt

def show_picture_cube_moves(puzzle_type, show_numbers=True):
    moves_src = get_moves(puzzle_type)
    N = len(next(iter(moves_src.values()))) #Ğ±ĞµÑ€ĞµĞ¼ Ğ´Ğ»Ğ¸Ğ½Ñƒ Ğ¿ĞµÑ€ĞµÑ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºĞ¸
    #assert N == 72, f"Expected 72 positions, got {N}"
    per_face = 12
    face_span = 3  # each face spans 3x3 units; center orientations are 0.5x0.5 inside

    # 12 local slots per face (dx, dy, size); (0,0) is bottom-left of the face
    # Indices on a face: 0 1 2 / 3 [4 5; 6 7] 8 / 9 10 11  (your front numbering)
    face_slots = [
    (0,2,1.0), (1,2,1.0), (2,2,1.0),        # 0,1,2
    (0,1,1.0),                               # 3
    (1.0,1.5,0.5), (1.5,1.5,0.5),            # 4,5 (Ğ²ĞµÑ€Ñ… Ñ†ĞµĞ½Ñ‚Ñ€Ğ°)
    (1.0,1.0,0.5), (1.5,1.0,0.5),            # 6,7 (Ğ½Ğ¸Ğ· Ñ†ĞµĞ½Ñ‚Ñ€Ğ°)
    (2,1,1.0),                               # 8
    (0,0,1.0), (1,0,1.0), (2,0,1.0),        # 9,10,11
]




    # Net bases (your cross): F, D, R, L, B, U
    xbases = [face_span, face_span, face_span*2, face_span*3, 0, face_span]
    ybases = [face_span*3, face_span*2, face_span*2,  face_span*2, face_span*2,   face_span]

    colors = ['gray','green','red','blue','orange','yellow']  # per destination face

    # build dict with inverses too
    moves = {"origin": list(range(N))}
    for name, perm in moves_src.items():
        moves[name] = perm
        inv = [None]*N
        for i in range(N):
            inv[perm[i]] = i
        moves["-"+name] = inv

    for name, perm in moves.items():
        if name == "origin": 
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, (title, p) in zip(axes, [("before", moves["origin"]), (f"after: {name}", perm)]):
            ax.set_xlim([0, face_span*4])
            ax.set_ylim([0, face_span*4])
            ax.set_xticks(range(face_span*4))
            ax.set_yticks(range(face_span*4))
            ax.grid(True, lw=0.5)
            ax.set_title(title)

            for face in range(6):
                bx, by = xbases[face], ybases[face]
                for i, (dx, dy, sz) in enumerate(face_slots):
                    ii = face*per_face + i
                    dst_face = p[ii] // per_face
                    x = bx + dx
                    y = by + dy
                    ax.add_patch(plt.Rectangle((x, y), sz, sz, 
                                               facecolor=colors[dst_face % 6], 
                                               edgecolor='black', lw=0.7))
                    if show_numbers:
                        ax.text(x+sz/2, y+sz/2, str(p[ii]), ha='center', va='center', fontsize=8, color="black")

        plt.tight_layout()
        plt.show()


# ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ñ�Ğ»Ğ¾Ğ²Ğ°Ñ€Ñ� Ñ…Ğ¾Ğ´Ğ¾Ğ² (Ñ�Ğ²Ğ¾Ñ‘ Ğ²Ñ�Ñ‚Ğ°Ğ²ÑŒ!)
def get_moves(puzzle_type):
    if puzzle_type == "piccube":
        return {'conz:=': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 52, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
                'move1:=': [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19,20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 40, 43, 41, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55,52, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
                'move2:=': [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 43, 42, 41, 40, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
                'f0:=': [2, 8, 11, 1, 5, 7, 4, 6, 10, 0, 3, 9, 48, 49, 50, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12, 13, 14, 27, 28, 29, 30, 31, 32, 33, 34, 35, 24, 25, 26, 39, 40, 41, 42, 43, 44, 45, 46, 47, 36, 37, 38, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71], 'f1:=': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 51, 52, 53, 54, 55, 56, 21, 22, 23, 24, 25, 26, 15, 16, 17, 18, 19, 20, 33, 34, 35, 36, 37, 38, 27, 28, 29, 30, 31, 32, 45, 46, 47, 48, 49, 50, 39, 40, 41, 42, 43, 44, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71], 'f2:=': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 57, 58, 59, 24, 25, 26, 27, 28, 29, 30, 31, 32, 21, 22, 23, 36, 37, 38, 39, 40, 41, 42, 43, 44, 33, 34, 35, 48, 49, 50, 51, 52, 53, 54, 55, 56, 45, 46, 47, 69, 63, 60, 70, 66, 64, 67, 65, 61, 71, 68, 62], 'r0:=': [0, 1, 45, 3, 4, 5, 6, 7, 39, 9, 10, 36, 12, 13, 2, 15, 16, 17, 18, 19, 8, 21, 22, 11, 26, 32, 35, 25, 29, 31, 28, 30, 34, 24, 27, 33, 71, 37, 38, 68, 40, 41, 42, 43, 44, 62, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 14, 63, 64, 65, 66, 67, 20, 69, 70, 23], 'r1:=': [0, 46, 2, 3, 43, 42, 41, 40, 8, 9, 37, 11, 12, 1, 14, 15, 4, 5, 6, 7, 20, 21, 10, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 70, 38, 39, 67, 66, 65, 64, 44, 45, 61, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 13, 62, 63, 16, 17, 18, 19, 68, 69, 22, 71], 'r2:=': [47, 1, 2, 44, 4, 5, 6, 7, 8, 38, 10, 11, 0, 13, 14, 3, 16, 17, 18, 19, 20, 9, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 69, 39, 40, 41, 42, 43, 63, 45, 46, 60, 57, 51, 48, 58, 54, 52, 55, 53, 49, 59, 56, 50, 12, 61, 62, 15, 64, 65, 66, 67, 68, 21, 70, 71], 'd0:=': [0, 1, 2, 3, 4, 5, 6, 7, 8, 24, 27, 33, 14, 20, 23, 13, 17, 19, 16, 18, 22, 12, 15, 21, 62, 25, 26, 61, 28, 29, 30, 31, 32, 60, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 11, 51, 52, 53, 54, 55, 10, 57, 58, 9, 50, 56, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71], 'd1:=': [0, 1, 2, 25, 29, 31, 28, 30, 34, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 68, 26, 27, 65, 67, 64, 66, 32, 33, 63, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 8, 50, 51, 5, 7, 4, 6, 56, 57, 3, 59, 60, 61, 62, 49, 53, 55, 52, 54, 58, 69, 70, 71], 'd2:=': [26, 32, 35, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 71, 27, 28, 29, 30, 31, 70, 33, 34, 69, 45, 39, 36, 46, 42, 40, 43, 41, 37, 47, 44, 38, 2, 49, 50, 1, 52, 53, 54, 55, 56, 0, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 48, 51, 57]}
    else:
        raise ValueError("unknown puzzle_type")

# Ğ Ğ¸Ñ�ÑƒĞµĞ¼
show_picture_cube_moves("piccube", show_numbers=True)





#show_cube_moves("cube_2/2/2")





show_cube_moves("cube_3/3/3")





show_cube_moves("cube_4/4/4")








# show_cube_moves("cube_19/19/19", show_numbers=False)

# for k,mv in get_moves("cube_19/19/19").items():
#     print(k, mv[:10])


print(get_moves("cube_33/33/33").keys())





    
def show_wreath_moves(puzzle_type, D, show_numbers = True):
    # D is the number of nodes in the left lease
    
    # ãƒ‘ã‚ºãƒ«ã�®å‹•ã��
    allowed_moves_dict = get_moves(puzzle_type)

    # è¦�ç´ æ•°
    N = len( list( allowed_moves_dict.values() )[0] )


    # é€†ã�®å‹•ã��å�«ã‚�ã�¦dictã‚’ä½œæˆ�
    moves = {"origin":[i for i in range(N)]}
    for k,mv in allowed_moves_dict.items():
        moves[k] = mv
        rev_mv = [None]*N
        for i in range(N):
            rev_mv[ mv[i] ] = i
        moves["-"+k] = rev_mv
    moves["origin"] = [i for i in range(N)]

#     for k,mv in moves.items():
#         print(k,mv)


    
    l = int((N+2)/2)

    grid_height = l*2
    grid_width = l*2.5

    r = l*0.5
    d = r *( 1- np.cos(D/l*np.pi) )

    
    for k,mv in moves.items():
        if k == "origin":
            continue
        fig, axes = plt.subplots(1,2,figsize=(16,8))

        
        for ax, mi in zip(axes, [("before",moves["origin"]), ("after : "+k, mv)] ):
            k,mv = mi

            ax.set_xlim([0, grid_width])
            ax.set_ylim([0, grid_height])

            ax.add_patch(plt.Circle((l,l), radius = r, color="black", fill=False))
            ax.add_patch(plt.Circle((l*2-d*2,l), radius = r, color="black", fill=False))

            
            for i in range(l):
                a = (i/l*2-D/l)
                x = l + r*np.cos( a*np.pi)
                y = l - r*np.sin( a*np.pi)
                c = 'b' if mv[i] < l else 'r'
                ax.plot(x, y, c+'o', markersize=15)
                ax.text(x, y, mv[i], ha='center', va='center', color='white')


            for i in range(l-2):
                a = (i/(l-1)*2-1+(D+2)/(l-1))
                x = l + r*np.cos( a*np.pi) + l - 2*d
                y = l - r*np.sin( a*np.pi)
                if i >= l-D-2:
                    x = l + r - 2*d
                    y = l + (i-l+D+2) - (D-1)/2
                c = 'b' if mv[l+i] < l else 'r'
                ax.plot(x, y, c+'o', markersize=15)
                ax.text(x, y, mv[l+i], ha='center', va='center', color='white')

            
            ax.grid(True)
            ax.set_title(k)
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

        plt.show()







show_wreath_moves("wreath_6/6", 2)





show_wreath_moves("wreath_7/7", 2)





show_wreath_moves("wreath_12/12",3)


show_wreath_moves("wreath_21/21",6)


show_wreath_moves("wreath_33/33",9)


show_wreath_moves("wreath_100/100", 25)


# print(get_moves("wreath_21/21")['r'])
# print(get_moves("wreath_100/100")['r'])





import matplotlib.colors as mcolors

def generate_gradient_colors(N):
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "orange", "yellow", "green", "blue", "indigo", "violet"])
    return [mcolors.rgb2hex(cmap(i/N)) for i in range(N)]




    
def show_globe_moves(puzzle_type,show_numbers = True):
    # ãƒ‘ã‚ºãƒ«ã�®å‹•ã��
    allowed_moves_dict = get_moves(puzzle_type)

    # è¦�ç´ æ•°
    N = len( list( allowed_moves_dict.values() )[0] )

    # é€†ã�®å‹•ã��å�«ã‚�ã�¦dictã‚’ä½œæˆ�
    moves = {"origin":[i for i in range(N)]}
    for k,mv in allowed_moves_dict.items():
        moves[k] = mv
        rev_mv = [None]*N
        for i in range(N):
            rev_mv[ mv[i] ] = i
        moves["-"+k] = rev_mv
    moves["origin"] = [i for i in range(N)]

    #for k,mv in moves.items():
    #    print(k,mv)

    
    W = int(puzzle_type.split('/')[-1])*2
    H = N//W

    grid_height = H
    grid_width = W

    colors = generate_gradient_colors(W)

    for k,mv in moves.items():
        if k == "origin":
            continue
        fig, axes = plt.subplots(1,2,figsize=(12,6))

        for ax, mi in zip(axes, [("before",moves["origin"]), ("after : "+k, mv)] ):
            k,mv = mi

            
            ax.set_xlim([0, grid_width])
            ax.set_ylim([0, grid_height])
            ax.set_xticks(range(grid_width))
            ax.set_yticks(range(grid_height))
            

            for h in range(H):
                for w in range(W):
                    x = w
                    y = H-h-1
                    ii = h*W + w
                    c = colors[mv[ii]%W]

                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color=c))
                    if show_numbers:
                        ax.text(x + 0.5, y + 0.5, mv[ii], ha='center', va='center', color='black')

            ax.grid(True)
            ax.set_title(k)
            # ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

        plt.show()




show_globe_moves("globe_1/16")


show_globe_moves("globe_2/6")


show_globe_moves("globe_3/4")





show_globe_moves("globe_6/4")








puzzle_info_df.loc[puzzle_info_df["puzzle_type"]=="globe_2/6"].iloc[0,1].split("'")


for puzzle_type in puzzle_info_df["puzzle_type"]:
    if not "globe" in puzzle_type:
        continue
    
    print("---")
    print(puzzle_type)
    mv = get_moves(puzzle_type)["f0"]
    N = len(mv)
    W = int(puzzle_type.split('/')[-1])*2
    H = N//W
    for i in range(H):
        print(mv[W*i : W*(i+1) ])









for puzzle_type in puzzle_info_df["puzzle_type"]:
    if not "globe" in puzzle_type:
        continue
    
    print(puzzle_type)
    print(get_moves(puzzle_type).keys())










