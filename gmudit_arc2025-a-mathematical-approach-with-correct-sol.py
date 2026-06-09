import json as js
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



from matplotlib import colors
cmap = colors.ListedColormap([

    '#8B00FF',  # Violet
    '#4B0082',  # Indigo
    '#0000FF',  # Blue
    '#FFFF00',  # Yellow
    '#00FF00',  # Green
    '#FF7F00',  # Orange
    '#FF0000',  # Red
    '#964B00',  # Golden
    '#000000',  # Black
    '#FFFFFF',  # White
])
norm = colors.Normalize(vmin=0, vmax=9)




with open('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json') as f:
    train_challenges = js.load(f)
    
with open('/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json') as f:
    train_sols = js.load(f)


with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json') as f:
    train_eval_challenges = js.load(f)

with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json') as f:
    train_eval_sols = js.load(f)


def visualize_challenges(df,num_visualize):
    i = 0
    for key,value in df.items():
        if i >=num_visualize:
            break
        print(f"ID = {key}\n")
        n = len(value['train'])
        print(f"TRAINING DATA :")
        for j in range(n):
            mat_inp = np.array(value['train'][j]['input']) 
            shape_i = mat_inp.shape
            mat_out = np.array(value['train'][j]['output'])
            shape_o = mat_out.shape
            plt.figure(figsize=(5, 5))
            plt.subplot(1,2,1)
            plt.title(f"INPUT MATRIX : {shape_i}")
            plt.imshow(mat_inp, cmap=cmap, norm=norm)
            plt.subplot(1,2,2)
            plt.title(f"OUTPUT MATRIX : {shape_o}")
            plt.imshow(mat_out, cmap=cmap, norm=norm)
            plt.axis('off')
            plt.show()
            print()
        m = len(value['test'])
        print(f"TESTING DATA : num ques = {m}")
        for k in range(m):
            mat_test = np.array(value['test'][k]['input'])
            shape_test = mat_test.shape 
            plt.figure(figsize = (5,5))
            plt.title(f"TEST COLOR MATRIX : ORDER = {shape_test}")
            plt.imshow(mat_test,cmap=cmap, norm=norm)
            plt.axis('off')
            plt.show()
        print()
        i+=1


def visualize_solutions(df_sols,df_challenges,num_visualize): 
    i = 0
    for key,value in df_sols.items():
        if i>=num_visualize:
            break
        print(f"ID = {key}")
        n = len(value)
        print(f"Number of questions = {n}")
        for j in range(n):
            mat_ques = np.array(df_challenges[key]['test'][j]['input'])
            shape_ques = mat_ques.shape
            mat_sol = np.array(value[j])
            shape_sol = mat_sol.shape 
            plt.figure(figsize=(5,5))
            plt.subplot(1,2,1)
            plt.title(f"QUESTION : {shape_ques}")
            plt.imshow(mat_ques, cmap=cmap, norm=norm)
            plt.subplot(1,2,2)
            plt.title(f"SOLUTION : {shape_sol}")
            plt.imshow(mat_sol, cmap=cmap, norm=norm)
            plt.axis('off')
            plt.show()
        i+=1


visualize_challenges(train_challenges,1)


visualize_solutions(train_sols,train_challenges,1)


with open("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json","r") as f:
    test = js.load(f)


visualize_challenges(test,1)


print("Trainaing\n")
i = 0
for key,value in train_challenges.items():
    if i >= 1:break
    print(f"KEY = {key}")
    n = len(value['train'])
    print(f"Number of training examples = {n}")
    print(value['train'])
    Q1,A1 = value['train'][0]['input'], value['train'][0]['output']
    Q2,A2 = value['train'][1]['input'], value['train'][1]['output']
    i+=1
print("\nTest")
i = 0
for key,value in test.items():
    if i>=1: break
    print(f"KEY = {key}")
    test_ques = np.array(value['test'][0]['input'])
    print(test_ques)
    i+=1


print(f"Trainiang Example 1 :\nInput 1: \n{np.array(Q1)}\nOutput 1: \n{np.array(A1)}\n\nInput 2 :\n{np.array(Q2)}\nOutput 2 :\n{np.array(A2)}")


# Vectorize Q1, Q2 and A1, A2
q1_vec = np.array(Q1).reshape(-1, 1)  # shape (4, 1)
q2_vec = np.array(Q2).reshape(-1, 1)  # shape (4, 1)
a1_vec = np.array(A1).reshape(-1, 1)  # shape (36, 1)
a2_vec = np.array(A2).reshape(-1, 1)  # shape (36, 1)


# Stack to form Q and A
Q_stacked = np.hstack([q1_vec, q2_vec])  # shape (4, 2)
A_stacked = np.hstack([a1_vec, a2_vec])  # shape (36, 2)


from numpy.linalg import pinv


T1 = A_stacked @ pinv(Q_stacked)
T1


(T1 @ q1_vec).reshape(6,6)


(T1 @ q2_vec).reshape(6,6)


reconstructed_a1 = T1 @ q1_vec
reconstructed_a2 = T1 @ q2_vec

# Compute errors
error1 = np.linalg.norm(reconstructed_a1 - a1_vec)
error2 = np.linalg.norm(reconstructed_a2 - a2_vec)

print(f"Error in solving 1st example ={error1}\nError in solving 2nd example = {error2}")



test_ques
plt.imshow(test_ques, cmap=cmap, norm=norm)


test_ans = np.round((T1@test_ques.reshape(-1,1)).reshape(6,6))
plt.imshow(test_ans, cmap=cmap, norm=norm)


def construct_T1():

    T1 = np.zeros((36, 4), dtype=int)
    indices = [
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,
        1, 0, 1, 0, 1, 0,
        3, 2, 3, 2, 3, 2,
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3
    ]
    for i, idx in enumerate(indices):
        T1[i, idx] = 1
    return T1
act_T1 = construct_T1()


print(f"Q1 :\n{np.array(Q1)}")
print()
print(f"Q2 :\n{np.array(Q2)}")


print(f"Ans 1 : \n{(act_T1@np.array(Q1).reshape(-1,1)).reshape(6,6)}")
print()
print(f"Ans 2 : \n{(act_T1@np.array(Q2).reshape(-1,1)).reshape(6,6)}")


plt.imshow(test_ques, cmap=cmap, norm=norm)


correct_ans = (act_T1 @ test_ques.reshape(-1,1)).reshape(6,6)
plt.imshow(correct_ans, cmap=cmap, norm=norm)


from sympy import symbols, Matrix, solve, transpose




q1 = np.array(train_challenges["00576224"]['train'][0]['input'])
print(f"Q1 : \n{q1}\n")
a1 = np.array(train_challenges["00576224"]['train'][0]['output'])
print(f"A1 : \n{a1}\n")
print()
q2 = np.array(train_challenges["00576224"]['train'][1]['input'])
print(f"Q2 : \n{q2}\n")
a2 = np.array(train_challenges["00576224"]['train'][1]['output'])
print(f"A2 : \n{a2}\n")
print()
test1 = np.array(train_challenges["00576224"]['test'][0]['input'])
print(f"Test_challenge 1 : \n{test1}\n")




def convert_ques_to_sym(mat):
    n,m = len(mat),len(mat[0])
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    
    uniques = list(set([x[0] for x in np.array(q1).reshape(-1,1)]))
    variables = list(alphabets[:len(uniques)])
    map_dict = dict(zip(uniques, variables))
    inverse_map_dict = dict(zip(variables,uniques))
    
    res = [[0 for i in range(m)] for j in range(n)]
    for i in range(n):
        for j in range(m):
            res[i][j] = map_dict[mat[i][j]]
    return res,map_dict,inverse_map_dict


def convert_ans_to_sym(mat, map_dict):
    mat = np.array(mat)
    n,m = len(mat),len(mat[0])
    res = [[0 for i in range(m)] for j in range(n)]
    for i in range(n):
        for j in range(m):
            res[i][j] = map_dict[mat[i][j]]
    return res


q1_symbollic, mappings_f, mappings_b = convert_ques_to_sym(q1)
q1_sym = Matrix(q1_symbollic).reshape(len(q1_symbollic)*len(q1_symbollic[0]),1)
q1_sym


a1_sym = Matrix(convert_ans_to_sym(a1,mappings_f)).reshape(len(a1)*len(a1[0]),1)
a1_sym


# Calculating psuedo-inverse

v = q1_sym
vTv = transpose(v) * v  
v_pinv = transpose(v) / vTv[0]
v_pinv


T1 = a1_sym * v_pinv
T1


q1_sym.reshape(2,2)


q1


q1_sym.subs(mappings_b).reshape(2,2)


a1_sym.subs(mappings_b).reshape(6,6)


T_1 = T1.subs(mappings_b)
(T_1 @ np.array(q1).reshape(-1,1)).reshape(6,6)


q1_sym.reshape(2,2)


q2


new_mapping = {'a':6, 'b':4, 'c':6, 'd':8}


T_2 = T1.subs(new_mapping)


(T_2 @ np.array(q2).reshape(-1,1)).reshape(6,6)


test1


test_mappings = {'d':3,'a':2,'c':7,'b':8}


T_test = T1.subs(test_mappings)


result = (T_test @ np.array(test1).reshape(-1,1)).reshape(6,6)
result


num_result = np.array(result.tolist(),dtype='int')


plt.subplot(1,2,1)
plt.imshow(test1, cmap=cmap, norm=norm)

plt.subplot(1,2,2)
plt.imshow(np.array(num_result),cmap=cmap, norm=norm)

plt.show()




