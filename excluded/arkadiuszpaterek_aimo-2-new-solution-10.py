!nvidia-smi


%%writefile dq_data4h.py
LD_tasks = [
	{'i_dataset':4, 'dataset':'reference.csv', 'i_task':5, 'question':'Let $ABC$ be a triangle with $BC=108$, $CA=126$, and $AB=39$. Point $X$ lies on segment $AC$ such that $BX$ bisects $\\angle CBA$. Let $\\omega$ be the circumcircle of triangle $ABX$. Let $Y$ be a point on $\\omega$ different from $X$ such that $CX=CY$. Line $XY$ meets $BC$ at $E$. The length of the segment $BE$ can be written as $\\frac{m}{n}$, where $m$ and $n$ are coprime positive integers. Find $m+n$.', 'correct_answer':751, },
	{'i_dataset':4, 'dataset':'reference.csv', 'i_task':4, 'question':'We call a sequence $a_1, a_2, \\ldots$ of non-negative integers \\textit{delightful} if there exists a positive integer $N$ such that for all $n > N$, $a_n = 0$, and for all $i \\geq 1$, $a_i$ counts the number of multiples of $i$ in $a_1, a_2, \\ldots, a_N$. How many delightful sequences of non-negative integers are there?', 'correct_answer':3, },
	{'i_dataset':2, 'dataset':'aime', 'i_task':18, 'question':'Let $a, b, x,$ and $y$ be real numbers with $a>4$ and $b>1$ such that\\[\\frac{x^2}{a^2}+\\frac{y^2}{a^2-16}=\\frac{(x-20)^2}{b^2-1}+\\frac{(y-11)^2}{b^2}=1.\\]Find the least possible value of $a+b.$', 'correct_answer':23, },
	{'i_dataset':2, 'dataset':'aime', 'i_task':43, 'question':'Rhombus $ABCD$ has $\\angle BAD < 90^\\circ.$ There is a point $P$ on the incircle of the rhombus such that the distances from $P$ to the lines $DA,AB,$ and $BC$ are $9,$ $5,$ and $16,$ respectively. Find the perimeter of $ABCD.$', 'correct_answer':125, },
	{'i_dataset':2, 'dataset':'aime', 'i_task':56, 'question':'Consider the L-shaped region formed by three unit squares joined at their sides, as shown below. Two points $A$ and $B$ are chosen independently and uniformly at random from inside the region. The probability that the midpoint of $\\overline{AB}$ also lies inside this L-shaped region can be expressed as $\\frac{m}{n},$ where $m$ and $n$ are relatively prime positive integers. Find $m+n.$\n[asy] unitsize(2cm); draw((0,0)--(2,0)--(2,1)--(1,1)--(1,2)--(0,2)--cycle); draw((0,1)--(1,1)--(1,0),dashed); [/asy]', 'correct_answer':35, },
	{'i_dataset':2, 'dataset':'aime', 'i_task':73, 'question':'Eight circles of radius $34$ are sequentially tangent, and two of the circles are tangent to $AB$ and $BC$ of triangle $ABC$, respectively. $2024$ circles of radius $1$ can be arranged in the same manner. The inradius of triangle $ABC$ can be expressed as $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.', 'correct_answer':197, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':7, 'question':'Find the least positive integer $n$ for which there exists a set $\\left\\{s_{1}, s_{2}, \\ldots, s_{n}\\right\\}$ consisting of $n$ distinct positive integers such that\n\n$$\n\\left(1-\\frac{1}{s_{1}}\\right)\\left(1-\\frac{1}{s_{2}}\\right) \\ldots\\left(1-\\frac{1}{s_{n}}\\right)=\\frac{51}{2010}\n$$', 'correct_answer':39, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':8, 'question':'Find the smallest number $n$ such that there exist polynomials $f_{1}, f_{2}, \\ldots, f_{n}$ with rational coefficients satisfying\n\n$$\nx^{2}+7=f_{1}(x)^{2}+f_{2}(x)^{2}+\\cdots+f_{n}(x)^{2} .\n$$', 'correct_answer':5, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':82, 'question':'Three thin metal rods of lengths 9,12 and 15 are welded together to form a right-angled triangle, which is held in a horizontal position. A solid sphere of radius 5 rests in the triangle so that it is tangent to each of the three sides. Assuming that the thickness of the rods can be neglected, how high above the plane of the triangle is the top of the sphere?', 'correct_answer':5, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':151, 'question':'Let $A R M L$ be a trapezoid with bases $\\overline{A R}$ and $\\overline{M L}$, such that $M R=R A=A L$ and $L R=$ $A M=M L$. Point $P$ lies inside the trapezoid such that $\\angle R M P=12^{\\circ}$ and $\\angle R A P=6^{\\circ}$. Diagonals $A M$ and $R L$ intersect at $D$. Compute the measure, in degrees, of angle $A P D$.', 'correct_answer':48, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':204, 'question':'In triangle $A B C, A B=B C$. A trisector of $\\angle B$ intersects $\\overline{A C}$ at $D$. If $A B, A C$, and $B D$ are integers and $A B-B D=7$, compute $A C$.', 'correct_answer':146, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':223, 'question':'Given that April $1^{\\text {st }}, 2012$ fell on a Sunday, what is the next year in which April $1^{\\text {st }}$ will fall on a Sunday?', 'correct_answer':18, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':236, 'question':'In $\\triangle A B C, D$ is on $\\overline{A C}$ so that $\\overline{B D}$ is the angle bisector of $\\angle B$. Point $E$ is on $\\overline{A B}$ and $\\overline{C E}$ intersects $\\overline{B D}$ at $P$. Quadrilateral $B C D E$ is cyclic, $B P=12$ and $P E=4$. Compute the ratio $\\frac{A C}{A E}$.', 'correct_answer':3, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':239, 'question':'Points $A, R, M$, and $L$ are consecutively the midpoints of the sides of a square whose area is 650. The coordinates of point $A$ are $(11,5)$. If points $R, M$, and $L$ are all lattice points, and $R$ is in Quadrant I, compute the number of possible ordered pairs $(x, y)$ of coordinates for point $R$.', 'correct_answer':10, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':255, 'question':'Let $x$ be a real number in the interval $[0,360]$ such that the four expressions $\\sin x^{\\circ}, \\cos x^{\\circ}$, $\\tan x^{\\circ}, \\cot x^{\\circ}$ take on exactly three distinct (finite) real values. Compute the sum of all possible values of $x$.', 'correct_answer':990, },
	{'i_dataset':3, 'dataset':'olympiadbench2.json', 'i_task':299, 'question':'Elizabeth is in an "escape room" puzzle. She is in a room with one door which is locked at the start of the puzzle. The room contains $n$ light switches, each of which is initially off. Each minute, she must flip exactly $k$ different light switches (to "flip" a switch means to turn it on if it is currently off, and off if it is currently on). At the end of each minute, if all of the switches are on, then the door unlocks and Elizabeth escapes from the room.\n\nLet $E(n, k)$ be the minimum number of minutes required for Elizabeth to escape, for positive integers $n, k$ with $k \\leq n$. For example, $E(2,1)=2$ because Elizabeth cannot escape in one minute (there are two switches and one must be flipped every minute) but she can escape in two minutes (by flipping Switch 1 in the first minute and Switch 2 in the second minute). Define $E(n, k)=\\infty$ if the puzzle is impossible to solve (that is, if it is impossible to have all switches on at the end of any minute).\n\nFor convenience, assume the $n$ light switches are numbered 1 through $n$.\nFind the $E(2020,1993)$', 'correct_answer':76, },
	{'i_dataset':5, 'dataset':'o.csv', 'i_task':0, 'question':'There are $2022$ equally spaced points on a circular track $\\gamma$ of circumference $2022$. The points are labeled $A_1, A_2, \\ldots, A_{2022}$ in some order, each label used once. Initially, Bunbun the Bunny begins at $A_1$. She hops along $\\gamma$ from $A_1$ to $A_2$, then from $A_2$ to $A_3$, until she reaches $A_{2022}$, after which she hops back to $A_1$. When hopping from $P$ to $Q$, she always hops along the shorter of the two arcs $\\widehat{PQ}$ of $\\gamma$; if $\\overline{PQ}$ is a diameter of $\\gamma$, she moves along either semicircle.\n\nDetermine the maximal possible sum of the lengths of the $2022$ arcs which Bunbun traveled, over all possible labellings of the $2022$ points.\n', 'correct_answer':222, },
	{'i_dataset':5, 'dataset':'o.csv', 'i_task':7, 'question':'Let $a_1,a_2,\\cdots,a_{41}\\in\\mathbb{R},$ such that $a_{41}=a_1, \\sum_{i=1}^{40}a_i=0,$ and for any $i=1,2,\\cdots,40, |a_i-a_{i+1}|\\leq 1.$ Determine the greatest possible value of\n$(1)a_{10}+a_{20}+a_{30}+a_{40};$\n$(2)a_{10}\\cdot a_{20}+a_{30}\\cdot a_{40}.$', 'correct_answer':10, },
	{'i_dataset':5, 'dataset':'o.csv', 'i_task':14, 'question':"$ S$ is a non-empty subset of the set $ \\{ 1, 2, \\cdots, 108 \\}$, satisfying:\n\n(1) For any two numbers $ a,b \\in S$ ( may not distinct), there exists $ c \\in S$, such that $ \\gcd(a,c)\\equal{}\\gcd(b,c)\\equal{}1$.\n\n(2) For any two numbers $ a,b \\in S$ ( may not distinct), there exists $ c' \\in S$, $ c' \\neq a$, $ c' \\neq b$, such that $ \\gcd(a, c') > 1$, $ \\gcd(b,c') >1$.\n\nFind the largest possible value of $ |S|$.", 'correct_answer':79, },
	{'i_dataset':5, 'dataset':'o.csv', 'i_task':20, 'question':'Suppose $A_1,A_2,\\cdots ,A_n \\subseteq \\left \\{ 1,2,\\cdots ,2018 \\right \\}$ and $\\left | A_i \\right |=2, i=1,2,\\cdots ,n$, satisfying that $$A_i + A_j, \\; 1 \\le i \\le j \\le n ,$$ are distinct from each other. $A + B = \\left \\{ a+b|a\\in A,\\,b\\in B \\right \\}$. Determine the maximal value of $n$.', 'correct_answer':33, },
	{'i_dataset':5, 'dataset':'o.csv', 'i_task':42, 'question':'For each positive integer $n$, let $k(n)$ be the number of ones in the binary representation of $2023 \\cdot n$. What is the minimum value of $k(n)$?', 'correct_answer':3, },
	{'i_dataset':6, 'dataset':'aime25', 'i_task':1, 'question':'On \\( \\triangle ABC \\), points \\( A, D, E, \\) and \\( B \\) lie in that order on side \\( \\overline{AB} \\) with \\( AD = 4 \\), \\( DE = 16 \\), and \\( EB = 8 \\). Points \\( A, F, G, \\) and \\( C \\) lie in that order on side \\( \\overline{AC} \\) with \\( AF = 13 \\), \\( FG = 52 \\), and \\( GC = 26 \\). Let \\( M \\) be the reflection of \\( D \\) through \\( F \\), and let \\( N \\) be the reflection of \\( G \\) through \\( E \\). Quadrilateral \\( DEGF \\) has area 288. Find the area of heptagon \\( AFNBCEM \\).', 'correct_answer':588, },
	{'i_dataset':6, 'dataset':'aime25', 'i_task':7, 'question':'Let \\( k \\) be a real number such that the system\n\\[\n|25 + 20i - z| = 5\n\\]\n\\[\n|z - 4 - k| = |z - 3i - k|\n\\]\nhas exactly one complex solution \\( z \\). The sum of all possible values of \\( k \\) can be written as \\( \\frac{m}{n} \\), where \\( m \\) and \\( n \\) are relatively prime positive integers. Find \\( m + n \\). Here \\( i = \\sqrt{-1} \\).', 'correct_answer':77, },
	{'i_dataset':6, 'dataset':'aime25', 'i_task':15, 'question':'Six points $A, B, C, D, E$ and $F$ lie in a straight line in that order. Suppose that $G$ is a point not on the line and that $AC = 26$, $BD = 22$, $CE = 31$, $DF = 33$, $AF = 73$, $CG = 40$, and $DG = 30$. Find the area of $\\triangle BGE$.', 'correct_answer':468, },
	{'i_dataset':6, 'dataset':'aime25', 'i_task':25, 'question':'Let $S$ be the set of vertices of a regular $24$-gon. Find the number of ways to draw $12$ segments of equal lengths so that each vertex in $S$ is an endpoint of exactly one of the $12$ segments.', 'correct_answer':113, },
]



T = True
F = False
l = len

submit = F
conf_pc = F

import os, sys, time, re

D_args_default = {
	'save_dir': '14',
	'save_prefix': 'X',
	'runs': '1',
	'gpus': '0,1,2,3',
	'model_size': '32',
}

if conf_pc:
	D_args = {}
	for arg in sys.argv[1:]:
		L_arg = arg.split('=')
		assert(l(L_arg) == 2)
		D_args[L_arg[0]] = L_arg[1]
	assert(all(key in D_args for key in D_args_default))
else:
	D_args = D_args_default

# before submitting set submit=T, time_available, time_extra

if submit:
	save_answers = F
	pass_exceptions = F
	use_control = T
	prior_time_per_task = 345
	time_available = 4.85*3600
	N_tasks = 50
	run_main_loop = F
else:
	save_answers = T if conf_pc else T
	pass_exceptions = T if conf_pc else F
	use_control = T if conf_pc else T
	prior_time_per_task = 560 if conf_pc else 345
	N_tasks = 25 if conf_pc else 25
	time_available = prior_time_per_task*N_tasks
	run_main_loop = T

time_extra = (80 if conf_pc else 250) + 50
use_parse_latex = F

time_available_final = time_available + time_extra
time_remaining = time_available
	
code_timeout = 12
code_timeout2 = 12

if conf_pc:
	llm_path = '../../dl/llm/deepseek-r1-distill-qwen-{}b-awq/'.format(D_args['model_size'])
else:
	llm_path = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-{}b-awq/1'.format(D_args['model_size'])

if submit:
	assert(not conf_pc)

N_runs = int(D_args['runs'])

prefix_filename_save_answers = D_args['save_prefix']
dir_save_answers = 'output_answers{}'.format(D_args['save_dir'])

import numpy as np
np.random.seed()

N_gpus = l(D_args['gpus'].split(','))

os.environ['CUDA_VISIBLE_DEVICES'] = D_args['gpus']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['DO_NOT_TRACK'] = '1'
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'
#os.environ['VLLM_LOGGING_LEVEL'] = 'INFO'

for dirname in [dir_save_answers, 'tmp', 'output_llm0']:
	os.system('mkdir -p ' + dirname)

default_answer = 9

# configure what to print
verbose = {
	'error': T,
	'warning': T,
	'exception': T,
	'answer': T,
	'answer_info': T,
	'result_summary': T,
	'result_task': T,
	'control': T,
	'output':T,
	'lean_info': F,
	'prompt_info': F,
	'info': T,
}

use_colors = T

env0 = os.environ.copy()
L_thread_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "ANYODE_NUM_THREADS"]
for s in L_thread_vars:
	os.environ[s] = "8"
	env0[s] = "8"

import asyncio, nest_asyncio, gc, psutil, pdb, traceback, sympy, math
from sympy.parsing.latex import parse_latex

nest_asyncio.apply()

N_max_threads = os.cpu_count()

np.set_printoptions(threshold=sys.maxsize)
r = np.round
l = len
ar = np.array
t = pdb.set_trace
exc = Exception('')

def p(conf_verbose, *args, **kwargs):
	if verbose[conf_verbose]:
		if conf_verbose == 'error':
			args = ['ERROR:'] + list(args)
		elif conf_verbose == 'warning':
			args = ['Warning:'] + list(args)
		print(*args, **kwargs, flush=T)

def init_time():
	global start_time, last_time
	start_time = time.time()
	last_time = start_time

def step_time():
	global start_time, last_time
	current_time = time.time()
	result = "step_time={:.2f}s total_time={:.0f}s={:.2f}h".format(current_time - last_time, current_time - start_time, (current_time - start_time)/3600)
	last_time = current_time
	return result

init_time()


N_max_tokens = (8+4)*1024

D_prompt_format = {
	0: 'Task:\n\n{}\n\nSolve it with a sympy-based program. Do not write functions that just return a number without any computation. The answer will be likely between 101 and 999, but it can be larger or smaller. Write the final answer in boxed. Always ask yourself what is wrong in the last step you made. Doubt yourself and recheck and recount every important calculation. Solve the task as fast as possible, but be very careful to not make any mistake. You will never give answer directly until you have enough confidence. When you are not certain of the last step, recheck it multiple times.\n\n',

	1: '<|im_start|>user\nTask:\n{}\n\nInstructions:\nSolve the problem step by step with deep reasoning and put your final answer within \\boxed after taking modulo 1000.<|im_end|>\n<|im_start|>assistant\n\n',

	2: '<|im_start|>user\nTask:\n{}\n\nInstructions:\nSolve the problem and put your final answer within \\boxed after taking modulo 1000. Doubt yourself and recheck and recount every important calculation. You will never give answer directly until you have enough confidence.<|im_end|>\n<|im_start|>assistant\n\n',

	3: 'Solve the task step by step.\n\nQuestion:\n\n{}\n\nSolution:\n\n',

	4: 'Task:\n\n{}\n\nFirst reflect on every sentence from the task and rewrite that sentence in mathematical notation.\n\nSolve the task by reasoning step by step, and then writing a program in Python, that calculates the correct answer in less than one second.\n\nThink how to make the program very fast. Remember that the answer must be given modulo 1000, so there are only 1000 possible answers, and all computations can be done modulo 1000. Perhaps the right way to solve it in one second is to loop over all possible answers, and each time try to calculate a contradiction with an answer assumed and fixed. Calculate approximately the total number of basic operations performed within the main loop. Algorithmically decrease the number of operations needed.\n\nThen write the code that is very fast. Write the final answer in \\boxed .\n\nSolution:\n\n',

	5: 'Task:\n\n{}\n\nSolve it with the sp.solve function from the sympy library.\n\nFirst reflect on every sentence from the task and rewrite that sentence in mathematical notation.\n\nSolve the task by reasoning step by step, and then writing a program in Python, that calculates the correct answer in less than one second.\n\nThink how to make the program very fast. Remember that the answer must be given modulo 1000, so there are only 1000 possible answers, and all computations can be done modulo 1000. Perhaps the right way to solve it in one second is to loop over all possible answers, and each time try to calculate a contradiction with an answer assumed and fixed. Calculate approximately the total number of basic operations performed within the main loop. Algorithmically decrease the number of operations needed.\n\nThen write the code that is very fast. Write the final answer in \\boxed .\n\nSolution:\n\n',

	8: 'Task:\n\n{} Give the answer modulo 1000.\n\nFirst reflect on every sentence from the task and rewrite that sentence in mathematical notation.\n\nSolve the task by reasoning step by step, and then writing a program in Python, that calculates the correct answer.\n\nWrite the final answer in \\boxed .\n\nSolution:\n\n',

	10: '{}',
	12: '{}',
	13: '{}\n\nNow there are three possible ways to solve this task:\n',
	14: '{}\n\nI see the following symmetries:\n',

	100: '{}',

	900: ('{}\n\nI understand enough to write a program in Python that calculates the final answer. To save space, I am writing the code without any comments.\n\n### Final Solution Code\n\n```python\n', ['```']),

	901: ('{}\n\nI understand enough to write a sympy-based program that calculates the final answer. The answer must be given modulo 1000, so there are only 1000 possible answers, and probably all computations can be done modulo 1000.\n\n### Final Solution Code\n\n```python\n', ['```']),

	902: ("{}\n\nEvery sentence in the task is important. The task was:\n{}\nThe answer must be given modulo 1000, so there are only 1000 possible answers, and probably all computations can be done modulo 1000.\n\nLet's verify the solution by writing and running a sympy-based program.\n\n### Final Solution Code\n\n```python\n", ['```']),

	903: ("{}\n\nEvery sentence in the task is important. The task was:\n{}\n\nLet's verify the solution by writing and running a sympy-based program.\n\n### Final Solution Code\n\n```python\n", ['```']),

	910: ('{}\n\nI understand enough to write a program in Python that calculates the final integer answer. To save space, I am writing the code without any comments.\n\n### Final Solution Code\n\n```python\n', ['```']),

	911: ('{}\n\nI understand enough to write a sympy-based program that calculates the final answer. The answer must be given modulo 1000, so there are only 1000 possible answers, and probably all computations can be done modulo 1000.\n\n### Final Solution Code\n\n```python\n', ['```']),
}

L_prompts1 = [i for i in range(10) if i in D_prompt_format]
L_prompts2 = [i for i in range(10,100) if i in D_prompt_format]
L_prompts3 = [i for i in range(100,200) if i in D_prompt_format]
L_prompts4 = [i for i in range(900,910) if i in D_prompt_format]
L_prompts5 = [i for i in range(910,920) if i in D_prompt_format]

t0 = .7
t1 = .5
t2 = .2
t3 = .15

default_tokens_phase1 = 5*1024+256
default_tokens_phase1_prompt3 = default_tokens_phase1 - 768

D_gen_conf = {
	10: {'tokens':768, 'chars_shift':0, 'n':1, 'temp':t1, 'next_after_stop':L_prompts5, 'next_after_no_stop':L_prompts5+[100]},
	12: {'tokens':3*768, 'chars_shift':0, 'n':1, 'temp':t1, 'next_after_stop':[], 'next_after_no_stop':[]},
	13: {'tokens':3*768, 'chars_shift':32, 'n':1, 'temp':t1, 'next_after_stop':[], 'next_after_no_stop':[]},
	14: {'tokens':3*768, 'chars_shift':32, 'n':1, 'temp':t1, 'next_after_stop':[], 'next_after_no_stop':[]},

	100: {'tokens':768, 'chars_shift':0, 'n':1, 'temp':t2, 'next_after_stop':[], 'next_after_no_stop':[]},
}

for i in L_prompts1:
	if i == 3:
		D_gen_conf[i] = {'tokens':default_tokens_phase1_prompt3, 'chars_shift':0, 'n':1, 'temp':t0, 'next_after_stop':L_prompts4, 'next_after_no_stop':[12,13,14]}
	else:
		D_gen_conf[i] = {'tokens':default_tokens_phase1, 'chars_shift':0, 'n':1, 'temp':t0, 'next_after_stop':L_prompts4, 'next_after_no_stop':L_prompts4+[10]}

for i in L_prompts4:
	D_gen_conf[i] = {'tokens':768, 'chars_shift':32, 'n':3, 'temp':t3, 'next_after_stop':[], 'next_after_no_stop':[]}

for i in L_prompts5:
	D_gen_conf[i] = {'tokens':768, 'chars_shift':32, 'n':3, 'temp':t3, 'next_after_stop':[], 'next_after_no_stop':[]}

L_i_prompt_initial = L_prompts1

def dfs(L):
	return L + [k2 for k in L for k2 in dfs(D_gen_conf[k]['next_after_stop'] + D_gen_conf[k]['next_after_no_stop'])]

def verify_D_gen_conf():
	L_reachable = list(np.unique(dfs(L_i_prompt_initial)))
	assert(all(k in L_reachable for k in D_gen_conf))
	assert(all(k in D_gen_conf for k in L_reachable))

verify_D_gen_conf()

assert(all(i_prompt in D_prompt_format for i_prompt in D_gen_conf))
assert(all(i_prompt in D_gen_conf for i_prompt in D_prompt_format))
assert(all(l(s_prompt.split('{}'))==2 for i_prompt,s_prompt in D_prompt_format.items() if i_prompt in L_i_prompt_initial))
assert(all(l(s_prompt.split('{}'))>=2 for i_prompt,x in D_prompt_format.items() if i_prompt not in L_i_prompt_initial for s_prompt in [x if type(x)==str else x[0]]))
assert(all(not s_prompt.startswith('{}') for i_prompt,s_prompt in D_prompt_format.items() if i_prompt in L_i_prompt_initial))
assert(all(s_prompt.startswith('{}') for i_prompt,x in D_prompt_format.items() if i_prompt not in L_i_prompt_initial for s_prompt in [x if type(x)==str else x[0]]))


def custom_assert(cond, s_cond, *values):
	if not cond:
		p('error', s_cond, *values)
		if conf_pc:
			assert(False)

def create_prompt(i_prompt, *args):
	obj = D_prompt_format[i_prompt]
	if type(obj) == tuple:
		s_format = obj[0]
	else:
		s_format = obj
	custom_assert(l(args) == l(s_format.split('{}'))-1, "l(args) == l(s_format.split('{}'))-1", l(args), l(s_format.split('{}'))-1)
	return s_format.format(*args)

c1 = 0,200,0
c2 = 0,0,50
esc = '\x1B'
def code_set_color_fg_rgb(r, g, b):
	return '{}[38;2;{};{};{}m'.format(esc, r, g, b)
def code_set_color_bg_rgb(r, g, b):
	return '{}[48;2;{};{};{}m'.format(esc, r, g, b)
def code_set_color_rgb(r, g, b, b_r, b_g, b_b):
	return code_set_color_fg_rgb(r, g, b) + code_set_color_bg_rgb(b_r, b_g, b_b)
code_reset_color = '{}[0m'.format(esc)
color1 = code_set_color_rgb(*c1, *c2)

def load_outputs(filename):
	d = {}
	# todo: utf-8? decode?
	with open(filename, 'r') as f:
		L = f.read().split('\n===end_output')
		output = L[0]
		for part in L[1:]:
			L2 = part.split('===\n')
			idx = L2[0][1:]
			d[idx] = output
			output = '===\n'.join(L2[1:])
	return d

def extract_output(L_prompt, line):
	a = re.match('^ready ([^ ]*) n=([^ ]*) max_tokens=([^ ]*) ', line)
	filename = a.group(1)

	D_outputs = load_outputs(filename)

	L_output = []
	for s_idx,s_output in D_outputs.items():
		i,_ = [int(k) for k in s_idx.split('_')]
		L_output.append((L_prompt[i], s_output))

	return L_output

def find_answer_in_str(text):
	# returns the fragment after boxed{ and before the closing } or \n
	L0 = text.split('boxed{}')
	text = ''.join(L0)
	L = text.split('boxed{')
	if l(L) <= 1:
		return None
	s_part = L[1]
	s_res = ''
	par_cnt = 1
	c = None
	for c in s_part:
		if c == '{':
			par_cnt += 1
		elif c == '}':
			par_cnt -= 1
		if par_cnt == 0 or c == '\n':
			break
		s_res += c
	if s_res == '':
		return None
	elif par_cnt >= 2 or (c != '\n' and par_cnt >= 1):
		p('info', "find_answer_in_str: invalid boxed answer: s_part[:l(s_res)+1]='{}'".format(s_part[:l(s_res)+1]))
		return None
	else:
		return s_res

def init_solve():
	global N_used_threads, L_code_to_run, D_used_prompt, D_boxed_run_result, D_code_run_result,  exit_coroutines, D_stats, solve_start_time, LD_gen_request, LD_gen_output, L_prompts_force_code
	N_used_threads = 0
	L_code_to_run = []
	D_used_prompt = {}
	D_boxed_run_result = {}
	D_code_run_result = {}
	exit_coroutines = F
	D_stats = {
		'cnt_run_code': 0,
		'cnt_code_1': 0,
		'cnt_code_2': 0,
		'cnt_code_unfinished_1': 0,
		'cnt_code_unfinished_2': 0,
	}
	solve_start_time = time.time()
	LD_gen_request = []
	LD_gen_output = []

def fix_indent(code):
	# counts the number of spaces mod 4, and if it's 3 or 1, then rounds it to 0 mod 4.
	lines = code.split('\n')
	lines_new = []
	for line in lines:
		cnt_spaces = 0
		if l(line) >= 1:
			for i in range(l(line)):
				if line[i] == ' ':
					cnt_spaces += 1
				else:
					if cnt_spaces%4 == 1:
						cnt_spaces -= 1
					elif cnt_spaces%4 == 3:
						cnt_spaces += 1
					break
			line_new = ' '*cnt_spaces + line[i:]
			lines_new.append(line_new)
		else:
			lines_new.append(line)
	# adds additional newline
	return '\n'.join(lines_new) + '\n'

def fix_return(code):
	# removes lines beginning with return
	lines = code.split('\n')
	lines_new = [line for line in lines if re.fullmatch('^return.*', line) is None]
	return '\n'.join(lines_new)

def fix_plot(code):
	# removes lines containing plotting
	lines = code.split('\n')
	lines_new = [line for line in lines if not any(s in line for s in ['.show', '.plot', '.subplot', '.scatter'])]
	return '\n'.join(lines_new)

def fix_input(code):
	# removes lines containing input()
	lines = code.split('\n')
	lines_new = [line for line in lines if not re.match('^.*[^a-zA-Z_.]input\\(', line)]
	return '\n'.join(lines_new)

def prepare_code(code):
	code = fix_indent(code)
	code = fix_return(code)
	code = fix_plot(code)
	code = fix_input(code)

	code = '\nopen = None\n' + code
	code = code.replace('\n', '\n    ')
	code = "try:\n    from functools import *\n    from sympy import *\n    import numpy as np\n{}\nexcept Exception as ex:\n    print('Exception:\\n{{}}'.format(ex))\n    print('FAIL')\n".format(code)	
	return code

async def run_code2(prefix, coroutine_id, code, timeout):
	global d_result

	try:
		process = None
		result = ''
		code = prepare_code(code)

		filename = 'tmp/code_{}{}.py'.format(prefix, coroutine_id)
		with open(filename, 'w', encoding='utf-8') as fout:
			fout.write(code)
			fout.flush()
			os.fsync(fout)

		cmd = 'ulimit -Sv 9000000; nice -n 19 ' + sys.executable + ' ' + filename
		process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
		try:
			stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
			stdout = stdout.decode()
			if any(stdout.startswith(s) for s in ['Exception', '  File']):
				result = ('', 'exception3: {}'.format(stdout))
			else:
				result = (stdout, '')
		except asyncio.TimeoutError:
			result = ('', 'timeout')
	except Exception as ex:
		result = ('', 'exception4: {}'.format(ex))
		p('info', code)
		p('info', 'exception="{}"'.format(result[1]))
		raise ex

	if process is not None:
		try:
			if process.returncode is None:
				parent = psutil.Process(process.pid)
				for child in parent.children(recursive=True):
					child.terminate()
				parent.terminate()
		except:
			None

		await asyncio.sleep(.05)

		try:
			await process.wait()
		except Exception as ex:
			p('exception', 'exception1:', ex, type(ex))
			if pass_exceptions:
				raise ex

	return result

def is_int_answer_valid(answer):
	return answer >= -1e10 and answer <= 1e10

def is_float_answer_valid(answer):
	return answer >= -1e10 and answer <= 1e10

def intfloat(s_answer):
	try:
		return int(s_answer)
	except:
		None

	try:
		return float(s_answer)
	except:
		None

async def evaluate_answer(coroutine_id, s_answer):
	s_answer = s_answer.strip()

	# when s_answer contains \n, do not run sympy.simplify, parse_latex, sympy.pycode
	try:
		for s in reversed(s_answer.split()):
			if '{' in s and '}' in s:
				if use_parse_latex:
					p('info', "parse latex: input='{}'".format(s_answer))
					sympy_answer0 = parse_latex(s_answer)
					p('info', "parse latex: parsed='{}'".format(sympy_answer0))
					sympy_answer = sympy.simplify(sympy_answer0)
					p('info', "parse latex: simplified='{}'".format(sympy_answer))
				else:
					sympy_answer = sympy.simplify(s_answer)
				if sympy_answer.is_integer:
					return int(sympy_answer)
				else:
					return float(sympy_answer)
				break
	except Exception as ex:
		p('info', 'simplify("{}"): exception7 in evaluate_answer: {} type={}'.format(s_answer, ex, type(ex)))

	res = intfloat(s_answer)
	if res is not None:
		return res

	if '\n' not in s_answer:
		try:
			s_answer = sympy.pycode(s_answer)
		except:
			None

	res = intfloat(s_answer)
	if res is not None:
		return res

	if '\n' not in s_answer:
		try:
			sympy_answer = sympy.simplify(s_answer)
			if sympy_answer.is_integer:
				return int(sympy_answer)
			else:
				return float(sympy_answer)
		except:
			None

	res = intfloat(s_answer)
	if res is not None:
		return res

	evaluated_answer = None
	if '\n' not in s_answer:
		code = 'from functools import *\nfrom numpy import *\nimport numpy as np\nimport math\nfrom sympy import *\nI=1j\nresult1 = simplify(' + s_answer + ')\nprint(result1)\n'
		s_code_output, _ = await run_code2('b', coroutine_id, code, timeout=code_timeout2)

		if not (s_code_output is None or s_code_output == ''):
			try:
				evaluated_answer = eval(s_code_output)
				gc.collect()
			except:
				None

	if evaluated_answer is None or type(evaluated_answer) not in [int, float, np.float16, np.float32, np.float64, np.float128]:
		code = 'from functools import *\nfrom numpy import *\nimport numpy as np\nimport math\nfrom sympy import *\nI=1j\nresult1 = ' + s_answer + '\nprint(result1)\n'
		s_code_output, _ = await run_code2('c', coroutine_id, code, timeout=code_timeout2)
		if s_code_output is None or s_code_output == '':
			return None
		
		try:
			evaluated_answer = eval(s_code_output)
			gc.collect()
		except:
			None

	return evaluated_answer

async def simplify_and_classify_answer0(coroutine_id, answer, repeat=T):
	# int, float, evaluated symbolic, symbolic
	if type(answer) == str and 'boxed{' in answer:
		answer_boxed = find_answer_in_str(answer)
		if answer_boxed is not None:
			answer = answer_boxed
	while type(answer) in [list, tuple, np.ndarray]:
		if l(answer) == 0:
			return '', 'none'
		else:
			answer = answer[0]
	answer_orig = answer
	if answer is None or answer == '' or (type(answer) == str and answer.strip() == 'None') or (type(answer) == float and (np.isnan(answer) or np.isinf(answer))):
		return '', 'none'
	try:
		if type(answer) == complex:
			answer = answer.real
		if type(answer) in [int, float]:
			evaluated_answer = answer
		elif type(answer) == str and l(answer) >= 150:
			return answer, 'symbolic'
		else:
			evaluated_answer = await evaluate_answer(coroutine_id, answer)
			while type(evaluated_answer) in [list, tuple, np.ndarray]:
				if l(evaluated_answer) == 0:
					return '', 'none'
				else:
					evaluated_answer = evaluated_answer[0]

		if evaluated_answer is None:
			return answer, 'symbolic'
		elif type(evaluated_answer) == float and (np.isnan(evaluated_answer) or np.isinf(evaluated_answer)):
			return '', 'none'
		elif type(evaluated_answer) not in [int, float]:
			try:			
				if type(evaluated_answer) == complex:
					evaluated_answer = evaluated_answer.real
				if type(evaluated_answer) not in [dict, set]:
					evaluated_answer = float(evaluated_answer)
			except Exception as ex:
				p('exception', 'evalutated_answer = {}'.format(evaluated_answer))
				p('exception', 'exception3: {} {}'.format(ex, type(ex)))

		if type(evaluated_answer) == int:
			type_answer = 'int'
		elif type(evaluated_answer) == float:
			if abs(evaluated_answer - round(evaluated_answer)) < 1e-2:
				evaluated_answer = round(evaluated_answer)
				type_answer = 'int'
				assert(type(evaluated_answer) == int)
			else:
				type_answer = 'float'
		else:
			type_answer = 'evaluated symbolic'
	except Exception as ex:
		p('exception', 'exception4 during evaluating "{}" (type = {}):'.format(answer, type(answer)))
		p('exception', 'exception4: "{}" {}'.format(ex, type(ex)))
		if repeat:
			try:
				evaluated_answer = await evaluate_answer(coroutine_id, answer)
				if evaluated_answer is not None:
					return await simplify_and_classify_answer0(coroutine_id, evaluated_answer, repeat=F)
			except Exception as ex2:
				p('exception', 'exception5 during evaluating "{}" (type = {}):'.format(answer, type(answer)))
				p('exception', 'exception5: "{}" {}'.format(ex2, type(ex2)))

		type_answer = 'symbolic'
		evaluated_answer = answer

	if type_answer == 'int':
		if not is_int_answer_valid(evaluated_answer):
			type_answer = 'symbolic'
			evaluated_answer = answer_orig

	if type_answer == 'float':
		if not is_float_answer_valid(evaluated_answer):
			type_answer = 'symbolic'
			evaluated_answer = answer_orig

	return evaluated_answer, type_answer

async def simplify_and_classify_answer(coroutine_id, answer):
	evaluated_answer, type_answer = await simplify_and_classify_answer0(coroutine_id, answer)

	if type_answer == 'symbolic' and type(answer) == str:
		# returns the last word that evaluates to an integer
		while l(answer) >= 1 and answer[-1] in [' ', ',', '.']:
			answer = answer[:-1]
		if l(answer) >= 1 and answer[-1] != '\\' and not re.search('mod|deprec|warning|no solution|not found|not work|not possible|time |sqrt|pi|cdot|times', answer, re.I):
			L_answer = answer.split()
			if l(L_answer) <= 1 or L_answer[-2] != '+':
				for s_answer in L_answer[::-1][:8]:
					evaluated_answer = await evaluate_answer(coroutine_id, s_answer)
					evaluated_answer_part, type_answer_part = await simplify_and_classify_answer0(coroutine_id, evaluated_answer, repeat=F)
					if type_answer_part == 'int':
						p('info', 'answer="{}" s_answer="{}" evaluated_answer_part={}'.format(answer, s_answer, evaluated_answer_part))
						return evaluated_answer_part, type_answer_part

	return evaluated_answer, type_answer

async def interpret_code_answer(coroutine_id, D_answer):
	answer, answer_type = await simplify_and_classify_answer(coroutine_id, D_answer['s_code_output'])
	D_answer['answer'] = answer
	D_answer['type'] = answer_type

async def run_code(coroutine_id, code, timeout):
	D_answer = {}
	s_code_output, s_exception = await run_code2('a', coroutine_id, code, timeout=timeout)
	D_answer['s_code_output'] = s_code_output
	D_answer['s_exception'] = s_exception

	await interpret_code_answer(coroutine_id, D_answer)
	return D_answer

def inc(d, key):
	if key not in d:
		d[key] = 0
	d[key] += 1

def get(d, key, default=0):
	if key in d:
		return d[key]
	else:
		return default

def r3(x):
	if abs(x-round(x)) < 1e-6:
		return '{}'.format(round(x+1e-6))
	elif abs(10*x-round(10*x)) < 1e-6:
		return '{:.1f}'.format(x)
	elif abs(100*x-round(100*x)) < 1e-6:
		return '{:.2f}'.format(x)
	else:
		return '{:.3f}'.format(x)

D_prompt_group_boxed = {
}

for i in L_prompts1 + [10]:
	D_prompt_group_boxed[i] = 1

for i in [12,13,14,100]:
	D_prompt_group_boxed[i] = 2

for i in L_prompts4 + L_prompts5:
	D_prompt_group_boxed[i] = 3

D_prompt_group_code = {}
for i,v in D_prompt_group_boxed.items():
	D_prompt_group_code[i] = 10+v

assert(all(i_prompt in D_prompt_format for i_prompt in D_prompt_group_boxed))
assert(all(i_prompt in D_prompt_group_boxed for i_prompt in D_prompt_format))

assert(all(i_prompt in D_prompt_format for i_prompt in D_prompt_group_code))
assert(all(i_prompt in D_prompt_group_code for i_prompt in D_prompt_format))

boxed_and_code_bonus = 0

D_default_conf_choose_answer = {
	('pair_bonus', (1,13)): 3,
	('pair_bonus', (2,13)): 3,
	'pair_bonus_max': 15,

	('pair_bonus2', (1,13)): 35,
	('pair_bonus2', (2,13)): 15,

	'shrink_same_prefix_boxed': .85,
	'shrink_same_prefix_code': .65,
	'shrink_same_prefix_code2': .65,
	'shrink_same_prefix_code3': .65,

	('penalty', 0): .5,		# ≤ -1
	('penalty', 1): .03,	# 0
	('penalty', 2): .08,	# 1
	('penalty', 3): .7,		# 10
	('penalty', 4): .95,	# 50
	('penalty', 5): 1,		# 100 (≥100 have penalty 1)
	('penalty', 6): .8,		# 1000
	('penalty', 7): .8,		# 10000

	'th_len_boxed_le': 300,

	'th_len_code_ge1': 100,
	'th_len_code_ge2': 250,
	'th_len_code_ge3': 350,
	'th_len_code_ge4': 130,
	'shrink_ge1': .05,
	'shrink_ge2': .5,

	('mul_boxed', 10): 12,
	('mul_boxed', 100): 18,

	('mul_boxed', 12): 25,
	('mul_boxed', 13): 18,
	('mul_boxed', 14): 18,

	('mul_code', 900): 8,
	('mul_code', 901): 8,
	('mul_code', 902): 8,
	('mul_code', 903): 8,
	('mul_code', 910): 8,
	('mul_code', 911): 9,
}

for i in L_prompts1:
	D_default_conf_choose_answer[('mul_boxed', i)] = 11
	D_default_conf_choose_answer[('mul_code', i)] = 5

for i in L_prompts2:
	D_default_conf_choose_answer[('mul_code', i)] = 7

for i in L_prompts3:
	D_default_conf_choose_answer[('mul_code', i)] = 10

for i in L_prompts4:
	D_default_conf_choose_answer[('mul_boxed', i)] = 5

for i in L_prompts5:
	D_default_conf_choose_answer[('mul_boxed', i)] = 5

assert(all(x[1] in D_prompt_format for x in D_default_conf_choose_answer if type(x) == tuple and x[0].startswith('mul_')))
assert(all(('mul_boxed',i_prompt) in D_default_conf_choose_answer for i_prompt in D_prompt_format))
assert(all(('mul_code',i_prompt) in D_default_conf_choose_answer for i_prompt in D_prompt_format))

def calc_code_length(code):
	code_wo_comments = '\n'.join([line for line in code.split('\n') if not ('#' in line and line.strip()[0]=='#')])
	return l(code_wo_comments)

def choose_answer(question, LD_answers, correct_answer=None, D_update_conf_choose_answer=None):
	D_conf = D_default_conf_choose_answer.copy()
	if D_update_conf_choose_answer is not None:
		D_conf.update(D_update_conf_choose_answer)

	th_len_code_ge1 = D_conf['th_len_code_ge1']
	th_len_code_ge2 = D_conf['th_len_code_ge2']
	th_len_code_ge3 = D_conf['th_len_code_ge3']
	th_len_code_ge4 = D_conf['th_len_code_ge4']
	shrink_ge1 = D_conf['shrink_ge1']
	shrink_ge2 = D_conf['shrink_ge2']

	D_mul_prompt_boxed = {}
	D_mul_prompt_code = {}

	L_penalty_range = 	[-np.inf,	0,	1,	10,	50,	100]
	L_penalty = np.zeros(l(L_penalty_range))
	penalty_1000 = D_conf['penalty', l(L_penalty)]
	penalty_10000 = D_conf['penalty', l(L_penalty)+1]

	for key, v in D_conf.items():
		if type(key) == tuple:
			s,i = key
			if s == 'mul_boxed':
				D_mul_prompt_boxed[i] = v
			elif s == 'mul_code':
				D_mul_prompt_code[i] = v
			elif s == 'penalty' and i < l(L_penalty):
				L_penalty[i] = v

	def get_penalty(answer):
		penalty = 1.0
		if answer <= -1:
			penalty = L_penalty[0]
		elif answer == 0:
			penalty = L_penalty[1]
		elif answer >= 10000:
			penalty = penalty_10000
		elif answer >= 1000:
			penalty = penalty_1000
		elif answer >= 1:
			for i in range(1, l(L_penalty_range)-1):
				if answer > L_penalty_range[i] and answer <= L_penalty_range[i+1]:
					lo = L_penalty_range[i]
					hi = L_penalty_range[i+1]
					penalty = L_penalty[i] + (L_penalty[i+1] - L_penalty[i])  * (answer - lo)/(hi-lo)
					break
		return penalty

	L_prompts_to_shrink_boxed = []
	L_prompts_to_shrink_code = list(range(900,930))
	D_cnt_parent = {}
	D_cnt_parent2 = {}
	D_cnt_parent3 = {}
	for i in L_prompts_to_shrink_boxed + L_prompts_to_shrink_code:
		if i in D_prompt_format:
			D_cnt_parent[i] = {}

	for D in LD_answers:
		if (D['i_prompt'] in L_prompts_to_shrink_boxed and 'boxed_answer' in D and D['boxed_answer']['type'] == 'int') or (D['i_prompt'] in L_prompts_to_shrink_code and 'code_answer' in D and D['code_answer']['type'] == 'int'):
			D1 = D_cnt_parent[D['i_prompt']]
			id_parent = id(D['D_parent'])
			if id_parent not in D1:
				D1[id_parent] = 0
			D1[id_parent] += 1

		if 'code_answer' in D and D['code_answer']['type'] == 'int':
			if D['i_prompt'] in L_prompts4:
				inc(D_cnt_parent2, id(D['D_parent']))
				inc(D_cnt_parent3, id(D['D_parent']))
			if D['i_prompt'] in L_prompts5:
				inc(D_cnt_parent2, id(D['D_parent']))
				inc(D_cnt_parent3, id(D['D_parent']['D_parent']))

	L_answer = []
	D_cnt_boxed_answer = {}
	D_cnt_800_boxed_answer = {}
	D_cnt_code_answer = {}
	for D in LD_answers:
		D_answer = {}
		for key in ['i_prompt']:
			D_answer[key] = D[key]
		D_answer['id_parent'] = id(D['D_parent'])
		if 'code' in D:
			D_answer['code'] = D['code']
			D_answer['code_length'] = calc_code_length(D['code'])

		if 'boxed_answer' in D and D['boxed_answer']['type'] == 'int':
			pos_begin_boxed = l(D['text'].split('boxed{')[0])
			if l(D['boxed_answer_text']) <= D_conf['th_len_boxed_le']:
				answer = D['boxed_answer']['answer']
				if answer >= 0:
					answer %= 1000
				D_answer['boxed_answer'] = answer
				if D['i_prompt'] >= 800 and D['i_prompt'] < 810:
					inc(D_cnt_800_boxed_answer, answer)
				else:
					inc(D_cnt_boxed_answer, answer)
				if D['i_prompt'] in L_prompts_to_shrink_boxed and id(D['D_parent']) in D_cnt_parent[D['i_prompt']]:
					D_answer['cnt_same_parent'] = D_cnt_parent[D['i_prompt']][id(D['D_parent'])]

		if 'code_answer' in D and D['code_answer']['type'] == 'int':
			#if D_answer['code_length'] >= D_conf['th_len_code_ge']:
			answer = D['code_answer']['answer']
			if answer >= 0:
				answer %= 1000
			D_answer['code_answer'] = answer
			inc(D_cnt_code_answer, answer)
			if D['i_prompt'] in L_prompts_to_shrink_code and id(D['D_parent']) in D_cnt_parent[D['i_prompt']]:
				D_answer['cnt_same_parent'] = D_cnt_parent[D['i_prompt']][id(D['D_parent'])]
			if D['i_prompt'] in L_prompts4:
				D_answer['cnt_same_parent2'] = D_cnt_parent2[id(D['D_parent'])]
				D_answer['cnt_same_parent3'] = D_cnt_parent3[id(D['D_parent'])]
			if D['i_prompt'] in L_prompts5:
				D_answer['cnt_same_parent2'] = D_cnt_parent2[id(D['D_parent'])]
				D_answer['cnt_same_parent3'] = D_cnt_parent3[id(D['D_parent']['D_parent'])]

		if any([s in D_answer for s in ['code_answer', 'boxed_answer']]):
			L_answer.append(D_answer)

	DD_prompt_to_cnt_boxed = {}
	DD_prompt_to_cnt_code = {}
	DD_boxed_prompts = {}
	DD_code_prompts = {}
	DD_boxed_prompts2 = {}
	DD_code_prompts2 = {}
	DL_boxed_prompts = {}
	DL_code_prompts = {}
	for D_answer in L_answer:
		i_prompt = D_answer['i_prompt']
		if 'boxed_answer' in D_answer:
			answer = D_answer['boxed_answer']
			if i_prompt not in DD_prompt_to_cnt_boxed:
				DD_prompt_to_cnt_boxed[i_prompt] = {}
			if answer not in DD_prompt_to_cnt_boxed[i_prompt]:
				DD_prompt_to_cnt_boxed[i_prompt][answer] = 0
			DD_prompt_to_cnt_boxed[i_prompt][answer] += 1
			if answer not in DD_boxed_prompts:
				DD_boxed_prompts[answer] = {}
				DD_boxed_prompts2[answer] = {}
				DL_boxed_prompts[answer] = []
			DD_boxed_prompts[answer][i_prompt] = 1
			DD_boxed_prompts2[answer][D_prompt_group_boxed[i_prompt]] = 1
			if 'cnt_same_parent' in D_answer:
				cnt = D_answer['cnt_same_parent']
				a = D_conf['shrink_same_prefix_boxed']
				shrink = (1-a)/cnt + a
			else:
				shrink = 1
			DL_boxed_prompts[answer].append((i_prompt,D_mul_prompt_boxed[i_prompt]*shrink))
		if 'code_answer' in D_answer:
			answer = D_answer['code_answer']
			if i_prompt not in DD_prompt_to_cnt_code:
				DD_prompt_to_cnt_code[i_prompt] = {}
			if answer not in DD_prompt_to_cnt_code[i_prompt]:
				DD_prompt_to_cnt_code[i_prompt][answer] = 0
			DD_prompt_to_cnt_code[i_prompt][answer] += 1
			if answer not in DD_code_prompts:
				DD_code_prompts[answer] = {}
				DD_code_prompts2[answer] = {}
				DL_code_prompts[answer] = []
			if D_answer['code_length'] >= th_len_code_ge4:
				DD_code_prompts[answer][i_prompt] = 1
				DD_code_prompts2[answer][D_prompt_group_code[i_prompt]] = 1
			shrink = 1
			if 'cnt_same_parent' in D_answer:
				cnt = D_answer['cnt_same_parent']
				a = D_conf['shrink_same_prefix_code']
				shrink *= (1-a)/cnt + a
			if 'cnt_same_parent2' in D_answer:
				cnt = D_answer['cnt_same_parent2']
				a = D_conf['shrink_same_prefix_code2']
				shrink *= (1-a)/cnt + a
			if 'cnt_same_parent3' in D_answer:
				cnt = D_answer['cnt_same_parent3']
				a = D_conf['shrink_same_prefix_code3']
				shrink *= (1-a)/cnt + a
			if D_answer['code_length'] < th_len_code_ge1:
				shrink *= shrink_ge1
			elif D_answer['code_length'] < th_len_code_ge2:
				l1 = D_answer['code_length'] - th_len_code_ge1
				l2 = th_len_code_ge2 - D_answer['code_length']
				l3 = th_len_code_ge2 - th_len_code_ge1
				shrink *= l2/l3 * shrink_ge1 + l1/l3 * shrink_ge2
			elif D_answer['code_length'] < th_len_code_ge3:
				l1 = D_answer['code_length'] - th_len_code_ge2
				l2 = th_len_code_ge3 - D_answer['code_length']
				l3 = th_len_code_ge3 - th_len_code_ge2
				shrink *= l2/l3 * shrink_ge2 + l1/l3
			DL_code_prompts[answer].append((i_prompt,D_mul_prompt_code[i_prompt]*shrink))

	p('answer_info', '='*50)
	p('answer_info', 'boxed answers:')
	for i_prompt,D in DD_prompt_to_cnt_boxed.items():
		if l(D) >= 1:
			s = '{}'.format('?' if correct_answer is None else 0 if correct_answer not in D else r3(D[correct_answer]))
			p('answer_info', ' prompt={:2d} {:3d} {}/{} ['.format(i_prompt,l(D),s,r3(sum(list(D.values())))),end='')
			L = ['{}{}:{}{}'.format(color1,x[0],r3(x[1]),code_reset_color) if x[0] == correct_answer and use_colors else '{}:{}'.format(x[0],r3(x[1])) for x in sorted(D.items(),key=lambda x:-x[1])]
			for s in L:
				p('answer_info', s, end=', ')
			p('answer_info', ']')
	p('answer_info', 'code answers:')
	for i_prompt,D in DD_prompt_to_cnt_code.items():
		if l(D) >= 1:
			s = '{}'.format('?' if correct_answer is None else 0 if correct_answer not in D else r3(D[correct_answer]))
			p('answer_info', ' prompt={:2d} {:3d} {}/{} ['.format(i_prompt,l(D),s,r3(sum(list(D.values())))),end='')
			L = ['{}{}:{}{}'.format(color1,x[0],r3(x[1]),code_reset_color) if x[0] == correct_answer and use_colors else '{}:{}'.format(x[0],r3(x[1])) for x in sorted(D.items(),key=lambda x:-x[1])]
			for s in L:
				p('answer_info', s, end=', ')
			p('answer_info', ']')

	D_score_penalty = {}
	for D_answer in L_answer:
		for s in ['code_answer', 'boxed_answer']:
			if s in D_answer:
				answer = D_answer[s]
				penalty = get_penalty(answer)
				D_score_penalty[answer] = penalty

	D_score_boxed = {}
	D_score_code = {}
	D_score_bonus = {}
	D_score_bonus2 = {}
	D_score = {}
	for answer in D_score_penalty:
		score_boxed = 0
		for i_prompt,score in get(DL_boxed_prompts, answer, []):
			score_boxed += score

		score_code = 0
		for i_prompt,score in get(DL_code_prompts, answer, []):
			score_code += score

		bonus = 0
		for i1_prompt in get(DD_boxed_prompts, answer, []):
			i1_prompt_group = D_prompt_group_boxed[i1_prompt]
			for i2_prompt in get(DD_boxed_prompts, answer, []):
				if i1_prompt != i2_prompt:
					i2_prompt_group = D_prompt_group_boxed[i2_prompt]
					key = (i1_prompt_group,i2_prompt_group)
					if ('pair_bonus', key) in D_conf:
						bonus += D_conf[('pair_bonus', key)]
		boxed_and_code = F
		for i1_prompt in get(DD_boxed_prompts, answer, []):
			i1_prompt_group = D_prompt_group_boxed[i1_prompt]
			for i2_prompt in get(DD_code_prompts, answer, []):
				boxed_and_code = T
				i2_prompt_group = D_prompt_group_code[i2_prompt]
				key = (i1_prompt_group,i2_prompt_group)
				if ('pair_bonus', key) in D_conf:
					bonus += D_conf[('pair_bonus', key)]
				key = (i2_prompt_group,i1_prompt_group)
				if ('pair_bonus', key) in D_conf:
					bonus += D_conf[('pair_bonus', key)]
		for i1_prompt in get(DD_code_prompts, answer, []):
			i1_prompt_group = D_prompt_group_code[i1_prompt]
			for i2_prompt in get(DD_code_prompts, answer, []):
				if i1_prompt != i2_prompt:
					i2_prompt_group = D_prompt_group_code[i2_prompt]
					key = (i1_prompt_group,i2_prompt_group)
					if ('pair_bonus', key) in D_conf:
						bonus += D_conf[('pair_bonus', key)]
		if boxed_and_code:
			bonus += boxed_and_code_bonus

		bonus2 = 0
		for i1_prompt_group in get(DD_boxed_prompts2, answer, []):
			for i2_prompt_group in get(DD_boxed_prompts2, answer, []):
				if i1_prompt_group != i2_prompt_group:
					key = (i1_prompt_group,i2_prompt_group)
					if ('pair_bonus2', key) in D_conf:
						bonus2 += D_conf[('pair_bonus2', key)]
		boxed_and_code = F
		for i1_prompt_group in get(DD_boxed_prompts2, answer, []):
			for i2_prompt_group in get(DD_code_prompts2, answer, []):
				boxed_and_code = T
				key = (i1_prompt_group,i2_prompt_group)
				if ('pair_bonus2', key) in D_conf:
					bonus2 += D_conf[('pair_bonus2', key)]
				key = (i2_prompt_group,i1_prompt_group)
				if ('pair_bonus2', key) in D_conf:
					bonus2 += D_conf[('pair_bonus2', key)]
		for i1_prompt_group in get(DD_code_prompts2, answer, []):
			for i2_prompt_group in get(DD_code_prompts2, answer, []):
				if i1_prompt_group != i2_prompt_group:
					key = (i1_prompt_group,i2_prompt_group)
					if ('pair_bonus2', key) in D_conf:
						bonus2 += D_conf[('pair_bonus2', key)]

		D_score_boxed[answer] = score_boxed
		D_score_code[answer] = score_code
		D_score_bonus[answer] = min(D_conf['pair_bonus_max'], D_score_penalty[answer] * bonus)
		D_score_bonus2[answer] = D_score_penalty[answer] * bonus2

		D_score[answer] = D_score_penalty[answer] * (D_score_boxed[answer] + D_score_code[answer] + D_score_bonus[answer] + D_score_bonus2[answer])

	L = list(D_score.items())
	np.random.shuffle(L)
	L_answers_scored_sorted = list(sorted(L, key=lambda x:-x[1]))

	i_correct_answer = None
	for i,(answer,score) in enumerate(L_answers_scored_sorted):
		boxed_answers = get(D_score_boxed, answer)
		code_answers = get(D_score_code, answer)
		cnt_boxed_answers = get(D_cnt_boxed_answer, answer)
		cnt_800_boxed_answers = get(D_cnt_800_boxed_answer, answer)
		cnt_code_answers = get(D_cnt_code_answer, answer)
		bonus = get(D_score_bonus, answer)
		bonus2 = get(D_score_bonus2, answer)
		penalty = get(D_score_penalty, answer)
		s = '{}{:5d}{}'.format(color1,answer,code_reset_color) if answer%1000 == correct_answer and use_colors else '{:5d}'.format(answer)
		p('answer', 'ans={} sco={:5.2f} cod={:5.2f}({:2d}) box={:5.2f}({:2d},{:2d}) bon={:5.2f} bon2={:5.2f} pen={:5.2f}'.format(s, score, code_answers, cnt_code_answers, boxed_answers, cnt_boxed_answers, cnt_800_boxed_answers, bonus, bonus2, penalty))
		if answer%1000 == correct_answer:
			i_correct_answer = i
		if i_correct_answer is not None and i >= i_correct_answer + 2 and i >= 5:
			break

	final_answer = default_answer
	for answer,score in L_answers_scored_sorted:
		final_answer = answer
		break

	final_answer %= 1000

	p('answer_info', 'choose_answer end', step_time())

	return final_answer, L_answers_scored_sorted

def end_run_code(max_time):
	if time.time() - solve_start_time >= max_time:
		return T
	if exit_coroutines and l(L_code_to_run) == 0:
		return T
	return F

def is_math_expression(s):
	return re.match('.*[0-9].*', s)

async def coroutine_run_code(coroutine_id, max_time):
	global N_used_threads

	try:
		while T:
			if l(L_code_to_run) >= 1 and N_used_threads + 1 <= N_max_threads:
				code_type, D_gen_output = L_code_to_run.pop()
				if code_type == 'code':
					D_answer = None
					h = hash(D_gen_output['code'])
					if h in D_code_run_result:
						if D_code_run_result[h] is None:
							# result from another coroutine not ready yet
							L_code_to_run.append((code_type, D_gen_output))
							await asyncio.sleep(0)
						else:
							D_answer = D_code_run_result[h]
					else:
						D_code_run_result[h] = None
						N_used_threads += 1
						D_stats['cnt_run_code'] += 1
						D_answer = await run_code(coroutine_id, D_gen_output['code'], timeout=code_timeout)				
						N_used_threads -= 1
						D_code_run_result[h] = D_answer

					if D_answer is not None:
						D_gen_output['code_answer'] = D_answer
						D_gen_output['interpret_code'] = F
						D_gen_output['changed'] = T
				elif code_type == 'boxed':
					answer_type = None
					h = hash(D_gen_output['boxed_answer_text'])
					if h in D_boxed_run_result:
						if D_boxed_run_result[h] is None:
							# result from another coroutine not ready yet
							L_code_to_run.append((code_type, D_gen_output))
							await asyncio.sleep(0)
						else:
							answer, answer_type = D_boxed_run_result[h]
					else:
						D_boxed_run_result[h] = None
						N_used_threads += 1
						answer, answer_type = await simplify_and_classify_answer(coroutine_id, D_gen_output['boxed_answer_text'])
						N_used_threads -= 1
						D_boxed_run_result[h] = (answer, answer_type)

					if answer_type is not None:
						D_gen_output['boxed_answer'] = {'answer':answer, 'type':answer_type}
						D_gen_output['interpret_boxed'] = F
						D_gen_output['changed'] = T
				else:
					raise exc

			if end_run_code(max_time):
				break

			await asyncio.sleep(.02)

	except Exception as ex:
		p('exception', 'exception6 in coroutine_run_code:', ex, type(ex))
		raise ex

def end_generate(max_time):
	if time.time() - solve_start_time >= max_time:
		return T
	all_requests_finished = all(D_gen_request['state'] == 'finished' for D_gen_request in LD_gen_request)
	all_interpret_finished = all(not (D_gen_output['interpret_boxed'] or D_gen_output['interpret_code']) for D_gen_output in LD_gen_output)
	if all_requests_finished and all_interpret_finished:
		return T
	return F

def cleanup_llm_engine(max_time):
	p('info', 'cleanup_llm_engine begin')
	assert(end_generate(max_time))
	if all(D_gen_request['state'] == 'finished' for D_gen_request in LD_gen_request) and llm_engine.has_unfinished_requests():
		p('error', 'llm_engine.has_unfinished_requests()')
	if llm_engine.has_unfinished_requests():
		for D_gen_request in LD_gen_request:
			if D_gen_request['state'] != 'finished':
				p('info', 'aborting', D_gen_request['id_gen_request'])
				llm_engine.abort_request(D_gen_request['id_gen_request'])
				D_gen_request['end_time'] = time.time()
				D_gen_request['state'] = 'aborted'

	if llm_engine.has_unfinished_requests():
		p('error', 'after aborting all requests llm_engine has unfinished requests')
		for i,sh in enumerate(llm_engine.scheduler):
			p('error', 'sh{} waiting: {}'.format(i, [g.request_id for g in sh.waiting]))
			p('error', 'sh{} running: {}'.format(i, [g.request_id for g in sh.running]))
			p('error', 'sh{} swapped: {}'.format(i, [g.request_id for g in sh.swapped]))
	p('info', 'cleanup_llm_engine end')

def deepcopy(x):
	if type(x) == dict:
		return {k2:deepcopy(v2) for k2,v2 in x.items()}
	elif type(x) == list:
		return [deepcopy(y) for y in x]
	elif type(x) == tuple:
		return tuple([deepcopy(y) for y in list(x)])
	else:
		return x

def find_code_in_str(input_text, output_text):
	L_part = output_text.split('```')
	code = None
	code_pos = None
	if '```' in input_text:
		if l(L_part) >= 2:
			code = L_part[0]
			if code.startswith('python'):
				code = code[l('python'):]
			code_pos = 0
			D_stats['cnt_code_2'] += 1
		else:
			D_stats['cnt_code_unfinished_2'] += 1
	else:
		if l(L_part) >= 3:
			if (L_part[1].startswith('python') or L_part[1].startswith('\n')):
				code = L_part[1]
				if code.startswith('python'):
					code = code[l('python'):]
				code_pos = l(L_part[0])
				D_stats['cnt_code_1'] += 1
		elif l(L_part) == 2:
			D_stats['cnt_code_unfinished_1'] += 1
	return code, code_pos

def remove_code_and_cut(s):
	s = s.split('```')[0]
	L = s.split('\n')[:-1]
	s_new = '\n'.join(L)
	return s_new

def prepare_prev_output_text(output_text, shift):
	output_text = output_text.split('boxed{')[0]
	if shift >= 1:
		output_text = output_text[:-shift]
	output_text = remove_code_and_cut(output_text)
	return output_text

def interpret_gen_output(output):
	D_gen_request = LD_gen_request[int(output.request_id)]
	D_gen_request['end_time'] = time.time()
	D_gen_request['state'] = 'finished'

	t = D_gen_request['end_time'] - D_gen_request['start_time']
	t0 = D_gen_request['end_time'] - solve_start_time
	L_tok = list(sorted([l(output_obj.token_ids) for output_obj in  output.outputs]))
	p('info', 't={:.1f} t0={:.1f} id_gen_request={} i_prompt={} generated L_tok={}'.format(t, t0, D_gen_request['id_gen_request'], D_gen_request['i_prompt'], L_tok))

	for j,output_obj in enumerate(output.outputs):
		input_text = D_gen_request['input_text']
		output_text = output_obj.text
		output_tokens = output_obj.token_ids
		text = input_text + output_text
		D_gen_output = {
			'id_gen_output': l(LD_gen_output),
			'idx_output': j,
			'interpret_boxed': F,
			'interpret_code': F,
			'D_parent': D_gen_request['D_parent_gen_output'],
			'i_prompt': D_gen_request['i_prompt'],
			'D_gen_request': D_gen_request,
			'input_text': input_text,
			'output_text': output_text,
			'output_tokens': output_tokens,
			'text': text,
		}
		LD_gen_output.append(D_gen_output)

		prev_i_prompt = D_gen_request['i_prompt']

		if 'boxed{' in text:
			boxed_answer_text = find_answer_in_str(text)
			if boxed_answer_text is not None:
				D_gen_output['boxed_answer_text'] = boxed_answer_text
				if l(boxed_answer_text) >= 1 and l(boxed_answer_text) <= 3 and all(c.isdigit() for c in boxed_answer_text):
					D_gen_output['boxed_answer'] = {'type':'int', 'answer':int(boxed_answer_text)}
				else:
					D_gen_output['interpret_boxed'] = T
					L_code_to_run.append(('boxed',D_gen_output))
			L_i_prompt = D_gen_conf[prev_i_prompt]['next_after_stop'].copy()
		else:
			stop = l(output_tokens) < D_gen_request['tokens']
			if stop:
				L_i_prompt = D_gen_conf[prev_i_prompt]['next_after_stop'].copy()
			else:
				L_i_prompt = D_gen_conf[prev_i_prompt]['next_after_no_stop'].copy()

		for i_prompt in L_i_prompt:
			new_D_gen_request, s_prompt_format = init_D_gen_request(D_gen_request['question'], i_prompt, D_parent_gen_output=D_gen_output)

			chars_shift = D_gen_conf[i_prompt]['chars_shift']

			cnt_args = l(s_prompt_format.split('{}')) - 1
			if cnt_args == 1:
				new_input_text = create_prompt(i_prompt, prepare_prev_output_text(text, chars_shift))
			elif cnt_args == 2:
				new_input_text = create_prompt(i_prompt, prepare_prev_output_text(text, chars_shift), new_D_gen_request['question'])
			elif cnt_args == 3:
				new_input_text = create_prompt(i_prompt, new_D_gen_request['question'], prepare_prev_output_text(output_text, chars_shift), new_D_gen_request['question'])
			else:
				raise exc
			new_D_gen_request['input_text'] = new_input_text

			params = create_params(new_D_gen_request)
			llm_engine.add_request(new_D_gen_request['id_gen_request'], new_input_text, params)

		code, code_pos = find_code_in_str(input_text, output_text)
		if code is not None:
			D_gen_output['code'] = code
			D_gen_output['code_pos'] = code_pos
			D_gen_output['interpret_code'] = T
			L_code_to_run.append(('code', D_gen_output))

async def llm_generate_programs(max_time):
	global exit_coroutines
	p('info', 'llm_generate_programs max_time={:.2f}'.format(max_time))
	prev_l_L_code_to_run = 0
	while not end_generate(max_time):
		outputs = llm_engine.step()
		for output in outputs:
			if output.finished:
				interpret_gen_output(output)

		await asyncio.sleep(0)
		if l(L_code_to_run) != prev_l_L_code_to_run:
			prev_l_L_code_to_run = l(L_code_to_run)

	exit_coroutines = T
	cleanup_llm_engine(max_time)
	await asyncio.sleep(0)

def create_params(D_gen_request):
	return vllm.SamplingParams(temperature=D_gen_request['temp'], skip_special_tokens=T, max_tokens=D_gen_request['tokens'], n=D_gen_request['n'], stop=D_gen_request['L_stop'], include_stop_str_in_output=T)

def init_D_gen_request(question, i_prompt, D_parent_gen_output):
	id_gen_request = str(l(LD_gen_request))

	obj = D_prompt_format[i_prompt]
	if type(obj) == tuple:
		s_prompt_format, L_stop = obj
	else:
		s_prompt_format = obj
		L_stop = []

	D_gen_request = {
		'id_gen_request': id_gen_request,
		'state': 'generate',
		'question': question,
		'D_parent_gen_output': D_parent_gen_output,
		'i_prompt': i_prompt,
		'L_stop': L_stop,
		'start_time': time.time(),
		'changed': T,
	}

	for key in ['tokens', 'n', 'temp']:
		D_gen_request[key] = D_gen_conf[i_prompt][key]

	LD_gen_request.append(D_gen_request)
	return D_gen_request, s_prompt_format

def launch_gen_prompt(question, i_prompt):
	D_gen_request, _ = init_D_gen_request(question, i_prompt, D_parent_gen_output=None)

	input_text = create_prompt(i_prompt, question)
	D_gen_request['input_text'] = input_text

	params = create_params(D_gen_request)
	llm_engine.add_request(D_gen_request['id_gen_request'], input_text, params)

cnt_solve = 0

def solve_inner(question, max_time, save_filename, correct_answer):
	global LD_gen_request, LD_gen_output, cnt_solve
	cnt_solve += 1

	init_solve()

	if save_filename != None and os.path.isfile(save_filename):
		D = np.load(save_filename, allow_pickle=T)[0]
		LD_gen_request = D['LD_gen_request']
		LD_gen_output = D['LD_gen_output']
		assert(D['question'] == question)
	else:
		for i_prompt in np.random.permutation((L_i_prompt_initial*N_runs).copy()):
			launch_gen_prompt(question, i_prompt)

		L_coroutine = [llm_generate_programs(max_time)]
		for i in range(N_max_threads):
			L_coroutine.append(coroutine_run_code(i, max_time))
		looper = asyncio.gather(*L_coroutine)
		asyncio.get_event_loop().run_until_complete(looper)
		gc.collect()

		if save_filename != None:
			D_answers = {
				'question': question,
				'time': time.time() - solve_start_time,
				'LD_gen_request': LD_gen_request,
				'LD_gen_output': LD_gen_output,
			}
			np.save(save_filename, ar([D_answers]))

			s = ''
			for D in LD_gen_output:
				s += D['text']
				s += '\n===end i_prompt={} j={} tok={} l(text)={}\n'.format(D['i_prompt'], D['idx_output'], l(D['output_tokens']), l(D['text']))
			save_filename2 = save_filename + '.txt'
			with open(save_filename2, 'a', encoding='utf-8') as f:
				f.write(s)

	cnt_timeout = l([x['code_answer']['s_exception'] for x in LD_gen_output if 'code_answer' in x and x['code_answer']['type'] in ['none'] and x['code_answer']['s_exception']=='timeout'])
	p('info', '#code1={} #code2={} #un_code1={} #un_code2={} #run_code={} #timeout={} {}'.format(D_stats['cnt_code_1'], D_stats['cnt_code_2'], D_stats['cnt_code_unfinished_1'], D_stats['cnt_code_unfinished_2'], D_stats['cnt_run_code'], cnt_timeout, step_time() ))

	for D in LD_gen_output:
		D['question'] = question

	# remove duplicate code
	D_hash_code_answer = {}
	LD_gen_output2 = []
	for D in LD_gen_output:
		if 'code_answer' in D:
			h = hash(D['code'])
			if h in D_hash_code_answer:
				continue
			D_hash_code_answer[h] = T
		LD_gen_output2.append(D)

	answer, L_answers_scored_sorted = choose_answer(question, LD_gen_output2, correct_answer)

	return answer, L_answers_scored_sorted

def init_control():
	global D_control
	D_control = {'cnt_tasks':0, 'L_change':[]}

def choose_max_time():
	timeout = 1500 if conf_pc else 1000
	return max(1, min(timeout, time_available_final - (time.time() - start_time_llm_loaded)))

def update_control():
	D_control['cnt_tasks'] += 1

	prior_cnt_tasks = 10

	regularized_avg_change = sum(D_control['L_change']) / (D_control['cnt_tasks'] - 1 + prior_cnt_tasks)

	time_used = time.time() - start_time_llm_loaded
	time_remaining = time_available - time_used

	N_tasks_remaining = N_tasks - D_control['cnt_tasks']

	regularized_time_used_per_task = (time_used + prior_cnt_tasks*prior_time_per_task)/(D_control['cnt_tasks'] + prior_cnt_tasks)

	surplus_time_per_task = (time_remaining - regularized_time_used_per_task*N_tasks_remaining) / (N_tasks_remaining + 1)

	term2 = (prior_time_per_task - regularized_time_used_per_task)/10
	term3 = surplus_time_per_task/15
	change = int(regularized_avg_change + term2 + term3)
	change = max(-3, min(4, change))
	D_control['L_change'].append(change)

	if use_control:
		for i in L_prompts1:
			if i == 3:
				D_gen_conf[i]['tokens'] = default_tokens_phase1_prompt3 + change*128
			else:
				D_gen_conf[i]['tokens'] = default_tokens_phase1 + change*128

		p('info', 'N_tasks_remaining={} time_used={:.1f} time_remaining={:.1f} reg_time_used_per_task={:.1f} surplus_time_per_task={:.1f} reg_avg_change={:.2f} term2={:.2f} term3={:.2f} change={}'.format(N_tasks_remaining, time_used, time_remaining, regularized_time_used_per_task, surplus_time_per_task, regularized_avg_change, term2, term3, change))
	else:
		p('info', 'N_tasks_remaining={} time_used={:.1f} time_remaining={:.1f} reg_time_used_per_task={:.1f}'.format(N_tasks_remaining, time_used, time_remaining, regularized_time_used_per_task))

llm_engine = None

def solve(question, D_task=None):
	#if question in ['Solve $4+x=4$ for $x$.', 'What is $0\\times10$?', 'What is $1-1$?']:
	#if question in ['Solve $4+x=4$ for $x$.', 'What is $1-1$?']:
	if question in ['Solve $4+x=4$ for $x$.']:
		return 0

	answer = None
	try:
		if save_answers:
			save_filename = '{}/{}{}_{}.npy'.format(dir_save_answers, prefix_filename_save_answers, D_task['i_dataset'], D_task['i_task'])
		else:
			save_filename = None

		correct_answer = D_task['correct_answer'] if D_task else None

		max_time = choose_max_time()

		answer, L_answers_scored_sorted = solve_inner(question, max_time, save_filename, correct_answer)

	except Exception as ex:
		p('error', 'exception8 in solve: {} type={}'.format(ex, type(ex)))
		extype, _, _ = sys.exc_info()
		if extype != SystemExit:
			traceback.print_exc()
		try:
			cleanup_llm_engine(0)
		except:
			None
		if pass_exceptions:
			raise ex

	update_control()

	if answer is None:
		answer = default_answer
	answer %= 1000

	if D_task:
		return answer, L_answers_scored_sorted
	else:
		return answer

def init_solve_tasks():
	init_control()

def init_tasks():
	from dq_data4h import LD_tasks

	p('info', 'l(LD_tasks) =', l(LD_tasks))
	assert(l(LD_tasks) == N_tasks)
	return LD_tasks

def solve_tasks():
	global N_tasks

	LD_tasks = init_tasks()
	N_tasks = l(LD_tasks)

	cnt_correct = 0
	sum_score = 0
	for idx,D_task in enumerate(LD_tasks):
		p('info', 'solving idx={} i_dataset={} i_task={}'.format(idx, D_task['i_dataset'], D_task['i_task']))
		answer, L_answers_scored_sorted = solve(D_task['question'], D_task)

		if 'correct_answer' in D_task:
			L = [i for i,x in enumerate(L_answers_scored_sorted) if x[0] == D_task['correct_answer']]

			if l(L) == 0:
				score = 10
			else:
				if L[0] == 0:
					score = -20
				elif L[0] == 1:
					score = -5
				else:
					score = min([L[0],10])
			sum_score += score

			correct = answer == D_task['correct_answer']
			cnt_correct += correct

			s_correct = 'CORRECT' if correct else 'WRONG'
			s_correct_answer = '' if correct else ' correct_answer={}'.format(D_task['correct_answer'])

			p('result_task', '{} {}/{}={:.2f} score_rank={:.1f} answer={}{}'.format(s_correct, cnt_correct, idx+1, cnt_correct/(idx+1), sum_score/(idx+1), answer, s_correct_answer))
		else:
			p('result_task', 'RESULT idx_task={} answer={}'.format(idx, answer))

try:
	p('info', 'before import vllm')
	import vllm
	p('info', 'after import vllm {} {}'.format(vllm.__version__, step_time()))

	p('info', 'loading llm')
	p('info', 'N_max_tokens =', N_max_tokens)
	engine_args = vllm.EngineArgs(
		dtype='bfloat16',
		model=llm_path,
		gpu_memory_utilization=.89,
		trust_remote_code=True,
		tensor_parallel_size=N_gpus,
		max_model_len=N_max_tokens,
		enable_prefix_caching=True,
	)
	llm_engine = vllm.LLMEngine.from_engine_args(engine_args)

	p('info', 'llm loaded ' + step_time())
	np.random.seed()

	start_time_llm_loaded = time.time()

	init_solve_tasks()

	if run_main_loop:
		solve_tasks()

except Exception as ex:
	print('exception:', ex)
	extype, _, tb = sys.exc_info()
	if extype != SystemExit:
		traceback.print_exc()
		if conf_pc:
			pdb.post_mortem(tb)



import os
import pandas as pd
import polars as pl
import kaggle_evaluation.aimo_2_inference_server

def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question = question.item(0)
    answer = solve(question)
    return pl.DataFrame({'id': id_, 'answer': answer})

if not run_main_loop:
    inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            (
                '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            )
        )


