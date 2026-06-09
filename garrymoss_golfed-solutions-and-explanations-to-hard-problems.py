!pip install zopfli

import os
import zipfile
import re
import zopfli.zlib
from rich import print as print_rich
from rich.syntax import Syntax
from rich.panel import Panel

# --- Official Utilities Setup ---
# Ensures the helper functions for loading and showing examples are available.
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *

# --- Configuration ---
SUBMISSION_DIR = "/kaggle/working/submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

def save_final_solution(task_id: int, solution_code: str):
    """
    Analyzes the solution code, compresses it with zopfli if beneficial, 
    and saves the final version to the submission directory.
    """
    file_path = os.path.join(SUBMISSION_DIR, f"task{task_id:03d}.py")
    
    solution_bytes = solution_code.strip().encode('utf-8')
    original_len = len(solution_bytes)
    
    # --- Compression Logic ---
    compressed_data = zopfli.zlib.compress(solution_bytes)
    compressed_data = b"'" + compressed_data + b"'" if b'"' in compressed_data else b'"' + compressed_data + b'"'
    compressed_code_str = b"#coding:L1\nimport zlib;exec(zlib.decompress(bytes(%s,'L1')))" % compressed_data
    compressed_len = len(compressed_code_str)

    final_bytes_to_save = solution_bytes
    used_compression = False
    
    if compressed_len < original_len:
        final_bytes_to_save = compressed_code_str
        used_compression = True
        print_rich(f"[bold green]Compression successful![/bold green] Original: {original_len} bytes -> Final: {compressed_len} bytes")
    else:
        print_rich(f"[bold yellow]No compression improvement.[/bold yellow] Original size: {original_len} bytes")
    
    with open(file_path, 'wb') as f:
        f.write(final_bytes_to_save)

    # Display what was saved for verification
    display_code = final_bytes_to_save.decode('utf-8', errors='replace') if used_compression else solution_code
    panel_title = f"[bold]✓ Saved solution for Task {task_id}[/bold] ({len(final_bytes_to_save)} bytes)"
    panel = Panel(
        Syntax(display_code.strip(), "python", theme="monakai", line_numbers=False),
        title=panel_title, border_style="green"
    )
    print_rich(panel)


def create_submission_zip():
    """
    Finds all saved task files and packages them into submission.zip.
    """
    zip_path = "/kaggle/working/submission.zip"
    print_rich(f"\nCreating submission file at [cyan]{zip_path}[/cyan]...")
    
    saved_solutions = [
        f for f in os.listdir(SUBMISSION_DIR) 
        if f.startswith('task') and f.endswith('.py')
    ]

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in sorted(saved_solutions):
            file_path = os.path.join(SUBMISSION_DIR, filename)
            zipf.write(file_path, arcname=filename)
    
    solved_count = len(saved_solutions)
    print_rich(f"[bold green]✓ Done![/bold green]")
    print_rich(f"Packaged [yellow]{solved_count}[/yellow] solutions into the submission file.")
    print_rich("You can now submit '/kaggle/working/submission.zip'.")

# Display the color legend for the grids
show_legend()


TASK_ID = 118
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 118 ---

# --- Explained Version ---
# The function is recursive. It first tries to solve the puzzle assuming the
# crosses have a "length" of 2 (L=2). If it fails, it calls itself to
# try again with a length of 3.
def p(g,L=2):
  # E, H, W are shortened names for range, height, and width for golfing.
  E=range
  H=len(g)
  W=len(g[0])

  # Based on the problem, we can assume colors are integers:
  # black=0, gray=1, red=2. The goal is to turn gray pixels that are part of
  # a cross into cyan (a new color).

  # R is the set of coordinates for all "red" pixels (color 2).
  # The bitwise-and `&2` is a short way to check if the color is 2 or 3.
  # In the input grid, this identifies only the red pixels.
  # These are the pixels that MUST be covered by the crosses we find.
  R={(r,c)for r in E(H)for c in E(W)if g[r][c]&2}

  # S is a recursive lambda function that solves a non-overlapping set-cover problem.
  # It tries to find a subset of candidate crosses (from Q) that perfectly covers
  # all the red pixels (in R) without any of the crosses overlapping.
  # Q: A list of candidate crosses, where each cross is a set of coordinates.
  # U: The union of crosses selected so far for the solution.
  # The expression `[...][R<=U]` is a trick: if R is a subset of U (all red pixels
  # are covered), it returns U. Otherwise, it executes the first part of the list.
  S=lambda Q,U:[
      # If Q is not empty, continue the search.
      Q and(
        # First, try to find a solution *without* using the current candidate cross Q[0].
        S(Q[1:],U)or
        # If that fails, check if the current candidate Q[0] overlaps with the
        # crosses already in our solution U. `{1}>Q[0]&U` is a golfed way to
        # check if the intersection `Q[0]&U` is empty.
        {1}>Q[0]&U and
        # If it doesn't overlap, try to find a solution *including* Q[0].
        S(Q[1:],U|Q[0])
      ),
      U # This is the value returned if the condition `R<=U` is true.
  ][R<=U]

  # This is the main return statement, structured as a chain of boolean logic.
  return(
    # The walrus operator `:=` assigns the result of the solver `S` to the variable U.
    # The solver is called with a list of all possible valid crosses and an initial empty set.
    (U:=S(
      # This is a list comprehension that generates all candidate crosses.
      [
        # C is the set of coordinates for one potential cross.
        C for r in E(H)for c in E(W)for C in[
          # Generate the coordinates of a cross of length L centered at (r, c).
          {(y,x)for k in E(-L,L+1)for y,x in[(r+k,c),(r,c+k)]if H>y>-1<x<W}
        ]
        # This `if` filters the candidates. A cross is only considered a valid candidate
        # if the minimum color value of its pixels, when bitwise-ANDed with 2, is non-zero.
        # With colors black=0, gray=5 and red=2, this means `min(colors)` must be 2.
        # This has the effect of rejecting crosses with any black and also requiring at
        # least one pixel to be red.
        if min(g[y][x]for y,x in C)&2
      ],
      set() # The initial call to S starts with an empty solution set U.
    ))
    # If the solver `S` found a solution, U will be a non-empty set (truthy).
    and
    # The `exec(...) or g` trick modifies the grid `g` in-place and then returns `g`.
    (exec(
      # This loop modifies the grid based on the solution `U`.
      # It iterates over every coordinate `(r,c)` in the solution crosses.
      'for r,c in U:'
      # `g[r][c]&1` is 1 if the color is odd (gray=1) and 0 if it is even (red=2).
      # It adds 3 to gray pixels (1 -> 4), effectively changing them to cyan.
      # It adds 0 to red pixels (2 -> 2), leaving them unchanged.
      'g[r][c]+=3*(g[r][c]&1)'
    ) or g)
    # If the solver `S` failed to find a solution for L=2 (it returned a falsy value),
    # this `or` clause executes, retrying the entire function for crosses of length 3.
    or p(g,3)
  )

# --- Optimized (No Compression) ---
def p(g,L=2):E=range;H=len(g);W=len(g[0]);R={(r,c)for r in E(H)for c in E(W)if g[r][c]&2};S=lambda Q,U:[Q and(S(Q[1:],U)or{1}>Q[0]&U and S(Q[1:],U|Q[0])),U][R<=U];return(U:=S([C for r in E(H)for c in E(W)for C in[{(y,x)for k in E(-L,L+1)for y,x in[(r+k,c),(r,c+k)]if H>y>-1<x<W}]if min(g[y][x]for y,x in C)&2],set()))and(exec('for r,c in U:g[r][c]+=3*(g[r][c]&1)')or g)or p(g,3)

# --- Optimized for Compression ---
def p(p):
 t={(l,n)for l in range(len(p))for n in range(len(p[0]))if p[l][n]&2};d=lambda i,l:[i and(d(i[1:],l)or not i[0]&l and d(i[1:],l|i[0])),l][t<=l]
 for n in 2,3:
  l=[l for d in range(len(p))for i in range(len(p[0]))for l in[{(l,n)for l in range(-n,n+1)for l,n in[(d+l,i),(d,i+l)]if len(p)>l>-1<n<len(p[0])}]if min(p[l][n]for l,n in l)&2]
  if l:=d(l,set()):
   for l,n in l:p[l][n]+=3*(p[l][n]&1)
   return p

# --- Save Final Solution ---
# We extract the last defined solution from this cell to save it.
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


TASK_ID = 133
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 133 ---

# --- Explained Version ---
def p(G):
 # Initialize P (pixel dict) and S (sprite list) via unpacking.
 P,T,*S={(r,c):v for r,g in enumerate(G)for c,v in enumerate(g)if v},[]
 # Find sprites using DFS.
 def B(z):
  if z in P:C[z]=P.pop(z);B((z[0]-1, z[1]));B((z[0]+1, z[1]));B((z[0], z[1]-1));B((z[0], z[1]+1))
 while P:S+=[C:={}];B([*P][0])
 # Iterate S+S: first pass learns T (from M=1 sprite), second pass reconstructs.
 for C in S+S:
  # K=Anchor color, V=Main color
  K,={*S[0].values()}&{*S[1].values()};V,={*C.values()}-{K}
  # Analysis: X=Anchors, M=Size, r,c=Top-left.
  r,c=min(X:=[z for z in C if C[z]==K]);M=max(X)[0]-r+1
  # Template update: (C[z]==V)==M is a golfed check for (M==1 and C[z]==V).
  T+=[(z[0]-r,z[1]-c)for z in C if(C[z]==V)==M]
  # Reconstruction
  for u,v in T:
   for i in range(M*M):G[r+u*M+i//M][c+v*M+i%M]=V
 return G

# --- Optimized (No Compression) ---
def p(G):
 W,E,*J=30,enumerate;C,*B={B*W+D:A for B,C in E(G)for D,A in E(C)if A},
 def F(z):
  if A:=C.pop(z,0):H[0][z]=A;H[1].add(A);[F(z+A%3-1+A//3*W-W)for A in range(9)]
 while C:B+=[H:=({},set())];F(min(C))
 for D,N in B+B:
  I,=B[0][1]&B[1][1];Z=min(O:=[A for A in D if D[A]==I]);A=int(len(O)**.5);L,=N-{I};J+=[B-Z for B in D if(D[B]==L)==A]
  for T in J:
   for M in range(A*A):Q=Z+T*A+M//A*W+M%A;G[Q//W][Q%W]=L
 return G
    
# --- Optimized for Compression ---
def p(u):
 e,t,*o={(p,m):v for p,u in enumerate(u)for m,v in enumerate(u)if v},[]
 def p(u):
  if u in e:r[u]=e.pop(u);p((u[0]-1,u[1]));p((u[0]+1,u[1]));p((u[0],u[1]-1));p((u[0],u[1]+1))
 while e:o+=[r:={}];p(min(e))
 for r in o+o:
  i,={*o[0].values()}&{*o[1].values()};g,={*r.values()}-{i};p,m=min(e:=[u for u in r if r[u]==i]);i=max(e)[0]+1-p;t+=[(u[0]-p,u[1]-m)for u in r if(r[u]==g)==i]
  for r,v in t:
   for e in range(i*i):u[p+r*i+e//i][m+v*i+e%i]=g
 return u

# --- Save Final Solution ---
# We extract the last defined solution from this cell to save it.
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


TASK_ID = 157
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 157 ---

# --- Explained Version ---
def p(g):
  # R is an alias for range, W is the grid width (15).
  # A is initialized as an empty list [], to be used to accumulate the
  # pixel indices of the current gray shape being scanned.
  R,W,*A=range,15
  
  # F is the flattened 1D grid, created by sum(g, []).
  # P is initialized to [[]] (a list containing one empty combination).
  # This works by unpacking the tuple (sum(g,A), A) where A is initially [].
  F,*P=sum(g,A),A

  # This is the main shape-finding loop, implemented as a list comprehension
  # that iterates through columns 0 to 15 (R(16)). The 16th "fake" column (x=15)
  # ensures that the logic for a completed shape triggers if a shape
  # ends in the very last column. The comprehension runs for its side-effects.
  [
    # This is a chained comparison used as a conditional. In Python,
    # `X op1 Y op2 Z` is equivalent to `(X op1 Y) and (Y op2 Z)`.
    # This entire expression is only true when a shape has just been completed.
    
    # Part 1: A == (A:=...). This checks if A's value has changed.
    # The inner part `A:=A+[...]` attempts to add all non-black pixels from
    # the current column `x` to A. The comparison is true only if no pixels
    # were added, which signifies we've hit a column without gray pixels.
    A==(A:=A+[z for z in R(x,150,W)if(W>x)&F[z]])
    
    # Part 2: ... > []. The right side of this is the newly updated A.
    # A non-empty list is lexicographically greater than an empty one.
    # This is a short way of checking `and A is not empty`.
    # So, the logic is: "no pixels were added this column, AND A is not empty".
    # This is the exact condition for having just finished scanning a shape.
    > []
    
    # Part 3: [] < [...]. This part only gets evaluated if the first two
    # conditions are true, due to short-circuiting.
    # The list on the right will always be lexicographically greater than [],
    # so this comparison is always true. Its purpose is to execute the
    # code within it for its side effects.
    < [
        # This rebuilds the list of all placement combinations P.
        # For every existing combination 'c', it creates new ones by appending
        # a pair: (anchor 'a', shape 'A'). 'a' is any valid black pixel
        # in the top 3 rows. 'A' is the raw pixel list of the shape just found.
        P:=[c+[(a,A)]for a in R(45)for c in(F[a]<1)*P],
        
        # After the shape is processed and stored in P, A is reset to an
        # empty list to begin searching for the next shape.
        A:=[]
      ]
    # The loop that drives this whole shape-scanning comprehension.
    for x in R(16)
  ]

  # After all shapes are found and all possible placement combinations (P)
  # are generated, find the correct output grid by iterating all combinations.
  # max() selects the "best" grid lexicographically.
  return max([
      # This is a golfed 1D-to-2D grid reshape.
      # It creates a list of W (15) references to the *same* pixel generator,
      # then zips them. This pulls 15 items at a time (one row) from the generator.
      *zip(*[
          # This generator produces the 150 pixels for one complete output grid
          # based on the current combination 'c'.
          ((
            # any() checks if the current pixel 'z' is covered by any shape
            # in this combination. It returns 1 (blue) if covered, 0 otherwise.
            any(
                # c is a list of (anchor, shape_pixels) pairs.
                # This is the hit-test: it checks if 'z', when offset by the
                # anchor 'a', exists in the shape's raw pixel list 'h'.
                z-a+min(h) in h
                # This is an optimization: if the pixel 'z' is too far
                # horizontally from the anchor 'a', skip the (expensive) 'in' check.
                * (a%W-z%W<6)
                for a,h in c
            )
            
            # The result (0 or 1) is bitwise OR'd with the original background.
            # F[z]%5 maps: Black(0)->0, Red(2)->2, Gray(5)->0.
            # This erases the gray squares but keeps the red bar and black background.
            | F[z]%5

            # This final modulo 3 is the validity check.
            # Legal pixels: 0|0=0, 0|2=2, 1|0=1. All are unchanged by %3.
            # ILLEGAL pixel (blue on red): 1|2 = 3.
            # 3 % 3 = 0.
            # This maps any illegal placement onto a black pixel (0).
            # Since this happens in the top rows, max() will discard this
            # invalid solution because it will be lexicographically "smaller"
            # than a valid solution which has only 1s and 2s in its top rows.
          ) % 3 for z in R(150)) # Iterate all 150 pixels
      ]*W)
    # Iterate through every possible combination 'c' in our list P.
    ] for c in P
  )

# --- Optimized (No Compression) ---
def p(g):R,W,*A=range,15;F,*P=sum(g,A),A;[A==(A:=A+[z for z in R(x,150,W)if(W>x)&F[z]])>[]<[P:=[c+[(a,A)]for a in R(45)for c in(F[a]<1)*P],A:=[]]for x in R(16)];return max([*zip(*[((any(z-a+min(h)in h*(a%W-z%W<6)for a,h in c)|F[z]%5)%3for z in R(150))]*W)]for c in P)

# --- Initial version before bizy-coder's improvements ---
def p(g):
 R=range;F=sum(g,l:=[]);G=[];P=[[]];W=15
 for x in R(16):
  if(5in F[x::W])>x/W:G+=[x]
  elif G:A=[z for z in R(150)if z%W in G*(F[z]&5)];l+=[[z-A[0]for z in A]];G=[];P=[c+[a]for c in P for a in R(45)if F[a]<1]
 for c in P:
  O=[any(z-x in h*(x%W-z%W<6)for x,h in zip(c,l))|F[z]%5for z in R(150)]
  if all(O[:45])&(O.count(0)==F.count(0)):return[O[i*W:][:W]for i in R(10)]

# --- Optimized for Compression ---
def p(n):p,i,*r=range,15;m=sum(n,[]);n=[[]];[(n:=[n+[(f,r)]for n in n for f in p(45)if m[f]<1],r:=[])for f in p(16)if r==(r:=r+[p for p in p(f,150,i)if(i>f)&m[p]])>[]];return max([*zip(*[((any(p-f+min(n)in n*(f%i-p%i<6)for f,n in n)|m[p]%5)%3for p in p(150))]*i)]for n in n)

# --- Save Final Solution ---
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


TASK_ID = 158
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 158 ---

# --- Explained Version ---
def p(g):
  # --- Setup and Initialization ---

  X=range # Alias range for brevity.
  R=len(g) # R = number of rows in the grid.
  C=len(v:=g[0]) # C = number of columns. The walrus operator also assigns the first row to 'v'.

  # Find the background color 'b' by finding the most frequent color in the first row.
  b=max(v,key=v.count)

  # H contains the original grid and a vertically-flipped version.
  # This allows searching for patterns without writing separate logic for vertical flips.
  H=g,g[::-1]
  
  # --- Find the 3x3 Sprite Pattern ---

  # This line finds the single, complete 3x3 sprite to learn the color pattern 'p'.
  # It iterates through every 3x3 window in both the normal and flipped grids.
  p,=[x for g in H # Iterate through the original grid and its vertical flip.
      for r in X(R-2) # Iterate through all possible top-left row positions 'r'.
      for c in X(C-2) # Iterate through all possible top-left column positions 'c'.
      # This is a single, valid, chained comparison that acts as a complex filter.
      # It is parsed as: (len(set(x)) > 3) and (3 > 0) and (0 < b) and (b != x[0]) and (x[0] != x[8]) and (x[8] != b)
      if len(set(x:=[g[r+i//3][c+i%3]for i in X(9)]))>3>0<b!=x[0]!=x[8]!=b]
      # Breakdown of the chained comparison's logic:
      # x:=[...] -> 'x' becomes a flattened list of the 9 colors in the 3x3 window.
      # len(set(x))>3 -> The window must have >3 unique colors (the 3 sprite colors + background).
      # 3>0 -> Always True. A trick to continue the chain.
      # 0<b -> True if background color isn't 0. Included to continue the chain.
      # b!=x[0] -> The top-left pixel (x[0]) is not the background.
      # x[0]!=x[8] -> The top-left and bottom-right corners are different colors.
      # x[8]!=b -> The bottom-right pixel (x[8]) is not the background.
  # The list comprehension finds the one matching 3x3 square, and it's unpacked into 'p'.

  # --- Find and Repair Corrupted Sprites ---

  # This is a list comprehension whose main purpose is the side effect of the 'exec'
  # statement, which modifies the grid 'g' in place. The list itself is created and discarded.
  [
    # This entire block is a clever use of boolean short-circuiting (`or`).
    # The structure is `(comparison) or (side_effect)`.
    # If `comparison` is True, the `or` stops and the `exec` is NOT run.
    # If `comparison` is False, the `or` continues and runs the `exec` statement.

    # W is a helper lambda to generate one row of a magnified sprite pattern.
    # It defaults to drawing background 'b' but can be overridden.
    (W:=lambda a=b,b=b,c=b,*_:[a]*u+[b]*u+[c]*u)
    # This generates the "corrupted" sprite template (corners only, middle is background).
    (W(p[0]),W(),W(c=p[8]))
    # The comparison: checks if the current grid slice is NOT EQUAL to the corrupted template.
    !=[g[r+q][f:=slice(c,[t:=c+3*u*s,None][t<0],s)]for q in X(3*u)]
    
    or # If the above was False (i.e., a corrupted sprite was found), run the 'exec'.

    # The repair step: This code is executed as a string.
    # It iterates through the rows of the identified corrupted area and replaces
    # each row with the correctly generated full sprite row.
    exec('for q in X(3*u):g[r+q][f]=W(*p[q//u*3:])')
    
    # --- Loop Definitions for the repair process ---
    for u in X(4) # Loop through magnifications 'u' from 0 to 3. (u=0 is effectively skipped).
    for s in[-1,1] # Loop through horizontal direction 's' (-1 for flipped, 1 for normal).
    for r in X(R-3*u+1) # Loop through all possible top row positions 'r' for this size.
    for c in X(C) # Loop through all possible start column positions 'c'.
    for g in H # Loop through the original and vertically-flipped grids.
  ]

  return g

# --- Optimized (No Compression) ---
def p(g):X=range;R=len(g);C=len(v:=g[0]);b=max(v,key=v.count);H=g,g[::-1];p,=[x for g in H for r in X(R-2)for c in X(C-2)if len(set(x:=[g[r+i//3][c+i%3]for i in X(9)]))>3>0<b!=x[0]!=x[8]!=b];[(W:=lambda a=b,b=b,c=b,*_:[a]*u+[b]*u+[c]*u)(W(p[0]),W(),W(c=p[8]))!=[g[r+q][f:=slice(c,[t:=c+3*u*s,None][t<0],s)]for q in X(3*u)]or exec('for q in X(3*u):g[r+q][f]=W(*p[q//u*3:])')for u in X(4)for s in[-1,1]for r in X(R-3*u+1)for c in X(C)for g in H];return g

# --- Optimized for Compression ---
def p(n):o=max(n[0],key=n[0].count);f,=[r for a in(n,n[::-1])for g in range(len(n)-2)for h in range(len(n[0])-2)if len(set(r:=[a[g+p//3][h+p%3]for p in range(9)]))>3>0<o!=r[0]!=r[8]!=o];[(t:=lambda d=o,b=o,c=o,*z:[d]*u+[b]*u+[c]*u)(t(f[0]),t(),t(c=f[8]))!=[a[g+p][h:[h+3*u*s,None][h+3*u*s<0]:s]for p in range(3*u)]or exec('for p in range(3*u):a[g+p][h:[h+3*u*s,None][h+3*u*s<0]:s]=t(*f[p//u*3:])')for u in range(4)for s in(1,-1)for g in range(len(n)-3*u+1)for h in range(len(n[0]))for a in(n,n[::-1])];return n

# --- Save Final Solution ---
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


TASK_ID = 173
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 173 ---

# --- Explained Version ---
def p(g):
  # R is a shortcut for the range function to save characters.
  R=range
  # N is range(9), for iterating through the 9 cells of a 3x3 grid (0-8).
  N=R(9)
  # H is the range of possible top-row indices for any 3x3 subgrid.
  H=R(len(g)-2)
  # W is the range of possible left-column indices for any 3x3 subgrid.
  W=R(len(g[0])-2)

  # This is a list comprehension whose result is discarded. Its purpose is to
  # loop through all pairs of 3x3 windows and perform a repair via `exec`.
  # (R, C) is the top-left of the window to potentially repair.
  # (x, y) is the top-left of the window to use as a potential template.
  [
    # The entire block is a single boolean expression used for conditional execution.
    # If this expression is True, the `exec` call after `and` is run.
    # Python's `*` on booleans acts as a logical AND.

    (
      # Define F as a flattened list of the 9 pixels from the template window at (x,y).
      # The walrus operator (:=) both creates and assigns the list to F.
      (F:=[g[x+i//3][y+i%3]for i in N])
      # 1. Test if the template is a valid sprite shape:
      #    - All sprites are symmetrical, which means the flattened 3x3 grid is a palindrome.
      ==F[::-1]
    )
    #    - All sprites have a non-black center pixel.
    *F[4]
    #    - All sprites have other non-black pixels besides the center.
    *any(F[:4])
    
    # 2. Test if this valid template `F` matches the damaged sprite at (R,C).
    *(
      # Case A: The sprite at (R,C) is missing its edges (only the center remains).
      # We match it to the template `F` by checking if their center colors are the same.
      F[4]==g[R+1][C+1]
      or
      # Case B: The sprite at (R,C) is missing its center.
      # We match by checking for high similarity (>7 of 9 pixels are identical).
      sum(g[R+i//3][C+i%3]==F[i]for i in N)>7
    )
    
    # If all conditions pass, we have found a valid template F for the damaged
    # sprite at (R,C). Now, execute the repair.
    and
    exec(
      # This string is executed as Python code. It iterates through the 9 cells (i in N)
      # and copies each pixel from the template `F` onto the grid `g` at location (R,C).
      'for i in N:g[R+i//3][C+i%3]=F[i]'
    )
    
    # These nested loops drive the entire process, comparing every possible
    # 3x3 window with every other possible 3x3 window.
    for R in H
    for x in H
    for C in W
    for y in W
  ]
  
  # After the loops complete and all repairs are made, return the grid.
  return g

# --- Optimized (No Compression) ---
def p(g):R=range;N=R(9);H=R(len(g)-2);W=R(len(g[0])-2);[((F:=[g[x+i//3][y+i%3]for i in N])==F[::-1])*F[4]*any(F[:4])*(F[4]==g[R+1][C+1]or sum(g[R+i//3][C+i%3]==F[i]for i in N)>7)and exec('for i in N:g[R+i//3][C+i%3]=F[i]')for R in H for x in H for C in W for y in W];return g

# --- Optimized for Compression ---
def p(n):[((r:=[n[y+f//3][t+f%3]for f in range(9)])==r[::-1])*r[4]*any(r[:4])*(r[4]==n[l+1][d+1]or sum(n[l+f//3][d+f%3]==r[f]for f in range(9))==8)and exec('for f in range(9):n[l+f//3][d+f%3]=r[f]')for l in range(len(n)-2) for y in range(len(n)-2) for d in range(len(n[0])-2) for t in range(len(n[0])-2)];return n

# --- Save Final Solution ---
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


TASK_ID = 233
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 233 ---

# --- Explained Version ---
def p(g):
 # R: alias for range. N: range(9) for 3x3 block indices.
 # b: alias for tuple. S: dictionary to store Sprite signatures and colors.
 R=range;N=R(9);b=tuple;S={}
 # E: lambda to crop a grid by removing all-zero rows, then all-zero columns.
 E=lambda M:[*map(list,zip(*filter(any,M)))]
 # W: lambda for a 3x3 sliding Window. It calls function f(row, col, block)
 # for each 3x3 block B in the grid g.
 W=lambda f:[f(r,c,[g[r+i//3][c+i%3]for i in N])for r in R(len(g)-2)for c in R(len(g[0])-2)]
 
 # A is the function to find and Analyze sprites in the input grid.
 def A(r,c,B):
  # A sprite is a 3x3 block with 2 colors: red (2) and a background color.
  # C is the set of colors in the block B. It must not contain black (0).
  if len(C:={*B})==2and C&{0,2}=={2}:
   # K is the Key (signature) of the sprite's red shape, a tuple of booleans.
   K=b(x==2for x in B)
   # D is a shared dictionary used to mark a sprite as "used" once drawn.
   # The key is the original shape, the value is the background color.
   D={K:sum(C)-2}
   # Store the sprite data for all 4 possible rotations.
   # All rotations will point to the *same* dictionary object D.
   for i in R(4):
    S[K]=D
    # K is updated to its 90-degree clockwise rotation for the next loop.
    K=b(K[j%3*3+2-j//3]for j in N)
   # Erase the found sprite from the input grid to avoid re-processing.
   for k in N:
    g[r+k//3][c+k%3]=0

 # First pass: run W(A) to find all source sprites and populate the map S.
 W(A)
 # Crop the grid to the target red rectangle by running E twice (rows, then cols).
 g=E(E(g))
 
 # Second pass: find black shapes in the target grid and draw the sprites.
 # Loop twice (k=0, k=1). The first pass handles non-rotated sprites,
 # the second handles any remaining rotated sprites.
 [W(lambda r,c,B:
  # P is the pattern of black pixels in the current window.
  # D is the shared "used" marker dictionary if P is a known sprite shape.
  (D:=S.get(P:=b(x<1for x in B)))
  # On pass k=0, only draw if shape P is the original, unrotated shape key.
  # On pass k=1, this is always true, so any valid rotation will be drawn.
  and(k or P in D)
  # Pop the item from D to get the color V and mark the sprite as used.
  and(V:=D.popitem()[1],
      # Draw the sprite: black pixels (where P[j] is True) become red (2),
      # and the red background becomes the sprite's background color V.
      [g[r+j//3].__setitem__(c+j%3,(V,2)[P[j]])for j in N])
  )for k in R(2)]
 
 # Return the final grid.
 return g

# --- Optimized (No Compression) ---
def p(g):
 R=range;N=R(9);b=tuple;S={};E=lambda M:[*map(list,zip(*filter(any,M)))];W=lambda f:[f(r,c,[g[r+i//3][c+i%3]for i in N])for r in R(len(g)-2)for c in R(len(g[0])-2)]
 def A(r,c,B):
  if len(C:={*B})==2and C&{0,2}=={2}:
   K=b(x==2for x in B);D={K:sum(C)-2}
   for i in R(4):S[K]=D;K=b(K[j%3*3+2-j//3]for j in N)
   for k in N:g[r+k//3][c+k%3]=0
 W(A);g=E(E(g));[W(lambda r,c,B:(D:=S.get(P:=b(x<1for x in B)))and(k or P in D)and(V:=D.popitem()[1],[g[r+j//3].__setitem__(c+j%3,(V,2)[P[j]])for j in N]))for k in R(2)];return g

# --- Optimized for Compression ---
def p(u):
 i=range(9);c={};f=lambda a:[a(p,s,[u[p+a//3][s+a%3]for a in i])for p in range(len(u)-2)for s in range(len(u[0])-2)]
 def p(p,s,n):
  if len(r:={*n})==2and r&{0,2}=={2}:
   n=tuple(a==2for a in n);k={n:sum(r)-2}
   for a in range(4):c[n]=k;n=tuple(n[a%3*3+2-a//3]for a in i)
   for a in i:u[p+a//3][s+a%3]=0
 f(p);u=[*map(list,zip(*filter(any,[*map(list,zip(*filter(any,u)))])))];[f(lambda p,s,n:(t:=c.get(n:=tuple(a==0for a in n)))and(a or n in t)and(r:=t.popitem()[1],[u[p+a//3].__setitem__(s+a%3,(r,2)[n[a]])for a in i]))for a in range(2)];return u

# --- Save Final Solution ---
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


TASK_ID = 328
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 328 ---

# --- Explained Version ---
def p(g):
  # R is a range for iterating over grid coordinates, assuming a square grid.
  R=range(len(g));
  # Build the output grid using a nested list comprehension.
  return[ # Start of outer list comprehension for rows
    [ # Start of inner list comprehension for columns
      # For each output cell (r, c), find all non-black pixels in the input grid `g`,
      # sort them by distance to (r, c), and calculate a new color.

      # The walrus operator `:=` assigns the sorted list of neighbors to `D`.
      (D:=sorted(
        # This generator expression creates a list of tuples to be sorted.
        (
          # 1. Sort key: The Manhattan distance from input pixel (x,y) to output pixel (r,c).
          # T is assigned the coordinate differences [dy, dx] for reuse.
          sum(T:=[abs(x-r),abs(y-c)]),
          # 2. Associated value: The color `v` of the input pixel, but modified.
          # The color `v` is kept only if the Chebyshev distance (`max(T)`) is even.
          # `~max(T)&1` is a golfed way to check if `max(T)` is even.
          v*(~max(T)&1)
        )
        # Iterate over every pixel (x, y) in the input grid `g`.
        for x in R for y in R
        # Filter for non-black pixels, assigning their color to `v`.
        if(v:=g[x][y])
      ))
      # `D` is now a list of `(distance, modified_color)` tuples, sorted by distance.
      # The `and` operator chain below determines the final pixel color.
      
      # First, `D` must be non-empty for the expression to proceed.
      # Then, check if the closest pixel is uniquely the closest by comparing
      # the distance of the first `D[0][0]` and second `D[1][0]` elements.
      and(D[0][0]<D[1][0])
      # The boolean result (True=1, False=0) is multiplied by the modified color
      # of the closest pixel, `D[0][1]`.
      # If there's a unique closest pixel, the result is its modified color.
      # If there's a tie in distance (or no colored pixels), the result is 0.
      *D[0][1]
    for c in R # End of inner list comprehension (columns)
  ] for r in R # End of outer list comprehension (rows)
]

# --- Optimized (No Compression) ---
def p(g):R=range(len(g));return[[(D:=sorted((sum(T:=[abs(x-r),abs(y-c)]),v*(~max(T)&1))for x in R for y in R if(v:=g[x][y])))and(D[0][0]<D[1][0])*D[0][1]for c in R]for r in R]

# --- Save Final Solution ---
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


TASK_ID = 361
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 361 ---

# --- Explained Version ---
def p(g):
  # Alias 'range' to 'R' for brevity, a common code golf practice.
  R=range;
  
  # Find the center of rotation (G,H) using a doubled coordinate system.
  # This system allows for centers that fall between pixels (e.g., for 2x2 shapes).
  # The center is identified by finding a solid 3x3 or 2x2 square at the pattern's core.
  G,H=[
    (2*b+a-1,2*d+a-1) # Calculate the doubled center from the square's top-left corner (b,d) and size (a).
    for a in(3,2) # Iterate through possible square sizes, 3x3 then 2x2.
    for b in R(8) # Iterate through possible top rows of the square.
    for d in R(8) # Iterate through possible left columns of the square.
    if all(all(r[d:d+a])for r in g[b:b+a]) # Check if the a*a subgrid is completely filled (non-zero).
  ][0]; # Take the first center found.

  # This list comprehension completes the pattern by filling in missing pixels.
  # It is used for its side effect of modifying the grid 'g', not for the list it produces.
  [
    (d:=2*a-G,e:=2*b-H) # Calculate the doubled vector (d,e) from the center (G,H) to pixel (a,b).
    and
    # Execute a command string three times to fill the 90, 180, and 270-degree rotated points.
    exec('d,e=-e,d;g[G+d>>1][H+e>>1]=k;'*3)
    # Inside the executed string:
    # 'd,e=-e,d' rotates the vector (d,e) by 90 degrees.
    # 'g[...]=k' places the original color 'k' at the new position.
    # '>>1' is a bitwise right shift, equivalent to integer division by 2,
    # which converts from doubled coordinates back to standard grid coordinates.
    for a in R(10) # Iterate through all grid rows 'a'.
    for b in R(10) # Iterate through all grid columns 'b'.
    if(k:=g[a][b]) # If the pixel at (a,b) is colored, assign its color to 'k' and proceed.
  ];
  
  # Return the now-completed grid.
  return g

# --- Optimized (No Compression) ---
def p(g):R=range;G,H=[(2*b+a-1,2*d+a-1)for a in(3,2)for b in R(8)for d in R(8)if all(all(r[d:d+a])for r in g[b:b+a])][0];[(d:=2*a-G,e:=2*b-H)and exec('d,e=-e,d;g[G+d>>1][H+e>>1]=k;'*3)for a in R(10)for b in R(10)if(k:=g[a][b])];return g

# --- Optimized for Compression ---
def p(i):g,d=[(2*n+e-1,2*x+e-1)for e in(3,2)for n in range(8)for x in range(8)if all(all(a[x:x+e])for a in i[n:n+e])][0];[(l:=2*n-g,e:=2*x-d)and exec('l,e=-e,l;i[g+l>>1][d+e>>1]=a;'*3)for n in range(10)for x in range(10)if(a:=i[n][x])];return i

# --- Save Final Solution ---
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


TASK_ID = 366
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 366 ---

# --- Explained Version ---
def p(g):
    # A, B, D are aliases for range, len, sorted to save space.
    # F is a function to find unique colors in a grid q, sorted by frequency (least to most).
    A=range;B=len;D=sorted;F=lambda q:D(set(c:=sum(q,[])),key=c.count);

    # R is True if the input grid g is tall (height > width), False otherwise.
    # I becomes the grid, rotated via transpose to be horizontal if it was tall.
    I=[*map(list,(g,zip(*g))[R:=B(g)>B(g[0])])];

    # h, w are the height and width of one half of the standardized grid I.
    h,w=B(I),B(I[0])//2;

    # Split I into two halves. Identify the Source (S) and Target (T) halves.
    # The source half is the one with more unique colors, so we sort by that.
    T,S=D([[r[:w]for r in I],[r[w:]for r in I]],key=lambda q:B(F(q)));

    # In the Source grid S, find the two most frequent colors.
    # f is the foreground color, b is the background color of the S-pane.
    f,b=F(S)[-2:];
    # In the Target grid T, find the most frequent color.
    # z is the background color of the T-pane.
    z=F(T)[-1];

    # This is the main logic, implemented as a nested list comprehension.
    # It finds boxes in S, finds their corresponding pattern in T, and copies them over.
    [
        # First part of 'or': erase the found box in S by filling with background b.
        # This prevents re-matching. exec returns None (falsy), so the second part always runs.
        exec('for r in S[y:y+H]:r[x:x+W]=[b]*W')
        or
        # Second part: find where to draw the box in T and draw it.
        [
            # Condition to find a match in T.
            # It checks if for every row in the box, the pattern matches.
            # The pattern is the box's dots, with the box's foreground (f) replaced by T's background (z).
            all(T[u+j][v:v+W]==[(m,z)[m==f]for m in t[j]]for j in A(H))
            # If a match is found, execute the drawing command.
            and
            exec('for j in A(H):T[u+j][v:v+W]=t[j]')
            # Loop through all possible top-left positions (u, v) in T to check for a match.
            for u in A(h-H+1)for v in A(w-W+1)
        ]
        # These outer loops iterate through all possible boxes in S.
        # n is the number of 'dot' pixels in a box (iterating 3, 2, 1).
        for n in[3,2,1]
        # (y, x) is the top-left corner of the box.
        for y in A(h)for x in A(w)
        # (H, W) are the height and width of the box (iterating largest to smallest).
        for H in A(h-y,0,-1)for W in A(w-x,0,-1)
        # This is the condition to identify a box 't' in S.
        # A box is a rectangle that does not contain S's background color (b),
        # and where the number of non-foreground pixels (the dots) is exactly n.
        if b not in(c:=sum(t:=[r[x:x+W]for r in S[y:y+H]],[]))and H*W-c.count(f)==n
    ];

    # Return the solved grid T. If the original grid was rotated (R is True),
    # rotate it back by transposing it before returning.
    return(T,[*zip(*T)])[R]

# --- Optimized (No Compression) ---
def p(g):A=range;B=len;D=sorted;F=lambda q:D(set(c:=sum(q,[])),key=c.count);I=[*map(list,(g,zip(*g))[R:=B(g)>B(g[0])])];h,w=B(I),B(I[0])//2;T,S=D([[r[:w]for r in I],[r[w:]for r in I]],key=lambda q:B(F(q)));f,b=F(S)[-2:];z=F(T)[-1];[exec('for r in S[y:y+H]:r[x:x+W]=[b]*W')or[all(T[u+j][v:v+W]==[(m,z)[m==f]for m in t[j]]for j in A(H))and exec('for j in A(H):T[u+j][v:v+W]=t[j]')for u in A(h-H+1)for v in A(w-W+1)]for n in[3,2,1]for y in A(h)for x in A(w)for H in A(h-y,0,-1)for W in A(w-x,0,-1)if b not in(c:=sum(t:=[r[x:x+W]for r in S[y:y+H]],[]))and H*W-c.count(f)==n];return(T,[*zip(*T)])[R]

# --- Optimized for Compression ---
def p(r):g=len(r)>len(r[0]);n=[*map(list,(r,zip(*r))[g])];f,o=len(n),len(n[0])>>1;i,r=sorted([[r[:o]for r in n],[r[o:]for r in n]],key=lambda r:len(set(sum(r,[]))));y,a=sorted(set(p:=sum(r,[])),key=p.count)[-2:];u=max(set(p:=sum(i,[])),key=p.count);[exec('for r in r[t:t+e]:r[n:n+s]=[a]*s')or[all(i[t+a][n:n+s]==[(a,u)[a==y]for a in k[a]]for a in range(e))and exec('for a in range(e):i[t+a][n:n+s]=k[a]')for t in range(f-e+1)for n in range(o-s+1)]for g in range(3,0,-1)for t in range(f)for n in range(o)for e in range(f-t,0,-1)for s in range(o-n,0,-1)if a not in(p:=sum(k:=[r[n:n+s]for r in r[t:t+e]],[]))and e*s-p.count(y)==g];return(i,[*zip(*i)])[g]

# --- Save Final Solution ---
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


TASK_ID = 392
current_task_data = load_examples(TASK_ID)
show_examples(current_task_data['train'] + current_task_data['test'])


# --- Solution Versions for Task 392 ---

# --- Explained Version ---
# p is a lambda function that takes the input grid `n`.
# r=range(10) is a default argument, a shorthand for iterating from 0-9.
p=lambda n,r=range(10):[
  # This is the outer list comprehension, creating the list of rows.
  # `f` is the row index of the cell we are generating.
  [
    # The color for each cell (f, m) is determined by this expression.
    # It selects a color from a tuple using an index.
    (
      # The tuple of possible colors: (pattern_color, gray, gray).
      # `a` is the pattern color, found by taking the max color value in the input grid `n`.
      # `sum(n, [])` flattens the grid to find the max value easily.
      # The walrus operator `:=` assigns this max value to `a` for later use.
      a:=max(sum(n,[])),
      5, # ARC's color code for gray.
      5  # Gray is used for the background if the index isn't 0.
    )[
      # The index into the color tuple is calculated here.
      # The logic is: (min_distance_to_pattern % step_size).
      # If the index is 0, we get the pattern color `a`. Otherwise, we get gray.

      # This calculates the minimum Chebyshev distance (max of differences in coords)
      # from the current cell (f, m) to any non-black pixel (b, s) in the input `n`.
      min(max(abs(f-b),abs(m-s))for b in r for s in r if n[b][s])

      % # Modulo operator.

      # This calculates the step size of the pattern (either 2 or 3).
      # It cleverly checks if the string pattern 'color, 0, color' (e.g., '8, 0, 8')
      # exists in the string representation of the grid.
      # If it exists, the expression is True (1), so step = 3-1 = 2.
      # If not, it's False (0), so step = 3-0 = 3.
      (3-(f'{a}, 0, {a}'in str(n)))
    ]
    # This is the inner list comprehension, creating a single row.
    # `m` is the column index of the cell we are generating.
    for m in r
  ]
  for f in r
]

# --- Optimized (No Compression) ---
p=lambda n,r=range(10):[[(a:=max(sum(n,[])),5,5)[min(max(abs(f-b),abs(m-s))for b in r for s in r if n[b][s])%(3-(f'{a}, 0, {a}'in str(n)))]for m in r]for f in r]

# --- Save Final Solution ---
cell_source = _ih[-1]
last_solution = re.findall(r"(def p\(.*?\):.*?|p=lambda.*?:.*?)(?=\n\n|\n# ---|\n$)", cell_source, re.S)[-1]
save_final_solution(TASK_ID, last_solution)


create_submission_zip()




