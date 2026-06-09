import re
p=lambda g:eval(re.sub('(?<=[12346789].{34})5','0',str(g)))


import re
p=lambda g:eval(re.sub('(?<=[^5],.{33})5','0',str(g)))


# 60b soln.
p=lambda g,h=[]:g*0!=0and[*map(p,g,g[9:]+h+g)]or(g%6<=h%6)*g

