import numpy as np
#reversing an array
arr=np.array([1,2,3,6,4,5])
print("arr=",arr)
#1st method:
rarr=arr[::-1]
print("reversed array=",rarr)
#2nd method:
rev=np.flip(arr)
print("reversed array=",rev)


#flatten an array
array=np.array([[1,2,3],[2,4,5],[1,2,3]])
print("array=",array)
flt=array.flatten()
print("Flattened array=",flt)
flt1=array.ravel()
print("Flattened array=",flt1)


#comparing 2 arrays
a1=np.array([[1,2],[3,4]])
a2=np.array([[1,2],[3,4]])
print(np.array_equal(a1,a2))


from os import MFD_HUGE_512MB
#most frequent value and its index
#numpy.bincount- to count occurrences of each value
#numpy.argmax-to find the value with max count
#numpy.where-to find the indices of the most frequent value
x = np.array([1,2,3,4,5,1,2,1,1,1])
print("Original array:",x)
mf=np.bincount(x).argmax()
print("Most frequent value",mf)
print("Index of most frequent value",np.where(x==mf))
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3, ])
print("Original array:",y)
mf=np.bincount(y).argmax()
print("Most frequent value",mf)
print("Index of most frequent value",np.where(y==mf))



#sum of all elements, row-wise, column-wise
gfg=np.matrix('[4,1,9;12,3,1;4,5,6]')
s=np.sum(gfg)
print("Sum of all elements",s);
sr=np.sum(gfg,axis=1) #row wise
print("Sum of all elements row-wise\n",sr)
sc=np.sum(gfg,axis=0) #column wise
print("Sum of all elements column-wise",sc)


n_array = np.array([[55, 25, 15],[30, 44, 2],[11, 45, 77]])
#sum of diagonal elements
sd=np.trace(n_array)
print("Sum of diagonal elements",sd)
#eigen values of matrix
eval=np.linalg.eigvals(n_array)
print("Eigen values of matrix",eval)
#eigen vectors of matrix
evec=np.linalg.eig(n_array)
print("Eigen vector of matrix\n",evec)
#Inverse of matrix
In=np.linalg.inv(n_array)
print("Inverse of matrix\n",In)
#determinant of matrix
d=np.linalg.det(n_array)
print("Determinant of matrix\n",d)


p1=np.array([[1,2],[3,4]])
q1=np.array([[4,5],[6,7]])
#matrix multiplication
#alternatively, m=p1@q1
m=np.dot(p1,q1)
print("Matrix multiplication\n",m)
#flatten matrix and find covariance
fp1=p1.flatten()
fq1=q1.flatten()
cov=np.cov(fp1,fq1)
print("Covariance\n",cov)
p2=np.array([[1,2],[2,3],[4,5]])
q2=np.array([[4,5,1],[6,7,2]])
m=np.dot(p2,q2)
print("Matrix multiplication\n",m)
fp2=p2.flatten()
fq2=q2.flatten()
cov=np.cov(fp2,fq2)
print("Covariance\n",cov)


import itertools
x=np.array([[2,3,4],[3,2,9]])
y=np.array([[1,5,0],[5,10,3]])
#inner product
ip=np.inner(x,y)
print("Inner product\n",ip)
#outer product
op=np.outer(x,y)
print("Outer product\n",op)
#cartesian product
#first flatten
xflt=x.flatten()
yflt=y.flatten()
cp=np.transpose([np.tile(xflt,len(yflt)),np.repeat(yflt,len(xflt))])
#alternative
cp1=np.array(list(itertools.product(xflt,yflt)))
print("Cartesian product\n",cp)
print("Cartesian product\n",cp1)


arr=np.array([[1,-2,3],[-4,5,-6]])


#absolute of every element of array
ab=np.abs(arr)
print("Element wise absolute value\n",ab)


#percentile:25,50,75
flt=arr.flatten()
per=np.percentile(flt,[25,50,75])
print("Percentile:complete array-",per)
perr=np.percentile(arr,[25,50,75],axis=0)
print("Percentile:column-wise-",perr)
perc=np.percentile(arr,[25,50,75],axis=1)
print("Percentile:column-wise",perc)

#mean,median,standard deviation
mean=np.mean(flt)
print("Mean:",mean)
median=np.median(flt)
print("Median:",median)
std=np.std(flt)
print("Standard deviation:",std)

meanc=np.mean(arr,axis=0)
meanr=np.mean(arr,axis=1)
print("Mean column-wise:",meanc)
print("Mean row-wise::",meanr)
medianc=np.median(arr,axis=0)
medianr=np.median(arr,axis=1)
print("Median column-wise:",medianc)
print("Median row-wise:",medianr)
stdc=np.std(arr,axis=0)
stdr=np.std(arr,axis=1)
print("Standard deviation column-wise:",stdc)
print("Standard deviation row-wise:",stdr)



a=np.array([-1.8,-1.6,-0.5,0.5,1.6,1.8,3.0])
fl=np.floor(a)
print("Floor values:",fl)
cl=np.ceil(a)
print("Ceil values:",cl)
tr=np.trunc(a)
print("Truncated values:",tr)
r=np.round(a)
print("Rounded values:",r)


#sorting
array=np.array([10,52,62,16,16,54,453])
print("Original array:",array)
s=np.sort(array)
print("Sorted array:",s)
#argsort: gives actual index of element in the sorted array
sind=np.argsort(array)
print("Indices of sorted array:",sind)
#4 smallest elements
sm=np.partition(array,4)[:4]
print("4 smallest elements:",sm)
#5 largest elements
lg=np.partition(array,-5)[-5:]
print("5 largest elements:",lg)


#finding integer elements only
array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
int_mask=array==np.floor(array)
int_array=array[int_mask]
print(int_array)
#finding float elements only
float_mask=array!=np.floor(array)
float_array=array[float_mask]
print(float_array)




