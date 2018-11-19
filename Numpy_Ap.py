# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:53:02 2018

@author: A-Bibeka
#numpy examples
"""

#******************************************************************************
#1 Understanding Data Types in Python
# List of int
L=list(range(10))
type(L[0])

#List of strings
L2=[str(c) for c in L]
type(L2[0])


#Hetrogeneous lists
L3=[True,"2",3.0,4]
[type(item) for item in L3]

#Fixed type arrays in Python
import array
L=list(range(10))
A=array.array('i',L)
A

import numpy as np
np.array([1,4,3,5,7])

np.array([1,3,4,6],dtype=np.float32)

#Multidimensional arrays
np.array([range(i,i+3) for i in [2,4,6]])

#Creating arrays

#Arrays of zeros
np.zeros(10,dtype=int)

# create a 3X5 floating point array filled with ones
np.ones((3,5),dtype=float)

# Create a full array filled with 3.1
np.full((3,5),3.1)

# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
np.arange(0,20,2)
np.linspace(0,20,11) #Need to know 11 in advance. np.arange is better

# Create an array of five values evenly spaced between 0 and 1
np.linspace(0,1,5)


# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
np.random.random((3,3))

# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0,1,(3,3))

# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0,10,(3,3))

# Create a 3x3 identity matrix
np.eye(3)

## Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)

#******************************************************************************
#2 The basics of NumPy Arrays

## Numpy Array Attributes
np.random.seed(0) #seed for reproducibility
x1=np.random.randint(10,size=6)
x2=np.random.randint(10,size=(3,4))
x3=np.random.randint(10,size=(3,4,5))

print("x3 ndim:", x3.ndim) # Num dimesions
print("x3 shape:",x3.shape) #Shape of the array
print("x3 size:",x3.size)  # Number of elements

print("x3 dtype:",x3.dtype)
print("itemsize:",x3.itemsize,"bytes") #Bytes of each element in the array
print("nbytes:",x3.nbytes,"bytes") #Total number of bytes in the array = (3*4*5)*8 bytes
###The following should be true
x3.nbytes==(x3.itemsize*x3.size)

## Array Indexing : Accessing Single Elements
x1[0]=3.14 # This will be truncated to int
x1[0] 

x1[0:6:2] # Returns 0th 2th, 4th element
x1
x1[0:6:1] # Returns all the elements

## 1-dimensional subarrays
x1[::2] # Every other element
x1[1::2] #Every other element starting at 1


x1[::-1] #all elements reversed
x1[5::-2] #reversed every other element from index 5

## Multi-dimensional subarrays
x2
x2[:2,:3] # 2 row, 3 col
x2[:3,::2] # all row, every other col

x2[::-1,::-1] # Reversing the subarray dimensions

### Accessing aray rows and columns
x2[:,0] # first col of all rows
x2[0,:] #first row of all col == x2[0]
x2[0]

## Subarrays as no-copy views
x2
x2_sub=x2[:2,:2]
x2_sub
x2_sub[0,0]=999 # Now if we modify this subarray, we'll see that the orignal array is changed
x2
x2_sub

# Creating copy of arrays
x2_sub_copy=x2[:2,:2].copy()
print(x2_sub_copy)

x2_sub_copy[0, 0] = 42
print(x2_sub_copy)
print(x2)


# Reshpaing of Array
buf=np.arange(1,10)
grid=buf.reshape(3,3)
grid
grid[0,0]=999
buf # np.array by default is no copy. So the orignal gets modified also


x=np.array([1,2,3])
# Row vector via reshape
x.reshape(1,3)
# Row vector va newaxis
x[np.newaxis,:]
#Col vector via reshape
x.reshape((3,1))
#Col vector via newaxis
x[:,np.newaxis]


# Array Concatenation and Splitting
x=np.array([1,2,3])
y=np.array([3,2,1])
np.concatenate([x,y])
z=np.array([99,99,99])
np.concatenate([x,y,z])

grid=np.array([[1,2,3],
                [4,5,6]])
np.concatenate([grid,grid]) #Concatenate along the first axis

np.concatenate([grid,grid],axis=1) #Concatenate along the second axis 

# Clearer to use np.vstack and np.hstack
x=np.array([1,2,3])
grid=np.array([[9,8,7],
               [6,5,4]])
# Vertically stack the arrays
np.vstack([x,grid])

# horizontally stack the arrays
np.hstack([x[:2,np.newaxis],grid]) # Use the 1st of element to match dimension and reshape using newaxis


# Splitting of arrays
x=[1,2,3,99,99,3,2,1]
x1,x2,x3=np.split(x,[3,5])
print(x1,x2,x3)


grid=np.arange(16).reshape((4,4))
grid
upper,lower=np.vsplit(grid,[2])
print(upper)
print(lower)

left,right=np.hsplit(grid,[2])
print(left)
print(right)

#******************************************************************************
#1 Computation on NumPy Arrays: Universal Functions

