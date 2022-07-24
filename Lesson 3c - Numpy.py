#!/usr/bin/env python
# coding: utf-8

# ## Numpy
# 
# Numpy is a numerical library that makes it easy to work with big arrays and matrices.
# 
# It can be used to make fast arithmetic operations with matrixes. Pandas and Numpy are usually used together, as Pandas builds on NumPy functionality to work with DataFrames.
# 
# Since Pandas is designed to work with Numpy, almost any Numpy function will work with Pandas Series and DataFrames, lets run some examples.

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


# Create a container for a pseudo random generator
rng = np.random.RandomState()
# Create a Pandas Series from our random generator
series = pd.Series(rng.randint(0, 10, 4))
series


# In[7]:


# Create a Pandas Dataframe with our Numpy random generator
# Integer numbers between 0 and 10, 3 rows and 4 columns
df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                 columns=['A', 'B', 'C', 'D'])
df


# We can use a Numpy function on these Pandas objects and still keep our indexes for order

# In[8]:


# Calculate e^x, where x is every element of our array
np.exp(series)


# In[9]:


# Calculate sin() of every value in the DataFrame multiplied by pi and divided by 4
np.sin(df * np.pi / 4)


# ### The Numpy Array
# Numpy provides mutidimentional arrays, with high efficiency and designed for scientific calculations.
# 
# An array is similar to a list in Python and can be created from a list.
# 
# Array have useful atributes we can use. Lets start by defining three random arrays, a single dimension, a two dimension and a tri dimensional array. We will use Numpy random number generator.

# In[10]:


# Import our Numpy package
import numpy as np
np.random.seed(0) #this will generate the same random arrays every time

x1 = np.random.randint(10, size=6) # one dimension
x2 = np.random.randint(10, size=(3, 4)) # two dimensions
x3 = np.random.randint(10, size=(3, 4, 5)) # tri dimensional array


# All arrays have the `ndim` (number of dimensions) attribute, `shape` the size of each dimension, and `size` the total array size

# In[11]:


print("x3 ndim:", x3.ndim)
print("x3 sahpe:", x3.shape)
print("x3 size:", x3.size)


# Other attributes for arrays are `itemsize` shows the byte size of every element in the array, and `nbytes` shows the total bytes size of the array:

# In[12]:


print("x3 itemsize:", x3.itemsize, "bytes")
print("x3 nbytes", x3.nbytes, "bytes")


# #### Optional Exercise
# - Create a 3x3x3 array that contains all random numbers.
# - Using the previous array, square all the numbers in the array.

# ### Creating arrays with Numpy methods
# Specially for largers arrays we can use the more efficient methods from Numpy.

# In[13]:


# Create a length 10 integer array filled with zeros
np.zeros(10, dtype=int)


# In[14]:


# Create a 3x5 float array filled with ones
np.ones((3, 5), dtype=float)


# In[15]:


# Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)


# In[16]:


# Create an array filled with a linear sequence
# Start at 0, end at 20, step size 2
np.arange(0, 20, 2)


# In[17]:


# Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)


# In[18]:


# Create a 3x3 array of uniformly distributed random values between 0 and 1
np.random.random((3, 3))


# In[19]:


# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))


# In[20]:


# Create a 3x3 array of random integer in the interval [0, 10]
np.random.randint(0, 10, (3, 3))


# In[21]:


# Create a 3x3 identity matrix
np.eye(3)


# In[22]:


# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)


# #### Array indexing
# Similar to Python lists we can access individual elements in the array. For single dimensional arrays we can use the indexing format using `[]`

# In[23]:


x1


# In[24]:


x1[0]


# In[25]:


x1[4]


# In[26]:


x1[-6]


# We can use a similar logic for multi dimensional arrays

# In[27]:


x2


# In[28]:


x2[0, 3] # Access row 0, column index 3


# In[29]:


x2[2, -1] # Access row index 2, column index -1


# We can use the same logic to change values using array indexing

# In[30]:


x2[2, -1] = 2


# #### Optional Exercise
# - Create a new array using `arange` this array should contain a total of 4 values.
# - Select only values in index 2 and index 4.
# - Change the value of index 3 to 99

# #### Sub arrays (slicing)
# We can also use a similar syntax to Python list slicing to access only parts of the array. The syntax goes as follows:
# 
# `x[start:stop:step]`
# Where default start value = 0, stop is the non inclusive stop index, and step the number of items we want to count

# In[31]:


x = np.arange(10)
x


# In[32]:


x[:5] # first five elements


# In[33]:


x[5:] # elements starting from the fifth index


# In[34]:


x[4:7] # elements between 4 and non inclusive 7


# In[35]:


x[::2] # all elements but step size 2


# In[36]:


x[1::2] # elements every two steps, starting from index 1


# We can also use negative step index. In this case the default start and stop values are inverted. This makes it a convenient way to invert an array.

# In[37]:


x[::-1] # reverse the array


# In[38]:


x[5::-2] # inverted array starting from the fifth index at minus two step interval


# We can also select multi dimension sectors of an array. The syntax is similar, with every sector separated by a comma

# In[39]:


x2


# In[40]:


x2[:2, :3] # rows with index 0 and 1, and columns with index 0, 1 and 2


# In[41]:


x2[:3, ::2] # all rows but step size 2


# #### Reshaping arrays
# Another useful operation is reshaping, we can use the `reshape()` method. If we wanted to reshape an array by a 3 x 3 array we can use the following syntax

# In[42]:


grid = np.arange(1, 10).reshape((3, 3))
grid


# For this to work the size of the initial array must match the reshaped array.
# 
# Another common form of reshaping is converting an unidimensional array of rows or columns. We can either use `reshape` or the `nexaxis` keyword inside a slicing operation

# In[43]:


x = np.array([1, 2, 3])

# row vector via reshape
x.reshape(1, 3)


# In[44]:


# row vector via newaxis
x[np.newaxis, :]


# In[45]:


# column vector via reshape
x.reshape((3, 1))


# In[46]:


# column vector via newaxis
x[:, np.newaxis]


# #### Optional exercise
# - Create a new 4x4 array.
# - Select and print only the first two rows.
# - Select and print only the last two columns.
# - Select only the last two rows and first two columns.
# - Reshape the array into a 8x2 array.

# #### Array concatenation and division
# We can also combine multiple arrays into one and vice versa
# 
# Concatenation can be achieved using the `np.concatenate`, `np.vstack` and `np.hstack`. `np.concatenate` takes a list of arrays as its first argument

# In[47]:


x = np.array([1, 2, 3]) # create an array from a list
y = np.array([3, 2, 1]) # create a second array from a list
np.concatenate([x, y]) # array list to concatenate


# In[48]:


# we can concatenate more than one array at a time
z = [99, 88, 77]
print(np.concatenate([x, y, z]))


# We can use the same logic for multidimensional arrays

# In[49]:


grid = np.array([[1, 2, 3],
                [4, 5, 6]])
np.concatenate([grid, grid]) # concatenate on the first axis. Rows


# In[50]:


grid


# In[51]:


# Concatenate on the second axis (zero indexed)
np.concatenate([grid, grid], axis=1) # concatenate on columns


# To work with diferent sized arrays it may be easier to work with `np.vstack` vertical stack, and `np.hstack` horizontal stack

# In[52]:


x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                [6, 5, 4]])
# stack vertically
np.vstack([x, grid])


# In[53]:


# stack horizontally
y = np.array([[99],
              [99]])
np.hstack([grid, y])


# #### Split arrays
# Finally we can split arrays using the `np.split`, `np.hplit` and `np.vsplit`. For each of these we need to pass a list of indexes that divide/split our array

# In[54]:


x = [1, 2, 3, 99, 88, 77, 4, 5, 6]
x1, x2, x3 = np.split(x, [3, 5]) # split at index 3 and 5, non inclusive
print(x1, x2, x3)


# Observe that N division/split points lead to N + 1 sub arrays. Similarly `np.hsplit` and `np.vsplit` can be used

# In[55]:


grid = np.arange(16).reshape((4, 4))


# In[56]:


grid


# In[57]:


upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)


# In[58]:


left, right = np.hsplit(grid, [2])
print(left)
print(right)


# ### Universal Functions
# Next we will look at why Numpy is important for data science and working with arrays.
# 
# The key to making computation with Numpy very fast is using vectorized operations with Numpy, they key to this is using Numpy Universal Functions. 
# 
# Here is a speed comparison between a traditional for loop and a vectorized operation in Numpy using and array that contains a million values.

# In[59]:


# Traditional implementation
import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
        
values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)


# In[60]:


big_array = np.random.randint(1, 100, size=1000000)
get_ipython().run_line_magic('timeit', 'compute_reciprocals(big_array)')


# The code above takes several seconds to run. Lets run it now with a vectorized operation.

# In[61]:


print(compute_reciprocals(values))
print(1.0 / values)


# In[62]:


get_ipython().run_line_magic('timeit', '(1.0 / big_array)')


# We can see that each loop and the consequent total execution time is orders of magnitude faster that the first iteration. Vectorized operations are implemented via ufuncs whose main function is to exectute repeteated operations on values in Numpy arrays.
# 
# Ufuncs (universal functions) can run between scalars and arrays, two arrays, and multi dimensional arrays.

# In[63]:


np.arange(5) / np.arange(1, 6)


# In[64]:


# multidimensional example
x = np.arange(9).reshape((3, 3))
2 ** x


# ### Numpy UFuncs
# #### Array arithmetic
# We have standard addition, substraction, multiplication and division

# In[65]:


x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # floor division


# In[66]:


# unary functions for negation, exponentiation, and modulus
print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)


# All these arithmetic operations are wrappers for Numpy functions
# 
# |Operator|	Equivalent ufunc |	Description |
# |:--------:|:--------|:--------|
# |+ |	np.add	| Addition (e.g., 1 + 1 = 2) |
# |- |	np.subtract | Subtraction (e.g., 3 - 2 = 1)
# |- |	np.negative |	Unary negation (e.g., -2)
# |* |	np.multiply |	Multiplication (e.g., 2 * 3 = 6)
# |/ |	np.divide |	Division (e.g., 3 / 2 = 1.5)
# |// |	np.floor_divide |	Floor division (e.g., 3 // 2 = 1)
# |** |	np.power |	Exponentiation (e.g., 2 ** 3 = 8)
# |% |	np.mod |	Modulus/remainder (e.g., 9 % 4 = 1)

# #### Trigonometric functions
# We will explore trigonometric functions. Lets start by defining an array of angles.

# In[67]:


theta = np.linspace(0, np.pi, 3)


# In[68]:


print("theta      = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))


# In[69]:


# inverse functions
x = [-1, 0, 1]
print("x         = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))


# #### Exponents and logarithms

# In[70]:


x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))


# In[71]:


# log functions
x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))


# #### Agregates
# We can reduce array operations of any ufunc. A reduce repeatedly applies a given operation to the elements of the array until a single result remains.
# 
# Calling `reduce` on the `add` ufunc results in the sum of all elements in the array

# In[72]:


x = np.arange(1, 6)
np.add.reduce(x)


# In[73]:


x


# In[74]:


# calling reduce on multiply
# results in the product of all array elements
np.multiply.reduce(x)


# In[75]:


# if we would like to store all the intermediate results
# we can use accumulate instead
np.add.accumulate(x)


# In[76]:


np.multiply.accumulate(x)


# ### Exercises for participation credit
# 1. Create a new random integer array with Numpy functions. Use a random seed to guarantee the same array every time. Array must be of size (20, 5).
# 2. Calculate the average value of the second column of the array created in exercise 1. Calculate the sum of all elements in columns 3 and 4. You can use built in ufuncs and indexing.
# 
# We will now use a simple IoT readings dataset for this exercise. This dataset contains IoT temperature readings, where the reading was made (room) and location (reading was outside or inside a room).
# 
# 3. Open the data file `IOT-temp.csv` inside the `iotdata-compressed` file, you will need some program like 7zip or Winrar to uncompress this `iotdata-compressed` folder. Open this file with Pandas and create a new Dataframe with this data. Show the first 10 rows of data. Show the last 10 rows of data. Show data points for index 300 to 350.
# 4. Create a new dataframe from index 1000 to 2000. Rename column `out/in` to `location`. Rename colum `room_id/id` to `room` How many of these readings were made outside? How many were inside?
# 5. Split your new dataframe into readings that were made inside and readings that were made outside. Create a new dataframe with only inside readings. Create a new dataframe only with outside readings. Print the mean value of the temperature for both new dataframes.
# 6. Save both dataframes created in exercise 5 to `csv` files.
# 
# More information about the dataset for exercises 3 and 4 in [this link](https://www.kaggle.com/datasets/atulanandjha/temperature-readings-iot-devices)

# In[85]:


rng.randint(0, 10, (20, 5))


# In[86]:


rad = rng.randint(0, 10,(20, 5))


# In[90]:


rad[:10][:2]


# In[89]:


np.add.reduce(rad[:,[2]])


# In[92]:


y = np.add.accumulate(rad[:,[3,4]])


# In[93]:


np.add.reduce(y)


# In[96]:


X = np.random.randint(0,101,(100,50))
X


# In[97]:


columns_average = X.mean(axis=0)


# In[98]:


columns_average


# In[99]:


np.mean(rad[:,[2]])


# In[16]:


import pandas as pd 

data = pd.read_csv('/Users/penelopesayago/Documents/IOT-temp.csv')
data.tail(10)


# In[17]:


data.head(10)


# In[18]:


data[300:350]


# In[19]:


new_data = data[1000:2000]
new_data


# In[21]:


new_data.columns


# In[23]:


new_data.rename(columns={'room_id/id':'room', 'out/in':'location'}, inplace=True )


# In[24]:


new_data.columns


# In[26]:


new_data.location.value_counts()


# In[28]:


out_data = new_data[new_data.location == 'Out']

out_data.head()


# In[29]:


in_data = new_data[new_data.location == 'In']
in_data.head()


# In[30]:


out_data.describe()


# In[31]:


in_data.describe()


# In[32]:


in_data.to_csv('Inside_data.csv')


# In[33]:


out_data.to_csv('Outside_data.csv')

