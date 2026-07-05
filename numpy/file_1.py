import numpy as np  
nums_list = [10,12,14,16,20]  
nums_array = np.array(nums_list)  
type(nums_array)
# op
#numpy.ndarray


row1 = [10,12,13]  
row2 = [45,32,16]  
row3 = [45,32,16]  
  
nums_2d = np.array([row1, row2, row3])  
nums_2d.shape

# op
# (3, 3)

nums_arr = np.arange(5,11)                    
print(nums_arr)
#op
# [ 5  6  7  8  9 10]


nums_arr = np.arange(5,12,2)  
print(nums_arr)
# op
# [ 5  7  9 11] 

ones_array = np.ones(6)  
print(ones_array) 
# op
# [1. 1. 1. 1. 1. 1.]

ones_array = np.ones((6,4))  
print(ones_array) 
# op
# [[1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]
#  [1. 1. 1. 1.]]


zeros_array = np.zeros(6)  
print(zeros_array) 
# op
# [0. 0. 0. 0. 0. 0.]

zeros_array = np.zeros((6,4))  
print(zeros_array) 
# op
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]


# The eye() method is used to create an identity matrix in the 
# form of a 2-dimensional NumPy array
eyes_array = np.eye(5)  
print(eyes_array)
# op
# [[1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1.]]


uniform_random = np.random.rand(4, 5)  
print(uniform_random)
# op
# [0.36728531 0.25376281 0.05039624 0.96432236 0.08579293]
#  [0.29194804 0.93016399 0.88781312 0.50209692 0.63069239]
#  [0.99952044 0.44384871 0.46041845 0.10246553 0.53461098]
#  [0.75817916 0.36505441 0.01683344 0.9887365  0.21490949]]

integer_random = np.random.randint(10, 50, 5)  
print(integer_random)
# op
# [25 49 21 35 17]



# A NumPy array can be reshaped using the reshape() function. 
# It is important to mention that the product of the rows and 
# columns in the reshaped array must be equal to the product 
# of rows and columns in the original array
uniform_random = np.random.rand(4, 6)  
uniform_random = uniform_random.reshape(3, 8)  
print(uniform_random) 
# op
# [[0.37576967 0.5425328  0.56087883 0.35265748 0.19677258 0.65107479  0.63287089  0.70649913]
#  [0.47830882 0.3570451  0.82151482 0.09622735 0.1269332  0.65866216  0.31875221  0.91781242]
#  [0.89785438 0.47306848 0.58350797 0.4604004  0.62352155 0.88064432  0.0859386  0.51918485]]


s = np.arange(1,11)  
print(s) 
# op
# [ 1  2  3  4  5  6  7  8  9 10]


print(s[1:9])  
# [2 3 4 5 6 7 8 9]

print(s[:5])  
print(s[5:]) 
# [1 2 3 4 5]
# [ 6  7  8  9 10]


row1 = [10,12,13]  
row2 = [45,32,16]  
row3 = [45,32,16]  
  
nums_2d = np.array([row1, row2, row3])  
print(nums_2d[:2,:]) 
# [[10 12 13]
#  [45 32 16]]


row1 = [10,12,13]  
row2 = [45,32,16]  
row3 = [45,32,16]  
  
nums_2d = np.array([row1, row2, row3])  
print(nums_2d[:,:2]) 
# op
# [[10 12]
#  [45 32]
#  [45 32]]


row1 = [10,12,13]  
row2 = [45,32,16]  
row3 = [45,32,16]  
  
nums_2d = np.array([row1, row2, row3])  
print(nums_2d[1:,1:]) 
# op
# [[32 16]
#  [32 16]]


nums = [10,20,30,40,50]  
np_sqr = np.sqrt(nums)  
print(np_sqr)
# op
# [3.16227766 4.47213595 5.47722558 6.32455532 7.07106781]


nums = [10,20,30,40,50]  
np_log = np.log(nums)  
print(np_log )
# op
# [2.30258509 2.99573227 3.40119738 3.68887945 3.91202301]


nums = [10,20,30,40,50]  
np_exp = np.exp(nums)  
print(np_exp) 
# op
# [2.20264658e+04 4.85165195e+08 1.06864746e+13 2.35385267e+17
#  5.18470553e+21]


# Finding Sine and Cosine

nums = [10,20,30,40,50]  
np_sine = np.sin(nums)  
print(np_sine)  
  
nums = [10,20,30,40,50]  
np_cos = np.cos(nums)  
print(np_cos)
# op
# [-0.54402111  0.91294525 -0.98803162  0.74511316 -0.26237485]
# [-0.83907153  0.40808206  0.15425145 -0.66693806  0.96496603]


# Finding Matrix Dot Product
A = np.random.randn(4,5)  
  
B = np.random.randn(5,4)  
  
Z =  np.dot(A,B)  
  
print(Z)  
# op
# [[ 1.43837722 -4.74991285  1.42127048 -0.41569506]
#  [-1.64613809  5.79380984 -1.33542482  1.53201023]
#  [-1.31518878  0.72397674 -2.01300047  0.61651047]
#  [-1.36765444  3.83694475 -0.56382045  0.21757162]]


# Element-wise Matrix Multiplication

row1 = [10,12,13]  
row2 = [45,32,16]  
row3 = [45,32,16]  
  
nums_2d = np.array([row1, row2, row3])  
multiply = np.multiply(nums_2d, nums_2d)  
print(multiply)
# op
# [[ 100  144  169]
#  [2025 1024  256]
#  [2025 1024  256]]


# Finding Matrix Inverse

row1 = [1,2,3]  
row2 = [4,5,6]  
row3 = [7,8,9]  
  
nums_2d = np.array([row1, row2, row3])  
  
inverse = np.linalg.inv(nums_2d)  
print(inverse) 
# op
# [[ 3.15251974e+15 -6.30503948e+15  3.15251974e+15]
#  [-6.30503948e+15  1.26100790e+16 -6.30503948e+15]
#  [ 3.15251974e+15 -6.30503948e+15  3.15251974e+15]]



#  Finding Matrix Determinant

row1 = [1,2,3]    
row2 = [4,5,6]    
row3 = [7,8,9]    
    
nums_2d = np.array([row1, row2, row3])    
    
determinant = np.linalg.det(nums_2d)    
print(determinant)
# op
# -9.51619735392994e-16


# Finding Matrix Trace 
# The trace of a matrix refers to the sum of all the elements 
# along the diagonal of a matrix.

row1 = [1,2,3]  
row2 = [4,5,6]  
row3 = [7,8,9]  
  
nums_2d = np.array([row1, row2, row3])  
  
trace = np.trace(nums_2d)  
print(trace) 
# op
# 15



