import numpy as np

# broadcasting
# array_2d = np.array([[1, 2, 3], [4, 5, 6]])
# scalar = 10
# result = array_2d + scalar
# print("Array + Scalar:\n", result)

# This process simplifies element-wise operations and improves efficiency by avoiding explicit loops.

# Working of Broadcasting in NumPy
# Broadcasting applies specific rules to find whether two arrays can be aligned for operations or not that are:

# Check Dimensions: Ensure the arrays have the same number of dimensions or expandable dimensions.
# Dimension Padding: If arrays have different numbers of dimensions the smaller array is left-padded with ones.
# Shape Compatibility: Two dimensions are compatible if they are equal or one of them is 1.


# Broadcasting a Scalar to a 1D Array
# array_1d = np.array([1, 2, 3])
# result = array_1d + 10
# print("Array + 10:\n", result)

# # Broadcasting a 1D Array to a 2D Array
# array_2d = np.array([[1, 2, 3], [4, 5, 6]])
# array_1d = np.array([10, 20, 30])
# result = array_2d + array_1d
# print("2D Array + 1D Array:\n", result)

# Broadcasting a 2D Array to a 3D Array
# array_3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# array_2d = np.array([[10, 20, 30], [40, 50, 60]])
# result = array_3d + array_2d
# print("3D Array + 2D Array:\n", result)

#  Broadcasting in Conditional Operations
# array = np.array([[1, 2, 3], [4, 5, 6]])
# mask = array > 3
# result = array[mask]
# print("Elements greater than 3:\n", result)

# ages = np.array([12, 24, 35, 45, 60, 72])
# age_group = np.array(["Adult", "Minor"])
# result = np.where(ages > 18, age_group[0], age_group[1])
# print(result)

# # Using Broadcasting for Matrix Multiplication

# matrix = np.array([[1, 2], [3, 4]])
# vector = np.array([10, 20])
# result = matrix * vector
# print(result)

# Scaling Data with Broadcasting

# food_data = np.array([
#     [0.8, 2.9, 3.9, 4.5, 5.1, 6.2, 7.4, 8.0, 9.6, 10.5],
#     [52.4, 23.6, 36.5, 45.2, 51.3, 62.1, 74.3, 80.0, 96.5, 105.4],
#     [55.2, 31.7, 23.9, 54.1, 61.2, 72.5, 83.1, 91.0, 101.2, 110.3],
#     [14.4, 11, 4.9, 9.5, 10.2, 12.3, 13.4, 14.0, 15.6, 16.5]]
# )
# caloric_values = np.array([9, 4, 4, 4, 4, 4, 4, 4, 4, 4])
# caloric_matrix = caloric_values
# calorie_breakdown = food_data * caloric_matrix
# print(calorie_breakdown)


# Adjusting Temperature Data Across Multiple Locations

# temperatures = np.array([
#     [30, 32, 34, 33, 31],  
#     [25, 27, 29, 28, 26], 
#     [20, 22, 24, 23, 21]  
# ])

# corrections = np.array([1.5, -0.5, 2.0])

# adjusted_temperatures = temperatures + corrections[:, np.newaxis]
# print(adjusted_temperatures)

# Normalizing Image Data
# image = np.array([
#     [100, 120, 130],
#     [90, 110, 140],
#     [80, 100, 120]
# ])

# mean = image.mean(axis=0)   
# std = image.std(axis=0)    

# normalized_image = (image - mean) / std
# print(normalized_image)


# Centering Data in Machine Learning

data = np.array([
    [10, 20],
    [15, 25],
    [20, 30]
])

feature_mean = data.mean(axis=0)
print(feature_mean)
centered_data = data - feature_mean
print(centered_data)