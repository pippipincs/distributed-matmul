import numpy as np

# Define the dimensions of the matrix
rows = 5
cols = 5000

# Generate a matrix with random integers between -10 and 10
matrix = np.random.randint(-10, 11, size=(rows, cols))

# Print the matrix (optional, as it will be very large)
# print(matrix)
output_filename = "matrixB.txt"

np.savetxt(output_filename, matrix, fmt="%d", delimiter="\t")

print(f"Matrix successfully saved to '{output_filename}'")
# You can also check the shape of the matrix
print("Matrix shape:", matrix.shape)

# And verify the range of values (e.g., min and max)
print("Minimum value:", matrix.min())
print("Maximum value:", matrix.max())