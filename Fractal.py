import numpy as np
import matplotlib.pyplot as plt
import random

def create_random_sierpinski_carpet(order):
    size = 3**order
    carpet = np.ones((size, size), dtype=int)
    for o in range(order):
        step = 3**(order - 1 - o)
        sub_grid_size = 3**o
        for r_grid in range(sub_grid_size):
            for c_grid in range(sub_grid_size):
                # Find the index of the current 3x3 subgrid
                sub_r = r_grid * 3 * step
                sub_c = c_grid * 3 * step
                
                # Choose a random index between 0 and 8 to remove a square
                idx_to_remove = random.randint(0, 8)
                
                # Convert index to local coordinates within the 3x3 subgrid
                local_r = (idx_to_remove // 3)
                local_c = (idx_to_remove % 3)
                
                # Absolute global coordinates of the square to remove
                abs_r = sub_r + local_r * step
                abs_c = sub_c + local_c * step
                
                # Remove the corresponding square (set to 0 - white)
                carpet[abs_r:abs_r+step, abs_c:abs_c+step] = 0
                
    return carpet

def calculate_fractal_dimension_numeric(carpet):
    """Numerically calculates the fractal dimension using a simplified box-counting approach."""
    # N(s) is the total number of black squares (pixels with value 1)
    number_of_black_squares = np.sum(carpet)
    # scale_factor = L = size of the grid
    size = carpet.shape[0]
    
    # Fractal dimension D = log(N) / log(scale_factor)
    # This is a reasonable approximation for fixed-scale objects like this
    D = np.log(number_of_black_squares) / np.log(size)
    return D

# --- SIMULATION CONFIGURATION ---
num_simulations = 500  # Run 500 times as requested
order = 4               # Carpet size: 3^4 = 81x81 pixels. Adequate for numerical calculation.
size = 3**order
# 1. GENERATE FRACTAL DIMENSION STATISTICS
fractal_dimensions = []
for i in range(num_simulations):
    carpet = create_random_sierpinski_carpet(order)
    dim = calculate_fractal_dimension_numeric(carpet)
    fractal_dimensions.append(dim)
    # Progress print every 100 simulations
    if (i+1) % 100 == 0:
        print(f"Completed {i+1} simulations...")

# Calculate the mean and standard deviation of the 500 dimensions
mean_dimension = np.mean(fractal_dimensions)
std_dimension = np.std(fractal_dimensions)
print(f"Calculated mean fractal dimension (D_mean): {mean_dimension:.5f}")
print(f"Standard deviation of the dimension: {std_dimension:.5f}")

# 2. GENERATE AN EXAMPLE IMAGE OF A RANDOM CARPET
example_carpet = create_random_sierpinski_carpet(order)

# Save the mean statistics to show in the image title
plt.figure(figsize=(7, 7))
plt.imshow(example_carpet, cmap='gray', interpolation='nearest')
plt.axis('off')

# Save and display
output_path = 'random_sierpinski_carpet_numeric.png'
plt.savefig(output_path)
plt.show()