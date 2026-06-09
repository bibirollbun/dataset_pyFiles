import numpy as np
import matplotlib.pyplot as plt

def visualize_grid(grid, title="Grid"):
    """
    Visualize a 2D grid using matplotlib.
    """
    plt.imshow(grid, cmap='tab20', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

def apply_transformation(grid):
    """
    Apply a transformation to the input grid.
    For demonstration, we'll implement a simple rule:
    - Flip the grid horizontally.
    - Replace all instances of color 2 with color 3.
    """
    transformed = np.fliplr(grid)
    transformed[transformed == 2] = 3
    return transformed

def main():
    # Example input grid (using integers to represent colors)
    input_grid = np.array([
        [0, 1, 2],
        [2, 1, 0],
        [1, 0, 2]
    ])

    # Expected output grid after transformation
    expected_output = np.array([
        [3, 1, 0],
        [0, 1, 3],
        [2, 0, 3]
    ])

    # Visualize input grid
    visualize_grid(input_grid, title="Input Grid")

    # Apply transformation
    output_grid = apply_transformation(input_grid)

    # Visualize output grid
    visualize_grid(output_grid, title="Output Grid")

    # Check if the transformation matches the expected output
    if np.array_equal(output_grid, expected_output):
        print("✅ Transformation successful! Output matches expected result.")
    else:
        print("❌ Transformation failed. Output does not match expected result.")

if __name__ == "__main__":
    main()
    

