from numba import jit

@jit(nopython=True)
def von_neumann_neighborhood_3d(center, r, x_max, y_max, z_max):
    """
    Generate the von Neumann neighborhood for a given center point in 3D space
    with a variable range r.

    Args:
    - center (tuple): The coordinates of the center point in 3D space (x, y, z).
    - r (int): The range for the neighborhood.
    - x_max (int): Upper bound for x coordinate.
    - y_max (int): Upper bound for y coordinate.
    - z_max (int): Upper bound for z coordinate.

    Returns:
    - List of tuples: The coordinates of the cells in the von Neumann neighborhood.
    """
    x_center, y_center, z_center = center

    neighborhood = []

    for x in range(max(0, x_center - r), min(x_max, x_center + r + 1)):
        for y in range(max(0, y_center - r), min(y_max, y_center + r + 1)):
            for z in range(max(0, z_center - r), min(z_max, z_center + r + 1)):
                # Check if the cell is within the von Neumann neighborhood
                if (abs(x - x_center) + abs(y - y_center) + abs(z - z_center)) <= r:
                    neighborhood.append((x, y, z))

    return neighborhood
