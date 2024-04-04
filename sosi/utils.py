from numba import jit
#jit(nopython=True)
def generate_lookup_table_3d(max_r):
    """
    Generate a lookup table for the Von Neumann neighborhood for distances up to max_r.

    Returns:
    - List of lists: The lookup table for the Von Neumann neighborhood.
    """
    lookup_table = []
    for r in range(1, max_r + 1):
        neighborhood = []
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                for z in range(-r, r + 1):
                    if abs(x) + abs(y) + abs(z) <= r:
                        neighborhood.append((x, y, z))
        lookup_table.append(neighborhood)
    return lookup_table

#@jit(nopython=True)
def von_neumann_neighborhood_3d(center, offsets):
    """
    Generate the von Neumann neighborhood for a given center point in 3D space
    with a variable range r using a lookup table.

    Args:
    - center (tuple): The coordinates of the center point in 3D space (x, y, z).
    - r (int): The range for the neighborhood.
    - x_max (int): Upper bound for x coordinate.
    - y_max (int): Upper bound for y coordinate.
    - z_max (int): Upper bound for z coordinate.
    - lookup_table (list of lists): Precomputed lookup table for the Von Neumann neighborhood.

    Returns:
    - List of tuples: The coordinates of the cells in the von Neumann neighborhood.
    """
    x_center, y_center, z_center = center

    neighborhood = []

    for offset in offsets:
        new_x, new_y, new_z = x_center + offset[0], y_center + offset[1], z_center + offset[2]
        if 0 <= new_x < 400 and 0 <= new_y < 400 and 0 <= new_z < 50:
            neighborhood.append((new_x, new_y, new_z))

    return neighborhood

@jit(nopython=True)
def von_neumann_neighborhood_r1(center):
    x, y, z = center
    
    neighbors = []
    # Define offsets for Von Neumann neighborhood
    for dx, dy, dz in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]:
        new_x, new_y, new_z = x + dx, y + dy, z + dz
        if 0 <= new_x < 400 and 0 <= new_y < 400 and 0 <= new_z < 50:
            neighbors.append((new_x, new_y, new_z))
    return neighbors