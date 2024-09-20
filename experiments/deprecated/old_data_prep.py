import pandas as pd
import numpy as np
import gc

filenames = ['random_1', 'random_2', 'random_3', 'random_4', 'random_5', 'random_6', 'random_7', 'random_8', 'random_9', 'random_10', 'random_11', 'random_12', 'random_13', 'random_14', 'random_15', 'random_16', 'random_17', 'random_18', 'random_19', 'random_20', 'clustered_1', 'clustered_2', 'clustered_3', 'clustered_4', 'clustered_5', 'clustered_6', 'clustered_7', 'clustered_8', 'clustered_9', 'clustered_10', 'clustered_11', 'clustered_12', 'clustered_13', 'clustered_14', 'clustered_15', 'clustered_16', 'clustered_17', 'clustered_18', 'clustered_19', 'clustered_20']
path = "../blossom/hpc/outputs2/"
rs = [1, 2, 3, 4, 5]
init_locs = ['R1', 'R1', 'R1', 'R1', 'R1', 'R1','R1', 'R1', 'R1', 'R1', 'R1', 'R1','R1', 'R1', 'R1', 'R1', 'R1', 'R1','R1', 'R1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1']
init_seed  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
sample_times = [252, 415, 310, 192, 241, 308, 338, 231, 241, 233, 241, 330, 246, 246, 161, 327, 158, 218, 291, 431, 206, 185, 307, 258, 287, 244, 335, 291, 151, 206, 167, 356, 208, 224, 271, 181, 179, 279, 305, 246]
x_max = 400
y_max = 400

### Sample Simulations
#
#
###
def retrieve_ids_per_sample_von_neumann(points, data, r):
    samples =  [ [] for _ in range(len(points)) ]
    for ind, row in data.iterrows():
        for index, point in enumerate(points):
            x1, y1 = point
            x2 = row.x
            y2 = row.y

            if (abs(x1 - x2) % 400 + abs(y1 - y2) % 400) <= r:
                samples[index].append(ind)
    return samples

def wageningen_w():
    return [(50, 50), (50, 150), (50, 250), (50, 350), (125, 150), (175, 250), (275, 150), (225, 250), (350, 50), (350, 150), (350, 250), (350, 350)]

def systematic_regular(no_of_samples, x, y):
    # Calculate the dimensions of the regular grid
    rows = int(np.sqrt(no_of_samples))
    cols = int(np.ceil(no_of_samples / rows))

    # Calculate the padding to ensure points are equally spaced
    padding_x = (x - (rows - 1)) // rows
    padding_y = (y - (cols - 1)) // cols

    # Initialize points list
    points = []

    # Select equally spaced points with padding
    for i in range(rows):
        for j in range(cols):
            points.append((int((i * (padding_x + 1)) + ((padding_x + 1)/2)), int((j * (padding_y + 1) + ((padding_y + 1)/2)))))

    return points[:no_of_samples]

print('Soil Sample Sims')
selected_points_w = wageningen_w()
sample_counts_w = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'diameter','sample_id', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
no_of_samples = 16
selected_points_reg = systematic_regular(no_of_samples, x_max, y_max)
sample_counts_reg = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'diameter','sample_id', '0', '1', '2', '3', '4', '5', '6', '7', '8'])

for idx, filename in enumerate(filenames):  
    df = pd.read_csv(path + filename + ".csv")
    data = df[df["tick"] == sample_times[idx]]
    data = data.reset_index(drop=True)
    
    for r in rs:
        samples_w = retrieve_ids_per_sample_von_neumann(selected_points_w, data, r)
        samples_reg = retrieve_ids_per_sample_von_neumann(selected_points_reg, data, r)
    
        sample_site_counts_w = []
        for i, sample in enumerate(samples_w):
            df2 = data.iloc[sample]
            counts = df2['type'].value_counts().reindex(range(len(df["type"].unique())), fill_value=0)
            sample_counts_w = pd.concat([sample_counts_w, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'diameter': [r], 'sample_id': [i], '0': [counts[0]], '1': [counts[1]], '2': [counts[2]], '3': [counts[3]], '4': [counts[4]], '5': [counts[5]], '6': [counts[6]], '7': [counts[7]], '8': [counts[8]]})])
        
        sample_site_counts_reg = []
        for i, sample in enumerate(samples_reg):
            df2 = data.iloc[sample]
            counts = df2['type'].value_counts().reindex(range(len(df["type"].unique())), fill_value=0)
            sample_counts_reg = pd.concat([sample_counts_reg, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'diameter': [r], 'sample_id': [i], '0': [counts[0]], '1': [counts[1]], '2': [counts[2]], '3': [counts[3]], '4': [counts[4]], '5': [counts[5]], '6': [counts[6]], '7': [counts[7]], '8': [counts[8]]})])
    del df
    del data
    gc.collect()
            
sample_counts_w.to_csv('outputs/sample_counts_w.csv', index=False)
sample_counts_reg.to_csv('outputs/sample_counts_reg.csv', index=False)           

### Abundance
#   Baseline
#
###
print('Abundance prep')
abundances_df = pd.DataFrame(columns=['filename', 'init_locs', 'seed', '0', '1', '2', '3', '4', '5', '6', '7', '8'])

for idx, filename in enumerate(filenames):  
    df = pd.read_csv(path + filename + ".csv")
    data = df[df["tick"] == sample_times[idx]]
    data = data.reset_index(drop=True)
    counts = data['type'].value_counts().reindex(range(len(df["type"].unique())), fill_value=0)
    
    counts /= 20000
    abundances_df = pd.concat([abundances_df, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], '0': counts[0], '1': counts[1], '2': counts[2], '3': counts[3], '4': counts[4], '5': counts[5], '6': counts[6], '7': counts[7], '8': counts[8]})])

    del df
    del data
    gc.collect()
    
abundances_df.to_csv('outputs/baseline_abundances.csv', index=False)

### Abundance
#   Estimates W
#
###
def compute_abundance_estimates(data, num_types, r):
    cells = (2*(r**2))+(2*r)+1
    sample_weight = cells * 0.125
    
    norm_abundances_per_sample = []
    for i in range(len(data)):
        sample = data[data['sample_id'] == i]
        abundances_norm = []
        for t in range(num_types):
            abundances_norm.append(float(sample[str(t)].iloc[0]/sample_weight))
        norm_abundances_per_sample.append(abundances_norm)
        
    return norm_abundances_per_sample

data = pd.read_csv('./outputs/sample_counts_w.csv')
abundances_df = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'r', 'sample_id', '0', '1', '2', '3', '4', '5', '6', '7', '8'])

for idx, filename in enumerate(filenames):
    df1 = data[data['filename'] == filename]
    file_abundances = []
    for r in rs:
        df2 = df1[df1['diameter'] == r]         
        plot_abundances_per_sample = compute_abundance_estimates(df2, 9, r)
        file_abundances.append(plot_abundances_per_sample)
            
        for i, sample_abundances in enumerate(plot_abundances_per_sample):
            abundances_df = pd.concat([abundances_df, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'r': r, 'sample_id': i, '0': sample_abundances[0], '1': sample_abundances[1], '2': sample_abundances[2], '3': sample_abundances[3], '4': sample_abundances[4], '5': sample_abundances[5], '6': sample_abundances[6], '7': sample_abundances[7], '8': sample_abundances[8]})])
            
abundances_df.to_csv('outputs/estimated_abundances_w.csv', index=False)

### Abundance
#   Estimates Sys Reg
#
###

data = pd.read_csv('./outputs/sample_counts_reg.csv')
abundances_df = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'r', 'sample_id', '0', '1', '2', '3', '4', '5', '6', '7', '8'])

for idx, filename in enumerate(filenames):
    df1 = data[data['filename'] == filename]
    file_abundances = []
    for r in rs:
        df2 = df1[df1['diameter'] == r]         
        plot_abundances = compute_abundance_estimates(df2, 9, r)
        file_abundances.append(plot_abundances)
            
        for i, sample_abundances in enumerate(plot_abundances):
            abundances_df = pd.concat([abundances_df, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'r': r, 'sample_id': i, '0': sample_abundances[0], '1': sample_abundances[1], '2': sample_abundances[2], '3': sample_abundances[3], '4': sample_abundances[4], '5': sample_abundances[5], '6': sample_abundances[6], '7': sample_abundances[7], '8': sample_abundances[8]})])

abundances_df.to_csv('outputs/estimated_abundances_reg.csv', index=False)

del data
del df1
gc.collect()

### Diversity
#   Baseline
#
###
print('Diversity Prep')
diversity_df = pd.DataFrame(columns=['filename', 'init_locs', 'seed', '0', '1', '2', '3', '4', '5', '6', '7', '8'])

for idx, filename in enumerate(filenames):  
    df = pd.read_csv(path + filename + ".csv")
    data = df[df["tick"] == sample_times[idx]]
    data = data.reset_index(drop=True)
    counts = data['type'].value_counts().reindex(range(len(df["type"].unique())), fill_value=0)
    
    diversity_df = pd.concat([diversity_df, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], '0': counts[0]>0, '1': counts[1]>0, '2': counts[2]>0, '3': counts[3]>0, '4': counts[4]>0, '5': counts[5]>0, '6': counts[6]>0, '7': counts[7]>0, '8': counts[8]>0})])

    del df
    del data
    gc.collect()
diversity_df.to_csv('outputs/baseline_diversity.csv', index=False)

### Diversity
#   Estimates W
#
###
def compute_abundance_estimates(data, num_types, r):
    cells = (2*(r**2))+(2*r)+1
    sample_weight = cells * 0.125
    
    norm_abundances_per_sample = []
    for i in range(len(data)):
        sample = data[data['sample_id'] == i]
        abundances_norm = []
        for t in range(num_types):
            abundances_norm.append(float(sample[str(t)].iloc[0]/sample_weight))
        norm_abundances_per_sample.append(abundances_norm)
        
    return norm_abundances_per_sample

data = pd.read_csv('./outputs/sample_counts_w.csv')
diversity_df = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'r', 'sample_id', '0', '1', '2', '3', '4', '5', '6', '7', '8'])

for idx, filename in enumerate(filenames):
    df1 = data[data['filename'] == filename]
    file_abundances = []
    for r in rs:
        df2 = df1[df1['diameter'] == r]         
        plot_abundances_per_sample = compute_abundance_estimates(df2, 9, r)
        file_abundances.append(plot_abundances_per_sample)
            
        for i, sample_abundances in enumerate(plot_abundances_per_sample):
            diversity_df = pd.concat([diversity_df, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'r': r, 'sample_id': i, '0': sample_abundances[0]>0, '1': sample_abundances[1]>0, '2': sample_abundances[2]>0, '3': sample_abundances[3]>0, '4': sample_abundances[4]>0, '5': sample_abundances[5]>0, '6': sample_abundances[6]>0, '7': sample_abundances[7]>0, '8': sample_abundances[8]>0})])
            
diversity_df.to_csv('outputs/estimated_diversity_w.csv', index=False)

### Diversity
#   Estimates Sys Reg
#
###

data = pd.read_csv('./outputs/sample_counts_reg.csv')
diversity_df = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'r', 'sample_id', '0', '1', '2', '3', '4', '5', '6', '7', '8'])

for idx, filename in enumerate(filenames):
    df1 = data[data['filename'] == filename]
    file_abundances = []
    for r in rs:
        df2 = df1[df1['diameter'] == r]         
        plot_abundances = compute_abundance_estimates(df2, 9, r)
        file_abundances.append(plot_abundances)
            
        for i, sample_abundances in enumerate(plot_abundances):
            diversity_df = pd.concat([diversity_df, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'r': r, 'sample_id': i, '0': sample_abundances[0]>0, '1': sample_abundances[1]>0, '2': sample_abundances[2]>0, '3': sample_abundances[3]>0, '4': sample_abundances[4]>0, '5': sample_abundances[5]>0, '6': sample_abundances[6]>0, '7': sample_abundances[7]>0, '8': sample_abundances[8]>0})])

diversity_df.to_csv('outputs/estimated_diversity_reg.csv', index=False)

### D_index
#   Estimates Baseline
#
###
print('D_index Prep')
def compute_d_index_pairwise(data, grid_dimensions, range_by_type, pseudo_count=1e-5):
    X, Y = grid_dimensions
    num_types = 9
    type_count = np.zeros((X, Y, num_types))  # Counts for each type at each spatial unit
    D_matrix = np.zeros((num_types, num_types))  # 9x9 matrix 
    neighborhood_counts = np.zeros((num_types, num_types))  # Count of neighborhoods for each type pair

    # Populate the 2D grid with agent counts
    for _, agent in data.iterrows():
        x, y, t = agent['x'], agent['y'], agent['type']
        type_count[x][y][t] += 1


    # Calculate the total number of agents for each type in the entire grid
    total_count_by_type = np.sum(type_count, axis=(0, 1))

    def compute_neighborhood_counts(x, y, r):
        """Compute counts of each type in the Von Neumann neighborhood of (x, y) with range r."""
        neighborhood_count = np.zeros(num_types)
        for dx in range(-r, r + 1):
            for dy in range(-r + abs(dx), r - abs(dx) + 1):
                    nx = (x + dx) % X
                    ny = (y + dy) % Y
                    neighborhood_count += type_count[nx][ny]
        return neighborhood_count

    # Calculate the Dissimilarity Index for each spatial unit
    for x in range(X):
        for y in range(Y):
            for t in range(num_types):
                r = range_by_type[t]
                neighborhood_count_at_unit = compute_neighborhood_counts(x, y, r)
                
                for t_prime in range(num_types):
                    if total_count_by_type[t] > 0 and total_count_by_type[t_prime] > 0:
                        prop_t = (neighborhood_count_at_unit[t] + pseudo_count) / total_count_by_type[t]
                        prop_t_prime = (neighborhood_count_at_unit[t_prime] + pseudo_count) / total_count_by_type[t_prime]
                        
                        # Calculate the absolute difference
                        D = abs(prop_t - prop_t_prime)
                        
                        # Accumulate the difference in the respective matrix
                        D_matrix[t][t_prime] += D
                        neighborhood_counts[t][t_prime] += 1
                        if t != t_prime:
                            # Ensure symmetric accumulation
                            D_matrix[t_prime][t] += D
                            neighborhood_counts[t_prime][t] += 1
    
    # Average the D-index values
    for t in range(num_types):
        for t_prime in range(num_types):
            if neighborhood_counts[t][t_prime] > 0:
                # Average the dissimilarity index
                D_matrix[t][t_prime] /= neighborhood_counts[t][t_prime]
            else:
                D_matrix[t][t_prime] = 0  # Handle cases where no neighborhoods were tested

    # Normalize by maximum possible dissimilarity
    max_possible_dissimilarity = np.max(D_matrix)
    if max_possible_dissimilarity > 0:
        D_matrix /= max_possible_dissimilarity  # Normalize to [0, 1]
    else:
        D_matrix = np.zeros_like(D_matrix)  # Handle cases with no data

    return D_matrix

indices = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'type_id', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
d_indices = []
for idx, filename in enumerate(filenames):  
    df = pd.read_csv(path + filename + ".csv")
    data = df[df["tick"] == sample_times[idx]]
    data = data.reset_index(drop=True)
    d_index = compute_d_index_pairwise(data, (x_max, y_max), [1,1,1,1,1,1,1,1,1])
    for type_id, row in enumerate(d_index):
        indices = pd.concat([indices, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'type_id': [type_id], '0': [row[0]], '1': [row[1]], '2': [row[2]], '3': [row[3]], '4': [row[4]], '5': [row[5]], '6': [row[6]], '7': [row[7]], '8': [row[8]]})])
    d_indices.append(d_index)
    
    del df
    del data
    gc.collect()

indices.to_csv('outputs/baseline_d_index.csv', index=False)
arr = np.asarray(d_indices)
np.save('outputs/baseline_d_index.npy', arr)

### D_index
#   Estimates W
#
###
def compute_d_index_pairwise_estimates_sample(counts, num_types, num_samples):
    total_count_by_type = []
    D_matrix = np.zeros((num_samples, num_types, num_types))
    neighborhood_counts = np.zeros((num_samples, num_types, num_types)) 
    
    for idx in range(num_types):
        total_count_by_type.append(counts[str(idx)].sum())
        
    for sample_id in range(len(df2['sample_id'].unique())):
        sample_counts = counts[counts['sample_id'] == sample_id]       
        for t in range(num_types):  
            t_count = sample_counts[str(t)].values[0]
            for t_prime in range(num_types):
                if t_count > 0 and total_count_by_type[t] > 0 and total_count_by_type[t_prime] > 0:
                    prop_t = t_count / total_count_by_type[t]
                    prop_t_prime = sample_counts[str(t_prime)].values[0] / total_count_by_type[t_prime]

                    # Calculate the absolute difference
                    D = abs(prop_t - prop_t_prime)

                    # Accumulate the difference in the respective matrix
                    D_matrix[sample_id][t][t_prime] += D
                    neighborhood_counts[sample_id][t][t_prime] += 1
                    if t != t_prime:
                        # Ensure symmetric accumulation
                        D_matrix[sample_id][t_prime][t] += D
                        neighborhood_counts[sample_id][t_prime][t] += 1
    
    # Average the D-index values
    for s in range(num_samples):
        for t in range(num_types):
            for t_prime in range(num_types):
                if neighborhood_counts[s][t][t_prime] > 0:
                    # Average the dissimilarity index
                    D_matrix[s][t][t_prime] /= neighborhood_counts[s][t][t_prime]
                else:
                    D_matrix[s][t][t_prime] = 0  # Handle cases where no neighborhoods were tested

    # Normalize by maximum possible dissimilarity
    max_possible_dissimilarity = np.max(D_matrix)
    if max_possible_dissimilarity > 0:
        D_matrix /= max_possible_dissimilarity  # Normalize to [0, 1]
    else:
        D_matrix = np.zeros_like(D_matrix)  # Handle cases with no data

    return D_matrix

def compute_d_index_pairwise_estimates_plot(counts, num_types):
    total_count_by_type = []
    D_matrix = np.zeros((num_types, num_types))
    neighborhood_counts = np.zeros((num_types, num_types)) 
    
    for idx in range(num_types):
        total_count_by_type.append(counts[str(idx)].sum())
        
    for sample_id in range(len(df2['sample_id'].unique())):
        sample_counts = counts[counts['sample_id'] == sample_id]       
        for t in range(num_types):  
            t_count = sample_counts[str(t)].values[0]
            for t_prime in range(num_types):
                if t_count > 0 and total_count_by_type[t] > 0 and total_count_by_type[t_prime] > 0:
                    prop_t = t_count / total_count_by_type[t]
                    prop_t_prime = sample_counts[str(t_prime)].values[0] / total_count_by_type[t_prime]

                    # Calculate the absolute difference
                    D = abs(prop_t - prop_t_prime)

                    # Accumulate the difference in the respective matrix
                    D_matrix[t][t_prime] += D
                    neighborhood_counts[t][t_prime] += 1
                    if t != t_prime:
                        # Ensure symmetric accumulation
                        D_matrix[t_prime][t] += D
                        neighborhood_counts[t_prime][t] += 1
    
    # Average the D-index values
    for t in range(num_types):
        for t_prime in range(num_types):
            if neighborhood_counts[t][t_prime] > 0:
                # Average the dissimilarity index
                D_matrix[t][t_prime] /= neighborhood_counts[t][t_prime]
            else:
                D_matrix[t][t_prime] = 0  # Handle cases where no neighborhoods were tested

    # Normalize by maximum possible dissimilarity
    max_possible_dissimilarity = np.max(D_matrix)
    if max_possible_dissimilarity > 0:
        D_matrix /= max_possible_dissimilarity  # Normalize to [0, 1]
    else:
        D_matrix = np.zeros_like(D_matrix)  # Handle cases with no data

    return D_matrix

data = pd.read_csv('./outputs/sample_counts_w.csv')
indices = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'sample_id', 'type_id', 'r', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
d_indices = []

for idx, filename in enumerate(filenames):
    df1 = data[data['filename'] == filename]
    file_indices = []
    for r in rs:
        df2 = df1[df1['diameter'] == r]         
        d_index = compute_d_index_pairwise_estimates_sample(df2, 9, len(df2['sample_id'].unique()))
        file_indices.append(d_index)
        
        for sample_id, rows in enumerate(d_index):
            for type_id, row in enumerate(rows):
                indices = pd.concat([indices, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'sample_id': [sample_id], 'type_id': [type_id], 'r': r, '0': [row[0]], '1': [row[1]], '2': [row[2]], '3': [row[3]], '4': [row[4]], '5': [row[5]], '6': [row[6]], '7': [row[7]], '8': [row[8]]})])
    d_indices.append(file_indices)
indices.to_csv('outputs/estimated_d_index_sample_w.csv', index=False)
arr = np.asarray(d_indices)
np.save('outputs/estimated_d_index_sample_w.npy', arr)

data = pd.read_csv('./outputs/sample_counts_w.csv')
indices = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'type_id', 'r', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
d_indices = []

for idx, filename in enumerate(filenames):
    df1 = data[data['filename'] == filename]
    file_indices = []
    for r in rs:
        df2 = df1[df1['diameter'] == r]         
        d_index = compute_d_index_pairwise_estimates_plot(df2, 9)
        file_indices.append(d_index)
            
        for type_id, row in enumerate(d_index):
            indices = pd.concat([indices, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'type_id': [type_id], 'r': r, '0': [row[0]], '1': [row[1]], '2': [row[2]], '3': [row[3]], '4': [row[4]], '5': [row[5]], '6': [row[6]], '7': [row[7]], '8': [row[8]]})])
    d_indices.append(file_indices)
indices.to_csv('outputs/estimated_d_index_plot_w.csv', index=False)
arr = np.asarray(d_indices)
np.save('outputs/estimated_d_index_plot_w.npy', arr)

### D_index
#   Estimates Sys Reg
#
###
data = pd.read_csv('./outputs/sample_counts_reg.csv')
indices = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'sample_id', 'type_id', 'r', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
d_indices = []

for idx, filename in enumerate(filenames):
    df1 = data[data['filename'] == filename]
    file_indices = []
    for r in rs:
        df2 = df1[df1['diameter'] == r]         
        d_index = compute_d_index_pairwise_estimates_sample(df2, 9, len(df2['sample_id'].unique()))
        file_indices.append(d_index)
        
        for sample_id, rows in enumerate(d_index):
            for type_id, row in enumerate(rows):
                indices = pd.concat([indices, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'sample_id': [sample_id], 'type_id': [type_id], 'r': r, '0': [row[0]], '1': [row[1]], '2': [row[2]], '3': [row[3]], '4': [row[4]], '5': [row[5]], '6': [row[6]], '7': [row[7]], '8': [row[8]]})])
    d_indices.append(file_indices)
indices.to_csv('outputs/estimated_d_index_sample_reg.csv', index=False)
arr = np.asarray(d_indices)
np.save('outputs/estimated_d_index_sample_reg.npy', arr)

data = pd.read_csv('./outputs/sample_counts_reg.csv')
indices = pd.DataFrame(columns=['filename', 'init_locs', 'seed', 'type_id', 'r', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
d_indices = []

for idx, filename in enumerate(filenames):
    df1 = data[data['filename'] == filename]
    file_indices = []
    for r in rs:
        df2 = df1[df1['diameter'] == r]         
        d_index = compute_d_index_pairwise_estimates_plot(df2, 9)
        file_indices.append(d_index)
            
        for type_id, row in enumerate(d_index):
            indices = pd.concat([indices, pd.DataFrame({'filename': [filename], 'init_locs': [init_locs[idx]], 'seed': [init_seed[idx]], 'type_id': [type_id], 'r': r, '0': [row[0]], '1': [row[1]], '2': [row[2]], '3': [row[3]], '4': [row[4]], '5': [row[5]], '6': [row[6]], '7': [row[7]], '8': [row[8]]})])
    d_indices.append(file_indices)
indices.to_csv('outputs/estimated_d_index_plot_reg.csv', index=False)
arr = np.asarray(d_indices)
np.save('outputs/estimated_d_index_plot_reg.npy', arr)