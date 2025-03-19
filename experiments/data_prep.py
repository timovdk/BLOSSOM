import numpy as np
import polars as pl
from tqdm import tqdm

init_types = ["random", "clustered"]
trials = 20
filenames = [
    f"{init_type}_{trial}" for init_type in init_types for trial in range(1, trials + 1)
]
path = "./raw_data/"
rs = [1, 2, 3, 4, 5]
sample_times = [0, 100, 200, 300, 400, 500, 600]
x_max = 400
y_max = 400
num_types = 9


# Retrieve DataFrame indices that are within Von Neumann neighbourhood defined by centers and r
def retrieve_ids_per_sample_von_neumann(centers, data_x, data_y, r):
    samples = [[] for _ in range(len(centers))]

    for i, (center_x, center_y) in enumerate(centers):
        distances = (np.abs(center_x - data_x) % 400) + (
            np.abs(center_y - data_y) % 400
        )
        samples[i] = np.where(distances <= r)[0].tolist()

    return samples


# Sample centers
selected_points_w = [
    (50, 50),
    (50, 150),
    (50, 250),
    (50, 350),
    (125, 150),
    (175, 250),
    (275, 150),
    (225, 250),
    (350, 50),
    (350, 150),
    (350, 250),
    (350, 350),
]
selected_points_reg = [
    (50, 50),
    (50, 150),
    (50, 250),
    (50, 350),
    (150, 50),
    (150, 150),
    (150, 250),
    (150, 350),
    (250, 50),
    (250, 150),
    (250, 250),
    (250, 350),
    (350, 50),
    (350, 150),
    (350, 250),
    (350, 350),
]

# Store results in lists
sample_counts_w_list = []
sample_counts_reg_list = []

# Process each file
for idx, filename in tqdm(
    enumerate(filenames), desc="Computing Abundances per Simulated Sample"
):
    df = pl.read_parquet(path + filename + ".parquet")

    data_x = df["x"].to_numpy()
    data_y = df["y"].to_numpy()
    data_type = df["type"].to_numpy()

    for st in sample_times:
        # Filter data by time step
        mask = df["tick"] == st
        filtered_x = data_x[mask]
        filtered_y = data_y[mask]
        filtered_type = data_type[mask]

        for r in rs:
            # Get DataFrame indices of organisms within Von Neumann neighbourhood of samples
            samples_w = retrieve_ids_per_sample_von_neumann(
                selected_points_w, filtered_x, filtered_y, r
            )
            samples_reg = retrieve_ids_per_sample_von_neumann(
                selected_points_reg, filtered_x, filtered_y, r
            )

            # Count organism types
            for i, sample in enumerate(samples_w):
                unique, counts = np.unique(filtered_type[sample], return_counts=True)
                type_counts = {str(t): 0 for t in range(9)}
                type_counts.update(dict(zip(map(str, unique), counts)))

                sample_counts_w_list.append(
                    {
                        "filename": filename,
                        "sample_time": st,
                        "r": r,
                        "sample_id": i,
                        **type_counts,
                    }
                )

            for i, sample in enumerate(samples_reg):
                unique, counts = np.unique(filtered_type[sample], return_counts=True)
                type_counts = {str(t): 0 for t in range(9)}
                type_counts.update(dict(zip(map(str, unique), counts)))

                sample_counts_reg_list.append(
                    {
                        "filename": filename,
                        "sample_time": st,
                        "r": r,
                        "sample_id": i,
                        **type_counts,
                    }
                )

# Convert lists to DataFrame and save to CSV
pl.DataFrame(sample_counts_w_list).write_csv("prep_out/sample_counts_w.csv")
pl.DataFrame(sample_counts_reg_list).write_csv("prep_out/sample_counts_reg.csv")

# Initialize a list to store the results
abundances_list = []

# Loop over filenames
for idx, filename in tqdm(enumerate(filenames), desc="Computing Baseline Abundances"):
    # Read the parquet file using polars
    df = pl.read_parquet(path + filename + ".parquet")

    # Loop over sample times
    for st in sample_times:
        # Filter the data for the current sample time
        data = df.filter(pl.col("tick") == st)

        # Count occurrences of each type (0-8)
        counts = (
            data.group_by("type")
            .agg(pl.len())
            .select(
                [pl.col("type"), (pl.col("len") / 20000).round(5).alias("abundance")]
            )
            .with_columns(
                pl.col("type").cast(pl.Int32)  # Ensuring 'type' is an integer
            )
        )

        # Ensure all organism types (0-8) are included even if missing
        # Create a range of types (0-8) and left join with the counts
        all_types = pl.DataFrame({"type": list(range(9))})
        counts = all_types.join(counts, on="type", how="left").fill_null(0)

        # Append the result to the list
        abundances_list.append(
            {
                "filename": filename,
                "sample_time": st,
                **{
                    str(i): counts.filter(pl.col("type") == i)["abundance"].to_list()[0]
                    for i in range(9)
                },
            }
        )

# Convert the list of dictionaries to a polars DataFrame and then to CSV
abundances_df = pl.DataFrame(abundances_list)
abundances_df.write_csv("prep_out/baseline_abundances.csv")


# Compute normalized abundance estimates
def compute_abundance_estimates(df, num_types, r):
    cells = (2 * (r**2)) + (2 * r) + 1
    sample_weight = cells * 0.125

    # Normalize and round to 5 decimals
    return df.select(
        "filename",
        "r",
        "sample_time",
        "sample_id",
        *[pl.col(str(t)).cast(pl.Float64) / sample_weight for t in range(num_types)],
    ).with_columns([pl.col(str(t)).round(5) for t in range(num_types)])


data_w = pl.read_csv("./prep_out/sample_counts_w.csv")
data_reg = pl.read_csv("./prep_out/sample_counts_reg.csv")

results_w = []
results_reg = []

for filename in tqdm(filenames, desc="Computing Estimated Abundances"):
    df1_w = data_w.filter(pl.col("filename") == filename)
    df1_reg = data_reg.filter(pl.col("filename") == filename)

    for st in sample_times:
        df2_w = df1_w.filter(pl.col("sample_time") == st)
        df2_reg = df1_reg.filter(pl.col("sample_time") == st)

        for r in rs:
            df3_w = df2_w.filter(pl.col("r") == r)
            df3_reg = df2_reg.filter(pl.col("r") == r)

            # Compute normalized abundances
            norm_abundances_w = compute_abundance_estimates(df3_w, num_types=9, r=r)
            norm_abundances_reg = compute_abundance_estimates(df3_reg, num_types=9, r=r)

            # Append to results
            results_w.append(norm_abundances_w)
            results_reg.append(norm_abundances_reg)

# Concatenate all results and save to CSV
pl.concat(results_w).write_csv("prep_out/estimated_abundances_w.csv")
pl.concat(results_reg).write_csv("prep_out/estimated_abundances_reg.csv")


def shannon_diversity(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    proportions = counts / total
    return -np.sum(proportions[proportions > 0] * np.log(proportions[proportions > 0]))


def simpson_diversity(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    proportions = counts / total
    return 1.0 - np.sum(proportions**2)


diversity_results = []

for filename in tqdm(filenames, desc="Computing Baseline Diversity Indices"):
    df = pl.read_parquet(path + filename + ".parquet")

    for st in sample_times:
        data = df.filter(pl.col("tick") == st)

        # Compute type counts and reindex to ensure all organism types 0-8 exist
        counts = (
            data["type"]
            .value_counts()
            .with_columns(pl.col("type").cast(pl.Int64))
            .sort("type")
        )

        full_range_df = pl.DataFrame({"type": range(9)})
        counts_df = full_range_df.join(counts, on="type", how="left").with_columns(
            pl.col("count").fill_null(0)
        )

        counts_array = counts_df["count"].to_numpy()
        diversity_results.append(
            (
                filename,
                st,
                shannon_diversity(counts_array).round(5),
                simpson_diversity(counts_array).round(5),
            )
        )

# Convert results to Polars DataFrame and save
pl.DataFrame(
    diversity_results,
    schema=["filename", "sample_time", "shannon", "simpson"],
    orient="row",
).write_csv("prep_out/baseline_diversity_indices.csv")

df_w = pl.read_csv("prep_out/sample_counts_w.csv")
df_reg = pl.read_csv("prep_out/sample_counts_reg.csv")

count_cols = [str(i) for i in range(9)]
diversity_dfs_w = []
diversity_dfs_reg = []

print("Computing Estimated Diversity Indices")

for group_cols, output_file in [
    (
        ["filename", "r", "sample_time", "sample_id"],
        "estimated_diversity_indices_sample",
    ),  # No pooling
    (["filename", "r", "sample_time"], "estimated_diversity_indices_plot"),  # Per plot
    (["filename", "r"], "estimated_diversity_indices_temporal"),  # Per plot over time
]:
    grouped_df_w = df_w.group_by(group_cols).agg(
        [pl.sum(col).alias(col) for col in count_cols]
    )
    grouped_df_reg = df_reg.group_by(group_cols).agg(
        [pl.sum(col).alias(col) for col in count_cols]
    )

    counts_array_w = grouped_df_w.select(count_cols).to_numpy()
    counts_array_reg = grouped_df_reg.select(count_cols).to_numpy()

    shannon_vals_w = np.round(
        np.apply_along_axis(shannon_diversity, 1, counts_array_w), 5
    )
    simpson_vals_w = np.round(
        np.apply_along_axis(simpson_diversity, 1, counts_array_w), 5
    )
    shannon_vals_reg = np.round(
        np.apply_along_axis(shannon_diversity, 1, counts_array_reg), 5
    )
    simpson_vals_reg = np.round(
        np.apply_along_axis(simpson_diversity, 1, counts_array_reg), 5
    )

    diversity_dfs_w.append(
        grouped_df_w.with_columns(
            [pl.Series("shannon", shannon_vals_w), pl.Series("simpson", simpson_vals_w)]
        )
        .drop(count_cols)
        .sort(group_cols)  # Drop organism count columns
    )
    diversity_dfs_reg.append(
        grouped_df_reg.with_columns(
            [
                pl.Series("shannon", shannon_vals_reg),
                pl.Series("simpson", simpson_vals_reg),
            ]
        )
        .drop(count_cols)
        .sort(group_cols)  # Drop organism count columns
    )

    diversity_dfs_w[-1].write_csv(f"prep_out/{output_file}_w.csv")
    diversity_dfs_reg[-1].write_csv(f"prep_out/{output_file}_reg.csv")


def compute_d_index_pairwise(data, range_by_type, pseudo_count=1e-5):
    type_count = np.zeros((x_max, y_max, num_types))  # (x, y, type)
    D_matrix = np.zeros((num_types, num_types))  # 9x9 dissimilarity matrix
    neighborhood_counts = np.zeros((num_types, num_types))  # Neighborhood count tracker

    # Populate the 2D grid with agent counts (adjusted for multiple organisms in the same location)
    x_vals, y_vals, type_vals = (
        data["x"].to_numpy(),
        data["y"].to_numpy(),
        data["type"].to_numpy(),
    )
    for x, y, t in zip(x_vals, y_vals, type_vals):
        type_count[x, y, t] += (
            1  # Increment the count for each organism type at the (x, y) location
        )

    # Calculate the total number of agents for each type in the entire grid
    total_count_by_type = np.sum(type_count, axis=(0, 1))

    # Vectorized neighborhood count function
    def compute_neighborhood_counts(x, y, r):
        """Vectorized Von Neumann neighborhood count computation."""
        neighborhood_count = np.zeros(num_types)
        for dx in range(-r, r + 1):
            for dy in range(-r + abs(dx), r - abs(dx) + 1):
                nx = (x + dx) % x_max
                ny = (y + dy) % y_max
                neighborhood_count += type_count[nx, ny]
        return neighborhood_count

    # Compute D-index
    for x in range(x_max):
        for y in range(y_max):
            for t in range(num_types):
                r = range_by_type[t]
                neighborhood_count_at_unit = compute_neighborhood_counts(x, y, r)

                for t_prime in range(num_types):
                    if total_count_by_type[t] > 0 and total_count_by_type[t_prime] > 0:
                        prop_t = (
                            neighborhood_count_at_unit[t] + pseudo_count
                        ) / total_count_by_type[t]
                        prop_t_prime = (
                            neighborhood_count_at_unit[t_prime] + pseudo_count
                        ) / total_count_by_type[t_prime]

                        # Compute absolute difference
                        D = abs(prop_t - prop_t_prime)

                        # Accumulate values symmetrically
                        D_matrix[t, t_prime] += D
                        neighborhood_counts[t, t_prime] += 1

    # Normalize D-index
    mask = neighborhood_counts > 0
    D_matrix[mask] /= neighborhood_counts[mask]  # Avoid division by zero

    # Normalize to [0,1]
    max_D = np.max(D_matrix)
    if max_D > 0:
        D_matrix /= max_D

    return D_matrix


d_index_results = []

for filename in tqdm(filenames, desc="Computing Baseline D-Index"):
    df = pl.read_parquet(path + filename + ".parquet")

    for st in sample_times:
        data = df.filter(pl.col("tick") == st)

        # Compute the D-index (using the optimized function)
        d_index = compute_d_index_pairwise(data, [1] * 9)

        # Store results efficiently in a list
        for type_id, row in enumerate(d_index):
            rounded_row = tuple(
                round(val, 5) for val in row
            )  # Round each value to 5 decimals
            d_index_results.append((filename, st, type_id, *rounded_row))

# Convert list to a Polars DataFrame in one step (much faster than looping `concat`)
indices_df = pl.DataFrame(
    d_index_results,
    schema=["filename", "sample_time", "type_id"] + [str(i) for i in range(9)],
    orient="row",
)

# Save to CSV
indices_df.write_csv("prep_out/baseline_d_index_test.csv")


def compute_d_index(counts, num_samples, mode):
    total_count_by_type = [counts[str(idx)].sum() for idx in range(num_types)]

    if mode == "sample":
        D_matrix = np.zeros((num_samples, num_types, num_types))
        neighborhood_counts = np.zeros((num_samples, num_types, num_types))
    else:  # "plot"
        D_matrix = np.zeros((num_types, num_types))
        neighborhood_counts = np.zeros((num_types, num_types))

    for sample_id in range(num_samples):
        sample_counts = counts.filter(pl.col("sample_id") == sample_id)
        for t in range(num_types):
            t_count = sample_counts[str(t)].sum()
            for t_prime in range(num_types):
                if all(
                    [
                        t_count > 0,
                        total_count_by_type[t] > 0,
                        total_count_by_type[t_prime] > 0,
                    ]
                ):
                    prop_t = t_count / total_count_by_type[t]
                    prop_t_prime = (
                        sample_counts[str(t_prime)].sum() / total_count_by_type[t_prime]
                    )
                    D = abs(prop_t - prop_t_prime)

                    if mode == "sample":
                        D_matrix[sample_id, t, t_prime] += D
                        neighborhood_counts[sample_id, t, t_prime] += 1
                        if t != t_prime:
                            D_matrix[sample_id, t_prime, t] += D
                            neighborhood_counts[sample_id, t_prime, t] += 1
                    else:
                        D_matrix[t, t_prime] += D
                        neighborhood_counts[t, t_prime] += 1
                        if t != t_prime:
                            D_matrix[t_prime, t] += D
                            neighborhood_counts[t_prime, t] += 1

    valid_counts = neighborhood_counts > 0
    D_matrix[valid_counts] /= neighborhood_counts[valid_counts]
    D_matrix[~valid_counts] = 0

    max_dissimilarity = np.max(D_matrix)
    if max_dissimilarity > 0:
        D_matrix /= max_dissimilarity
    else:
        D_matrix.fill(0)

    return D_matrix


def collect_indices_from_d_index(data, filenames, sample_times, rs, mode, sample_sim):
    if mode == "temporal":
        data = data.group_by(["filename", "r", "sample_id"]).agg(
            [pl.col(str(i)).sum().alias(str(i)) for i in range(9)]
        )
    indices = []
    for filename in filenames:
        for r in rs:
            df_filtered = data.filter(pl.col("filename") == filename).filter(
                pl.col("r") == r
            )
            if mode != "temporal":
                for st in sample_times:
                    df_st_filtered = df_filtered.filter(pl.col("sample_time") == st)
                    d_index_result = compute_d_index(
                        df_st_filtered,
                        len(df_st_filtered["sample_id"].unique()),
                        mode=mode,
                    )
                    if mode == "sample":
                        for sample_index, sample_id in enumerate(
                            df_st_filtered["sample_id"].unique(maintain_order=True)
                        ):
                            for type_id in range(9):
                                row = d_index_result[sample_index, type_id]
                                indices.append(
                                    {
                                        "filename": filename,
                                        "r": r,
                                        "sample_time": st,
                                        "sample_id": sample_id,
                                        "type_id": type_id,
                                        **{str(i): row[i].round(5) for i in range(9)},
                                    }
                                )

                    elif mode == "plot":
                        for type_id, row in enumerate(d_index_result):
                            indices.append(
                                {
                                    "filename": filename,
                                    "r": r,
                                    "sample_time": st,
                                    "type_id": type_id,
                                    **{str(i): row[i].round(5) for i in range(9)},
                                }
                            )
            else:
                d_index_result = compute_d_index(
                    df_filtered, len(df_filtered["sample_id"].unique()), mode="plot"
                )
                for type_id, row in enumerate(d_index_result):
                    indices.append(
                        {
                            "filename": filename,
                            "r": r,
                            "type_id": type_id,
                            **{str(i): row[i].round(5) for i in range(9)},
                        }
                    )

    df_result = pl.DataFrame(indices)
    df_result = pl.DataFrame(
        [
            {
                k: str(v) if isinstance(v, (list, np.ndarray)) else v
                for k, v in entry.items()
            }
            for entry in indices
        ]
    )
    df_result.write_csv(f"prep_out/estimated_d_index_{mode}_{sample_sim}.csv")


print("Computing Estimated D-Index")

# Load data
data_w = pl.read_csv("./prep_out/sample_counts_w.csv")
data_reg = pl.read_csv("./prep_out/sample_counts_reg.csv")

# Compute D-index with different pooling strategies
collect_indices_from_d_index(
    data_w, filenames, sample_times, rs, mode="sample", sample_sim="w"
)
collect_indices_from_d_index(
    data_w, filenames, sample_times, rs, mode="plot", sample_sim="w"
)
collect_indices_from_d_index(
    data_w, filenames, sample_times, rs, mode="temporal", sample_sim="w"
)

collect_indices_from_d_index(
    data_reg, filenames, sample_times, rs, mode="sample", sample_sim="reg"
)
collect_indices_from_d_index(
    data_reg, filenames, sample_times, rs, mode="plot", sample_sim="reg"
)
collect_indices_from_d_index(
    data_reg, filenames, sample_times, rs, mode="temporal", sample_sim="reg"
)
