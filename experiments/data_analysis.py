import polars as pl
from sklearn.metrics import mean_absolute_error, median_absolute_error
import numpy as np


def calculate_normalized_absolute_error(y_true, y_pred, pop_counts):
    np.seterr(invalid="raise", divide="raise")
    y_true, y_pred, pop_counts = map(np.array, (y_true, y_pred, pop_counts))
    absolute_errors = np.abs(y_pred - y_true)

    normalized_errors = np.zeros_like(absolute_errors)
    valid_indices = pop_counts != 0
    if not np.any(valid_indices):
        return normalized_errors
    normalized_errors[valid_indices] = (
        absolute_errors[valid_indices] / pop_counts[valid_indices]
    )
    return normalized_errors


def calculate_normalized_error(y_true, y_pred, pop_counts):
    y_true, y_pred, pop_counts = map(np.array, (y_true, y_pred, pop_counts))
    errors = y_pred - y_true

    normalized_errors = np.zeros_like(errors)
    valid_indices = pop_counts != 0
    if not np.any(valid_indices):
        return normalized_errors
    normalized_errors[valid_indices] = errors[valid_indices] / pop_counts[valid_indices]
    return normalized_errors


def process_metrics_data_sample(
    es_r_r,
    es_w_r,
    bl_st_ab,
    values,
    sample_time,
    file,
    rs,
    metrics_reg_list,
    metrics_w_list,
    metrics_reg_type_list,
    metrics_w_type_list,
):
    """
    Function to process and append metrics for both reg and w data
    """
    for r in rs:
        es_r_r_filtered = es_r_r.filter(pl.col("r") == r)
        es_w_r_filtered = es_w_r.filter(pl.col("r") == r)

        for es, metrics_list, metrics_type_list in [
            (es_r_r_filtered, metrics_reg_list, metrics_reg_type_list),
            (es_w_r_filtered, metrics_w_list, metrics_w_type_list),
        ]:
            for i in es["sample_id"].unique():
                es_ab = (
                    es.filter(pl.col("sample_id") == i)
                    .select(pl.col(["0", "1", "2", "3", "4", "5", "6", "7", "8"]))
                    .to_numpy()
                    .flatten()
                    * 1000
                )

                # Calculate both MAE (normalized absolute error) and MSE (normalized error)
                ae = calculate_normalized_absolute_error([bl_st_ab], es_ab, [values])[0]
                mse = calculate_normalized_error([bl_st_ab], es_ab, [values])[0]

                metrics_list.append(
                    {
                        "filename": file,
                        "sample_time": sample_time,
                        "r": r,
                        "sample_id": i,
                        "mae": np.mean(ae),
                        "mdae": np.median(ae),
                        "mser": np.mean(mse),
                        "mdser": np.median(mse),
                    }
                )

                for t in range(9):
                    es_ab_t = es_ab[t]

                    ae = calculate_normalized_absolute_error(
                        [bl_st_ab[t]], [es_ab_t], [values[t]]
                    )[0]
                    ser = calculate_normalized_error(
                        [bl_st_ab[t]], [es_ab_t], [values[t]]
                    )[0]
                    metrics_type_list.append(
                        {
                            "filename": file,
                            "sample_time": sample_time,
                            "r": r,
                            "sample_id": i,
                            "type_id": t,
                            "ae": ae,
                            "ser": ser,
                        }
                    )


def process_raw_data_sample(
    ground_truth_dict,
    estimate_reg_dict,
    estimate_w_dict,
    rs=[1, 2, 3, 4, 5],
    sample_times=[0, 100, 200, 300, 400, 500, 600],
):
    """
    Function to process raw data for a given sample time and calculate error metrics
    """
    metrics_reg_list = []
    metrics_w_list = []
    metrics_reg_type_list = []
    metrics_w_type_list = []

    # Read raw data, ground_truth, and estimate data for the given file and sample time
    for file in ground_truth_dict.keys():
        raw_data = pl.read_parquet(f"./raw_data/{file}.parquet")
        for sample_time in sample_times:
            raw_data_st = raw_data.filter(pl.col("tick") == sample_time)
            bl = ground_truth_dict[file]
            es_r = estimate_reg_dict[file]
            es_w = estimate_w_dict[file]

            # Precompute unique types and their counts
            type_counts = raw_data_st.group_by("type").agg(
                pl.col("type").count().alias("count")
            )
            unique_types = raw_data["type"].unique(maintain_order=True)

            # Create a fast lookup array for type counts
            values = np.zeros(len(unique_types), dtype=int)
            values[np.searchsorted(unique_types, type_counts["type"])] = type_counts[
                "count"
            ].to_numpy()

            es_st_r = es_r.filter(pl.col("sample_time") == sample_time)
            es_st_w = es_w.filter(pl.col("sample_time") == sample_time)
            bl_st = bl.filter(pl.col("sample_time") == sample_time)

            bl_st_ab = (
                bl_st.select(pl.col(["0", "1", "2", "3", "4", "5", "6", "7", "8"]))
                .to_numpy()
                .flatten()
                * 1000
            )

            # Process for each sample time
            process_metrics_data_sample(
                es_st_r,
                es_st_w,
                bl_st_ab,
                values,
                sample_time,
                file,
                rs,
                metrics_reg_list,
                metrics_w_list,
                metrics_reg_type_list,
                metrics_w_type_list,
            )

    return metrics_reg_list, metrics_w_list, metrics_reg_type_list, metrics_w_type_list


# Read data using Polars LazyFrames
ground_truth = pl.read_csv("prep_out/ground_truth_abundances.csv")
estimate_reg = pl.read_csv("prep_out/estimated_abundances_reg.csv")
estimate_w = pl.read_csv("prep_out/estimated_abundances_w.csv")

# Convert groupby results into dictionaries for fast lookup
ground_truth_dict = {df["filename"][0]: df for df in ground_truth.partition_by("filename")}
estimate_reg_dict = {
    df["filename"][0]: df for df in estimate_reg.partition_by("filename")
}
estimate_w_dict = {df["filename"][0]: df for df in estimate_w.partition_by("filename")}

metrics_reg_list = []
metrics_w_list = []
metrics_reg_type_list = []
metrics_w_type_list = []

metrics_reg, metrics_w, metrics_reg_type, metrics_w_type = process_raw_data_sample(
    ground_truth_dict,
    estimate_reg_dict,
    estimate_w_dict,
)
metrics_reg_list.extend(metrics_reg)
metrics_w_list.extend(metrics_w)
metrics_reg_type_list.extend(metrics_reg_type)
metrics_w_type_list.extend(metrics_w_type)

# Convert the lists to Polars DataFrames
metrics_reg_df = pl.DataFrame(metrics_reg_list)
metrics_w_df = pl.DataFrame(metrics_w_list)
metrics_reg_type_df = pl.DataFrame(metrics_reg_type_list)
metrics_w_type_df = pl.DataFrame(metrics_w_type_list)

# Save the results as CSV
metrics_reg_df.write_csv("./analysis_out/abundances_sample_reg.csv")
metrics_w_df.write_csv("./analysis_out/abundances_sample_w.csv")
metrics_reg_type_df.write_csv("./analysis_out/abundances_sample_reg_type.csv")
metrics_w_type_df.write_csv("./analysis_out/abundances_sample_w_type.csv")


def process_metrics_data_plot(
    es_r_r,
    es_w_r,
    bl_st_ab,
    values,
    sample_time,
    file,
    rs,
    metrics_reg_list,
    metrics_w_list,
    metrics_reg_type_list,
    metrics_w_type_list,
):
    """
    Function to process and append metrics for both reg and w data
    """
    for r in rs:
        es_r_r_filtered = es_r_r.filter(pl.col("r") == r)
        es_w_r_filtered = es_w_r.filter(pl.col("r") == r)

        for es, metrics_list, metrics_type_list in [
            (es_r_r_filtered, metrics_reg_list, metrics_reg_type_list),
            (es_w_r_filtered, metrics_w_list, metrics_w_type_list),
        ]:
            es_ab = (
                es.select(pl.sum(["0", "1", "2", "3", "4", "5", "6", "7", "8"]))
                .to_numpy()
                .flatten()
                / len(es["sample_id"].unique())
                * 1000
            )
            # Calculate both MAE (normalized absolute error) and MSE (normalized error)
            ae = calculate_normalized_absolute_error([bl_st_ab], es_ab, [values])[0]
            mse = calculate_normalized_error([bl_st_ab], es_ab, [values])[0]
            metrics_list.append(
                {
                    "filename": file,
                    "sample_time": sample_time,
                    "r": r,
                    "mae": np.mean(ae),
                    "mdae": np.median(ae),
                    "mser": np.mean(mse),
                    "mdser": np.median(mse),
                }
            )
            for t in range(9):
                es_ab_t = es_ab[t]
                ae = calculate_normalized_absolute_error(
                    [bl_st_ab[t]], [es_ab_t], [values[t]]
                )[0]
                ser = calculate_normalized_error([bl_st_ab[t]], [es_ab_t], [values[t]])[
                    0
                ]
                metrics_type_list.append(
                    {
                        "filename": file,
                        "sample_time": sample_time,
                        "r": r,
                        "type_id": t,
                        "ae": ae,
                        "ser": ser,
                    }
                )


def process_raw_data_plot(
    ground_truth_dict,
    estimate_reg_dict,
    estimate_w_dict,
    rs=[1, 2, 3, 4, 5],
    sample_times=[0, 100, 200, 300, 400, 500, 600],
):
    """
    Function to process raw data for a given sample time and calculate error metrics
    """
    metrics_reg_list = []
    metrics_w_list = []
    metrics_reg_type_list = []
    metrics_w_type_list = []

    # Read raw data, ground_truth, and estimate data for the given file and sample time
    for file in ground_truth_dict.keys():
        raw_data = pl.read_parquet(f"./raw_data/{file}.parquet")
        for sample_time in sample_times:
            raw_data_st = raw_data.filter(pl.col("tick") == sample_time)
            bl = ground_truth_dict[file]
            es_r = estimate_reg_dict[file]
            es_w = estimate_w_dict[file]

            # Precompute unique types and their counts
            type_counts = raw_data_st.group_by("type").agg(
                pl.col("type").count().alias("count")
            )
            unique_types = raw_data["type"].unique(maintain_order=True)

            # Create a fast lookup array for type counts
            values = np.zeros(len(unique_types), dtype=int)
            values[np.searchsorted(unique_types, type_counts["type"])] = type_counts[
                "count"
            ].to_numpy()

            es_st_r = es_r.filter(pl.col("sample_time") == sample_time)
            es_st_w = es_w.filter(pl.col("sample_time") == sample_time)
            bl_st = bl.filter(pl.col("sample_time") == sample_time)

            bl_st_ab = (
                bl_st.select(pl.col(["0", "1", "2", "3", "4", "5", "6", "7", "8"]))
                .to_numpy()
                .flatten()
                * 1000
            )

            # Process for each sample time
            process_metrics_data_plot(
                es_st_r,
                es_st_w,
                bl_st_ab,
                values,
                sample_time,
                file,
                rs,
                metrics_reg_list,
                metrics_w_list,
                metrics_reg_type_list,
                metrics_w_type_list,
            )

    return metrics_reg_list, metrics_w_list, metrics_reg_type_list, metrics_w_type_list


# Read data using Polars LazyFrames
ground_truth = pl.read_csv("prep_out/ground_truth_abundances.csv")
estimate_reg = pl.read_csv("prep_out/estimated_abundances_reg.csv")
estimate_w = pl.read_csv("prep_out/estimated_abundances_w.csv")

# Convert groupby results into dictionaries for fast lookup
ground_truth_dict = {df["filename"][0]: df for df in ground_truth.partition_by("filename")}
estimate_reg_dict = {
    df["filename"][0]: df for df in estimate_reg.partition_by("filename")
}
estimate_w_dict = {df["filename"][0]: df for df in estimate_w.partition_by("filename")}

metrics_reg_list = []
metrics_w_list = []
metrics_reg_type_list = []
metrics_w_type_list = []

# Iterate through sample times and process metrics data

metrics_reg, metrics_w, metrics_reg_type, metrics_w_type = process_raw_data_plot(
    ground_truth_dict,
    estimate_reg_dict,
    estimate_w_dict,
)
metrics_reg_list.extend(metrics_reg)
metrics_w_list.extend(metrics_w)
metrics_reg_type_list.extend(metrics_reg_type)
metrics_w_type_list.extend(metrics_w_type)

# Convert the lists to Polars DataFrames
metrics_reg_df = pl.DataFrame(metrics_reg_list)
metrics_w_df = pl.DataFrame(metrics_w_list)
metrics_reg_type_df = pl.DataFrame(metrics_reg_type_list)
metrics_w_type_df = pl.DataFrame(metrics_w_type_list)

# Save the results as CSV
metrics_reg_df.write_csv("./analysis_out/abundances_plot_reg.csv")
metrics_w_df.write_csv("./analysis_out/abundances_plot_w.csv")
metrics_reg_type_df.write_csv("./analysis_out/abundances_plot_reg_type.csv")
metrics_w_type_df.write_csv("./analysis_out/abundances_plot_w_type.csv")

def process_metrics_data_temporal(
    es_r,
    es_w,
    bl_ab,
    values,
    sample_times,
    file,
    rs,
    metrics_reg_list,
    metrics_w_list,
    metrics_reg_type_list,
    metrics_w_type_list,
):
    """
    Function to process and append metrics for both reg and w data
    """
    for r in rs:
        es_r_r = es_r.filter(pl.col("r") == r)
        es_w_r = es_w.filter(pl.col("r") == r)

        for es, metrics_list, metrics_type_list in [
            (es_r_r, metrics_reg_list, metrics_reg_type_list),
            (es_w_r, metrics_w_list, metrics_w_type_list),
        ]:
            es_ab = (
                es.select(pl.sum(["0", "1", "2", "3", "4", "5", "6", "7", "8"]))
                .to_numpy()
                .flatten()
                / len(es["sample_id"].unique())
                / len(sample_times)
                * 1000
            )
            # Calculate both MAE (normalized absolute error) and MSE (normalized error)
            ae = calculate_normalized_absolute_error([bl_ab], es_ab, [values])[0]
            mse = calculate_normalized_error([bl_ab], es_ab, [values])[0]
            metrics_list.append(
                {
                    "filename": file,
                    "r": r,
                    "mae": np.mean(ae),
                    "mdae": np.median(ae),
                    "mser": np.mean(mse),
                    "mdser": np.median(mse),
                }
            )
            for t in range(9):
                es_ab_t = es_ab[t]
                ae = calculate_normalized_absolute_error(
                    [bl_ab[t]], [es_ab_t], [values[t]]
                )[0]
                ser = calculate_normalized_error([bl_ab[t]], [es_ab_t], [values[t]])[0]
                metrics_type_list.append(
                    {
                        "filename": file,
                        "r": r,
                        "type_id": t,
                        "ae": ae,
                        "ser": ser,
                    }
                )


def process_raw_data_temporal(
    ground_truth_dict,
    estimate_reg_dict,
    estimate_w_dict,
    rs=[1, 2, 3, 4, 5],
    sample_times=[0, 100, 200, 300, 400, 500, 600],
):
    """
    Function to process raw data for a given sample time and calculate error metrics
    """
    metrics_reg_list = []
    metrics_w_list = []
    metrics_reg_type_list = []
    metrics_w_type_list = []

    # Read raw data, ground_truth, and estimate data for the given file and sample time
    for file in ground_truth_dict.keys():
        raw_data = pl.read_parquet(f"./raw_data/{file}.parquet")
        raw_data_st = raw_data.filter(pl.col("tick").is_in(sample_times))
        bl = ground_truth_dict[file]
        es_r = estimate_reg_dict[file]
        es_w = estimate_w_dict[file]
        # Precompute unique types and their counts
        type_counts = raw_data_st.group_by("type").agg(
            pl.col("type").count().alias("count")
        )
        unique_types = raw_data["type"].unique(maintain_order=True)
        # Create a fast lookup array for type counts
        values = np.zeros(len(unique_types), dtype=int)
        values[np.searchsorted(unique_types, type_counts["type"])] = type_counts[
            "count"
        ].to_numpy()

        values = values.astype(float)
        values /= len(sample_times)

        bl_ab = (
            bl.select(pl.sum(["0", "1", "2", "3", "4", "5", "6", "7", "8"]))
            .to_numpy()
            .flatten()
            / len(sample_times)
            * 1000
        )
        # Process for each sample time
        process_metrics_data_temporal(
            es_r,
            es_w,
            bl_ab,
            values,
            sample_times,
            file,
            rs,
            metrics_reg_list,
            metrics_w_list,
            metrics_reg_type_list,
            metrics_w_type_list,
        )

    return metrics_reg_list, metrics_w_list, metrics_reg_type_list, metrics_w_type_list


# Read data using Polars LazyFrames
ground_truth = pl.read_csv("prep_out/ground_truth_abundances.csv")
estimate_reg = pl.read_csv("prep_out/estimated_abundances_reg.csv")
estimate_w = pl.read_csv("prep_out/estimated_abundances_w.csv")

# Convert groupby results into dictionaries for fast lookup
ground_truth_dict = {df["filename"][0]: df for df in ground_truth.partition_by("filename")}
estimate_reg_dict = {
    df["filename"][0]: df for df in estimate_reg.partition_by("filename")
}
estimate_w_dict = {df["filename"][0]: df for df in estimate_w.partition_by("filename")}

metrics_reg_list = []
metrics_w_list = []
metrics_reg_type_list = []
metrics_w_type_list = []

# Iterate through sample times and process metrics data

metrics_reg, metrics_w, metrics_reg_type, metrics_w_type = process_raw_data_temporal(
    ground_truth_dict,
    estimate_reg_dict,
    estimate_w_dict,
)
metrics_reg_list.extend(metrics_reg)
metrics_w_list.extend(metrics_w)
metrics_reg_type_list.extend(metrics_reg_type)
metrics_w_type_list.extend(metrics_w_type)

# Convert the lists to Polars DataFrames
metrics_reg_df = pl.DataFrame(metrics_reg_list)
metrics_w_df = pl.DataFrame(metrics_w_list)
metrics_reg_type_df = pl.DataFrame(metrics_reg_type_list)
metrics_w_type_df = pl.DataFrame(metrics_w_type_list)

# Save the results as CSV
metrics_reg_df.write_csv("./analysis_out/abundances_temporal_reg.csv")
metrics_w_df.write_csv("./analysis_out/abundances_temporal_w.csv")
metrics_reg_type_df.write_csv("./analysis_out/abundances_temporal_reg_type.csv")
metrics_w_type_df.write_csv("./analysis_out/abundances_temporal_w_type.csv")

ground_truth = pl.read_csv("prep_out/ground_truth_diversity_indices.csv")
estimate_reg = pl.read_csv("prep_out/estimated_diversity_indices_sample_reg.csv")
estimate_w = pl.read_csv("prep_out/estimated_diversity_indices_sample_w.csv")

rs = [1, 2, 3, 4, 5]
sample_times = [0, 100, 200, 300, 400, 500, 600]

metrics_list_reg = []
metrics_list_w = []

for file in ground_truth["filename"].unique(maintain_order=True):
    bl_f = ground_truth.filter(pl.col("filename") == file)
    es_r = estimate_reg.filter(pl.col("filename") == file)
    es_w = estimate_w.filter(pl.col("filename") == file)

    for st in sample_times:
        bl_st = bl_f.filter(pl.col("sample_time") == st)
        bl_sh, bl_si = bl_st.select(["shannon", "simpson"]).row(0)

        es_st_r = es_r.filter(pl.col("sample_time") == st)
        es_st_w = es_w.filter(pl.col("sample_time") == st)

        for r in rs:
            es_r_r = es_st_r.filter(pl.col("r") == r)
            es_w_r = es_st_w.filter(pl.col("r") == r)

            for sample_id in es_r_r["sample_id"].unique(maintain_order=True):
                es_r_s = es_r_r.filter(pl.col("sample_id") == sample_id)
                sh, si = es_r_s.select(["shannon", "simpson"]).row(0)

                metrics_list_reg.append(
                    {
                        "filename": file,
                        "r": r,
                        "sample_time": st,
                        "sample_id": sample_id,
                        "mae_sh": mean_absolute_error([bl_sh], [sh]),
                        "mae_si": mean_absolute_error([bl_si], [si]),
                        "mdae_sh": median_absolute_error([bl_sh], [sh]),
                        "mdae_si": median_absolute_error([bl_si], [si]),
                    }
                )

            for sample_id in es_w_r["sample_id"].unique(maintain_order=True):
                es_w_s = es_w_r.filter(pl.col("sample_id") == sample_id)
                sh, si = es_w_s.select(["shannon", "simpson"]).row(0)

                metrics_list_w.append(
                    {
                        "filename": file,
                        "r": r,
                        "sample_time": st,
                        "sample_id": sample_id,
                        "mae_sh": mean_absolute_error([bl_sh], [sh]),
                        "mae_si": mean_absolute_error([bl_si], [si]),
                        "mdae_sh": median_absolute_error([bl_sh], [sh]),
                        "mdae_si": median_absolute_error([bl_si], [si]),
                    }
                )

# Convert lists to Polars DataFrame and write to CSV
pl.DataFrame(metrics_list_reg).write_csv(
    "./analysis_out/diversity_indices_sample_reg.csv"
)
pl.DataFrame(metrics_list_w).write_csv("./analysis_out/diversity_indices_sample_w.csv")

ground_truth = pl.read_csv("prep_out/ground_truth_diversity_indices.csv")
estimate_reg = pl.read_csv("prep_out/estimated_diversity_indices_plot_reg.csv")
estimate_w = pl.read_csv("prep_out/estimated_diversity_indices_plot_w.csv")

rs = [1, 2, 3, 4, 5]
sample_times = [0, 100, 200, 300, 400, 500, 600]

metrics_list_reg = []
metrics_list_w = []

for file in ground_truth["filename"].unique(maintain_order=True):
    bl_f = ground_truth.filter(pl.col("filename") == file)
    es_r = estimate_reg.filter(pl.col("filename") == file)
    es_w = estimate_w.filter(pl.col("filename") == file)

    for st in sample_times:
        bl_st = bl_f.filter(pl.col("sample_time") == st)
        bl_sh, bl_si = bl_st.select(["shannon", "simpson"]).row(0)

        es_st_r = es_r.filter(pl.col("sample_time") == st)
        es_st_w = es_w.filter(pl.col("sample_time") == st)

        for r in rs:
            es_r_r = es_st_r.filter(pl.col("r") == r)
            es_w_r = es_st_w.filter(pl.col("r") == r)

            sh, si = es_r_r.select(["shannon", "simpson"]).row(0)
            metrics_list_reg.append(
                {
                    "filename": file,
                    "r": r,
                    "sample_time": st,
                    "mae_sh": mean_absolute_error([bl_sh], [sh]),
                    "mae_si": mean_absolute_error([bl_si], [si]),
                    "mdae_sh": median_absolute_error([bl_sh], [sh]),
                    "mdae_si": median_absolute_error([bl_si], [si]),
                }
            )

            sh, si = es_w_r.select(["shannon", "simpson"]).row(0)
            metrics_list_w.append(
                {
                    "filename": file,
                    "r": r,
                    "sample_time": st,
                    "mae_sh": mean_absolute_error([bl_sh], [sh]),
                    "mae_si": mean_absolute_error([bl_si], [si]),
                    "mdae_sh": median_absolute_error([bl_sh], [sh]),
                    "mdae_si": median_absolute_error([bl_si], [si]),
                }
            )

# Convert lists to Polars DataFrame and write to CSV
pl.DataFrame(metrics_list_reg).write_csv(
    "./analysis_out/diversity_indices_plot_reg.csv"
)
pl.DataFrame(metrics_list_w).write_csv("./analysis_out/diversity_indices_plot_w.csv")

ground_truth = pl.read_csv("prep_out/ground_truth_diversity_indices.csv")
estimate_reg = pl.read_csv("prep_out/estimated_diversity_indices_temporal_reg.csv")
estimate_w = pl.read_csv("prep_out/estimated_diversity_indices_temporal_w.csv")

rs = [1, 2, 3, 4, 5]

metrics_list_reg = []
metrics_list_w = []

for file in ground_truth["filename"].unique(maintain_order=True):
    bl_f = ground_truth.filter(pl.col("filename") == file)
    bl_sh = bl_f["shannon"].mean()
    bl_si = bl_f["simpson"].mean()

    es_r = estimate_reg.filter(pl.col("filename") == file)
    es_w = estimate_w.filter(pl.col("filename") == file)

    for r in rs:
        es_r_r = es_r.filter(pl.col("r") == r)
        es_w_r = es_w.filter(pl.col("r") == r)

        sh, si = es_r_r.select(["shannon", "simpson"]).row(0)
        metrics_list_reg.append(
            {
                "filename": file,
                "r": r,
                "mae_sh": mean_absolute_error([bl_sh], [sh]),
                "mae_si": mean_absolute_error([bl_si], [si]),
                "mdae_sh": median_absolute_error([bl_sh], [sh]),
                "mdae_si": median_absolute_error([bl_si], [si]),
            }
        )

        sh, si = es_w_r.select(["shannon", "simpson"]).row(0)
        metrics_list_w.append(
            {
                "filename": file,
                "r": r,
                "mae_sh": mean_absolute_error([bl_sh], [sh]),
                "mae_si": mean_absolute_error([bl_si], [si]),
                "mdae_sh": median_absolute_error([bl_sh], [sh]),
                "mdae_si": median_absolute_error([bl_si], [si]),
            }
        )

# Convert lists to Polars DataFrame and write to CSV
pl.DataFrame(metrics_list_reg).write_csv(
    "./analysis_out/diversity_indices_temporal_reg.csv"
)
pl.DataFrame(metrics_list_w).write_csv(
    "./analysis_out/diversity_indices_temporal_w.csv"
)

ground_truth = pl.read_csv("prep_out/ground_truth_d_index.csv")
estimate_reg = pl.read_csv("prep_out/estimated_d_index_sample_reg.csv")
estimate_w = pl.read_csv("prep_out/estimated_d_index_sample_w.csv")

rs = [1, 2, 3, 4, 5]
sample_times = [0, 100, 200, 300, 400, 500, 600]

metrics_reg = []
metrics_w = []

# Process each unique filename
for file in ground_truth["filename"].unique(maintain_order=True):
    bl_f = ground_truth.filter(pl.col("filename") == file)
    es_r = estimate_reg.filter(pl.col("filename") == file)
    es_w = estimate_w.filter(pl.col("filename") == file)

    for t in ground_truth["type_id"].unique(maintain_order=True):
        bl_t = bl_f.filter(pl.col("type_id") == t)
        es_r_t = es_r.filter(pl.col("type_id") == t)
        es_w_t = es_w.filter(pl.col("type_id") == t)

        for st in sample_times:
            es_st_r = es_r_t.filter(pl.col("sample_time") == st)
            es_st_w = es_w_t.filter(pl.col("sample_time") == st)

            bl_st = (
                bl_t.filter(pl.col("sample_time") == st)
                .select([str(i) for i in range(9)])
                .to_numpy()
                .flatten()
            )

            for sample_id in es_st_r["sample_id"].unique(maintain_order=True):
                es_r_s = es_st_r.filter(pl.col("sample_id") == sample_id)

                for r in rs:
                    es_r_r = (
                        es_r_s.filter(pl.col("r") == r)
                        .select([str(i) for i in range(9)])
                        .to_numpy()
                        .flatten()
                    )

                    metrics_reg.append(
                        {
                            "filename": file,
                            "sample_time": st,
                            "r": r,
                            "sample_id": sample_id,
                            "type_id": t,
                            "mae": mean_absolute_error(bl_st, es_r_r),
                            "mdae": median_absolute_error(bl_st, es_r_r),
                        }
                    )

            for sample_id in es_st_w["sample_id"].unique(maintain_order=True):
                es_w_s = es_st_w.filter(pl.col("sample_id") == sample_id)

                for r in rs:
                    es_w_r = (
                        es_w_s.filter(pl.col("r") == r)
                        .select([str(i) for i in range(9)])
                        .to_numpy()
                        .flatten()
                    )

                    metrics_w.append(
                        {
                            "filename": file,
                            "sample_time": st,
                            "r": r,
                            "sample_id": sample_id,
                            "type_id": t,
                            "mae": mean_absolute_error(bl_st, es_w_r),
                            "mdae": median_absolute_error(bl_st, es_w_r),
                        }
                    )

# Convert lists to Polars DataFrame and save
pl.DataFrame(metrics_reg).write_csv("./analysis_out/d_index_sample_reg.csv")
pl.DataFrame(metrics_w).write_csv("./analysis_out/d_index_sample_w.csv")

ground_truth = pl.read_csv("prep_out/ground_truth_d_index.csv")
estimate_reg = pl.read_csv("prep_out/estimated_d_index_plot_reg.csv")
estimate_w = pl.read_csv("prep_out/estimated_d_index_plot_w.csv")

rs = [1, 2, 3, 4, 5]
sample_times = [0, 100, 200, 300, 400, 500, 600]

# Initialize lists for storing results
metrics_reg_list = []
metrics_w_list = []

# Iterate over unique filenames
for file in ground_truth["filename"].unique(maintain_order=True):
    bl_f = ground_truth.filter(pl.col("filename") == file)
    es_r = estimate_reg.filter(pl.col("filename") == file)
    es_w = estimate_w.filter(pl.col("filename") == file)

    for st in sample_times:
        es_st_r = es_r.filter(pl.col("sample_time") == st)
        es_st_w = es_w.filter(pl.col("sample_time") == st)
        bl_st = bl_f.filter(pl.col("sample_time") == st)

        for t in ground_truth["type_id"].unique(maintain_order=True):
            bl = (
                bl_st.filter(pl.col("type_id") == t)
                .select([str(i) for i in range(9)])
                .to_numpy()
                .flatten()
            )

            es_r_t = es_st_r.filter(pl.col("type_id") == t)
            es_w_t = es_st_w.filter(pl.col("type_id") == t)

            for r in rs:
                es_r_r = (
                    es_r_t.filter(pl.col("r") == r)
                    .select([str(i) for i in range(9)])
                    .to_numpy()
                    .flatten()
                )
                es_w_r = (
                    es_w_t.filter(pl.col("r") == r)
                    .select([str(i) for i in range(9)])
                    .to_numpy()
                    .flatten()
                )

                # Compute metrics and store in list
                metrics_reg_list.append(
                    {
                        "filename": file,
                        "sample_time": st,
                        "r": r,
                        "type_id": t,
                        "mae": mean_absolute_error(bl, es_r_r),
                        "mdae": median_absolute_error(bl, es_r_r),
                    }
                )

                metrics_w_list.append(
                    {
                        "filename": file,
                        "sample_time": st,
                        "r": r,
                        "type_id": t,
                        "mae": mean_absolute_error(bl, es_w_r),
                        "mdae": median_absolute_error(bl, es_w_r),
                    }
                )

# Convert lists to Polars DataFrames
metrics_reg = pl.DataFrame(metrics_reg_list)
metrics_w = pl.DataFrame(metrics_w_list)

# Save to CSV
metrics_reg.write_csv("./analysis_out/d_index_plot_reg.csv")
metrics_w.write_csv("./analysis_out/d_index_plot_w.csv")

ground_truth = pl.read_csv("prep_out/ground_truth_d_index.csv")
estimate_reg = pl.read_csv("prep_out/estimated_d_index_temporal_reg.csv")
estimate_w = pl.read_csv("prep_out/estimated_d_index_temporal_w.csv")

rs = [1, 2, 3, 4, 5]

# Initialize lists for storing results
metrics_reg_list = []
metrics_w_list = []

# Iterate over unique filenames
for file in ground_truth["filename"].unique(maintain_order=True):
    bl_f = ground_truth.filter(pl.col("filename") == file)
    es_r = estimate_reg.filter(pl.col("filename") == file)
    es_w = estimate_w.filter(pl.col("filename") == file)

    for t in ground_truth["type_id"].unique(maintain_order=True):
        # Compute mean of ground_truth for type_id == 0
        bl = (
            bl_f.filter(pl.col("type_id") == 0)
            .select([str(i) for i in range(9)])
            .mean()
            .to_numpy()
            .flatten()
        )

        es_r_t = es_r.filter(pl.col("type_id") == t)
        es_w_t = es_w.filter(pl.col("type_id") == t)

        for r in rs:
            # Compute mean for each r
            es_r_r = (
                es_r_t.filter(pl.col("r") == r)
                .select([str(i) for i in range(9)])
                .mean()
                .to_numpy()
                .flatten()
            )
            es_w_r = (
                es_w_t.filter(pl.col("r") == r)
                .select([str(i) for i in range(9)])
                .mean()
                .to_numpy()
                .flatten()
            )

            # Compute metrics and store in list
            metrics_reg_list.append(
                {
                    "filename": file,
                    "r": r,
                    "type_id": t,
                    "mae": mean_absolute_error(bl, es_r_r),
                    "mdae": median_absolute_error(bl, es_r_r),
                }
            )

            metrics_w_list.append(
                {
                    "filename": file,
                    "r": r,
                    "type_id": t,
                    "mae": mean_absolute_error(bl, es_w_r),
                    "mdae": median_absolute_error(bl, es_w_r),
                }
            )

# Convert lists to Polars DataFrames
metrics_reg = pl.DataFrame(metrics_reg_list)
metrics_w = pl.DataFrame(metrics_w_list)

# Save to CSV
metrics_reg.write_csv("./analysis_out/d_index_temporal_reg.csv")
metrics_w.write_csv("./analysis_out/d_index_temporal_w.csv")
