import argparse
import glob
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

dtype_agent = np.dtype(
    [
        ("id", "u4"),
        ("biomass", "f4"),
        ("tick", "u2"),
        ("x", "u2"),
        ("y", "u2"),
        ("type", "u1"),
        ("age", "u1"),
    ]
)

dtype_agent_small = np.dtype(
    [
        ("tick", "u2"),
        ("x", "u2"),
        ("y", "u2"),
        ("type", "u1"),
    ]
)

dtype_som = np.dtype(
    [
        ("som_value", "f4"),
        ("tick", "u2"),
        ("x", "u2"),
        ("y", "u2"),
    ]
)


def bin_to_parquet(idx, d_agent, filename, output_dir):
    for sub in ["agent/", "som/"]:
        files = glob.glob(output_dir + sub + f"{str(idx)}_" + filename + "_*_*.bin")

        log_files = []

        for file in files:
            base = os.path.basename(file)
            match = re.match(r"(\d+)_" + filename + r"_(\d+)_(\d+)\.bin", base)
            if match:
                _, _, rotation_id = match.groups()
                log_files.append((int(rotation_id), file))

        print(f"Processing setup {str(idx)} with {len(log_files)} rotated files...")
        log_files.sort()
        dfs = []
        for _, path in log_files:
            data = np.fromfile(path, dtype=d_agent if sub == "agent/" else dtype_som)
            df = pd.DataFrame(data)
            dfs.append(df)

        full_df = pd.concat(dfs)
        output_filename = f"{filename}_{str(idx)}{'_SOM' if sub == 'som/' else ''}.parquet"
        full_df.to_parquet(output_dir + output_filename, compression="zstd")

        # Clean up .bin files
        for _, path in log_files:
            os.remove(path)
            print(f"Deleted {path}")


def load_base_config(filename):
    base_params = {}
    string_keys = ["output_dir", "output_file_name"]

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            key, value = line.split("=")
            key = key.strip()
            value = value.strip()

            if (
                "preys" in key.lower()
                or "predators" in key.lower()
                or key in string_keys
            ):
                base_params[key] = value
            else:
                base_params[key] = float(value)

    return base_params


def run_blossom(params, seed, extended_log):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
        for key, value in params.items():
            tmpfile.write(f"{key}={value}\n")
        config_path = tmpfile.name

    try:
        subprocess.run(
            [
                "./bin/blossom",
                "--config",
                config_path,
                "--seed",
                str(seed),
                "--extended_logging",
                str(int(extended_log)),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        os.remove(config_path)


def prepare_parameters(id, base_params):
    params = base_params.copy()
    params["output_file_name"] = f"{id}_{base_params['output_file_name']}"

    return params


def main(args):
    rng = np.random.default_rng(args.seed)

    seeds = rng.integers(0, 2**16, size=args.n_trials).tolist()

    base_params = load_base_config("optimized_config.props")

    def task(idx_seed):
        idx, seed = idx_seed
        try:
            params = prepare_parameters(idx, base_params)
            run_blossom(params, seed, args.extended_logs)
            bin_to_parquet(
                idx,
                dtype_agent if args.extended_logs else dtype_agent_small,
                base_params["output_file_name"],
                base_params["output_dir"],
            )
        except Exception as e:
            print(f"[ERROR] Seed {seed} failed with: {e}")

    with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
        executor.map(task, zip(range(len(seeds)), seeds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extended_logs", type=bool, default=False)
    args = parser.parse_args()

    main(args)
