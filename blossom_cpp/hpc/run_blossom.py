import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np


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


def run_blossom(params, seed):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
        for key, value in params.items():
            tmpfile.write(f"{key}={value}\n")
        config_path = tmpfile.name

    try:
        subprocess.run(
            ["./bin/blossom", "--config", config_path, "--seed", str(seed)],
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
            run_blossom(params, seed)
        except Exception as e:
            print(f"[ERROR] Seed {seed} failed with: {e}")

    with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
        executor.map(task, zip(range(len(seeds)), seeds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
