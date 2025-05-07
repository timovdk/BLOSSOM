import argparse
import datetime
import math
import os
import re
import subprocess
import tempfile

import optuna


def load_base_config(filename):
    base_params = {}
    string_keys = ["output_dir", "output_file_name"]

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line or line.startswith("#"):  # Skip empty lines and comments
                continue

            key, value = line.split("=")
            key = key.strip()
            value = value.strip()

            # Check if the key contains "preys" or "predators" (case insensitive)
            if (
                "preys" in key.lower()
                or "predators" in key.lower()
                or key in string_keys
            ):
                base_params[key] = value  # Store as string
            else:
                base_params[key] = float(value)  # Store as float for numerical values

    return base_params


def evaluate(params, num_trials, seed):
    logs = []
    # Write the full params dict to a temp config file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
        for key, value in params.items():
            tmpfile.write(f"{key}={value}\n")
        config_path = tmpfile.name

    try:
        # Run simulation
        result = subprocess.run(
            [
                "./bin/blossom",
                "--config",
                config_path,
                "--trials",
                str(num_trials),
                "--seed",
                str(seed),
                "--logging",
                str(0),
            ],
            capture_output=True,
            text=True,
        )

        # Parse the survivors count using regex
        survivors = 0
        tick = 0
        tick_survivor_pattern = re.compile(r"Tick: (\d+) Survivors: (\d+)")
        for line in result.stdout.splitlines():
            # Match the formatted output "Tick:<tick_number> Survivors:<survivor_count>"
            match = tick_survivor_pattern.search(line)
            if match:
                tick = int(match.group(1))
                survivors = int(match.group(2))
                logs.append({"tick": tick, "survivors": survivors})

        return logs
    finally:
        os.remove(config_path)


def objective(
    trial: optuna.trial.FrozenTrial, base_params, orgs, num_trials=1, seed=42
):
    params = base_params.copy()

    for i in orgs:
        biomass_max_base = base_params[f"organism_{i}_biomass_max"]
        age_max_base = base_params[f"organism_{i}_age_max"]
        biomass_repr_base = base_params[f"organism_{i}_biomass_reproduction"]
        age_repr_base = base_params[f"organism_{i}_age_reproduction"]
        k_base = base_params[f"organism_{i}_k"]

        biomass_max = trial.suggest_float(
            f"organism_{i}_biomass_max", biomass_max_base * 0.8, biomass_max_base * 1.2
        )
        age_max = trial.suggest_int(
            f"organism_{i}_age_max",
            math.floor(age_max_base * 0.8),
            math.ceil(age_max_base * 1.2),
        )

        biomass_reproduction = trial.suggest_float(
            f"organism_{i}_biomass_reproduction",
            biomass_repr_base * 0.8,
            min(biomass_repr_base * 1.2, biomass_max),
        )

        age_reproduction = trial.suggest_int(
            f"organism_{i}_age_reproduction",
            math.floor(age_repr_base * 0.8),
            min(math.ceil(age_repr_base * 1.2), age_max),
        )

        k = trial.suggest_float(f"organism_{i}_k", k_base * 0.8, k_base * 1.2)

        params[f"organism_{i}_biomass_max"] = round(biomass_max, 8)
        params[f"organism_{i}_biomass_reproduction"] = round(biomass_reproduction, 8)
        params[f"organism_{i}_age_max"] = age_max
        params[f"organism_{i}_age_reproduction"] = age_reproduction
        params[f"organism_{i}_k"] = k

    logs = evaluate(params, num_trials=num_trials, seed=seed)

    for log in logs:
        tick = log["tick"]
        survivors = log["survivors"]
        trial.report(survivors, step=tick)

    final_log = logs[-1] if logs else {"tick": 0, "survivors": 0}

    return (0.7 * (final_log["tick"] / 1000)) * (0.3 * (final_log["survivors"] / 9))


parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=20)
parser.add_argument("--n_jobs", type=int, default=2)
args = parser.parse_args()

storage_url = "postgresql://localhost:5433/optuna_study"

tpe_sampler = optuna.samplers.TPESampler(
    n_startup_trials=250,
    n_ei_candidates=64,
    multivariate=True,
    consider_magic_clip=True,
    consider_endpoints=True,
    constant_liar=True,
    seed=42,
)

study = optuna.create_study(
    sampler=tpe_sampler,
    direction="maximize",
    study_name=f"{datetime.datetime.now().strftime('%b-%d-%H-%M')}_3_orgs",
    storage=storage_url,
    load_if_exists=True,
)

base_params = load_base_config("base_config.props")

study.optimize(
    lambda trial: objective(
        trial, base_params, orgs=[0, 1, 2, 3, 4, 5, 6, 7, 8], num_trials=1
    ),
    n_trials=args.n_trials,
    n_jobs=args.n_jobs,
)
