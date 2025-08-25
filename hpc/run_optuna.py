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


def evaluate(params, num_trials, seed, impossible_trial):
    logs = []
    if impossible_trial:
        return logs
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
        for key, value in params.items():
            tmpfile.write(f"{key}={value}\n")
        config_path = tmpfile.name

    try:
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
    impossible_trial = False

    for i in orgs:
        biomass_max_base = base_params[f"organism_{i}_biomass_max"]
        age_max_base = base_params[f"organism_{i}_age_max"]
        biomass_repr_base = base_params[f"organism_{i}_biomass_reproduction"]
        age_repr_base = base_params[f"organism_{i}_age_reproduction"]
        k_base = base_params[f"organism_{i}_k"]

        biomass_max = trial.suggest_float(
            f"organism_{i}_biomass_max", biomass_max_base * 0.75, biomass_max_base * 1.25
        )
        age_max = trial.suggest_int(
            f"organism_{i}_age_max",
            math.floor(age_max_base * 0.75),
            math.ceil(age_max_base * 1.25),
        )

        biomass_reproduction = trial.suggest_float(
            f"organism_{i}_biomass_reproduction",
            biomass_repr_base * 0.75,
            biomass_repr_base * 1.25,
        )

        if biomass_reproduction > biomass_max:
            impossible_trial = True

        age_reproduction = trial.suggest_int(
            f"organism_{i}_age_reproduction",
            math.floor(age_repr_base * 0.75),
            math.ceil(age_repr_base * 1.25),
        )

        if age_reproduction > age_max:
            impossible_trial = True

        k = trial.suggest_float(f"organism_{i}_k", k_base * 0.75, k_base * 1.25)

        params[f"organism_{i}_biomass_max"] = round(biomass_max, 8)
        params[f"organism_{i}_biomass_reproduction"] = round(biomass_reproduction, 8)
        params[f"organism_{i}_age_max"] = age_max
        params[f"organism_{i}_age_reproduction"] = age_reproduction
        params[f"organism_{i}_k"] = k

    logs = evaluate(params, num_trials=num_trials, seed=seed, impossible_trial=impossible_trial)

    # For impossible trials or simulations that produced no logs, return a tiny non-zero value.
    # This penalizes invalid configurations while keeping the search space fully visible to 
    # some samplers (e.g., CMA-ES), helping the sampler explore more effectively.
    if impossible_trial or len(logs) == 0:
        return 1e-6

    # We use area under the curve (AUC) as the objective.
    # The maximum number of ticks is 1000, and we log every 50 ticks.
    # Therefore, we have at most 21 logs (0, 50, 100, ..., 1000).
    # Each log can have at most 9 organisms surviving.
    # The maximum number of survivors across all logs is 21 * 9 = 189.
    # We normalize the total survivors by dividing it by the maximum possible value.
    # This ensures that the objective value is in the range [0, 1].
    total_survivors = 0
    for log in logs:
        tick = log["tick"]
        survivors = log["survivors"]
        trial.report(survivors, step=tick)
        total_survivors += log["survivors"]

    maximum_survivors = 21 * 9  
    
    return total_survivors / maximum_survivors  # Returns value in [0, 1]


parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=20)
parser.add_argument("--n_jobs", type=int, default=2)
args = parser.parse_args()

storage_url = "postgresql://localhost:5433/optuna_study"

tpe_sampler = optuna.samplers.TPESampler(
    n_startup_trials=300,
    n_ei_candidates=64,
    consider_magic_clip=True,
    consider_endpoints=True,
    constant_liar=True,
    seed=42,
)

cma_sampler = optuna.samplers.CmaEsSampler(n_startup_trials=200, popsize=64, restart_strategy="ipop", seed=42)

study = optuna.create_study(
    #sampler=tpe_sampler,
    sampler=cma_sampler,
    direction="maximize",
    study_name=f"[{datetime.datetime.now().strftime('%b-%d-%H-%M')}] 9 Organisms, Regular Objective",
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
