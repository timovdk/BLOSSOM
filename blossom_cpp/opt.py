import itertools
import math
import os
import re
import subprocess
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Helper function to evaluate the simulation for a given parameter set
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
            timeout=600,
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
                print(f"Tick {tick}: {survivors} survivors")  # Debugging output
                logs.append({"tick": tick, "survivors": survivors})

        return logs
    finally:
        os.remove(config_path)


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


# Define the grid search range (±20%, 10% intervals)
def generate_independent_grids(
    base_params,
    n_orgs,
    biomass_max_variation=[-20, -10, 0, 10, 20],
    age_max_variation=[-20, -10, 0, 10, 20],
):
    grid = {}

    # Create a grid for each organism's biomass_max and age_max
    for i in range(n_orgs):  # There are n_orgs organisms
        # Generate the biomass_max and age_max variations for this organism
        biomass_base = base_params[
            f"organism_{i}_biomass_max"
        ]  # biomass_max is the 3rd parameter
        age_max_base = base_params[
            f"organism_{i}_age_max"
        ]  # age_max is the 2nd parameter

        grid[i] = {
            "biomass_max": [
                biomass_base * (1 + p / 100) for p in biomass_max_variation
            ],
            "age_max": [round(age_max_base * (1 + p / 100)) for p in age_max_variation],
        }

    return grid


# Perform grid search and write the results to a CSV file, with survivors for each organism
def grid_search_and_write_to_file(
    base_config_file="base_config.props",
    output_file="organism_gridsearch_results.csv",
    num_trials=1,
    seed=42,
    n_orgs=9,
):
    # Load base configuration from file
    base_params = load_base_config(base_config_file)

    # Generate the independent grid of parameter variations
    grid = generate_independent_grids(
        base_params,
        n_orgs,
        biomass_max_variation=[-10, 0, 10],
        age_max_variation=[-10, 0, 10],
    )

    # Store results
    results = []

    # Iterate over each organism type and its parameter grid
    for i in range(n_orgs):  # There are n_orgs organisms
        organism_results = []

        for biomass_max, age_max in set(
            itertools.product(grid[i]["biomass_max"], grid[i]["age_max"])
        ):
            print(
                f"Evaluating organism {i}: biomass_max={biomass_max}, age_max={age_max}"
            )

            # Set up the parameter set for the current organism's grid combination
            params = base_params.copy()
            params[f"organism_{i}_biomass_max"] = float(
                biomass_max
            )  # Set the biomass_max for this organism
            params[f"organism_{i}_biomass_reproduction"] = float(
                biomass_max / 2
            )  # Set the biomass_max for this organism
            params[f"organism_{i}_age_max"] = int(
                age_max
            )  # Set the age_max for this organism
            params[f"organism_{i}_age_reproduction"] = int(math.floor(age_max / 2))

            # Evaluate the combination
            logs = evaluate(params, num_trials=num_trials, seed=seed)

            for log in logs:
                organism_results.append(
                    [i, biomass_max, age_max, log["tick"], log["survivors"]]
                )

        # Append the organism results to the main results list
        results.extend(organism_results)

    # Write results to CSV file
    df = pd.DataFrame(
        results, columns=["organism", "biomass_max", "age_max", "tick", "survivors"]
    )
    df.to_csv(output_file, index=False)
    print(f"Grid search results saved to {output_file}")


# Create a heatmap from the results for each organism
def create_heatmap_from_results(
    input_file="organism_gridsearch_results.csv",
    output_pdf="heatmap_per_organism.pdf",
    selected_tick=-1,
):
    # Load the results from the CSV
    df = pd.read_csv(input_file)

    # Iterate over each organism and create a heatmap for its results
    for organism in df["organism"].unique():
        organism_df = df[df["organism"] == organism]

        if selected_tick == -1:
            max_tick = organism_df["tick"].max()
            organism_df = organism_df[organism_df["tick"] == max_tick]
        else:
            organism_df = organism_df[organism_df["tick"] == selected_tick]

        # Pivot the DataFrame to make a grid with age_max as rows, biomass_max as columns, and survivors as values
        heatmap_data = organism_df.pivot(
            index="age_max", columns="biomass_max", values="survivors"
        )

        # Create a heatmap for this organism
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            cbar_kws={"label": "Survivors"},
        )

        # Save the heatmap to a PDF
        plt.title(f"Organism {organism} Survivors Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_pdf}_organism_{organism}.pdf")
        plt.close()
        print(f"Heatmap for organism {organism} saved.")


# Run the grid search and create heatmaps
def run_gridsearch_and_heatmap():
    # Perform the grid search and write results to file
    grid_search_and_write_to_file(
        base_config_file="base_config.props",
        output_file="organism_gridsearch_results.csv",
        num_trials=1,
        seed=42,
        n_orgs=1,
    )

    # Create a heatmap from the grid search results
    create_heatmap_from_results(
        input_file="organism_gridsearch_results.csv",
        output_pdf="heatmap_per_organism.pdf",
    )


# Run the entire process
run_gridsearch_and_heatmap()
