#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --time=48:00:00

# Load modules (modify these as needed)
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

source $HOME/venvs/blossom/bin/activate

cp -r $HOME/BLOSSOM/blossom_cpp "$TMPDIR"

cd $TMPDIR/blossom_cpp
make clean
make

# Run your Optuna script (which should connect to this DB)
echo "Running BLOSSOM..."
python ./hpc/run_blossom.py --n_trials 500 --n_jobs 23 --seed 135432

cp -r $TMPDIR/blossom_cpp $HOME/

cleanup
