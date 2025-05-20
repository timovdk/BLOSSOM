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

# Trap to copy results on unexpected exit
trap 'echo "Script exiting, saving crash snapshot..."; rsync -a "$TMPDIR/blossom_cpp/" "$HOME/blossom_cpp_crashdump/"' EXIT

# Run BLOSSOM
echo "Running BLOSSOM..."
python ./hpc/run_blossom.py --n_trials 47 --n_jobs 47 --seed 135432

# MAKE SURE WE COPY THIS TO THE RIGHT PLACE, THIS IS FOR SURE BIG DATA! (~5GB per full run (oops))
cp -r $TMPDIR/blossom_cpp $HOME/

exit 0
