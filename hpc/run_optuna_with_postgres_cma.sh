#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa
#SBATCH --time=48:00:00

# Load modules (modify these as needed)
module load 2025 PostgreSQL/17.5-GCCcore-14.2.0 Python/3.13.1-GCCcore-14.2.0

source $HOME/venvs/blossom/bin/activate

PG_DATA="$HOME/pgsql/data"
LOGFILE="$HOME/pgsql/postgres.log"

# Optional: ensure PostgreSQL uses a local socket in home dir to avoid conflicts
export PGHOST=127.0.0.1
export PGPORT=5433  # Use a non-standard port if needed

cp -r $HOME/BLOSSOM/blossom "$TMPDIR"
cp -r $HOME/BLOSSOM/hpc "$TMPDIR/blossom"

cd $TMPDIR/blossom
make clean
make

( while true; do
    sleep 60
    pg_isready -p $PGPORT
done ) &

echo "Creating DB..."
srun -n 1 python ./hpc/run_optuna_cma.py --init-db

# Run Optuna script
echo "Running Optuna optimization..."

srun --ntasks=47 --cpus-per-task=1 \
     python ./hpc/run_optuna_cma.py --n_trials 5000 --n_jobs 1 &

wait
