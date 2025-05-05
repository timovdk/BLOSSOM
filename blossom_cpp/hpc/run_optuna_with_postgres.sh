#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --time=10:00:00

# Load modules (modify these as needed)
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load PostgreSQL/16.4-GCCcore-13.3.0

source $HOME/venvs/optuna/bin/activate

PG_DATA="$HOME/pgsql/data"
LOGFILE="$HOME/pgsql/postgres.log"

# Optional: ensure PostgreSQL uses a local socket in home dir to avoid conflicts
export PGHOST=$PG_DATA
export PGPORT=5433  # Use a non-standard port if needed

# Start PostgreSQL server
echo "Starting PostgreSQL..."
pg_ctl -D $PG_DATA -l $LOGFILE -o "-p $PGPORT" start

# Wait to ensure the server has started
sleep 5

# Create the DB if it doesn't exist
createdb -p $PGPORT optuna_study

mkdir -p /scratch-shared/$USER
cp -r $HOME/BLOSSOM/blossom_cpp /scratch-shared/$USER

cd /scratch-shared/$USER/blossom_cpp

# Run your Optuna script (which should connect to this DB)
echo "Running Optuna optimization..."
python run_optuna.py --n_trials 1000 --n_jobs 4

# Stop PostgreSQL server cleanly
echo "Stopping PostgreSQL..."
$PG_BIN/pg_ctl -D $PG_DATA stop
