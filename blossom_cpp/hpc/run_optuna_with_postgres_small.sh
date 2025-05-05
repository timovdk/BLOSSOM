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

source $HOME/BLOSSOM/.venv/bin/activate

PG_DATA="$HOME/pgsql/data"
LOGFILE="$HOME/pgsql/postgres.log"

# Optional: ensure PostgreSQL uses a local socket in home dir to avoid conflicts
export PGHOST=127.0.0.1
export PGPORT=5433  # Use a non-standard port if needed

# Start PostgreSQL server
echo "Starting PostgreSQL..."
pg_ctl -D $PG_DATA -l $LOGFILE -o "-p $PGPORT" start

# Wait to ensure the server has started
sleep 5

# Check if PostgreSQL is ready
pg_isready -p $PGPORT
if [ $? -ne 0 ]; then
    echo "PostgreSQL did not start successfully!"
    exit 1
fi

# Create the DB if it doesn't exist
if ! psql -p $PGPORT -lqt | cut -d \| -f 1 | grep -qw optuna_study; then
    createdb -p $PGPORT optuna_study
    echo "Created the optuna_study database."
else
    echo "Database optuna_study already exists."
fi

mkdir -p /scratch-shared/$USER

rm -rf /scratch-shared/$USER/blossom_cpp
cp -r $HOME/BLOSSOM/blossom_cpp /scratch-shared/$USER

cd /scratch-shared/$USER/blossom_cpp/hpc

# Run your Optuna script (which should connect to this DB)
echo "Running Optuna optimization..."
python run_optuna.py --n_trials 2 --n_jobs 2

# Stop PostgreSQL server cleanly
echo "Stopping PostgreSQL..."
pg_ctl -D $PG_DATA stop
