#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa
#SBATCH --time=24:00:00

# Load modules (modify these as needed)
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load PostgreSQL/16.4-GCCcore-13.3.0

source $HOME/venvs/blossom/bin/activate

PG_DATA="$HOME/pgsql/data"
LOGFILE="$HOME/pgsql/postgres.log"

# Optional: ensure PostgreSQL uses a local socket in home dir to avoid conflicts
export PGHOST=127.0.0.1
export PGPORT=5433  # Use a non-standard port if needed

# Clean exit on interrupt
function cleanup() {
    echo "Stopping PostgreSQL..."
    pg_ctl -D $PG_DATA -m fast stop
    exit 0
}
trap cleanup SIGINT SIGTERM

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

cp -r $HOME/BLOSSOM/blossom_cpp "$TMPDIR"

cd $TMPDIR/blossom_cpp
make clean
make

( while true; do
    sleep 60
    pg_isready -p $PGPORT
done ) &

# Run your Optuna script (which should connect to this DB)
echo "Running Optuna optimization..."
python ./hpc/run_optuna.py --n_trials 1250 --n_jobs 23

cleanup
