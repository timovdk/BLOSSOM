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
# OR install dependencies manually
# pip install numpy pandas pyarrow

cp -r $HOME/BLOSSOM/blossom_cpp "$TMPDIR"

cd $TMPDIR/blossom_cpp
make clean
make

# Run BLOSSOM
echo "Running BLOSSOM..."
python ./hpc/run_blossom.py --n_trials 10 --n_jobs 10 --seed 135432

# Should copy to deepstore (maybe with rsync/rclone?)
# Make dir (if not exists) and copy input configs
mkdir -p $HOME/blossom_out/configs
mv $TMPDIR/blossom_cpp/configs/*.props $HOME/blossom_out/configs/
# Make dir (if not exists) and copy output parquets
mkdir -p $HOME/blossom_out/outputs
mv $TMPDIR/blossom_cpp/outputs/*.parquet $HOME/blossom_out/outputs/

exit 0
