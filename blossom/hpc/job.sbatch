#!/bin/bash
# parameters for slurm
#SBATCH -c 1
#SBATCH -n 4
#SBATCH --mem=6gb
#SBATCH --mail-type=END,FAIL
#SBATCH --time=72:00:00

# Create a directory for this job on the node
cd /local
mkdir ${SLURM_JOBID}
cd ${SLURM_JOBID}
# Copy input and executable to the node
cp -f -r ${SLURM_SUBMIT_DIR}/* .

# load all modules needed
module load openmpi/4.1.5
module load python/3.10.7

# It's nice to have some information logged for debugging
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Number of nodes used        : "$SLURM_NNODES
echo "Number of threads           : "$SLURM_CPUS_PER_TASK
echo "Number of MPI ranks per node: "$SLURM_TASKS_PER_NODE
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Starting worker: "

# Run the job, first the 4 runs for stochasticity analysis, then the remaining.
echo "1/44"
mpirun python3 blossom_2d.py inputs/random_1.yaml
mv outputs/random_1.csv ${SLURM_SUBMIT_DIR}/outputs

echo "2/44"
mpirun python3 blossom_2d.py inputs/clustered_1.yaml
mv outputs/clustered_1.csv ${SLURM_SUBMIT_DIR}/outputs

echo "3/44"
mpirun python3 blossom_2d.py inputs/random_1_2.yaml
mv outputs/random_1_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "4/44"
mpirun python3 blossom_2d.py inputs/clustered_1_2.yaml
mv outputs/clustered_1_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "5/44"
mpirun python3 blossom_2d.py inputs/random_2.yaml
mv outputs/random_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "6/44"
mpirun python3 blossom_2d.py inputs/clustered_2.yaml
mv outputs/clustered_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "7/44"
mpirun python3 blossom_2d.py inputs/random_2_2.yaml
mv outputs/random_2_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "8/44"
mpirun python3 blossom_2d.py inputs/clustered_2_2.yaml
mv outputs/clustered_2_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "9/44"
mpirun python3 blossom_2d.py inputs/random_3.yaml
mv outputs/random_3.csv ${SLURM_SUBMIT_DIR}/outputs

echo "10/44"
mpirun python3 blossom_2d.py inputs/random_4.yaml
mv outputs/random_4.csv ${SLURM_SUBMIT_DIR}/outputs

echo "11/44"
mpirun python3 blossom_2d.py inputs/random_5.yaml
mv outputs/random_5.csv ${SLURM_SUBMIT_DIR}/outputs

echo "12/44"
mpirun python3 blossom_2d.py inputs/random_6.yaml
mv outputs/random_6.csv ${SLURM_SUBMIT_DIR}/outputs

echo "13/44"
mpirun python3 blossom_2d.py inputs/random_7.yaml
mv outputs/random_7.csv ${SLURM_SUBMIT_DIR}/outputs

echo "14/44"
mpirun python3 blossom_2d.py inputs/random_8.yaml
mv outputs/random_8.csv ${SLURM_SUBMIT_DIR}/outputs

echo "15/44"
mpirun python3 blossom_2d.py inputs/random_9.yaml
mv outputs/random_9.csv ${SLURM_SUBMIT_DIR}/outputs

echo "16/44"
mpirun python3 blossom_2d.py inputs/random_10.yaml
mv outputs/random_10.csv ${SLURM_SUBMIT_DIR}/outputs

echo "17/44"
mpirun python3 blossom_2d.py inputs/random_11.yaml
mv outputs/random_11.csv ${SLURM_SUBMIT_DIR}/outputs

echo "18/44"
mpirun python3 blossom_2d.py inputs/random_12.yaml
mv outputs/random_12.csv ${SLURM_SUBMIT_DIR}/outputs

echo "19/44"
mpirun python3 blossom_2d.py inputs/random_13.yaml
mv outputs/random_13.csv ${SLURM_SUBMIT_DIR}/outputs

echo "20/44"
mpirun python3 blossom_2d.py inputs/random_14.yaml
mv outputs/random_14.csv ${SLURM_SUBMIT_DIR}/outputs

echo "21/44"
mpirun python3 blossom_2d.py inputs/random_15.yaml
mv outputs/random_15.csv ${SLURM_SUBMIT_DIR}/outputs

echo "22/44"
mpirun python3 blossom_2d.py inputs/random_16.yaml
mv outputs/random_16.csv ${SLURM_SUBMIT_DIR}/outputs

echo "23/44"
mpirun python3 blossom_2d.py inputs/random_17.yaml
mv outputs/random_17.csv ${SLURM_SUBMIT_DIR}/outputs

echo "24/44"
mpirun python3 blossom_2d.py inputs/random_18.yaml
mv outputs/random_18.csv ${SLURM_SUBMIT_DIR}/outputs

echo "25/44"
mpirun python3 blossom_2d.py inputs/random_19.yaml
mv outputs/random_19.csv ${SLURM_SUBMIT_DIR}/outputs

echo "26/44"
mpirun python3 blossom_2d.py inputs/random_20.yaml
mv outputs/random_20.csv ${SLURM_SUBMIT_DIR}/outputs

echo "27/44"
mpirun python3 blossom_2d.py inputs/clustered_3.yaml
mv outputs/clustered_3.csv ${SLURM_SUBMIT_DIR}/outputs

echo "28/44"
mpirun python3 blossom_2d.py inputs/clustered_4.yaml
mv outputs/clustered_4.csv ${SLURM_SUBMIT_DIR}/outputs

echo "29/44"
mpirun python3 blossom_2d.py inputs/clustered_5.yaml
mv outputs/clustered_5.csv ${SLURM_SUBMIT_DIR}/outputs

echo "30/44"
mpirun python3 blossom_2d.py inputs/clustered_6.yaml
mv outputs/clustered_6.csv ${SLURM_SUBMIT_DIR}/outputs

echo "31/44"
mpirun python3 blossom_2d.py inputs/clustered_7.yaml
mv outputs/clustered_7.csv ${SLURM_SUBMIT_DIR}/outputs

echo "32/44"
mpirun python3 blossom_2d.py inputs/clustered_8.yaml
mv outputs/clustered_8.csv ${SLURM_SUBMIT_DIR}/outputs

echo "33/44"
mpirun python3 blossom_2d.py inputs/clustered_9.yaml
mv outputs/clustered_9.csv ${SLURM_SUBMIT_DIR}/outputs

echo "34/44"
mpirun python3 blossom_2d.py inputs/clustered_10.yaml
mv outputs/clustered_10.csv ${SLURM_SUBMIT_DIR}/outputs

echo "35/44"
mpirun python3 blossom_2d.py inputs/clustered_11.yaml
mv outputs/clustered_11.csv ${SLURM_SUBMIT_DIR}/outputs

echo "36/44"
mpirun python3 blossom_2d.py inputs/clustered_12.yaml
mv outputs/clustered_12.csv ${SLURM_SUBMIT_DIR}/outputs

echo "37/44"
mpirun python3 blossom_2d.py inputs/clustered_13.yaml
mv outputs/clustered_13.csv ${SLURM_SUBMIT_DIR}/outputs

echo "38/44"
mpirun python3 blossom_2d.py inputs/clustered_14.yaml
mv outputs/clustered_14.csv ${SLURM_SUBMIT_DIR}/outputs

echo "39/44"
mpirun python3 blossom_2d.py inputs/clustered_15.yaml
mv outputs/clustered_15.csv ${SLURM_SUBMIT_DIR}/outputs

echo "40/44"
mpirun python3 blossom_2d.py inputs/clustered_16.yaml
mv outputs/clustered_16.csv ${SLURM_SUBMIT_DIR}/outputs

echo "41/44"
mpirun python3 blossom_2d.py inputs/clustered_17.yaml
mv outputs/clustered_17.csv ${SLURM_SUBMIT_DIR}/outputs

echo "42/44"
mpirun python3 blossom_2d.py inputs/clustered_18.yaml
mv outputs/clustered_18.csv ${SLURM_SUBMIT_DIR}/outputs

echo "43/44"
mpirun python3 blossom_2d.py inputs/clustered_19.yaml
mv outputs/clustered_19.csv ${SLURM_SUBMIT_DIR}/outputs

echo "44/44"
mpirun python3 blossom_2d.py inputs/clustered_20.yaml
mv outputs/clustered_20.csv ${SLURM_SUBMIT_DIR}/outputs

# we step out of the scratch directory and remove it
cd ..
rm -rf ${SLURM_JOBID}

# happy end
exit 0

