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
echo "1/16"
mpirun python3 blossom_2d.py inputs/random_1.yaml
mv outputs/random_1.csv ${SLURM_SUBMIT_DIR}/outputs

echo "2/16"
mpirun python3 blossom_2d.py inputs/clustered_1.yaml
mv outputs/clustered_1.csv ${SLURM_SUBMIT_DIR}/outputs

echo "3/16"
mpirun python3 blossom_2d.py inputs/random_1_2.yaml
mv outputs/random_1_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "4/16"
mpirun python3 blossom_2d.py inputs/clustered_1_2.yaml
mv outputs/clustered_1_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "5/16"
mpirun python3 blossom_2d.py inputs/random_2.yaml
mv outputs/random_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "6/16"
mpirun python3 blossom_2d.py inputs/clustered_2.yaml
mv outputs/clustered_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "7/16"
mpirun python3 blossom_2d.py inputs/random_2_2.yaml
mv outputs/random_2_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "8/16"
mpirun python3 blossom_2d.py inputs/clustered_2_2.yaml
mv outputs/clustered_2_2.csv ${SLURM_SUBMIT_DIR}/outputs

echo "9/16"
mpirun python3 blossom_2d.py inputs/random_3.yaml
mv outputs/random_3.csv ${SLURM_SUBMIT_DIR}/outputs

echo "10/16"
mpirun python3 blossom_2d.py inputs/random_4.yaml
mv outputs/random_4.csv ${SLURM_SUBMIT_DIR}/outputs

echo "11/16"
mpirun python3 blossom_2d.py inputs/random_5.yaml
mv outputs/random_5.csv ${SLURM_SUBMIT_DIR}/outputs

echo "12/16"
mpirun python3 blossom_2d.py inputs/random_6.yaml
mv outputs/random_6.csv ${SLURM_SUBMIT_DIR}/outputs

echo "13/16"
mpirun python3 blossom_2d.py inputs/clustered_3.yaml
mv outputs/clustered_3.csv ${SLURM_SUBMIT_DIR}/outputs

echo "14/16"
mpirun python3 blossom_2d.py inputs/clustered_4.yaml
mv outputs/clustered_4.csv ${SLURM_SUBMIT_DIR}/outputs

echo "15/16"
mpirun python3 blossom_2d.py inputs/clustered_5.yaml
mv outputs/clustered_5.csv ${SLURM_SUBMIT_DIR}/outputs

echo "16/16"
mpirun python3 blossom_2d.py inputs/clustered_6.yaml
mv outputs/clustered_6.csv ${SLURM_SUBMIT_DIR}/outputs

# we step out of the scratch directory and remove it
cd ..
rm -rf ${SLURM_JOBID}

# happy end
exit 0

