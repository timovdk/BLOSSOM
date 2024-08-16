#!/bin/bash
# parameters for slurm
#SBATCH -c 1
#SBATCH -n 4
#SBATCH --mem=6gb
#SBATCH --mail-type=END,FAIL
#SBATCH --time=48:00:00

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

# Run the job
mpirun python3 blossom_2d.py inputs/data_collection_2d.yaml

# Copy output back to the master, comment with # if not used
mv outputs ${SLURM_SUBMIT_DIR}

# we step out of the scratch directory and remove it
cd ..
rm -rf ${SLURM_JOBID}

# happy end
exit 0

