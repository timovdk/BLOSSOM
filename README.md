# Soil Simulator (SoSi)
![A 2D line diagram with ticks from 0 to 1000 on the x axis, and counts of agents in the model from 0 to 4000 on the y axis. There are twelve coloroued lines that show the evolution of counts per organism group over time.](output.png "Output of SoSi, visualized using the code in viz.ipynb")

## Install/Run instructions
Tested for Windows with WSL2 Ubuntu 22.04, and Python 3.10.7

1. Install MPI (in this case mpich, but openMPI should also work fine (not tested)): `sudo apt install mpich`
2. Install Python libraries `env CC=mpicxx pip install -r requirements.txt`
3. Run the model with `mpirun -n 1 python main.py full_run.yaml`, where `-n <#_processes` sets the number of processes that are used. 