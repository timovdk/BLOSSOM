# BLOSSOM - BioLogical Organism Simulation in SOil Model
A soil biota agent based model (ABM) based on [Repast4Py](https://repast.github.io/repast4py.site/index.html)
The `./blossom` folder contains the model itself, input files, and output files. The `./experiments` folder contains notebooks for visualizations and analysis.


<img src="output.png" alt="A 2D line diagram with ticks from 0 to 1000 on the x axis, and counts of agents in the model from 0 to 4000 on the y axis. There are twelve coloroued lines that show the evolution of counts per organism group over time." width="400"/>

## Install/Run instructions
Tested for Windows with [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) Ubuntu 22.04, and Python 3.10.7

1. Install MPI (in this case [mpich](https://www.mpich.org/), but openMPI should also work fine (not tested)): `sudo apt install mpich`
2. Install Python libraries (consider making a venv) `env CC=mpicxx pip3 install -r requirements.txt`
     - If you encounter issues with Repast4Py, try following the [Repast4Py install instructions](https://repast.github.io/repast4py.site/index.html)
4. Run the model from the `./blossom` folder with the command `mpirun -n 2 python3 blossom.py ./inputs/run.yaml`, where `-n <#_processes>` sets the number of processes that are used. NOTE: `-n 1` is not supported!
