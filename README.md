# Soil Simulator (SoSi)
A soil biota agent based model (ABM) based on [Repast4Py](https://repast.github.io/repast4py.site/index.html)
The sosi folder contains the model itself, and the experiments folder contains inputs, outputs, and notebooks for visualizations and analysis.


<img src="output.png" alt="A 2D line diagram with ticks from 0 to 1000 on the x axis, and counts of agents in the model from 0 to 4000 on the y axis. There are twelve coloroued lines that show the evolution of counts per organism group over time." width="400"/>

## Install/Run instructions
Tested for Windows with WSL2 Ubuntu 22.04, and Python 3.10.7

1. Install MPI (in this case [mpich](https://www.mpich.org/), but openMPI should also work fine (not tested)): `sudo apt install mpich`
2. Install Python libraries (consider making a venv) `env CC=mpicxx pip3 install -r requirements.txt`
     - If you encounter issues with Repast4Py, try following the [Repast4Py install instructions](https://repast.github.io/repast4py.site/index.html)
4. Run the model from the `./experiments` folder with the command `mpirun -n 1 python3 main.py inputs/full_run.yaml`, where `-n <#_processes` sets the number of processes that are used. 
