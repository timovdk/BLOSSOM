# BLOSSOM - BioLOgical Simulation in SOil Model
A soil biota agent based model (ABM) based on [Repast4Py](https://repast.github.io/repast4py.site/index.html)

The `./blossom` folder contains the model itself, notebook to create trophic networks, input files, and output files. The `./blossom_viz` folder contains notebooks for model run visualizations. The `./experiments` folder contains notebooks for analysis. The `./figures` folder contains notebooks used to create the thesis figures, including result and discussion visualizations.

![alt text](random_1.gif)

## Install/Run instructions
Tested for Windows with [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) Ubuntu 22.04, and Python 3.10.7. 

1. Install MPI (in this case [mpich](https://www.mpich.org/), but openMPI should also work fine (not tested)): `sudo apt install mpich`
2. Install Python libraries to run the BLOSSOM model (consider making a venv) `env CC=mpicxx pip3 install repast4py scikit-learn`
     - If you encounter issues with Repast4Py, try following the [Repast4Py install instructions](https://repast.github.io/repast4py.site/index.html)
3. Install Python libraries to run the Jupyter Notebooks (consider making a venv) `pip3 install jupyter pandas matplotlib seaborn statsmodels`
4. Run the model from the `./blossom` folder with the command `mpirun -n 4 python3 blossom_2d.py ./inputs/run_2d.yaml`, where `-n <#_processes>` sets the number of processes that are used. NOTE: `-n 1` is not supported!
