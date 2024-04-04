from typing import Dict
from mpi4py import MPI
from .model import Model


def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    model.start()
