from typing import Tuple
import numpy as np
from repast4py import core

from repast4py.space import DiscretePoint as dpt


class OrganismGroup(core.Agent):
    OFFSETS = np.array([-1, 0, 1])

    def __init__(self, local_id: int, type: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=type, rank=rank)
        self.pt = pt
        self.age = 0
        self.biomass = 0

    def save(self) -> Tuple:
        """Saves the state of this OrganismGroup as a Tuple.

        Returns:
            The saved state of this OrganismGroup.
        """
        return (self.uid, self.pt.coordinates, self.age, self.biomass)


organism_group_cache = {}


def restore_organism_group(organism_group_data: Tuple):
    """
    Args:
        organism_group_data: tuple containing the data returned by OrganismGroup.save.
        0: uid, 1: location, 2: age, 3: biomass
    """
    # o_g_d is a 4 element tuple: 0 is uid, 1 is location, 2 is age, 3 is biomass
    # uid is a 3 element tuple: 0 is id, 1 is type, 2 is rank
    uid = organism_group_data[0]
    pt_array = organism_group_data[1]
    pt = dpt(pt_array[0], pt_array[1], pt_array[2])

    if uid in organism_group_cache:
        organism_group = organism_group_cache[uid]
    else:
        organism_group = OrganismGroup(local_id=uid[0], type=uid[1], rank=uid[2], pt=pt)
        organism_group_cache[uid] = organism_group

    organism_group.pt = pt
    organism_group.age = organism_group_data[2]
    organism_group.biomass = organism_group_data[3]
    return organism_group
