from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
import networkx as nx
import json

import numba
from numba import int32, int64
from numba.experimental import jitclass

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt

@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]

spec = [
    ('xo', int32[:]),
    ('yo', int32[:]),
    ('zo', int32[:]),
    ('xmin', int32),
    ('ymin', int32),
    ('zmin', int32),
    ('ymax', int32),
    ('xmax', int32),
    ('zmax', int32)
]


@jitclass(spec)
class GridNghFinder:

    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.xo = np.array([0, 0, -1, 0, 1, 0, 0], dtype=np.int32)
        self.yo = np.array([0, 1, 0, 0, 0, -1, 0], dtype=np.int32)
        self.zo = np.array([1, 0, 0, 0, 0, 0, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax

    def find(self, x, y, z):
        xs = self.xo + x
        ys = self.yo + y
        zs = self.zo + z

        xd = (xs >= self.xmin) & (xs <= self.xmax)
        xs = xs[xd]
        ys = ys[xd]
        zs = zs[xd]

        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]
        zs = zs[yd]

        zd = (zs >= self.zmin) & (zs <= self.zmax)
        xs = xs[zd]
        ys = ys[zd]
        zs = zs[zd]

        return np.stack((xs, ys, zs), axis=-1)

def reduce_ratio(ratio_list, total_num, min_num=1):
    output = [min_num for _ in ratio_list]
    total_num -= sum(output)
    if total_num < 0:
        raise Exception('Could not satisfy min_num')
    elif total_num == 0:
        return output

    nloads = len(ratio_list)
    for ii in range(nloads):
        load_sum = float( sum(ratio_list) )
        load = ratio_list.pop(0)
        value = int( round(total_num*load/load_sum) )
        output[ii] += value
        total_num -= value
    return output


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

    def check_death(self, death_age):
        if self.age >= death_age:
            return self.uid

    def eat(self, food):
        self.biomass += food

    # todo: implement biomass cutoff
    def reproduce(self, reproduction_age):
        #print('repr')
        if self.age > reproduction_age:
            self.age+=1
            return self.save()
        else:
            self.age+=1

    def compete(self, food_opts):
        self.age+=1

        if food_opts:
            eat = random.default_rng.choice(food_opts, size=1)[0]
            return(eat.uid)

    def disperse(self, grid, range):
        # choose two elements from the OFFSET array
        # to select the direction to disperse in the
        # x,y,z dimensions
        xyz_dirs = random.default_rng.choice(OrganismGroup.OFFSETS, size=3) * range
        self.pt = grid.move(self, dpt(self.pt.x + xyz_dirs[0], self.pt.y + xyz_dirs[1], self.pt.z + xyz_dirs[2]))
        
        self.age+=1
    

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


class Model:
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """

    def __init__(self, comm: MPI.Intracomm, params: Dict):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = comm.Get_rank()

        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_repeating_event(1.1, 1, self.log_agents)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        self.rates = list(params['organism_group.rates'].values())
        self.organism_range_list = list(params['organism_group.ranges'].values())
        self.organism_reproduction_age = list(params['organism_group.repr_age'].values())
        self.organism_death_age = list(params['organism_group.death_age'].values())
        self.co_occ_network = nx.node_link_graph(json.loads(params['co_occ_network']))
        self.sorted_occ_net = sorted(self.co_occ_network.degree, key=lambda x: x[1], reverse=True)
        self.trophic_net = nx.node_link_graph(json.loads(params['trophic_network']))
        self.width = params['world.width']
        self.height = params['world.height']
        self.depth = params['world.depth']

        self.nutrient_grid = np.random.rand(self.width, self.height, self.depth)*0.25

        self.food_dependency = []
        for x in range(len(self.trophic_net.nodes())):
            edges = self.trophic_net.in_edges(nbunch=x)
            food = []
            if edges:
                for k in edges:
                    food.append(k[0])
            self.food_dependency.append(food)

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(0, params['world.width']-1, 0, params['world.height']-1, 0, params['world.depth']-1)
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple, buffer_size=2, comm=comm)
        
        self.ngh_finder = GridNghFinder(0, 0, 0, box.xextent, box.yextent, box.zextent)

        self.context.add_projection(self.grid)

        self.populate()

        # initialize the logging
        self.agent_logger = logging.TabularLogger(self.comm, params['agent_log_file'], ['tick', 'agent_id', 'type', 'agent_uid_rank', 'x', 'y', 'z', 'age'])
        
        self.log_agents()

    def populate(self):
        rng = random.default_rng
        # calculate no. of agents per rank/process
        ranks = self.comm.Get_size()
        count_per_rank = int(params['organism_group.count']/ranks)

        # convert ratios of organism_group.types to number of agents for this rank
        types = dict(params['organism_group.types'])
        counts = reduce_ratio(ratio_list=list(types.values()), total_num=count_per_rank)
        
        self.organism_id = 0
        og_types_added = []

        # for each entry i in counts, add v agents
        for i, v in enumerate(counts):
            og_type_to_add = self.sorted_occ_net[i][0]

            # If any of the already added types co-occur with the type we are adding currently, ensure placement is based on co-occurrence
            if(any([og_type_to_add in self.co_occ_network[p] for p in og_types_added])):
                og_types_added.append(og_type_to_add)

                # Find the types that are already placed
                intersection = list(set(self.co_occ_network[og_type_to_add]).intersection(og_types_added))

                co_occurring_agents = []
                for d in intersection:
                    co_occurring_agents.extend(list(self.context.agents(agent_type=d)))
                
                for _ in range(v):
                    # Get the location of a random co-occurring agent and get a random offset
                    # offset is based on Von Neumann distance of r=1, including the 0,0,0
                    co_occ_loc = self.grid.get_location(rng.choice(co_occurring_agents, size=1)[0])
                    offsets = rng.choice([[0, 0, 1], [0, 1, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, -1]], size=1)[0]
                    
                    # get a location based on the co_occ_loc and offset, and add an agent here
                    pt = dpt(min(self.height-1, co_occ_loc.x + offsets[0]), min(self.width-1, co_occ_loc.y + offsets[1]), min(self.depth-1, co_occ_loc.z + offsets[2]))
                    self.add_agent(og_type_to_add, pt)
           
            # else add the agents of this type on a random location
            else:
                og_types_added.append(og_type_to_add)
                for _ in range(v):
                    # get a random x,y,z location in the grid and add an agent here
                    pt = self.grid.get_random_local_pt(rng)
                    self.add_agent(og_type_to_add, pt)

    def step(self):
        ogs_to_kill = []
        # First loop through all organisms to process check_death and nutrient uptake
        for organism_group in self.context.agents():
            # First determine whether this organism_group survives
            uid = organism_group.check_death(self.organism_death_age[organism_group.type])
            if uid is not None:
                ogs_to_kill.append(uid)
        
        for uid in ogs_to_kill:
            agent = self.context.agent(uid)
            if agent is not None:
                self.remove_agent(agent)
        
        self.context.synchronize(restore_organism_group)

        #todo: make nutrient uptake based on monod's eq
        for organism_group in self.context.agents():
            # If the organism_group survives, determine nutrient uptake
            coords = organism_group.pt.coordinates
            food = self.nutrient_grid[coords[0]][coords[1]][coords[2]]
            organism_group.eat(food)

        ogs_to_add = []
        ogs_to_kill = []

        #todo: make this loop random
        # Then loop through all organisms to process reproduction/competition/dispersion
        for organism_group in self.context.agents():
            # Pick one of these at random based on rate of events
            choice = random.default_rng.choice([0, 1, 2], p=self.rates[organism_group.type])
            if choice == 0:
                og = organism_group.reproduce(self.organism_reproduction_age[organism_group.type])
                if og is not None:
                    ogs_to_add.append(og)
            elif choice == 1:
                coords = organism_group.pt.coordinates
                nghs = self.ngh_finder.find(coords[0], coords[1], coords[2])

                food_types = self.food_dependency[organism_group.type]
                food_opts = []

                if len(food_types) != 0:
                    at = dpt(0, 0, 0)
                    for ngh in nghs:
                        at._reset_from_array(ngh)

                        for obj in self.grid.get_agents(at):
                            if obj.type in food_types:
                                food_opts.append(obj)
                
                uid = organism_group.compete(food_opts)
                if uid is not None:
                    ogs_to_kill.append(uid)

            else:
                organism_group.disperse(self.grid, self.organism_range_list[organism_group.type])

        for og in ogs_to_add:
            uid = og[0]
            pt_array = og[1]
            xyz_dirs = random.default_rng.choice(OrganismGroup.OFFSETS, size=3)
            
            pt = dpt(min(self.width-1, pt_array[0] + xyz_dirs[0]), min(self.height-1, pt_array[1] + xyz_dirs[1]), min(self.depth-1, pt_array[2] + xyz_dirs[2]))
            self.add_agent(uid[1], pt)

        for uid in list(set(ogs_to_kill)):
            agent = self.context.agent(uid)
            if agent is not None:
                self.remove_agent(agent)
        
        self.context.synchronize(restore_organism_group)

    def log_agents(self):
        tick = int(self.runner.schedule.tick)
        for organism_group in self.context.agents():
            coords = organism_group.pt.coordinates
            self.agent_logger.log_row(tick, organism_group.id, organism_group.type, organism_group.uid_rank, coords[0], coords[1], coords[2], organism_group.age)

        self.agent_logger.write()

    def at_end(self):
        self.agent_logger.close()

    def start(self):
        self.runner.execute()
    
    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_agent(self, type, pt):
        coords = pt.coordinates

        og = OrganismGroup(self.organism_id, type, self.rank, pt)
        self.organism_id += 1
        self.context.add(og)
        self.grid.move(og, pt)


def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
