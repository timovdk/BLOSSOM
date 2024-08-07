import json
import pickle
import networkx as nx
import numpy as np
import repast4py.context as ctx

from mpi4py import MPI
from numba import njit
from typing import Tuple, Dict
from repast4py import core, parameters, random, schedule, space, logging, value_layer
from repast4py.space import DiscretePoint as dpt

X = 0
Y = 0
Z = 0


@njit
def less_than(a, b):
    return a < b


@njit
def geq(a, b):
    return a >= b


@njit
def max_numba(a, b):
    return max(a, b)


@njit
def min_numba(a, b):
    return min(a, b)


@njit
def normalize_list(input_list):
    total_sum = np.sum(input_list)
    if total_sum == 0:
        raise ValueError("Sum of the list elements is zero, can't normalize.")
    return input_list / total_sum


@njit()
def precompute_offsets(r):
    offsets = set()
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    frontier = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    offsets.update(
        [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    )

    for _ in range(1, r):
        next_frontier = []
        for x, y, z in frontier:
            for dx, dy, dz in directions:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor not in offsets:
                    offsets.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return list(offsets)


@njit()
def von_neumann_neighborhood_3d(x, y, z, r):
    if r == 1:
        xm = x % X
        ym = y % Y
        zm = z % Z
        return [
            (xm, ym, zm),
            (xm, ym, (z + 1) % Z),
            (xm, ym, (z + -1) % Z),
            (xm, (y + 1) % Y, zm),
            (xm, (y + -1) % Y, zm),
            ((x + 1) % X, ym, zm),
            ((x + -1) % X, ym, zm),
        ]

    else:
        neighbors = []

        for dx, dy, dz in precompute_offsets(r):
            neighbors.append(((x + dx) % X, (y + dy) % Y, (z + dz) % Z))

        return neighbors


class OrganismGroup(core.Agent):
    def __init__(
        self, local_id: int, type: int, rank: int, pt: dpt, biomass, age: int = 0
    ):
        super().__init__(id=local_id, type=type, rank=rank)
        self.pt = pt
        self.age = age
        self.biomass = biomass

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
        organism_group.pt = pt
        organism_group.age = organism_group_data[2]
        organism_group.biomass = organism_group_data[3]

    else:
        organism_group = OrganismGroup(
            local_id=uid[0],
            type=uid[1],
            rank=uid[2],
            pt=pt,
            biomass=organism_group_data[3],
            age=organism_group_data[2],
        )

    organism_group_cache[uid] = organism_group

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
        self.context = ctx.SharedContext(self.comm)
        self.rank = self.comm.Get_rank()

        # load parameters
        self.organism_parameters = params["organism_parameters"]

        # Global so Numba functions can use them as well
        global X
        X = params["world.width"]
        global Y
        Y = params["world.height"]
        global Z
        Z = params["world.depth"]

        # Set the rng
        random.init(rng_seed=params["random.seed"])
        self.rng = random.default_rng

        # create the schedule
        self.runner = schedule.init_schedule_runner(self.comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params["stop.at"])
        self.runner.schedule_end_event(self.at_end)

        # load trophic network
        self.trophic_net = nx.node_link_graph(json.loads(params["trophic_network"]))

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(0, X, 0, Y, 0, Z)

        # create a SharedGrid of 'box' size with Periodic borders (wrapping) and multiple occupancy
        self.grid = space.SharedGrid(
            name="grid",
            bounds=box,
            borders=space.BorderType.Periodic,
            occupancy=space.OccupancyType.Multiple,
            buffer_size=params["buffer_size"],
            comm=self.comm,
        )
        self.context.add_projection(self.grid)

        # Create #Z 2D value layers (3D value layers are not yet supported in Repast4Py)
        box_2d = space.BoundingBox(0, X, 0, Y)
        self.value_layers = []
        for _ in range(Z):
            vl = value_layer.SharedValueLayer(
                bounds=box_2d,
                borders=space.BorderType.Periodic,
                buffer_size=params["buffer_size"],
                init_value=params["nutrient.max"],
                comm=self.comm,
            )
            self.context.add_value_layer(vl)
            self.value_layers.append(vl)

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            self.comm,
            params["agent_log_file"],
            [
                "tick",
                "type",
                "x",
                "y",
                "z",
            ],
        )

        # networks preprocessing
        self.trophic_net_pre_processing()

        # populate the model
        self.populate(params["initial_location"], params["initial_locations_folder"])

        # Synchronize the ranks and log initial state
        self.context.synchronize(restore_organism_group)
        self.log_agents()

    def trophic_net_pre_processing(self):
        # Create a list of lists of connected nodes for each node. i.e. preys[0] will return a list of incoming connected nodes for node 0
        self.preys = []
        # Iterate over all nodes
        for node in self.trophic_net.nodes():
            # Get the nodes that have edges pointing to the current node
            self.preys.append([n for n in self.trophic_net.predecessors(node)])
        # Create a list of lists of connected nodes for each node. i.e. predators[0] will return a list of outgoing connected nodes for node 0
        self.predators = []
        # Iterate over all nodes
        for node in self.trophic_net.nodes():
            # Get the nodes that have edges going out of the current node
            self.predators.append([n for n in self.trophic_net.successors(node)])

        # SOM is always the last id in the trophic net
        self.som_id = len(self.trophic_net.nodes()) - 1

    def populate(self, init_type, init_folder):
        ranks = self.comm.Get_size()
        self.organism_id = 0

        # init_type == 1: load clusters
        if init_type == 1:
            with open(init_folder + "/clustered_locations.pkl", "rb") as f:
                locations = pickle.load(f)

        # else, load random
        else:
            with open(init_folder + "/random_locations.pkl", "rb") as f:
                locations = pickle.load(f)

        # For each agent type, add the defined number of agents for one rank
        for type_i in range(len(self.organism_parameters)):
            # Agents are initialized with biomass = biomass_reproduction / 2
            bm = self.organism_parameters[type_i]["biomass_reproduction"] / 2
            slice_size = int(self.organism_parameters[type_i]["count"] / ranks)

            # Precompute the start index for each type
            start_index = self.rank * slice_size
            end_index = start_index + slice_size
            locs = locations[type_i][start_index:end_index]

            # Add agents for agent_type
            for i in range(slice_size):
                self.add_agent(type_i, dpt(locs[i][0], locs[i][1], locs[i][2]), bm)

    def step(self):
        ogs_to_add = []
        ogs_to_kill = []

        # Loop through all agents (shuffled, using rng seed)
        for organism_group in self.context.agents(shuffle=True):
            # If the agent is not in the to_kill list, execute a timestep
            if organism_group.uid not in ogs_to_kill:
                # initialize some variables for this timestep
                organism_parameters = self.organism_parameters[organism_group.type]
                x, y, z = organism_group.pt.coordinates

                dispersal_range = organism_parameters["range_dispersal"]
                biomass_max = organism_parameters["biomass_max"]
                k = organism_parameters["k"]
                preys = set(self.preys[organism_group.type])
                predators = set(self.predators[organism_group.type])
                vls = [vl for vl in self.value_layers]

                # First, run dispersal submodel if the dispersal range is not 0
                if dispersal_range:
                    # get dispersal options
                    options = von_neumann_neighborhood_3d(x, y, z, dispersal_range)

                    # initialize probabilities
                    probs = np.full(len(options), 0.01)

                    # create a list of lists with the agent types at each dispersal option
                    get_agents = self.grid.get_agents
                    agents = [
                        [
                            agent.type
                            for agent in get_agents(dpt(opt[0], opt[1], opt[2]))
                        ]
                        for opt in options
                    ]

                    # If agent biomass < maximum biomass, disperse based on food availability
                    if less_than(organism_group.biomass, biomass_max):
                        # If agent feeds on SOM, set probabilities as the SOM availability
                        if self.som_id in preys:
                            for i, opt in enumerate(options):
                                probs[i] = max_numba(
                                    0.00001, float(vls[opt[2]].get(dpt(opt[0], opt[1])))
                                )

                        # If agent feeds on other agents, count the number of preys at dispersal options
                        else:
                            for i, agent_types in enumerate(agents):
                                probs[i] = max_numba(
                                    0.01, sum(el in agent_types for el in preys)
                                )

                    # create a mask of save and none save dispersal options
                    danger = [
                        any(each in predators for each in agent_types_at_location)
                        for agent_types_at_location in agents
                    ]
                    probs[danger] = 0.00001

                    # Finally, normalize probabilities so that they sum up to 1 and chose dispersal location
                    probs = normalize_list(probs)
                    x_d, y_d, z_d = options[self.rng.choice(len(options), p=probs)]

                    # move to the found location
                    organism_group.pt = self.grid.move(
                        organism_group, dpt(x_d, y_d, z_d)
                    )

                # If biomass is less than biomass_max
                if less_than(organism_group.biomass, biomass_max):
                    # If SOM feeder, calculate uptake using Monod and update value_layer
                    if self.som_id in preys:
                        food_available = float(vls[z].get(dpt(x, y)))

                        uptake = biomass_max * food_available / (k + food_available)

                        organism_group.biomass += min_numba(food_available, uptake)
                        self.value_layers[z].set(
                            dpt(x, y), max_numba(0, food_available - uptake)
                        )

                    # Else run agent-agent feeding submodel
                    else:
                        food_opts = []
                        food_probs = []

                        # get agents at current loc, and check whether they are a prey and have not been killed already
                        for obj in self.grid.get_agents(dpt(x, y, z)):
                            if (obj.type in preys) and (obj.uid not in ogs_to_kill):
                                food_opts.append(obj)
                                food_probs.append(max_numba(0.0000001, obj.biomass))

                        # If there are food options
                        if food_probs:
                            food_probs = normalize_list(np.array(food_probs))

                            # Pick a semi random target, with probabilities decided by biomass of targets
                            target_og = food_opts[
                                self.rng.choice(len(food_opts), p=food_probs)
                            ]
                            x_t, y_t, z_t = target_og.pt.coordinates

                            # Calculate uptake
                            uptake = (
                                biomass_max
                                * target_og.biomass
                                / (k + target_og.biomass)
                            )

                            # Eat organism and calculate how much SOM should be added to the current location (if the prey was not fully eaten)
                            organism_group.biomass += min_numba(
                                target_og.biomass, uptake
                            )
                            self.value_layers[z_t].set(
                                dpt(x_t, y_t),
                                vls[z].get(dpt(x, y))
                                + max_numba(0, (target_og.biomass - uptake)),
                            )

                            # finally, add the target organism group to the to kill list
                            ogs_to_kill.append(target_og.uid)

                # If agent's age is >= reproduction age, and biomass >= reproduction biomass, run reproduction submodel
                if geq(
                    organism_group.age, organism_parameters["age_reproduction"]
                ) and geq(
                    organism_group.biomass, organism_parameters["biomass_reproduction"]
                ):
                    # divide agent's biomass by 2, and transfer half to the new agent
                    organism_group.biomass /= 2
                    ogs_to_add.append(organism_group.save())

                # If the agent's age has reached age_max, add it to the kill list and add its biomass to the value_layer
                if geq(organism_group.age, organism_parameters["age_max"]):
                    self.value_layers[z].set(
                        dpt(x, y),
                        self.value_layers[z].get(dpt(x, y)) + organism_group.biomass,
                    )
                    ogs_to_kill.append(organism_group.uid)

                # Increase agent's age
                organism_group.age += 1

        # loop for adding organism groups (from reproduction step)
        for og in ogs_to_add:
            pt_array = og[1]
            # If organism is Fungi, replicate at a cell next to parent cell
            if og[0][1] == 1:
                nghs = von_neumann_neighborhood_3d(
                    pt_array[0], pt_array[1], pt_array[2], r=1
                )
                pt_array = nghs[self.rng.choice(len(nghs))]

            pt = dpt(pt_array[0], pt_array[1], pt_array[2])
            self.add_agent(og[0][1], pt, og[3])

        # loop for removing agents that were added to the kill list
        for uid in ogs_to_kill:
            self.remove_agent(self.context.agent(uid))
            organism_group_cache.pop(uid, None)

        # Synchronize ranks and log agents
        self.context.synchronize(restore_organism_group)
        self.log_agents()

    def log_agents(self):
        tick = int(self.runner.schedule.tick)
        for organism_group in self.context.agents():
            x, y, z = organism_group.pt.coordinates
            self.agent_logger.log_row(
                tick,
                organism_group.type,
                x,
                y,
                z,
            )

        self.agent_logger.write()

    def at_end(self):
        self.agent_logger.close()

    def start(self):
        self.runner.execute()

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_agent(self, type, pt, biomass):
        og = OrganismGroup(self.organism_id, type, self.rank, pt, biomass)
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
