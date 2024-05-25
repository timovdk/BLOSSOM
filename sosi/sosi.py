import json
from mpi4py import MPI
import networkx as nx
from numba import njit
import numpy as np

import repast4py.context as ctx
from repast4py import core, parameters, random, schedule, space, logging
from repast4py.space import DiscretePoint as dpt

from typing import Tuple, Dict


@njit
def precompute_offsets(r):
    offsets = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if abs(dx) + abs(dy) + abs(dz) <= r:
                    offsets.append((dx, dy, dz))
    return np.array(offsets)


@njit()
def von_neumann_neighborhood_3d(x, y, z, r):
    offsets = precompute_offsets(r)
    num_offsets = offsets.shape[0]
    neighbors = []

    for i in range(num_offsets):
        dx, dy, dz = offsets[i]
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < 399 and 0 <= ny < 399 and 0 <= nz < 49:
            neighbors.append((nx, ny, nz))

    return neighbors


@njit()
def von_neumann_neighborhood_r1(center):
    x, y, z = center

    neighbors = []
    # Define offsets for Von Neumann neighborhood
    for dx, dy, dz in [
        (0, 0, 1),
        (0, 0, -1),
        (0, 1, 0),
        (0, -1, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 0, 0),
    ]:
        new_x, new_y, new_z = x + dx, y + dy, z + dz
        if 0 <= new_x < 400 and 0 <= new_y < 400 and 0 <= new_z < 50:
            neighbors.append((new_x, new_y, new_z))
    return neighbors


class OrganismGroup(core.Agent):
    def __init__(self, local_id: int, type: int, rank: int, pt: dpt, biomass):
        super().__init__(id=local_id, type=type, rank=rank)
        self.pt = pt
        self.age = 0
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
    else:
        organism_group = OrganismGroup(
            local_id=uid[0],
            type=uid[1],
            rank=uid[2],
            pt=pt,
            biomass=organism_group_data[3],
        )
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

        # load parameters
        self.organism_parameters = params["organism_parameters"]
        self.nutrient_grid_filename = params["nutrient_log_file"]
        self.x_max = params["world.width"]
        self.y_max = params["world.height"]
        self.z_max = params["world.depth"]

        # initialize the nutrient grid (SOM)
        self.initialize_nutrients()

        # Set the rng
        random.init(rng_seed=params["random.seed"])
        self.rng = random.default_rng

        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(
            1, 1, self.step, priority_type=schedule.PriorityType.FIRST
        )
        self.runner.schedule_repeating_event(
            1, 1, self.log_agents, priority_type=schedule.PriorityType.LAST
        )
        self.runner.schedule_stop(params["stop.at"])
        self.runner.schedule_end_event(self.at_end)

        # load networks
        self.co_occ_network = nx.node_link_graph(json.loads(params["co_occ_network"]))
        self.trophic_net = nx.node_link_graph(json.loads(params["trophic_network"]))

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(
            0,
            self.x_max - 1,
            0,
            self.y_max - 1,
            0,
            self.z_max - 1,
        )
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(
            name="grid",
            bounds=box,
            borders=space.BorderType.Sticky,
            occupancy=space.OccupancyType.Multiple,
            buffer_size=12,
            comm=comm,
        )
        self.context.add_projection(self.grid)

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            self.comm,
            params["agent_log_file"],
            [
                "tick",
                "rank",
                "type",
                "x",
                "y",
                "z",
            ],
        )

        # networks preprocessing
        self.co_occurrence_pre_processing()
        self.trophic_net_pre_processing()

        # populate the model
        self.populate()

        # log the initial state
        self.log_agents()

    def initialize_nutrients(self):
        # if random distro: init random gen and fill a grid with random values between 0 and max nutrient value
        if int(params["nutrient.type"]) == 0:
            rng = np.random.default_rng(params["nutrient.seed"])

            # Initialize som grid
            self.nutrient_grid = rng.random(
                (self.x_max, self.y_max, self.z_max), dtype=np.float64
            ) * float(params["nutrient.max"])
        # else: uniform distribution of half max nutrient value
        else:
            self.nutrient_grid = np.full(
                (self.x_max, self.y_max, self.z_max), float(params["nutrient.max"]) / 2
            )

    def co_occurrence_pre_processing(self):
        # Create a list of tuples [(org_id, degree), ...] sorted on degree
        self.sorted_co_occurrence = sorted(
            self.co_occ_network.degree, key=lambda x: x[1], reverse=True
        )

        # Create a list of lists of co_occurring types i.e. co_occurrence[0] returns a list of all co_occurring type_ids for type 0 (bacteria)
        self.co_occurrence = []
        # Iterate over all nodes
        for node in self.co_occ_network.nodes():
            # Get the nodes that have edges pointing to the current node
            self.co_occurrence.append([n for n in self.co_occ_network.neighbors(node)])

        # Create a list of only positive co_occurring types
        self.positive_co_occurrence = []
        # Iterate over all nodes
        for node in self.co_occ_network.nodes():
            # Get the nodes that have edges pointing to the current node
            pos_ngh = []
            for n in self.co_occ_network.neighbors(node):
                if self.co_occ_network.get_edge_data(node, n)["weight"] > 0:
                    pos_ngh.append(n)
            self.positive_co_occurrence.append(pos_ngh)

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
            # Get the nodes that have edges pointing to the current node
            self.predators.append([n for n in self.trophic_net.predecessors(node)])

        # SOM is always the last id in the trophic net
        self.som_id = len(self.trophic_net.nodes()) - 1

    def populate(self):
        ranks = self.comm.Get_size()

        self.organism_id = 0
        og_types_added = []

        # for each organism group type, add entities
        for i in range(len(self.organism_parameters)):
            # Start with the most connected organism by using the sorted_occ list
            type_to_add = self.sorted_co_occurrence[i][0]
            # Find the co_occurring types for this og type
            co_occurring_types = self.co_occurrence[type_to_add]

            # number of ogs to add is the initial count divided by the number of ranks
            number_to_add = int(self.organism_parameters[type_to_add]["count"] / ranks)

            # Find the types that are already placed and co_occur with og_type_to_add
            co_occurring_types_added = list(
                set(og_types_added).intersection(co_occurring_types)
            )

            # If any of the already added types co-occur with the type we are adding currently, ensure placement is based on co-occurrence weights
            if len(co_occurring_types_added) > 0:
                # Find placed agents per type and weights per type
                co_occurring_agents = {}
                weights = {}
                for type_id in co_occurring_types_added:
                    co_occurring_agents[type_id] = list(
                        self.context.agents(agent_type=type_id)
                    )
                    weights[type_id] = self.co_occ_network.get_edge_data(
                        type_to_add, type_id
                    )
                # For each og that should be added for this og_type:
                for _ in range(number_to_add):
                    self.informed_placement(
                        type_to_add,
                        co_occurring_types_added,
                        co_occurring_agents,
                        weights,
                    )

            # else add the agents of this type on a random location
            else:
                # For each og that should be added for this og_type:
                for _ in range(number_to_add):
                    self.random_placement(type_to_add)

            # Finally, add this type to the list of added organism types
            if number_to_add != 0:
                og_types_added.append(type_to_add)

    def informed_placement(
        self, type_to_add, co_occurring_types_added, co_occurring_agents, weights
    ):
        # Randomly pick a co occurring type
        co_occ_og_id = self.rng.choice(co_occurring_types_added)
        # Get all placed agents of this type
        co_occurring_ogs = co_occurring_agents.get(co_occ_og_id)

        # Decide between positive, negative, random placement from this type
        placement_type = self.decide_placement_type(weights.get(co_occ_og_id)["weight"])

        if placement_type == 0:
            self.positive_placement(type_to_add, co_occurring_ogs)
        elif placement_type == 1:
            self.negative_placement(type_to_add, co_occurring_ogs)
        else:
            self.random_placement(type_to_add)

    def decide_placement_type(self, weight):
        p_pos = weight if 0 <= weight <= 1 else 0
        p_neg = abs(weight) if -1 <= weight < 0 else 0
        p_rand = (
            1 - abs(weight)
            if -1 <= weight < 0
            else 1 - weight if 0 <= weight <= 1 else 0
        )
        return self.rng.choice(3, p=[p_pos, p_neg, p_rand])

    def positive_placement(self, type_to_add, co_occurring_ogs):
        # Get the location of a random co-occurring agent and get a random offset
        # offset is based on Von Neumann distance of r=1, including the 0,0,0
        co_occ_loc = self.grid.get_location(
            co_occurring_ogs[self.rng.choice(len(co_occurring_ogs))]
        )
        nghs = von_neumann_neighborhood_r1((co_occ_loc.x, co_occ_loc.y, co_occ_loc.z))
        loc = nghs[self.rng.choice(len(nghs))]
        # get a location based on the co_occ_loc and offset, and add an agent here
        pt = dpt(loc[0], loc[1], loc[2])
        self.add_agent(
            type_to_add,
            pt,
            self.organism_parameters[type_to_add]["biomass_reproduction"],
        )

    def negative_placement(self, type_to_add, co_occurring_ogs):
        loc_found = False
        pt = dpt(0, 0, 0)
        while not loc_found:
            pt = self.grid.get_random_local_pt(self.rng)
            nghs = von_neumann_neighborhood_r1((pt.x, pt.y, pt.z))

            # If the random location is not near any negative co_occurring orgs, pick this
            for og in co_occurring_ogs:
                loc = self.grid.get_location(og)
                if not ((loc.x, loc.y, loc.z) in nghs):
                    loc_found = True

        # Add the agent at this location
        self.add_agent(
            type_to_add,
            pt,
            self.organism_parameters[type_to_add]["biomass_reproduction"],
        )

    def random_placement(self, type_to_add):
        # get a random x,y,z location in the grid and add an agent here
        pt = self.grid.get_random_local_pt(self.rng)
        self.add_agent(
            type_to_add,
            pt,
            self.organism_parameters[type_to_add]["biomass_reproduction"],
        )

    def step(self):
        ogs_to_add = []
        ogs_to_kill = []

        # Loop through all agents
        for organism_group in self.context.agents(shuffle=True):
            # If the agent got eaten, don't execute the step for this agent.
            if organism_group.uid in ogs_to_kill:
                continue

            organism_parameters = self.organism_parameters[organism_group.type]
            coords = organism_group.pt.coordinates

            # First, run dispersal submodel if the dispersal range is not 0
            if organism_parameters["range_dispersal"] != 0:
                self.disperse(
                    organism_group,
                    coords,
                    organism_parameters,
                    organism_group.biomass >= organism_parameters["biomass_max"],
                    self.som_id in self.preys[organism_group.type],
                )

            # If the agent eats som, the biomass is smaller than max biomass, and there is som, handle uptake
            if (
                self.som_id in self.preys[organism_group.type]
                and organism_group.biomass < organism_parameters["biomass_max"]
                and self.nutrient_grid[coords[0], coords[1], coords[2]]
            ):
                food_available = self.nutrient_grid[coords[0], coords[1], coords[2]]

                uptake = (
                    organism_parameters["mu_max"]
                    * food_available
                    / (organism_parameters["k"] + food_available)
                )
                organism_group.biomass += min(food_available, uptake)
                self.nutrient_grid[coords[0], coords[1], coords[2]] = max(
                    0, food_available - uptake
                )

            # Else, if the agent does not eat som, run competition submodel
            elif self.som_id not in self.preys[organism_group.type]:
                ogs_to_kill.append(
                    self.compete(
                        organism_group,
                        coords,
                        organism_parameters["mu_max"],
                        organism_parameters["k"],
                    )
                )

            # If agent's age is >= reproduction age, and biomas >= reproduction biomass, run reproduction submodel
            if (
                organism_group.age >= organism_parameters["age_reproduction"]
                and organism_group.biomass
                >= organism_parameters["biomass_reproduction"]
            ):
                organism_group.biomass /= 2
                ogs_to_add.append(self.reproduce(organism_group))

        # Loop through all organisms and determine whether they survive for next step
        for organism_group in self.context.agents():
            coords = organism_group.pt.coordinates

            if (
                organism_group.age
                >= self.organism_parameters[organism_group.type]["age_max"]
            ):
                self.nutrient_grid[
                    coords[0], coords[1], coords[2]
                ] += organism_group.biomass
                ogs_to_kill.append(organism_group.uid)

            organism_group.age += 1

        ogs_to_add = filter(None, ogs_to_add)
        ogs_to_kill = filter(None, ogs_to_kill)

        # loop for adding organism groups (from reproduction step)
        for og in ogs_to_add:
            pt_array = og[1]

            # If organism is Fungi, replicate at a cell next to parent cell
            if og[0][1] == 1:
                nghs = von_neumann_neighborhood_r1(
                    (pt_array[0], pt_array[1], pt_array[2])
                )
                pt_array = nghs[self.rng.choice(len(nghs))]

            pt = dpt(pt_array[0], pt_array[1], pt_array[2])
            self.add_agent(og[0][1], pt, og[3])

        # loop for removing organism groups that were added to the kill list
        for uid in list(set(ogs_to_kill)):
            agent = self.context.agent(uid)
            if agent is not None:
                self.remove_agent(agent)

        self.context.synchronize(restore_organism_group)

    def reproduce(self, organism_group):
        return organism_group.save()

    def compete(self, organism_group, coords, mu_max, k):

        food_types = self.preys[organism_group.type]
        food_opts = []
        food_probs = []

        if len(food_types) != 0:
            for obj in self.grid.get_agents(dpt(coords[0], coords[1], coords[2])):
                if obj.type in food_types and obj.uid and obj.biomass > 0:
                    food_opts.append(obj)
                    food_probs.append(obj.biomass)

        # If there are food options
        if food_opts:
            food_probs /= np.sum(food_probs)
            # Pick a semi random target, with probabilities decided by biomass of targets
            target_og = food_opts[self.rng.choice(len(food_opts), p=food_probs)]
            target_coords = target_og.pt.coordinates

            # Calculate uptake
            uptake = mu_max * target_og.biomass / (k + target_og.biomass)

            # If organism_group's biomass is smaller than max, eat part of the target
            organism_group.biomass += min(target_og.biomass, uptake)

            self.nutrient_grid[
                target_coords[0], target_coords[1], target_coords[2]
            ] += max(0, (target_og.biomass - uptake))

            # finally, add the target organism group to the to kill list
            return target_og.uid

    def disperse(
        self, organism_group, coords, organism_parameters, fully_satisfied, som_feeder
    ):
        disperse_location = (coords[0], coords[1], coords[2])
        preys = self.preys[organism_group.type]
        predators = self.predators[organism_group.type]

        # for each range step, run the search code (i.e. range=5, search 5 times for food)
        # for _ in range(organism_parameters["range_dispersal"]):
        options = von_neumann_neighborhood_3d(
            coords[0], coords[1], coords[2], organism_parameters["range_dispersal"]
        )
        probs = np.full(len(options), 0.001)
        if not fully_satisfied:
            # now check which probability should be boosted (much food, co_occ, or target)
            for i, opt in enumerate(options):
                # If som feeder, first check whether food availability is not 0, else check whether there is no predator
                if som_feeder:

                    # predator check
                    for obj in self.grid.get_agents(dpt(opt[0], opt[1], opt[2])):
                        if obj.type in predators:
                            probs[i] = 0.000001
                            break

                    # food check
                    if self.nutrient_grid[opt[0], opt[1], opt[2]] == 0:
                        probs[i] = 0.000001
                    elif (
                        self.nutrient_grid[opt[0], opt[1], opt[2]]
                        >= organism_parameters["biomass_max"]
                    ):
                        probs[i] = 1
                else:
                    # if any obj type is in preys, add 1 to probability, but if there is also a predator, set probability low and break the loop
                    for obj in self.grid.get_agents(dpt(opt[0], opt[1], opt[2])):
                        if obj.type in preys:
                            probs[i] += 1
                        elif obj.type in predators:
                            probs[i] = 0.000001
                            break

        # Finally, normalize probabilities so that they sum up to 1
        probs /= np.sum(probs)
        disperse_location = options[self.rng.choice(len(options), p=probs)]

        # move to the found location
        organism_group.pt = self.grid.move(
            organism_group,
            dpt(disperse_location[0], disperse_location[1], disperse_location[2]),
        )

    def log_agents(self):
        tick = int(self.runner.schedule.tick)
        for organism_group in self.context.agents():
            coords = organism_group.pt.coordinates
            self.agent_logger.log_row(
                tick,
                organism_group.uid_rank,
                organism_group.type,
                coords[0],
                coords[1],
                coords[2],
            )

        self.agent_logger.write()

    def at_end(self):
        self.agent_logger.close()
        np.save(self.nutrient_grid_filename, self.nutrient_grid)

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
