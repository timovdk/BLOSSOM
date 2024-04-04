import json
from mpi4py import MPI
import networkx as nx
from numba import jit
import numpy as np

import repast4py.context as ctx
from repast4py import core, parameters, random, schedule, space, logging
from repast4py.space import DiscretePoint as dpt

from typing import Tuple, Dict


def generate_lookup_table_3d(max_r):
    """
    Generate a lookup table for the Von Neumann neighborhood for distances up to max_r.

    Returns:
    - List of lists: The lookup table for the Von Neumann neighborhood.
    """
    lookup_table = []
    for r in range(1, max_r + 1):
        neighborhood = []
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                for z in range(-r, r + 1):
                    if abs(x) + abs(y) + abs(z) <= r:
                        neighborhood.append((x, y, z))
        lookup_table.append(neighborhood)
    return lookup_table


def von_neumann_neighborhood_3d(center, offsets):
    """
    Generate the von Neumann neighborhood for a given center point in 3D space
    with a variable range r using a lookup table.

    Args:
    - center (tuple): The coordinates of the center point in 3D space (x, y, z).
    - r (int): The range for the neighborhood.
    - x_max (int): Upper bound for x coordinate.
    - y_max (int): Upper bound for y coordinate.
    - z_max (int): Upper bound for z coordinate.
    - lookup_table (list of lists): Precomputed lookup table for the Von Neumann neighborhood.

    Returns:
    - List of tuples: The coordinates of the cells in the von Neumann neighborhood.
    """
    x_center, y_center, z_center = center

    neighborhood = []

    for offset in offsets:
        new_x, new_y, new_z = (
            x_center + offset[0],
            y_center + offset[1],
            z_center + offset[2],
        )
        if 0 <= new_x < 400 and 0 <= new_y < 400 and 0 <= new_z < 50:
            neighborhood.append((new_x, new_y, new_z))

    return neighborhood


@jit(nopython=True)
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
    ]:
        new_x, new_y, new_z = x + dx, y + dy, z + dz
        if 0 <= new_x < 400 and 0 <= new_y < 400 and 0 <= new_z < 50:
            neighbors.append((new_x, new_y, new_z))
    return neighbors


class OrganismGroup(core.Agent):
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

        rng = np.random.default_rng(params["nutrient.seed"])

        # load parameters
        self.organism_parameters = params["organism_parameters"]
        self.x_max = params["world.width"]
        self.y_max = params["world.height"]
        self.z_max = params["world.depth"]

        # Initialize som grid
        self.nutrient_grid = (
            rng.random((self.x_max, self.y_max, self.z_max), dtype=np.float64) * 0.005
        )

        # Assign each rank its own seed
        random.init(rng_seed=params["random.seed"] * (self.rank + 1))
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
        self.sorted_occ_net = sorted(
            self.co_occ_network.degree, key=lambda x: x[1], reverse=True
        )
        self.trophic_net = nx.node_link_graph(json.loads(params["trophic_network"]))

        self.food_dependency = []
        # Iterate over all nodes
        for node in self.trophic_net.nodes():
            # Get the nodes that have edges pointing to the current node
            self.food_dependency.append(
                [n for n in self.trophic_net.predecessors(node)]
            )

        self.co_occurrence = []
        # Iterate over all nodes
        for node in self.co_occ_network.nodes():
            # Get the nodes that have edges pointing to the current node
            self.co_occurrence.append([n for n in self.co_occ_network.neighbors(node)])

        self.positive_co_occurrence = []
        # Iterate over all nodes
        for node in self.co_occ_network.nodes():
            # Get the nodes that have edges pointing to the current node
            pos_ngh = []
            for n in self.co_occ_network.neighbors(node):
                if self.co_occ_network.get_edge_data(node, n)["weight"] > 0:
                    pos_ngh.append(n)
            self.positive_co_occurrence.append(pos_ngh)

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
            buffer_size=20,
            comm=comm,
        )
        self.context.add_projection(self.grid)

        # Generate lookup table for von-neumann distance
        self.lookup_table_von_neumann = generate_lookup_table_3d(9)

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

        # populate the model
        self.populate()

        # log the initial state
        self.log_agents()

    def populate(self):
        ranks = self.comm.Get_size()

        self.organism_id = 0
        og_types_added = []

        # for each organism group type, add entities
        for i in range(len(self.organism_parameters)):
            og_type_to_add = self.sorted_occ_net[i][0]
            co_occurring_types = self.co_occurrence[og_type_to_add]

            # number of ogs to add is the initial count divided by the number of ranks
            num_ogs_to_add = int(
                self.organism_parameters[og_type_to_add]["count"] / ranks
            )

            # If any of the already added types co-occur with the type we are adding currently, ensure placement is based on co-occurrence weights
            if any(
                co_occurring_type in og_types_added
                for co_occurring_type in co_occurring_types
            ):
                # Find the types that are already placed and co-occur with og_type_to_add
                relevant_og_types_added = list(
                    set(og_types_added).intersection(co_occurring_types)
                )

                # Find placed agents per type and weights per type
                co_occurring_agents = {}
                weights = {}
                for type_id in relevant_og_types_added:
                    co_occurring_agents[type_id] = list(
                        self.context.agents(agent_type=type_id)
                    )
                    weights[type_id] = self.co_occ_network.get_edge_data(
                        og_type_to_add, type_id
                    )

                # For each og that should be added for this og_type:
                for _ in range(num_ogs_to_add):
                    # Randomly pick a co occurring type
                    co_occ_og_id = self.rng.choice(relevant_og_types_added)

                    co_occurring_ogs = co_occurring_agents.get(co_occ_og_id)

                    coords = []
                    for og in co_occurring_ogs:
                        loc = self.grid.get_location(og)
                        coords.append((loc.x, loc.y, loc.z))

                    # Decide between positive, negative, random placement from this type
                    weight = weights.get(co_occ_og_id)["weight"]

                    p_pos = weight if 0 <= weight <= 1 else 0
                    p_neg = abs(weight) if -1 <= weight < 0 else 0
                    p_rand = (
                        1 - abs(weight)
                        if -1 <= weight < 0
                        else 1 - weight if 0 <= weight <= 1 else 0
                    )

                    choice = self.rng.choice(3, p=[p_pos, p_neg, p_rand])

                    # If positive placement
                    if choice == 0:
                        # Get the location of a random co-occurring agent and get a random offset
                        # offset is based on Von Neumann distance of r=1, including the 0,0,0
                        co_occ_loc = self.grid.get_location(
                            co_occurring_ogs[self.rng.choice(len(co_occurring_ogs))]
                        )

                        nghs = von_neumann_neighborhood_r1(
                            (co_occ_loc.x, co_occ_loc.y, co_occ_loc.z)
                        )
                        loc = nghs[self.rng.choice(len(nghs))]
                        # get a location based on the co_occ_loc and offset, and add an agent here
                        pt = dpt(loc[0], loc[1], loc[2])
                        self.add_agent(og_type_to_add, pt)

                    # If negative placement:
                    elif choice == 1:
                        co_occ_loc = self.grid.get_location(
                            co_occurring_ogs[self.rng.choice(len(co_occurring_ogs))]
                        )

                        nghs = von_neumann_neighborhood_3d(
                            (co_occ_loc.x, co_occ_loc.y, co_occ_loc.z),
                            self.lookup_table_von_neumann[3],
                        )

                        max_min_distance = 0
                        loc = None

                        # Calculate the minimum Von Neumann distance from each point in point_list to all points in target_list
                        for point in nghs:
                            min_distance = min(
                                sum(
                                    abs(c1 - c2)
                                    for c1, c2 in zip(point, target_point)
                                    if c1 != c2
                                )
                                for target_point in coords
                            )
                            if min_distance > max_min_distance:
                                max_min_distance = min_distance
                                loc = point

                        # get a location based on the co_occ_loc and add an agent here
                        pt = dpt(loc[0], loc[1], loc[2])
                        self.add_agent(og_type_to_add, pt)

                    # else, random placement
                    else:
                        # get a random x,y,z location in the grid and add an agent here
                        pt = self.grid.get_random_local_pt(self.rng)
                        self.add_agent(og_type_to_add, pt)

            # else add the agents of this type on a random location
            else:
                og_types_added.append(og_type_to_add)
                for _ in range(num_ogs_to_add):
                    # get a random x,y,z location in the grid and add an agent here
                    pt = self.grid.get_random_local_pt(self.rng)
                    self.add_agent(og_type_to_add, pt)

            # Finally, add this type to the list of added organism types
            og_types_added.append(og_type_to_add)

    def step(self):
        ogs_to_add = []
        ogs_to_kill = []

        # First loop through all organisms to process check_death and nutrient uptake
        for organism_group in self.context.agents():
            organism_parameters = self.organism_parameters[organism_group.type]
            coords = organism_group.pt.coordinates

            # First determine whether this organism_group survives
            if organism_group.age >= organism_parameters["age_max"]:
                self.nutrient_grid[
                    coords[0], coords[1], coords[2]
                ] += organism_group.biomass
                ogs_to_kill.append(organism_group.uid)
                continue

            # If the organism_group survives, determine nutrient uptake
            food_available = self.nutrient_grid[coords[0], coords[1], coords[2]]
            uptake = (
                organism_parameters["mu_max"]
                * food_available
                / (organism_parameters["k"] + food_available)
            )

            # If the organism_group's biomass is smaller than max biomass, handle uptake
            if organism_group.biomass < organism_parameters["biomass_max"]:
                organism_group.biomass += uptake
                self.nutrient_grid[coords[0], coords[1], coords[2]] = (
                    food_available - uptake
                )

            # Pick one of these at random based on rate of events
            choice = self.rng.choice(
                3,
                p=[
                    organism_parameters["rate_reproduction"],
                    organism_parameters["rate_competition"],
                    organism_parameters["rate_dispersal"],
                ],
            )

            # If reproduction
            if choice == 0:
                if (
                    organism_group.age > organism_parameters["age_reproduction"]
                    and organism_group.biomass
                    >= organism_parameters["biomass_reproduction"]
                ):
                    ogs_to_add.append(organism_group.save())

            # If competition
            elif choice == 1:
                nghs = von_neumann_neighborhood_r1(coords)

                food_types = self.food_dependency[organism_group.type]
                food_opts = []

                # If food_types is not empty, look for nearby food options (target organism groups)
                if len(food_types) != 0:
                    at = dpt(0, 0, 0)
                    for ngh in nghs:
                        at._reset_from_array(np.asarray(ngh))

                        for obj in self.grid.get_agents(at):
                            if obj.type in food_types and obj.uid:
                                food_opts.append(obj)

                # If there are food options
                if food_opts:
                    # Pick a random nearby target organism_group
                    target_og = food_opts[self.rng.choice(len(food_opts))]
                    target_coords = target_og.pt.coordinates

                    # Calculate uptake
                    uptake = (
                        organism_parameters["mu_max"]
                        * target_og.biomass
                        / (organism_parameters["k"] + food_available)
                    )

                    # If organism_group's biomass is smaller than max, eat part of the target
                    if organism_group.biomass < organism_parameters["biomass_max"]:
                        organism_group.biomass += uptake

                        self.nutrient_grid[
                            target_coords[0], target_coords[1], target_coords[2]
                        ] += (target_og.biomass - uptake)

                    # else, the target's biomass is added in full to the current location
                    else:
                        self.nutrient_grid[
                            target_coords[0], target_coords[1], target_coords[2]
                        ] += target_og.biomass

                    # finally, add the target organism group to the to kill list
                    ogs_to_kill.append(target_og.uid)

            # If dispersal
            else:
                disperse_location = (coords[0], coords[1], coords[2])
                food_types = self.food_dependency[organism_group.type]
                co_occurring_types = self.positive_co_occurrence[organism_group.type]
                k = organism_parameters["k"]

                # for each range step, run the search code (i.e. range=5, search 5 times for food)
                for _ in range(organism_parameters["range_dispersal"]):
                    options = von_neumann_neighborhood_r1(disperse_location)

                    probs = np.full(len(options), 0.1)

                    # now check which probability should be boosted (much food, co_occ, or target)
                    for i, opt in enumerate(options):
                        # Low probability for current location and og location to boost mobility
                        if opt == disperse_location or all(opt == coords):
                            probs[i] = 0.01
                            break
                        # If food availibility is higher than max feeding rate
                        if self.nutrient_grid[opt[0], opt[1], opt[2]] >= k:
                            probs[i] = 0.4

                        # if any obj type is in either food_types or co_occurring_types, add 0.5
                        for obj in self.grid.get_agents(dpt(opt[0], opt[1], opt[2])):
                            if obj.type in co_occurring_types:
                                probs[i] += 0.8
                            if obj.type in food_types:
                                probs[i] += 0.4

                    # Finally, normalize probabilities so that they sum up to 1
                    probs /= np.sum(probs)
                    disperse_location = options[self.rng.choice(len(options), p=probs)]

                # move to the found location
                organism_group.pt = self.grid.move(
                    organism_group,
                    dpt(
                        disperse_location[0], disperse_location[1], disperse_location[2]
                    ),
                )

            organism_group.age += 1

        # loop for adding organism groups (from reproduction step)
        for og in ogs_to_add:
            pt_array = og[1]
            nghs = von_neumann_neighborhood_r1((pt_array[0], pt_array[1], pt_array[2]))
            loc = nghs[self.rng.choice(len(nghs))]
            pt = dpt(loc[0], loc[1], loc[2])

            self.add_agent(og[0][1], pt)

        # loop for removing organism groups that were added to the kill list
        for uid in list(set(ogs_to_kill)):
            agent = self.context.agent(uid)
            if agent is not None:
                self.remove_agent(agent)

        self.context.synchronize(restore_organism_group)

    def log_agents(self):
        tick = int(self.runner.schedule.tick)
        for organism_group in self.context.agents():
            coords = organism_group.pt.coordinates
            self.agent_logger.log_row(
                tick,
                organism_group.type,
                coords[0],
                coords[1],
                coords[2],
            )

        self.agent_logger.write()

    def at_end(self):
        self.agent_logger.close()
        np.save("output/nutrient_grid_end_.npy", self.nutrient_grid)

    def start(self):
        self.runner.execute()

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_agent(self, type, pt):
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
