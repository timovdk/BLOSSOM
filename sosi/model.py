from typing import Dict
from mpi4py import MPI

import repast4py.context as ctx
from repast4py import random, schedule, space, logging
from repast4py.space import DiscretePoint as dpt

import json

import networkx as nx
from .agent import OrganismGroup, restore_organism_group

from .utils import GridNghFinder

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

        # Assign each rank its own seed
        random.init(rng_seed=params["random.seed"] * (self.rank + 1))
        self.rng = random.default_rng

        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_repeating_event(1.1, 1, self.log_agents)
        self.runner.schedule_stop(params["stop.at"])
        self.runner.schedule_end_event(self.at_end)

        # load parameters
        self.organism_parameters = params["organism_parameters"]
        self.width = params["world.width"]
        self.height = params["world.height"]
        self.depth = params["world.depth"]

        # load networks
        self.co_occ_network = nx.node_link_graph(json.loads(params["co_occ_network"]))
        self.sorted_occ_net = sorted(
            self.co_occ_network.degree, key=lambda x: x[1], reverse=True
        )
        self.trophic_net = nx.node_link_graph(json.loads(params["trophic_network"]))

        # Initialize som grid
        self.nutrient_grid = (
            self.rng.random((self.width, self.height, self.depth)) * 0.25
        )

        # determine the food dependency per organism type
        self.food_dependency = []
        for x in range(len(self.trophic_net.nodes())):
            edges = self.trophic_net.in_edges(nbunch=x)
            food = []
            if edges:
                for k in edges:
                    food.append(k[0])
            self.food_dependency.append(food)

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(
            0,
            params["world.width"] - 1,
            0,
            params["world.height"] - 1,
            0,
            params["world.depth"] - 1,
        )
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(
            name="grid",
            bounds=box,
            borders=space.BorderType.Sticky,
            occupancy=space.OccupancyType.Multiple,
            buffer_size=2,
            comm=comm,
        )
        self.context.add_projection(self.grid)

        # initialize grid searcher
        self.ngh_finder = GridNghFinder(0, 0, 0, box.xextent, box.yextent, box.zextent)

        # populate the model
        self.populate()

        # initialize the logging
        self.agent_logger = logging.TabularLogger(
            self.comm,
            params["agent_log_file"],
            [
                "tick",
                "agent_id",
                "type",
                "agent_uid_rank",
                "x",
                "y",
                "z",
                "age",
                "biomass",
            ],
        )

        self.log_agents()

    def populate(self):
        ranks = self.comm.Get_size()

        self.organism_id = 0
        og_types_added = []

        # for each organism group type, add entities
        for i in range(len(self.organism_parameters)):
            og_type_to_add = self.sorted_occ_net[i][0]

            # number of ogs to add is the initial count divided by the number of ranks
            num_ogs_to_add = int(
                self.organism_parameters[og_type_to_add]["count"] / ranks
            )

            # If any of the already added types co-occur with the type we are adding currently, ensure placement is based on co-occurrence
            if any([og_type_to_add in self.co_occ_network[p] for p in og_types_added]):
                og_types_added.append(og_type_to_add)

                # Find the types that are already placed
                intersection = list(
                    set(self.co_occ_network[og_type_to_add]).intersection(
                        og_types_added
                    )
                )

                co_occurring_agents = []
                for d in intersection:
                    co_occurring_agents.extend(list(self.context.agents(agent_type=d)))

                for _ in range(num_ogs_to_add):
                    choice = self.rng.choice([0, 1], size=1, p=[0.75, 0.25])[0]
                    if choice == 0:
                        # Get the location of a random co-occurring agent and get a random offset
                        # offset is based on Von Neumann distance of r=1, including the 0,0,0
                        co_occ_loc = self.grid.get_location(
                            self.rng.choice(co_occurring_agents, size=1)[0]
                        )
                        offsets = self.rng.choice(
                            [
                                [0, 0, 1],
                                [0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, 0],
                                [1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1],
                            ],
                            size=1,
                        )[0]

                        # get a location based on the co_occ_loc and offset, and add an agent here
                        pt = dpt(
                            min(self.height - 1, co_occ_loc.x + offsets[0]),
                            min(self.width - 1, co_occ_loc.y + offsets[1]),
                            min(self.depth - 1, co_occ_loc.z + offsets[2]),
                        )
                        self.add_agent(og_type_to_add, pt)
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

    def step(self):
        ogs_to_add = []
        ogs_to_kill = []

        # First loop through all organisms to process check_death and nutrient uptake
        for organism_group in self.context.agents():
            organism_parameters = self.organism_parameters[organism_group.type]
            coords = organism_group.pt.coordinates

            # First determine whether this organism_group survives
            if organism_group.age >= organism_parameters["age_max"]:
                ogs_to_kill.append(organism_group.uid)
                # todo: check whether continue is correct here
                continue

            # If the organism_group survives, determine nutrient uptake
            # todo: make nutrient uptake based on monod's eq
            food = self.nutrient_grid[coords[0]][coords[1]][coords[2]]

            if organism_group.biomass < organism_parameters["biomass_max"]:
                organism_group.biomass += food

            # Pick one of these at random based on rate of events
            choice = self.rng.choice(
                [0, 1, 2],
                p=[
                    organism_parameters["rate_reproduction"],
                    organism_parameters["rate_competition"],
                    organism_parameters["rate_dispersal"],
                ],
            )

            if choice == 0:
                if (
                    organism_group.age > organism_parameters["age_reproduction"]
                    and organism_group.biomass
                    >= organism_parameters["biomass_reproduction"]
                ):
                    ogs_to_add.append(organism_group.save())

            elif choice == 1:
                nghs = self.ngh_finder.find(coords[0], coords[1], coords[2])

                food_types = self.food_dependency[organism_group.type]
                food_opts = []

                if len(food_types) != 0:
                    at = dpt(0, 0, 0)
                    for ngh in nghs:
                        at._reset_from_array(ngh)

                        for obj in self.grid.get_agents(at):
                            if obj.type in food_types and obj.uid:
                                food_opts.append(obj)

                if food_opts:
                    ogs_to_kill.append(self.rng.choice(food_opts, size=1)[0].uid)

            else:
                # choose two elements from the OFFSET array
                # to select the direction to disperse in the
                # x,y,z dimensions
                xyz_dirs = (
                    self.rng.choice(OrganismGroup.OFFSETS, size=3)
                    * organism_parameters["range_dispersal"]
                )
                organism_group.pt = self.grid.move(
                    organism_group,
                    dpt(
                        coords[0] + xyz_dirs[0],
                        coords[1] + xyz_dirs[1],
                        coords[2] + xyz_dirs[2],
                    ),
                )

            organism_group.age += 1

        for og in ogs_to_add:
            pt_array = og[1]
            xyz_dirs = self.rng.choice(OrganismGroup.OFFSETS, size=3)

            pt = dpt(
                min(self.width - 1, pt_array[0] + xyz_dirs[0]),
                min(self.height - 1, pt_array[1] + xyz_dirs[1]),
                min(self.depth - 1, pt_array[2] + xyz_dirs[2]),
            )
            self.add_agent(og[0][1], pt)

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
                organism_group.id,
                organism_group.type,
                organism_group.uid_rank,
                coords[0],
                coords[1],
                coords[2],
                organism_group.age,
                organism_group.biomass,
            )

        self.agent_logger.write()

    def at_end(self):
        self.agent_logger.close()

    def start(self):
        self.runner.execute()

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_agent(self, type, pt):
        og = OrganismGroup(self.organism_id, type, self.rank, pt)
        self.organism_id += 1
        self.context.add(og)
        self.grid.move(og, pt)
