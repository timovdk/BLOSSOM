#include "blossom.hpp"
#include "logger.hpp"
#include "neighbourhoods.hpp"
#include "organism.hpp"
#include "utils.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

BLOSSOM::BLOSSOM(const int index)
{
    trialID = index;
    // Load configuration
    loadConfig("./configs/config_" + std::to_string(trialID) + ".props");

    // Initialize RNGs
    defaultRNG = std::mt19937(defaultSeed);
    initDistRNG = std::mt19937(initialDistributionSeed);
    nutrientRNG = std::mt19937(nutrientSeed);

    // Initialize simulation
    init();
}

void BLOSSOM::run()
{ // Run the simulation
    bool earlyStop = false;
    while ((currentStep < maxSteps) && !earlyStop)
    {
        // Increment step (0 == init step)
        currentStep++;
        if (currentStep % earlyStopInterval == 0)
        {
            earlyStop = shouldStopEarly(agents, earlyStopMinTypes);
        }
        // Run one step
        step();
        // Log state
        logger->log(currentStep, agents, somGrid);
    }

    if (earlyStop)
    {
        std::cerr << "Trial: " << trialID << " stopped early! Not enough organism types are alive." << std::endl;
    }
}

// Setup
void BLOSSOM::loadConfig(const std::string &filename)
{
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    std::string line;

    // Read the configuration file
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string key, value;
        // Split the lines into keys and values
        if (std::getline(iss, key, '=') && std::getline(iss, value))
        {
            config[key] = value;
        }
    }

    // Parse the configuration values
    outputDir = config["output_dir"];
    outputFileName = config["output_file_name"];
    defaultSeed = std::stoul(config["default_seed"]);
    initialDistributionType = std::stoi(config["initial_distribution_type"]);
    initialDistributionSeed = std::stoul(config["initial_distribution_seed"]);
    nutrientType = std::stoi(config["nutrient_type"]);
    nutrientSeed = std::stoul(config["nutrient_seed"]);
    nutrientMean = std::stod(config["nutrient_mean"]);
    maxSteps = std::stoi(config["max_steps"]);
    gridWidth = std::stoi(config["grid_width"]);
    gridHeight = std::stoi(config["grid_height"]);
    earlyStopInterval = std::stoi(config["early_stop_interval"]);
    earlyStopMinTypes = std::stoi(config["early_stop_min_types"]);

    // Initialize organism data
    for (int i = 0; i <= 8; ++i)
    {
        OrganismData data;
        std::string prefix = "organism_" + std::to_string(i) + "_";

        data.params["count"] = std::stoi(config[prefix + "count"]);
        data.params["range_dispersal"] = std::stoi(config[prefix + "range_dispersal"]);
        data.params["age_reproduction"] = std::stoi(config[prefix + "age_reproduction"]);
        data.params["age_max"] = std::stoi(config[prefix + "age_max"]);
        data.params["biomass_max"] = std::stod(config[prefix + "biomass_max"]);
        data.params["biomass_reproduction"] = std::stod(config[prefix + "biomass_reproduction"]);
        data.params["k"] = std::stod(config[prefix + "k"]);
        data.params["som_feeder"] = std::stoi(config[prefix + "som_feeder"]);

        // Convert the comma separated string to a vector of ints
        if (config.contains(prefix + "preys"))
        {
            data.preys = parseIntList(config[prefix + "preys"]);
        }

        if (config.contains(prefix + "predators"))
        {
            data.predators = parseIntList(config[prefix + "predators"]);
        }

        organismData.push_back(data);
    }
}

void BLOSSOM::init()
{
    // First initialize the SOM grid
    initializeSOM();
    // Then, populate the agent grid
    populate();
    // Finally, initialize logger and log the initial state
    logger = std::make_unique<Logger>(outputDir, outputFileName, gridWidth, gridHeight);
    logger->log(currentStep, agents, somGrid);
}

void BLOSSOM::initializeSOM()
{
    std::uniform_real_distribution<> dist(0, 2 * nutrientMean);
    somGrid.resize(gridWidth, std::vector<double>(gridHeight, 0));

    // Initialize the SOM grid with random values if nutrientType is 0
    // Otherwise, set each grid site to the nutrientMean
    for (int x = 0; x < gridWidth; ++x)
    {
        for (int y = 0; y < gridHeight; ++y)
        {
            somGrid[x][y] = (nutrientType == 0) ? dist(nutrientRNG) : nutrientMean;
        }
    }
}

void BLOSSOM::populate()
{
    agentGrid.resize(gridWidth, std::vector<std::vector<int>>(gridHeight));

    // Populate the grid with organisms in clusters if initialDistributionType is 1
    // Otherwise, populate the grid randomly
    if (initialDistributionType == 1)
    {
        for (size_t type_i = 0; type_i < organismData.size(); ++type_i)
        {
            double biomass = organismData[type_i].params["biomass_reproduction"] / 2.0;
            int numAgents = static_cast<int>(organismData[type_i].params["count"]);

            // Create random clusters of agents
            auto locations = createRandomClusters(numAgents);
            for (const auto &loc : locations)
            {
                addAgent(OrganismGroup(organismId, type_i, dpt(loc.first, loc.second), 0, biomass));
                organismId++; // Increment the organism ID for the new agent
            }
        }
    }
    else
    {
        std::uniform_int_distribution<> distX(0, gridWidth - 1);
        std::uniform_int_distribution<> distY(0, gridHeight - 1);

        for (size_t type_i = 0; type_i < organismData.size(); ++type_i)
        {
            double biomass = organismData[type_i].params["biomass_reproduction"] / 2.0;
            int numAgents = static_cast<int>(organismData[type_i].params["count"]);

            // Randomly place agents in the grid
            for (int i = 0; i < numAgents; ++i)
            {
                int x = distX(initDistRNG);
                int y = distY(initDistRNG);

                addAgent(OrganismGroup(organismId, type_i, dpt(x, y), 0, biomass));
                organismId++; // Increment the organism ID for the new agent
            }
        }
    }
}

const std::vector<std::pair<int, int>> BLOSSOM::createRandomClusters(const int num_individuals)
{
    std::vector<std::pair<int, int>> clusters;
    std::set<std::pair<int, int>> occupied_locations;
    std::uniform_int_distribution<> max_cluster_size(std::max(1, num_individuals / 1000), num_individuals / 50);

    int locations_remaining = num_individuals;

    // While we do not have enough locations
    while (locations_remaining > 0)
    {
        int cluster_size = std::min(locations_remaining, max_cluster_size(initDistRNG));

        std::uniform_real_distribution<> distCenterX(0, gridWidth);
        std::uniform_real_distribution<> distCenterY(0, gridHeight);
        double centerX = distCenterX(initDistRNG);
        double centerY = distCenterY(initDistRNG);

        // Create a cluster of agents around the center
        for (int i = 0; i < cluster_size; ++i)
        {
            std::normal_distribution<> clusterDist(0.0, 1.0);
            int x = static_cast<int>(centerX + clusterDist(initDistRNG));
            int y = static_cast<int>(centerY + clusterDist(initDistRNG));

            x = (x + gridWidth) % gridWidth;
            y = (y + gridHeight) % gridHeight;

            std::pair<int, int> point(x, y);

            // Check if the location is already occupied, if not, add it to the cluster
            if (occupied_locations.find(point) == occupied_locations.end())
            {
                clusters.push_back(point);
                occupied_locations.insert(point);
                locations_remaining--;
            }
        }
    }

    return clusters;
}

// Simulation loop
void BLOSSOM::step()
{
    std::cout << "Step: " << currentStep << " Organisms: " << agents.size() << std::endl;

    // Create a vector of agentIds and shuffle it to ensure random order of processing
    std::vector<int> agentIds;
    for (const auto &p : agents)
    {
        agentIds.push_back(p.first);
    }
    std::shuffle(agentIds.begin(), agentIds.end(), defaultRNG);

    std::vector<OrganismGroup> ogs_to_add;
    std::set<int> ogs_to_kill;

    for (auto &agentId : agentIds)
    {
        // Get a reference to the current agent and check if it is already marked for killing
        auto &agent = agents.find(agentId)->second;
        if (ogs_to_kill.find(agent.getId()) == ogs_to_kill.end())
        {
            // Simulate the agent's step
            simulateAgentStep(agent, ogs_to_kill, ogs_to_add);
        }
    }

    // After processing all agents, handle the new and killed agents
    handleNewAgents(ogs_to_add);
    ogs_to_add.clear();
    handleKilledAgents(ogs_to_kill);
    ogs_to_kill.clear();
}

void BLOSSOM::simulateAgentStep(OrganismGroup &agent, std::set<int> &ogs_to_kill,
                                std::vector<OrganismGroup> &ogs_to_add)
{
    const auto &params = organismData[agent.getType()].params;
    const auto &predators = organismData[agent.getType()].predators;
    const auto &preys = organismData[agent.getType()].preys;
    const auto &location = agent.getLocation();

    // Dispersal logic
    // If bacteria, move randomly in von Neumann neighbourhood
    if (agent.getType() == 0)
    {
        const auto options = vonNeumannR1(location.x, location.y, gridWidth, gridHeight);
        moveAgent(agent, options[defaultRNG() % options.size()]);
    }
    // If not bacteria and fungi, calculate movement probabilities in von Neumann neighbourhood and move
    else if (agent.getType() != 1)
    {
        auto options = vonNeumannRn(location.x, location.y, params.at("range_dispersal"), gridWidth, gridHeight);
        std::vector<double> probs(options.size(), 0.01);

        calculateMovementProbabilities(agent, probs, options, ogs_to_kill, params, preys, predators);

        normalize(probs);
        moveAgent(agent, options[weightedChoice(probs, defaultRNG)]);
    }

    // Feeding logic
    // If the agent is not fully fed
    if (agent.getBiomass() < params.at("biomass_max"))
    {
        // If the agent is a SOM feeder, feed on SOM
        // Otherwise, feed on other agents
        if (params.at("som_feeder") == 1)
        {
            feedOnSOM(agent, location, params);
        }
        else
        {
            feedOnAgents(agent, location, params, preys, ogs_to_kill);
        }
    }

    // Reproduction logic
    // Reproduce if old enough and its biomass is sufficient
    if (agent.getAge() >= params.at("age_reproduction") && agent.getBiomass() >= params.at("biomass_reproduction"))
    {
        reproduce(agent, ogs_to_add);
    }

    // Age logic
    // Kill the agent if it is too old
    if (agent.getAge() >= params.at("age_max"))
    {
        ogs_to_kill.insert(agent.getId());
    }

    // Increment the agent's age
    agent.incrementAge();
}

// Movement
void BLOSSOM::calculateMovementProbabilities(const OrganismGroup &agent, std::vector<double> &probs,
                                             const std::vector<dpt> &options, const std::set<int> &ogs_to_kill,
                                             const std::map<std::string, double> &params, const std::set<int> &preys,
                                             const std::set<int> &predators)
{
    const double biomass_max = params.at("biomass_max");

    // Loop through each option
    for (size_t i = 0; i < options.size(); ++i)
    {
        // First, check if current agent is a SOM feeder
        // If so, set the probability to the SOM value at the option location
        if (params.at("som_feeder") == 1 && agent.getBiomass() < biomass_max)
        {
            probs[i] = somGrid[options[i].x][options[i].y];
        }

        // Then, loop through each agent at the current option location
        for (const auto &neighbour_agent : getAgentsAtLocation(options[i]))
        {
            // Skip if the agent is the same or if it is marked for killing
            if (agent.getId() == neighbour_agent.getId() ||
                ogs_to_kill.find(neighbour_agent.getId()) != ogs_to_kill.end())
            {
                continue;
            }
            // Check if the neighbour agent is a predator of this agent
            // If so, set the probability to a very low value
            // and break the loop for neighbours at the current option location
            if (predators.find(neighbour_agent.getType()) != predators.end())
            {
                probs[i] = 0.00001;
                break;
            }
            // If current agent does not feed on SOM, and the neighbour agent is a prey
            // and the current agent is not fully fed, increase the probability
            if (params.at("som_feeder") == 0 && preys.find(neighbour_agent.getType()) != preys.end() &&
                agent.getBiomass() < biomass_max)
            {
                probs[i] += 1.0;
            }
        }
    }
}

void BLOSSOM::moveAgent(OrganismGroup &agent, const dpt &new_location)
{
    dpt old_location = agent.getLocation();

    // Remove the agent's ID from the old location in the agent grid
    auto &old_grid_agents = agentGrid[old_location.x][old_location.y];
    old_grid_agents.erase(std::remove(old_grid_agents.begin(), old_grid_agents.end(), agent.getId()),
                          old_grid_agents.end());

    // Update the agent's location
    agent.setLocation(new_location);

    // Add the agent's ID to the new location in the agent grid
    agentGrid[new_location.x][new_location.y].push_back(agent.getId());
}

// Feeding
void BLOSSOM::feedOnSOM(OrganismGroup &agent, const dpt &location, const std::map<std::string, double> &params)
{
    // Check if there is SOM available
    // If so, increase the agent's biomass and decrease the SOM value
    // at the current location using Monod's equation
    double food_value = somGrid[location.x][location.y];
    if (food_value > 0.0)
    {
        double biomass_increase = params.at("biomass_max") * food_value / (params.at("k") + food_value);
        biomass_increase = std::min(biomass_increase, food_value);

        agent.increaseBiomass(biomass_increase);
        somGrid[location.x][location.y] -= biomass_increase;

        if (somGrid[location.x][location.y] < 0.0)
        {
            somGrid[location.x][location.y] = 0.0;
        }
    }
}

void BLOSSOM::feedOnAgents(OrganismGroup &agent, const dpt &location, const std::map<std::string, double> &params,
                           const std::set<int> &preys, std::set<int> &ogs_to_kill)
{
    auto agents_at_location = getAgentsAtLocation(location);

    std::vector<double> food_probs;
    std::vector<OrganismGroup> food_opts;

    // Loop through each agent at the current location
    for (auto &agent_at_location : agents_at_location)
    {
        // If the agent is not the same as the current agent
        // and is not marked for killing, and is a prey
        // add it to the food options and its biomass to the food probabilities
        if (agent.getId() != agent_at_location.getId() &&
            ogs_to_kill.find(agent_at_location.getId()) == ogs_to_kill.end() &&
            preys.find(agent_at_location.getType()) != preys.end())
        {
            food_opts.push_back(agent_at_location);
            food_probs.push_back(agent_at_location.getBiomass());
        }
    }

    // If there is at least one prey available, select one pseudorandomly using weightedChoice
    // and increase the agent's biomass and decrease the prey's biomass
    // using Monod's equation, and add the prey to the kill list
    if (!food_probs.empty())
    {
        normalize(food_probs);
        auto prey = food_opts[weightedChoice(food_probs, defaultRNG)];

        double biomass_increase = params.at("biomass_max") * prey.getBiomass() / (params.at("k") + prey.getBiomass());
        biomass_increase = std::min(biomass_increase, prey.getBiomass());

        agent.increaseBiomass(biomass_increase);
        prey.decreaseBiomass(biomass_increase);

        ogs_to_kill.insert(prey.getId());
    }
}

// Reproduction
void BLOSSOM::reproduce(OrganismGroup &agent, std::vector<OrganismGroup> &ogs_to_add)
{
    // Then, if the agent is a fungi, reproduce in a random neighbouring location
    // Otherwise, reproduce in the same location
    auto new_loc = agent.getLocation();
    if (agent.getType() == 1)
    {
        new_loc = vonNeumannR1(agent.getLocation().x, agent.getLocation().y, gridWidth, gridHeight)[defaultRNG() % 5];
    }
    ogs_to_add.push_back(agent.reproduce(organismId, new_loc));
    organismId++; // Increment the organism ID for the new agent
}

// Agent management
void BLOSSOM::handleNewAgents(const std::vector<OrganismGroup> &ogs_to_add)
{
    for (const auto &new_agent : ogs_to_add)
    {
        addAgent(new_agent);
    }
}

void BLOSSOM::handleKilledAgents(const std::set<int> &ogs_to_kill)
{
    for (const auto &agent_id : ogs_to_kill)
    {
        removeAgent(agent_id);
    }
}

void BLOSSOM::addAgent(const OrganismGroup &agent)
{
    // Add the new agent to the agents map and the agent grid
    agents.emplace(std::pair<int, OrganismGroup>(agent.getId(), agent));
    agentGrid[agent.getLocation().x][agent.getLocation().y].push_back(agent.getId());
}

void BLOSSOM::removeAgent(const int agent_id)
{
    // Get a reference to the agent
    auto &agent = agents.find(agent_id)->second;

    // Remove the agent from the agent grid and decrease the SOM value
    auto &location_agents = agentGrid[agent.getLocation().x][agent.getLocation().y];
    location_agents.erase(std::remove(location_agents.begin(), location_agents.end(), agent_id), location_agents.end());

    // Increase the SOM value at the agent's location
    somGrid[agent.getLocation().x][agent.getLocation().y] += agent.getBiomass();

    // Remove the agent from the agents map
    agents.erase(agent_id);
}

// Utility functions
const std::vector<OrganismGroup> BLOSSOM::getAgentsAtLocation(const dpt &location)
{
    std::vector<OrganismGroup> result;

    // Loop through all agentIds at the location
    for (int agent_id : agentGrid[location.x][location.y])
    {
        // add the agent to the result vector
        result.push_back(agents.find(agent_id)->second);
    }
    return result;
}

bool BLOSSOM::shouldStopEarly(const std::unordered_map<int, OrganismGroup> &agents, int min_types) const
{
    std::unordered_set<int> types;
    for (const auto& agentPair : agents) {
        types.insert(agentPair.second.getType());
        if (static_cast<int>(types.size()) >= min_types) return false;
    }

    return true;
}
