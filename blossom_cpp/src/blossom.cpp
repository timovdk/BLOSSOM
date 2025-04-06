#include "blossom.hpp"
#include "neighbourhoods.hpp"
#include "organism.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

BLOSSOM::BLOSSOM()
{
    loadConfig("./config/config.props");

    defaultRNG = std::mt19937(defaultSeed);
    initDistRNG = std::mt19937(initialDistributionSeed);
    nutrientRNG = std::mt19937(nutrientSeed);

    init();
}

void BLOSSOM::run()
{
    while (currentStep < maxSteps)
    {
        currentStep++;
        step();
        log(false);
    }
}

// Setup
void BLOSSOM::loadConfig(const std::string &filename)
{
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    std::string line;

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value))
        {
            config[key] = value;
        }
    }

    outputFile = config["output_file"];
    defaultSeed = std::stoi(config["default_seed"]);
    initialDistributionType = std::stoi(config["initial_distribution_type"]);
    initialDistributionSeed = std::stoi(config["initial_distribution_seed"]);
    nutrientType = std::stoi(config["nutrient_type"]);
    nutrientSeed = std::stoi(config["nutrient_seed"]);
    nutrientMean = std::stod(config["nutrient_mean"]);
    maxSteps = std::stoi(config["max_steps"]);
    gridWidth = std::stoi(config["grid_width"]);
    gridHeight = std::stoi(config["grid_height"]);

    for (int i = 1; i <= 9; ++i)
    {
        OrganismData data;
        std::string prefix = "organism_" + std::to_string(i) + "_";

        data.params["biomass_reproduction"] = std::stod(config[prefix + "biomass_reproduction"]);
        data.params["count"] = std::stoi(config[prefix + "count"]);
        data.params["range_dispersal"] = std::stoi(config[prefix + "range_dispersal"]);
        data.params["age_reproduction"] = std::stoi(config[prefix + "age_reproduction"]);
        data.params["age_max"] = std::stoi(config[prefix + "age_max"]);
        data.params["biomass_max"] = std::stod(config[prefix + "biomass_max"]);
        data.params["k"] = std::stod(config[prefix + "k"]);
        data.params["som_feeder"] = std::stoi(config[prefix + "som_feeder"]);

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
    initializeSOM();
    populate();
    log(true);
}

void BLOSSOM::initializeSOM()
{
    std::uniform_real_distribution<> dist(0, 2 * nutrientMean);
    somGrid.resize(gridWidth, std::vector<double>(gridHeight, 0));

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

    if (initialDistributionType == 1)
    {
        for (int type_i = 0; static_cast<size_t>(type_i) < organismData.size(); ++type_i)
        {
            double biomass = organismData[type_i].params["biomass_reproduction"] / 2.0;
            int numAgents = static_cast<int>(organismData[type_i].params["count"]);
            auto locations = createRandomClusters(numAgents);
            for (const auto &loc : locations)
            {
                addAgent(OrganismGroup(organismId, type_i, dpt(loc.first, loc.second), 0, biomass));
                organismId++;
            }
        }
    }
    else
    {
        std::uniform_int_distribution<> distX(0, gridWidth - 1);
        std::uniform_int_distribution<> distY(0, gridHeight - 1);

        for (int type_i = 0; static_cast<size_t>(type_i) < organismData.size(); ++type_i)
        {
            double biomass = organismData[type_i].params["biomass_reproduction"] / 2.0;
            int numAgents = static_cast<int>(organismData[type_i].params["count"]);

            for (int i = 0; i < numAgents; ++i)
            {
                int x = distX(initDistRNG);
                int y = distY(initDistRNG);

                addAgent(OrganismGroup(organismId, type_i, dpt(x, y), 0, biomass));
                organismId++;
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

    while (locations_remaining > 0)
    {
        int cluster_size = std::min(locations_remaining, max_cluster_size(initDistRNG));

        std::uniform_real_distribution<> distCenterX(0, gridWidth);
        std::uniform_real_distribution<> distCenterY(0, gridHeight);
        double centerX = distCenterX(initDistRNG);
        double centerY = distCenterY(initDistRNG);

        for (int i = 0; i < cluster_size; ++i)
        {
            std::normal_distribution<> clusterDist(0.0, 1.0);
            int x = static_cast<int>(centerX + clusterDist(initDistRNG));
            int y = static_cast<int>(centerY + clusterDist(initDistRNG));

            x = (x + gridWidth) % gridWidth;
            y = (y + gridHeight) % gridHeight;

            std::pair<int, int> point(x, y);

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
        auto it = agents.find(agentId);
        if (it != agents.end())
        {
            auto &agent = it->second;
            if (ogs_to_kill.find(agent.getId()) == ogs_to_kill.end())
            {
                simulateAgentStep(agent, ogs_to_kill, ogs_to_add);
            }
        }
    }

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
    if (agent.getType() == 0)
    {
        const auto options = vonNeumannR1(location.x, location.y, gridWidth, gridHeight);
        moveAgent(agent, options[defaultRNG() % options.size()]);
    }
    else if (agent.getType() != 1)
    {
        auto options = vonNeumannRn(location.x, location.y, params.at("range_dispersal"), gridWidth, gridHeight);
        std::vector<double> probs(options.size(), 0.01);

        calculateMovementProbabilities(agent, probs, options, ogs_to_kill, params, preys, predators);

        normalize(probs);
        moveAgent(agent, options[weighted_choice(probs, defaultRNG)]);
    }

    // Feeding logic
    if (agent.getBiomass() < params.at("biomass_max"))
    {
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
    if (agent.getAge() >= params.at("age_reproduction") && agent.getBiomass() >= params.at("biomass_reproduction"))
    {
        reproduce(agent, ogs_to_add);
    }

    // Age logic
    if (agent.getAge() >= params.at("age_max"))
    {
        ogs_to_kill.insert(agent.getId());
    }

    agent.incrementAge();
}

// Movement
void BLOSSOM::calculateMovementProbabilities(const OrganismGroup &agent, std::vector<double> &probs,
                                             const std::vector<dpt> &options, const std::set<int> &ogs_to_kill,
                                             const std::map<std::string, double> &params, const std::set<int> &preys,
                                             const std::set<int> &predators)
{
    const double biomass_max = params.at("biomass_max");

    for (size_t i = 0; i < options.size(); ++i)
    {
        for (const auto &neighbor_agent : getAgentsAtLocation(options[i]))
        {
            if (ogs_to_kill.find(neighbor_agent.getId()) != ogs_to_kill.end())
            {
                continue;
            }
            if (predators.find(neighbor_agent.getType()) != predators.end())
            {
                probs[i] = 0.00001;
                break;
            }
            if (params.at("som_feeder") == 0 && preys.find(neighbor_agent.getType()) != preys.end() &&
                agent.getBiomass() < biomass_max)
            {
                probs[i] += 1.0;
            }
            if (params.at("som_feeder") == 1 && agent.getBiomass() < biomass_max)
            {
                probs[i] = somGrid[options[i].x][options[i].y];
            }
        }
    }
}

void BLOSSOM::moveAgent(OrganismGroup &agent, const dpt &new_location)
{
    dpt old_location = agent.getLocation();

    auto &old_grid_agents = agentGrid[old_location.x][old_location.y];
    old_grid_agents.erase(std::remove(old_grid_agents.begin(), old_grid_agents.end(), agent.getId()),
                          old_grid_agents.end());

    agent.setLocation(new_location);

    agentGrid[new_location.x][new_location.y].push_back(agent.getId());
}

// Feeding
void BLOSSOM::feedOnSOM(OrganismGroup &agent, const dpt &location, const std::map<std::string, double> &params)
{
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

    for (auto &agent_at_location : agents_at_location)
    {
        if (agent.getId() != agent_at_location.getId() &&
            ogs_to_kill.find(agent_at_location.getId()) == ogs_to_kill.end() &&
            preys.find(agent_at_location.getType()) != preys.end())
        {
            food_opts.push_back(agent_at_location);
            food_probs.push_back(agent_at_location.getBiomass());
        }
    }

    if (!food_probs.empty())
    {
        normalize(food_probs);
        auto prey = food_opts[weighted_choice(food_probs, defaultRNG)];

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
    agent.divideBiomass();
    auto new_loc = agent.getType() == 1 ? vonNeumannR1(agent.getLocation().x, agent.getLocation().y, gridWidth,
                                                       gridHeight)[defaultRNG() % 5]
                                        : agent.getLocation();

    ogs_to_add.push_back(agent.reproduce(organismId, new_loc));
    organismId++;
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
    agents.emplace(std::pair<int, OrganismGroup>(agent.getId(), agent));
    agentGrid[agent.getLocation().x][agent.getLocation().y].push_back(agent.getId());
}

void BLOSSOM::removeAgent(const int agent_id)
{
    auto it = agents.find(agent_id);
    if (it != agents.end())
    {
        auto &agent = it->second;
        auto &location_agents = agentGrid[agent.getLocation().x][agent.getLocation().y];
        location_agents.erase(std::remove(location_agents.begin(), location_agents.end(), agent_id),
                              location_agents.end());

        somGrid[agent.getLocation().x][agent.getLocation().y] += agent.getBiomass();
        agents.erase(agent_id);
    }
}

// Utility functions
const std::vector<OrganismGroup> BLOSSOM::getAgentsAtLocation(const dpt &location)
{
    std::vector<OrganismGroup> result;
    for (int agent_id : agentGrid[location.x][location.y])
    {
        auto it = agents.find(agent_id);
        if (it != agents.end())
        {
            result.push_back(it->second);
        }
    }
    return result;
}

void BLOSSOM::log(const bool first_time)
{
    if (first_time)
    {
        std::ofstream outFile(outputFile);
        if (!outFile.is_open())
        {
            std::cerr << "Failed to open log file.\n";
            return;
        }
        outFile << "tick,id,type,x,y,age,biomass\n";
        outFile.close();
    }

    std::ofstream outFile(outputFile, std::ios::app);

    if (!outFile.is_open())
    {
        std::cerr << "Failed to open log file.\n";
        return;
    }

    for (const auto &p : agents)
    {
        const auto &agent = p.second;
        outFile << currentStep << "," << agent.getId() << "," << agent.getType() << "," << agent.getLocation().x << ","
                << agent.getLocation().y << "," << agent.getAge() << "," << agent.getBiomass() << "\n";
    }
    outFile.close();
}
