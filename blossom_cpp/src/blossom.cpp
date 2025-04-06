#include "blossom.hpp"
#include "utils.hpp"
#include "neighbourhoods.hpp"
#include "organism.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

BLOSSOM::BLOSSOM()
{
    loadConfig("./config/config.props");

    // Initialize RNGs
    defaultRNG = std::mt19937(defaultSeed);
    initDistRNG = std::mt19937(initialDistributionSeed);
    nutrientRNG = std::mt19937(nutrientSeed);

    init();
}

// Load configuration from the props file
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

    // Load general configuration
    outputFile = config["output_file"];
    defaultSeed = std::stoi(config["default_seed"]);
    initialDistributionType = std::stoi(config["initial_distribution_type"]);
    initialDistributionSeed = std::stoi(config["initial_distribution_seed"]);
    nutrientType = std::stoi(config["nutrient_type"]);
    nutrientSeed = std::stoi(config["nutrient_seed"]);
    nutrientMax = std::stod(config["nutrient_max"]);
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
    initialize_som();
    populate(initialDistributionType);
}

void BLOSSOM::initialize_som()
{
    std::uniform_int_distribution<> dist(0, 2 * nutrientMax);

    somGrid.resize(gridWidth, std::vector<int>(gridHeight, 0));

    for (int x = 0; x < gridWidth; ++x)
    {
        for (int y = 0; y < gridHeight; ++y)
        {
            somGrid[x][y] = (nutrientType == 0) ? dist(nutrientRNG) : nutrientMax;
        }
    }
}

// Populate function (initialize agents)
void BLOSSOM::populate(int init)
{
    agentGrid.resize(gridWidth, std::vector<std::vector<int>>(gridHeight));

    std::uniform_real_distribution<> distX(0, gridWidth - 1);
    std::uniform_real_distribution<> distY(0, gridHeight - 1);

    if (init == 1)
    { // Clustered initialization
        for (int type_i = 0; static_cast<size_t>(type_i) < organismData.size(); ++type_i)
        {
            double biomass = organismData[type_i].params["biomass_reproduction"] / 2.0;
            int numAgents = static_cast<int>(organismData[type_i].params["count"]);
            // Create clustered locations
            auto locations = create_random_clusters(numAgents, initDistRNG);
            // Add agents at those locations
            for (const auto &loc : locations)
            {
                agents.emplace(std::pair<int, OrganismGroup>(organismId, OrganismGroup(organismId, type_i, dpt(loc.first, loc.second), 0, biomass)));
                agentGrid[loc.first][loc.second].push_back(organismId);
                organismId++;
                // Ensure unique IDs
            }
        }
    }
    else
    { // Random initialization
        for (int type_i = 0; static_cast<size_t>(type_i) < organismData.size(); ++type_i)
        {
            double biomass = organismData[type_i].params["biomass_reproduction"] / 2.0;
            int numAgents = static_cast<int>(organismData[type_i].params["count"]);

            for (int i = 0; i < numAgents; ++i)
            {
                int x = static_cast<int>(distX(initDistRNG));
                int y = static_cast<int>(distY(initDistRNG));

                // Add random agents
                agents.emplace(std::pair<int, OrganismGroup>(organismId, OrganismGroup(organismId, type_i, dpt(x, y), 0, biomass)));
                agentGrid[x][y].push_back(organismId);
                organismId++;
                // Ensure unique IDs
            }
        }
    }
}

// Cluster creation function
std::vector<std::pair<int, int>> BLOSSOM::create_random_clusters(int num_individuals, std::mt19937 &rng)
{
    std::vector<std::pair<int, int>> clusters;
    std::set<std::pair<int, int>> occupied_locations;

    int locations_remaining = num_individuals;

    while (locations_remaining > 0)
    {
        int cluster_size = std::min(locations_remaining, 100); // Cluster size range

        // Randomly determine the center of the next cluster
        std::uniform_real_distribution<> distCenterX(0, gridWidth);
        std::uniform_real_distribution<> distCenterY(0, gridHeight);
        double centerX = distCenterX(rng);
        double centerY = distCenterY(rng);

        for (int i = 0; i < cluster_size; ++i)
        {
            std::normal_distribution<> clusterDist(0.0, 1.0);
            int x = static_cast<int>(centerX + clusterDist(rng));
            int y = static_cast<int>(centerY + clusterDist(rng));

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

std::vector<OrganismGroup> BLOSSOM::get_agents_at_location(dpt location)
{
    std::vector<OrganismGroup> result;
    // Iterate over the agent IDs stored in the agentGrid at the specified location
    for (int agent_id : agentGrid[location.x][location.y])
    {
        // Check if the agent_id exists in the agents map using find
        auto it = agents.find(agent_id);
        if (it != agents.end())
        {
            result.push_back(it->second); // Push the reference to the agent
        }
        // Optionally handle the case when the agent_id doesn't exist in the map
        else
        {
            // For example, log or skip if needed
            // std::cerr << "Agent ID " << agent_id << " not found in the agents map.\n";
        }
    }
    return result;
}

void BLOSSOM::step()
{
    std::cout << "Step: " << currentStep << " Organisms: " << agents.size() << std::endl;

    std::vector<int> agentIds;
    for (const auto &p : agents)
    {
        agentIds.push_back(p.first);
    }

    std::shuffle(agentIds.begin(), agentIds.end(), defaultRNG);

    // Lists to hold agents to add and remove
    std::vector<OrganismGroup> ogs_to_add;
    std::set<int> ogs_to_kill;

    for (auto &agentId : agentIds)
    {
        auto it = agents.find(agentId);
        if (it != agents.end())
        {
            auto &agent = it->second; // Use reference to the found OrganismGroup
            if (ogs_to_kill.find(agent.getId()) == ogs_to_kill.end())
            {
                simulate_agent(agent, ogs_to_kill, ogs_to_add);
            }
        }
        else
        {
            // Handle case where agentId is not found in the map (optional)
            std::cerr << "Agent ID " << agentId << " not found in the map.\n";
        }
    }

    // Handle adding new agents and removing killed agents
    handle_new_agents(ogs_to_add);
    ogs_to_add.clear();
    handle_killed_agents(ogs_to_kill);
    ogs_to_kill.clear();
}

void BLOSSOM::simulate_agent(OrganismGroup &agent, std::set<int> &ogs_to_kill, std::vector<OrganismGroup> &ogs_to_add)
{
    // Fetch agent parameters from a config or predefined array
    const auto &organism_data = organismData[agent.getType()];
    dpt location = agent.getLocation();

    // Dispersal logic (like the von Neumann dispersal in Python)
    if (agent.getType() == 0)
    {
        // Special dispersal for bacteria
        auto options = von_neumann_r1(location.x, location.y, gridWidth, gridHeight);
        // Move agent using random selection
        agent.move(options[defaultRNG() % options.size()]);
    }
    else if (agent.getType() != 1)
    {
        // Regular dispersal
        auto options = von_neumann_neighborhood_2d(location.x, location.y, organism_data.params.at("range_dispersal"), gridWidth, gridHeight);
        std::vector<double> probs(options.size(), 0.01);

        // Modify probabilities based on prey and predator interactions
        update_probabilities(probs, options, ogs_to_kill, organism_data);

        // Normalize probabilities and move agent
        normalize(probs);
        agent.move(options[weighted_choice(probs, defaultRNG)]);
    }

    // Feeding logic (if biomass is lower than biomass_max)
    if (agent.getBiomass() < organism_data.params.at("biomass_max"))
    {
        if (organism_data.params.at("som_feeder") == 1)
        {
            // SOM feeding logic
            feed_from_som(agent, location, organism_data);
        }
        else
        {
            // Run agent-agent feeding logic
            feed_from_other_agents(agent, location, organism_data, ogs_to_kill);
        }
    }

    // Reproduction logic
    if (agent.getAge() >= organism_data.params.at("age_reproduction") && agent.getBiomass() >= organism_data.params.at("biomass_reproduction"))
    {
        reproduce(agent, ogs_to_add);
    }

    // Age logic
    if (agent.getAge() >= organism_data.params.at("age_max"))
    {
        ogs_to_kill.insert(agent.getId());
    }

    agent.incrementAge();
}

// This function updates the probabilities based on prey and predator interactions
void BLOSSOM::update_probabilities(std::vector<double> &probs, const std::vector<dpt> &options, const std::set<int> &ogs_to_kill,
                          const OrganismData &organism_data)
{
    double biomass_max = organism_data.params.at("biomass_max");
    std::set<int> preys = organism_data.preys;
    std::set<int> predators = organism_data.predators;

    // Iterate through each dispersal option
    for (size_t i = 0; i < options.size(); ++i)
    {
        // Check if the location is already occupied by an agent to be killed
        for (const auto &neighbor_agent : get_agents_at_location(options[i]))
        {
            if (ogs_to_kill.find(neighbor_agent.getId()) != ogs_to_kill.end())
            {
                continue;
            }
            // If predator is found, decrease the probability significantly
            if (predators.find(neighbor_agent.getType()) != predators.end())
            {
                probs[i] = 0.00001;
                break;
            }
            // If prey is found, increase the probability
            if (preys.find(neighbor_agent.getType()) != preys.end())
            {
                probs[i] += 1.0; // Increase probability if prey found
            }
        }

        // If som feeder (based on biomass), consider value layer for food availability
        if (organism_data.params.at("som_feeder") == 1)
        {
            double food_available = somGrid[options[i].x][options[i].y]; // Assume this function
            double uptake = biomass_max * food_available / (organism_data.params.at("k") + food_available);
            probs[i] = std::min(1.0, probs[i] + uptake); // Update with food uptake
        }
    }
}

void BLOSSOM::feed_from_som(OrganismGroup &agent, dpt location, const OrganismData &organism_data)
{
    // Get food value from SOM grid
    double food_value = somGrid[location.x][location.y]; // Assume this function exists
    if (food_value > 0.0)
    {
        double biomass_increase = organism_data.params.at("biomass_max") * food_value / (organism_data.params.at("k") + food_value);
        agent.increaseBiomass(biomass_increase);
        somGrid[location.x][location.y] -= food_value; // Decrease SOM value
        if (somGrid[location.x][location.y] < 0.0)
        {
            somGrid[location.x][location.y] = 0.0; // Ensure non-negative SOM value
        }
    }
}

void BLOSSOM::feed_from_other_agents(OrganismGroup &agent, dpt location, const OrganismData &organism_data, std::set<int> &ogs_to_kill)
{
    // Get agents at current location
    auto food_opts = get_agents_at_location(location);
    std::vector<double> food_probs;

    for (auto &food_agent : food_opts)
    {
        if (food_agent.getId() != agent.getId() &&
            ogs_to_kill.find(food_agent.getId()) == ogs_to_kill.end() &&
            organism_data.preys.find(food_agent.getType()) != organism_data.preys.end())
        { // Check if food agent is prey
            food_probs.push_back(food_agent.getBiomass());
        }
    }

    if (!food_probs.empty())
    {
        // Normalize probabilities and choose prey
        normalize(food_probs);
        auto prey = food_opts[weighted_choice(food_probs, defaultRNG)];

        // Feed on prey
        agent.increaseBiomass(prey.getBiomass());
        ogs_to_kill.insert(prey.getId());
    }
}

void BLOSSOM::reproduce(OrganismGroup &agent, std::vector<OrganismGroup> &ogs_to_add)
{
    agent.divideBiomass();
    ogs_to_add.push_back(agent.save(organismId)); // Save the new agent
    organismId++;
}

void BLOSSOM::handle_new_agents(const std::vector<OrganismGroup> &ogs_to_add)
{
    for (const auto &new_agent : ogs_to_add)
    {
        add_agent(new_agent);
    }
}

void BLOSSOM::handle_killed_agents(const std::set<int> &ogs_to_kill)
{
    for (const auto &agent_id : ogs_to_kill)
    {
        remove_agent(agent_id);
    }
}

void BLOSSOM::add_agent(const OrganismGroup &agent)
{
    agents.emplace(std::pair<int, OrganismGroup>(agent.getId(), agent));
    agentGrid[agent.getLocation().x][agent.getLocation().y].push_back(agent.getId());
}

void BLOSSOM::remove_agent(int agent_id)
{
    auto it = agents.find(agent_id);
    if (it != agents.end())
    {
        auto &agent = it->second; // Use reference to the found OrganismGroup
        auto &location_agents = agentGrid[agent.getLocation().x][agent.getLocation().y];
        location_agents.erase(std::remove(location_agents.begin(), location_agents.end(), agent_id), location_agents.end());
        agents.erase(agent_id);
    }
    else
    {
        // Handle case where agentId is not found in the map (optional)
        std::cerr << "Agent ID " << agent_id << " not found in the map.\n";
    }
}

void BLOSSOM::log()
{
    std::ofstream outFile(outputFile, std::ios::app); // Open in append mode

    if (!outFile.is_open())
    {
        std::cerr << "Failed to open log file.\n";
        return;
    }
    if (currentStep == 0)
    {
        outFile << "tick,id,type,x,y,age,biomass\n"; // Header
    }

    for (const auto &p : agents)
    {
        const auto &agent = p.second;
        outFile << currentStep << ","
                << agent.getId() << ","
                << agent.getType() << ","
                << agent.getLocation().x << ","
                << agent.getLocation().y << ","
                << agent.getAge() << ","
                << agent.getBiomass() << "\n";
    }
    outFile.close();
}

void BLOSSOM::run()
{
    while (currentStep < maxSteps)
    {
        step();
        log();
        currentStep++;
    }
}