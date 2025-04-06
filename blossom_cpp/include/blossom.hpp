#pragma once
#include "organism.hpp"
#include <unordered_map>
#include <vector>
#include <random>
#include <set>
#include <map>
#include <string>

class BLOSSOM {
private:
    std::unordered_map<int, OrganismGroup> agents;
    std::vector<std::vector<std::vector<int>>> agentGrid;
    std::vector<std::vector<int>> somGrid;
    std::vector<OrganismData> organismData;

    int currentStep = 0;
    unsigned long long organismId = 0;
    std::string outputFile;
    unsigned int defaultSeed, initialDistributionSeed, nutrientSeed;
    int initialDistributionType, nutrientType;
    double nutrientMax;
    int maxSteps, gridWidth, gridHeight;

    std::mt19937 defaultRNG, initDistRNG, nutrientRNG;

    void loadConfig(const std::string& filename);
    void init();
    void initialize_som();
    void populate(int init);
    std::vector<std::pair<int, int>> create_random_clusters(int num_individuals, std::mt19937& rng);
    void simulate_agent(OrganismGroup& agent, std::set<int>& to_kill, std::vector<OrganismGroup>& to_add);
    void feed_from_som(OrganismGroup& agent, dpt location, const OrganismData& data);
    void feed_from_other_agents(OrganismGroup& agent, dpt location, const OrganismData& data, std::set<int>& to_kill);
    void reproduce(OrganismGroup& agent, std::vector<OrganismGroup>& to_add);
    void update_probabilities(std::vector<double>& probs, const std::vector<dpt>& options, const std::set<int>& to_kill, const OrganismData& data);
    void handle_killed_agents(const std::set<int>& to_kill);
    void handle_new_agents(const std::vector<OrganismGroup>& to_add);
    void add_agent(const OrganismGroup &agent);
    void remove_agent(int agent_id);
    void log();
    void step();
    std::vector<OrganismGroup> get_agents_at_location(dpt location);


public:
    BLOSSOM();
    void run();
};
