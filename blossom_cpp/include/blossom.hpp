#pragma once

#include "logger.hpp"
#include "organism.hpp"
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

class BLOSSOM
{
  private:
    // Core data structures
    std::unordered_map<int, OrganismGroup> agents;
    std::vector<std::vector<std::vector<int>>> agentGrid;
    std::vector<std::vector<double>> somGrid;
    std::vector<OrganismData> organismData;

    // Simulation parameters
    int currentStep = 0;
    int trialID = 0;
    unsigned long long organismId = 0;
    std::string outputDir, outputFileName;

    // Config parameters
    unsigned int defaultSeed, initialDistributionSeed, nutrientSeed;
    int initialDistributionType, nutrientType;
    double nutrientMean;
    int maxSteps, gridWidth, gridHeight;
    int earlyStopInterval, earlyStopMinTypes;

    // RNGs
    std::mt19937 defaultRNG, initDistRNG, nutrientRNG;

    // Logger
    std::unique_ptr<Logger> logger;

    // Setup
    void loadConfig(const std::string &filename);
    void init();
    void initializeSOM();
    void populate();
    const std::vector<std::pair<int, int>> createRandomClusters(const int num_individuals);

    // Simulation loop
    void step();
    void simulateAgentStep(OrganismGroup &agent, std::set<int> &to_kill, std::vector<OrganismGroup> &to_add);

    // Movement
    void calculateMovementProbabilities(const OrganismGroup &agent, std::vector<double> &probs,
                                        const std::vector<dpt> &options, const std::set<int> &ogs_to_kill,
                                        const std::map<std::string, double> &params, const std::set<int> &preys,
                                        const std::set<int> &predators);
    void moveAgent(OrganismGroup &agent, const dpt &new_location);

    // Feeding
    void feedOnSOM(OrganismGroup &agent, const dpt &location, const std::map<std::string, double> &params);
    void feedOnAgents(OrganismGroup &agent, const dpt &location, const std::map<std::string, double> &params,
                      const std::set<int> &preys, std::set<int> &ogs_to_kill);

    // Reproduction
    void reproduce(OrganismGroup &agent, std::vector<OrganismGroup> &to_add);

    // Agent management
    void handleNewAgents(const std::vector<OrganismGroup> &to_add);
    void handleKilledAgents(const std::set<int> &to_kill);
    void addAgent(const OrganismGroup &agent);
    void removeAgent(const int agent_id);

    // Utility functions
    const std::vector<OrganismGroup> getAgentsAtLocation(const dpt &location);
    bool shouldStopEarly(const std::unordered_map<int, OrganismGroup> &agents, int min_types) const;

  public:
    BLOSSOM(const int index);
    void run();
};
