#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <set>
#include "NeighborhoodOffsets.hpp"

unsigned long long index_counter = 0;

const int seed = 28910743;
const int max_tick = 600;
const int x_max = 400;
const int y_max = 400;
const int z_max = 1;
const int num_types = 9;
const int max_nutrients = 0.0075;
const auto types = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8};
const auto counts = std::vector<int>{40000, 15000, 5500, 7500, 5500, 4000, 1000, 500, 1000};
const auto ranges = std::vector<int>{1, 0, 3, 3, 3, 4, 5, 6, 6};
const auto max_ages = std::vector<int>{1, 9, 10, 15, 21, 22, 60, 70, 17};
const auto repr_ages = std::vector<int>{0, 1, 8, 10, 19, 16, 40, 35, 15};
const auto bm_max = std::vector<double>{0.000975, 0.001112, 0.001129, 0.001019, 0.000829, 0.001037, 0.001056, 0.00132, 0.001063};
const auto bm_repr = std::vector<double>{0.0005, 0.001112, 0.0006, 0.00051, 0.00035, 0.000519, 0.000528, 0.000728, 0.000582};
const auto ks = std::vector<double>{0.026, 0.047, 0.15, 0.00295, 0.0025, 0.009, 0.048, 0.025, 0.02};

const auto preys = std::vector<std::vector<int>>{{9}, {9}, {9}, {0}, {1}, {0, 2, 3, 4}, {1}, {2, 3, 4, 5, 6, 8}, {1}};
const auto predators = std::vector<std::vector<int>>{{3, 5}, {8, 4, 6}, {5, 7}, {5, 7}, {5, 7}, {7}, {7}, {}, {7}};

enum AgentType
{
    BACTERIA = 0,
    FUNGI = 1,
    RF_NEMATODES = 2,
    B_NEMATODES = 3,
    F_NEMATODES = 4,
    O_NEMATODES = 5,
    F_MITES = 6,
    O_MITES = 7,
    COLLEMBOLANS = 8
};

// Agent structure
struct Agent
{
    int id;
    AgentType type;
    int x, y, z;
    double biomass;
    int age;

    Agent(int id, AgentType t, int x, int y, int z, double biomass, int age) : id(id), type(t), x(x), y(y), z(z), biomass(biomass), age(age) {}
};

// 3D grid flattened into a 1D vector
using Agent_grid = std::vector<std::vector<Agent *>>;
using SOM_grid = std::vector<double>;

// Function to create a grid of given size
Agent_grid createAgentGrid()
{
    Agent_grid grid(x_max * y_max * z_max);
    for (auto &cell : grid)
    {
        cell.reserve(100); // Reserve memory for agent pointers
    }
    return grid;
}

// Function to create a grid of given size
SOM_grid createSOMGrid()
{
    SOM_grid grid(x_max * y_max * z_max, max_nutrients);
    
    return grid;
}

// Convert 3D coordinates to a 1D index for flattened grid with unequal dimensions
inline int index3D(int x, int y, int z)
{
    return x * y_max * z_max + y * z_max + z;
}

// Wrap around the grid edges (toroidal wrapping) for unequal dimensions
inline int wrapAround(int coordinate, int maxVal)
{
    return (coordinate + maxVal) % maxVal;
}

// Function to get von Neumann neighbors for an agent
std::vector<Agent *> getVonNeumannNeighbors(const Agent &agent, const Agent_grid &grid)
{
    std::vector<Agent *> neighbors;
    neighbors.reserve(30); // Pre-allocate memory for performance

    // Get agent-specific range based on type
    int r = ranges[agent.type];

    // Get precomputed offsets
    const auto &offsets = getOffsetsForR(r);

    // Calculate neighbors and collect agents
    for (const auto &offset : offsets)
    {
        int dx, dy, dz;
        std::tie(dx, dy, dz) = offset;

        int nx = wrapAround(agent.x + dx, x_max);
        int ny = wrapAround(agent.y + dy, y_max);
        int nz = wrapAround(agent.z + dz, z_max);

        int idx = index3D(nx, ny, nz);
        neighbors.insert(neighbors.end(), grid[idx].begin(), grid[idx].end());
    }

    return neighbors;
}

// Function to move an agent if no Y agent is in the target cell
void moveAgent(Agent *agent, Agent_grid &grid)
{
    int oldX = agent->x, oldY = agent->y, oldZ = agent->z;

    // Randomly pick a neighboring cell within von Neumann neighborhood
    int dx = (rand() % 3) - 1;
    int dy = (rand() % 3) - 1;
    int dz = (rand() % 3) - 1;

    int newX = wrapAround(oldX + dx, x_max);
    int newY = wrapAround(oldY + dy, y_max);
    int newZ = wrapAround(oldZ + dz, z_max);

    int oldIdx = index3D(oldX, oldY, oldZ);
    int newIdx = index3D(newX, newY, newZ);
    // Move agent to the new position
    grid[oldIdx].erase(std::remove(grid[oldIdx].begin(), grid[oldIdx].end(), agent), grid[oldIdx].end());
    agent->x = newX;
    agent->y = newY;
    agent->z = newZ;
    grid[newIdx].push_back(agent);
}

// Function to move an agent if no Y agent is in the target cell
void moveAgentLoc(Agent *agent, Agent_grid &grid, int x, int y, int z)
{
    int oldX = agent->x, oldY = agent->y, oldZ = agent->z;

    // Randomly pick a neighboring cell within von Neumann neighborhood
    int dx = (rand() % 3) - 1;
    int dy = (rand() % 3) - 1;
    int dz = (rand() % 3) - 1;

    int newX = wrapAround(oldX + dx, x_max);
    int newY = wrapAround(oldY + dy, y_max);
    int newZ = wrapAround(oldZ + dz, z_max);

    int oldIdx = index3D(oldX, oldY, oldZ);
    int newIdx = index3D(newX, newY, newZ);
    // Move agent to the new position
    grid[oldIdx].erase(std::remove(grid[oldIdx].begin(), grid[oldIdx].end(), agent), grid[oldIdx].end());
    agent->x = newX;
    agent->y = newY;
    agent->z = newZ;
    grid[newIdx].push_back(agent);
}

// Function to log the agent's data to CSV
void logAgentsToCSV(const std::vector<Agent> &agents, std::ofstream &outFile)
{
    for (const auto &agent : agents)
    {
        outFile << agent.id << "," << agent.type << ","
                << agent.x << "," << agent.y << "," << agent.z << "\n";
    }
}

int main()
{

    Agent_grid agent_grid = createAgentGrid();
    SOM_grid som_grid = createSOMGrid();

    // Create some agents of type X and Y
    std::vector<Agent> agents;
    for (int type = 0; type < num_types; ++type)
    {
        for (int count = 0; count < counts[type]; ++count)
        {
            agents.emplace_back(index_counter, static_cast<AgentType>(type), rand() % x_max, rand() % y_max, rand() % z_max, bm_repr[type] / 2, 0);
        }
    }

    // Place agents in the grid
    for (Agent &agent : agents)
    {
        int idx = index3D(agent.x, agent.y, agent.z);
        agent_grid[idx].push_back(&agent);
    }

    // Open CSV file for logging
    std::ofstream outFile("agent_log.csv");
    outFile << "ID,Type,X,Y,Z\n"; // CSV Header

    // Run the simulation for a few time steps
    const int timeSteps = 10;
    for (int t = 0; t < timeSteps; ++t)
    {
        auto agents_to_kill = std::vector<Agent *>{};
        auto agents_to_add = std::vector<Agent *>{};
        auto ids_to_kill = std::set<int>{};
        // Move each agent
        for (Agent &agent : agents)
        {
            if(ids_to_kill.count(agent.id))
            {
                if(agent.type == AgentType::BACTERIA)
                {
                    moveAgent(&agent, agent_grid);
                }
                else if (agent.type != AgentType::FUNGI)
                {
                    auto options = getVonNeumannNeighbors(agent, agent_grid)

                }
                

            }
            
            
            if(agent.type != AgentType::FUNGI){
                moveAgent(&agent, agent_grid);
            }
            agent.age++;            
        }

        // Log agents to CSV
        logAgentsToCSV(agents, outFile);
    }

    outFile.close();
    return 0;
}
