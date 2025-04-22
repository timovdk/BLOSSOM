#pragma once

#include "organism.hpp"
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <string>

#pragma pack(push, 1)
struct AgentLogEntry
{
    uint32_t id;
    float biomass;
    uint16_t tick;
    uint16_t x;
    uint16_t y;
    uint8_t type;
    uint8_t age;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct SOMLogEntry
{
    float som_value;
    uint16_t tick;
    uint16_t x;
    uint16_t y;
};
#pragma pack(pop)

class Logger
{
  public:
    Logger(std::string outputDir, std::string outputFileName, int gridWidth, int gridHeight, const bool logging);
    ~Logger();
    void log(int currentStep, const std::unordered_map<int, OrganismGroup> &agents,
             const std::vector<std::vector<double>> &somGrid);
    void reset();

  private:
    std::string agentDir;
    std::string somDir;
    std::string outputFileName;

    std::ofstream outFileAgents;
    std::ofstream outFileSOM;

    size_t linesLogged = 0;
    size_t maxLinesPerFile = 100000000; // 100M lines
    int ticksLogged = 0;
    int maxTicksPerFile = 1000; // 1000 ticks
    int fileIndex = 0;
    int fileIndexSOM = 0;
    int gridWidth;
    int gridHeight;
    bool shouldLog;

    void logAgents(int currentStep, const std::unordered_map<int, OrganismGroup> &agents);
    void logSOM(int currentStep, const std::vector<std::vector<double>> &somGrid);
};
