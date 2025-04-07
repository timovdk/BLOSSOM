#pragma once

#include "organism.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#pragma pack(push, 1)
struct AgentLogEntry
{
    int tick;
    int id;
    int type;
    int x;
    int y;
    int age;
    float biomass;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct SOMLogEntry
{
    int tick;
    int x;
    int y;
    float som_value;
};
#pragma pack(pop)

class Logger
{
  public:
    Logger(std::string outputDir, std::string outputFileName, int gridWidth, int gridHeight);
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
    size_t maxLinesPerFile = 40000000;
    int ticksLogged = 0;
    int maxTicksPerFile = 250;
    int fileIndex = 0;
    int fileIndexSOM = 0;
    int gridWidth;
    int gridHeight;

    void logAgents(int currentStep, const std::unordered_map<int, OrganismGroup> &agents);
    void logSOM(int currentStep, const std::vector<std::vector<double>> &somGrid);
};
