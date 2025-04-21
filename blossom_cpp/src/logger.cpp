#include "logger.hpp"
#include <iostream>

Logger::Logger(std::string outputDir, std::string outputFileName, int gridWidth, int gridHeight)
    : agentDir(outputDir + "agent/"), somDir(outputDir + "som/"), outputFileName(std::move(outputFileName)),
      gridWidth(gridWidth), gridHeight(gridHeight)
{
}
Logger::~Logger()
{
    if (outFileAgents.is_open())
        outFileAgents.close();
    if (outFileSOM.is_open())
        outFileSOM.close();
}
void Logger::log(int currentStep, const std::unordered_map<int, OrganismGroup> &agents,
                 const std::vector<std::vector<double>> &somGrid)
{
    logAgents(currentStep, agents);
    logSOM(currentStep, somGrid);
}
void Logger::reset()
{
    linesLogged = 0;
    ticksLogged = 0;
    fileIndex = 0;
    fileIndexSOM = 0;

    if (outFileAgents.is_open())
        outFileAgents.close();

    if (outFileSOM.is_open())
        outFileSOM.close();
}

// Private methods
void Logger::logAgents(int currentStep, const std::unordered_map<int, OrganismGroup> &agents)
{
    if (!outFileAgents.is_open() || linesLogged >= maxLinesPerFile)
    {
        if (outFileAgents.is_open())
            outFileAgents.close();

        std::string fileName = agentDir + outputFileName + "_" + std::to_string(fileIndex++) + ".bin";
        outFileAgents.open(fileName, std::ios::binary);
        if (!outFileAgents)
        {
            std::cerr << "Failed to open binary log file.\n";
            return;
        }

        linesLogged = 0;
    }

    for (const auto &p : agents)
    {
        const auto &agent = p.second;

        AgentLogEntry entry{static_cast<uint32_t>(agent.getId()),         static_cast<float>(agent.getBiomass()),
                            static_cast<uint16_t>(currentStep),           static_cast<uint16_t>(agent.getLocation().x),
                            static_cast<uint16_t>(agent.getLocation().y), static_cast<uint8_t>(agent.getType()),
                            static_cast<uint8_t>(agent.getAge())};

        outFileAgents.write(reinterpret_cast<char *>(&entry), sizeof(AgentLogEntry));
        ++linesLogged;
    }
}

void Logger::logSOM(int currentStep, const std::vector<std::vector<double>> &somGrid)
{
    if (!outFileSOM.is_open() || ticksLogged >= maxTicksPerFile)
    {
        if (outFileSOM.is_open())
            outFileSOM.close();

        std::string fileName = somDir + outputFileName + "_" + std::to_string(fileIndexSOM++) + ".bin";
        outFileSOM.open(fileName, std::ios::binary);
        if (!outFileSOM)
        {
            std::cerr << "Failed to open binary log file.\n";
            return;
        }

        ticksLogged = 0;
    }

    for (int x = 0; x < gridWidth; ++x)
    {
        for (int y = 0; y < gridHeight; ++y)
        {
            SOMLogEntry entry{static_cast<float>(somGrid[x][y]), static_cast<uint16_t>(currentStep),
                              static_cast<uint16_t>(x), static_cast<uint16_t>(y)};
            outFileSOM.write(reinterpret_cast<char *>(&entry), sizeof(SOMLogEntry));
        }
    }
    ++ticksLogged;
}
