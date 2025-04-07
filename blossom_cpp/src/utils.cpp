#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

std::vector<unsigned int> generate_seeds(unsigned int initial_seed, size_t num_seeds)
{
    std::mt19937 rng(initial_seed);
    std::vector<unsigned int> seeds;

    for (size_t i = 0; i < num_seeds; ++i)
    {
        seeds.push_back(rng());
    }

    return seeds;
}

std::string get_random_within_percentage(int original_value, double percentage)
{
    int deviation = static_cast<int>(original_value * percentage);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(original_value - deviation, original_value + deviation);

    return std::to_string(dis(gen));
}

void modify_config(const std::string &input_file, const std::string &output_file, const int index,
                   std::vector<unsigned int> &seeds)
{
    std::ifstream infile(input_file);
    if (!infile.is_open())
    {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    std::unordered_map<std::string, std::string> config_map;
    std::string line;

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string key, value;

        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        if (std::getline(iss, key, '=') && std::getline(iss, value))
        {
            config_map[key] = value;
        }
    }

    infile.close();

    config_map["output_file_name"] = std::stoi(config_map["initial_distribution_type"]) == 0
                                         ? "random_" + std::to_string(index)
                                         : "clustered_" + std::to_string(index);
    config_map["default_seed"] = std::to_string(seeds.back());
    seeds.pop_back();

    config_map["initial_distribution_seed"] = std::to_string(seeds.back());
    seeds.pop_back();

    config_map["nutrient_seed"] = std::to_string(seeds.back());
    seeds.pop_back();

    float deviation = std::stof(config_map["deviation"]);
    for (int i = 0; i <= 8; ++i)
    {
        std::string key = "organism_" + std::to_string(i) + "_count";
        config_map[key] = get_random_within_percentage(std::stoi(config_map[key]), deviation);
    }

    std::ofstream outfile(output_file);
    if (!outfile.is_open())
    {
        std::cerr << "Could not open the output file!" << std::endl;
        return;
    }

    for (const auto &line : config_map)
    {
        outfile << line.first << "=" << line.second << std::endl;
    }

    outfile.close();
}

std::set<int> parseIntList(const std::string &str)
{
    std::set<int> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
            result.insert(std::stoi(item));
    }
    return result;
}

void normalize(std::vector<double> &probs)
{
    double total = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (total > 0.0)
    {
        for (auto &prob : probs)
            prob /= total;
    }
    else
    {
        std::fill(probs.begin(), probs.end(), 1.0 / probs.size());
    }
}

int weightedChoice(const std::vector<double> &probs, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double random_value = dist(rng);

    double cumulative_sum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i)
    {
        cumulative_sum += probs[i];
        if (random_value <= cumulative_sum)
            return i;
    }

    return probs.size() - 1;
}