#include "utils.hpp"
#include <sstream>

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