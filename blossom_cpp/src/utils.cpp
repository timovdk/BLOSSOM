#include "utils.hpp"
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cmath>

std::set<int> parseIntList(const std::string &str) {
    std::set<int> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty())
            result.insert(std::stoi(item));
    }
    return result;
}

void normalize(std::vector<double>& probs) {
    double total = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (total > 0.0) {
        for (auto& prob : probs) prob /= total;
    } else {
        std::fill(probs.begin(), probs.end(), 1.0 / probs.size());
    }
}

int weighted_choice(const std::vector<double>& probs, std::mt19937& rng) {
    std::vector<double> cumulative_probs(probs.size());
    cumulative_probs[0] = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
        cumulative_probs[i] = cumulative_probs[i - 1] + probs[i];
    }
    std::uniform_real_distribution<double> dist(0.0, cumulative_probs.back());
    double random_value = dist(rng);
    for (size_t i = 0; i < cumulative_probs.size(); ++i) {
        if (random_value <= cumulative_probs[i]) return i;
    }
    return cumulative_probs.size() - 1;
}
