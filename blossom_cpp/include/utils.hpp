#pragma once

#include <random>
#include <set>
#include <string>
#include <vector>

std::set<int> parseIntList(const std::string &str);
void normalize(std::vector<double> &probs);
int weightedChoice(const std::vector<double> &probs, std::mt19937 &rng);
