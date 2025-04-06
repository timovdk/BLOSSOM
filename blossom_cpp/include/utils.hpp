#pragma once
#include <vector>
#include <set>
#include <string>
#include <random>

std::set<int> parseIntList(const std::string &str);
void normalize(std::vector<double> &probs);
int weighted_choice(const std::vector<double> &probs, std::mt19937 &rng);
