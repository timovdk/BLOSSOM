#pragma once

#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

std::vector<unsigned int> generate_seeds(unsigned int initial_seed, size_t num_seeds);
std::string get_random_within_percentage(int original_value, double percentage, std::mt19937 &rng);
std::string modify_config(const std::string &input_file, const int index, std::vector<unsigned int> &seeds,
                          const unsigned int initial_seed);
std::set<int> parseIntList(const std::string &str);
void normalize(std::vector<double> &probs);
int weightedChoice(const std::vector<double> &probs, std::mt19937 &rng);
std::unordered_map<std::string, std::string> parse_args(int argc, char *argv[]);