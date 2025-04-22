#include "blossom.hpp"
#include "utils.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    std::string base_config_path = "./calibrated_config.props";
    int num_trials = 1;
    unsigned int initial_seed = 42;
    bool logging = true;

    auto args = parse_args(argc, argv);

    if (args.count("--config")) {
        base_config_path = args["--config"];
    }
    if (args.count("--trials")) {
        num_trials = std::stoi(args["--trials"]);
    }
    if (args.count("--seed")) {
        initial_seed = static_cast<unsigned int>(std::stoul(args["--seed"]));
    }
    if (args.count("--logging")) {
        logging = static_cast<bool>(std::stoi(args["--logging"]));
    }

    size_t num_seeds = num_trials * 3;
    std::vector<unsigned int> seeds = generate_seeds(initial_seed, num_seeds);

    for (int i = 0; i < num_trials; ++i)
    {
        std::cout << "Running simulation " << i << std::endl;

        std::string out = "./configs/config_" + std::to_string(i) + ".props";
        modify_config(base_config_path, out, i, seeds);

        BLOSSOM model(i, logging);
        model.run();
    }

    return 0;
}