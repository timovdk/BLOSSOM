#include "blossom.hpp"
#include "utils.hpp"
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    std::string base_config_path = "./optimized_config.props";
    int num_trials = 1;
    unsigned int initial_seed = 42;
    bool logging = true;
    bool extended_logging = false;

    auto args = parse_args(argc, argv);

    if (args.count("--config"))
    {
        base_config_path = args["--config"];
    }
    if (args.count("--trials"))
    {
        num_trials = std::stoi(args["--trials"]);
    }
    if (args.count("--seed"))
    {
        initial_seed = static_cast<unsigned int>(std::stoul(args["--seed"]));
    }
    if (args.count("--logging"))
    {
        logging = static_cast<bool>(std::stoi(args["--logging"]));
    }
    if (args.count("--extended_logging"))
    {
        extended_logging = static_cast<bool>(std::stoi(args["--extended_logging"]));
    }

    size_t num_seeds = num_trials * 3;
    std::vector<unsigned int> seeds = generate_seeds(initial_seed, num_seeds);

    for (int i = 0; i < num_trials; ++i)
    {
        std::cout << "Running simulation " << i << std::endl;

        auto config_file_name = modify_config(base_config_path, i, seeds, initial_seed);

        BLOSSOM model(i, config_file_name, logging, extended_logging);
        model.run();
    }

    return 0;
}