#include "blossom.hpp"
#include "utils.hpp"

int main()
{
    const int num_trials = 5;
    unsigned int initial_seed = 42;
    size_t num_seeds = num_trials * 3;

    std::vector<unsigned int> seeds = generate_seeds(initial_seed, num_seeds);

    for (int i = 0; i < num_trials; ++i)
    {
        std::cout << "Running simulation " << i << std::endl;

        std::string out = "./configs/config_" + std::to_string(i) + ".props";
        modify_config("./base_config.props", out, i, seeds);

        BLOSSOM model(i);
        model.run();
    }

    return 0;
}