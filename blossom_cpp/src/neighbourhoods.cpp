#include "neighbourhoods.hpp"

dpt::dpt(int x, int y) : x(x), y(y)
{
}

int wrapAround(const int value, const int gridSize)
{
    return (value % gridSize + gridSize) % gridSize;
}

const std::vector<dpt> vonNeumannR1(const int x, const int y, const int gridWidth, const int gridHeight)
{
    return {{wrapAround(x, gridWidth), wrapAround(y, gridHeight)},
            {wrapAround(x, gridWidth), wrapAround(y - 1, gridHeight)},
            {wrapAround(x, gridWidth), wrapAround(y + 1, gridHeight)},
            {wrapAround(x - 1, gridWidth), wrapAround(y, gridHeight)},
            {wrapAround(x + 1, gridWidth), wrapAround(y, gridHeight)}};
}

const std::vector<dpt> vonNeumannRn(const int x, const int y, const int r, const int gridWidth, const int gridHeight)
{
    std::vector<dpt> neighbors;

    for (int dx = -r; dx <= r; ++dx)
    {
        for (int dy = -r; dy <= r; ++dy)
        {
            if (std::abs(dx) + std::abs(dy) <= r)
            {
                neighbors.push_back({wrapAround(x + dx, gridWidth), wrapAround(y + dy, gridHeight)});
            }
        }
    }

    return neighbors;
}