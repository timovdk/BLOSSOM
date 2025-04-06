#include "neighbourhoods.hpp"

dpt::dpt(int x, int y) : x(x), y(y) {}

int wrapAround(int value, int gridSize)
{
    return (value % gridSize + gridSize) % gridSize;
}

std::vector<dpt> von_neumann_r1(int x, int y, int gridWidth, int gridHeight)
{
    return {
        {wrapAround(x, gridWidth), wrapAround(y - 1, gridHeight)},
        {wrapAround(x, gridWidth), wrapAround(y + 1, gridHeight)},
        {wrapAround(x - 1, gridWidth), wrapAround(y, gridHeight)},
        {wrapAround(x + 1, gridWidth), wrapAround(y, gridHeight)}};
}

std::vector<dpt> von_neumann_neighborhood_2d(int x, int y, int r, int gridWidth, int gridHeight)
{
    std::vector<dpt> neighbors;
    for (int dx = -r; dx <= r; ++dx)
    {
        for (int dy = -r; dy <= r; ++dy)
        {
            if (dx == 0 && dy == 0)
                continue;
            if (std::abs(dx) == r || std::abs(dy) == r)
                neighbors.emplace_back(wrapAround(x + dx, gridWidth), wrapAround(y + dy, gridHeight));
        }
    }
    return neighbors;
}