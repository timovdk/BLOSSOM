#pragma once

#include <vector>

struct dpt
{
    int x, y;
    dpt(int x = 0, int y = 0);
};

int wrapAround(const int value, const int gridSize);
const std::vector<dpt> vonNeumannR1(const int x, const int y, const int gridWidth, const int gridHeight);
const std::vector<dpt> vonNeumannRn(const int x, const int y, const int r, const int gridWidth, const int gridHeight);