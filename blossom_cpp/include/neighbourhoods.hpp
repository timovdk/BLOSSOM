#pragma once
#include <vector>

struct dpt
{
    int x, y;
    dpt(int x = 0, int y = 0);
};

int wrapAround(int value, int gridSize);
std::vector<dpt> von_neumann_r1(int x, int y, int gridWidth, int gridHeight);
std::vector<dpt> von_neumann_neighborhood_2d(int x, int y, int r, int gridWidth, int gridHeight);