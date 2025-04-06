#pragma once

#include "neighbourhoods.hpp"
#include <map>
#include <set>
#include <string>

struct OrganismData
{
    std::map<std::string, double> params;
    std::set<int> preys;
    std::set<int> predators;
};

class OrganismGroup
{
  private:
    int id, type, age;
    double biomass;
    dpt location;

  public:
    OrganismGroup(int id, int type, dpt location, int age, double biomass);

    // Getters and Setters
    int getId() const;
    int getType() const;
    dpt getLocation() const;
    void setLocation(const dpt new_location);
    int getAge() const;
    double getBiomass() const;

    // State modification
    void divideBiomass();
    void increaseBiomass(const double amount);
    void decreaseBiomass(const double amount);
    void incrementAge();

    // Reproduction
    OrganismGroup reproduce(const int new_id, const dpt new_loc) const;
};
