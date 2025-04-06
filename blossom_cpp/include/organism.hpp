#pragma once
#include "neighbourhoods.hpp"
#include <map>
#include <set>
#include <string>

struct OrganismData {
    std::map<std::string, double> params;
    std::set<int> preys;
    std::set<int> predators;
};

class OrganismGroup {
private:
    int id, type, age;
    double biomass;
    dpt location;

public:
    OrganismGroup(int id, int type, dpt location, int age, double biomass);
    int getId() const;
    int getType() const;
    dpt getLocation() const;
    double getBiomass() const;
    int getAge() const;
    void incrementAge();
    void divideBiomass();
    void move(const dpt& new_location);
    OrganismGroup save(int new_id) const;
    void increaseBiomass(double amount);
};
