#include "organism.hpp"

OrganismGroup::OrganismGroup(int id, int type, dpt location, int age, double biomass)
    : id(id), type(type), age(age), biomass(biomass), location(location) {}

// Getters
int OrganismGroup::getId() const { return id; }
int OrganismGroup::getType() const { return type; }
dpt OrganismGroup::getLocation() const { return location; }
double OrganismGroup::getBiomass() const { return biomass; }
int OrganismGroup::getAge() const { return age; }

void OrganismGroup::incrementAge() { ++age; }
void OrganismGroup::divideBiomass() { biomass /= 2.0; }
void OrganismGroup::increaseBiomass(double amount)
{
    biomass += amount;
}
// TODO: Whenever move is called, check if the new location is reflected in the agents and ids list
void OrganismGroup::move(const dpt &new_location) { location = new_location; }
OrganismGroup OrganismGroup::save(int new_id) const
{
    return OrganismGroup(new_id, type, location, 0, biomass);
}