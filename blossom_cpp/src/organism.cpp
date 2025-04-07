#include "organism.hpp"

OrganismGroup::OrganismGroup(int id, int type, dpt location, int age, double biomass)
    : id(id), type(type), age(age), biomass(biomass), location(location)
{
}

// Getters and Setters
int OrganismGroup::getId() const
{
    return id;
}
int OrganismGroup::getType() const
{
    return type;
}
dpt OrganismGroup::getLocation() const
{
    return location;
}
void OrganismGroup::setLocation(const dpt new_location)
{
    location = new_location;
}
int OrganismGroup::getAge() const
{
    return age;
}
double OrganismGroup::getBiomass() const
{
    return biomass;
}

// State modification
void OrganismGroup::increaseBiomass(const double amount)
{
    biomass += amount;
}
void OrganismGroup::decreaseBiomass(const double amount)
{
    biomass -= amount;
    if (biomass < 0.0)
    {
        biomass = 0.0;
    }
}
void OrganismGroup::incrementAge()
{
    ++age;
}

// Reproduction
OrganismGroup OrganismGroup::reproduce(const int new_id, const dpt new_loc)
{
    double old_biomass = biomass;
    biomass *= 0.5;
    return OrganismGroup(new_id, type, new_loc, 0, (old_biomass - biomass));
}