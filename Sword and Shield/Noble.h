#ifndef NOBLE_H
#define NOBLE_H

#include <iostream>   
#include <string>
#include <vector>
#include <fstream>

namespace WarriorCraft {
    class Protector;

    // Noble class
    class Noble {
    public:
        Noble(const std::string& name);

        const std::string& getName() const;
        virtual double getStrength() const = 0;
        bool getAlive() const;

        virtual void battle(Noble& another);
        virtual void defend() const = 0;
        virtual void die();
        virtual void setStrength(double proportion) = 0;

    private:
        std::string name;
        bool alive;
    };

    // Lord class
    class Lord : public Noble {
        friend std::ostream& operator<<(std::ostream& os, const Lord& rhs);

    public:
        Lord(const std::string& name);
        double getStrength() const;

        bool hires(Protector& protector);
        bool fire(Protector& protector);
        void defend() const;
        void die();
        void setStrength(double proportion);
        bool removeProtector(const Protector& protector);

    private:
        std::vector<Protector*> protectors;
    };

    // PersonWithStrengthToFight class
    class PersonWithStrengthToFight : public Noble {
        friend std::ostream& operator<<(std::ostream& os, const PersonWithStrengthToFight& rhs);

    public:
        PersonWithStrengthToFight(const std::string& name, double strength);

        double getStrength() const;

        void defend() const;
        void die();
        void setStrength(double proportion);

    private:
        double strength;
    };
}

#endif