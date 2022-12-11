#ifndef PROTECTOR_H
#define PROTECTOR_H

#include <iostream>   
#include <string>
#include <vector>
#include <fstream>    

namespace WarriorCraft {
    class Noble;
    class Lord;

    // Protector class
    class Protector {
        friend std::ostream& operator<<(std::ostream& os, const Protector& rhs);

    public:
        Protector(const std::string& name, double strength);

        std::string getName() const;
        double getStrength() const;
        Lord* getLord() const;

        void setStrength(double theStrength);
        void setLord(Lord& theLord);

        void removeLord();
        bool runaway();

        virtual void defend() const = 0;

    private:
        std::string name;
        double strength;
        Lord* lord;
    };

    // Wizard class
    class Wizard : public Protector {
    public:
        Wizard(const std::string& name, double strength);
        void defend() const;
    };

    // Warrior class
    class Warrior : public Protector {
    public:
        Warrior(const std::string& name, double strength);
        void defend() const;
    };

    // Archer class
    class Archer :public Warrior {
    public:
        Archer(const std::string& name, double strength);
        void defend() const;
    };

    // Swordsman class
    class Swordsman : public Warrior {
    public:
        Swordsman(const std::string& name, double strength);
        void defend() const;
    };
}

#endif