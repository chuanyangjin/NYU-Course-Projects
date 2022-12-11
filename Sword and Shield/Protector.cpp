#include "Noble.h"
#include "Protector.h"

#include <iostream>   
#include <string>
#include <vector>
#include <fstream>    
using namespace std;

namespace WarriorCraft {
    // Protector methods & output operator
    ostream& operator<<(ostream& os, const Protector& rhs) {
        os << "          " << rhs.name << ": " << rhs.strength << endl;
        return os;
    }

    Protector::Protector(const string& name, double strength) : name(name), strength(strength), lord(nullptr) {}

    string Protector::getName() const { return name; }
    double Protector::getStrength() const { return strength; }
    Lord* Protector::getLord() const { return lord; }

    void Protector::setStrength(double theStrength) { strength = theStrength; }
    void Protector::setLord(Lord& theLord) { lord = &theLord; }
    void Protector::removeLord() { lord = nullptr; }

    // when protector runs away
    bool Protector::runaway() {
        if (lord == nullptr) {
            cout << "I am not hired!" << endl;
            return false;
        }
        else {
            cout << name << " flees in terror, abandoning his lord, " << lord->getName() << endl;
            lord->removeProtector(*this);
            lord = nullptr;
            return true;
        }
    }

    // Wizard methods
    Wizard::Wizard(const string& name, double strength) : Protector(name, strength) {}
    void Wizard::defend() const { cout << "POOF!" << endl; }

    // Warrior methods
    Warrior::Warrior(const string& name, double strength) : Protector(name, strength) {}
    void Warrior::defend() const { cout << getName() << " says: Take that in the name of my lord, " << getLord()->getName() << endl; }

    // Archer methods
    Archer::Archer(const string& name, double strength) : Warrior(name, strength) {}
    void Archer::defend() const {
        cout << "TWANG!  ";
        Warrior::defend();
    }

    // Swordsman methods
    Swordsman::Swordsman(const string& name, double strength) : Warrior(name, strength) {}
    void Swordsman::defend() const {
        cout << "CLANG!  ";
        Warrior::defend();
    }
}
