#include "Noble.h"
#include "Protector.h"

#include <iostream>   
#include <string>
#include <vector>
#include <fstream>    
using namespace std;

namespace WarriorCraft {
    // Noble methods
    Noble::Noble(const string& name) : name(name), alive(true) {}

    const string& Noble::getName() const { return name; }
    bool Noble::getAlive() const { return alive; }

    // battle between nobles
    void Noble::battle(Noble& another) {
        cout << name << " battles " << another.name << endl;
        // both are dead
        if (!alive && !another.alive) {
            cout << "Oh, NO!  They're both dead!  Yuck!\n";
        }
        // the original warrior is dead
        else if (!alive) {
            another.defend();
            cout << "He's dead " << another.name << endl;
        }
        // another warrior is dead
        else if (!another.alive) {
            defend();
            cout << "He's dead " << name << endl;
        }
        // neither is dead
        else {
            // get the nobles' total strength
            double strength1 = getStrength();
            double strength2 = another.getStrength();
            defend();
            another.defend();

            // compare their strength, get and display the result, and update the strengths and status
            if (strength1 == strength2) {
                die();
                another.die();
                cout << "Mutual Annihilation: " << name << " and " << another.name << " die at each other's hands\n";
            }
            else if (strength1 > strength2) {
                setStrength(1 - strength2 / strength1);
                another.die();
                cout << name << " defeats " << another.name << endl;
            }
            else if (strength1 < strength2) {
                another.setStrength(1 - strength1 / strength2);
                die();
                cout << another.name << " defeats " << name << endl;
            }
        }
    }
    void Noble::die() { alive = false; }

    // Lord methods & output operator
    ostream& operator<<(ostream& os, const Lord& rhs) {
        os << rhs.getName() << " has an army of " << rhs.protectors.size() << endl;
        for (Protector* ptr : rhs.protectors) {
            os << *ptr;
        }
        return os;
    }

    Lord::Lord(const string& name) : Noble(name) {}

    double Lord::getStrength() const {
        double count = 0;
        for (size_t i = 0; i < protectors.size(); i++) {
            count += protectors[i]->getStrength();
        }
        return count;
    }

    // hire a protector
    bool Lord::hires(Protector& protector) {
        if (!getAlive()) {
            return false;
        }
        else if (protector.getLord() != nullptr) {
            cout << "Warriors who are already employed cannot be hired!" << endl;
            return false;
        }
        else {
            protectors.push_back(&protector);
            protector.setLord(*this);
            return true;
        }
    }

    // fire a protector
    bool Lord::fire(Protector& protector) {
        if (!getAlive()) {
            return false;
        }
        size_t i = 0;
        while (i < protectors.size()) {
            if (protectors[i]->getName() == protector.getName()) {
                protectors[i]->removeLord();
                while (i < protectors.size() - 1) {
                    protectors[i] = protectors[i + 1];
                    i++;
                }
                protectors.pop_back();
                return true;
            }
            i++;
        }
        return false;
    }
    // defend and yell in battles
    void Lord::defend() const {
        for (size_t i = 0; i < protectors.size(); i++) {
            protectors[i]->defend();
        }
    }
    void Lord::die() {
        Noble::die();
        for (size_t i = 0; i < protectors.size(); i++) {
            protectors[i]->setStrength(0);
        }
    }
    void Lord::setStrength(double proportion) {
        for (size_t i = 0; i < protectors.size(); i++) {
            protectors[i]->setStrength(protectors[i]->getStrength() * proportion);
        }
    }
    // when protector runs away
    bool Lord::removeProtector(const Protector& protector) {
        int i = 0;
        while (i < protectors.size()) {
            if (protectors[i]->getName() == protector.getName()) {
                while (i < protectors.size() - 1) {
                    protectors[i] = protectors[i + 1];
                    i++;
                }
                protectors.pop_back();
                return true;
            }
            i++;
        }
        return false;
    }

    // PersonWithStrengthToFight methods
    ostream& operator<<(ostream& os, const PersonWithStrengthToFight& rhs) {
        os << rhs.getName() << " has a strength of " << rhs.strength << endl;
        return os;
    }

    PersonWithStrengthToFight::PersonWithStrengthToFight(const string& name, double strength) : Noble(name), strength(strength) {}
    double PersonWithStrengthToFight::getStrength() const { return strength; }

    void PersonWithStrengthToFight::defend() const { cout << "Ugh!" << endl; }
    void PersonWithStrengthToFight::die() {
        Noble::die();
        strength = 0;
    }
    void PersonWithStrengthToFight::setStrength(double proportion) {
        strength *= proportion;
    }
}