#include "ControlTower.h"

Define_Module(ControlTower);

void ControlTower::initialize()
{
}

void ControlTower::handleMessage(cMessage *msg)
{
    // la torre di controllo riceve e cancella il messaggio
    delete msg;
}
