#include "ControlTower.h"

Define_Module(ControlTower);

void ControlTower::initialize()
{
}

void ControlTower::handleMessage(cMessage *msg)
{
    // La torre di controllo semplicemente butta il messaggio
    delete msg;
}
