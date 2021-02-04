#include "ControlTower.h"

Define_Module(ControlTower);

void ControlTower::initialize()
{
}

void ControlTower::handleMessage(cMessage *msg)
{
    // control tower just drop the message
    EV << "received and dropped \n";
    delete msg;
}
