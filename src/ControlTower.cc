#include "ControlTower.h"

Define_Module(ControlTower);

void ControlTower::initialize()
{
}

void ControlTower::handleMessage(cMessage *msg)
{
    // the control tower just drop the message
    delete msg;
}
