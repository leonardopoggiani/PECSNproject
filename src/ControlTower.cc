#include "ControlTower.h"

Define_Module(ControlTower);

void ControlTower::initialize()
{
    cMessage* msg = new cMessage("End");
}

void ControlTower::handleMessage(cMessage *msg)
{
    // the control tower just drop the message
    delete msg;
}
