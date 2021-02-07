#include "ControlTower.h"

Define_Module(ControlTower);

void ControlTower::initialize()
{
    computeReceivedPackets_ = registerSignal("computeReceivedPackets");
}

void ControlTower::handleMessage(cMessage *msg)
{
    // la torre di controllo riceve e cancella il messaggio
    emit(computeReceivedPackets_,1);

    delete msg;
}
