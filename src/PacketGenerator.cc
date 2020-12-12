#include "PacketGenerator.h"

Define_Module(PacketGenerator);

void PacketGenerator::initialize()
{
    // solo roba per provare che effettivamente viene cambiato il dataLink con la massima capacit�
    cMessage* msg = new cMessage("Messaggio");
    scheduleAt(simTime()+100,msg);
}

void PacketGenerator::handleMessage(cMessage *msg)
{
    send(msg,"out");
    scheduleAt(simTime()+100,msg);
}

