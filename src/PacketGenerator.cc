#include "PacketGenerator.h"

Define_Module(PacketGenerator);

void PacketGenerator::initialize()
{
    // solo roba per provare che effettivamente viene cambiato il dataLink con la massima capacità
    cMessage* msg = new cMessage("Messaggio");
    scheduleAt(simTime()+10,msg);
}

void PacketGenerator::handleMessage(cMessage *msg)
{
    send(msg,"out");
}

