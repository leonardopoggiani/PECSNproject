#include "PacketGenerator.h"
#include <string>
#include "AircraftPacket_m.h"

using namespace std;

Define_Module(PacketGenerator);

void PacketGenerator::initialize()
{
    // ** SIGNAL ** //
    registerSignal("computeArrivalTime");

    cMessage* msg = new cMessage("Messaggio");
    k = getAncestorPar("k").doubleValue();
    scheduleAt(simTime() + exponential(k, 0), msg );

}

void PacketGenerator::handleMessage(cMessage* msg)
{
    if (msg->isSelfMessage()){
        createSendPacket(msg);
    }
}

void PacketGenerator::createSendPacket(cMessage* msg){
    AircraftPacket* ap = new AircraftPacket("AircraftPacket");
    ap->setAircraftID(getIndex());
    ap->setSendTime(simTime().dbl()); // nome

    //Send the packet to LinkSelector
    send(ap, "out");

    //Riattivo il timer
    scheduleAt(simTime() + exponential(k, 0), msg );

}
