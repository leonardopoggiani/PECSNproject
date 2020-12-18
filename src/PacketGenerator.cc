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
    distribution = getAncestorPar("distribution").stdstringValue();
    scheduleAt(simTime() + k,msg);

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
    scheduleCreateSendPacket(msg);

}

void PacketGenerator::scheduleCreateSendPacket(cMessage* msg){
   if ( strcmp(distribution.c_str(), "lognormal") == 0)
        scheduleAt(simTime() + lognormal(k,0,0), msg);
   else if (strcmp(distribution.c_str(), "exponential") == 0 )
        scheduleAt(simTime() + exponential(k, 0), msg );
}

