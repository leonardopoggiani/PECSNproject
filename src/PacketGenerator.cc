#include "PacketGenerator.h"
#include <string>
#include "AircraftPacket_m.h"

using namespace std;

Define_Module(PacketGenerator);

void PacketGenerator::initialize()
{
    // solo roba per provare che effettivamente viene cambiato il dataLink con la massima capacita
   cMessage* msg = new cMessage("Messaggio");
    k_ = getAncestorPar("k").doubleValue();
    distribution = getAncestorPar("distribution").stdstringValue();
    scheduleAt(simTime()+100,msg);
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

    //Send the packet to LinkSelector
    send(ap, "out");
    EV << "Sono PackGenerator, ho inviato il pacchetto: " << ap->getAircraftID();

    //Riattivo il timer
    scheduleCreateSendPacket(msg);

}

void PacketGenerator::scheduleCreateSendPacket(cMessage* msg){


   if ( strcmp(distribution.c_str(), "lognormal") == 0)
        scheduleAt(simTime() + lognormal(k_,0,0), msg);
   else if (strcmp(distribution.c_str(), "exponential") == 0 )
        scheduleAt(simTime() + exponential(k_, 0), msg );
}

