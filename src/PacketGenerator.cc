#include "PacketGenerator.h"
#include <string>

using namespace std;

Define_Module(PacketGenerator);

void PacketGenerator::initialize()
{
    // solo roba per provare che effettivamente viene cambiato il dataLink con la massima capacita
    cMessage* msg = new cMessage("Messaggio");
    scheduleAt(simTime()+100,msg);
}

void PacketGenerator::handleMessage(cMessage* msg)
{
    if (msg->isSelfMessage()){
        createSendPacket();
    }
}

void PacketGenerator::createSendPacket(){
    cMessage* msg = new cMessage("PacketSend");

    //Send the packet to LinkSelector
    send(msg, "out");

    //Riattivo il timer
    scheduleArrival(new cMessage("PacketArrival"));

}

void PacketGenerator::scheduleArrival(cMessage* msg){

    string distribution = getAncestorPar("distribution").stdstringValue();
    double k = getAncestorPar("k").doubleValue();

       if ( strcmp(distribution.c_str(), "lognormal") == 0)
            scheduleAt(simTime() + lognormal(k,0,0), msg);
       else if (strcmp(distribution.c_str(), "exponential") == 0 )
            scheduleAt(simTime() + exponential(k, 0), msg );
}

