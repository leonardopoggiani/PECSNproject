#include "PacketGenerator.h"
#include <string>
#include "AircraftPacket_m.h"

using namespace std;

Define_Module(PacketGenerator);

void PacketGenerator::initialize()
{
    // ** SIGNAL ** //
    computeArrivalTime_ = registerSignal("computeArrivalTime");

    cMessage* msg = new cMessage("msg");
    k = getAncestorPar("k").doubleValue();
    simtime_t arrivalTime = exponential(k,0);

    scheduleAt(simTime() + arrivalTime, msg );
    emit(computeArrivalTime_, arrivalTime);

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
    ap->setSendTime(simTime().dbl());

    send(ap, "out");

    //Riattivo il timer
    simtime_t arrivalTime = exponential(k,0);

    scheduleAt(simTime() + arrivalTime, msg );
    emit(computeArrivalTime_, arrivalTime);


}
