#include "LinkSelector.h"

Define_Module(LinkSelector);

void LinkSelector::initialize()
{

}

void LinkSelector::handleMessage(cMessage *msg){
}

/*void LinkSelector::handlePacketArrival(AircraftPacket* ap){

}*/

void LinkSelector::handlePacketArrival(cMessage* msg){

}

void LinkSelector::handleSetCapacity(cMessage* msg){

}

int LinkSelector::getCapacity(){
    return 1;
}

void LinkSelector::scheduleCheckCapacity(){

}

