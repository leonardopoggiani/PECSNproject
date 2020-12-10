#include "LinkSelector.h"

Define_Module(LinkSelector);

void LinkSelector::initialize()
{
    operationMode = par("operationMode");
    if(operationMode == 1){
        // non monitoro, ricerco il DL con capacità più alta e tengo quello
        EV << "Monitoraggio non attivo\n";
    } else {
        // monitoraggio ogni m secondi
        cModule* aircraft = getModuleByPath("Aircraft"); //God mode, inserisci una getter
        double m = aircraft->par("m");
        cMessage* checkingMaxCapacity = new cMessage("schedule");

        scheduleAt(simTime() + m, checkingMaxCapacity);
        EV << "Schedulato monitoraggio ogni" << m << " secondi\n";
    }
}

void LinkSelector::handleMessage(cMessage* msg){
    if( msg->isName("schedule")){
            handleSetCapacity();
    } else if(strcmp(msg->getName(),"AircraftPacket")){
            handlePacketArrival(msg);
    }
}

void LinkSelector::handlePacketArrival(cMessage* msg){

}


void LinkSelector::handleSetCapacity(){

}

int LinkSelector::getCapacity(){
    return 1;
}

void LinkSelector::scheduleCheckCapacity(){

}

