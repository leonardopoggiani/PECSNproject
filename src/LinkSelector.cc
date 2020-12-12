#include "LinkSelector.h"

Define_Module(LinkSelector);

void LinkSelector::initialize()
{
    maxCapacityDataLinkIndex = 1;
    operationMode = par("operationMode");
    nDL = par("nDL");
    if(operationMode == 1){
        // non monitoro, ricerco il DL con capacità più alta e tengo quello
        EV << "Monitoraggio non attivo\n";
        // scelgo il DL una volta e mai più
        handleSetCapacity();
    } else {
        // monitoraggio ogni m secondi
        monitoringTime = getAncestorPar("m");
        scheduleCheckCapacity();
    }
}

void LinkSelector::handleMessage(cMessage* msg){
    if(msg->isSelfMessage()){
        EV << "Monitoraggio del data link\n";

        scheduleCheckCapacity();
        handleSetCapacity();
        delete msg;
    } else {
        // mi è arrivato un messaggio da packetGenerator
        EV << "Arrivato un pacchetto da packetGenerator\n";
        handlePacketArrival(msg);
    }
}

void LinkSelector::handlePacketArrival(cMessage* msg){
    // qui mi è arrivato il pacchetto da packetGenerator, adesso inoltro verso il DL scelto
    send(msg, "out", maxCapacityDataLinkIndex);
}


void LinkSelector::handleSetCapacity(){
    // ad ogni monitoringTime controllo la capacità dei DL e aggiorno il DL a capacità max
    // dovrei prendere le capacità attuali dei DL e trovo il max
    int max = getMaxIndexCapacity();
    EV << "The index of the highest capacity DL is " << max;
    maxCapacityDataLinkIndex = max;

}

int LinkSelector::getMaxIndexCapacity(){
    std::vector<int> capacities;

    for(int i = 0; i < nDL; i++){
        cModule* temp;
        temp = getModuleByPath("dataLink[i]");
        DataLink* dl;
        dl = check_and_cast<DataLink*> (temp);
        int actualCapacity = dl->getCapacity();
        EV << actualCapacity << "\n";
        capacities.push_back(actualCapacity);
    }

    return std::max_element(capacities.begin(),capacities.end()) - capacities.begin();
}

void LinkSelector::scheduleCheckCapacity(){
    cMessage* checkingMaxCapacity = new cMessage("schedule");

    scheduleAt(simTime() + monitoringTime, checkingMaxCapacity);
    EV << "Schedulato monitoraggio ogni " << monitoringTime << " secondi\n";
}

