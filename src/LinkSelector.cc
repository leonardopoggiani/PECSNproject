#include "LinkSelector.h"
#include <vector>
#include <algorithm>

Define_Module(LinkSelector);

void LinkSelector::initialize()
{
    maxCapacityDataLinkIndex = 1;
    operationMode = par("operationMode");
    nDL = par("nDL");
    if(operationMode == 1){
        // non monitoro, ricerco il DL con capacit� pi� alta e tengo quello
        EV << "Monitoraggio non attivo\n";
        // scelgo il DL una volta e mai pi�
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
        // mi � arrivato un messaggio da packetGenerator
        EV << "Arrivato un pacchetto da packetGenerator\n";
        handlePacketArrival(msg);
    }
}

void LinkSelector::handlePacketArrival(cMessage* msg){
    // qui mi � arrivato il pacchetto da packetGenerator, adesso inoltro verso il DL scelto
    send(msg, "out", maxCapacityDataLinkIndex);
}


void LinkSelector::handleSetCapacity(){
    // ad ogni monitoringTime controllo la capacit� dei DL e aggiorno il DL a capacit� max
    // dovrei prendere le capacit� attuali dei DL e trovo il max
    std::vector<int> capacities;

    for(int i = 0; i < nDL; i++){
        cModule* temp;
        temp = getModuleByPath("dataLink[i]");
        int actualCapacity = temp->par("actualCapacity");
        capacities.push_back(actualCapacity);
    }

    int max = std::max_element(capacities.begin(),capacities.end()) - capacities.begin();
    EV << "The index of the highest capacity DL is " << max;
    maxCapacityDataLinkIndex = max;

}

int LinkSelector::getCapacity(){
    return 1;
}

void LinkSelector::scheduleCheckCapacity(){
    cMessage* checkingMaxCapacity = new cMessage("schedule");

    scheduleAt(simTime() + monitoringTime, checkingMaxCapacity);
    EV << "Schedulato monitoraggio ogni " << monitoringTime << " secondi\n";
}

