#include "LinkSelector.h"

Define_Module(LinkSelector);

void LinkSelector::initialize()
{
    maxCapacityDataLinkIndex = -1; // DataLink su cui invierò perchè ha la capacità più alta
    operationMode = par("operationMode"); // 0-> monitoraggio costante dei DataLink attivo, 1-> non attivo, scelgo un DL e quello rimane
    nDL = par("nDL");
    handleSetCapacity();

    if(operationMode == 0){
        // monitoraggio ogni m secondi
        m = getAncestorPar("m");
        // avvio la schedulazione dei pacchetti di monitoraggio dei DataLink
        scheduleCheckCapacity();
    }
}

void LinkSelector::handleMessage(cMessage* msg){
    if(msg->isSelfMessage()){
        // ricevuto un pacchetto di monitoraggio, rischedulo il prossimo e analizzo la capacità dei dataLink
        scheduleCheckCapacity();
        handleSetCapacity();
        delete msg;
    } else {
        // mi è arrivato un messaggio da packetGenerator che va inoltrato
        handlePacketArrival(msg);
    }
}

void LinkSelector::handlePacketArrival(cMessage* msg){
    // qui mi è arrivato il pacchetto da packetGenerator, adesso inoltro verso il DL scelto
    send(msg, "out", maxCapacityDataLinkIndex);
}


void LinkSelector::handleSetCapacity(){
    // ad ogni m_ controllo la capacità dei DL e aggiorno il DL a capacità max
    // dovrei prendere le capacità attuali dei DL e trovo il max
    int max = getMaxIndexCapacity();
    // EV_INFO << "The index of the highest capacity DL is " << max;
    maxCapacityDataLinkIndex = max;

}

int LinkSelector::getMaxIndexCapacity(){
    std::vector<int> capacities;

    for(int i = 0; i < nDL; i++){
        cModule* temp;
        // scorro tutti i dataLink
        std::string path = "dataLink[" + std::to_string(i) + "]";
        temp =  gate("out",i)->getPathEndGate()->getOwnerModule();
        DataLink* dl;
        dl = check_and_cast<DataLink*> (temp);
        int actualCapacity = dl->getCapacity();
        // EV_INFO << dl << ", la sua actualCapacity: " << actualCapacity << endl;
        capacities.push_back(actualCapacity);

        cMessage* newmsg = new cMessage("startMalusPenality");
        send(newmsg,"out",i);

    }
    // indice che corrisponde al dataLink di capacità maggiore
    return std::max_element(capacities.begin(),capacities.end()) - capacities.begin();
}

void LinkSelector::scheduleCheckCapacity(){
    cMessage* checkingMaxCapacity = new cMessage("schedule");
    scheduleAt(simTime() + m, checkingMaxCapacity);
}

