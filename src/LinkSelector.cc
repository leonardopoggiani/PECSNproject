#include "LinkSelector.h"

Define_Module(LinkSelector);

void LinkSelector::initialize()
{
    maxCapacityDataLinkIndex = -1; // il datalink che si occupera' dell'invio, indice del datalink a capacita' piu' alta
    operationMode = getAncestorPar("operationMode"); // 0-> monitoraggio costante dei DataLink attivo, 1-> non attivo, scelgo un DL e inviero' sempre su quello

    nDL = getAncestorPar("nDL");

    if(operationMode == 0){
        // monitoraggio ogni m secondi
        m = getAncestorPar("m");
        // avvio la schedulazione dei pacchetti di monitoraggio dei DataLink
        scheduleCheckCapacity();
    }

    cMessage* checkingMaxCapacity = new cMessage("setMaxIndexCapacity");
    scheduleAt(simTime() + 0.000001, checkingMaxCapacity);

}

void LinkSelector::handleMessage(cMessage* msg){
    if(msg->isSelfMessage()){
        if( strcmp(msg->getName(), "setMaxIndexCapacity") == 0 ){
            // sto settando per la prima volta l'indice del dl a capacita' massima
            handleSetCapacity();
        } else {
            // ricevuto un pacchetto di monitoraggio, rischedulo il prossimo e analizzo la capacita' dei dataLink
            scheduleCheckCapacity();
            handleSetCapacity();
        }

        delete msg;
    } else {
        // mi e' arrivato un messaggio da packetGenerator che va inoltrato
        handlePacketArrival(msg);
    }
}

void LinkSelector::handlePacketArrival(cMessage* msg){
    // qui mi e' arrivato il pacchetto da packetGenerator, adesso inoltro verso il DL scelto
    if(nDL > 0){
        send(msg, "out", maxCapacityDataLinkIndex);
    } else {
        delete msg;
    }
}


void LinkSelector::handleSetCapacity(){
    // ad ogni m_ controllo la capacita' dei DL e aggiorno il DL a capacita' max
    // dovrei prendere le capacita' attuali dei DL e trovo il max
    int max = getMaxIndexCapacity();
    EV_INFO << "The index of the highest capacity DL is " << max;
    maxCapacityDataLinkIndex = max;
}

int LinkSelector::getMaxIndexCapacity(){
    std::vector<int> capacities;
    DataLink* dl;
    int actualCapacity;

    for(int i = 0; i < nDL; i++){
        cModule* temp;
        // scorro tutti i dataLink
        std::string path = "dataLink[" + std::to_string(i) + "]";
        temp =  gate("out",i)->getPathEndGate()->getOwnerModule();
        dl = check_and_cast<DataLink*> (temp);
        actualCapacity = dl->getCapacity();
        EV_INFO << dl << ", la sua actualCapacity: " << actualCapacity << endl;
        capacities.push_back(actualCapacity);

        if( nDL > 0) {
            cMessage* newmsg = new cMessage("startMalusPenality"); // ho monitorato la capacita', quindi devo far iniziare la penalita'
            send(newmsg,"out",i);
        }
    }
    // indice che corrisponde al dataLink di capacita' maggiore
    return std::max_element(capacities.begin(),capacities.end()) - capacities.begin();
}

void LinkSelector::scheduleCheckCapacity(){
    cMessage* checkingMaxCapacity = new cMessage("schedule");
    scheduleAt(simTime() + m, checkingMaxCapacity);
}

