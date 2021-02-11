#include "LinkSelector.h"

Define_Module(LinkSelector);

void LinkSelector::initialize()
{
    maxCapacityDataLinkIndex = -1; // il datalink che si occupera' dell'invio, indice del datalink a capacita' piu' alta
    operationMode = getAncestorPar("operationMode"); // 0-> monitoraggio costante dei DataLink attivo, 1-> non attivo, scelgo un DL e inviero' sempre su quello
    malusX = getAncestorPar("X").doubleValue();
    penalty = false;
    nDL = getAncestorPar("nDL");

    computeQueueLength_ = registerSignal("computeQueueLength");
    computeWaitingTime_ = registerSignal("computeWaitingTime");
    computeMeanMalus_ = registerSignal("computeMeanMalus");
    computeServiceTime_ = registerSignal("computeServiceTime");

    size = getAncestorPar("s").doubleValue(); // la dimensione di un pacchetto


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
        if ( strcmp(msg->getName(), "malus") == 0 ) {
            // malus terminato
            penalty = false;
            sendPacket();
            delete msg;
        } else if ( strcmp(msg->getName(), "setMaxIndexCapacity") == 0 ){
            // non sto monitorando, setto una volta e basta
            getMaxIndexCapacity();
            delete msg;
        }else if ( strcmp(msg->getName(), "packetToSend") == 0 ){
            // serviceTime passato, ora posso inviare davvero
            sendPacketToDataLink(msg);
        } else {
            // ricevuto un pacchetto di monitoraggio, rischedulo il prossimo e analizzo la capacita' dei dataLink
            scheduleCheckCapacity();
            getMaxIndexCapacity();
            delete msg;
        }
    } else {
        // mi e' arrivato un messaggio da packetGenerator che va inoltrato
        handlePacketArrival(msg);
    }
}

void LinkSelector::sendPacketToDataLink(cMessage* msg){
    send(msg,"out",maxCapacityDataLinkIndex);
    sendPacket();
}

void LinkSelector::handlePacketArrival(cMessage* msg) {
    // qui mi e' arrivato il pacchetto da packetGenerator, adesso inoltro verso il DL scelto
    AircraftPacket* pa = check_and_cast<AircraftPacket*>(msg);
    pa->setArrivalTime(simTime().dbl());
    pa->setName("packetToSend");
    queue.insert(pa);
    EV << "queueLength " << queue.getLength() << endl;
    emit(computeQueueLength_, queue.getLength());
    // provo subito ad inviarlo
    sendPacket();
}

void LinkSelector::sendPacket() {
    if ( !queue.isEmpty() && !penalty) {
        // la coda non e' vuota e non sto scontando una  penalita'
        AircraftPacket* ap = (AircraftPacket*) queue.front();
        queue.pop();
        EV << "WaitingTime: " << simTime() - ap->getArrivalTime()<< endl; // tempo attuale - tempo in cui il pacchetto e' entrato in coda

        double s = (double) size;
        std::string path = "dataLink[" + std::to_string(maxCapacityDataLinkIndex) + "]";
        cModule* temp =  gate("out",maxCapacityDataLinkIndex)->getPathEndGate()->getOwnerModule();
        DataLink* dl = check_and_cast<DataLink*> (temp);
        double ac = dl->getCapacity();

        double serviceTime = s/ac;

        emit(computeServiceTime_, serviceTime);
        EV <<"Service time is: " << serviceTime << ",size: " << size << ", actualCapacity: " << ac << endl;

        scheduleAt(simTime() + serviceTime, ap);
    } else {
    }
}

void LinkSelector::getMaxIndexCapacity(){
    // monitoring
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
    }

    // indice che corrisponde al dataLink di capacita' maggiore
    maxCapacityDataLinkIndex = std::max_element(capacities.begin(),capacities.end()) - capacities.begin();
    MaxIndexActualCapacity = capacities.at(maxCapacityDataLinkIndex);
    // faccio partire il malus perché ho monitorato
    EV << "monitoraggio: " << maxCapacityDataLinkIndex << ", capacita " << MaxIndexActualCapacity << endl;
    penalty = true;
    emit(computeMeanMalus_,malusX);
    scheduleAt(simTime() + malusX, new cMessage("malus"));
}

void LinkSelector::scheduleCheckCapacity(){
    cMessage* checkingMaxCapacity = new cMessage("schedule");
    scheduleAt(simTime() + m, checkingMaxCapacity);
}

