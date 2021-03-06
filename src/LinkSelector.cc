#include "LinkSelector.h"

Define_Module(LinkSelector);

void LinkSelector::initialize()
{
    maxCapacityDataLinkIndex = -1; // il datalink che si occupera' dell'invio, indice del datalink a capacita' piu' alta
    operationMode = getAncestorPar("operationMode"); // 0-> monitoraggio costante dei DataLink attivo, 1-> non attivo, scelgo un DL e inviero' sempre su quello
    malusX = getAncestorPar("X").doubleValue();
    penalty = false;
    schedulePenalty = false;
    nDL = getAncestorPar("nDL");
    transmitting = false;
    computeQueueLength_ = registerSignal("computeQueueLength");
    computeWaitingTime_ = registerSignal("computeWaitingTime");
    computeMeanMalus_ = registerSignal("computeMeanMalus");
    computeServiceTime_ = registerSignal("computeServiceTime");
    computeResponseTime_ = registerSignal("computeResponseTime");
    packetDelivered_ = registerSignal("packetDelivered");

    size = getAncestorPar("s").doubleValue(); // la dimensione di un pacchetto

    if(operationMode == 0 && nDL > 0){
        // monitoraggio ogni m secondi
        m = getAncestorPar("m");
        // avvio la schedulazione dei pacchetti di monitoraggio dei DataLink
        scheduleCheckCapacity();
    }

    if(nDL > 0){
        cMessage* checkingMaxCapacity = new cMessage("setMaxIndexCapacity");
        scheduleAt(simTime(), checkingMaxCapacity);
    }

}

void LinkSelector::handleMessage(cMessage* msg){
    if(msg->isSelfMessage()){
        // Monitoraggio prima volta
       if ( strcmp(msg->getName(), "setMaxIndexCapacity") == 0 ){
            getMaxIndexCapacity();
            delete msg;
        }else if ( strcmp(msg->getName(), "serviceTimeElapsed") == 0 ){
            // passato il service time posso gestire un nuovo pacchetto
            handleServiceTimeElapsed(msg);
        } else if( strcmp(msg->getName(), "malusElapsed") == 0 ){
            // passato il malus per il monitoraggio, posso gestire un nuovo pacchetto
            handleMalusElapsed();
            delete msg;
        } else if(strcmp(msg->getName(), "schedule") == 0 ) {
            // ricevuto un pacchetto di monitoraggio, rischedulo il prossimo e analizzo la capacita' dei dataLink
            scheduleCheckCapacity();
            getMaxIndexCapacity();
            delete msg;
        }

    } else {
        handlePacketArrival(msg);
    }
}

void LinkSelector::sendPacketToDataLink(cMessage* msg){
    AircraftPacket* ap = check_and_cast<AircraftPacket*>(msg);
    emit(packetDelivered_,1);
    emit(computeResponseTime_, simTime() - ap->getArrival());
    EV << "responseTime: " << simTime() - ap->getArrival() << endl;

    send(msg,"out",maxCapacityDataLinkIndex);
    sendPacket();
}

void LinkSelector::handlePacketArrival(cMessage* msg) {
    if(nDL > 0){
        // qui mi e' arrivato il pacchetto da packetGenerator, adesso inoltro verso il DL scelto
        AircraftPacket* pa = check_and_cast<AircraftPacket*>(msg);
        pa->setArrival(simTime().dbl());
        pa->setName("packetToSend");
        queue.insert(pa);

       if ( !transmitting ) {
            // provo a mandare un pacchetto, se non sto trasmettendo (sta scadendo il serviceTime)
            sendPacket();
        }
    } else {
        delete msg;
    }
}

void LinkSelector::sendPacket() {
    if ( !queue.isEmpty() && !penalty ) {
        // la coda non e' vuota e non sto scontando una  penalita'
        AircraftPacket* ap = (AircraftPacket*) queue.front();
        queue.pop();
        emit(computeWaitingTime_, simTime() - ap->getArrival());
        emit(computeQueueLength_, queue.getLength());

        double s = (double) size;
        std::string path = "dataLink[" + std::to_string(maxCapacityDataLinkIndex) + "]";
        cModule* temp =  gate("out",maxCapacityDataLinkIndex)->getPathEndGate()->getOwnerModule();
        DataLink* dl = check_and_cast<DataLink*> (temp);
        double ac = dl->getCapacity();
        double serviceTime = s/ac;

        emit(computeServiceTime_, serviceTime);

        ap->setName("serviceTimeElapsed");
        scheduleAt(simTime() + serviceTime, ap);
        transmitting = true; // sto trasmettendo
    }
}

/*
* Ritorna l'indice del datalink a capacit� maggiore.
*/

void LinkSelector::getMaxIndexCapacity(){
    // monitoring
    std::vector<int> capacities;
    DataLink* dl;
    penalty = true;
    int actualCapacity;

    for(int i = 0; i < nDL; i++){
        cModule* temp;
        // scorro tutti i dataLink
        std::string path = "dataLink[" + std::to_string(i) + "]";
        temp =  gate("out",i)->getPathEndGate()->getOwnerModule();
        dl = check_and_cast<DataLink*> (temp);
        actualCapacity = dl->getCapacity();
        capacities.push_back(actualCapacity);
    }

    // indice che corrisponde al dataLink di capacita' maggiore
    maxCapacityDataLinkIndex = std::max_element(capacities.begin(),capacities.end()) - capacities.begin();
    MaxIndexActualCapacity = capacities.at(maxCapacityDataLinkIndex);
    // faccio partire il malus perch� ho monitorato

    handleStartMalusPenalty();
}

void LinkSelector::scheduleCheckCapacity(){
    cMessage* checkingMaxCapacity = new cMessage("schedule");
    scheduleAt(simTime() + m, checkingMaxCapacity);
}

void LinkSelector::handleServiceTimeElapsed(cMessage* msg){

    transmitting = false;

    sendPacketToDataLink(msg);

    if (schedulePenalty) {
       scheduleAt(simTime() + malusX, new cMessage("malusElapsed"));
    }
}

void LinkSelector::handleStartMalusPenalty() {
    if ( !transmitting ) {
        scheduleAt(simTime() + malusX, new cMessage("malusElapsed"));
        schedulePenalty = false;
    } else {
        schedulePenalty = true;
    }
}

void LinkSelector::handleMalusElapsed() {
    emit(computeMeanMalus_, malusX);
    penalty = false;
    sendPacket();
}

