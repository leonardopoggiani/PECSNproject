#include "LinkSelector.h"

Define_Module(LinkSelector);

void LinkSelector::initialize()
{
    maxCapacityDataLinkIndex = -1; // il datalink che si occupera' dell'invio, indice del datalink a capacita' piu' alta
    operationMode = getAncestorPar("operationMode"); // 0-> monitoraggio costante dei DataLink attivo, 1-> non attivo, scelgo un DL e inviero' sempre su quello
    malusX = getAncestorPar("X").doubleValue();
    penalty = false;
    nDL = getAncestorPar("nDL");
    transmitting = false;
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
        //Monitoraggio prima volta
       if ( strcmp(msg->getName(), "setMaxIndexCapacity") == 0 ){
            getMaxIndexCapacity();
            delete msg;
        }else if ( strcmp(msg->getName(), "serviceTimeElapsed") == 0 ){
            // passato il service time posso gestire un nuovo pacchetto
            handleServiceTimeElapsed();
            delete msg;

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
    send(msg,"out",maxCapacityDataLinkIndex);
    sendPacket();
}

void LinkSelector::handlePacketArrival(cMessage* msg) {
    // qui mi e' arrivato il pacchetto da packetGenerator, adesso inoltro verso il DL scelto
    AircraftPacket* pa = check_and_cast<AircraftPacket*>(msg);
    pa->setArrivalTime(simTime().dbl());
    pa->setName("packetToSend");
    queue.insert(pa);

   if ( !transmitting ) {
        // provo a mandare un pacchetto, se non sto trasmettendo (sta scadendo il serviceTime)
        sendPacket();
    }

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

        scheduleAt(simTime() + serviceTime, new cMessage("serviceTimeElapsed"));
        transmitting = true; // sto trasmettendo
        //La chiamo qua senn� non riuscivo a ragionare, puoi rispostarla se vuoi
        sendPacketToDataLink(ap);

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
    // faccio partire il malus perch� ho monitorato
    EV << "monitoraggio: " << maxCapacityDataLinkIndex << ", capacita " << MaxIndexActualCapacity << endl;
    penalty = true;
    handleStartMalusPenalty();

}

void LinkSelector::scheduleCheckCapacity(){
    cMessage* checkingMaxCapacity = new cMessage("schedule");
    scheduleAt(simTime() + m, checkingMaxCapacity);
}

void LinkSelector::handleServiceTimeElapsed(){

    transmitting = false;

    sendPacket(); // mando il prossimo pacchetto in coda
    //Se ritorno � perch� penalty a true o perch� transmitting a true
    //Se � penalty ad essere a true allora vado avanti e comincio il malus, altrimenti niente

    if (penalty) {
       EV_INFO << "Penalty started, "<< simTime() << endl;
       EV_INFO << "Penalty should end at " << simTime().dbl() + malusX << endl;
       scheduleAt(simTime() + malusX, new cMessage("malusElapsed"));
       penalty = false;
    }
}

void LinkSelector::handleStartMalusPenality() {
    if ( !transmitting ) {
        EV_INFO << "Penalty started, "<< simTime() << endl;
        EV_INFO << "Penalty should end at " << simTime().dbl() + malusX << endl;
        scheduleAt(simTime() + malusX, new cMessage("malusElapsed"));
    } else {
        EV_INFO << "Penalty starting after finishing the current transmission" << endl;
        penalty = true;
    }
}

void LinkSelector::handleMalusElapsed() {
    EV_INFO << "==> PenaltyTimeElapsed: handover completed, transmissions restored, "<< simTime() << endl;
    emit(computeMeanMalus_, malusX);
    penalty = false;
    sendPacket();
}

