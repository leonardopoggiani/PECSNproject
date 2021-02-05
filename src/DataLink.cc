#include "DataLink.h"
#include "AircraftPacket_m.h"

Define_Module(DataLink);

void DataLink::initialize()
{
   //** SIGNALS **//
   computeResponseTime_ = registerSignal("computeResponseTime");
   computeWaitingTime_ = registerSignal("computeWaitingTime");
   computeQueueLength_ = registerSignal("computeQueueLength");
   computeTDistribution_ = registerSignal("computeTDistribution");
   computeActualCapacity_ = registerSignal("computeActualCapacity");
   computeMeanMalus_ = registerSignal("computeMeanMalus");
   computeServiceTime_ = registerSignal("computeServiceTime");
   computeThroughput_ = registerSignal("computeThroughput");
   computeSentPackets_ = registerSignal("computeSentPackets");

   operationMode = par("operationMode");
   transmitting = false;
   malusPenalty = false;
   scheduleMalus = false;
   malusX = par("X").doubleValue();
   t = getAncestorPar("t").doubleValue(); // il valore della media per lognormal ed exponential
   size = par("s").doubleValue(); // la dimensione di un pacchetto
   dimPoolMax = par("dimPoolMax"); // massima capacita del DL
   dimPoolMin = par("dimPoolMin"); // minima capacita del DL
   lastCapacity = uniform(dimPoolMin,dimPoolMax,1); // ultima capacita, in occasione della initialize va estratta casualmente
   nextCapacity = uniform(dimPoolMin,dimPoolMax,1); // verra riestratta ogni monitoringTime

   lastCapacityTime = 0; // tempo in cui si e' effettuato l'ultimo aggiornamento della capacita'

   int tempLast = 0;
   // Li ordino per trovare actualCapacity
   if(lastCapacity > nextCapacity) {
       tempLast = lastCapacity;
       lastCapacity = nextCapacity;
       nextCapacity = tempLast;
   }

   actualCapacity = uniform(lastCapacity,nextCapacity,1); // capacita' attuale del DL, la prima va estratta, poi variera' linearmente
   emit(computeActualCapacity_,actualCapacity);
   EV <<"Capacita' attuale " << actualCapacity << endl;

   double s = (double) size;
   double ac = (double) actualCapacity;
   serviceTime = s/ac;

   emit(computeServiceTime_,serviceTime);
   EV <<"Service time is: " << serviceTime <<endl;

   tDistribution = par("tDistribution").stdstringValue(); // il tipo di distribuzione che si intende usare

   cMessage * msg = new cMessage("setNextCapacity");
   scheduleSetNextCapacity(msg); // schedulazione del prossimo aggiornamento della capacita'
}

void DataLink::handleMessage(cMessage *msg)
{
    if ( msg->isSelfMessage() ) {
        if ( strcmp(msg->getName(), "setNextCapacity") == 0 ){
            handleSetNextCapacity(msg);
        } else if ( strcmp(msg->getName(), "serviceTimeElapsed") == 0 ){
            // passato il service time posso gestire un nuovo pacchetto
            handleServiceTimeElapsed();
            delete msg;
        } else if( strcmp(msg->getName(), "malusElapsed") == 0 ){
            // passato il malus per il monitoraggio, posso gestire un nuovo pacchetto
            handleMalusElapsed();
            delete msg;
        }
    }
    else {
        if( strcmp(msg->getName(), "startMalusPenality") == 0 ){
            // inizio a scontare la penalita'
            handleStartMalusPenality();
            delete msg;
        } else {
            // nuovo pacchetto arrivato da Aircraft
            handlePacketArrival(msg);
        }
    }
}

/* Aggiornamento della capacita: viene estratta la prossima capacita da raggiungere e si varia linearmente dall'ultima capacita
 * estratta alla prossima. In qualsiasi tempo tra l'ultimo aggiornamento e il prossimo la capacita viene ritornata da getCapacity() e
 * sara' un valore compreso dalla capacita' precedente e la successiva.
 */
void DataLink::handleSetNextCapacity(cMessage *msg)
{
    lastCapacity = nextCapacity; // l'ultima capacita' viene aggiornata, se sto estraendo ho raggiunto la capacita' estratta precedentemente
    nextCapacity = uniform(dimPoolMin,dimPoolMax,1); // estratta la capacita da raggiungere tra t_
    lastCapacityTime = simTime(); // tempo dell'ultimo aggiornamento di capacita', ora
    scheduleSetNextCapacity(msg);
}

void DataLink::handlePacketArrival(cMessage *msg) {
    // arrivato un pacchetto da packetGenerator
    EV_INFO << "queue length: " << queue.getLength() << endl;
    emit(computeQueueLength_, queue.getLength());

    AircraftPacket* pa = check_and_cast<AircraftPacket*>(msg);
    pa->setArrivalTime(simTime().dbl());

    queue.insert(pa);
    if ( !transmitting ) {
        // provo a mandare un pacchetto, se non sto trasmettendo (sta scadendo il serviceTime)
        sendPacket();
    }
}

void DataLink::sendPacket() {
    if ( !queue.isEmpty() && !malusPenalty ) {
        // la coda non e' vuota e non sto scontando una  penalita'
        AircraftPacket* ap = (AircraftPacket*) queue.front();
        queue.pop();
        EV << "WaitingTime: " << simTime() - ap->getArrivalTime()<< endl; // tempo attuale - tempo in cui il pacchetto e' entrato in coda
        emit(computeWaitingTime_, simTime() - ap->getArrivalTime());

        transmitting = true; // sto trasmettendo

        actualCapacity = getCapacity();
        double s = (double) size;
        double ac = (double) actualCapacity;
        serviceTime = (s/ac);
        EV << "serviceTime " << serviceTime << endl;
        emit(computeServiceTime_,serviceTime);
        processing = ap; // il pacchetto che sto processando e' quello attuale

        scheduleAt(simTime() + serviceTime, new cMessage("serviceTimeElapsed"));
        send(processing,"out");

        EV << "ResponseTime: " << simTime().dbl() - processing->getSendTime()  + serviceTime << endl;
        emit(computeResponseTime_, simTime().dbl() - processing->getSendTime()  + serviceTime);

        EV_INFO << "==> SendPacket " << processing->getId() << " with service time "<< serviceTime << ", packet exit at: "<< simTime() + serviceTime << ", capacity: " << actualCapacity << endl;
    }
}

void DataLink::handleServiceTimeElapsed(){

    transmitting = false;
    emit(computeSentPackets_, 1);

    sendPacket(); // mando il prossimo pacchetto in coda

    if (malusPenalty) {
       EV_INFO << "Penalty started, "<< simTime() << endl;
       EV_INFO << "Penalty should end at " << simTime().dbl() + malusX << endl;
       scheduleAt(simTime() + malusX, new cMessage("malusElapsed"));
       emit(computeMeanMalus_, malusX);
       malusPenalty = false;
    }
}

void DataLink::handleStartMalusPenality() {
    if ( !transmitting ) {
        EV_INFO << "Penalty started, "<< simTime() << endl;
        EV_INFO << "Penalty should end at " << simTime().dbl() + malusX << endl;
        emit(computeMeanMalus_, malusX);
        scheduleAt(simTime() + malusX, new cMessage("malusElapsed"));
    } else {
        EV_INFO << "Penalty starting after finishing the current transmission" << endl;
        malusPenalty = true;
    }
}

void DataLink::handleMalusElapsed() {
    EV_INFO << "==> PenaltyTimeElapsed: handover completed, transmissions restored, "<< simTime() << endl;
    malusPenalty = false;
    sendPacket();
}

/***********************************************
***************** UTILITY **********************
************************************************/

void DataLink::scheduleSetNextCapacity(cMessage *msg)
{
    if ( strcmp(tDistribution.c_str(), "lognormal") == 0){
        interval = lognormal(t,2);
        scheduleAt(simTime() + interval, msg);
        emit(computeTDistribution_,  interval);
    } else if (strcmp(tDistribution.c_str(), "exponential") == 0 ){
        interval = exponential(t,2);
        scheduleAt(simTime() + interval, msg);
        emit(computeTDistribution_,  interval);
    }
}

int DataLink::getCapacity()
{
    bool discesa = false; // se sto diminuendo la capacita' rispetto a quella precedente
    simtime_t timeInterval = simTime() - lastCapacityTime;
    int deltaCapacity = nextCapacity - lastCapacity;

    if(deltaCapacity < 0) {
        int tmp = abs(deltaCapacity);
        deltaCapacity = tmp;
        discesa = true;
    }

    double ratio = (timeInterval.dbl()/interval);
    double increment = ratio*deltaCapacity;

    int ret = 0;
    if(discesa){
        ret = lastCapacity - increment;
    } else {
        ret = lastCapacity + increment;
    }

    emit(computeActualCapacity_,ret);
    return ret;
}
