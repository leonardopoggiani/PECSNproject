#include "DataLink.h"
#include "simtime.h"
#include "AircraftPacket_m.h"

Define_Module(DataLink);

void DataLink::initialize()
{
   //** SIGNALS **//
   computeResponseTime_ = registerSignal("computeResponseTime");
   computeWaitingTime_ = registerSignal("computeWaitingTime");
   computeQueueLength_ = registerSignal("computeQueueLength");

   operationMode = par("operationMode");
   transmitting = false;
   malusPenality = false;
   scheduleMalus = false;
   malusX = par("X").doubleValue();
   t = getAncestorPar("t").doubleValue(); // il valore della media per lognormal ed exponential
   size = par("s").doubleValue(); // la dimensione di un pacchetto
   dimPoolMax = par("dimPoolMax"); // massima capacitï¿½ del DL
   dimPoolMin = par("dimPoolMin"); // minima capacitï¿½ del DL
   lastCapacity = uniform(dimPoolMin,dimPoolMax,1); // ultima capacitï¿½, in occasione della initialize va estratta casualmente
   nextCapacity = uniform(dimPoolMin,dimPoolMax,1); // verrï¿½ riestratta ogni monitoringTime

   lastCapacityTime = 0; // tempo in cui si ï¿½ effettuato l'ultimo aggiornamento della capacitï¿½

   int tempLast = 0;
   // Li ordino per trovare actualCapacity
   if(lastCapacity > nextCapacity)
   {
       tempLast = lastCapacity;
       lastCapacity = nextCapacity;
       nextCapacity = tempLast;
   }

   actualCapacity = uniform(lastCapacity,nextCapacity,1); // capacitï¿½ attuale del DL, la prima va estratta, poi varierï¿½ linearmente
   // EV << "First Actual capacity is: " << actualCapacity << endl;

   serviceTime = size/actualCapacity;
   EV <<"Service time is: " << serviceTime <<endl;

   setCapacityDistribution_ = par("setCapacityDistribution").stdstringValue(); // il tipo di distribuzione che si intende usare

   cMessage * msg = new cMessage("setNextCapacity");
   scheduleSetNextCapacity(msg); // schedulazione del prossimo aggiornamento della capacitï¿½

}

void DataLink::handleMessage(cMessage *msg)
{
    if ( msg->isSelfMessage() ) {
        if ( strcmp(msg->getName(), "setNextCapacity") == 0 )
            handleSetNextCapacity(msg);
        else if (strcmp(msg->getName(), "serviceTimeElapsed") == 0){
            handleServiceTimeElapsed();
            delete msg;
        } else if(strcmp(msg->getName(), "malusElapsed") == 0){
            handleMalusElapsed();
            delete msg;
        }
    }
    else
    {
        if(strcmp(msg->getName(), "startMalusPenality") == 0){
            handleStartMalusPenality();
            delete msg;
        } else {
            // Pacchetto arrivato da Aircraft
            handlePacketArrival(msg);
        }

    }
}

/* Aggiornamento della capacitï¿½: viene estratta la prossima capacitï¿½ da raggiungere e si varia linearmente dall'ultima capacitï¿½
 * estratta alla prossima. In qualsiasi tempo tra l'ultimo aggiornamento e il prossimo la capacitï¿½ viene ritornata da getCapacity().
 */
void DataLink::handleSetNextCapacity(cMessage *msg)
{

    lastCapacity = nextCapacity; // l'ultima capacitï¿½ viene aggiornata
    nextCapacity = uniform(dimPoolMin,dimPoolMax,1); // estratta la capacitï¿½ da raggiungere tra t_
    lastCapacityTime = simTime();
    scheduleSetNextCapacity(msg);
}

void DataLink::handlePacketArrival(cMessage *msg) {
    EV_INFO << "queue length: " << queue.getLength() << endl;
    emit(computeQueueLength_, queue.getLength());
    AircraftPacket* pa = check_and_cast<AircraftPacket*>(msg);
    pa->setArrivalTime(simTime().dbl());

    queue.insert(pa);
    if ( !transmitting ) {
        // Try to send a new packet
        sendPacket();
    }
}

void DataLink::sendPacket() { //elaboratePacket
    if ( !queue.isEmpty() && !malusPenality ) {

        AircraftPacket* ap = (AircraftPacket*) queue.front();
        queue.pop();
        EV << "WaitingTime: " << simTime() - ap->getArrivalTime()<< endl;
        emit(computeWaitingTime_, simTime() - ap->getArrivalTime());

        transmitting = true;

        actualCapacity = getCapacity();
        serviceTime = size/actualCapacity;
        processing = ap;

        scheduleAt(simTime() + serviceTime, new cMessage("serviceTimeElapsed"));
        send(processing,"out");
        EV << "ResponseTime: " << simTime() - processing->getSendTime()  + serviceTime << endl;
        emit(computeResponseTime_, simTime() - processing->getSendTime()  + serviceTime);

        EV_INFO << "==> SendPacket " << processing->getId() << " with service time "<< serviceTime << ", packet exit at: "<< simTime() + serviceTime << ", capacity: " << actualCapacity << endl;
    } else {
        EV << "Non sono entrato " << endl;
    }
}

void DataLink::handleServiceTimeElapsed(){

    transmitting = false;

    sendPacket();

    if (malusPenality) {
       EV_INFO << "Penalty started, "<< simTime() <<endl;
       EV_INFO << "Penalty should end at " << simTime().dbl() + malusX << endl;
       scheduleAt(simTime() + malusX, new cMessage("malusElapsed"));
       malusPenality = false;
    }
}

void DataLink::handleStartMalusPenality() {
    if ( !transmitting ) {
        EV_INFO << "Penalty started, "<< simTime() <<endl;
        EV_INFO << "Penalty should end at " << simTime().dbl() + malusX << endl;
        scheduleAt(simTime() + malusX, new cMessage("malusElapsed"));
    } else {
        EV_INFO << "Penalty starting after finishing the current transmission" << endl;
        malusPenality = true;
    }
}

void DataLink::handleMalusElapsed() {
    EV_INFO << "==> PenaltyTimeElapsed: handover completed, transmissions restored, "<< simTime() << endl;
    malusPenality = false;
    sendPacket();
}

/***********************************************
***************** UTILITY **********************
************************************************/

void DataLink::scheduleSetNextCapacity(cMessage *msg)
{
    if ( strcmp(setCapacityDistribution_.c_str(), "lognormal") == 0){
        interval = lognormal(t,2);
        scheduleAt(simTime() + interval, msg);
    } else if (strcmp(setCapacityDistribution_.c_str(), "exponential") == 0 ){
        interval = exponential(t,2);
        scheduleAt(simTime() + interval, msg);
    }
}

int DataLink::getCapacity()
{
    bool discesa = false;
    simtime_t timeInterval = simTime() - lastCapacityTime;
    int deltaCapacity = nextCapacity - lastCapacity;

    if(deltaCapacity < 0){
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

    return ret;
}
