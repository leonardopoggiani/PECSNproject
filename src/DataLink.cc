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

   transmitting = false;
   t = getAncestorPar("t").doubleValue(); // il valore della media per lognormal ed exponential
   k = getAncestorPar("k").doubleValue();
   size = par("s"); // la dimensione di un pacchetto
   dimPoolMax = par("dimPoolMax"); // massima capacità del DL
   dimPoolMin = par("dimPoolMin"); // minima capacità del DL
   lastCapacity = uniform(dimPoolMin,dimPoolMax); // ultima capacità, in occasione della initialize va estratta casualmente
   nextCapacity = uniform(dimPoolMin,dimPoolMax); // verrà riestratta ogni monitoringTime

   lastCapacityTime = 0; // tempo in cui si è effettuato l'ultimo aggiornamento della capacità

   int tempLast = 0;
   // Li ordino per trovare actualCapacity
   if(lastCapacity > nextCapacity)
   {
       tempLast = lastCapacity;
       lastCapacity = nextCapacity;
       nextCapacity = tempLast;
   }

   actualCapacity = uniform(lastCapacity,nextCapacity); // capacità attuale del DL, la prima va estratta, poi varierà linearmente
   // EV << "First Actual capacity is: " << actualCapacity << endl;

   serviceTime = size/actualCapacity;
   EV <<"Service time is: " << serviceTime <<endl;

   setCapacityDistribution_ = par("setCapacityDistribution").stdstringValue(); // il tipo di distribuzione che si intende usare

   cMessage * msg = new cMessage("setNextCapacity");
   scheduleSetNextCapacity(msg); // schedulazione del prossimo aggiornamento della capacità

}

void DataLink::handleMessage(cMessage *msg)
{
    if ( msg->isSelfMessage() ) {
            if ( strcmp(msg->getName(), "setNextCapacity") == 0 )
                handleSetNextCapacity(msg);
            else if (strcmp(msg->getName(), "serviceTimeElapsed") == 0)
                handleServiceTimeElapsed(msg);
    }
    else
    {
        // Pacchetto arrivato da Aircraft
        handlePacketArrival(msg);
    }
}

/* Aggiornamento della capacità: viene estratta la prossima capacità da raggiungere e si varia linearmente dall'ultima capacità
 * estratta alla prossima. In qualsiasi tempo tra l'ultimo aggiornamento e il prossimo la capacità viene ritornata da getCapacity().
 */
void DataLink::handleSetNextCapacity(cMessage *msg)
{

    lastCapacity = nextCapacity; // l'ultima capacità viene aggiornata
    nextCapacity = uniform(dimPoolMin,dimPoolMax); // estratta la capacità da raggiungere tra t_
    lastCapacityTime = simTime();
    scheduleSetNextCapacity(msg);
}

void DataLink::handlePacketArrival(cMessage *msg) {
    EV_INFO << ", queue length: " << queue.getLength() << endl;
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
    if ( !queue.isEmpty()) {

        AircraftPacket* ap = (AircraftPacket*) queue.front();
        queue.pop();
        EV << "WaitingTime: " << simTime() - ap->getArrivalTime()<< endl;
        emit(computeWaitingTime_, simTime() - ap->getArrivalTime());

        transmitting = true;

        actualCapacity = getCapacity();
        serviceTime = size/actualCapacity;
        processing = ap;
        scheduleAt(simTime() + serviceTime, new cMessage("serviceTimeElapsed"));
        EV_INFO << "==> SendPacket " << processing->getId() << " with service time "<< serviceTime << ", packet exit at: "<< simTime() + serviceTime << ", capacity: " << actualCapacity << endl;

    }
}

void DataLink::handleServiceTimeElapsed(cMessage *msg){

    emit(computeResponseTime_, simTime() - processing->getSendTime()  + serviceTime);
    EV << "ResponseTime: " << simTime() - processing->getSendTime()  + serviceTime << endl;

    send(processing, "out");
    transmitting = false;
    delete msg;
    sendPacket();
}

void DataLink::handlePacketSent(cMessage *msg) {
    transmitting = false;
    // Try to send a new packet

   /* if (schedulePenalty) {
        EV_INFO << "Penalty started, "<< simTime() <<endl;
        EV_INFO << "Penalty should end at " << simTime().dbl() +p << endl;
        scheduleAt(simTime().dbl() + p, new cMessage("penaltyTimeElapsed"));
        schedulePenalty = false;
    }*/
    delete msg;
}


/***********************************************
***************** UTILITY **********************
************************************************/

void DataLink::scheduleSetNextCapacity(cMessage *msg)
{
    if ( strcmp(setCapacityDistribution_.c_str(), "lognormal") == 0){
        interval = lognormal(t,0);
        scheduleAt(simTime() + interval, msg);
    } else if (strcmp(setCapacityDistribution_.c_str(), "exponential") == 0 ){
        interval = exponential(t,0);
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
    // EV << "increment: " << increment << " , ratio: " << ratio << endl;
    // EV << "last: " << lastCapacity << " , next: " << nextCapacity << " , actual: " << ret << endl;

    return ret;
}
