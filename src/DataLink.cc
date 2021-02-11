#include "DataLink.h"
#include "AircraftPacket_m.h"

Define_Module(DataLink);

void DataLink::initialize()
{
   //** SIGNALS **//
   computeResponseTime_ = registerSignal("computeResponseTime");
   computeTDistribution_ = registerSignal("computeTDistribution");
   computeActualCapacity_ = registerSignal("computeActualCapacity");
   computeSentPackets_ = registerSignal("computeSentPackets");
   computeThroughput_ = registerSignal("computeThroughput");

   operationMode = getAncestorPar("operationMode");
   transmitting = false;
   malusPenalty = false;
   scheduleMalus = false;
   t = getAncestorPar("t").doubleValue(); // il valore della media per lognormal ed exponential
   dimPoolMax = getAncestorPar("dimPoolMax"); // massima capacita del DL
   dimPoolMin = getAncestorPar("dimPoolMin"); // minima capacita del DL
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
   tDistribution = getAncestorPar("tDistribution").stdstringValue(); // il tipo di distribuzione che si intende usare

   cMessage * msg = new cMessage("setNextCapacity");
   scheduleSetNextCapacity(msg); // schedulazione del prossimo aggiornamento della capacita'
}

void DataLink::handleMessage(cMessage *msg)
{
    if ( msg->isSelfMessage() ) {
        handleSetNextCapacity(msg);
    } else {
        // nuovo pacchetto arrivato da Aircraft
        handlePacketArrival(msg);
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
    EV << "ATTUALE " << lastCapacity << ", DA RAGGIUNGERE " << nextCapacity << endl;
    EV << "next capacity " << getFullName() << ", " << nextCapacity << endl;
    scheduleSetNextCapacity(msg);
}

void DataLink::handlePacketArrival(cMessage *msg) {
    emit(computeSentPackets_, 1);
    emit(computeThroughput_, 1);
    send(msg,"out");
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
