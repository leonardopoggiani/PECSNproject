#include "DataLink.h"
#include "simtime.h"
#include "AircraftPacket_m.h"

Define_Module(DataLink);

void DataLink::initialize()
{
   transmitting = false;
   mean = par("mean").doubleValue(); // il valore della media per lognormal ed exponential
   k_ = par("k").doubleValue();
   size_ = par("size"); // la dimensione di un pacchetto
   dimPoolMax_ = par("dimPoolMax"); // massima capacità del DL
   dimPoolMin_ = par("dimPoolMin"); // minima capacità del DL
   lastCapacity = uniform(dimPoolMin_,dimPoolMax_); // ultima capacità, in occasione della initialize va estratta casualmente
   nextCapacity = uniform(dimPoolMin_,dimPoolMax_); // verrà riestratta ogni monitoringTime

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
   EV << "First Actual capacity is: " << actualCapacity << endl;

   //serviceTime = size_/actualCapacity;
   serviceTime = 2;
   //EV <<"Service time is: " << serviceTime <<endl;

   setCapacityDistribution_ = par("setCapacityDistribution").stdstringValue(); // il tipo di distribuzione che si intende usare

   cMessage * msg = new cMessage("setNextCapacity");
   scheduleSetNextCapacity(msg); // schedulazione del prossimo aggiornamento della capacità

}

void DataLink::handleMessage(cMessage *msg)
{
    if ( msg->isSelfMessage() ) {
            if ( strcmp(msg->getName(), "setNextCapacity") == 0 )
                handleSetNextCapacity(msg);
            else if ( strcmp(msg->getName(), "packetSent") == 0 )
                handlePacketSent(msg);
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
    nextCapacity = uniform(dimPoolMin_,dimPoolMax_); // estratta la capacità da raggiungere tra t_
    lastCapacityTime = simTime();
    scheduleSetNextCapacity(msg);
}

void DataLink::handlePacketArrival(cMessage *msg) {
    EV_INFO << "==> PacketArrival";
    EV_INFO << ", queue length: " << queue.getLength() << ", transmitting: " << transmitting << endl;
    AircraftPacket* pa = check_and_cast<AircraftPacket*>(msg);
    queue.insert(pa);
    if ( !transmitting ) {
           // Try to send a new packet
          sendPacket();
       }
}

void DataLink::sendPacket() {
    if ( !queue.isEmpty()) {

        AircraftPacket* ap = (AircraftPacket*) queue.front();
        queue.pop();

        transmitting = true;
        //scheduleAt(simTime() + s, new cMessage("packetSent"));
        EV_INFO << "==> SendPacket " << ap->getId() << " with service time "<< serviceTime << ", packet exit at: "<< simTime() + serviceTime <<endl;
        scheduleAt(simTime() + serviceTime, new cMessage("packetSent"));

        send(ap, "out");
    }
}

void DataLink::handlePacketSent(cMessage *msg) {
    transmitting = false;
    // Try to send a new packet
    sendPacket();

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
    // TODO
    if ( strcmp(setCapacityDistribution_.c_str(), "lognormal") == 0){
                t_ = lognormal(mean,0);
                scheduleAt(simTime() + t_, msg);
    } else if (strcmp(setCapacityDistribution_.c_str(), "exponential") == 0 ){
                t_ = exponential(mean,0);
                scheduleAt(simTime() + t_, msg);
    }
}

int DataLink::getCapacity()
{
    bool discesa = false;
    simtime_t timeInterval = simTime() - lastCapacityTime;
    int deltaCapacity = nextCapacity - lastCapacity;

    if(deltaCapacity < 0){
        EV << " scendo " << endl;
        int tmp = abs(deltaCapacity);
        deltaCapacity = tmp;
        discesa = true;
    }

    double ratio = (timeInterval.dbl()/t_);
    double increment = ratio*deltaCapacity;

    int ret = 0;
    if(discesa){
        ret = lastCapacity - increment;
    } else {
        ret = lastCapacity + increment;
    }
    EV << "increment: " << increment << " , ratio: " << ratio << endl;
    EV << "last: " << lastCapacity << " , next: " << nextCapacity << " , actual: " << ret << endl;

    return ret;
}
