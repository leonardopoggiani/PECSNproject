#include "DataLink.h"
#include "simtime.h"
#include "AircraftPacket_m.h"

Define_Module(DataLink);

void DataLink::initialize()
{
   transmitting = false;
   mean = par("mean").doubleValue();
   k_ = par("k").doubleValue();
   size_ = par("size");
   dimPoolMax_ = par("dimPoolMax");
   dimPoolMin_ = par("dimPoolMin");
   // lastCapacity = rand() % dimPoolMax_ + dimPoolMin_;
   lastCapacity = uniform(dimPoolMin_,dimPoolMax_);
   nextCapacity = uniform(dimPoolMin_,dimPoolMax_);
   // nextCapacity = rand() % dimPoolMax_ + dimPoolMin_;

   // lastCapacityTime =simTime()+uniform(0,2);
   lastCapacityTime = 0;

   int tempLast;
   //Li ordino per trovare actualCapacity
   if(lastCapacity > nextCapacity)
   {
       tempLast = lastCapacity;
       lastCapacity = nextCapacity;
       nextCapacity = tempLast;
   }
   actualCapacity = uniform(lastCapacity,nextCapacity);
   EV << "First Actual capacity is: " << actualCapacity << endl;

   //serviceTime = size_/actualCapacity;
   serviceTime = 2;
   //EV <<"Service time is: " << serviceTime <<endl;
   EV << "First last capacity is: " << lastCapacity <<endl;
   EV << "First next capacity is: " << nextCapacity <<endl;


   setCapacityDistribution_ = par("setCapacityDistribution").stdstringValue();
   cMessage * msg = new cMessage("setNextCapacity");
   scheduleSetNextCapacity(msg);

}

void DataLink::handleMessage(cMessage *msg)
{
    if ( msg->isSelfMessage() ) {
            if ( strcmp(msg->getName(), "setNextCapacity") == 0 )
                handleSetNextCapacity(msg);
            else if ( strcmp(msg->getName(), "packetSent") == 0 )
                handlePacketSent(msg);
    }
    //Packet arrived from aircraft
    else
    {
        handlePacketArrival(msg);
    }
}

void DataLink::handleSetNextCapacity(cMessage *msg)
{

    lastCapacity = nextCapacity;
    nextCapacity = uniform(dimPoolMin_,dimPoolMax_);
    lastCapacityTime = simTime();
    EV << "Actual capacity is: " << actualCapacity <<endl;
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
