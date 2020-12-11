#include "DataLink.h"

Define_Module(DataLink);

void DataLink::initialize()
{
   t_ = par("t");
   k_ = par("k");
   dimPoolMax_ = par("dimPoolMax");
   dimPoolMin_ = par("dimPoolMin");
   lastCapacity = rand() % dimPoolMax_ + dimPoolMin_;
   nextCapacity = rand() % dimPoolMax_ + dimPoolMin_;
   lastCapacityTime = simTime();
   actualCapacity = getCapacity();
   serviceTime = k_/actualCapacity;
   setCapacityDistribution_ = par("setCapacityDistribution").stdstringValue();

}

void DataLink::handleMessage(cMessage *msg)
{
    if ( msg->isSelfMessage() ) {
            if ( strcmp(msg->getName(), "setNewCapacity") == 0 ) {
                handleSetNextCapacity(msg);
        }
    }
}

void DataLink::handleSetNextCapacity(cMessage *msg)
{
    lastCapacity = nextCapacity;
    nextCapacity = rand() % dimPoolMax_ + dimPoolMin_;
    lastCapacityTime = simTime();
    scheduleSetNextCapacity(msg);
}

/***********************************************
***************** UTILITY **********************
************************************************/

void DataLink::scheduleSetNextCapacity(cMessage *msg)
{
    if ( strcmp(setCapacityDistribution_.c_str(), "lognormal") == 0)
                scheduleAt(simTime() + lognormal(k_,0), msg);
    else if (strcmp(setCapacityDistribution_.c_str(), "exponential") == 0 )
                scheduleAt(simTime() + exponential(k_, 0), msg );
}

int DataLink::getCapacity()
{
    int actualCapacity = (lastCapacity*simTime())/lastCapacityTime;
    return actualCapacity;
}
