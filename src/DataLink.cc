#include "DataLink.h"
#include "simtime.h"

Define_Module(DataLink);

void DataLink::initialize()
{
   t_ = par("t");
   k_ = par("k");
   size_ = par("size");
   dimPoolMax_ = par("dimPoolMax");
   dimPoolMin_ = par("dimPoolMin");
   lastCapacity = rand() % dimPoolMax_ + dimPoolMin_;
   nextCapacity = rand() % dimPoolMax_ + dimPoolMin_;
   EV << "First last capacity is: " << lastCapacity <<endl;
   EV << "First next capacity is: " << nextCapacity <<endl;

   lastCapacityTime =simTime()+uniform(0,2);

   int tempLast;
   //Li ordino per trovare actualCapacity
   if(lastCapacity > nextCapacity)
   {
       tempLast = lastCapacity;
       lastCapacity = nextCapacity;
       nextCapacity = tempLast;
   }
   actualCapacity = rand()%nextCapacity+lastCapacity;
   EV << "First Actual capacity is: " << actualCapacity <<endl;

   //serviceTime = size_/actualCapacity;

   //EV <<"Service time is: " << serviceTime <<endl;

   setCapacityDistribution_ = par("setCapacityDistribution").stdstringValue();
   cMessage * msg = new cMessage("setNextCapacity");
   scheduleSetNextCapacity(msg);

}

void DataLink::handleMessage(cMessage *msg)
{
    if ( msg->isSelfMessage() ) {
            if ( strcmp(msg->getName(), "setNextCapacity") == 0 ) {
                handleSetNextCapacity(msg);
        }
    }
}

void DataLink::handleSetNextCapacity(cMessage *msg)
{
    lastCapacity = nextCapacity;
    nextCapacity = rand() % dimPoolMax_ + dimPoolMin_;
    //lastCapacityTime = simTime();
    actualCapacity = getCapacity();
    EV << "Actual capacity is: " << actualCapacity <<endl;
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
    int actualCapacityL =nextCapacity/(simTime()-lastCapacityTime);
    return actualCapacityL;
}
