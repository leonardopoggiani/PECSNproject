#ifndef __AEROCOMSYSTEMPROJECT_DATALINK_H_
#define __AEROCOMSYSTEMPROJECT_DATALINK_H_

#include <omnetpp.h>

using namespace omnetpp;

class LinkSelector : public cSimpleModule
{
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);
  private:
    // quando definiremo AircraftPacket void handlePacketArrival(AircraftPacket* ap);
    void handlePacketArrival(cMessage *msg);
    int getCapacity();
    void handleSetCapacity(cMessage* msg);
    void scheduleCheckCapacity();
};

#endif

