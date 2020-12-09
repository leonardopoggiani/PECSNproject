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
    void handlePacketArrival(AircraftPacket* ap);
    int getCapacity();
    void handleSetCapacity(cMessage* msg);
    void scheduleCheckCapacity();
};

#endif

