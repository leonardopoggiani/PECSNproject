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
    int operationMode;
    double monitoringTime;
    int maxCapacityDataLinkIndex;
    int nDL;

    void handlePacketArrival(cMessage *msg);
    int getCapacity();
    void handleSetCapacity();
    void scheduleCheckCapacity();
};

#endif

