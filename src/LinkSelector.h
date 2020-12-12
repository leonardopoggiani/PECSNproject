#ifndef LINKSELECTOR_H_
#define LINKSELECTOR_H_

#include <omnetpp.h>
#include <vector>
#include <algorithm>
#include "DataLink.h"

using namespace omnetpp;

class LinkSelector : public cSimpleModule
{
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

  private:
    int operationMode;
    double m_;
    int maxCapacityDataLinkIndex;
    int nDL_;

    void handlePacketArrival(cMessage *msg);
    void handleSetCapacity();
    void scheduleCheckCapacity();
    int getMaxIndexCapacity();
};

#endif

