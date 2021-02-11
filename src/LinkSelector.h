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
    cPacketQueue queue;
    int operationMode;
    double m;
    int maxCapacityDataLinkIndex;
    int nDL;
    bool penalty;
    double malusX;
    int MaxIndexActualCapacity;
    int size;

    void handlePacketArrival(cMessage *msg);
    void scheduleCheckCapacity();
    void getMaxIndexCapacity();
    void sendPacket();
    void sendPacketToDataLink(cMessage* msg);


    simsignal_t computeQueueLength_;
    simsignal_t computeServiceTime_;

};

#endif

