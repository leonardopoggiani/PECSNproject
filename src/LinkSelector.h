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
    bool transmitting;
    bool schedulePenalty;

    void handlePacketArrival(cMessage *msg);
    void scheduleCheckCapacity();
    void getMaxIndexCapacity();
    void sendPacket();
    void sendPacketToDataLink(cMessage* msg);
    void handleMalusElapsed();
    void handleStartMalusPenalty();
    void handleServiceTimeElapsed(cMessage* msg);

    simsignal_t computeQueueLength_;
    simsignal_t computeServiceTime_;
    simsignal_t computeWaitingTime_;
    simsignal_t computeMeanMalus_;
    simsignal_t computeResponseTime_;
    simsignal_t packetDelivered_;

};

#endif

