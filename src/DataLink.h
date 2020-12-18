#ifndef __AEROCOMSYSTEMPROJECT_DATALINK_H_
#define __AEROCOMSYSTEMPROJECT_DATALINK_H_

#include <omnetpp.h>
#include "AircraftPacket_m.h"

using namespace omnetpp;

class DataLink : public cSimpleModule
{
  public:
    int getCapacity();

  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

  private:
    double t_;
    cPacketQueue queue;
    double mean;
    double k_;
    int size_;
    std::string setCapacityDistribution_;
    int nextCapacity;
    int lastCapacity;
    int actualCapacity; //Sempre vecchia ma mi serve come appoggio
    simtime_t serviceTime; //Sempre vecchio ma mi serve come appoggio
    simtime_t lastCapacityTime;
    int dimPoolMax_;
    int dimPoolMin_;
    bool transmitting;
    AircraftPacket* processing;

    void handlePacketArrival(cMessage* msg);
    void sendPacket();
    void handleSetNextCapacity(cMessage* msg);
    void scheduleSetNextCapacity(cMessage* msg);
    void handlePacketSent(cMessage *msg);
    void handleServiceTimeElapsed(cMessage* msg);

    simsignal_t computeResponseTime_;
    simsignal_t computeWaitingTime_;
    simsignal_t computeQueueLength_;
};

#endif
