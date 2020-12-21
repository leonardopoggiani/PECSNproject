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
    double t;
    double interval;
    cPacketQueue queue;
    int size;
    std::string setCapacityDistribution_;
    int nextCapacity;
    int lastCapacity;
    int actualCapacity; //Sempre vecchia ma mi serve come appoggio
    simtime_t serviceTime; //Sempre vecchio ma mi serve come appoggio
    simtime_t lastCapacityTime;
    int dimPoolMax;
    int dimPoolMin;
    bool transmitting;
    AircraftPacket* processing;
    double malusX;
    bool malusPenality;
    bool scheduleMalus;

    void handlePacketArrival(cMessage* msg);
    void sendPacket();
    void handleSetNextCapacity(cMessage* msg);
    void scheduleSetNextCapacity(cMessage* msg);
    void handlePacketSent(cMessage *msg);
    void handleServiceTimeElapsed();
    void handleStartMalusPenality();
    void handleMalusElapsed();

    simsignal_t computeResponseTime_;
    simsignal_t computeWaitingTime_;
    simsignal_t computeQueueLength_;
};

#endif
