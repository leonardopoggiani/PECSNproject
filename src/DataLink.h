#ifndef __AEROCOMSYSTEMPROJECT_DATALINK_H_
#define __AEROCOMSYSTEMPROJECT_DATALINK_H_

#include <omnetpp.h>
#include <vector>
#include <algorithm>
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
    std::string tDistribution;
    int nextCapacity;
    int lastCapacity;
    int actualCapacity;
    double serviceTime;
    simtime_t lastCapacityTime;
    int dimPoolMax;
    int dimPoolMin;
    bool transmitting;
    AircraftPacket* processing;
    bool malusPenalty;
    bool scheduleMalus;
    int operationMode;

    void handlePacketArrival(cMessage* msg);
    void sendPacket();
    void handleSetNextCapacity(cMessage* msg);
    void scheduleSetNextCapacity(cMessage* msg);
    void handlePacketSent(cMessage *msg);
    void handleMalusElapsed();
    double obtainServiceTime();
    void handleSentPacket(cMessage* msg);

    simsignal_t computeTDistribution_;
    simsignal_t computeMeanMalus_;
    simsignal_t computeActualCapacity_;
    simsignal_t computeSentPackets_;

};

#endif
