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
    double malusX;
    bool malusPenality;
    bool scheduleMalus;
    int operationMode;

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
    simsignal_t computeTDistribution_;
    simsignal_t computeMeanMalus_;
    simsignal_t computeActualCapacity_;
    simsignal_t computeServiceTime_;
    simsignal_t computeUtilization_;

};

#endif
