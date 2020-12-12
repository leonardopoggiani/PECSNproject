#ifndef __AEROCOMSYSTEMPROJECT_DATALINK_H_
#define __AEROCOMSYSTEMPROJECT_DATALINK_H_

#include <omnetpp.h>

using namespace omnetpp;

class DataLink : public cSimpleModule
{
  public:
    int getCapacity();

  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

  private:
    cPacketQueue queue;
    double t_;
    double k_;
    std::string setCapacityDistribution_;
    int nextCapacity;
    int lastCapacity;
    int actualCapacity; //Sempre vecchia ma mi serve come appoggio
    simtime_t serviceTime; //Sempre vecchio ma mi serve come appoggio
    simtime_t lastCapacityTime;
    int dimPoolMax_;
    int dimPoolMin_;

    void handlePacketArrival(cMessage* msg);
    void sendPacket();
    void handleSetNextCapacity(cMessage* msg);
    void scheduleSetNextCapacity(cMessage* msg);
};

#endif
