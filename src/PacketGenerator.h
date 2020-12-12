#ifndef PACKETGENERATOR_H_
#define PACKETGENERATOR_H_

#include <omnetpp.h>

using namespace omnetpp;

class PacketGenerator : public cSimpleModule
{
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage* msg);
    virtual void createSendPacket();
    virtual void scheduleArrival(cMessage* msg);
  private:
    double k_;

};

#endif /* PACKETGENERATOR_H_ */
