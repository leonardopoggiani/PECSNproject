#ifndef PACKETGENERATOR_H_
#define PACKETGENERATOR_H_

#include <omnetpp.h>

using namespace omnetpp;

class PacketGenerator : public cSimpleModule
{
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage* msg);
    virtual void createSendPacket(cMessage *msg);
    virtual void scheduleCreateSendPacket(cMessage* msg);
  private:
    double k_;
    std::string distribution;

};

#endif /* PACKETGENERATOR_H_ */
