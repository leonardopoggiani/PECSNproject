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
  private:
    double k;
    std::string distribution;
    simsignal_t  computeArrivalTime_;


};

#endif /* PACKETGENERATOR_H_ */
