#ifndef PACKETGENERATOR_H_
#define PACKETGENERATOR_H_

#include <omnetpp.h>

using namespace omnetpp;

class PacketGenerator : public cSimpleModule
{
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

};

#endif /* PACKETGENERATOR_H_ */
