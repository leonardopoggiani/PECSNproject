#ifndef CONTROLTOWER_H_
#define CONTROLTOWER_H_

#include <omnetpp.h>

using namespace omnetpp;

class ControlTower : public cSimpleModule
{
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

    simsignal_t computeReceivedPackets_;

};
#endif /* CONTROLTOWER_H_ */
