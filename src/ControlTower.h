#ifndef __AEROCOMSYSTEMPROJECT_CONTROLTOWER_H_
#define __AEROCOMSYSTEMPROJECT_CONTROLTOWER_H_

#include <omnetpp.h>

using namespace omnetpp;

class ControlTower : public cSimpleModule
{
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

  private:
    simsignal_t computeReceivedPackets_;

};

#endif /* __AEROCOMSYSTEMPROJECT_CONTROLTOWER_H_ */
