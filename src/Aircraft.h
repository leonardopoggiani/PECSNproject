#ifndef AIRCRAFT_H_
#define AIRCRAFT_H_

#include <omnetpp.h>

using namespace omnetpp;

class Aircraft : public cSimpleModule
{
  public:
    int getMonitorTime();
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);
  private:
    int m;
};

#endif /* AIRCRAFT_H_ */
