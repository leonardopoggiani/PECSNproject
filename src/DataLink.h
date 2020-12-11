//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#ifndef __AEROCOMSYSTEMPROJECT_DATALINK_H_
#define __AEROCOMSYSTEMPROJECT_DATALINK_H_

#include <omnetpp.h>

using namespace omnetpp;
using namespace std;

class DataLink : public cSimpleModule
{
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);
  private:
    cPacketQueue queue;
    double t_;
    double k_;
    string setCapacityDistribution_;
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
    int getCapacity();
};

#endif
