#include "Aircraft.h"

Define_Module(Aircraft);

void Aircraft::initialize(){ m = par("m"); }

void Aircraft::handleMessage(cMessage *msg){}
