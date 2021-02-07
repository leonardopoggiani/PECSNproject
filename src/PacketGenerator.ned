package aerocomsystemproject;

simple PacketGenerator
{
    parameters:     
        @signal[computeArrivalTime](type=long);
        @statistic[arrivalTime](source=computeArrivalTime; record=vector;);	
        
    gates:
       output out;
}