package aerocomsystemproject;

simple PacketGenerator
{
    parameters:     
        @signal[computeArrivalTime](type=long);
        @statistic[arrivalTime](source=computeArrivalTime; record=vector;);
        
        @signal[computeSentPackets](type=int);
		@statistic[sentPackets](source=computeSentPackets;record=count);	
        
    gates:
       output out;
}