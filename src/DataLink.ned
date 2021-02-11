package aerocomsystemproject;

simple DataLink
{
    parameters:
		         
		 @signal[computeResponseTime](type=simtime_t);
		 @statistic[responseTime](source=computeResponseTime; record=vector,mean;);
		
		 @signal[computeTDistribution](type=long);
		 @statistic[tDistribution](source=computeTDistribution; record=vector;);
		       	
		 @signal[computeActualCapacity](type=long);
		 @statistic[actualCapacity](source=computeActualCapacity; record=mean,vector;);
		 
		 @signal[computeThroughput](type=long);
		 @statistic[throughput](source=computeThroughput; record=vector,mean;);
		 
         @signal[computeSentPackets](type=int);
		 @statistic[sentPackets](source=computeSentPackets;record=count,vector);
       	
     gates:
         input in;
         output out;
        
}
