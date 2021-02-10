package aerocomsystemproject;

simple DataLink
{
    parameters:
		         
		 @signal[computeResponseTime](type=simtime_t);
		 @statistic[responseTime](source=computeResponseTime; record=vector,mean;);
		 
		 @signal[computeWaitingTime](type=long);
		 @statistic[waitingTime](source=computeWaitingTime; record=vector,mean;);
		 
		 @signal[computeQueueLength](type=int);
		 @statistic[queueLength](source=computeQueueLength; record=vector,mean;);
			
		 @signal[computeTDistribution](type=long);
		 @statistic[tDistribution](source=computeTDistribution; record=vector;);
		       	
		 @signal[computeActualCapacity](type=long);
		 @statistic[actualCapacity](source=computeActualCapacity; record=mean;);
		
		 @signal[computeMeanMalus](type=int);
		 @statistic[meanMalus](source=computeMeanMalus; record=count;);
		
		 @signal[computeServiceTime](type=long);
		 @statistic[serviceTime](source=computeServiceTime; record=vector,mean;);
		 
		 @signal[computeThroughput](type=long);
		 @statistic[throughput](source=computeThroughput; record=vector;);
		 
        @signal[computeSentPackets](type=int);
		@statistic[sentPackets](source=computeSentPackets;record=count);
       	
     gates:
         input in;
         output out;
        
}
