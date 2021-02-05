package aerocomsystemproject;

simple DataLink
{
    parameters:
		 double t @unit(s); // ogni quanto controlla la capacita' dei data-link (VA)
		 int dimPoolMax; // const
		 int dimPoolMin; // const
		 int operationMode;
		 double X @unit(s);
		 double s @unit(byte);
		 int nDL;
		 string tDistribution = default("exponential");     
		         
		 @signal[computeResponseTime](type=simtime_t);
		 @statistic[responseTime](source=computeResponseTime; record=vector;);
		 
		 @signal[computeWaitingTime](type=long);
		 @statistic[waitingTime](source=computeWaitingTime; record=vector;);
		 
		 @signal[computeQueueLength](type=int);
		 @statistic[queueLength](source=computeQueueLength; record=vector;);
			
		 @signal[computeTDistribution](type=long);
		 @statistic[tDistribution](source=computeTDistribution; record=vector;);
		       	
		 @signal[computeActualCapacity](type=long);
		 @statistic[actualCapacity](source=computeActualCapacity; record=vector;);
		
		 @signal[computeMeanMalus](type=long);
		 @statistic[meanMalus](source=computeMeanMalus; record=vector;);
		
		 @signal[computeServiceTime](type=long);
		 @statistic[serviceTime](source=computeServiceTime; record=vector;);
		
		 @signal[computeUtilization](type=long);
		 @statistic[utilization](source=computeUtilization; record=vector;);
       	
     gates:
         input in;
         output out;
        
}
