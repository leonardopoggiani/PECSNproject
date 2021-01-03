package aerocomsystemproject;

simple DataLink
{
    parameters:
         double t @unit(s); //Ogni quanto controlla la capacità dei data-link (VA)
         int dimPoolMax; //Const
         int dimPoolMin; //Const
         int operationMode;
         double X;
         double s @unit(byte);
         int nDL;
         string tDistribution = default("exponential");     
         
         @signal[computeResponseTime](type="simtime_t");
       	 @statistic[responseTime](source=computeResponseTime; record=vector;);
       	 
       	 @signal[computeWaitingTime](type="long");
       	 @statistic[waitingTime](source=computeWaitingTime; record=vector;);
       	 
       	 @signal[computeQueueLength](type=int);
       	 @statistic[queueLength](source=computeQueueLength; record=vector;);
        	
         @signal[computeTDistribution](type=long);
       	 @statistic[tDistribution](source=computeTDistribution; record=vector;);
       	
        
        //@signal[computeCapacity](type=long);
       	//@statistic[capacity](source=computeCapacity; record=vector;);
       	
       	@signal[computeActualCapacity](type=long);
       	@statistic[actualCapacity](source=computeActualCapacity; record=vector;);
       	
       	@signal[computeMeanMalus](type=long);
       	@statistic[meanMalus](source=computeMeanMalus; record=vector;);
       	
     gates:
         input in;
         output out;
        
}
