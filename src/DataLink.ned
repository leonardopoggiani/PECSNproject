package aerocomsystemproject;

simple DataLink
{
    parameters:
         double t @unit(s); //Ogni quanto controlla la capacità dei data-link (VA)
         double k @unit(s);
         //int actualCapacity; //Capacità attuale del DL (VA)
         int dimPoolMax; //Const
         int dimPoolMin; //Const
         double X;
         int nDL;
         int s @unit(byte);
         string setCapacityDistribution = default("exponential");     
         
         @signal[computeResponseTime](type=long);
       	 @statistic[responseTime](source=computeResponseTime; record=vector;);
       	 
       	 @signal[computeWaitingTime](type=long);
       	 @statistic[waitingTime](source=computeWaitingTime; record=vector;);
       	 
       	 @signal[computeQueueLength](type=long);
       	 @statistic[queueLength](source=computeQueueLength; record=vector;);
        
     gates:
         input in[nDL];
         output out;
        
}
