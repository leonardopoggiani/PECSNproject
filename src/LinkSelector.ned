package aerocomsystemproject;
// the link selector is the module responsible for the scheduling of the highest capacity DataLink
simple LinkSelector
{
    parameters:
      int nDL;
      
      @signal[computeQueueLength](type=int);
	  @statistic[queueLength](source=computeQueueLength; record=vector,sum;);
	  
	  @signal[computeServiceTime](type=long);
      @statistic[serviceTime](source=computeServiceTime; record=vector,mean;);
      
	  @signal[computeWaitingTime](type=long);
	  @statistic[waitingTime](source=computeWaitingTime; record=vector,mean;);
	  
	  @signal[computeMeanMalus](type=int);
      @statistic[meanMalus](source=computeMeanMalus; record=sum;);
   	gates:
      output out[nDL];	//collegamento con gate dell'aereo
      input in; //collegamento con packetGenerator    
}