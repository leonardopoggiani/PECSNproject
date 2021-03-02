package aerocomsystemproject;
// the link selector is the module responsible for the scheduling of the highest capacity DataLink
simple LinkSelector
{
    parameters:
      int nDL;
      
      @signal[computeQueueLength](type=int);
	  @statistic[queueLength](source=computeQueueLength; record=vector,mean;);
	  
	  @signal[computeServiceTime](type=long);
      @statistic[serviceTime](source=computeServiceTime; record=vector,mean;);
      
	  @signal[computeWaitingTime](type=long);
	  @statistic[waitingTime](source=computeWaitingTime; record=vector,mean;);
	  
	  @signal[computeMeanMalus](type=int);
      @statistic[meanMalus](source=computeMeanMalus; record=sum;);
      
      @signal[computeResponseTime](type=simtime_t);
      @statistic[responseTime](source=computeResponseTime; record=vector,mean;);
      
      @signal[packetDelivered](type=int);
      @statistic[delivery](source=packetDelivered; record=sum;);
   	gates:
      output out[nDL];	//collegamento con gate dell'aereo
      input in; //collegamento con packetGenerator    
}