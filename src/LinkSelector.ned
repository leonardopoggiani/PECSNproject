package aerocomsystemproject;
// the link selector is the module responsible for the scheduling of the highest capacity DataLink
simple LinkSelector
{
    parameters:
      int nDL;
      
      @signal[computeQueueLength](type=int);
	  @statistic[queueLength](source=computeQueueLength; record=vector,sum;);
   	gates:
      output out[nDL];	//collegamento con gate dell'aereo
      input in; //collegamento con packetGenerator    
}