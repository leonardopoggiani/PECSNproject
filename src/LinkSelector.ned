package aerocomsystemproject;
// the link selector is the module responsible for the scheduling of the highest capacity DataLink
simple LinkSelector
{
    parameters:
      int DLMaxCapacity;
      bool trasmitting;
      int nDL;
      int operationMode;
   	gates:
      output out[nDL];	//collegamento con gate dell'aereo
      input in; //collegamento con packetGenerator    
}