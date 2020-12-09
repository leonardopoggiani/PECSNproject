package aerocomsystemproject;
// the link selector is the module responsible for the scheduling of the highest capacity DataLink
simple LinkSelector
{
    parameters:
      int DLMaxCapacity;
      double t; 
      bool trasmitting;
<<<<<<< HEAD
      int nDL;
   	gates:
      output out[nDL];	//collegamento con gate dell'aereo
      input in; //collegamento con pachetGenerator
          
=======
>>>>>>> 343b348c6a25c26a8b3c24d332d768cd8a585b92
      
}