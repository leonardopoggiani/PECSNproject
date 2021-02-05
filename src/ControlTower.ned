package aerocomsystemproject;

simple ControlTower
{
    parameters: 
        int nDL = default(2);	//Numero di dataLink
        int nA = default(1);	//Numero di aircraft
        
        @signal[computeReceivedPackets](type=int);
        @statistic[receivedPackets](source=computeReceivedPackets; record=count);

    gates:
        input in[nDL*nA];  
}