package aerocomsystemproject;

simple ControlTower
{
    parameters: 
        int nDL = default(2);	//Numero di dalaLink
        int nA = default(1);	//Numero di aircraft
        
    gates:
        input in[nDL*nA];  
}