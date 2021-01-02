package aerocomsystemproject;

simple ControlTower
{
    parameters: 
        int nDL = default(2);
        int nA = default(1);	//Numero di data-link
        
    gates:
        input in[nDL*nA];  
}