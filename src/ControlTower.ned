package aerocomsystemproject;

simple ControlTower
{
    parameters: 
        int nDL;	//Numero di data-link
        
    gates:
        input in[nDL*nDL];  
}