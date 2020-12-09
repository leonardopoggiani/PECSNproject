package aerocomsystemproject;

simple Aircraft
{
    int nDL = default(2);//(Cost)
    double k @unit(s); // Ogni quanto genera i pacchetti (VA)
   	double m @unit(s); //Tempo di monitoraggio (Cost)
    int size;	//Dimensione dei pacchetti (Cost)
    int serving; //Interfaccia sulla quale sta trasmettendo (1..nDL)
    
    gates:
        output out[nDL];
    
    
}