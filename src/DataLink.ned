package aerocomsystemproject;

simple DataLink
{
     double mean @unit(s); //Ogni quanto controlla la capacità dei data-link (VA)
     double k @unit(s);
     //int actualCapacity; //Capacità attuale del DL (VA)
     int dimPoolMax; //Const
     int dimPoolMin; //Const
     double X;
     int nDL;
     int size @unit(byte);
     string setCapacityDistribution = default("exponential");
     
     
     gates:
         input in[nDL];
         output out;
        
}
